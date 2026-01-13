"""
Rate Limiting Middleware for FastAPI

Implements token bucket algorithm for API rate limiting
"""

from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Dict, Tuple
import time
import logging

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Token bucket rate limiter

    Allows bursts while maintaining average rate limit
    """

    def __init__(self, rate: int = 60, period: int = 60):
        """
        Args:
            rate: Number of requests allowed per period
            period: Time period in seconds (default: 60s)
        """
        self.rate = rate
        self.period = period
        self.buckets: Dict[str, Tuple[float, int]] = {}

    def _get_client_id(self, request: Request) -> str:
        """
        Get unique client identifier from request

        Uses IP address or API key if available
        """
        # Try to get IP from X-Forwarded-For header (if behind proxy)
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()

        # Fall back to client host
        if request.client:
            return request.client.host

        return "unknown"

    def is_allowed(self, request: Request) -> Tuple[bool, Dict]:
        """
        Check if request is allowed under rate limit

        Returns:
            Tuple of (is_allowed, rate_limit_info)
        """
        client_id = self._get_client_id(request)
        current_time = time.time()

        # Get or create bucket for this client
        if client_id not in self.buckets:
            self.buckets[client_id] = (current_time, self.rate)

        last_check, tokens = self.buckets[client_id]

        # Calculate token replenishment
        time_passed = current_time - last_check
        tokens_to_add = time_passed * (self.rate / self.period)
        tokens = min(self.rate, tokens + tokens_to_add)

        # Check if request can be allowed
        if tokens >= 1:
            tokens -= 1
            self.buckets[client_id] = (current_time, tokens)

            rate_limit_info = {
                "limit": self.rate,
                "remaining": int(tokens),
                "reset": int(current_time + (self.period * (1 - tokens / self.rate)))
            }

            return True, rate_limit_info
        else:
            # Rate limit exceeded
            retry_after = int((1 - tokens) * (self.period / self.rate))

            rate_limit_info = {
                "limit": self.rate,
                "remaining": 0,
                "reset": int(current_time + retry_after),
                "retry_after": retry_after
            }

            return False, rate_limit_info

    def cleanup_old_buckets(self, max_age: int = 3600):
        """
        Remove buckets that haven't been used recently

        Args:
            max_age: Maximum age in seconds before removal (default: 1 hour)
        """
        current_time = time.time()
        to_remove = []

        for client_id, (last_check, _) in self.buckets.items():
            if current_time - last_check > max_age:
                to_remove.append(client_id)

        for client_id in to_remove:
            del self.buckets[client_id]

        if to_remove:
            logger.info(f"Cleaned up {len(to_remove)} old rate limit buckets")


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for rate limiting
    """

    def __init__(self, app, rate: int = 60, period: int = 60):
        super().__init__(app)
        self.limiter = RateLimiter(rate=rate, period=period)

    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for health check endpoints
        if request.url.path in ["/health", "/docs", "/openapi.json", "/redoc"]:
            return await call_next(request)

        # Check rate limit
        is_allowed, rate_info = self.limiter.is_allowed(request)

        if not is_allowed:
            client_id = self.limiter._get_client_id(request)
            logger.warning(
                f"Rate limit exceeded for {client_id} "
                f"on {request.url.path}"
            )

            # Log security event
            try:
                from security_monitor import log_rate_limit_violation
                log_rate_limit_violation(
                    client_id=client_id,
                    endpoint=request.url.path,
                    request_count=self.limiter.rate
                )
            except Exception as e:
                logger.error(f"Failed to log rate limit violation: {e}")

            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "message": f"Too many requests. Please try again in {rate_info['retry_after']} seconds.",
                    "retry_after": rate_info["retry_after"]
                },
                headers={
                    "X-RateLimit-Limit": str(rate_info["limit"]),
                    "X-RateLimit-Remaining": str(rate_info["remaining"]),
                    "X-RateLimit-Reset": str(rate_info["reset"]),
                    "Retry-After": str(rate_info["retry_after"])
                }
            )

        # Add rate limit headers to response
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(rate_info["limit"])
        response.headers["X-RateLimit-Remaining"] = str(rate_info["remaining"])
        response.headers["X-RateLimit-Reset"] = str(rate_info["reset"])

        return response


# Endpoint-specific rate limiters for finer control
def create_endpoint_limiter(rate: int = 10, period: int = 60):
    """
    Create a rate limiter for specific endpoints

    Usage:
        limiter = create_endpoint_limiter(rate=10, period=60)

        @app.get("/expensive-endpoint")
        async def endpoint(request: Request):
            if not limiter.is_allowed(request)[0]:
                raise HTTPException(status_code=429, detail="Rate limit exceeded")
            ...
    """
    return RateLimiter(rate=rate, period=period)


# Cleanup task (run periodically)
import asyncio

async def cleanup_rate_limiters_task(limiter: RateLimiter, interval: int = 3600):
    """
    Periodic task to cleanup old rate limit buckets

    Args:
        limiter: RateLimiter instance
        interval: Cleanup interval in seconds
    """
    while True:
        await asyncio.sleep(interval)
        limiter.cleanup_old_buckets()
