"""Rate limiting utilities to prevent API throttling."""

import time
import logging
from typing import Callable, Any
from functools import wraps


class RateLimiter:
    """Token bucket rate limiter to enforce minimum interval between API calls."""

    def __init__(self, min_interval: float):
        """
        Initialize rate limiter.

        Args:
            min_interval: Minimum seconds between calls
        """
        self.min_interval = min_interval
        self.last_call = 0.0
        self.logger = logging.getLogger(__name__)

    def wait_if_needed(self) -> None:
        """Sleep if necessary to respect rate limit."""
        current_time = time.time()
        elapsed = current_time - self.last_call

        if elapsed < self.min_interval:
            sleep_time = self.min_interval - elapsed
            self.logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f}s")
            time.sleep(sleep_time)

        self.last_call = time.time()

    def __call__(self, func: Callable) -> Callable:
        """Decorator for rate-limited functions."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            self.wait_if_needed()
            return func(*args, **kwargs)
        return wrapper


class RetryHandler:
    """Exponential backoff retry logic for API calls."""

    def __init__(self, max_retries: int = 3, base_delay: float = 1.0):
        """
        Initialize retry handler.

        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Base delay in seconds for exponential backoff
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.logger = logging.getLogger(__name__)

    def execute_with_retry(
        self,
        func: Callable,
        *args,
        retryable_exceptions: tuple = (Exception,),
        **kwargs
    ) -> Any:
        """
        Execute function with exponential backoff retry.

        Args:
            func: Function to execute
            *args: Positional arguments for func
            retryable_exceptions: Tuple of exceptions that should trigger retry
            **kwargs: Keyword arguments for func

        Returns:
            Result of func if successful

        Raises:
            Last exception if all retries exhausted
        """
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)

            except retryable_exceptions as e:
                last_exception = e

                if attempt < self.max_retries:
                    delay = self.base_delay * (2 ** attempt)  # Exponential backoff
                    self.logger.warning(
                        f"Attempt {attempt + 1}/{self.max_retries + 1} failed: {e}. "
                        f"Retrying in {delay}s..."
                    )
                    time.sleep(delay)
                else:
                    self.logger.error(
                        f"All {self.max_retries + 1} attempts failed. Last error: {e}"
                    )

        # If we get here, all retries failed
        raise last_exception
