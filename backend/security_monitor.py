"""
Security Monitoring Utilities

Provides logging and alerting for security-related events:
- Rate limit violations
- Authentication failures
- Suspicious activity patterns
- Security header validation
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from collections import defaultdict
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class SecurityMonitor:
    """
    Monitor and log security events.

    Can be extended to send alerts via email, Slack, etc.
    """

    def __init__(self, log_file: Optional[str] = None):
        """
        Initialize security monitor.

        Args:
            log_file: Path to security log file (optional)
        """
        self.log_file = log_file or str(Path(__file__).parent / "security.log")
        self._setup_logging()

        # Track violations in memory for pattern detection
        self.rate_limit_violations: Dict[str, List[datetime]] = defaultdict(list)
        self.auth_failures: Dict[str, List[datetime]] = defaultdict(list)

    def _setup_logging(self):
        """Set up dedicated security logging."""
        security_logger = logging.getLogger("security")
        security_logger.setLevel(logging.INFO)

        # File handler for security events
        handler = logging.FileHandler(self.log_file)
        handler.setLevel(logging.INFO)

        # JSON format for easy parsing
        formatter = logging.Formatter(
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "event": "%(message)s"}'
        )
        handler.setFormatter(formatter)

        security_logger.addHandler(handler)
        self.security_logger = security_logger

    def log_rate_limit_violation(self, client_id: str, endpoint: str, request_count: int):
        """
        Log rate limit violation.

        Args:
            client_id: Client IP or identifier
            endpoint: API endpoint that was rate limited
            request_count: Number of requests made
        """
        event = {
            "type": "rate_limit_violation",
            "client_id": client_id,
            "endpoint": endpoint,
            "request_count": request_count,
            "timestamp": datetime.utcnow().isoformat()
        }

        self.security_logger.warning(json.dumps(event))

        # Track for pattern detection
        self.rate_limit_violations[client_id].append(datetime.utcnow())

        # Check for repeated violations
        if self._check_repeated_violations(client_id):
            self.alert_repeated_violations(client_id)

    def log_auth_failure(self, client_id: str, user_id: Optional[str] = None, reason: str = ""):
        """
        Log authentication failure.

        Args:
            client_id: Client IP or identifier
            user_id: User ID that failed auth (if available)
            reason: Reason for failure
        """
        event = {
            "type": "auth_failure",
            "client_id": client_id,
            "user_id": user_id,
            "reason": reason,
            "timestamp": datetime.utcnow().isoformat()
        }

        self.security_logger.warning(json.dumps(event))

        # Track for pattern detection
        self.auth_failures[client_id].append(datetime.utcnow())

        # Check for brute force attempts
        if self._check_brute_force_attempt(client_id):
            self.alert_brute_force(client_id)

    def log_suspicious_activity(self, client_id: str, activity_type: str, details: Dict):
        """
        Log suspicious activity.

        Args:
            client_id: Client IP or identifier
            activity_type: Type of suspicious activity
            details: Additional details about the activity
        """
        event = {
            "type": "suspicious_activity",
            "activity_type": activity_type,
            "client_id": client_id,
            "details": details,
            "timestamp": datetime.utcnow().isoformat()
        }

        self.security_logger.warning(json.dumps(event))

    def log_security_event(self, event_type: str, details: Dict):
        """
        Log general security event.

        Args:
            event_type: Type of security event
            details: Event details
        """
        event = {
            "type": event_type,
            "details": details,
            "timestamp": datetime.utcnow().isoformat()
        }

        self.security_logger.info(json.dumps(event))

    def _check_repeated_violations(self, client_id: str, threshold: int = 5, window_minutes: int = 10) -> bool:
        """
        Check if client has repeated rate limit violations.

        Args:
            client_id: Client identifier
            threshold: Number of violations to trigger alert
            window_minutes: Time window to check

        Returns:
            True if repeated violations detected
        """
        violations = self.rate_limit_violations.get(client_id, [])
        cutoff = datetime.utcnow() - timedelta(minutes=window_minutes)
        recent_violations = [v for v in violations if v > cutoff]

        return len(recent_violations) >= threshold

    def _check_brute_force_attempt(self, client_id: str, threshold: int = 10, window_minutes: int = 5) -> bool:
        """
        Check if client is attempting brute force attack.

        Args:
            client_id: Client identifier
            threshold: Number of failures to trigger alert
            window_minutes: Time window to check

        Returns:
            True if brute force detected
        """
        failures = self.auth_failures.get(client_id, [])
        cutoff = datetime.utcnow() - timedelta(minutes=window_minutes)
        recent_failures = [f for f in failures if f > cutoff]

        return len(recent_failures) >= threshold

    def alert_repeated_violations(self, client_id: str):
        """
        Alert for repeated rate limit violations.

        Args:
            client_id: Client identifier
        """
        message = f"ALERT: Client {client_id} has repeated rate limit violations"
        self.security_logger.error(json.dumps({
            "type": "alert",
            "alert_type": "repeated_violations",
            "client_id": client_id,
            "message": message,
            "timestamp": datetime.utcnow().isoformat()
        }))

        # TODO: Send alert via email/Slack/PagerDuty
        logger.error(message)

    def alert_brute_force(self, client_id: str):
        """
        Alert for potential brute force attack.

        Args:
            client_id: Client identifier
        """
        message = f"ALERT: Potential brute force attack from {client_id}"
        self.security_logger.error(json.dumps({
            "type": "alert",
            "alert_type": "brute_force",
            "client_id": client_id,
            "message": message,
            "timestamp": datetime.utcnow().isoformat()
        }))

        # TODO: Send urgent alert via email/Slack/PagerDuty
        logger.error(message)

    def get_security_summary(self, hours: int = 24) -> Dict:
        """
        Get summary of security events.

        Args:
            hours: Number of hours to summarize

        Returns:
            Dictionary with security event counts
        """
        cutoff = datetime.utcnow() - timedelta(hours=hours)

        # Count recent violations
        recent_rate_limit = sum(
            len([v for v in violations if v > cutoff])
            for violations in self.rate_limit_violations.values()
        )

        recent_auth_failures = sum(
            len([f for f in failures if f > cutoff])
            for failures in self.auth_failures.values()
        )

        # Get unique clients
        rate_limit_clients = set(
            client_id
            for client_id, violations in self.rate_limit_violations.items()
            if any(v > cutoff for v in violations)
        )

        auth_failure_clients = set(
            client_id
            for client_id, failures in self.auth_failures.items()
            if any(f > cutoff for f in failures)
        )

        return {
            "period_hours": hours,
            "rate_limit_violations": recent_rate_limit,
            "auth_failures": recent_auth_failures,
            "rate_limit_clients": len(rate_limit_clients),
            "auth_failure_clients": len(auth_failure_clients),
            "top_violators": self._get_top_violators(cutoff, limit=5)
        }

    def _get_top_violators(self, cutoff: datetime, limit: int = 5) -> List[Dict]:
        """Get top rate limit violators."""
        violator_counts = {}

        for client_id, violations in self.rate_limit_violations.items():
            recent = [v for v in violations if v > cutoff]
            if recent:
                violator_counts[client_id] = len(recent)

        # Sort by count descending
        top = sorted(violator_counts.items(), key=lambda x: x[1], reverse=True)[:limit]

        return [
            {"client_id": client_id, "violation_count": count}
            for client_id, count in top
        ]

    def cleanup_old_events(self, days: int = 7):
        """
        Clean up old events from memory.

        Args:
            days: Keep events from last N days
        """
        cutoff = datetime.utcnow() - timedelta(days=days)

        # Clean rate limit violations
        for client_id in list(self.rate_limit_violations.keys()):
            self.rate_limit_violations[client_id] = [
                v for v in self.rate_limit_violations[client_id] if v > cutoff
            ]
            if not self.rate_limit_violations[client_id]:
                del self.rate_limit_violations[client_id]

        # Clean auth failures
        for client_id in list(self.auth_failures.keys()):
            self.auth_failures[client_id] = [
                f for f in self.auth_failures[client_id] if f > cutoff
            ]
            if not self.auth_failures[client_id]:
                del self.auth_failures[client_id]


# Global security monitor instance
_security_monitor: Optional[SecurityMonitor] = None


def get_security_monitor() -> SecurityMonitor:
    """Get or create global security monitor instance."""
    global _security_monitor
    if _security_monitor is None:
        _security_monitor = SecurityMonitor()
    return _security_monitor


# Convenience functions
def log_rate_limit_violation(client_id: str, endpoint: str, request_count: int):
    """Log rate limit violation."""
    get_security_monitor().log_rate_limit_violation(client_id, endpoint, request_count)


def log_auth_failure(client_id: str, user_id: Optional[str] = None, reason: str = ""):
    """Log authentication failure."""
    get_security_monitor().log_auth_failure(client_id, user_id, reason)


def log_suspicious_activity(client_id: str, activity_type: str, details: Dict):
    """Log suspicious activity."""
    get_security_monitor().log_suspicious_activity(client_id, activity_type, details)


def get_security_summary(hours: int = 24) -> Dict:
    """Get security summary."""
    return get_security_monitor().get_security_summary(hours)
