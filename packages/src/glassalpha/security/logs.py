"""Log sanitization and secure logging utilities.

This module provides utilities for sanitizing log messages to prevent
information leakage, injection attacks, and compliance violations.
"""

import json
import logging
import re
from typing import Any


class SecureFormatter(logging.Formatter):
    """Secure log formatter that sanitizes sensitive information."""

    def __init__(self, *args, **kwargs):
        """Initialize secure formatter."""
        super().__init__(*args, **kwargs)
        self.sanitizer = LogSanitizer()

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with sanitization."""
        # Sanitize the message
        if hasattr(record, "msg") and record.msg:
            record.msg = self.sanitizer.sanitize(str(record.msg))

        # Sanitize arguments
        if hasattr(record, "args") and record.args:
            sanitized_args = []
            for arg in record.args:
                if isinstance(arg, str):
                    sanitized_args.append(self.sanitizer.sanitize(arg))
                else:
                    sanitized_args.append(arg)
            record.args = tuple(sanitized_args)

        return super().format(record)


class LogSanitizer:
    """Sanitizes log messages to prevent information leakage."""

    def __init__(self):
        """Initialize log sanitizer with default patterns."""
        self.patterns = self._build_sanitization_patterns()

    def _build_sanitization_patterns(self) -> list[tuple[re.Pattern, str]]:
        """Build list of sanitization patterns.

        Returns:
            List of (pattern, replacement) tuples

        """
        patterns = []

        # API keys and tokens
        patterns.extend(
            [
                (re.compile(r"\b[A-Za-z0-9]{32,}\b"), "[REDACTED_TOKEN]"),
                (re.compile(r'api[_-]?key["\']?\s*[:=]\s*["\']?([A-Za-z0-9]+)', re.IGNORECASE), 'api_key="[REDACTED]"'),
                (re.compile(r'token["\']?\s*[:=]\s*["\']?([A-Za-z0-9]+)', re.IGNORECASE), 'token="[REDACTED]"'),
                (re.compile(r'secret["\']?\s*[:=]\s*["\']?([A-Za-z0-9]+)', re.IGNORECASE), 'secret="[REDACTED]"'),
            ]
        )

        # Passwords
        patterns.extend(
            [
                (re.compile(r'password["\']?\s*[:=]\s*["\']?([^\s"\']+)', re.IGNORECASE), 'password="[REDACTED]"'),
                (re.compile(r'passwd["\']?\s*[:=]\s*["\']?([^\s"\']+)', re.IGNORECASE), 'passwd="[REDACTED]"'),
                (re.compile(r'pwd["\']?\s*[:=]\s*["\']?([^\s"\']+)', re.IGNORECASE), 'pwd="[REDACTED]"'),
            ]
        )

        # Email addresses (partial redaction)
        patterns.append(
            (
                re.compile(r"\b([A-Za-z0-9._%+-]+)@([A-Za-z0-9.-]+\.[A-Z|a-z]{2,})\b"),
                r"\1***@\2",
            )
        )

        # IP addresses (partial redaction)
        patterns.append(
            (
                re.compile(r"\b(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})\b"),
                r"\1.\2.***.\4",
            )
        )

        # File paths (redact user directories)
        patterns.extend(
            [
                (re.compile(r"/Users/([^/\s]+)"), "/Users/[USER]"),
                (re.compile(r"/home/([^/\s]+)"), "/home/[USER]"),
                (re.compile(r"C:\\\\Users\\\\([^\\\\s]+)", re.IGNORECASE), r"C:\\Users\\[USER]"),
            ]
        )

        # Credit card numbers (full redaction)
        patterns.append(
            (
                re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
                "[REDACTED_CC]",
            )
        )

        # Social Security Numbers (full redaction)
        patterns.append(
            (
                re.compile(r"\b\d{3}-?\d{2}-?\d{4}\b"),
                "[REDACTED_SSN]",
            )
        )

        # Control characters (security)
        patterns.append(
            (
                re.compile(r"[\x00-\x1f\x7f-\x9f]"),
                "[CTRL]",
            )
        )

        return patterns

    def sanitize(self, message: str) -> str:
        """Sanitize a log message.

        Args:
            message: Original log message

        Returns:
            Sanitized log message

        """
        if not isinstance(message, str):
            return str(message)

        sanitized = message

        # Apply all sanitization patterns
        for pattern, replacement in self.patterns:
            sanitized = pattern.sub(replacement, sanitized)

        return sanitized

    def add_pattern(self, pattern: str | re.Pattern, replacement: str) -> None:
        """Add custom sanitization pattern.

        Args:
            pattern: Regular expression pattern (string or compiled)
            replacement: Replacement string

        """
        if isinstance(pattern, str):
            pattern = re.compile(pattern)

        self.patterns.append((pattern, replacement))


def sanitize_log_message(message: str) -> str:
    """Sanitize a single log message.

    Args:
        message: Log message to sanitize

    Returns:
        Sanitized log message

    """
    sanitizer = LogSanitizer()
    return sanitizer.sanitize(message)


def setup_secure_logging(
    logger_name: str | None = None,
    level: int = logging.INFO,
    format_string: str | None = None,
    enable_json: bool = False,
) -> logging.Logger:
    """Set up secure logging with sanitization.

    Args:
        logger_name: Name of logger (default: root logger)
        level: Logging level (default: INFO)
        format_string: Custom format string
        enable_json: Whether to use JSON formatting

    Returns:
        Configured logger with secure formatting

    """
    # Get or create logger
    if logger_name:
        logger = logging.getLogger(logger_name)
    else:
        logger = logging.getLogger()

    # Set level
    logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create console handler
    handler = logging.StreamHandler()
    handler.setLevel(level)

    # Set up formatter
    if enable_json:
        formatter = JSONSecureFormatter()
    else:
        if format_string is None:
            format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        formatter = SecureFormatter(format_string)

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


class JSONSecureFormatter(SecureFormatter):
    """JSON formatter with secure sanitization."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as sanitized JSON."""
        # First sanitize using parent class
        super().format(record)

        # Create JSON log entry
        log_entry = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in {
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "exc_info",
                "exc_text",
                "stack_info",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
                "getMessage",
            }:
                # Sanitize extra fields
                if isinstance(value, str):
                    value = self.sanitizer.sanitize(value)
                log_entry[key] = value

        return json.dumps(log_entry, default=str, ensure_ascii=False)


def create_audit_logger(audit_id: str) -> logging.Logger:
    """Create a dedicated logger for audit operations.

    Args:
        audit_id: Unique identifier for the audit

    Returns:
        Configured audit logger

    """
    logger_name = f"glassalpha.audit.{audit_id}"
    logger = setup_secure_logging(
        logger_name=logger_name,
        level=logging.INFO,
        enable_json=True,
    )

    # Add audit ID to all log records
    class AuditFilter(logging.Filter):
        def filter(self, record):
            record.audit_id = audit_id
            return True

    logger.addFilter(AuditFilter())
    return logger


def log_security_event(
    event_type: str,
    details: dict[str, Any],
    severity: str = "INFO",
) -> None:
    """Log a security-related event.

    Args:
        event_type: Type of security event
        details: Event details (will be sanitized)
        severity: Event severity (INFO, WARNING, ERROR, CRITICAL)

    """
    security_logger = logging.getLogger("glassalpha.security")

    # Sanitize details
    sanitizer = LogSanitizer()
    sanitized_details = {}
    for key, value in details.items():
        if isinstance(value, str):
            sanitized_details[key] = sanitizer.sanitize(value)
        else:
            sanitized_details[key] = value

    # Log the event
    log_message = f"Security Event: {event_type}"
    extra = {
        "event_type": event_type,
        "details": sanitized_details,
        "security_event": True,
    }

    level = getattr(logging, severity.upper(), logging.INFO)
    security_logger.log(level, log_message, extra=extra)


def get_secure_logging_config() -> dict[str, Any]:
    """Get secure default configuration for logging.

    Returns:
        Dictionary with secure logging configuration

    """
    return {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "enable_json": False,
        "sanitize_messages": True,
        "redact_patterns": [
            "api_key",
            "token",
            "secret",
            "password",
            "passwd",
            "pwd",
        ],
        "max_message_length": 10000,  # Prevent log flooding
    }
