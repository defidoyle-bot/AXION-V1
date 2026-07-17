"""
AXION QUANT V4 - Production Logging System
Structured, async-safe logging with secret masking and rotation.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import structlog
from pythonjsonlogger import jsonlogger

from config.settings import get_config, LogLevel, Environment


# =============================================================================
# SECRET MASKING
# =============================================================================

SENSITIVE_PATTERNS = [
    re.compile(r'(api[_-]?key|access[_-]?key|secret[_-]?key|api[_-]?secret|token|password|auth)[:\s=]+([^\s&]+)', re.IGNORECASE),
    re.compile(r'(mx0[a-zA-Z0-9]+)', re.IGNORECASE),  # MEXC keys
    re.compile(r'(CG-[a-zA-Z0-9]+)', re.IGNORECASE),  # CoinGecko keys
    re.compile(r'([0-9]+:[A-Za-z0-9_-]{35})', re.IGNORECASE),  # Telegram bot tokens
]

class SecretMaskingFilter(logging.Filter):
    """Masks sensitive information in log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Mask secrets in log message and args."""
        if isinstance(record.msg, str):
            record.msg = self._mask_secrets(record.msg)

        if record.args:
            masked_args = []
            for arg in record.args:
                if isinstance(arg, str):
                    masked_args.append(self._mask_secrets(arg))
                else:
                    masked_args.append(arg)
            record.args = tuple(masked_args)

        return True

    def _mask_secrets(self, text: str) -> str:
        """Replace sensitive patterns with [REDACTED]."""
        for pattern in SENSITIVE_PATTERNS:
            text = pattern.sub(r'\1=[REDACTED]', text)
        return text


# =============================================================================
# LOG FORMATTERS
# =============================================================================

class ColoredFormatter(logging.Formatter):
    """Colored console formatter for development."""

    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m',
    }

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']

        timestamp = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        location = f"{record.name}:{record.funcName}:{record.lineno}"

        formatted = f"{color}[{timestamp}]{reset} {color}[{record.levelname}]{reset} [{location}] {record.getMessage()}"

        if record.exc_info:
            formatted += "\n" + self.formatException(record.exc_info)

        return formatted


class StructuredJsonFormatter(jsonlogger.JsonFormatter):
    """JSON formatter for production log aggregation."""

    def add_fields(self, log_record: Dict[str, Any], record: logging.LogRecord, message_dict: Dict[str, Any]) -> None:
        super().add_fields(log_record, record, message_dict)

        log_record['timestamp'] = datetime.fromtimestamp(record.created).isoformat()
        log_record['level'] = record.levelname
        log_record['logger'] = record.name
        log_record['function'] = record.funcName
        log_record['line'] = record.lineno
        log_record['thread'] = record.thread
        log_record['thread_name'] = record.threadName

        if hasattr(record, 'event_data'):
            log_record['event_data'] = record.event_data

        if hasattr(record, 'correlation_id'):
            log_record['correlation_id'] = record.correlation_id


# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(
    log_level: Optional[Union[str, LogLevel]] = None,
    logs_dir: Optional[Path] = None,
    environment: Optional[str] = None,
) -> logging.Logger:
    """Configure production-grade logging."""

    config = get_config()

    level = log_level or config.log_level
    if isinstance(level, LogLevel):
        level = level.value

    log_path = logs_dir or config.logs_dir
    env = environment or config.environment.value

    # Ensure logs directory exists
    log_path.mkdir(parents=True, exist_ok=True)

    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove existing handlers to prevent duplicates on reload
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Secret masking filter
    secret_filter = SecretMaskingFilter()

    # Console handler (colored for dev, plain for prod)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    console_handler.addFilter(secret_filter)

    if env in (Environment.DEVELOPMENT.value, Environment.TESTING.value):
        console_handler.setFormatter(ColoredFormatter())
    else:
        console_handler.setFormatter(logging.Formatter(
            '%(asctime)s [%(levelname)s] %(name)s:%(funcName)s:%(lineno)d - %(message)s'
        ))

    root_logger.addHandler(console_handler)

    # File handler with rotation (daily, keep 30 days)
    app_log_file = log_path / "axion_quant.log"
    file_handler = logging.handlers.TimedRotatingFileHandler(
        app_log_file,
        when='midnight',
        interval=1,
        backupCount=30,
        encoding='utf-8',
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.addFilter(secret_filter)

    if env == Environment.PRODUCTION.value:
        file_handler.setFormatter(StructuredJsonFormatter(
            '%(timestamp)s %(level)s %(logger)s %(function)s %(line)d %(message)s'
        ))
    else:
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s [%(levelname)s] %(name)s:%(funcName)s:%(lineno)d - %(message)s'
        ))

    root_logger.addHandler(file_handler)

    # Error log file (separate, keep 90 days)
    error_log_file = log_path / "errors.log"
    error_handler = logging.handlers.TimedRotatingFileHandler(
        error_log_file,
        when='midnight',
        interval=1,
        backupCount=90,
        encoding='utf-8',
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.addFilter(secret_filter)
    error_handler.setFormatter(StructuredJsonFormatter())
    root_logger.addHandler(error_handler)

    # Structured logging with structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer() if env == Environment.PRODUCTION.value else structlog.dev.ConsoleRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    logger = logging.getLogger("axion_quant")
    logger.info(
        "Logging initialized",
        extra={
            "event_data": {
                "level": level,
                "environment": env,
                "log_dir": str(log_path),
            }
        }
    )

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the AXION prefix."""
    return logging.getLogger(f"axion_quant.{name}")


# =============================================================================
# EVENT LOGGING
# =============================================================================

class EventLogger:
    """Structured event logging for the event-driven pipeline."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or get_logger("events")
        self._event_counts: Dict[str, int] = {}

    def log_event(self, event_type: str, event_data: Dict[str, Any], correlation_id: Optional[str] = None) -> None:
        """Log a structured event."""
        self._event_counts[event_type] = self._event_counts.get(event_type, 0) + 1

        extra = {
            "event_data": event_data,
            "correlation_id": correlation_id or self._generate_correlation_id(),
        }

        self.logger.info(f"EVENT: {event_type}", extra=extra)

    def log_event_error(self, event_type: str, error: Exception, event_data: Optional[Dict[str, Any]] = None) -> None:
        """Log an event processing error."""
        self.logger.error(
            f"EVENT_ERROR: {event_type} - {str(error)}",
            extra={"event_data": event_data or {}},
            exc_info=True,
        )

    def get_event_stats(self) -> Dict[str, int]:
        """Get event occurrence statistics."""
        return self._event_counts.copy()

    def _generate_correlation_id(self) -> str:
        """Generate a unique correlation ID."""
        import uuid
        return str(uuid.uuid4())[:8]
