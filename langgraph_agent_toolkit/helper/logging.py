import logging
import os
import sys
from typing import Any, Dict, Optional

from loguru import logger as loguru_logger

from langgraph_agent_toolkit.helper.types import EnvironmentMode


class SingletonMeta(type):
    """Singleton metaclass to ensure only one instance of LoggerConfig exists."""

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class LoggerConfig(metaclass=SingletonMeta):
    """Singleton class for configuring and managing the logger."""

    def __init__(self):
        self.env = EnvironmentMode(os.environ.get("ENV_MODE", EnvironmentMode.PRODUCTION))
        self.log_level = os.environ.get("LOG_LEVEL", "INFO" if self.env == EnvironmentMode.PRODUCTION else "DEBUG")
        self.json_logs = os.environ.get("JSON_LOGS", "false").lower() == "true"
        self.colorize = os.environ.get("COLORIZE", "true").lower() == "true"

        # Configure logger on initialization
        self._setup_logger()

    def _format_record(self, record: Dict[str, Any]) -> str:
        """Format log records for human-readable output (text mode only)."""
        # This formatter is only used when NOT in JSON mode
        return (
            "<green>[{level}]</green> <blue>{time:YYYY-MM-DD HH:mm:ss.SS}</blue>"
            " | <cyan>{module}:{function}:{line}</cyan>"
            " | <white>{message}</white>\n"
        )

    def _setup_logger(self) -> None:
        """Configure Loguru logger based on environment settings."""
        # Remove default handlers
        try:
            loguru_logger.remove(0)
        except Exception:
            pass

        # Configure stderr handler with the correct settings for JSON or text mode
        if self.json_logs:
            # For JSON output, don't provide a formatter function - let Loguru handle serialization
            loguru_logger.add(
                sys.stderr,
                level=self.log_level,
                format=self._format_record,  # No custom formatter for JSON mode
                serialize=True,  # Enable JSON serialization
                colorize=False,  # No color in JSON mode
                enqueue=True,
                backtrace=True,
                diagnose=True,
            )
        else:
            # For text output, use the formatter function
            loguru_logger.add(
                sys.stderr,
                level=self.log_level,
                format=self._format_record,
                serialize=False,  # Disable JSON serialization
                colorize=self.colorize,
                enqueue=True,
                backtrace=True,
                diagnose=True,
            )

        # Log startup configuration
        loguru_logger.debug(
            "Logger initialized",
            extra={
                "environment": self.env,
                "log_level": self.log_level,
                "json_logs": self.json_logs,
                "colorize": self.colorize,
            },
        )

    @property
    def logger(self):
        """Get the configured logger instance."""
        return loguru_logger


class InterceptHandler(logging.Handler):
    """Redirect FastAPI's built-in logger to Loguru."""

    def emit(self, record: Optional[logging.LogRecord]) -> None:
        if record is None:
            return

        loguru_level = record.levelname.upper()
        if record.exc_info is not None:
            logger.opt(exception=record.exc_info).log(loguru_level, record.getMessage())
        else:
            logger.log(loguru_level, record.getMessage())


# Initialize singleton and expose logger
_logger_config = LoggerConfig()
logger = _logger_config.logger
