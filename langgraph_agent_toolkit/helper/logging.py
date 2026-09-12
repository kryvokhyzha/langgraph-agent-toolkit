import logging
import os
import sys
from pathlib import Path

from loguru import logger as loguru_logger

from langgraph_agent_toolkit.helper.types import EnvironmentMode


class DuplicateFilter:
    """Filter duplicate log messages."""

    def __init__(self):
        self.msgs: set[str] = set()

    def __call__(self, record) -> bool:
        k = f"{record['level']}{record['message']}"
        if k in self.msgs:
            return False
        else:
            self.msgs.add(k)
            return True


class Formatter:
    """Format log messages by log level."""

    def __init__(self, debug: bool = False):
        """Initialize the formatter.

        Args:
            debug (bool): Use the detailed format with function and line data.

        """
        if debug:
            self.fmt = (
                "<green>{time:YYYY-MM-DD HH:mm:ss.SS}</green>"
                " | <level>{level: <8}</level>"
                " | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan>"
                " | <level>{message}</level>"
            )
        else:
            self.fmt = (
                "<green>{time:YYYY-MM-DD HH:mm:ss.SS}</green>"
                " | <level>{level: <8}</level>"
                " | <cyan>{module}</cyan>"
                " | <level>{message}</level>"
            )
        self.fmt += "\n{exception}"

    def format(self, record):
        """Format a log record.

        Args:
            record: Log record.

        Returns:
            str: Formatted log message.

        """
        if record["level"].no == LoggerConfig.WARN_ONCE_NO:
            return self.fmt.replace("{level: <8}", "WARNING ")
        return self.fmt


class SingletonMeta(type):
    """Create one `LoggerConfig` instance."""

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class LoggerConfig(metaclass=SingletonMeta):
    """Configure and manage the singleton logger."""

    WARN_ONCE_NO = 25
    DEPRECATED_NO = 26

    def __init__(self):
        self.env = EnvironmentMode(os.environ.get("ENV_MODE", EnvironmentMode.PRODUCTION))
        self.log_level = os.environ.get("LOG_LEVEL", "INFO" if self.env == EnvironmentMode.PRODUCTION else "DEBUG")
        self.json_logs = os.environ.get("JSON_LOGS", "false").lower() == "true"
        self.colorize = os.environ.get("COLORIZE", "true").lower() == "true"
        self._duplicate_filter = DuplicateFilter()
        self._logger_initialized = False

        self._setup_logger()

    def _setup_logger(self, log_file: str | None = None) -> None:
        """Configure the Loguru logger from environment settings.

        Args:
            log_file (Optional[str]): Optional path for a log file.

        """
        try:
            loguru_logger.remove()
        except Exception:
            pass

        is_debug = loguru_logger.level(self.log_level.upper()).no <= loguru_logger.level("DEBUG").no
        formatter = Formatter(debug=is_debug)

        if not self._logger_initialized:
            loguru_logger.level("WARNONCE", no=self.WARN_ONCE_NO, color="<yellow><bold>")
            loguru_logger.level("DEPRECATED", no=self.DEPRECATED_NO, color="<yellow><bold>")

            loguru_logger.add(
                sys.stderr,
                level=self.log_level.upper(),
                format=formatter.format,
                filter=lambda r: r["level"].no not in {self.WARN_ONCE_NO, self.DEPRECATED_NO},
                serialize=self.json_logs,
                colorize=self.colorize,
                enqueue=True,
                backtrace=True,
                diagnose=True,
            )

        loguru_logger.add(
            sys.stderr,
            level=max(loguru_logger.level(self.log_level.upper()).no, self.WARN_ONCE_NO),
            format=formatter.format,
            filter=lambda r: r["level"].no == self.WARN_ONCE_NO and self._duplicate_filter(r),
            colorize=self.colorize,
        )

        loguru_logger.add(
            sys.stderr,
            level=max(loguru_logger.level(self.log_level.upper()).no, self.DEPRECATED_NO),
            format=formatter.format,
            filter=lambda r: r["level"].no == self.DEPRECATED_NO and self._duplicate_filter(r),
            colorize=self.colorize,
        )

        if log_file is not None:
            log_dir = Path(log_file).parent
            log_dir.mkdir(parents=True, exist_ok=True)

            loguru_logger.add(
                log_file,
                level=self.log_level.upper(),
                format=formatter.format,
                filter=lambda r: r["level"].no != self.WARN_ONCE_NO,
                serialize=self.json_logs,
                rotation="10 MB",
                retention=5,
                compression="zip",
                enqueue=True,
            )

        if not self._logger_initialized:
            loguru_logger.debug(
                "Logger initialized",
                extra={
                    "environment": self.env,
                    "log_level": self.log_level,
                    "json_logs": self.json_logs,
                    "colorize": self.colorize,
                    "log_file": log_file,
                },
            )

        self._logger_initialized = True

    def configure_file_logging(self, log_file: str) -> None:
        """Add file logging to the logger configuration.

        Args:
            log_file (str): Log file path.

        """
        self._setup_logger(log_file=log_file)

    @property
    def logger(self):
        """Get the configured logger.

        Returns:
            loguru_logger: Configured logger.

        """
        return loguru_logger


def warn_once(message, *args, **kwargs):
    """Log a warning message once.

    Args:
        message (str): Warning message.
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    """
    try:
        logger.log("WARNONCE", message, *args, **kwargs)
    except ValueError:
        logger.warning(message, *args, **kwargs)


def log_deprecated(message, *args, **kwargs):
    """Log a deprecation warning.

    Args:
        message (str): Deprecation message.
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    """
    try:
        logger.log("DEPRECATED", message, *args, **kwargs)
    except ValueError:
        logger.warning(message, *args, **kwargs)


class InterceptHandler(logging.Handler):
    """Redirect FastAPI's built-in logger to Loguru."""

    def emit(self, record: logging.LogRecord | None) -> None:
        if record is None:
            return

        loguru_level = record.levelname.upper()
        if record.exc_info is not None:
            logger.opt(exception=record.exc_info).log(loguru_level, record.getMessage())
        else:
            logger.log(loguru_level, record.getMessage())


_logger_config = LoggerConfig()
logger = _logger_config.logger


def get_logger(log_file: str | None = None):
    """Get the configured logger and optionally add file logging.

    Args:
        log_file (Optional[str]): Optional log file path.

    Returns:
        logger: Configured logger.

    """
    if log_file is not None:
        _logger_config.configure_file_logging(log_file)
    return logger


__all__ = ["logger", "get_logger", "warn_once", "log_deprecated", "LoggerConfig", "InterceptHandler"]
