"""Structured logging setup.

The console receives human-readable output; the optional log file receives
line-delimited JSON suitable for machine parsing. Both share one processor
chain, so a field added to a log call appears identically in both sinks.
"""

import logging
import sys
from pathlib import Path

import structlog
from structlog.typing import Processor

_SHARED_PROCESSORS: list[Processor] = [
    structlog.contextvars.merge_contextvars,
    structlog.stdlib.add_log_level,
    structlog.stdlib.add_logger_name,
    structlog.processors.TimeStamper(fmt="iso", utc=True),
    structlog.processors.StackInfoRenderer(),
    structlog.processors.format_exc_info,
]


def setup_logging(level: str = "INFO", log_file: Path | None = None) -> None:
    """Configure structlog and the stdlib root logger for this process.

    Calling this more than once replaces the previously installed handlers
    rather than stacking duplicates.

    Args:
        level: Threshold name for the root logger, e.g. ``"INFO"``.
        log_file: Path for JSON logs. Parent directories are created as needed.
            ``None`` leaves the console as the only sink.
    """
    structlog.configure(
        processors=[*_SHARED_PROCESSORS, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=_SHARED_PROCESSORS,
            processor=structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty()),
        )
    )
    handlers: list[logging.Handler] = [console]

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(
            structlog.stdlib.ProcessorFormatter(
                foreign_pre_chain=_SHARED_PROCESSORS,
                processor=structlog.processors.JSONRenderer(),
            )
        )
        handlers.append(file_handler)

    root = logging.getLogger()
    for installed in list(root.handlers):
        root.removeHandler(installed)
        installed.close()
    for handler in handlers:
        root.addHandler(handler)
    root.setLevel(level.upper())


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a logger bound to ``name``.

    Args:
        name: Logger name, conventionally the calling module's ``__name__``.

    Returns:
        A structlog logger writing through the stdlib handlers.
    """
    return structlog.stdlib.get_logger(name)
