"""Application configuration.

Configuration is a pydantic model, never a raw dict, so an invalid value fails
at process start instead of halfway through a backtest.

There is deliberately no module-level singleton: a :class:`Settings` instance is
constructed once at the entry point and passed explicitly to the components that
need it.
"""

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class Settings(BaseSettings):
    """Process-wide settings, read from the environment or a ``.env`` file.

    Environment variables are prefixed with ``TS_``; for example ``TS_LOG_LEVEL``
    populates :attr:`log_level`.

    Attributes:
        debug: Enable developer-facing behaviour such as verbose tracebacks.
        log_level: Threshold for the root logger.
        data_dir: Root directory for local parquet market data.
        log_file: Destination for JSON logs. ``None`` disables file logging.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="TS_",
        case_sensitive=False,
        extra="ignore",
    )

    debug: bool = False
    log_level: LogLevel = "INFO"
    data_dir: Path = Path("data")
    log_file: Path | None = None
