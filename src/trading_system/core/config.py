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
        instruments_path: Contract specifications the Risk Engine sizes against,
            loaded by :func:`~trading_system.core.instruments.load_instruments`.
            A setting rather than a constant because tick sizes, lot bounds and
            swap rates are broker-specific: two brokers' EURUSD are two files,
            and hardcoding one of them would silently size against the wrong one.
        runs_dir: Root directory backtest and walk-forward runs are stored
            under, by
            :func:`~trading_system.backtest.reproducibility.write_run` and
            :class:`~trading_system.validation.walkforward.WalkForwardRunner`
            alike. A setting rather than a literal ``"runs"`` in the CLI —
            the same reasoning as ``data_dir``.
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
    instruments_path: Path = Path("configs/instruments.yaml")
    runs_dir: Path = Path("runs")
