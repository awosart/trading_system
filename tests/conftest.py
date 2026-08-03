"""Shared fixtures."""

from datetime import UTC, datetime

import pytest

from trading_system.core.types import Bar, Price, Volume


@pytest.fixture
def bar_open_time() -> datetime:
    """Reference bar OPEN time, tz-aware UTC."""
    return datetime(2024, 1, 1, tzinfo=UTC)


@pytest.fixture
def sample_bar(bar_open_time: datetime) -> Bar:
    """A well-formed OHLCV bar."""
    return Bar(
        timestamp=bar_open_time,
        open=Price(1.0),
        high=Price(1.1),
        low=Price(0.9),
        close=Price(1.05),
        volume=Volume(1000.0),
    )
