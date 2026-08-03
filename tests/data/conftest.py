"""Fixtures for the data-layer tests."""

from datetime import UTC, datetime, timedelta
from typing import cast

import polars as pl
import pytest

from trading_system.core.types import Timeframe
from trading_system.data.models import TIMESTAMP_COLUMN, OHLCVFrame


def timestamp_zone(df: pl.DataFrame) -> str | None:
    """Read the timestamp column's timezone, narrowing polars' DataType union."""
    return cast(pl.Datetime, df.schema[TIMESTAMP_COLUMN]).time_zone


def make_frame(
    start: datetime,
    count: int,
    *,
    timeframe: Timeframe = Timeframe.M1,
    symbol: str = "EURUSD",
    step: timedelta | None = None,
) -> OHLCVFrame:
    """Build a synthetic frame whose values encode their own bar index.

    Bar ``i`` has ``open=i``, ``high=i+2``, ``low=i-2``, ``close=i+1`` and
    ``volume=1``. Encoding the index into the prices lets a resampling assertion
    state the exact expected OHLCV rather than a tolerance.
    """
    interval = step or timeframe.duration
    timestamps = [start + interval * i for i in range(count)]
    df = pl.DataFrame(
        {
            "timestamp": timestamps,
            "open": [float(i) for i in range(count)],
            "high": [float(i) + 2 for i in range(count)],
            "low": [float(i) - 2 for i in range(count)],
            "close": [float(i) + 1 for i in range(count)],
            "volume": [1.0] * count,
        }
    ).with_columns(pl.col("timestamp").dt.convert_time_zone("UTC"))
    return OHLCVFrame(df, symbol, timeframe)


@pytest.fixture
def minute_frame() -> OHLCVFrame:
    """Three and a half hours of 1m bars starting 2024-01-01 09:00 UTC."""
    return make_frame(datetime(2024, 1, 1, 9, 0, tzinfo=UTC), 210)
