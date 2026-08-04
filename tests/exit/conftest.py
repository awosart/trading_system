"""Fixtures for the Exit Engine tests.

Bars are written out as explicit OHLC tuples rather than derived from closes.
Exit rules are triggered by highs and lows, and the gap cases — a bar that opens
beyond a level it never brackets — are exactly the ones a close-only helper
cannot express.
"""

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from decimal import Decimal

import polars as pl

from trading_system.core.types import Price, Side, Timeframe
from trading_system.data.models import OHLCVFrame
from trading_system.entry.context import PRICE_FIELDS, BarSeries
from trading_system.exit.context import Tick
from trading_system.exit.position import ManagedPosition

SYMBOL = "TESTFX"
START = datetime(2024, 1, 2, 9, 0, tzinfo=UTC)
TIMEFRAME = Timeframe.M15

#: One bar as ``(open, high, low, close)``.
type Bar = tuple[float, float, float, float]


def bar_open_ts(index: int) -> datetime:
    """Open time of the bar at ``index``.

    Args:
        index: Bar position in the series.

    Returns:
        The bar's open timestamp.
    """
    return START + TIMEFRAME.duration * index


def bar_close_ts(index: int) -> datetime:
    """Close time of the bar at ``index``.

    Args:
        index: Bar position in the series.

    Returns:
        The bar's close timestamp, which is the next bar's open.
    """
    return bar_open_ts(index) + TIMEFRAME.duration


def series(
    bars: Sequence[Bar],
    symbol: str = SYMBOL,
    *,
    features: Mapping[str, Sequence[float | None]] | None = None,
) -> BarSeries:
    """Build a bar series from explicit OHLC tuples.

    Args:
        bars: One ``(open, high, low, close)`` per bar.
        symbol: Instrument identifier.
        features: Hand-written feature columns, aligned with ``bars``. Written
            by hand rather than computed, so a rule test can state the exact
            ATR/swing/MA values it means — including the warmup nulls a real
            indicator would produce.

    Returns:
        The series.
    """
    frame = OHLCVFrame.from_raw(
        pl.DataFrame(
            {
                "timestamp": [bar_open_ts(index) for index in range(len(bars))],
                "open": [bar[0] for bar in bars],
                "high": [bar[1] for bar in bars],
                "low": [bar[2] for bar in bars],
                "close": [bar[3] for bar in bars],
                "volume": [1_000.0] * len(bars),
            }
        ),
        symbol,
        TIMEFRAME,
    )
    return BarSeries(
        symbol=frame.symbol,
        timeframe=frame.timeframe,
        timestamps=frame.df["timestamp"].to_list(),
        prices={field: frame.df[field].to_list() for field in PRICE_FIELDS},
        features=features or {},
    )


def ticks(index: int, prices: Sequence[float]) -> list[Tick]:
    """Ticks spread evenly across one bar.

    Args:
        index: Bar the ticks belong to.
        prices: Traded prices, in the order they printed.

    Returns:
        One tick per price, ascending in time and inside the bar.
    """
    step = TIMEFRAME.duration / (len(prices) + 1)
    return [
        Tick(ts=bar_open_ts(index) + step * (offset + 1), price=price)
        for offset, price in enumerate(prices)
    ]


def long_position(
    *,
    entry: float = 1.1000,
    stop: float = 1.0900,
    size: Decimal = Decimal("1"),
    opened_at: datetime | None = None,
) -> ManagedPosition:
    """A long with one R spanning ``entry - stop``.

    Args:
        entry: Entry fill price.
        stop: Initial protective level, below ``entry``.
        size: Size in instrument units.
        opened_at: Fill time; defaults to the first bar's open.

    Returns:
        The position.
    """
    return ManagedPosition(
        symbol=SYMBOL,
        side=Side.BUY,
        entry_price=Price(entry),
        size=size,
        initial_stop=Price(stop),
        opened_at=opened_at or START,
        strategy_id="test-strategy",
    )


def short_position(
    *,
    entry: float = 1.1000,
    stop: float = 1.1100,
    size: Decimal = Decimal("1"),
    opened_at: datetime | None = None,
) -> ManagedPosition:
    """A short with one R spanning ``stop - entry``.

    Args:
        entry: Entry fill price.
        stop: Initial protective level, above ``entry``.
        size: Size in instrument units.
        opened_at: Fill time; defaults to the first bar's open.

    Returns:
        The position.
    """
    return ManagedPosition(
        symbol=SYMBOL,
        side=Side.SELL,
        entry_price=Price(entry),
        size=size,
        initial_stop=Price(stop),
        opened_at=opened_at or START,
        strategy_id="test-strategy",
    )
