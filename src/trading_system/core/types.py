"""Core domain types.

Conventions enforced here:
    * Every timestamp is tz-aware UTC. A bar timestamp is the bar's OPEN time.
    * Prices and indicator values are ``float``; sizes and money are ``Decimal``.
    * Value objects are frozen — state transitions produce new instances.
"""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from enum import StrEnum
from typing import NewType

Price = NewType("Price", float)
Volume = NewType("Volume", float)


def ensure_utc(timestamp: datetime) -> datetime:
    """Normalise a tz-aware timestamp to UTC.

    Args:
        timestamp: Timestamp to normalise.

    Returns:
        The same instant expressed in UTC.

    Raises:
        ValueError: If ``timestamp`` is naive, since a naive value has no
            defined instant and silently assuming a zone corrupts alignment.
    """
    if timestamp.tzinfo is None:
        raise ValueError("timestamp must be tz-aware; naive datetimes are rejected")
    return timestamp.astimezone(UTC)


class Side(StrEnum):
    """Trade direction."""

    BUY = "BUY"
    SELL = "SELL"


class OrderType(StrEnum):
    """Order type."""

    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"


class Timeframe(StrEnum):
    """Trading timeframe."""

    M1 = "M1"
    M5 = "M5"
    M15 = "M15"
    H1 = "H1"
    H4 = "H4"
    D1 = "D1"

    @property
    def duration(self) -> timedelta:
        """Nominal wall-clock length of one bar.

        ``D1`` reports 24 hours. A calendar day spanning a DST transition is
        23 or 25 hours long, so callers doing day-boundary arithmetic must work
        in the session timezone rather than adding this value.
        """
        return _TIMEFRAME_DURATIONS[self]


_TIMEFRAME_DURATIONS: dict[Timeframe, timedelta] = {
    Timeframe.M1: timedelta(minutes=1),
    Timeframe.M5: timedelta(minutes=5),
    Timeframe.M15: timedelta(minutes=15),
    Timeframe.H1: timedelta(hours=1),
    Timeframe.H4: timedelta(hours=4),
    Timeframe.D1: timedelta(days=1),
}


@dataclass(frozen=True)
class Bar:
    """A single OHLCV bar.

    Attributes:
        timestamp: Bar OPEN time, tz-aware UTC. A bar is only considered closed
            once the clock has passed ``timestamp + timeframe``.
        open: Opening price.
        high: Highest traded price within the bar.
        low: Lowest traded price within the bar.
        close: Closing price.
        volume: Traded volume.
    """

    timestamp: datetime
    open: Price
    high: Price
    low: Price
    close: Price
    volume: Volume

    def __post_init__(self) -> None:
        """Normalise the timestamp and reject malformed OHLC relationships."""
        object.__setattr__(self, "timestamp", ensure_utc(self.timestamp))
        if self.high < self.low:
            raise ValueError(f"high {self.high} is below low {self.low}")
        if not self.low <= self.open <= self.high:
            raise ValueError(f"open {self.open} outside [{self.low}, {self.high}]")
        if not self.low <= self.close <= self.high:
            raise ValueError(f"close {self.close} outside [{self.low}, {self.high}]")
        if self.volume < 0:
            raise ValueError(f"volume must be non-negative, got {self.volume}")


@dataclass(frozen=True)
class Signal:
    """A strategy's directional idea, scored but unsized.

    A signal carries no notion of money. The Risk Engine derives position size
    downstream from ``quality`` and the distance between entry and
    ``invalidation_price``; the strategy never decides how much is at stake.

    Attributes:
        strategy_id: Identifier of the strategy that produced the signal.
        symbol: Instrument the signal applies to.
        bar_close_ts: Close time of the bar the signal was derived from (UTC).
            Execution is not allowed before open[t+1]. Note this differs from
            :attr:`Bar.timestamp`, which is a bar's OPEN time; the name is
            explicit so the two can never be confused at a module boundary.
        direction: Intended trade direction.
        quality: Confidence score in ``[0.0, 1.0]``.
        invalidation_price: Price at which the signal's thesis is void.
    """

    strategy_id: str
    symbol: str
    bar_close_ts: datetime
    direction: Side
    quality: float
    invalidation_price: Price

    def __post_init__(self) -> None:
        """Normalise the bar close time and bound-check the quality score."""
        object.__setattr__(self, "bar_close_ts", ensure_utc(self.bar_close_ts))
        if not 0.0 <= self.quality <= 1.0:
            raise ValueError(f"quality must be in [0.0, 1.0], got {self.quality}")


def can_execute_at(signal: Signal, execution_ts: datetime) -> bool:
    """Whether an order derived from ``signal`` may be filled at ``execution_ts``.

    A signal is knowable only once bar ``t`` has closed, and bar ``t+1`` opens at
    that same instant, so fills are permitted from ``bar_close_ts`` onwards.
    Filling any earlier lands inside the bar the signal came from, which is
    lookahead: the strategy would be trading on a close it could not yet have
    observed.

    Args:
        signal: The signal an order would originate from.
        execution_ts: Instant the fill would occur at. Must be tz-aware.

    Returns:
        ``True`` when the fill is at or after the signal's bar close.

    Raises:
        ValueError: If ``execution_ts`` is naive.
    """
    return ensure_utc(execution_ts) >= signal.bar_close_ts


@dataclass(frozen=True)
class Order:
    """An instruction handed to the execution layer.

    Attributes:
        symbol: Instrument to trade.
        side: Trade direction.
        size: Size in instrument units. ``Decimal`` because it feeds accounting.
        order_type: How the order should be filled.
        price: Limit or stop price; ``None`` for market orders.
    """

    symbol: str
    side: Side
    size: Decimal
    order_type: OrderType = OrderType.MARKET
    price: Price | None = None

    def __post_init__(self) -> None:
        """Reject non-positive sizes and prices missing for non-market orders."""
        if self.size <= 0:
            raise ValueError(f"order size must be positive, got {self.size}")
        if self.order_type is not OrderType.MARKET and self.price is None:
            raise ValueError(f"{self.order_type.value} order requires a price")


@dataclass(frozen=True)
class Position:
    """An open position snapshot.

    Attributes:
        symbol: Instrument held.
        side: Direction of the exposure.
        size: Size in instrument units.
        entry_price: Average fill price of the position.
        opened_at: When the position was opened, UTC.
    """

    symbol: str
    side: Side
    size: Decimal
    entry_price: Price
    opened_at: datetime

    def __post_init__(self) -> None:
        """Normalise the open timestamp and reject non-positive sizes."""
        object.__setattr__(self, "opened_at", ensure_utc(self.opened_at))
        if self.size <= 0:
            raise ValueError(f"position size must be positive, got {self.size}")
