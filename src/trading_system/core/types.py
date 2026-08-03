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
        timestamp: Close time of the bar the signal was derived from, UTC.
        direction: Intended trade direction.
        quality: Confidence score in ``[0.0, 1.0]``.
        invalidation_price: Price at which the signal's thesis is void.
    """

    strategy_id: str
    symbol: str
    timestamp: datetime
    direction: Side
    quality: float
    invalidation_price: Price

    def __post_init__(self) -> None:
        """Normalise the timestamp and bound-check the quality score."""
        object.__setattr__(self, "timestamp", ensure_utc(self.timestamp))
        if not 0.0 <= self.quality <= 1.0:
            raise ValueError(f"quality must be in [0.0, 1.0], got {self.quality}")


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
