"""Core domain type invariants."""

from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal

import pytest

from trading_system.core.types import (
    Bar,
    Order,
    OrderType,
    Position,
    Price,
    Side,
    Signal,
    Timeframe,
    Volume,
    ensure_utc,
)


def test_bar_creation(sample_bar: Bar) -> None:
    assert sample_bar.close == 1.05
    assert sample_bar.timestamp == datetime(2024, 1, 1, tzinfo=UTC)


def test_bar_normalises_non_utc_timestamp_to_utc() -> None:
    moscow = timezone(timedelta(hours=3))
    bar = Bar(
        timestamp=datetime(2024, 1, 1, 3, 0, tzinfo=moscow),
        open=Price(1.0),
        high=Price(1.1),
        low=Price(0.9),
        close=Price(1.05),
        volume=Volume(1000.0),
    )
    assert bar.timestamp == datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
    assert bar.timestamp.tzinfo is UTC


def test_bar_rejects_naive_timestamp() -> None:
    with pytest.raises(ValueError, match="tz-aware"):
        Bar(
            timestamp=datetime(2024, 1, 1),  # noqa: DTZ001 - deliberately naive
            open=Price(1.0),
            high=Price(1.1),
            low=Price(0.9),
            close=Price(1.05),
            volume=Volume(1000.0),
        )


@pytest.mark.parametrize(
    ("open_", "high", "low", "close", "match"),
    [
        (1.0, 0.9, 1.1, 1.0, "below low"),
        (1.5, 1.1, 0.9, 1.0, r"open .* outside"),
        (1.0, 1.1, 0.9, 1.5, r"close .* outside"),
    ],
)
def test_bar_rejects_inconsistent_ohlc(
    bar_open_time: datetime,
    open_: float,
    high: float,
    low: float,
    close: float,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        Bar(
            timestamp=bar_open_time,
            open=Price(open_),
            high=Price(high),
            low=Price(low),
            close=Price(close),
            volume=Volume(1.0),
        )


def test_bar_rejects_negative_volume(bar_open_time: datetime) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        Bar(
            timestamp=bar_open_time,
            open=Price(1.0),
            high=Price(1.1),
            low=Price(0.9),
            close=Price(1.05),
            volume=Volume(-1.0),
        )


def test_bar_is_frozen(sample_bar: Bar) -> None:
    with pytest.raises(AttributeError):
        sample_bar.close = Price(2.0)  # type: ignore[misc]


def test_ensure_utc_rejects_naive() -> None:
    with pytest.raises(ValueError, match="tz-aware"):
        ensure_utc(datetime(2024, 1, 1))  # noqa: DTZ001 - deliberately naive


def test_side_enum() -> None:
    assert Side.BUY.value == "BUY"
    assert Side.SELL.value == "SELL"


def test_timeframe_and_order_type_enums() -> None:
    assert Timeframe.H1.value == "H1"
    assert OrderType.MARKET.value == "MARKET"


def test_signal_accepts_boundary_quality(bar_open_time: datetime) -> None:
    for quality in (0.0, 1.0):
        signal = Signal(
            strategy_id="s1",
            symbol="EURUSD",
            timestamp=bar_open_time,
            direction=Side.BUY,
            quality=quality,
            invalidation_price=Price(0.95),
        )
        assert signal.quality == quality


@pytest.mark.parametrize("quality", [-0.01, 1.01])
def test_signal_rejects_quality_outside_unit_interval(
    bar_open_time: datetime, quality: float
) -> None:
    with pytest.raises(ValueError, match=r"quality must be in"):
        Signal(
            strategy_id="s1",
            symbol="EURUSD",
            timestamp=bar_open_time,
            direction=Side.BUY,
            quality=quality,
            invalidation_price=Price(0.95),
        )


def test_order_defaults_to_market_without_price() -> None:
    order = Order(symbol="EURUSD", side=Side.BUY, size=Decimal("1.5"))
    assert order.order_type is OrderType.MARKET
    assert order.price is None


def test_order_rejects_non_positive_size() -> None:
    with pytest.raises(ValueError, match="size must be positive"):
        Order(symbol="EURUSD", side=Side.BUY, size=Decimal("0"))


def test_limit_order_requires_price() -> None:
    with pytest.raises(ValueError, match="LIMIT order requires a price"):
        Order(
            symbol="EURUSD",
            side=Side.BUY,
            size=Decimal("1"),
            order_type=OrderType.LIMIT,
        )


def test_position_normalises_opened_at(bar_open_time: datetime) -> None:
    position = Position(
        symbol="EURUSD",
        side=Side.BUY,
        size=Decimal("2"),
        entry_price=Price(1.0),
        opened_at=bar_open_time.astimezone(timezone(timedelta(hours=-5))),
    )
    assert position.opened_at == bar_open_time
