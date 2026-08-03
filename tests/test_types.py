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
    can_execute_at,
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
            bar_close_ts=bar_open_time,
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
            bar_close_ts=bar_open_time,
            direction=Side.BUY,
            quality=quality,
            invalidation_price=Price(0.95),
        )


class TestNoLookaheadContract:
    """A signal derived from bar t must not be executable on bar t.

    Bar timestamps are OPEN times; a signal's ``bar_close_ts`` is a CLOSE time.
    Bar t therefore spans ``[timestamp, timestamp + duration)`` and is only
    knowable at its end — which is exactly when bar t+1 opens. Every assertion
    below pins one side of that boundary.
    """

    timeframe = Timeframe.M1
    bar_t_open = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)

    def bar_t(self) -> Bar:
        """The bar the signal is derived from."""
        return Bar(
            timestamp=self.bar_t_open,
            open=Price(1.0),
            high=Price(1.1),
            low=Price(0.9),
            close=Price(1.05),
            volume=Volume(100.0),
        )

    def bar_t_plus_1(self) -> Bar:
        """The next bar, the earliest one an order may be filled on."""
        return Bar(
            timestamp=self.bar_t_open + self.timeframe.duration,
            open=Price(1.05),
            high=Price(1.2),
            low=Price(1.0),
            close=Price(1.15),
            volume=Volume(120.0),
        )

    def signal_from_bar_t(self) -> Signal:
        """A signal produced by observing bar t's close."""
        return Signal(
            strategy_id="s1",
            symbol="EURUSD",
            bar_close_ts=self.bar_t_open + self.timeframe.duration,
            direction=Side.BUY,
            quality=0.8,
            invalidation_price=Price(0.95),
        )

    def test_bar_close_is_the_next_bar_open(self) -> None:
        """The two timestamps name the same instant, read from opposite sides."""
        assert self.signal_from_bar_t().bar_close_ts == self.bar_t_plus_1().timestamp

    def test_signal_cannot_be_executed_at_bar_t_open(self) -> None:
        """Filling at bar t's open would trade on a close not yet observed."""
        assert not can_execute_at(self.signal_from_bar_t(), self.bar_t().timestamp)

    @pytest.mark.parametrize("seconds_into_bar", [1, 30, 59])
    def test_signal_cannot_be_executed_inside_bar_t(self, seconds_into_bar: int) -> None:
        """No instant strictly inside bar t is a legal fill."""
        moment = self.bar_t_open + timedelta(seconds=seconds_into_bar)
        assert moment < self.signal_from_bar_t().bar_close_ts
        assert not can_execute_at(self.signal_from_bar_t(), moment)

    def test_signal_cannot_be_executed_one_microsecond_early(self) -> None:
        """The boundary is exact, not approximate."""
        just_before = self.signal_from_bar_t().bar_close_ts - timedelta(microseconds=1)
        assert not can_execute_at(self.signal_from_bar_t(), just_before)

    def test_signal_may_be_executed_at_bar_t_plus_1_open(self) -> None:
        """open[t+1] is the earliest permitted fill."""
        assert can_execute_at(self.signal_from_bar_t(), self.bar_t_plus_1().timestamp)

    def test_later_execution_remains_allowed(self) -> None:
        """Delaying a fill is legal; only filling early is not."""
        later = self.bar_t_plus_1().timestamp + timedelta(hours=3)
        assert can_execute_at(self.signal_from_bar_t(), later)

    def test_execution_time_is_compared_across_timezones(self) -> None:
        """A non-UTC execution time is judged on its instant, not its wall clock."""
        tokyo = timezone(timedelta(hours=9))
        legal = self.bar_t_plus_1().timestamp.astimezone(tokyo)
        assert can_execute_at(self.signal_from_bar_t(), legal)
        assert not can_execute_at(self.signal_from_bar_t(), legal - timedelta(microseconds=1))

    def test_naive_execution_time_is_rejected(self) -> None:
        """A naive instant has no defined position relative to the bar close."""
        with pytest.raises(ValueError, match="tz-aware"):
            can_execute_at(self.signal_from_bar_t(), datetime(2024, 1, 1, 12, 1))  # noqa: DTZ001


def test_signal_bar_close_ts_is_normalised_to_utc() -> None:
    """A signal built with a non-UTC close time stores the same instant in UTC."""
    tokyo = timezone(timedelta(hours=9))
    signal = Signal(
        strategy_id="s1",
        symbol="EURUSD",
        bar_close_ts=datetime(2024, 1, 1, 21, 0, tzinfo=tokyo),
        direction=Side.BUY,
        quality=0.5,
        invalidation_price=Price(0.95),
    )
    assert signal.bar_close_ts == datetime(2024, 1, 1, 12, 0, tzinfo=UTC)


def test_signal_rejects_naive_bar_close_ts() -> None:
    with pytest.raises(ValueError, match="tz-aware"):
        Signal(
            strategy_id="s1",
            symbol="EURUSD",
            bar_close_ts=datetime(2024, 1, 1, 12, 0),  # noqa: DTZ001
            direction=Side.BUY,
            quality=0.5,
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
