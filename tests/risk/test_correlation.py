"""Correlation: measured as of an instant, never as a standing fact."""

import math
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from trading_system.core.types import Timeframe
from trading_system.data.models import OHLCVFrame
from trading_system.data.resample import DayOrigin
from trading_system.risk.correlation import (
    CorrelationConfig,
    CorrelationProvider,
    daily_returns,
)

START = datetime(2024, 1, 1, tzinfo=UTC)
#: UTC-anchored days, so a test's "day N" is unambiguous.
UTC_DAYS = DayOrigin(tz="UTC")


def series(closes: list[float], *, symbol: str = "A", start: datetime = START) -> OHLCVFrame:
    """Build a daily series with the given closes, one bar per day."""
    rows = [
        (start + timedelta(days=i), close, close, close, close, 0.0)
        for i, close in enumerate(closes)
    ]
    return OHLCVFrame.from_raw(
        pl.DataFrame(
            rows,
            schema=["timestamp", "open", "high", "low", "close", "volume"],
            orient="row",
        ),
        symbol,
        Timeframe.D1,
    )


def config(**overrides: object) -> CorrelationConfig:
    """Config with a short window so tests need few bars."""
    base: dict[str, object] = {"window": 20, "min_periods": 5, "day_origin": UTC_DAYS}
    return CorrelationConfig(**(base | overrides))  # type: ignore[arg-type]


def after_day(index: int) -> datetime:
    """An instant just after the close of the bar at ``index``."""
    return START + timedelta(days=index + 1)


class TestNoLookahead:
    """The property the whole module is arranged around."""

    def test_a_matrix_cannot_be_obtained_without_an_instant(self) -> None:
        # as_of is keyword-only and has no default, so "the correlation matrix"
        # is not a thing that exists detached from a moment.
        provider = CorrelationProvider({"A": series([1.0])}, config=config())
        with pytest.raises(TypeError):
            provider.matrix()  # type: ignore[call-arg]

    @settings(max_examples=50, deadline=None)
    @given(cut=st.integers(min_value=8, max_value=25))
    def test_truncating_the_future_changes_nothing(self, cut: int) -> None:
        # The P06 BarContext test, applied here. A matrix measured at t over the
        # full history must equal one measured at t over history that stops at
        # t. Any bar from after t reaching the window separates them, and no
        # amount of docstring can substitute for this.
        closes_a = [100.0 * math.exp(0.01 * math.sin(i)) for i in range(30)]
        closes_b = [50.0 * math.exp(0.01 * math.sin(i + 0.3)) for i in range(30)]
        as_of = after_day(cut)

        full = CorrelationProvider(
            {"A": series(closes_a, symbol="A"), "B": series(closes_b, symbol="B")},
            config=config(),
        ).matrix(as_of=as_of)
        truncated = CorrelationProvider(
            {
                "A": series(closes_a[: cut + 1], symbol="A"),
                "B": series(closes_b[: cut + 1], symbol="B"),
            },
            config=config(),
        ).matrix(as_of=as_of)

        assert full.values == truncated.values
        assert full.observations == truncated.observations

    def test_a_bar_that_has_not_closed_is_excluded(self) -> None:
        # Day 5's bar opens at day 5 and closes at day 6. Asked at day 5 12:00,
        # its close is not knowable.
        frame = series([100.0 + i for i in range(8)])
        mid_bar = START + timedelta(days=5, hours=12)
        returns = daily_returns(frame, as_of=mid_bar, config=config())
        assert max(returns) == (START + timedelta(days=4)).date()

    def test_the_cache_is_keyed_by_day_not_held_as_current(self) -> None:
        # Two different instants must not be able to return one another's
        # matrix. A single mutable "current matrix" slot is exactly the leak.
        closes_a = [100.0 * (1.0 + 0.01 * i) for i in range(30)]
        closes_b = [50.0 * (1.0 - 0.01 * i) for i in range(30)]
        provider = CorrelationProvider(
            {"A": series(closes_a, symbol="A"), "B": series(closes_b, symbol="B")},
            config=config(),
        )
        early = provider.matrix(as_of=after_day(9))
        late = provider.matrix(as_of=after_day(25))
        assert early.as_of != late.as_of
        assert early.observations[("A", "B")] < late.observations[("A", "B")]
        # And re-asking the earlier instant still gets the earlier answer.
        assert provider.matrix(as_of=after_day(9)).observations == early.observations


class TestMeasurement:
    def test_two_identical_series_correlate_perfectly(self) -> None:
        closes = [100.0 * math.exp(0.01 * math.sin(i)) for i in range(20)]
        provider = CorrelationProvider(
            {"A": series(closes, symbol="A"), "B": series(closes, symbol="B")},
            config=config(),
        )
        matrix = provider.matrix(as_of=after_day(19))
        assert matrix.get("A", "B") == pytest.approx(1.0)

    def test_mirrored_series_correlate_at_minus_one(self) -> None:
        closes_a = [100.0 * math.exp(0.01 * math.sin(i)) for i in range(20)]
        closes_b = [100.0 * math.exp(-0.01 * math.sin(i)) for i in range(20)]
        provider = CorrelationProvider(
            {"A": series(closes_a, symbol="A"), "B": series(closes_b, symbol="B")},
            config=config(),
        )
        assert provider.matrix(as_of=after_day(19)).get("A", "B") == pytest.approx(-1.0)

    def test_an_instrument_correlates_perfectly_with_itself(self) -> None:
        provider = CorrelationProvider({"A": series([1.0, 2.0])}, config=config())
        assert provider.matrix(as_of=after_day(1)).get("A", "A") == 1.0

    def test_the_matrix_is_symmetric(self) -> None:
        closes_a = [100.0 * math.exp(0.01 * math.sin(i)) for i in range(20)]
        closes_b = [50.0 * math.exp(0.01 * math.cos(i)) for i in range(20)]
        provider = CorrelationProvider(
            {"A": series(closes_a, symbol="A"), "B": series(closes_b, symbol="B")},
            config=config(),
        )
        matrix = provider.matrix(as_of=after_day(19))
        assert matrix.get("A", "B") == matrix.get("B", "A")

    def test_only_the_trailing_window_is_measured(self) -> None:
        closes = [100.0 + i for i in range(40)]
        provider = CorrelationProvider(
            {"A": series(closes, symbol="A"), "B": series(closes, symbol="B")},
            config=config(window=10, min_periods=5),
        )
        matrix = provider.matrix(as_of=after_day(39))
        assert matrix.observations[("A", "B")] == 10


class TestInsufficientHistory:
    """A missing measurement is absent, never a zero."""

    def test_a_pair_below_min_periods_is_absent_not_zero(self) -> None:
        # Zero would read as "independent", which is the most permissive value
        # available. Absence forces the caller to fall back to its prior.
        closes = [100.0 + i for i in range(4)]
        provider = CorrelationProvider(
            {"A": series(closes, symbol="A"), "B": series(closes, symbol="B")},
            config=config(min_periods=10, window=20),
        )
        matrix = provider.matrix(as_of=after_day(3))
        assert matrix.get("A", "B") is None
        assert ("A", "B") not in matrix.values

    def test_overlap_is_counted_pairwise_not_per_series(self) -> None:
        # B trades every day; A only the first few. The pair has as much history
        # as the shorter of the two, not as much as the longer.
        long_series = series([100.0 + i for i in range(30)], symbol="B")
        short_series = series([50.0 + i for i in range(6)], symbol="A")
        provider = CorrelationProvider(
            {"A": short_series, "B": long_series}, config=config(min_periods=10)
        )
        assert provider.matrix(as_of=after_day(29)).get("A", "B") is None

    def test_a_flat_series_carries_no_information_and_is_absent(self) -> None:
        # Zero variance means the sample correlation is undefined, not zero.
        flat = series([100.0] * 20, symbol="A")
        moving = series([100.0 + i for i in range(20)], symbol="B")
        provider = CorrelationProvider({"A": flat, "B": moving}, config=config())
        assert provider.matrix(as_of=after_day(19)).get("A", "B") is None


class TestConfig:
    def test_a_minimum_the_window_cannot_reach_is_rejected(self) -> None:
        # It would disable correlation permanently and silently.
        with pytest.raises(ValueError, match="exceeds window"):
            CorrelationConfig(window=10, min_periods=20)

    def test_returns_are_logarithmic(self) -> None:
        frame = series([100.0, 110.0])
        returns = daily_returns(frame, as_of=after_day(1), config=config())
        assert returns[(START + timedelta(days=1)).date()] == pytest.approx(math.log(1.1))
