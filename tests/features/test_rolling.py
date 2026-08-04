"""Incremental primitives: windows, monotonic extremes, seeded averages."""

import math

import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from trading_system.features.rolling import (
    RollingExtreme,
    RollingWindow,
    SeededEma,
    linear_weights,
    wilder_alpha,
)

VALUES = st.lists(
    st.floats(min_value=-1e4, max_value=1e4, allow_nan=False, allow_infinity=False),
    min_size=1,
    max_size=120,
)


def test_window_evicts_the_oldest_value_once_full() -> None:
    window = RollingWindow(3)
    assert window.push(1.0) is None
    assert window.push(2.0) is None
    assert window.push(3.0) is None
    assert window.full
    assert window.push(4.0) == 1.0
    assert window.values == (2.0, 3.0, 4.0)
    assert window.oldest == 2.0
    assert window.newest == 4.0


def test_window_aggregates_match_the_definitions() -> None:
    window = RollingWindow(4)
    for value in (1.0, 2.0, 3.0, 4.0):
        window.push(value)
    assert window.sum() == 10.0
    assert window.mean() == 2.5
    assert window.std(0) == pytest.approx(math.sqrt(1.25))
    assert window.std(1) == pytest.approx(math.sqrt(5 / 3))
    assert window.mean_absolute_deviation() == pytest.approx(1.0)
    assert window.weighted_mean((1.0, 2.0, 3.0, 4.0)) == pytest.approx(30 / 10)


def test_window_clear_empties_it() -> None:
    window = RollingWindow(2)
    window.push(1.0)
    window.clear()
    assert len(window) == 0
    assert not window.full


def test_window_rejects_a_non_positive_size() -> None:
    with pytest.raises(ValueError, match="positive"):
        RollingWindow(0)


def test_weighted_mean_rejects_a_mismatched_weight_count() -> None:
    window = RollingWindow(3)
    window.push(1.0)
    with pytest.raises(ValueError, match="weights"):
        window.weighted_mean((1.0, 2.0, 3.0))


@given(values=VALUES, size=st.integers(min_value=1, max_value=12))
@settings(max_examples=60, deadline=None)
def test_rolling_extreme_matches_a_brute_force_scan(values: list[float], size: int) -> None:
    """The monotonic deque must agree with recomputing max/min over the window."""
    largest = RollingExtreme(size, largest=True)
    smallest = RollingExtreme(size, largest=False)
    for index, value in enumerate(values):
        high = largest.push(value)
        low = smallest.push(value)
        if index + 1 < size:
            assert high is None
            assert low is None
        else:
            window = values[index - size + 1 : index + 1]
            assert high == max(window)
            assert low == min(window)


def test_rolling_extreme_clear_restarts_the_window() -> None:
    extreme = RollingExtreme(2, largest=True)
    extreme.push(5.0)
    extreme.push(1.0)
    extreme.clear()
    assert extreme.push(2.0) is None


@given(values=VALUES, period=st.integers(min_value=1, max_value=15))
@settings(max_examples=60, deadline=None)
def test_seeded_ema_matches_polars_ewm_with_an_sma_seed(values: list[float], period: int) -> None:
    """The incremental average must reproduce what the vectorised helper computes.

    Both are compared against polars directly rather than against each other, so
    a shared misunderstanding of the seeding rule cannot pass.
    """
    alpha = 2.0 / (period + 1)
    average = SeededEma(period, alpha)
    incremental = [average.push(value) for value in values]

    series = pl.Series("x", values, dtype=pl.Float64)
    sma = series.rolling_mean(period)
    index = pl.int_range(0, len(values), eager=True)
    seeded = pl.select(
        pl.when(pl.lit(index) < period - 1)
        .then(None)
        .when(pl.lit(index) == period - 1)
        .then(pl.lit(sma))
        .otherwise(pl.lit(series))
        .alias("z")
    ).to_series()
    expected = seeded.ewm_mean(alpha=alpha, adjust=False, ignore_nulls=False).to_list()

    assert len(incremental) == len(expected)
    for got, want in zip(incremental, expected, strict=True):
        if want is None:
            assert got is None
        else:
            assert got is not None
            assert got == pytest.approx(want, abs=1e-12)


def test_seeded_ema_starts_from_the_mean_not_the_first_value() -> None:
    average = SeededEma(3, 0.5)
    assert average.push(1.0) is None
    assert average.push(2.0) is None
    assert average.push(3.0) == pytest.approx(2.0)
    assert average.push(4.0) == pytest.approx(3.0)
    assert average.value == pytest.approx(3.0)


def test_seeded_ema_clear_returns_it_to_seeding() -> None:
    average = SeededEma(2)
    average.push(1.0)
    average.push(3.0)
    average.clear()
    assert average.value is None
    assert average.push(5.0) is None


@pytest.mark.parametrize(("period", "alpha"), [(0, None), (3, 0.0), (3, 1.5)])
def test_seeded_ema_rejects_impossible_parameters(period: int, alpha: float | None) -> None:
    with pytest.raises(ValueError, match="must be"):
        SeededEma(period, alpha)


def test_helper_constants() -> None:
    assert wilder_alpha(14) == pytest.approx(1 / 14)
    assert linear_weights(4) == (1.0, 2.0, 3.0, 4.0)
