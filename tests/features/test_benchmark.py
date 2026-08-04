"""Performance budgets on million-bar frames.

Deselected by default — a million bars is slower to build than the entire rest
of the suite is to run. Execute with ``pytest -m benchmark``.

The budget exists because a research loop evaluates a feature set thousands of
times across a walk-forward grid. An indicator that takes two seconds instead of
two hundred milliseconds does not cost two seconds; it costs the difference
between running the study and not bothering.
"""

import time

import polars as pl
import pytest

from trading_system.core.types import Timeframe
from trading_system.data.models import OHLCVFrame
from trading_system.features.registry import build_indicator, default_indicators

BARS = 1_000_000

#: Twenty indicators spanning all five families, as a realistic feature set
#: would. Supertrend is included deliberately: its ratcheting bands have no
#: vectorised form, so it exercises the Python-level recursive path rather than
#: letting the budget be met entirely by polars expressions.
BENCHMARK_SET: list[tuple[str, dict[str, object]]] = [
    ("sma", {"period": 20}),
    ("sma", {"period": 50}),
    ("ema", {"period": 20}),
    ("ema", {"period": 50}),
    ("wma", {"period": 20}),
    ("hma", {"period": 20}),
    ("vwma", {"period": 20}),
    ("macd", {}),
    ("adx", {"period": 14}),
    ("supertrend", {}),
    ("ichimoku", {}),
    ("donchian", {"period": 20}),
    ("atr", {"period": 14}),
    ("bbands", {}),
    ("keltner", {}),
    ("chop", {"period": 14}),
    ("rsi", {"period": 14}),
    ("stoch", {}),
    ("cci", {"period": 20}),
    ("willr", {"period": 14}),
]

#: Definition-of-done budget for :data:`BENCHMARK_SET` over :data:`BARS` bars.
BUDGET_SECONDS = 5.0

#: Budget for the entire registry, which additionally runs the two fractal
#: indicators. Both walk the bars in Python to keep pivot confirmation exact,
#: and cost roughly 1.4 microseconds per bar each.
FULL_REGISTRY_BUDGET_SECONDS = 10.0


def synthetic_frame(bars: int) -> OHLCVFrame:
    """Build a large frame without leaving polars.

    A Python loop over a million bars would dominate the measurement it is meant
    to support, so the random walk is generated with a hash-like sine expression
    and a cumulative sum. The series is deterministic, which keeps the benchmark
    comparable between runs.

    Args:
        bars: Number of bars to generate.

    Returns:
        A frame of one-minute bars.
    """
    index = pl.int_range(0, bars).cast(pl.Float64)
    fractional = (index * 127.1 + 311.7).sin() * 43758.5453
    df = (
        pl.select(
            pl.datetime_range(
                pl.datetime(2020, 1, 1),
                pl.datetime(2020, 1, 1) + pl.duration(minutes=bars - 1),
                interval="1m",
                time_zone="UTC",
            ).alias("timestamp"),
            (fractional - fractional.floor()).alias("noise"),
        )
        .with_columns((1.1 + ((pl.col("noise") - 0.5) * 0.0004).cum_sum()).alias("close"))
        .with_columns(
            pl.col("close").shift(1).fill_null(1.1).alias("open"),
            (pl.col("noise") * 0.0002).alias("wick"),
            (pl.col("noise") * 2000).alias("volume"),
        )
        .with_columns(
            (pl.max_horizontal("open", "close") + pl.col("wick")).alias("high"),
            (pl.min_horizontal("open", "close") - pl.col("wick")).alias("low"),
        )
        .select("timestamp", "open", "high", "low", "close", "volume")
    )
    return OHLCVFrame.from_raw(df, "BENCH", Timeframe.M1)


@pytest.fixture(scope="module")
def million_bars() -> OHLCVFrame:
    """One million one-minute bars. Built once; construction is not timed."""
    return synthetic_frame(BARS)


@pytest.mark.benchmark
def test_twenty_indicators_over_a_million_bars_fit_the_budget(
    million_bars: OHLCVFrame,
) -> None:
    indicators = [build_indicator(kind, params) for kind, params in BENCHMARK_SET]
    assert len(indicators) == 20

    timings: dict[str, float] = {}
    started = time.perf_counter()
    for indicator in indicators:
        indicator_started = time.perf_counter()
        indicator.compute_frame(million_bars)
        timings[indicator.name] = time.perf_counter() - indicator_started
    elapsed = time.perf_counter() - started

    slowest = sorted(timings.items(), key=lambda item: -item[1])[:5]
    assert elapsed < BUDGET_SECONDS, (
        f"20 indicators over {BARS} bars took {elapsed:.2f}s, budget {BUDGET_SECONDS}s. "
        f"Slowest: {[f'{name} {seconds:.2f}s' for name, seconds in slowest]}"
    )


@pytest.mark.benchmark
def test_the_whole_registry_stays_within_its_own_budget(
    million_bars: OHLCVFrame,
) -> None:
    """Every registered indicator, including the two that walk bars in Python."""
    started = time.perf_counter()
    for indicator in default_indicators():
        indicator.compute_frame(million_bars)
    elapsed = time.perf_counter() - started
    assert elapsed < FULL_REGISTRY_BUDGET_SECONDS, (
        f"the full registry over {BARS} bars took {elapsed:.2f}s, "
        f"budget {FULL_REGISTRY_BUDGET_SECONDS}s"
    )
