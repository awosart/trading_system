"""Fixtures and generators for the feature-layer tests.

Prices are generated in integer ticks of 1e-5 and only converted to float at the
end. Real FX quotes are discrete that way, and it keeps hypothesis from spending
its budget on denormals and 17-significant-digit values that no feed produces
and that make a 1e-9 parity tolerance meaningless.

Volumes are allowed to reach zero on purpose: that is the branch every
volume-weighted indicator has to defend, and FX tick volume really does report
it in dead sessions.
"""

from collections.abc import Sequence
from datetime import UTC, datetime

import polars as pl
import pytest
from hypothesis import strategies as st

from trading_system.core.types import Timeframe
from trading_system.data.models import OHLCVFrame

TICK = 1e-5
_MIN_TICKS = 20_000
_START = datetime(2024, 1, 2, 21, 0, tzinfo=UTC)

#: One bar's worth of movement, in ticks: body, upper wick, lower wick, the gap
#: to the next open, and volume.
BAR_MOVE = st.tuples(
    st.integers(min_value=-200, max_value=200),
    st.integers(min_value=0, max_value=120),
    st.integers(min_value=0, max_value=120),
    st.integers(min_value=-40, max_value=40),
    st.integers(min_value=0, max_value=10_000),
)


def build_frame(
    moves: Sequence[tuple[int, int, int, int, int]],
    *,
    symbol: str = "TESTFX",
    timeframe: Timeframe = Timeframe.M1,
    start: datetime = _START,
) -> OHLCVFrame:
    """Fold a sequence of tick moves into a valid OHLCV frame.

    The start time sits just before the 17:00 New York rollover so that frames
    of a couple of hundred bars cross a session boundary, which is the branch
    the session VWAP would otherwise never take in a test.

    Args:
        moves: Per-bar ``(body, upper_wick, lower_wick, gap, volume)`` in ticks.
        symbol: Instrument identifier for the frame.
        timeframe: Bar size, used only for the timestamp step.
        start: Open time of the first bar.

    Returns:
        A frame whose bars all satisfy ``low <= open, close <= high``.
    """
    timestamps: list[datetime] = []
    opens: list[float] = []
    highs: list[float] = []
    lows: list[float] = []
    closes: list[float] = []
    volumes: list[float] = []

    price = 100_000
    for index, (body, upper, lower, gap, volume) in enumerate(moves):
        open_ticks = price
        close_ticks = max(_MIN_TICKS, open_ticks + body)
        high_ticks = max(open_ticks, close_ticks) + upper
        low_ticks = max(1, min(open_ticks, close_ticks) - lower)
        timestamps.append(start + timeframe.duration * index)
        opens.append(open_ticks * TICK)
        highs.append(high_ticks * TICK)
        lows.append(low_ticks * TICK)
        closes.append(close_ticks * TICK)
        volumes.append(float(volume))
        price = max(_MIN_TICKS, close_ticks + gap)

    df = pl.DataFrame(
        {
            "timestamp": timestamps,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        }
    )
    return OHLCVFrame.from_raw(df, symbol, timeframe)


def frame_from_bars(
    rows: Sequence[tuple[float, float, float, float, float]],
    *,
    symbol: str = "TESTFX",
    timeframe: Timeframe = Timeframe.M1,
    start: datetime = _START,
) -> OHLCVFrame:
    """Build a frame from hand-written ``(open, high, low, close, volume)`` bars.

    Args:
        rows: One tuple per bar, in order.
        symbol: Instrument identifier.
        timeframe: Bar size, used for the timestamp step.
        start: Open time of the first bar.

    Returns:
        The frame.
    """
    df = pl.DataFrame(
        {
            "timestamp": [start + timeframe.duration * index for index in range(len(rows))],
            "open": [row[0] for row in rows],
            "high": [row[1] for row in rows],
            "low": [row[2] for row in rows],
            "close": [row[3] for row in rows],
            "volume": [row[4] for row in rows],
        }
    )
    return OHLCVFrame.from_raw(df, symbol, timeframe)


def frame_from_closes(
    closes: Sequence[float],
    *,
    volumes: Sequence[float] | None = None,
    start: datetime = _START,
) -> OHLCVFrame:
    """Build a frame of zero-range bars, each flat at its close.

    Useful where an assertion is about the averaging itself and the intrabar
    range would only add noise to the arithmetic.

    Args:
        closes: One price per bar.
        volumes: Per-bar volume; defaults to 1.0 everywhere.
        start: Open time of the first bar.

    Returns:
        The frame.
    """
    weights = list(volumes) if volumes is not None else [1.0] * len(closes)
    return frame_from_bars(
        [
            (price, price, price, price, weight)
            for price, weight in zip(closes, weights, strict=True)
        ],
        start=start,
    )


@st.composite
def ohlcv_frames(draw: st.DrawFn, *, min_bars: int = 120, max_bars: int = 170) -> OHLCVFrame:
    """Generate a valid frame long enough to warm up every registered indicator.

    Args:
        draw: Hypothesis draw function.
        min_bars: Fewest bars to generate. The default clears Ichimoku's
            77-bar warmup with room to check the bars after it.
        max_bars: Most bars to generate.

    Returns:
        A drawn frame.
    """
    moves = draw(st.lists(BAR_MOVE, min_size=min_bars, max_size=max_bars))
    return build_frame(moves)


def deterministic_moves(count: int, *, seed: int = 12_345) -> list[tuple[int, int, int, int, int]]:
    """Produce reproducible bar moves without touching the global RNG.

    A small linear congruential generator keeps the reference data identical
    across runs and machines, which matters when a parity failure has to be
    reproduced from a test report.

    Args:
        count: Number of bars.
        seed: Generator seed.

    Returns:
        Moves suitable for :func:`build_frame`.
    """
    state = seed
    moves: list[tuple[int, int, int, int, int]] = []

    def next_value(modulus: int) -> int:
        nonlocal state
        state = (state * 1_103_515_245 + 12_345) % (2**31)
        return state % modulus

    for _ in range(count):
        moves.append(
            (
                next_value(401) - 200,
                next_value(121),
                next_value(121),
                next_value(81) - 40,
                next_value(10_001),
            )
        )
    return moves


@pytest.fixture(scope="session")
def reference_frame() -> OHLCVFrame:
    """A 600-bar frame, long enough for every warmup and two session rollovers."""
    return build_frame(deterministic_moves(600))


@pytest.fixture(scope="session")
def short_frame() -> OHLCVFrame:
    """A 40-bar frame, shorter than the slowest indicator's warmup."""
    return build_frame(deterministic_moves(40, seed=999))


def assert_series_close(
    actual: pl.Series, expected: pl.Series, *, atol: float = 1e-9, context: str = ""
) -> None:
    """Assert two float series agree, treating nulls as a value in their own right.

    Args:
        actual: Series under test.
        expected: Series to compare against.
        atol: Absolute tolerance for non-null pairs.
        context: Label included in the failure message.

    Raises:
        AssertionError: On the first differing element.
    """
    left = actual.to_list()
    right = expected.to_list()
    assert len(left) == len(right), f"{context}: lengths differ, {len(left)} vs {len(right)}"
    for index, (first, second) in enumerate(zip(left, right, strict=True)):
        if first is None and second is None:
            continue
        assert first is not None and second is not None, (
            f"{context}[{index}]: {first!r} vs {second!r}; one side is null"
        )
        assert abs(first - second) <= atol, (
            f"{context}[{index}]: {first!r} vs {second!r}, diff {abs(first - second):.3e}"
        )
