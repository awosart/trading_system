"""3 worked strategy examples × 8 bundled presets — the N×M promise, exercised.

Entry and Exit stay decoupled even here: this does not run the Entry compiler
against the example specs to produce a real signal (that pulls in the whole
feature pipeline and belongs to a full backtest-engine test, not P07). Instead
each spec's own ``entries[0].direction`` and its declared risk profile are
enough to open a plausible :class:`~trading_system.exit.position.ManagedPosition`
by hand — what this test is actually proving is that any of the three real,
schema-valid ``StrategySpec`` examples can pair with any of the eight real
presets and run to completion without failing, which is exactly the promise
``exit_ref`` composition makes.
"""

import math
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from trading_system.core.types import Price, Side, Timeframe
from trading_system.entry.context import BarSeries
from trading_system.exit.context import exit_contexts
from trading_system.exit.library import load_library
from trading_system.exit.position import ManagedPosition
from trading_system.exit.rules._features import atr_key, ma_key, swing_keys
from trading_system.strategies import schema as strategy_schema_module
from trading_system.strategies.schema import Direction, StrategySpec

EXAMPLES_DIR = Path(strategy_schema_module.__file__).parent / "examples"
EXAMPLE_PATHS = sorted(EXAMPLES_DIR.glob("*.json"))

ATR_14 = atr_key(14)
ATR_10 = atr_key(10)
SWING_LOW_5, SWING_HIGH_5 = swing_keys(5)
EMA_20 = ma_key(20, "close")

LIBRARY = load_library()
PRESET_IDS = sorted(LIBRARY)

START = datetime(2024, 1, 8, 0, 0, tzinfo=UTC)  # a Monday, clear of any weekend close
BARS = 400
ENTRY_PRICE = 1.1000
RISK_DISTANCE = 0.0100


def _synthetic_closes(count: int) -> list[float]:
    """A trending, oscillating close series with plenty for exits to react to.

    Enough range to threaten every kind of exit at least once: stops,
    R-multiple targets, activation thresholds, and a few hundred bars of
    runway for time-based exits.
    """
    closes = []
    for index in range(count):
        drift = 0.00004 * index
        wave = 0.0090 * math.sin(2 * math.pi * index / 40)
        closes.append(ENTRY_PRICE + drift + wave)
    return closes


def _synthetic_series(timeframe: Timeframe) -> BarSeries:
    """A bar series at ``timeframe``, carrying every feature a bundled preset reads."""
    closes = _synthetic_closes(BARS)
    timestamps = [START + timeframe.duration * index for index in range(BARS)]
    highs = [close + 0.0015 for close in closes]
    lows = [close - 0.0015 for close in closes]
    opens = [closes[index - 1] if index > 0 else closes[0] for index in range(BARS)]

    window = 20
    ema_column = [
        sum(closes[max(0, index - window + 1) : index + 1])
        / len(closes[max(0, index - window + 1) : index + 1])
        for index in range(BARS)
    ]

    features = {
        ATR_14: [0.0050] * BARS,
        ATR_10: [0.0045] * BARS,
        SWING_LOW_5: [close - 0.0200 for close in closes],
        SWING_HIGH_5: [close + 0.0200 for close in closes],
        EMA_20: ema_column,
    }
    return BarSeries(
        symbol="TESTFX",
        timeframe=timeframe,
        timestamps=timestamps,
        prices={
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": [1_000.0] * BARS,
        },
        features=features,
    )


def _load_example(path: Path) -> StrategySpec:
    return StrategySpec.model_validate_json(path.read_text(encoding="utf-8"))


def _open_position(spec: StrategySpec) -> ManagedPosition:
    """A plausible position for ``spec``'s first entry leg.

    Stands in for the Risk Engine, which does not exist yet: the actual entry
    price and distance are arbitrary, since what this test exercises is
    whether the (strategy, preset) pairing runs, not any particular strategy's
    edge.
    """
    side = Side.BUY if spec.entries[0].direction is Direction.LONG else Side.SELL
    stop = ENTRY_PRICE - RISK_DISTANCE if side is Side.BUY else ENTRY_PRICE + RISK_DISTANCE
    return ManagedPosition(
        symbol="TESTFX",
        side=side,
        entry_price=Price(ENTRY_PRICE),
        size=Decimal("1"),
        initial_stop=Price(stop),
        opened_at=START,
        strategy_id=spec.id,
    )


COMBINATIONS = [(path, preset_id) for path in EXAMPLE_PATHS for preset_id in PRESET_IDS]


class TestCombinatorial:
    def test_exactly_three_examples_and_eight_presets(self) -> None:
        assert len(EXAMPLE_PATHS) == 3
        assert len(PRESET_IDS) == 8
        assert len(COMBINATIONS) == 24

    @pytest.mark.parametrize(
        ("path", "preset_id"), COMBINATIONS, ids=[f"{p.stem}-{i}" for p, i in COMBINATIONS]
    )
    def test_every_combination_runs_to_a_sensible_result(self, path: Path, preset_id: str) -> None:
        spec = _load_example(path)
        position = _open_position(spec)
        series = _synthetic_series(spec.timeframes.signal_tf)
        plan = LIBRARY[preset_id]

        result = plan.run(position, exit_contexts(series))

        # Ran to completion: every fill is internally consistent and the
        # ledger never exceeds the position.
        assert result.bars > 0
        assert 0 <= position.remaining_fraction <= 1
        assert sum(leg.fraction for leg in position.legs) == 1 - position.remaining_fraction
        for leg in position.legs:
            assert math.isfinite(leg.price)
            assert math.isfinite(leg.r_multiple)

        # PnL is computed and finite either way.
        last_close = series.context(result.bars - 1).price("close") or ENTRY_PRICE
        assert position.realized_quote_move.is_finite()
        assert math.isfinite(position.total_r(Price(last_close)))

        # closed <=> nothing remains, and vice versa.
        assert result.closed == (position.remaining_fraction == 0)
        assert result.closed == (not position.is_open)


class TestMostCombinationsActuallyResolve:
    """A check that the synthetic series is doing its job.

    Not a strict per-combo requirement, but a fixture where nothing ever
    closes would let every assertion above pass vacuously.
    """

    def test_most_of_the_matrix_produces_at_least_one_fill(self) -> None:
        produced_fills = 0
        for path, preset_id in COMBINATIONS:
            spec = _load_example(path)
            position = _open_position(spec)
            series = _synthetic_series(spec.timeframes.signal_tf)
            result = LIBRARY[preset_id].run(position, exit_contexts(series))
            if result.fills:
                produced_fills += 1
        assert produced_fills >= len(COMBINATIONS) * 0.8
