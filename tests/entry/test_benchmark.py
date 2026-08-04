"""Per-bar budget for a compiled entry.

Deselected by default alongside the feature-layer benchmarks; run with
``pytest -m benchmark``.

The budget is on ``evaluate()`` plus the context construction that precedes it,
because that pair is what a live loop and a backtest both do once per bar. Feature
computation is deliberately outside it: that cost is the feature layer's budget,
already measured in :mod:`tests.features.test_benchmark`, and folding it in here
would hide the thing this test exists to protect.

Why it matters: a walk-forward study evaluates an entry across a parameter grid
over years of minute bars. At 50 microseconds a bar, a million-bar pass costs 50
seconds; at 500 it costs eight minutes, and the study stops being something
anyone runs.
"""

import math
import time

import pytest

from trading_system.entry.compiler import compile_entry
from trading_system.entry.context import BarSeries
from trading_system.entry.features import FeatureRegistry
from trading_system.strategies.schema import (
    AllOf,
    AnyOf,
    Invalidation,
    LimitOrder,
    Not,
    QualityModifier,
    StrategySpec,
)

from .conftest import frame_from_closes, leaf, ref, strategy_spec

BARS = 200_000

#: Definition-of-done budget, per bar, for the realistic spec below.
BUDGET_MICROSECONDS = 50.0

EMA_20 = ref("ema", period=20)
EMA_50 = ref("ema", period=50)
RSI_14 = ref("rsi", period=14)
ATR_14 = ref("atr", period=14)
ADX_14 = ref("adx", channel="adx", period=14)
MACD_LINE = ref("macd", channel="macd")
MACD_SIGNAL = ref("macd", channel="signal")


def benchmark_spec() -> StrategySpec:
    """A spec with more structure than any real one is likely to have.

    Fourteen leaves across a nested trigger, two confirmations, an invalidation
    condition and two quality modifiers, over six indicators including two
    multi-output ones. If this clears the budget, an ordinary spec is not close
    to it.
    """
    return strategy_spec(
        trigger=AllOf(
            conditions=[
                leaf("cross_above", "price:close", EMA_20),
                leaf("gt", "price:close", EMA_50),
                AnyOf(
                    conditions=[
                        leaf("between", RSI_14, (40.0, 60.0)),
                        leaf("rising", RSI_14, 3.0),
                        Not(condition=leaf("gt", RSI_14, 80.0)),
                    ]
                ),
                Not(condition=leaf("lt", ATR_14, 0.0001)),
            ]
        ),
        confirmation=[
            leaf("cross_above", MACD_LINE, MACD_SIGNAL),
            leaf("gt", ADX_14, 20.0),
        ],
        confirmation_window_bars=5,
        invalidation=Invalidation(price_level=EMA_50, condition=leaf("falling", ADX_14, 2.0)),
        order=LimitOrder(offset=0.00005),
        quality_modifiers=[
            QualityModifier(condition=leaf("gt", ADX_14, 25.0), delta=0.15, reason="trend"),
            QualityModifier(condition=leaf("lt", RSI_14, 45.0), delta=0.05, reason="pullback"),
        ],
    )


@pytest.mark.benchmark
def test_evaluate_stays_under_the_per_bar_budget() -> None:
    spec = benchmark_spec()
    registry = FeatureRegistry.from_strategy(spec)
    closes = [
        1.1000 + 0.0040 * math.sin(index / 37.0) + 0.0015 * math.sin(index / 7.0)
        for index in range(BARS)
    ]
    frame = frame_from_closes(
        closes,
        lows=[close - 0.0002 for close in closes],
        highs=[close + 0.0002 for close in closes],
    )
    series = BarSeries.from_frame(frame, registry.pipeline().compute(frame))
    evaluator = compile_entry(spec, registry)

    started = time.perf_counter()
    signals = [
        signal for ctx in series.contexts() if (signal := evaluator.evaluate(ctx)) is not None
    ]
    elapsed = time.perf_counter() - started

    per_bar = elapsed / BARS * 1e6
    assert signals, "a benchmark that never fires the signal path measures the wrong thing"
    assert per_bar < BUDGET_MICROSECONDS, (
        f"evaluate() took {per_bar:.1f} us/bar over {BARS} bars, budget is "
        f"{BUDGET_MICROSECONDS:.0f} us ({len(signals)} signals)"
    )
