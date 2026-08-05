"""Full-run budget: one real strategy, five years of H1 bars, three instruments.

Deselected by default alongside the feature-layer and entry-layer benchmarks;
run with ``pytest -m benchmark``.

The clock starts at :class:`~trading_system.backtest.orchestrator.Orchestrator`
construction — which is where the per-stream feature pipelines actually run,
vectorised, once — and stops when
:meth:`~trading_system.backtest.orchestrator.Orchestrator.run` returns. That is
the whole cost a user waits on. Building the synthetic bars themselves is
scaffolding this test needs and a real run does not (data comes from Parquet,
already on its own budget elsewhere), so it is excluded from the timed section,
the same way :mod:`tests.entry.test_benchmark` excludes feature computation from
``evaluate()``'s own budget.

**The strategy is the shipped ``ema_pullback.json`` example**, not a spec
invented for this test — the same one :mod:`tests.backtest.test_features`
exercises for correctness. Its own ``max_concurrent_positions`` is 2, which
would leave two positions opened once and never closed for the rest of a
trending five-year run — a real number, but one that barely reaches the
exit-management code this budget is also supposed to cover. It is raised to 20
here, disclosed rather than silently changed, so the run actually opens,
manages and closes positions throughout instead of mostly sizing signals it
immediately refuses.

Its shipped ``timeframes.signal_tf`` is ``H4``, but this budget trades every
symbol at ``H1`` — throughput at H4 across five years is a much smaller loop
and would not exercise per-bar cost the same way. ``signal_tf`` is overridden
to ``H1`` alongside the concurrency cap, for the same reason: disclosed here
rather than left for :class:`~trading_system.backtest.orchestrator.Orchestrator`
to reject at construction, which it now does for any binding whose stream
timeframe does not match the spec's own ``signal_tf``.
"""

import json
import time
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import polars as pl
import pytest

from trading_system.backtest.clock import StreamKey
from trading_system.backtest.config import BacktestConfig
from trading_system.backtest.orchestrator import Orchestrator, StrategyBinding
from trading_system.core.instruments import load_instruments
from trading_system.core.types import Timeframe
from trading_system.data.models import OHLCVFrame
from trading_system.execution.config import (
    CostConfig,
    GapConfig,
    LimitFillConfig,
    SlippageConfig,
    SpreadConfig,
)
from trading_system.execution.costs import CostModel
from trading_system.exit.library import ExitLibrarySpec
from trading_system.risk.conversion import SameCurrencyConverter
from trading_system.risk.engine import RiskEngine
from trading_system.risk.sizing.methods import FixedFractional
from trading_system.strategies.schema import StrategySpec

ROOT = Path(__file__).resolve().parents[2]
INSTRUMENTS_PATH = ROOT / "configs" / "instruments.yaml"
LIBRARY_PATH = ROOT / "src" / "trading_system" / "exit" / "library.json"
EXAMPLE_PATH = ROOT / "src" / "trading_system" / "strategies" / "examples" / "ema_pullback.json"

#: Five years of hourly bars, continuously — a backtest does not need weekend
#: gaps modelled to stress the event loop, and a continuous clock is the harder
#: case for the merged stream besides (no bars ever coincide in close time
#: across streams the way a shared FX calendar occasionally gives for free).
YEARS = 5
BARS_PER_INSTRUMENT = YEARS * 365 * 24

#: Three asset classes, all quoted in USD in the shipped registry — this
#: benchmark is about event-loop and feature throughput, not FX conversion,
#: which has its own tests in P10.
SYMBOLS = ("EURUSD", "XAUUSD", "NAS100")
BASES = {"EURUSD": 1.10, "XAUUSD": 2000.0, "NAS100": 18000.0}

BUDGET_SECONDS = 60.0

#: Fractional trend and oscillation, scaled per instrument by its own base
#: price. A short period relative to ema_pullback's ema_50/adx_14 warmup is
#: deliberate: a wave that only ever produces one pullback in five years would
#: time an event loop that mostly does nothing, which is the same mismeasurement
#: a near-empty run would give.
TREND_PER_BAR = 0.00025 / 1.10
WAVE_AMPLITUDE = 0.006 / 1.10
WAVE_PERIOD_BARS = 9.0


def _wave_frame(symbol: str, base: float) -> OHLCVFrame:
    """A trending, oscillating series built with vectorised column construction.

    Row-by-row dict construction over 40 000+ bars is itself real time before
    the run under test even starts; building the columns with ``polars``
    expressions keeps the untimed setup fast without smoothing over the event
    loop's own per-bar cost, which still runs one Python bar at a time exactly
    as a live loop would.
    """
    n = BARS_PER_INSTRUMENT
    start = datetime(2019, 1, 1, tzinfo=UTC)
    base_df = pl.DataFrame(
        {
            "timestamp": pl.datetime_range(
                start, start + timedelta(hours=n - 1), interval="1h", eager=True
            ),
            "index": pl.int_range(0, n, eager=True).cast(pl.Float64),
        }
    )
    close_expr = (
        base
        + base * TREND_PER_BAR * pl.col("index")
        + base * WAVE_AMPLITUDE * (pl.col("index") / WAVE_PERIOD_BARS).sin()
    )
    spread = base * 0.0006
    df = (
        base_df.with_columns(close=close_expr)
        .with_columns(open=pl.col("close").shift(1).fill_null(base))
        .select(
            "timestamp",
            "open",
            (pl.max_horizontal("open", "close") + spread).alias("high"),
            (pl.min_horizontal("open", "close") - spread).alias("low"),
            "close",
            pl.lit(1000.0).alias("volume"),
        )
    )
    return OHLCVFrame(df, symbol=symbol, timeframe=Timeframe.H1)


def _strategy() -> StrategySpec:
    """The shipped ``ema_pullback.json``, with benchmark-only overrides.

    Both deviations from the shipped file are disclosed in the module
    docstring rather than made silently: the concurrency cap, and
    ``signal_tf`` moved from its native ``H4`` to the ``H1`` this benchmark
    actually trades at.
    """
    raw = json.loads(EXAMPLE_PATH.read_text())
    raw["risk_profile"]["max_concurrent_positions"] = 20
    raw["timeframes"]["signal_tf"] = "H1"
    return StrategySpec.model_validate(raw)


@pytest.mark.benchmark
def test_five_years_h1_three_instruments_under_budget() -> None:
    registry = load_instruments(INSTRUMENTS_PATH)
    library = ExitLibrarySpec.model_validate_json(LIBRARY_PATH.read_text())
    preset = next(item for item in library.presets if item.id == "atr_trail_aggressive")
    spec = _strategy()

    keys = tuple(StreamKey(symbol, Timeframe.H1) for symbol in SYMBOLS)
    streams = {key: _wave_frame(key.symbol, BASES[key.symbol]) for key in keys}

    costs = CostConfig(
        spread=SpreadConfig(
            session_multipliers={}, off_session_multiplier=1.0, volatility_beta=0.0
        ),
        slippage=SlippageConfig(),
        gap=GapConfig(),
        limit_fill=LimitFillConfig(),
        run_seed=11,
    )

    started = time.perf_counter()
    orchestrator = Orchestrator(
        config=BacktestConfig(
            atr_period=14, atr_baseline_bars=200, starting_balance=Decimal(100_000)
        ),
        streams=streams,
        bindings=[StrategyBinding(spec=spec, exit_preset=preset, keys=keys)],
        instruments=registry,
        risk_engine=RiskEngine(
            instruments=registry,
            sizing=FixedFractional(risk_pct=0.01),
            converter=SameCurrencyConverter(),
        ),
        cost_model=CostModel({symbol: registry[symbol] for symbol in SYMBOLS}, costs),
        converter=SameCurrencyConverter(),
        run_seed=11,
    )
    result = orchestrator.run()
    elapsed = time.perf_counter() - started

    print(
        f"\n5y H1 x {len(SYMBOLS)} instruments: {elapsed:.2f}s, {result.instants} instants, "
        f"{result.fills} fills, {len(result.trades)} trades, {result.open_at_end} open at end"
    )
    assert result.instants == BARS_PER_INSTRUMENT, "all three streams share one calendar here"
    # A run that traded nothing would time an event loop that mostly does
    # nothing, which is not the budget this test is measuring.
    assert result.fills > 0, "the scenario must actually trade, or it proves nothing"
    assert elapsed < BUDGET_SECONDS, f"{elapsed:.2f}s over the {BUDGET_SECONDS:.0f}s budget"
