"""Fixtures for the backtest layer.

The instruments and the exit library come from the shipped files, for the reason
the risk and execution fixtures give: a hand-built copy drifts from the one that
is actually traded, and the tests then pass against a universe nobody runs.

The strategies here are **price-only** — their conditions compare ``price:close``
against ``price:open`` or a constant. That is not a simplification of what a
strategy is; it is what lets these tests exercise the loop, the portfolio and the
execution wiring without also depending on the feature pipeline, which stage 1
does not yet assemble per stream.
"""

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from trading_system.backtest.clock import StreamKey
from trading_system.backtest.config import BacktestConfig
from trading_system.backtest.orchestrator import BacktestResult, Orchestrator, StrategyBinding
from trading_system.core.instruments import InstrumentRegistry, load_instruments
from trading_system.core.types import Timeframe
from trading_system.data.models import OHLCVFrame
from trading_system.execution.config import (
    CostConfig,
    GapConfig,
    LimitFillConfig,
    PerLotRolloverSwap,
    SlippageConfig,
    SlippageParams,
    SpreadConfig,
)
from trading_system.execution.costs import CostModel
from trading_system.exit.library import ExitLibrarySpec, ExitPresetSpec
from trading_system.risk.circuit_breakers import CircuitBreakerConfig, CircuitBreakers
from trading_system.risk.conversion import SameCurrencyConverter
from trading_system.risk.engine import RiskEngine
from trading_system.risk.portfolio_risk import PortfolioLimitsConfig, PortfolioRisk
from trading_system.risk.sizing.methods import FixedAmount, FixedFractional
from trading_system.strategies.schema import StrategySpec

ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = ROOT / "configs" / "instruments.yaml"
LIBRARY_PATH = ROOT / "src" / "trading_system" / "exit" / "library.json"
EMA_PULLBACK_PATH = (
    ROOT / "src" / "trading_system" / "strategies" / "examples" / "ema_pullback.json"
)

#: Start of every synthetic series. A Monday, so trading-day labels and session
#: lookups do not depend on when the suite runs.
START = datetime(2024, 3, 4, tzinfo=UTC)

#: The stream almost every test trades.
EURUSD_H1 = StreamKey("EURUSD", Timeframe.H1)

#: The streams the no-lookahead harness trades. ``ema_pullback`` reasons on H4,
#: and the orchestrator refuses a stream at any other timeframe.
EURUSD_H4 = StreamKey("EURUSD", Timeframe.H4)
XAUUSD_H4 = StreamKey("XAUUSD", Timeframe.H4)


@pytest.fixture(scope="session")
def registry() -> InstrumentRegistry:
    """The bundled instrument registry."""
    return load_instruments(REGISTRY_PATH)


@pytest.fixture(scope="session")
def library() -> ExitLibrarySpec:
    """The bundled exit library, as specs rather than built plans.

    Specs, because a plan is built per position — exit rules carry state, so two
    concurrent positions cannot share one.
    """
    return ExitLibrarySpec.model_validate_json(LIBRARY_PATH.read_text())


@pytest.fixture
def preset(library: ExitLibrarySpec) -> ExitPresetSpec:
    """``conservative_2r``: a protective stop and a 2R target, nothing else.

    Chosen because both of its rules are legible in a hand computation: one level
    below the entry, one above, no trailing and no ladder.
    """
    return next(spec for spec in library.presets if spec.id == "conservative_2r")


def flat_costs() -> CostConfig:
    """Costs with every *multiplier* neutral and no slippage.

    The base spread is not here — it comes from the instrument's
    ``typical_spread_points``, which is why :func:`costless_registry` exists.
    What this switches off is everything that would make a fill's price depend
    on the session, the volatility or a random draw, so a hand computation stays
    a hand computation. Tests that are *about* costs turn the relevant one on.
    """
    return CostConfig(
        spread=SpreadConfig(
            session_multipliers={},
            off_session_multiplier=1.0,
            volatility_beta=0.0,
        ),
        slippage=SlippageConfig(),
        gap=GapConfig(),
        limit_fill=LimitFillConfig(),
        swap={},
        run_seed=17,
    )


def costless_registry(registry: InstrumentRegistry) -> InstrumentRegistry:
    """The shipped registry with EURUSD's spread and commission zeroed.

    Derived from the real specification rather than invented, so every other
    field — tick size, point size, lot bounds, contract size — is the one that is
    actually traded. Only the two figures a test is deliberately holding at zero
    are changed, and they are changed visibly here rather than by a config knob
    that would also silently affect a run somebody cares about.
    """
    eurusd = registry["EURUSD"].model_copy(
        update={"typical_spread_points": 0.0, "commission_per_lot": Decimal(0)}
    )
    return InstrumentRegistry({**{s: registry[s] for s in registry.symbols}, "EURUSD": eurusd})


def bars(
    closes: Sequence[float],
    *,
    symbol: str = "EURUSD",
    timeframe: Timeframe = Timeframe.H1,
    start: datetime = START,
    spread: float = 0.0005,
    step: timedelta | None = None,
    first_open: float | None = None,
) -> OHLCVFrame:
    """Build a frame whose closes are exactly ``closes``.

    Each bar opens at the previous close, so the series is gapless and a test
    reasoning about "the price the next bar opens at" can name it. Highs and lows
    bracket the open/close range by ``spread``, so an extreme never surprises a
    test that is reasoning about closes.

    ``first_open`` decides whether bar 0 is an up bar. It matters more than it
    looks: left at the first close, bar 0 opens and closes at the same price, so
    a ``close > open`` trigger does **not** fire on it and every subsequent index
    in a hand computation shifts by one.

    Args:
        closes: One close per bar.
        symbol: Instrument the frame is for.
        timeframe: Bar size.
        start: Open time of the first bar.
        spread: How far the high and low sit outside the open/close range.
        step: Distance between bar opens. Defaults to the timeframe's duration.
        first_open: Open of bar 0. Defaults to its own close, making bar 0 a doji.

    Returns:
        The frame.
    """
    stride = step if step is not None else timeframe.duration
    rows: list[dict[str, Any]] = []
    previous = closes[0] if first_open is None else first_open
    for index, close in enumerate(closes):
        opening = previous
        rows.append(
            {
                "timestamp": start + stride * index,
                "open": opening,
                "high": max(opening, close) + spread,
                "low": min(opening, close) - spread,
                "close": close,
                "volume": 1000.0,
            }
        )
        previous = close
    return OHLCVFrame(pl.DataFrame(rows), symbol=symbol, timeframe=timeframe)


def strategy(
    *,
    strategy_id: str = "price-only",
    trigger: dict[str, Any] | None = None,
    invalidation: float = 1.0,
    order: dict[str, Any] | None = None,
    expire_after_bars: int | None = None,
    max_concurrent_positions: int = 1,
    signal_tf: Timeframe = Timeframe.H1,
    stop_pips: float = 20.0,
) -> StrategySpec:
    """A price-only long strategy, parameterised where the tests differ.

    Args:
        strategy_id: Id carried into every signal and position.
        trigger: Trigger condition. Defaults to "the bar closed up".
        invalidation: Absolute price the setup is disproven at. A constant, so
            the stop is predictable in a hand computation.
        order: Entry order spec. Defaults to ``MARKET``.
        expire_after_bars: Order lifetime, or ``None`` to leave the field unset
            so the run's fallback applies.
        max_concurrent_positions: Cap on simultaneous positions.
        signal_tf: Timeframe the entry reasons on.
        stop_pips: Fixed stop distance the risk profile asks for.

    Returns:
        The validated spec.
    """
    entry_order: dict[str, Any] = {"order": order or {"type": "MARKET"}}
    if expire_after_bars is not None:
        entry_order["expire_after_bars"] = expire_after_bars
    return StrategySpec.model_validate(
        {
            "id": strategy_id,
            "name": "Price Only",
            "version": "1.0.0",
            "author": "tests",
            "type": "INTRADAY",
            "status": "DRAFT",
            "timeframes": {"signal_tf": signal_tf.value, "entry_tf": signal_tf.value},
            "instruments": {
                "allowed_classes": ["FX"],
                "allowed_symbols": ["EURUSD"],
                "denied_symbols": [],
            },
            "entries": [
                {
                    "direction": "LONG",
                    "trigger": trigger
                    or {"type": "leaf", "op": "gt", "left": "price:close", "right": "price:open"},
                    "invalidation": {"price_level": invalidation},
                    "entry_order": entry_order,
                }
            ],
            "exit_ref": "conservative_2r",
            "filters": [],
            "risk_profile": {
                "base_quality": 0.6,
                "stop_reference": {"kind": "FIXED_PIPS", "pips": stop_pips},
                "max_concurrent_positions": max_concurrent_positions,
            },
        }
    )


def orchestrator(
    *,
    registry: InstrumentRegistry,
    streams: dict[StreamKey, OHLCVFrame],
    bindings: Sequence[StrategyBinding],
    config: BacktestConfig | None = None,
    costs: CostConfig | None = None,
    risk_pct: float = 0.01,
) -> Orchestrator:
    """Assemble a run over synthetic streams.

    Args:
        registry: Instrument specifications.
        streams: Bars per stream.
        bindings: Strategies and the streams they trade.
        config: Run parameters. Defaults to a short ATR baseline, since a
            synthetic series is far shorter than a real one.
        costs: Cost configuration. Free by default.
        risk_pct: Fraction of equity per trade.

    Returns:
        The orchestrator, ready to run.
    """
    cost_config = costs if costs is not None else flat_costs()
    symbols = {key.symbol for key in streams}
    return Orchestrator(
        config=config
        if config is not None
        else BacktestConfig(atr_baseline_bars=5, atr_period=3, starting_balance=Decimal(100_000)),
        streams=streams,
        bindings=bindings,
        instruments=registry,
        risk_engine=RiskEngine(
            instruments=registry,
            sizing=FixedFractional(risk_pct=risk_pct),
            converter=SameCurrencyConverter(),
        ),
        cost_model=CostModel({symbol: registry[symbol] for symbol in symbols}, cost_config),
        converter=SameCurrencyConverter(),
        run_seed=cost_config.run_seed,
    )


# ----------------------------------------------------------------------------
# The no-lookahead harness, and the reproducibility tests, run one shared setup:
# the shipped ``ema_pullback`` strategy over a synthetic H4 series. Shipped
# rather than hand-built for the reason the registry and the exit library are
# shipped — a leak the harness cannot see in the strategy people actually run is
# a leak the harness does not check for.
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class RunInputs:
    """Everything one run is wired from, in one picklable place.

    Held as data rather than as an assembled :class:`Orchestrator` so the same
    description can be digested into a run id, varied one field at a time, and
    (from P13 stage 2's parallel half) shipped to another process.

    Attributes:
        config: Run parameters.
        streams: Bars per stream.
        bindings: Strategies, exit presets and the streams they trade.
        instruments: Contract specifications.
        costs: Spread, slippage, commission and financing.
        sizing: How much money goes at stake per trade.
        limits: Concurrent-risk ceilings.
        breakers: When trading stops altogether.
    """

    config: BacktestConfig
    streams: Mapping[StreamKey, OHLCVFrame]
    bindings: tuple[StrategyBinding, ...]
    instruments: InstrumentRegistry
    costs: CostConfig
    sizing: FixedFractional | FixedAmount
    limits: PortfolioLimitsConfig
    breakers: CircuitBreakerConfig

    def orchestrator(self) -> Orchestrator:
        """Assemble the run these inputs describe."""
        return Orchestrator(
            config=self.config,
            streams=self.streams,
            bindings=self.bindings,
            instruments=self.instruments,
            risk_engine=RiskEngine(
                instruments=self.instruments,
                sizing=self.sizing,
                converter=SameCurrencyConverter(),
                portfolio=PortfolioRisk(self.limits),
                breakers=CircuitBreakers(self.breakers),
            ),
            cost_model=CostModel(
                {key.symbol: self.instruments[key.symbol] for key in self.streams}, self.costs
            ),
            converter=SameCurrencyConverter(),
            run_seed=self.costs.run_seed,
        )

    def run(self) -> BacktestResult:
        """Walk the whole run and return what it produced."""
        return self.orchestrator().run()

    def components(self) -> dict[str, object]:
        """The named components a run manifest digests beyond bars and strategies."""
        return {
            "costs": self.costs,
            "sizing": self.sizing,
            "converter": SameCurrencyConverter(),
            "portfolio_limits": self.limits,
            "circuit_breakers": self.breakers,
        }

    def with_streams(self, streams: Mapping[StreamKey, OHLCVFrame]) -> "RunInputs":
        """The same run over different bars — what truncation and poisoning vary."""
        return replace(self, streams=dict(streams))


def ema_pullback(
    *,
    strategy_id: str | None = None,
    asset_class: str | None = None,
    symbol: str | None = None,
) -> StrategySpec:
    """The shipped ``ema_pullback`` example, optionally re-scoped to another instrument.

    Re-validated rather than copied field by field, so a re-scoping typo fails
    here instead of producing a strategy that silently never trades — the defect
    P06 stage 1.5 found in ``SessionFilter.sessions``.

    Args:
        strategy_id: Id to give the spec. Its own by default.
        asset_class: Instrument class to allow instead of FX.
        symbol: Symbol to allow instead of EURUSD/GBPUSD.

    Returns:
        The validated spec.
    """
    payload: dict[str, Any] = json.loads(EMA_PULLBACK_PATH.read_text())
    if strategy_id is not None:
        payload["id"] = strategy_id
    if asset_class is not None or symbol is not None:
        payload["instruments"] = {
            "allowed_classes": [asset_class] if asset_class is not None else ["FX"],
            "allowed_symbols": [symbol] if symbol is not None else ["EURUSD"],
            "denied_symbols": [],
        }
    return StrategySpec.model_validate(payload)


def swing_series(
    length: int,
    *,
    symbol: str = "EURUSD",
    base: float = 1.10,
    scale: float = 1.0,
    phase: float = 0.0,
    macro_period: int = 90,
    micro_period: int = 18,
    timeframe: Timeframe = Timeframe.H4,
    start: datetime = START,
) -> OHLCVFrame:
    """A trending series that turns over, so positions both open and close.

    Two sine components: a slow one that produces up-legs for the strategy to buy
    and down-legs for the structure trail to stop out on, and a fast one that
    produces the pullbacks the entry is looking for. Deterministic rather than a
    random walk, for the reason P07 stage 3's combinatorial test gives — a run
    that produces no trades proves nothing, and a random series makes "no trades"
    a property of the seed.

    Args:
        length: How many bars.
        symbol: Instrument the frame is for.
        base: Price the series oscillates around.
        scale: Multiplier on both amplitudes and on the bar's range, so the same
            shape works at EURUSD's 1.10 and at gold's 2000.
        phase: Phase offset applied to both components, in radians, for making a
            second instrument's series its own series rather than a copy of the
            first. Not every offset produces trades — the entry is looking for a
            pullback that recrosses, and some alignments of the two components
            never present one — so a new value belongs with a check that the
            run it produces is not empty.
        macro_period: Bars per slow cycle.
        micro_period: Bars per pullback cycle.
        timeframe: Bar size.
        start: Open time of bar 0.

    Returns:
        The frame.
    """
    closes = [
        base
        + 0.02 * scale * math.sin(2 * math.pi * (index / macro_period) + phase)
        + 0.01 * scale * math.sin(2 * math.pi * (index / micro_period) + phase)
        for index in range(length)
    ]
    rows: list[dict[str, Any]] = []
    previous = closes[0] - 0.001 * scale
    for index, close in enumerate(closes):
        opening = previous
        rows.append(
            {
                "timestamp": start + timeframe.duration * index,
                "open": opening,
                "high": max(opening, close) + 0.0006 * scale,
                "low": min(opening, close) - 0.0006 * scale,
                "close": close,
                "volume": 1000.0 + 10 * (index % 7),
            }
        )
        previous = close
    return OHLCVFrame(pl.DataFrame(rows), symbol=symbol, timeframe=timeframe)


def noisy_costs(seed: int = 11) -> CostConfig:
    """Costs with every random and volatility-dependent component switched **on**.

    The opposite of :func:`flat_costs`, and deliberately so. A harness run under
    flat costs would exercise none of the paths where a leak is cheapest to
    introduce: ``jitter_points`` is what makes a fill consult
    :func:`~trading_system.execution.rng.fill_seed` at all, and
    ``atr_coefficient`` is what makes a fill's price depend on a rolling
    statistic. With both at zero the permutation check cannot see a shared random
    stream and the poisoning check cannot see a whole-series normalisation.

    Args:
        seed: The run seed.

    Returns:
        The configuration.
    """
    return CostConfig(
        slippage=SlippageConfig(
            market=SlippageParams(base_points=0.3, atr_coefficient=0.2, jitter_points=0.4),
            stop=SlippageParams(base_points=0.5, atr_coefficient=0.3, jitter_points=0.6),
            limit=SlippageParams(),
        ),
        swap={"FX": PerLotRolloverSwap(), "COMMODITY": PerLotRolloverSwap()},
        run_seed=seed,
    )


def harness_inputs(
    registry: InstrumentRegistry,
    *,
    streams: Mapping[StreamKey, OHLCVFrame],
    bindings: Sequence[StrategyBinding],
    costs: CostConfig | None = None,
    sizing: FixedFractional | FixedAmount | None = None,
    limits: PortfolioLimitsConfig | None = None,
    breakers: CircuitBreakerConfig | None = None,
) -> RunInputs:
    """Wire one harness run.

    Args:
        registry: Instrument specifications.
        streams: Bars per stream.
        bindings: Strategies and the streams they trade.
        costs: Cost configuration. :func:`noisy_costs` by default.
        sizing: Sizing method. One percent of equity by default.
        limits: Concurrent-risk ceilings. The conservative defaults unless a
            test needs them out of the way.
        breakers: Circuit breakers. The defaults unless a test needs them off.

    Returns:
        The inputs.
    """
    return RunInputs(
        config=BacktestConfig(atr_period=14, atr_baseline_bars=30),
        streams=dict(streams),
        bindings=tuple(bindings),
        instruments=registry,
        costs=costs if costs is not None else noisy_costs(),
        sizing=sizing if sizing is not None else FixedFractional(risk_pct=0.01),
        limits=limits if limits is not None else PortfolioLimitsConfig(),
        breakers=breakers if breakers is not None else CircuitBreakerConfig(),
    )
