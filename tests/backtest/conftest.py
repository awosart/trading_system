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

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from trading_system.backtest.clock import StreamKey
from trading_system.backtest.config import BacktestConfig
from trading_system.backtest.orchestrator import Orchestrator, StrategyBinding
from trading_system.core.instruments import InstrumentRegistry, load_instruments
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
from trading_system.exit.library import ExitLibrarySpec, ExitPresetSpec
from trading_system.risk.conversion import SameCurrencyConverter
from trading_system.risk.engine import RiskEngine
from trading_system.risk.sizing.methods import FixedFractional
from trading_system.strategies.schema import StrategySpec

ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = ROOT / "configs" / "instruments.yaml"
LIBRARY_PATH = ROOT / "src" / "trading_system" / "exit" / "library.json"

#: Start of every synthetic series. A Monday, so trading-day labels and session
#: lookups do not depend on when the suite runs.
START = datetime(2024, 3, 4, tzinfo=UTC)

#: The stream almost every test trades.
EURUSD_H1 = StreamKey("EURUSD", Timeframe.H1)


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
