"""Feature assembly per stream: dedup across strategies, and the exit side of it.

Two gaps closed here, both of the same shape — a column an evaluator or a rule
reads by name that the stream's :class:`~trading_system.entry.context.BarSeries`
did not actually carry, which used to fail not at compile time but on the first
bar that read it:

* An **entry** trigger, confirmation, invalidation or quality modifier
  referencing an indicator. Fixed by building one merged
  :class:`~trading_system.entry.features.FeatureRegistry` per stream, from the
  union of every strategy trading it.
* An **exit** rule or stop modifier reading a column through
  :mod:`trading_system.exit.rules._features` — which Entry's registry never
  sees at all, since Exit deliberately keeps its own naming convention rather
  than importing Entry's. Fixed by
  :func:`~trading_system.backtest.orchestrator._merged_feature_specs`.
"""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from tests.backtest.conftest import bars, orchestrator
from trading_system.backtest.clock import StreamKey
from trading_system.backtest.config import BacktestConfig
from trading_system.backtest.orchestrator import (
    StrategyBinding,
    _exit_feature_specs,
    _merged_feature_specs,
    _merged_registry,
)
from trading_system.core.instruments import InstrumentRegistry
from trading_system.core.types import Timeframe
from trading_system.exit.library import ExitLibrarySpec, ExitPresetSpec
from trading_system.features.base import BaseIndicator
from trading_system.features.indicators.trend import EMA
from trading_system.strategies.schema import StrategySpec

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "src" / "trading_system" / "strategies" / "examples"


def _ema_strategy(strategy_id: str, *, extra_ref_period: int | None = None) -> StrategySpec:
    """A strategy whose trigger reads ``ema(period=50)``, and optionally more.

    Args:
        strategy_id: Id for this spec.
        extra_ref_period: If given, an additional condition reads ``rsi`` of
            this period, so two strategies can share one reference (``ema_50``)
            while each also asking for something the other does not.
    """
    conditions = [
        {
            "type": "leaf",
            "op": "gt",
            "left": "price:close",
            "right": {"indicator": "ema", "params": {"period": 50}},
        }
    ]
    if extra_ref_period is not None:
        conditions.append(
            {
                "type": "leaf",
                "op": "lt",
                "left": {"indicator": "rsi", "params": {"period": extra_ref_period}},
                "right": 90.0,
            }
        )
    trigger = (
        conditions[0] if len(conditions) == 1 else {"type": "all_of", "conditions": conditions}
    )
    return StrategySpec.model_validate(
        {
            "id": strategy_id,
            "name": strategy_id,
            "version": "1.0.0",
            "author": "tests",
            "type": "INTRADAY",
            "status": "DRAFT",
            "timeframes": {"signal_tf": "H1", "entry_tf": "H1"},
            "instruments": {
                "allowed_classes": ["FX"],
                "allowed_symbols": ["EURUSD"],
                "denied_symbols": [],
            },
            "entries": [
                {
                    "direction": "LONG",
                    "trigger": trigger,
                    "invalidation": {"price_level": 1.0},
                    "entry_order": {"order": {"type": "MARKET"}},
                }
            ],
            "exit_ref": "conservative_2r",
            "filters": [],
            "risk_profile": {
                "base_quality": 0.6,
                "stop_reference": {"kind": "FIXED_PIPS", "pips": 20.0},
                "max_concurrent_positions": 5,
            },
        }
    )


class TestEntrySideDedup:
    """Two strategies asking for the same indicator collapse onto one spec."""

    def test_two_strategies_sharing_ema_50_produce_one_feature_spec(self) -> None:
        a = _ema_strategy("strategy-a", extra_ref_period=14)
        b = _ema_strategy("strategy-b", extra_ref_period=21)
        registry = _merged_registry([a, b])

        names = [spec.name for spec in registry.specs]
        assert names.count("ema_50") == 1, (
            f"ema_50 should be requested once for the stream, not once per strategy; got {names}"
        )
        # Each strategy's OWN extra reference still made it through the union.
        assert "rsi_14" in names
        assert "rsi_21" in names
        assert len(registry.specs) == 3

    def test_the_shared_column_is_computed_once_not_once_per_strategy(
        self, monkeypatch: pytest.MonkeyPatch, registry: InstrumentRegistry, preset: ExitPresetSpec
    ) -> None:
        """The stronger claim: not just one spec, one actual computation.

        Patches the indicator's own ``compute_frame`` — the entry point every
        vectorised computation goes through — and counts calls specifically for
        ``EMA(period=50)``. Two strategies sharing that reference on the same
        stream must produce exactly one call, or a run with twenty strategies
        computing the same EMA-50 twenty times would pass every spec-level check
        and still be the defect this test exists to catch.
        """
        calls = {"ema_50": 0}
        original = BaseIndicator.compute_frame

        def counting(self: BaseIndicator, frame: object) -> object:  # type: ignore[no-untyped-def]
            if isinstance(self, EMA) and self.period == 50:
                calls["ema_50"] += 1
            return original(self, frame)  # type: ignore[arg-type]

        monkeypatch.setattr(BaseIndicator, "compute_frame", counting)

        key = StreamKey("EURUSD", Timeframe.H1)
        closes = [1.1000 + 0.0005 * i for i in range(120)]
        a = _ema_strategy("strategy-a", extra_ref_period=14)
        b = _ema_strategy("strategy-b", extra_ref_period=21)
        orchestrator(
            registry=registry,
            streams={key: bars(closes)},
            bindings=[
                StrategyBinding(spec=a, exit_preset=preset, keys=(key,)),
                StrategyBinding(spec=b, exit_preset=preset, keys=(key,)),
            ],
        )
        assert calls["ema_50"] == 1


class TestExitSideFeatures:
    """The half of the requirement Entry's own registry cannot see."""

    def test_structure_stop_needs_its_swing_and_its_gated_atr(
        self, library: ExitLibrarySpec
    ) -> None:
        preset = next(item for item in library.presets if item.id == "structure_trail")
        specs = {spec.name for spec in _exit_feature_specs(preset)}
        # structure_trail is lookback=5, buffer_atr_multiple=0.3 in the shipped
        # library — positive, so the ATR is required alongside the swing.
        assert specs == {"swing_5", "atr_14"}

    def test_a_zero_buffer_does_not_pull_in_an_atr_nobody_reads(self) -> None:
        """The mirror case: buffer_atr_multiple == 0 means no ATR column at all.

        `StructureStop.on_bar` never calls `ctx.feature(atr_key(...))` when its
        buffer is zero — the branch is unreached — so requiring the column here
        would compute something no rule in the preset can ever ask for.
        """
        preset = ExitPresetSpec.model_validate(
            {
                "id": "zero-buffer",
                "name": "zero-buffer",
                "protective_stop": {"kind": "PROTECTIVE_STOP"},
                "stop_modifiers": [
                    {"kind": "STRUCTURE_STOP", "lookback": 5, "buffer_atr_multiple": 0.0}
                ],
            }
        )
        specs = {spec.name for spec in _exit_feature_specs(preset)}
        assert specs == {"swing_5"}

    def test_atr_trail_needs_only_its_own_atr(self, library: ExitLibrarySpec) -> None:
        preset = next(item for item in library.presets if item.id == "atr_trail_aggressive")
        specs = {spec.name for spec in _exit_feature_specs(preset)}
        assert specs == {"atr_10"}

    def test_swing_trail_needs_its_swing_and_no_atr_when_unbuffered(
        self, library: ExitLibrarySpec
    ) -> None:
        preset = next(item for item in library.presets if item.id == "swing_partial_ladder")
        specs = {spec.name for spec in _exit_feature_specs(preset)}
        assert specs == {"swing_5"}

    def test_rules_with_no_feature_contribute_nothing(self, library: ExitLibrarySpec) -> None:
        for preset_id in ("conservative_2r", "scalp_quick", "session_close", "time_boxed"):
            preset = next(item for item in library.presets if item.id == preset_id)
            assert _exit_feature_specs(preset) == []

    def test_the_merge_prefers_the_entry_specs_own_version_on_a_name_collision(self) -> None:
        """Entry and exit both wanting atr_14 must not compute it twice either."""
        from trading_system.features.pipeline import FeatureSpec

        entry_specs = (FeatureSpec(name="atr_14", kind="atr", params={"period": 14}),)
        preset = ExitPresetSpec.model_validate(
            {
                "id": "shares-atr",
                "name": "shares-atr",
                "protective_stop": {"kind": "PROTECTIVE_STOP"},
                "stop_modifiers": [{"kind": "ATR_STOP", "period": 14, "multiple": 1.5}],
            }
        )
        merged = _merged_feature_specs(entry_specs, [preset])
        names = [spec.name for spec in merged]
        assert names.count("atr_14") == 1


class TestEmaPullbackExample:
    """One of the shipped example strategies, indicator-driven trigger and all.

    Before the feature pipeline was wired per stream, this crashed on the first
    bar the trigger was evaluated on: the strategy's ``BarSeries`` carried
    prices only, so ``ctx.feature("ema_50")`` raised ``ValidationError`` rather
    than returning a value or ``None``. This exercises the whole chain — entry
    trigger, confirmation, invalidation, quality modifier, and (through
    ``structure_trail``) the exit side's own ATR and swing columns — end to end.
    """

    @pytest.fixture
    def spec(self) -> StrategySpec:
        return StrategySpec.model_validate_json((EXAMPLES / "ema_pullback.json").read_text())

    @pytest.fixture
    def structure_trail(self, library: ExitLibrarySpec) -> ExitPresetSpec:
        return next(item for item in library.presets if item.id == "structure_trail")

    def test_it_runs_to_completion_without_a_feature_resolution_error(
        self,
        registry: InstrumentRegistry,
        spec: StrategySpec,
        structure_trail: ExitPresetSpec,
    ) -> None:
        import math

        key = StreamKey("EURUSD", Timeframe.H4)
        start = datetime(2024, 1, 1, tzinfo=UTC)
        rows = []
        price = 1.1000
        for index in range(260):
            close = 1.1000 + 0.00025 * index + 0.0060 * math.sin(index / 9.0)
            rows.append((start + timedelta(hours=4 * index), price, close))
            price = close
        frame = bars(
            [close for _, _, close in rows],
            symbol="EURUSD",
            timeframe=Timeframe.H4,
            start=start,
            step=timedelta(hours=4),
            spread=0.0006,
        )

        result = orchestrator(
            registry=registry,
            streams={key: frame},
            bindings=[StrategyBinding(spec=spec, exit_preset=structure_trail, keys=(key,))],
            config=BacktestConfig(
                atr_period=14, atr_baseline_bars=20, starting_balance=Decimal(100_000)
            ),
        ).run()

        assert result.instants == 260
        assert result.entry_drops == {} or all(v == 0 for v in result.entry_drops.values())
        # The point of the scenario: it actually reached the point of placing
        # and filling orders, not merely surviving without raising.
        assert result.fills + result.expired_orders > 0
