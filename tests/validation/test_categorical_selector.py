"""The selector under a categorical axis, through the real engine.

Two things are checked here that no unit test of ``SearchSpace`` can reach: that
a chosen exit actually arrives in the out-of-sample run, and that a sizing axis
reaching past the engine's own cap is refused before a fold is searched.

The first is the defect this stage was most exposed to. ``_with_params`` used to
write only the strategy spec, which is correct for every space that existed
before this stage and silently wrong for one with an exit axis: the search would
optimise eight exits and the out-of-sample run would trade the template's. The
report would look entirely normal.
"""

from datetime import timedelta
from pathlib import Path

import pytest

from tests.backtest.conftest import ema_pullback, harness_inputs, swing_series
from trading_system.backtest.clock import StreamKey
from trading_system.backtest.orchestrator import StrategyBinding
from trading_system.backtest.spec import RunInputs
from trading_system.core.instruments import InstrumentRegistry
from trading_system.core.types import Timeframe
from trading_system.data.resample import FX_DAY_ORIGIN
from trading_system.exit.library import ExitLibrarySpec
from trading_system.risk.sizing.config import FixedFractionalConfig
from trading_system.validation.objective import SortinoTimesSqrtTrades
from trading_system.validation.optimization import (
    AxisTarget,
    GridSearch,
    ParameterAxis,
    SearchSpace,
    VariationTargets,
)
from trading_system.validation.splitting import WalkForwardMode, WalkForwardSplitter
from trading_system.validation.walkforward import OptimizingSelector

EURUSD_H4 = StreamKey("EURUSD", Timeframe.H4)
EXITS = ("conservative_2r", "breakeven_runner", "scalp_quick")


def _base(registry: InstrumentRegistry, library: ExitLibrarySpec, length: int = 900) -> RunInputs:
    spec = ema_pullback()
    preset = next(item for item in library.presets if item.id == spec.exit_ref)
    return harness_inputs(
        registry,
        streams={EURUSD_H4: swing_series(length)},
        bindings=[StrategyBinding(spec=spec, exit_preset=preset, keys=(EURUSD_H4,))],
    )


def _libraries(library: ExitLibrarySpec, base: RunInputs) -> VariationTargets:
    binding = base.bindings[0]
    return VariationTargets(
        spec=binding.spec,
        exit_preset=binding.exit_preset,
        sizing=FixedFractionalConfig(risk_pct=0.01),
        exit_library={item.id: item for item in library.presets},
        sizing_library={
            "half": FixedFractionalConfig(risk_pct=0.005),
            "full": FixedFractionalConfig(risk_pct=0.01),
            "over_cap": FixedFractionalConfig(risk_pct=0.05),
        },
    )


def _exit_space() -> SearchSpace:
    return SearchSpace(
        axes=(
            ParameterAxis(name="exit", target=AxisTarget.EXIT_PRESET, values=EXITS),
            ParameterAxis(
                name="ema_slow",
                paths=(
                    "/entries/0/trigger/conditions/0/right/params/period",
                    "/entries/0/invalidation/price_level/params/period",
                ),
                values=(30, 50),
            ),
        )
    )


def _fold(base: RunInputs):
    splitter = WalkForwardSplitter(
        mode=WalkForwardMode.ROLLING,
        is_span=timedelta(days=60),
        oos_span=timedelta(days=20),
        step=timedelta(days=20),
        embargo=timedelta(days=2),
        warmup=timedelta(days=10),
    )
    frame = base.streams[EURUSD_H4]
    assert frame.start is not None and frame.end is not None
    return splitter.split((frame.start, frame.end), day_origin=FX_DAY_ORIGIN)[0]


class TestTheChosenExitReachesTheOutOfSampleRun:
    """Optimising one exit and trading another would look completely normal."""

    def test_the_returned_binding_carries_the_selected_preset(
        self, registry: InstrumentRegistry, library: ExitLibrarySpec, tmp_path: Path
    ) -> None:
        base = _base(registry, library)
        space = _exit_space()
        selector = OptimizingSelector(
            base=base,
            space=space,
            search=GridSearch(),
            objective=SortinoTimesSqrtTrades(),
            trial_budget=space.feasible_size(),
            store_root=tmp_path,
            libraries=_libraries(library, base),
        )
        chosen = selector.select(_fold(base), is_result=None)  # type: ignore[arg-type]
        binding = chosen.bindings[0]
        assert binding.exit_preset.id in EXITS
        assert binding.spec.exit_ref == binding.exit_preset.id, (
            "the recorded strategy must name the exit the run actually executed"
        )

    def test_every_reachable_point_produces_a_consistent_pair(
        self, registry: InstrumentRegistry, library: ExitLibrarySpec, tmp_path: Path
    ) -> None:
        base = _base(registry, library)
        space = _exit_space()
        selector = OptimizingSelector(
            base=base,
            space=space,
            search=GridSearch(),
            objective=SortinoTimesSqrtTrades(),
            trial_budget=space.feasible_size(),
            store_root=tmp_path,
            libraries=_libraries(library, base),
        )
        seen = set()
        for point in space.enumerate():
            inputs = selector._with_params(point)
            binding = inputs.bindings[0]
            assert binding.spec.exit_ref == binding.exit_preset.id
            seen.add(binding.exit_preset.id)
        assert seen == set(EXITS)


class TestASizingAxisMayNotReachPastTheEnginesCap:
    """Above the cap every value is trimmed to the same run — a manufactured plateau."""

    def _selector(
        self,
        registry: InstrumentRegistry,
        library: ExitLibrarySpec,
        tmp_path: Path,
        values: tuple[str, ...],
    ) -> OptimizingSelector:
        base = _base(registry, library, length=400)
        return OptimizingSelector(
            base=base,
            space=SearchSpace(
                axes=(ParameterAxis(name="sizing", target=AxisTarget.SIZING_METHOD, values=values),)
            ),
            search=GridSearch(),
            objective=SortinoTimesSqrtTrades(),
            trial_budget=len(values),
            store_root=tmp_path,
            libraries=_libraries(library, base),
        )

    def test_a_variant_above_max_risk_pct_is_refused_at_construction(
        self, registry: InstrumentRegistry, library: ExitLibrarySpec, tmp_path: Path
    ) -> None:
        with pytest.raises(ValueError, match="above the engine's max_risk_pct"):
            self._selector(registry, library, tmp_path, ("half", "over_cap"))

    def test_variants_at_or_below_the_cap_are_accepted(
        self, registry: InstrumentRegistry, library: ExitLibrarySpec, tmp_path: Path
    ) -> None:
        selector = self._selector(registry, library, tmp_path, ("half", "full"))
        assert selector.space.categorical_mask == (True,)

    def test_a_run_axis_is_checked_the_same_way(
        self, registry: InstrumentRegistry, library: ExitLibrarySpec, tmp_path: Path
    ) -> None:
        base = _base(registry, library, length=400)
        with pytest.raises(ValueError, match="above the engine's max_risk_pct"):
            OptimizingSelector(
                base=base,
                space=SearchSpace(
                    axes=(
                        ParameterAxis(
                            name="risk",
                            target=AxisTarget.RUN,
                            paths=("/sizing/risk_pct",),
                            values=(0.01, 0.02, 0.03),
                        ),
                    )
                ),
                search=GridSearch(),
                objective=SortinoTimesSqrtTrades(),
                trial_budget=3,
                store_root=tmp_path,
                libraries=_libraries(library, base),
            )


class TestTheSelectorRefusesWhatItCannotResolve:
    """A misconfigured search must fail before it spends a fold's budget."""

    def test_a_categorical_space_without_libraries_is_refused(
        self, registry: InstrumentRegistry, library: ExitLibrarySpec, tmp_path: Path
    ) -> None:
        base = _base(registry, library)
        selector = OptimizingSelector(
            base=base,
            space=_exit_space(),
            search=GridSearch(),
            objective=SortinoTimesSqrtTrades(),
            trial_budget=6,
            store_root=tmp_path,
        )
        with pytest.raises(ValueError, match="no libraries to resolve them against"):
            selector.select(_fold(base), is_result=None)  # type: ignore[arg-type]

    def test_two_categorical_axes_are_refused_at_construction(
        self, registry: InstrumentRegistry, library: ExitLibrarySpec, tmp_path: Path
    ) -> None:
        base = _base(registry, library, length=400)
        with pytest.raises(ValueError, match="one categorical axis"):
            OptimizingSelector(
                base=base,
                space=SearchSpace(
                    axes=(
                        ParameterAxis(name="exit", target=AxisTarget.EXIT_PRESET, values=EXITS),
                        ParameterAxis(
                            name="sizing",
                            target=AxisTarget.SIZING_METHOD,
                            values=("half", "full"),
                        ),
                    )
                ),
                search=GridSearch(),
                objective=SortinoTimesSqrtTrades(),
                trial_budget=6,
                store_root=tmp_path,
                libraries=_libraries(library, base),
            )


class TestTheSelectorsIdentityFollowsWhatChangesItsChoice:
    """A knob that moves the selection and not the id is the stage 2 defect, repeated."""

    def _selector(self, base: RunInputs, library: ExitLibrarySpec, tmp_path: Path, **kwargs):
        return OptimizingSelector(
            base=base,
            space=_exit_space(),
            search=GridSearch(),
            objective=SortinoTimesSqrtTrades(),
            trial_budget=6,
            store_root=tmp_path,
            libraries=_libraries(library, base),
            **kwargs,
        )

    def test_turning_the_baseline_correction_on_changes_the_key(
        self, registry: InstrumentRegistry, library: ExitLibrarySpec, tmp_path: Path
    ) -> None:
        from trading_system.validation.null_baseline import BaselineRequest

        base = _base(registry, library, length=400)
        plain = self._selector(base, library, tmp_path)
        corrected = self._selector(base, library, tmp_path, baseline_request=BaselineRequest())
        assert plain.key() != corrected.key()

    def test_a_different_exit_library_changes_the_key(
        self, registry: InstrumentRegistry, library: ExitLibrarySpec, tmp_path: Path
    ) -> None:
        base = _base(registry, library, length=400)
        wide = self._selector(base, library, tmp_path)
        narrow = OptimizingSelector(
            base=base,
            space=_exit_space(),
            search=GridSearch(),
            objective=SortinoTimesSqrtTrades(),
            trial_budget=6,
            store_root=tmp_path,
            libraries=VariationTargets(
                spec=base.bindings[0].spec,
                exit_preset=base.bindings[0].exit_preset,
                sizing=FixedFractionalConfig(risk_pct=0.01),
                exit_library={item.id: item for item in library.presets if item.id in EXITS},
            ),
        )
        assert wide.key() != narrow.key()


class TestAFoldIsRecordedOnceAndItsCalibrationCounted:
    """The bug the first real run found: the ledger was written twice for one fold.

    ``TrialLedger.record`` refuses a second entry per fold, which is exactly what
    it is for — a double count would misstate what a search cost. The
    calibration's backtests therefore have to be *added* to the fold's run
    count, not recorded separately. They are not a rounding error: on the
    measured run they were 1144 against the search's own 96.
    """

    def test_a_calibrated_fold_records_once_and_includes_the_calibration_runs(
        self, registry: InstrumentRegistry, library: ExitLibrarySpec, tmp_path: Path
    ) -> None:
        from trading_system.validation.null_baseline import BaselineRequest, read_calibration

        base = _base(registry, library)
        space = _exit_space()
        selector = OptimizingSelector(
            base=base,
            space=space,
            search=GridSearch(),
            objective=SortinoTimesSqrtTrades(),
            trial_budget=space.feasible_size(),
            store_root=tmp_path,
            libraries=_libraries(library, base),
            baseline_request=BaselineRequest(min_seeds=2, max_seeds=4),
        )
        fold = _fold(base)
        selector.select(fold, is_result=None)  # type: ignore[arg-type]

        calibration = read_calibration(selector.fold_dir(fold.index))
        assert calibration is not None
        outcome = selector.outcomes[fold.index]
        assert selector.ledger.per_fold[fold.index] == outcome.n_trials
        assert selector.ledger.runs_per_fold[fold.index] == outcome.n_runs + calibration.runs
        assert calibration.runs > 0

    def test_a_resumed_calibrated_fold_reports_the_same_run_count(
        self, registry: InstrumentRegistry, library: ExitLibrarySpec, tmp_path: Path
    ) -> None:
        """Resuming must not lose the calibration's cost, or the ledger disagrees with itself."""
        from trading_system.validation.null_baseline import BaselineRequest

        base = _base(registry, library)
        space = _exit_space()

        def build() -> OptimizingSelector:
            return OptimizingSelector(
                base=base,
                space=space,
                search=GridSearch(),
                objective=SortinoTimesSqrtTrades(),
                trial_budget=space.feasible_size(),
                store_root=tmp_path,
                libraries=_libraries(library, base),
                baseline_request=BaselineRequest(min_seeds=2, max_seeds=4),
            )

        fold = _fold(base)
        first = build()
        first.select(fold, is_result=None)  # type: ignore[arg-type]
        second = build()
        second.select(fold, is_result=None)  # type: ignore[arg-type]
        assert second.ledger.runs_per_fold == first.ledger.runs_per_fold
        assert second.ledger.per_fold == first.ledger.per_fold


class TestTheCalibrationPublishesWhatItCannotVouchFor:
    """The reuse assumption is monitored, not asserted."""

    def test_every_baseline_reports_its_trade_count_alongside_the_spread(
        self, registry: InstrumentRegistry, library: ExitLibrarySpec, tmp_path: Path
    ) -> None:
        from trading_system.validation.null_baseline import BaselineRequest, read_calibration

        base = _base(registry, library)
        space = _exit_space()
        selector = OptimizingSelector(
            base=base,
            space=space,
            search=GridSearch(),
            objective=SortinoTimesSqrtTrades(),
            trial_budget=space.feasible_size(),
            store_root=tmp_path,
            libraries=_libraries(library, base),
            baseline_request=BaselineRequest(min_seeds=2, max_seeds=4),
        )
        fold = _fold(base)
        selector.select(fold, is_result=None)  # type: ignore[arg-type]
        calibration = read_calibration(selector.fold_dir(fold.index))
        assert calibration is not None
        for item in calibration.baselines:
            assert item.trade_count_spread >= 1.0
            assert item.median_trade_count >= 0.0
            assert item.stopped_because in {"target", "budget", "unscoreable"}
