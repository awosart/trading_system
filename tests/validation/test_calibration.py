"""Calibration: deterministic seeding, idempotence, and a real percentile against real nulls."""

from decimal import Decimal
from pathlib import Path

import pytest

from tests.backtest.conftest import EURUSD_H4, ema_pullback, harness_inputs, swing_series
from trading_system.backtest.orchestrator import BacktestResult, StrategyBinding
from trading_system.backtest.spec import RunInputs
from trading_system.core.instruments import InstrumentRegistry
from trading_system.core.types import Timeframe
from trading_system.data.resample import FX_DAY_ORIGIN
from trading_system.execution.config import CostConfig
from trading_system.exit.library import ExitLibrarySpec
from trading_system.validation.calibration import (
    POSITION_COUNT_DIVERGENCE_THRESHOLD,
    CalibrationResult,
    NullKind,
    iteration_seed,
    run_calibration,
)
from trading_system.validation.nulls.random_entry import (
    EntryTraceProfile,
    build_entry_trace_profile,
    real_signals,
)
from trading_system.validation.objective import SortinoTimesSqrtTrades

LENGTH = 2000

#: (base, binding, real result, profile) — one real run, shared read-only
#: across the tests below.
RealRun = tuple[RunInputs, StrategyBinding, BacktestResult, EntryTraceProfile]


class TestIterationSeed:
    def test_deterministic(self) -> None:
        assert iteration_seed("cal-1", 5) == iteration_seed("cal-1", 5)

    def test_differs_by_iteration(self) -> None:
        assert iteration_seed("cal-1", 5) != iteration_seed("cal-1", 6)

    def test_differs_by_calibration_id(self) -> None:
        assert iteration_seed("cal-1", 5) != iteration_seed("cal-2", 5)

    def test_negative_one_is_reserved_and_still_deterministic(self) -> None:
        assert iteration_seed("cal-1", -1) == iteration_seed("cal-1", -1)
        assert iteration_seed("cal-1", -1) != iteration_seed("cal-1", 0)


@pytest.fixture(scope="module")
def real_run(registry: InstrumentRegistry, library: ExitLibrarySpec) -> RealRun:
    spec = ema_pullback()
    preset = next(item for item in library.presets if item.id == spec.exit_ref)
    binding = StrategyBinding(spec=spec, exit_preset=preset, keys=(EURUSD_H4,))
    base = harness_inputs(registry, streams={EURUSD_H4: swing_series(LENGTH)}, bindings=[binding])
    result = base.run()
    signals = real_signals(base.streams, binding, EURUSD_H4)
    profile = build_entry_trace_profile(signals, result.trades, EURUSD_H4.timeframe)
    return base, binding, result, profile


class TestRunCalibrationRandomEntry:
    def test_produces_n_records(self, real_run: RealRun, tmp_path: Path) -> None:
        base, binding, result, profile = real_run
        cal = run_calibration(
            NullKind.RANDOM_ENTRY,
            base,
            objective=SortinoTimesSqrtTrades(),
            real_result=result,
            n=6,
            calibration_id="test-random-entry-n",
            store_root=tmp_path,
            key=EURUSD_H4,
            real_binding=binding,
            profile=profile,
        )
        assert cal.n == 6
        assert len(cal.records) == 6
        assert [r.iteration for r in cal.records] == list(range(6))

    def test_real_percentile_is_computed(self, real_run: RealRun, tmp_path: Path) -> None:
        base, binding, result, profile = real_run
        cal = run_calibration(
            NullKind.RANDOM_ENTRY,
            base,
            objective=SortinoTimesSqrtTrades(),
            real_result=result,
            n=6,
            calibration_id="test-random-entry-percentile",
            store_root=tmp_path,
            key=EURUSD_H4,
            real_binding=binding,
            profile=profile,
        )
        assert cal.real_score is not None
        assert cal.percentile is not None
        assert 0.0 <= cal.percentile <= 100.0

    def test_median_ci_brackets_the_median(self, real_run: RealRun, tmp_path: Path) -> None:
        base, binding, result, profile = real_run
        cal = run_calibration(
            NullKind.RANDOM_ENTRY,
            base,
            objective=SortinoTimesSqrtTrades(),
            real_result=result,
            n=10,
            calibration_id="test-random-entry-ci",
            store_root=tmp_path,
            key=EURUSD_H4,
            real_binding=binding,
            profile=profile,
        )
        assert cal.median_score is not None
        assert cal.median_ci_low is not None
        assert cal.median_ci_high is not None
        assert cal.median_ci_low <= cal.median_score <= cal.median_ci_high

    def test_repeated_calibration_is_a_bit_for_bit_no_op(
        self, real_run: RealRun, tmp_path: Path
    ) -> None:
        base, binding, result, profile = real_run

        def run() -> CalibrationResult:
            return run_calibration(
                NullKind.RANDOM_ENTRY,
                base,
                objective=SortinoTimesSqrtTrades(),
                real_result=result,
                n=5,
                calibration_id="test-random-entry-idempotent",
                store_root=tmp_path,
                key=EURUSD_H4,
                real_binding=binding,
                profile=profile,
            )

        first = run()
        written = {
            path: path.stat().st_mtime_ns for path in first.directory.rglob("*") if path.is_file()
        }
        second = run()
        still = {
            path: path.stat().st_mtime_ns for path in first.directory.rglob("*") if path.is_file()
        }

        assert second.records == first.records
        assert second.percentile == first.percentile
        assert still == written

    def test_position_count_divergence_is_finite_and_computable(
        self, real_run: RealRun, tmp_path: Path
    ) -> None:
        base, binding, result, profile = real_run
        cal = run_calibration(
            NullKind.RANDOM_ENTRY,
            base,
            objective=SortinoTimesSqrtTrades(),
            real_result=result,
            n=6,
            calibration_id="test-random-entry-divergence",
            store_root=tmp_path,
            key=EURUSD_H4,
            real_binding=binding,
            profile=profile,
        )
        assert cal.real_trade_count == len(result.trades)
        divergence = cal.position_count_divergence
        assert divergence is not None
        assert divergence >= 0.0
        # Not asserting it stays under POSITION_COUNT_DIVERGENCE_THRESHOLD here —
        # that is a property of the specific fixture's data, not of the
        # calibration mechanism; the threshold constant exists for a caller
        # (the CLI, a report) to compare against.
        assert isinstance(POSITION_COUNT_DIVERGENCE_THRESHOLD, float)


class TestRunCalibrationFixedHold:
    def test_produces_scored_records(self, real_run: RealRun, tmp_path: Path) -> None:
        base, _binding, result, profile = real_run
        cal = run_calibration(
            NullKind.RANDOM_ENTRY_FIXED_HOLD,
            base,
            objective=SortinoTimesSqrtTrades(),
            real_result=result,
            n=6,
            calibration_id="test-fixed-hold",
            store_root=tmp_path,
            key=EURUSD_H4,
            profile=profile,
        )
        assert cal.n_scored > 0

    def test_differs_from_the_plain_random_entry_null(
        self, real_run: RealRun, tmp_path: Path
    ) -> None:
        """The two null variants isolate different things and are not the same calibration."""
        base, binding, result, profile = real_run
        objective = SortinoTimesSqrtTrades()
        plain = run_calibration(
            NullKind.RANDOM_ENTRY,
            base,
            objective=objective,
            real_result=result,
            n=6,
            calibration_id="test-compare-plain",
            store_root=tmp_path,
            key=EURUSD_H4,
            real_binding=binding,
            profile=profile,
        )
        fixed = run_calibration(
            NullKind.RANDOM_ENTRY_FIXED_HOLD,
            base,
            objective=objective,
            real_result=result,
            n=6,
            calibration_id="test-compare-fixed",
            store_root=tmp_path,
            key=EURUSD_H4,
            profile=profile,
        )
        plain_scores = [r.score for r in plain.records]
        fixed_scores = [r.score for r in fixed.records]
        assert plain_scores != fixed_scores


class TestRunCalibrationPermutation:
    def test_produces_scored_records_on_zero_cost(
        self, registry: InstrumentRegistry, library: ExitLibrarySpec, tmp_path: Path
    ) -> None:
        spec = ema_pullback()
        preset = next(item for item in library.presets if item.id == spec.exit_ref)
        zero_registry = InstrumentRegistry(
            {
                **{s: registry[s] for s in registry.symbols},
                "EURUSD": registry["EURUSD"].model_copy(
                    update={"typical_spread_points": 0.0, "commission_per_lot": Decimal(0)}
                ),
            }
        )
        base = harness_inputs(
            zero_registry,
            streams={EURUSD_H4: swing_series(LENGTH)},
            bindings=[StrategyBinding(spec=spec, exit_preset=preset, keys=(EURUSD_H4,))],
        )
        zero_cost_base = base.__class__(
            **{
                **{f: getattr(base, f) for f in base.__dataclass_fields__},
                "costs": CostConfig(run_seed=base.costs.run_seed),
            }
        )
        real_result = zero_cost_base.run()

        cal = run_calibration(
            NullKind.PERMUTATION,
            zero_cost_base,
            objective=SortinoTimesSqrtTrades(),
            real_result=real_result,
            n=6,
            calibration_id="test-permutation",
            store_root=tmp_path,
            finest=Timeframe.H4,
            day_origin=FX_DAY_ORIGIN,
        )
        assert cal.n_scored > 0
        assert cal.kind is NullKind.PERMUTATION


class TestContract:
    def test_permutation_without_finest_is_rejected(
        self, real_run: RealRun, tmp_path: Path
    ) -> None:
        base, _binding, result, _profile = real_run
        with pytest.raises(ValueError, match="finest"):
            run_calibration(
                NullKind.PERMUTATION,
                base,
                objective=SortinoTimesSqrtTrades(),
                real_result=result,
                n=2,
                calibration_id="test-missing-finest",
                store_root=tmp_path,
            )

    def test_random_entry_without_profile_is_rejected(
        self, real_run: RealRun, tmp_path: Path
    ) -> None:
        base, binding, result, _profile = real_run
        with pytest.raises(ValueError, match="profile"):
            run_calibration(
                NullKind.RANDOM_ENTRY,
                base,
                objective=SortinoTimesSqrtTrades(),
                real_result=result,
                n=2,
                calibration_id="test-missing-profile",
                store_root=tmp_path,
                key=EURUSD_H4,
                real_binding=binding,
            )


class TestResultIsAPlainDataclass:
    def test_returned_type(self, real_run: RealRun, tmp_path: Path) -> None:
        base, binding, result, profile = real_run
        cal = run_calibration(
            NullKind.RANDOM_ENTRY,
            base,
            objective=SortinoTimesSqrtTrades(),
            real_result=result,
            n=3,
            calibration_id="test-type",
            store_root=tmp_path,
            key=EURUSD_H4,
            real_binding=binding,
            profile=profile,
        )
        assert isinstance(cal, CalibrationResult)
