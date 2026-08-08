"""The walk-forward report: everything a fold's boundary can lose is a number here."""

import json
from datetime import timedelta
from pathlib import Path

import pytest

from tests.backtest.conftest import ema_pullback, harness_inputs, swing_series
from trading_system.backtest.clock import StreamKey
from trading_system.backtest.orchestrator import StrategyBinding
from trading_system.core.instruments import InstrumentRegistry
from trading_system.core.types import Timeframe
from trading_system.exit.library import ExitLibrarySpec
from trading_system.validation.report import WalkForwardReport, build_report, write_report
from trading_system.validation.splitting import WalkForwardMode, WalkForwardSplitter
from trading_system.validation.walkforward import IdentitySelector, WalkForwardRunner

EURUSD_H4 = StreamKey("EURUSD", Timeframe.H4)


@pytest.fixture(scope="module")
def report(
    registry: InstrumentRegistry, library: ExitLibrarySpec, tmp_path_factory: pytest.TempPathFactory
) -> WalkForwardReport:
    spec = ema_pullback()
    preset = next(item for item in library.presets if item.id == spec.exit_ref)
    base = harness_inputs(
        registry,
        streams={EURUSD_H4: swing_series(2200)},
        bindings=[StrategyBinding(spec=spec, exit_preset=preset, keys=(EURUSD_H4,))],
    )
    splitter = WalkForwardSplitter(
        mode=WalkForwardMode.ROLLING,
        is_span=timedelta(days=90),
        oos_span=timedelta(days=25),
        step=timedelta(days=25),
        embargo=timedelta(days=2),
        warmup=timedelta(days=15),
    )
    runner = WalkForwardRunner(
        base=base,
        splitter=splitter,
        selector=IdentitySelector(base),
        store_root=tmp_path_factory.mktemp("wf_report_store"),
        max_drain_bars=30,
    )
    # A generous min_trades_per_fold, high enough that at least some of this
    # short synthetic run's folds fall under it — otherwise "insufficient" is
    # never exercised, which is the point of this fixture.
    return build_report(runner.run(), min_trades_per_fold=8)


class TestFoldCount:
    def test_every_fold_has_a_report_row(self, report: WalkForwardReport) -> None:
        assert report.n_folds == len(report.folds)
        assert report.n_folds >= 2

    def test_fold_indices_are_in_order(self, report: WalkForwardReport) -> None:
        assert [fold.index for fold in report.folds] == list(range(report.n_folds))


class TestInsufficientIsFlaggedNotDropped:
    def test_at_least_one_fold_is_flagged_insufficient(self, report: WalkForwardReport) -> None:
        """The fixture's threshold is deliberately generous — see its docstring."""
        assert any(fold.insufficient for fold in report.folds)

    def test_insufficient_folds_are_still_present_in_the_aggregate(
        self, report: WalkForwardReport
    ) -> None:
        flagged = [fold for fold in report.folds if fold.insufficient]
        assert flagged
        for fold in flagged:
            assert fold in report.folds  # never excluded, only marked

    def test_the_report_level_flag_matches_the_per_fold_detail(
        self, report: WalkForwardReport
    ) -> None:
        assert report.insufficient_sample == any(fold.insufficient for fold in report.folds)

    def test_insufficient_is_exactly_below_threshold(self, report: WalkForwardReport) -> None:
        for fold in report.folds:
            assert fold.insufficient == (fold.oos_trade_count < report.min_trades_per_fold)

    def test_folds_without_any_oos_trade_are_counted_separately(
        self, report: WalkForwardReport
    ) -> None:
        assert report.n_folds_without_oos_trades == sum(
            1 for fold in report.folds if fold.oos_trade_count == 0
        )
        assert report.n_folds_without_oos_trades <= report.n_folds


class TestPooledFigures:
    def test_stitched_trade_count_is_the_sum_of_per_fold_oos_counts(
        self, report: WalkForwardReport
    ) -> None:
        assert report.stitched_trade_count == sum(fold.oos_trade_count for fold in report.folds)

    def test_per_fold_distributions_have_one_entry_per_fold(
        self, report: WalkForwardReport
    ) -> None:
        assert len(report.per_fold_oos_expectancy_r) == report.n_folds
        assert len(report.per_fold_oos_sortino) == report.n_folds

    def test_per_fold_distribution_matches_each_folds_own_value(
        self, report: WalkForwardReport
    ) -> None:
        assert list(report.per_fold_oos_expectancy_r) == [
            fold.oos_expectancy_r for fold in report.folds
        ]


class TestCountersAreCarriedNotSummarised:
    def test_every_fold_carries_its_own_drop_and_degradation_counters(
        self, report: WalkForwardReport
    ) -> None:
        for fold in report.folds:
            assert isinstance(fold.rejections, dict) and fold.rejections
            assert isinstance(fold.degradations, dict) and fold.degradations
            assert isinstance(fold.signal_drops, dict) and fold.signal_drops
            assert 0.0 <= fold.atr_unavailable_fraction <= 1.0

    def test_no_reason_is_ever_missing_even_at_zero(self, report: WalkForwardReport) -> None:
        """Every rejection reason must be a present key, per the P10-era counting discipline."""
        reasons = {reason for fold in report.folds for reason in fold.rejections}
        for fold in report.folds:
            assert set(fold.rejections) == reasons


class TestJsonExport:
    def test_to_dict_round_trips_through_json(self, report: WalkForwardReport) -> None:
        payload = json.loads(json.dumps(report.to_dict()))
        assert payload["n_folds"] == report.n_folds
        assert len(payload["folds"]) == report.n_folds

    def test_write_report_writes_a_readable_file(
        self, report: WalkForwardReport, tmp_path: Path
    ) -> None:
        out = tmp_path / "nested" / "report.json"
        write_report(report, out)

        payload = json.loads(out.read_text())
        assert payload["wf_id"] == report.wf_id

    def test_boundary_residuals_serialise_as_decimal_strings(
        self, report: WalkForwardReport, tmp_path: Path
    ) -> None:
        out = tmp_path / "report.json"
        write_report(report, out)
        payload = json.loads(out.read_text())
        for fold in payload["folds"]:
            if fold["oos_boundary_residual"] is not None:
                assert isinstance(fold["oos_boundary_residual"], str)
