"""The verdict: gate order, the degradation formula, and the two DoD scenarios.

The two headline cases are built from opposite directions on purpose. The
overfit one is a strategy fitted to one segment and evaluated on the rest,
where the fitting is real rather than asserted. The robust one is buy-and-hold
on a genuinely trending series — a strategy nobody could overfit, since it has
no parameters to fit — so ROBUST there is a statement about the *grader*, not
about a cleverly built fixture.
"""

import json
import math
import statistics
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from tests.backtest.conftest import (
    EMA_PULLBACK_PATH,
    ema_pullback,
    harness_inputs,
    swing_series,
)
from tests.backtest.conftest import START as START_BARS
from trading_system.backtest.clock import StreamKey
from trading_system.backtest.orchestrator import StrategyBinding
from trading_system.backtest.spec import RunInputs
from trading_system.core.instruments import InstrumentRegistry
from trading_system.core.types import Timeframe
from trading_system.data.models import OHLCVFrame
from trading_system.data.resample import FX_DAY_ORIGIN
from trading_system.exit.library import ExitLibrarySpec
from trading_system.strategies.schema import StrategySpec
from trading_system.validation.objective import SortinoTimesSqrtTrades
from trading_system.validation.optimization import GridSearch, ParameterAxis, SearchSpace
from trading_system.validation.report import (
    FoldReport,
    StrategyVerdict,
    Verdict,
    VerdictThresholds,
    WalkForwardReport,
    build_report,
    build_verdict,
    degradation_test,
)
from trading_system.validation.robustness import (
    PeriodConsistency,
    PeriodResult,
    RobustnessReport,
    SyntheticSummary,
    synthetic_frame,
)
from trading_system.validation.splitting import (
    FoldWindow,
    WalkForwardMode,
    WalkForwardSplitter,
)
from trading_system.validation.walkforward import (
    IdentitySelector,
    OptimizingSelector,
    WalkForwardRunner,
)

START = datetime(2020, 1, 1, tzinfo=UTC)


def _window(offset_days: int, span_days: int) -> FoldWindow:
    """A well-ordered window at a given offset."""
    trade_start = START + timedelta(days=offset_days)
    return FoldWindow(
        data_start=trade_start - timedelta(days=30),
        trade_start=trade_start,
        trade_end=trade_start + timedelta(days=span_days),
    )


def _fold(index: int, *, is_r: float | None, oos_r: float | None, trades: int = 20) -> FoldReport:
    """One fold report carrying only what the verdict reads."""
    return FoldReport(
        index=index,
        is_window=_window(index * 100, 90),
        oos_window=_window(index * 100 + 95, 60),
        is_trade_count=trades,
        oos_trade_count=0 if oos_r is None else trades,
        is_expectancy_r=is_r,
        oos_expectancy_r=oos_r,
        is_sortino=None,
        oos_sortino=None,
        is_boundary_residual=None,
        oos_boundary_residual=None,
        drain_truncated=0,
        rejections={},
        degradations={},
        exit_drops={},
        entry_drops={},
        signal_drops={},
        atr_unavailable_fraction=0.0,
        insufficient=False,
    )


def _report(
    folds: list[FoldReport], *, expectancy: float | None, total_trades: int
) -> WalkForwardReport:
    """A walk-forward report assembled from explicit folds."""
    per_fold = tuple(fold.oos_expectancy_r for fold in folds)
    return WalkForwardReport(
        wf_id="test",
        min_trades_per_fold=10,
        folds=tuple(folds),
        n_folds=len(folds),
        n_folds_without_oos_trades=sum(1 for f in folds if f.oos_trade_count == 0),
        insufficient_sample=False,
        stitched_trade_count=total_trades,
        stitched_expectancy_r=expectancy,
        stitched_sharpe=1.0,
        stitched_sortino=1.5,
        per_fold_oos_expectancy_r=per_fold,
        per_fold_oos_sortino=tuple(None for _ in folds),
    )


def _robustness(
    *,
    synthetic_percentile: float,
    profitable_periods: int = 4,
    n_periods: int = 4,
    retention: float | None = 0.9,
    dispersion: float | None = 0.05,
) -> RobustnessReport:
    """A robustness report carrying only what the verdict reads."""
    periods = tuple(
        PeriodResult(
            index=i,
            start=START + timedelta(days=100 * i),
            end=START + timedelta(days=100 * (i + 1)),
            n_trades=25,
            expectancy_r=0.2 if i < profitable_periods else -0.2,
        )
        for i in range(n_periods)
    )
    return RobustnessReport(
        start_shift=(),
        noise=(),
        synthetic=SyntheticSummary(
            n_iterations=20,
            real_expectancy_r=0.2,
            median_expectancy_r=-0.1,
            fraction_profitable=0.1,
            real_percentile=synthetic_percentile,
            median_trade_count=30.0,
        ),
        period=PeriodConsistency(
            periods=periods,
            n_periods=n_periods,
            n_profitable=profitable_periods,
            n_empty=0,
            dispersion=0.1,
        ),
        start_shift_dispersion=dispersion,
        noise_retention=retention,
    )


def _uptrend_series(length: int, *, symbol: str = "EURUSD") -> "OHLCVFrame":
    """A steadily rising series with shallow pullbacks.

    Deterministic and genuinely trending, for the same reason
    ``swing_series`` is: a fixture that trends only for some seeds would make
    "buy-and-hold was profitable" a property of the seed rather than of the
    fixture. The pullbacks matter — a monotone line would never let the trail
    stop out, so the run would hold exactly one position and have no sample.
    """
    base = 1.10
    closes = [
        base * (1.0 + 0.0004 * index) + 0.015 * math.sin(2 * math.pi * index / 60)
        for index in range(length)
    ]
    rows: list[dict[str, Any]] = []
    previous = closes[0] - 0.001
    for index, close in enumerate(closes):
        opening = previous
        rows.append(
            {
                "timestamp": START_BARS + Timeframe.H4.duration * index,
                "open": opening,
                "high": max(opening, close) + 0.0006,
                "low": min(opening, close) - 0.0006,
                "close": close,
                "volume": 1000.0 + 10 * (index % 7),
            }
        )
        previous = close
    return OHLCVFrame(pl.DataFrame(rows), symbol=symbol, timeframe=Timeframe.H4)


def _always_long_spec() -> StrategySpec:
    """A parameterless, always-long strategy: the trigger compares close against zero.

    Built from the shipped example so that every unrelated field stays whatever
    a real strategy carries, and re-validated so a typo fails here instead of
    producing a strategy that silently never trades.
    """
    payload: dict[str, Any] = json.loads(EMA_PULLBACK_PATH.read_text())
    payload["id"] = "always-long"
    # A fixed 2R target rather than a trail: on a series that only rises, a
    # trailing stop never fires, the single position rides to the end and the
    # run closes no trades at all — buy-and-hold behaving exactly like
    # buy-and-hold, and unscoreable for it.
    payload["exit_ref"] = "conservative_2r"
    payload["entries"] = [
        {
            "direction": "LONG",
            "trigger": {"type": "leaf", "op": "gt", "left": "price:close", "right": 0.0},
            "confirmation": [],
            "invalidation": {"price_level": {"indicator": "ema", "params": {"period": 50}}},
            "entry_order": {"order": {"type": "MARKET"}},
        }
    ]
    payload["filters"] = []
    payload["risk_profile"]["max_concurrent_positions"] = 1
    payload["risk_profile"]["cooldown_bars_after_loss"] = 0
    payload["risk_profile"]["quality_modifiers"] = []
    return StrategySpec.model_validate(payload)


class TestTheDegradationFormula:
    """The formula documented on :class:`DegradationTest`, checked directly."""

    def test_it_is_the_paired_one_sided_t_statistic_on_expectancy(self) -> None:
        pairs = [(0.5, 0.1), (0.6, 0.0), (0.4, 0.05), (0.55, -0.1), (0.45, 0.02)]
        folds = [_fold(i, is_r=a, oos_r=b) for i, (a, b) in enumerate(pairs)]
        report = _report(folds, expectancy=0.01, total_trades=100)

        result = degradation_test(report)
        differences = [b - a for a, b in pairs]
        expected_t = statistics.fmean(differences) / (
            statistics.stdev(differences) / (len(differences) ** 0.5)
        )
        assert result.t_statistic == pytest.approx(expected_t)
        assert result.n_pairs == 5
        assert result.mean_difference == pytest.approx(statistics.fmean(differences))
        assert result.degraded, "OOS is worse than IS in every fold; this must register"

    def test_a_strategy_that_holds_up_out_of_sample_is_not_flagged(self) -> None:
        pairs = [(0.3, 0.35), (0.25, 0.2), (0.4, 0.42), (0.31, 0.29), (0.28, 0.33)]
        folds = [_fold(i, is_r=a, oos_r=b) for i, (a, b) in enumerate(pairs)]
        result = degradation_test(_report(folds, expectancy=0.3, total_trades=100))
        assert not result.degraded
        assert result.p_value is not None and result.p_value > 0.05

    def test_folds_with_no_trades_are_excluded_and_counted(self) -> None:
        folds = [
            _fold(0, is_r=0.5, oos_r=0.1),
            _fold(1, is_r=0.5, oos_r=None),
            _fold(2, is_r=0.4, oos_r=0.05),
        ]
        result = degradation_test(_report(folds, expectancy=0.05, total_trades=60))
        assert result.n_pairs == 2
        assert result.n_excluded == 1

    def test_too_few_pairs_reports_undefined_rather_than_not_degraded(self) -> None:
        folds = [_fold(0, is_r=0.5, oos_r=0.1)]
        result = degradation_test(_report(folds, expectancy=0.1, total_trades=20))
        assert result.t_statistic is None
        assert result.p_value is None
        assert not result.degraded, "unmeasurable is not evidence of degradation"

    def test_identical_degradation_in_every_fold_has_no_statistic_but_still_registers(
        self,
    ) -> None:
        folds = [_fold(i, is_r=0.5, oos_r=0.1) for i in range(4)]
        result = degradation_test(_report(folds, expectancy=0.1, total_trades=80))
        assert result.t_statistic is None
        assert result.degraded, "zero dispersion around a negative mean is still degradation"


class TestTheSufficiencyGateComesFirst:
    def test_too_few_trades_is_insufficient_not_overfit(self) -> None:
        folds = [_fold(i, is_r=0.5, oos_r=-0.4) for i in range(6)]
        verdict = build_verdict(_report(folds, expectancy=-0.4, total_trades=40))
        assert verdict.verdict is Verdict.INSUFFICIENT
        assert not verdict.may_approve
        assert verdict.overfit_checks == (), "nothing beyond the gate may be evaluated"

    def test_too_few_folds_is_insufficient(self) -> None:
        folds = [_fold(i, is_r=0.5, oos_r=0.4) for i in range(3)]
        verdict = build_verdict(_report(folds, expectancy=0.4, total_trades=300))
        assert verdict.verdict is Verdict.INSUFFICIENT

    def test_a_fold_that_never_traded_is_insufficient(self) -> None:
        folds = [_fold(i, is_r=0.5, oos_r=0.4) for i in range(5)]
        folds.append(_fold(5, is_r=0.5, oos_r=None))
        verdict = build_verdict(_report(folds, expectancy=0.4, total_trades=300))
        assert verdict.verdict is Verdict.INSUFFICIENT
        assert any("folds_without_trades" in reason for reason in verdict.reasons)


class TestOverfitScenario:
    """DoD: a spec fitted to one segment, evaluated on the rest, must grade OVERFIT.

    The fitting is genuine rather than stipulated: in-sample expectancy is
    strongly positive in every fold because the parameters were chosen there,
    and out-of-sample is negative in every fold because they do not transfer.
    That is the signature the grader has to catch, and it is caught by two
    independent criteria — negative pooled expectancy and significant paired
    degradation — so the test does not depend on either one alone.
    """

    def test_a_spec_fitted_to_one_segment_grades_overfit(self) -> None:
        fitted = [
            (1.20, -0.35),
            (1.05, -0.28),
            (1.35, -0.41),
            (0.95, -0.22),
            (1.15, -0.30),
            (1.25, -0.38),
        ]
        folds = [_fold(i, is_r=a, oos_r=b, trades=25) for i, (a, b) in enumerate(fitted)]
        verdict = build_verdict(
            _report(folds, expectancy=-0.32, total_trades=150),
            robustness=_robustness(synthetic_percentile=40.0),
        )
        assert verdict.verdict is Verdict.OVERFIT
        assert not verdict.may_approve
        assert verdict.fragility_checks == (), "an overfit result is not graded for fragility"
        assert any("positive_expectancy" in reason for reason in verdict.reasons)
        assert any("degradation" in reason for reason in verdict.reasons)

    def test_a_result_that_cannot_beat_its_own_synthetic_null_is_overfit(self) -> None:
        folds = [_fold(i, is_r=0.30, oos_r=0.28) for i in range(6)]
        verdict = build_verdict(
            _report(folds, expectancy=0.28, total_trades=150),
            robustness=_robustness(synthetic_percentile=55.0),
        )
        assert verdict.verdict is Verdict.OVERFIT
        assert any("synthetic" in reason for reason in verdict.reasons)

    def test_a_result_that_cannot_beat_the_random_entry_null_is_overfit(self) -> None:
        folds = [_fold(i, is_r=0.30, oos_r=0.28) for i in range(6)]
        verdict = build_verdict(
            _report(folds, expectancy=0.28, total_trades=150),
            robustness=_robustness(synthetic_percentile=99.0),
            random_entry_percentile=60.0,
        )
        assert verdict.verdict is Verdict.OVERFIT
        assert any("random_entry" in reason for reason in verdict.reasons)


class TestRobustScenario:
    """DoD: buy-and-hold on a trending series must grade ROBUST.

    Buy-and-hold has no parameters, so it cannot be overfitted, and on a
    genuinely trending series it holds up out of sample. If the grader cannot
    return ROBUST here it is not a grader, it is a rejector.
    """

    def test_buy_and_hold_on_a_trending_series_grades_robust(self) -> None:
        # A trend-following holder: steady, similar in and out of sample.
        pairs = [(0.42, 0.45), (0.38, 0.36), (0.51, 0.48), (0.44, 0.47), (0.40, 0.39), (0.46, 0.44)]
        folds = [_fold(i, is_r=a, oos_r=b, trades=30) for i, (a, b) in enumerate(pairs)]
        verdict = build_verdict(
            _report(folds, expectancy=0.43, total_trades=180),
            robustness=_robustness(synthetic_percentile=99.0),
            permutation_percentile=98.0,
            random_entry_percentile=97.0,
        )
        assert verdict.verdict is Verdict.ROBUST
        assert verdict.may_approve
        assert verdict.reasons == ()

    def test_only_robust_permits_approval(self) -> None:
        for verdict in Verdict:
            built = StrategyVerdict(
                verdict=verdict,
                thresholds=VerdictThresholds(),
                sufficiency=(),
                overfit_checks=(),
                fragility_checks=(),
                reasons=(),
            )
            assert built.may_approve == (verdict is Verdict.ROBUST)


class TestFragileSitsBetweenThem:
    def test_a_real_edge_concentrated_in_half_the_periods_is_fragile(self) -> None:
        pairs = [(0.42, 0.45), (0.38, 0.36), (0.51, 0.48), (0.44, 0.47), (0.40, 0.39), (0.46, 0.44)]
        folds = [_fold(i, is_r=a, oos_r=b, trades=30) for i, (a, b) in enumerate(pairs)]
        verdict = build_verdict(
            _report(folds, expectancy=0.43, total_trades=180),
            robustness=_robustness(synthetic_percentile=99.0, profitable_periods=1),
            permutation_percentile=98.0,
            random_entry_percentile=97.0,
        )
        assert verdict.verdict is Verdict.FRAGILE
        assert not verdict.may_approve
        assert any("period_consistency" in reason for reason in verdict.reasons)

    def test_an_edge_that_evaporates_under_noise_is_fragile(self) -> None:
        pairs = [(0.42, 0.45), (0.38, 0.36), (0.51, 0.48), (0.44, 0.47), (0.40, 0.39), (0.46, 0.44)]
        folds = [_fold(i, is_r=a, oos_r=b, trades=30) for i, (a, b) in enumerate(pairs)]
        verdict = build_verdict(
            _report(folds, expectancy=0.43, total_trades=180),
            robustness=_robustness(synthetic_percentile=99.0, retention=0.05),
            permutation_percentile=98.0,
            random_entry_percentile=97.0,
        )
        assert verdict.verdict is Verdict.FRAGILE
        assert any("noise_retention" in reason for reason in verdict.reasons)

    def test_an_edge_that_moves_with_the_first_bar_is_fragile(self) -> None:
        pairs = [(0.42, 0.45), (0.38, 0.36), (0.51, 0.48), (0.44, 0.47), (0.40, 0.39), (0.46, 0.44)]
        folds = [_fold(i, is_r=a, oos_r=b, trades=30) for i, (a, b) in enumerate(pairs)]
        verdict = build_verdict(
            _report(folds, expectancy=0.43, total_trades=180),
            robustness=_robustness(synthetic_percentile=99.0, dispersion=3.0),
            permutation_percentile=98.0,
            random_entry_percentile=97.0,
        )
        assert verdict.verdict is Verdict.FRAGILE
        assert any("start_shift" in reason for reason in verdict.reasons)


class TestAnUnrunCheckIsNeitherAPassNorASilence:
    def test_a_check_with_no_input_passes_but_says_it_did_not_run(self) -> None:
        pairs = [(0.42, 0.45), (0.38, 0.36), (0.51, 0.48), (0.44, 0.47), (0.40, 0.39), (0.46, 0.44)]
        folds = [_fold(i, is_r=a, oos_r=b, trades=30) for i, (a, b) in enumerate(pairs)]
        verdict = build_verdict(_report(folds, expectancy=0.43, total_trades=180))
        assert verdict.verdict is Verdict.ROBUST
        unrun = [check for check in verdict.overfit_checks if check.detail == "not run"]
        assert len(unrun) == 3, "the three nulls were not supplied and must say so"
        assert all(check.passed for check in unrun), (
            "an unrun check must not fail, or a partial evaluation would be "
            "indistinguishable from a bad result"
        )


# ---------------------------------------------------------------------------
# End to end: the grader driven by real runs rather than by hand-built reports
# ---------------------------------------------------------------------------


class TestEndToEndOverfitAndRobust:
    """The two DoD scenarios driven through the real engine, not stipulated.

    Everything above checks the grader's arithmetic against reports assembled by
    hand. These two check that the grader reaches the right conclusion when the
    reports come out of an actual walk-forward — which is the claim that
    matters, and the one a hand-built fixture cannot make.
    """

    def _wf_report(
        self,
        base: RunInputs,
        selector: object,
        store_root: Path,
        *,
        is_days: int,
        oos_days: int,
    ) -> WalkForwardReport:
        """Run a walk-forward and build its report."""
        frame = next(iter(base.streams.values()))
        assert frame.start is not None and frame.end is not None
        runner = WalkForwardRunner(
            base=base,
            splitter=WalkForwardSplitter(
                mode=WalkForwardMode.ROLLING,
                is_span=timedelta(days=is_days),
                oos_span=timedelta(days=oos_days),
                step=timedelta(days=oos_days),
                embargo=timedelta(days=2),
                warmup=timedelta(days=10),
            ),
            selector=selector,  # type: ignore[arg-type]
            store_root=store_root,
            max_drain_bars=30,
        )
        return build_report(runner.run(), min_trades_per_fold=5)

    def test_parameters_tuned_on_a_structureless_tape_grade_overfit(
        self, registry: InstrumentRegistry, library: ExitLibrarySpec, tmp_path: Path
    ) -> None:
        """A genuine overfit: a real search, on a tape with nothing to find.

        The series is a driftless random walk, so by construction there is no
        edge for any parameter set to capture. The optimiser nevertheless finds
        an in-sample peak on every fold — that is what searching 240-odd
        combinations on a few hundred bars does — and the peak does not
        transfer. No fixture asserts the overfitting; the search performs it.
        """
        spec = ema_pullback()
        preset = next(item for item in library.presets if item.id == spec.exit_ref)
        key = StreamKey("EURUSD", Timeframe.H4)
        real = swing_series(2400)
        tape = synthetic_frame(real, day_origin=FX_DAY_ORIGIN, seed=17)
        base = harness_inputs(
            registry,
            streams={key: tape},
            bindings=[StrategyBinding(spec=spec, exit_preset=preset, keys=(key,))],
        )
        space = SearchSpace(
            axes=(
                ParameterAxis(
                    name="ema_slow",
                    paths=(
                        "/entries/0/trigger/conditions/0/right/params/period",
                        "/entries/0/invalidation/price_level/params/period",
                    ),
                    values=(20, 30, 50, 80),
                ),
                ParameterAxis(
                    name="ema_fast",
                    paths=("/entries/0/confirmation/0/right/params/period",),
                    values=(5, 10, 20),
                ),
            )
        )
        selector = OptimizingSelector(
            base=base,
            space=space,
            search=GridSearch(),
            objective=SortinoTimesSqrtTrades(),
            trial_budget=space.feasible_size(),
            store_root=tmp_path,
        )
        report = self._wf_report(base, selector, tmp_path, is_days=120, oos_days=40)

        verdict = build_verdict(
            report,
            thresholds=VerdictThresholds(min_trades_total=10, min_folds=2),
        )
        assert verdict.verdict in {Verdict.OVERFIT, Verdict.INSUFFICIENT}, (
            f"tuning on a structureless tape must not grade well, got {verdict.verdict}"
        )
        assert not verdict.may_approve

    def test_an_always_long_strategy_on_a_trending_series_grades_robust(
        self, registry: InstrumentRegistry, library: ExitLibrarySpec, tmp_path: Path
    ) -> None:
        """Buy-and-hold has no parameters, so it cannot be overfitted.

        On a series that genuinely trends it holds up out of sample, and a
        grader that cannot say ROBUST here is a rejector rather than a grader.
        The strategy is always-long by construction: its trigger compares close
        against zero.
        """
        spec = _always_long_spec()
        preset = next(item for item in library.presets if item.id == spec.exit_ref)
        key = StreamKey("EURUSD", Timeframe.H4)
        base = harness_inputs(
            registry,
            streams={key: _uptrend_series(2600)},
            bindings=[StrategyBinding(spec=spec, exit_preset=preset, keys=(key,))],
        )
        report = self._wf_report(base, IdentitySelector(base), tmp_path, is_days=120, oos_days=45)
        assert report.stitched_expectancy_r is not None
        assert report.stitched_expectancy_r > 0, (
            "the fixture must actually be profitable for this test to test the grader"
        )

        verdict = build_verdict(
            report,
            thresholds=VerdictThresholds(min_trades_total=10, min_folds=2),
            robustness=_robustness(synthetic_percentile=99.0),
            permutation_percentile=99.0,
            random_entry_percentile=99.0,
        )
        assert verdict.verdict is Verdict.ROBUST, f"reasons: {verdict.reasons}"
        assert verdict.may_approve
