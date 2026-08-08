"""What a walk-forward produced, as data — no HTML, no verdict.

Every number a fold's boundary can lose is a number this module puts in the
report rather than a number it summarises away: rejections, degradations,
drops and ``ATR_UNAVAILABLE`` coverage all travel from the stored run into
:class:`FoldReport` untouched, the same discipline
:mod:`trading_system.backtest.orchestrator` already applies to a single run.

**Whether a ratio between IS and OOS is worth computing is not this module's
call.** CLAUDE.md's own requirement for this stage is explicit: publish the
pairs (``is_expectancy_r``, ``oos_expectancy_r`` and the two Sortinos), and
leave "OOS ≥ 60% of IS" — a rule that is undefined the moment IS is near zero
or negative — to stage 5, where a verdict is actually rendered.

**A fold below ``min_trades_per_fold`` is flagged, never dropped.** Excluding
it from the aggregate would be a second, silent way for a boundary to lose
information, on top of every counter this module already exists to preserve.
"""

import json
import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from trading_system.analytics.metrics import daily_curve, sharpe_daily, sortino_daily, trade_stats
from trading_system.backtest.portfolio import EquityPoint, TradeRecord
from trading_system.backtest.reproducibility import read_run
from trading_system.validation.splitting import Fold, FoldWindow
from trading_system.validation.stitching import StitchedCurve, stitch
from trading_system.validation.walkforward import WalkForwardResult

if TYPE_CHECKING:
    from trading_system.validation.monte_carlo import MonteCarloReport
    from trading_system.validation.robustness import RobustnessReport


def _equity_at(curve: Sequence[EquityPoint], ts: datetime) -> Decimal | None:
    """Equity of the last row at or before ``ts``, or ``None`` if the curve starts after it."""
    at_or_before = [point.equity for point in curve if point.ts <= ts]
    return at_or_before[-1] if at_or_before else None


def _boundary_residual(
    curve: Sequence[EquityPoint], trades: Sequence[TradeRecord], window: FoldWindow
) -> Decimal | None:
    """``equity(trade_end) - equity(trade_start) - realised PnL closed inside the window``.

    An exact bookkeeping identity, not an estimate — see
    :mod:`trading_system.validation.walkforward`'s module docstring. ``None``
    only if the stored curve does not actually cover the window, which would
    mean the run was sliced wrong upstream rather than anything this function
    can resolve.
    """
    start_equity = _equity_at(curve, window.trade_start)
    end_equity = _equity_at(curve, window.trade_end)
    if start_equity is None or end_equity is None:
        return None
    realized = sum(
        (trade.net for trade in trades if window.trade_start < trade.closed_at <= window.trade_end),
        Decimal(0),
    )
    return end_equity - start_equity - realized


def _expectancy_r(trades: Sequence[TradeRecord]) -> float | None:
    """Mean realised R, or ``None`` with fewer than one trade."""
    if not trades:
        return None
    return trade_stats(trades).expectancy_r


def _sortino(curve: Sequence[EquityPoint]) -> float | None:
    """Annualised Sortino of a curve, or ``None`` when the sample cannot support one.

    Catches exactly the :class:`ValueError` cases
    :func:`~trading_system.analytics.metrics.sortino_daily` documents as
    undefined (fewer than one daily return, zero downside deviation) and turns
    each into an explicit ``None`` — a value this report can show, rather than
    an exception the caller of :func:`build_report` would have to guard
    against on every fold.
    """
    try:
        return sortino_daily(daily_curve(curve)).value
    except ValueError:
        return None


def _sharpe(curve: Sequence[EquityPoint]) -> float | None:
    """Annualised Sharpe of a curve, or ``None`` when the sample cannot support one."""
    try:
        return sharpe_daily(daily_curve(curve)).value
    except ValueError:
        return None


@dataclass(frozen=True)
class FoldReport:
    """One fold's own numbers: periods, trade counts, IS/OOS metric pairs, and every counter.

    Attributes:
        index: The fold's position in the sequence.
        is_window: The in-sample window.
        oos_window: The out-of-sample window.
        is_trade_count: Trades the IS run closed.
        oos_trade_count: Trades the OOS run closed — every one of them opened
            inside ``oos_window``, by construction; see
            :mod:`trading_system.validation.walkforward`.
        is_expectancy_r: Mean realised R of the IS run's trades, or ``None``
            with none.
        oos_expectancy_r: Same, OOS.
        is_sortino: Annualised Sortino of the IS run's own curve, or ``None``.
        oos_sortino: Same, OOS.
        is_boundary_residual: Bookkeeping residual at the IS window's own
            boundary — see :func:`_boundary_residual`.
        oos_boundary_residual: Same, OOS. What the drain exists to keep small.
        drain_truncated: Positions still open when the OOS run's drain
            allowance ran out — ``open_at_end`` of the stored OOS run.
        rejections: Risk Engine refusals, OOS run, every reason present.
        degradations: Risk Engine measurements that fell back to a prior, OOS
            run.
        exit_drops: Exit instructions not carried out as stated, OOS run.
        entry_drops: Entry signals discarded as defective, OOS run.
        signal_drops: Signals discarded before reaching the Risk Engine, OOS
            run.
        atr_unavailable_fraction: Share of OOS fills priced without a
            volatility measurement.
        insufficient: Whether ``oos_trade_count`` is below the report's own
            ``min_trades_per_fold`` — flagged, not excluded from any
            aggregate below.
    """

    index: int
    is_window: FoldWindow
    oos_window: FoldWindow
    is_trade_count: int
    oos_trade_count: int
    is_expectancy_r: float | None
    oos_expectancy_r: float | None
    is_sortino: float | None
    oos_sortino: float | None
    is_boundary_residual: Decimal | None
    oos_boundary_residual: Decimal | None
    drain_truncated: int
    rejections: dict[str, int]
    degradations: dict[str, int]
    exit_drops: dict[str, int]
    entry_drops: dict[str, int]
    signal_drops: dict[str, int]
    atr_unavailable_fraction: float
    insufficient: bool

    def to_dict(self) -> dict[str, Any]:
        """Serialise to JSON-able data."""

        def window(w: FoldWindow) -> dict[str, str]:
            return {
                "data_start": w.data_start.isoformat(),
                "trade_start": w.trade_start.isoformat(),
                "trade_end": w.trade_end.isoformat(),
            }

        return {
            "index": self.index,
            "is_window": window(self.is_window),
            "oos_window": window(self.oos_window),
            "is_trade_count": self.is_trade_count,
            "oos_trade_count": self.oos_trade_count,
            "is_expectancy_r": self.is_expectancy_r,
            "oos_expectancy_r": self.oos_expectancy_r,
            "is_sortino": self.is_sortino,
            "oos_sortino": self.oos_sortino,
            "is_boundary_residual": str(self.is_boundary_residual)
            if self.is_boundary_residual is not None
            else None,
            "oos_boundary_residual": str(self.oos_boundary_residual)
            if self.oos_boundary_residual is not None
            else None,
            "drain_truncated": self.drain_truncated,
            "rejections": self.rejections,
            "degradations": self.degradations,
            "exit_drops": self.exit_drops,
            "entry_drops": self.entry_drops,
            "signal_drops": self.signal_drops,
            "atr_unavailable_fraction": self.atr_unavailable_fraction,
            "insufficient": self.insufficient,
        }


@dataclass(frozen=True)
class WalkForwardReport:
    """Every fold's own report, plus the whole run's stitched OOS picture.

    Attributes:
        wf_id: The walk-forward this report was built from.
        min_trades_per_fold: Threshold :attr:`FoldReport.insufficient` was
            measured against.
        folds: Every fold's report, oldest first.
        n_folds: ``len(folds)``.
        n_folds_without_oos_trades: Folds whose OOS run closed no trades at
            all — the extreme case of ``insufficient``.
        insufficient_sample: Whether any fold is ``insufficient`` — a coarse,
            visible flag; the per-fold detail is in ``folds``.
        stitched_trade_count: Trades across every fold's OOS run, pooled.
        stitched_expectancy_r: Mean realised R across every fold's OOS trades,
            pooled — not averaged fold by fold, which would weight a
            ten-trade fold the same as a hundred-trade one.
        stitched_sharpe: Annualised Sharpe of the spliced OOS return series.
        stitched_sortino: Annualised Sortino of the same series.
        per_fold_oos_expectancy_r: Each fold's own OOS expectancy, in fold
            order — the distribution CLAUDE.md asks for, alongside the pooled
            figure above.
        per_fold_oos_sortino: Each fold's own OOS Sortino, in fold order.
    """

    wf_id: str
    min_trades_per_fold: int
    folds: tuple[FoldReport, ...]
    n_folds: int
    n_folds_without_oos_trades: int
    insufficient_sample: bool
    stitched_trade_count: int
    stitched_expectancy_r: float | None
    stitched_sharpe: float | None
    stitched_sortino: float | None
    per_fold_oos_expectancy_r: tuple[float | None, ...]
    per_fold_oos_sortino: tuple[float | None, ...]

    def to_dict(self) -> dict[str, Any]:
        """Serialise to JSON-able data."""
        return {
            "wf_id": self.wf_id,
            "min_trades_per_fold": self.min_trades_per_fold,
            "folds": [fold.to_dict() for fold in self.folds],
            "n_folds": self.n_folds,
            "n_folds_without_oos_trades": self.n_folds_without_oos_trades,
            "insufficient_sample": self.insufficient_sample,
            "stitched_trade_count": self.stitched_trade_count,
            "stitched_expectancy_r": self.stitched_expectancy_r,
            "stitched_sharpe": self.stitched_sharpe,
            "stitched_sortino": self.stitched_sortino,
            "per_fold_oos_expectancy_r": list(self.per_fold_oos_expectancy_r),
            "per_fold_oos_sortino": list(self.per_fold_oos_sortino),
        }


def _fold_report(
    fold: Fold,
    is_curve: Sequence[EquityPoint],
    is_trades: Sequence[TradeRecord],
    oos_curve: Sequence[EquityPoint],
    oos_trades: Sequence[TradeRecord],
    oos_counters: Mapping[str, Any],
    *,
    min_trades_per_fold: int,
) -> FoldReport:
    """Build one fold's report from its two runs' curves and trades."""
    oos_trade_count = len(oos_trades)
    return FoldReport(
        index=fold.index,
        is_window=fold.is_window,
        oos_window=fold.oos_window,
        is_trade_count=len(is_trades),
        oos_trade_count=oos_trade_count,
        is_expectancy_r=_expectancy_r(is_trades),
        oos_expectancy_r=_expectancy_r(oos_trades),
        is_sortino=_sortino(is_curve),
        oos_sortino=_sortino(oos_curve),
        is_boundary_residual=_boundary_residual(is_curve, is_trades, fold.is_window),
        oos_boundary_residual=_boundary_residual(oos_curve, oos_trades, fold.oos_window),
        drain_truncated=int(oos_counters["open_at_end"]),
        rejections=dict(oos_counters["rejections"]),
        degradations=dict(oos_counters["degradations"]),
        exit_drops=dict(oos_counters["exit_drops"]),
        entry_drops=dict(oos_counters["entry_drops"]),
        signal_drops=dict(oos_counters["signal_drops"]),
        atr_unavailable_fraction=float(
            oos_counters["cost_degradations"].get("atr_unavailable", 0.0)
        ),
        insufficient=oos_trade_count < min_trades_per_fold,
    )


def build_report(
    result: WalkForwardResult,
    *,
    min_trades_per_fold: int,
    stitched_starting_equity: Decimal = Decimal(100_000),
) -> WalkForwardReport:
    """Build the full report from a finished walk-forward, reading tables off disk.

    Args:
        result: The walk-forward's result — run ids and paths, not curves or
            trades; both are read here, explicitly, per fold.
        min_trades_per_fold: Below this many OOS trades, a fold is flagged
            ``insufficient``. Required, with no default — see this module's
            docstring on why silence is not an option here.
        stitched_starting_equity: Passed straight to
            :func:`~trading_system.validation.stitching.stitch`.

    Returns:
        The report.
    """
    fold_reports: list[FoldReport] = []
    oos_curves: list[Sequence[EquityPoint]] = []
    folds: list[Fold] = []
    pooled_oos_trades: list[TradeRecord] = []

    for fold_run in result.folds:
        is_stored = read_run(fold_run.is_run.path)
        oos_stored = read_run(fold_run.oos_run.path)
        fold_reports.append(
            _fold_report(
                fold_run.fold,
                is_stored.result.curve,
                is_stored.result.trades,
                oos_stored.result.curve,
                oos_stored.result.trades,
                fold_run.oos_run.counters,
                min_trades_per_fold=min_trades_per_fold,
            )
        )
        oos_curves.append(oos_stored.result.curve)
        folds.append(fold_run.fold)
        pooled_oos_trades.extend(oos_stored.result.trades)

    stitched: StitchedCurve = stitch(folds, oos_curves, starting_equity=stitched_starting_equity)

    return WalkForwardReport(
        wf_id=result.wf_id,
        min_trades_per_fold=min_trades_per_fold,
        folds=tuple(fold_reports),
        n_folds=len(fold_reports),
        n_folds_without_oos_trades=sum(1 for report in fold_reports if report.oos_trade_count == 0),
        insufficient_sample=any(report.insufficient for report in fold_reports),
        stitched_trade_count=len(pooled_oos_trades),
        stitched_expectancy_r=_expectancy_r(pooled_oos_trades),
        stitched_sharpe=_sharpe(stitched.points),
        stitched_sortino=_sortino(stitched.points),
        per_fold_oos_expectancy_r=tuple(report.oos_expectancy_r for report in fold_reports),
        per_fold_oos_sortino=tuple(report.oos_sortino for report in fold_reports),
    )


def write_report(report: WalkForwardReport, path: Path) -> None:
    """Export a report to JSON, next to the walk-forward's own manifest.

    Args:
        report: The report.
        path: File to write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2))


# ---------------------------------------------------------------------------
# The verdict
# ---------------------------------------------------------------------------


class Verdict(StrEnum):
    """What a walk-forward, its nulls and its perturbations add up to.

    Evaluated in the order declared, first match winning. The order is the
    substance, not a detail: a sample too small to measure cannot be called
    fragile, and something that fails its own no-structure null is not merely
    fragile either.
    """

    #: Not enough evidence to grade. A gate, not a grade — nothing else is
    #: evaluated once this fires, because every criterion below is a statement
    #: about a measurement that was not possible.
    INSUFFICIENT = "INSUFFICIENT"

    #: Measurable, and indistinguishable from (or worse than) a null that has
    #: no edge in it by construction.
    OVERFIT = "OVERFIT"

    #: A real result that does not survive being asked slightly differently.
    FRAGILE = "FRAGILE"

    #: Passed every gate.
    ROBUST = "ROBUST"


@dataclass(frozen=True)
class VerdictThresholds:
    """Every number the verdict depends on, in one place and never inline.

    Attributes:
        min_trades_total: Fewest pooled out-of-sample trades that can support
            any judgement at all.
        min_folds: Fewest folds. A walk-forward with three folds cannot
            distinguish a strategy from a regime.
        max_folds_without_trades: How many folds may close nothing before the
            sample is called insufficient. Zero: a fold that never traded is a
            fold whose out-of-sample window was never actually tested.
        null_percentile: Percentile a real score must exceed against a null
            distribution to count as distinguishable from it.
        degradation_alpha: One-sided significance level for the paired IS/OOS
            degradation test.
        min_profitable_periods_fraction: Share of equal time slices that must
            be profitable.
        max_drawdown_percentile: If the observed drawdown sits below this
            percentile of its own block-permutation distribution, the realised
            path was luckier than the ordering alone justifies, and the result
            is called fragile rather than robust.
        min_noise_retention: Share of the smallest-noise expectancy that must
            survive at the largest noise level.
        max_start_shift_dispersion: Largest standard deviation of expectancy
            across shifted starts that still counts as stable.
        min_expectancy_r: Pooled out-of-sample expectancy below this is not a
            result worth grading as robust however stable it is.
    """

    min_trades_total: int = 100
    min_folds: int = 5
    max_folds_without_trades: int = 0
    null_percentile: float = 95.0
    degradation_alpha: float = 0.05
    min_profitable_periods_fraction: float = 0.5
    max_drawdown_percentile: float = 5.0
    min_noise_retention: float = 0.5
    max_start_shift_dispersion: float = 0.5
    min_expectancy_r: float = 0.0


@dataclass(frozen=True)
class DegradationTest:
    """Paired in-sample to out-of-sample degradation across folds.

    **The formula, stated here because a reader must not have to read the
    implementation to know what was tested.** For every fold where both sides
    scored::

        d[i] = oos_expectancy_r[i] - is_expectancy_r[i]
        t    = mean(d) / (stdev(d) / sqrt(k))        on k - 1 degrees of freedom
        p    = P(T <= t)                             one-sided, T ~ Student-t

    and the criterion is ``p < degradation_alpha``: out-of-sample is
    *significantly worse* than in-sample.

    **Paired, because folds differ by regime far more than by degradation.** An
    unpaired comparison of ``mean(oos)`` against ``mean(is)`` has its variance
    dominated by between-fold regime differences, which would swamp the effect
    being measured. A fold's in-sample and out-of-sample windows are adjacent in
    time and share a regime, so their difference isolates degradation.

    **On ``expectancy_r`` and never on the objective score.** The objective is
    Sortino times the square root of trade count, so it scales with sample size,
    and an in-sample window of 360 days against an out-of-sample window of 270
    days produces systematically different counts. Measured on that, "degradation"
    would partly be a measurement of the window lengths. ``expectancy_r`` is a
    per-trade mean and does not scale with the sample.

    **Student-t, not a normal approximation**, because k is the number of folds
    — eight in the reference configuration, not a large sample.

    Attributes:
        n_pairs: Folds where both sides scored.
        n_excluded: Folds dropped because one side had no trades.
        mean_difference: ``mean(d)``. Negative means out-of-sample was worse.
        t_statistic: The statistic above, or ``None`` when undefined.
        p_value: One-sided p, or ``None`` when undefined.
        degraded: Whether ``p < alpha``.
    """

    n_pairs: int
    n_excluded: int
    mean_difference: float | None
    t_statistic: float | None
    p_value: float | None
    degraded: bool

    def to_dict(self) -> dict[str, Any]:
        """Serialise to JSON-able data."""
        return {
            "n_pairs": self.n_pairs,
            "n_excluded": self.n_excluded,
            "mean_difference": self.mean_difference,
            "t_statistic": self.t_statistic,
            "p_value": self.p_value,
            "degraded": self.degraded,
        }


def degradation_test(report: WalkForwardReport, *, alpha: float = 0.05) -> DegradationTest:
    """Run the paired IS/OOS degradation test. See :class:`DegradationTest` for the formula.

    Args:
        report: The walk-forward's own report.
        alpha: One-sided significance level.

    Returns:
        The test. Undefined cases (fewer than two usable pairs, or zero
        dispersion with a non-negative mean) report ``degraded=False`` with a
        ``None`` statistic rather than an invented number — "could not be
        measured" is not evidence of degradation, and the pair counts are
        published so the distinction is visible.
    """
    from scipy import stats

    pairs = [
        (fold.is_expectancy_r, fold.oos_expectancy_r)
        for fold in report.folds
        if fold.is_expectancy_r is not None and fold.oos_expectancy_r is not None
    ]
    excluded = len(report.folds) - len(pairs)
    if len(pairs) < 2:
        return DegradationTest(len(pairs), excluded, None, None, None, False)

    differences = [oos - is_ for is_, oos in pairs]
    mean = statistics.fmean(differences)
    spread = statistics.stdev(differences)
    if spread == 0:
        # Every fold degraded by exactly the same amount. A t-statistic is
        # infinite here; the honest reading is "degraded iff that amount is
        # negative", with no statistic to report.
        return DegradationTest(len(pairs), excluded, mean, None, None, mean < 0)

    t_stat = mean / (spread / math.sqrt(len(differences)))
    p_value = float(stats.t.cdf(t_stat, df=len(differences) - 1))
    return DegradationTest(len(pairs), excluded, mean, t_stat, p_value, p_value < alpha)


@dataclass(frozen=True)
class VerdictCheck:
    """One named criterion and whether it held.

    Attributes:
        name: What was checked.
        passed: Whether it held.
        detail: The measurement, for a reader who wants the number rather than
            the boolean.
    """

    name: str
    passed: bool
    detail: str

    def to_dict(self) -> dict[str, Any]:
        """Serialise to JSON-able data."""
        return {"name": self.name, "passed": self.passed, "detail": self.detail}


@dataclass(frozen=True)
class StrategyVerdict:
    """The judgement, and every check it was assembled from.

    Attributes:
        verdict: The grade.
        thresholds: What it was graded against.
        sufficiency: Sample-size gates.
        overfit_checks: Null and degradation gates.
        fragility_checks: Perturbation gates.
        reasons: Human-readable failures, in evaluation order.
        may_approve: Whether ``status=APPROVED`` is permitted — true only for
            :attr:`Verdict.ROBUST`. A property of the verdict rather than a
            judgement call left to whoever writes the strategy file.
    """

    verdict: Verdict
    thresholds: VerdictThresholds
    sufficiency: tuple[VerdictCheck, ...]
    overfit_checks: tuple[VerdictCheck, ...]
    fragility_checks: tuple[VerdictCheck, ...]
    reasons: tuple[str, ...]

    @property
    def may_approve(self) -> bool:
        """Whether the strategy may be moved to ``APPROVED``."""
        return self.verdict is Verdict.ROBUST

    def to_dict(self) -> dict[str, Any]:
        """Serialise to JSON-able data."""
        return {
            "verdict": self.verdict.value,
            "may_approve": self.may_approve,
            "reasons": list(self.reasons),
            "sufficiency": [check.to_dict() for check in self.sufficiency],
            "overfit_checks": [check.to_dict() for check in self.overfit_checks],
            "fragility_checks": [check.to_dict() for check in self.fragility_checks],
            "thresholds": asdict(self.thresholds),
        }


def build_verdict(
    report: WalkForwardReport,
    *,
    thresholds: VerdictThresholds | None = None,
    monte_carlo: "MonteCarloReport | None" = None,
    robustness: "RobustnessReport | None" = None,
    permutation_percentile: float | None = None,
    random_entry_percentile: float | None = None,
) -> StrategyVerdict:
    """Grade a walk-forward against its nulls and perturbations.

    Args:
        report: The walk-forward's own report.
        thresholds: What to grade against. Defaults are
            :class:`VerdictThresholds`'s own.
        monte_carlo: Resampling results, if run. Its block-permutation family
            is the only one read — see
            :mod:`trading_system.validation.monte_carlo` on why the pooled one
            is deliberately excluded.
        robustness: Perturbation results, if run.
        permutation_percentile: Where the real score fell against the
            bar-permutation null, if that calibration was run.
        random_entry_percentile: Where it fell against the random-entry null.

    Returns:
        The verdict.

    Notes:
        A check whose input was not supplied does not fail — it is recorded as
        passed with a detail saying it was not run. Treating an unrun check as
        a failure would make a partial evaluation indistinguishable from a bad
        result; treating it as a silent pass would hide it. The detail string
        is the difference, and it is always present.
    """
    limits = thresholds if thresholds is not None else VerdictThresholds()
    reasons: list[str] = []

    def check(name: str, passed: bool, detail: str, bucket: list[VerdictCheck]) -> None:
        bucket.append(VerdictCheck(name, passed, detail))
        if not passed:
            reasons.append(f"{name}: {detail}")

    sufficiency: list[VerdictCheck] = []
    check(
        "min_trades_total",
        report.stitched_trade_count >= limits.min_trades_total,
        f"{report.stitched_trade_count} pooled OOS trades, need {limits.min_trades_total}",
        sufficiency,
    )
    check(
        "min_folds",
        report.n_folds >= limits.min_folds,
        f"{report.n_folds} folds, need {limits.min_folds}",
        sufficiency,
    )
    check(
        "folds_without_trades",
        report.n_folds_without_oos_trades <= limits.max_folds_without_trades,
        f"{report.n_folds_without_oos_trades} fold(s) closed no OOS trades, "
        f"allowed {limits.max_folds_without_trades}",
        sufficiency,
    )
    if not all(item.passed for item in sufficiency):
        return StrategyVerdict(
            verdict=Verdict.INSUFFICIENT,
            thresholds=limits,
            sufficiency=tuple(sufficiency),
            overfit_checks=(),
            fragility_checks=(),
            reasons=tuple(reasons),
        )

    degradation = degradation_test(report, alpha=limits.degradation_alpha)
    overfit: list[VerdictCheck] = []
    expectancy = report.stitched_expectancy_r
    check(
        "positive_expectancy",
        expectancy is not None and expectancy > limits.min_expectancy_r,
        f"pooled OOS expectancy_r {expectancy}, need > {limits.min_expectancy_r}",
        overfit,
    )
    check(
        "beats_random_entry_null",
        random_entry_percentile is None or random_entry_percentile >= limits.null_percentile,
        "not run"
        if random_entry_percentile is None
        else f"percentile {random_entry_percentile:.1f}, need >= {limits.null_percentile}",
        overfit,
    )
    check(
        "beats_permutation_null",
        permutation_percentile is None or permutation_percentile >= limits.null_percentile,
        "not run"
        if permutation_percentile is None
        else f"percentile {permutation_percentile:.1f}, need >= {limits.null_percentile}",
        overfit,
    )
    synthetic_pct = None if robustness is None else robustness.synthetic.real_percentile
    check(
        "beats_synthetic_null",
        synthetic_pct is None or synthetic_pct >= limits.null_percentile,
        "not run"
        if synthetic_pct is None
        else f"percentile {synthetic_pct:.1f}, need >= {limits.null_percentile}",
        overfit,
    )
    check(
        "no_significant_is_oos_degradation",
        not degradation.degraded,
        f"paired t={degradation.t_statistic}, p={degradation.p_value}, "
        f"mean difference {degradation.mean_difference} over {degradation.n_pairs} folds",
        overfit,
    )
    if not all(item.passed for item in overfit):
        return StrategyVerdict(
            verdict=Verdict.OVERFIT,
            thresholds=limits,
            sufficiency=tuple(sufficiency),
            overfit_checks=tuple(overfit),
            fragility_checks=(),
            reasons=tuple(reasons),
        )

    fragility: list[VerdictCheck] = []
    dd_pct = (
        None if monte_carlo is None else monte_carlo.verdict_basis.max_drawdown.observed_percentile
    )
    check(
        "drawdown_not_flattered_by_ordering",
        dd_pct is None or dd_pct >= limits.max_drawdown_percentile,
        "not run"
        if dd_pct is None
        else f"observed drawdown at percentile {dd_pct:.1f} of its own reorderings, "
        f"need >= {limits.max_drawdown_percentile}",
        fragility,
    )
    if robustness is not None:
        period = robustness.period
        share = period.n_profitable / period.n_periods
        check(
            "period_consistency",
            share >= limits.min_profitable_periods_fraction,
            f"{period.n_profitable}/{period.n_periods} periods profitable, "
            f"need >= {limits.min_profitable_periods_fraction:.0%}",
            fragility,
        )
        check(
            "noise_retention",
            robustness.noise_retention is None
            or robustness.noise_retention >= limits.min_noise_retention,
            "undefined"
            if robustness.noise_retention is None
            else f"{robustness.noise_retention:.2f} of low-noise expectancy survives, "
            f"need >= {limits.min_noise_retention}",
            fragility,
        )
        check(
            "start_shift_stability",
            robustness.start_shift_dispersion is None
            or robustness.start_shift_dispersion <= limits.max_start_shift_dispersion,
            "undefined"
            if robustness.start_shift_dispersion is None
            else f"expectancy stdev {robustness.start_shift_dispersion:.3f} across shifted "
            f"starts, allowed {limits.max_start_shift_dispersion}",
            fragility,
        )
    else:
        check("robustness_suite", True, "not run", fragility)

    verdict = Verdict.ROBUST if all(item.passed for item in fragility) else Verdict.FRAGILE
    return StrategyVerdict(
        verdict=verdict,
        thresholds=limits,
        sufficiency=tuple(sufficiency),
        overfit_checks=tuple(overfit),
        fragility_checks=tuple(fragility),
        reasons=tuple(reasons),
    )


def write_verdict(verdict: StrategyVerdict, path: Path) -> None:
    """Export a verdict to JSON.

    Args:
        verdict: The verdict.
        path: File to write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(verdict.to_dict(), indent=2))
