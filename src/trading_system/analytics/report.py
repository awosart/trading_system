"""One run, or one walk-forward, rendered as a self-contained HTML page.

**A flat run and a walk-forward are two report kinds, not one report with a
flag.** A flat run is one strategy with one set of parameters, and its equity
curve is an account that existed: it is drawn in account currency. A
walk-forward is a *procedure* — "re-optimise every 270 days" — and its stitched
curve is a splice of returns from eight separate runs, each starting from the
same synthetic capital, each carrying different parameters. Handing both to one
template would give one heading two meanings.

Four things keep the stitched curve from being read as a single strategy's
account, and they are structural rather than a caption nobody reads:

* the fold seams are drawn on the chart, each labelled with the parameters that
  segment was traded on;
* the axis is a multiple of the starting point, never money — ``stitching.py``
  says of its own ``starting_equity`` that it "does not represent an account any
  run actually held";
* every metric appears three times, as the pooled figure, the median across
  folds and the fold-to-fold range, because the pooled expectancy weights folds
  by trade count and the median does not, and which of the two a reader meant is
  not this module's guess to make;
* IS and OOS appear as a pair, and no ratio between them is computed anywhere —
  the same decision P15 stage 1 took, for the same reason: the ratio is
  undefined the moment IS approaches zero, which it does in half the folds.

**A metric is flagged unreliable by walking the sample-size metadata, not by a
hand-maintained list.** ``metrics.field_meta`` already requires every derived
statistic to name the field carrying its sample size; this module reads that
metadata to decide which numbers to mark thin. A metric added upstream is
therefore flagged correctly without this file being edited, and one that forgets
its metadata fails the invariant test rather than rendering unmarked.
"""

import json
import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
from jinja2 import Environment, FileSystemLoader, select_autoescape
from markupsafe import Markup
from plotly.offline import get_plotlyjs

from trading_system.analytics.attribution import (
    ExcursionStats,
    QualityAnalysis,
    attribute_all,
    excursions,
    quality_vs_r,
)
from trading_system.analytics.metrics import (
    calmar_ratio,
    daily_curve,
    drawdown_stats,
    monthly_returns,
    r_distribution,
    sharpe_daily,
    sortino_daily,
    trade_stats,
    ulcer_index,
)
from trading_system.analytics.statistical import (
    DeflatedSharpeResult,
    ProbabilisticSharpeResult,
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
)
from trading_system.backtest.clock import StreamKey
from trading_system.backtest.portfolio import EquityPoint, TradeRecord
from trading_system.backtest.reproducibility import read_run
from trading_system.data.models import OHLCVFrame
from trading_system.validation.optimization import SearchSpace
from trading_system.validation.report import (
    FoldReport,
    StrategyVerdict,
    WalkForwardReport,
    build_report,
)
from trading_system.validation.stitching import stitch
from trading_system.validation.walkforward import read_result

#: Pooled trades below which every trade-derived statistic on the page is
#: marked thin. The same floor the verdict uses for ``min_trades_total``, so a
#: reader does not have to hold two numbers.
RELIABLE_TRADES = 100

#: Where the Jinja template lives.
TEMPLATE_DIR = Path(__file__).resolve().parents[3] / "docs"
TEMPLATE_NAME = "report_template.html"

_POSITIVE = "#1b7f5e"
_NEGATIVE = "#b3402f"
_NEUTRAL = "#4a6fa5"
_SEAM = "#9aa5b1"


class SourceKind(StrEnum):
    """Which of the two report kinds a page is."""

    #: One backtest: one parameter set, one account, money on the axis.
    RUN = "run"

    #: A walk-forward: a procedure, spliced returns, multiples on the axis.
    WALKFORWARD = "walkforward"


@dataclass(frozen=True)
class FoldSelection:
    """What the parameter search did on one fold's in-sample window.

    Separate from the fold's in-sample **run**, and the distinction is the
    whole point of this class. :class:`~trading_system.validation.walkforward.WalkForwardRunner`
    executes one in-sample run per fold using the **strategy file's own**
    parameters, and only then hands that fold to the selector, which evaluates
    its own trials. The two are different samples over the same window, and a
    reader who sees the first and assumes it is the second draws exactly the
    wrong conclusion — that a fold whose baseline traded nothing was optimised
    on nothing.

    Attributes:
        parameters: The point handed to the out-of-sample run.
        n_trials: Parameter sets the search evaluated.
        n_scored: How many of them produced a score at all. A trial that closed
            no trades is unscoreable and counts here as absent, so this is the
            number that answers "did the search have a sample".
        selected_score: The chosen point's in-sample score.
    """

    parameters: Mapping[str, Any]
    n_trials: int
    n_scored: int
    selected_score: float | None


@dataclass(frozen=True)
class FoldSegment:
    """One fold's stretch of the stitched curve, and what it was traded on.

    Attributes:
        index: Fold number.
        start: First instant of the fold's out-of-sample trade window.
        end: Last instant of it.
        parameters: The parameters the optimiser selected for this fold, or an
            empty mapping for a run that selected none.
        trade_count: Trades the fold closed out of sample.
        expectancy_r: The fold's own out-of-sample expectancy, or ``None``.
        insufficient: Whether the fold fell below ``min_trades_per_fold``.
        selection: What the search did on this fold, or ``None`` when nothing
            was searched.
        baseline_is_traded: Whether the fold's in-sample run — the one on the
            strategy file's own parameters — closed any trades. False is not a
            statement about the search; see :class:`FoldSelection`.
    """

    index: int
    start: datetime
    end: datetime
    parameters: Mapping[str, Any]
    trade_count: int
    expectancy_r: float | None
    insufficient: bool
    selection: FoldSelection | None = None
    baseline_is_traded: bool = True


@dataclass(frozen=True)
class SearchSummary:
    """What the parameter search did across the whole walk-forward, not just one fold.

    The per-fold ``Search scored / trials`` column already answers "did this
    fold's search have a sample". This answers a different question: what
    objective was optimised, how large was the space, and did the search see
    all of it. ``n_trials`` here is the number the Deflated Sharpe Ratio in
    this report is deflated by — stated once, in one place, rather than
    silently assumed identical to whatever the fold table shows.

    Attributes:
        method: ``"grid"``, ``"random"`` or ``"optuna"``.
        objective: What a trial was scored by.
        trial_budget: Trials each fold was allowed.
        feasible_size: Points in the search space after constraints — the
            whole space a grid search over this space could visit.
        truncated: Whether ``trial_budget`` was below ``feasible_size``,
            meaning even a grid search did not cover every point. Always
            ``False`` for a grid search, which this project's CLI refuses to
            start under-budgeted; meaningful for random/optuna search.
        total_trials: Trials evaluated, summed across every fold. Each fold's
            in-sample window is a different sample, so these are not
            duplicates of one another — this is the real count of independent
            evaluations behind the walk-forward's selections.
        total_scored: Of those, how many produced a score at all.
        n_folds_searched: Folds a search actually ran on.
    """

    method: str
    objective: str
    trial_budget: int
    feasible_size: int
    truncated: bool
    total_trials: int
    total_scored: int
    n_folds_searched: int


@dataclass(frozen=True)
class CostSensitivityPoint:
    """One cost multiplier's worth of a full re-run, not a resample.

    Attributes:
        multiplier: Applied to the instrument's declared spread and to every
            slippage/gap magnitude in the cost config.
        trades: Trades the run closed at this cost level — costs change which
            trades exist (an order that stops moving may miss a TTL fill), not
            only what they earn.
        expectancy_r: Mean realised R, or ``None`` with no trades.
        profit_factor: Gross profit over gross loss, or ``None`` with no
            trades or no losses to divide by.
        base_spread_points: The instrument's typical spread at this multiplier,
            in points — so the table shows the absolute cost alongside the
            relative one.
    """

    multiplier: float
    trades: int
    expectancy_r: float | None
    profit_factor: float | None
    base_spread_points: float


@dataclass(frozen=True)
class CostSensitivityReport:
    """A strategy's expectancy and profit factor as declared costs are scaled up.

    **Answers a different question than Monte Carlo cost perturbation does.**
    :func:`~trading_system.validation.monte_carlo.run_cost_sensitivity` draws
    random multipliers from a band and reports a distribution — it asks "how
    sensitive is this run to cost *uncertainty*". This asks "does the edge
    survive a *specific*, named cost level" — ×1.5 spread, ×2 spread — which is
    the question that decides whether a strategy lives only on today's
    execution quality.

    Attributes:
        points: One entry per multiplier, ascending.
        note: Caveats about how the multiplier was applied — see
            :func:`cost_sensitivity_from_disk` on why this is not optional.
    """

    points: tuple[CostSensitivityPoint, ...]
    note: str


@dataclass(frozen=True)
class ReportSource:
    """Everything a page is rendered from, with its own provenance attached.

    Attributes:
        kind: Flat run or walk-forward.
        label: What to call it — a strategy id, a run id, a wf id.
        curve: The equity curve. For a walk-forward this is the stitched
            return series, not an account.
        trades: Closed trades. For a walk-forward, every fold's out-of-sample
            trades pooled — the only sample large enough to slice.
        streams: Bars, for excursions. Empty means the MAE/MFE section reports
            that it had nothing to measure against.
        money_is_real: Whether ``curve`` is an account that existed. False for
            a stitched curve, which switches the axis to multiples and
            suppresses currency formatting.
        verdict: The grade, if one was rendered for this source.
        folds: Per-fold reports, walk-forward only.
        segments: Fold seams with their parameters, walk-forward only.
        is_oos_pairs: ``(is_expectancy_r, oos_expectancy_r)`` per fold, with
            ``None`` where a side had no trades. Never divided.
        search: What the search did across the whole walk-forward, or
            ``None`` for a flat run or one that selected nothing.
        cost_sensitivity: Expectancy and profit factor at named cost
            multipliers, or ``None`` when it was not computed for this source.
    """

    kind: SourceKind
    label: str
    curve: Sequence[EquityPoint]
    trades: Sequence[TradeRecord]
    streams: Mapping[StreamKey, OHLCVFrame]
    money_is_real: bool
    verdict: StrategyVerdict | None = None
    folds: tuple[FoldReport, ...] = ()
    segments: tuple[FoldSegment, ...] = ()
    is_oos_pairs: tuple[tuple[float | None, float | None], ...] = ()
    search: SearchSummary | None = None
    cost_sensitivity: CostSensitivityReport | None = None


# ---------------------------------------------------------------------------
# Loading a source off disk
# ---------------------------------------------------------------------------


def source_from_run(
    directory: Path,
    *,
    streams: Mapping[StreamKey, OHLCVFrame] | None = None,
    label: str | None = None,
    verdict: StrategyVerdict | None = None,
    cost_sensitivity: CostSensitivityReport | None = None,
) -> ReportSource:
    """Load one stored backtest as a report source.

    Args:
        directory: The run's directory, as :func:`write_run` produced it.
        streams: The run's bars, for excursions. Omitted means the excursion
            section says it had nothing to measure against, rather than being
            absent.
        label: What to call it. Defaults to the run id.
        verdict: A grade, if one was rendered for this run.
        cost_sensitivity: Expectancy/profit-factor at named cost multipliers,
            if computed — see :func:`cost_sensitivity_from_disk`.

    Returns:
        The source, with money on the axis: a flat run's curve is an account
        that existed.
    """
    stored = read_run(directory)
    return ReportSource(
        kind=SourceKind.RUN,
        label=label or directory.name,
        curve=stored.result.curve,
        trades=stored.result.trades,
        streams=streams or {},
        money_is_real=True,
        verdict=verdict,
        cost_sensitivity=cost_sensitivity,
    )


def source_from_walkforward(
    wf_directory: Path,
    *,
    store_root: Path,
    min_trades_per_fold: int,
    streams: Mapping[StreamKey, OHLCVFrame] | None = None,
    label: str | None = None,
    verdict: StrategyVerdict | None = None,
    selections: Mapping[int, FoldSelection] | None = None,
    search: SearchSummary | None = None,
    cost_sensitivity: CostSensitivityReport | None = None,
) -> ReportSource:
    """Load a finished walk-forward as a report source.

    The curve is the stitched out-of-sample return series and the trades are
    every fold's out-of-sample trades pooled. ``money_is_real`` is False, which
    is what puts multiples rather than currency on the axis.

    Args:
        wf_directory: ``runs/walkforward/{wf_id}``.
        store_root: Where the per-fold runs live — ``runs``.
        min_trades_per_fold: Threshold folds are flagged against.
        streams: Bars, for excursions.
        label: What to call it. Defaults to the wf id.
        verdict: The grade, if one was rendered.
        selections: What the search did on each fold, by fold index — read
            from ``optimization.json`` by :func:`fold_selections_from_disk`.
            Absent means the seams are drawn without labels, which is the
            honest rendering of a walk-forward that selected nothing.
        search: What the search did across the whole walk-forward — read by
            :func:`search_summary_from_disk`.
        cost_sensitivity: Expectancy/profit-factor at named cost multipliers,
            if computed — see :func:`cost_sensitivity_from_disk`.

    Returns:
        The source.
    """
    wf_id = wf_directory.name
    result = read_result(wf_directory / "manifest.json", wf_id, store_root)
    report = build_report(result, min_trades_per_fold=min_trades_per_fold)

    curves: list[Sequence[EquityPoint]] = []
    pooled: list[TradeRecord] = []
    for fold_run in result.folds:
        stored = read_run(fold_run.oos_run.path)
        curves.append(stored.result.curve)
        pooled.extend(stored.result.trades)
    stitched = stitch([fold_run.fold for fold_run in result.folds], curves)

    chosen = selections or {}
    segments = tuple(
        FoldSegment(
            index=fold.index,
            start=fold.oos_window.trade_start,
            end=fold.oos_window.trade_end,
            parameters=chosen[fold.index].parameters if fold.index in chosen else {},
            trade_count=fold.oos_trade_count,
            expectancy_r=fold.oos_expectancy_r,
            insufficient=fold.insufficient,
            selection=chosen.get(fold.index),
            baseline_is_traded=fold.is_trade_count > 0,
        )
        for fold in report.folds
    )
    return ReportSource(
        kind=SourceKind.WALKFORWARD,
        label=label or wf_id,
        curve=stitched.points,
        trades=pooled,
        streams=streams or {},
        money_is_real=False,
        verdict=verdict,
        folds=report.folds,
        segments=segments,
        is_oos_pairs=tuple((fold.is_expectancy_r, fold.oos_expectancy_r) for fold in report.folds),
        search=search,
        cost_sensitivity=cost_sensitivity,
    )


def fold_selections_from_disk(wf_directory: Path) -> dict[int, FoldSelection]:
    """What the search did on each fold, from a walk-forward's ``optimization.json``.

    Reads ``n_trials`` and ``n_scored`` alongside the chosen point, because the
    chosen point alone cannot say whether it was chosen from anything. Those two
    numbers are the fold's real search sample; the fold's in-sample **run** is a
    different sample on the same window — see :class:`FoldSelection`.

    Args:
        wf_directory: ``runs/walkforward/{wf_id}``.

    Returns:
        Fold index to its selection, empty when the walk-forward selected
        nothing (an ``IdentitySelector`` run writes no such file, and that is
        not an error).
    """
    path = wf_directory / "optimization.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    return {
        int(fold["fold_index"]): FoldSelection(
            parameters=dict(fold.get("selected") or {}),
            n_trials=int(fold.get("n_trials", 0)),
            n_scored=int(fold.get("n_scored", 0)),
            selected_score=fold.get("selected_score"),
        )
        for fold in payload.get("folds", [])
    }


def search_summary_from_disk(wf_directory: Path) -> SearchSummary | None:
    """The whole-walk-forward search summary, from ``optimization.json``.

    Args:
        wf_directory: ``runs/walkforward/{wf_id}``.

    Returns:
        The summary, or ``None`` when the walk-forward selected nothing (an
        ``IdentitySelector`` run writes no ``optimization.json``).
    """
    path = wf_directory / "optimization.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    space = SearchSpace.model_validate(payload["space"])
    folds = payload.get("folds", [])
    total_trials = sum(int(fold.get("n_trials", 0)) for fold in folds)
    total_scored = sum(int(fold.get("n_scored", 0)) for fold in folds)
    trial_budget = int(payload["trial_budget"])
    feasible_size = space.feasible_size()
    return SearchSummary(
        method=str(payload["method"]),
        objective=str(payload["objective"]),
        trial_budget=trial_budget,
        feasible_size=feasible_size,
        truncated=trial_budget < feasible_size,
        total_trials=total_trials,
        total_scored=total_scored,
        n_folds_searched=len(folds),
    )


def cost_sensitivity_from_disk(path: Path) -> CostSensitivityReport:
    """Load a cost-sensitivity table written by an offline sweep.

    **Why this is a loader over a JSON file rather than a function this module
    calls directly.** Computing a point costs a full backtest — this module
    otherwise only reads already-materialised results, never executes one
    (the same boundary that keeps Exit from importing Entry: a renderer that
    can also run backtests is a renderer that can no longer be trusted to be
    cheap). The sweep is a separate offline step; this only reads its output.

    Args:
        path: The JSON file the sweep wrote.

    Returns:
        The report.
    """
    payload = json.loads(path.read_text())
    points = tuple(
        CostSensitivityPoint(
            multiplier=float(row["multiplier"]),
            trades=int(row["trades"]),
            expectancy_r=row.get("expectancy_r"),
            profit_factor=row.get("profit_factor"),
            base_spread_points=float(row["base_spread_points"]),
        )
        for row in payload["points"]
    )
    return CostSensitivityReport(points=points, note=str(payload["note"]))


# ---------------------------------------------------------------------------
# Metrics, and which of them the sample cannot support
# ---------------------------------------------------------------------------


#: Two-to-three sentence explanations for the metrics named in the P14 stage 3
#: brief, keyed by ``(group, field name)`` exactly as :func:`metric_rows`
#: produces them. Deliberately not exhaustive: a metric with no entry here
#: renders without a "?" icon rather than with a guessed explanation, and
#: guessing what a number means is worse than saying nothing about it.
METRIC_TOOLTIPS: dict[tuple[str, str], str] = {
    ("sharpe", "value"): (
        "Mean return over its own standard deviation, annualised. Treats a good "
        "week and a bad week as equally risky as long as they're the same size — "
        "it does not know the difference between volatility on the way up and on "
        "the way down. Above 1 is usually called workable, above 2 is rare "
        "outside overfitting."
    ),
    ("sortino", "value"): (
        "Sharpe's cousin, but only downside moves count as risk — a big winning "
        "day does not get penalised. Reads higher than Sharpe on a strategy with "
        "an asymmetric return shape, which is most trend and breakout systems. "
        "Compare the two: a large gap says the strategy's upside is unusually "
        "lumpy."
    ),
    ("calmar", "value"): (
        "CAGR divided by the worst drawdown seen in the trailing window (3 years "
        "by convention). Answers a very concrete question: how much did you make "
        "for the worst pain you had to sit through. Above 1 means the annual "
        "return exceeded the worst drawdown; that is a low bar, not a good one."
    ),
    ("ulcer", "value"): (
        "Root-mean-square of the drawdown-from-peak percentage, one term per "
        "day. Unlike max drawdown, it penalises a long, shallow underwater "
        "stretch almost as much as a short, deep one — it is measuring how much "
        "time was spent hurting, not just how badly. Lower is better; there is "
        "no universal good value, only comparisons between strategies on the "
        "same data."
    ),
    ("psr", "value"): (
        "Probability the true Sharpe ratio is actually positive, given how "
        "noisy this sample is — not the Sharpe itself, a confidence about it. "
        "0.95 is the conventional bar. A strategy can have a flattering Sharpe "
        "and a low PSR at the same time; that combination means the sample is "
        "too short or too skewed to trust the Sharpe."
    ),
    ("dsr", "value"): (
        "Probabilistic Sharpe Ratio, but benchmarked against the best of "
        "n_trials independent attempts rather than against zero — see n_trials "
        "on this same row. An ordinary Sharpe computed after 125 parameter "
        "combinations were tried per fold is comparing the winner of a lottery "
        "to a strategy that only ever bought one ticket; DSR corrects for how "
        "many tickets were actually bought. It is always lower than PSR unless "
        "n_trials is 1, and the gap between them is the size of the correction."
    ),
    ("trades", "profit_factor"): (
        "Gross profit divided by gross loss, both taken as positive numbers. "
        "1.0 is exactly break-even before you count what it cost to be wrong "
        "more often than right, or right more often than wrong — it says "
        "nothing about win rate on its own. Infinite means no losing trade in "
        "the sample, which is a small-sample artefact, not an achievement."
    ),
    ("trades", "expectancy_r"): (
        "Mean realised R per trade — the number that answers 'if I take this "
        "trade a thousand times, what do I make on average, in units of what I "
        "risked'. Positive is necessary but not sufficient: it says nothing "
        "about how the wins and losses are distributed, which is what the R "
        "histogram and MAE/MFE are for."
    ),
}


@dataclass(frozen=True)
class MetricRow:
    """One metric as the page shows it.

    Attributes:
        group: Which statistics object it came from.
        name: Field name.
        value: Formatted for display.
        sample: The sample size it was computed over, or ``None`` for a field
            that is not derived from one.
        unreliable: Whether that sample is below :data:`RELIABLE_TRADES`.
        tooltip: A short explanation, or ``None`` when :data:`METRIC_TOOLTIPS`
            has none for this ``(group, name)``.
    """

    group: str
    name: str
    value: str
    sample: int | None
    unreliable: bool
    tooltip: str | None = None


def _format(value: object) -> str:
    """Render a metric value for a table cell."""
    if value is None:
        return "—"
    if isinstance(value, float):
        if math.isinf(value):
            return "∞" if value > 0 else "−∞"
        return f"{value:,.4f}"
    if isinstance(value, Decimal):
        return f"{value:,.2f}"
    return str(value)


def metric_rows(group: str, stats: object) -> list[MetricRow]:
    """Flatten a statistics dataclass into rows, marking those the sample cannot support.

    Reads the ``field_meta`` metadata every dataclass in
    :mod:`trading_system.analytics.metrics` is required to carry: a field of
    kind ``"value"`` names the sibling holding its sample size, and that size
    decides whether the row is flagged. Nothing here enumerates metric names,
    so a metric added upstream arrives flagged correctly.

    Args:
        group: Label for the block of rows.
        stats: A dataclass instance from the analytics layer.

    Returns:
        One row per field, excluding raw sample fields (a tuple of every
        observation is not a table cell).
    """
    if not is_dataclass(stats) or isinstance(stats, type):
        raise TypeError(f"{group}: expected a stats dataclass instance, got {type(stats)!r}")
    rows: list[MetricRow] = []
    for entry in fields(stats):
        kind = entry.metadata.get("kind")
        if kind == "raw":
            continue
        value = getattr(stats, entry.name)
        sample: int | None = None
        if kind == "value":
            sibling = entry.metadata.get("n_field")
            sample = int(getattr(stats, sibling)) if sibling else None
        elif kind == "n":
            sample = int(value)
        rows.append(
            MetricRow(
                group=group,
                name=entry.name,
                value=_format(value),
                sample=sample,
                unreliable=sample is not None and sample < RELIABLE_TRADES,
                tooltip=METRIC_TOOLTIPS.get((group, entry.name)),
            )
        )
    return rows


def all_metric_rows(curve: Sequence[EquityPoint], trades: Sequence[TradeRecord]) -> list[MetricRow]:
    """Every metric the stage-1 layer can produce for this sample.

    A statistic the sample cannot support at all raises upstream by design, and
    every one of those cases is a shape a real run takes: Sortino with no
    downside deviation, Sharpe with a single daily return, **drawdown of a curve
    that never dipped below its running peak**. Each becomes an absent group
    rather than a crash — a page that refuses to render because one ratio is
    undefined is a page nobody can read, and the missing group is visible in the
    table by not being there.

    The guard is applied uniformly rather than to the two groups that were
    expected to need it: which statistics are undefined on which sample is a
    property of the sample, not something this function should be predicting.
    """
    rows: list[MetricRow] = []
    if curve:
        daily = daily_curve(curve)
        for label, compute in (
            ("drawdown", drawdown_stats),
            ("ulcer", ulcer_index),
            ("sharpe", sharpe_daily),
            ("sortino", sortino_daily),
        ):
            try:
                rows += metric_rows(label, compute(daily))
            except ValueError:
                continue
        try:
            calmar = calmar_ratio(daily)
        except ValueError:
            pass
        else:
            rows.append(
                MetricRow(
                    group="calmar",
                    name="value",
                    value=_format(calmar),
                    sample=None,
                    unreliable=False,
                    tooltip=METRIC_TOOLTIPS.get(("calmar", "value")),
                )
            )
    if trades:
        for label, over_trades in (("trades", trade_stats), ("r_distribution", r_distribution)):
            try:
                rows += metric_rows(label, over_trades(trades))
            except ValueError:
                continue
    return rows


# ---------------------------------------------------------------------------
# The Sharpe family: ordinary, probability-adjusted, deflated for multiple testing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SharpeFamily:
    """Ordinary Sharpe next to its two more careful cousins, so the gap between them is visible.

    Each of the three can be individually undefined, and each carries its own
    reason rather than a shared "n/a" — the reasons are different (too few
    daily returns against no search having been recorded) and a reader
    deciding what to do next needs to know which one applies.

    Attributes:
        sharpe: Ordinary annualised Sharpe, or ``None``.
        sharpe_reason: Why ``sharpe`` is ``None``, when it is.
        psr: Probabilistic Sharpe, or ``None``.
        psr_reason: Why ``psr`` is ``None``, when it is.
        dsr: Deflated Sharpe, or ``None``.
        dsr_reason: Why ``dsr`` is ``None``, when it is — in particular,
            "no parameter search was recorded for this source" when
            ``n_trials`` was never supplied, which is the ordinary case for a
            flat run and for a walk-forward that selected nothing.
        n_trials: What ``dsr`` deflated against, echoed here so the number
            sits next to the ratio it explains rather than only in the search
            summary section.
    """

    sharpe: float | None
    sharpe_reason: str | None
    psr: ProbabilisticSharpeResult | None
    psr_reason: str | None
    dsr: DeflatedSharpeResult | None
    dsr_reason: str | None
    n_trials: int | None


def sharpe_family(curve: Sequence[EquityPoint], n_trials: int | None) -> SharpeFamily | None:
    """Ordinary Sharpe, PSR and DSR from one curve, each computed independently.

    Args:
        curve: The equity curve.
        n_trials: Total parameter combinations the reported result was
            selected from — :attr:`SearchSummary.total_trials` for a
            walk-forward that searched, ``None`` for one that did not and for
            every flat run. ``None`` does not mean "assume 1"; it means "this
            question does not apply here", and the family says so rather than
            silently reporting PSR(0) relabelled as DSR.

    Returns:
        The family, or ``None`` when there is no curve at all.
    """
    if not curve:
        return None
    daily = daily_curve(curve)

    sharpe: float | None = None
    sharpe_reason: str | None = None
    try:
        sharpe = sharpe_daily(daily).value
    except ValueError as error:
        sharpe_reason = str(error)

    psr: ProbabilisticSharpeResult | None = None
    psr_reason: str | None = None
    try:
        psr = probabilistic_sharpe_ratio(daily)
    except ValueError as error:
        psr_reason = str(error)

    dsr: DeflatedSharpeResult | None = None
    dsr_reason: str | None = None
    if n_trials is None:
        dsr_reason = (
            "no parameter search was recorded for this result — DSR only applies "
            "once there is a number of trials to deflate against; see PSR above "
            "for the same question without a multiple-testing correction"
        )
    else:
        try:
            dsr = deflated_sharpe_ratio(daily, n_trials=n_trials)
        except ValueError as error:
            dsr_reason = str(error)

    return SharpeFamily(
        sharpe=sharpe,
        sharpe_reason=sharpe_reason,
        psr=psr,
        psr_reason=psr_reason,
        dsr=dsr,
        dsr_reason=dsr_reason,
        n_trials=n_trials,
    )


# ---------------------------------------------------------------------------
# Money, in units a reader does not have to convert in their head
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MoneySummary:
    """Starting capital to ending capital, in currency and in per cent, up top.

    **Real for a flat run, hypothetical for a walk-forward, and the field that
    says which is which is :attr:`hypothetical`.** A flat run's curve is an
    account that existed — these numbers are what it did. A walk-forward's
    stitched curve is a splice of eight OOS runs' *returns*, compounded onto a
    nominal starting balance nobody's sizing ever actually used (each fold's
    own OOS run sized from its own reset balance, not from this running
    total). The dollar figures here are that nominal balance carried through
    the same return sequence the multiples chart already plots — not a new
    computation, a relabelling of one — and they are shown because the
    request for "simple money at a glance" is legitimate even for a
    procedure, provided the reader is told, prominently, which kind of number
    they are looking at.

    Attributes:
        starting_equity: First point's equity.
        ending_equity: Last point's equity.
        pnl_money: ``ending_equity - starting_equity``.
        pnl_pct: The same, as a fraction of ``starting_equity``.
        max_drawdown_money: Deepest peak-to-trough drop, negative.
        max_drawdown_pct: The same, as a fraction of the peak it fell from.
        trade_count: Trades closed.
        period_start: First instant on the curve.
        period_end: Last instant on the curve.
        hypothetical: Whether the money figures are a real account
            (``False``) or a nominal-capital relabelling of a stitched return
            series (``True``).
    """

    starting_equity: Decimal
    ending_equity: Decimal
    pnl_money: Decimal
    pnl_pct: float
    max_drawdown_money: Decimal
    max_drawdown_pct: float
    trade_count: int
    period_start: datetime
    period_end: datetime
    hypothetical: bool


def money_summary(
    curve: Sequence[EquityPoint], trades: Sequence[TradeRecord], *, hypothetical: bool
) -> MoneySummary | None:
    """The top-of-page money block.

    Args:
        curve: The equity curve — an account for a flat run, a stitched
            return series for a walk-forward.
        trades: Closed trades, for the count.
        hypothetical: Whether ``curve`` is a real account. Echoed onto
            :attr:`MoneySummary.hypothetical` for the template to caption
            correctly; this function computes the same arithmetic either way,
            because the numbers are honest in both cases — only their
            meaning differs.

    Returns:
        The summary, or ``None`` when the curve is empty.
    """
    if not curve:
        return None
    starting = curve[0].equity
    ending = curve[-1].equity
    max_dd_money = Decimal(0)
    max_dd_pct = 0.0
    try:
        drawdown = drawdown_stats(daily_curve(curve))
    except ValueError:
        pass
    else:
        max_dd_money = drawdown.max_drawdown_money
        max_dd_pct = drawdown.max_drawdown_pct
    return MoneySummary(
        starting_equity=starting,
        ending_equity=ending,
        pnl_money=ending - starting,
        pnl_pct=float((ending - starting) / starting) if starting != 0 else 0.0,
        max_drawdown_money=max_dd_money,
        max_drawdown_pct=max_dd_pct,
        trade_count=len(trades),
        period_start=curve[0].ts,
        period_end=curve[-1].ts,
        hypothetical=hypothetical,
    )


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _axis_title(source: ReportSource) -> str:
    """What the equity axis is denominated in."""
    return "Equity (account currency)" if source.money_is_real else "Multiple of starting point"


def _curve_values(source: ReportSource) -> tuple[list[datetime], list[float]]:
    """The curve as plottable values, normalised to multiples when money is not real."""
    times = [point.ts for point in source.curve]
    equities = [float(point.equity) for point in source.curve]
    if source.money_is_real or not equities or equities[0] == 0.0:
        return times, equities
    base = equities[0]
    return times, [value / base for value in equities]


def equity_figure(source: ReportSource) -> go.Figure:
    """Equity with drawdown shaded, and fold seams where the source has them."""
    times, values = _curve_values(source)
    figure = go.Figure()
    if values:
        peaks: list[float] = []
        running = values[0]
        for value in values:
            running = max(running, value)
            peaks.append(running)
        figure.add_trace(
            go.Scatter(
                x=times,
                y=peaks,
                line={"width": 0},
                hoverinfo="skip",
                showlegend=False,
                name="peak",
            )
        )
        figure.add_trace(
            go.Scatter(
                x=times,
                y=values,
                fill="tonexty",
                fillcolor="rgba(179, 64, 47, 0.18)",
                line={"color": _NEUTRAL, "width": 1.6},
                name="equity",
            )
        )
    for segment in source.segments:
        figure.add_vline(x=segment.start, line={"color": _SEAM, "width": 1, "dash": "dot"})
        figure.add_annotation(
            x=segment.start,
            yref="paper",
            y=1.0,
            showarrow=False,
            yanchor="bottom",
            font={"size": 9, "color": "#5b6672"},
            text=f"f{segment.index}<br>{_short_params(segment.parameters)}",
        )
    figure.update_layout(
        title=(
            "Out-of-sample, stitched fold to fold — parameters are re-selected at every seam"
            if source.kind is SourceKind.WALKFORWARD
            else "Equity, with drawdown shaded"
        ),
        yaxis_title=_axis_title(source),
        xaxis_title="",
    )
    return _styled(figure)


def _short_params(parameters: Mapping[str, Any]) -> str:
    """Fold parameters, compact enough to sit above a seam."""
    if not parameters:
        return "—"
    return "<br>".join(f"{name}={value}" for name, value in parameters.items())


def r_distribution_figure(trades: Sequence[TradeRecord]) -> go.Figure:
    """Histogram of realised R with the expectancy drawn on it."""
    values = [trade.realized_r for trade in trades]
    figure = go.Figure()
    if values:
        figure.add_trace(go.Histogram(x=values, nbinsx=40, marker={"color": _NEUTRAL}, name="R"))
        mean = math.fsum(values) / len(values)
        figure.add_vline(
            x=mean,
            line={"color": _POSITIVE if mean > 0 else _NEGATIVE, "width": 2},
            annotation_text=f"expectancy {mean:+.4f}R",
            annotation_position="top right",
        )
        figure.add_vline(x=0.0, line={"color": _SEAM, "width": 1, "dash": "dot"})
    figure.update_layout(
        title="Realised R per trade", xaxis_title="R", yaxis_title="trades", bargap=0.05
    )
    return _styled(figure)


def monthly_heatmap_figure(curve: Sequence[EquityPoint]) -> go.Figure:
    """Monthly returns as a year-by-month grid."""
    figure = go.Figure()
    if curve:
        months = monthly_returns(daily_curve(curve))
        years = sorted({month.year for month in months})
        grid: list[list[float | None]] = [[None] * 12 for _ in years]
        index = {year: position for position, year in enumerate(years)}
        for month in months:
            grid[index[month.year]][month.month - 1] = round(month.return_pct * 100.0, 2)
        figure.add_trace(
            go.Heatmap(
                z=grid,
                x=[
                    "Jan",
                    "Feb",
                    "Mar",
                    "Apr",
                    "May",
                    "Jun",
                    "Jul",
                    "Aug",
                    "Sep",
                    "Oct",
                    "Nov",
                    "Dec",
                ],
                y=[str(year) for year in years],
                colorscale=[[0.0, _NEGATIVE], [0.5, "#f2f2f0"], [1.0, _POSITIVE]],
                zmid=0.0,
                texttemplate="%{z}",
                textfont={"size": 10},
                colorbar={"title": "%"},
            )
        )
    figure.update_layout(title="Monthly return, per cent")
    return _styled(figure)


def underwater_figure(curve: Sequence[EquityPoint]) -> go.Figure:
    """Percent below the running peak, every day, under the equity curve.

    A single max-drawdown number cannot distinguish one deep episode from
    five shallow ones that never individually reached it — and that
    distinction is exactly what decides whether a prop desk's daily-loss
    circuit breaker would have tripped once or five times. This draws every
    day's distance from its own running peak, not only the worst one.
    """
    figure = go.Figure()
    if curve:
        daily = daily_curve(curve)
        peak = daily.equity[0]
        pct: list[float] = []
        for equity in daily.equity:
            peak = max(peak, equity)
            pct.append(float(equity / peak - 1) * 100.0 if peak != 0 else 0.0)
        figure.add_trace(
            go.Scatter(
                x=list(daily.days),
                y=pct,
                fill="tozeroy",
                fillcolor="rgba(179, 64, 47, 0.28)",
                line={"color": _NEGATIVE, "width": 1.2},
                name="drawdown",
            )
        )
    figure.update_layout(
        title="Underwater curve: % below the running peak, every day",
        yaxis_title="% from peak",
        xaxis_title="",
    )
    return _styled(figure)


def duration_histogram_figure(trades: Sequence[TradeRecord]) -> go.Figure:
    """Histogram of time held, in hours."""
    figure = go.Figure()
    if trades:
        hours = [(trade.closed_at - trade.opened_at).total_seconds() / 3600.0 for trade in trades]
        figure.add_trace(go.Histogram(x=hours, nbinsx=40, marker={"color": _NEUTRAL}, name="hours"))
    figure.update_layout(
        title="Time held, per trade", xaxis_title="hours", yaxis_title="trades", bargap=0.05
    )
    return _styled(figure)


def duration_vs_r_figure(trades: Sequence[TradeRecord]) -> go.Figure:
    """Time held against realised R — does the strategy sit longer in losers than winners."""
    figure = go.Figure()
    if trades:
        hours = [(trade.closed_at - trade.opened_at).total_seconds() / 3600.0 for trade in trades]
        rs = [trade.realized_r for trade in trades]
        figure.add_trace(
            go.Scatter(
                x=hours,
                y=rs,
                mode="markers",
                marker={
                    "size": 6,
                    "color": rs,
                    "colorscale": [[0.0, _NEGATIVE], [0.5, "#e8e8e4"], [1.0, _POSITIVE]],
                    "cmid": 0.0,
                    "line": {"width": 0.5, "color": "#ffffff"},
                },
                name="trades",
            )
        )
        figure.add_hline(y=0.0, line={"color": _SEAM, "width": 1, "dash": "dot"})
    figure.update_layout(
        title="Time held against realised R",
        xaxis_title="hours held",
        yaxis_title="R",
    )
    return _styled(figure)


def excursion_figure(stats: ExcursionStats) -> go.Figure:
    """MAE against MFE, one point per trade, coloured by realised R."""
    figure = go.Figure()
    if stats.excursions:
        figure.add_trace(
            go.Scatter(
                x=[item.mae_r for item in stats.excursions],
                y=[item.mfe_r for item in stats.excursions],
                mode="markers",
                marker={
                    "size": 7,
                    "color": [item.realized_r for item in stats.excursions],
                    "colorscale": [[0.0, _NEGATIVE], [0.5, "#e8e8e4"], [1.0, _POSITIVE]],
                    "cmid": 0.0,
                    "colorbar": {"title": "R"},
                    "line": {"width": 0.5, "color": "#ffffff"},
                },
                text=[item.position_id for item in stats.excursions],
                hovertemplate="MAE %{x:.2f}R<br>MFE %{y:.2f}R<br>%{text}<extra></extra>",
                name="trades",
            )
        )
        figure.add_vline(
            x=1.0,
            line={"color": _SEAM, "width": 1, "dash": "dot"},
            annotation_text="1R against",
            annotation_position="top",
        )
    figure.update_layout(
        title="Excursions: how far each trade ran against and for its holder",
        xaxis_title="Maximum adverse excursion (R)",
        yaxis_title="Maximum favourable excursion (R)",
    )
    return _styled(figure)


def per_fold_figure(source: ReportSource) -> go.Figure:
    """Each fold's own out-of-sample expectancy, thin folds hatched."""
    figure = go.Figure()
    if source.segments:
        values = [
            0.0 if segment.expectancy_r is None else segment.expectancy_r
            for segment in source.segments
        ]
        figure.add_trace(
            go.Bar(
                x=[f"fold {segment.index}" for segment in source.segments],
                y=values,
                marker={
                    "color": [_POSITIVE if value > 0 else _NEGATIVE for value in values],
                    "pattern": {
                        "shape": [
                            "/" if segment.insufficient else "" for segment in source.segments
                        ]
                    },
                },
                text=[f"n={segment.trade_count}" for segment in source.segments],
                textposition="outside",
                name="OOS expectancy",
            )
        )
        figure.add_hline(y=0.0, line={"color": _SEAM, "width": 1})
    figure.update_layout(
        title="Out-of-sample expectancy by fold (hatched: below the per-fold trade floor)",
        yaxis_title="expectancy (R)",
    )
    return _styled(figure)


def is_oos_figure(source: ReportSource) -> go.Figure:
    """In-sample against out-of-sample expectancy, one point per fold, with ``y = x``."""
    pairs = [(a, b) for a, b in source.is_oos_pairs if a is not None and b is not None]
    figure = go.Figure()
    if pairs:
        xs = [a for a, _ in pairs]
        ys = [b for _, b in pairs]
        span = [min(xs + ys), max(xs + ys)]
        figure.add_trace(
            go.Scatter(
                x=span, y=span, mode="lines", line={"color": _SEAM, "dash": "dot"}, name="y = x"
            )
        )
        figure.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers+text",
                marker={"size": 10, "color": _NEUTRAL},
                text=[str(index) for index, _ in enumerate(pairs)],
                textposition="top center",
                name="folds",
            )
        )
    figure.update_layout(
        title="In-sample against out-of-sample expectancy, by fold (no ratio is taken)",
        xaxis_title="IS expectancy (R)",
        yaxis_title="OOS expectancy (R)",
    )
    return _styled(figure)


def comparison_figure(sources: Sequence[ReportSource]) -> go.Figure:
    """Every source's curve on one axis, each normalised to its own starting point."""
    figure = go.Figure()
    for source in sources:
        times = [point.ts for point in source.curve]
        equities = [float(point.equity) for point in source.curve]
        if not equities or equities[0] == 0.0:
            continue
        base = equities[0]
        figure.add_trace(
            go.Scatter(
                x=times, y=[value / base for value in equities], mode="lines", name=source.label
            )
        )
    figure.update_layout(
        title="Equity, each normalised to its own starting point",
        yaxis_title="Multiple of starting point",
    )
    return _styled(figure)


def _styled(figure: go.Figure) -> go.Figure:
    """The one place layout defaults are set, so every chart on a page matches."""
    figure.update_layout(
        template="plotly_white",
        margin={"l": 60, "r": 30, "t": 60, "b": 50},
        height=420,
        font={"family": "Inter, -apple-system, Segoe UI, sans-serif", "size": 12},
        title={"font": {"size": 14}},
        hovermode="closest",
    )
    return figure


# ---------------------------------------------------------------------------
# Aggregates a walk-forward reports three ways
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PooledAndPerFold:
    """One metric as pooled, as a median across folds, and as a range.

    Three columns rather than one because they answer different questions and
    disagree in a way that is itself informative: the pooled figure weights a
    twenty-trade fold twenty times as heavily as a one-trade fold, and the
    median weights them alike.

    Attributes:
        name: What is measured.
        pooled: Over every out-of-sample trade at once, or ``None``.
        median: Median of the per-fold values, or ``None``.
        low: Smallest per-fold value.
        high: Largest per-fold value.
        n_folds_scored: Folds that produced a value at all.
    """

    name: str
    pooled: float | None
    median: float | None
    low: float | None
    high: float | None
    n_folds_scored: int


def pooled_and_per_fold(report: WalkForwardReport) -> list[PooledAndPerFold]:
    """The stitched figures beside the fold-by-fold distribution behind them."""
    rows: list[PooledAndPerFold] = []
    for name, pooled, per_fold in (
        ("expectancy_r", report.stitched_expectancy_r, report.per_fold_oos_expectancy_r),
        ("sortino", report.stitched_sortino, report.per_fold_oos_sortino),
    ):
        scored = [value for value in per_fold if value is not None]
        rows.append(
            PooledAndPerFold(
                name=name,
                pooled=pooled,
                median=statistics.median(scored) if scored else None,
                low=min(scored) if scored else None,
                high=max(scored) if scored else None,
                n_folds_scored=len(scored),
            )
        )
    rows.append(
        PooledAndPerFold(
            name="sharpe",
            pooled=report.stitched_sharpe,
            median=None,
            low=None,
            high=None,
            n_folds_scored=report.n_folds,
        )
    )
    return rows


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _environment() -> Environment:
    """The Jinja environment, autoescaping on."""
    return Environment(
        loader=FileSystemLoader(TEMPLATE_DIR),
        autoescape=select_autoescape(["html"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )


def _figure_html(figure: go.Figure) -> Markup:
    """One figure as a div, with the library loaded once by the page rather than per chart.

    Returned as :class:`~markupsafe.Markup` so the template renders it as
    markup instead of escaping it into visible angle brackets. Marking it here
    rather than writing ``| safe`` at each of six call sites keeps the decision
    that this HTML is trusted in one place — and it is trusted for a concrete
    reason: plotly serialises every value that came from the run into JSON,
    so no label reaches the document as raw markup.
    """
    html: str = figure.to_html(full_html=False, include_plotlyjs=False, default_width="100%")
    return Markup(html)


@dataclass(frozen=True)
class RenderedReport:
    """A page and the data behind it.

    Attributes:
        html: The complete document.
        metrics: The same numbers as JSON-able data, for
            :func:`export_metrics`.
    """

    html: str
    metrics: dict[str, Any]


def build(source: ReportSource) -> RenderedReport:
    """Render one source to a page.

    Args:
        source: What to report on.

    Returns:
        The page and its metrics.
    """
    stats = excursions(source.trades, source.streams)
    quality = quality_vs_r(source.trades)
    slices = attribute_all(source.trades)
    rows = all_metric_rows(source.curve, source.trades)
    money = money_summary(source.curve, source.trades, hypothetical=not source.money_is_real)
    n_trials = source.search.total_trials if source.search else None
    family = sharpe_family(source.curve, n_trials)

    figures: dict[str, Markup] = {
        "equity": _figure_html(equity_figure(source)),
        "underwater": _figure_html(underwater_figure(source.curve)),
        "r_distribution": _figure_html(r_distribution_figure(source.trades)),
        "monthly": _figure_html(monthly_heatmap_figure(source.curve)),
        "excursions": _figure_html(excursion_figure(stats)),
        "duration_hist": _figure_html(duration_histogram_figure(source.trades)),
        "duration_vs_r": _figure_html(duration_vs_r_figure(source.trades)),
    }
    if source.kind is SourceKind.WALKFORWARD:
        figures["per_fold"] = _figure_html(per_fold_figure(source))
        figures["is_oos"] = _figure_html(is_oos_figure(source))

    template = _environment().get_template(TEMPLATE_NAME)
    html = template.render(
        plotly_js=Markup(get_plotlyjs()),
        source=source,
        kind=source.kind.value,
        figures=figures,
        metric_rows=rows,
        slices=slices,
        quality=quality,
        excursion_stats=stats,
        verdict=source.verdict,
        reliable_trades=RELIABLE_TRADES,
        thin_sample=len(source.trades) < RELIABLE_TRADES,
        comparison=None,
        money=money,
        sharpe_family=family,
        search=source.search,
        cost_sensitivity=source.cost_sensitivity,
    )
    return RenderedReport(
        html=html, metrics=metrics_payload(source, rows, quality, stats, money, family)
    )


def build_comparison(sources: Sequence[ReportSource]) -> RenderedReport:
    """Render several sources side by side.

    Args:
        sources: What to compare. Curves are normalised to their own starting
            points, because they are different accounts and different periods
            and only their shapes are comparable.

    Returns:
        The page and every source's metrics, keyed by label.
    """
    per_source = {source.label: all_metric_rows(source.curve, source.trades) for source in sources}
    template = _environment().get_template(TEMPLATE_NAME)
    html = template.render(
        plotly_js=Markup(get_plotlyjs()),
        source=None,
        kind="comparison",
        figures={"comparison": _figure_html(comparison_figure(sources))},
        metric_rows=[],
        slices={},
        quality=None,
        excursion_stats=None,
        verdict=None,
        reliable_trades=RELIABLE_TRADES,
        thin_sample=False,
        comparison={
            "sources": list(sources),
            "rows": _comparison_rows(sources, per_source),
            "verdicts": [(source.label, source.verdict) for source in sources],
        },
    )
    return RenderedReport(
        html=html,
        metrics={
            source.label: metrics_payload(source, per_source[source.label], None, None, None, None)
            for source in sources
        },
    )


def _comparison_rows(
    sources: Sequence[ReportSource], per_source: Mapping[str, Sequence[MetricRow]]
) -> list[dict[str, Any]]:
    """Metrics aligned across sources: one row per metric, one cell per source."""
    order: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for source in sources:
        for row in per_source[source.label]:
            key = (row.group, row.name)
            if key not in seen:
                seen.add(key)
                order.append(key)
    aligned: list[dict[str, Any]] = []
    for group, name in order:
        cells = []
        for source in sources:
            match = next(
                (
                    row
                    for row in per_source[source.label]
                    if row.group == group and row.name == name
                ),
                None,
            )
            cells.append(
                {
                    "value": match.value if match else "—",
                    "unreliable": bool(match and match.unreliable),
                }
            )
        aligned.append({"group": group, "name": name, "cells": cells})
    return aligned


def metrics_payload(
    source: ReportSource,
    rows: Sequence[MetricRow],
    quality: QualityAnalysis | None,
    stats: ExcursionStats | None,
    money: MoneySummary | None = None,
    family: SharpeFamily | None = None,
) -> dict[str, Any]:
    """The page's numbers as JSON-able data."""
    payload: dict[str, Any] = {
        "label": source.label,
        "kind": source.kind.value,
        "n_trades": len(source.trades),
        "money_is_real": source.money_is_real,
        "metrics": [
            {
                "group": row.group,
                "name": row.name,
                "value": row.value,
                "sample": row.sample,
                "unreliable": row.unreliable,
            }
            for row in rows
        ],
    }
    if money is not None:
        payload["money"] = {
            "starting_equity": money.starting_equity,
            "ending_equity": money.ending_equity,
            "pnl_money": money.pnl_money,
            "pnl_pct": money.pnl_pct,
            "max_drawdown_money": money.max_drawdown_money,
            "max_drawdown_pct": money.max_drawdown_pct,
            "trade_count": money.trade_count,
            "hypothetical": money.hypothetical,
        }
    if family is not None:
        payload["sharpe_family"] = {
            "sharpe": family.sharpe,
            "psr": family.psr.value if family.psr else None,
            "dsr": family.dsr.value if family.dsr else None,
            "dsr_reason": family.dsr_reason,
            "n_trials": family.n_trials,
        }
    if source.search is not None:
        payload["search"] = {
            "method": source.search.method,
            "objective": source.search.objective,
            "total_trials": source.search.total_trials,
            "total_scored": source.search.total_scored,
            "truncated": source.search.truncated,
        }
    if quality is not None:
        payload["quality"] = {
            "mode": str(quality.mode),
            "n_levels": quality.n_levels,
            "correlation": quality.correlation,
            "note": quality.note,
            "groups": [
                {
                    "label": group.label,
                    "count": group.count,
                    "expectancy_r": group.expectancy_r,
                    "win_rate": group.win_rate,
                }
                for group in quality.groups
            ],
        }
    if stats is not None:
        payload["excursions"] = {
            "count": stats.count,
            "skipped": stats.skipped,
            "mean_mae_r": stats.mean_mae_r,
            "mean_mfe_r": stats.mean_mfe_r,
            "capture": stats.capture,
            "median_capture": stats.median_capture,
            "losers_with_favourable_run": stats.losers_with_favourable_run,
        }
    if source.verdict is not None:
        payload["verdict"] = source.verdict.to_dict()
    return payload


def export_metrics(rendered: RenderedReport, path: Path) -> Path:
    """Write a rendered report's metrics to JSON.

    Args:
        rendered: What :func:`build` or :func:`build_comparison` returned.
        path: Destination file.

    Returns:
        The path written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rendered.metrics, indent=2, default=str))
    return path


def write(rendered: RenderedReport, path: Path) -> Path:
    """Write a rendered page to disk.

    Args:
        rendered: What :func:`build` returned.
        path: Destination ``.html`` file.

    Returns:
        The path written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered.html)
    return path


__all__ = [
    "METRIC_TOOLTIPS",
    "CostSensitivityPoint",
    "CostSensitivityReport",
    "FoldSegment",
    "FoldSelection",
    "MetricRow",
    "MoneySummary",
    "PooledAndPerFold",
    "RenderedReport",
    "ReportSource",
    "SearchSummary",
    "SharpeFamily",
    "SourceKind",
    "all_metric_rows",
    "build",
    "build_comparison",
    "cost_sensitivity_from_disk",
    "export_metrics",
    "fold_selections_from_disk",
    "metric_rows",
    "money_summary",
    "pooled_and_per_fold",
    "search_summary_from_disk",
    "sharpe_family",
    "source_from_run",
    "source_from_walkforward",
    "write",
]
