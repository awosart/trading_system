"""Summarising a screen into one page that cannot be read as a result.

The page answers one question — *which of these are worth the cost of a
walk-forward on the holdout* — and is built so that it cannot be mistaken for
answering another. Four things enforce that, and each of them is there because
the alternative is a page somebody quotes:

* every figure is in-sample, over one window, with no folds and no null, and the
  page says so in its own heading rather than in a footnote;
* the size of the search sits beside every ranking, together with the effective
  number of independent trials, because "best of sixteen thousand" and "best of
  eight" are different claims about the same number;
* the cross-sectional z-score is published with the sentence naming what it does
  not remove — a bias shared by every strategy on that instrument survives it
  intact, so a top-decile z is not evidence of an edge;
* the holdout boundary is printed, so a reader can see which bars are still
  clean and that nothing here touched them.
"""

import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from trading_system.validation.screen import ScreenRow
from trading_system.validation.trials import EffectiveTrials

#: Where the page's template lives.
TEMPLATE_DIR = Path(__file__).resolve().parents[3] / "docs"

#: The template file.
TEMPLATE_NAME = "screen_report_template.html"

#: Trades below which a row's ratios are noise rather than an estimate. Not a
#: verdict threshold — the verdict's own gate is 100 trades over a whole
#: walk-forward — but the point below which a mean of R multiples has no
#: standard error worth printing.
MIN_TRADES_FOR_RANKING = 30

#: How many rows each ranking table shows.
TOP_N = 30


@dataclass(frozen=True)
class GroupSummary:
    """One slice of the screen, summarised.

    Attributes:
        name: What the slice is.
        runs: Rows in it.
        traded: Rows that produced at least one trade.
        rankable: Rows with enough trades for their ratios to mean anything.
        positive: Rankable rows with positive expectancy.
        median_expectancy_r: Median expectancy over rankable rows.
        median_trades: Median trade count over rows that traded.
        median_z: Median cross-sectional z over rows that have one.
    """

    name: str
    runs: int
    traded: int
    rankable: int
    positive: int
    median_expectancy_r: float | None
    median_trades: float | None
    median_z: float | None

    @property
    def positive_share(self) -> float | None:
        """Share of rankable rows above zero, or ``None`` when none rank."""
        if self.rankable == 0:
            return None
        return self.positive / self.rankable


@dataclass(frozen=True)
class ScreenReport:
    """Everything the page needs, computed once.

    Attributes:
        rows: Every run, in the order the log holds them.
        meta: Fidelity, family and type by spec id, from the corpus manifest.
        z_scores: Cross-sectional z by task key.
        trials: The effective-trials estimate.
        bar_budget: Bars each run was given.
        holdout_fraction: Share of every series withheld.
        holdout_note: One line naming where the earliest holdout starts.
        by_symbol: Summaries by instrument.
        by_timeframe: Summaries by bar size.
        by_family: Summaries by what the entry bets on.
        by_fidelity: Summaries by how much of the spec came from its page.
        top: Best rankable rows by cross-sectional z.
        bottom: Worst rankable rows by z.
        busiest: Rows with the most trades, whatever their result.
        failures: Rows whose run raised.
        blocked: Most common dominant refusal, with counts.
        skipped: Specs that produced no task at all, by reason.
    """

    rows: tuple[ScreenRow, ...]
    meta: Mapping[str, Mapping[str, str]]
    z_scores: Mapping[str, float]
    trials: EffectiveTrials | None
    bar_budget: int
    holdout_fraction: float
    holdout_note: str
    by_symbol: tuple[GroupSummary, ...]
    by_timeframe: tuple[GroupSummary, ...]
    by_family: tuple[GroupSummary, ...]
    by_fidelity: tuple[GroupSummary, ...]
    top: tuple[ScreenRow, ...]
    bottom: tuple[ScreenRow, ...]
    busiest: tuple[ScreenRow, ...]
    failures: tuple[ScreenRow, ...]
    blocked: tuple[tuple[str, int], ...]
    skipped: Mapping[str, str]

    @property
    def total(self) -> int:
        """Runs attempted."""
        return len(self.rows)

    @property
    def completed(self) -> tuple[ScreenRow, ...]:
        """Runs that finished without raising."""
        return tuple(row for row in self.rows if row.error is None)

    @property
    def traded(self) -> tuple[ScreenRow, ...]:
        """Runs that produced at least one trade."""
        return tuple(row for row in self.completed if row.trades > 0)

    @property
    def thin(self) -> tuple[ScreenRow, ...]:
        """Runs that traded but too little to rank."""
        return tuple(row for row in self.traded if row.trades < MIN_TRADES_FOR_RANKING)

    @property
    def rankable(self) -> tuple[ScreenRow, ...]:
        """Runs with enough trades for their ratios to be worth reading."""
        return tuple(
            row
            for row in self.traded
            if row.trades >= MIN_TRADES_FOR_RANKING and row.expectancy_r is not None
        )

    @property
    def positive(self) -> tuple[ScreenRow, ...]:
        """Rankable runs above zero expectancy."""
        return tuple(row for row in self.rankable if (row.expectancy_r or 0.0) > 0)

    @property
    def total_seconds(self) -> float:
        """Wall time summed over runs — not elapsed, which parallelism cuts."""
        return sum(row.seconds for row in self.rows)

    def z_of(self, row: ScreenRow) -> float | None:
        """This row's cross-sectional z, if it has one."""
        return self.z_scores.get(row.key)


def _summarise(name: str, rows: Sequence[ScreenRow], z: Mapping[str, float]) -> GroupSummary:
    """Reduce one slice to its counts and medians."""
    completed = [row for row in rows if row.error is None]
    traded = [row for row in completed if row.trades > 0]
    rankable = [
        row
        for row in traded
        if row.trades >= MIN_TRADES_FOR_RANKING and row.expectancy_r is not None
    ]
    expectancies = [row.expectancy_r for row in rankable if row.expectancy_r is not None]
    scores = [z[row.key] for row in rankable if row.key in z]
    return GroupSummary(
        name=name,
        runs=len(rows),
        traded=len(traded),
        rankable=len(rankable),
        positive=sum(1 for value in expectancies if value > 0),
        median_expectancy_r=statistics.median(expectancies) if expectancies else None,
        median_trades=statistics.median([row.trades for row in traded]) if traded else None,
        median_z=statistics.median(scores) if scores else None,
    )


def _grouped(
    rows: Sequence[ScreenRow], label: Mapping[str, str], z: Mapping[str, float], fallback: str = "?"
) -> tuple[GroupSummary, ...]:
    """Summarise rows grouped by a per-spec label, largest group first."""
    buckets: dict[str, list[ScreenRow]] = {}
    for row in rows:
        buckets.setdefault(label.get(row.spec_id, fallback), []).append(row)
    summaries = [_summarise(name, group, z) for name, group in buckets.items()]
    return tuple(sorted(summaries, key=lambda summary: (-summary.runs, summary.name)))


def build_report(
    rows: Sequence[ScreenRow],
    meta: Mapping[str, Mapping[str, str]],
    z_scores: Mapping[str, float],
    *,
    trials: EffectiveTrials | None,
    bar_budget: int,
    holdout_fraction: float,
    holdout_note: str,
    skipped: Mapping[str, str] | None = None,
) -> ScreenReport:
    """Compute every table the page shows.

    Args:
        rows: What the screen produced.
        meta: Per-spec labels from the corpus manifest, possibly empty.
        z_scores: Cross-sectional z by task key.
        trials: The effective-trials estimate, or ``None`` when not computed.
        bar_budget: Bars each run was given.
        holdout_fraction: Share of every series withheld.
        holdout_note: One line naming where the holdout starts.
        skipped: Specs that produced no task, by reason.

    Returns:
        The report.
    """
    ordered = tuple(rows)
    families = {spec: labels.get("family", "?") for spec, labels in meta.items()}
    fidelities = {spec: labels.get("fidelity", "?") for spec, labels in meta.items()}

    rankable = [
        row
        for row in ordered
        if row.error is None
        and row.trades >= MIN_TRADES_FOR_RANKING
        and row.expectancy_r is not None
        and row.key in z_scores
    ]
    ranked = sorted(rankable, key=lambda row: -z_scores[row.key])
    busiest = sorted((row for row in ordered if row.error is None), key=lambda row: -row.trades)
    blocked: dict[str, int] = {}
    for row in ordered:
        if row.error is None and row.dominant_reason:
            blocked[row.dominant_reason] = blocked.get(row.dominant_reason, 0) + 1

    return ScreenReport(
        rows=ordered,
        meta=meta,
        z_scores=z_scores,
        trials=trials,
        bar_budget=bar_budget,
        holdout_fraction=holdout_fraction,
        holdout_note=holdout_note,
        by_symbol=_grouped(ordered, {row.spec_id: row.symbol for row in ordered}, z_scores),
        by_timeframe=_grouped(ordered, {row.spec_id: row.timeframe for row in ordered}, z_scores),
        by_family=_grouped(ordered, families, z_scores),
        by_fidelity=_grouped(ordered, fidelities, z_scores),
        top=tuple(ranked[:TOP_N]),
        bottom=tuple(reversed(ranked[-TOP_N:])) if ranked else (),
        busiest=tuple(busiest[:TOP_N]),
        failures=tuple(row for row in ordered if row.error is not None),
        blocked=tuple(sorted(blocked.items(), key=lambda pair: -pair[1])[:12]),
        skipped=dict(skipped or {}),
    )


@dataclass(frozen=True)
class RenderedScreenReport:
    """A rendered page and the report behind it.

    Attributes:
        html: The self-contained page.
        report: What it was built from.
    """

    html: str
    report: ScreenReport


def render(report: ScreenReport) -> RenderedScreenReport:
    """Render the screen to a self-contained HTML page."""
    environment = Environment(
        loader=FileSystemLoader(TEMPLATE_DIR),
        autoescape=select_autoescape(["html"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = environment.get_template(TEMPLATE_NAME)
    html = template.render(
        report=report,
        generated_at=datetime.now(UTC),
        min_trades=MIN_TRADES_FOR_RANKING,
        top_n=TOP_N,
    )
    return RenderedScreenReport(html=html, report=report)


def write(rendered: RenderedScreenReport, path: Path) -> Path:
    """Write a rendered page to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered.html, encoding="utf-8")
    return path
