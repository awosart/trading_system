"""What is in the strategy library, and what each entry is worth.

**Not** :func:`~trading_system.analytics.report.build_comparison`. That function
compares a handful of run ids a caller already chose; this module answers a
different question — "what does the whole library hold, and what is missing" —
by walking every entry :class:`~trading_system.strategies.repository.StrategyRepository`
knows about, whether or not anyone has ever run it.

**Three states a library entry can be in, and none of them is hidden.** A
strategy with zero recorded results is not an error — it is a fact about the
library, shown as :data:`NOT_MEASURED` rather than folded into a table row with
blank cells. A result whose run id can no longer be read back from ``runs/`` is
:data:`ORPHANED` — this module attempts the full reconstruction (every fold's
stored curve, not just the walk-forward's own top-level manifest) before
calling something resolved, because a manifest that parses is not the same
claim as a report that can be built from it. A verdict distribution with zero
``ROBUST`` entries says so in words — ``"nothing to approve"`` — rather than
rendering an empty table a reader has to interpret.

**The correlation matrix answers a design question, not a portfolio question.**
See the module's own section below for the reasoning; the short version is that
it measures shared market-risk exposure between strategies, which is defined and
informative even when none of them has a confirmed edge — a point worth stating
because "these results have no edge" is exactly the moment a reader is tempted
to wave the whole diagnostic away as noise.

**The selected-strategies section is a placeholder, not a result.** It renders
and says plainly that there is nothing to select while zero strategies carry
``ROBUST``. It is not filled with the least-bad of the rejected — CLAUDE.md's
own anti-pattern list is explicit that a `nothing approved` state must read as
that, not be quietly upgraded into a ranked shortlist.

---

**Correlation of equity curves with no confirmed edge is not noise-correlation,
and the reasoning is arithmetic, not intuition.** A strategy's daily return is
zero on every day it holds no position — most days, for these four. Covariance
between two such series is ``E[r_A r_B] - E[r_A] E[r_B]``; the product
``r_A(t) r_B(t)`` is exactly zero whenever *either* series is flat that day, so
the numerator is driven entirely by days both strategies were actually in the
market. With expectancy near zero, ``E[r_A]`` and ``E[r_B]`` are themselves near
zero, so the covariance is close to the raw sum over the mutually-active days
alone. The denominator (each series' own variance) is computed over its *full*
calendar, active or not. The ratio therefore behaves like
``ρ_active-days × (n_both_active / sqrt(n_A_active · n_B_active))`` — the
correlation computed over the full calendar, flat days included, is
automatically discounted by how rarely the two strategies are ever both
exposed. That is exactly the quantity "do twenty strategies add up to one bet"
needs: a pair that agrees strongly on the rare day they overlap but is almost
never simultaneously active gets a *low* score, correctly, because it rarely
stacks risk in practice. Restricting the computation to only the mutually-active
days would throw away that discount and answer a narrower, less useful
question — "when they do overlap, do they agree" — which is not what portfolio
heat cares about.

None of this depends on either strategy having a confirmed edge. "No edge" is a
statement about the *mean* of a return series (expectancy indistinguishable
from zero); it says nothing about the *variance*, which is what a correlation
coefficient is actually built from, and both series are still deterministic
functions of the same underlying price action regardless of whether either one's
average outcome beats a null. Two zero-edge trend-followers on the same
instrument can be, and typically are, genuinely correlated in when and which way
they are exposed — measuring that is the whole point of this section.

What correlation over the full calendar does **not** fix is sample size: the
covariance sum is only as reliable as the number of mutually-active days behind
it, and a large calendar overlap can still rest on very few of them. Every
pairwise cell therefore carries, alongside the coefficient, both the calendar
overlap it was computed over and the count of days both strategies actually held
a position — the second number is what a reader should look at before trusting
the first. A pair is reported as :data:`None` (never as a fabricated zero) when
the calendar overlap itself falls under :data:`MIN_SHARED_DAYS`, the same
"absent, not zero" discipline :class:`~trading_system.risk.correlation.CorrelationMatrix`
already applies to instrument correlation, for the same reason: zero reads as
"measured independent", and a missing measurement must never be more permissive
than a present one.

**Not** :class:`~trading_system.risk.correlation.CorrelationProvider`. That
class exists to answer a no-lookahead question — "what could this correlation
have looked like at the instant a live decision was taken" — with an ``as_of``
argument, a rolling window and per-day caching, none of which apply here: this
module runs once, after every fold of every walk-forward has already finished,
over the whole out-of-sample history at once. Reusing the live-sizing machinery
for a retrospective diagnostic would conflate two different disciplines under
one name.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from itertools import combinations
from pathlib import Path
from statistics import StatisticsError
from statistics import correlation as _statistics_correlation

from jinja2 import Environment, FileSystemLoader, select_autoescape

from trading_system.analytics.metrics import daily_curve, simple_returns
from trading_system.analytics.report import (
    SearchSummary,
    load_walkforward_curve,
    search_summary_from_disk,
    sharpe_family,
)
from trading_system.analytics.statistical import DeflatedSharpeResult
from trading_system.backtest.portfolio import EquityPoint
from trading_system.backtest.reproducibility import read_run
from trading_system.strategies.repository import StrategyRecord, StrategyRepository
from trading_system.strategies.results_link import (
    RUN_KIND_WALKFORWARD,
    ResultRecord,
    ResultsLink,
)
from trading_system.validation.report import Verdict
from trading_system.validation.walkforward import WF_MANIFEST_FILE

#: Where the Jinja template lives — the same directory
#: :mod:`trading_system.analytics.report` reads its own template from, but a
#: different file: this report's shape (library-wide tables) has nothing in
#: common with a single run's page.
TEMPLATE_DIR = Path(__file__).resolve().parents[3] / "docs"
TEMPLATE_NAME = "library_report_template.html"

#: Fewest overlapping calendar days two strategies' out-of-sample windows must
#: share before a correlation is reported at all. Not a statement about how many
#: days both strategies were *active* — see the module docstring on why that is
#: a different, unenforced number reported alongside instead — this is the
#: coarser gate against comparing two curves that barely coexist in time at all.
#: Sixty trading days is roughly a quarter, the same order of magnitude as this
#: project's own fold spans (``oos_span`` is commonly 270 days); at the several
#: years of overlap these walk-forwards actually produce it does not bind, and
#: it exists for the day a much shorter-lived strategy joins the library.
MIN_SHARED_DAYS = 60

#: Sentence shown in place of a portfolio section while nothing has been
#: approved. Not blank, and not filled with the least-bad rejected strategy —
#: CLAUDE.md's own list of anti-patterns rules out dressing up "nothing
#: approved" as a ranked shortlist.
NOTHING_TO_APPROVE = (
    "No strategy in the library carries a ROBUST verdict. There is nothing to "
    "approve, and this section will stay empty until one does — it is not "
    "filled with the best of the rejected."
)


@dataclass(frozen=True)
class LibraryRow:
    """One strategy as the report's top table shows it.

    Attributes:
        strategy_id: The library entry's id.
        version: Current spec version.
        spec_type: Holding-period class, e.g. ``"SWING"``.
        status: Current lifecycle stage.
        n_runs: Every recorded result for this strategy, superseded ones
            included — a strategy re-measured after an engine change shows two,
            not one, because both are real, distinct measurements.
        last_run: The most recently recorded result, or ``None`` when
            :attr:`n_runs` is zero.
        last_run_resolves: Whether :attr:`last_run` could actually be read back
            from ``runs/``. ``False`` only when the newest measurement itself is
            gone; older, superseded results going missing does not set this —
            that is what the orphaned section is for.
        expectancy_r: From :attr:`last_run`'s own stored metrics.
        sharpe: From :attr:`last_run`'s own stored metrics.
        dsr: Deflated Sharpe against the search that produced
            :attr:`last_run`, recomputed from its curve — never stored on the
            result, since DSR needs a curve and a trial count neither
            ``validate walkforward`` nor ``validate optimize`` persists as a
            single number. ``None`` when :attr:`last_run_resolves` is false or
            no search was recorded for it.
        dsr_reason: Why :attr:`dsr` is ``None``, when it is.
        n_trades: From :attr:`last_run`'s own stored metrics.
        last_run_at: When :attr:`last_run` was recorded.
    """

    strategy_id: str
    version: str
    spec_type: str
    status: str
    n_runs: int
    last_run: ResultRecord | None
    last_run_resolves: bool
    expectancy_r: float | None
    sharpe: float | None
    dsr: float | None
    dsr_reason: str | None
    n_trades: float | None
    last_run_at: datetime | None


@dataclass(frozen=True)
class OrphanedResult:
    """One recorded result whose run can no longer be read back.

    Attributes:
        strategy_id: Which strategy it was evidence for.
        run_id: The run or walk-forward id, unresolvable.
        run_kind: ``"run"`` or ``"walkforward"``.
        verdict: The verdict it carried, if any — a graded, orphaned result is
            a different kind of loss than an ungraded one.
        created_at: When it was recorded.
        reason: What actually failed while reading it back, for a human tracing
            the gap manually.
    """

    strategy_id: str
    run_id: str
    run_kind: str
    verdict: str | None
    created_at: datetime
    reason: str


@dataclass(frozen=True)
class VerdictCounts:
    """How the library's measured strategies distribute across verdicts.

    Counted **per strategy**, on the last recorded result — a strategy
    measured twice contributes one verdict, its most recent, not two. Strategies
    with zero runs are not counted here at all; they are
    :data:`NOT_MEASURED`, a different kind of absence.

    Attributes:
        robust: Strategies whose last verdict is ``ROBUST``.
        overfit: Strategies whose last verdict is ``OVERFIT``.
        fragile: Strategies whose last verdict is ``FRAGILE``.
        insufficient: Strategies whose last verdict is ``INSUFFICIENT``.
        ungraded: Strategies with at least one run but no verdict on the last
            one — recorded, never graded.
    """

    robust: int
    overfit: int
    fragile: int
    insufficient: int
    ungraded: int

    @property
    def total(self) -> int:
        """Every measured strategy counted here."""
        return self.robust + self.overfit + self.fragile + self.insufficient + self.ungraded

    @property
    def nothing_to_approve(self) -> bool:
        """Whether the library holds zero ``ROBUST`` strategies."""
        return self.robust == 0


@dataclass(frozen=True)
class CorrelationCell:
    """One pair's measured co-movement, or the reason it is unmeasured.

    Attributes:
        left: First strategy id.
        right: Second strategy id.
        correlation: Pearson correlation of daily returns over the full
            calendar overlap — see the module docstring — or ``None`` when
            :attr:`n_shared_days` falls under :data:`MIN_SHARED_DAYS`, or when
            either series has zero variance over the overlap.
        n_shared_days: Trading days both strategies' out-of-sample windows
            cover, whether or not either held a position that day.
        n_both_active: Of those, days **both** held a nonzero return —
            the sample size the coefficient's numerator actually rests on. Can
            be small even when :attr:`n_shared_days` is large; a reader should
            look at this before trusting :attr:`correlation`.
    """

    left: str
    right: str
    correlation: float | None
    n_shared_days: int
    n_both_active: int


@dataclass(frozen=True)
class LibraryReport:
    """Everything ``ts strategy report`` renders.

    Attributes:
        rows: One per library entry with at least one run, id order.
        not_measured: Strategy ids with zero recorded runs, id order.
        orphaned: Every result whose run can no longer be read back.
        verdicts: Distribution of last-recorded-verdict across measured
            strategies.
        correlation: One cell per unordered pair of strategies whose *last*
            run resolves — pairs where either side's last run is orphaned or
            the strategy is unmeasured are absent, not zero.
        portfolio_note: What the selected-strategies section says. Always
            :data:`NOTHING_TO_APPROVE` until :attr:`verdicts` holds a
            ``ROBUST`` entry — this module does not compute anything past that
            point yet.
    """

    rows: tuple[LibraryRow, ...]
    not_measured: tuple[str, ...]
    orphaned: tuple[OrphanedResult, ...]
    verdicts: VerdictCounts
    correlation: tuple[CorrelationCell, ...]
    portfolio_note: str


def _resolves(
    record: ResultRecord, store_root: Path
) -> tuple[Sequence[EquityPoint] | None, str | None]:
    """Attempt the full reconstruction of one result's curve.

    Args:
        record: The result to resolve.
        store_root: Where runs live — ``runs``.

    Returns:
        ``(curve, None)`` on success, ``(None, reason)`` on any failure. A
        walk-forward is resolved by rebuilding its **stitched** curve, which
        means opening every fold's own stored run — a top-level
        ``manifest.json`` that parses but references a deleted fold directory
        is still a failure here, because a report cannot be built from it
        either.
    """
    try:
        if record.run_kind == RUN_KIND_WALKFORWARD:
            directory = store_root / "walkforward" / record.run_id
            manifest = directory / WF_MANIFEST_FILE
            if not manifest.exists():
                raise FileNotFoundError(f"no walk-forward manifest at {manifest}")
            _, stitched, _ = load_walkforward_curve(directory, store_root)
            return stitched.points, None
        directory = store_root / record.run_id
        stored = read_run(directory)
        return stored.result.curve, None
    except Exception as error:  # noqa: BLE001 - the failure itself is the report
        return None, str(error)


def _dsr_for(
    record: ResultRecord, curve: Sequence[EquityPoint], store_root: Path
) -> tuple[float | None, str | None]:
    """Deflated Sharpe for one resolved result, recomputed from its curve.

    Args:
        record: The result — only its ``run_kind``/``run_id`` are used, to find
            the search summary a walk-forward may have recorded.
        curve: The result's own curve, already resolved by :func:`_resolves`.
        store_root: Where runs live.

    Returns:
        ``(dsr, None)`` or ``(None, reason)``. A flat run (``run_kind ==
        "run"``) never carries a search, so it is always the latter, with a
        reason distinct from "search ran but produced no ``optimization.json``".
    """
    search: SearchSummary | None = None
    if record.run_kind == RUN_KIND_WALKFORWARD:
        search = search_summary_from_disk(store_root / "walkforward" / record.run_id)
    family = sharpe_family(curve, search.total_trials if search is not None else None)
    if family is None:
        return None, "no equity curve to compute a Sharpe family from"
    dsr: DeflatedSharpeResult | None = family.dsr
    return (dsr.value if dsr is not None else None), family.dsr_reason


def _last_run(records: Sequence[ResultRecord]) -> ResultRecord | None:
    """The most recently created of a strategy's results, or ``None``."""
    return max(records, key=lambda record: record.created_at) if records else None


def _verdict_bucket(verdict: str | None) -> str:
    """Which :class:`VerdictCounts` field a stored verdict string falls into."""
    if verdict is None:
        return "ungraded"
    try:
        member = Verdict(verdict)
    except ValueError:
        return "ungraded"
    return member.value.lower()


def _daily_returns_by_day(curve: Sequence[EquityPoint]) -> dict[date, float]:
    """A curve's day-over-day simple returns, keyed by the day the return lands on.

    Reuses :func:`~trading_system.analytics.metrics.daily_curve` and
    :func:`~trading_system.analytics.metrics.simple_returns` rather than
    re-bucketing by ``EquityPoint.day`` here — this module does not invent a
    second definition of "day" or "return".

    Args:
        curve: An equity curve, chronologically ordered.

    Returns:
        Trading day to that day's simple return. Empty for a curve with fewer
        than two distinct days.
    """
    if len(curve) < 2:
        return {}
    daily = daily_curve(curve)
    if len(daily.days) < 2:
        return {}
    returns = simple_returns(daily)
    return dict(zip(daily.days[1:], returns, strict=True))


def _pearson(pairs: Sequence[tuple[float, float]]) -> float | None:
    """Pearson correlation of paired observations, or ``None`` when undefined.

    A thin wrapper over :func:`statistics.correlation` — the same stdlib
    function :mod:`trading_system.analytics.attribution` already uses for
    quality-vs-outcome correlation — rather than a third hand-rolled
    implementation of the same formula in this codebase.

    Args:
        pairs: ``(x, y)`` observations.

    Returns:
        The correlation, or ``None`` when either series has zero variance —
        flat carries no information about co-movement, and reporting zero
        would misrepresent that as evidence of independence.
    """
    xs = [x for x, _ in pairs]
    ys = [y for _, y in pairs]
    try:
        return _statistics_correlation(xs, ys)
    except StatisticsError:
        return None


def correlation_matrix(
    curves: Mapping[str, Sequence[EquityPoint]],
) -> tuple[CorrelationCell, ...]:
    """Pairwise daily-return correlation between every pair of curves given.

    Args:
        curves: Strategy id to its resolved out-of-sample equity curve. Callers
            filter to resolved, measured strategies before this point — this
            function has no notion of "unmeasured" or "orphaned", only curves.

    Returns:
        One cell per unordered pair, in :func:`itertools.combinations` order
        over the sorted ids.
    """
    returns = {strategy_id: _daily_returns_by_day(curve) for strategy_id, curve in curves.items()}
    cells: list[CorrelationCell] = []
    for left, right in combinations(sorted(returns), 2):
        days_left, days_right = returns[left], returns[right]
        shared = sorted(set(days_left) & set(days_right))
        both_active = sum(1 for day in shared if days_left[day] != 0.0 and days_right[day] != 0.0)
        if len(shared) < MIN_SHARED_DAYS:
            cells.append(
                CorrelationCell(
                    left=left,
                    right=right,
                    correlation=None,
                    n_shared_days=len(shared),
                    n_both_active=both_active,
                )
            )
            continue
        value = _pearson([(days_left[day], days_right[day]) for day in shared])
        cells.append(
            CorrelationCell(
                left=left,
                right=right,
                correlation=value,
                n_shared_days=len(shared),
                n_both_active=both_active,
            )
        )
    return tuple(cells)


def build_library_report(
    repository: StrategyRepository, link: ResultsLink, store_root: Path
) -> LibraryReport:
    """Walk the whole library and every recorded result, and grade the library itself.

    Args:
        repository: The strategy library.
        link: The result log.
        store_root: Where runs live — ``runs``.

    Returns:
        The report.
    """
    records: list[StrategyRecord] = repository.records()
    all_results = link.records()

    orphaned: list[OrphanedResult] = []
    resolved_curve_by_run: dict[str, Sequence[EquityPoint] | None] = {}
    for result in all_results:
        curve, reason = _resolves(result, store_root)
        resolved_curve_by_run[result.run_id] = curve
        if curve is None:
            orphaned.append(
                OrphanedResult(
                    strategy_id=result.strategy_id,
                    run_id=result.run_id,
                    run_kind=result.run_kind,
                    verdict=result.verdict,
                    created_at=result.created_at,
                    reason=reason or "unresolvable",
                )
            )

    rows: list[LibraryRow] = []
    not_measured: list[str] = []
    counts = {"robust": 0, "overfit": 0, "fragile": 0, "insufficient": 0, "ungraded": 0}
    correlation_curves: dict[str, Sequence[EquityPoint]] = {}

    for record in records:
        for_strategy = [item for item in all_results if item.strategy_id == record.id]
        if not for_strategy:
            not_measured.append(record.id)
            continue

        last = _last_run(for_strategy)
        assert last is not None  # for_strategy is non-empty
        curve = resolved_curve_by_run.get(last.run_id)
        resolves = curve is not None

        dsr: float | None = None
        dsr_reason: str | None = None
        if curve is not None:
            dsr, dsr_reason = _dsr_for(last, curve, store_root)
            correlation_curves[record.id] = curve
        else:
            dsr_reason = "the last recorded run cannot be read back; see orphaned results"

        counts[_verdict_bucket(last.verdict)] += 1

        rows.append(
            LibraryRow(
                strategy_id=record.id,
                version=record.spec.version,
                spec_type=record.spec.type.value,
                status=record.status.value,
                n_runs=len(for_strategy),
                last_run=last,
                last_run_resolves=resolves,
                expectancy_r=last.metric("expectancy_r"),
                sharpe=last.metric("sharpe"),
                dsr=dsr,
                dsr_reason=dsr_reason,
                n_trades=last.metric("trades"),
                last_run_at=last.created_at,
            )
        )

    verdicts = VerdictCounts(
        robust=counts["robust"],
        overfit=counts["overfit"],
        fragile=counts["fragile"],
        insufficient=counts["insufficient"],
        ungraded=counts["ungraded"],
    )

    portfolio_note = (
        NOTHING_TO_APPROVE
        if verdicts.nothing_to_approve
        else (
            f"{verdicts.robust} strategy(ies) carry a ROBUST verdict, but this section does "
            "not compute a portfolio yet — see the module docstring."
        )
    )

    return LibraryReport(
        rows=tuple(sorted(rows, key=lambda row: row.strategy_id)),
        not_measured=tuple(sorted(not_measured)),
        orphaned=tuple(orphaned),
        verdicts=verdicts,
        correlation=correlation_matrix(correlation_curves),
        portfolio_note=portfolio_note,
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RenderedLibraryReport:
    """A page and the data behind it.

    Attributes:
        html: The complete document.
        report: The :class:`LibraryReport` it was built from, for a caller
            that wants the numbers without re-parsing the page.
    """

    html: str
    report: LibraryReport


def _environment() -> Environment:
    """The Jinja environment for this report's own template, autoescaping on."""
    return Environment(
        loader=FileSystemLoader(TEMPLATE_DIR),
        autoescape=select_autoescape(["html"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )


def render(report: LibraryReport) -> RenderedLibraryReport:
    """Render a library report to a self-contained HTML page.

    Args:
        report: What :func:`build_library_report` produced.

    Returns:
        The page.
    """
    left_ids = {cell.left for cell in report.correlation}
    right_ids = {cell.right for cell in report.correlation}
    ids = sorted(left_ids | right_ids)
    # A plain dict keyed both ways, built here rather than looked up inside the
    # template: Jinja's `{% set %}` does not leak out of a `{% for %}` body (a
    # well-known scoping gap, unlike Python), so a template-side search would
    # need `namespace()` gymnastics for what is a one-line dict comprehension
    # in Python.
    lookup = {(cell.left, cell.right): cell for cell in report.correlation}
    lookup.update({(cell.right, cell.left): cell for cell in report.correlation})

    template = _environment().get_template(TEMPLATE_NAME)
    html = template.render(
        report=report,
        generated_at=datetime.now(UTC),
        min_shared_days=MIN_SHARED_DAYS,
        correlation_ids=ids,
        correlation_lookup=lookup,
    )
    return RenderedLibraryReport(html=html, report=report)


def write(rendered: RenderedLibraryReport, path: Path) -> Path:
    """Write a rendered page to disk.

    Args:
        rendered: What :func:`render` returned.
        path: Destination ``.html`` file.

    Returns:
        The path written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered.html)
    return path
