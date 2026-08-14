"""One page over every stored run, linking to the per-run pages.

``ts report run`` answers "what did this one run do". After twenty strategies on
two symbols there are forty answers and no question that reaches them: opening
forty pages to find which ones are worth reading is not analysis, it is filing.
This is the index — a sortable table of what is on disk, and a link into the
detail page for anything worth opening.

Three things it deliberately does not do.

**It does not curate.** The rows come from walking ``runs/``, not from a list
someone maintains, so the page cannot disagree with what is stored. A run that
exists appears; a run that was deleted stops appearing without anyone editing
anything.

**It does not hide a broken run.** A directory whose tables no longer match the
result schema, or whose digest disagrees with its tables, is listed with the
reason instead of being skipped. Silently omitting it would make "not in the
index" mean both "never ran" and "cannot be read", which are opposite problems —
the same reasoning that puts ORPHANED rows in the library report rather than
dropping them.

**It does not link to a page that is not there.** A link into a detail page is
written only when that file exists next to the index. A dead link costs a reader
a click and a moment of doubt about whether the run is broken; no link costs
neither.
"""

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from trading_system.analytics.metrics import daily_curve, drawdown_stats, sharpe_daily, trade_stats
from trading_system.backtest.reproducibility import MANIFEST_FILE, read_run
from trading_system.core.exceptions import TradingSystemError

#: Where the index template lives, alongside the other report templates.
TEMPLATE_DIR = Path(__file__).resolve().parents[3] / "docs"
TEMPLATE_NAME = "run_index_template.html"

#: Subdirectories of ``runs/`` that hold something other than a flat run.
NON_RUN_DIRECTORIES = frozenset({"walkforward", "calibration", "optimization"})


@dataclass(frozen=True)
class RunRow:
    """One stored run, summarised for the table.

    Attributes:
        run_id: The run's directory name.
        strategy_id: Strategy that produced it. Several would mean a portfolio
            run; the id column joins them so the row still names what ran.
        stream: ``"SYMBOL@TF"``, as the manifest keys its data digests.
        trades: Closed trades.
        expectancy_r: Mean R per trade.
        profit_factor: Gross win over gross loss.
        win_rate: Fraction of trades closed positive.
        sharpe: Annualised Sharpe of the daily curve, or ``None`` when the curve
            is too short to have one.
        max_drawdown_pct: Deepest peak-to-trough fall, negative.
        pnl_pct: Total return over the run.
        detail_href: Relative link to the run's own page, or ``None`` when that
            page has not been rendered next to this index.
        error: Why the run could not be read, when it could not be. A row
            carries metrics or an error, never both.
    """

    run_id: str
    strategy_id: str
    stream: str
    trades: int
    expectancy_r: float | None
    profit_factor: float | None
    win_rate: float | None
    sharpe: float | None
    max_drawdown_pct: float | None
    pnl_pct: float | None
    detail_href: str | None
    error: str | None = None

    @property
    def readable(self) -> bool:
        """Whether this run could be read back at all."""
        return self.error is None


@dataclass(frozen=True)
class RunIndex:
    """Every stored run, worst-readable-first behind the best.

    Attributes:
        rows: Readable runs, ordered by ``expectancy_r`` descending.
        broken: Runs that could not be read, with the reason.
        generated: When the page was built, UTC.
        root: The runs directory that was walked.
    """

    rows: tuple[RunRow, ...]
    broken: tuple[RunRow, ...]
    generated: datetime
    root: Path

    @property
    def total(self) -> int:
        """How many run directories were found, readable or not."""
        return len(self.rows) + len(self.broken)


def fold_run_ids(root: Path) -> frozenset[str]:
    """Run ids that belong to a walk-forward rather than standing on their own.

    Folds are stored as ordinary runs in ``runs/`` by design — nothing is
    duplicated on disk, and a walk-forward manifest references them by id. The
    consequence only shows up here: after four walk-forwards of twenty folds,
    the runs directory is 90% folds, and no filter on strategy or stream can
    separate them because a fold shares both with the flat run beside it. They
    are excluded by default and the reason is this, not tidiness: a fold is one
    slice of a procedure, and reading its expectancy as a result would be
    reading a single fold as if it were the walk-forward.

    Args:
        root: The runs directory.

    Returns:
        Every run id referenced by a walk-forward manifest.
    """
    ids: set[str] = set()
    walkforward = root / "walkforward"
    if not walkforward.is_dir():
        return frozenset()
    for manifest_path in walkforward.glob("*/manifest.json"):
        try:
            payload = json.loads(manifest_path.read_text())
        except (OSError, ValueError):
            continue
        for fold in payload.get("folds", []):
            if not isinstance(fold, dict):
                continue
            for key in ("is_run", "oos_run"):
                run = fold.get(key)
                run_id = run.get("run_id") if isinstance(run, dict) else None
                if isinstance(run_id, str):
                    ids.add(run_id)
    return frozenset(ids)


def run_directories(root: Path, *, exclude: frozenset[str] = frozenset()) -> list[Path]:
    """Every flat-run directory under ``root``.

    Walk-forwards, calibrations and optimisation outputs live under ``runs/``
    too but are not runs: they hold manifests of a different shape, and reading
    them as runs would fail for a reason that has nothing to do with the run
    being broken.

    Args:
        root: The runs directory.
        exclude: Run ids to leave out — normally :func:`fold_run_ids`.

    Returns:
        Directories holding a run manifest, sorted by name.
    """
    if not root.is_dir():
        return []
    return sorted(
        directory
        for directory in root.iterdir()
        if directory.is_dir()
        and directory.name not in NON_RUN_DIRECTORIES
        and directory.name not in exclude
        and (directory / MANIFEST_FILE).exists()
    )


def _summarise(directory: Path, *, detail_dir: Path | None) -> RunRow:
    """Read one run and reduce it to a table row.

    Args:
        directory: The run's directory.
        detail_dir: Where per-run pages live, for the link. ``None`` links
            nothing.

    Returns:
        A row carrying metrics, or one carrying the reason it has none.
    """
    run_id = directory.name
    href = None
    if detail_dir is not None and (detail_dir / f"{run_id}.html").exists():
        href = f"{run_id}.html"

    try:
        stored = read_run(directory)
    except (TradingSystemError, OSError, ValueError, KeyError) as error:
        return RunRow(
            run_id=run_id,
            strategy_id="?",
            stream="?",
            trades=0,
            expectancy_r=None,
            profit_factor=None,
            win_rate=None,
            sharpe=None,
            max_drawdown_pct=None,
            pnl_pct=None,
            detail_href=href,
            error=str(error).splitlines()[0][:200],
        )

    manifest, result = stored.manifest, stored.result
    trades = list(result.trades)
    stats = trade_stats(trades) if trades else None
    daily = daily_curve(result.curve) if result.curve else None

    sharpe: float | None = None
    drawdown: float | None = None
    pnl: float | None = None
    if daily is not None and len(daily.equity) >= 2:
        try:
            sharpe = sharpe_daily(daily).value
        except ValueError:
            sharpe = None
        try:
            drawdown = drawdown_stats(daily).max_drawdown_pct
        except ValueError:
            drawdown = None
        start, end = daily.equity[0], daily.equity[-1]
        pnl = float((end - start) / start) if start else None

    return RunRow(
        run_id=run_id,
        strategy_id=", ".join(sorted(manifest.strategies)) or "?",
        stream=", ".join(sorted(manifest.data)) or "?",
        trades=len(trades),
        expectancy_r=stats.expectancy_r if stats else None,
        profit_factor=stats.profit_factor if stats else None,
        win_rate=stats.winrate if stats else None,
        sharpe=sharpe,
        max_drawdown_pct=drawdown,
        pnl_pct=pnl,
        detail_href=href,
    )


def build_index(
    directories: Iterable[Path], *, root: Path, detail_dir: Path | None = None
) -> RunIndex:
    """Summarise every run into one index.

    Args:
        directories: Run directories to include — see :func:`run_directories`.
        root: The runs directory they came from, for the header.
        detail_dir: Where per-run pages live, for the links.

    Returns:
        The index, readable rows sorted by expectancy descending so the table
        opens on what is worth reading.
    """
    summarised = [_summarise(directory, detail_dir=detail_dir) for directory in directories]
    readable = [row for row in summarised if row.readable]
    broken = [row for row in summarised if not row.readable]
    readable.sort(key=lambda row: (row.expectancy_r is None, -(row.expectancy_r or 0.0)))
    return RunIndex(
        rows=tuple(readable),
        broken=tuple(broken),
        generated=datetime.now(UTC),
        root=root,
    )


def _environment() -> Environment:
    """Jinja environment for the index template."""
    return Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=select_autoescape(["html"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )


def render(index: RunIndex) -> str:
    """Render the index to a self-contained HTML page.

    Args:
        index: What to render.

    Returns:
        The page. No plots and no external assets, so it stays a few kilobytes
        however many runs it lists — the heavy pages are the ones it links to.
    """
    return _environment().get_template(TEMPLATE_NAME).render(index=index)


def write(index: RunIndex, path: Path) -> Path:
    """Render and write the index.

    Args:
        index: What to render.
        path: Where to write it. Parent directories are created.

    Returns:
        ``path``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(index), encoding="utf-8")
    return path


def filter_rows(
    directories: Sequence[Path], *, strategy: str | None, stream: str | None
) -> list[Path]:
    """Keep only run directories matching a strategy or stream substring.

    Filtering on the manifest rather than the directory name because a run id is
    a digest and carries nothing a human would filter on.

    Args:
        directories: Candidates.
        strategy: Substring of a strategy id, or ``None`` for all.
        stream: Substring of a ``"SYMBOL@TF"`` key, or ``None`` for all.

    Returns:
        The matching directories, order preserved.
    """
    if strategy is None and stream is None:
        return list(directories)

    kept: list[Path] = []
    for directory in directories:
        try:
            manifest = json.loads((directory / MANIFEST_FILE).read_text())
        except (OSError, ValueError):
            continue
        strategies = " ".join(manifest.get("strategies", {}))
        streams = " ".join(manifest.get("data", {}))
        if strategy is not None and strategy not in strategies:
            continue
        if stream is not None and stream not in streams:
            continue
        kept.append(directory)
    return kept
