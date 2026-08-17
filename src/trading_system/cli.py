"""Command-line entry point.

The root callback builds the single :class:`Settings` instance for the process
and hands it to subcommands through the typer context, keeping configuration an
explicit dependency rather than a module-level global.
"""

import json
import time
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from enum import StrEnum
from pathlib import Path

import polars as pl
import typer
from pydantic import BaseModel, ConfigDict, Field
from pydantic import ValidationError as PydanticValidationError

from trading_system.analytics.library_report import build_library_report
from trading_system.analytics.library_report import render as render_library_report
from trading_system.analytics.library_report import write as write_library_report
from trading_system.analytics.metrics import daily_curve, total_return, trade_stats
from trading_system.analytics.report import (
    build,
    build_comparison,
    cost_sensitivity_from_disk,
    export_metrics,
    fold_selections_from_disk,
    search_summary_from_disk,
    source_from_run,
    source_from_walkforward,
    write,
)
from trading_system.analytics.run_index import (
    build_index,
    filter_rows,
    fold_run_ids,
    run_directories,
)
from trading_system.analytics.run_index import write as write_index
from trading_system.analytics.screen_report import MIN_TRADES_FOR_RANKING
from trading_system.analytics.screen_report import build_report as build_screen_report
from trading_system.analytics.screen_report import render as render_screen_report
from trading_system.analytics.screen_report import write as write_screen_report
from trading_system.backtest.clock import StreamKey
from trading_system.backtest.config import BacktestConfig
from trading_system.backtest.fx import build_converter
from trading_system.backtest.orchestrator import StrategyBinding
from trading_system.backtest.reproducibility import code_version, read_run, write_run
from trading_system.backtest.spec import RunInputs
from trading_system.core.config import Settings
from trading_system.core.exceptions import TradingSystemError, ValidationError
from trading_system.core.instruments import InstrumentRegistry, load_instruments
from trading_system.core.logging import get_logger, setup_logging
from trading_system.core.types import Timeframe
from trading_system.data.models import OHLCVFrame
from trading_system.data.providers.csv_provider import CSVProvider, CSVSchema
from trading_system.data.providers.dukascopy_provider import DukascopyProvider
from trading_system.data.quality import QualityConfig, check_frame
from trading_system.data.resample import FX_DAY_ORIGIN, resample
from trading_system.data.sessions import AssetClass, TradingCalendar
from trading_system.data.store import ParquetStore
from trading_system.execution.config import CostConfig
from trading_system.exit.library import (
    DEFAULT_LIBRARY_PATH,
    ExitLibrarySpec,
    ExitPresetSpec,
    known_exit_ids,
)
from trading_system.prop.rules import (
    DEFAULT_RULES_PATH,
    day_origin_divergence,
    load_prop_rules,
)
from trading_system.prop.simulator import (
    REPORT_ITERATIONS,
    sample_from_walkforward,
    simulate,
)
from trading_system.risk.engine import RiskEngineConfig
from trading_system.risk.margin import load_prop_profiles
from trading_system.risk.sizing.methods import FixedFractional
from trading_system.strategies.ingest import load_cards, load_overrides, triage
from trading_system.strategies.ingest import render as render_ingest
from trading_system.strategies.normalize import (
    Fidelity,
    measure_coverage,
    normalise_card,
    write_corpus,
)
from trading_system.strategies.normalize.coverage import DEFAULT_MAX_COST_RATIO
from trading_system.strategies.repository import META_SUFFIX, Status, StrategyRepository
from trading_system.strategies.results_link import (
    RUN_KIND_WALKFORWARD,
    ResultsLink,
    approve_from_result,
    build_record,
)
from trading_system.strategies.schema import (
    SCHEMA_JSON_PATH,
    Regime,
    StrategySpec,
    StrategyType,
    strategy_json_schema,
)
from trading_system.strategies.validator import Severity, validate_paths
from trading_system.validation import trials as trials_module
from trading_system.validation.calibration import (
    POSITION_COUNT_DIVERGENCE_THRESHOLD,
    NullKind,
    run_calibration,
)
from trading_system.validation.holdout import DEFAULT_HOLDOUT_FRACTION, HoldoutBoundary

# Two different samplers share a name: prop.simulator's builds a prop episode,
# this one builds the trade sample the resamplers replay. Aliased so neither
# call site can pick up the other by accident.
from trading_system.validation.monte_carlo import run_all as run_monte_carlo_all
from trading_system.validation.monte_carlo import (
    sample_from_walkforward as monte_carlo_sample,
)
from trading_system.validation.nulls.random_entry import (
    EntryTraceProfile,
    build_entry_trace_profile,
    real_signals,
)
from trading_system.validation.objective import ExpectancyR, SortinoTimesSqrtTrades
from trading_system.validation.optimization import (
    GridSearch,
    OptunaSearch,
    ParameterSearch,
    RandomSearch,
    SearchSpace,
    read_fold_selection,
)
from trading_system.validation.report import (
    StrategyVerdict,
    WalkForwardReport,
    build_report,
    build_verdict,
    write_report,
)
from trading_system.validation.robustness import run_all
from trading_system.validation.screen import (
    DEFAULT_BAR_BUDGET,
    SCREEN_ROOT,
    ScreenManifest,
    ScreenRow,
    ScreenStore,
    ScreenTask,
    cross_sectional_z,
    distinct_signatures,
    load_corpus_meta,
    load_manifest,
    now_iso,
    plan_tasks,
    run_screen,
    screen_id,
)
from trading_system.validation.space_builder import (
    build_candidates,
    build_space_document,
    prune,
    render,
    verify,
    write_space,
)
from trading_system.validation.splitting import PurgedKFold, WalkForwardMode, WalkForwardSplitter
from trading_system.validation.walkforward import (
    WF_MANIFEST_FILE,
    IdentitySelector,
    OptimizingSelector,
    WalkForwardRunner,
    read_result,
)

#: Where the strategy library lives when ``--library`` is not given. A repo-root
#: relative default, because the library is version-controlled alongside the code.
DEFAULT_STRATEGY_LIBRARY = Path("strategies")

#: Where the scrape and the reviewer's answers to it live by default.
DEFAULT_CARDS_DIR = Path("strategies/scraped_strategies_v3")
DEFAULT_OVERRIDES_DIR = Path("strategies/ingest_overrides")

#: Prop-firm profile a run config assumes when it names none. Swing rather than
#: standard because every strategy in this repository holds through the weekend,
#: which is the only plan type that permits it, and because it is the strictest
#: complete profile on file — a default that errs towards refusing a trade.
DEFAULT_PROP_PROFILE = "ftmo_swing"


def _risk_config(settings: Settings, profile_name: str | None) -> RiskEngineConfig:
    """Build the engine config for a run, resolving its prop profile by name.

    Args:
        settings: Process settings, naming the profiles file.
        profile_name: Profile to trade under, or ``None`` to trade under the
            venue leverage each instrument already declares.

    Returns:
        The config. A name that resolves to nothing raises out of
        :meth:`~trading_system.risk.margin.PropProfileLibrary.get` rather than
        falling back, since a run silently trading under rules nobody chose is
        worse than one that will not start.
    """
    if profile_name is None:
        return RiskEngineConfig()
    library = load_prop_profiles(settings.prop_profiles_path)
    return RiskEngineConfig(prop_profile=library.get(profile_name))


app = typer.Typer(help="Modular trading system.")
data_app = typer.Typer(help="Market data management.")
strategy_app = typer.Typer(help="Strategy spec management.")
validate_app = typer.Typer(help="Out-of-sample validation.")
ingest_app = typer.Typer(help="Scraped strategy cards: triage and conversion.")
report_app = typer.Typer(help="HTML reports over stored runs.")
prop_app = typer.Typer(help="Prop-firm account rules: would this survive one.")
app.add_typer(data_app, name="data")
app.add_typer(strategy_app, name="strategy")
strategy_app.add_typer(ingest_app, name="ingest")
app.add_typer(validate_app, name="validate")
app.add_typer(report_app, name="report")
app.add_typer(prop_app, name="prop")

logger = get_logger(__name__)


def _as_utc(moment: datetime | None) -> datetime | None:
    """Interpret a CLI-supplied datetime as UTC."""
    if moment is None:
        return None
    return moment.replace(tzinfo=UTC) if moment.tzinfo is None else moment.astimezone(UTC)


def _store(ctx: typer.Context) -> ParquetStore:
    """Build a store rooted at the configured data directory."""
    settings: Settings = ctx.obj
    return ParquetStore(settings.data_dir)


@app.callback()
def main(ctx: typer.Context) -> None:
    """Load settings and configure logging before any subcommand runs."""
    settings = Settings()
    setup_logging(level=settings.log_level, log_file=settings.log_file)
    ctx.obj = settings


#: Escape sequences a shell will not conveniently pass through as themselves.
#: Only the ones that are plausible CSV delimiters — a mapping wide enough to
#: cover "\n" would let a file be split on something no vendor uses.
_SEPARATOR_ESCAPES = {r"\t": "\t"}


def _unescape_separator(text: str) -> str:
    r"""Resolve a delimiter written as an escape sequence.

    Args:
        text: The ``--sep`` value as typed, e.g. ``","`` or ``"\t"``.

    Returns:
        The single character to split on.

    Raises:
        typer.BadParameter: If the result is not exactly one character. polars
            takes a single byte, and a two-character delimiter would otherwise
            fail deep inside the reader rather than at the flag that set it.
    """
    resolved = _SEPARATOR_ESCAPES.get(text, text)
    if len(resolved) != 1:
        raise typer.BadParameter(f"--sep takes a single character, got {text!r}")
    return resolved


def _parse_columns(text: str | None) -> tuple[str | None, ...] | None:
    """Read ``--columns`` into the schema's positional naming.

    Args:
        text: Comma-separated canonical names in file order, ``-`` for a field
            that is present and not stored. ``None`` when the flag was omitted.

    Returns:
        One entry per field, or ``None`` to leave naming to the header.
    """
    if text is None:
        return None
    names = (name.strip().lower() for name in text.split(","))
    return tuple(None if name in {"-", ""} else name for name in names)


@data_app.command("import")
def data_import(
    ctx: typer.Context,
    symbol: str = typer.Option(..., "--symbol", help="Instrument identifier."),
    timeframe: Timeframe = typer.Option(..., "--tf", help="Bar size of the source file."),
    path: Path = typer.Option(..., "--path", help="CSV file to import."),
    source_tz: str = typer.Option("UTC", "--tz", help="IANA zone the file's timestamps use."),
    date_column: str | None = typer.Option(None, "--date-col", help="Separate date column."),
    time_column: str | None = typer.Option(None, "--time-col", help="Separate time column."),
    timestamp_format: str | None = typer.Option(None, "--format", help="chrono parse format."),
    separator: str = typer.Option(",", "--sep", help=r"Field delimiter. Write a tab as '\t'."),
    drop_unnamed_fields: bool = typer.Option(
        False,
        "--drop-unnamed-fields",
        help=(
            "Accept rows carrying more fields than the header names, discarding "
            "the surplus. Without this such a file is an error, since a row wider "
            "than its header usually means --sep is wrong."
        ),
    ),
    no_header: bool = typer.Option(
        False, "--no-header", help="The first line is data, not column names."
    ),
    columns: str | None = typer.Option(
        None,
        "--columns",
        help=(
            "Canonical name of every field, in file order, comma-separated: "
            "'timestamp,open,high,low,close,volume'. Write '-' for a field that is "
            "present and not stored. Needed when the file names no columns, or "
            "names one of them wrongly."
        ),
    ),
    no_volume: bool = typer.Option(
        False,
        "--no-volume",
        help=(
            "The file carries no volume. It is stored as 0.0, which data quality "
            "reports on every bar; a column that is not volume must never be "
            "imported as one."
        ),
    ),
) -> None:
    """Import bars from a CSV file into the local store."""
    provider = CSVProvider(
        path,
        CSVSchema(
            source_tz=source_tz,
            date_column=date_column,
            time_column=time_column,
            timestamp_format=timestamp_format,
            separator=_unescape_separator(separator),
            has_header=not no_header,
            drop_unnamed_fields=drop_unnamed_fields,
            positional_columns=_parse_columns(columns),
            absent_volume=no_volume,
        ),
    )
    frame = provider.fetch(symbol, timeframe)
    written = _store(ctx).upsert(frame)
    logger.info("data_imported", symbol=symbol, timeframe=str(timeframe), bars=len(frame))
    typer.echo(f"Imported {len(frame)} bars for {symbol} {timeframe}; partition holds {written}.")


@data_app.command("download")
def data_download(
    ctx: typer.Context,
    symbol: str = typer.Option(..., "--symbol", help="Instrument identifier, e.g. EURUSD."),
    timeframe: Timeframe = typer.Option(..., "--tf", help="Bar size to fetch."),
    start: datetime = typer.Option(..., "--start", help="Inclusive lower bound, e.g. 2024-08-05."),
    end: datetime | None = typer.Option(
        None, "--end", help="Exclusive upper bound. Defaults to now."
    ),
) -> None:
    """Download bars from Dukascopy and store them.

    Fetches bid and ask separately and averages them into mid OHLCV bars, per
    the project convention that stored bars are mid prices.
    """
    provider = DukascopyProvider()
    frame = provider.fetch(symbol, timeframe, _as_utc(start), _as_utc(end))
    if frame.is_empty:
        typer.echo(f"No data returned for {symbol} {timeframe} in the requested range.")
        raise typer.Exit(code=1)
    written = _store(ctx).upsert(frame)
    logger.info("data_downloaded", symbol=symbol, timeframe=str(timeframe), bars=len(frame))
    typer.echo(f"Downloaded {len(frame)} bars for {symbol} {timeframe}; partition holds {written}.")


@data_app.command("coverage")
def data_coverage(
    ctx: typer.Context,
    symbol: str | None = typer.Option(None, "--symbol", help="Limit to one instrument."),
    timeframe: Timeframe | None = typer.Option(None, "--tf", help="Limit to one bar size."),
) -> None:
    """Report what the local store holds."""
    store = _store(ctx)
    symbols = [symbol] if symbol else store.symbols()
    if not symbols:
        typer.echo("Store is empty.")
        return
    for name in symbols:
        timeframes = [timeframe] if timeframe else store.timeframes(name)
        for bar_size in timeframes:
            found = store.coverage(name, bar_size)
            if found is None:
                continue
            typer.echo(
                f"{name} {bar_size}: {found.bars} bars, "
                f"{found.start:%Y-%m-%d %H:%M} .. {found.end:%Y-%m-%d %H:%M} UTC"
            )


@data_app.command("quality")
def data_quality(
    ctx: typer.Context,
    symbol: str = typer.Option(..., "--symbol", help="Instrument identifier."),
    timeframe: Timeframe = typer.Option(..., "--tf", help="Bar size to inspect."),
    calendar: AssetClass | None = typer.Option(
        None,
        "--calendar",
        help=(
            "Asset class of the instrument. Excludes its regular closed periods "
            "(weekends for FX/EQUITY) from missing_bars, which otherwise reports "
            "every closed hour as a defect."
        ),
    ),
) -> None:
    """Run data-quality detectors over stored bars."""
    frame = _store(ctx).get(symbol, timeframe)
    if frame.is_empty:
        typer.echo(f"No data stored for {symbol} {timeframe}.")
        raise typer.Exit(code=1)
    config = QualityConfig(calendar=TradingCalendar(asset_class=calendar)) if calendar else None
    report = check_frame(frame, config)
    typer.echo(f"{symbol} {timeframe}: {report.bars_checked} bars checked")
    if not report.issues:
        typer.echo("  no issues found")
        return
    for issue in report.issues:
        typer.echo(f"  [{issue.severity}] {issue.code} (n={issue.count}): {issue.message}")
    if report.has_errors:
        raise typer.Exit(code=1)


@data_app.command("resample")
def data_resample(
    ctx: typer.Context,
    symbol: str = typer.Option(..., "--symbol", help="Instrument identifier."),
    source: Timeframe = typer.Option(..., "--from-tf", help="Source bar size."),
    target: Timeframe = typer.Option(..., "--to-tf", help="Target bar size."),
    fx_day: bool = typer.Option(False, "--fx-day", help="Anchor the day to 17:00 New York."),
) -> None:
    """Resample stored bars into a coarser timeframe and store the result."""
    store = _store(ctx)
    frame = store.get(symbol, source)
    if frame.is_empty:
        typer.echo(f"No data stored for {symbol} {source}.")
        raise typer.Exit(code=1)
    resampled = resample(frame, target, origin=FX_DAY_ORIGIN if fx_day else None)
    store.upsert(resampled)
    typer.echo(f"Resampled {len(frame)} {source} bars into {len(resampled)} {target} bars.")


@strategy_app.command("validate")
def strategy_validate(
    paths: list[Path] = typer.Argument(..., help="Strategy spec JSON files."),
    exit_library: Path = typer.Option(
        DEFAULT_LIBRARY_PATH,
        "--exit-library",
        help="Exit preset library exit_ref is checked against.",
    ),
) -> None:
    """Validate strategy specs: schema, feature refs, timeframe order, id uniqueness, exit_ref.

    ``exit_ref`` is checked against the preset ids declared in
    ``--exit-library``, the bundled ``exit/library.json`` by default.

    Library bookkeeping files are skipped, and the count is printed rather than
    swallowed: ``strategies/library/*/*.json`` is the natural glob to type and
    it matches ``{id}.meta.json`` too, which is not a spec and would report
    dozens of schema errors that mean nothing.
    """
    try:
        ids = known_exit_ids(exit_library)
    except ValidationError as error:
        typer.echo(f"exit library {exit_library}: {error}")
        raise typer.Exit(code=1) from error

    specs = [path for path in paths if not path.name.endswith(META_SUFFIX)]
    skipped = len(paths) - len(specs)
    if skipped:
        typer.echo(f"Skipped {skipped} bookkeeping file(s) (*{META_SUFFIX}).")
    if not specs:
        typer.echo("No strategy specs to validate.")
        return

    results = validate_paths(specs, known_exit_ids=ids)
    has_errors = False
    for path, issues in results.items():
        if not issues:
            typer.echo(f"{path}: OK")
            continue
        for issue in issues:
            typer.echo(f"{path}: [{issue.severity}] {issue.code}: {issue.message}")
            if issue.severity is Severity.ERROR:
                has_errors = True
    if has_errors:
        raise typer.Exit(code=1)


@strategy_app.command("add")
def strategy_add(
    path: Path = typer.Argument(..., help="Strategy spec JSON file to import."),
    name: str = typer.Option(..., "--name", help="Human-readable name."),
    author: str = typer.Option(..., "--author", help="Who owns the entry."),
    library: Path = typer.Option(DEFAULT_STRATEGY_LIBRARY, "--library", help="Repository root."),
    source: str | None = typer.Option(None, "--source", help="Where the idea came from."),
    tag: list[str] = typer.Option([], "--tag", help="Label to file it under; repeatable."),
    notes: str | None = typer.Option(None, "--notes", help="Prose about the entry."),
) -> None:
    """Import a strategy spec into the library.

    The spec is copied, not moved: the file it came from keeps whatever role it
    had. A new entry always lands at ``DRAFT`` — every other stage needs
    evidence this command has not been given.
    """
    repository = StrategyRepository(library)
    try:
        spec = StrategySpec.model_validate_json(path.read_text(encoding="utf-8"))
        record = repository.add(
            spec, name=name, author=author, source=source, tags=tuple(tag), notes=notes
        )
    except (ValidationError, PydanticValidationError) as error:
        typer.echo(f"{path}: {error}")
        raise typer.Exit(code=1) from error
    typer.echo(f"Added {record.id} v{record.spec.version} [{record.status.value}] -> {record.path}")


@strategy_app.command("list")
def strategy_list(
    library: Path = typer.Option(DEFAULT_STRATEGY_LIBRARY, "--library", help="Repository root."),
    results: Path | None = typer.Option(
        None, "--results", help="Result log, for --min-sharpe. Defaults to the library root."
    ),
    spec_type: StrategyType | None = typer.Option(None, "--type", help="Holding-period class."),
    status: Status | None = typer.Option(None, "--status", help="Lifecycle stage."),
    regime: Regime | None = typer.Option(None, "--regime", help="Permitted market regime."),
    instrument: str | None = typer.Option(None, "--instrument", help="Symbol it may trade."),
    tag: str | None = typer.Option(None, "--tag", help="Label."),
    author: str | None = typer.Option(None, "--author", help="Owner."),
    min_sharpe: float | None = typer.Option(
        None, "--min-sharpe", help="Keep strategies with a recorded run at or above this Sharpe."
    ),
    metric: str = typer.Option("sharpe", "--metric", help="Metric --min-sharpe applies to."),
    walkforward_only: bool = typer.Option(
        False, "--oos", help="Judge --min-sharpe on walk-forward runs only."
    ),
) -> None:
    """List library entries matching every filter given.

    ``--min-sharpe`` is answered from the result log rather than the library:
    the strategy files hold no measurements, and a filter that silently needed
    a result store would answer differently depending on what had been run.
    """
    repository = StrategyRepository(library)
    records = repository.list(
        type=spec_type,
        status=status,
        regime=regime,
        instrument=instrument,
        tag=tag,
        author=author,
    )
    if min_sharpe is not None:
        link = ResultsLink(results if results is not None else library)
        kind = RUN_KIND_WALKFORWARD if walkforward_only else None
        qualified = {
            item.strategy_id for item in link.find(metric=metric, minimum=min_sharpe, run_kind=kind)
        }
        records = [record for record in records if record.id in qualified]

    if not records:
        typer.echo("No strategies match.")
        return
    for record in records:
        flags = "" if record.meta_present else "  (no meta file)"
        tags = ",".join(record.meta.tags) or "-"
        typer.echo(
            f"{record.id:<24} v{record.spec.version:<8} {record.spec.type.value:<9} "
            f"{record.status.value:<9} tags={tags:<28} {record.meta.name}{flags}"
        )


@strategy_app.command("show")
def strategy_show(
    strategy_id: str = typer.Argument(..., help="Strategy id."),
    library: Path = typer.Option(DEFAULT_STRATEGY_LIBRARY, "--library", help="Repository root."),
    results: Path | None = typer.Option(None, "--results", help="Result log. Defaults to library."),
    version: str | None = typer.Option(None, "--version", help="Read an archived version."),
) -> None:
    """Show one entry: spec identity, bookkeeping, lifecycle log and its runs."""
    repository = StrategyRepository(library)
    try:
        record = repository.get(strategy_id, version)
    except KeyError as error:
        typer.echo(str(error))
        raise typer.Exit(code=1) from error

    typer.echo(f"{record.id}  v{record.spec.version}  {record.spec.type.value}")
    typer.echo(f"  name       {record.meta.name}")
    typer.echo(f"  author     {record.meta.author}")
    if record.meta.source:
        typer.echo(f"  source     {record.meta.source}")
    typer.echo(f"  status     {record.status.value}")
    typer.echo(f"  spec       {record.digest}")
    typer.echo(f"  exit_ref   {record.spec.exit_ref}")
    typer.echo(f"  tags       {','.join(record.meta.tags) or '-'}")
    if not record.meta_present:
        typer.echo("  WARNING    no meta file; bookkeeping shown is a default, not a record")
    if record.meta.notes:
        typer.echo(f"  notes      {record.meta.notes}")

    typer.echo("  versions:")
    for entry in record.meta.versions:
        note = f"  {entry.note}" if entry.note else ""
        typer.echo(f"    {entry.version:<8} {entry.at:%Y-%m-%d} {entry.spec_digest[:12]}{note}")

    typer.echo("  lifecycle:")
    if not record.meta.lifecycle:
        typer.echo("    (none recorded — reads as DRAFT)")
    for event in record.meta.lifecycle:
        detail = event.reason or ""
        if event.run_id:
            detail = f"run={event.run_id} selector={event.selector_key} verdict={event.verdict}"
        typer.echo(f"    {event.status.value:<9} {event.at:%Y-%m-%d} {detail}")

    link = ResultsLink(results if results is not None else library)
    runs = link.for_strategy(strategy_id)
    typer.echo(f"  runs ({len(runs)}):")
    for run in runs:
        stale = "" if run.spec_digest == record.digest else "  [stale: spec changed since]"
        verdict = run.verdict or "-"
        typer.echo(
            f"    {run.run_id[:16]:<18} {run.run_kind:<12} {verdict:<12} "
            f"{run.period_start:%Y-%m-%d}..{run.period_end:%Y-%m-%d} "
            f"{','.join(run.symbols)}{stale}"
        )


@strategy_app.command("report")
def strategy_report(
    ctx: typer.Context,
    library: Path = typer.Option(DEFAULT_STRATEGY_LIBRARY, "--library", help="Repository root."),
    results: Path | None = typer.Option(None, "--results", help="Result log. Defaults to library."),
    out: Path | None = typer.Option(
        None, "--out", help="Where to write the page. Defaults to <library>/report.html."
    ),
) -> None:
    """Render the whole library as one page: what exists, what is missing, what is worth.

    Not ``ts report compare`` — that command takes run ids a caller already
    picked. This one walks every entry the library holds, whether or not
    anyone has ever run it, and reports what is not measured and what is
    recorded but can no longer be read back from ``runs/``, alongside a
    strategy-by-strategy table and a cross-strategy correlation matrix.
    """
    settings: Settings = ctx.obj
    repository = StrategyRepository(library)
    link = ResultsLink(results if results is not None else library)
    report = build_library_report(repository, link, settings.runs_dir)
    rendered = render_library_report(report)
    destination = out or library / "report.html"
    write_library_report(rendered, destination)
    typer.echo(
        f"{len(report.rows)} measured, {len(report.not_measured)} not measured, "
        f"{len(report.orphaned)} orphaned result(s) -> {destination}"
    )
    if report.orphaned:
        typer.echo(f"  WARNING: {len(report.orphaned)} recorded result(s) cannot be read back")
    if report.verdicts.nothing_to_approve:
        typer.echo("  nothing to approve: no ROBUST verdict in the library")


@strategy_app.command("diff")
def strategy_diff(
    strategy_id: str = typer.Argument(..., help="Strategy id."),
    left: str = typer.Argument(..., help="Version on the left."),
    right: str = typer.Argument(..., help="Version on the right."),
    library: Path = typer.Option(DEFAULT_STRATEGY_LIBRARY, "--library", help="Repository root."),
) -> None:
    """Show what changed between two versions of a spec."""
    repository = StrategyRepository(library)
    try:
        typer.echo(repository.diff(strategy_id, left, right))
    except KeyError as error:
        typer.echo(str(error))
        raise typer.Exit(code=1) from error


@strategy_app.command("retire")
def strategy_retire(
    strategy_id: str = typer.Argument(..., help="Strategy id."),
    reason: str = typer.Option(..., "--reason", help="Why it is being retired."),
    library: Path = typer.Option(DEFAULT_STRATEGY_LIBRARY, "--library", help="Repository root."),
) -> None:
    """Retire a strategy, recording why."""
    repository = StrategyRepository(library)
    try:
        record = repository.retire(strategy_id, reason)
    except (KeyError, ValidationError) as error:
        typer.echo(str(error))
        raise typer.Exit(code=1) from error
    typer.echo(f"Retired {record.id}: {reason}")


@strategy_app.command("approve")
def strategy_approve(
    run_id: str = typer.Argument(..., help="The run whose verdict justifies approval."),
    library: Path = typer.Option(DEFAULT_STRATEGY_LIBRARY, "--library", help="Repository root."),
    results: Path | None = typer.Option(None, "--results", help="Result log. Defaults to library."),
) -> None:
    """Approve the strategy a ROBUST run evaluated.

    Approval names a run rather than a strategy on purpose: the verdict, the
    selector and the strategy are all read from the recorded run, so nothing
    about the approval is supplied by whoever wants it.
    """
    repository = StrategyRepository(library)
    link = ResultsLink(results if results is not None else library)
    try:
        record = approve_from_result(repository, link, run_id)
    except (KeyError, ValidationError) as error:
        typer.echo(str(error))
        raise typer.Exit(code=1) from error
    typer.echo(f"Approved {record.id} v{record.spec.version} on run {run_id}")


@strategy_app.command("space")
def strategy_space(
    strategy_id: str = typer.Argument(..., help="Strategy id in the library."),
    library: Path = typer.Option(DEFAULT_STRATEGY_LIBRARY, "--library", help="Repository root."),
    out: Path | None = typer.Option(None, "--out", help="Where to write the draft space JSON."),
    axis: list[str] = typer.Option(
        [], "--axis", help="Keep only these axes; repeatable. Omit to keep every candidate."
    ),
    force: bool = typer.Option(False, "--force", help="Overwrite an existing --out file."),
) -> None:
    """Derive a draft search space from a strategy, pointers included.

    Nobody types a JSON pointer: the tool walks the spec, groups every tunable
    position by role, and writes the pointers itself. What is left to a human is
    deleting axes that should not vary and editing ranges — both safe, neither
    able to produce a pointer that does not resolve.

    Ranges are proposals. The rule behind each is printed next to it, because an
    author who disagrees with a range needs to know what they are disagreeing
    with.
    """
    repository = StrategyRepository(library)
    try:
        record = repository.get(strategy_id)
    except KeyError as error:
        typer.echo(str(error))
        raise typer.Exit(code=1) from error

    spec = record.spec
    candidates = build_candidates(spec)
    typer.echo(render(spec, candidates))

    document = build_space_document(spec, keep=axis or None)
    document, notes = prune(spec, document)
    for note in notes:
        typer.echo(f"  pruned: {note}")

    try:
        verify(spec, document)
    except ValidationError as error:
        typer.echo(f"generated space does not apply cleanly: {error}")
        raise typer.Exit(code=1) from error
    typer.echo(
        f"\nverified: {len(document['axes'])} axes, "
        f"{sum(len(item['paths']) for item in document['axes'])} pointers all resolve, "
        "every value leaves a valid spec."
    )

    if out is None:
        typer.echo(json.dumps(document, indent=2))
        return
    if out.exists() and not force:
        typer.echo(f"{out} exists; pass --force to overwrite.")
        raise typer.Exit(code=1)
    write_space(document, out)
    typer.echo(f"Wrote {out}")


@strategy_app.command("schema-export")
def strategy_schema_export(
    out: Path = typer.Option(SCHEMA_JSON_PATH, "--out", help="Where to write the JSON Schema."),
) -> None:
    """(Re-)generate the JSON Schema editors use to validate strategy files."""
    out.write_text(json.dumps(strategy_json_schema(), indent=2) + "\n", encoding="utf-8")
    typer.echo(f"Wrote schema to {out}")


@ingest_app.command("report")
def ingest_report(
    cards: Path = typer.Option(DEFAULT_CARDS_DIR, "--cards", help="Directory of scraped cards."),
    overrides: Path = typer.Option(
        DEFAULT_OVERRIDES_DIR, "--overrides", help="Directory of reviewer overrides."
    ),
    exit_library: Path = typer.Option(
        DEFAULT_LIBRARY_PATH,
        "--exit-library",
        help="Exit preset library exit_ref is checked against.",
    ),
    out: Path | None = typer.Option(None, "--out", help="Write the report here as well."),
    examples: int = typer.Option(2, "--examples", help="Example cards shown per obstacle."),
) -> None:
    """Count what a scrape holds and what stops each card from converting.

    Prints one line per obstacle: how many cards it stopped first, how many it
    affects at all, and how many it is the *only* reason for — the last being
    the number that says what removing it would buy.
    """
    if not cards.is_dir():
        typer.echo(f"{cards} is not a directory of scraped cards.")
        raise typer.Exit(code=1)
    report = triage(
        [card for _, card in load_cards(cards)],
        known_exit_ids(exit_library),
        load_overrides(overrides),
    )
    text = render_ingest(report, examples)
    typer.echo(text)
    if out is not None:
        out.write_text(text + "\n", encoding="utf-8")
        typer.echo(f"Wrote {out}")


@ingest_app.command("convert")
def ingest_convert(
    out: Path = typer.Argument(..., help="Directory converted specs are written to."),
    cards: Path = typer.Option(DEFAULT_CARDS_DIR, "--cards", help="Directory of scraped cards."),
    overrides: Path = typer.Option(
        DEFAULT_OVERRIDES_DIR, "--overrides", help="Directory of reviewer overrides."
    ),
    exit_library: Path = typer.Option(
        DEFAULT_LIBRARY_PATH,
        "--exit-library",
        help="Exit preset library exit_ref is checked against.",
    ),
    library: Path | None = typer.Option(
        None, "--library", help="Also import each converted spec into this strategy repository."
    ),
    author: str = typer.Option("ingest", "--author", help="Author recorded on imported entries."),
) -> None:
    """Convert every card that converts, and say nothing about the ones that do not.

    A spec is written only after it passes the same validation
    ``ts strategy validate`` runs. With ``--library`` each one is also imported
    as a ``DRAFT``, carrying the card's URL as its source and the conversion's
    assumptions as its notes — so what the converter had to decide for itself
    stays attached to the entry.
    """
    if not cards.is_dir():
        typer.echo(f"{cards} is not a directory of scraped cards.")
        raise typer.Exit(code=1)
    report = triage(
        [card for _, card in load_cards(cards)],
        known_exit_ids(exit_library),
        load_overrides(overrides),
    )
    out.mkdir(parents=True, exist_ok=True)
    repository = StrategyRepository(library) if library is not None else None
    for conversion in report.converted:
        spec = conversion.spec
        assert spec is not None
        path = out / f"{spec.id}.json"
        path.write_text(spec.model_dump_json(indent=2) + "\n", encoding="utf-8")
        typer.echo(f"{conversion.card_id} -> {path}")
        if repository is not None:
            record = repository.add(
                spec,
                name=conversion.title or conversion.card_id,
                author=author,
                source=conversion.source_url,
                tags=("scraped",),
                notes="\n".join(conversion.assumptions),
            )
            typer.echo(f"  imported as {record.id} [{record.status.value}] -> {record.path}")
    typer.echo(
        f"Converted {len(report.converted)} of {report.total}; "
        f"{len(report.review_shortlist)} card(s) await review."
    )


@strategy_app.command("normalize")
def strategy_normalize(
    ctx: typer.Context,
    out: Path = typer.Argument(..., help="Directory the normalised tree is written to."),
    cards: Path = typer.Option(DEFAULT_CARDS_DIR, "--cards", help="Directory of scraped cards."),
    exit_library: Path = typer.Option(
        DEFAULT_LIBRARY_PATH,
        "--exit-library",
        help="Exit preset library exit_ref is checked against.",
    ),
    max_cost_ratio: float = typer.Option(
        DEFAULT_MAX_COST_RATIO,
        "--max-cost-ratio",
        help="Refuse a symbol at a bar size once the round turn is this fraction of the "
        "median bar's range.",
    ),
) -> None:
    """Render every scraped card as a spec, filling in what the page never said.

    The strict reader (``ts strategy ingest convert``) refuses a card the moment
    it is silent or unreadable, and converts one card in 883. This command
    answers a different question — what is the most defensible strategy each
    page implies, on data this store actually holds — and so supplies the
    missing values instead of refusing.

    Nothing it writes is a measurement or a library record. Each spec carries a
    fidelity grade in ``manifest.json``: ``read`` means every condition is a
    sentence the page wrote, ``partial`` means some were dropped or salvaged,
    and ``archetype`` means the entry was written here from the card's
    indicators or family and its page never stated a rule at all.
    """
    settings: Settings = ctx.obj
    if not cards.is_dir():
        typer.echo(f"{cards} is not a directory of scraped cards.")
        raise typer.Exit(code=1)

    store = ParquetStore(settings.data_dir)
    instruments = load_instruments(settings.instruments_path)
    symbols = sorted(store.symbols())
    coverage = measure_coverage(store.get, instruments, symbols, list(Timeframe))
    typer.echo(f"{len(coverage.series)} series measured across {len(symbols)} symbol(s)")

    exit_ids = known_exit_ids(exit_library)
    results = [
        normalise_card(
            card, coverage=coverage, known_exit_ids=exit_ids, max_cost_ratio=max_cost_ratio
        )
        for _path, card in load_cards(cards)
    ]
    corpus = write_corpus(results, out, source=str(cards))

    typer.echo(f"normalised {len(corpus.normalised)} of {len(corpus.results)} card(s) -> {out}")
    for fidelity in Fidelity:
        found = corpus.by_fidelity(fidelity)
        if found:
            typer.echo(f"  {fidelity.value:10} {len(found)}")
    if corpus.refused:
        typer.echo(f"  refused    {len(corpus.refused)}")
    typer.echo(f"Manifest: {out / 'manifest.json'}   index: {out / 'index.csv'}")


@strategy_app.command("screen")
def strategy_screen(
    ctx: typer.Context,
    specs: Path = typer.Argument(..., help="Directory of strategy specs, searched recursively."),
    out: Path = typer.Option(
        Path("reports/screen.html"), "--out", help="Where the HTML report is written."
    ),
    manifest: Path | None = typer.Option(
        None,
        "--manifest",
        help="Normalised corpus manifest, for fidelity and family labels. "
        "Defaults to manifest.json beside the specs' root.",
    ),
    bar_budget: int = typer.Option(
        DEFAULT_BAR_BUDGET, "--bars", help="Bars each run gets, back from the holdout boundary."
    ),
    holdout: float = typer.Option(
        DEFAULT_HOLDOUT_FRACTION,
        "--holdout",
        help="Share of every series withheld from the screen. Cut before anything runs.",
    ),
    symbols_per_spec: int = typer.Option(
        1,
        "--symbols",
        help="Instruments each strategy is run on, cheapest round turn first. 0 means every "
        "instrument its universe names and the store holds.",
    ),
    keep_max_dd: float | None = typer.Option(
        None,
        "--keep-max-dd",
        help="Write the whole run to the store when its maximum drawdown is below this "
        "fraction, e.g. 0.10. Every task still runs — a drawdown is not knowable without "
        "running — so this decides what is kept, not what is measured. Rows that do not "
        "qualify are counted and reported, never silently dropped.",
    ),
    risk_pct: float = typer.Option(0.01, "--risk", help="Fraction of equity risked per trade."),
    workers: int = typer.Option(1, "--workers", help="Processes to spread the runs over."),
    returns_sample: int = typer.Option(
        200,
        "--returns-sample",
        help="Tasks that bring back daily returns, for the effective-trials estimate.",
    ),
    limit: int | None = typer.Option(
        None, "--limit", help="Run only the first N tasks. For sizing the job before committing."
    ),
    exit_library: Path = typer.Option(
        DEFAULT_LIBRARY_PATH, "--exit-library", help="Exit preset library."
    ),
) -> None:
    """Screen a shelf of strategies over the pre-holdout history, and report.

    Hypotheses, not evidence: one window, the parameters each file names, no
    folds, no selection, no null. The last ``--holdout`` of every series is cut
    off before any run is built, and the frames a run walks do not contain those
    bars — so the holdout stays clean for the walk-forward that confirms a
    candidate later.

    Re-running with the same inputs is a no-op: the screen id is a digest of the
    code, the spec set, the universe and the screen's own settings, and a
    finished screen is not recomputed. An interrupted one resumes from its own
    row log, having lost at most the task that was in flight.
    """
    settings: Settings = ctx.obj
    if not specs.is_dir():
        typer.echo(f"{specs} is not a directory of strategy specs.")
        raise typer.Exit(code=1)

    spec_paths = sorted(p for p in specs.rglob("*.json") if not p.name.endswith(".meta.json"))
    if not spec_paths:
        typer.echo(f"No strategy specs under {specs}.")
        raise typer.Exit(code=1)

    store = ParquetStore(settings.data_dir)
    instruments = load_instruments(settings.instruments_path)
    universe = sorted(store.symbols())
    coverage = measure_coverage(store.get, instruments, universe, list(Timeframe))

    tasks, skipped = plan_tasks(
        spec_paths,
        coverage,
        symbols_per_spec=symbols_per_spec,
        bar_budget=bar_budget,
        holdout_fraction=holdout,
        risk_pct=risk_pct,
        returns_sample=returns_sample,
        keep_run_max_dd=keep_max_dd,
    )
    if limit is not None:
        tasks = tasks[:limit]

    identity = screen_id(
        spec_paths,
        universe=universe,
        bar_budget=bar_budget,
        holdout_fraction=holdout,
        risk_pct=risk_pct,
        symbols_per_spec=symbols_per_spec,
    )
    root = settings.runs_dir / SCREEN_ROOT / identity
    finished = load_manifest(root)
    if finished is not None:
        typer.echo(f"screen {identity} is already complete at {root}; nothing re-run")
        raise typer.Exit(code=0)

    screen_store = ScreenStore(root)
    already = len(screen_store.completed())
    typer.echo(
        f"screen {identity}: {len(spec_paths)} spec(s), {len(tasks)} run(s) planned"
        + (f", {already} already done" if already else "")
        + (f", {len(skipped)} spec(s) had no stored instrument" if skipped else "")
    )

    done = already
    started = time.monotonic()

    def progress(row: ScreenRow) -> None:
        del row
        nonlocal done
        done += 1
        if done % 50 == 0 or done == len(tasks):
            elapsed = time.monotonic() - started
            rate = (done - already) / elapsed if elapsed > 0 else 0.0
            left = (len(tasks) - done) / rate if rate > 0 else 0.0
            typer.echo(
                f"  {done}/{len(tasks)}  {elapsed / 60:.1f} min elapsed, ~{left / 60:.1f} min left"
            )

    rows = run_screen(
        tasks,
        screen_store,
        data_dir=settings.data_dir,
        instruments_path=settings.instruments_path,
        exit_library=exit_library,
        workers=workers,
        runs_dir=settings.runs_dir if keep_max_dd is not None else None,
        on_row=progress,
    )
    rows_path, returns_path = screen_store.materialise()

    manifest_path = manifest if manifest is not None else specs.parent / "manifest.json"
    meta = load_corpus_meta(manifest_path)
    z_scores = cross_sectional_z(rows)

    # Only the markets actually screened: a factor measured over instruments
    # nobody ran would describe a screen that did not happen.
    market_returns = _market_returns(store, sorted({row.symbol for row in rows}), holdout)
    estimate = trials_module.estimate(
        strategy_returns={row.key: row.daily_returns for row in rows if row.daily_returns},
        market_returns=market_returns,
        n_strategies=len({row.spec_id for row in rows}),
        n_markets=len({row.symbol for row in rows}),
        n_trials_raw=len(rows),
        signature_floor=distinct_signatures(spec_paths, meta),
    )

    boundaries = tuple(boundary for boundary in _screen_boundaries(store, rows, holdout))
    earliest = min((item.boundary for item in boundaries), default=None)
    note = (
        f"Отрезано {holdout:.0%} каждой ленты; самая ранняя граница — "
        f"{earliest:%Y-%m-%d %H:%M} UTC."
        if earliest is not None
        else f"Отрезано {holdout:.0%} каждой ленты."
    )
    ScreenManifest(
        screen_id=identity,
        generated=now_iso(),
        source_digest=code_version().source_digest,
        holdout_fraction=holdout,
        boundaries=boundaries,
        n_specs=len(spec_paths),
        n_tasks=len(tasks),
        bar_budget=bar_budget,
        symbols_per_spec=symbols_per_spec,
        trials=estimate.as_record(),
        skipped=skipped,
    ).write(root)

    report = build_screen_report(
        rows,
        meta,
        z_scores,
        trials=estimate,
        bar_budget=bar_budget,
        holdout_fraction=holdout,
        holdout_note=note,
        skipped=skipped,
    )
    path = write_screen_report(render_screen_report(report), out)

    if keep_max_dd is not None:
        kept = sum(1 for row in rows if row.kept_run)
        traded = sum(1 for row in rows if row.error is None and row.trades > 0)
        typer.echo(
            f"drawdown filter < {keep_max_dd:.0%}: kept {kept} full run(s), "
            f"{traded - kept} of {traded} that traded did not qualify"
        )
    typer.echo(
        f"{len(report.completed)}/{report.total} completed, {len(report.traded)} traded, "
        f"{len(report.rankable)} with >= {MIN_TRADES_FOR_RANKING} trades, "
        f"{len(report.positive)} of those above zero"
    )
    typer.echo(
        f"trials: {estimate.n_trials_raw} raw, {estimate.n_trials_effective:.1f} effective "
        f"(S_eff {estimate.strategies_effective:.1f} x M_eff {estimate.markets_effective:.2f})"
    )
    typer.echo(f"rows: {rows_path}" + (f"   returns: {returns_path}" if returns_path else ""))
    typer.echo(f"Report: {path}")


def _market_returns(
    store: ParquetStore, universe: Sequence[str], holdout: float
) -> dict[str, tuple[float, ...]]:
    """Daily returns per market, over the pre-holdout history alone.

    The market factor of the effective-trials estimate must be measured on the
    same bars the screen saw; taking it from the whole series would let the
    holdout inform a number the screen publishes.
    """
    from trading_system.validation.holdout import screen_frame as _cut

    found: dict[str, tuple[float, ...]] = {}
    for symbol in universe:
        frame = store.get(symbol, Timeframe.D1)
        if frame.is_empty or len(frame) < 30:
            continue
        visible, _boundary = _cut(frame, fraction=holdout)
        closes = visible.df["close"].to_list()
        found[symbol] = tuple(
            (closes[index] / closes[index - 1]) - 1.0
            for index in range(1, len(closes))
            if closes[index - 1]
        )
    return found


def _screen_boundaries(
    store: ParquetStore, rows: Sequence[ScreenRow], holdout: float
) -> list[HoldoutBoundary]:
    """Where each series a screen touched was cut."""
    from trading_system.validation.holdout import screen_frame as _cut

    seen: dict[tuple[str, str], HoldoutBoundary] = {}
    for row in rows:
        if not row.timeframe:
            continue
        key = (row.symbol, row.timeframe)
        if key in seen:
            continue
        frame = store.get(row.symbol, Timeframe(row.timeframe))
        if frame.is_empty:
            continue
        seen[key] = _cut(frame, fraction=holdout)[1]
    return [seen[key] for key in sorted(seen)]


@strategy_app.command("keep-top")
def strategy_keep_top(
    ctx: typer.Context,
    specs: Path = typer.Argument(..., help="The same directory of specs the screen ran."),
    screen: str = typer.Option(..., "--screen-id", help="Screen whose rows the choice is made on."),
    top: int = typer.Option(
        5, "--top", help="Runs kept per (instrument, timeframe) pair, best first."
    ),
    min_trades: int = typer.Option(
        100, "--min-trades", help="Trades a row needs before it is eligible to be kept."
    ),
    workers: int = typer.Option(1, "--workers", help="Processes to spread the re-runs over."),
    exit_library: Path = typer.Option(
        DEFAULT_LIBRARY_PATH, "--exit-library", help="Exit preset library."
    ),
) -> None:
    """Store the full runs of the best rows per market, by re-running them.

    The screen's second pass. The first writes nothing but compact rows, because
    a run directory per task is thousands of directories for a few numbers each
    — the cost P15 stage 1.5 already paid once. This pass picks a bounded set
    and reconstructs it: re-running is deterministic and the row's stored
    ``result_digest`` is what makes the reconstruction checkable rather than a
    fresh measurement.

    **Chosen by cross-sectional z within each instrument, not by drawdown.**
    A raw Sortino on XAUUSD and on EURUSD are different quantities; z is the
    unit that is comparable across markets. And a drawdown gate was measured on
    the delivered screen to select strategies that barely traded — positive at
    42.2% against a 42.0% base rate — while moving with ``risk_pct`` rather than
    with the strategy. Top-N *per pair* rather than top-N overall, so the kept
    set covers every market instead of collapsing into whichever one happened to
    suit the shelf.
    """
    settings: Settings = ctx.obj
    root = settings.runs_dir / SCREEN_ROOT / screen
    rows_path = root / "rows.parquet"
    if not rows_path.is_file():
        typer.echo(f"no screen rows at {rows_path}")
        raise typer.Exit(code=1)

    table = pl.read_parquet(rows_path)
    rows = [ScreenRow.from_record(record) for record in table.to_dicts()]
    scores = cross_sectional_z(rows, statistic="expectancy_r")
    eligible = [
        row for row in rows if row.error is None and row.trades >= min_trades and row.key in scores
    ]
    buckets: dict[tuple[str, str], list[ScreenRow]] = {}
    for row in eligible:
        buckets.setdefault((row.symbol, row.timeframe), []).append(row)
    chosen: list[ScreenRow] = []
    for group in buckets.values():
        group.sort(key=lambda item: -scores[item.key])
        chosen.extend(group[:top])
    typer.echo(
        f"{len(rows)} row(s), {len(eligible)} eligible (>= {min_trades} trades), "
        f"{len(buckets)} (instrument, timeframe) pair(s), keeping {len(chosen)}"
    )
    if not chosen:
        raise typer.Exit(code=0)

    paths = {p.stem: p for p in specs.rglob("*.json") if not p.name.endswith(".meta.json")}
    tasks = []
    missing = 0
    for row in chosen:
        path = paths.get(row.spec_id)
        if path is None:
            missing += 1
            continue
        tasks.append(
            ScreenTask(
                spec_path=path,
                symbol=row.symbol,
                bar_budget=row.bars,
                cost_ratio=row.cost_ratio,
                keep_run=True,
            )
        )
    if missing:
        typer.echo(f"  {missing} chosen row(s) have no spec file under {specs}; skipped")

    store = ScreenStore(root / f"kept_top{top}")
    done = 0

    def progress(row: ScreenRow) -> None:
        del row  # the row is kept by the store; this only counts
        nonlocal done
        done += 1
        if done % 25 == 0 or done == len(tasks):
            typer.echo(f"  {done}/{len(tasks)}")

    kept_rows = run_screen(
        tasks,
        store,
        data_dir=settings.data_dir,
        instruments_path=settings.instruments_path,
        exit_library=exit_library,
        workers=workers,
        runs_dir=settings.runs_dir,
        on_row=progress,
    )
    store.materialise()
    stored = sum(1 for row in kept_rows if row.kept_run)
    typer.echo(f"stored {stored} full run(s) under {settings.runs_dir}")
    typer.echo("Render the catalogue with: ts report index --out reports/runs/index.html")


class BacktestCliConfig(BaseModel):
    """Everything one run needs beyond its strategy and its bars.

    Shared by the flat ``ts backtest`` and, through
    :class:`WalkForwardCliConfig`, by the fold runners: a field that changes a
    run's result has to reach every command that builds one, and a second copy
    of this list is a second place to forget it.

    Attributes:
        symbol: Instrument to trade, read from the local store.
        timeframe: Bar size to trade.
        account_currency: Denomination of the account.
        starting_balance: Opening balance the run starts from.
        risk_pct: Fraction of equity risked per trade
            (:class:`~trading_system.risk.sizing.methods.FixedFractional`).
        run_seed: Seed for the per-fill random streams.
        atr_period: ATR period the cost model's volatility ratio is built
            from.
        atr_baseline_bars: Rolling window the ATR is divided by.
        prop_profile: Name of the prop-firm profile in
            ``configs/prop_profiles.yaml`` this account trades under.
            ``ftmo_swing`` by default: every strategy in the repository holds
            through the weekend, which is what a swing plan is for, and it is
            also the strictest complete profile on file — a default that errs
            towards refusing. ``null`` trades under the venue leverage declared
            on each instrument alone.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str
    timeframe: Timeframe
    account_currency: str = "USD"
    starting_balance: Decimal = Field(default=Decimal(100_000), gt=0)
    risk_pct: float = Field(gt=0, le=1)
    run_seed: int = 0
    atr_period: int = Field(default=14, gt=0)
    atr_baseline_bars: int = Field(default=500, gt=1)
    prop_profile: str | None = DEFAULT_PROP_PROFILE


class WalkForwardCliConfig(BacktestCliConfig):
    """What ``ts validate walkforward`` needs on top of a single run's settings.

    The fold geometry (``--mode``, ``--is``, ``--oos``, ``--step``,
    ``--embargo``, ``--warmup``) is on the command line, per the CLAUDE.md
    spec for this command; what is here has no natural single-flag form.

    Attributes:
        max_drain_bars: Bars past each OOS window's ``trade_end`` a position
            already open may still be managed on.
        min_trades_per_fold: Below this many OOS trades, a fold is flagged
            insufficient in the report.
        parallel_threshold_seconds: Below this many seconds for the first run
            of a batch, the rest of that batch runs sequentially rather than
            paying a process pool's spawn cost.
    """

    max_drain_bars: int = Field(gt=0)
    min_trades_per_fold: int = Field(ge=0)
    parallel_threshold_seconds: float = Field(default=2.0, gt=0)


def _load_binding(
    strategy: Path, exit_library: Path
) -> tuple[StrategySpec, ExitPresetSpec, ExitLibrarySpec]:
    """Read a strategy and resolve the exit preset it names.

    Args:
        strategy: Strategy spec JSON file.
        exit_library: Exit preset library the spec's ``exit_ref`` is looked up in.

    Returns:
        The spec, its preset, and the library the preset came from.

    Raises:
        typer.Exit: If ``exit_ref`` names no preset in the library.
    """
    spec = StrategySpec.model_validate_json(strategy.read_text())
    library = ExitLibrarySpec.model_validate_json(exit_library.read_text())
    preset = next((item for item in library.presets if item.id == spec.exit_ref), None)
    if preset is None:
        typer.echo(f"exit_ref {spec.exit_ref!r} not found in {exit_library}")
        raise typer.Exit(code=1)
    return spec, preset, library


def _run_inputs(
    settings: Settings,
    cli_config: BacktestCliConfig,
    *,
    spec: StrategySpec,
    preset: ExitPresetSpec,
    frame: OHLCVFrame,
) -> RunInputs:
    """Assemble the description of one run from a config file and a strategy.

    One function rather than a copy per command, because
    :meth:`~trading_system.backtest.spec.RunInputs.components` folds every field
    here into the run id: a command that built its inputs slightly differently
    would produce runs that are not comparable with the others' and would not
    say so.

    Args:
        settings: Process settings, naming the instrument and profile files.
        cli_config: The account, sizing and cost settings.
        spec: The strategy to trade.
        preset: Its exit preset.
        frame: The bars, already sliced to the period being run.

    Returns:
        The run, not yet walked.
    """
    key = StreamKey(cli_config.symbol, cli_config.timeframe)
    instruments = load_instruments(settings.instruments_path)
    store = ParquetStore(settings.data_dir)
    return RunInputs(
        config=BacktestConfig(
            account_currency=cli_config.account_currency,
            starting_balance=cli_config.starting_balance,
            atr_period=cli_config.atr_period,
            atr_baseline_bars=cli_config.atr_baseline_bars,
        ),
        streams={key: frame},
        bindings=(StrategyBinding(spec=spec, exit_preset=preset, keys=(key,)),),
        instruments=instruments,
        costs=CostConfig(run_seed=cli_config.run_seed),
        sizing=FixedFractional(risk_pct=cli_config.risk_pct),
        risk=_risk_config(settings, cli_config.prop_profile),
        converter=build_converter(
            (instruments[cli_config.symbol],),
            account_currency=cli_config.account_currency,
            timeframe=cli_config.timeframe,
            load=store.get,
        ),
    )


@app.command()
def backtest(
    ctx: typer.Context,
    config: Path = typer.Option(..., "--config", help="Run config JSON file."),
    strategy: Path = typer.Option(..., "--strategy", help="Strategy spec JSON file."),
    exit_library: Path = typer.Option(
        DEFAULT_LIBRARY_PATH, "--exit-library", help="Exit preset library."
    ),
    start: datetime | None = typer.Option(None, "--start", help="Inclusive lower bound."),
    end: datetime | None = typer.Option(None, "--end", help="Exclusive upper bound."),
) -> None:
    """Walk one strategy over one stream once, and store what it produced.

    A single period on the parameters the strategy file names — no folds, no
    selection, no null. That makes it an exploratory instrument and not
    evidence: nothing here is out of sample, so a good number produced by this
    command is a reason to run ``ts validate walkforward``, never a result.
    """
    settings: Settings = ctx.obj
    cli_config = BacktestCliConfig.model_validate_json(config.read_text())
    spec, preset, _ = _load_binding(strategy, exit_library)

    frame = ParquetStore(settings.data_dir).get(
        cli_config.symbol, cli_config.timeframe, _as_utc(start), _as_utc(end)
    )
    if frame.is_empty:
        typer.echo(f"No data stored for {cli_config.symbol} {cli_config.timeframe}.")
        raise typer.Exit(code=1)

    inputs = _run_inputs(settings, cli_config, spec=spec, preset=preset, frame=frame)
    result = inputs.run()
    manifest = inputs.manifest()
    path = write_run(settings.runs_dir, manifest, result)

    typer.echo(
        f"{spec.id} on {cli_config.symbol} {cli_config.timeframe}: "
        f"{len(frame)} bars, {frame.start:%Y-%m-%d} .. {frame.end:%Y-%m-%d}"
    )
    typer.echo(f"run {manifest.run_id} -> {path}")
    typer.echo(
        f"  trades: {len(result.trades)}   fills: {result.fills}   "
        f"open_at_end: {result.open_at_end}   expired_orders: {result.expired_orders}"
    )
    if result.trades:
        stats = trade_stats(result.trades)
        typer.echo(
            f"  expectancy_r: {stats.expectancy_r:+.4f}   winrate: {stats.winrate:.3f}   "
            f"profit_factor: {stats.profit_factor:.3f}"
        )
        daily = daily_curve(result.curve)
        typer.echo(f"  total_return: {total_return(daily):+.2%} over {len(daily.days)} days")
    # Non-zero only; the run's own JSON keeps every reason, including the
    # zeros, which is where "this never happened" is a recorded fact.
    for label, counts in (
        ("signal_drops", result.signal_drops),
        ("entry_drops", result.entry_drops),
        ("rejections", result.rejections),
        ("degradations", result.degradations),
        ("exit_drops", result.exit_drops),
    ):
        fired = {str(reason): count for reason, count in counts.items() if count}
        if fired:
            typer.echo(f"  {label}: {fired}")
    typer.echo(f"Render it with: ts report run {manifest.run_id}")


def _parse_duration(text: str) -> timedelta:
    """Parse a compact duration like ``"2y"``, ``"6m"``, ``"5d"``, ``"3w"``.

    Args:
        text: The duration string: an integer followed by one of
            ``d`` (day), ``w`` (7 days), ``m`` (30 days), ``y`` (365 days).
            Calendar months and years are approximated rather than resolved
            against a calendar — a fold's own boundaries are snapped to a
            real day boundary by :mod:`trading_system.validation.splitting`
            regardless, so the approximation only decides roughly how far
            each fold advances, not where it actually lands.

    Returns:
        The duration.

    Raises:
        typer.BadParameter: If the text does not parse.
    """
    days_per_unit = {"d": 1, "w": 7, "m": 30, "y": 365}
    if len(text) < 2 or text[-1] not in days_per_unit:
        raise typer.BadParameter(f"invalid duration {text!r}; expected e.g. '2y', '6m', '5d'")
    try:
        value = int(text[:-1])
    except ValueError as error:
        raise typer.BadParameter(
            f"invalid duration {text!r}; expected e.g. '2y', '6m', '5d'"
        ) from error
    if value <= 0:
        raise typer.BadParameter(f"invalid duration {text!r}: must be positive")
    return timedelta(days=value * days_per_unit[text[-1]])


@validate_app.command("walkforward")
def validate_walkforward(
    ctx: typer.Context,
    config: Path = typer.Option(..., "--config", help="Walk-forward run config JSON file."),
    strategy: Path = typer.Option(..., "--strategy", help="Strategy spec JSON file."),
    mode: WalkForwardMode = typer.Option(..., "--mode", help="anchored or rolling."),
    is_span: str = typer.Option(..., "--is", help="In-sample span, e.g. '2y'."),
    oos_span: str = typer.Option(..., "--oos", help="Out-of-sample span, e.g. '6m'."),
    step: str = typer.Option(..., "--step", help="Advance per fold, e.g. '6m'."),
    embargo: str = typer.Option(
        ..., "--embargo", help="Gap enforced after each IS window, e.g. '5d'."
    ),
    warmup: str = typer.Option("30d", "--warmup", help="Prefix reserved for indicator warmup."),
    exit_library: Path = typer.Option(
        DEFAULT_LIBRARY_PATH, "--exit-library", help="Exit preset library."
    ),
    out: Path | None = typer.Option(
        None, "--out", help="Report path. Defaults next to the walk-forward's own manifest."
    ),
    record: bool = typer.Option(
        False, "--record", help="Bind this run's metrics to the strategy in the results log."
    ),
    library_root: Path = typer.Option(
        DEFAULT_STRATEGY_LIBRARY, "--library", help="Strategy library and results log root."
    ),
) -> None:
    """Walk history in folds: IS run, OOS run, and a report of every window's boundary.

    No parameter selection happens here — every fold's OOS run uses the same
    parameters the strategy file names. See CLAUDE.md P15 stage 1: this stage
    is the harness, not the optimiser.
    """
    settings: Settings = ctx.obj
    cli_config = WalkForwardCliConfig.model_validate_json(config.read_text())

    frame = ParquetStore(settings.data_dir).get(cli_config.symbol, cli_config.timeframe)
    if frame.is_empty:
        typer.echo(f"No data stored for {cli_config.symbol} {cli_config.timeframe}.")
        raise typer.Exit(code=1)

    spec, preset, _ = _load_binding(strategy, exit_library)
    base = _run_inputs(settings, cli_config, spec=spec, preset=preset, frame=frame)
    splitter = WalkForwardSplitter(
        mode=mode,
        is_span=_parse_duration(is_span),
        oos_span=_parse_duration(oos_span),
        step=_parse_duration(step),
        embargo=_parse_duration(embargo),
        warmup=_parse_duration(warmup),
    )
    runner = WalkForwardRunner(
        base=base,
        splitter=splitter,
        selector=IdentitySelector(base),
        store_root=settings.runs_dir,
        max_drain_bars=cli_config.max_drain_bars,
        parallel_threshold_seconds=cli_config.parallel_threshold_seconds,
    )
    result = runner.run()
    report = build_report(result, min_trades_per_fold=cli_config.min_trades_per_fold)
    out_path = out if out is not None else result.manifest_path.parent / "report.json"
    write_report(report, out_path)

    typer.echo(f"walk-forward {result.wf_id}: {report.n_folds} folds, report written to {out_path}")
    if report.insufficient_sample:
        flagged = sum(1 for fold in report.folds if fold.insufficient)
        typer.echo(
            f"  {flagged}/{report.n_folds} fold(s) below "
            f"min_trades_per_fold={cli_config.min_trades_per_fold}"
        )
    if record:
        _record_walkforward(
            spec=spec,
            base=base,
            report=report,
            wf_id=result.wf_id,
            selector_key=IdentitySelector(base).key(),
            library=library_root,
        )


class SearchMethod(StrEnum):
    """Which :class:`~trading_system.validation.optimization.ParameterSearch` to use."""

    GRID = "grid"
    RANDOM = "random"
    OPTUNA = "optuna"


class ObjectiveName(StrEnum):
    """Which objective scores a trial.

    One member, because one objective exists. Adding a flag value here without
    adding a :class:`~trading_system.validation.objective.Objective` behind it
    would be a command-line option that silently does nothing.
    """

    SORTINO_SQRT_N = "sortino_sqrt_n"


def _build_search(method: SearchMethod, seed: int, startup_trials: int) -> ParameterSearch:
    """The search a ``--method`` flag names."""
    if method is SearchMethod.GRID:
        return GridSearch()
    if method is SearchMethod.RANDOM:
        return RandomSearch(seed=seed)
    return OptunaSearch(seed=seed, n_startup_trials=startup_trials)


@validate_app.command("optimize")
def validate_optimize(
    ctx: typer.Context,
    config: Path = typer.Option(..., "--config", help="Walk-forward run config JSON file."),
    strategy: Path = typer.Option(..., "--strategy", help="Strategy spec JSON file."),
    space: Path = typer.Option(..., "--space", help="Search space JSON file."),
    method: SearchMethod = typer.Option(SearchMethod.GRID, "--method", help="Search method."),
    trial_budget: int = typer.Option(
        ...,
        "--trial-budget",
        help="Parameter sets each fold may evaluate. Per fold, never carried.",
    ),
    objective: ObjectiveName = typer.Option(
        ObjectiveName.SORTINO_SQRT_N, "--objective", help="How a trial is scored."
    ),
    mode: WalkForwardMode = typer.Option(..., "--mode", help="anchored or rolling."),
    is_span: str = typer.Option(..., "--is", help="In-sample span, e.g. '360d'."),
    oos_span: str = typer.Option(..., "--oos", help="Out-of-sample span, e.g. '270d'."),
    step: str = typer.Option(..., "--step", help="Advance per fold, e.g. '270d'."),
    embargo: str = typer.Option(..., "--embargo", help="Gap after each IS window, e.g. '3d'."),
    warmup: str = typer.Option("30d", "--warmup", help="Prefix reserved for indicator warmup."),
    cv_k: int = typer.Option(
        0, "--cv-k", help="PurgedKFold pieces inside each IS window. 0 disables cross-validation."
    ),
    cv_embargo: str = typer.Option("2d", "--cv-embargo", help="Embargo after each CV test piece."),
    cv_label_span: str = typer.Option(
        "5d", "--cv-label-span", help="Longest a position may stay open; purged before each piece."
    ),
    min_cv_test_span: str = typer.Option(
        "30d", "--min-cv-test-span", help="Shortest CV piece worth scoring."
    ),
    penalty_weight: float = typer.Option(
        0.5, "--penalty-weight", help="How hard neighbourhood instability is penalised."
    ),
    tolerance_sigmas: float = typer.Option(
        1.0, "--tolerance-sigmas", help="Plateau membership margin, in score standard deviations."
    ),
    search_seed: int = typer.Option(0, "--search-seed", help="Seed for random/optuna proposals."),
    startup_trials: int = typer.Option(
        10, "--startup-trials", help="Random trials before TPE starts modelling."
    ),
    exit_library: Path = typer.Option(
        DEFAULT_LIBRARY_PATH, "--exit-library", help="Exit preset library."
    ),
    out: Path | None = typer.Option(
        None, "--out", help="Report path. Defaults next to the walk-forward's own manifest."
    ),
    record: bool = typer.Option(
        False, "--record", help="Bind this run's metrics to the strategy in the results log."
    ),
    library_root: Path = typer.Option(
        DEFAULT_STRATEGY_LIBRARY, "--library", help="Strategy library and results log root."
    ),
) -> None:
    """Tune parameters on each fold's in-sample window, then score the choice out-of-sample.

    The search never sees an out-of-sample bar, and not by convention: see
    :class:`~trading_system.validation.walkforward.OptimizingSelector` on the
    two constructors that refuse to hold one.
    """
    settings: Settings = ctx.obj
    cli_config = WalkForwardCliConfig.model_validate_json(config.read_text())
    search_space = SearchSpace.model_validate_json(space.read_text())

    registry = load_instruments(settings.instruments_path)
    frame = ParquetStore(settings.data_dir).get(cli_config.symbol, cli_config.timeframe)
    if frame.is_empty:
        typer.echo(f"No data stored for {cli_config.symbol} {cli_config.timeframe}.")
        raise typer.Exit(code=1)

    spec = StrategySpec.model_validate_json(strategy.read_text())
    library = ExitLibrarySpec.model_validate_json(exit_library.read_text())
    preset = next((item for item in library.presets if item.id == spec.exit_ref), None)
    if preset is None:
        typer.echo(f"exit_ref {spec.exit_ref!r} not found in {exit_library}")
        raise typer.Exit(code=1)

    if method is SearchMethod.GRID:
        # Counted once, not twice: on a constrained space this enumerates, and
        # enumerating a space twice to build one message is the kind of waste
        # that hides until somebody profiles it.
        needed = search_space.feasible_size()
        if trial_budget < needed:
            typer.echo(
                f"--method grid over this space needs --trial-budget of at least "
                f"{needed}; got {trial_budget}."
            )
            raise typer.Exit(code=1)

    key = StreamKey(cli_config.symbol, cli_config.timeframe)
    base = RunInputs(
        config=BacktestConfig(
            account_currency=cli_config.account_currency,
            starting_balance=cli_config.starting_balance,
            atr_period=cli_config.atr_period,
            atr_baseline_bars=cli_config.atr_baseline_bars,
        ),
        streams={key: frame},
        bindings=(StrategyBinding(spec=spec, exit_preset=preset, keys=(key,)),),
        instruments=registry,
        costs=CostConfig(run_seed=cli_config.run_seed),
        sizing=FixedFractional(risk_pct=cli_config.risk_pct),
        risk=_risk_config(settings, cli_config.prop_profile),
    )
    splitter = WalkForwardSplitter(
        mode=mode,
        is_span=_parse_duration(is_span),
        oos_span=_parse_duration(oos_span),
        step=_parse_duration(step),
        embargo=_parse_duration(embargo),
        warmup=_parse_duration(warmup),
    )
    selector = OptimizingSelector(
        base=base,
        space=search_space,
        search=_build_search(method, search_seed, startup_trials),
        objective=SortinoTimesSqrtTrades(),
        trial_budget=trial_budget,
        store_root=settings.runs_dir,
        cv=None
        if cv_k == 0
        else PurgedKFold(
            k=cv_k,
            embargo=_parse_duration(cv_embargo),
            label_span=_parse_duration(cv_label_span),
        ),
        min_cv_test_span=_parse_duration(min_cv_test_span),
        tolerance_sigmas=tolerance_sigmas,
        penalty_weight=penalty_weight,
    )
    runner = WalkForwardRunner(
        base=base,
        splitter=splitter,
        selector=selector,
        store_root=settings.runs_dir,
        max_drain_bars=cli_config.max_drain_bars,
        parallel_threshold_seconds=cli_config.parallel_threshold_seconds,
    )
    result = runner.run()
    report = build_report(result, min_trades_per_fold=cli_config.min_trades_per_fold)
    out_path = out if out is not None else result.manifest_path.parent / "report.json"
    write_report(report, out_path)

    selections = [
        read_fold_selection(selector.fold_dir(fold_run.fold.index)) for fold_run in result.folds
    ]
    optimization_path = out_path.parent / "optimization.json"
    optimization_path.write_text(
        json.dumps(
            {
                "wf_id": result.wf_id,
                "method": method.value,
                "objective": objective.value,
                "trial_budget": trial_budget,
                "space": search_space.model_dump(mode="json"),
                "ledger": selector.ledger.to_dict(),
                "folds": selections,
            },
            indent=2,
        )
    )

    typer.echo(f"walk-forward {result.wf_id}: {report.n_folds} folds, report at {out_path}")
    typer.echo(
        f"  {method.value} search, {selector.ledger.total_trials} trials / "
        f"{selector.ledger.total_runs} runs, selections at {optimization_path}"
    )
    for fold_report, selection in zip(report.folds, selections, strict=True):
        if selection is None:
            continue
        plateau = selection["plateau"] or {}
        typer.echo(
            f"  fold {fold_report.index}: IS {selection['selected_score']:.3f} "
            f"{selection['selected']} | plateau {plateau.get('plateau_size')}/"
            f"{selection.get('n_scored')} shift {plateau.get('selection_shift')} "
            f"| OOS expectancy_r {fold_report.oos_expectancy_r}"
        )

    if record:
        _record_walkforward(
            spec=spec,
            base=base,
            report=report,
            wf_id=result.wf_id,
            selector_key=selector.key(),
            library=library_root,
        )


def _record_walkforward(
    *,
    spec: StrategySpec,
    base: RunInputs,
    report: WalkForwardReport,
    wf_id: str,
    selector_key: str,
    library: Path,
) -> None:
    """Bind a walk-forward's stitched metrics to its strategy in the results log.

    Records the run **ungraded**: a verdict needs the nulls and the
    perturbations, which neither ``walkforward`` nor ``optimize`` runs. ``ts
    validate verdict`` computes it and attaches it with
    :meth:`~trading_system.strategies.results_link.ResultsLink.grade`.

    Args:
        spec: The strategy that ran — the template, since each fold ran its own
            materialised parameters.
        base: The run inputs, for the manifest and the coverage.
        report: The walk-forward's report, for the stitched metrics.
        wf_id: The walk-forward's id, which is this row's run id.
        selector_key: What chose each fold's parameters. Required, because an
            optimising run is evidence about the pair (spec, selector).
        library: Repository root holding the results log.
    """
    metrics = {
        "expectancy_r": report.stitched_expectancy_r,
        "sharpe": report.stitched_sharpe,
        "sortino": report.stitched_sortino,
        "trades": float(report.stitched_trade_count),
        "folds": float(report.n_folds),
    }
    link = ResultsLink(library)
    try:
        stored = link.record(
            build_record(
                spec=spec,
                manifest=base.manifest(),
                streams=base.streams,
                run_id=wf_id,
                run_kind=RUN_KIND_WALKFORWARD,
                selector_key=selector_key,
                metrics={name: value for name, value in metrics.items() if value is not None},
            )
        )
    except ValidationError as error:
        typer.echo(f"  not recorded: {error}")
        raise typer.Exit(code=1) from error
    typer.echo(
        f"  recorded {stored.strategy_id} v{stored.version} spec={stored.spec_digest[:12]} "
        f"dataset={stored.dataset_hash[:12]} selector={stored.selector_key} -> {link.path}"
    )
    if stored.verdict is None:
        typer.echo("  ungraded: run 'ts validate verdict' to attach one")


class NullCliConfig(BaseModel):
    """Everything ``ts validate null`` needs beyond ``--kind``/``--n``/``--calibration-id``.

    Attributes:
        symbol: Instrument to trade, read from the local store.
        timeframe: Bar size to trade.
        account_currency: Denomination of the account.
        starting_balance: Opening balance.
        risk_pct: Fraction of equity risked per trade.
        run_seed: Seed for the per-fill random streams — shared with the real
            run and every null iteration, per CLAUDE.md P15 stage 1.5.
        atr_period: ATR period the cost model's volatility ratio is built from.
        atr_baseline_bars: Rolling window the ATR is divided by.
        stop_pips: Distance from reference to invalidation the random-entry
            null variants use — see
            :mod:`trading_system.validation.nulls.random_entry`.
        max_concurrent_positions: Cap for the random-entry null's own strategy.
        parallel_threshold_seconds: Below this many seconds for the first
            iteration, the rest of the batch runs sequentially.
        prop_profile: Prop-firm profile the null runs under. The same one the
            real run uses, necessarily: a null calibrated under looser margin
            rules than the run it is compared against would be measuring the
            rules, not the entries.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str
    timeframe: Timeframe
    account_currency: str = "USD"
    starting_balance: Decimal = Field(default=Decimal(100_000), gt=0)
    risk_pct: float = Field(gt=0, le=1)
    run_seed: int = 0
    atr_period: int = Field(default=14, gt=0)
    atr_baseline_bars: int = Field(default=500, gt=1)
    stop_pips: float = Field(default=20.0, gt=0)
    max_concurrent_positions: int = Field(default=50, gt=0)
    parallel_threshold_seconds: float = Field(default=2.0, gt=0)
    prop_profile: str | None = DEFAULT_PROP_PROFILE


def _zero_cost_registry(registry: InstrumentRegistry, symbol: str) -> InstrumentRegistry:
    """``registry`` with ``symbol``'s spread and commission zeroed — the null-A cost model.

    Args:
        registry: The real registry.
        symbol: Instrument the permutation null trades.

    Returns:
        A registry identical to ``registry`` except ``symbol``'s
        ``typical_spread_points`` and ``commission_per_lot`` are zero. Every
        other field — tick size, lot bounds, contract size — is the real
        instrument's own, so nothing about position sizing changes, only cost.
    """
    zeroed = registry[symbol].model_copy(
        update={"typical_spread_points": 0.0, "commission_per_lot": Decimal(0)}
    )
    return InstrumentRegistry({**{s: registry[s] for s in registry.symbols}, symbol: zeroed})


@validate_app.command("null")
def validate_null(
    ctx: typer.Context,
    config: Path = typer.Option(..., "--config", help="Null run config JSON file."),
    strategy: Path = typer.Option(..., "--strategy", help="Strategy spec JSON file."),
    kind: NullKind = typer.Option(
        ..., "--kind", help="permutation | random-entry | random-entry-fixed-hold."
    ),
    n: int = typer.Option(..., "--n", help="Iterations to run.", min=2),
    calibration_id: str = typer.Option(
        ..., "--calibration-id", help="Identifies this calibration; also seeds every iteration."
    ),
    exit_library: Path = typer.Option(
        DEFAULT_LIBRARY_PATH, "--exit-library", help="Exit preset library."
    ),
) -> None:
    """Run one zero process n times and report where the real run sits in its distribution.

    ``permutation`` runs with spread and commission forced to zero (CLAUDE.md
    P15 stage 1.5: the fold-sequence acceptance test needs a strictly zero
    null, not an estimate of expected turnover cost). ``random-entry`` and
    ``random-entry-fixed-hold`` run with the config's own real costs, matched
    against the real strategy's own signal trace.
    """
    settings: Settings = ctx.obj
    cli_config = NullCliConfig.model_validate_json(config.read_text())

    registry = load_instruments(settings.instruments_path)
    frame = ParquetStore(settings.data_dir).get(cli_config.symbol, cli_config.timeframe)
    if frame.is_empty:
        typer.echo(f"No data stored for {cli_config.symbol} {cli_config.timeframe}.")
        raise typer.Exit(code=1)

    spec = StrategySpec.model_validate_json(strategy.read_text())
    library = ExitLibrarySpec.model_validate_json(exit_library.read_text())
    preset = next((item for item in library.presets if item.id == spec.exit_ref), None)
    if preset is None:
        typer.echo(f"exit_ref {spec.exit_ref!r} not found in {exit_library}")
        raise typer.Exit(code=1)

    key = StreamKey(cli_config.symbol, cli_config.timeframe)
    if kind is NullKind.PERMUTATION:
        run_registry = _zero_cost_registry(registry, cli_config.symbol)
    else:
        run_registry = registry
    base = RunInputs(
        config=BacktestConfig(
            account_currency=cli_config.account_currency,
            starting_balance=cli_config.starting_balance,
            atr_period=cli_config.atr_period,
            atr_baseline_bars=cli_config.atr_baseline_bars,
        ),
        streams={key: frame},
        bindings=(StrategyBinding(spec=spec, exit_preset=preset, keys=(key,)),),
        instruments=run_registry,
        costs=CostConfig(run_seed=cli_config.run_seed),
        sizing=FixedFractional(risk_pct=cli_config.risk_pct),
        risk=_risk_config(settings, cli_config.prop_profile),
    )
    real_result = base.run()
    objective = SortinoTimesSqrtTrades()

    finest: Timeframe | None = None
    real_binding: StrategyBinding | None = None
    profile: EntryTraceProfile | None = None
    if kind is NullKind.PERMUTATION:
        finest = cli_config.timeframe
    else:
        binding = base.bindings[0]
        signals = real_signals(base.streams, binding, key)
        profile = build_entry_trace_profile(signals, real_result.trades, key.timeframe)
        if kind is NullKind.RANDOM_ENTRY:
            real_binding = binding

    result = run_calibration(
        kind,
        base,
        objective=objective,
        real_result=real_result,
        n=n,
        calibration_id=calibration_id,
        store_root=settings.runs_dir,
        parallel_threshold_seconds=cli_config.parallel_threshold_seconds,
        finest=finest,
        day_origin=base.config.day_origin,
        key=key,
        real_binding=real_binding,
        profile=profile,
        stop_pips=cli_config.stop_pips,
        max_concurrent_positions=cli_config.max_concurrent_positions,
    )

    typer.echo(
        f"{kind.value}: {result.n_scored}/{result.n} scored, real_score={result.real_score}, "
        f"percentile={result.percentile}, median={result.median_score} "
        f"CI=[{result.median_ci_low}, {result.median_ci_high}]"
    )
    divergence = result.position_count_divergence
    if divergence is not None:
        flag = " (exceeds threshold)" if divergence > POSITION_COUNT_DIVERGENCE_THRESHOLD else ""
        typer.echo(f"  position count divergence: {divergence:.2%}{flag}")
    typer.echo(f"  written to {result.directory}")


@validate_app.command("verdict")
def validate_verdict(
    ctx: typer.Context,
    wf_id: str = typer.Option(..., "--wf-id", help="Walk-forward to grade."),
    config: Path = typer.Option(..., "--config", help="The same run config the walk-forward used."),
    strategy: Path = typer.Option(..., "--strategy", help="The same strategy spec it used."),
    n_null: int = typer.Option(20, "--n", help="Iterations per null."),
    mc_iterations: int = typer.Option(
        10_000,
        "--mc-iterations",
        help="Draws per Monte Carlo family. Resampling replays trades that were already "
        "computed, so ten thousand costs under a second; zero skips it and the verdict's "
        "drawdown check then reports 'not run' rather than passing on a measurement.",
    ),
    exit_library: Path = typer.Option(
        DEFAULT_LIBRARY_PATH, "--exit-library", help="Exit preset library."
    ),
    record: bool = typer.Option(
        False, "--record", help="Attach the verdict to this run in the results log."
    ),
    library_root: Path = typer.Option(
        DEFAULT_STRATEGY_LIBRARY, "--library", help="Strategy library and results log root."
    ),
) -> None:
    """Grade a walk-forward: run both nulls, perturb the data, and decide.

    Separate from ``walkforward`` and ``optimize`` because a verdict is a
    second, far more expensive measurement over the same run — the nulls and
    the perturbations are dozens of extra backtests. Those commands record the
    metrics; this one attaches the grade.

    Both nulls are calibrated on a **flat** run's ``expectancy_r``, not on the
    walk-forward's, so the two are compared on one number over one span. See
    CLAUDE.md "Решения P15 этапы 3–5".
    """
    settings: Settings = ctx.obj
    cli_config = WalkForwardCliConfig.model_validate_json(config.read_text())
    spec = StrategySpec.model_validate_json(strategy.read_text())

    library = ExitLibrarySpec.model_validate_json(exit_library.read_text())
    preset = next((item for item in library.presets if item.id == spec.exit_ref), None)
    if preset is None:
        typer.echo(f"exit_ref {spec.exit_ref!r} not found in {exit_library}")
        raise typer.Exit(code=1)

    frame = ParquetStore(settings.data_dir).get(cli_config.symbol, cli_config.timeframe)
    # start/end are None exactly when the frame is empty; naming both keeps the
    # perturbation span below a (datetime, datetime) rather than an assertion.
    if frame.is_empty or frame.start is None or frame.end is None:
        typer.echo(f"No data stored for {cli_config.symbol} {cli_config.timeframe}.")
        raise typer.Exit(code=1)
    span = (frame.start, frame.end)

    key = StreamKey(cli_config.symbol, cli_config.timeframe)
    base = RunInputs(
        config=BacktestConfig(
            account_currency=cli_config.account_currency,
            starting_balance=cli_config.starting_balance,
            atr_period=cli_config.atr_period,
            atr_baseline_bars=cli_config.atr_baseline_bars,
        ),
        streams={key: frame},
        bindings=(StrategyBinding(spec=spec, exit_preset=preset, keys=(key,)),),
        instruments=load_instruments(settings.instruments_path),
        costs=CostConfig(run_seed=cli_config.run_seed),
        sizing=FixedFractional(risk_pct=cli_config.risk_pct),
        risk=_risk_config(settings, cli_config.prop_profile),
    )

    manifest_path = settings.runs_dir / "walkforward" / wf_id / WF_MANIFEST_FILE
    if not manifest_path.exists():
        typer.echo(f"no walk-forward {wf_id} under {settings.runs_dir}")
        raise typer.Exit(code=1)
    result = read_result(manifest_path, wf_id, settings.runs_dir)
    report = build_report(result, min_trades_per_fold=cli_config.min_trades_per_fold)

    objective = ExpectancyR()
    flat = base.run()
    typer.echo(f"flat run: {len(flat.trades)} trades, expectancy_r {objective.score(flat):.4f}")

    permutation = run_calibration(
        NullKind.PERMUTATION,
        base,
        objective=objective,
        real_result=flat,
        n=n_null,
        calibration_id=f"verdict-perm-{wf_id[:12]}",
        store_root=settings.runs_dir,
        finest=cli_config.timeframe,
        day_origin=base.config.day_origin,
    )
    typer.echo(f"permutation null: percentile {permutation.percentile}")

    robustness = run_all(
        base,
        trades=flat.trades,
        coverage=span,
        synthetic_iterations=n_null,
        seed=cli_config.run_seed,
    )
    typer.echo(f"synthetic null: percentile {robustness.synthetic.real_percentile}")

    monte_carlo = None
    if mc_iterations > 0:
        sample = monte_carlo_sample(
            result, risk_pct=cli_config.risk_pct, starting_equity=cli_config.starting_balance
        )
        monte_carlo = run_monte_carlo_all(
            sample, n_iterations=mc_iterations, seed=cli_config.run_seed
        )
        basis = monte_carlo.verdict_basis
        typer.echo(
            f"monte carlo: block permutation over {basis.n_trades} trades in "
            f"{len(basis.fold_sizes)} fold(s), observed max drawdown at percentile "
            f"{basis.max_drawdown.observed_percentile}"
        )
        (manifest_path.parent / "monte_carlo.json").write_text(
            json.dumps(monte_carlo.to_dict(), indent=2) + "\n", encoding="utf-8"
        )

    verdict = build_verdict(
        report,
        monte_carlo=monte_carlo,
        robustness=robustness,
        permutation_percentile=permutation.percentile,
    )
    verdict_path = manifest_path.parent / "verdict.json"
    verdict_path.write_text(json.dumps(verdict.to_dict(), indent=2) + "\n", encoding="utf-8")

    typer.echo(f"\n{wf_id}: {verdict.verdict.value} (may_approve={verdict.may_approve})")
    for group in (verdict.sufficiency, verdict.overfit_checks, verdict.fragility_checks):
        for item in group:
            typer.echo(f"  {'PASS' if item.passed else 'FAIL'}  {item.name}: {item.detail}")
    typer.echo(f"  written to {verdict_path}")

    if record:
        link = ResultsLink(library_root)
        measured = {
            "permutation_percentile": permutation.percentile,
            "synthetic_percentile": robustness.synthetic.real_percentile,
            "null_iterations": float(n_null),
        }
        try:
            graded = link.grade(
                wf_id,
                verdict.verdict.value,
                metrics={name: value for name, value in measured.items() if value is not None},
            )
        except (KeyError, ValidationError) as error:
            typer.echo(f"  not graded: {error}")
            raise typer.Exit(code=1) from error
        typer.echo(
            f"  graded {graded.strategy_id} -> {graded.verdict} "
            f"(perm {graded.metric('permutation_percentile')}, "
            f"synth {graded.metric('synthetic_percentile')})"
        )


@report_app.command("index")
def report_index(
    ctx: typer.Context,
    out: Path = typer.Option(Path("reports/index.html"), "--out", help="Where to write the page."),
    strategy: str | None = typer.Option(
        None, "--strategy", help="Keep runs whose strategy id contains this."
    ),
    stream: str | None = typer.Option(
        None, "--stream", help="Keep runs whose stream contains this, e.g. 'EURUSD' or '@H4'."
    ),
    with_details: bool = typer.Option(
        False, "--with-details", help="Also render each run's own page next to the index."
    ),
    include_folds: bool = typer.Option(
        False,
        "--include-folds",
        help="Also list walk-forward folds. Off by default: folds are stored as ordinary "
        "runs and outnumber standalone runs many times over, and one fold's expectancy "
        "is not a result.",
    ),
    limit: int = typer.Option(
        40,
        "--limit",
        help="Refuse to render more detail pages than this. Each inlines plotly and runs "
        "several megabytes; runs/ also holds every walk-forward fold, so an unfiltered "
        "--with-details is gigabytes.",
    ),
) -> None:
    """One page listing every stored run, linking into the per-run pages.

    ``ts report run`` answers what one run did. This answers which of them is
    worth opening — the rows are walked off disk rather than curated, so the
    page cannot disagree with what is stored, and a run that will not read back
    is listed with the reason instead of quietly missing.

    Detail pages are heavy (plotly is inlined), so they are rendered only with
    ``--with-details``; without it the index links only to pages already sitting
    next to it.
    """
    settings: Settings = ctx.obj
    exclude = frozenset() if include_folds else fold_run_ids(settings.runs_dir)
    directories = filter_rows(
        run_directories(settings.runs_dir, exclude=exclude), strategy=strategy, stream=stream
    )
    if not directories:
        typer.echo(f"no runs under {settings.runs_dir} matching the filters")
        raise typer.Exit(code=1)

    detail_dir = out.parent
    if with_details and len(directories) > limit:
        typer.echo(
            f"{len(directories)} runs match, over the --limit of {limit}. Detail pages inline "
            "plotly and run to several megabytes each, and runs/ holds every walk-forward fold "
            "as a run of its own. Narrow with --strategy/--stream, or raise --limit knowingly."
        )
        raise typer.Exit(code=1)
    if with_details:
        detail_dir.mkdir(parents=True, exist_ok=True)
        for directory in directories:
            page = detail_dir / f"{directory.name}.html"
            if page.exists():
                continue
            try:
                stored = read_run(directory)
                symbol, _, timeframe = next(iter(stored.manifest.data), "@").partition("@")
                streams = _report_streams(
                    settings, symbol or None, Timeframe(timeframe) if timeframe else None
                )
                write(
                    build(source_from_run(directory, streams=streams, label=directory.name)),
                    page,
                )
            except (TradingSystemError, OSError, ValueError, KeyError) as error:
                typer.echo(f"  skipped {directory.name}: {str(error).splitlines()[0]}")
                continue

    index = build_index(directories, root=settings.runs_dir, detail_dir=detail_dir)
    write_index(index, out)
    linked = sum(1 for row in index.rows if row.detail_href)
    typer.echo(
        f"indexed {index.total} run(s) -> {out} ({linked} linked, {len(index.broken)} unreadable)"
    )


def _report_streams(
    settings: Settings, symbol: str | None, timeframe: Timeframe | None
) -> dict[StreamKey, OHLCVFrame]:
    """Bars for the excursion section, or nothing when the caller named no stream.

    Excursions need the price path, which a closed trade does not carry (P14
    stage 2). Absent bars produce a section saying so, never a silently missing
    one.
    """
    if symbol is None or timeframe is None:
        return {}
    frame = ParquetStore(settings.data_dir).get(symbol, timeframe)
    return {} if frame.is_empty else {StreamKey(symbol, timeframe): frame}


def _stored_verdict(directory: Path) -> StrategyVerdict | None:
    """A walk-forward's verdict, if ``validate verdict`` has been run over it."""
    path = directory / "verdict.json"
    if not path.exists():
        return None
    return StrategyVerdict.from_dict(json.loads(path.read_text()))


@report_app.command("run")
def report_run(
    ctx: typer.Context,
    run_id: str = typer.Argument(..., help="Run id under the runs directory."),
    out: Path | None = typer.Option(None, "--out", help="Where to write the page."),
    symbol: str | None = typer.Option(None, "--symbol", help="Symbol whose bars to load."),
    timeframe: Timeframe | None = typer.Option(None, "--timeframe", help="Its timeframe."),
    metrics_out: Path | None = typer.Option(None, "--metrics", help="Also export metrics JSON."),
) -> None:
    """Render one stored backtest as a self-contained HTML page."""
    settings: Settings = ctx.obj
    directory = settings.runs_dir / run_id
    if not directory.exists():
        typer.echo(f"no run {run_id} under {settings.runs_dir}")
        raise typer.Exit(code=1)

    source = source_from_run(
        directory, streams=_report_streams(settings, symbol, timeframe), label=run_id
    )
    rendered = build(source)
    destination = out or directory / "report.html"
    write(rendered, destination)
    typer.echo(f"{run_id}: {len(source.trades)} trades → {destination}")
    if metrics_out is not None:
        export_metrics(rendered, metrics_out)
        typer.echo(f"  metrics → {metrics_out}")


@report_app.command("walkforward")
def report_walkforward(
    ctx: typer.Context,
    wf_id: str = typer.Argument(..., help="Walk-forward id."),
    min_trades_per_fold: int = typer.Option(
        10, "--min-trades-per-fold", help="Below this, a fold is flagged insufficient."
    ),
    out: Path | None = typer.Option(None, "--out", help="Where to write the page."),
    symbol: str | None = typer.Option(None, "--symbol", help="Symbol whose bars to load."),
    timeframe: Timeframe | None = typer.Option(None, "--timeframe", help="Its timeframe."),
    metrics_out: Path | None = typer.Option(None, "--metrics", help="Also export metrics JSON."),
    cost_sensitivity_path: Path | None = typer.Option(
        None,
        "--cost-sensitivity",
        help="JSON written by an offline cost-sensitivity sweep, to embed as a table.",
    ),
) -> None:
    """Render a finished walk-forward: stitched out-of-sample curve, folds, verdict.

    The page presents the walk-forward as a **procedure**, not as one
    strategy's account — the axis is a multiple of the starting point and every
    fold seam is labelled with the parameters that segment traded on.
    """
    settings: Settings = ctx.obj
    directory = settings.runs_dir / "walkforward" / wf_id
    if not (directory / WF_MANIFEST_FILE).exists():
        typer.echo(f"no walk-forward {wf_id} under {settings.runs_dir}")
        raise typer.Exit(code=1)

    source = source_from_walkforward(
        directory,
        store_root=settings.runs_dir,
        min_trades_per_fold=min_trades_per_fold,
        streams=_report_streams(settings, symbol, timeframe),
        label=wf_id,
        verdict=_stored_verdict(directory),
        selections=fold_selections_from_disk(directory),
        search=search_summary_from_disk(directory),
        cost_sensitivity=cost_sensitivity_from_disk(cost_sensitivity_path)
        if cost_sensitivity_path is not None
        else None,
    )
    rendered = build(source)
    destination = out or directory / "report.html"
    write(rendered, destination)
    typer.echo(
        f"{wf_id}: {len(source.folds)} folds, {len(source.trades)} pooled OOS trades "
        f"→ {destination}"
    )
    if metrics_out is not None:
        export_metrics(rendered, metrics_out)
        typer.echo(f"  metrics → {metrics_out}")


@report_app.command("compare")
def report_compare(
    ctx: typer.Context,
    ids: list[str] = typer.Argument(..., help="Walk-forward ids to compare."),
    labels: str | None = typer.Option(
        None, "--labels", help="Comma-separated display names, in the same order as the ids."
    ),
    min_trades_per_fold: int = typer.Option(10, "--min-trades-per-fold"),
    out: Path = typer.Option(Path("comparison.html"), "--out", help="Where to write the page."),
    metrics_out: Path | None = typer.Option(None, "--metrics", help="Also export metrics JSON."),
) -> None:
    """Put several walk-forwards on one page: curves, metrics and verdicts side by side.

    Curves are normalised to their own starting points. They are different
    accounts over different periods, so only their shapes compare.
    """
    settings: Settings = ctx.obj
    names = labels.split(",") if labels else []
    if names and len(names) != len(ids):
        typer.echo(f"--labels has {len(names)} names for {len(ids)} ids")
        raise typer.Exit(code=1)

    sources = []
    for index, wf_id in enumerate(ids):
        directory = settings.runs_dir / "walkforward" / wf_id
        if not (directory / WF_MANIFEST_FILE).exists():
            typer.echo(f"no walk-forward {wf_id} under {settings.runs_dir}")
            raise typer.Exit(code=1)
        sources.append(
            source_from_walkforward(
                directory,
                store_root=settings.runs_dir,
                min_trades_per_fold=min_trades_per_fold,
                label=names[index] if names else wf_id,
                verdict=_stored_verdict(directory),
                selections=fold_selections_from_disk(directory),
            )
        )

    rendered = build_comparison(sources)
    write(rendered, out)
    typer.echo(f"compared {len(sources)} runs → {out}")
    for source in sources:
        grade = source.verdict.verdict.value if source.verdict else "not graded"
        typer.echo(f"  {source.label}: {len(source.trades)} trades, {grade}")
    if metrics_out is not None:
        export_metrics(rendered, metrics_out)
        typer.echo(f"  metrics → {metrics_out}")


@prop_app.command("simulate")
def prop_simulate(
    ctx: typer.Context,
    wf_id: str = typer.Argument(..., help="Walk-forward whose out-of-sample trades to replay."),
    config: Path = typer.Option(..., "--config", help="The same run config the walk-forward used."),
    rules: list[str] = typer.Option(
        [], "--rules", help="Rule sets to apply. Every shipped one when omitted."
    ),
    iterations: int = typer.Option(
        REPORT_ITERATIONS, "--iterations", help="Simulated attempts per rule set."
    ),
    seed: int = typer.Option(0, "--seed", help="Permutation seed, shared across rule sets."),
    rules_path: Path = typer.Option(
        DEFAULT_RULES_PATH, "--rules-file", help="Prop rule sets YAML."
    ),
) -> None:
    """Replay a finished walk-forward's trades through prop-firm rulebooks.

    Reports the odds of passing each account and of losing it, from a block
    permutation of the out-of-sample trades — the same resampling unit P15
    uses, shuffling within a fold and preserving fold order.
    """
    settings: Settings = ctx.obj
    cli_config = WalkForwardCliConfig.model_validate_json(config.read_text())

    manifest_path = settings.runs_dir / "walkforward" / wf_id / WF_MANIFEST_FILE
    if not manifest_path.exists():
        typer.echo(f"no walk-forward {wf_id} under {settings.runs_dir}")
        raise typer.Exit(code=1)

    library = load_prop_rules(rules_path)
    wanted = list(rules) if rules else list(library.names)
    try:
        chosen = [library.get(name) for name in wanted]
    except ValidationError as error:
        typer.echo(str(error))
        raise typer.Exit(code=1) from error

    result = read_result(manifest_path, wf_id, settings.runs_dir)
    sample = sample_from_walkforward(result, risk_pct=cli_config.risk_pct)
    typer.echo(
        f"{wf_id}: {sample.n_trades} out-of-sample trades over {len(sample.folds)} folds, "
        f"risked at {cli_config.risk_pct:.2%} of equity"
    )

    run_origin = BacktestConfig(
        account_currency=cli_config.account_currency,
        starting_balance=cli_config.starting_balance,
    ).day_origin
    for item in chosen:
        divergence = day_origin_divergence(item, run_origin)
        simulation = simulate(sample, item, iterations=iterations, seed=seed)
        typer.echo("")
        typer.echo(f"  {simulation.summary()}")
        typer.echo(
            f"    drawdown median {simulation.drawdown_median:.1%}, "
            f"p95 {simulation.drawdown_p95:.1%}, worst {simulation.drawdown_worst:.1%}"
        )
        days = simulation.mean_calendar_days_to_pass
        typer.echo(
            f"    mean trading days {simulation.mean_trading_days:.1f}, "
            + (f"mean calendar days to pass {days:.0f}" if days is not None else "never passed")
        )
        share = simulation.observed_day_share
        typer.echo(
            "    best single day, as it actually happened: "
            + (f"{share:.1%} of total profit" if share is not None else "no profit to share")
        )
        typer.echo(
            f"    trades blocked by the daily limit, median {simulation.blocked_trades_median:.0f}"
        )
        if divergence is not None:
            typer.echo(f"    NOTE  {divergence}")


@app.command()
def ui() -> None:
    """Launch UI."""
    typer.echo("UI")


if __name__ == "__main__":
    app()
