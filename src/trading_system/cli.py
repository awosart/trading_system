"""Command-line entry point.

The root callback builds the single :class:`Settings` instance for the process
and hands it to subcommands through the typer context, keeping configuration an
explicit dependency rather than a module-level global.
"""

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from enum import StrEnum
from pathlib import Path

import typer
from pydantic import BaseModel, ConfigDict, Field

from trading_system.backtest.clock import StreamKey
from trading_system.backtest.config import BacktestConfig
from trading_system.backtest.orchestrator import StrategyBinding
from trading_system.backtest.spec import RunInputs
from trading_system.core.config import Settings
from trading_system.core.exceptions import ValidationError
from trading_system.core.instruments import InstrumentRegistry, load_instruments
from trading_system.core.logging import get_logger, setup_logging
from trading_system.core.types import Timeframe
from trading_system.data.providers.csv_provider import CSVProvider, CSVSchema
from trading_system.data.providers.dukascopy_provider import DukascopyProvider
from trading_system.data.quality import QualityConfig, check_frame
from trading_system.data.resample import FX_DAY_ORIGIN, resample
from trading_system.data.sessions import AssetClass, TradingCalendar
from trading_system.data.store import ParquetStore
from trading_system.execution.config import CostConfig
from trading_system.exit.library import DEFAULT_LIBRARY_PATH, ExitLibrarySpec, known_exit_ids
from trading_system.risk.sizing.methods import FixedFractional
from trading_system.strategies.schema import SCHEMA_JSON_PATH, StrategySpec, strategy_json_schema
from trading_system.strategies.validator import Severity, validate_paths
from trading_system.validation.calibration import (
    POSITION_COUNT_DIVERGENCE_THRESHOLD,
    NullKind,
    run_calibration,
)
from trading_system.validation.nulls.random_entry import (
    EntryTraceProfile,
    build_entry_trace_profile,
    real_signals,
)
from trading_system.validation.objective import SortinoTimesSqrtTrades
from trading_system.validation.optimization import (
    GridSearch,
    OptunaSearch,
    ParameterSearch,
    RandomSearch,
    SearchSpace,
    read_fold_selection,
)
from trading_system.validation.report import build_report, write_report
from trading_system.validation.splitting import PurgedKFold, WalkForwardMode, WalkForwardSplitter
from trading_system.validation.walkforward import (
    IdentitySelector,
    OptimizingSelector,
    WalkForwardRunner,
)

app = typer.Typer(help="Modular trading system.")
data_app = typer.Typer(help="Market data management.")
strategy_app = typer.Typer(help="Strategy spec management.")
validate_app = typer.Typer(help="Out-of-sample validation.")
app.add_typer(data_app, name="data")
app.add_typer(strategy_app, name="strategy")
app.add_typer(validate_app, name="validate")

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
) -> None:
    """Import bars from a CSV file into the local store."""
    provider = CSVProvider(
        path,
        CSVSchema(
            source_tz=source_tz,
            date_column=date_column,
            time_column=time_column,
            timestamp_format=timestamp_format,
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
    """
    try:
        ids = known_exit_ids(exit_library)
    except ValidationError as error:
        typer.echo(f"exit library {exit_library}: {error}")
        raise typer.Exit(code=1) from error

    results = validate_paths(paths, known_exit_ids=ids)
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


@strategy_app.command("schema-export")
def strategy_schema_export(
    out: Path = typer.Option(SCHEMA_JSON_PATH, "--out", help="Where to write the JSON Schema."),
) -> None:
    """(Re-)generate the JSON Schema editors use to validate strategy files."""
    out.write_text(json.dumps(strategy_json_schema(), indent=2) + "\n", encoding="utf-8")
    typer.echo(f"Wrote schema to {out}")


@app.command()
def backtest() -> None:
    """Run backtest."""
    typer.echo("Backtest")


class WalkForwardCliConfig(BaseModel):
    """Everything ``ts validate walkforward`` needs beyond the fold geometry.

    The fold geometry (``--mode``, ``--is``, ``--oos``, ``--step``,
    ``--embargo``, ``--warmup``) is on the command line, per the CLAUDE.md
    spec for this command; everything else a run needs to be built at all —
    which instrument and timeframe, the account, the sizing — has no natural
    single-flag form and lives in this file instead.

    Attributes:
        symbol: Instrument to trade, read from the local store.
        timeframe: Bar size to trade.
        account_currency: Denomination of the account.
        starting_balance: Opening balance of every fold's own run.
        risk_pct: Fraction of equity risked per trade
            (:class:`~trading_system.risk.sizing.methods.FixedFractional`).
            No optimiser exists yet to choose this per fold, so it is fixed
            for the whole walk-forward.
        run_seed: Seed for the per-fill random streams. One for the whole
            walk-forward — see CLAUDE.md's "Решения P15 этап 1" on why.
        atr_period: ATR period the cost model's volatility ratio is built
            from.
        atr_baseline_bars: Rolling window the ATR is divided by.
        max_drain_bars: Bars past each OOS window's ``trade_end`` a position
            already open may still be managed on.
        min_trades_per_fold: Below this many OOS trades, a fold is flagged
            insufficient in the report.
        parallel_threshold_seconds: Below this many seconds for the first run
            of a batch, the rest of that batch runs sequentially rather than
            paying a process pool's spawn cost.
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
    max_drain_bars: int = Field(gt=0)
    min_trades_per_fold: int = Field(ge=0)
    parallel_threshold_seconds: float = Field(default=2.0, gt=0)


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
) -> None:
    """Walk history in folds: IS run, OOS run, and a report of every window's boundary.

    No parameter selection happens here — every fold's OOS run uses the same
    parameters the strategy file names. See CLAUDE.md P15 stage 1: this stage
    is the harness, not the optimiser.
    """
    settings: Settings = ctx.obj
    cli_config = WalkForwardCliConfig.model_validate_json(config.read_text())

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
    )
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

    if method is SearchMethod.GRID and trial_budget < search_space.feasible_size():
        typer.echo(
            f"--method grid over this space needs --trial-budget of at least "
            f"{search_space.feasible_size()}; got {trial_budget}."
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


@app.command()
def ui() -> None:
    """Launch UI."""
    typer.echo("UI")


if __name__ == "__main__":
    app()
