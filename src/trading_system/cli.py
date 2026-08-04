"""Command-line entry point.

The root callback builds the single :class:`Settings` instance for the process
and hands it to subcommands through the typer context, keeping configuration an
explicit dependency rather than a module-level global.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import typer

from trading_system.core.config import Settings
from trading_system.core.logging import get_logger, setup_logging
from trading_system.core.types import Timeframe
from trading_system.data.providers.csv_provider import CSVProvider, CSVSchema
from trading_system.data.quality import check_frame
from trading_system.data.resample import FX_DAY_ORIGIN, resample
from trading_system.data.store import ParquetStore
from trading_system.strategies.schema import SCHEMA_JSON_PATH, strategy_json_schema
from trading_system.strategies.validator import Severity, validate_paths

app = typer.Typer(help="Modular trading system.")
data_app = typer.Typer(help="Market data management.")
strategy_app = typer.Typer(help="Strategy spec management.")
app.add_typer(data_app, name="data")
app.add_typer(strategy_app, name="strategy")

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
) -> None:
    """Run data-quality detectors over stored bars."""
    frame = _store(ctx).get(symbol, timeframe)
    if frame.is_empty:
        typer.echo(f"No data stored for {symbol} {timeframe}.")
        raise typer.Exit(code=1)
    report = check_frame(frame)
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
) -> None:
    """Validate strategy specs: schema, feature refs, timeframe order, id uniqueness.

    ``exit_ref`` existence is not checked yet — there is no Exit DB to consult
    until the Exit Engine module exists.
    """
    results = validate_paths(paths)
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


@app.command()
def validate() -> None:
    """Validate strategies."""
    typer.echo("Validate")


@app.command()
def ui() -> None:
    """Launch UI."""
    typer.echo("UI")


if __name__ == "__main__":
    app()
