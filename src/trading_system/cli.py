"""Command-line entry point.

The root callback builds the single :class:`Settings` instance for the process
and hands it to subcommands through the typer context, keeping configuration an
explicit dependency rather than a module-level global.
"""

import typer

from trading_system.core.config import Settings
from trading_system.core.logging import setup_logging

app = typer.Typer(help="Modular trading system.")


@app.callback()
def main(ctx: typer.Context) -> None:
    """Load settings and configure logging before any subcommand runs."""
    settings = Settings()
    setup_logging(level=settings.log_level, log_file=settings.log_file)
    ctx.obj = settings


@app.command()
def data() -> None:
    """Data management."""
    typer.echo("Data commands")


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
