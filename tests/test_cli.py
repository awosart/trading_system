"""CLI wiring: every command runs and settings reach the context."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from trading_system.cli import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Prevent the CLI callback from picking up a developer's real .env."""
    monkeypatch.chdir(tmp_path)


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("data", "Data commands"),
        ("backtest", "Backtest"),
        ("validate", "Validate"),
        ("ui", "UI"),
    ],
)
def test_commands_run(command: str, expected: str) -> None:
    result = runner.invoke(app, [command])
    assert result.exit_code == 0
    assert expected in result.stdout


def test_help_lists_all_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("data", "backtest", "validate", "ui"):
        assert command in result.stdout
