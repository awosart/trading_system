"""CLI wiring, including a full CSV-to-store-to-resample round trip."""

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from typer.testing import CliRunner

from trading_system.cli import app
from trading_system.strategies import schema as strategy_schema_module

runner = CliRunner()

EXAMPLES_DIR = Path(strategy_schema_module.__file__).parent / "examples"
EXAMPLE_PATHS = sorted(str(path) for path in EXAMPLES_DIR.glob("*.json"))

#: Examples that set ``htf_filter_tf`` as a stated intent the Entry Engine
#: cannot yet act on — a legitimate WARNING, not a defect. See
#: ``check_htf_filter_unused`` in ``strategies/validator.py``.
HTF_FILTER_EXAMPLES = {
    path for path in EXAMPLE_PATHS if "htf_filter_tf" in Path(path).read_text(encoding="utf-8")
}

MINUTE_CSV_HEADER = "timestamp,open,high,low,close,volume\n"


def write_minute_csv(path: Path, bars: int = 300) -> None:
    """Write a synthetic 1m CSV starting 2020-01-01 00:00."""
    from datetime import UTC, datetime, timedelta

    start = datetime(2020, 1, 1, tzinfo=UTC)
    lines = [MINUTE_CSV_HEADER]
    for index in range(bars):
        moment = start + timedelta(minutes=index)
        price = 1.1 + index * 0.0001
        lines.append(
            f"{moment:%Y-%m-%dT%H:%M:%S},{price:.5f},{price + 0.0002:.5f},"
            f"{price - 0.0002:.5f},{price + 0.0001:.5f},{100 + index}\n"
        )
    path.write_text("".join(lines), encoding="utf-8")


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Point the CLI at a scratch data directory and away from any real .env."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TS_DATA_DIR", str(tmp_path / "store"))


@pytest.mark.parametrize(
    ("command", "expected"),
    [("backtest", "Backtest"), ("validate", "Validate"), ("ui", "UI")],
)
def test_placeholder_commands_run(command: str, expected: str) -> None:
    result = runner.invoke(app, [command])
    assert result.exit_code == 0
    assert expected in result.stdout


def test_help_lists_command_groups() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("data", "strategy", "backtest", "validate", "ui"):
        assert command in result.stdout


def test_data_help_lists_subcommands() -> None:
    result = runner.invoke(app, ["data", "--help"])
    assert result.exit_code == 0
    for command in ("import", "coverage", "quality", "resample"):
        assert command in result.stdout


def test_strategy_help_lists_subcommands() -> None:
    result = runner.invoke(app, ["strategy", "--help"])
    assert result.exit_code == 0
    for command in ("validate", "schema-export"):
        assert command in result.stdout


class TestStrategyValidate:
    def test_worked_examples_all_pass(self) -> None:
        """Passes means exit code 0. A WARNING is not a failure to validate.

        Two examples set ``htf_filter_tf``, which prints a warning line instead
        of the bare ``OK`` — see ``HTF_FILTER_EXAMPLES``.
        """
        result = runner.invoke(app, ["strategy", "validate", *EXAMPLE_PATHS])
        assert result.exit_code == 0, result.stdout
        assert "[ERROR]" not in result.stdout
        for path in EXAMPLE_PATHS:
            if path in HTF_FILTER_EXAMPLES:
                assert f"{path}: [WARNING] htf_filter_unused" in result.stdout
            else:
                assert f"{path}: OK" in result.stdout

    def test_syntactically_broken_file_exits_nonzero(self, tmp_path: Path) -> None:
        broken = tmp_path / "broken.json"
        broken.write_text('{"id": "Not A Slug"}', encoding="utf-8")
        result = runner.invoke(app, ["strategy", "validate", str(broken)])
        assert result.exit_code == 1
        assert "[ERROR]" in result.stdout

    def test_missing_file_exits_nonzero(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["strategy", "validate", str(tmp_path / "nope.json")])
        assert result.exit_code == 1
        assert "[ERROR]" in result.stdout
        assert "read_error" in result.stdout

    def test_unknown_exit_ref_exits_nonzero(self, tmp_path: Path) -> None:
        # P07 stage 3: exit_ref is now checked against the real exit library
        # by default, not skipped.
        spec = json.loads(Path(EXAMPLE_PATHS[0]).read_text(encoding="utf-8"))
        spec["exit_ref"] = "no_such_preset"
        broken = tmp_path / "bad_exit_ref.json"
        broken.write_text(json.dumps(spec), encoding="utf-8")

        result = runner.invoke(app, ["strategy", "validate", str(broken)])

        assert result.exit_code == 1
        assert "unknown_exit_ref" in result.stdout
        assert "no_such_preset" in result.stdout

    def test_a_real_preset_id_passes(self, tmp_path: Path) -> None:
        """Passing means exit code 0 and no ERROR line, not a bare OK.

        The base spec is ``EXAMPLE_PATHS[0]`` (``ema_pullback.json``), which
        sets ``htf_filter_tf`` and so prints a WARNING line rather than OK —
        the point under test here is ``exit_ref``, not that field.
        """
        spec = json.loads(Path(EXAMPLE_PATHS[0]).read_text(encoding="utf-8"))
        spec["exit_ref"] = "conservative_2r"
        fixed = tmp_path / "good_exit_ref.json"
        fixed.write_text(json.dumps(spec), encoding="utf-8")

        result = runner.invoke(app, ["strategy", "validate", str(fixed)])

        assert result.exit_code == 0, result.stdout
        assert "[ERROR]" not in result.stdout

    def test_a_broken_exit_library_is_reported_and_exits_nonzero(self, tmp_path: Path) -> None:
        broken_library = tmp_path / "library.json"
        broken_library.write_text("{not valid json", encoding="utf-8")

        result = runner.invoke(
            app,
            [
                "strategy",
                "validate",
                "--exit-library",
                str(broken_library),
                EXAMPLE_PATHS[0],
            ],
        )

        assert result.exit_code == 1
        assert "exit library" in result.stdout


class TestStrategySchemaExport:
    def test_writes_a_valid_draft_2020_12_schema(self, tmp_path: Path) -> None:
        out = tmp_path / "schema.json"
        result = runner.invoke(app, ["strategy", "schema-export", "--out", str(out)])
        assert result.exit_code == 0, result.stdout
        schema = json.loads(out.read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)


class TestEndToEnd:
    """CSV on disk to queryable, resampled bars in the store."""

    def test_import_then_coverage_then_resample(self, tmp_path: Path) -> None:
        source = tmp_path / "EURUSD.csv"
        write_minute_csv(source, bars=300)

        imported = runner.invoke(
            app,
            ["data", "import", "--symbol", "EURUSD", "--tf", "M1", "--path", str(source)],
        )
        assert imported.exit_code == 0, imported.stdout
        assert "Imported 300 bars" in imported.stdout

        coverage = runner.invoke(app, ["data", "coverage"])
        assert coverage.exit_code == 0
        assert "EURUSD M1: 300 bars" in coverage.stdout
        assert "2020-01-01 00:00" in coverage.stdout

        quality = runner.invoke(app, ["data", "quality", "--symbol", "EURUSD", "--tf", "M1"])
        assert quality.exit_code == 0
        assert "300 bars checked" in quality.stdout

        resampled = runner.invoke(
            app,
            ["data", "resample", "--symbol", "EURUSD", "--from-tf", "M1", "--to-tf", "H1"],
        )
        assert resampled.exit_code == 0
        # 300 minutes covers five whole hours; the sixth is partial and withheld.
        assert "into 5 H1 bars" in resampled.stdout

        after = runner.invoke(app, ["data", "coverage", "--symbol", "EURUSD", "--tf", "H1"])
        assert after.exit_code == 0
        assert "EURUSD H1: 5 bars" in after.stdout

    def test_reimport_is_idempotent(self, tmp_path: Path) -> None:
        source = tmp_path / "EURUSD.csv"
        write_minute_csv(source, bars=120)
        arguments = [
            "data",
            "import",
            "--symbol",
            "EURUSD",
            "--tf",
            "M1",
            "--path",
            str(source),
        ]
        runner.invoke(app, arguments)
        runner.invoke(app, arguments)

        coverage = runner.invoke(app, ["data", "coverage"])
        assert "EURUSD M1: 120 bars" in coverage.stdout

    def test_quality_exits_nonzero_on_broken_data(self, tmp_path: Path) -> None:
        source = tmp_path / "EURUSD.csv"
        source.write_text(
            MINUTE_CSV_HEADER
            + "2020-01-01T00:00:00,1.1,0.5,1.5,1.1,100\n"  # high below low
            + "2020-01-01T00:01:00,1.1,1.2,1.0,1.1,100\n",
            encoding="utf-8",
        )
        runner.invoke(
            app, ["data", "import", "--symbol", "EURUSD", "--tf", "M1", "--path", str(source)]
        )
        result = runner.invoke(app, ["data", "quality", "--symbol", "EURUSD", "--tf", "M1"])
        assert result.exit_code == 1
        assert "high_below_low" in result.stdout
        assert "[ERROR]" in result.stdout


def test_coverage_on_empty_store() -> None:
    result = runner.invoke(app, ["data", "coverage"])
    assert result.exit_code == 0
    assert "Store is empty" in result.stdout


def test_quality_without_data_exits_nonzero() -> None:
    result = runner.invoke(app, ["data", "quality", "--symbol", "EURUSD", "--tf", "M1"])
    assert result.exit_code == 1
    assert "No data stored" in result.stdout
