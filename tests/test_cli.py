"""CLI wiring, including a full CSV-to-store-to-resample round trip."""

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from typer.testing import CliRunner

from trading_system import cli as cli_module
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
    """Point the CLI at a scratch data directory and away from any real .env.

    Logging setup is stubbed out for the same reason: the root callback
    configures structlog process-wide with ``cache_logger_on_first_use``, so a
    module logger first used while a command runs stays bound to that
    configuration for the rest of the session, and
    ``structlog.testing.capture_logs`` elsewhere in the suite then intercepts
    nothing — an assertion about a warning that really was emitted fails, in a
    file that never mentioned the CLI. What ``setup_logging`` installs is
    ``tests/test_logging.py``'s subject; these tests are about command wiring.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TS_DATA_DIR", str(tmp_path / "store"))
    monkeypatch.setattr(cli_module, "setup_logging", lambda **_: None)


@pytest.mark.parametrize(("command", "expected"), [("ui", "UI")])
def test_placeholder_commands_run(command: str, expected: str) -> None:
    result = runner.invoke(app, [command])
    assert result.exit_code == 0
    assert expected in result.stdout


def test_backtest_is_no_longer_a_placeholder() -> None:
    """It used to echo "Backtest" and exit 0, having run nothing.

    A command that reports success without doing its job is worse than a
    missing one: the missing one sends you to the API, this one sends you
    looking for the output.
    """
    result = runner.invoke(app, ["backtest"])
    assert result.exit_code != 0
    assert "Backtest" not in result.stdout


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


def test_validate_help_lists_subcommands() -> None:
    result = runner.invoke(app, ["validate", "--help"])
    assert result.exit_code == 0
    assert "walkforward" in result.stdout


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

    def test_library_bookkeeping_files_are_skipped_and_the_count_is_printed(
        self, tmp_path: Path
    ) -> None:
        # "strategies/library/*/*.json" is the natural glob to type and it
        # matches {id}.meta.json too. Validating those as specs would report
        # dozens of meaningless schema errors; skipping them silently would
        # hide that files the user named were not checked.
        from trading_system.strategies.repository import META_SUFFIX, StrategyRepository
        from trading_system.strategies.schema import StrategySpec

        repository = StrategyRepository(tmp_path)
        spec = StrategySpec.model_validate_json(Path(EXAMPLE_PATHS[0]).read_text(encoding="utf-8"))
        record = repository.add(spec, name="x", author="ts")
        meta = record.path.with_name(f"{record.path.stem}{META_SUFFIX}")

        result = runner.invoke(app, ["strategy", "validate", str(record.path), str(meta)])

        assert result.exit_code == 0, result.stdout
        assert "[ERROR]" not in result.stdout
        assert "Skipped 1 bookkeeping file" in result.stdout

    def test_a_run_of_only_bookkeeping_files_validates_nothing_and_says_so(
        self, tmp_path: Path
    ) -> None:
        from trading_system.strategies.repository import META_SUFFIX

        meta = tmp_path / f"whatever{META_SUFFIX}"
        meta.write_text("{}", encoding="utf-8")
        result = runner.invoke(app, ["strategy", "validate", str(meta)])
        assert result.exit_code == 0
        assert "No strategy specs to validate" in result.stdout

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

    def test_a_tab_separated_export_with_an_unnamed_column_imports(self, tmp_path: Path) -> None:
        """The shape a broker export arrives in, end to end through the flags.

        Six names in the header, seven fields per row, tabs between them. Both
        flags are needed and neither is a default: without ``--sep`` the file
        parses as one column, and without ``--drop-unnamed-fields`` a row wider
        than its header is an error, because that is usually what a wrong
        ``--sep`` looks like.
        """
        source = tmp_path / "EURUSD.tsv"
        lines = ["Time\tOpen\tHigh\tLow\tClose\tVolume\n"]
        for index in range(120):
            price = 1.1 + index * 0.0001
            lines.append(
                f"2020-01-01 {index // 60:02d}:{index % 60:02d}:00\t{price:.5f}\t"
                f"{price + 0.0002:.5f}\t{price - 0.0002:.5f}\t{price + 0.0001:.5f}\t"
                f"{100 + index}\t3\n"
            )
        source.write_text("".join(lines), encoding="utf-8")
        arguments = [
            "data",
            "import",
            "--symbol",
            "EURUSD",
            "--tf",
            "M1",
            "--path",
            str(source),
            "--sep",
            r"\t",
            "--format",
            "%Y-%m-%d %H:%M:%S",
        ]

        refused = runner.invoke(app, arguments)
        assert refused.exit_code != 0

        imported = runner.invoke(app, [*arguments, "--drop-unnamed-fields"])
        assert imported.exit_code == 0, imported.stdout
        assert "Imported 120 bars" in imported.stdout

        coverage = runner.invoke(app, ["data", "coverage"])
        assert "EURUSD M1: 120 bars" in coverage.stdout

    def test_a_multi_character_separator_is_refused_at_the_flag(self, tmp_path: Path) -> None:
        """A reader takes one byte; a longer one must fail where it was typed."""
        source = tmp_path / "EURUSD.csv"
        write_minute_csv(source, bars=10)
        result = runner.invoke(
            app,
            [
                "data",
                "import",
                "--symbol",
                "EURUSD",
                "--tf",
                "M1",
                "--path",
                str(source),
                "--sep",
                "||",
            ],
        )
        assert result.exit_code != 0
        assert "single character" in result.output + result.stderr

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

    def test_calendar_option_suppresses_weekend_missing_bars(self, tmp_path: Path) -> None:
        from datetime import UTC, datetime, timedelta

        from trading_system.data.sessions import AssetClass, TradingCalendar

        calendar = TradingCalendar(asset_class=AssetClass.FX)
        start = datetime(2024, 1, 1, tzinfo=UTC)  # Monday
        hours = [start + timedelta(hours=i) for i in range(24 * 9)]  # spans one weekend
        open_hours = [hour for hour in hours if calendar.is_open(hour)]

        lines = [MINUTE_CSV_HEADER]
        for index, ts in enumerate(open_hours):
            price = 1.1 + index * 0.0001
            lines.append(
                f"{ts:%Y-%m-%dT%H:%M:%S},{price:.5f},{price + 0.0002:.5f},"
                f"{price - 0.0002:.5f},{price + 0.0001:.5f},{100 + index}\n"
            )
        source = tmp_path / "EURUSD.csv"
        source.write_text("".join(lines), encoding="utf-8")
        runner.invoke(
            app, ["data", "import", "--symbol", "EURUSD", "--tf", "H1", "--path", str(source)]
        )

        without_calendar = runner.invoke(
            app, ["data", "quality", "--symbol", "EURUSD", "--tf", "H1"]
        )
        assert "missing_bars" in without_calendar.stdout

        with_calendar = runner.invoke(
            app, ["data", "quality", "--symbol", "EURUSD", "--tf", "H1", "--calendar", "FX"]
        )
        assert "missing_bars" not in with_calendar.stdout


def test_coverage_on_empty_store() -> None:
    result = runner.invoke(app, ["data", "coverage"])
    assert result.exit_code == 0
    assert "Store is empty" in result.stdout


def test_quality_without_data_exits_nonzero() -> None:
    result = runner.invoke(app, ["data", "quality", "--symbol", "EURUSD", "--tf", "M1"])
    assert result.exit_code == 1
    assert "No data stored" in result.stdout


class TestFlatBacktest:
    """``ts backtest``: one strategy, one stream, one period, stored."""

    def _prepare(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
        """Fill the store and write the spec and config the command reads."""
        from tests.backtest.conftest import strategy
        from trading_system.core.types import Timeframe

        repository_root = Path(__file__).resolve().parent.parent
        monkeypatch.setenv("TS_INSTRUMENTS_PATH", str(repository_root / "configs/instruments.yaml"))
        monkeypatch.setenv(
            "TS_PROP_PROFILES_PATH", str(repository_root / "configs/prop_profiles.yaml")
        )
        monkeypatch.setenv("TS_RUNS_DIR", str(tmp_path / "runs"))

        source = tmp_path / "EURUSD.csv"
        write_minute_csv(source, bars=600)
        imported = runner.invoke(
            app,
            ["data", "import", "--symbol", "EURUSD", "--tf", "M1", "--path", str(source)],
        )
        assert imported.exit_code == 0, imported.stdout

        spec_path = tmp_path / "spec.json"
        spec_path.write_text(
            strategy(signal_tf=Timeframe.M1, invalidation=1.09).model_dump_json(),
            encoding="utf-8",
        )
        config_path = tmp_path / "run.json"
        config_path.write_text(
            json.dumps({"symbol": "EURUSD", "timeframe": "M1", "risk_pct": 0.01}),
            encoding="utf-8",
        )
        return spec_path, config_path

    def test_it_runs_and_stores_what_it_produced(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        spec_path, config_path = self._prepare(tmp_path, monkeypatch)
        result = runner.invoke(
            app,
            ["backtest", "--config", str(config_path), "--strategy", str(spec_path)],
        )
        assert result.exit_code == 0, result.stdout
        assert "trades:" in result.stdout
        stored = list((tmp_path / "runs").iterdir())
        assert len(stored) == 1
        assert (stored[0] / "curve.parquet").exists()

    def test_the_same_run_twice_lands_on_one_id(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The store's idempotence, reached through the command that fills it."""
        spec_path, config_path = self._prepare(tmp_path, monkeypatch)
        arguments = ["backtest", "--config", str(config_path), "--strategy", str(spec_path)]
        first = runner.invoke(app, arguments)
        second = runner.invoke(app, arguments)
        assert first.exit_code == 0 and second.exit_code == 0, second.stdout
        assert len(list((tmp_path / "runs").iterdir())) == 1

    def test_a_period_with_no_bars_is_refused_rather_than_run_empty(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        spec_path, config_path = self._prepare(tmp_path, monkeypatch)
        result = runner.invoke(
            app,
            [
                "backtest",
                "--config",
                str(config_path),
                "--strategy",
                str(spec_path),
                "--start",
                "2019-01-01",
                "--end",
                "2019-02-01",
            ],
        )
        assert result.exit_code != 0
        assert "No data stored" in result.stdout

    def test_a_walkforward_config_is_refused_by_name(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The two configs are not interchangeable, and say which field is extra.

        ``max_drain_bars`` and ``min_trades_per_fold`` describe fold geometry a
        flat run has none of. Accepting them silently would let a config be
        edited for one command and run by the other.
        """
        spec_path, config_path = self._prepare(tmp_path, monkeypatch)
        config_path.write_text(
            json.dumps(
                {
                    "symbol": "EURUSD",
                    "timeframe": "M1",
                    "risk_pct": 0.01,
                    "max_drain_bars": 200,
                    "min_trades_per_fold": 10,
                }
            ),
            encoding="utf-8",
        )
        result = runner.invoke(
            app,
            ["backtest", "--config", str(config_path), "--strategy", str(spec_path)],
        )
        assert result.exit_code != 0
        assert "max_drain_bars" in str(result.exception)
