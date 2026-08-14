"""CLI wiring, including a full CSV-to-store-to-resample round trip."""

import json
from pathlib import Path

import pytest
import typer
from jsonschema import Draft202012Validator
from typer.testing import CliRunner

from trading_system.cli import ObjectiveName, _build_objective, app
from trading_system.prop.objective import PropObjective
from trading_system.prop.rules import DEFAULT_RULES_PATH
from trading_system.strategies import schema as strategy_schema_module
from trading_system.validation.objective import SortinoTimesSqrtTrades
from trading_system.validation.optimization import AxisTarget, ParameterAxis, SearchSpace
from trading_system.validation.report import VerdictThresholds

REPO_ROOT = Path(__file__).resolve().parents[1]
PROP_RULES = REPO_ROOT / DEFAULT_RULES_PATH

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


@pytest.mark.parametrize(("command", "expected"), [("backtest", "Backtest"), ("ui", "UI")])
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


class TestTheObjectiveFlagHasSomethingBehindEveryValue:
    """A flag value with no objective behind it would be an option that does nothing."""

    def _space(self, *, sizing: bool) -> SearchSpace:
        axes = [ParameterAxis(name="p", paths=("/risk_profile/base_quality",), values=(0.4, 0.5))]
        if sizing:
            axes.append(
                ParameterAxis(
                    name="risk",
                    target=AxisTarget.RUN,
                    paths=("/sizing/risk_pct",),
                    values=(0.005, 0.01),
                )
            )
        return SearchSpace(axes=tuple(axes))

    def test_the_default_objective_needs_no_extra_configuration(self) -> None:
        built = _build_objective(
            ObjectiveName.SORTINO_SQRT_N,
            rules_path=PROP_RULES,
            space=self._space(sizing=False),
            prop_rules=None,
            risk_pct=0.01,
            null_percentile=None,
        )
        assert isinstance(built, SortinoTimesSqrtTrades)

    def test_prop_without_a_rule_set_is_refused(self) -> None:
        with pytest.raises(typer.BadParameter, match="--prop-rules"):
            _build_objective(
                ObjectiveName.PROP,
                rules_path=PROP_RULES,
                space=self._space(sizing=False),
                prop_rules=None,
                risk_pct=0.01,
                null_percentile=None,
            )

    def test_prop_over_a_space_that_varies_sizing_is_refused(self) -> None:
        with pytest.raises(typer.BadParameter, match="varies sizing"):
            _build_objective(
                ObjectiveName.PROP,
                rules_path=PROP_RULES,
                space=self._space(sizing=True),
                prop_rules="ftmo_normal",
                risk_pct=0.01,
                null_percentile=None,
            )

    def test_prop_refuses_an_unmeasured_edge_rather_than_ranking_an_all_ties_surface(
        self,
    ) -> None:
        """The gate is per spec, so a failing spec scores every trial identically.

        Left to run, that flat surface still produces a selection: zero
        dispersion means the plateau swallows the space and the centroid
        tie-breaks. Measured before this refusal existed: 96 trials all at
        -1.95, eight folds of ordinary-looking selections, out-of-sample numbers
        reported for parameters nothing chose.
        """
        with pytest.raises(typer.BadParameter, match="needs an edge percentile"):
            _build_objective(
                ObjectiveName.PROP,
                rules_path=PROP_RULES,
                space=self._space(sizing=False),
                prop_rules="ftmo_normal",
                risk_pct=0.01,
                null_percentile=None,
            )

    def test_prop_refuses_a_percentile_below_its_own_threshold(self) -> None:
        with pytest.raises(typer.BadParameter, match="is 60.0"):
            _build_objective(
                ObjectiveName.PROP,
                rules_path=PROP_RULES,
                space=self._space(sizing=False),
                prop_rules="ftmo_normal",
                risk_pct=0.01,
                null_percentile=60.0,
            )

    def test_prop_builds_once_the_edge_clears_the_same_threshold_the_verdict_uses(self) -> None:
        threshold = VerdictThresholds().null_percentile
        built = _build_objective(
            ObjectiveName.PROP,
            rules_path=PROP_RULES,
            space=self._space(sizing=False),
            prop_rules="ftmo_normal",
            risk_pct=0.01,
            null_percentile=threshold,
        )
        assert isinstance(built, PropObjective)
        assert built.edge_threshold == threshold, (
            "the objective and the verdict must not hold two opinions on what an edge is"
        )

    def test_the_optimize_command_advertises_the_flag(self) -> None:
        result = runner.invoke(app, ["validate", "optimize", "--help"])
        assert result.exit_code == 0
        assert "--objective" in result.stdout
