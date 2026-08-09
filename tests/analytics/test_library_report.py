"""Library-wide report: not-measured, orphaned, verdict counts, correlation.

Orphan detection is proven against a real, small walk-forward — production
data currently has zero orphans (every recorded result still resolves), so the
only way to prove the mechanism works is to build a genuine one and delete a
directory out from under it.
"""

from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from tests.analytics.conftest import point
from tests.backtest.conftest import EURUSD_H4, ema_pullback, harness_inputs, swing_series
from trading_system.analytics.library_report import (
    MIN_SHARED_DAYS,
    NOTHING_TO_APPROVE,
    build_library_report,
    correlation_matrix,
    render,
)
from trading_system.backtest.orchestrator import StrategyBinding
from trading_system.backtest.reproducibility import write_run
from trading_system.backtest.spec import RunInputs
from trading_system.core.instruments import InstrumentRegistry
from trading_system.exit.library import ExitLibrarySpec
from trading_system.strategies.repository import StrategyRepository
from trading_system.strategies.results_link import (
    RUN_KIND_FLAT,
    RUN_KIND_WALKFORWARD,
    ResultsLink,
    build_record,
)
from trading_system.validation.report import Verdict
from trading_system.validation.splitting import WalkForwardMode, WalkForwardSplitter
from trading_system.validation.walkforward import IdentitySelector, WalkForwardRunner

LENGTH = 2200


def _base_inputs(
    registry: InstrumentRegistry, library: ExitLibrarySpec, strategy_id: str
) -> RunInputs:
    """A modest real ``ema_pullback``/``swing_series`` run, long enough for several folds."""
    spec = ema_pullback(strategy_id=strategy_id)
    preset = next(item for item in library.presets if item.id == spec.exit_ref)
    return harness_inputs(
        registry,
        streams={EURUSD_H4: swing_series(LENGTH)},
        bindings=[StrategyBinding(spec=spec, exit_preset=preset, keys=(EURUSD_H4,))],
    )


def _run_walkforward(base: RunInputs, store_root: Path) -> str:
    """A real, small walk-forward, stored under ``store_root``. Returns its wf_id."""
    splitter = WalkForwardSplitter(
        mode=WalkForwardMode.ROLLING,
        is_span=timedelta(days=90),
        oos_span=timedelta(days=25),
        step=timedelta(days=25),
        embargo=timedelta(days=2),
        warmup=timedelta(days=15),
    )
    runner = WalkForwardRunner(
        base=base,
        splitter=splitter,
        selector=IdentitySelector(base),
        store_root=store_root,
        max_drain_bars=30,
    )
    result = runner.run()
    assert len(result.folds) >= 2, "the fixture must actually produce more than one fold"
    return result.wf_id


def _ensure_registered(repository: StrategyRepository, spec: Any) -> None:
    """Add ``spec`` to the library unless it is already there."""
    try:
        repository.get(spec.id)
    except KeyError:
        repository.add(spec, name=spec.id, author="test")


def _record_walkforward(
    repository: StrategyRepository,
    link: ResultsLink,
    base: RunInputs,
    wf_id: str,
    *,
    metrics: dict[str, float] | None = None,
    verdict: str | None = None,
) -> None:
    """Register the strategy (if new) and bind one walk-forward result to it."""
    spec = base.bindings[0].spec
    _ensure_registered(repository, spec)
    record = build_record(
        spec=spec,
        manifest=base.manifest(),
        streams=base.streams,
        run_id=wf_id,
        run_kind=RUN_KIND_WALKFORWARD,
        selector_key=IdentitySelector(base).key(),
        metrics=metrics or {"expectancy_r": 0.1, "sharpe": 0.5, "trades": 20.0},
    )
    stored = link.record(record)
    if verdict is not None:
        link.grade(stored.run_id, verdict)


def _record_flat_run(
    repository: StrategyRepository, link: ResultsLink, base: RunInputs, store_root: Path
) -> str:
    """Register the strategy and bind one flat (non-walk-forward) result to it. Returns run_id."""
    spec = base.bindings[0].spec
    _ensure_registered(repository, spec)
    result = base.run()
    manifest = base.manifest()
    write_run(store_root, manifest, result)
    record = build_record(
        spec=spec,
        manifest=manifest,
        streams=base.streams,
        run_id=manifest.run_id,
        run_kind=RUN_KIND_FLAT,
        selector_key="identity",
        metrics={"expectancy_r": 0.05, "sharpe": 0.2, "trades": float(len(result.trades))},
    )
    link.record(record)
    return manifest.run_id


class TestNotMeasured:
    def test_a_strategy_with_zero_runs_is_listed_separately_not_blank_in_the_table(
        self, tmp_path: Path
    ) -> None:
        repository = StrategyRepository(tmp_path)
        link = ResultsLink(tmp_path)
        spec = ema_pullback(strategy_id="never-run")
        repository.add(spec, name="never run", author="test")

        report = build_library_report(repository, link, tmp_path / "runs")

        assert report.not_measured == ("never-run",)
        assert report.rows == ()

    def test_a_measured_strategy_does_not_appear_in_not_measured(
        self, registry: InstrumentRegistry, library: ExitLibrarySpec, tmp_path: Path
    ) -> None:
        repository = StrategyRepository(tmp_path / "lib")
        link = ResultsLink(tmp_path / "lib")
        store_root = tmp_path / "runs"
        base = _base_inputs(registry, library, "measured-one")
        wf_id = _run_walkforward(base, store_root)
        _record_walkforward(repository, link, base, wf_id)

        report = build_library_report(repository, link, store_root)

        assert "measured-one" not in report.not_measured
        assert [row.strategy_id for row in report.rows] == ["measured-one"]


class TestOrphanDetection:
    """Both run kinds, proven by deleting a real stored run out from under a record."""

    def test_a_resolvable_walkforward_is_not_orphaned(
        self, registry: InstrumentRegistry, library: ExitLibrarySpec, tmp_path: Path
    ) -> None:
        repository = StrategyRepository(tmp_path / "lib")
        link = ResultsLink(tmp_path / "lib")
        store_root = tmp_path / "runs"
        base = _base_inputs(registry, library, "intact-wf")
        wf_id = _run_walkforward(base, store_root)
        _record_walkforward(repository, link, base, wf_id)

        report = build_library_report(repository, link, store_root)

        assert report.orphaned == ()
        row = next(r for r in report.rows if r.strategy_id == "intact-wf")
        assert row.last_run_resolves
        assert row.n_runs == 1

    def test_deleting_the_walkforward_manifest_orphans_it(
        self, registry: InstrumentRegistry, library: ExitLibrarySpec, tmp_path: Path
    ) -> None:
        repository = StrategyRepository(tmp_path / "lib")
        link = ResultsLink(tmp_path / "lib")
        store_root = tmp_path / "runs"
        base = _base_inputs(registry, library, "broken-wf")
        wf_id = _run_walkforward(base, store_root)
        _record_walkforward(
            repository,
            link,
            base,
            wf_id,
            metrics={"expectancy_r": 0.42, "sharpe": 1.1, "trades": 30.0},
        )

        import shutil

        shutil.rmtree(store_root / "walkforward" / wf_id)

        report = build_library_report(repository, link, store_root)

        assert len(report.orphaned) == 1
        orphan = report.orphaned[0]
        assert orphan.strategy_id == "broken-wf"
        assert orphan.run_id == wf_id
        assert orphan.run_kind == RUN_KIND_WALKFORWARD

        row = next(r for r in report.rows if r.strategy_id == "broken-wf")
        assert not row.last_run_resolves
        # The stored summary metrics still show — they come from results.parquet,
        # not from re-reading the curve — but DSR, which needs the curve, does not.
        assert row.expectancy_r == 0.42
        assert row.sharpe == 1.1
        assert row.dsr is None
        assert row.dsr_reason is not None and "cannot be read back" in row.dsr_reason

    def test_deleting_one_folds_flat_run_still_orphans_the_walkforward(
        self, registry: InstrumentRegistry, library: ExitLibrarySpec, tmp_path: Path
    ) -> None:
        """A top-level manifest.json that parses is not enough — every fold must resolve too."""
        repository = StrategyRepository(tmp_path / "lib")
        link = ResultsLink(tmp_path / "lib")
        store_root = tmp_path / "runs"
        base = _base_inputs(registry, library, "half-broken-wf")
        wf_id = _run_walkforward(base, store_root)
        _record_walkforward(repository, link, base, wf_id)

        import json
        import shutil

        manifest_path = store_root / "walkforward" / wf_id / "manifest.json"
        payload = json.loads(manifest_path.read_text())
        first_oos_path = Path(payload["folds"][0]["oos_run"]["path"])
        assert first_oos_path.exists()
        shutil.rmtree(first_oos_path)

        report = build_library_report(repository, link, store_root)

        assert len(report.orphaned) == 1
        assert report.orphaned[0].run_id == wf_id

    def test_a_resolvable_flat_run_is_not_orphaned(
        self, registry: InstrumentRegistry, library: ExitLibrarySpec, tmp_path: Path
    ) -> None:
        repository = StrategyRepository(tmp_path / "lib")
        link = ResultsLink(tmp_path / "lib")
        store_root = tmp_path / "runs"
        base = _base_inputs(registry, library, "intact-flat")
        _record_flat_run(repository, link, base, store_root)

        report = build_library_report(repository, link, store_root)

        assert report.orphaned == ()
        row = next(r for r in report.rows if r.strategy_id == "intact-flat")
        assert row.last_run_resolves

    def test_deleting_a_flat_run_directory_orphans_it(
        self, registry: InstrumentRegistry, library: ExitLibrarySpec, tmp_path: Path
    ) -> None:
        repository = StrategyRepository(tmp_path / "lib")
        link = ResultsLink(tmp_path / "lib")
        store_root = tmp_path / "runs"
        base = _base_inputs(registry, library, "broken-flat")
        run_id = _record_flat_run(repository, link, base, store_root)

        import shutil

        shutil.rmtree(store_root / run_id)

        report = build_library_report(repository, link, store_root)

        assert len(report.orphaned) == 1
        assert report.orphaned[0].run_id == run_id
        assert report.orphaned[0].run_kind == RUN_KIND_FLAT


class TestSupersededResultsAreNotOrphans:
    def test_two_runs_of_one_strategy_count_as_two_and_use_the_newest_for_the_summary(
        self, registry: InstrumentRegistry, library: ExitLibrarySpec, tmp_path: Path
    ) -> None:
        repository = StrategyRepository(tmp_path / "lib")
        link = ResultsLink(tmp_path / "lib")
        store_root = tmp_path / "runs"
        base = _base_inputs(registry, library, "measured-twice")

        first_wf = _run_walkforward(base, store_root)
        _record_walkforward(
            repository,
            link,
            base,
            first_wf,
            metrics={"expectancy_r": -0.1, "sharpe": -0.3, "trades": 10.0},
        )

        # A second, distinct walk-forward over a slightly different window —
        # a different wf_id, standing in for "re-measured after an engine change".
        base2 = _base_inputs(registry, library, "measured-twice")
        base2 = base2.with_streams({EURUSD_H4: swing_series(LENGTH + 10)})
        second_wf = _run_walkforward(base2, store_root)
        _record_walkforward(
            repository,
            link,
            base2,
            second_wf,
            metrics={"expectancy_r": 0.2, "sharpe": 0.6, "trades": 15.0},
        )

        report = build_library_report(repository, link, store_root)

        assert report.orphaned == ()
        row = next(r for r in report.rows if r.strategy_id == "measured-twice")
        assert row.n_runs == 2
        assert row.expectancy_r == 0.2  # the newer of the two
        assert row.last_run is not None
        assert row.last_run.run_id == second_wf


class TestVerdictCounts:
    def test_counts_by_last_verdict_per_strategy_not_per_run(
        self, registry: InstrumentRegistry, library: ExitLibrarySpec, tmp_path: Path
    ) -> None:
        repository = StrategyRepository(tmp_path / "lib")
        link = ResultsLink(tmp_path / "lib")
        store_root = tmp_path / "runs"

        for strategy_id, verdict in (
            ("s-overfit", Verdict.OVERFIT.value),
            ("s-insufficient", Verdict.INSUFFICIENT.value),
            ("s-ungraded", None),
        ):
            base = _base_inputs(registry, library, strategy_id)
            wf_id = _run_walkforward(base, store_root)
            _record_walkforward(repository, link, base, wf_id, verdict=verdict)

        report = build_library_report(repository, link, store_root)

        assert report.verdicts.overfit == 1
        assert report.verdicts.insufficient == 1
        assert report.verdicts.ungraded == 1
        assert report.verdicts.robust == 0
        assert report.verdicts.fragile == 0
        assert report.verdicts.total == 3

    def test_zero_robust_says_nothing_to_approve_in_the_portfolio_note(
        self, registry: InstrumentRegistry, library: ExitLibrarySpec, tmp_path: Path
    ) -> None:
        repository = StrategyRepository(tmp_path / "lib")
        link = ResultsLink(tmp_path / "lib")
        store_root = tmp_path / "runs"
        base = _base_inputs(registry, library, "not-robust")
        wf_id = _run_walkforward(base, store_root)
        _record_walkforward(repository, link, base, wf_id, verdict=Verdict.OVERFIT.value)

        report = build_library_report(repository, link, store_root)

        assert report.verdicts.nothing_to_approve
        assert report.portfolio_note == NOTHING_TO_APPROVE

    def test_a_robust_verdict_produces_a_different_note_and_is_still_not_a_computed_portfolio(
        self, registry: InstrumentRegistry, library: ExitLibrarySpec, tmp_path: Path
    ) -> None:
        repository = StrategyRepository(tmp_path / "lib")
        link = ResultsLink(tmp_path / "lib")
        store_root = tmp_path / "runs"
        base = _base_inputs(registry, library, "robust-one")
        wf_id = _run_walkforward(base, store_root)
        _record_walkforward(repository, link, base, wf_id, verdict=Verdict.ROBUST.value)

        report = build_library_report(repository, link, store_root)

        assert not report.verdicts.nothing_to_approve
        assert report.verdicts.robust == 1
        assert report.portfolio_note != NOTHING_TO_APPROVE
        assert "does not compute a portfolio yet" in report.portfolio_note


def _points_from_returns(
    returns: list[float], *, start: Decimal = Decimal(100), start_day: date = date(2024, 1, 1)
) -> list[Any]:
    """One :class:`EquityPoint` per day, whose ``simple_returns`` are exactly ``returns``."""
    equity = [start]
    for r in returns:
        equity.append(equity[-1] * (1 + Decimal(str(r))))
    return [point(start_day + timedelta(days=i), float(e)) for i, e in enumerate(equity)]


class TestCorrelationMath:
    """Pure math over synthetic curves — no backtest involved."""

    def test_identical_series_correlate_perfectly(self) -> None:
        returns = [0.01, -0.02, 0.0, 0.03, -0.01, 0.02, 0.0, 0.01, -0.03, 0.02] * 7
        curves = {
            "a": _points_from_returns(returns),
            "b": _points_from_returns(returns),
        }
        cells = correlation_matrix(curves)
        assert len(cells) == 1
        assert cells[0].correlation == pytest.approx(1.0)

    def test_mirrored_series_anticorrelate(self) -> None:
        returns = [0.01, -0.02, 0.0, 0.03, -0.01, 0.02, 0.0, 0.01, -0.03, 0.02] * 7
        curves = {
            "a": _points_from_returns(returns),
            "b": _points_from_returns([-r for r in returns]),
        }
        cells = correlation_matrix(curves)
        assert cells[0].correlation == pytest.approx(-1.0)

    def test_flat_days_are_zero_return_and_do_not_poison_the_coefficient(self) -> None:
        """Mostly-flat, no-edge-looking series whose ACTIVE days perfectly agree."""
        active = [0.01, -0.02, 0.03, -0.01, 0.02] * 15
        n = 90
        # Both series flat on most days, both active (and identical) on a
        # sixth of them — exactly the "no confirmed edge, still real
        # co-movement" scenario the module docstring's arithmetic is about.
        both = [active[i % len(active)] if i % 6 == 0 else 0.0 for i in range(n)]
        curves = {
            "a": _points_from_returns(both, start_day=date(2024, 1, 1)),
            "b": _points_from_returns(both, start_day=date(2024, 1, 1)),
        }
        cells = correlation_matrix(curves)
        assert cells[0].correlation == pytest.approx(1.0)
        assert cells[0].n_both_active < cells[0].n_shared_days

    def test_rarely_co_active_but_locally_agreeing_pair_scores_lower_than_always_co_active(
        self,
    ) -> None:
        """The discount the module docstring's arithmetic predicts, demonstrated.

        Two pairs agree identically on every day they are BOTH active. One pair
        is active together every day; the other is active together on only a
        quarter of days — the other three quarters, exactly one side holds a
        position with an unrelated move while the other is flat. The
        full-calendar correlation must be lower for the second pair — that
        discount is the entire point of not restricting the computation to
        mutually-active days.
        """
        n = 200
        agreement = [0.01 if i % 2 == 0 else -0.01 for i in range(n)]
        noise = [((-1) ** i) * 0.015 for i in range(n)]

        # Pair 1: always co-active, always agrees.
        always = {
            "x1": _points_from_returns(agreement, start_day=date(2024, 1, 1)),
            "y1": _points_from_returns(agreement, start_day=date(2024, 1, 1)),
        }
        always_cell = correlation_matrix(always)[0]

        # Pair 2: agree on 1 day in 4; on the other 3, exactly one side is
        # active alone (unrelated noise) and the other is flat that day.
        x2: list[float] = []
        y2: list[float] = []
        for i in range(n):
            bucket = i % 4
            if bucket == 0:
                x2.append(agreement[i])
                y2.append(agreement[i])
            elif bucket == 1:
                x2.append(noise[i])
                y2.append(0.0)
            elif bucket == 2:
                x2.append(0.0)
                y2.append(noise[i])
            else:
                x2.append(0.0)
                y2.append(0.0)
        rarely = {
            "x2": _points_from_returns(x2, start_day=date(2024, 1, 1)),
            "y2": _points_from_returns(y2, start_day=date(2024, 1, 1)),
        }
        rarely_cell = correlation_matrix(rarely)[0]

        assert always_cell.n_both_active == n
        assert rarely_cell.n_both_active == n // 4
        assert rarely_cell.correlation is not None
        assert always_cell.correlation is not None
        assert rarely_cell.correlation < always_cell.correlation

    def test_below_the_shared_day_floor_is_none_not_a_fabricated_number(self) -> None:
        short = [0.01, -0.01, 0.02]
        assert len(short) < MIN_SHARED_DAYS
        curves = {
            "a": _points_from_returns(short),
            "b": _points_from_returns(short),
        }
        cells = correlation_matrix(curves)
        assert cells[0].correlation is None
        assert cells[0].n_shared_days == len(short)

    def test_a_constant_series_is_undefined_not_zero(self) -> None:
        """Zero variance -> None, per the module's absent-not-zero discipline."""
        varying = [0.01, -0.02, 0.03, -0.01] * 20
        flat = [0.0] * 80
        curves = {
            "varying": _points_from_returns(varying),
            "flat": _points_from_returns(flat),
        }
        cells = correlation_matrix(curves)
        assert cells[0].correlation is None

    def test_three_curves_give_three_pairs(self) -> None:
        base = [0.01, -0.01, 0.02, -0.02] * 20
        curves = {
            "a": _points_from_returns(base),
            "b": _points_from_returns(base),
            "c": _points_from_returns([-r for r in base]),
        }
        cells = correlation_matrix(curves)
        pairs = {(c.left, c.right) for c in cells}
        assert pairs == {("a", "b"), ("a", "c"), ("b", "c")}


class TestRendering:
    def test_the_page_renders_with_zero_measured_strategies(self, tmp_path: Path) -> None:
        repository = StrategyRepository(tmp_path / "lib")
        link = ResultsLink(tmp_path / "lib")
        report = build_library_report(repository, link, tmp_path / "runs")

        rendered = render(report)

        assert "Strategy library" in rendered.html
        # Zero strategies means zero ROBUST, so the note shows even here.
        assert NOTHING_TO_APPROVE in rendered.html
        assert "nothing to correlate" in rendered.html.lower()

    def test_a_real_report_renders_and_shows_the_no_robust_note(
        self, registry: InstrumentRegistry, library: ExitLibrarySpec, tmp_path: Path
    ) -> None:
        repository = StrategyRepository(tmp_path / "lib")
        link = ResultsLink(tmp_path / "lib")
        store_root = tmp_path / "runs"
        base = _base_inputs(registry, library, "render-me")
        wf_id = _run_walkforward(base, store_root)
        _record_walkforward(repository, link, base, wf_id, verdict=Verdict.OVERFIT.value)

        rendered = render(build_library_report(repository, link, store_root))

        assert "render-me" in rendered.html
        assert "Nothing to approve" in rendered.html
        assert "OVERFIT" in rendered.html
