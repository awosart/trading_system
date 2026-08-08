"""Running a walk-forward: no leakage across the fold boundary, deterministic fills, idempotence."""

from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from tests.backtest.conftest import ema_pullback, harness_inputs, swing_series
from trading_system.backtest.clock import StreamKey
from trading_system.backtest.orchestrator import StrategyBinding
from trading_system.backtest.reproducibility import result_digest
from trading_system.backtest.spec import RunInputs
from trading_system.core.instruments import InstrumentRegistry
from trading_system.core.types import Timeframe
from trading_system.data.models import OHLCVFrame
from trading_system.data.resample import FX_DAY_ORIGIN
from trading_system.exit.library import ExitLibrarySpec
from trading_system.validation.splitting import WalkForwardMode, WalkForwardSplitter
from trading_system.validation.walkforward import (
    IdentitySelector,
    WalkForwardResult,
    WalkForwardRunner,
)

EURUSD_H4 = StreamKey("EURUSD", Timeframe.H4)


def _ema_pullback_base(
    registry: InstrumentRegistry, library: ExitLibrarySpec, length: int
) -> RunInputs:
    """A modest ema_pullback/swing_series run, long enough for several folds."""
    spec = ema_pullback()
    preset = next(item for item in library.presets if item.id == spec.exit_ref)
    return harness_inputs(
        registry,
        streams={EURUSD_H4: swing_series(length)},
        bindings=[StrategyBinding(spec=spec, exit_preset=preset, keys=(EURUSD_H4,))],
    )


def _coverage(frame: OHLCVFrame) -> tuple[datetime, datetime]:
    """``(frame.start, frame.end)``, proven non-empty rather than merely assumed."""
    assert frame.start is not None
    assert frame.end is not None
    return frame.start, frame.end


class TestFoldWarmupIsBoundedByDataStart:
    """The truncation-equivalence property, lifted to a fold's own boundary.

    Extra history sitting before a fold's ``data_start`` in the base run's own
    streams must have zero effect on that fold's result: :meth:`WalkForwardRunner._is_inputs`
    slices it away before a single feature is computed. If it leaked in, the
    fold's warmup would not be provably sufficient — it would just be however
    much history the base run happened to carry.
    """

    def test_padding_before_data_start_does_not_change_the_is_run(
        self, registry: InstrumentRegistry, library: ExitLibrarySpec, tmp_path: Path
    ) -> None:
        base = _ema_pullback_base(registry, library, length=1400)
        splitter = WalkForwardSplitter(
            mode=WalkForwardMode.ROLLING,
            is_span=timedelta(days=60),
            oos_span=timedelta(days=20),
            step=timedelta(days=20),
            embargo=timedelta(days=2),
            warmup=timedelta(days=10),
        )
        full = base.streams[EURUSD_H4]
        folds = splitter.split(_coverage(full), day_origin=FX_DAY_ORIGIN)
        fold = folds[0]
        assert fold.is_window.data_start > _coverage(full)[0], (
            "the fixture must carry real spare history"
        )

        tight = full.slice(fold.is_window.data_start, None)
        base_tight = base.with_streams({EURUSD_H4: tight})
        base_padded = base  # carries everything from full.start, strictly more than needed

        runner_tight = WalkForwardRunner(
            base=base_tight,
            splitter=splitter,
            selector=IdentitySelector(base_tight),
            store_root=tmp_path / "tight",
            max_drain_bars=20,
        )
        runner_padded = WalkForwardRunner(
            base=base_padded,
            splitter=splitter,
            selector=IdentitySelector(base_padded),
            store_root=tmp_path / "padded",
            max_drain_bars=20,
        )

        result_tight = runner_tight._is_inputs(fold).run()
        result_padded = runner_padded._is_inputs(fold).run()

        assert result_digest(result_tight) == result_digest(result_padded)

    def test_the_fold_actually_trades(
        self, registry: InstrumentRegistry, library: ExitLibrarySpec, tmp_path: Path
    ) -> None:
        """Without this, the equality above could hold for two empty runs."""
        base = _ema_pullback_base(registry, library, length=1400)
        splitter = WalkForwardSplitter(
            mode=WalkForwardMode.ROLLING,
            is_span=timedelta(days=60),
            oos_span=timedelta(days=20),
            step=timedelta(days=20),
            embargo=timedelta(days=2),
            warmup=timedelta(days=10),
        )
        full = base.streams[EURUSD_H4]
        fold = splitter.split(_coverage(full), day_origin=FX_DAY_ORIGIN)[0]
        runner = WalkForwardRunner(
            base=base,
            splitter=splitter,
            selector=IdentitySelector(base),
            store_root=tmp_path,
            max_drain_bars=20,
        )
        assert runner._is_inputs(fold).run().trades


class TestFillIdentityAcrossOverlappingFolds:
    """A calendar bar sitting in two folds' overlapping IS windows draws the same entry fill.

    Scoped to the entry fill deliberately: ``position_id`` (the order id,
    derived from ``close_ts`` since the fix this test guards) and
    ``entry_price`` are what :func:`~trading_system.execution.rng.fill_seed`
    and the spread/slippage model determine from the bar alone, and both must
    match exactly for a trade recognised on the same calendar bar in either
    fold. What is *not* asserted is the eventual ``net`` or ``size`` of that
    trade: sizing is equity-relative
    (:class:`~trading_system.risk.sizing.methods.FixedFractional`, the
    harness default) and the two folds do not share an equity path outside
    their overlap, and the exit itself is read off ``structure_trail``'s own
    ATR/swing features, which — like any indicator seeded from a different
    ``data_start`` — converge rather than match bit for bit. Both are
    legitimate fold-to-fold differences, not leaks; the entry fill is the
    claim P15 actually makes.
    """

    def test_overlapping_folds_agree_on_shared_entry_fills(
        self, registry: InstrumentRegistry, library: ExitLibrarySpec, tmp_path: Path
    ) -> None:
        base = _ema_pullback_base(registry, library, length=1400)
        splitter = WalkForwardSplitter(
            mode=WalkForwardMode.ROLLING,
            is_span=timedelta(days=60),
            oos_span=timedelta(days=20),
            step=timedelta(days=20),
            embargo=timedelta(days=2),
            warmup=timedelta(days=10),
        )
        full = base.streams[EURUSD_H4]
        folds = splitter.split(_coverage(full), day_origin=FX_DAY_ORIGIN)
        assert len(folds) >= 2
        fold0, fold1 = folds[0], folds[1]
        assert fold1.is_window.trade_start < fold0.is_window.trade_end, "folds must overlap"

        runner = WalkForwardRunner(
            base=base,
            splitter=splitter,
            selector=IdentitySelector(base),
            store_root=tmp_path,
            max_drain_bars=20,
        )
        result0 = runner._is_inputs(fold0).run()
        result1 = runner._is_inputs(fold1).run()

        overlap_start = fold1.is_window.trade_start
        overlap_end = fold0.is_window.trade_end
        by_open0 = {
            trade.opened_at: trade
            for trade in result0.trades
            if overlap_start < trade.opened_at < overlap_end
        }
        by_open1 = {
            trade.opened_at: trade
            for trade in result1.trades
            if overlap_start < trade.opened_at < overlap_end
        }
        shared = set(by_open0) & set(by_open1)
        assert shared, "no trade opened on a bar shared by both folds; the test proves nothing"
        for ts in shared:
            a, b = by_open0[ts], by_open1[ts]
            assert a.position_id == b.position_id
            assert a.entry_price == b.entry_price


class TestIdempotentRun:
    def test_repeated_run_is_a_no_op(
        self, registry: InstrumentRegistry, library: ExitLibrarySpec, tmp_path: Path
    ) -> None:
        base = _ema_pullback_base(registry, library, length=1400)
        splitter = WalkForwardSplitter(
            mode=WalkForwardMode.ROLLING,
            is_span=timedelta(days=60),
            oos_span=timedelta(days=20),
            step=timedelta(days=20),
            embargo=timedelta(days=2),
            warmup=timedelta(days=10),
        )
        runner = WalkForwardRunner(
            base=base,
            splitter=splitter,
            selector=IdentitySelector(base),
            store_root=tmp_path,
            max_drain_bars=20,
        )

        first = runner.run()
        written = {path: path.stat().st_mtime_ns for path in tmp_path.rglob("*") if path.is_file()}
        assert written, "the first run must have written something to compare against"

        second = runner.run()
        still = {path: path.stat().st_mtime_ns for path in tmp_path.rglob("*") if path.is_file()}

        assert second.wf_id == first.wf_id
        assert still == written, "a repeated run() touched a file it should not have"
        is_ids = [fr.is_run.run_id for fr in second.folds]
        oos_ids = [fr.oos_run.run_id for fr in second.folds]
        assert is_ids == [fr.is_run.run_id for fr in first.folds]
        assert oos_ids == [fr.oos_run.run_id for fr in first.folds]


@pytest.fixture(scope="module")
def real_result(
    registry: InstrumentRegistry, library: ExitLibrarySpec, tmp_path_factory: pytest.TempPathFactory
) -> WalkForwardResult:
    """One real walk-forward, shared read-only across the tests below."""
    base = _ema_pullback_base(registry, library, length=2200)
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
        store_root=tmp_path_factory.mktemp("wf_store"),
        max_drain_bars=30,
    )
    return runner.run()


class TestRealRun:
    def test_more_than_one_fold(self, real_result: WalkForwardResult) -> None:
        assert len(real_result.folds) >= 2

    def test_at_least_one_fold_closed_oos_trades(self, real_result: WalkForwardResult) -> None:
        assert any(fr.oos_run.counters["fills"] > 0 for fr in real_result.folds)

    def test_every_fold_reports_its_own_trade_count(self, real_result: WalkForwardResult) -> None:
        from trading_system.backtest.reproducibility import read_run

        for fold_run in real_result.folds:
            oos = read_run(fold_run.oos_run.path)
            # Every trade opened inside this fold's own OOS window, by
            # construction — see the module docstring on evaluation_start/end.
            for trade in oos.result.trades:
                assert fold_run.fold.oos_window.trade_start <= trade.opened_at
                assert trade.opened_at < fold_run.fold.oos_window.trade_end


class TestBoundaryResidualInvariant:
    """``equity(trade_end) - equity(trade_start) == realised-in-window + boundary_residual``.

    Computed independently here from the raw stored curve and trades, and
    checked against every fold's own report row — not against
    ``report._boundary_residual`` itself, which would just be testing the
    function against its own definition.
    """

    def test_holds_for_every_fold_both_windows(self, real_result: WalkForwardResult) -> None:
        from trading_system.backtest.reproducibility import read_run
        from trading_system.validation.report import build_report

        report = build_report(real_result, min_trades_per_fold=1)

        for fold_report, fold_run in zip(report.folds, real_result.folds, strict=True):
            for window, run, reported in (
                (fold_run.fold.is_window, fold_run.is_run, fold_report.is_boundary_residual),
                (fold_run.fold.oos_window, fold_run.oos_run, fold_report.oos_boundary_residual),
            ):
                stored = read_run(run.path)
                curve = stored.result.curve
                trades = stored.result.trades
                start_equity = next(
                    point.equity for point in reversed(curve) if point.ts <= window.trade_start
                )
                end_equity = next(
                    point.equity for point in reversed(curve) if point.ts <= window.trade_end
                )
                realized = sum(
                    (
                        trade.net
                        for trade in trades
                        if window.trade_start < trade.closed_at <= window.trade_end
                    ),
                    Decimal(0),
                )
                independent_residual = end_equity - start_equity - realized
                assert reported == independent_residual
