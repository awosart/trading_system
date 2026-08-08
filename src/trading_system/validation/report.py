"""What a walk-forward produced, as data — no HTML, no verdict.

Every number a fold's boundary can lose is a number this module puts in the
report rather than a number it summarises away: rejections, degradations,
drops and ``ATR_UNAVAILABLE`` coverage all travel from the stored run into
:class:`FoldReport` untouched, the same discipline
:mod:`trading_system.backtest.orchestrator` already applies to a single run.

**Whether a ratio between IS and OOS is worth computing is not this module's
call.** CLAUDE.md's own requirement for this stage is explicit: publish the
pairs (``is_expectancy_r``, ``oos_expectancy_r`` and the two Sortinos), and
leave "OOS ≥ 60% of IS" — a rule that is undefined the moment IS is near zero
or negative — to stage 5, where a verdict is actually rendered.

**A fold below ``min_trades_per_fold`` is flagged, never dropped.** Excluding
it from the aggregate would be a second, silent way for a boundary to lose
information, on top of every counter this module already exists to preserve.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from trading_system.analytics.metrics import daily_curve, sharpe_daily, sortino_daily, trade_stats
from trading_system.backtest.portfolio import EquityPoint, TradeRecord
from trading_system.backtest.reproducibility import read_run
from trading_system.validation.splitting import Fold, FoldWindow
from trading_system.validation.stitching import StitchedCurve, stitch
from trading_system.validation.walkforward import WalkForwardResult


def _equity_at(curve: Sequence[EquityPoint], ts: datetime) -> Decimal | None:
    """Equity of the last row at or before ``ts``, or ``None`` if the curve starts after it."""
    at_or_before = [point.equity for point in curve if point.ts <= ts]
    return at_or_before[-1] if at_or_before else None


def _boundary_residual(
    curve: Sequence[EquityPoint], trades: Sequence[TradeRecord], window: FoldWindow
) -> Decimal | None:
    """``equity(trade_end) - equity(trade_start) - realised PnL closed inside the window``.

    An exact bookkeeping identity, not an estimate — see
    :mod:`trading_system.validation.walkforward`'s module docstring. ``None``
    only if the stored curve does not actually cover the window, which would
    mean the run was sliced wrong upstream rather than anything this function
    can resolve.
    """
    start_equity = _equity_at(curve, window.trade_start)
    end_equity = _equity_at(curve, window.trade_end)
    if start_equity is None or end_equity is None:
        return None
    realized = sum(
        (trade.net for trade in trades if window.trade_start < trade.closed_at <= window.trade_end),
        Decimal(0),
    )
    return end_equity - start_equity - realized


def _expectancy_r(trades: Sequence[TradeRecord]) -> float | None:
    """Mean realised R, or ``None`` with fewer than one trade."""
    if not trades:
        return None
    return trade_stats(trades).expectancy_r


def _sortino(curve: Sequence[EquityPoint]) -> float | None:
    """Annualised Sortino of a curve, or ``None`` when the sample cannot support one.

    Catches exactly the :class:`ValueError` cases
    :func:`~trading_system.analytics.metrics.sortino_daily` documents as
    undefined (fewer than one daily return, zero downside deviation) and turns
    each into an explicit ``None`` — a value this report can show, rather than
    an exception the caller of :func:`build_report` would have to guard
    against on every fold.
    """
    try:
        return sortino_daily(daily_curve(curve)).value
    except ValueError:
        return None


def _sharpe(curve: Sequence[EquityPoint]) -> float | None:
    """Annualised Sharpe of a curve, or ``None`` when the sample cannot support one."""
    try:
        return sharpe_daily(daily_curve(curve)).value
    except ValueError:
        return None


@dataclass(frozen=True)
class FoldReport:
    """One fold's own numbers: periods, trade counts, IS/OOS metric pairs, and every counter.

    Attributes:
        index: The fold's position in the sequence.
        is_window: The in-sample window.
        oos_window: The out-of-sample window.
        is_trade_count: Trades the IS run closed.
        oos_trade_count: Trades the OOS run closed — every one of them opened
            inside ``oos_window``, by construction; see
            :mod:`trading_system.validation.walkforward`.
        is_expectancy_r: Mean realised R of the IS run's trades, or ``None``
            with none.
        oos_expectancy_r: Same, OOS.
        is_sortino: Annualised Sortino of the IS run's own curve, or ``None``.
        oos_sortino: Same, OOS.
        is_boundary_residual: Bookkeeping residual at the IS window's own
            boundary — see :func:`_boundary_residual`.
        oos_boundary_residual: Same, OOS. What the drain exists to keep small.
        drain_truncated: Positions still open when the OOS run's drain
            allowance ran out — ``open_at_end`` of the stored OOS run.
        rejections: Risk Engine refusals, OOS run, every reason present.
        degradations: Risk Engine measurements that fell back to a prior, OOS
            run.
        exit_drops: Exit instructions not carried out as stated, OOS run.
        entry_drops: Entry signals discarded as defective, OOS run.
        signal_drops: Signals discarded before reaching the Risk Engine, OOS
            run.
        atr_unavailable_fraction: Share of OOS fills priced without a
            volatility measurement.
        insufficient: Whether ``oos_trade_count`` is below the report's own
            ``min_trades_per_fold`` — flagged, not excluded from any
            aggregate below.
    """

    index: int
    is_window: FoldWindow
    oos_window: FoldWindow
    is_trade_count: int
    oos_trade_count: int
    is_expectancy_r: float | None
    oos_expectancy_r: float | None
    is_sortino: float | None
    oos_sortino: float | None
    is_boundary_residual: Decimal | None
    oos_boundary_residual: Decimal | None
    drain_truncated: int
    rejections: dict[str, int]
    degradations: dict[str, int]
    exit_drops: dict[str, int]
    entry_drops: dict[str, int]
    signal_drops: dict[str, int]
    atr_unavailable_fraction: float
    insufficient: bool

    def to_dict(self) -> dict[str, Any]:
        """Serialise to JSON-able data."""

        def window(w: FoldWindow) -> dict[str, str]:
            return {
                "data_start": w.data_start.isoformat(),
                "trade_start": w.trade_start.isoformat(),
                "trade_end": w.trade_end.isoformat(),
            }

        return {
            "index": self.index,
            "is_window": window(self.is_window),
            "oos_window": window(self.oos_window),
            "is_trade_count": self.is_trade_count,
            "oos_trade_count": self.oos_trade_count,
            "is_expectancy_r": self.is_expectancy_r,
            "oos_expectancy_r": self.oos_expectancy_r,
            "is_sortino": self.is_sortino,
            "oos_sortino": self.oos_sortino,
            "is_boundary_residual": str(self.is_boundary_residual)
            if self.is_boundary_residual is not None
            else None,
            "oos_boundary_residual": str(self.oos_boundary_residual)
            if self.oos_boundary_residual is not None
            else None,
            "drain_truncated": self.drain_truncated,
            "rejections": self.rejections,
            "degradations": self.degradations,
            "exit_drops": self.exit_drops,
            "entry_drops": self.entry_drops,
            "signal_drops": self.signal_drops,
            "atr_unavailable_fraction": self.atr_unavailable_fraction,
            "insufficient": self.insufficient,
        }


@dataclass(frozen=True)
class WalkForwardReport:
    """Every fold's own report, plus the whole run's stitched OOS picture.

    Attributes:
        wf_id: The walk-forward this report was built from.
        min_trades_per_fold: Threshold :attr:`FoldReport.insufficient` was
            measured against.
        folds: Every fold's report, oldest first.
        n_folds: ``len(folds)``.
        n_folds_without_oos_trades: Folds whose OOS run closed no trades at
            all — the extreme case of ``insufficient``.
        insufficient_sample: Whether any fold is ``insufficient`` — a coarse,
            visible flag; the per-fold detail is in ``folds``.
        stitched_trade_count: Trades across every fold's OOS run, pooled.
        stitched_expectancy_r: Mean realised R across every fold's OOS trades,
            pooled — not averaged fold by fold, which would weight a
            ten-trade fold the same as a hundred-trade one.
        stitched_sharpe: Annualised Sharpe of the spliced OOS return series.
        stitched_sortino: Annualised Sortino of the same series.
        per_fold_oos_expectancy_r: Each fold's own OOS expectancy, in fold
            order — the distribution CLAUDE.md asks for, alongside the pooled
            figure above.
        per_fold_oos_sortino: Each fold's own OOS Sortino, in fold order.
    """

    wf_id: str
    min_trades_per_fold: int
    folds: tuple[FoldReport, ...]
    n_folds: int
    n_folds_without_oos_trades: int
    insufficient_sample: bool
    stitched_trade_count: int
    stitched_expectancy_r: float | None
    stitched_sharpe: float | None
    stitched_sortino: float | None
    per_fold_oos_expectancy_r: tuple[float | None, ...]
    per_fold_oos_sortino: tuple[float | None, ...]

    def to_dict(self) -> dict[str, Any]:
        """Serialise to JSON-able data."""
        return {
            "wf_id": self.wf_id,
            "min_trades_per_fold": self.min_trades_per_fold,
            "folds": [fold.to_dict() for fold in self.folds],
            "n_folds": self.n_folds,
            "n_folds_without_oos_trades": self.n_folds_without_oos_trades,
            "insufficient_sample": self.insufficient_sample,
            "stitched_trade_count": self.stitched_trade_count,
            "stitched_expectancy_r": self.stitched_expectancy_r,
            "stitched_sharpe": self.stitched_sharpe,
            "stitched_sortino": self.stitched_sortino,
            "per_fold_oos_expectancy_r": list(self.per_fold_oos_expectancy_r),
            "per_fold_oos_sortino": list(self.per_fold_oos_sortino),
        }


def _fold_report(
    fold: Fold,
    is_curve: Sequence[EquityPoint],
    is_trades: Sequence[TradeRecord],
    oos_curve: Sequence[EquityPoint],
    oos_trades: Sequence[TradeRecord],
    oos_counters: Mapping[str, Any],
    *,
    min_trades_per_fold: int,
) -> FoldReport:
    """Build one fold's report from its two runs' curves and trades."""
    oos_trade_count = len(oos_trades)
    return FoldReport(
        index=fold.index,
        is_window=fold.is_window,
        oos_window=fold.oos_window,
        is_trade_count=len(is_trades),
        oos_trade_count=oos_trade_count,
        is_expectancy_r=_expectancy_r(is_trades),
        oos_expectancy_r=_expectancy_r(oos_trades),
        is_sortino=_sortino(is_curve),
        oos_sortino=_sortino(oos_curve),
        is_boundary_residual=_boundary_residual(is_curve, is_trades, fold.is_window),
        oos_boundary_residual=_boundary_residual(oos_curve, oos_trades, fold.oos_window),
        drain_truncated=int(oos_counters["open_at_end"]),
        rejections=dict(oos_counters["rejections"]),
        degradations=dict(oos_counters["degradations"]),
        exit_drops=dict(oos_counters["exit_drops"]),
        entry_drops=dict(oos_counters["entry_drops"]),
        signal_drops=dict(oos_counters["signal_drops"]),
        atr_unavailable_fraction=float(
            oos_counters["cost_degradations"].get("atr_unavailable", 0.0)
        ),
        insufficient=oos_trade_count < min_trades_per_fold,
    )


def build_report(
    result: WalkForwardResult,
    *,
    min_trades_per_fold: int,
    stitched_starting_equity: Decimal = Decimal(100_000),
) -> WalkForwardReport:
    """Build the full report from a finished walk-forward, reading tables off disk.

    Args:
        result: The walk-forward's result — run ids and paths, not curves or
            trades; both are read here, explicitly, per fold.
        min_trades_per_fold: Below this many OOS trades, a fold is flagged
            ``insufficient``. Required, with no default — see this module's
            docstring on why silence is not an option here.
        stitched_starting_equity: Passed straight to
            :func:`~trading_system.validation.stitching.stitch`.

    Returns:
        The report.
    """
    fold_reports: list[FoldReport] = []
    oos_curves: list[Sequence[EquityPoint]] = []
    folds: list[Fold] = []
    pooled_oos_trades: list[TradeRecord] = []

    for fold_run in result.folds:
        is_stored = read_run(fold_run.is_run.path)
        oos_stored = read_run(fold_run.oos_run.path)
        fold_reports.append(
            _fold_report(
                fold_run.fold,
                is_stored.result.curve,
                is_stored.result.trades,
                oos_stored.result.curve,
                oos_stored.result.trades,
                fold_run.oos_run.counters,
                min_trades_per_fold=min_trades_per_fold,
            )
        )
        oos_curves.append(oos_stored.result.curve)
        folds.append(fold_run.fold)
        pooled_oos_trades.extend(oos_stored.result.trades)

    stitched: StitchedCurve = stitch(folds, oos_curves, starting_equity=stitched_starting_equity)

    return WalkForwardReport(
        wf_id=result.wf_id,
        min_trades_per_fold=min_trades_per_fold,
        folds=tuple(fold_reports),
        n_folds=len(fold_reports),
        n_folds_without_oos_trades=sum(1 for report in fold_reports if report.oos_trade_count == 0),
        insufficient_sample=any(report.insufficient for report in fold_reports),
        stitched_trade_count=len(pooled_oos_trades),
        stitched_expectancy_r=_expectancy_r(pooled_oos_trades),
        stitched_sharpe=_sharpe(stitched.points),
        stitched_sortino=_sortino(stitched.points),
        per_fold_oos_expectancy_r=tuple(report.oos_expectancy_r for report in fold_reports),
        per_fold_oos_sortino=tuple(report.oos_sortino for report in fold_reports),
    )


def write_report(report: WalkForwardReport, path: Path) -> None:
    """Export a report to JSON, next to the walk-forward's own manifest.

    Args:
        report: The report.
        path: File to write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2))
