"""Splicing OOS windows into one return series, for P14's metrics.

**Stitched by return, not by money.** Every fold's OOS run starts from the
same starting balance (P15 stage 1's own reset-between-folds decision — see
CLAUDE.md), so concatenating equity values directly would draw a discontinuity
at every fold seam: fold 2 does not actually start over at the base capital,
it continues from wherever fold 1 left off. Each fold's own trajectory is
converted to a return relative to *its own* first point, then compounded onto
the running total left by the folds before it.

**Only the tradeable window is stitched.** A fold's stored OOS curve extends
from ``data_start`` (warmup, no signals yet) through ``trade_end`` plus the
drain allowance (no new signals, existing ones still closing) — neither belongs
in a performance series. Only ``[trade_start, trade_end]`` is spliced; the
drain's own effect on P&L is what
:mod:`trading_system.validation.walkforward`'s ``boundary_residual`` already
accounts for, in the fold's own report row rather than in this series.

**The result is not a run, and :func:`~trading_system.backtest.reproducibility.write_run`
will not take it.** No single :class:`~trading_system.backtest.spec.RunInputs`
produced this curve, so it earns no ``run_id``. Rather than fabricate a
sentinel manifest for something that must never be storable,
:class:`StitchedCurve` simply is not a
:class:`~trading_system.backtest.orchestrator.BacktestResult` — it carries no
``trades``, no drop counters, nothing ``write_run`` needs, so passing it there
fails the moment ``write_run`` reaches for an attribute this type does not
have, rather than needing a bespoke guard to say so.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal

from trading_system.backtest.portfolio import EquityPoint
from trading_system.validation.splitting import Fold


@dataclass(frozen=True)
class StitchedCurve:
    """OOS equity, spliced fold to fold as a compounded return series.

    Attributes:
        points: One :class:`~trading_system.backtest.portfolio.EquityPoint`
            per stitched instant, chronological. ``equity`` and ``balance``
            are equal on every point and carry no meaning beyond the series'
            own compounded return — there is no financing, no commission and
            no open position behind this curve, only a splice of other
            curves' returns.
        starting_equity: The synthetic capital the series starts from.
    """

    points: tuple[EquityPoint, ...]
    starting_equity: Decimal


def stitch(
    folds: Sequence[Fold],
    oos_curves: Sequence[Sequence[EquityPoint]],
    *,
    starting_equity: Decimal = Decimal(100_000),
) -> StitchedCurve:
    """Splice each fold's OOS trade window into one compounded return series.

    Args:
        folds: The folds, any order — sorted here by ``index``.
        oos_curves: Each fold's own stored OOS curve, **unsliced** — this
            function restricts each to its own ``[trade_start, trade_end]``
            itself, so the caller does not have to duplicate that boundary.
            Same length and order as ``folds``.
        starting_equity: Synthetic capital the stitched series starts from.
            Money-shaped only so :mod:`trading_system.analytics.metrics`
            (which expects a curve of positive ``Decimal`` equity) runs on
            this series unmodified; it does not represent an account any run
            actually held.

    Returns:
        The stitched curve. A fold whose OOS curve has no row inside its own
        trade window contributes nothing — the running total simply carries
        flat through the gap, which is the honest reading of "no observation",
        not zero return.

    Raises:
        ValueError: If ``folds`` and ``oos_curves`` disagree in length, if
            ``starting_equity`` is not positive, or if two folds' OOS trade
            windows overlap — splicing a bar's return into the series twice
            would make the stitched Sharpe a function of how the folds were
            cut rather than of what the strategy did.
    """
    if len(folds) != len(oos_curves):
        raise ValueError(
            f"folds ({len(folds)}) and oos_curves ({len(oos_curves)}) must be the same length"
        )
    if starting_equity <= 0:
        raise ValueError(f"starting_equity must be positive, got {starting_equity}")

    ordered = sorted(zip(folds, oos_curves, strict=True), key=lambda pair: pair[0].index)
    for (prev_fold, _), (next_fold, _) in zip(ordered, ordered[1:], strict=False):
        if prev_fold.oos_window.trade_end > next_fold.oos_window.trade_start:
            raise ValueError(
                f"fold {prev_fold.index}'s OOS window ends {prev_fold.oos_window.trade_end!r}, "
                f"after fold {next_fold.index}'s OOS window starts "
                f"{next_fold.oos_window.trade_start!r}; stitching overlapping OOS windows "
                "would count bars in both, making the result a function of how the folds "
                "were cut rather than of what the strategy did — reduce step, or read the "
                "per-fold report instead of the stitched series"
            )

    points: list[EquityPoint] = []
    running_equity = starting_equity
    for fold, curve in ordered:
        window = fold.oos_window
        segment = [point for point in curve if window.trade_start <= point.ts <= window.trade_end]
        if not segment:
            continue
        fold_start_equity = segment[0].equity
        if fold_start_equity <= 0:
            raise ValueError(
                f"fold {fold.index}: OOS equity at window start is {fold_start_equity}, "
                "not positive — cannot express the rest of the window as a return of it"
            )
        for point in segment:
            fold_return = point.equity / fold_start_equity - 1
            stitched_equity = running_equity * (1 + fold_return)
            points.append(
                EquityPoint(
                    ts=point.ts,
                    day=point.day,
                    balance=stitched_equity,
                    equity=stitched_equity,
                    realized=Decimal(0),
                    unrealized=Decimal(0),
                    commission_paid=Decimal(0),
                    swap_paid=Decimal(0),
                    open_positions=point.open_positions,
                )
            )
        running_equity = points[-1].equity

    return StitchedCurve(points=tuple(points), starting_equity=starting_equity)
