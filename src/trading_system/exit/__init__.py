"""Exit Engine: managing a position that is already open.

An independent library. It does not import the Entry Engine and the Entry Engine
does not import it — a strategy names its exit by ``exit_ref``, so entries and
exits combine N×M. What crosses the boundary is a position, opened by the
backtest layer from an entry signal, with the signal's ``invalidation_price``
installed as its stop.

A run has three steps::

    plan     = ExitPlan(exit_id="conservative_2r", protective_stop=ProtectiveStop(),
                        rules=[FixedRR(2.0)])
    position = ManagedPosition(symbol=..., side=..., entry_price=..., size=...,
                               initial_stop=..., opened_at=...)
    result   = plan.run(position, exit_contexts(series))   # .fills and .drops

Two decisions shape everything here. Exits are either resting orders filled
inside a bar or decisions taken on its close and filled at the next open, and
which one a decision is, is a field on it rather than something the execution
layer infers. And when several resting orders are touched in the same bar, the
order of events is unknown, so it is a stated policy — pessimistic by default,
because an optimistic default is how a backtest reports money that never existed.
"""

from trading_system.exit.base import (
    REASON_PRIORITY,
    ExitDecision,
    ExitDropReason,
    ExitKind,
    ExitPriority,
    ExitReason,
    ExitRule,
    ExitTrigger,
    IntrabarPolicy,
    StopModifier,
    empty_drop_counts,
)
from trading_system.exit.context import ExitContext, Tick, exit_contexts
from trading_system.exit.fills import (
    approaches_from_below,
    resting_fill_price,
    resting_order_filled,
)
from trading_system.exit.plan import (
    BarOutcome,
    DeferredExit,
    ExitFill,
    ExitPlan,
    ExitRun,
)
from trading_system.exit.position import WHOLE, ClosedLeg, ManagedPosition
from trading_system.exit.rules import (
    ATRStop,
    AtrTrail,
    BreakevenMove,
    Chandelier,
    FixedRR,
    MaTrail,
    PartialClose,
    PartialRung,
    ProtectiveStop,
    SignalReverseExit,
    StructureStop,
    SwingTrail,
    TimeExit,
    TimeExitMode,
    TrailingSource,
    TrailingStop,
)

__all__ = [
    "REASON_PRIORITY",
    "WHOLE",
    "ATRStop",
    "AtrTrail",
    "BarOutcome",
    "BreakevenMove",
    "Chandelier",
    "ClosedLeg",
    "DeferredExit",
    "ExitContext",
    "ExitDecision",
    "ExitDropReason",
    "ExitFill",
    "ExitKind",
    "ExitPlan",
    "ExitPriority",
    "ExitReason",
    "ExitRule",
    "ExitRun",
    "ExitTrigger",
    "FixedRR",
    "IntrabarPolicy",
    "ManagedPosition",
    "MaTrail",
    "PartialClose",
    "PartialRung",
    "ProtectiveStop",
    "SignalReverseExit",
    "StopModifier",
    "StructureStop",
    "SwingTrail",
    "Tick",
    "TimeExit",
    "TimeExitMode",
    "TrailingSource",
    "TrailingStop",
    "approaches_from_below",
    "empty_drop_counts",
    "exit_contexts",
    "resting_fill_price",
    "resting_order_filled",
]
