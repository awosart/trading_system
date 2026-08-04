"""The exit rule library.

Rules split into two roles, and the split is load-bearing rather than cosmetic:

* :class:`~trading_system.exit.base.ExitRule` implementations close all or part
  of a position: :class:`ProtectiveStop`, :class:`FixedRR`,
  :class:`PartialClose`, :class:`TimeExit`, :class:`SignalReverseExit`.
* :class:`~trading_system.exit.base.StopModifier` implementations move the
  position's stop and close nothing: :class:`ATRStop`, :class:`StructureStop`,
  :class:`TrailingStop`, :class:`BreakevenMove`.

Exactly one rule converts a stop level into an exit —
:class:`~trading_system.exit.rules.protective_stop.ProtectiveStop` — so there is
never more than one stop level in the market for a position, regardless of how
many modifiers are proposing levels to it.
"""

from trading_system.exit.rules.atr_stop import ATRStop
from trading_system.exit.rules.breakeven import BreakevenMove
from trading_system.exit.rules.fixed_rr import FixedRR
from trading_system.exit.rules.partial_close import PartialClose, PartialRung
from trading_system.exit.rules.protective_stop import ProtectiveStop
from trading_system.exit.rules.signal_reverse import SignalReverseExit
from trading_system.exit.rules.structure_stop import StructureStop
from trading_system.exit.rules.time_exit import TimeExit, TimeExitMode
from trading_system.exit.rules.trailing_stop import (
    AtrTrail,
    Chandelier,
    MaTrail,
    SwingTrail,
    TrailingSource,
    TrailingStop,
)

__all__ = [
    "ATRStop",
    "AtrTrail",
    "BreakevenMove",
    "Chandelier",
    "FixedRR",
    "MaTrail",
    "PartialClose",
    "PartialRung",
    "ProtectiveStop",
    "SignalReverseExit",
    "StructureStop",
    "SwingTrail",
    "TimeExit",
    "TimeExitMode",
    "TrailingSource",
    "TrailingStop",
]
