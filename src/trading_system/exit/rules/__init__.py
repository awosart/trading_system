"""The exit rule library.

Rules split into two roles, and the split is load-bearing rather than cosmetic:

* :class:`~trading_system.exit.base.ExitRule` implementations close all or part
  of a position.
* :class:`~trading_system.exit.base.StopModifier` implementations move the
  position's stop and close nothing. Trailing and breakeven rules are these.

Exactly one rule converts a stop level into an exit —
:class:`~trading_system.exit.rules.protective_stop.ProtectiveStop` — so there is
never more than one stop level in the market for a position.
"""

from trading_system.exit.rules.fixed_rr import FixedRR
from trading_system.exit.rules.protective_stop import ProtectiveStop

__all__ = ["FixedRR", "ProtectiveStop"]
