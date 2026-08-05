"""Execution: what an order costs and where it fills.

Stage 1 is the modelling half — costs and fill models. Broker adapters and the
order router are stage 2, and nothing here talks to a venue.

The import discipline this package is held to, and which its tests assert::

    execution/ imports:       core, data.sessions, data.resample, exit.fills
    execution/ never imports: features, entry, risk

Volatility reaches the cost model as a scalar on
:class:`~trading_system.execution.market_state.MarketState`, assembled by
whoever already holds the feature frame, rather than by importing the feature
pipeline — the same move that kept the Entry Engine out of the Exit Engine.
``exit.fills`` is the one exception, and only for
:func:`~trading_system.exit.fills.approaches_from_below`, which states which side
of a level an order is reached from. That is a fact about order types, not about
exits, and duplicating it would give the system two tables that can disagree
about which way a stop fills.
"""

from trading_system.execution.config import (
    CostConfig,
    ExecutionRunConfig,
    GapConfig,
    LimitFillConfig,
    SlippageConfig,
    SlippageParams,
    SpreadConfig,
    SpreadSource,
    load_execution_config,
)
from trading_system.execution.costs import (
    CostDegradation,
    CostModel,
    CostStats,
    SwapAccrual,
    accrue_swap,
    realized_points,
)
from trading_system.execution.fill_model import (
    GapFill,
    LimitTouch,
    NextBarOpen,
    RestingFill,
    RestingOrderModel,
    SameBarClose,
    order_from_fill,
)
from trading_system.execution.market_state import MarketState, NewsSeverity
from trading_system.execution.orders import ExecutionOrder, Fill
from trading_system.execution.rng import fill_rng, fill_seed

__all__ = [
    "CostConfig",
    "CostDegradation",
    "CostModel",
    "CostStats",
    "ExecutionOrder",
    "ExecutionRunConfig",
    "Fill",
    "GapConfig",
    "GapFill",
    "LimitFillConfig",
    "LimitTouch",
    "MarketState",
    "NewsSeverity",
    "NextBarOpen",
    "RestingFill",
    "RestingOrderModel",
    "SameBarClose",
    "SlippageConfig",
    "SlippageParams",
    "SpreadConfig",
    "SpreadSource",
    "SwapAccrual",
    "accrue_swap",
    "fill_rng",
    "fill_seed",
    "realized_points",
    "load_execution_config",
    "order_from_fill",
]
