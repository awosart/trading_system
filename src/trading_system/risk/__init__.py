"""The Risk Engine: how much to trade, and whether to trade at all.

Knows about signals, accounts and instruments. Knows nothing about indicators —
whichever strategy produced a signal, and on what, is invisible here.
"""

from trading_system.risk.conversion import (
    BarFxConverter,
    FxConverter,
    FxRateUnavailableError,
    SameCurrencyConverter,
    StaticFxConverter,
    convert,
)
from trading_system.risk.engine import RiskEngine, RiskEngineConfig
from trading_system.risk.models import AccountState, RiskDecision, RiskReason
from trading_system.risk.pnl import leg_pnl, realized_pnl
from trading_system.risk.stop_calculator import StopBufferConfig, calculate_stop

__all__ = [
    "AccountState",
    "BarFxConverter",
    "FxConverter",
    "FxRateUnavailableError",
    "RiskDecision",
    "RiskEngine",
    "RiskEngineConfig",
    "RiskReason",
    "SameCurrencyConverter",
    "StaticFxConverter",
    "StopBufferConfig",
    "calculate_stop",
    "convert",
    "leg_pnl",
    "realized_pnl",
]
