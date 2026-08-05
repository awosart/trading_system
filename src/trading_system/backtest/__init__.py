"""Backtesting: one chronological event loop, and a portfolio that marks every bar.

The layer where every engine finally meets. Its two structural commitments:

**Lookahead is a raised exception, not a convention.** Streams are merged on bar
*close* — never on the ``timestamp`` field, which is a bar's open, and merging on
which would publish a daily bar twenty-four hours before it finishes — and
:class:`~trading_system.backtest.engine.BarStore` refuses to hand out a bar the
merged clock has not reached.

**And it is proven rather than asserted.** ``tests/test_no_lookahead.py`` runs
the shipped strategy twice over truncated, poisoned and permuted data and demands
the results agree byte for byte; each of its checks has been made to fail against
a leak deliberately introduced into this package and then reverted. What the
checks agree *about* is :func:`~trading_system.backtest.reproducibility.digest`,
which is also what gives a run its id and what makes two runs of one id a
comparison rather than an impression.

**The same strategy code runs here and live.** A bar arrives, engines read a
context that counts backwards only, and orders execute no earlier than the next
open. What differs in live trading is the broker behind the fill, not the loop.

The phase order inside one instant — financing, deferred fills, recognition,
exits, marking, snapshot, sizing — is written out in
:mod:`trading_system.backtest.engine`. It decides the equity curve and it decides
whether the Risk Engine sees a limit freed by an exit, so it lives in one
readable place rather than being spread across the callers.
"""

from trading_system.backtest.clock import StreamKey, bar_close_ts, day_close_ts
from trading_system.backtest.config import BacktestConfig
from trading_system.backtest.engine import (
    BacktestEngine,
    BarEvent,
    BarStore,
    DataHandler,
    Instant,
    LookaheadError,
    Phases,
)
from trading_system.backtest.orchestrator import (
    BacktestResult,
    Orchestrator,
    PendingEntry,
    SignalDrop,
    StrategyBinding,
)
from trading_system.backtest.parallel import run_one, run_parallel, run_sequential
from trading_system.backtest.portfolio import (
    EquityPoint,
    OpenPosition,
    Portfolio,
    TradeRecord,
)
from trading_system.backtest.reproducibility import (
    CodeVersion,
    ReproducibilityError,
    RunManifest,
    StoredRun,
    code_version,
    digest,
    digest_frame,
    read_run,
    result_digest,
    write_run,
)
from trading_system.backtest.spec import RunInputs

__all__ = [
    "BacktestConfig",
    "BacktestEngine",
    "BacktestResult",
    "BarEvent",
    "BarStore",
    "CodeVersion",
    "DataHandler",
    "EquityPoint",
    "Instant",
    "LookaheadError",
    "OpenPosition",
    "Orchestrator",
    "PendingEntry",
    "Phases",
    "Portfolio",
    "ReproducibilityError",
    "RunInputs",
    "RunManifest",
    "SignalDrop",
    "StoredRun",
    "StrategyBinding",
    "StreamKey",
    "TradeRecord",
    "bar_close_ts",
    "code_version",
    "day_close_ts",
    "digest",
    "digest_frame",
    "read_run",
    "result_digest",
    "run_one",
    "run_parallel",
    "run_sequential",
    "write_run",
]
