"""What the cost model is told about the moment it is pricing.

**This module exists so that :mod:`trading_system.execution` never imports the
feature pipeline.** Slippage scales with volatility, volatility is an ATR, and an
ATR is a P03 indicator — so the naive implementation reaches into
:mod:`trading_system.features` and Execution acquires a dependency on every
indicator in the system. :class:`MarketState` breaks that the same way P07 broke
the Exit Engine's dependency on the Entry Engine: the value arrives as a plain
scalar, computed by whoever already holds both, and this layer only has to
accept it.

The assembler is the backtest orchestrator (P13). It already builds the
``BarSeries`` that Entry and Exit walk, so reading two feature columns and
dividing them is one line on its side. The import graph this layer is held to::

    execution/ imports:      core.types, core.instruments, data.sessions, data.resample
    execution/ never imports: features, entry, exit, risk

``session`` is *not* a field. It is derived here through
:func:`~trading_system.data.sessions.session_of`, which is pure arithmetic on a
timestamp — the same class of computation P07 allowed ``TimeExit`` to do with
``TradingCalendar`` — so making it a field would add one more thing an assembler
can fill in wrongly for no gain. ``atr_ratio`` and ``news_severity`` are fields
precisely because neither is derivable from a timestamp.

**``atr_ratio`` is a ratio, not an ATR.** It is ``atr(t) / mean(atr, N)(t)``,
dimensionless and approximately ``1.0`` in typical volatility. That is what lets
the slippage coefficient ``k`` carry one meaning across EURUSD and BTCUSD; a raw
ATR would need ``k`` recalibrated per instrument, and the recalibration would
silently not happen the first time an instrument was added.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from trading_system.core.types import ensure_utc
from trading_system.data.sessions import Session, session_of


class NewsSeverity(StrEnum):
    """How disruptive the scheduled news around this instant is.

    A graded scale rather than a boolean because the cost difference between a
    second-tier release and an FOMC decision is larger than the difference
    between a quiet bar and a second-tier release. Every value is ``NONE`` until
    P16 supplies a calendar; the grades exist now so that the configuration
    shape does not change when it does.
    """

    NONE = "NONE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


@dataclass(frozen=True)
class MarketState:
    """Conditions at the instant an order is being priced.

    Attributes:
        ts: The instant, tz-aware UTC. For a fill at the open of bar ``t+1``
            this is that bar's open time — the moment the fill happens, not the
            moment the decision was taken.
        atr_ratio: Current ATR over its own rolling mean, dimensionless and
            around ``1.0`` in typical conditions. ``None`` when the indicator
            had not warmed up, which the cost model degrades on and **counts**
            rather than silently reading as ``1.0``.
        quoted_spread_points: Observed spread in points, when the run has real
            quote data. Takes priority over the instrument's
            ``typical_spread_points`` whenever the configured spread source asks
            for it. ``None`` on a run without quote data, which is the normal
            case.
        news_severity: Grade of scheduled news around ``ts``. ``NONE`` until a
            news calendar exists to say otherwise.
    """

    ts: datetime
    atr_ratio: float | None = None
    quoted_spread_points: float | None = None
    news_severity: NewsSeverity = NewsSeverity.NONE

    def __post_init__(self) -> None:
        """Normalise the timestamp and reject impossible market readings.

        Raises:
            ValueError: If ``ts`` is naive, ``atr_ratio`` is not positive, or
                ``quoted_spread_points`` is negative. A non-positive ATR ratio
                is a division that went wrong upstream, and letting it through
                would produce a negative spread multiplier — a cost model that
                pays the trader.
        """
        object.__setattr__(self, "ts", ensure_utc(self.ts))
        if self.atr_ratio is not None and self.atr_ratio <= 0:
            raise ValueError(
                f"atr_ratio must be positive, got {self.atr_ratio}; use None for "
                "'not warmed up' rather than a zero, which reads as 'no volatility'"
            )
        if self.quoted_spread_points is not None and self.quoted_spread_points < 0:
            raise ValueError(
                f"quoted_spread_points must not be negative, got {self.quoted_spread_points}"
            )

    @property
    def sessions(self) -> frozenset[Session]:
        """Every trading session active at :attr:`ts`.

        Derived, never supplied: a timestamp determines this completely, so a
        field would only be a way for an assembler to disagree with the clock.
        """
        return frozenset(session_of(self.ts))
