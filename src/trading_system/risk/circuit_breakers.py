"""When to stop trading altogether, and when to start again.

Four breakers, in the order a run is likely to hit them: a loss limit for the
day, the week and the month, a pause after a run of consecutive losses, and a
pause when fills come back far worse than they were priced at.

**The period breakers hold no "blocked" flag.** Each recomputes from the trade
ledger every time it is asked: "the P&L booked inside the current trading day is
at or below minus the limit". A mutable flag would need clearing at the right
instant, and a clear that is forgotten blocks the remainder of the run — the
failure is silent and looks exactly like a strategy that stopped finding setups.
Derived from the ledger, the reset is not an event anyone has to remember: it is
a consequence of the day label changing.

**The trading day is configurable, and it is an IANA zone plus a local time.**
Never UTC midnight by default and never a stored UTC offset. Prop firms each
define their own boundary — midnight in the firm's own city, or 17:00 New York
tracking the CME roll — and getting it wrong by a few hours moves trades between
days under the one rule that closes accounts. A stored offset would be right for
half the year: 17:00 New York is 22:00 UTC in winter and 21:00 UTC in summer. The
boundary is :class:`~trading_system.data.resample.DayOrigin` and the day label
comes from :func:`~trading_system.data.resample.trading_day` — the same function
a session VWAP resets on, so the two can never disagree.

The consecutive-loss breaker is the one exception to statelessness, because its
pause is measured in **bars** rather than in time, and bars are not recoverable
from timestamps alone. It records the bar index the pause began at, which is why
``bar_index`` is an argument to the engine's evaluation.

The slippage breaker's inputs do not exist yet: nothing in the system executes
orders until P12, so nothing calls :meth:`CircuitBreakers.record_fill`. The
interface is what P12 will call and is tested against synthetic reports; it is
written down rather than implied so that the producer arriving later has a shape
to fill rather than a decision to make.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import IntEnum

from pydantic import BaseModel, ConfigDict, Field

from trading_system.core.logging import get_logger
from trading_system.core.types import ensure_utc
from trading_system.data.resample import FX_DAY_ORIGIN, DayOrigin, trading_day
from trading_system.risk.models import RiskReason

logger = get_logger(__name__)


class Weekday(IntEnum):
    """Day a trading week starts on, matching :meth:`datetime.date.weekday`."""

    MONDAY = 0
    TUESDAY = 1
    WEDNESDAY = 2
    THURSDAY = 3
    FRIDAY = 4
    SATURDAY = 5
    SUNDAY = 6


@dataclass(frozen=True)
class ClosedTrade:
    """One realised trade, as the breakers need to see it.

    Attributes:
        closed_at: When the trade was booked, tz-aware. What places it in a
            trading day, week and month.
        pnl: Realised result in account currency, signed. Net of costs where the
            caller knows them — the breakers compare it against a limit and do
            not care how it was assembled.
        symbol: Instrument, for forensics.
        strategy_id: Strategy, for forensics.
    """

    closed_at: datetime
    pnl: Decimal
    symbol: str = ""
    strategy_id: str = ""

    def __post_init__(self) -> None:
        """Normalise the close time.

        Raises:
            ValueError: If ``closed_at`` is naive — a trade with no defined
                instant cannot be placed in a trading day.
        """
        object.__setattr__(self, "closed_at", ensure_utc(self.closed_at))


@dataclass(frozen=True)
class SlippageReport:
    """One fill, compared against the price it was expected at.

    Attributes:
        at: When the fill happened, tz-aware.
        slippage_points: How much worse than expected the fill was, in the
            instrument's points. **Signed, and positive means worse** — a fill
            better than expected is negative and never trips the breaker.
        symbol: Instrument filled.
    """

    at: datetime
    slippage_points: float
    symbol: str = ""

    def __post_init__(self) -> None:
        """Normalise the fill time.

        Raises:
            ValueError: If ``at`` is naive.
        """
        object.__setattr__(self, "at", ensure_utc(self.at))


class CircuitBreakerConfig(BaseModel):
    """When trading stops, and for how long.

    Attributes:
        trading_day: Where the trading day starts — an IANA zone and a local
            wall-clock time. Defaults to the FX convention of 17:00 New York; a
            prop account should set the firm's own boundary, which is the whole
            reason this is a parameter.
        week_starts_on: Day the trading week rolls over. Sunday by default,
            following the firm conventions this is most often measured against
            rather than the ISO calendar.
        max_daily_loss_pct: Fraction of the day's starting equity that may be
            lost before trading stops until the next trading day. ``None``
            disables the breaker.
        max_weekly_loss_pct: The same, per trading week.
        max_monthly_loss_pct: The same, per calendar month of the trading day.
        max_consecutive_losses: Losing trades in a row that trigger a pause.
            ``None`` disables it.
        consecutive_loss_pause_bars: How many bars the pause lasts.
        max_slippage_points: Slippage on a single fill, in points, that triggers
            a pause. ``None`` disables it.
        slippage_pause_bars: How many bars that pause lasts.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    trading_day: DayOrigin = FX_DAY_ORIGIN
    week_starts_on: Weekday = Weekday.SUNDAY
    max_daily_loss_pct: float | None = Field(default=0.05, gt=0, le=1)
    max_weekly_loss_pct: float | None = Field(default=0.08, gt=0, le=1)
    max_monthly_loss_pct: float | None = Field(default=0.10, gt=0, le=1)
    max_consecutive_losses: int | None = Field(default=None, gt=0)
    consecutive_loss_pause_bars: int = Field(default=0, ge=0)
    max_slippage_points: float | None = Field(default=None, gt=0)
    slippage_pause_bars: int = Field(default=0, ge=0)


@dataclass(frozen=True)
class BreakerTrip:
    """A breaker saying no.

    Attributes:
        reason: Which breaker.
        detail: The numbers behind it.
    """

    reason: RiskReason
    detail: str


def week_label(day: date, *, starts_on: Weekday) -> date:
    """The date labelling the trading week containing ``day``.

    Args:
        day: A trading-day label.
        starts_on: Weekday the week rolls over on.

    Returns:
        The date of that week's first day.
    """
    offset = (day.weekday() - int(starts_on)) % 7
    return day.fromordinal(day.toordinal() - offset)


class CircuitBreakers:
    """Evaluates whether trading is permitted at all, before anything is sized."""

    __slots__ = (
        "_config",
        "_pause_reason",
        "_paused_until_bar",
        "_slippage_events",
        "_streak_armed_after",
    )

    def __init__(self, config: CircuitBreakerConfig | None = None) -> None:
        """Configure the breakers.

        Args:
            config: Limits and pause lengths. Defaults enable the three period
                limits and leave the consecutive-loss and slippage pauses off,
                since both need a length chosen against a real timeframe.
        """
        self._config = config if config is not None else CircuitBreakerConfig()
        self._paused_until_bar: int | None = None
        self._pause_reason: RiskReason | None = None
        self._slippage_events: list[SlippageReport] = []
        self._streak_armed_after = 0

    def __repr__(self) -> str:
        """Compact description naming the day boundary."""
        origin = self._config.trading_day
        return (
            f"CircuitBreakers(day={origin.tz}@{origin.at}, paused_until={self._paused_until_bar})"
        )

    @property
    def config(self) -> CircuitBreakerConfig:
        """The configured limits."""
        return self._config

    @property
    def slippage_events(self) -> tuple[SlippageReport, ...]:
        """Fills that exceeded the slippage threshold, since the last reset."""
        return tuple(self._slippage_events)

    def reset(self) -> None:
        """Clear the bar-counted pauses, for a new run or walk-forward fold."""
        self._paused_until_bar = None
        self._pause_reason = None
        self._slippage_events.clear()
        self._streak_armed_after = 0

    def record_fill(self, report: SlippageReport) -> bool:
        """Take one execution report, and pause trading if it was abnormal.

        Nothing calls this before P12: no part of the system executes an order
        yet. It is the shape that layer will fill.

        Args:
            report: The fill and how much worse than expected it came back.

        Returns:
            Whether this report tripped the breaker.
        """
        threshold = self._config.max_slippage_points
        if threshold is None or report.slippage_points <= threshold:
            return False
        self._slippage_events.append(report)
        logger.warning(
            "risk.slippage_anomaly",
            symbol=report.symbol,
            slippage_points=report.slippage_points,
            threshold=threshold,
        )
        return True

    def note_bar(self, bar_index: int) -> None:
        """Arm the slippage pause against a bar index.

        Slippage is observed at execution, which knows nothing of bar indices,
        so the pause is anchored the next time the engine sees a bar. Splitting
        it this way keeps :meth:`record_fill` callable from the execution layer
        without that layer having to carry a bar counter.

        Args:
            bar_index: Index of the bar currently being evaluated.
        """
        if self._slippage_events and self._pause_reason is not RiskReason.SLIPPAGE_ANOMALY_PAUSE:
            self._pause(
                bar_index, self._config.slippage_pause_bars, RiskReason.SLIPPAGE_ANOMALY_PAUSE
            )

    def check(
        self,
        *,
        at: datetime,
        bar_index: int,
        equity: Decimal,
        trades: Sequence[ClosedTrade],
    ) -> BreakerTrip | None:
        """Whether trading is blocked at this instant.

        Args:
            at: The signal's bar close time, tz-aware. Everything is measured
                against this rather than against the wall clock, so a backtest
                reproduces exactly.
            bar_index: Index of the current bar, for the bar-counted pauses.
            equity: Account equity now, which the percentage limits are taken of.
            trades: Realised trades so far. Only those inside the current period
                are counted; the caller may pass the whole history.

        Returns:
            The trip, or ``None`` when trading is permitted.

        Raises:
            ValueError: If ``at`` is naive.
        """
        moment = ensure_utc(at)
        self.note_bar(bar_index)

        paused = self._paused_trip(bar_index)
        if paused is not None:
            return paused

        today = trading_day(moment, self._config.trading_day)
        periods = (
            (
                self._config.max_daily_loss_pct,
                RiskReason.DAILY_LOSS_LIMIT,
                "trading day",
                lambda day: day == today,
            ),
            (
                self._config.max_weekly_loss_pct,
                RiskReason.WEEKLY_LOSS_LIMIT,
                "trading week",
                lambda day: (
                    week_label(day, starts_on=self._config.week_starts_on)
                    == week_label(today, starts_on=self._config.week_starts_on)
                ),
            ),
            (
                self._config.max_monthly_loss_pct,
                RiskReason.MONTHLY_LOSS_LIMIT,
                "month",
                lambda day: (day.year, day.month) == (today.year, today.month),
            ),
        )

        for limit_pct, reason, label, in_period in periods:
            if limit_pct is None:
                continue
            realised = sum(
                (
                    trade.pnl
                    for trade in trades
                    if in_period(trading_day(trade.closed_at, self._config.trading_day))
                ),
                Decimal(0),
            )
            allowance = equity * Decimal(str(limit_pct))
            if realised <= -allowance:
                return BreakerTrip(
                    reason=reason,
                    detail=(
                        f"{realised} realised this {label} against a limit of {-allowance} "
                        f"({limit_pct:.4%} of equity {equity}); trading stops until the "
                        f"{label} rolls over at {self._config.trading_day.at} "
                        f"{self._config.trading_day.tz}"
                    ),
                )

        streak = self._losing_streak(trades)
        maximum = self._config.max_consecutive_losses
        # Re-arm only once the ledger has actually grown. Without this the pause
        # is permanent rather than M bars long: the streak that caused it is
        # still the tail of an unchanged ledger the moment the pause expires, so
        # every subsequent bar would start a fresh one and trading would resume
        # only after a win — which is not what "pause for M bars" means.
        if maximum is not None and streak >= maximum and len(trades) > self._streak_armed_after:
            self._streak_armed_after = len(trades)
            self._pause(
                bar_index,
                self._config.consecutive_loss_pause_bars,
                RiskReason.CONSECUTIVE_LOSS_PAUSE,
            )
            trip = self._paused_trip(bar_index)
            if trip is not None:
                return trip

        return None

    def _pause(self, bar_index: int, bars: int, reason: RiskReason) -> None:
        """Block trading for ``bars`` bars from ``bar_index``.

        Args:
            bar_index: Bar the pause starts on.
            bars: How long it lasts. Zero means no pause at all.
            reason: Which breaker imposed it.
        """
        if bars <= 0:
            return
        self._paused_until_bar = bar_index + bars
        self._pause_reason = reason
        logger.info("risk.paused", reason=reason.value, until_bar=self._paused_until_bar)

    def _paused_trip(self, bar_index: int) -> BreakerTrip | None:
        """The active pause, if any, expiring it once its bars have elapsed.

        Args:
            bar_index: Index of the current bar.

        Returns:
            The trip while the pause holds, ``None`` once it has lapsed.
        """
        if self._paused_until_bar is None or self._pause_reason is None:
            return None
        if bar_index >= self._paused_until_bar:
            self._paused_until_bar = None
            self._pause_reason = None
            return None
        return BreakerTrip(
            reason=self._pause_reason,
            detail=(
                f"paused until bar {self._paused_until_bar}, currently bar {bar_index} "
                f"({self._paused_until_bar - bar_index} to go)"
            ),
        )

    @staticmethod
    def _losing_streak(trades: Sequence[ClosedTrade]) -> int:
        """How many of the most recent trades were losses, consecutively.

        A break-even trade ends the streak rather than extending it: the streak
        is meant to detect a run of the strategy being wrong, and a scratch is
        not evidence of that.

        Args:
            trades: Realised trades, oldest first.

        Returns:
            The length of the trailing run of losses.
        """
        streak = 0
        for trade in reversed(trades):
            if trade.pnl >= 0:
                break
            streak += 1
        return streak
