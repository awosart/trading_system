"""Close a position for reasons that have nothing to do with price.

All three variants below decide on ``close(t)`` and can only fill at
``open(t+1)`` — there is no level to rest an order at, so
:attr:`~trading_system.exit.base.ExitTrigger.BAR_CLOSE` is the only trigger any
of them can honestly use. The session-boundary and before-weekend checks read
only bar timestamps and :class:`~trading_system.data.sessions.Session` windows,
never price — the same category of computation as
:attr:`~trading_system.entry.context.BarContext.bar_close_ts` itself
(``open + duration``), which P06 already treats as arithmetic, not lookahead:
a calendar is knowable arbitrarily far in advance, unlike a price.
"""

from decimal import Decimal
from enum import StrEnum

from trading_system.core.exceptions import ValidationError
from trading_system.data.sessions import AssetClass, Session, TradingCalendar, is_in_session
from trading_system.exit.base import ExitDecision, ExitKind, ExitReason, ExitTrigger
from trading_system.exit.context import ExitContext
from trading_system.exit.position import ManagedPosition


class TimeExitMode(StrEnum):
    """Which time-based condition closes the position."""

    #: Close once the position has been open for this many bars, counting the
    #: bar it opened on as bar 1.
    MAX_BARS_HELD = "MAX_BARS_HELD"

    #: Close on the last bar a named session is active — the bar whose open was
    #: inside the session and whose close is not.
    SESSION_CLOSE = "SESSION_CLOSE"

    #: Close on the last bar before the market stops accepting orders for the
    #: weekend.
    BEFORE_WEEKEND = "BEFORE_WEEKEND"


class TimeExit:
    """Close full, on the bar's close, for a reason unrelated to price."""

    def __init__(
        self,
        mode: TimeExitMode,
        *,
        max_bars_held: int | None = None,
        session: Session | None = None,
        asset_class: AssetClass = AssetClass.FX,
    ) -> None:
        """Configure one time-based condition.

        Args:
            mode: Which condition to check.
            max_bars_held: Bars to hold before closing. Required by, and only
                read under, ``MAX_BARS_HELD``.
            session: Session whose close ends the trade. Required by, and only
                read under, ``SESSION_CLOSE``.
            asset_class: Determines the weekly close used by
                ``BEFORE_WEEKEND``. Ignored by the other two modes.

        Raises:
            ValidationError: If the mode's required parameter is missing or
                ``max_bars_held`` is not positive.
        """
        if mode is TimeExitMode.MAX_BARS_HELD:
            if max_bars_held is None or max_bars_held < 1:
                raise ValidationError(
                    f"MAX_BARS_HELD requires a positive max_bars_held, got {max_bars_held!r}"
                )
        elif mode is TimeExitMode.SESSION_CLOSE and session is None:
            raise ValidationError("SESSION_CLOSE requires a session")

        self._mode = mode
        self._max_bars_held = max_bars_held
        self._session = session
        self._asset_class = asset_class
        self._bars_held = 0

    @property
    def name(self) -> str:
        """Stable identifier, e.g. ``time_exit_max_bars_held``."""
        return f"time_exit_{self._mode.value.lower()}"

    @property
    def partial_fractions(self) -> tuple[Decimal, ...]:
        """None: a time exit always closes in full."""
        return ()

    def on_bar(self, position: ManagedPosition, ctx: ExitContext) -> ExitDecision | None:  # noqa: ARG002
        """Close in full if this bar's close meets the configured condition.

        Args:
            position: Position being managed. Unused — every condition here is
                a function of time alone, never of the position's price or side.
            ctx: View of the closed bar ``t``.

        Returns:
            A ``BAR_CLOSE`` full exit, or ``None``.
        """
        self._bars_held += 1
        if self._mode is TimeExitMode.MAX_BARS_HELD:
            if self._max_bars_held is None:
                raise ValidationError("MAX_BARS_HELD requires max_bars_held")
            fired = self._bars_held >= self._max_bars_held
        elif self._mode is TimeExitMode.SESSION_CLOSE:
            if self._session is None:
                raise ValidationError("SESSION_CLOSE requires a session")
            fired = is_in_session(ctx.bar_open_ts, self._session) and not is_in_session(
                ctx.bar_close_ts, self._session
            )
        else:
            calendar = TradingCalendar(asset_class=self._asset_class)
            next_open = ctx.bar_close_ts + ctx.timeframe.duration
            fired = calendar.is_open(ctx.bar_close_ts) and not calendar.is_open(next_open)

        if not fired:
            return None
        return ExitDecision(
            reason=ExitReason.TIME_EXIT,
            kind=ExitKind.FULL,
            trigger=ExitTrigger.BAR_CLOSE,
            context={"mode": self._mode.value, "bars_held": self._bars_held},
        )

    def reset(self) -> None:
        """Zero the bar counter for a fresh run."""
        self._bars_held = 0
