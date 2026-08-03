"""Trading sessions and market calendars.

Every boundary is expressed as a local wall-clock time in an IANA timezone and
resolved through :mod:`zoneinfo`. Fixed UTC offsets are never used: the London
session opens at 08:00 London time all year, which is 08:00 UTC in winter and
07:00 UTC in summer. Hard-coding either value silently misaligns half the year.
"""

from dataclasses import dataclass, field
from datetime import date, datetime, time
from enum import StrEnum
from zoneinfo import ZoneInfo

NEW_YORK = "America/New_York"


class Session(StrEnum):
    """A named FX trading session."""

    SYDNEY = "SYDNEY"
    TOKYO = "TOKYO"
    LONDON = "LONDON"
    NEWYORK = "NEWYORK"
    LONDON_NY_OVERLAP = "LONDON_NY_OVERLAP"


class AssetClass(StrEnum):
    """Asset class, which determines when the market is open at all."""

    FX = "FX"
    EQUITY = "EQUITY"
    CRYPTO = "CRYPTO"


@dataclass(frozen=True)
class SessionWindow:
    """Daily local-time window during which a session is active.

    Attributes:
        tz: IANA timezone the window is defined in.
        opens: Local time the session starts, inclusive.
        closes: Local time the session ends, exclusive.
    """

    tz: str
    opens: time
    closes: time

    def contains(self, timestamp: datetime) -> bool:
        """Whether ``timestamp`` falls inside this window on a weekday.

        Args:
            timestamp: A tz-aware instant.

        Returns:
            ``True`` when the instant's local time is within the window and the
            local date is a weekday.

        Raises:
            ValueError: If ``timestamp`` is naive.
        """
        if timestamp.tzinfo is None:
            raise ValueError("timestamp must be tz-aware")
        local = timestamp.astimezone(ZoneInfo(self.tz))
        if local.weekday() >= 5:
            return False
        return self.opens <= local.time() < self.closes


# Conventional retail-FX session hours, stated in each centre's own local time.
SESSION_WINDOWS: dict[Session, SessionWindow] = {
    Session.SYDNEY: SessionWindow("Australia/Sydney", time(8, 0), time(17, 0)),
    Session.TOKYO: SessionWindow("Asia/Tokyo", time(9, 0), time(18, 0)),
    Session.LONDON: SessionWindow("Europe/London", time(8, 0), time(17, 0)),
    Session.NEWYORK: SessionWindow(NEW_YORK, time(8, 0), time(17, 0)),
}

_BASE_SESSIONS = (Session.SYDNEY, Session.TOKYO, Session.LONDON, Session.NEWYORK)


def session_of(timestamp: datetime) -> set[Session]:
    """Return every session active at ``timestamp``.

    Sessions overlap, so the result is a set. ``LONDON_NY_OVERLAP`` is derived:
    it appears exactly when both London and New York are open, which is the
    highest-liquidity window of the FX day.

    Args:
        timestamp: A tz-aware instant.

    Returns:
        The set of active sessions, empty outside all of them.

    Raises:
        ValueError: If ``timestamp`` is naive.
    """
    active = {session for session in _BASE_SESSIONS if SESSION_WINDOWS[session].contains(timestamp)}
    if Session.LONDON in active and Session.NEWYORK in active:
        active.add(Session.LONDON_NY_OVERLAP)
    return active


def is_in_session(timestamp: datetime, session: Session) -> bool:
    """Whether ``session`` is active at ``timestamp``.

    Args:
        timestamp: A tz-aware instant.
        session: Session to test.

    Returns:
        ``True`` when the session is active.

    Raises:
        ValueError: If ``timestamp`` is naive.
    """
    if session is Session.LONDON_NY_OVERLAP:
        return Session.LONDON_NY_OVERLAP in session_of(timestamp)
    return SESSION_WINDOWS[session].contains(timestamp)


@dataclass(frozen=True)
class TradingCalendar:
    """Which instants a market accepts orders at, by asset class.

    This models the weekly open/closed cycle and full-day holidays only.
    Intraday exchange hours are out of scope; use :func:`session_of` for those.

    Attributes:
        asset_class: Determines the weekly rule applied.
        holidays: Full days on which the market does not trade, expressed as
            dates in :attr:`reference_tz`. Supplied by the caller rather than
            hard-coded, since holiday sets differ per venue and per year.
    """

    asset_class: AssetClass
    holidays: frozenset[date] = field(default_factory=frozenset)

    @property
    def reference_tz(self) -> str:
        """Timezone in which this calendar's day boundaries are defined."""
        return NEW_YORK

    def is_open(self, timestamp: datetime) -> bool:
        """Whether the market trades at ``timestamp``.

        FX runs continuously from Sunday 17:00 New York to Friday 17:00 New York.
        Equities trade on weekdays. Crypto never closes.

        Args:
            timestamp: A tz-aware instant.

        Returns:
            ``True`` when the market is open.

        Raises:
            ValueError: If ``timestamp`` is naive.
        """
        if timestamp.tzinfo is None:
            raise ValueError("timestamp must be tz-aware")
        if self.asset_class is AssetClass.CRYPTO:
            return True

        local = timestamp.astimezone(ZoneInfo(self.reference_tz))
        if local.date() in self.holidays:
            return False

        weekday = local.weekday()
        if self.asset_class is AssetClass.EQUITY:
            return weekday < 5

        # FX: the week opens Sunday evening and closes Friday evening.
        if weekday == 5:  # Saturday
            return False
        if weekday == 4 and local.time() >= time(17, 0):  # Friday after the close
            return False
        return not (weekday == 6 and local.time() < time(17, 0))  # Sunday before the open
