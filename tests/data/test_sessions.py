"""Session boundaries and market calendars, with DST as the headline concern."""

from datetime import UTC, date, datetime

import pytest

from trading_system.data.sessions import (
    AssetClass,
    Session,
    TradingCalendar,
    is_in_session,
    session_of,
)


def utc(year: int, month: int, day: int, hour: int, minute: int = 0) -> datetime:
    """Build a UTC instant."""
    return datetime(year, month, day, hour, minute, tzinfo=UTC)


class TestLondonDST:
    """London opens at 08:00 local all year; its UTC window moves twice a year."""

    def test_winter_window_is_utc_aligned(self) -> None:
        """In January London is on GMT, so 08:00 local is 08:00 UTC."""
        # 2024-01-10 is a Wednesday.
        assert not is_in_session(utc(2024, 1, 10, 7, 59), Session.LONDON)
        assert is_in_session(utc(2024, 1, 10, 8, 0), Session.LONDON)
        assert is_in_session(utc(2024, 1, 10, 16, 59), Session.LONDON)
        assert not is_in_session(utc(2024, 1, 10, 17, 0), Session.LONDON)

    def test_summer_window_shifts_one_hour_earlier_in_utc(self) -> None:
        """In July London is on BST, so 08:00 local is 07:00 UTC."""
        # 2024-07-10 is a Wednesday.
        assert not is_in_session(utc(2024, 7, 10, 6, 59), Session.LONDON)
        assert is_in_session(utc(2024, 7, 10, 7, 0), Session.LONDON)
        assert is_in_session(utc(2024, 7, 10, 15, 59), Session.LONDON)
        assert not is_in_session(utc(2024, 7, 10, 16, 0), Session.LONDON)

    def test_march_transition_moves_the_boundary(self) -> None:
        """UK DST began 2024-03-31; the Monday after opens an hour earlier in UTC."""
        # 2024-03-28 (Thu, GMT) vs 2024-04-01 (Mon, BST).
        assert not is_in_session(utc(2024, 3, 28, 7, 30), Session.LONDON)
        assert is_in_session(utc(2024, 4, 1, 7, 30), Session.LONDON)

    def test_october_transition_moves_it_back(self) -> None:
        """UK DST ended 2024-10-27; the Monday after opens an hour later in UTC."""
        # 2024-10-25 (Fri, BST) vs 2024-10-28 (Mon, GMT).
        assert is_in_session(utc(2024, 10, 25, 7, 30), Session.LONDON)
        assert not is_in_session(utc(2024, 10, 28, 7, 30), Session.LONDON)


def test_new_york_session_tracks_us_dst() -> None:
    """US DST began 2024-03-10, shifting NY's 08:00 open from 13:00 to 12:00 UTC."""
    assert not is_in_session(utc(2024, 3, 8, 12, 30), Session.NEWYORK)  # Fri, EST
    assert is_in_session(utc(2024, 3, 11, 12, 30), Session.NEWYORK)  # Mon, EDT


def test_london_ny_overlap_is_the_intersection() -> None:
    """The overlap appears exactly when both centres are open."""
    # 2024-07-10, BST/EDT: London 07:00-16:00 UTC, New York 12:00-21:00 UTC.
    assert session_of(utc(2024, 7, 10, 10, 0)) == {Session.LONDON}
    overlap = session_of(utc(2024, 7, 10, 13, 0))
    assert overlap == {Session.LONDON, Session.NEWYORK, Session.LONDON_NY_OVERLAP}
    assert session_of(utc(2024, 7, 10, 17, 0)) == {Session.NEWYORK}


def test_overlap_lasts_four_hours_in_both_dst_regimes() -> None:
    """The London/NY overlap is four hours long in summer and in winter."""
    for day, first_open, last_open in [
        (date(2024, 7, 10), 12, 15),  # BST/EDT: 12:00-16:00 UTC
        (date(2024, 1, 10), 13, 16),  # GMT/EST: 13:00-17:00 UTC
    ]:
        hours = [
            hour
            for hour in range(24)
            if Session.LONDON_NY_OVERLAP
            in session_of(datetime(day.year, day.month, day.day, hour, tzinfo=UTC))
        ]
        assert hours == list(range(first_open, last_open + 1))


def test_tokyo_and_sydney_use_their_own_zones() -> None:
    """Asia-Pacific sessions are evaluated in local time, spanning UTC midnight."""
    # 2024-07-10 09:00 Tokyo == 2024-07-10 00:00 UTC.
    assert is_in_session(utc(2024, 7, 10, 0, 0), Session.TOKYO)
    # 2024-07-10 08:00 Sydney (AEST, UTC+10) == 2024-07-09 22:00 UTC.
    assert is_in_session(utc(2024, 7, 9, 22, 0), Session.SYDNEY)


def test_sessions_are_closed_at_the_weekend() -> None:
    """2024-07-13 is a Saturday; no centre is open."""
    assert session_of(utc(2024, 7, 13, 12, 0)) == set()


def test_naive_timestamp_is_rejected() -> None:
    with pytest.raises(ValueError, match="tz-aware"):
        session_of(datetime(2024, 7, 10, 12, 0))  # noqa: DTZ001 - deliberately naive


class TestFXCalendar:
    """FX trades continuously from Sunday evening to Friday evening, New York time."""

    calendar = TradingCalendar(AssetClass.FX)

    def test_open_midweek(self) -> None:
        assert self.calendar.is_open(utc(2024, 7, 10, 3, 0))  # Wednesday

    def test_closed_on_saturday(self) -> None:
        assert not self.calendar.is_open(utc(2024, 7, 13, 12, 0))

    def test_closes_friday_evening_new_york(self) -> None:
        """2024-07-12 17:00 EDT is 21:00 UTC."""
        assert self.calendar.is_open(utc(2024, 7, 12, 20, 59))
        assert not self.calendar.is_open(utc(2024, 7, 12, 21, 0))

    def test_opens_sunday_evening_new_york(self) -> None:
        """2024-07-14 17:00 EDT is 21:00 UTC."""
        assert not self.calendar.is_open(utc(2024, 7, 14, 20, 59))
        assert self.calendar.is_open(utc(2024, 7, 14, 21, 0))

    def test_weekly_boundary_follows_dst(self) -> None:
        """In winter the Friday close is 22:00 UTC, an hour later than in summer."""
        winter = TradingCalendar(AssetClass.FX)
        assert winter.is_open(utc(2024, 1, 12, 21, 59))  # Friday, EST
        assert not winter.is_open(utc(2024, 1, 12, 22, 0))


def test_equity_calendar_excludes_weekends_and_holidays() -> None:
    calendar = TradingCalendar(AssetClass.EQUITY, frozenset({date(2024, 7, 4)}))
    assert calendar.is_open(utc(2024, 7, 3, 14, 0))
    assert not calendar.is_open(utc(2024, 7, 4, 14, 0))  # Independence Day
    assert not calendar.is_open(utc(2024, 7, 6, 14, 0))  # Saturday


def test_crypto_never_closes() -> None:
    calendar = TradingCalendar(AssetClass.CRYPTO)
    assert calendar.is_open(utc(2024, 7, 13, 3, 0))  # Saturday night
    assert calendar.is_open(utc(2024, 12, 25, 0, 0))


def test_holidays_apply_to_fx_too() -> None:
    calendar = TradingCalendar(AssetClass.FX, frozenset({date(2024, 12, 25)}))
    assert not calendar.is_open(utc(2024, 12, 25, 12, 0))


def test_calendar_rejects_naive_timestamp() -> None:
    with pytest.raises(ValueError, match="tz-aware"):
        TradingCalendar(AssetClass.FX).is_open(datetime(2024, 7, 10, 12))  # noqa: DTZ001
