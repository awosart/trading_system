"""Circuit breakers: when trading stops, and when it starts again.

The DST tests are the ones that matter most. A daily loss limit is the rule that
ends a prop account, and a boundary that is right for half the year moves trades
between days without anyone noticing until the account is closed.
"""

from datetime import UTC, datetime, time, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo

import pytest

from trading_system.data.resample import FX_DAY_ORIGIN, DayOrigin, trading_day
from trading_system.risk.circuit_breakers import (
    CircuitBreakerConfig,
    CircuitBreakers,
    ClosedTrade,
    SlippageReport,
    Weekday,
    week_label,
)
from trading_system.risk.models import RiskReason

EQUITY = Decimal("100000")
PRAGUE_MIDNIGHT = DayOrigin(tz="Europe/Prague", at=time(0, 0))


def loss(at: datetime, amount: str = "-1000") -> ClosedTrade:
    """A losing trade booked at ``at``."""
    return ClosedTrade(closed_at=at, pnl=Decimal(amount))


def win(at: datetime, amount: str = "1000") -> ClosedTrade:
    """A winning trade booked at ``at``."""
    return ClosedTrade(closed_at=at, pnl=Decimal(amount))


class TestTradingDayBoundary:
    """The boundary is an IANA zone plus a local time, and it tracks DST."""

    def test_utc_midnight_is_not_the_default(self) -> None:
        # A default that looks neutral but is wrong for every prop firm is worse
        # than one that is obviously a choice.
        assert CircuitBreakerConfig().trading_day == FX_DAY_ORIGIN

    def test_seventeen_hundred_new_york_is_twenty_two_utc_in_winter(self) -> None:
        # A day is labelled by the date it BEGAN on, so 16:59 New York on the
        # 15th still belongs to the day that opened at 17:00 on the 14th.
        before = datetime(2024, 1, 15, 21, 59, tzinfo=UTC)  # 16:59 EST
        after = datetime(2024, 1, 15, 22, 0, tzinfo=UTC)  # 17:00 EST
        assert trading_day(before, FX_DAY_ORIGIN).day == 14
        assert trading_day(after, FX_DAY_ORIGIN).day == 15

    def test_and_twenty_one_utc_in_summer(self) -> None:
        # The same local boundary, an hour earlier in UTC. A stored UTC offset
        # would have put the rollover at 22:00 here too and been wrong for half
        # the year.
        before = datetime(2024, 7, 15, 20, 59, tzinfo=UTC)  # 16:59 EDT
        after = datetime(2024, 7, 15, 21, 0, tzinfo=UTC)  # 17:00 EDT
        assert trading_day(before, FX_DAY_ORIGIN).day == 14
        assert trading_day(after, FX_DAY_ORIGIN).day == 15

    def test_the_spring_forward_trading_day_is_twenty_three_hours_long(self) -> None:
        # US DST begins on 10 March 2024. The day labelled the 9th opens at
        # 17:00 EST (22:00 UTC) and closes at 17:00 EDT (21:00 UTC on the 10th):
        # 23 hours of real time. That is not a defect to be smoothed over -- the
        # firm's day really is short that week.
        opens = datetime(2024, 3, 9, 22, 0, tzinfo=UTC)
        closes = datetime(2024, 3, 10, 21, 0, tzinfo=UTC)
        assert trading_day(opens, FX_DAY_ORIGIN).day == 9
        assert trading_day(opens - timedelta(minutes=1), FX_DAY_ORIGIN).day == 8
        assert trading_day(closes - timedelta(minutes=1), FX_DAY_ORIGIN).day == 9
        assert trading_day(closes, FX_DAY_ORIGIN).day == 10
        assert closes - opens == timedelta(hours=23)

    def test_an_anchor_at_a_local_time_that_does_not_exist_still_works(self) -> None:
        # 02:30 America/New_York never happens on 10 March 2024. The day label
        # is computed by subtracting an offset from the localised instant and
        # never by constructing the anchor, so there is nothing to be undefined.
        origin = DayOrigin(tz="America/New_York", at=time(2, 30))
        zone = ZoneInfo("America/New_York")
        around_the_gap = [
            datetime(2024, 3, 10, 1, 30, tzinfo=zone),
            datetime(2024, 3, 10, 3, 30, tzinfo=zone),
            datetime(2024, 3, 10, 12, 0, tzinfo=zone),
        ]
        labels = [trading_day(moment, origin) for moment in around_the_gap]
        assert labels[0].day == 9  # before the (missing) 02:30 anchor
        assert labels[1].day == 10
        assert labels[2].day == 10

    def test_a_prop_firms_own_midnight_is_expressible(self) -> None:
        # FTMO-style: midnight in the firm's city, which is CET or CEST.
        winter = datetime(2024, 1, 10, 22, 59, tzinfo=UTC)  # 23:59 Prague
        winter_next = datetime(2024, 1, 10, 23, 0, tzinfo=UTC)  # 00:00 Prague
        assert trading_day(winter, PRAGUE_MIDNIGHT).day == 10
        assert trading_day(winter_next, PRAGUE_MIDNIGHT).day == 11

        summer = datetime(2024, 7, 10, 21, 59, tzinfo=UTC)  # 23:59 Prague
        summer_next = datetime(2024, 7, 10, 22, 0, tzinfo=UTC)  # 00:00 Prague
        assert trading_day(summer, PRAGUE_MIDNIGHT).day == 10
        assert trading_day(summer_next, PRAGUE_MIDNIGHT).day == 11

    def test_a_naive_instant_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="tz-aware"):
            trading_day(datetime(2024, 1, 1, 12, 0), FX_DAY_ORIGIN)  # noqa: DTZ001


class TestDailyLossLimit:
    """DoD: after the limit every signal is refused; after the reset they pass."""

    @pytest.fixture
    def breakers(self) -> CircuitBreakers:
        return CircuitBreakers(
            CircuitBreakerConfig(
                trading_day=PRAGUE_MIDNIGHT,
                max_daily_loss_pct=0.02,
                max_weekly_loss_pct=None,
                max_monthly_loss_pct=None,
            )
        )

    def test_below_the_limit_trading_continues(self, breakers: CircuitBreakers) -> None:
        at = datetime(2024, 1, 10, 12, 0, tzinfo=UTC)
        trades = [loss(at, "-1500")]
        assert breakers.check(at=at, bar_index=0, equity=EQUITY, trades=trades) is None

    def test_at_the_limit_trading_stops(self, breakers: CircuitBreakers) -> None:
        at = datetime(2024, 1, 10, 12, 0, tzinfo=UTC)
        trades = [loss(at, "-2000")]
        trip = breakers.check(at=at, bar_index=0, equity=EQUITY, trades=trades)
        assert trip is not None
        assert trip.reason is RiskReason.DAILY_LOSS_LIMIT

    def test_the_next_trading_day_starts_clean(self, breakers: CircuitBreakers) -> None:
        # The reset is not an event anyone has to fire: it is a consequence of
        # the day label changing. Same ledger, later instant, trading allowed.
        blown = datetime(2024, 1, 10, 12, 0, tzinfo=UTC)
        trades = [loss(blown, "-3000")]
        assert breakers.check(at=blown, bar_index=0, equity=EQUITY, trades=trades) is not None

        next_day = datetime(2024, 1, 11, 12, 0, tzinfo=UTC)
        assert breakers.check(at=next_day, bar_index=1, equity=EQUITY, trades=trades) is None

    def test_the_reset_happens_at_the_configured_local_midnight(
        self, breakers: CircuitBreakers
    ) -> None:
        # 22:59 UTC is still 23:59 in Prague on the 10th, so still blocked.
        # 23:00 UTC is 00:00 on the 11th, so clear. A UTC-midnight boundary
        # would have kept trading blocked for another hour.
        trades = [loss(datetime(2024, 1, 10, 12, 0, tzinfo=UTC), "-3000")]
        still_blocked = datetime(2024, 1, 10, 22, 59, tzinfo=UTC)
        cleared = datetime(2024, 1, 10, 23, 0, tzinfo=UTC)
        assert (
            breakers.check(at=still_blocked, bar_index=0, equity=EQUITY, trades=trades) is not None
        )
        assert breakers.check(at=cleared, bar_index=1, equity=EQUITY, trades=trades) is None

    def test_the_reset_moves_with_dst(self, breakers: CircuitBreakers) -> None:
        # DoD: the same configured boundary, in July. Prague is CEST, so local
        # midnight is 22:00 UTC rather than 23:00 -- and the blocked/cleared
        # pair from the winter test lands the other way round.
        trades = [loss(datetime(2024, 7, 10, 12, 0, tzinfo=UTC), "-3000")]
        still_blocked = datetime(2024, 7, 10, 21, 59, tzinfo=UTC)
        cleared = datetime(2024, 7, 10, 22, 0, tzinfo=UTC)
        assert (
            breakers.check(at=still_blocked, bar_index=0, equity=EQUITY, trades=trades) is not None
        )
        assert breakers.check(at=cleared, bar_index=1, equity=EQUITY, trades=trades) is None

    def test_profit_earlier_in_the_day_offsets_a_later_loss(
        self, breakers: CircuitBreakers
    ) -> None:
        at = datetime(2024, 1, 10, 12, 0, tzinfo=UTC)
        trades = [win(at, "1500"), loss(at, "-3000")]
        assert breakers.check(at=at, bar_index=0, equity=EQUITY, trades=trades) is None

    def test_a_previous_days_loss_does_not_count(self, breakers: CircuitBreakers) -> None:
        yesterday = datetime(2024, 1, 9, 12, 0, tzinfo=UTC)
        today = datetime(2024, 1, 10, 12, 0, tzinfo=UTC)
        assert (
            breakers.check(at=today, bar_index=0, equity=EQUITY, trades=[loss(yesterday, "-9000")])
            is None
        )

    def test_disabling_the_limit_disables_the_breaker(self) -> None:
        breakers = CircuitBreakers(
            CircuitBreakerConfig(
                max_daily_loss_pct=None, max_weekly_loss_pct=None, max_monthly_loss_pct=None
            )
        )
        at = datetime(2024, 1, 10, 12, 0, tzinfo=UTC)
        assert (
            breakers.check(at=at, bar_index=0, equity=EQUITY, trades=[loss(at, "-99999")]) is None
        )


class TestWeeklyAndMonthlyLimits:
    def test_a_week_accumulates_across_days(self) -> None:
        breakers = CircuitBreakers(
            CircuitBreakerConfig(
                trading_day=PRAGUE_MIDNIGHT,
                max_daily_loss_pct=0.05,
                max_weekly_loss_pct=0.04,
                max_monthly_loss_pct=None,
            )
        )
        # Three days of losses, each under the daily limit, together over the
        # weekly one.
        trades = [
            loss(datetime(2024, 1, 9, 12, 0, tzinfo=UTC), "-1500"),
            loss(datetime(2024, 1, 10, 12, 0, tzinfo=UTC), "-1500"),
            loss(datetime(2024, 1, 11, 12, 0, tzinfo=UTC), "-1500"),
        ]
        at = datetime(2024, 1, 11, 13, 0, tzinfo=UTC)
        trip = breakers.check(at=at, bar_index=0, equity=EQUITY, trades=trades)
        assert trip is not None
        assert trip.reason is RiskReason.WEEKLY_LOSS_LIMIT

    def test_the_week_start_is_configurable(self) -> None:
        # 7 January 2024 is a Sunday. Under a Sunday-start week it opens a new
        # week; under a Monday-start one it closes the previous.
        sunday = datetime(2024, 1, 7).date()
        assert week_label(sunday, starts_on=Weekday.SUNDAY) == sunday
        assert week_label(sunday, starts_on=Weekday.MONDAY).day == 1

    def test_a_month_accumulates_across_weeks(self) -> None:
        breakers = CircuitBreakers(
            CircuitBreakerConfig(
                trading_day=PRAGUE_MIDNIGHT,
                max_daily_loss_pct=None,
                max_weekly_loss_pct=None,
                max_monthly_loss_pct=0.03,
            )
        )
        trades = [
            loss(datetime(2024, 1, 3, 12, 0, tzinfo=UTC), "-1500"),
            loss(datetime(2024, 1, 17, 12, 0, tzinfo=UTC), "-1500"),
        ]
        at = datetime(2024, 1, 20, 12, 0, tzinfo=UTC)
        trip = breakers.check(at=at, bar_index=0, equity=EQUITY, trades=trades)
        assert trip is not None
        assert trip.reason is RiskReason.MONTHLY_LOSS_LIMIT

        # The next month starts clean off the same ledger.
        february = datetime(2024, 2, 1, 12, 0, tzinfo=UTC)
        assert breakers.check(at=february, bar_index=1, equity=EQUITY, trades=trades) is None


class TestConsecutiveLossPause:
    """DoD: N losses in a row pauses; the pause lifts after M bars."""

    @pytest.fixture
    def breakers(self) -> CircuitBreakers:
        return CircuitBreakers(
            CircuitBreakerConfig(
                max_daily_loss_pct=None,
                max_weekly_loss_pct=None,
                max_monthly_loss_pct=None,
                max_consecutive_losses=3,
                consecutive_loss_pause_bars=5,
            )
        )

    def test_below_the_streak_nothing_happens(self, breakers: CircuitBreakers) -> None:
        at = datetime(2024, 1, 10, 12, 0, tzinfo=UTC)
        trades = [loss(at), loss(at)]
        assert breakers.check(at=at, bar_index=0, equity=EQUITY, trades=trades) is None

    def test_the_streak_starts_the_pause(self, breakers: CircuitBreakers) -> None:
        at = datetime(2024, 1, 10, 12, 0, tzinfo=UTC)
        trades = [loss(at), loss(at), loss(at)]
        trip = breakers.check(at=at, bar_index=10, equity=EQUITY, trades=trades)
        assert trip is not None
        assert trip.reason is RiskReason.CONSECUTIVE_LOSS_PAUSE

    def test_the_pause_holds_for_exactly_m_bars(self, breakers: CircuitBreakers) -> None:
        at = datetime(2024, 1, 10, 12, 0, tzinfo=UTC)
        trades = [loss(at), loss(at), loss(at)]
        breakers.check(at=at, bar_index=10, equity=EQUITY, trades=trades)

        for bar in range(11, 15):
            trip = breakers.check(at=at, bar_index=bar, equity=EQUITY, trades=trades)
            assert trip is not None, f"bar {bar} should still be paused"
            assert trip.reason is RiskReason.CONSECUTIVE_LOSS_PAUSE

        # Bar 15 is 10 + 5: the pause has served its term.
        assert breakers.check(at=at, bar_index=15, equity=EQUITY, trades=trades) is None

    def test_a_win_ends_the_streak(self, breakers: CircuitBreakers) -> None:
        at = datetime(2024, 1, 10, 12, 0, tzinfo=UTC)
        trades = [loss(at), loss(at), loss(at), win(at)]
        assert breakers.check(at=at, bar_index=0, equity=EQUITY, trades=trades) is None

    def test_a_scratch_ends_the_streak_too(self, breakers: CircuitBreakers) -> None:
        # The streak is meant to detect a run of the strategy being wrong, and
        # a break-even trade is not evidence of that.
        at = datetime(2024, 1, 10, 12, 0, tzinfo=UTC)
        trades = [loss(at), loss(at), loss(at), ClosedTrade(closed_at=at, pnl=Decimal(0))]
        assert breakers.check(at=at, bar_index=0, equity=EQUITY, trades=trades) is None

    def test_reset_clears_an_armed_pause(self, breakers: CircuitBreakers) -> None:
        # A pause left armed from the previous walk-forward fold would suppress
        # the opening bars of the next one, silently.
        at = datetime(2024, 1, 10, 12, 0, tzinfo=UTC)
        trades = [loss(at), loss(at), loss(at)]
        breakers.check(at=at, bar_index=10, equity=EQUITY, trades=trades)
        breakers.reset()
        assert breakers.check(at=at, bar_index=11, equity=EQUITY, trades=[]) is None


class TestSlippageBreaker:
    """The producer of these reports arrives in P12; the shape is fixed now."""

    @pytest.fixture
    def breakers(self) -> CircuitBreakers:
        return CircuitBreakers(
            CircuitBreakerConfig(
                max_daily_loss_pct=None,
                max_weekly_loss_pct=None,
                max_monthly_loss_pct=None,
                max_slippage_points=10.0,
                slippage_pause_bars=3,
            )
        )

    def test_normal_slippage_is_ignored(self, breakers: CircuitBreakers) -> None:
        at = datetime(2024, 1, 10, 12, 0, tzinfo=UTC)
        assert not breakers.record_fill(SlippageReport(at=at, slippage_points=4.0))
        assert breakers.check(at=at, bar_index=0, equity=EQUITY, trades=[]) is None

    def test_a_favourable_fill_never_trips_it(self, breakers: CircuitBreakers) -> None:
        # Positive means worse. A fill better than expected is negative, and
        # being pleasantly surprised is not an anomaly to stop trading over.
        at = datetime(2024, 1, 10, 12, 0, tzinfo=UTC)
        assert not breakers.record_fill(SlippageReport(at=at, slippage_points=-40.0))

    def test_an_anomalous_fill_pauses_trading(self, breakers: CircuitBreakers) -> None:
        at = datetime(2024, 1, 10, 12, 0, tzinfo=UTC)
        assert breakers.record_fill(SlippageReport(at=at, slippage_points=25.0, symbol="EURUSD"))
        trip = breakers.check(at=at, bar_index=7, equity=EQUITY, trades=[])
        assert trip is not None
        assert trip.reason is RiskReason.SLIPPAGE_ANOMALY_PAUSE

    def test_that_pause_also_expires(self, breakers: CircuitBreakers) -> None:
        at = datetime(2024, 1, 10, 12, 0, tzinfo=UTC)
        breakers.record_fill(SlippageReport(at=at, slippage_points=25.0))
        breakers.check(at=at, bar_index=7, equity=EQUITY, trades=[])
        assert breakers.check(at=at, bar_index=9, equity=EQUITY, trades=[]) is not None
        assert breakers.check(at=at, bar_index=10, equity=EQUITY, trades=[]) is None

    def test_the_anomalies_are_recorded_for_the_alert(self, breakers: CircuitBreakers) -> None:
        at = datetime(2024, 1, 10, 12, 0, tzinfo=UTC)
        breakers.record_fill(SlippageReport(at=at, slippage_points=25.0, symbol="EURUSD"))
        assert [event.symbol for event in breakers.slippage_events] == ["EURUSD"]
