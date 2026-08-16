"""The period totals are accumulated, and must still answer the old question.

Before this, every check summed the whole journal three times — once per period
— and re-derived each trade's trading day while doing it. That was O(bars ×
trades) and 58% of a real run. The totals are now folded in as trades arrive,
which is only sound if the fold reproduces the filter exactly: the boundary
between one period and the next must not move by a second, and no trade may be
counted twice or dropped.

These tests state the equivalence directly by re-deriving what the old scan
would have returned, rather than by asserting on a number someone wrote down.
"""

from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from trading_system.backtest.portfolio import Portfolio
from trading_system.core.instruments import load_instruments
from trading_system.data.resample import FX_DAY_ORIGIN, DayOrigin, trading_day
from trading_system.risk.circuit_breakers import (
    CircuitBreakerConfig,
    CircuitBreakers,
    ClosedTrade,
    Weekday,
    week_label,
)
from trading_system.risk.conversion import SameCurrencyConverter

EQUITY = Decimal("100000")
PRAGUE_MIDNIGHT = DayOrigin(tz="Europe/Prague", at=time(0, 0))
NY_CLOSE = DayOrigin(tz="America/New_York", at=time(17, 0))


def trade(at: datetime, amount: str) -> ClosedTrade:
    """A realised trade booked at ``at``."""
    return ClosedTrade(closed_at=at, pnl=Decimal(amount))


def scanned(
    trades: list[ClosedTrade], today: date, origin: DayOrigin, period: str, starts_on: Weekday
) -> Decimal:
    """What the replaced full-journal scan would have summed for one period.

    This is the pre-change expression, kept verbatim so the equivalence is
    checked against the old code's arithmetic and not against a restatement of
    the new code's.
    """

    def in_period(day: date) -> bool:
        if period == "day":
            return day == today
        if period == "week":
            return week_label(day, starts_on=starts_on) == week_label(today, starts_on=starts_on)
        return (day.year, day.month) == (today.year, today.month)

    return sum(
        (item.pnl for item in trades if in_period(trading_day(item.closed_at, origin))),
        Decimal(0),
    )


def totals_of(breakers: CircuitBreakers, trades: list[ClosedTrade], at: datetime) -> None:
    """Drive one check so the accumulator absorbs ``trades``."""
    breakers.check(at=at, bar_index=0, equity=EQUITY, trades=trades)


class TestTheFoldMatchesTheScanItReplaced:
    """The accumulated total equals what summing the journal would have given."""

    @pytest.fixture
    def breakers(self) -> CircuitBreakers:
        return CircuitBreakers(
            CircuitBreakerConfig(
                trading_day=PRAGUE_MIDNIGHT,
                max_daily_loss_pct=0.9,
                max_weekly_loss_pct=0.9,
                max_monthly_loss_pct=0.9,
                week_starts_on=Weekday.MONDAY,
            )
        )

    @pytest.mark.parametrize("period", ["day", "week", "month"])
    def test_a_journal_spanning_days_weeks_and_months(
        self, breakers: CircuitBreakers, period: str
    ) -> None:
        start = datetime(2024, 1, 8, 9, 30, tzinfo=UTC)
        trades = [
            trade(start + timedelta(hours=13 * index), f"{-70 + index * 11}") for index in range(60)
        ]
        # Absorb the journal one trade at a time, as a run does.
        for index in range(1, len(trades) + 1):
            prefix = trades[:index]
            at = prefix[-1].closed_at + timedelta(minutes=5)
            totals_of(breakers, prefix, at)

        at = trades[-1].closed_at + timedelta(minutes=5)
        today = trading_day(at, PRAGUE_MIDNIGHT)
        expected = scanned(trades, today, PRAGUE_MIDNIGHT, period, Weekday.MONDAY)
        got = {
            "day": breakers._daily.get(today, Decimal(0)),
            "week": breakers._weekly.get(week_label(today, starts_on=Weekday.MONDAY), Decimal(0)),
            "month": breakers._monthly.get(today.replace(day=1), Decimal(0)),
        }[period]
        assert got == expected


class TestPeriodBoundaries:
    """A rollover must land where it landed before, to the second."""

    def _breakers(self, origin: DayOrigin) -> CircuitBreakers:
        return CircuitBreakers(
            CircuitBreakerConfig(
                trading_day=origin,
                max_daily_loss_pct=0.02,
                max_weekly_loss_pct=None,
                max_monthly_loss_pct=None,
            )
        )

    def test_the_day_rolls_over_and_the_previous_days_loss_stops_counting(self) -> None:
        breakers = self._breakers(PRAGUE_MIDNIGHT)
        blown = datetime(2024, 1, 10, 12, 0, tzinfo=UTC)
        trades = [trade(blown, "-3000")]
        assert breakers.check(at=blown, bar_index=0, equity=EQUITY, trades=trades) is not None
        later = datetime(2024, 1, 11, 12, 0, tzinfo=UTC)
        assert breakers.check(at=later, bar_index=1, equity=EQUITY, trades=trades) is None

    def test_the_boundary_itself_is_not_off_by_one_bar(self) -> None:
        # 23:59:59 Prague still belongs to the blown day; 00:00:00 does not.
        breakers = self._breakers(PRAGUE_MIDNIGHT)
        blown = datetime(2024, 1, 10, 12, 0, tzinfo=UTC)
        trades = [trade(blown, "-3000")]
        prague = PRAGUE_MIDNIGHT.tz
        last = datetime(2024, 1, 10, 23, 59, 59, tzinfo=ZoneInfo(prague)).astimezone(UTC)
        first = datetime(2024, 1, 11, 0, 0, 0, tzinfo=ZoneInfo(prague)).astimezone(UTC)
        assert breakers.check(at=last, bar_index=1, equity=EQUITY, trades=trades) is not None
        assert breakers.check(at=first, bar_index=2, equity=EQUITY, trades=trades) is None

    def test_a_weekend_does_not_start_a_new_week(self) -> None:
        breakers = CircuitBreakers(
            CircuitBreakerConfig(
                trading_day=PRAGUE_MIDNIGHT,
                max_daily_loss_pct=None,
                max_weekly_loss_pct=0.02,
                max_monthly_loss_pct=None,
                week_starts_on=Weekday.MONDAY,
            )
        )
        # Friday's loss must still bind on the following Sunday, which belongs
        # to the same trading week, and stop binding on the Monday.
        friday = datetime(2024, 1, 12, 12, 0, tzinfo=UTC)
        sunday = datetime(2024, 1, 14, 12, 0, tzinfo=UTC)
        monday = datetime(2024, 1, 15, 12, 0, tzinfo=UTC)
        trades = [trade(friday, "-3000")]
        assert breakers.check(at=friday, bar_index=0, equity=EQUITY, trades=trades) is not None
        assert breakers.check(at=sunday, bar_index=1, equity=EQUITY, trades=trades) is not None
        assert breakers.check(at=monday, bar_index=2, equity=EQUITY, trades=trades) is None

    def test_a_month_rolls_over_on_its_own_first_day(self) -> None:
        breakers = CircuitBreakers(
            CircuitBreakerConfig(
                trading_day=PRAGUE_MIDNIGHT,
                max_daily_loss_pct=None,
                max_weekly_loss_pct=None,
                max_monthly_loss_pct=0.02,
            )
        )
        blown = datetime(2024, 1, 31, 12, 0, tzinfo=UTC)
        trades = [trade(blown, "-3000")]
        assert breakers.check(at=blown, bar_index=0, equity=EQUITY, trades=trades) is not None
        february = datetime(2024, 2, 1, 12, 0, tzinfo=UTC)
        assert breakers.check(at=february, bar_index=1, equity=EQUITY, trades=trades) is None

    def test_the_clock_change_does_not_move_a_trade_between_days(self) -> None:
        # 17:00 New York is 21:00 UTC in summer and 22:00 UTC in winter. A trade
        # booked at 21:30 UTC on the Friday before the autumn change belongs to
        # the *next* trading day in summer terms and the same one in winter
        # terms; whichever it is, the fold and the scan must agree.
        breakers = CircuitBreakers(
            CircuitBreakerConfig(
                trading_day=NY_CLOSE,
                max_daily_loss_pct=0.9,
                max_weekly_loss_pct=None,
                max_monthly_loss_pct=None,
            )
        )
        around = [
            datetime(2024, 11, 1, 20, 30, tzinfo=UTC),
            datetime(2024, 11, 1, 21, 30, tzinfo=UTC),
            datetime(2024, 11, 4, 20, 30, tzinfo=UTC),
            datetime(2024, 11, 4, 21, 30, tzinfo=UTC),
            datetime(2024, 11, 4, 22, 30, tzinfo=UTC),
        ]
        trades = [trade(at, "-100") for at in around]
        for index in range(1, len(trades) + 1):
            totals_of(breakers, trades[:index], trades[index - 1].closed_at)
        for at in around:
            day = trading_day(at, NY_CLOSE)
            assert breakers._daily[day] == scanned(trades, day, NY_CLOSE, "day", Weekday.MONDAY)


class TestTheAccumulatorFollowsTheRightJournal:
    """Absorbing a tail is only safe while it is the same history."""

    @pytest.fixture
    def breakers(self) -> CircuitBreakers:
        return CircuitBreakers(
            CircuitBreakerConfig(
                trading_day=PRAGUE_MIDNIGHT,
                max_daily_loss_pct=0.9,
                max_weekly_loss_pct=None,
                max_monthly_loss_pct=None,
            )
        )

    def test_the_same_journal_seen_twice_is_not_counted_twice(
        self, breakers: CircuitBreakers
    ) -> None:
        at = datetime(2024, 1, 10, 12, 0, tzinfo=UTC)
        trades = [trade(at, "-100")]
        for index in range(5):
            breakers.check(at=at, bar_index=index, equity=EQUITY, trades=trades)
        assert breakers._daily[trading_day(at, PRAGUE_MIDNIGHT)] == Decimal("-100")

    def test_a_shorter_journal_rebuilds_rather_than_keeping_a_stale_total(
        self, breakers: CircuitBreakers
    ) -> None:
        at = datetime(2024, 1, 10, 12, 0, tzinfo=UTC)
        long = [trade(at, "-100"), trade(at, "-200"), trade(at, "-300")]
        breakers.check(at=at, bar_index=0, equity=EQUITY, trades=long)
        breakers.check(at=at, bar_index=1, equity=EQUITY, trades=long[:1])
        assert breakers._daily[trading_day(at, PRAGUE_MIDNIGHT)] == Decimal("-100")

    def test_a_different_journal_of_the_same_length_rebuilds(
        self, breakers: CircuitBreakers
    ) -> None:
        at = datetime(2024, 1, 10, 12, 0, tzinfo=UTC)
        first = [trade(at, "-100"), trade(at, "-200")]
        second = [trade(at, "-1"), trade(at, "-2")]
        breakers.check(at=at, bar_index=0, equity=EQUITY, trades=first)
        breakers.check(at=at, bar_index=1, equity=EQUITY, trades=second)
        assert breakers._daily[trading_day(at, PRAGUE_MIDNIGHT)] == Decimal("-3")

    def test_reset_clears_the_totals_as_well_as_the_pauses(self, breakers: CircuitBreakers) -> None:
        # A walk-forward fold that inherited the previous fold's realised loss
        # would measure its limit against a history it never traded.
        at = datetime(2024, 1, 10, 12, 0, tzinfo=UTC)
        trades = [trade(at, "-100")]
        breakers.check(at=at, bar_index=0, equity=EQUITY, trades=trades)
        breakers.reset()
        assert breakers._daily == {}
        breakers.check(at=at, bar_index=0, equity=EQUITY, trades=trades)
        assert breakers._daily[trading_day(at, PRAGUE_MIDNIGHT)] == Decimal("-100")


class TestNothingCanShortenTheJournal:
    """The append-only assumption the fold rests on, asserted rather than trusted."""

    def test_the_portfolio_exposes_no_way_to_remove_a_trade(self) -> None:
        from trading_system.backtest.portfolio import Portfolio

        surface = [name for name in dir(Portfolio) if not name.startswith("_")]
        forbidden = ("remove", "delete", "drop", "clear", "purge", "rewrite", "pop")
        offenders = [name for name in surface if any(word in name.lower() for word in forbidden)]
        assert offenders == []


class TestTheProjectionIsBuiltOncePerTrade:
    """`closed_trades` is accumulated, not rebuilt on every read."""

    @pytest.fixture
    def portfolio(self) -> Portfolio:
        return Portfolio(
            currency="USD",
            starting_balance=Decimal(100_000),
            instruments=load_instruments(Path("configs/instruments.yaml")),
            converter=SameCurrencyConverter(),
            day_origin=FX_DAY_ORIGIN,
        )

    def test_two_reads_without_a_new_trade_return_the_same_object(
        self, portfolio: Portfolio
    ) -> None:
        first = portfolio.closed_trades
        second = portfolio.closed_trades
        # Identity, not equality: a rebuild would produce an equal-but-new
        # tuple, and it is the rebuild that cost 15% of a run.
        assert first is second

    def test_the_projection_matches_the_records_it_came_from(self, portfolio: Portfolio) -> None:
        assert [item.pnl for item in portfolio.closed_trades] == [
            record.net for record in portfolio.trades
        ]
