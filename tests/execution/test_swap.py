"""Financing: the tripled Wednesday, and the week that must sum to seven."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from trading_system.core.instruments import InstrumentRegistry, InstrumentSpec
from trading_system.core.types import Price, Side
from trading_system.execution.config import FundingRateSwap, PerLotRolloverSwap
from trading_system.execution.costs import accrue_swap

ACCOUNT = "USD"

#: Rollovers happen at 17:00 New York. On 2024-03-06 — a Wednesday, four days
#: before US daylight saving began — that is 22:00 UTC. Which UTC instant it
#: lands on is exactly what a stored offset would get wrong; see
#: :class:`TestDaylightSaving`.
BEFORE_WED_ROLLOVER = datetime(2024, 3, 6, 20, 0, tzinfo=UTC)
AFTER_WED_ROLLOVER = datetime(2024, 3, 6, 22, 0, tzinfo=UTC)


def _accrue(
    instrument: InstrumentSpec,
    held_from: datetime,
    held_to: datetime,
    *,
    side: Side = Side.BUY,
    size: str = "1",
) -> tuple[Decimal, int]:
    """Accrue per-lot financing and return the amount with the charge count.

    Args:
        instrument: Contract specification.
        held_from: Open time.
        held_to: Close time.
        side: Direction held.
        size: Size in lots.

    Returns:
        The signed amount and how many days of financing were charged.
    """
    accrual = accrue_swap(
        instrument,
        side=side,
        size=Decimal(size),
        held_from=held_from,
        held_to=held_to,
        model=PerLotRolloverSwap(),
        account_currency=ACCOUNT,
    )
    return accrual.amount, accrual.charges


class TestTripleWednesday:
    """Value date T+2 from Wednesday lands on Monday, so the rollover carries three."""

    def test_crossing_the_wednesday_rollover_charges_three_days(
        self, eurusd: InstrumentSpec
    ) -> None:
        """The headline convention, on a two-hour hold that crosses one boundary."""
        amount, charges = _accrue(eurusd, BEFORE_WED_ROLLOVER, AFTER_WED_ROLLOVER)
        assert charges == 3
        assert amount == eurusd.swap_long * 3

    def test_crossing_a_tuesday_rollover_charges_one_day(self, eurusd: InstrumentSpec) -> None:
        """The comparison that makes the previous test mean something."""
        amount, charges = _accrue(
            eurusd,
            BEFORE_WED_ROLLOVER - timedelta(days=1),
            AFTER_WED_ROLLOVER - timedelta(days=1),
        )
        assert charges == 1
        assert amount == eurusd.swap_long

    def test_the_wednesday_charge_is_exactly_three_times_the_tuesday_one(
        self, eurusd: InstrumentSpec
    ) -> None:
        """Not merely larger: a factor of three, which is what T+2 implies."""
        wednesday, _ = _accrue(eurusd, BEFORE_WED_ROLLOVER, AFTER_WED_ROLLOVER)
        tuesday, _ = _accrue(
            eurusd,
            BEFORE_WED_ROLLOVER - timedelta(days=1),
            AFTER_WED_ROLLOVER - timedelta(days=1),
        )
        assert wednesday == tuesday * 3

    def test_a_short_is_charged_its_own_side_of_the_swap(self, eurusd: InstrumentSpec) -> None:
        """EURUSD pays to hold a long and receives to hold a short."""
        long_amount, _ = _accrue(eurusd, BEFORE_WED_ROLLOVER, AFTER_WED_ROLLOVER)
        short_amount, _ = _accrue(eurusd, BEFORE_WED_ROLLOVER, AFTER_WED_ROLLOVER, side=Side.SELL)
        assert long_amount == eurusd.swap_long * 3
        assert short_amount == eurusd.swap_short * 3
        assert long_amount < 0 < short_amount


class TestAFullWeekCostsSevenDays:
    """The arithmetic check that the convention was implemented, not approximated."""

    def test_a_week_held_charges_exactly_seven_days(self, eurusd: InstrumentSpec) -> None:
        """Sunday, Monday, Tuesday, Thursday at one each, Wednesday at three.

        Friday and Saturday cover a shut market and carry nothing, which is the
        whole reason Wednesday is tripled in the first place. Any other
        arrangement of skips and triples fails to reach seven.
        """
        start = datetime(2024, 3, 3, 12, 0, tzinfo=UTC)  # Sunday, before the open
        _, charges = _accrue(eurusd, start, start + timedelta(days=7))
        assert charges == 7

    def test_a_weekend_hold_charges_nothing(self, eurusd: InstrumentSpec) -> None:
        """The Friday and Saturday labels cover a market that is not open."""
        friday_evening = datetime(2024, 3, 8, 23, 0, tzinfo=UTC)
        sunday_afternoon = datetime(2024, 3, 10, 18, 0, tzinfo=UTC)
        _, charges = _accrue(eurusd, friday_evening, sunday_afternoon)
        assert charges == 0

    def test_an_intraday_hold_crosses_no_rollover(self, eurusd: InstrumentSpec) -> None:
        """Opening and closing inside one trading day costs no financing."""
        _, charges = _accrue(eurusd, BEFORE_WED_ROLLOVER, BEFORE_WED_ROLLOVER + timedelta(hours=1))
        assert charges == 0

    def test_financing_scales_with_size(self, eurusd: InstrumentSpec) -> None:
        """Two lots pay twice."""
        one, _ = _accrue(eurusd, BEFORE_WED_ROLLOVER, AFTER_WED_ROLLOVER, size="1")
        two, _ = _accrue(eurusd, BEFORE_WED_ROLLOVER, AFTER_WED_ROLLOVER, size="2")
        assert two == one * 2


class TestDaylightSaving:
    """The boundary is a local wall-clock time, never a stored UTC offset."""

    def test_the_rollover_follows_new_york_across_the_spring_transition(
        self, eurusd: InstrumentSpec
    ) -> None:
        """17:00 New York is 22:00 UTC in winter and 21:00 UTC in summer.

        The same UTC window — 20:30 to 21:30 — therefore crosses the rollover in
        summer and does not in winter. Both dates are Wednesdays, so a model
        storing a fixed offset does not merely mistime the charge: it books
        three days of financing on one of the two and none on the other.
        """
        # 2024-06-12, a Wednesday inside US daylight saving: rollover at 21:00 UTC.
        summer = datetime(2024, 6, 12, 21, 30, tzinfo=UTC)
        _, summer_charges = _accrue(eurusd, summer - timedelta(hours=1), summer)
        # 2024-01-10, a Wednesday in standard time: rollover at 22:00 UTC, so the
        # same wall-clock window crosses nothing at all.
        winter = datetime(2024, 1, 10, 21, 30, tzinfo=UTC)
        _, winter_charges = _accrue(eurusd, winter - timedelta(hours=1), winter)

        assert summer_charges == 3
        assert winter_charges == 0


class TestFundingRate:
    """Perpetuals exchange funding on notional and never roll over."""

    def test_funding_is_charged_at_each_scheduled_instant(
        self, registry: InstrumentRegistry
    ) -> None:
        """A day at the eight-hourly schedule is three exchanges."""
        btc = registry["BTCUSD"]
        model = FundingRateSwap(rate_per_interval=0.0001)
        accrual = accrue_swap(
            btc,
            side=Side.BUY,
            size=Decimal("1"),
            held_from=datetime(2024, 3, 6, 0, 0, tzinfo=UTC),
            held_to=datetime(2024, 3, 7, 0, 0, tzinfo=UTC),
            model=model,
            account_currency=ACCOUNT,
            mark_price=Price(60000.0),
        )
        assert accrual.charges == 3
        # Positive rate means longs pay: 3 * 0.0001 * 60000 = 18 leaving the account.
        assert accrual.amount == Decimal("-18.0000")

    def test_a_short_receives_what_the_long_pays(self, registry: InstrumentRegistry) -> None:
        """Funding is an exchange between the two sides, not a fee to the venue."""
        btc = registry["BTCUSD"]
        window = {
            "held_from": datetime(2024, 3, 6, 0, 0, tzinfo=UTC),
            "held_to": datetime(2024, 3, 7, 0, 0, tzinfo=UTC),
            "model": FundingRateSwap(rate_per_interval=0.0001),
            "account_currency": ACCOUNT,
            "mark_price": Price(60000.0),
        }
        long_side = accrue_swap(btc, side=Side.BUY, size=Decimal("1"), **window)
        short_side = accrue_swap(btc, side=Side.SELL, size=Decimal("1"), **window)
        assert long_side.amount == -short_side.amount

    def test_no_weekend_is_skipped_and_no_day_is_tripled(
        self, registry: InstrumentRegistry
    ) -> None:
        """Crypto never closes, so the schedule is uniform."""
        btc = registry["BTCUSD"]
        model = FundingRateSwap(rate_per_interval=0.0001)
        week = accrue_swap(
            btc,
            side=Side.BUY,
            size=Decimal("1"),
            held_from=datetime(2024, 3, 3, 0, 0, tzinfo=UTC),
            held_to=datetime(2024, 3, 10, 0, 0, tzinfo=UTC),
            model=model,
            account_currency=ACCOUNT,
            mark_price=Price(60000.0),
        )
        assert week.charges == 21  # seven days at three a day, weekend included

    def test_funding_is_denominated_in_the_quote_currency(
        self, registry: InstrumentRegistry
    ) -> None:
        """A fraction of a notional priced in the quote currency is quote money.

        Stated on the accrual rather than assumed, because a per-lot rollover is
        account currency and the two models produce different currencies from
        the same call.
        """
        btc = registry["BTCUSD"]
        accrual = accrue_swap(
            btc,
            side=Side.BUY,
            size=Decimal("1"),
            held_from=datetime(2024, 3, 6, 0, 0, tzinfo=UTC),
            held_to=datetime(2024, 3, 6, 9, 0, tzinfo=UTC),
            model=FundingRateSwap(),
            account_currency=ACCOUNT,
            mark_price=Price(60000.0),
        )
        assert accrual.currency == btc.quote_currency

    def test_funding_without_a_mark_price_refuses(self, registry: InstrumentRegistry) -> None:
        """A rate on notional with no notional is not a number."""
        with pytest.raises(ValueError, match="mark_price"):
            accrue_swap(
                registry["BTCUSD"],
                side=Side.BUY,
                size=Decimal("1"),
                held_from=datetime(2024, 3, 6, 0, 0, tzinfo=UTC),
                held_to=datetime(2024, 3, 7, 0, 0, tzinfo=UTC),
                model=FundingRateSwap(),
                account_currency=ACCOUNT,
            )


class TestConfigRefusals:
    """Contradictory financing configurations do not load."""

    def test_a_tripled_weekday_cannot_also_be_closed(self) -> None:
        """A rollover that never happens cannot be the one carrying three days."""
        with pytest.raises(ValueError, match="closed_weekdays"):
            PerLotRolloverSwap(triple_weekday=4, closed_weekdays=(4, 5))

    def test_an_empty_funding_schedule_is_rejected(self) -> None:
        """A perpetual with no funding instants is not a perpetual."""
        with pytest.raises(ValueError, match="at least one"):
            FundingRateSwap(times_utc=())

    def test_a_backwards_holding_period_is_rejected(self, eurusd: InstrumentSpec) -> None:
        """Closing before opening is a caller bug, not a zero accrual."""
        with pytest.raises(ValueError, match="backwards"):
            _accrue(eurusd, AFTER_WED_ROLLOVER, BEFORE_WED_ROLLOVER)
