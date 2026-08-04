"""Turning the Exit Engine's price movement into account money.

The multiplier is the whole point. A P&L calculation that is right on an
instrument with ``contract_size`` of 1 and wrong by five orders of magnitude on
one with 100 000 is worse than one that is wrong everywhere, because it looks
correct in the first test anyone writes.
"""

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from tests.risk.conftest import NOW, USDJPY
from trading_system.core.instruments import InstrumentRegistry
from trading_system.core.types import Price, Side
from trading_system.exit.base import ExitReason
from trading_system.exit.position import ManagedPosition
from trading_system.risk.conversion import (
    FxRateUnavailableError,
    SameCurrencyConverter,
    StaticFxConverter,
)
from trading_system.risk.pnl import leg_pnl, realized_pnl


def opened(symbol: str, *, entry: float, stop: float, size: str) -> ManagedPosition:
    """Open a long position of ``size`` lots."""
    return ManagedPosition(
        symbol=symbol,
        side=Side.BUY,
        entry_price=Price(entry),
        size=Decimal(size),
        initial_stop=Price(stop),
        opened_at=NOW,
    )


class TestContractSizeMultiplier:
    """DoD: NAS100 with a contract size other than one."""

    def test_nas100_pnl_carries_the_multiplier(self, registry: InstrumentRegistry) -> None:
        # 2 lots of NAS100 from 18 000 to 18 050 is a 50-point move.
        # contract_size is 20 USD per index point, so:
        #   50 points * 20 USD * 2 lots = 2 000 USD.
        # Without the multiplier this would report 100 USD -- a twentieth.
        position = opened("NAS100", entry=18000.0, stop=17900.0, size="2")
        position.close_all(price=Price(18050.0), ts=NOW, reason=ExitReason.TAKE_PROFIT)

        assert realized_pnl(
            position,
            instrument=registry["NAS100"],
            account_currency="USD",
            converter=SameCurrencyConverter(),
        ) == Decimal("2000")

    def test_the_exit_engines_own_figure_deliberately_lacks_the_multiplier(self) -> None:
        # ManagedPosition holds no InstrumentSpec, so 100 is the most it can
        # honestly report. The name says quote_move, not pnl, for this reason.
        position = opened("NAS100", entry=18000.0, stop=17900.0, size="2")
        position.close_all(price=Price(18050.0), ts=NOW, reason=ExitReason.TAKE_PROFIT)
        assert position.realized_quote_move == Decimal("100")

    def test_eurusd_pnl_carries_a_multiplier_of_a_hundred_thousand(
        self, registry: InstrumentRegistry
    ) -> None:
        # 1 lot from 1.0850 to 1.0900 is 50 pips: 0.0050 * 100 000 = 500 USD.
        position = opened("EURUSD", entry=1.0850, stop=1.0800, size="1")
        position.close_all(price=Price(1.0900), ts=NOW, reason=ExitReason.TAKE_PROFIT)
        assert realized_pnl(
            position,
            instrument=registry["EURUSD"],
            account_currency="USD",
            converter=SameCurrencyConverter(),
        ) == pytest.approx(Decimal("500"))

    def test_xauusd_pnl_carries_a_multiplier_of_a_hundred_ounces(
        self, registry: InstrumentRegistry
    ) -> None:
        # 1 lot from 2050 to 2060 is a 10 dollar move on 100 ounces = 1 000 USD.
        position = opened("XAUUSD", entry=2050.0, stop=2040.0, size="1")
        position.close_all(price=Price(2060.0), ts=NOW, reason=ExitReason.TAKE_PROFIT)
        assert realized_pnl(
            position,
            instrument=registry["XAUUSD"],
            account_currency="USD",
            converter=SameCurrencyConverter(),
        ) == Decimal("1000")

    def test_a_loss_is_negative(self, registry: InstrumentRegistry) -> None:
        position = opened("NAS100", entry=18000.0, stop=17900.0, size="1")
        position.close_all(price=Price(17900.0), ts=NOW, reason=ExitReason.PROTECTIVE_STOP)
        assert realized_pnl(
            position,
            instrument=registry["NAS100"],
            account_currency="USD",
            converter=SameCurrencyConverter(),
        ) == Decimal("-2000")

    def test_a_short_profits_when_price_falls(self, registry: InstrumentRegistry) -> None:
        position = ManagedPosition(
            symbol="NAS100",
            side=Side.SELL,
            entry_price=Price(18000.0),
            size=Decimal("1"),
            initial_stop=Price(18100.0),
            opened_at=NOW,
        )
        position.close_all(price=Price(17950.0), ts=NOW, reason=ExitReason.TAKE_PROFIT)
        assert realized_pnl(
            position,
            instrument=registry["NAS100"],
            account_currency="USD",
            converter=SameCurrencyConverter(),
        ) == Decimal("1000")


class TestCurrencyConversion:
    def test_a_jpy_quoted_instrument_is_converted(self, registry: InstrumentRegistry) -> None:
        # 1 lot of GBPJPY over 50 pips: 0.50 JPY * 100 000 = 50 000 JPY.
        # At USDJPY 150 that is 333.33 USD.
        position = opened("GBPJPY", entry=190.00, stop=189.50, size="1")
        position.close_all(price=Price(190.50), ts=NOW, reason=ExitReason.TAKE_PROFIT)
        converted = realized_pnl(
            position,
            instrument=registry["GBPJPY"],
            account_currency="USD",
            converter=StaticFxConverter({("USD", "JPY"): USDJPY}),
        )
        assert converted == pytest.approx(Decimal("333.3333"), abs=Decimal("0.001"))

    def test_each_leg_is_converted_at_its_own_fill_time(self, registry: InstrumentRegistry) -> None:
        # A leg closed in March and one closed in June are two conversions of
        # two sums. Averaging them over one rate would smear a currency move
        # across a trade that never experienced it.
        march = datetime(2024, 3, 1, 12, 0, tzinfo=UTC)
        june = datetime(2024, 6, 1, 12, 0, tzinfo=UTC)

        class DatedConverter:
            """USDJPY at 150 in March and 160 in June."""

            def rate(self, *, base: str, quote: str, at: datetime) -> Decimal:
                del base, quote
                return Decimal(1) / (Decimal("150") if at < june else Decimal("160"))

        position = ManagedPosition(
            symbol="GBPJPY",
            side=Side.BUY,
            entry_price=Price(190.00),
            size=Decimal("1"),
            initial_stop=Price(189.50),
            opened_at=march,
        )
        partial = ExitReason.PARTIAL_TAKE_PROFIT
        position.close(Decimal("0.5"), price=Price(190.50), ts=march, reason=partial)
        position.close(Decimal("0.5"), price=Price(190.50), ts=june, reason=partial)

        legs = position.legs
        first = leg_pnl(
            legs[0],
            instrument=registry["GBPJPY"],
            account_currency="USD",
            converter=DatedConverter(),
        )
        second = leg_pnl(
            legs[1],
            instrument=registry["GBPJPY"],
            account_currency="USD",
            converter=DatedConverter(),
        )
        # Identical price movement, different rate, therefore different money.
        assert legs[0].gross_quote_move == legs[1].gross_quote_move
        assert first > second

    def test_a_usd_quoted_instrument_never_consults_the_converter(
        self, registry: InstrumentRegistry
    ) -> None:
        # SameCurrencyConverter refuses every real pair, so this passing at all
        # proves no conversion was attempted.
        position = opened("NAS100", entry=18000.0, stop=17900.0, size="1")
        position.close_all(price=Price(18050.0), ts=NOW, reason=ExitReason.TAKE_PROFIT)
        assert realized_pnl(
            position,
            instrument=registry["NAS100"],
            account_currency="USD",
            converter=SameCurrencyConverter(),
        ) == Decimal("1000")

    def test_an_unconvertible_leg_raises_rather_than_counting_as_zero(
        self, registry: InstrumentRegistry
    ) -> None:
        # Silently skipping it would understate a loss.
        position = opened("GBPJPY", entry=190.00, stop=189.50, size="1")
        position.close_all(price=Price(189.50), ts=NOW, reason=ExitReason.PROTECTIVE_STOP)
        with pytest.raises(FxRateUnavailableError):
            realized_pnl(
                position,
                instrument=registry["GBPJPY"],
                account_currency="USD",
                converter=StaticFxConverter({}),
            )


class TestPartialCloses:
    def test_legs_sum(self, registry: InstrumentRegistry) -> None:
        # 1 lot of NAS100: half at +50 points, half at +100.
        # (0.5 * 50 + 0.5 * 100) * 20 = 1 500 USD.
        position = opened("NAS100", entry=18000.0, stop=17900.0, size="1")
        partial = ExitReason.PARTIAL_TAKE_PROFIT
        position.close(Decimal("0.5"), price=Price(18050.0), ts=NOW, reason=partial)
        position.close(Decimal("0.5"), price=Price(18100.0), ts=NOW, reason=partial)
        assert realized_pnl(
            position,
            instrument=registry["NAS100"],
            account_currency="USD",
            converter=SameCurrencyConverter(),
        ) == Decimal("1500")

    def test_a_position_with_no_closes_has_booked_nothing(
        self, registry: InstrumentRegistry
    ) -> None:
        position = opened("NAS100", entry=18000.0, stop=17900.0, size="1")
        assert realized_pnl(
            position,
            instrument=registry["NAS100"],
            account_currency="USD",
            converter=SameCurrencyConverter(),
        ) == Decimal(0)
