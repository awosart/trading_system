"""The reference fill model: when a resting order fires, and at what price."""

import pytest

from trading_system.core.types import OrderType, Price, Side
from trading_system.exit.fills import (
    approaches_from_below,
    resting_fill_price,
    resting_order_filled,
)

LONG_STOP = (Side.SELL, OrderType.STOP)
LONG_TAKE = (Side.SELL, OrderType.LIMIT)
SHORT_STOP = (Side.BUY, OrderType.STOP)
SHORT_TAKE = (Side.BUY, OrderType.LIMIT)


class TestDirection:
    @pytest.mark.parametrize(
        ("order", "from_below"),
        [(LONG_STOP, False), (LONG_TAKE, True), (SHORT_STOP, True), (SHORT_TAKE, False)],
    )
    def test_which_side_price_approaches_a_level_from(
        self, order: tuple[Side, OrderType], from_below: bool
    ) -> None:
        assert approaches_from_below(*order) is from_below


class TestFiring:
    def test_a_long_stop_fires_when_the_low_reaches_it(self) -> None:
        assert resting_order_filled(
            Price(1.0900), high=1.1050, low=1.0900, exit_side=Side.SELL, order_type=OrderType.STOP
        )
        assert not resting_order_filled(
            Price(1.0900), high=1.1050, low=1.0901, exit_side=Side.SELL, order_type=OrderType.STOP
        )

    def test_a_gap_through_the_level_fires_even_though_the_bar_never_brackets_it(self) -> None:
        # The bug a symmetric low <= level <= high test would have: this bar
        # opened below the stop and traded lower still, and is precisely the bar
        # the stop was hit on.
        assert resting_order_filled(
            Price(1.0900), high=1.0850, low=1.0800, exit_side=Side.SELL, order_type=OrderType.STOP
        )

    def test_a_long_take_fires_when_the_high_reaches_it(self) -> None:
        assert resting_order_filled(
            Price(1.1200), high=1.1200, low=1.1000, exit_side=Side.SELL, order_type=OrderType.LIMIT
        )
        assert not resting_order_filled(
            Price(1.1200), high=1.1199, low=1.1000, exit_side=Side.SELL, order_type=OrderType.LIMIT
        )

    def test_a_short_stop_fires_when_the_high_reaches_it(self) -> None:
        assert resting_order_filled(
            Price(1.1100), high=1.1100, low=1.1000, exit_side=Side.BUY, order_type=OrderType.STOP
        )

    def test_a_short_take_fires_when_the_low_reaches_it(self) -> None:
        assert resting_order_filled(
            Price(1.0800), high=1.1000, low=1.0800, exit_side=Side.BUY, order_type=OrderType.LIMIT
        )


class TestFillPrice:
    def test_an_ordinary_touch_fills_at_the_level(self) -> None:
        for exit_side, order_type in (LONG_STOP, LONG_TAKE, SHORT_STOP, SHORT_TAKE):
            level = Price(1.1000)
            # The bar opened on the ordinary side of the level in every case.
            bar_open = 1.1000
            assert resting_fill_price(
                level, bar_open=bar_open, exit_side=exit_side, order_type=order_type
            ) == pytest.approx(1.1000)

    def test_a_long_stop_gapped_through_fills_worse_than_its_level(self) -> None:
        assert resting_fill_price(
            Price(1.0900), bar_open=1.0850, exit_side=Side.SELL, order_type=OrderType.STOP
        ) == pytest.approx(1.0850)

    def test_a_short_stop_gapped_through_fills_worse_than_its_level(self) -> None:
        assert resting_fill_price(
            Price(1.1100), bar_open=1.1180, exit_side=Side.BUY, order_type=OrderType.STOP
        ) == pytest.approx(1.1180)

    def test_a_long_take_gapped_through_fills_better_than_its_level(self) -> None:
        assert resting_fill_price(
            Price(1.1200), bar_open=1.1260, exit_side=Side.SELL, order_type=OrderType.LIMIT
        ) == pytest.approx(1.1260)

    def test_a_short_take_gapped_through_fills_better_than_its_level(self) -> None:
        assert resting_fill_price(
            Price(1.0800), bar_open=1.0750, exit_side=Side.BUY, order_type=OrderType.LIMIT
        ) == pytest.approx(1.0750)

    def test_a_stop_never_fills_better_than_its_level_and_a_limit_never_worse(self) -> None:
        # The single sentence the whole table reduces to, over a grid of opens.
        for bar_open in (1.0700, 1.0900, 1.1000, 1.1100, 1.1300):
            level = Price(1.1000)
            long_stop = resting_fill_price(
                level, bar_open=bar_open, exit_side=Side.SELL, order_type=OrderType.STOP
            )
            long_take = resting_fill_price(
                level, bar_open=bar_open, exit_side=Side.SELL, order_type=OrderType.LIMIT
            )
            assert long_stop <= level
            assert long_take >= level
