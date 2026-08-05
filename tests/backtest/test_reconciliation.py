"""The equity curve and the trade list must be the same money, twice.

The DoD asks for agreement "to the cent". These tests claim something stronger:
agreement **exactly**, in decimal. Money only ever enters the balance through a
booked leg, a commission or a financing charge, each of which is a
:class:`~decimal.Decimal`, so the identity is arithmetic rather than a comparison
against a tolerance. Needing a tolerance would itself be the finding.
"""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from tests.backtest.conftest import (
    EURUSD_H1,
    bars,
    costless_registry,
    flat_costs,
    orchestrator,
    strategy,
)
from trading_system.backtest.config import BacktestConfig
from trading_system.backtest.orchestrator import BacktestResult, StrategyBinding
from trading_system.core.instruments import InstrumentRegistry
from trading_system.core.types import Side
from trading_system.data.resample import FX_DAY_ORIGIN, trading_day
from trading_system.execution.config import CostConfig, PerLotRolloverSwap
from trading_system.exit.library import ExitLibrarySpec, ExitPresetSpec

#: A rising series that ends with one sharp drop. The rise makes the trigger fire
#: and the instrument trend; the drop takes out whatever stop is live on the last
#: bar, so the run ends flat. That matters: a position still open at the end holds
#: commission and financing that no trade record carries, and the trade list would
#: then legitimately fail to add up to the balance.
RISING = [1.1000 + 0.0020 * i for i in range(59)] + [1.0500]

#: Where the setups are disproven. Close enough to the opening price that the
#: preset's 2R target is reachable inside the series — with the invalidation far
#: away, R is so wide that neither the stop nor the target is ever touched and the
#: run produces one open position and no trades, which proves nothing.
RISING_INVALIDATION = 1.0950

#: Buy-and-hold starts in the evening so the holding period crosses exactly one
#: 17:00 New York rollover — into a Monday label, which carries a single day of
#: financing rather than the tripled Wednesday or the unbilled weekend. That is
#: what makes the swap term a number this test can name.
HOLD_START = datetime(2024, 3, 4, 18, 0, tzinfo=UTC)

#: Rising hourly closes, and an open below the first of them so that bar 0 is an
#: up bar and the trigger fires on it. Left at the default, bar 0 would open and
#: close at the same price and every index below would shift.
#:
#: The series stops one bar after the exit fills, deliberately. The strategy
#: re-triggers on every up bar, so a longer series is not buy-and-hold at all —
#: it is a sequence of trades, and the hand computation would be checking the
#: first of several rather than the whole run.
HOLD_CLOSES = [1.1000 + 0.0010 * i for i in range(1, 13)]
HOLD_FIRST_OPEN = 1.1000

#: Bars the ``time_boxed`` preset is pinned to for the hand computation.
HOLD_BARS = 10


def run_rising(
    registry: InstrumentRegistry,
    preset: ExitPresetSpec,
    *,
    costs: CostConfig | None = None,
) -> BacktestResult:
    """One run of the price-only strategy over a rising EURUSD H1 series."""
    spec = strategy(invalidation=RISING_INVALIDATION)
    return orchestrator(
        registry=registry,
        streams={EURUSD_H1: bars(RISING)},
        bindings=[StrategyBinding(spec=spec, exit_preset=preset, keys=(EURUSD_H1,))],
        costs=costs,
    ).run()


class TestCurveReconciles:
    """Every row of the curve is a restatement of what has been booked."""

    def test_the_final_balance_is_exactly_the_sum_of_the_trades(
        self, registry: InstrumentRegistry, preset: ExitPresetSpec
    ) -> None:
        result = run_rising(registry, preset)
        assert result.trades, "the scenario must actually trade, or it proves nothing"
        assert result.open_at_end == 0, "an open position holds costs no trade record carries"

        booked = sum((trade.net for trade in result.trades), Decimal(0))
        assert result.curve[-1].balance == BacktestConfig().starting_balance + booked

    def test_every_row_satisfies_balance_equals_realized_less_costs(
        self, registry: InstrumentRegistry, preset: ExitPresetSpec
    ) -> None:
        """Not just the last row: the identity holds at every instant."""
        result = run_rising(registry, preset)
        start = BacktestConfig().starting_balance
        for point in result.curve:
            assert point.balance == (
                start + point.realized - point.commission_paid + point.swap_paid
            )

    def test_every_row_satisfies_equity_equals_balance_plus_unrealized(
        self, registry: InstrumentRegistry, preset: ExitPresetSpec
    ) -> None:
        result = run_rising(registry, preset)
        for point in result.curve:
            assert point.equity == point.balance + point.unrealized

    def test_each_trades_net_is_its_own_decomposition(
        self, registry: InstrumentRegistry, preset: ExitPresetSpec
    ) -> None:
        result = run_rising(registry, preset)
        for trade in result.trades:
            assert trade.net == trade.gross - trade.commission + trade.swap

    def test_realized_across_the_run_is_the_sum_of_the_trades_gross(
        self, registry: InstrumentRegistry, preset: ExitPresetSpec
    ) -> None:
        """Ties the curve's running totals to the records, not only the balance."""
        result = run_rising(registry, preset)
        final = result.curve[-1]
        assert final.realized == sum((trade.gross for trade in result.trades), Decimal(0))
        assert final.commission_paid == sum(
            (trade.commission for trade in result.trades), Decimal(0)
        )
        assert final.swap_paid == sum((trade.swap for trade in result.trades), Decimal(0))

    def test_the_curve_has_one_row_per_instant_not_one_per_event(
        self, registry: InstrumentRegistry, preset: ExitPresetSpec
    ) -> None:
        """Several rows at one instant would make drawdown depend on phase order."""
        result = run_rising(registry, preset)
        stamps = [point.ts for point in result.curve]
        assert len(stamps) == len(set(stamps)) == result.instants == len(RISING)

    def test_every_row_carries_the_trading_day_p14_will_group_by(
        self, registry: InstrumentRegistry, preset: ExitPresetSpec
    ) -> None:
        """P14 takes the last row per label and defines no day of its own."""
        origin = BacktestConfig().day_origin
        result = run_rising(registry, preset)
        for point in result.curve:
            assert point.day == trading_day(point.ts, origin)

    def test_the_curve_marks_on_every_bar_rather_than_on_every_trade(
        self, registry: InstrumentRegistry, preset: ExitPresetSpec
    ) -> None:
        """Equity moves while a position is open, not only when one closes."""
        result = run_rising(registry, preset)
        held = [point for point in result.curve if point.open_positions > 0]
        assert len(held) > len(result.trades)
        assert len({point.equity for point in held}) > 1


class TestBuyAndHold:
    """One position, opened once and closed by the clock, against hand arithmetic.

    Every expected figure below is built from the bar prices, the shipped
    instrument specification and the rollover convention — never read back off
    the result it is checking. The one figure taken from the run is the position
    *size*, which is the Risk Engine's answer and has its own tests in P10; what
    is under test here is the money chain that follows from it.
    """

    @pytest.fixture
    def hold_preset(self, library: ExitLibrarySpec) -> ExitPresetSpec:
        """A protective stop plus "close after ten bars", and nothing else.

        Built from the shipped ``time_boxed`` preset with its bar count pinned,
        so the exit fires at a bar this test names rather than at one that
        happens to be configured.
        """
        spec = next(item for item in library.presets if item.id == "time_boxed")
        rules = [rule.model_copy(update={"max_bars_held": HOLD_BARS}) for rule in spec.rules]
        return spec.model_copy(update={"rules": rules})

    def _run(
        self, registry: InstrumentRegistry, hold_preset: ExitPresetSpec, costs: CostConfig
    ) -> BacktestResult:
        """Buy and hold one position over the evening series."""
        return orchestrator(
            registry=registry,
            streams={EURUSD_H1: bars(HOLD_CLOSES, start=HOLD_START, first_open=HOLD_FIRST_OPEN)},
            bindings=[StrategyBinding(spec=strategy(), exit_preset=hold_preset, keys=(EURUSD_H1,))],
            costs=costs,
        ).run()

    @staticmethod
    def _bar_opens() -> list[float]:
        """Open price of each bar, which is the previous close.

        The same rule :func:`~tests.backtest.conftest.bars` builds the frame by,
        restated here rather than imported, so the expectation does not inherit a
        mistake from the builder.
        """
        return [HOLD_FIRST_OPEN, *HOLD_CLOSES[:-1]]

    @staticmethod
    def _bar_index(ts: datetime) -> int:
        """Which bar of the hold series an instant is the open of."""
        return int((ts - HOLD_START) // timedelta(hours=1))

    def test_the_position_opens_and_closes_at_the_bars_the_clock_names(
        self, registry: InstrumentRegistry, hold_preset: ExitPresetSpec
    ) -> None:
        """The signal fires on bar 0's close; the exit is decided on bar 10's."""
        result = self._run(costless_registry(registry), hold_preset, flat_costs())
        trade = result.trades[0]

        # Bar 0 closes up, so the signal fires on its close and the market order
        # fills at open(bar 1).
        assert self._bar_index(trade.opened_at) == 1
        assert trade.opened_at == HOLD_START + timedelta(hours=1)

        # Held through bars 1..10 — ten bars — so TimeExit decides on bar 10's
        # close. A BAR_CLOSE decision names no price and fills at the next open.
        assert self._bar_index(trade.closed_at) == 1 + HOLD_BARS
        assert trade.closed_at == HOLD_START + timedelta(hours=1 + HOLD_BARS)
        assert trade.side is Side.BUY
        assert trade.legs == 1

    def test_with_costs_removed_the_result_is_the_bare_price_move(
        self, registry: InstrumentRegistry, hold_preset: ExitPresetSpec
    ) -> None:
        """Isolates the price arithmetic from the cost arithmetic.

        Spread and commission at zero and no financing configured, so a fill is
        its mid price exactly and the gross is a move between two bar opens this
        test names.
        """
        free = costless_registry(registry)
        result = self._run(free, hold_preset, flat_costs())
        trade = result.trades[0]

        opens = self._bar_opens()
        entry_mid = Decimal(str(opens[self._bar_index(trade.opened_at)]))
        exit_mid = Decimal(str(opens[self._bar_index(trade.closed_at)]))

        assert trade.commission == 0
        assert trade.swap == 0
        assert trade.gross == (exit_mid - entry_mid) * trade.size * free["EURUSD"].contract_size
        assert result.curve[-1].balance == BacktestConfig().starting_balance + trade.net

    def test_buy_and_hold_matches_a_hand_computation_including_costs_and_swap(
        self, registry: InstrumentRegistry, hold_preset: ExitPresetSpec
    ) -> None:
        """The whole chain priced by hand: fills across the spread, commission, financing.

        Costs are switched on deliberately — the shipped EURUSD spread and
        commission — because a buy-and-hold that reconciles only when everything
        is free proves nothing about the arithmetic that actually runs.
        """
        costs = flat_costs().model_copy(update={"swap": {"FX": PerLotRolloverSwap()}})
        result = self._run(registry, hold_preset, costs)
        trade = result.trades[0]
        instrument = registry["EURUSD"]

        # 1. Fills. A BUY lifts the offer and a SELL hits the bid, each by half
        #    the spread, snapped away from the mid onto the tick grid.
        opens = self._bar_opens()
        half = instrument.typical_spread_points / 2
        entry_price = instrument.shift_price(opens[self._bar_index(trade.opened_at)], half)
        exit_price = instrument.shift_price(opens[self._bar_index(trade.closed_at)], -half)
        assert trade.entry_price == entry_price

        expected_gross = (
            (Decimal(str(exit_price)) - Decimal(str(entry_price)))
            * trade.size
            * instrument.contract_size
        )
        assert trade.gross == expected_gross

        # 2. Commission, per fill: one entry and one closing leg.
        assert trade.commission == instrument.commission_per_side * trade.size * 2

        # 3. Financing: exactly one rollover, into a Monday label, at one day.
        crossed = _labels_crossed(trade.opened_at, trade.closed_at)
        assert crossed == [datetime(2024, 3, 4).date()]
        assert crossed[0].weekday() == 0, "a Monday label: neither tripled nor closed"
        assert trade.swap == instrument.swap_long * trade.size * 1

        # 4. The chain closes: the curve says what the trade says.
        assert trade.net == trade.gross - trade.commission + trade.swap
        assert result.curve[-1].balance == BacktestConfig().starting_balance + trade.net

    def test_the_round_turn_costs_one_spread_not_two_and_not_none(
        self, registry: InstrumentRegistry, hold_preset: ExitPresetSpec
    ) -> None:
        """P12's invariant, observed end to end rather than at the cost model.

        The difference between the costed and the costless run is the spread
        across two fills plus the commission on two fills — one spread, not the
        two a "full spread per side" implementation would charge.
        """
        costed = self._run(registry, hold_preset, flat_costs()).trades[0]
        free = self._run(costless_registry(registry), hold_preset, flat_costs()).trades[0]
        instrument = registry["EURUSD"]

        assert costed.size == free.size
        spread_cost = (
            Decimal(str(instrument.typical_spread_points))
            * Decimal(str(instrument.point_size))
            * costed.size
            * instrument.contract_size
        )
        assert free.gross - costed.gross == pytest.approx(spread_cost, abs=Decimal("0.02"))


def _labels_crossed(opened: datetime, closed: datetime) -> list[object]:
    """Trading-day labels entered strictly after the one the position opened in.

    The same rule the financing model applies, written out here rather than
    imported from it, so the expectation is independent of the code under test.
    """
    first = trading_day(opened, FX_DAY_ORIGIN)
    last = trading_day(closed, FX_DAY_ORIGIN)
    labels: list[object] = []
    label = first + timedelta(days=1)
    while label <= last:
        labels.append(label)
        label += timedelta(days=1)
    return labels
