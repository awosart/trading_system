"""Where an order would have executed, in the price series a backtest reads.

Every model here answers in **mid** space and stops there. What the account
actually receives is :mod:`trading_system.execution.costs`' answer, and keeping
the two apart is what makes the spread accounting checkable: there is exactly one
place a price stops being a mid, and it is not in this module.

Two families, because two different questions are being asked.

**Market timing** — a decision was taken on ``close(t)``; when does it execute?
:class:`NextBarOpen` says ``open(t+1)``, which is the only honest answer from bar
data and the default everywhere. :class:`SameBarClose` says ``close(t)``, which is
the model that makes every backtest look better than the account and is therefore
noisy about itself: it warns on construction, naming what it is doing, because a
model chosen by editing one config line should not be able to change the result
by twenty percent without saying so.

**Resting orders** — an order sat in the market at a level; did the bar reach it,
and at what price? This replaces P07's
:func:`~trading_system.exit.fills.resting_fill_price`, which said in its own
docstring that it was a reference model optimistic about bad fills. Two changes,
both in the same direction:

* A **stop** gapped through no longer fills at the bar's open. The open is the
  first price that printed, not the price a market order swept through an empty
  book got, and the difference grows with the hole — so the penalty is
  proportional to the gap rather than flat. See
  :class:`~trading_system.execution.config.GapConfig`.
* A **limit** gapped through no longer fills at the bar's open either. P07 gave
  it the better of level and open; here it fills at its level, because free
  positive slippage on every gap through a target is the most flattering
  assumption available and it is not one anybody measured. A venue that really
  fills this way sets ``grant_improvement_to_limits``.

:class:`LimitTouch` adds the boundary the reference model glossed over. A limit
fills when price traded **past** it, not when price merely reached it: at the
level you are behind everything already queued at that price, and a bar that
turned around exactly there is a bar in which the queue was not cleared. The
touch is not impossible, just improbable, so it is a configured probability drawn
from the fill's own random stream — which means it is reproducible per fill and
independent of what else the run contains.
"""

from dataclasses import dataclass
from decimal import Decimal

from trading_system.core.instruments import InstrumentSpec
from trading_system.core.logging import get_logger
from trading_system.core.types import Bar, OrderType, Price, Side
from trading_system.execution.config import GapConfig, LimitFillConfig
from trading_system.execution.orders import ExecutionOrder
from trading_system.execution.rng import fill_rng
from trading_system.exit.fills import approaches_from_below

logger = get_logger(__name__)


class MarketFillModel:
    """When a decision taken on a bar's close reaches the market.

    Subclasses answer with a mid price and nothing else; costs are applied
    afterwards.
    """

    def reference_price(self, decision_bar: Bar, next_bar: Bar | None) -> Price | None:
        """The mid price a market order taken on ``decision_bar``'s close fills at.

        Args:
            decision_bar: The bar whose close the decision was taken on.
            next_bar: The following bar, or ``None`` at the end of the series.

        Returns:
            The mid fill price, or ``None`` when the order cannot execute.
        """
        raise NotImplementedError


class NextBarOpen(MarketFillModel):
    """Execute at the open of the bar after the decision. The default.

    The only timing a bar series actually supports. A decision taken on
    ``close(t)`` is knowable at ``close(t)`` and not before, and the first price
    at which it can transact is ``open(t+1)``.

    A decision on the final bar of a series returns ``None`` rather than falling
    back to that bar's close. The run simply ended before the order could
    execute, and inventing a fill there would put one free trade at the end of
    every backtest — always taken at a price the strategy already knew.
    """

    def reference_price(self, decision_bar: Bar, next_bar: Bar | None) -> Price | None:
        """The next bar's open.

        Args:
            decision_bar: The bar whose close the decision was taken on.
            next_bar: The following bar, or ``None`` at the end of the series.

        Returns:
            ``next_bar.open``, or ``None`` if the series ended.
        """
        _ = decision_bar
        return None if next_bar is None else next_bar.open


class SameBarClose(MarketFillModel):
    """Execute at the close the decision was taken on. Optimistic, and says so.

    This fills at a price that was only knowable at the instant the decision
    became knowable — no time passes between deciding and transacting. It is not
    a fraud, because a strategy running on the close of a liquid bar really can
    get something near it; it is optimistic, because it removes the entire gap
    between the signal and the fill, which is where a meaningful share of a
    short-timeframe strategy's edge quietly lives.

    Selecting it logs a warning naming the effect, once per instance rather than
    once per bar: a per-bar warning in a run of two hundred thousand bars is a
    line nobody reads, which is the same reasoning that made P06 count dropped
    signals instead of only logging them.
    """

    def __init__(self) -> None:
        """Warn that this model is optimistic, at the moment it is chosen."""
        logger.warning(
            "execution.optimistic_fill_model",
            model="SameBarClose",
            detail=(
                "fills at close(t), the price the decision was derived from, so no time "
                "passes between signal and execution; results are an upper bound, and "
                "NextBarOpen is the model a live run can reproduce"
            ),
        )

    def reference_price(self, decision_bar: Bar, next_bar: Bar | None) -> Price | None:
        """The decision bar's own close.

        Args:
            decision_bar: The bar whose close the decision was taken on.
            next_bar: Unused; this model never looks forward.

        Returns:
            ``decision_bar.close``.
        """
        _ = next_bar
        return decision_bar.close


@dataclass(frozen=True)
class RestingFill:
    """A resting order's execution, before costs.

    Attributes:
        mid_price: Price the order executed at, in mid space.
        gap_points: How far beyond the level the market had already moved when
            the bar opened, in points. Zero for an ordinary fill.
    """

    mid_price: Price
    gap_points: float


class LimitTouch:
    """A limit order fills when price trades past its level, not when it reaches it.

    The strict inequality is the point. ``low <= level`` credits a limit with
    every bar whose extreme landed exactly on it, and those are precisely the
    bars where the order was at the back of a queue that never cleared. The
    difference looks small on one trade and is systematic across a mean-reversion
    strategy, whose best entries are all at exactly this boundary.
    """

    __slots__ = ("_config", "_gap", "_run_seed")

    def __init__(self, config: LimitFillConfig, gap: GapConfig, run_seed: int) -> None:
        """Bind the model to its configuration.

        Args:
            config: Touch-fill probability.
            gap: What a gap through the level does — for a limit, whether the
                improvement is granted at all.
            run_seed: Seed for the per-fill random streams.
        """
        self._config = config
        self._gap = gap
        self._run_seed = run_seed

    def fill(
        self,
        level: Price,
        *,
        bar: Bar,
        side: Side,
        instrument: InstrumentSpec,
        order_id: str,
    ) -> RestingFill | None:
        """Whether a limit resting at ``level`` filled during ``bar``.

        Args:
            level: Price the order rests at, in mid space.
            bar: The bar to test.
            side: Side of the limit order.
            instrument: Contract specification, for the point size.
            order_id: Identifier, seeding the touch draw.

        Returns:
            The fill, or ``None`` if the order did not execute.
        """
        rising = approaches_from_below(side, OrderType.LIMIT)
        extreme = bar.high if rising else bar.low
        beyond = extreme > level if rising else extreme < level

        if not beyond:
            if extreme != level:
                return None
            # Reached exactly and no further: the queue at that price may or may
            # not have cleared, and the bar does not record which.
            rng = fill_rng(
                self._run_seed, symbol=instrument.symbol, ts=bar.timestamp, order_id=order_id
            )
            if rng.random() >= self._config.touch_fill_probability:
                return None

        gap_price = self._gap_beyond(level, bar_open=bar.open, rising=rising)
        gap_points = instrument.price_to_points(gap_price)
        if gap_price > 0 and self._gap.grant_improvement_to_limits:
            return RestingFill(mid_price=bar.open, gap_points=gap_points)
        return RestingFill(mid_price=level, gap_points=gap_points)

    @staticmethod
    def _gap_beyond(level: Price, *, bar_open: float, rising: bool) -> float:
        """How far the bar opened past the level, in price units.

        Args:
            level: The resting level.
            bar_open: The bar's open.
            rising: Whether the level is reached by a rising market.

        Returns:
            The distance past the level, or zero if the open was on the ordinary
            side of it.
        """
        beyond = bar_open - level if rising else level - bar_open
        return max(beyond, 0.0)


class GapFill:
    """A stop order, including one the market gapped straight through.

    An untriggered stop rests at a level and fills there when price reaches it
    the ordinary way. When a bar **opens** past the level — a weekend gap, a news
    print — the order was already stale before the first trade, and P07's
    reference model filled it at that first trade. This model does not: it fills
    worse than the open, by an amount proportional to the hole, because a hole
    forty points wide is not one in which the first printable price was
    available in size.

    The penalty is applied by the cost model rather than here, so that a fill's
    slippage stays decomposed into what the gap contributed and what ordinary
    execution did. This model's job is to report the mid price and the size of
    the gap; :attr:`~trading_system.execution.orders.ExecutionOrder.gap_points`
    carries the second across.
    """

    __slots__ = ()

    def fill(
        self,
        level: Price,
        *,
        bar: Bar,
        side: Side,
        instrument: InstrumentSpec,
        order_id: str,
    ) -> RestingFill | None:
        """Whether a stop resting at ``level`` filled during ``bar``.

        Args:
            level: Price the order rests at, in mid space.
            bar: The bar to test.
            side: Side of the stop order.
            instrument: Contract specification, for the point size.
            order_id: Unused; a stop that is reached always fills, so there is
                no draw to seed.

        Returns:
            The fill, or ``None`` if the level was not reached.
        """
        _ = order_id
        rising = approaches_from_below(side, OrderType.STOP)
        reached = bar.high >= level if rising else bar.low <= level
        if not reached:
            return None

        # Directional on purpose: a bar that opens below a long's stop never
        # brackets it, and that is exactly the bar the stop was hit on.
        beyond = bar.open - level if rising else level - bar.open
        if beyond <= 0:
            return RestingFill(mid_price=level, gap_points=0.0)
        return RestingFill(mid_price=bar.open, gap_points=instrument.price_to_points(beyond))


@dataclass(frozen=True)
class RestingOrderModel:
    """Dispatches a resting order to the model for its type.

    Attributes:
        limit: Model for limit orders.
        stop: Model for stop orders.
    """

    limit: LimitTouch
    stop: GapFill

    def fill(
        self,
        level: Price,
        *,
        bar: Bar,
        side: Side,
        order_type: OrderType,
        instrument: InstrumentSpec,
        order_id: str,
    ) -> RestingFill | None:
        """Execute a resting order of either type.

        Args:
            level: Price the order rests at.
            bar: The bar to test.
            side: Side of the order.
            order_type: Whether it rests as a stop or a limit.
            instrument: Contract specification.
            order_id: Identifier, seeding any draw the model makes.

        Returns:
            The fill, or ``None``.

        Raises:
            ValueError: If ``order_type`` is ``MARKET``, which does not rest.
        """
        if order_type is OrderType.MARKET:
            raise ValueError("a MARKET order does not rest at a level; use a MarketFillModel")
        model = self.stop if order_type is OrderType.STOP else self.limit
        return model.fill(level, bar=bar, side=side, instrument=instrument, order_id=order_id)


def order_from_fill(
    resting: RestingFill,
    *,
    order_id: str,
    symbol: str,
    side: Side,
    order_type: OrderType,
    size: Decimal,
) -> ExecutionOrder:
    """Package a resting fill as an order ready for costing.

    Args:
        resting: What the fill model resolved.
        order_id: Identifier.
        symbol: Instrument.
        side: Side of the order.
        order_type: How the order reached the market.
        size: Size in lots.

    Returns:
        The order, carrying the gap the fill model measured.
    """
    return ExecutionOrder(
        order_id=order_id,
        symbol=symbol,
        side=side,
        order_type=order_type,
        size=size,
        mid_price=resting.mid_price,
        gap_points=resting.gap_points,
    )
