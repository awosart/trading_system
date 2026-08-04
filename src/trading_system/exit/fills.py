"""Reference fill model for orders resting inside a bar.

**This is a backtest model, not a fact.** It answers two questions from OHLC
alone — was a level reached during the bar, and at what price would the order
have filled — and OHLC does not contain the answer to either with certainty. In
particular the gap case here is *optimistic about a bad fill*: it assumes an
order gapped through fills at the bar's open, whereas in reality a stop swept
through a hole in the book fills below it, sometimes far below. Weekend gaps and
news prints are where the difference lives. P13 replaces this with a slippage
model that widens the fill; nothing above this module should treat the price it
returns as executed truth.

What the model does get exactly right is direction. Which side of a level price
approaches from is fully determined by two things the decision already carries:
the side that closes the position, and whether the order is a stop or a limit.

===============  ===========  ==========================  ============================
Closing side     Order type   Level sits                  A gap through it fills
===============  ===========  ==========================  ============================
``SELL``         ``STOP``     below (a long's stop)       lower — worse
``SELL``         ``LIMIT``    above (a long's target)     higher — better
``BUY``          ``STOP``     above (a short's stop)      higher — worse
``BUY``          ``LIMIT``    below (a short's target)    lower — better
===============  ===========  ==========================  ============================

Both columns collapse to one sentence: **a stop fills at the worse of its level
and the open, a limit at the better.** That is why ``order_type`` is a load-
bearing field of :class:`~trading_system.exit.base.ExitDecision` rather than
decoration — drop it and the gap direction becomes a guess.
"""

from trading_system.core.types import OrderType, Price, Side


def approaches_from_below(exit_side: Side, order_type: OrderType) -> bool:
    """Whether price has to rise to reach a resting order's level.

    Args:
        exit_side: Side of the order that closes the position — ``SELL`` closes
            a long.
        order_type: Whether the order rests as a stop or a limit.

    Returns:
        ``True`` if the level sits above price and is reached by a rising
        market, ``False`` if it sits below and is reached by a falling one.
    """
    return (exit_side is Side.BUY) == (order_type is OrderType.STOP)


def resting_order_filled(
    level: Price, *, high: float, low: float, exit_side: Side, order_type: OrderType
) -> bool:
    """Whether a bar reached the level an order was resting at.

    Directional on purpose. A symmetric ``low <= level <= high`` test would miss
    every gap: a bar that opens below a long's stop never brackets it, and yet
    that is precisely the bar the stop was hit on.

    Args:
        level: Price the order rests at.
        high: Bar high.
        low: Bar low.
        exit_side: Side of the closing order.
        order_type: Whether the order rests as a stop or a limit.

    Returns:
        Whether the order would have been filled during the bar.
    """
    if approaches_from_below(exit_side, order_type):
        return high >= level
    return low <= level


def resting_fill_price(
    level: Price, *, bar_open: float, exit_side: Side, order_type: OrderType
) -> Price:
    """Price a resting order would have filled at, given the bar's open.

    A backtest estimate — see this module's docstring. A stop takes the worse of
    its level and the open, a limit the better; when the bar opened on the
    ordinary side of the level, both reduce to the level itself.

    Args:
        level: Price the order rests at.
        bar_open: Open of the bar the order filled in.
        exit_side: Side of the closing order.
        order_type: Whether the order rests as a stop or a limit.

    Returns:
        The modelled fill price.
    """
    # A level approached from below is passed by an open *above* it, and one
    # approached from above by an open *below* it. Both readings of the table in
    # this module's docstring reduce to that single asymmetry.
    if approaches_from_below(exit_side, order_type):
        return Price(max(level, bar_open))
    return Price(min(level, bar_open))
