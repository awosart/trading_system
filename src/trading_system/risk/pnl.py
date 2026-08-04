"""Turning the Exit Engine's price movement into money in the account.

:class:`~trading_system.exit.position.ManagedPosition` books every close as a
:attr:`~trading_system.exit.position.ClosedLeg.gross_quote_move`: size times
price movement, in the instrument's **quote** currency, with no contract-size
multiplier and no costs. That is not an oversight and it is not P&L — it is the
most that layer can honestly compute, since it holds no
:class:`~trading_system.core.instruments.InstrumentSpec` and no FX rate. This
module supplies both and finishes the calculation:

    ``account P&L = gross_quote_move * contract_size * fx(quote -> account)``

The multiplier is why this exists. On NAS100 with ``contract_size`` of 1 the
factor is invisible; on EURUSD it is 100 000 and on XAUUSD 100, and a figure that
is wrong by five orders of magnitude on one instrument and right on another is
worse than one that is wrong everywhere.

**Each leg is converted at its own fill time**, not at the position's entry or at
the end of the run. A leg closed in March and a leg closed in June are two
separate conversions of two separate sums, which is what actually happened to the
account; averaging them over one rate would smear a currency move across trades
that never experienced it.

Commission and swap are not deducted here. They are per-lot figures already in
account currency, they need a position size in lots and a holding period rather
than a price movement, and folding them into the same function would produce a
number that is neither gross nor net. Cost accounting belongs with the execution
and portfolio layers.
"""

from decimal import Decimal

from trading_system.core.instruments import InstrumentSpec
from trading_system.exit.position import ClosedLeg, ManagedPosition
from trading_system.risk.conversion import FxConverter


def leg_pnl(
    leg: ClosedLeg,
    *,
    instrument: InstrumentSpec,
    account_currency: str,
    converter: FxConverter,
) -> Decimal:
    """Gross profit of one closed leg, in account currency.

    Args:
        leg: The realised close.
        instrument: Contract specification of what was traded.
        account_currency: Currency to express the result in.
        converter: Rate source, consulted at the leg's own fill time.

    Returns:
        Gross profit, signed. Before commission and swap.

    Raises:
        FxRateUnavailableError: If the quote currency cannot be converted at the fill
            time. Raised rather than skipped: an unconvertible leg silently
            treated as zero would understate a loss.
    """
    quote_pnl = leg.gross_quote_move * instrument.contract_size
    if instrument.quote_currency == account_currency:
        return quote_pnl
    rate = converter.rate(base=instrument.quote_currency, quote=account_currency, at=leg.ts)
    return quote_pnl * rate


def realized_pnl(
    position: ManagedPosition,
    *,
    instrument: InstrumentSpec,
    account_currency: str,
    converter: FxConverter,
) -> Decimal:
    """Gross profit booked so far on a position, in account currency.

    Args:
        position: The managed position, closed or partly closed.
        instrument: Contract specification of what was traded.
        account_currency: Currency to express the result in.
        converter: Rate source, consulted once per leg at that leg's fill time.

    Returns:
        Sum of every closed leg's profit, signed. Zero for a position with no
        closes yet. Before commission and swap.

    Raises:
        FxRateUnavailableError: If any leg cannot be converted at its fill time.
    """
    return sum(
        (
            leg_pnl(
                leg,
                instrument=instrument,
                account_currency=account_currency,
                converter=converter,
            )
            for leg in position.legs
        ),
        Decimal(0),
    )
