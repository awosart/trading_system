"""Assembling the rate source a run needs from the bars a store holds.

:class:`~trading_system.risk.conversion.BarFxConverter` deliberately takes
loaded frames rather than a store — its own docstring says the wiring "belongs to
the backtest orchestrator, which knows what date range the run needs". This is
that wiring, kept out of ``risk/`` so the sizing layer still does not import a
data store, and kept off :class:`~trading_system.data.store.ParquetStore` so it
can be exercised without one: the caller passes a ``load`` callable.

**A run that needs no conversion keeps the converter it always had.** When every
instrument is quoted in the account currency,
:func:`build_converter` returns :class:`~trading_system.risk.conversion.SameCurrencyConverter`
rather than an empty :class:`~trading_system.risk.conversion.BarFxConverter`.
The two behave identically — both refuse every genuine pair — but the converter
is folded into ``run_id`` by
:meth:`~trading_system.backtest.spec.RunInputs.components`, so swapping one for
the other would renumber every EURUSD run ever stored while changing no
decision. Identity has to move only when behaviour does.

**Which conversion timeframe.** The rate series is read at its own bar closes,
so a coarser series is staler and a finer one is not more accurate for a
decision taken on a coarse bar. The default is therefore the traded timeframe
itself: a rate whose bar closes on the same schedule as the decision is the one
that was actually observable when the decision was taken.
"""

from collections.abc import Callable, Iterable
from datetime import timedelta

from trading_system.core.exceptions import DataError
from trading_system.core.instruments import InstrumentSpec
from trading_system.core.types import Timeframe
from trading_system.data.models import OHLCVFrame
from trading_system.risk.conversion import (
    DEFAULT_MAX_STALENESS,
    BarFxConverter,
    FxConverter,
    SameCurrencyConverter,
    conversion_pair_candidates,
    required_conversion_currencies,
)

#: Loads one symbol's bars, or returns an empty frame when the store has none.
FrameLoader = Callable[[str, Timeframe], OHLCVFrame]


class ConversionSeriesMissingError(DataError):
    """A run needs a rate series that the store does not carry.

    Raised at assembly time rather than left to surface as a refused signal per
    bar. The two failures look the same in a counter — ``fx_rate_unavailable``
    either way — but they are different problems: a gap inside a loaded series
    is a data condition affecting some signals, while an entirely absent series
    means the run was configured to trade something it can never size. The first
    deserves a counter, the second deserves a stop.
    """


def load_conversion_series(
    currencies: Iterable[str],
    *,
    account_currency: str,
    timeframe: Timeframe,
    load: FrameLoader,
) -> dict[str, OHLCVFrame]:
    """Load one rate series per currency, in whichever direction the store has it.

    Args:
        currencies: Quote currencies needing conversion — see
            :func:`~trading_system.risk.conversion.required_conversion_currencies`.
        account_currency: Currency the account is denominated in.
        timeframe: Bar size to load the rate series at.
        load: Reads one symbol's bars; an empty frame means "not carried".

    Returns:
        Pair symbol to its bars, ready for :class:`BarFxConverter`.

    Raises:
        ConversionSeriesMissingError: If neither direction of a needed pair is
            carried, naming both spellings that were looked for.
    """
    series: dict[str, OHLCVFrame] = {}
    for currency in currencies:
        candidates = conversion_pair_candidates(currency, account_currency)
        for symbol in candidates:
            frame = load(symbol, timeframe)
            if not frame.is_empty:
                series[symbol] = frame
                break
        else:
            raise ConversionSeriesMissingError(
                f"{currency}/{account_currency}: the store carries neither "
                f"{' nor '.join(candidates)} at {timeframe.value}, so nothing quoted in "
                f"{currency} can be sized on a {account_currency} account"
            )
    return series


def build_converter(
    instruments: Iterable[InstrumentSpec],
    *,
    account_currency: str,
    timeframe: Timeframe,
    load: FrameLoader,
    max_staleness: timedelta = DEFAULT_MAX_STALENESS,
) -> FxConverter:
    """The rate source for a run over ``instruments``.

    Args:
        instruments: Specifications of everything the run may trade. Pass only
            what the run actually trades: an unused instrument quoted in a third
            currency would otherwise demand a series nobody reads.
        account_currency: Currency the account is denominated in.
        timeframe: Bar size to load rate series at.
        load: Reads one symbol's bars; an empty frame means "not carried".
        max_staleness: How old a rate may be before it stops counting.

    Returns:
        :class:`SameCurrencyConverter` when no instrument needs conversion —
        see the module docstring on why that is not an empty
        :class:`BarFxConverter` — and a :class:`BarFxConverter` otherwise.

    Raises:
        ConversionSeriesMissingError: If a needed series is not carried.
    """
    currencies = required_conversion_currencies(instruments, account_currency=account_currency)
    if not currencies:
        return SameCurrencyConverter()
    series = load_conversion_series(
        currencies, account_currency=account_currency, timeframe=timeframe, load=load
    )
    return BarFxConverter(series, max_staleness=max_staleness)
