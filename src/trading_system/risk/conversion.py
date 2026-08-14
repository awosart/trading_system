"""Where an FX rate comes from, and what happens when there isn't one.

Position size is money divided by money. The numerator is a fraction of account
equity; the denominator is what one point of the instrument is worth — and that
figure is born in the instrument's *quote* currency
(:attr:`~trading_system.core.instruments.InstrumentSpec.value_per_point_quote`).
When the two currencies differ, converting between them takes a rate, and a rate
is the only input to sizing that is neither configuration nor arithmetic: it has
to be observed.

**A missing rate is never silently 1.0.** For GBPJPY on a USD account, assuming
parity oversizes the position by roughly the whole USDJPY rate — two orders of
magnitude — and produces a number that looks entirely plausible on the way out.
So :meth:`FxConverter.rate` raises :class:`FxRateUnavailableError` rather than
guessing, and the Risk Engine turns that into a counted refusal rather than a
crashed run: a gap in one conversion series is a data condition affecting some
signals, not a broken configuration affecting all of them, and EURUSD in the same
run still sizes correctly.

**Staleness is measured from the rate bar's close, not from when the data was
loaded.** The question a staleness bound answers is "how old was this quote at
the moment we needed it", and only bar time can answer it — wall-clock at load
time would make every rate in a backtest of 2019 infinitely stale. The default
bound has to clear an FX weekend to be usable at all: the last Friday bar closes
around 21:00 UTC and the market does not reopen until Sunday evening, so a rate
requested at Monday 00:00 is already 51 hours old through no fault of the data.
A bound under about two and a half days therefore rejects every Monday, which is
why :data:`DEFAULT_MAX_STALENESS` is three days — long enough for a weekend plus
a public holiday, short enough that a series which genuinely stopped is caught
within the week.
"""

from collections.abc import Iterable, Mapping
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Protocol, runtime_checkable

from trading_system.core.exceptions import DataError
from trading_system.core.instruments import InstrumentSpec
from trading_system.core.types import ensure_utc
from trading_system.data.models import OHLCVFrame

#: How old a conversion rate may be before it stops counting as a rate.
#: Three days rather than one: see the module docstring on FX weekends.
DEFAULT_MAX_STALENESS = timedelta(days=3)


def conversion_pair_candidates(quote_currency: str, account_currency: str) -> tuple[str, ...]:
    """The pair symbols either of which prices ``quote_currency`` in ``account_currency``.

    Two candidates rather than one because the market quotes each pair in one
    direction only and which direction that is, is a convention rather than a
    rule: JPY against USD is published as ``USDJPY``, GBP against USD as
    ``GBPUSD``. :class:`BarFxConverter` already inverts whichever it is handed,
    so the caller's job is only to load the one that exists.

    Args:
        quote_currency: Currency the instrument's point value is born in.
        account_currency: Currency the account is denominated in.

    Returns:
        Both spellings, direct first; empty when no conversion is needed.
    """
    if quote_currency == account_currency:
        return ()
    return (f"{quote_currency}{account_currency}", f"{account_currency}{quote_currency}")


def required_conversion_currencies(
    instruments: Iterable[InstrumentSpec], *, account_currency: str
) -> tuple[str, ...]:
    """Every quote currency in ``instruments`` that is not the account's own.

    The question this answers is which rate series a run has to carry, and it is
    asked of the **quote** currency alone. An instrument's base currency does not
    enter position sizing at all — one point is worth ``point_size ×
    contract_size`` in the quote currency and nothing else — so a run trading
    GBPJPY on a USD account needs JPY/USD and never GBP/USD.

    Args:
        instruments: Specifications of everything the run may trade.
        account_currency: Currency the account is denominated in.

    Returns:
        The distinct currencies needing conversion, sorted, so that a caller
        building a converter produces the same set every time.
    """
    return tuple(
        sorted(
            {spec.quote_currency for spec in instruments if spec.quote_currency != account_currency}
        )
    )


class FxRateUnavailableError(DataError):
    """No usable rate exists for a currency pair at the requested instant.

    Raised for a pair that is not carried at all, for an instant before the
    series begins, and for a rate older than the converter's staleness bound.
    All three are the same thing to a caller: the conversion cannot be done, and
    the alternative to refusing is inventing a position size.
    """


@runtime_checkable
class FxConverter(Protocol):
    """Something that can price one currency in another at a point in time."""

    def rate(self, *, base: str, quote: str, at: datetime) -> Decimal:
        """How many units of ``quote`` one unit of ``base`` was worth at ``at``.

        ``rate(base="USD", quote="JPY")`` is the USDJPY quote: convert an amount
        in USD to JPY by multiplying, and an amount in JPY to USD by dividing —
        or equivalently by asking for ``rate(base="JPY", quote="USD")``, which
        every implementation here answers by inverting.

        Args:
            base: Currency being priced.
            quote: Currency it is priced in.
            at: Instant the rate is needed for, tz-aware. Implementations must
                not use information from after this instant.

        Returns:
            The rate, strictly positive.

        Raises:
            FxRateUnavailableError: If no usable rate exists.
        """
        ...


def convert(
    amount: Decimal,
    *,
    from_currency: str,
    to_currency: str,
    at: datetime,
    converter: FxConverter,
) -> Decimal:
    """Express ``amount`` in another currency.

    Args:
        amount: Sum in ``from_currency``.
        from_currency: Currency ``amount`` is denominated in.
        to_currency: Currency to express it in.
        at: Instant to price the conversion at, tz-aware.
        converter: Rate source.

    Returns:
        The equivalent sum in ``to_currency``. Returned unchanged, without
        consulting the converter, when the two currencies are the same.

    Raises:
        FxRateUnavailableError: If the converter has no usable rate.
    """
    if from_currency == to_currency:
        return amount
    return amount * converter.rate(base=from_currency, quote=to_currency, at=at)


class SameCurrencyConverter:
    """A converter that only handles the trivial case, and says so otherwise.

    For runs whose whole universe is quoted in the account currency. Choosing it
    is a statement that no conversion is expected; if one turns out to be needed,
    that is a fact worth learning as a refusal rather than papering over. It is
    never a default anywhere — a converter is always passed in explicitly.
    """

    def rate(self, *, base: str, quote: str, at: datetime) -> Decimal:
        """Return one for a currency against itself, and refuse everything else.

        Args:
            base: Currency being priced.
            quote: Currency it is priced in.
            at: Ignored; a currency's value against itself does not move.

        Returns:
            ``Decimal(1)`` when the currencies match.

        Raises:
            FxRateUnavailableError: For any genuine pair.
        """
        del at
        if base == quote:
            return Decimal(1)
        raise FxRateUnavailableError(
            f"{base}/{quote}: this converter handles no conversions; the run was configured "
            "as if every instrument were quoted in the account currency"
        )


class StaticFxConverter:
    """A fixed rate table, constant over time.

    For tests and for one-off calculations. Not for a backtest of any length: a
    single USDJPY rate applied across two years prices every trade in the run at
    one arbitrary instant's exchange rate.
    """

    __slots__ = ("_rates",)

    def __init__(self, rates: Mapping[tuple[str, str], Decimal]) -> None:
        """Build a converter over an explicit rate table.

        Args:
            rates: ``(base, quote)`` to rate. Only one direction of each pair
                need be supplied; the other is derived by inversion.

        Raises:
            ValueError: If any rate is not strictly positive. A zero or negative
                rate is not a rate, and inverting one divides by zero.
        """
        for (base, quote), value in rates.items():
            if value <= 0:
                raise ValueError(f"{base}/{quote}: rate must be positive, got {value}")
        self._rates = dict(rates)

    def __repr__(self) -> str:
        """Compact description naming the pairs carried."""
        pairs = sorted(f"{base}{quote}" for base, quote in self._rates)
        return f"StaticFxConverter({pairs})"

    def rate(self, *, base: str, quote: str, at: datetime) -> Decimal:
        """Look the pair up directly, then inverted.

        Args:
            base: Currency being priced.
            quote: Currency it is priced in.
            at: Ignored; this converter is constant over time.

        Returns:
            The rate.

        Raises:
            FxRateUnavailableError: If neither direction of the pair is in the table.
        """
        del at
        if base == quote:
            return Decimal(1)
        direct = self._rates.get((base, quote))
        if direct is not None:
            return direct
        inverse = self._rates.get((quote, base))
        if inverse is not None:
            return Decimal(1) / inverse
        raise FxRateUnavailableError(
            f"{base}/{quote}: no rate in the static table "
            f"({sorted(f'{b}{q}' for b, q in self._rates)})"
        )


class BarFxConverter:
    """Rates read from OHLCV series of the conversion pairs themselves.

    The realistic backtest source. Two properties matter and both are structural
    rather than documented:

    * **No lookahead.** The rate used at ``at`` is the close of the last bar to
      have *closed* at or before ``at``. A bar's timestamp is its open time, so
      the cutoff is ``at - timeframe``; the bar containing ``at`` is still in
      progress and its close is not yet knowable.
    * **Staleness is bar time.** The gap measured is ``at`` minus the rate bar's
      close, so a backtest of historical data is not uniformly stale and a
      conversion series that stops mid-run does not go unnoticed.

    Wiring this to a :class:`~trading_system.data.store.ParquetStore` belongs to
    the backtest orchestrator, which knows what date range the run needs. This
    class takes the loaded frames.
    """

    __slots__ = ("_max_staleness", "_series")

    def __init__(
        self,
        series: Mapping[str, OHLCVFrame],
        *,
        max_staleness: timedelta = DEFAULT_MAX_STALENESS,
    ) -> None:
        """Build a converter over loaded conversion series.

        Args:
            series: Pair symbol (``"USDJPY"``) to its bars. Only one direction of
                each pair need be supplied; the other is derived by inversion.
            max_staleness: How far ``at`` may exceed a rate bar's close before
                the rate stops counting. Must clear an FX weekend — see the
                module docstring.

        Raises:
            ValueError: If ``max_staleness`` is not positive.
        """
        if max_staleness <= timedelta(0):
            raise ValueError(f"max_staleness must be positive, got {max_staleness}")
        self._series = dict(series)
        self._max_staleness = max_staleness

    def __repr__(self) -> str:
        """Compact description naming the pairs carried."""
        return f"BarFxConverter({sorted(self._series)}, max_staleness={self._max_staleness})"

    def rate(self, *, base: str, quote: str, at: datetime) -> Decimal:
        """Read the rate off the last conversion bar to close at or before ``at``.

        Args:
            base: Currency being priced.
            quote: Currency it is priced in.
            at: Instant the rate is needed for, tz-aware.

        Returns:
            The rate, inverted if only the opposite pair is carried.

        Raises:
            FxRateUnavailableError: If neither direction is carried, no bar of it has
                closed by ``at``, or the newest such bar is too old.
        """
        if base == quote:
            return Decimal(1)
        moment = ensure_utc(at)

        direct = self._series.get(f"{base}{quote}")
        if direct is not None:
            return self._close_as_of(direct, moment)

        inverse = self._series.get(f"{quote}{base}")
        if inverse is not None:
            return Decimal(1) / self._close_as_of(inverse, moment)

        raise FxRateUnavailableError(
            f"{base}/{quote}: neither {base}{quote} nor {quote}{base} is loaded "
            f"(have {sorted(self._series)})"
        )

    def _close_as_of(self, frame: OHLCVFrame, at: datetime) -> Decimal:
        """Close of the last bar in ``frame`` to have closed at or before ``at``.

        Args:
            frame: Conversion pair bars.
            at: Instant the rate is needed for, UTC.

        Returns:
            The close, as ``Decimal``.

        Raises:
            FxRateUnavailableError: If no bar has closed by ``at``, or the newest one
                that has is older than the staleness bound.
        """
        duration = frame.timeframe.duration
        # A bar's timestamp is its OPEN, so the newest bar that has closed by
        # `at` is the newest one that OPENED by `at - duration`.
        cutoff = at - duration
        timestamps = frame.timestamps
        position = int(timestamps.search_sorted(cutoff, side="right")) - 1
        if position < 0:
            first = frame.start
            raise FxRateUnavailableError(
                f"{frame.symbol}: no bar had closed by {at.isoformat()}"
                + (f"; the series starts at {first.isoformat()}" if first is not None else "")
            )

        bar_close_ts = timestamps[position] + duration
        age = at - bar_close_ts
        if age > self._max_staleness:
            raise FxRateUnavailableError(
                f"{frame.symbol}: newest rate closed {bar_close_ts.isoformat()}, "
                f"{age} before {at.isoformat()} — older than the {self._max_staleness} "
                "staleness bound, so the series has a gap rather than a quote"
            )
        return Decimal(str(frame.df["close"][position]))
