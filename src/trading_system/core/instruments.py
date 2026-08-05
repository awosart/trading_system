"""What a tradable instrument actually is: ticks, lots, contract size, currencies.

This is the metadata every layer above the feature pipeline needs and none of
them may invent. The Entry Engine deliberately knows nothing about it — a signal
carries absolute price levels precisely because pip size is not its business —
and the Exit Engine speaks in fractions of a position because it has no idea
what a lot is. Both of those omissions are only defensible if the knowledge
exists somewhere concrete, and this is that place.

**``point_size`` is not ``tick_size``, and the difference is load-bearing.**

* ``tick_size`` is the smallest price increment the instrument can be quoted or
  filled at. It is what :meth:`InstrumentSpec.round_price` snaps to.
* ``point_size`` is the unit distances are *measured* in — what "a 50 point
  stop" means, and the unit of ``typical_spread_points`` and
  ``min_stop_distance_points``.

For EURUSD on a five-digit feed these differ by a factor of ten (``0.0001``
against ``0.00001``); for NAS100 they differ by ten as well (``1.0`` against
``0.1``). Collapsing them into one field forces a choice between two defects:
either "50 points" silently means half a pip, or ``round_price`` starts
truncating EURUSD at the fourth decimal and discards a digit the quote really
has. ``point_size`` is required to be a whole multiple of ``tick_size``, which
turns a typo in either into a load-time failure.

**Point value is one formula, not one per asset class.** The money a one-point
move is worth, per lot, in the instrument's *quote* currency is
``point_size * contract_size`` — see :attr:`InstrumentSpec.value_per_point_quote`.
EURUSD, XAUUSD, NAS100 and BTCUSD differ only in what those two numbers are.
The single genuine branch is currency: converting that figure into the account's
currency needs an FX rate whenever ``quote_currency`` is not the account's, and
sourcing that rate is :mod:`trading_system.risk.conversion`'s job, not this
module's.

``base_currency`` takes no part in point value at all. It is needed for margin
(``contract_size * price_of_base_in_account_currency * margin_rate``), which
belongs to the portfolio limits of stage 2.

**Commission states its basis, and there is no default.** ``commission_per_lot``
alone is half a number: brokers quote the same figure as "7 dollars per lot"
meaning either seven per side or seven for the whole round turn, and the two
readings differ by exactly a factor of two in the direction that flatters a
backtest. :attr:`InstrumentSpec.commission_basis` is therefore required, so
every registry file has to say which it means rather than inherit a guess.
Everything downstream reads :attr:`InstrumentSpec.commission_per_side`, which is
the normalised figure charged on each of the two fills.
"""

from collections.abc import Iterator, Mapping
from decimal import ROUND_CEILING, ROUND_DOWN, ROUND_FLOOR, ROUND_HALF_UP, Decimal
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any

import pydantic
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from trading_system.core.exceptions import ValidationError
from trading_system.core.types import Price

#: ISO-4217-shaped currency code. Not validated against a real code list: a
#: broker's account currency occasionally is not one (metal accounts), and a
#: wrong-but-well-formed code fails loudly at the first conversion anyway.
Currency = Annotated[str, Field(pattern=r"^[A-Z]{3,4}$")]


class InstrumentClass(StrEnum):
    """Broad category an instrument belongs to.

    Deliberately *not* a settlement wrapper. There is no ``INDEX_CFD`` member
    beside ``INDEX``: whether NAS100 reaches the account as a CFD, a future or a
    cash index changes ``contract_size``, ``margin_rate``, ``quote_currency`` and
    the swap fields, every one of which is already a field here. Adding the
    wrapper as a class would instead produce a strategy scoped to ``INDEX`` that
    silently never trades NAS100 — a defect that looks from outside like a quiet
    strategy, which is the same failure mode a mis-spelled session name once
    produced in :mod:`trading_system.entry.labels`.
    """

    FX = "FX"
    CRYPTO = "CRYPTO"
    EQUITY = "EQUITY"
    FUTURES = "FUTURES"
    INDEX = "INDEX"
    COMMODITY = "COMMODITY"


class CommissionBasis(StrEnum):
    """Whether a per-lot commission figure covers one fill or two.

    Deliberately has no "unspecified" member and no default anywhere. A broker's
    symbol table quoting "7.00 per lot" is genuinely ambiguous, and picking
    either reading silently doubles or halves every trade's cost — an error that
    survives any test which does not compare against a hand-computed figure,
    because both readings produce entirely plausible equity curves.
    """

    #: The figure is charged on each fill. A round turn costs twice it.
    PER_SIDE = "PER_SIDE"

    #: The figure covers the whole round turn. Each fill is charged half.
    ROUND_TURN = "ROUND_TURN"


def _decimal(value: float) -> Decimal:
    """Convert a float to ``Decimal`` through ``str``, never through the binary value.

    Args:
        value: Float to convert.

    Returns:
        The decimal spelling of ``value`` as written, not its binary neighbour.
    """
    return Decimal(str(value))


class InstrumentSpec(BaseModel):
    """The contract specification of one tradable instrument.

    Note this is not :class:`~trading_system.strategies.schema.InstrumentScope`,
    which says *which* instruments a strategy may be applied to. This says what
    one instrument *is*.

    Attributes:
        symbol: Broker's identifier, the key everything else joins on.
        asset_class: Broad category. See :class:`InstrumentClass` for why CFD is
            not one of the options.
        tick_size: Smallest quotable/fillable price increment.
        point_size: Unit that distances are quoted in — the "point" of
            ``typical_spread_points``, ``min_stop_distance_points`` and of any
            stop expressed in points. Must be a whole multiple of ``tick_size``.
        lot_step: Smallest increment of position size.
        min_lot: Smallest tradable size. A computed size below this is a refused
            trade, never a size rounded up.
        max_lot: Largest size accepted in one order.
        contract_size: Units of the underlying per lot — 100 000 for a standard
            FX lot, 100 ounces for gold, 1 for a typical index CFD.
        base_currency: The currency being bought. Used for margin, never for
            point value. Equal to ``quote_currency`` for an instrument whose
            notional is already denominated in the quote currency — an index CFD
            has no separate underlying currency, and saying so is more truthful
            than inventing a code for "the index".
        quote_currency: The currency the price is expressed in, and therefore
            the currency profit accrues in before conversion.
        margin_rate: Fraction of notional required as margin, in ``(0, 1]``.
        typical_spread_points: Representative spread, in points. A backtest
            default; a run with real spread data should use that instead.
        commission_per_lot: Commission per lot, in account currency. What it
            covers is stated by ``commission_basis`` and is not assumed; read
            :attr:`commission_per_side` rather than this field.
        commission_basis: Whether ``commission_per_lot`` is charged per fill or
            per round turn. Required, with no default — see
            :class:`CommissionBasis`.
        swap_long: Overnight financing per lot for a long, account currency.
            Signed — usually negative.
        swap_short: Overnight financing per lot for a short, account currency.
        min_stop_distance_points: Broker's minimum distance between the market
            and a stop order, in points. A stop closer than this is widened, and
            the size recomputed against the wider stop; it is never the stop
            that gives way.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str = Field(min_length=1)
    asset_class: InstrumentClass
    tick_size: float = Field(gt=0)
    point_size: float = Field(gt=0)
    lot_step: Decimal = Field(gt=0)
    min_lot: Decimal = Field(gt=0)
    max_lot: Decimal = Field(gt=0)
    contract_size: Decimal = Field(gt=0)
    base_currency: Currency
    quote_currency: Currency
    margin_rate: float = Field(gt=0, le=1)
    typical_spread_points: float = Field(ge=0)
    commission_per_lot: Decimal = Field(ge=0)
    commission_basis: CommissionBasis
    swap_long: Decimal
    swap_short: Decimal
    min_stop_distance_points: float = Field(ge=0)

    @model_validator(mode="after")
    def _check_consistency(self) -> "InstrumentSpec":
        """Reject internally contradictory metadata at load time.

        Returns:
            The validated spec.

        Raises:
            ValueError: If a point is not a whole number of ticks, the lot bounds
                contradict each other, or ``min_lot`` is not reachable in whole
                ``lot_step`` increments.
        """
        point, tick = _decimal(self.point_size), _decimal(self.tick_size)
        if point < tick or point % tick != 0:
            raise ValueError(
                f"{self.symbol}: point_size {self.point_size} must be a whole multiple of "
                f"tick_size {self.tick_size}; a point that is not a whole number of ticks "
                "means one of the two is a typo"
            )
        if self.max_lot < self.min_lot:
            raise ValueError(
                f"{self.symbol}: max_lot {self.max_lot} is below min_lot {self.min_lot}"
            )
        if self.min_lot % self.lot_step != 0:
            raise ValueError(
                f"{self.symbol}: min_lot {self.min_lot} is not a whole multiple of lot_step "
                f"{self.lot_step}, so the smallest tradable size is not tradable"
            )
        return self

    @property
    def ticks_per_point(self) -> int:
        """How many ticks make up one point. At least one, by validation."""
        return int(_decimal(self.point_size) / _decimal(self.tick_size))

    @property
    def value_per_point_quote(self) -> Decimal:
        """Money one point of price movement is worth per lot, in quote currency.

        The whole of the "point value" question, minus currency conversion:
        ``point_size * contract_size``. EURUSD gives 10 USD, GBPJPY 1 000 JPY,
        XAUUSD 1 USD, NAS100 1 USD — one formula, different inputs.

        ``Decimal`` because the result is money and is about to be multiplied by
        a position size. The float ``point_size`` crosses into decimal here, via
        ``str``, at this single boundary.
        """
        return _decimal(self.point_size) * self.contract_size

    @property
    def commission_per_side(self) -> Decimal:
        """Commission charged on **one** fill, per lot, in account currency.

        The single place the round-turn halving happens, and therefore the
        single place the factor-of-two error can live. Charging this on each of
        the two fills makes the round turn equal ``commission_per_lot`` under
        ``ROUND_TURN`` and twice it under ``PER_SIDE``, which is what each
        basis asserts.

        Kept at full decimal precision rather than quantised to cents: an odd
        round-turn figure halves to a fraction of a cent, and rounding it here
        would make the sum of two sides disagree with the stated round turn.
        Quantisation belongs at the reporting boundary.
        """
        if self.commission_basis is CommissionBasis.PER_SIDE:
            return self.commission_per_lot
        return self.commission_per_lot / 2

    @property
    def min_stop_distance_price(self) -> float:
        """The broker's minimum stop distance, expressed in price units."""
        return self.points_to_price(self.min_stop_distance_points)

    @property
    def typical_spread_price(self) -> float:
        """The representative spread, expressed in price units."""
        return self.points_to_price(self.typical_spread_points)

    def points_to_price(self, points: float) -> float:
        """Convert a distance in points into a distance in price units.

        Args:
            points: Distance in points. May be negative; the sign is preserved.

        Returns:
            The same distance in the instrument's price units.
        """
        return points * self.point_size

    def price_to_points(self, distance: float) -> float:
        """Convert a distance in price units into points.

        Args:
            distance: Distance in price units. May be negative.

        Returns:
            The same distance in points.
        """
        return distance / self.point_size

    def round_price(self, price: float) -> Price:
        """Snap a price to the nearest tick.

        Use for prices where neither direction is safer — a reference price, a
        reported fill. A protective level has a safe direction and should use
        :meth:`round_price_down` or :meth:`round_price_up` instead.

        Args:
            price: Price to snap.

        Returns:
            The nearest quotable price.
        """
        return self._quantize_price(price, ROUND_HALF_UP)

    def round_price_down(self, price: float) -> Price:
        """Snap a price down to the next tick at or below it.

        Args:
            price: Price to snap.

        Returns:
            The greatest quotable price not above ``price``.
        """
        return self._quantize_price(price, ROUND_FLOOR)

    def round_price_up(self, price: float) -> Price:
        """Snap a price up to the next tick at or above it.

        Args:
            price: Price to snap.

        Returns:
            The least quotable price not below ``price``.
        """
        return self._quantize_price(price, ROUND_CEILING)

    def shift_price(self, price: float, points: float) -> Price:
        """Move ``price`` by a signed number of points, never shortening the shift.

        The rounding direction follows the sign: a positive shift snaps up, a
        negative one snaps down, so the result is always at least as far from
        ``price`` as asked for. That is what a cost wants — snapping a fill back
        toward the trader hands back up to a tick per fill, in a direction that
        looks like noise and is not.

        **The arithmetic is decimal throughout**, and that is load-bearing rather
        than tidy. Adding half a spread to a price as floats gives
        ``1.1 + 0.00005 == 1.1000500000000001``; snapping that away from the
        entry rounds up a full tick, and a round turn that should cost exactly
        one spread comes back costing one spread plus two ticks. Both operands
        cross into decimal through ``str``, so a shift that lands on the tick
        grid stays on it.

        Args:
            price: Starting price.
            points: Signed distance to move, in the instrument's points.

        Returns:
            The shifted price, on the tick grid.
        """
        shifted = _decimal(price) + _decimal(points) * _decimal(self.point_size)
        if points > 0:
            rounding = ROUND_CEILING
        elif points < 0:
            rounding = ROUND_FLOOR
        else:
            rounding = ROUND_HALF_UP
        tick = _decimal(self.tick_size)
        ticks = (shifted / tick).to_integral_value(rounding=rounding)
        return Price(float(ticks * tick))

    def round_volume(self, volume: Decimal) -> Decimal:
        """Snap a size **down** to a whole number of lot steps.

        Always down, never to nearest. Rounding a size up increases the money at
        risk past the figure the sizing method computed, which is the one thing
        the Risk Engine's cap exists to prevent; rounding down only ever risks
        less than planned. A size that rounds below ``min_lot`` is a refused
        trade, decided by the caller — this method returns the honest zero
        rather than inventing a tradable size.

        Args:
            volume: Unquantised size in lots. Must not be negative.

        Returns:
            The largest whole multiple of ``lot_step`` not exceeding ``volume``.

        Raises:
            ValueError: If ``volume`` is negative.
        """
        if volume < 0:
            raise ValueError(f"{self.symbol}: volume must not be negative, got {volume}")
        steps = (volume / self.lot_step).to_integral_value(rounding=ROUND_DOWN)
        return steps * self.lot_step

    def _quantize_price(self, price: float, rounding: str) -> Price:
        """Snap ``price`` to the tick grid under an explicit rounding mode.

        Args:
            price: Price to snap.
            rounding: A :mod:`decimal` rounding mode.

        Returns:
            The snapped price.
        """
        tick = _decimal(self.tick_size)
        ticks = (_decimal(price) / tick).to_integral_value(rounding=rounding)
        return Price(float(ticks * tick))


class InstrumentRegistry:
    """The loaded instrument universe, keyed by symbol.

    Immutable once built. Held by whoever needs it and passed explicitly, like
    every other dependency in this system — there is no module-level default
    instance to reach for.
    """

    __slots__ = ("_by_symbol",)

    def __init__(self, instruments: Mapping[str, InstrumentSpec]) -> None:
        """Wrap an already-deduplicated mapping.

        Args:
            instruments: Symbol to spec. Use :func:`load_instruments` to build
                one from a file.
        """
        self._by_symbol = dict(instruments)

    def __repr__(self) -> str:
        """Compact description naming the size and the symbols."""
        return f"InstrumentRegistry({len(self._by_symbol)}: {sorted(self._by_symbol)})"

    def __len__(self) -> int:
        """How many instruments are registered."""
        return len(self._by_symbol)

    def __iter__(self) -> Iterator[str]:
        """Iterate over the registered symbols, in insertion order."""
        return iter(self._by_symbol)

    def __contains__(self, symbol: object) -> bool:
        """Whether ``symbol`` is registered."""
        return symbol in self._by_symbol

    def __getitem__(self, symbol: str) -> InstrumentSpec:
        """Look up one instrument.

        Args:
            symbol: Symbol to look up.

        Returns:
            Its specification.

        Raises:
            KeyError: If the symbol is not registered. Never a stand-in spec:
                guessing a contract size is guessing a position size.
        """
        try:
            return self._by_symbol[symbol]
        except KeyError:
            raise KeyError(
                f"unknown instrument {symbol!r}; registered: {sorted(self._by_symbol)}"
            ) from None

    def get(self, symbol: str) -> InstrumentSpec | None:
        """Look up one instrument, or ``None`` if it is not registered.

        Args:
            symbol: Symbol to look up.

        Returns:
            Its specification, or ``None``.
        """
        return self._by_symbol.get(symbol)

    @property
    def symbols(self) -> tuple[str, ...]:
        """Every registered symbol, in insertion order."""
        return tuple(self._by_symbol)


class _RegistryFile(BaseModel):
    """The on-disk shape of the instrument registry.

    A wrapper object rather than a bare list so the file can grow a version or a
    comment key later without every reader changing shape.
    """

    model_config = ConfigDict(extra="forbid")

    instruments: list[InstrumentSpec] = Field(min_length=1)


def load_instruments(path: Path) -> InstrumentRegistry:
    """Load and validate the instrument registry from a YAML file.

    Structure mirrors the Exit DB's: pydantic checks the syntax (known fields,
    right types, per-field bounds) and :meth:`InstrumentSpec._check_consistency`
    checks the semantics (a point is a whole number of ticks, the lot bounds
    agree). A file that fails either does not load; there is no partial registry.

    Args:
        path: Location of the YAML file.

    Returns:
        The loaded registry.

    Raises:
        ValidationError: If the file is missing, is not a mapping, fails
            validation, or declares the same symbol twice.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise ValidationError(f"instrument registry {path}: cannot be read: {error}") from error

    loaded: Any = yaml.safe_load(text)
    if not isinstance(loaded, dict):
        raise ValidationError(
            f"instrument registry {path}: expected a mapping with an 'instruments' key, "
            f"got {type(loaded).__name__}"
        )

    try:
        parsed = _RegistryFile.model_validate(loaded)
    except pydantic.ValidationError as error:
        raise ValidationError(f"instrument registry {path}: {error}") from error

    by_symbol: dict[str, InstrumentSpec] = {}
    for instrument in parsed.instruments:
        if instrument.symbol in by_symbol:
            raise ValidationError(
                f"instrument registry {path}: {instrument.symbol} is declared twice; "
                "a duplicate symbol makes which contract size applies a matter of file order"
            )
        by_symbol[instrument.symbol] = instrument
    return InstrumentRegistry(by_symbol)
