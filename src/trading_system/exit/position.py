"""The position an exit plan manages, and the ledger of how it was closed.

A :class:`ManagedPosition` is the only mutable object in the Exit Engine. It owns
three things no rule may own privately: the current stop level, how much of the
original position is still open, and the record of what has already been closed.
Centralising them is what makes the two invariants of this layer enforceable in
one place rather than in every rule:

* **The stop ratchets.** :meth:`ManagedPosition.tighten_stop` moves the level
  toward the entry and never away from it. A trailing rule that proposes a looser
  level on a pullback is behaving normally, not erroneously, so the request is
  declined and reported rather than raised on — but a level that would widen the
  risk the Risk Engine already sized against cannot be installed by any caller.
* **Fractions sum to at most one.** Every close is a fraction *of the original
  size*, and :meth:`ManagedPosition.close` refuses one that exceeds what remains.
  The refusal is atomic: nothing is booked, nothing is partially applied.

Fractions are :class:`~decimal.Decimal`, not float. The sum-to-100% invariant is
exact only in decimal arithmetic — a ladder of ``0.5 + 0.25 + 0.25`` written in
binary floating point leaves a residue, and "the position is flat" would become a
comparison against a tolerance. A fraction is also one multiplication away from
being a size in lots, and sizes are money's neighbours.

**This module does not import the Entry Engine.** A position is opened from an
:class:`~trading_system.entry.signal.EntrySignal` by the backtest layer, which
knows both; making the Exit Engine reach for the Entry Engine would defeat the
N×M combination the two libraries exist to provide.
"""

import math
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from trading_system.core.types import Price, Side, ensure_utc
from trading_system.exit.base import ExitReason

#: The whole position, as a fraction of itself.
WHOLE = Decimal("1")


@dataclass(frozen=True)
class ClosedLeg:
    """One realised close — partial or final — of a managed position.

    Attributes:
        fraction: Share of the ORIGINAL position size this leg closed, in
            ``(0, 1]``. Never a share of the remainder: a ladder written as
            "50% at 1R, 25% at 2R, 25% at 3R" sums to one only under that
            reading, and under the other it would leave a tail forever.
        price: Fill price in instrument price units.
        ts: When the fill happened, UTC.
        reason: Which class of rule closed it.
        r_multiple: This leg's result in units of the position's initial risk,
            at ``price``. Unweighted — multiply by ``fraction`` to contribute it
            to the position's total.
        pnl: This leg's profit in quote currency, ``size * fraction * move``.
    """

    fraction: Decimal
    price: Price
    ts: datetime
    reason: ExitReason
    r_multiple: float
    pnl: Decimal


class ManagedPosition:
    """An open position, its live stop, and the legs already closed out of it.

    Deliberately *not* a copy of :class:`~trading_system.core.types.Position`,
    which is a frozen snapshot for the portfolio to hold. This one is the
    stateful thing an exit plan advances bar by bar.

    There is no ``targets`` field. A target is a rule's output — ``FixedRR``
    computes it from the initial risk, a partial ladder from its rungs — and
    storing it here would create a second copy that drifts the moment a rule
    recomputes. What the position exposes instead is :meth:`r_multiple`, from
    which any rule derives its own levels, and :attr:`legs`, which records the
    targets that were actually reached.
    """

    __slots__ = (
        "_entry_price",
        "_initial_risk_distance",
        "_initial_stop",
        "_legs",
        "_opened_at",
        "_remaining",
        "_side",
        "_size",
        "_stop",
        "_strategy_id",
        "_symbol",
    )

    def __init__(
        self,
        *,
        symbol: str,
        side: Side,
        entry_price: Price,
        size: Decimal,
        initial_stop: Price,
        opened_at: datetime,
        strategy_id: str = "",
    ) -> None:
        """Open a position with a stop already in place.

        A position without a stop is not constructible: ``initial_stop`` has no
        default and no ``None``. This continues P06's decision to make
        ``invalidation`` mandatory — the level the Risk Engine sized against is
        the level the Exit Engine protects, and an unprotected position would be
        one whose size was computed from a distance nobody is enforcing.

        Args:
            symbol: Instrument held.
            side: Direction of the exposure.
            entry_price: Average fill price of the entry.
            size: Size in instrument units, positive.
            initial_stop: Protective level, strictly below ``entry_price`` for a
                ``BUY`` and strictly above it for a ``SELL``. This is normally
                the originating signal's ``invalidation_price``.
            opened_at: Fill time of the entry, tz-aware.
            strategy_id: Strategy the position came from, for forensics.

        Raises:
            ValueError: If the size is not positive, a price is not finite, the
                stop is not strictly on the losing side of the entry, or
                ``opened_at`` is naive.
        """
        if size <= 0:
            raise ValueError(f"position size must be positive, got {size}")
        for label, value in (("entry_price", entry_price), ("initial_stop", initial_stop)):
            if not math.isfinite(value):
                raise ValueError(f"{label} must be finite, got {value!r}")

        expected = "below" if side is Side.BUY else "above"
        wrong_side = (
            initial_stop >= entry_price if side is Side.BUY else initial_stop <= entry_price
        )
        if wrong_side:
            raise ValueError(
                f"{symbol}: initial_stop {initial_stop!r} must be strictly {expected} "
                f"entry_price {entry_price!r} for a {side.value}. Both are absolute price "
                "levels in instrument units, not distances."
            )

        self._symbol = symbol
        self._side = side
        self._entry_price = entry_price
        self._size = size
        self._initial_stop = initial_stop
        self._initial_risk_distance = abs(entry_price - initial_stop)
        self._opened_at = ensure_utc(opened_at)
        self._strategy_id = strategy_id
        self._stop = initial_stop
        self._remaining = WHOLE
        self._legs: list[ClosedLeg] = []

    def __repr__(self) -> str:
        """Compact description naming the symbol, side and what is left."""
        state = "flat" if not self.is_open else f"{self._remaining} open"
        return f"ManagedPosition({self._symbol} {self._side.value} @{self._entry_price} {state})"

    @property
    def symbol(self) -> str:
        """Instrument held."""
        return self._symbol

    @property
    def side(self) -> Side:
        """Direction of the exposure."""
        return self._side

    @property
    def exit_side(self) -> Side:
        """Direction an order closing this position would take."""
        return Side.SELL if self._side is Side.BUY else Side.BUY

    @property
    def strategy_id(self) -> str:
        """Strategy the position came from."""
        return self._strategy_id

    @property
    def entry_price(self) -> Price:
        """Average fill price of the entry."""
        return self._entry_price

    @property
    def size(self) -> Decimal:
        """Original size in instrument units. Never reduced by a partial close."""
        return self._size

    @property
    def opened_at(self) -> datetime:
        """Fill time of the entry, UTC."""
        return self._opened_at

    @property
    def initial_stop(self) -> Price:
        """Stop as it stood at entry. Frozen: this is R's denominator."""
        return self._initial_stop

    @property
    def stop(self) -> Price:
        """Current protective level, after any tightening."""
        return self._stop

    @property
    def initial_risk_distance(self) -> float:
        """Entry-to-initial-stop distance in price units — one R.

        Positive by the side check in the constructor, so callers may divide by
        it unguarded. Fixed at entry and never recomputed from the *current*
        stop: a breakeven move would otherwise drive it to zero and every
        R-based threshold in the plan to infinity.
        """
        return self._initial_risk_distance

    @property
    def remaining_fraction(self) -> Decimal:
        """Share of the original position still open, in ``[0, 1]``."""
        return self._remaining

    @property
    def remaining_size(self) -> Decimal:
        """Open size in instrument units, unquantised.

        Not necessarily a whole multiple of the instrument's lot step — rounding
        to something tradable needs ``min_lot``/``lot_step`` from the instrument
        metadata, which arrives with the Risk Engine and which this layer must
        not invent.
        """
        return self._size * self._remaining

    @property
    def is_open(self) -> bool:
        """Whether any of the position is still open."""
        return self._remaining > 0

    @property
    def legs(self) -> tuple[ClosedLeg, ...]:
        """Every realised close, oldest first."""
        return tuple(self._legs)

    def r_multiple(self, price: Price) -> float:
        """Result at ``price`` in units of the position's INITIAL risk.

        Measured from :attr:`entry_price` against :attr:`initial_risk_distance`
        — the entry-to-stop distance *as it stood at entry*. Neither term moves
        when the position is partially closed or when the stop is tightened, and
        that is the whole point:

        * **Size-independent**, so closing 50% at 1R does not shift the 2R and 3R
          rungs of the same ladder. Measuring against the remainder would
          re-scale every threshold in the plan each time a rung fired.
        * **Stop-independent**, so a breakeven move does not collapse the
          denominator to zero.

        This is the quantity rules trigger on. For the money-weighted figure
        analytics aggregates, use :meth:`realized_r` — 50% closed at 1R plus 50%
        at 3R is 2.0R, not 4.0R.

        Args:
            price: Price to evaluate at.

        Returns:
            Signed R multiple: positive in the position's favour.
        """
        direction = 1.0 if self._side is Side.BUY else -1.0
        return direction * (price - self._entry_price) / self._initial_risk_distance

    def realized_r(self) -> float:
        """R booked so far, each leg weighted by its share of the original size.

        Returns:
            Sum of ``fraction * r_multiple`` over the closed legs; zero while
            nothing has been closed.
        """
        return sum(float(leg.fraction) * leg.r_multiple for leg in self._legs)

    def open_r(self, price: Price) -> float:
        """R the still-open remainder is worth at ``price``, weighted by size.

        Args:
            price: Price to evaluate at.

        Returns:
            ``remaining_fraction * r_multiple(price)``.
        """
        return float(self._remaining) * self.r_multiple(price)

    def total_r(self, price: Price) -> float:
        """Booked plus open R at ``price``, the position's result if closed now.

        Args:
            price: Price to evaluate at.

        Returns:
            ``realized_r() + open_r(price)``.
        """
        return self.realized_r() + self.open_r(price)

    @property
    def realized_pnl(self) -> Decimal:
        """Profit booked so far, in quote currency, as ``Decimal``.

        Money, so decimal — the float price move is converted at this one
        boundary via ``str`` and never mixed back into a float total. The figure
        is ``size * fraction * move`` and therefore assumes size is already in
        instrument units; a contract-size multiplier, where an instrument has
        one, belongs to the Risk Engine's instrument metadata and is applied
        above this layer.
        """
        return sum((leg.pnl for leg in self._legs), Decimal(0))

    def tighten_stop(self, level: Price) -> bool:
        """Move the stop toward the entry, never away from it.

        Declining a looser level is the defined behaviour of a ratchet, not an
        error: a trailing rule recomputing its level on a pullback proposes a
        wider stop as a matter of course, and making that raise would push the
        same guard into every trailing rule. Widening is refused because the
        Risk Engine sized this position against the distance to the stop, and a
        stop that retreats silently increases the money at stake after the fact.

        Args:
            level: Proposed stop level, absolute price.

        Returns:
            Whether the stop actually moved.

        Raises:
            ValueError: If ``level`` is not finite, or the position is flat.
        """
        if not math.isfinite(level):
            raise ValueError(f"stop level must be finite, got {level!r}")
        if not self.is_open:
            raise ValueError(f"{self._symbol}: cannot move the stop of a closed position")
        tighter = level > self._stop if self._side is Side.BUY else level < self._stop
        if not tighter:
            return False
        self._stop = level
        return True

    def close(
        self, fraction: Decimal, *, price: Price, ts: datetime, reason: ExitReason
    ) -> ClosedLeg:
        """Book a close of ``fraction`` of the ORIGINAL position size.

        Args:
            fraction: Share of the original size, in ``(0, remaining_fraction]``.
            price: Fill price.
            ts: Fill time, tz-aware.
            reason: Which class of rule closed it.

        Returns:
            The booked leg.

        Raises:
            ValueError: If the position is already flat, ``fraction`` is not a
                positive ``Decimal``, ``price`` is not finite, or ``fraction``
                exceeds what remains. The refusal is atomic — a rejected close
                leaves the ledger and the remainder exactly as they were.
        """
        if not self.is_open:
            raise ValueError(f"{self._symbol}: position is already flat")
        if not isinstance(fraction, Decimal):
            raise ValueError(
                f"fraction must be a Decimal, got {type(fraction).__name__}; float fractions "
                "cannot sum to exactly one"
            )
        if fraction <= 0:
            raise ValueError(f"fraction must be positive, got {fraction}")
        if not math.isfinite(price):
            raise ValueError(f"fill price must be finite, got {price!r}")
        if fraction > self._remaining:
            raise ValueError(
                f"{self._symbol}: cannot close {fraction} of the original position, "
                f"{self._remaining} remains"
            )

        direction = 1 if self._side is Side.BUY else -1
        move = Decimal(str(price)) - Decimal(str(self._entry_price))
        leg = ClosedLeg(
            fraction=fraction,
            price=price,
            ts=ensure_utc(ts),
            reason=reason,
            r_multiple=self.r_multiple(price),
            pnl=self._size * fraction * move * direction,
        )
        self._remaining -= fraction
        self._legs.append(leg)
        return leg

    def close_all(self, *, price: Price, ts: datetime, reason: ExitReason) -> ClosedLeg:
        """Book a close of whatever remains.

        Args:
            price: Fill price.
            ts: Fill time, tz-aware.
            reason: Which class of rule closed it.

        Returns:
            The booked leg.

        Raises:
            ValueError: If the position is already flat or the price is not
                finite.
        """
        if not self.is_open:
            raise ValueError(f"{self._symbol}: position is already flat")
        return self.close(self._remaining, price=price, ts=ts, reason=reason)
