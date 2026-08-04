"""The Entry Engine's output: a scored, unsized directional idea.

An :class:`EntrySignal` carries no position size, no monetary risk and no account
state. That is the whole point of the split — the Risk Engine decides how much is
at stake, from ``quality`` and the distance between :attr:`EntrySignal.reference_price`
and :attr:`EntrySignal.invalidation_price`, and the strategy never gets a vote.

**Both prices are absolute levels in the instrument's own price units** —
``1.0847``, not ``15`` points and not ``0.0015`` of distance. A distance would be
half a number: its meaning depends on the instrument's pip size, which the Entry
Engine does not know and has no business knowing, and it needs a sign convention
on top. Four things make the mistake hard to commit rather than merely
documented:

* Both fields are :data:`~trading_system.core.types.Price`, a ``NewType`` over
  float, so ``mypy --strict`` refuses a bare float at the call site.
* They are named ``..._price``, never ``..._distance``.
* :meth:`EntrySignal.__post_init__` rejects an invalidation on the wrong side of
  the reference. A pip distance substituted for a level (``15.0`` against a
  reference of ``1.0847``) lands above the entry of a long and fails there.
* The distance is a derived :attr:`EntrySignal.risk_distance` property, so it
  exists in exactly one place and cannot disagree with the levels it came from.

A wrong-sided or zero-distance invalidation raises rather than being repaired.
The only repair available downstream is ``abs()``, which turns a defective setup
into a plausible position size computed from a meaningless distance — silently
wrong money, which is worse than no signal.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from types import MappingProxyType
from typing import Any

from trading_system.core.types import Price, Side, Signal, ensure_utc


@dataclass(frozen=True)
class EntrySignal:
    """One entry opportunity, recognised on a closed bar and scored but unsized.

    Attributes:
        strategy_id: Id of the :class:`~trading_system.strategies.schema.StrategySpec`
            that produced this.
        symbol: Instrument the signal applies to.
        bar_close_ts: Close time of bar ``t`` the signal was derived from, UTC.
            Execution may not happen before this instant — see
            :func:`~trading_system.core.types.can_execute_at`. Note this is a bar
            *close*, whereas :attr:`~trading_system.core.types.Bar.timestamp` is a
            bar *open*.
        side: Direction to trade.
        reference_price: The price the entry is anchored to, in instrument price
            units. For a market order this is the signal bar's close; for a limit
            or stop order it is that close displaced by the order's offset, so it
            is the price the order actually intends to fill at. This is the price
            risk is measured from.
        invalidation_price: The price at which the setup is disproven, in the
            same instrument price units — an absolute level, never a distance.
            Must sit below :attr:`reference_price` for a ``BUY`` and above it for
            a ``SELL``, strictly.
        quality: Confidence in ``[0, 1]``: the spec's ``base_quality`` plus the
            deltas of whichever ``quality_modifiers`` held on the signal bar,
            clamped. Risk Engine modifiers (portfolio heat, regime, correlation)
            are applied on top of this and are not reflected here.
        context: What actually fired, for trade forensics. Never read by any
            control-flow decision; excluded from equality for the same reason.
            Stored read-only so a consumer cannot edit the audit trail.
    """

    strategy_id: str
    symbol: str
    bar_close_ts: datetime
    side: Side
    reference_price: Price
    invalidation_price: Price
    quality: float
    context: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        """Normalise the bar close time and enforce every signal invariant.

        Raises:
            ValueError: If the timestamp is naive, a price is not finite, the
                quality is outside ``[0, 1]``, or the invalidation price is not
                strictly on the losing side of the reference price.
        """
        object.__setattr__(self, "bar_close_ts", ensure_utc(self.bar_close_ts))
        object.__setattr__(self, "context", MappingProxyType(dict(self.context)))

        if not 0.0 <= self.quality <= 1.0:
            raise ValueError(f"quality must be in [0.0, 1.0], got {self.quality}")
        for label, value in (
            ("reference_price", self.reference_price),
            ("invalidation_price", self.invalidation_price),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{label} must be finite, got {value!r}")

        expected = "below" if self.side is Side.BUY else "above"
        wrong_side = (
            self.invalidation_price >= self.reference_price
            if self.side is Side.BUY
            else self.invalidation_price <= self.reference_price
        )
        if wrong_side:
            raise ValueError(
                f"{self.strategy_id}/{self.symbol}: invalidation_price "
                f"{self.invalidation_price!r} must be strictly {expected} reference_price "
                f"{self.reference_price!r} for a {self.side.value}. Both are absolute price "
                "levels in instrument units, not distances or pips — a distance passed here "
                "lands on the wrong side of the entry and is rejected rather than silently "
                "turned into a position size."
            )

    @property
    def risk_distance(self) -> float:
        """Distance between entry and invalidation, in instrument price units.

        Derived rather than stored so it can never disagree with the two levels
        it comes from. Guaranteed positive by the side check in
        :meth:`__post_init__`, so the Risk Engine may divide by it unguarded.
        """
        return abs(self.reference_price - self.invalidation_price)

    def to_signal(self) -> Signal:
        """Project onto the core :class:`~trading_system.core.types.Signal`.

        The core type is the narrower contract every downstream module already
        agrees on; this projection exists so the two never drift into being two
        independent notions of "a signal". The entry-specific fields
        (``reference_price``, ``context``) are dropped.

        Returns:
            The equivalent core signal.
        """
        return Signal(
            strategy_id=self.strategy_id,
            symbol=self.symbol,
            bar_close_ts=self.bar_close_ts,
            direction=self.side,
            quality=self.quality,
            invalidation_price=self.invalidation_price,
        )
