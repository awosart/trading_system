"""Where the protective stop goes — and therefore what the position size is.

One rule governs this module, and it is the reason sizing works at all:

    **The stop is placed by structure. The size adapts to the stop. The stop is
    never moved closer in order to justify a larger size.**

That is not enforced by a comment. Every candidate level this module computes is
a *distance from the reference price*, and the answer is the **furthest** of
them. There is no branch that selects a nearer level, so no future edit can
tighten a stop by accident: to break the rule someone would have to replace
:func:`max` with :func:`min` and delete a test named after the invariant.

Three candidates compete, and each answers a different question:

* **The invalidation level, plus a buffer.** Where the setup is disproven, per
  P06 — pushed out by the spread and by whatever noise allowance is configured,
  because a stop sitting exactly on the level everyone can see gets taken out by
  the spread alone.
* **What ``stop_reference`` names.** The strategy's volatility view: two ATRs, a
  fixed distance, a percentage. If that is wider than the structural level, the
  structure is too close for current volatility, and the wider level wins.
* **The broker's minimum stop distance.** Not negotiable and not a policy — an
  order closer than this is rejected at the venue.

Taking the furthest of the three means each can only ever widen the stop, and
widening a stop shrinks the size for a fixed risk budget. The failure mode this
forecloses — quietly tightening a stop so a rejected trade becomes tradable — is
precisely how a risk limit gets respected on paper while the account takes a
larger loss than the number said.

Rounding follows the same logic: a stop is snapped **away** from the entry, never
to the nearest tick. Snapping toward the entry would shrink the very distance the
size was just computed from, by up to a tick, after the fact.
"""

from dataclasses import dataclass
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field

from trading_system.core.instruments import InstrumentSpec
from trading_system.core.types import Price, Side
from trading_system.strategies.schema import (
    AtrStop,
    FixedPipsStop,
    PercentStop,
    StopReference,
    StructureStop,
)


class StopBufferConfig(BaseModel):
    """How far beyond the invalidation level the stop is pushed.

    A pydantic model rather than three loose arguments, so a bad buffer fails
    when the config loads instead of in the middle of a run.

    Attributes:
        spread_multiple: Multiples of the instrument's typical spread. One by
            default: a stop resting exactly on a visible level is hit by the
            spread widening alone, without the price ever trading there.
        fixed_points: Flat noise allowance in points, on top of the spread.
        atr_multiple: Multiples of ATR, for a volatility-scaled allowance. Zero
            by default, because a non-zero value makes ATR a required input and
            a run without it a refusal rather than a silent zero.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    spread_multiple: float = Field(default=1.0, ge=0)
    fixed_points: float = Field(default=0.0, ge=0)
    atr_multiple: float = Field(default=0.0, ge=0)


@dataclass(frozen=True)
class StopCalculation:
    """Where the stop went and what set it there.

    Attributes:
        stop_price: Absolute level, snapped to the tick grid away from the entry.
        distance_price: Distance from the reference price, in price units.
            Strictly positive.
        distance_points: The same distance in the instrument's points, which is
            the unit point value is quoted per. Float, for the sizing methods,
            which work in price space.
        distance_points_exact: The same figure again, in exact decimal — and the
            one the money arithmetic uses. Both endpoints are exact decimal
            quantities (``stop_price`` sits on the tick grid by construction),
            so subtracting them in decimal is exact, whereas subtracting them as
            floats is not: ``1.0850 - 1.0800`` is ``0.004999999999999893``, which
            turns a risk of exactly 500 into 499.99999999998934. The size is
            unaffected — the error is a thousand times smaller than a lot step —
            but a risk figure that reports the cap as very slightly under it is
            an invitation to wonder why for the life of the project.
        binding: Which candidate turned out to be furthest — the thing that
            actually decided the size.
        reasons: Human-readable lines describing what was computed.
    """

    stop_price: Price
    distance_price: float
    distance_points: float
    distance_points_exact: Decimal
    binding: str
    reasons: tuple[str, ...]


#: Names reported in :attr:`StopCalculation.binding`.
INVALIDATION_BUFFERED = "invalidation_plus_buffer"
STOP_REFERENCE = "stop_reference"
BROKER_MINIMUM = "broker_minimum_distance"


def requires_atr(stop_reference: StopReference, buffer: StopBufferConfig) -> bool:
    """Whether computing this stop needs an ATR value.

    Checked by the caller *before* sizing, so a missing ATR becomes a counted
    refusal naming the real cause rather than a zero buffer that silently
    produces a tighter stop and a larger position.

    Args:
        stop_reference: How the strategy derives its stop distance.
        buffer: Buffer configuration.

    Returns:
        Whether an ATR value must be supplied.
    """
    if buffer.atr_multiple > 0:
        return True
    if isinstance(stop_reference, AtrStop):
        return True
    return isinstance(stop_reference, StructureStop) and stop_reference.buffer_atr_multiple > 0


def calculate_stop(
    *,
    side: Side,
    reference_price: Price,
    invalidation_price: Price,
    instrument: InstrumentSpec,
    stop_reference: StopReference,
    buffer: StopBufferConfig,
    atr_price: float | None = None,
) -> StopCalculation:
    """Place the protective stop at the furthest of the competing candidates.

    Args:
        side: Direction of the intended position.
        reference_price: Price the entry is anchored to — what distance is
            measured from. Not the bar close unless the order is a market order;
            see :class:`~trading_system.entry.signal.EntrySignal`.
        invalidation_price: Level at which the setup is disproven, strictly on
            the losing side of ``reference_price``.
        instrument: Contract specification, for spread, points and the tick grid.
        stop_reference: How the strategy derives its own stop distance.
        buffer: How far past the invalidation level to push the stop.
        atr_price: ATR in price units on the signal bar. Required when
            :func:`requires_atr` says so.

    Returns:
        The placed stop and what decided it.

    Raises:
        ValueError: If ``invalidation_price`` is not strictly on the losing side
            of ``reference_price``, or if an ATR value is needed and absent. The
            latter is a caller error — :func:`requires_atr` answers it in advance
            — so it raises rather than defaulting the buffer to zero, which would
            silently tighten the stop and enlarge the position.
    """
    _check_sides(side, reference_price, invalidation_price)
    if requires_atr(stop_reference, buffer) and atr_price is None:
        raise ValueError(
            "this stop configuration needs an ATR value and none was supplied; call "
            "requires_atr() first and refuse the signal rather than treating the missing "
            "ATR as a zero buffer, which would tighten the stop and enlarge the position"
        )
    atr = atr_price if atr_price is not None else 0.0

    structural = abs(reference_price - invalidation_price) + _buffer_distance(
        instrument, buffer, atr
    )
    referenced = _stop_reference_distance(
        reference_price=reference_price,
        invalidation_price=invalidation_price,
        instrument=instrument,
        stop_reference=stop_reference,
        atr=atr,
    )
    broker_minimum = instrument.min_stop_distance_price

    candidates = {
        INVALIDATION_BUFFERED: structural,
        STOP_REFERENCE: referenced,
        BROKER_MINIMUM: broker_minimum,
    }
    # Furthest wins, always. Never the nearest: see the module docstring.
    binding = max(candidates, key=lambda name: candidates[name])
    distance = candidates[binding]

    # Snap away from the entry so quantisation can only widen, never tighten.
    raw_stop = reference_price - distance if side is Side.BUY else reference_price + distance
    stop_price = (
        instrument.round_price_down(raw_stop)
        if side is Side.BUY
        else instrument.round_price_up(raw_stop)
    )
    final_distance = abs(reference_price - stop_price)

    reasons = [
        f"stop from {binding} at {stop_price}, "
        f"{instrument.price_to_points(final_distance):.4g} points from {reference_price}"
    ]
    if binding == BROKER_MINIMUM:
        reasons.append(
            f"widened to the broker minimum of {instrument.min_stop_distance_points} points; "
            "the size was recomputed against the wider stop, the stop was not brought in"
        )
    exact_distance = (Decimal(str(reference_price)) - Decimal(str(stop_price))).copy_abs()
    return StopCalculation(
        stop_price=stop_price,
        distance_price=final_distance,
        distance_points=instrument.price_to_points(final_distance),
        distance_points_exact=exact_distance / Decimal(str(instrument.point_size)),
        binding=binding,
        reasons=tuple(reasons),
    )


def _check_sides(side: Side, reference_price: Price, invalidation_price: Price) -> None:
    """Reject an invalidation level that is not on the losing side of the entry.

    Args:
        side: Direction of the intended position.
        reference_price: Entry anchor.
        invalidation_price: Level the setup is disproven at.

    Raises:
        ValueError: If the invalidation is not strictly on the losing side.
    """
    wrong_side = (
        invalidation_price >= reference_price
        if side is Side.BUY
        else invalidation_price <= reference_price
    )
    if wrong_side:
        expected = "below" if side is Side.BUY else "above"
        raise ValueError(
            f"invalidation_price {invalidation_price} must be strictly {expected} "
            f"reference_price {reference_price} for a {side.value}"
        )


def _buffer_distance(instrument: InstrumentSpec, buffer: StopBufferConfig, atr: float) -> float:
    """How far past the invalidation level the stop is pushed, in price units.

    Args:
        instrument: Contract specification, for the typical spread.
        buffer: Buffer configuration.
        atr: ATR in price units, or zero when the config does not use it.

    Returns:
        The buffer distance, non-negative.
    """
    points = buffer.spread_multiple * instrument.typical_spread_points + buffer.fixed_points
    return instrument.points_to_price(points) + buffer.atr_multiple * atr


def _stop_reference_distance(
    *,
    reference_price: Price,
    invalidation_price: Price,
    instrument: InstrumentSpec,
    stop_reference: StopReference,
    atr: float,
) -> float:
    """The distance the strategy's ``stop_reference`` asks for, in price units.

    Each variant answers "how far from the entry should the stop be" in its own
    terms. ``STRUCTURE`` is the exception: the structural level is already what
    the entry's invalidation *is*, so its only contribution is its ATR buffer —
    re-deriving structure here would compute a second, competing swing level from
    data the Entry Engine has already read.

    Args:
        reference_price: Entry anchor.
        invalidation_price: Level the setup is disproven at.
        instrument: Contract specification, for point size.
        stop_reference: The strategy's stop derivation.
        atr: ATR in price units, or zero when unused.

    Returns:
        The requested distance, non-negative.
    """
    match stop_reference:
        case AtrStop():
            return stop_reference.multiple * atr
        case FixedPipsStop():
            # The schema calls them pips; this layer's unit is points, and
            # `point_size` is defined as one pip for every FX symbol in the
            # registry, so the two coincide by construction.
            return instrument.points_to_price(stop_reference.pips)
        case PercentStop():
            return abs(reference_price) * stop_reference.percent / 100.0
        case StructureStop():
            return (
                abs(reference_price - invalidation_price) + stop_reference.buffer_atr_multiple * atr
            )
