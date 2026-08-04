"""The four sizing methods.

``FractionalKelly`` is deliberately absent until P14. Kelly needs a win rate and
a payoff ratio measured over a real sample of closed trades; before the analytics
layer exists there is nothing to measure, and a Kelly fraction computed from
assumed inputs is a leverage multiplier wearing a formula's authority.
"""

from decimal import Decimal

from trading_system.risk.models import RiskReason
from trading_system.risk.sizing.base import SizingOutcome, SizingRequest


class FixedFractional:
    """Risk a constant fraction of equity on every trade.

    The default, and the baseline every other method should be compared against.
    Its one property worth stating: because the fraction is of *equity* and not
    of balance, a losing streak shrinks the next position before any of the
    losers is closed.
    """

    __slots__ = ("_risk_pct",)

    def __init__(self, risk_pct: float) -> None:
        """Configure the fraction of equity to risk.

        Args:
            risk_pct: Fraction of equity, not a percentage — ``0.005`` is half a
                percent. Must be in ``(0, 1]``.

        Raises:
            ValueError: If the fraction is outside ``(0, 1]``.
        """
        if not 0 < risk_pct <= 1:
            raise ValueError(
                f"risk_pct is a fraction of equity in (0, 1], got {risk_pct}; "
                "0.005 means half a percent"
            )
        self._risk_pct = risk_pct

    def __repr__(self) -> str:
        """Compact description naming the fraction."""
        return f"FixedFractional({self._risk_pct})"

    @property
    def name(self) -> str:
        """Stable identifier."""
        return "FIXED_FRACTIONAL"

    def size(self, request: SizingRequest) -> SizingOutcome:
        """Risk ``risk_pct`` of equity.

        Args:
            request: Sizing inputs.

        Returns:
            The amount to risk.
        """
        amount = request.equity * Decimal(str(self._risk_pct))
        return SizingOutcome(
            risk_amount=amount,
            reason=RiskReason.SIZED,
            detail=f"{self.name} at {self._risk_pct:.4%} of equity {request.equity}",
        )


class FixedAmount:
    """Risk the same sum of money on every trade, regardless of equity.

    Does not compound and does not de-risk on a drawdown, which is the point: it
    makes a strategy's raw trade sequence readable without equity-curve feedback
    mixed into it. For live trading it is what a fixed-stake mandate looks like.
    """

    __slots__ = ("_amount",)

    def __init__(self, amount: Decimal) -> None:
        """Configure the flat stake.

        Args:
            amount: Money to risk per trade, in account currency.

        Raises:
            ValueError: If the amount is not positive.
        """
        if amount <= 0:
            raise ValueError(f"amount must be positive, got {amount}")
        self._amount = amount

    def __repr__(self) -> str:
        """Compact description naming the stake."""
        return f"FixedAmount({self._amount})"

    @property
    def name(self) -> str:
        """Stable identifier."""
        return "FIXED_AMOUNT"

    def size(self, request: SizingRequest) -> SizingOutcome:
        """Risk the configured amount.

        Args:
            request: Sizing inputs. Unused beyond reporting: the whole point of
                this method is that it does not vary.

        Returns:
            The amount to risk. The engine still caps it against equity, so a
            flat stake on a shrunken account is reduced rather than honoured.
        """
        del request
        return SizingOutcome(
            risk_amount=self._amount,
            reason=RiskReason.SIZED,
            detail=f"{self.name} at a flat {self._amount} per trade",
        )


class VolatilityTargeting:
    """Size so that an ATR-sized move is worth a target fraction of equity.

    Position size comes out inversely proportional to ATR: the same instrument in
    a quiet week gets a larger position than in a violent one, so the *money*
    swing of a typical bar stays roughly constant across regimes. That is a
    different objective from fixed-fractional, which holds the loss at the stop
    constant and lets bar-to-bar volatility do what it likes.

    Expressed as an amount of risk rather than a size — see
    :mod:`trading_system.risk.sizing.base` for the algebra and for why every
    method returns money.
    """

    __slots__ = ("_target_pct",)

    def __init__(self, target_pct: float) -> None:
        """Configure the volatility target.

        Args:
            target_pct: Fraction of equity one ATR of movement should be worth.
                Must be in ``(0, 1]``.

        Raises:
            ValueError: If the fraction is outside ``(0, 1]``.
        """
        if not 0 < target_pct <= 1:
            raise ValueError(f"target_pct is a fraction of equity in (0, 1], got {target_pct}")
        self._target_pct = target_pct

    def __repr__(self) -> str:
        """Compact description naming the target."""
        return f"VolatilityTargeting({self._target_pct})"

    @property
    def name(self) -> str:
        """Stable identifier."""
        return "VOLATILITY_TARGETING"

    def size(self, request: SizingRequest) -> SizingOutcome:
        """Risk the amount that makes one ATR worth ``target_pct`` of equity.

        Args:
            request: Sizing inputs. ``atr_price`` is required.

        Returns:
            The amount to risk, or a refusal when ATR is unavailable or zero. A
            zero ATR is refused rather than divided by: it means a flat window in
            the data, not an instrument that cannot move.
        """
        if request.atr_price is None:
            return SizingOutcome(
                risk_amount=None,
                reason=RiskReason.ATR_UNAVAILABLE,
                detail=f"{self.name} sizes from ATR and none was computed for this bar",
            )
        if request.atr_price <= 0:
            return SizingOutcome(
                risk_amount=None,
                reason=RiskReason.ATR_UNAVAILABLE,
                detail=(
                    f"{self.name} needs a positive ATR, got {request.atr_price}; a flat "
                    "window is not a zero-volatility instrument"
                ),
            )
        scale = Decimal(str(request.stop_distance_price)) / Decimal(str(request.atr_price))
        amount = request.equity * Decimal(str(self._target_pct)) * scale
        return SizingOutcome(
            risk_amount=amount,
            reason=RiskReason.SIZED,
            detail=(
                f"{self.name} targeting {self._target_pct:.4%} of equity per ATR of "
                f"{request.atr_price:.6g}, against a stop {request.stop_distance_price:.6g} away"
            ),
        )


class QualityScaled:
    """Risk a fraction of equity that rises linearly with the signal's quality.

    Two decisions are worth spelling out.

    **The floor is a gate, not a small size.** Below ``quality_floor`` the trade
    is not taken at all. Sizing it tiny instead would still pay the spread and
    the commission, still occupy a slot against
    ``max_concurrent_positions``, and still add a trade to the statistics — for
    an edge the strategy itself scored as absent.

    **The scale runs from the floor, not from zero.** Quality is mapped over
    ``[quality_floor, 1]`` onto ``[min_risk_pct, max_risk_pct]``, so the worst
    signal that gets traded is the one that risks ``min_risk_pct``. Mapping from
    zero instead would make ``min_risk_pct`` unreachable and leave a jump at the
    floor: a signal a hair above it would open at some arbitrary interior size
    rather than the smallest one.
    """

    __slots__ = ("_floor", "_max_risk_pct", "_min_risk_pct")

    def __init__(self, *, min_risk_pct: float, max_risk_pct: float, quality_floor: float) -> None:
        """Configure the quality-to-risk mapping.

        Args:
            min_risk_pct: Fraction of equity risked at ``quality_floor``.
            max_risk_pct: Fraction of equity risked at quality ``1.0``.
            quality_floor: Below this quality, no trade at all. Must be in
                ``[0, 1)`` — a floor of exactly one would leave no signal
                tradable and no range to scale over.

        Raises:
            ValueError: If a fraction is outside ``(0, 1]``, the maximum is below
                the minimum, or the floor is outside ``[0, 1)``.
        """
        for label, value in (("min_risk_pct", min_risk_pct), ("max_risk_pct", max_risk_pct)):
            if not 0 < value <= 1:
                raise ValueError(f"{label} is a fraction of equity in (0, 1], got {value}")
        if max_risk_pct < min_risk_pct:
            raise ValueError(f"max_risk_pct {max_risk_pct} is below min_risk_pct {min_risk_pct}")
        if not 0 <= quality_floor < 1:
            raise ValueError(
                f"quality_floor must be in [0, 1), got {quality_floor}; a floor of 1 leaves "
                "nothing tradable and no range to scale risk over"
            )
        self._min_risk_pct = min_risk_pct
        self._max_risk_pct = max_risk_pct
        self._floor = quality_floor

    def __repr__(self) -> str:
        """Compact description naming the range and the floor."""
        return f"QualityScaled({self._min_risk_pct}..{self._max_risk_pct}, floor={self._floor})"

    @property
    def name(self) -> str:
        """Stable identifier."""
        return "QUALITY_SCALED"

    def size(self, request: SizingRequest) -> SizingOutcome:
        """Scale risk linearly with quality, refusing anything below the floor.

        Args:
            request: Sizing inputs.

        Returns:
            The amount to risk, or a refusal when quality is below the floor.
        """
        if request.quality < self._floor:
            return SizingOutcome(
                risk_amount=None,
                reason=RiskReason.QUALITY_BELOW_FLOOR,
                detail=(
                    f"quality {request.quality:.4g} is below the floor {self._floor:.4g}; "
                    "not taken at all rather than taken small"
                ),
            )
        span = 1.0 - self._floor
        position = (request.quality - self._floor) / span
        risk_pct = self._min_risk_pct + position * (self._max_risk_pct - self._min_risk_pct)
        amount = request.equity * Decimal(str(risk_pct))
        return SizingOutcome(
            risk_amount=amount,
            reason=RiskReason.SIZED,
            detail=(
                f"{self.name} at {risk_pct:.4%} of equity for quality {request.quality:.4g} "
                f"in [{self._floor:.4g}, 1]"
            ),
        )
