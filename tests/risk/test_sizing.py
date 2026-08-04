"""The four sizing methods, in isolation from the engine that divides by a stop."""

from decimal import Decimal

import pytest

from trading_system.risk.models import RiskReason
from trading_system.risk.sizing import (
    FixedAmount,
    FixedAmountConfig,
    FixedFractional,
    FixedFractionalConfig,
    QualityScaled,
    QualityScaledConfig,
    SizingRequest,
    VolatilityTargeting,
    VolatilityTargetingConfig,
    build_sizing_method,
)

EQUITY = Decimal("100000")


def request(
    *, quality: float = 0.8, stop: float = 0.0050, atr: float | None = 0.0030
) -> SizingRequest:
    """Build a sizing request with EURUSD-shaped defaults."""
    return SizingRequest(equity=EQUITY, quality=quality, stop_distance_price=stop, atr_price=atr)


class TestFixedFractional:
    def test_it_risks_the_configured_fraction_of_equity(self) -> None:
        outcome = FixedFractional(0.005).size(request())
        assert outcome.risk_amount == Decimal("500.000")

    def test_the_fraction_is_a_fraction_not_a_percentage(self) -> None:
        # 0.005 means half a percent. Reading it as 0.005% would risk 5 USD on
        # a 100k account and reading it as 5% would risk 5 000.
        assert FixedFractional(0.005).size(request()).risk_amount == Decimal("500.000")

    def test_a_fraction_above_one_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="fraction of equity"):
            FixedFractional(5.0)

    def test_a_non_positive_fraction_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="fraction of equity"):
            FixedFractional(0.0)


class TestFixedAmount:
    def test_it_ignores_equity_entirely(self) -> None:
        method = FixedAmount(Decimal("250"))
        small = SizingRequest(
            equity=Decimal("5000"), quality=0.8, stop_distance_price=0.005, atr_price=None
        )
        assert method.size(request()).risk_amount == Decimal("250")
        assert method.size(small).risk_amount == Decimal("250")

    def test_a_non_positive_stake_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            FixedAmount(Decimal("0"))


class TestVolatilityTargeting:
    def test_risk_scales_with_the_stop_to_atr_ratio(self) -> None:
        # target 1% of 100k = 1 000 per ATR. A stop of 1.5 ATR therefore risks
        # 1 500, which is the size that makes one ATR of movement worth 1%.
        outcome = VolatilityTargeting(0.01).size(request(stop=0.0045, atr=0.0030))
        assert outcome.risk_amount == Decimal("1500.00")

    def test_size_comes_out_inversely_proportional_to_atr(self) -> None:
        # The defining property. Holding the stop fixed, doubling ATR halves
        # the risk taken and therefore halves the position.
        method = VolatilityTargeting(0.01)
        quiet = method.size(request(stop=0.0050, atr=0.0025)).risk_amount
        wild = method.size(request(stop=0.0050, atr=0.0050)).risk_amount
        assert quiet is not None and wild is not None
        assert quiet == wild * 2

    def test_a_missing_atr_refuses_rather_than_dividing(self) -> None:
        outcome = VolatilityTargeting(0.01).size(request(atr=None))
        assert outcome.risk_amount is None
        assert outcome.reason is RiskReason.ATR_UNAVAILABLE

    def test_a_zero_atr_refuses_rather_than_dividing_by_zero(self) -> None:
        # A flat window in the data is not a zero-volatility instrument.
        outcome = VolatilityTargeting(0.01).size(request(atr=0.0))
        assert outcome.risk_amount is None
        assert outcome.reason is RiskReason.ATR_UNAVAILABLE


class TestQualityScaled:
    @pytest.fixture
    def method(self) -> QualityScaled:
        return QualityScaled(min_risk_pct=0.002, max_risk_pct=0.01, quality_floor=0.6)

    def test_below_the_floor_refuses_rather_than_sizing_small(self, method: QualityScaled) -> None:
        # A refusal, not a risk of zero: a zero would reach the engine as a
        # size below the minimum lot and be reported as "too small to trade",
        # which is a different fact from "not good enough to trade".
        outcome = method.size(request(quality=0.5))
        assert outcome.risk_amount is None
        assert outcome.reason is RiskReason.QUALITY_BELOW_FLOOR

    def test_at_the_floor_it_risks_exactly_the_minimum(self, method: QualityScaled) -> None:
        assert method.size(request(quality=0.6)).risk_amount == Decimal("200.000")

    def test_at_perfect_quality_it_risks_exactly_the_maximum(self, method: QualityScaled) -> None:
        assert method.size(request(quality=1.0)).risk_amount == Decimal("1000.00")

    def test_halfway_up_the_range_is_halfway_up_the_risk(self, method: QualityScaled) -> None:
        # Quality 0.8 is halfway from the 0.6 floor to 1.0, so risk is halfway
        # from 0.2% to 1.0%, i.e. 0.6% = 600.
        outcome = method.size(request(quality=0.8))
        assert outcome.risk_amount == pytest.approx(Decimal("600"))

    def test_the_scale_starts_at_the_floor_not_at_zero(self, method: QualityScaled) -> None:
        # Scaling from zero would make min_risk_pct unreachable and leave a
        # jump at the floor: the worst tradable signal would open at some
        # arbitrary interior size rather than the smallest one.
        at_floor = method.size(request(quality=0.6)).risk_amount
        assert at_floor == EQUITY * Decimal("0.002")

    def test_a_maximum_below_the_minimum_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="below min_risk_pct"):
            QualityScaled(min_risk_pct=0.01, max_risk_pct=0.002, quality_floor=0.5)

    def test_a_floor_of_one_leaves_nothing_tradable(self) -> None:
        with pytest.raises(ValueError, match="quality_floor must be in"):
            QualityScaled(min_risk_pct=0.002, max_risk_pct=0.01, quality_floor=1.0)


class TestConfiguration:
    """A method is chosen in config, never in code."""

    def test_each_config_builds_its_method(self) -> None:
        assert isinstance(build_sizing_method(FixedFractionalConfig()), FixedFractional)
        assert isinstance(
            build_sizing_method(FixedAmountConfig(amount=Decimal("100"))), FixedAmount
        )
        assert isinstance(
            build_sizing_method(VolatilityTargetingConfig(target_pct=0.01)),
            VolatilityTargeting,
        )
        assert isinstance(
            build_sizing_method(
                QualityScaledConfig(min_risk_pct=0.002, max_risk_pct=0.01, quality_floor=0.5)
            ),
            QualityScaled,
        )

    def test_an_unknown_field_is_rejected_rather_than_ignored(self) -> None:
        import pydantic

        with pytest.raises(pydantic.ValidationError):
            FixedFractionalConfig(risk_pct=0.005, target_pct=0.01)  # type: ignore[call-arg]

    def test_a_semantic_error_surfaces_from_the_methods_own_constructor(self) -> None:
        # Field bounds cannot see this: both fractions are individually valid.
        # A config-built method rejects exactly what a hand-built one rejects.
        config = QualityScaledConfig(min_risk_pct=0.01, max_risk_pct=0.002, quality_floor=0.5)
        with pytest.raises(ValueError, match="below min_risk_pct"):
            build_sizing_method(config)

    def test_the_amount_lands_on_the_decimal_written(self) -> None:
        # Written as a string so it does not pass through float first.
        config = FixedAmountConfig.model_validate({"method": "FIXED_AMOUNT", "amount": "0.1"})
        assert config.amount == Decimal("0.1")
