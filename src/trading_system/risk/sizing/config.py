"""Choosing a sizing method from configuration.

Same two-layer split P04, P06 and P07 all use: the pydantic models are the
syntax — one discriminated union, unknown fields rejected, per-field bounds
checked — and :func:`build_sizing_method` is the semantics, constructing the real
object whose own constructor raises on a combination no field bound can see (a
maximum risk below the minimum, a quality floor of one).

The discriminator is a literal ``method`` field rather than pydantic's smart
union. In a union whose members differ only by which float fields they carry,
"smart" matching would happily accept a ``FIXED_FRACTIONAL`` block that had been
given a ``target_pct`` key by silently choosing a different method, and the run
would size correctly-looking positions by a rule nobody selected.
"""

from decimal import Decimal
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter
from pydantic import ValidationError as PydanticValidationError

from trading_system.core.exceptions import ValidationError
from trading_system.risk.sizing.base import SizingMethod
from trading_system.risk.sizing.methods import (
    FixedAmount,
    FixedFractional,
    QualityScaled,
    VolatilityTargeting,
)


class FixedFractionalConfig(BaseModel):
    """Config for :class:`~trading_system.risk.sizing.methods.FixedFractional`.

    Attributes:
        risk_pct: Fraction of equity to risk per trade — ``0.005`` is 0.5%.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    method: Literal["FIXED_FRACTIONAL"] = "FIXED_FRACTIONAL"
    risk_pct: float = Field(default=0.005, gt=0, le=1)


class FixedAmountConfig(BaseModel):
    """Config for :class:`~trading_system.risk.sizing.methods.FixedAmount`.

    Attributes:
        amount: Money to risk per trade, in account currency. Written as a JSON
            or YAML **string** so it lands on the exact decimal spelled, not on
            the nearest float to it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    method: Literal["FIXED_AMOUNT"] = "FIXED_AMOUNT"
    amount: Decimal = Field(gt=0)


class VolatilityTargetingConfig(BaseModel):
    """Config for :class:`~trading_system.risk.sizing.methods.VolatilityTargeting`.

    Attributes:
        target_pct: Fraction of equity one ATR of movement should be worth.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    method: Literal["VOLATILITY_TARGETING"] = "VOLATILITY_TARGETING"
    target_pct: float = Field(gt=0, le=1)


class QualityScaledConfig(BaseModel):
    """Config for :class:`~trading_system.risk.sizing.methods.QualityScaled`.

    Attributes:
        min_risk_pct: Fraction of equity risked at ``quality_floor``.
        max_risk_pct: Fraction of equity risked at quality ``1.0``.
        quality_floor: Below this quality, the trade is refused outright.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    method: Literal["QUALITY_SCALED"] = "QUALITY_SCALED"
    min_risk_pct: float = Field(gt=0, le=1)
    max_risk_pct: float = Field(gt=0, le=1)
    quality_floor: float = Field(ge=0, lt=1)


#: How position size is computed. Selected in configuration, never in code.
SizingConfig = Annotated[
    FixedFractionalConfig | FixedAmountConfig | VolatilityTargetingConfig | QualityScaledConfig,
    Field(discriminator="method"),
]


#: Where the named sizing variants live. A library file rather than values
#: written into a search space, for the same reason exit presets are a library:
#: a categorical axis names entries, and the entries have to be authored
#: somewhere a person can read them side by side.
DEFAULT_SIZING_METHODS_PATH = Path("configs/sizing_methods.yaml")


def load_sizing_methods(path: Path = DEFAULT_SIZING_METHODS_PATH) -> dict[str, SizingConfig]:
    """Load the named sizing variants a categorical axis chooses between.

    Args:
        path: The YAML file. A missing file yields an empty library rather than
            an error: a space with no sizing axis never looks one up, and
            failing to start over a file nothing needed would be a rule for a
            case that does not exist.

    Returns:
        Configs by name.

    Raises:
        ValidationError: If the file is not a mapping of name to sizing block,
            or a block fails validation.
    """
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValidationError(
            f"{path}: expected a mapping of name to sizing block, got {type(raw)}"
        )
    adapter: TypeAdapter[SizingConfig] = TypeAdapter(SizingConfig)
    out: dict[str, SizingConfig] = {}
    for name, block in raw.items():
        try:
            out[str(name)] = adapter.validate_python(block)
        except PydanticValidationError as error:
            raise ValidationError(f"{path}: sizing method {name!r} is invalid: {error}") from error
    return out


def declared_risk_fraction(config: SizingConfig) -> float | None:
    """The largest equity fraction a config asks for per trade, if it names one.

    Comparable against
    :attr:`~trading_system.risk.engine.RiskEngineConfig.max_risk_pct`, the
    ceiling the engine applies after the method has had its say. Two methods
    name such a fraction and two do not: ``FIXED_AMOUNT`` asks for money rather
    than a share of equity, and ``VOLATILITY_TARGETING`` asks for a share of
    equity per *ATR of movement*, which becomes a per-trade fraction only once a
    stop distance exists. ``None`` for those two is the honest answer, not a
    missing case — a number invented for them would be compared against a cap it
    has no relation to.

    Args:
        config: Validated configuration block.

    Returns:
        The fraction, or ``None`` when the method does not express one.
    """
    match config:
        case FixedFractionalConfig():
            return config.risk_pct
        case QualityScaledConfig():
            return config.max_risk_pct
        case _:
            return None


def build_sizing_method(config: SizingConfig) -> SizingMethod:
    """Construct the sizing method a validated config describes.

    Args:
        config: Validated configuration block.

    Returns:
        The method.

    Raises:
        ValueError: If the values are individually valid but jointly are not —
            ``max_risk_pct`` below ``min_risk_pct``, for instance. Raised by the
            method's own constructor, so a method built from config and a method
            built by hand reject exactly the same things.
    """
    match config:
        case FixedFractionalConfig():
            return FixedFractional(config.risk_pct)
        case FixedAmountConfig():
            return FixedAmount(config.amount)
        case VolatilityTargetingConfig():
            return VolatilityTargeting(config.target_pct)
        case QualityScaledConfig():
            return QualityScaled(
                min_risk_pct=config.min_risk_pct,
                max_risk_pct=config.max_risk_pct,
                quality_floor=config.quality_floor,
            )
