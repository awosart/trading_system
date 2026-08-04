"""The strategy contract: one pydantic model every downstream module agrees on.

A :class:`StrategySpec` describes *what to trade and when a signal fires* — it
never touches money. Position sizing, stop placement and trading permission are
decided downstream by the Risk Engine and Prop Guard from the
:class:`~trading_system.core.types.Signal` a strategy emits, not from this spec.
Entry Engine and Exit Engine consume this contract independently: a spec names
its exit by ``exit_ref`` rather than embedding one, so the two combine N×M
without either importing the other.

Condition trees (``Condition = LeafCondition | AllOf | AnyOf | Not``) are the one
recursive piece. Every leaf compares two operands, each of which is a
:class:`FeatureRef` (a structured pointer to one indicator, its parameters and,
for multi-output indicators, a channel), a price reference (``"price:close"``),
or a bare numeric constant. A :class:`FeatureRef` deliberately does not collapse
back into a string like ``"feature:rsi_14"``: a string forces a reader (human or
validator) to reverse-engineer which indicator and parameters produced it, and
cannot express *which channel* of a multi-output indicator like MACD is meant at
all. :mod:`trading_system.strategies.validator` checks a ``FeatureRef``'s
``indicator``, ``params`` and ``channel`` against the real indicator registry.

Every model here is frozen and rejects unknown fields (``extra="forbid"``): a
strategy spec is version-controlled configuration, not a scratch object, and a
typo in a field name must fail at load time rather than being silently ignored.
"""

import re
from collections.abc import Iterator
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator

from trading_system.core.types import Timeframe

#: Location of the generated JSON Schema, kept alongside this module so editors
#: can point a ``$schema`` reference at a stable relative path.
SCHEMA_JSON_PATH = Path(__file__).parent / "schema.json"


class StrategyType(StrEnum):
    """Holding-period class a strategy belongs to."""

    SWING = "SWING"
    INTRADAY = "INTRADAY"
    SCALP = "SCALP"
    POSITION = "POSITION"


class Status(StrEnum):
    """Lifecycle stage of a strategy definition."""

    DRAFT = "DRAFT"
    TESTING = "TESTING"
    APPROVED = "APPROVED"
    RETIRED = "RETIRED"


class Regime(StrEnum):
    """Market regime a strategy is permitted to trade in."""

    TREND_UP = "TREND_UP"
    TREND_DOWN = "TREND_DOWN"
    RANGE = "RANGE"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"


class Direction(StrEnum):
    """Trade direction(s) an entry is willing to take."""

    LONG = "LONG"
    SHORT = "SHORT"
    BOTH = "BOTH"


class ConditionOp(StrEnum):
    """Comparison performed by a :class:`LeafCondition`."""

    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"
    CROSS_ABOVE = "cross_above"
    CROSS_BELOW = "cross_below"
    BETWEEN = "between"
    RISING = "rising"
    FALLING = "falling"
    INSIDE_RANGE = "inside_range"
    PATTERN_IS = "pattern_is"
    REGIME_IS = "regime_is"
    SESSION_IS = "session_is"


class InstrumentClass(StrEnum):
    """Broad instrument category an entry is scoped to."""

    FX = "FX"
    CRYPTO = "CRYPTO"
    EQUITY = "EQUITY"
    FUTURES = "FUTURES"
    INDEX = "INDEX"
    COMMODITY = "COMMODITY"


_PRICE_FIELDS = ("open", "high", "low", "close", "volume")
_PRICE_REF_PATTERN = re.compile(rf"^price:({'|'.join(_PRICE_FIELDS)})$")
_SLUG_PATTERN = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
_SEMVER_PATTERN = re.compile(r"^\d+\.\d+\.\d+$")
_TIME_OF_DAY_PATTERN = re.compile(r"^([01]\d|2[0-3]):([0-5]\d)$")


def _validate_price_ref(value: str) -> str:
    """Reject any string that isn't a ``price:<field>`` reference."""
    if _PRICE_REF_PATTERN.match(value):
        return value
    raise ValueError(f"{value!r} must match 'price:<{'|'.join(_PRICE_FIELDS)}>'")


def _validate_time_of_day(value: str) -> str:
    """Reject anything but a 24-hour ``HH:MM`` timestamp."""
    if not _TIME_OF_DAY_PATTERN.match(value):
        raise ValueError(f"{value!r} must be 'HH:MM' in 24-hour UTC time")
    return value


class FeatureRef(BaseModel):
    """A structural pointer to one channel of one registered indicator.

    Replaces an earlier ``"feature:<name>"`` string convention, which could not
    distinguish a legitimate unusual parameter (``rsi`` with ``period=500``)
    from a typo, and had no way to say *which* channel of a multi-output
    indicator like MACD or Bollinger Bands was meant — both are things
    :mod:`trading_system.strategies.validator` must be able to check against
    the real indicator registry, not guess at from a string.

    Attributes:
        indicator: Registry key, e.g. ``"rsi"`` — see
            :data:`~trading_system.features.registry.INDICATOR_TYPES`.
        params: Constructor arguments for the indicator, e.g. ``{"period": 14}``.
            Empty means every parameter takes its default.
        channel: Which output channel to read, e.g. ``"signal"`` for MACD.
            Required for multi-output indicators; must be omitted for
            single-output ones, which publish under their own name with no
            channel to choose between.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    indicator: str = Field(min_length=1)
    params: dict[str, Any] = Field(default_factory=dict)
    channel: str | None = Field(default=None, min_length=1)


#: A price reference string, e.g. ``"price:close"``.
PriceRef = Annotated[str, AfterValidator(_validate_price_ref)]

#: A leaf operand: a feature reference, a price reference, or a bare constant.
Operand = FeatureRef | PriceRef | float

#: Inclusive numeric bounds, e.g. the ``[30, 70]`` in an RSI ``between`` check.
RangeBound = tuple[float, float]

TimeOfDay = Annotated[str, AfterValidator(_validate_time_of_day)]


def operand_feature_ref(operand: Operand | RangeBound) -> FeatureRef | None:
    """The :class:`FeatureRef` carried by ``operand``, or ``None`` if it isn't one.

    Args:
        operand: A leaf condition's ``left`` or ``right`` value.

    Returns:
        ``operand`` itself when it is a :class:`FeatureRef`, else ``None``.
    """
    return operand if isinstance(operand, FeatureRef) else None


def operand_price_field(operand: Operand | RangeBound) -> str | None:
    """The price field referenced by ``operand``, or ``None`` if it isn't one.

    Args:
        operand: A leaf condition's ``left`` or ``right`` value.

    Returns:
        The field after ``price:`` (e.g. ``"close"``), or ``None`` for a
        feature reference, constant, or range bound.
    """
    if isinstance(operand, str) and operand.startswith("price:"):
        return operand.removeprefix("price:")
    return None


class LeafCondition(BaseModel):
    """A single comparison against a feature, a price, or a constant.

    Attributes:
        type: Discriminator tag, always ``"leaf"``.
        op: The comparison performed.
        left: The primary operand. ``None`` only for ops such as
            ``regime_is``/``session_is`` that test global bar context rather
            than a specific series.
        right: The comparison target. A plain float for most ops; a
            ``[low, high]`` pair for ``between``/``inside_range``; ``None`` for
            unary tests like ``rising``/``falling`` with no explicit lookback.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    type: Literal["leaf"] = "leaf"
    op: ConditionOp
    left: Operand | None = None
    right: Operand | RangeBound | None = None


class AllOf(BaseModel):
    """Every child condition must hold.

    Attributes:
        type: Discriminator tag, always ``"all_of"``.
        conditions: Children to AND together.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    type: Literal["all_of"] = "all_of"
    conditions: list["Condition"] = Field(min_length=1)


class AnyOf(BaseModel):
    """At least one child condition must hold.

    Attributes:
        type: Discriminator tag, always ``"any_of"``.
        conditions: Children to OR together.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    type: Literal["any_of"] = "any_of"
    conditions: list["Condition"] = Field(min_length=1)


class Not(BaseModel):
    """Negates a child condition.

    Attributes:
        type: Discriminator tag, always ``"not"``.
        condition: The condition to negate.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    type: Literal["not"] = "not"
    condition: "Condition"


#: A trigger, confirmation, or invalidation condition tree.
Condition = Annotated[LeafCondition | AllOf | AnyOf | Not, Field(discriminator="type")]

AllOf.model_rebuild()
AnyOf.model_rebuild()
Not.model_rebuild()


def iter_leaf_conditions(condition: Condition) -> Iterator[LeafCondition]:
    """Walk a condition tree, yielding every leaf.

    Args:
        condition: Root of the tree to walk.

    Yields:
        Each :class:`LeafCondition` found, in depth-first order.
    """
    if isinstance(condition, LeafCondition):
        yield condition
    elif isinstance(condition, Not):
        yield from iter_leaf_conditions(condition.condition)
    else:
        for child in condition.conditions:
            yield from iter_leaf_conditions(child)


class Invalidation(BaseModel):
    """A condition or price level that voids a pending setup before entry.

    At least one of ``condition`` and ``price_level`` must be set.

    Attributes:
        condition: A condition that, once true, cancels the setup.
        price_level: A price beyond which the setup is void. Its side
            (above/below entry) is implied by :attr:`EntrySpec.direction`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    condition: Condition | None = None
    price_level: Operand | None = None

    @model_validator(mode="after")
    def _require_condition_or_price(self) -> "Invalidation":
        """Reject an invalidation rule that would never trigger."""
        if self.condition is None and self.price_level is None:
            raise ValueError("invalidation requires a condition, a price_level, or both")
        return self


class MarketOrder(BaseModel):
    """Enter at the next available price."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    type: Literal["MARKET"] = "MARKET"


class LimitOrder(BaseModel):
    """Enter at an offset better than the trigger price.

    Attributes:
        offset: Distance from the trigger price, in price units. Always
            positive; the favourable side is implied by direction.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    type: Literal["LIMIT"] = "LIMIT"
    offset: float = Field(gt=0)


class StopOrder(BaseModel):
    """Enter at an offset beyond the trigger price, on confirmation of momentum.

    Attributes:
        offset: Distance from the trigger price, in price units.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    type: Literal["STOP"] = "STOP"
    offset: float = Field(gt=0)


#: How the entry order is placed once the trigger and confirmation fire.
EntryOrder = Annotated[MarketOrder | LimitOrder | StopOrder, Field(discriminator="type")]


class EntryOrderSpec(BaseModel):
    """The order placed once a setup is confirmed.

    Attributes:
        order: Order type and its type-specific parameters.
        expire_after_bars: Cancel the pending order if unfilled after this many
            bars. ``None`` means it never expires on its own.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    order: EntryOrder
    expire_after_bars: int | None = Field(default=None, gt=0)


class EntrySpec(BaseModel):
    """Everything needed to recognise and place one entry.

    Attributes:
        direction: Direction(s) this entry is willing to take.
        trigger: The condition that opens a setup.
        confirmation: Additional conditions that must all hold within
            ``confirmation_window_bars`` bars of the trigger. Empty means the
            trigger alone is sufficient.
        confirmation_window_bars: Bars, counted from the trigger bar, allowed
            for every entry of ``confirmation`` to become true. Must be
            positive exactly when ``confirmation`` is non-empty — a window
            without conditions or conditions without a window is a
            contradictory config, not a default.
        invalidation: Condition or level that cancels a pending setup.
        entry_order: How the resulting order is placed.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    direction: Direction
    trigger: Condition
    confirmation: list[Condition] = Field(default_factory=list)
    confirmation_window_bars: int = Field(default=0, ge=0)
    invalidation: Invalidation | None = None
    entry_order: EntryOrderSpec

    @model_validator(mode="after")
    def _confirmation_window_consistent(self) -> "EntrySpec":
        """Reject a confirmation window that doesn't match its conditions."""
        if self.confirmation and self.confirmation_window_bars <= 0:
            raise ValueError("confirmation_window_bars must be positive when confirmation is set")
        if not self.confirmation and self.confirmation_window_bars:
            raise ValueError("confirmation_window_bars is set but confirmation is empty")
        return self


class SessionFilter(BaseModel):
    """Only allow entries during named trading sessions.

    Attributes:
        sessions: Session names, e.g. ``"LONDON"``, ``"NEW_YORK"``, ``"TOKYO"``.
            Interpretation is the trading calendar's; this spec only carries
            the name.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["session"] = "session"
    sessions: list[str] = Field(min_length=1)


class NewsBlackoutFilter(BaseModel):
    """Block entries around scheduled news releases.

    Attributes:
        minutes_before: Minutes before a release during which entry is blocked.
        minutes_after: Minutes after a release during which entry is blocked.
        min_impact: Lowest news-impact tier that triggers the blackout.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["news_blackout"] = "news_blackout"
    minutes_before: int = Field(ge=0)
    minutes_after: int = Field(ge=0)
    min_impact: Literal["LOW", "MEDIUM", "HIGH"] = "HIGH"


class VolatilityRangeFilter(BaseModel):
    """Only allow entries while a volatility feature sits inside a band.

    Attributes:
        feature: Volatility feature reference, e.g. ATR.
        min_value: Inclusive lower bound. ``None`` means no lower bound.
        max_value: Inclusive upper bound. ``None`` means no upper bound.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["volatility_range"] = "volatility_range"
    feature: FeatureRef
    min_value: float | None = None
    max_value: float | None = None

    @model_validator(mode="after")
    def _has_a_bound(self) -> "VolatilityRangeFilter":
        """Reject a band with neither bound, which would filter nothing."""
        if self.min_value is None and self.max_value is None:
            raise ValueError("volatility_range needs min_value, max_value, or both")
        return self


class SpreadMaxFilter(BaseModel):
    """Block entries when the quoted spread is too wide.

    Attributes:
        max_spread_pips: Widest tolerable spread, in pips.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["spread_max"] = "spread_max"
    max_spread_pips: float = Field(gt=0)


class CorrelationMaxFilter(BaseModel):
    """Block entries that would over-concentrate risk in correlated instruments.

    Attributes:
        max_correlation: Highest tolerable absolute correlation to an existing
            position.
        lookback_bars: Bars of return history used to estimate correlation.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["correlation_max"] = "correlation_max"
    max_correlation: float = Field(ge=0.0, le=1.0)
    lookback_bars: int = Field(gt=0)


class DayOfWeekFilter(BaseModel):
    """Only allow entries on named weekdays.

    Attributes:
        allowed_days: Weekdays entries are permitted on.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["day_of_week"] = "day_of_week"
    allowed_days: list[Literal["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]] = Field(
        min_length=1
    )


class TimeOfDayFilter(BaseModel):
    """Only allow entries within a UTC time-of-day window.

    Attributes:
        start: Window start, ``"HH:MM"`` UTC, inclusive.
        end: Window end, ``"HH:MM"`` UTC, exclusive. May be earlier than
            ``start`` to express a window crossing midnight.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["time_of_day"] = "time_of_day"
    start: TimeOfDay
    end: TimeOfDay


class MinLiquidityFilter(BaseModel):
    """Block entries when recent volume is too thin to trust or fill.

    Attributes:
        min_volume: Lowest tolerable average volume.
        lookback_bars: Bars averaged to estimate current liquidity.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["min_liquidity"] = "min_liquidity"
    min_volume: float = Field(gt=0)
    lookback_bars: int = Field(default=20, gt=0)


#: A filter can only forbid an entry, never make one more likely.
FilterSpec = Annotated[
    SessionFilter
    | NewsBlackoutFilter
    | VolatilityRangeFilter
    | SpreadMaxFilter
    | CorrelationMaxFilter
    | DayOfWeekFilter
    | TimeOfDayFilter
    | MinLiquidityFilter,
    Field(discriminator="kind"),
]


class QualityModifier(BaseModel):
    """A condition-gated adjustment to a signal's base quality score.

    Attributes:
        condition: When true, ``delta`` is added to ``base_quality``.
        delta: Adjustment applied. The final quality is still clamped to
            ``[0, 1]`` by the Risk Engine; this spec does not enforce the
            bound on any partial sum.
        reason: Human-readable justification, for review and logging.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    condition: Condition
    delta: float = Field(ge=-1.0, le=1.0)
    reason: str | None = None


class AtrStop(BaseModel):
    """Stop distance derived from a multiple of ATR.

    Attributes:
        period: ATR lookback.
        multiple: Multiple of ATR used as the stop distance.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["ATR"] = "ATR"
    period: int = Field(default=14, gt=0)
    multiple: float = Field(default=1.5, gt=0)


class StructureStop(BaseModel):
    """Stop placed beyond the nearest swing structure.

    Attributes:
        lookback_bars: How far back to look for structure.
        buffer_atr_multiple: Extra ATR-scaled buffer beyond the structure
            level, to absorb noise around it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["STRUCTURE"] = "STRUCTURE"
    lookback_bars: int = Field(default=20, gt=0)
    buffer_atr_multiple: float = Field(default=0.0, ge=0)


class FixedPipsStop(BaseModel):
    """A stop at a fixed pip distance.

    Attributes:
        pips: Stop distance in pips.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["FIXED_PIPS"] = "FIXED_PIPS"
    pips: float = Field(gt=0)


class PercentStop(BaseModel):
    """A stop at a fixed percentage of entry price.

    Attributes:
        percent: Stop distance as a percentage of entry price.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["PERCENT"] = "PERCENT"
    percent: float = Field(gt=0, le=100)


#: How the initial stop distance is derived; the Risk Engine turns this into a
#: price and, from there, a position size.
StopReference = Annotated[
    AtrStop | StructureStop | FixedPipsStop | PercentStop, Field(discriminator="kind")
]


class RiskProfileSpec(BaseModel):
    """Inputs the Risk Engine uses to size and gate a strategy's signals.

    Attributes:
        base_quality: Starting quality score before modifiers, in ``[0, 1]``.
        quality_modifiers: Condition-gated adjustments applied on top.
        stop_reference: How the initial stop distance is derived.
        max_concurrent_positions: Most open positions this strategy may hold
            at once.
        cooldown_bars_after_loss: Bars this strategy must stay flat after a
            losing trade before it may signal again.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    base_quality: float = Field(ge=0.0, le=1.0)
    quality_modifiers: list[QualityModifier] = Field(default_factory=list)
    stop_reference: StopReference
    max_concurrent_positions: int = Field(default=1, gt=0)
    cooldown_bars_after_loss: int = Field(default=0, ge=0)


class TimeframeSpec(BaseModel):
    """The timeframes an entry reasons across.

    Attributes:
        signal_tf: Bar size the entry's trigger and confirmation evaluate on.
        htf_filter_tf: Coarser timeframe used only to filter (e.g. a trend
            bias); ``None`` if the strategy uses none.
        entry_tf: Bar size execution actually fills on. Never coarser than
            ``signal_tf``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    signal_tf: Timeframe
    htf_filter_tf: Timeframe | None = None
    entry_tf: Timeframe


class InstrumentSpec(BaseModel):
    """Which instruments a strategy may be applied to.

    Attributes:
        allowed_classes: Instrument classes this strategy is designed for.
        allowed_symbols: Explicit allow-list. Empty means every symbol of an
            allowed class is eligible.
        denied_symbols: Explicit deny-list, checked after the allow-list.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    allowed_classes: list[InstrumentClass] = Field(min_length=1)
    allowed_symbols: list[str] = Field(default_factory=list)
    denied_symbols: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _no_symbol_both_allowed_and_denied(self) -> "InstrumentSpec":
        """Reject a symbol appearing on both lists, which is unresolvable."""
        overlap = sorted(set(self.allowed_symbols) & set(self.denied_symbols))
        if overlap:
            raise ValueError(f"symbols cannot be both allowed and denied: {overlap}")
        return self


class StrategySpec(BaseModel):
    """The unified description of one trading strategy.

    This is the contract Entry Engine, Exit Engine, the strategy database and
    validation all depend on. A spec never manages money: it names an
    ``exit_ref`` rather than embedding an exit, and its output downstream is a
    :class:`~trading_system.core.types.Signal`, not an order.

    Attributes:
        id: Unique, stable slug (lower-kebab-case).
        name: Human-readable name.
        version: Semantic version of this spec, ``"X.Y.Z"``.
        author: Who defined the strategy.
        source: Where the idea came from (URL, book, paper); optional.
        type: Holding-period class.
        status: Lifecycle stage.
        timeframes: Timeframes the entry reasons across.
        instruments: Which instruments this strategy may trade.
        market_regimes: Regimes this strategy is permitted to trade in. Empty
            means unrestricted.
        entry: Trigger, confirmation, invalidation and order placement.
        exit_ref: Id of the exit strategy this pairs with, looked up in the
            Exit DB — never embedded, so entries and exits combine N×M.
        filters: Permission gates; each can only forbid, never encourage.
        risk_profile: Inputs to the Risk Engine's sizing and gating.
        metadata: Free-form tags and notes. Never read by any control-flow
            decision; a strategy field belongs above this dict, not inside it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    name: str = Field(min_length=1)
    version: str
    author: str = Field(min_length=1)
    source: str | None = None
    type: StrategyType
    status: Status = Status.DRAFT
    timeframes: TimeframeSpec
    instruments: InstrumentSpec
    market_regimes: list[Regime] = Field(default_factory=list)
    entry: EntrySpec
    exit_ref: str = Field(min_length=1)
    filters: list[FilterSpec] = Field(default_factory=list)
    risk_profile: RiskProfileSpec
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _id_is_a_slug(self) -> "StrategySpec":
        """Reject an id that isn't a stable lower-kebab-case slug."""
        if not _SLUG_PATTERN.match(self.id):
            raise ValueError(f"id must be a lowercase slug (e.g. 'ema-pullback'), got {self.id!r}")
        return self

    @model_validator(mode="after")
    def _version_is_semver(self) -> "StrategySpec":
        """Reject a version that isn't ``X.Y.Z``."""
        if not _SEMVER_PATTERN.match(self.version):
            raise ValueError(f"version must be semver 'X.Y.Z', got {self.version!r}")
        return self


def strategy_json_schema() -> dict[str, Any]:
    """Build the JSON Schema (draft 2020-12) editors validate strategy files against.

    Returns:
        The schema as a plain dict, with an explicit ``$schema`` key since
        pydantic does not set one.
    """
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        **StrategySpec.model_json_schema(),
    }
