"""What is filled in when a page does not say, and the record of having filled it.

:mod:`trading_system.strategies.ingest` refuses a card that leaves a required
field unsaid, and that discipline bought a conversion rate of one card in 883.
This module takes the opposite instruction — supply the missing value, choose
the most defensible reading, keep the card — and pays for it the only way that
keeps the result honest: every departure from the page is an
:class:`Inference` carrying a code and a sentence, attached to the spec that
resulted, so that "what the page said" and "what was decided here" stay
separable after the fact.

The distinction that matters is not big versus small. It is whether the
inference is about *encoding* (which preset expresses a 1.3R target) or about
*content* (what this page's entry rule is, when the page never wrote one down).
:class:`~trading_system.strategies.normalize.normalise.Fidelity` grades that,
and a spec built by :func:`archetype_conditions` is never presented as
something the page said.
"""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

from trading_system.core.types import Timeframe
from trading_system.features.patterns import Pattern
from trading_system.features.registry import build_indicator
from trading_system.strategies.schema import (
    Condition,
    ConditionOp,
    Direction,
    FeatureRef,
    LabelSet,
    LeafCondition,
    Operand,
)


class InferenceCode(StrEnum):
    """Kinds of departure from what a card states.

    Read as a scale of how far from the page a spec has travelled: the first
    group encodes something the page said, the middle group drops something it
    said, and :data:`ARCHETYPE_FROM_INDICATORS` writes a rule the page did not.
    """

    TIMEFRAME_FROM_CATEGORY = "timeframe_from_category"
    TIMEFRAME_FLOOR_OF_SEVERAL = "timeframe_floor_of_several"
    UNIVERSE_FROM_STORE = "universe_from_store"
    SIDE_SPLIT_FROM_UNSIDED = "side_split_from_unsided"
    SIDE_MIRRORED = "side_mirrored"
    RULES_FROM_PROSE = "rules_from_prose"
    INDICATOR_SUBSTITUTED = "indicator_substituted"
    CLAUSE_SALVAGED = "clause_salvaged"
    CLAUSE_DROPPED_UNREADABLE = "clause_dropped_unreadable"
    CLAUSE_DROPPED_VISUAL = "clause_dropped_visual"
    CLAUSE_DROPPED_CROSS_TIMEFRAME = "clause_dropped_cross_timeframe"
    CLAUSE_DROPPED_BAR_ARITHMETIC = "clause_dropped_bar_arithmetic"
    CLAUSE_DROPPED_REGIME = "clause_dropped_regime"
    HTF_FILTER_DROPPED = "htf_filter_dropped"
    INVALIDATION_FROM_TRIGGER_LINE = "invalidation_from_trigger_line"
    INVALIDATION_FROM_SWING = "invalidation_from_swing"
    EXIT_SNAPPED_TO_RATIO = "exit_snapped_to_ratio"
    EXIT_FROM_TYPE = "exit_from_type"
    ARCHETYPE_FROM_INDICATORS = "archetype_from_indicators"
    DEFAULTS_APPLIED = "defaults_applied"


@dataclass(frozen=True)
class Inference:
    """One departure from what the card states.

    Attributes:
        code: What kind of departure it is.
        detail: What was decided, in enough words that a reader can disagree
            with it without opening the page.
    """

    code: InferenceCode
    detail: str


#: Bar size assumed for a card that names none, by site category. Each is the
#: smallest size the category's pages routinely name, so the assumption errs
#: towards more bars and more trades rather than towards a comfortable-looking
#: sample. A category not listed falls to :data:`FALLBACK_TIMEFRAME`.
CATEGORY_TIMEFRAME: dict[str, Timeframe] = {
    "scalping-forex-strategies": Timeframe.M5,
    "pivot-forex-strategies": Timeframe.M15,
    "breakout-forex-strategies": Timeframe.H1,
    "bollinger-bands-forex-strategies": Timeframe.H1,
    "forex-strategies-based-on-indicators": Timeframe.H1,
    "support-and-resistance-forex-strategies": Timeframe.H1,
    "volatility-forex-strategies": Timeframe.H1,
    "trend-following-forex-strategies": Timeframe.H4,
    "trend-following-forex-strategies-ii": Timeframe.H4,
    "patterns-forex-strategies": Timeframe.H4,
    "candlestick-forex-strategies": Timeframe.H4,
}

#: Bar size for a card whose category is unknown too.
FALLBACK_TIMEFRAME = Timeframe.H1

#: Indicators the corpus names that this system does not implement, mapped onto
#: one it does. A substitution is only listed where the replacement answers the
#: same question — not where it merely looks similar on a chart. Parabolic SAR
#: and HalfTrend are stop-and-reverse trend followers and ``supertrend`` is the
#: implemented one; the Awesome Oscillator is a two-mean momentum difference
#: around zero and ``macd`` is the implemented one. Heiken Ashi, Gann and
#: Fibonacci are absent on purpose: the first is a bar transform rather than an
#: indicator, and the other two are drawn levels, so no substitution would be a
#: reading of the same rule.
INDICATOR_SUBSTITUTIONS: dict[str, tuple[str, str]] = {
    "parabolic sar": ("supertrend", "a stop-and-reverse trend follower, as SAR is"),
    "parabolic": ("supertrend", "a stop-and-reverse trend follower, as SAR is"),
    "psar": ("supertrend", "a stop-and-reverse trend follower, as SAR is"),
    "half trend": ("supertrend", "a stop-and-reverse trend follower, as HalfTrend is"),
    "halftrend": ("supertrend", "a stop-and-reverse trend follower, as HalfTrend is"),
    "awesome oscillator": ("macd", "a difference of two means oscillating about zero, as AO is"),
    "awesome": ("macd", "a difference of two means oscillating about zero, as AO is"),
}

#: Parameters an indicator is given when a card names it without any. These are
#: the periods the pages themselves use most often, not the constructors'
#: defaults, so that a spec built from a bare mention reads like the corpus it
#: came from. Anything absent here is built with its own defaults.
ARCHETYPE_PARAMS: dict[str, dict[str, int | float]] = {
    "ema": {"period": 50},
    "sma": {"period": 50},
    "hma": {"period": 21},
    "wma": {"period": 21},
    "vwma": {"period": 20},
    "rsi": {"period": 14},
    "cci": {"period": 20},
    "willr": {"period": 14},
    "mfi": {"period": 14},
    "stoch": {"k_period": 14, "d_period": 3},
    "adx": {"period": 14},
    "atr": {"period": 14},
    "bbands": {"period": 20, "num_std": 2.0},
    "keltner": {"period": 20},
    "donchian": {"period": 20},
    "rvol": {"period": 20},
    "roc": {"period": 12},
    "swing": {"lookback": 5},
    "supertrend": {"period": 10, "multiplier": 3.0},
}

#: Oscillators with a fixed scale, and the levels a card that names no level
#: would be understood to mean: ``(oversold, midpoint, overbought)``.
OSCILLATOR_BANDS: dict[str, tuple[float, float, float]] = {
    "rsi": (30.0, 50.0, 70.0),
    "stoch": (20.0, 50.0, 80.0),
    "mfi": (20.0, 50.0, 80.0),
    "willr": (-80.0, -50.0, -20.0),
    "cci": (-100.0, 0.0, 100.0),
}

#: Indicators that publish a price-level line an entry can be invalidated
#: against, and the channel to read on each side. A trigger built on one of
#: these already names the level at which the idea is wrong, which is a better
#: invalidation than a generic swing because it is the card's own line.
TRIGGER_LINES: dict[str, tuple[str | None, str | None]] = {
    "ema": (None, None),
    "sma": (None, None),
    "hma": (None, None),
    "wma": (None, None),
    "vwma": (None, None),
    "supertrend": ("line", "line"),
    "keltner": ("lower", "upper"),
    "bbands": ("lower", "upper"),
    "donchian": ("lower", "upper"),
    "ichimoku": ("kijun", "kijun"),
    "vwap_session": (None, None),
}

#: Exit presets by the reward-to-risk ratio they express, for a card that
#: states a target ratio.
RATIO_PRESETS: tuple[tuple[float, str], ...] = (
    (1.0, "rr_1r"),
    (1.2, "rr_1_2r"),
    (1.25, "rr_1_25r"),
    (1.5, "rr_1_5r"),
    (2.0, "conservative_2r"),
    (2.5, "rr_2_5r"),
    (3.0, "rr_3r"),
    (4.0, "rr_4r"),
    (5.0, "rr_5r"),
    (6.0, "rr_6r"),
    (8.0, "rr_8r"),
    (10.0, "rr_10r"),
)

#: Exit preset a card that states no exit is given, by holding-period class.
#: A scalp cannot wait for structure and a swing should not be squared at a
#: fixed multiple of a minute's noise; these are the presets whose own
#: docstrings describe that holding period.
TYPE_EXITS: dict[str, str] = {
    "SCALP": "scalp_quick",
    "INTRADAY": "conservative_2r",
    "SWING": "structure_trail",
    "POSITION": "swing_partial_ladder",
}

_RATIO = re.compile(r"1\s*[:/]\s*(\d+(?:[.,]\d+)?)|\b(\d+(?:[.,]\d+)?)\s*:\s*1\b")


def snap_exit_ratio(ratio: float, known: Sequence[str]) -> tuple[str | None, str]:
    """Pick the preset closest to a stated reward-to-risk ratio.

    The strict pipeline refuses to round, and it is right that rounding changes
    the strategy: the channel-breakout ablation in CLAUDE.md measured a fixed
    2R target against an open trail as +0.099 against -0.017 expectancy on the
    same entry. Here the instruction is to keep the card, so the nearest preset
    is taken and the distance is stated — a reader can see a 1.3R page wearing
    a 1.25R preset and decide whether that is the same idea.

    Args:
        ratio: The reward-to-risk multiple the card states.
        known: Preset ids the library actually holds.

    Returns:
        ``(exit_ref, detail)``; ``exit_ref`` is ``None`` when no listed preset
        exists in ``known``.
    """
    available = [(value, name) for value, name in RATIO_PRESETS if name in known]
    if not available:
        return None, "the exit library holds no fixed-ratio preset"
    value, name = min(available, key=lambda pair: abs(pair[0] - ratio))
    if abs(value - ratio) < 1e-9:
        return name, f"the card's 1:{ratio:g} target is preset {name!r} exactly"
    return name, (
        f"the card targets 1:{ratio:g}; nearest preset is {name!r} at 1:{value:g}, "
        f"a difference of {abs(value - ratio):.2f}R that changes what is traded"
    )


def read_ratio(text: str) -> float | None:
    """Find a reward-to-risk ratio in a card's exit prose.

    Args:
        text: The exit or take-profit section.

    Returns:
        The reward multiple, or ``None`` when the text states none.
    """
    found = _RATIO.search(text.lower())
    if found is None:
        return None
    raw = found.group(1) or found.group(2)
    try:
        return float(raw.replace(",", "."))
    except ValueError:  # pragma: no cover - the pattern only matches numerals
        return None


def feature(indicator: str, channel: str | None = None) -> FeatureRef | None:
    """Build a feature reference for a bare indicator mention.

    Args:
        indicator: Registry name.
        channel: Output to read, for a multi-output indicator.

    Returns:
        The reference, or ``None`` when the registry cannot build the
        indicator with these parameters — which is the same authority
        :func:`~trading_system.strategies.validator.validate_spec` consults, so
        a reference this returns is one the validator will accept.
    """
    params = dict(ARCHETYPE_PARAMS.get(indicator, {}))
    try:
        built = build_indicator(indicator, params)
    except Exception:  # noqa: BLE001 - the registry raises several types
        return None
    outputs = tuple(built.outputs)
    if len(outputs) > 1:
        if channel is None or channel not in outputs:
            return None
        return FeatureRef(indicator=indicator, params=params, channel=channel)
    if channel is not None:
        return None
    return FeatureRef(indicator=indicator, params=params)


def _leaf(op: ConditionOp, left: Operand | None, right: Operand | None) -> LeafCondition:
    """One comparison, spelled the way the schema wants it."""
    return LeafCondition(op=op, left=left, right=right)


def archetype_conditions(
    indicators: Sequence[str], direction: Direction, mean_reverting: bool
) -> tuple[list[Condition], list[str]]:
    """Write the entry a card's indicator list implies, when the card wrote none.

    This is the one place that invents content rather than encoding it. It is
    reached only by cards that name indicators and never state a rule — 219 of
    883 in the delivered corpus — and what it produces is the reading a trader
    would give that indicator list, not a reading of the page.

    The construction is deliberately shallow: one trigger from the first
    indicator that can carry one, plus at most two filters. A longer inferred
    rule is not more faithful, only more specific about something nobody said.

    Args:
        indicators: Registry names the card declared, in the card's order.
        direction: Leg being built.
        mean_reverting: Whether the family fades extremes rather than following
            them. It flips the sense of every oscillator and of a band touch,
            which is the whole difference between two strategies drawn with the
            same two indicators.

    Returns:
        ``(conditions, notes)``. Empty conditions mean the indicator list
        carries nothing this can build on.
    """
    long = direction is Direction.LONG
    conditions: list[Condition] = []
    notes: list[str] = []

    for name in indicators:
        if conditions:
            break
        if name in ("ema", "sma", "hma", "wma", "vwma", "vwap_session"):
            ref = feature(name)
            if ref is None:
                continue
            conditions.append(_leaf(ConditionOp.GT if long else ConditionOp.LT, "price:close", ref))
            notes.append(f"trigger: price on the working side of {name}")
        elif name == "supertrend":
            ref = feature(name, "line")
            if ref is None:
                continue
            conditions.append(_leaf(ConditionOp.GT if long else ConditionOp.LT, "price:close", ref))
            notes.append("trigger: price on the working side of the Supertrend line")
        elif name == "macd":
            macd, signal = feature(name, "macd"), feature(name, "signal")
            if macd is None or signal is None:
                continue
            conditions.append(
                _leaf(ConditionOp.CROSS_ABOVE if long else ConditionOp.CROSS_BELOW, macd, signal)
            )
            notes.append("trigger: MACD crossing its signal line")
        elif name == "donchian":
            ref = feature(name, "upper" if long else "lower")
            if ref is None:
                continue
            shifted = FeatureRef(
                indicator=ref.indicator, params=ref.params, channel=ref.channel, shift=1
            )
            conditions.append(
                _leaf(
                    ConditionOp.CROSS_ABOVE if long else ConditionOp.CROSS_BELOW,
                    "price:close",
                    shifted,
                )
            )
            notes.append(
                "trigger: close crossing the previous bar's Donchian edge — the channel of the "
                "bar being tested contains that bar by construction, so an unshifted comparison "
                "can never fire"
            )
        elif name in ("bbands", "keltner"):
            if mean_reverting:
                ref = feature(name, "lower" if long else "upper")
                op = ConditionOp.LT if long else ConditionOp.GT
                notes.append(f"trigger: price outside the far {name} band, faded")
            else:
                ref = feature(name, "upper" if long else "lower")
                op = ConditionOp.CROSS_ABOVE if long else ConditionOp.CROSS_BELOW
                notes.append(f"trigger: price breaking the near {name} band")
            if ref is None:
                notes.pop()
                continue
            conditions.append(_leaf(op, "price:close", ref))
        elif name in OSCILLATOR_BANDS:
            low, mid, high = OSCILLATOR_BANDS[name]
            channel = "k" if name == "stoch" else None
            ref = feature(name, channel)
            if ref is None:
                continue
            if mean_reverting:
                conditions.append(
                    _leaf(ConditionOp.LT if long else ConditionOp.GT, ref, low if long else high)
                )
                band = "oversold" if long else "overbought"
                notes.append(f"trigger: {name} beyond its {band} band")
            else:
                conditions.append(
                    _leaf(ConditionOp.CROSS_ABOVE if long else ConditionOp.CROSS_BELOW, ref, mid)
                )
                notes.append(f"trigger: {name} crossing its midpoint")

    if not conditions:
        return [], []

    for name in indicators:
        if len(conditions) >= 3:
            break
        if name == "adx":
            ref = feature(name, "adx")
            if ref is not None:
                conditions.append(_leaf(ConditionOp.GT, ref, 25.0))
                notes.append("filter: ADX above 25, the level the corpus uses for 'trending'")
        elif name == "rvol":
            ref = feature(name)
            if ref is not None:
                conditions.append(_leaf(ConditionOp.GT, ref, 1.5))
                notes.append("filter: relative volume above 1.5")
    return conditions, notes


#: Candlestick phrases a page uses to the label the entry engine publishes.
#: "Pin bar" is the substitution worth naming: the corpus uses it for a long
#: wick rejecting a level, which this system labels ``HAMMER`` on the bullish
#: side and ``SHOOTING_STAR`` on the bearish one. That is the same bar, named
#: by two different traditions, and it is the only pattern phrase here that
#: does not map onto a label of its own name.
PATTERN_ARCHETYPES: dict[str, tuple[Pattern, Pattern]] = {
    "pin bar": (Pattern.HAMMER, Pattern.SHOOTING_STAR),
    "pinbar": (Pattern.HAMMER, Pattern.SHOOTING_STAR),
    "hammer": (Pattern.HAMMER, Pattern.SHOOTING_STAR),
    "shooting star": (Pattern.HAMMER, Pattern.SHOOTING_STAR),
    "engulfing": (Pattern.BULLISH_ENGULFING, Pattern.BEARISH_ENGULFING),
    "inside bar": (Pattern.INSIDE_BAR, Pattern.INSIDE_BAR),
    "outside bar": (Pattern.OUTSIDE_BAR, Pattern.OUTSIDE_BAR),
    "doji": (Pattern.DOJI, Pattern.DOJI),
    "morning star": (Pattern.MORNING_STAR, Pattern.EVENING_STAR),
    "evening star": (Pattern.MORNING_STAR, Pattern.EVENING_STAR),
    "star": (Pattern.MORNING_STAR, Pattern.EVENING_STAR),
}


def family_archetype(
    family: str, direction: Direction, prose: str
) -> tuple[list[Condition], str] | None:
    """Write an entry from what kind of strategy a card is, when it names no indicator.

    Reached by cards that carry no rule and no indicator list — a page about
    pin bars at support, or about the Asian range, draws its rule with nothing
    the registry has a name for. What the page does still say is what kind of
    bet it is, and each family has one canonical form: this returns that form.

    It is the furthest inference in this module and produces a spec that shares
    with its page only a family and, for candlestick pages, a bar shape.

    Args:
        family: The family's value, as classified.
        direction: The leg.
        prose: Title and description, searched for a named candle pattern.

    Returns:
        ``(conditions, reading)``, or ``None`` when the family has no canonical
        form to fall back on.
    """
    long = direction is Direction.LONG
    lowered = prose.lower()

    if family == "PATTERN":
        for phrase, (bullish, bearish) in PATTERN_ARCHETYPES.items():
            if phrase in lowered:
                label = bullish if long else bearish
                return (
                    [_leaf(ConditionOp.PATTERN_IS, None, LabelSet(labels=(label.value,)))],
                    f"the page is about the {phrase!r} candle; the leg fires on the "
                    f"{label.value} label with nothing else asked of the market",
                )
        return None

    canonical: dict[str, tuple[str, str | None, ConditionOp, ConditionOp, str]] = {
        "BREAKOUT": (
            "donchian",
            "upper" if long else "lower",
            ConditionOp.CROSS_ABOVE,
            ConditionOp.CROSS_BELOW,
            "a close through the previous bar's 20-bar channel edge",
        ),
        "PIVOT": (
            "pivots",
            "pivot",
            ConditionOp.CROSS_ABOVE,
            ConditionOp.CROSS_BELOW,
            "a close through the daily pivot",
        ),
        "VOLATILITY": (
            "keltner",
            "upper" if long else "lower",
            ConditionOp.CROSS_ABOVE,
            ConditionOp.CROSS_BELOW,
            "a close leaving the Keltner channel, which is what an expansion looks like",
        ),
        "MEAN_REVERSION": (
            "bbands",
            "lower" if long else "upper",
            ConditionOp.LT,
            ConditionOp.GT,
            "a close outside the far Bollinger band, faded",
        ),
        "MOMENTUM": ("macd", "macd", ConditionOp.CROSS_ABOVE, ConditionOp.CROSS_BELOW, ""),
        "TREND": (
            "ema",
            None,
            ConditionOp.GT,
            ConditionOp.LT,
            "price on the working side of EMA50",
        ),
    }
    found = canonical.get(family)
    if found is None:
        return None
    name, channel, long_op, short_op, note = found
    ref = feature(name, channel)
    if ref is None:
        return None
    op = long_op if long else short_op

    if name == "macd":
        signal = feature("macd", "signal")
        if signal is None:
            return None
        return (
            [_leaf(op, ref, signal)],
            "the page names no indicator; MACD crossing its signal is the family's canonical form",
        )
    if name == "donchian":
        ref = FeatureRef(indicator=ref.indicator, params=ref.params, channel=ref.channel, shift=1)
    return (
        [_leaf(op, "price:close", ref)],
        f"the page names no indicator; the {family} family's canonical form is {note}",
    )


def invalidation_level(
    trigger_indicators: Sequence[str], direction: Direction
) -> tuple[Operand, InferenceCode, str]:
    """Choose the price level at which a leg is wrong, when the card names none.

    Preference is not arbitrary. A trigger built on a line already names the
    level the idea depends on, and using that line is the reading closest to
    what the card traded. Only when no such line exists does this fall back to
    structure, and the fallback is loud because it is the single largest
    substitution this module makes: 417 of 883 cards state their stop as a pip
    distance, which the schema has no operand for.

    Args:
        trigger_indicators: Registry names the leg's trigger reads.
        direction: The leg.

    Returns:
        ``(operand, code, detail)``.
    """
    long = direction is Direction.LONG
    for name in trigger_indicators:
        if name not in TRIGGER_LINES:
            continue
        low_channel, high_channel = TRIGGER_LINES[name]
        channel = low_channel if long else high_channel
        ref = feature(name, channel)
        if ref is None:
            continue
        return (
            ref,
            InferenceCode.INVALIDATION_FROM_TRIGGER_LINE,
            f"the card states no level; the trigger's own {name} line is used, so the leg is "
            "given up exactly where the condition that opened it stops holding",
        )

    channel = "swing_low" if long else "swing_high"
    ref = feature("swing", channel)
    assert ref is not None, "the swing indicator is in the registry"
    return (
        ref,
        InferenceCode.INVALIDATION_FROM_SWING,
        "the card states no level a stop could be placed at — usually a pip distance, which is "
        f"not an operand — so the last confirmed {channel.replace('_', ' ')} is used instead. "
        "This is a substitution of content, not of notation: where the page stopped out and "
        "where this spec stops out are different places",
    )


def substitute_indicator(phrase: str) -> tuple[str, str] | None:
    """Map an unimplemented indicator name onto an implemented one.

    Args:
        phrase: The phrase found in the card, lower-cased.

    Returns:
        ``(registry_name, reason)``, or ``None`` when no substitution is
        defensible and the clause naming it has to be dropped.
    """
    for needle, (name, reason) in INDICATOR_SUBSTITUTIONS.items():
        if needle in phrase:
            return name, reason
    return None


def timeframe_for(category: str, stated: Timeframe | None) -> tuple[Timeframe, Inference | None]:
    """Settle on the bar size a spec is written for.

    Args:
        category: The page's site section.
        stated: What the card's own timeframe line resolved to, if anything.

    Returns:
        ``(timeframe, inference)``; ``inference`` is ``None`` when the card
        stated one and it was taken as written.
    """
    if stated is not None:
        return stated, None
    chosen = CATEGORY_TIMEFRAME.get(category, FALLBACK_TIMEFRAME)
    return chosen, Inference(
        InferenceCode.TIMEFRAME_FROM_CATEGORY,
        f"the card names no timeframe; {chosen.value} assumed from page category {category!r}",
    )


def merge(
    values: Mapping[str, int | float], overrides: Mapping[str, int | float]
) -> dict[str, int | float]:
    """Parameters with overrides applied, for building a reference from a mention."""
    merged = dict(values)
    merged.update(overrides)
    return merged
