"""One scraped card in, one strategy spec out — or one stated reason it is not.

The conversion is total in the sense that matters: every card ends as either a
spec that :func:`~trading_system.strategies.validator.validate_spec` accepts or
a list of refusals naming what stopped it. Nothing is filled in to keep a card
alive. Where the card is silent on something the schema demands, that silence
is a refusal (no timeframe, no invalidation level, no exit), and where the card
says something the schema cannot express, that is a refusal too — the three
limitations CLAUDE.md already records (cross-timeframe rules, arithmetic over
bar fields, market regimes) plus the ones this corpus adds.

Three readings are *not* refusals, because they are decisions about encoding
rather than about what is traded, and each is recorded on the conversion as an
assumption so it is visible on the spec it produced:

* the holding-period class, derived from the stated timeframe and the page's
  own category;
* ``risk_profile.stop_reference``, which every spec must carry and no card
  states — :data:`DEFAULT_STOP_REFERENCE`, the schema's own default, and the
  axis a search space sweeps first;
* ``base_quality``, which ranks setups against each other and which a card that
  ranks nothing cannot supply.

Everything else comes from the card or the card does not convert.
"""

import re
from collections.abc import Collection
from dataclasses import dataclass
from enum import StrEnum

from pydantic import ValidationError as PydanticValidationError

from trading_system.core.instruments import InstrumentClass
from trading_system.core.types import Timeframe
from trading_system.strategies.ingest.card import EXIT_HEADERS, STOP_HEADERS, ScrapedCard
from trading_system.strategies.ingest.lexicon import (
    BAR_ARITHMETIC_MARKERS,
    CHART_TIMEFRAME_PATTERN,
    CROSS_TIMEFRAME_MARKERS,
    OFF_REGISTRY_INDICATORS,
    REGIME_MARKERS,
    VISUAL_MARKERS,
)
from trading_system.strategies.ingest.overrides import CardOverride
from trading_system.strategies.ingest.rules import ClauseResult, is_noise, read_clause
from trading_system.strategies.ingest.terms import (
    DeclaredIndicators,
    declared_indicators,
    read_operand,
)
from trading_system.strategies.ingest.text import clauses, contains_any, normalise
from trading_system.strategies.schema import (
    AllOf,
    AtrStop,
    Condition,
    Direction,
    EntryOrderSpec,
    EntrySpec,
    FeatureRef,
    InstrumentScope,
    Invalidation,
    MarketOrder,
    RiskProfileSpec,
    StrategySpec,
    StrategyType,
    TimeframeSpec,
)
from trading_system.strategies.validator import Severity, validate_spec

#: Stop distance every converted spec declares. No card states one — they name
#: a level, which becomes the invalidation — but the Risk Engine takes the
#: farthest of the invalidation, this, and the broker minimum, so something has
#: to be here. The schema's own defaults are used rather than a number invented
#: for this pipeline, and the multiple is the first axis a generated search
#: space sweeps.
DEFAULT_STOP_REFERENCE = AtrStop(period=14, multiple=1.5)

#: Quality every converted spec starts at. A card that does not rank its own
#: setups cannot supply this, and a mid-scale constant with no modifiers says
#: exactly that: every setup this strategy finds is worth the same.
DEFAULT_BASE_QUALITY = 0.5

#: Fractal lookback used when a card says "the previous swing high/low" without
#: saying how a swing is confirmed — which no card does. It is the same value
#: the bundled exit presets use, so a converted spec and the exit it pairs with
#: read structure the same way.
DEFAULT_SWING_LOOKBACK = 5

#: Version every converted spec carries: below 1.0.0, because nothing about a
#: converted card has been measured yet.
CONVERTED_VERSION = "0.1.0"


class RefusalCode(StrEnum):
    """Why a card did not convert.

    Grouped by :data:`REFUSAL_GROUPS` into what the group is evidence *of*: a
    limitation of the schema, content this system does not implement, a card
    that never stated what it needs to, or a sentence the grammar could not
    read.
    """

    NO_RULES = "no_rules"
    CROSS_TIMEFRAME = "cross_timeframe"
    BAR_ARITHMETIC = "bar_arithmetic"
    REGIME = "regime"
    OFF_REGISTRY_INDICATOR = "off_registry_indicator"
    VISUAL_RULE = "visual_rule"
    UNREADABLE_CLAUSE = "unreadable_clause"
    UNREAD_SECTION = "unread_section"
    NO_TIMEFRAME = "no_timeframe"
    NO_INVALIDATION = "no_invalidation"
    EXIT_UNSTATED = "exit_unstated"
    EXIT_NOT_IN_LIBRARY = "exit_not_in_library"
    SPEC_REJECTED = "spec_rejected"


#: Refusal codes grouped by what they say about the corpus, in report order.
REFUSAL_GROUPS: dict[str, tuple[RefusalCode, ...]] = {
    "schema_cannot_express": (
        RefusalCode.CROSS_TIMEFRAME,
        RefusalCode.BAR_ARITHMETIC,
        RefusalCode.REGIME,
    ),
    "not_implemented_here": (
        RefusalCode.OFF_REGISTRY_INDICATOR,
        RefusalCode.VISUAL_RULE,
    ),
    "card_never_said": (
        RefusalCode.NO_RULES,
        RefusalCode.NO_TIMEFRAME,
        RefusalCode.NO_INVALIDATION,
        RefusalCode.EXIT_UNSTATED,
        RefusalCode.EXIT_NOT_IN_LIBRARY,
    ),
    "unreadable": (
        RefusalCode.UNREAD_SECTION,
        RefusalCode.UNREADABLE_CLAUSE,
        RefusalCode.SPEC_REJECTED,
    ),
}

#: Order a card's refusals are ranked in when one has to be called the reason.
#: A card blocked by several things is reported under the first of them, so a
#: rule needing another timeframe is never filed as an unreadable sentence.
REFUSAL_PRECEDENCE: tuple[RefusalCode, ...] = (
    RefusalCode.NO_RULES,
    RefusalCode.CROSS_TIMEFRAME,
    RefusalCode.REGIME,
    RefusalCode.BAR_ARITHMETIC,
    RefusalCode.OFF_REGISTRY_INDICATOR,
    RefusalCode.VISUAL_RULE,
    RefusalCode.UNREAD_SECTION,
    RefusalCode.UNREADABLE_CLAUSE,
    RefusalCode.NO_TIMEFRAME,
    RefusalCode.NO_INVALIDATION,
    RefusalCode.EXIT_NOT_IN_LIBRARY,
    RefusalCode.EXIT_UNSTATED,
    RefusalCode.SPEC_REJECTED,
)


@dataclass(frozen=True)
class Refusal:
    """One reason a card did not convert.

    Attributes:
        code: What kind of obstacle this is.
        detail: The card's own words, or the grammar's complaint about them.
    """

    code: RefusalCode
    detail: str


@dataclass(frozen=True)
class Conversion:
    """The outcome of putting one card through the converter.

    Attributes:
        card_id: The scraper's slug for the card.
        source_url: Where the card came from.
        title: The card's title, for reports.
        spec: The strategy spec, or ``None`` when the card was refused.
        refusals: Every obstacle found, not only the first.
        assumptions: Readings the card did not spell out, on a spec that
            converted.
    """

    card_id: str
    source_url: str
    title: str
    spec: StrategySpec | None
    refusals: tuple[Refusal, ...]
    assumptions: tuple[str, ...]

    @property
    def converted(self) -> bool:
        """Whether a spec came out."""
        return self.spec is not None

    @property
    def primary(self) -> RefusalCode | None:
        """The refusal this card is reported under, by :data:`REFUSAL_PRECEDENCE`."""
        codes = {refusal.code for refusal in self.refusals}
        return next((code for code in REFUSAL_PRECEDENCE if code in codes), None)


_TIMEFRAME_PHRASES: tuple[tuple[str, Timeframe], ...] = (
    ("m1", Timeframe.M1),
    ("1 min", Timeframe.M1),
    ("1min", Timeframe.M1),
    ("1 minute", Timeframe.M1),
    ("m5", Timeframe.M5),
    ("5 min", Timeframe.M5),
    ("5min", Timeframe.M5),
    ("5 minute", Timeframe.M5),
    ("m15", Timeframe.M15),
    ("15 min", Timeframe.M15),
    ("15min", Timeframe.M15),
    ("15 minute", Timeframe.M15),
    ("h1", Timeframe.H1),
    ("1 h", Timeframe.H1),
    ("1h", Timeframe.H1),
    ("60 min", Timeframe.H1),
    ("1 hour", Timeframe.H1),
    ("hourly", Timeframe.H1),
    ("h4", Timeframe.H4),
    ("4 h", Timeframe.H4),
    ("4h", Timeframe.H4),
    ("240", Timeframe.H4),
    ("4 hour", Timeframe.H4),
    ("d1", Timeframe.D1),
    ("daily", Timeframe.D1),
    ("1 day", Timeframe.D1),
)

#: Timeframes this system has no bar size for. A card written for M30 or W1 is
#: refused by name rather than rounded onto a neighbour.
_UNSUPPORTED_TIMEFRAMES: tuple[str, ...] = (
    "m30",
    "30 min",
    "30min",
    "30 minute",
    "m10",
    "10 min",
    "10min",
    "m2",
    "2 min",
    "3 min",
    "weekly",
    "w1",
    "monthly",
    "mn1",
)

_ORDINALS = {tf: index for index, tf in enumerate(Timeframe)}

_RR = re.compile(r"\b1\s*[:/]\s*(\d+(?:[.,]\d+)?)")
_PIPS = re.compile(r"\b(\d+(?:[.,]\d+)?)\s*pips?\b")
_FX_PAIR = re.compile(
    r"\b(AUD|CAD|CHF|EUR|GBP|JPY|NZD|USD)\s*[/\-]?\s*(AUD|CAD|CHF|EUR|GBP|JPY|NZD|USD)\b"
)


#: Bare minute counts a timeframe line uses when it drops the unit, as in
#: "15 or higher". Only the ones this system has a bar size for.
_BARE_MINUTES: dict[int, Timeframe] = {
    1: Timeframe.M1,
    5: Timeframe.M5,
    15: Timeframe.M15,
    60: Timeframe.H1,
    240: Timeframe.H4,
}


def _bare_minutes(text: str) -> list[Timeframe]:
    """Read a timeframe line that gives a number and no unit.

    Args:
        text: The normalised, lower-cased timeframe line.

    Returns:
        Every timeframe named this way, in reading order. A number this system
        has no bar size for contributes nothing, so "30 or higher" stays
        unreadable rather than being rounded onto M15 or H1.
    """
    found: list[Timeframe] = []
    for raw in re.findall(r"\b(\d{1,3})\b", text):
        timeframe = _BARE_MINUTES.get(int(raw))
        if timeframe is not None and timeframe not in found:
            found.append(timeframe)
    return found


def read_timeframe(card: ScrapedCard) -> tuple[Timeframe | None, str]:
    """Read the timeframe a card is written for.

    A card stating a floor ("15 min or higher") is read as that floor: it is
    the one bar size the page actually names. A card naming two timeframes is
    refused rather than resolved, because the second one is either a
    higher-timeframe filter this system cannot honour or a genuine ambiguity.

    Args:
        card: The card.

    Returns:
        ``(timeframe, detail)``. ``timeframe`` is ``None`` when the card names
        none, several, or one this system has no bar size for; ``detail`` then
        says which.
    """
    raw = card.timeframe_raw
    if not raw or not raw.strip():
        return None, "the card states no timeframe"
    text = normalise(raw).lower()
    unsupported = contains_any(text, _UNSUPPORTED_TIMEFRAMES)
    found: list[Timeframe] = []
    for phrase, timeframe in _TIMEFRAME_PHRASES:
        if re.search(rf"(?<![\w.]){re.escape(phrase)}", text) and timeframe not in found:
            found.append(timeframe)
    if unsupported and not found:
        return None, f"written for {unsupported[0]!r}, which this system has no bar size for"
    if not found:
        found = _bare_minutes(text)
    if not found:
        return None, f"no timeframe recognised in {raw.strip()!r}"
    if len(found) > 1:
        names = [tf.value for tf in found]
        return None, f"names several timeframes {names} in {raw.strip()!r}"
    return found[0], found[0].value


def read_instruments(card: ScrapedCard) -> tuple[InstrumentScope, tuple[str, ...]]:
    """Read which instruments a card is written for.

    Args:
        card: The card.

    Returns:
        The scope and any assumptions made. A card naming specific pairs gets
        them as an allow-list; one saying "any" — or saying nothing — gets the
        FX class with no allow-list, which is what "any" means for a corpus of
        forex pages.
    """
    raw = card.instruments_raw or ""
    pairs = sorted({f"{base}{quote}" for base, quote in _FX_PAIR.findall(raw.upper())})
    if pairs:
        return InstrumentScope(allowed_classes=[InstrumentClass.FX], allowed_symbols=pairs), ()
    return (
        InstrumentScope(allowed_classes=[InstrumentClass.FX]),
        ("the card names no instrument; scoped to FX with no symbol allow-list",),
    )


def read_type(card: ScrapedCard, timeframe: Timeframe) -> tuple[StrategyType, str]:
    """Classify a card's holding period.

    Args:
        card: The card, whose site category is the only stated evidence.
        timeframe: The bar size the card is written for.

    Returns:
        The type and the assumption recording how it was chosen.
    """
    if "scalping" in card.category:
        chosen = StrategyType.SCALP
    elif timeframe is Timeframe.D1:
        chosen = StrategyType.POSITION
    elif timeframe is Timeframe.H4:
        chosen = StrategyType.SWING
    else:
        chosen = StrategyType.INTRADAY
    return chosen, (
        f"holding-period class {chosen.value} derived from timeframe {timeframe.value} "
        f"and page category {card.category!r}; the card does not state one"
    )


def detect_blockers(text: str) -> list[Refusal]:
    """Find phrases proving a rule needs something unavailable.

    Run over the card's own rule text before any parsing, so that a card
    blocked by a limitation is reported under that limitation rather than as a
    sentence the grammar happened not to read.

    Args:
        text: The card's rule sections, joined.

    Returns:
        One refusal per kind of obstacle found, naming the phrase that proved it.
    """
    checks = (
        (RefusalCode.CROSS_TIMEFRAME, CROSS_TIMEFRAME_MARKERS),
        (RefusalCode.BAR_ARITHMETIC, BAR_ARITHMETIC_MARKERS),
        (RefusalCode.REGIME, REGIME_MARKERS),
        (RefusalCode.OFF_REGISTRY_INDICATOR, OFF_REGISTRY_INDICATORS),
        (RefusalCode.VISUAL_RULE, VISUAL_MARKERS),
    )
    refusals: list[Refusal] = []
    for code, markers in checks:
        found = contains_any(text, markers)
        if found:
            refusals.append(Refusal(code, f"the rules say {', '.join(repr(f) for f in found[:3])}"))
    chart = re.search(CHART_TIMEFRAME_PATTERN, normalise(text).lower())
    if chart is not None and not any(r.code is RefusalCode.CROSS_TIMEFRAME for r in refusals):
        refusals.append(Refusal(RefusalCode.CROSS_TIMEFRAME, f"the rules say {chart.group(0)!r}"))
    return refusals


def _orient(text: str, direction: Direction) -> tuple[str, tuple[str, ...]]:
    """Resolve the "above/below the swing high/low" shorthand onto one side.

    The pages write both legs into one sentence. For a long entry the stop
    sits below the swing low, for a short above the swing high; picking the
    side that matches the leg is the only reading of a paired phrase, but it is
    a reading, so it is recorded.

    Args:
        text: The stop sentence.
        direction: The leg being built.

    Returns:
        ``(oriented_text, assumptions)``.
    """
    pairs = (
        (r"above\s*/\s*below", r"below\s*/\s*above", "above", "below"),
        (r"high\s*/\s*low", r"low\s*/\s*high", "high", "low"),
        (r"highs\s*/\s*lows", r"lows\s*/\s*highs", "highs", "lows"),
        (r"upper\s*/\s*lower", r"lower\s*/\s*upper", "upper", "lower"),
    )
    notes: list[str] = []
    oriented = text
    for first, second, short_word, long_word in pairs:
        keep = long_word if direction is Direction.LONG else short_word
        for pattern in (first, second):
            if re.search(pattern, oriented, flags=re.IGNORECASE):
                oriented = re.sub(pattern, keep, oriented, flags=re.IGNORECASE)
                notes.append(
                    f"the stop names both sides; the {direction.value.lower()} leg took {keep!r}"
                )
    return oriented, tuple(dict.fromkeys(notes))


#: Indicators whose protective level is a band, so that "the opposite band" and
#: a bare mention in a stop sentence both mean the side away from the trade.
_BANDED = {
    "bbands": ("lower", "upper"),
    "keltner": ("lower", "upper"),
    "donchian": ("lower", "upper"),
}

_STOP_PREPOSITIONS = ("above", "below", "beyond", "under", "on the", "at the", "on", "at")


def read_invalidation(
    card: ScrapedCard, direction: Direction, declared: DeclaredIndicators
) -> tuple[Invalidation | None, tuple[str, ...], str]:
    """Read the price level at which a card gives the trade up.

    Args:
        card: The card.
        direction: The leg being built.
        declared: The card's declared indicators.

    Returns:
        ``(invalidation, assumptions, detail)``. ``invalidation`` is ``None``
        when no level was found, and ``detail`` then says what the card offered
        instead — a pip distance being the common case, and one this schema has
        no operand for.
    """
    text = card.section_text(STOP_HEADERS)
    if not text:
        return None, (), "the card states no stop"

    stop_clauses = [clause for clause in clauses(text) if "stop" in clause.lower()]
    if not stop_clauses:
        return None, (), "no sentence in the exit section mentions a stop"

    for clause in stop_clauses:
        oriented, notes = _orient(clause, direction)
        level, level_notes = _level_from_clause(oriented, direction, declared)
        if level is not None:
            return Invalidation(price_level=level), notes + level_notes, "read from the card"

    joined = " ".join(stop_clauses)
    pips = _PIPS.search(joined)
    if pips:
        return (
            None,
            (),
            f"the stop is stated only as a distance ({pips.group(0)}), and an invalidation "
            "is an absolute level: the schema has no operand for entry price plus an offset",
        )
    return None, (), f"no level in {joined.strip()[:120]!r}"


def _level_from_clause(
    clause: str, direction: Direction, declared: DeclaredIndicators
) -> tuple[FeatureRef | str | None, tuple[str, ...]]:
    """Read the level a stop sentence names, if it names one."""
    lowered = clause.lower()
    if re.search(r"\bswing\b|\bfractal\b", lowered):
        channel = "swing_low" if direction is Direction.LONG else "swing_high"
        return (
            FeatureRef(
                indicator="swing", params={"lookback": DEFAULT_SWING_LOOKBACK}, channel=channel
            ),
            (
                f"'swing' read as a {DEFAULT_SWING_LOOKBACK}-bar fractal; the card does not say "
                "how a swing is confirmed",
            ),
        )

    for preposition in _STOP_PREPOSITIONS:
        position = lowered.find(f" {preposition} ")
        if position < 0:
            continue
        tail = clause[position + len(preposition) + 2 :]
        result = read_operand(tail, declared)
        if result.ok and isinstance(result.operand, FeatureRef):
            ref = result.operand
            if ref.channel is None and ref.indicator in _BANDED:
                low, high = _BANDED[ref.indicator]
                side = low if direction is Direction.LONG else high
                ref = FeatureRef(indicator=ref.indicator, params=ref.params, channel=side)
                return ref, result.assumptions + (
                    f"a bare {ref.indicator!r} in a stop sentence read as its {side!r} band",
                )
            return ref, result.assumptions
        if result.ok and isinstance(result.operand, str):
            field = "low" if direction is Direction.LONG else "high"
            return f"price:{field}", (
                f"the stop sits at the signal bar's {field}, as the card names the bar itself",
            )
    return None, ()


def read_exit_ref(card: ScrapedCard) -> tuple[str | None, tuple[str, ...], RefusalCode, str]:
    """Pick the exit preset a card's own exit prose names.

    Nothing is rounded onto a neighbour: a card asking for a 1.3R target is
    refused by name rather than given the 2R preset, because the exit is part
    of the strategy and the difference between a fixed target and a trail has
    already been measured to matter (see the channel-breakout ablation in
    CLAUDE.md).

    Args:
        card: The card.

    Returns:
        ``(exit_ref, assumptions, code, detail)``. ``exit_ref`` is ``None``
        when the card's exit has no preset, and ``code``/``detail`` then say
        why.
    """
    text = card.section_text(EXIT_HEADERS)
    if not text:
        return None, (), RefusalCode.EXIT_UNSTATED, "the card states no exit"
    lowered = normalise(text).lower()

    trailing = "trail" in lowered
    if trailing and ("atr" in lowered or "chandelier" in lowered):
        return (
            "atr_trail_aggressive",
            ("the card asks for an ATR trail",),
            RefusalCode.EXIT_UNSTATED,
            "",
        )
    if trailing and ("swing" in lowered or "structure" in lowered):
        return (
            "structure_trail",
            ("the card asks for a structure trail",),
            RefusalCode.EXIT_UNSTATED,
            "",
        )

    match = _RR.search(lowered)
    if match is not None:
        ratio = float(match.group(1).replace(",", "."))
        if 1.75 <= ratio <= 2.5:
            return (
                "conservative_2r",
                (f"the card's target ratio 1:{ratio:g} taken as the 2R preset",),
                RefusalCode.EXIT_UNSTATED,
                "",
            )
        return (
            None,
            (),
            RefusalCode.EXIT_NOT_IN_LIBRARY,
            f"the card targets 1:{ratio:g}, and the exit library has no preset at that ratio",
        )
    if (
        "end of the session" in lowered
        or "end of day" in lowered
        or "close of the session" in lowered
    ):
        return (
            "session_close",
            ("the card exits at the session close",),
            RefusalCode.EXIT_UNSTATED,
            "",
        )
    return (
        None,
        (),
        RefusalCode.EXIT_UNSTATED,
        f"no exit this library has a preset for in {text.strip()[:120]!r}",
    )


def _read_side(
    text: str, declared: DeclaredIndicators, title: str = ""
) -> tuple[list[Condition], list[str], tuple[str, ...]]:
    """Read every clause of one side's rules.

    Args:
        text: The rule section for that side.
        declared: The card's declared indicators.
        title: The card's title, so a caption repeating it is dropped.

    Returns:
        ``(conditions, problems, assumptions)``. A non-empty ``problems`` means
        the side is refused: a rule section is read whole or not at all.
    """
    conditions: list[Condition] = []
    problems: list[str] = []
    assumptions: list[str] = []
    for clause in clauses(text):
        result: ClauseResult = read_clause(clause, declared, title)
        if result.noise:
            continue
        if result.problem is not None:
            problems.append(result.problem)
            continue
        assert result.condition is not None
        conditions.append(result.condition)
        assumptions.extend(result.assumptions)
    if not conditions and not problems:
        problems.append("the side's section carries no rule at all")
    return conditions, problems, tuple(dict.fromkeys(assumptions))


def _slug(card_id: str) -> str:
    """Turn a scraper slug into the lower-kebab-case id the schema demands."""
    slug = re.sub(r"[^a-z0-9]+", "-", card_id.lower()).strip("-")
    return slug or "card"


def convert_card(
    card: ScrapedCard,
    known_exit_ids: Collection[str] | None = None,
    override: CardOverride | None = None,
) -> Conversion:
    """Convert one card, or state why it does not convert.

    Args:
        card: The card to convert.
        known_exit_ids: Exit preset ids ``exit_ref`` is checked against, passed
            through to :func:`~trading_system.strategies.validator.validate_spec`.
        override: A reviewer's answers to what this card leaves unsaid. It can
            only supply missing values and dismiss sections it has read — never
            change a rule — so a converted trigger still says what the page said.

    Returns:
        The conversion outcome. Every obstacle found is reported, not only the
        first, so a report can say what a card would still need after the
        obvious blocker is removed.
    """
    refusals: list[Refusal] = []
    assumptions: list[str] = []
    declared = declared_indicators(card.indicators_raw)

    sides = {
        direction: text
        for direction in Direction
        if (text := card.rules_for(direction)) is not None
    }
    if not sides:
        return Conversion(
            card.strategy_id,
            card.source_url,
            card.title,
            None,
            (Refusal(RefusalCode.NO_RULES, "no section names a side to trade"),),
            (),
        )

    dismissed = {
        header.strip().lower() for header in (override.dismiss_sections if override else ())
    }
    unsided = tuple(
        (header, text)
        for header, text in card.unsided_rule_sections()
        if header.strip().lower() not in dismissed
    )
    # Blockers are read across every rule-bearing section, not only the sides:
    # a card whose Buy section is clean and whose Entry section says "on the
    # 60 min chart" is a cross-timeframe strategy, and converting the clean half
    # of it would produce a spec that trades something else.
    refusals.extend(detect_blockers("\n".join([*sides.values(), *(text for _, text in unsided)])))
    for header, text in unsided:
        carried = [clause for clause in clauses(text) if not is_noise(clause, card.title)]
        if carried:
            refusals.append(
                Refusal(
                    RefusalCode.UNREAD_SECTION,
                    f"section {header!r} carries {len(carried)} rule(s) that name no side, "
                    f"starting {carried[0][:80]!r}",
                )
            )

    timeframe, tf_detail = read_timeframe(card)
    if override is not None and override.timeframe is not None:
        timeframe = override.timeframe
    if timeframe is None:
        refusals.append(Refusal(RefusalCode.NO_TIMEFRAME, tf_detail))

    exit_ref, exit_notes, exit_code, exit_detail = read_exit_ref(card)
    if override is not None and override.exit_ref is not None:
        exit_ref = override.exit_ref
    elif exit_ref is None:
        refusals.append(Refusal(exit_code, exit_detail))
    if exit_ref is not None:
        assumptions.extend(exit_notes)

    entries: list[EntrySpec] = []
    for direction, text in sides.items():
        conditions, problems, side_notes = _read_side(text, declared, card.title)
        assumptions.extend(side_notes)
        for problem in problems[:3]:
            refusals.append(Refusal(RefusalCode.UNREADABLE_CLAUSE, f"{direction.value}: {problem}"))
        invalidation, stop_notes, stop_detail = read_invalidation(card, direction, declared)
        supplied = (override.invalidation if override else {}).get(direction)
        if supplied is not None:
            invalidation = Invalidation(price_level=supplied)
            stop_notes = ()
        if invalidation is None:
            refusals.append(Refusal(RefusalCode.NO_INVALIDATION, stop_detail))
        else:
            assumptions.extend(stop_notes)
        if problems or invalidation is None or not conditions:
            continue
        trigger = conditions[0] if len(conditions) == 1 else AllOf(conditions=conditions)
        entries.append(
            EntrySpec(
                direction=direction,
                trigger=trigger,
                invalidation=invalidation,
                entry_order=EntryOrderSpec(order=MarketOrder(), expire_after_bars=1),
            )
        )

    if refusals or not entries:
        return Conversion(
            card.strategy_id,
            card.source_url,
            card.title,
            None,
            tuple(refusals),
            (),
        )

    assert timeframe is not None and exit_ref is not None
    scope, scope_notes = read_instruments(card)
    if override is not None and override.instruments is not None:
        scope, scope_notes = override.instruments, ()
    assumptions.extend(scope_notes)
    strategy_type, type_note = read_type(card, timeframe)
    assumptions.append(type_note)
    assumptions.append(
        f"stop_reference is the pipeline default {DEFAULT_STOP_REFERENCE.kind} "
        f"({DEFAULT_STOP_REFERENCE.period}, {DEFAULT_STOP_REFERENCE.multiple}); no card states one"
    )
    assumptions.append(
        f"base_quality is the flat default {DEFAULT_BASE_QUALITY} with no modifiers; "
        "the card ranks no setup above another"
    )
    assumptions.append(
        "every rule of a side became a trigger condition; cards do not separate a trigger "
        "from a confirmation window"
    )
    if override is not None:
        assumptions.extend(override.provenance())

    try:
        spec = StrategySpec(
            id=_slug(card.strategy_id),
            version=CONVERTED_VERSION,
            type=strategy_type,
            timeframes=TimeframeSpec(signal_tf=timeframe, entry_tf=timeframe),
            instruments=scope,
            entries=entries,
            exit_ref=exit_ref,
            risk_profile=RiskProfileSpec(
                base_quality=DEFAULT_BASE_QUALITY,
                stop_reference=DEFAULT_STOP_REFERENCE,
            ),
        )
    except PydanticValidationError as error:
        return Conversion(
            card.strategy_id,
            card.source_url,
            card.title,
            None,
            (Refusal(RefusalCode.SPEC_REJECTED, str(error).replace("\n", " ")[:300]),),
            (),
        )

    issues = [
        issue
        for issue in validate_spec(spec, known_exit_ids=known_exit_ids)
        if issue.severity is Severity.ERROR
    ]
    if issues:
        return Conversion(
            card.strategy_id,
            card.source_url,
            card.title,
            None,
            tuple(
                Refusal(RefusalCode.SPEC_REJECTED, f"{issue.code}: {issue.message}")
                for issue in issues
            ),
            (),
        )
    return Conversion(
        card.strategy_id,
        card.source_url,
        card.title,
        spec,
        (),
        tuple(dict.fromkeys(assumptions)),
    )
