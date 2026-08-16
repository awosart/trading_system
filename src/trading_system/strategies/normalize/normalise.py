"""One scraped card in, one strategy spec out — filling in what the page left unsaid.

This is the lenient counterpart of
:func:`trading_system.strategies.ingest.convert.convert_card`. That one refuses
a card the moment the page is silent or unreadable, and converts one card in
883. This one keeps the card and records what it had to decide, because the
question being asked of the corpus changed: not "what does this page say" but
"what is the most defensible strategy this page implies, given the data we can
actually test it on".

Four sources of rule text are tried in order, and which one produced a leg is
what :class:`Fidelity` grades:

1. a section headed with the side's name — the strict pipeline's only source;
2. an unsided rule section split at its own inline "long"/"short" markers;
3. the card's description and indicator prose, which on this corpus regularly
   carries the rule the page never filed under a header;
4. an archetype built from the declared indicators, which is not a reading of
   the page at all and is labelled so.

A card that gets to none of them is refused, and the refusal says which of the
four it failed.
"""

import re
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from pydantic import ValidationError as PydanticValidationError

from trading_system.core.instruments import InstrumentClass
from trading_system.core.types import Timeframe
from trading_system.strategies.ingest.card import EXIT_HEADERS, STOP_HEADERS, ScrapedCard
from trading_system.strategies.ingest.convert import (
    DEFAULT_BASE_QUALITY,
    DEFAULT_STOP_REFERENCE,
    read_invalidation,
    read_timeframe,
)
from trading_system.strategies.ingest.lexicon import (
    BAR_ARITHMETIC_MARKERS,
    CROSS_TIMEFRAME_MARKERS,
    OFF_REGISTRY_INDICATORS,
    REGIME_MARKERS,
    VISUAL_MARKERS,
)
from trading_system.strategies.ingest.rules import read_clause
from trading_system.strategies.ingest.terms import DeclaredIndicators, declared_indicators
from trading_system.strategies.ingest.text import clauses, contains_any, normalise
from trading_system.strategies.normalize.classify import (
    UNPLACED_REASON,
    Family,
    classify_family,
    classify_type,
)
from trading_system.strategies.normalize.coverage import DEFAULT_MAX_COST_RATIO, MarketCoverage
from trading_system.strategies.normalize.inference import (
    TYPE_EXITS,
    Inference,
    InferenceCode,
    archetype_conditions,
    family_archetype,
    invalidation_level,
    read_ratio,
    snap_exit_ratio,
    substitute_indicator,
    timeframe_for,
)
from trading_system.strategies.normalize.salvage import salvage_clause, sniff_indicators
from trading_system.strategies.schema import (
    AllOf,
    Condition,
    Direction,
    EntryOrderSpec,
    EntrySpec,
    FeatureRef,
    InstrumentScope,
    Invalidation,
    LeafCondition,
    MarketOrder,
    Not,
    RiskProfileSpec,
    StrategySpec,
    StrategyType,
    TimeframeSpec,
)
from trading_system.strategies.validator import Severity, validate_spec

#: Version a normalised card carries. Below 1.0.0 for the same reason the
#: strict pipeline's is: nothing here has been measured.
NORMALISED_VERSION = "0.1.0"

#: Families whose entries fade an extreme instead of following it. Passed into
#: the archetype builder, where it decides the sense of every oscillator.
_MEAN_REVERTING = (Family.MEAN_REVERSION, Family.PIVOT)

#: Concurrent positions and post-loss cooldown by holding class. A scalp that
#: may hold three positions at once is a different risk profile from one that
#: may hold one, and no card states either.
_TYPE_RISK: dict[StrategyType, tuple[int, int]] = {
    StrategyType.SCALP: (1, 6),
    StrategyType.INTRADAY: (2, 4),
    StrategyType.SWING: (2, 3),
    StrategyType.POSITION: (3, 2),
}

_LONG_MARKER = re.compile(
    r"(?im)^[\s\-*#]*(long|buy|bullish)(?:\s+(?:entry|entries|trade|trades|signal|"
    r"position|setup|rules?|side))?\s*[:.\-–—]?\s*$"
)
_SHORT_MARKER = re.compile(
    r"(?im)^[\s\-*#]*(short|sell|bearish)(?:\s+(?:entry|entries|trade|trades|signal|"
    r"position|setup|rules?|side))?\s*[:.\-–—]?\s*$"
)
_LONG_INLINE = re.compile(r"(?i)\b(?:go\s+)?long\b|\bbuy\b|\bbullish\b|\bcall\b")
_SHORT_INLINE = re.compile(r"(?i)\b(?:go\s+)?short\b|\bsell\b|\bbearish\b|\bput\b")


class Fidelity(StrEnum):
    """How much of a spec came from the page.

    Attributes are ordered by distance from the source, and a reader deciding
    what to trust needs no other field: ``READ`` means every condition is a
    sentence the page wrote, ``ARCHETYPE`` means none of them are.
    """

    READ = "read"
    PARTIAL = "partial"
    ARCHETYPE = "archetype"


@dataclass(frozen=True)
class Normalisation:
    """What one card became.

    Attributes:
        card_id: The scraper's slug.
        spec: The strategy, or ``None`` when the card was refused.
        fidelity: How much of the spec the page supplied.
        family: What the entry bets on.
        universe: Symbols the spec may be run on, from the local store.
        refused_symbols: Symbols the store holds at this bar size that the
            spec may not use, by refusal code.
        inferences: Every departure from the page, in the order decided.
        refusal: Why there is no spec, when there is none.
        dropped: Clause text that was read and discarded, verbatim.
    """

    card_id: str
    spec: StrategySpec | None
    fidelity: Fidelity
    family: Family
    universe: tuple[str, ...] = ()
    refused_symbols: dict[str, str] = field(default_factory=dict)
    inferences: tuple[Inference, ...] = ()
    refusal: str | None = None
    dropped: tuple[str, ...] = ()

    @property
    def normalised(self) -> bool:
        """Whether a spec was produced."""
        return self.spec is not None


@dataclass
class _Leg:
    """A side under construction, and what it cost to build."""

    conditions: list[Condition] = field(default_factory=list)
    indicators: list[str] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)
    inferences: list[Inference] = field(default_factory=list)
    salvaged: int = 0


def _blocker_code(clause: str) -> tuple[InferenceCode, str] | None:
    """Why a clause cannot be read, when the reason is a known limitation.

    Args:
        clause: One sentence of rule text.

    Returns:
        ``(code, phrase)`` for the first limitation the clause trips, or
        ``None`` when it trips none — in which case the grammar's own failure
        is what stopped it.
    """
    checks = (
        (InferenceCode.CLAUSE_DROPPED_CROSS_TIMEFRAME, CROSS_TIMEFRAME_MARKERS),
        (InferenceCode.CLAUSE_DROPPED_REGIME, REGIME_MARKERS),
        (InferenceCode.CLAUSE_DROPPED_BAR_ARITHMETIC, BAR_ARITHMETIC_MARKERS),
        (InferenceCode.CLAUSE_DROPPED_VISUAL, VISUAL_MARKERS),
    )
    for code, markers in checks:
        found = contains_any(clause, markers)
        if found:
            return code, found[0]
    found = contains_any(clause, OFF_REGISTRY_INDICATORS)
    if found:
        return InferenceCode.INDICATOR_SUBSTITUTED, found[0]
    return None


def _indicators_of(condition: Condition) -> list[str]:
    """Every registry name a condition reads, in reading order."""
    names: list[str] = []
    if isinstance(condition, LeafCondition):
        for side in (condition.left, condition.right):
            if isinstance(side, FeatureRef) and side.indicator not in names:
                names.append(side.indicator)
        return names
    children = [condition.condition] if isinstance(condition, Not) else list(condition.conditions)
    for child in children:
        for name in _indicators_of(child):
            if name not in names:
                names.append(name)
    return names


def _read_text(text: str, declared: DeclaredIndicators, title: str, direction: Direction) -> _Leg:
    """Read every clause of one side, keeping what parses and recording what does not.

    This is where the strict pipeline's all-or-nothing rule is deliberately
    broken. There, one unreadable sentence refuses the card; here it is dropped
    and named. The cost is real and is not hidden: a spec whose page carried
    four conditions and whose grammar read three trades a strategy nobody wrote.
    :class:`Fidelity` and the dropped text are what let a reader see that.

    Args:
        text: The side's rule text.
        declared: The card's declared indicators.
        title: The card title, so a caption repeating it is dropped as noise.
        direction: The leg being built.

    Returns:
        The leg.
    """
    leg = _Leg()
    for clause in clauses(text):
        result = read_clause(clause, declared, title)
        if result.noise:
            continue
        if result.condition is not None:
            leg.conditions.append(result.condition)
            continue
        blocker = _blocker_code(clause)
        if blocker is None:
            rescued = salvage_clause(clause, direction)
            if rescued is not None:
                condition, reading = rescued
                leg.conditions.append(condition)
                leg.salvaged += 1
                leg.inferences.append(
                    Inference(
                        InferenceCode.CLAUSE_SALVAGED,
                        f"{reading}; the grammar refused the sentence whole: "
                        f"{clause.strip()[:120]!r}",
                    )
                )
                continue
            leg.dropped.append(clause.strip())
            leg.inferences.append(
                Inference(
                    InferenceCode.CLAUSE_DROPPED_UNREADABLE,
                    f"dropped, neither the grammar nor the salvage reader finds a rule in it: "
                    f"{clause.strip()[:120]!r}",
                )
            )
            continue
        code, phrase = blocker
        if code is InferenceCode.INDICATOR_SUBSTITUTED:
            swap = substitute_indicator(phrase)
            if swap is not None:
                rescued = salvage_clause(clause, direction)
                if rescued is not None:
                    condition, reading = rescued
                    leg.conditions.append(condition)
                    leg.salvaged += 1
                    if swap[0] not in leg.indicators:
                        leg.indicators.append(swap[0])
                    leg.inferences.append(
                        Inference(
                            InferenceCode.INDICATOR_SUBSTITUTED,
                            f"the clause names {phrase!r}, which is not implemented; "
                            f"{swap[0]!r} put in its place ({swap[1]}) and the clause then "
                            f"{reading}",
                        )
                    )
                    continue
            detail = (
                f"the clause names {phrase!r}; {swap[0]!r} would stand in for it ({swap[1]}), "
                "but the clause itself still does not read as a comparison, so it is dropped"
                if swap is not None
                else f"the clause names {phrase!r}, which is not implemented; dropped"
            )
            if swap is not None and swap[0] not in leg.indicators:
                leg.indicators.append(swap[0])
        else:
            detail = f"dropped, the clause is about {phrase!r}: {clause.strip()[:100]!r}"
        leg.dropped.append(clause.strip())
        leg.inferences.append(Inference(code, detail))
    return leg


def _split_sides(text: str) -> tuple[str, str] | None:
    """Cut a section that describes both legs at once into its two halves.

    The pages that do this write a bare "Long" on its own line, then the long
    rules, then a bare "Short". Where the markers are inline instead, the first
    occurrence of a short word is taken as the boundary — cruder, and the
    reason a leg built this way is never graded ``READ``.

    Args:
        text: An unsided rule section.

    Returns:
        ``(long_text, short_text)``, or ``None`` when no boundary is found.
    """
    long_at = _LONG_MARKER.search(text)
    short_at = _SHORT_MARKER.search(text)
    if long_at is not None and short_at is not None and long_at.start() != short_at.start():
        if long_at.start() < short_at.start():
            return text[long_at.end() : short_at.start()], text[short_at.end() :]
        return text[long_at.end() :], text[short_at.end() : long_at.start()]

    inline_long = _LONG_INLINE.search(text)
    inline_short = _SHORT_INLINE.search(text)
    if inline_long is None or inline_short is None:
        return None
    if inline_long.start() < inline_short.start():
        return text[: inline_short.start()], text[inline_short.start() :]
    return text[inline_long.start() :], text[: inline_long.start()]


def _side_texts(card: ScrapedCard) -> tuple[dict[Direction, str], InferenceCode | None]:
    """Find rule text for both legs, from whichever source carries it.

    Args:
        card: The card.

    Returns:
        ``(texts, code)``. ``code`` names the source when it was not a sided
        section, and is ``None`` when it was.
    """
    sided = {
        direction: text
        for direction in Direction
        if (text := card.rules_for(direction)) is not None
    }
    if sided:
        return sided, None

    unsided = "\n".join(text for _, text in card.unsided_rule_sections())
    if unsided.strip():
        split = _split_sides(unsided)
        if split is not None:
            return (
                {Direction.LONG: split[0], Direction.SHORT: split[1]},
                InferenceCode.SIDE_SPLIT_FROM_UNSIDED,
            )
        return {Direction.LONG: unsided}, InferenceCode.SIDE_SPLIT_FROM_UNSIDED

    prose = "\n".join(
        part for part in (card.description, card.indicators_raw) if part and part.strip()
    )
    if prose.strip():
        split = _split_sides(prose)
        if split is not None:
            return {Direction.LONG: split[0], Direction.SHORT: split[1]}, (
                InferenceCode.RULES_FROM_PROSE
            )
        return {Direction.LONG: prose}, InferenceCode.RULES_FROM_PROSE

    return {}, None


def _entry(
    direction: Direction, conditions: Sequence[Condition], indicators: Sequence[str]
) -> tuple[EntrySpec, Inference]:
    """Assemble one leg from the conditions read for it."""
    trigger: Condition = (
        conditions[0] if len(conditions) == 1 else AllOf(conditions=list(conditions))
    )
    level, code, detail = invalidation_level(indicators, direction)
    return (
        EntrySpec(
            direction=direction,
            trigger=trigger,
            confirmation=[],
            confirmation_window_bars=0,
            invalidation=Invalidation(price_level=level),
            entry_order=EntryOrderSpec(order=MarketOrder(), expire_after_bars=1),
        ),
        Inference(code, f"{direction.value.lower()} leg: {detail}"),
    )


def _exit_ref(
    card: ScrapedCard, strategy_type: StrategyType, known: Collection[str]
) -> tuple[str, Inference]:
    """Choose the exit preset, from the card's own words where it has any."""
    text = card.section_text(EXIT_HEADERS) or ""
    lowered = normalise(text).lower()
    if "trail" in lowered and ("atr" in lowered or "chandelier" in lowered):
        return "atr_trail_aggressive", Inference(
            InferenceCode.EXIT_FROM_TYPE, "the card asks for an ATR trail"
        )
    if "trail" in lowered and ("swing" in lowered or "structure" in lowered):
        return "structure_trail", Inference(
            InferenceCode.EXIT_FROM_TYPE, "the card asks for a structure trail"
        )
    ratio = read_ratio(lowered) if lowered else None
    if ratio is not None and ratio > 0:
        name, detail = snap_exit_ratio(ratio, tuple(known))
        if name is not None:
            return name, Inference(InferenceCode.EXIT_SNAPPED_TO_RATIO, detail)
    fallback = TYPE_EXITS[strategy_type.value]
    return fallback, Inference(
        InferenceCode.EXIT_FROM_TYPE,
        f"the card states no exit this library has a preset for; {fallback!r} chosen for a "
        f"{strategy_type.value} holding period. The exit is a free axis on this spec, not "
        "something the page decided",
    )


def _slug(card_id: str) -> str:
    """The scraper's slug in the lower-kebab-case the schema demands."""
    slug = re.sub(r"[^a-z0-9]+", "-", card_id.lower()).strip("-")
    return slug or "card"


def normalise_card(
    card: ScrapedCard,
    *,
    coverage: MarketCoverage,
    known_exit_ids: Collection[str],
    max_cost_ratio: float = DEFAULT_MAX_COST_RATIO,
) -> Normalisation:
    """Turn one card into a spec, deciding whatever the page left open.

    Args:
        card: The scraped page.
        coverage: What the local store holds, which decides the universe.
        known_exit_ids: Exit preset ids the library holds.
        max_cost_ratio: Cost-to-bar-range ratio above which a symbol is refused
            for this bar size.

    Returns:
        The outcome. A refusal here means the card offered no rule text of any
        kind and no indicator list to build an archetype from — after four
        sources were tried, not after the first.
    """
    declared = declared_indicators(card.indicators_raw)
    stated_tf, _ = read_timeframe(card)
    timeframe, tf_inference = timeframe_for(card.category, stated_tf)
    strategy_type, type_reason = classify_type(card.category, timeframe)

    inferences: list[Inference] = []
    if tf_inference is not None:
        inferences.append(tf_inference)
    stated_line = (card.timeframe_raw or "").strip()
    if stated_tf is None and stated_line:
        inferences.append(
            Inference(
                InferenceCode.TIMEFRAME_FLOOR_OF_SEVERAL,
                f"the card's timeframe line {stated_line[:60]!r} names none this "
                f"system has a bar size for, or several; {timeframe.value} used",
            )
        )

    texts, source_code = _side_texts(card)
    if source_code is not None and texts:
        inferences.append(
            Inference(
                source_code,
                "the card files no section under a side's name; rules taken from "
                + (
                    "an unsided rule section, split at its own long/short markers"
                    if source_code is InferenceCode.SIDE_SPLIT_FROM_UNSIDED
                    else "the page's description and indicator prose"
                ),
            )
        )

    legs: dict[Direction, _Leg] = {}
    for direction, text in texts.items():
        leg = _read_text(text, declared, card.title, direction)
        if leg.conditions:
            legs[direction] = leg
        inferences.extend(leg.inferences)

    dropped: list[str] = [line for leg in legs.values() for line in leg.dropped]
    read_any = bool(legs)
    salvaged_any = any(leg.salvaged for leg in legs.values())
    fidelity = Fidelity.READ if read_any else Fidelity.ARCHETYPE

    prose = f"{card.title}\n{card.description}"
    resolved = [name for leg in legs.values() for name in leg.indicators] + [
        name for leg in legs.values() for c in leg.conditions for name in _indicators_of(c)
    ]
    family, family_reason = classify_family(card.category, set(resolved), prose)

    if not read_any:
        names = _merge_names(
            _declared_names(declared),
            sniff_indicators(
                "\n".join(
                    part
                    for part in (card.indicators_raw, card.title, card.description, *texts.values())
                    if part
                )
            ),
        )
        for direction in (Direction.LONG, Direction.SHORT):
            conditions, notes = archetype_conditions(names, direction, family in _MEAN_REVERTING)
            detail = (
                f"{direction.value.lower()} leg written from the indicators {names} the card "
                "names, not read from the page: " + "; ".join(notes)
            )
            if not conditions and family_reason is not UNPLACED_REASON:
                fallback = family_archetype(family.value, direction, prose)
                if fallback is not None:
                    conditions, reading = fallback
                    detail = (
                        f"{direction.value.lower()} leg written from the family alone: {reading}"
                    )
            if conditions:
                legs[direction] = _Leg(conditions=list(conditions), indicators=list(names))
                inferences.append(Inference(InferenceCode.ARCHETYPE_FROM_INDICATORS, detail))
        if not legs:
            return Normalisation(
                card_id=card.strategy_id,
                spec=None,
                fidelity=Fidelity.ARCHETYPE,
                family=family,
                inferences=tuple(inferences),
                refusal=(
                    "no rule under a side's name, none in an unsided section, none in the "
                    f"page's prose, no archetype from its indicators {names}, and "
                    + (
                        "nothing about the page places it in a family whose canonical form "
                        "could stand in"
                        if family_reason is UNPLACED_REASON
                        else f"no canonical form for the {family.value} family"
                    )
                ),
            )

    if fidelity is Fidelity.READ and (dropped or source_code is not None or salvaged_any):
        fidelity = Fidelity.PARTIAL

    entries: list[EntrySpec] = []
    for direction in (Direction.LONG, Direction.SHORT):
        built = legs.get(direction)
        if built is None:
            continue
        names = built.indicators + [name for c in built.conditions for name in _indicators_of(c)]
        entry, inference = _entry(direction, built.conditions, names)
        entries.append(entry)
        inferences.append(inference)

    stop_text = card.section_text(STOP_HEADERS)
    if stop_text:
        for index, entry in enumerate(list(entries)):
            stated, _notes, _detail = read_invalidation(card, entry.direction, declared)
            if stated is not None:
                entries[index] = entry.model_copy(update={"invalidation": stated})
                inferences = [
                    inf
                    for inf in inferences
                    if not (
                        inf.code
                        in (
                            InferenceCode.INVALIDATION_FROM_SWING,
                            InferenceCode.INVALIDATION_FROM_TRIGGER_LINE,
                        )
                        and inf.detail.startswith(entry.direction.value.lower())
                    )
                ]

    exit_ref, exit_inference = _exit_ref(card, strategy_type, known_exit_ids)
    inferences.append(exit_inference)

    needs_volume = any(
        name in ("rvol", "mfi", "obv", "vwma", "volume_ma", "vwap_session", "vwap_anchored")
        for leg in legs.values()
        for c in leg.conditions
        for name in _indicators_of(c)
    )
    universe, refused = coverage.admissible(
        timeframe, needs_volume=needs_volume, max_cost_ratio=max_cost_ratio
    )
    inferences.append(
        Inference(
            InferenceCode.UNIVERSE_FROM_STORE,
            f"universe is what the store holds at {timeframe.value} and the cost model does not "
            f"drown: {list(universe)}"
            + (f"; refused {refused}" if refused else "")
            + (
                "; the spec reads volume, so zero-volume series are excluded"
                if needs_volume
                else ""
            ),
        )
    )
    max_positions, cooldown = _TYPE_RISK[strategy_type]
    inferences.append(
        Inference(
            InferenceCode.DEFAULTS_APPLIED,
            f"holding class {strategy_type.value} ({type_reason}); family {family.value} "
            f"({family_reason}); market entry expiring after 1 bar; stop reference "
            f"ATR(14)x1.5; base quality {DEFAULT_BASE_QUALITY} with no modifiers; "
            f"{max_positions} concurrent position(s); {cooldown}-bar cooldown after a loss",
        )
    )

    scope = InstrumentScope(
        allowed_classes=_classes_for(universe, coverage, timeframe),
        allowed_symbols=list(universe),
    )
    try:
        spec = StrategySpec(
            id=_slug(card.strategy_id),
            version=NORMALISED_VERSION,
            type=strategy_type,
            timeframes=TimeframeSpec(signal_tf=timeframe, entry_tf=timeframe),
            instruments=scope,
            market_regimes=[],
            entries=entries,
            exit_ref=exit_ref,
            filters=[],
            risk_profile=RiskProfileSpec(
                base_quality=DEFAULT_BASE_QUALITY,
                quality_modifiers=[],
                stop_reference=DEFAULT_STOP_REFERENCE,
                max_concurrent_positions=max_positions,
                cooldown_bars_after_loss=cooldown,
            ),
        )
    except (PydanticValidationError, ValueError) as error:
        return Normalisation(
            card_id=card.strategy_id,
            spec=None,
            fidelity=fidelity,
            family=family,
            inferences=tuple(inferences),
            refusal=f"the assembled spec is not valid: {error}",
            dropped=tuple(dropped),
        )

    errors = [
        issue
        for issue in validate_spec(spec, known_exit_ids=set(known_exit_ids))
        if issue.severity is Severity.ERROR
    ]
    if errors:
        return Normalisation(
            card_id=card.strategy_id,
            spec=None,
            fidelity=fidelity,
            family=family,
            inferences=tuple(inferences),
            refusal="; ".join(issue.message for issue in errors),
            dropped=tuple(dropped),
        )

    return Normalisation(
        card_id=card.strategy_id,
        spec=spec,
        fidelity=fidelity,
        family=family,
        universe=universe,
        refused_symbols=refused,
        inferences=tuple(inferences),
        dropped=tuple(dropped),
    )


def _declared_names(declared: DeclaredIndicators) -> list[str]:
    """Registry names a card declared, in declaration order."""
    return list(declared.by_key)


def _merge_names(declared: Sequence[str], sniffed: Sequence[str]) -> list[str]:
    """Declared indicators first, then any the prose names and the list missed.

    The declaration is the better evidence — it carries parameters — but this
    corpus routinely names in the rules an indicator the Indicators section
    forgot, and an archetype built from half the list is a different strategy
    from one built from all of it.
    """
    return list(dict.fromkeys([*declared, *sniffed]))


def _classes_for(
    symbols: Sequence[str], coverage: MarketCoverage, timeframe: Timeframe
) -> list[InstrumentClass]:
    """Asset classes an allow-list spans.

    The scope carries both the classes and the symbols because they answer
    different questions: the allow-list is what this spec may be run on today,
    the class is what kind of instrument it was written for and stays true when
    the store changes. An empty allow-list — every symbol at this bar size was
    refused — keeps FX, the class of the corpus these pages came from, rather
    than claiming no class at all.

    Args:
        symbols: The allow-list.
        coverage: Where each symbol's class is read from.
        timeframe: Bar size the spec trades.

    Returns:
        The distinct classes, in the enum's own order.
    """
    found = {
        found_series.asset_class
        for symbol in symbols
        if (found_series := coverage.get(symbol, timeframe)) is not None
    }
    if not found:
        return [InstrumentClass.FX]
    return [member for member in InstrumentClass if member in found]
