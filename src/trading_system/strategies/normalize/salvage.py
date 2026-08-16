"""Reading a sentence the strict grammar gave up on, more crudely and on purpose.

:mod:`trading_system.strategies.ingest.rules` parses a clause by structure and
refuses anything it cannot account for whole. That is the right discipline for
a reader whose output is evidence about a page, and it is why "ADX reading >=25"
— a sentence whose meaning no trader would dispute — comes back unread.

This module reads the same sentence by looking for three things and ignoring
everything between them: an operand, a comparator, another operand. It will
therefore read sentences that mean something else, which is exactly the trade
this normalisation is instructed to make. What keeps it honest is that a
condition built here is recorded as salvaged and the original sentence is kept
verbatim beside it, so the reading can be checked against the words.
"""

import re
from collections.abc import Sequence

from trading_system.features.registry import build_indicator
from trading_system.strategies.ingest.lexicon import (
    DEFAULT_CHANNELS,
    INDICATOR_PHRASES,
    PRICE_PHRASES,
    ZERO_PHRASES,
)
from trading_system.strategies.normalize.inference import (
    ARCHETYPE_PARAMS,
    INDICATOR_SUBSTITUTIONS,
    feature,
)
from trading_system.strategies.schema import (
    Condition,
    ConditionOp,
    Direction,
    FeatureRef,
    LeafCondition,
    Operand,
)

#: Comparator phrases to the operator they name, longest first so that
#: "crosses above" is never read as "above".
_COMPARATORS: tuple[tuple[str, ConditionOp], ...] = tuple(
    sorted(
        (
            ("crosses above", ConditionOp.CROSS_ABOVE),
            ("cross above", ConditionOp.CROSS_ABOVE),
            ("crossing above", ConditionOp.CROSS_ABOVE),
            ("crosses over", ConditionOp.CROSS_ABOVE),
            ("crosses up through", ConditionOp.CROSS_ABOVE),
            ("cuts above", ConditionOp.CROSS_ABOVE),
            ("crosses below", ConditionOp.CROSS_BELOW),
            ("cross below", ConditionOp.CROSS_BELOW),
            ("crossing below", ConditionOp.CROSS_BELOW),
            ("crosses under", ConditionOp.CROSS_BELOW),
            ("crosses down through", ConditionOp.CROSS_BELOW),
            ("cuts below", ConditionOp.CROSS_BELOW),
            ("greater than or equal", ConditionOp.GTE),
            ("less than or equal", ConditionOp.LTE),
            ("greater than", ConditionOp.GT),
            ("higher than", ConditionOp.GT),
            ("more than", ConditionOp.GT),
            ("rises above", ConditionOp.GT),
            ("less than", ConditionOp.LT),
            ("lower than", ConditionOp.LT),
            ("falls below", ConditionOp.LT),
            (">=", ConditionOp.GTE),
            ("<=", ConditionOp.LTE),
            ("above", ConditionOp.GT),
            ("below", ConditionOp.LT),
            ("under", ConditionOp.LT),
            ("over", ConditionOp.GT),
            (">", ConditionOp.GT),
            ("<", ConditionOp.LT),
        ),
        key=lambda pair: -len(pair[0]),
    )
)

#: Words meaning the value is climbing or falling, with no second operand.
_TREND_WORDS: tuple[tuple[str, ConditionOp], ...] = (
    ("rising", ConditionOp.RISING),
    ("increasing", ConditionOp.RISING),
    ("turns up", ConditionOp.RISING),
    ("pointing up", ConditionOp.RISING),
    ("falling", ConditionOp.FALLING),
    ("decreasing", ConditionOp.FALLING),
    ("turns down", ConditionOp.FALLING),
    ("pointing down", ConditionOp.FALLING),
)

#: Bands a bare indicator name is read as, by which way the comparison points.
#: A page saying "price above the Bollinger" means the upper band; saying
#: "below" it means the lower. Naming the near band instead would invert every
#: rule of this shape in the corpus.
_SIDED_BANDS: dict[str, tuple[str, str]] = {
    "bbands": ("lower", "upper"),
    "keltner": ("lower", "upper"),
    "donchian": ("lower", "upper"),
}

#: Phrases this reader adds on top of the strict lexicon's. They live here and
#: not in :mod:`trading_system.strategies.ingest.lexicon` on purpose: that table
#: is what the strict reader means by "the page said so", and widening it would
#: change what that reader converts. These are readings, not vocabulary — "the
#: price channel" is a Donchian channel to anyone reading the corpus, but it is
#: an inference all the same.
EXTRA_PHRASES: dict[str, str] = {
    "price channel": "donchian",
    "donchian channels": "donchian",
    "moving average": "sma",
    "moving averages": "sma",
    "envelope": "keltner",
    "envelopes": "keltner",
    "bollinger bands": "bbands",
    "stochastics": "stoch",
    "williams": "willr",
    "average directional movement": "adx",
    "momentum": "roc",
    "true range": "atr",
    "pivot": "pivots",
    "pivots": "pivots",
    "central pivot": "pivots",
    "camarilla": "pivots",
    "vwap": "vwap_session",
    "volume": "rvol",
}

_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def _phrase_hits(text: str) -> list[tuple[int, str, str]]:
    """Every operand phrase in a sentence, as ``(position, kind, name)``.

    Args:
        text: Lower-cased clause.

    Returns:
        Hits in reading order. Longer phrases win over shorter ones starting at
        the same place, so "moving average" never becomes two hits.
    """
    hits: list[tuple[int, str, str]] = []
    for phrase, name in INDICATOR_PHRASES.items():
        for found in re.finditer(rf"(?<![a-z]){re.escape(phrase)}(?![a-z])", text):
            hits.append((found.start(), "indicator", name))
    for phrase, field in PRICE_PHRASES.items():
        for found in re.finditer(rf"(?<![a-z]){re.escape(phrase)}(?![a-z])", text):
            hits.append((found.start(), "price", field))
    for phrase, (name, _reason) in INDICATOR_SUBSTITUTIONS.items():
        for found in re.finditer(rf"(?<![a-z]){re.escape(phrase)}(?![a-z])", text):
            hits.append((found.start(), "indicator", name))
    for phrase, name in EXTRA_PHRASES.items():
        for found in re.finditer(rf"(?<![a-z]){re.escape(phrase)}(?![a-z])", text):
            hits.append((found.start(), "indicator", name))
    hits.sort(key=lambda hit: hit[0])
    deduped: list[tuple[int, str, str]] = []
    for hit in hits:
        if deduped and hit[0] == deduped[-1][0]:
            continue
        deduped.append(hit)
    return deduped


def sniff_indicators(text: str) -> list[str]:
    """Every registry indicator a piece of prose mentions, in reading order.

    Used when a card files no rules at all: its title, description and
    indicator line still name what it is drawn with, and that list is what an
    archetype is built from.

    Args:
        text: Any prose from the card.

    Returns:
        Registry names, de-duplicated, first mention first.
    """
    names: list[str] = []
    for _position, kind, name in _phrase_hits(text.lower()):
        if kind == "indicator" and name not in names:
            names.append(name)
    return names


def _operand(
    kind: str, name: str, op: ConditionOp, params: Sequence[tuple[str, float | int]] = ()
) -> Operand | None:
    """Build one side of a comparison from a phrase hit."""
    if kind == "price":
        return f"price:{name}"
    channel = DEFAULT_CHANNELS.get(name)
    if name in _SIDED_BANDS:
        low, high = _SIDED_BANDS[name]
        channel = high if op in (ConditionOp.GT, ConditionOp.GTE, ConditionOp.CROSS_ABOVE) else low
    ref = feature(name, channel)
    if ref is None:
        return None
    if params:
        merged = dict(ARCHETYPE_PARAMS.get(name, {}))
        merged.update(dict(params))
        try:
            build_indicator(name, merged)
        except Exception:  # noqa: BLE001 - the registry raises several types
            return ref
        return FeatureRef(indicator=name, params=merged, channel=ref.channel)
    return ref


def salvage_clause(clause: str, direction: Direction) -> tuple[Condition, str] | None:
    """Read a clause by operand-comparator-operand, ignoring the rest.

    Args:
        clause: The sentence the strict grammar refused.
        direction: The leg being built, used only for a trend word that names
            no direction of its own.

    Returns:
        ``(condition, reading)`` where ``reading`` says what was matched, or
        ``None`` when the sentence holds no comparison at all.
    """
    text = clause.lower()
    hits = _phrase_hits(text)
    if not hits:
        return None

    for word, op in _TREND_WORDS:
        at = text.find(word)
        if at < 0:
            continue
        before = [hit for hit in hits if hit[0] < at]
        if not before:
            continue
        _position, kind, name = before[-1]
        left = _operand(kind, name, ConditionOp.GT)
        if left is None:
            continue
        return (
            LeafCondition(op=op, left=left, right=None),
            f"salvaged as {name} {op.value}, from the word {word!r}",
        )

    for phrase, op in _COMPARATORS:
        at = text.find(phrase)
        if at < 0:
            continue
        before = [hit for hit in hits if hit[0] < at]
        after = [hit for hit in hits if hit[0] >= at + len(phrase)]
        if not before:
            continue
        _position, kind, name = before[-1]
        left = _operand(kind, name, op)
        if left is None:
            continue

        tail = text[at + len(phrase) :]
        right: Operand | None = None
        number = _NUMBER.search(tail)
        first_after = after[0] if after else None
        if any(zero in tail for zero in ZERO_PHRASES) and (
            number is None or number.start() > tail.find("zero")
        ):
            right = 0.0
        elif number is not None and (
            first_after is None or number.start() < first_after[0] - (at + len(phrase))
        ):
            right = float(number.group(0))
        elif first_after is not None:
            right = _operand(first_after[1], first_after[2], op)
        if right is None:
            continue
        if isinstance(left, str) and isinstance(right, float):
            # "price above 1.2500" is a level from a chart, not a rule this
            # spec can carry onto another instrument.
            continue
        return (
            LeafCondition(op=op, left=left, right=right),
            f"salvaged as {_describe(left)} {op.value} {_describe(right)}, on the word {phrase!r}",
        )
    del direction
    return None


def _describe(operand: Operand) -> str:
    """A short name for an operand, for the reading string."""
    if isinstance(operand, FeatureRef):
        return f"{operand.indicator}{'.' + operand.channel if operand.channel else ''}"
    return str(operand)
