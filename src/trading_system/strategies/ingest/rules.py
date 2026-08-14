"""Turning one clause into one condition, or into a stated reason it cannot be.

The grammar is small on purpose. A clause is read only when it has the shape
"<something> <comparator> <something>", both sides read cleanly by
:mod:`trading_system.strategies.ingest.terms`, and nothing is left over. Two
readings of the same clause never compete: comparators are searched
longest-first and the crossing forms are tried before the plain ones, since
"crosses above" contains "above" and the two mean different things.

What the grammar refuses is as much of the design as what it accepts. A clause
that carries a rule but no comparator ("Bollinger Bands and Keltner Channels
narrow") is a refusal, not an empty condition; a clause that is a caption
("in the pictures below") is dropped without comment. The difference matters —
one is a rule the converter could not read, the other is not a rule at all —
and merging them would make an unreadable card look like a simple one.
"""

import re
from dataclasses import dataclass

from trading_system.strategies.ingest.lexicon import NOISE_MARKERS
from trading_system.strategies.ingest.terms import DeclaredIndicators, read_operand
from trading_system.strategies.schema import Condition, ConditionOp, LabelSet, LeafCondition
from trading_system.strategies.schema import Operand as SpecOperand

#: Words that give a crossing its side.
_UPWARD = ("above", "over", "upwards", "upward", "up", "higher")
_DOWNWARD = ("below", "under", "downwards", "downward", "down", "lower")

_CROSS = re.compile(r"\b(?:cross\w*|breaks?|breakout|penetrat\w*)\b")

#: Plain comparators, longest first so that "greater than" is not read as the
#: bare ">" hiding inside it.
_COMPARATORS: tuple[tuple[str, ConditionOp], ...] = (
    ("greater than or equal to", ConditionOp.GTE),
    ("less than or equal to", ConditionOp.LTE),
    ("is greater than", ConditionOp.GT),
    ("is less than", ConditionOp.LT),
    ("greater than", ConditionOp.GT),
    ("higher than", ConditionOp.GT),
    ("less than", ConditionOp.LT),
    ("lower than", ConditionOp.LT),
    ("is above", ConditionOp.GT),
    ("is below", ConditionOp.LT),
    ("stays above", ConditionOp.GT),
    ("stays below", ConditionOp.LT),
    ("closes above", ConditionOp.GT),
    ("closes below", ConditionOp.LT),
    ("close above", ConditionOp.GT),
    ("close below", ConditionOp.LT),
    ("above", ConditionOp.GT),
    ("below", ConditionOp.LT),
    (">=", ConditionOp.GTE),
    ("<=", ConditionOp.LTE),
    (">", ConditionOp.GT),
    ("<", ConditionOp.LT),
)

#: Words that carry no rule on their own. A clause made only of these is a
#: heading the scraper kept ("Buy", "Rule 1", "Example"), not a sentence the
#: grammar failed to read.
_NOISE_WORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "and",
        "buy",
        "sell",
        "long",
        "short",
        "entry",
        "entries",
        "exit",
        "rule",
        "rules",
        "example",
        "examples",
        "note",
        "notes",
        "first",
        "second",
        "third",
        "step",
        "setup",
        "trade",
        "trades",
        "trading",
        "signal",
        "signals",
        "chart",
        "charts",
        "on",
        "in",
        "the",
        "for",
        "of",
        "or",
        "position",
        "strategy",
        "system",
        "method",
    }
)

#: Trailing words that qualify a comparison without changing it.
_TRAILING_NOISE = re.compile(
    r"\b(?:for entry|at the close|on the close|of the candle|of the bar|at the market)\b"
)


@dataclass(frozen=True)
class ClauseResult:
    """What one clause turned into.

    Exactly one of ``condition``, ``noise`` and ``problem`` is meaningful:
    a condition read, a caption dropped, or a reason the clause was refused.

    Attributes:
        condition: The condition read, or ``None``.
        noise: Whether the clause carries no rule at all.
        problem: Why the clause was refused; ``None`` otherwise.
        assumptions: Readings the card did not spell out.
    """

    condition: Condition | None = None
    noise: bool = False
    problem: str | None = None
    assumptions: tuple[str, ...] = ()


def is_noise(clause: str, title: str = "") -> bool:
    """Whether a clause is a caption, a filename or a call to action.

    The pages repeat their own title between rules, as a caption under each
    screenshot. A clause whose words are all in the title is that caption: it
    is dropped rather than refused, since it never was a sentence about
    trading.

    Args:
        clause: One clause of a rule section.
        title: The card's title, if the caller has it.

    Returns:
        ``True`` when the clause carries no rule.
    """
    lowered = clause.lower()
    if len(lowered) < 3:
        return True
    if any(marker in lowered for marker in NOISE_MARKERS):
        return True
    words = set(re.findall(r"[a-z]+", lowered)) - _NOISE_WORDS
    if not words:
        return True
    # A caption is prose. Anything carrying a number or a comparison symbol is
    # a candidate rule and must be read or refused, never dropped for looking
    # like the page title — "ADX (14) > 25" on a page called "EMA and ADX
    # Trading System" is a rule whose every word is in the title.
    if any(char.isdigit() for char in lowered) or any(char in lowered for char in "<>="):
        return False
    title_words = set(re.findall(r"[a-z]+", title.lower()))
    return bool(title_words) and words <= title_words


def _cross_op(rest: str) -> tuple[ConditionOp | None, str, str | None]:
    """Read the side of a crossing and strip the word that gave it.

    Args:
        rest: Clause text after the crossing verb.

    Returns:
        ``(op, remaining_text, problem)``.
    """
    tokens = re.findall(r"[a-z]+", rest.lower())
    up = [word for word in tokens if word in _UPWARD]
    down = [word for word in tokens if word in _DOWNWARD]
    if up and down:
        return None, rest, f"names both sides of a crossing: {up + down}"
    if not up and not down:
        return None, rest, "a crossing without a side"
    op = ConditionOp.CROSS_ABOVE if up else ConditionOp.CROSS_BELOW
    stripped = re.sub(rf"\b(?:{'|'.join(_UPWARD + _DOWNWARD)})\b", " ", rest, flags=re.IGNORECASE)
    return op, stripped, None


def _leaf(op: ConditionOp, left: SpecOperand, right: SpecOperand) -> Condition:
    """Build a leaf condition."""
    return LeafCondition(op=op, left=left, right=right)


def read_clause(clause: str, declared: DeclaredIndicators, title: str = "") -> ClauseResult:
    """Read one clause of a rule section.

    Args:
        clause: The clause, as split by
            :func:`trading_system.strategies.ingest.text.clauses`.
        declared: The card's declared indicators, for mentions without
            parameters.
        title: The card's title, so a caption repeating it is dropped.

    Returns:
        The condition, a noise marker, or the reason for refusal.
    """
    if is_noise(clause, title):
        return ClauseResult(noise=True)

    body = _TRAILING_NOISE.sub(" ", clause)

    cross = _CROSS.search(body)
    if cross is not None:
        op, rest, problem = _cross_op(body[cross.end() :])
        if problem is not None:
            return ClauseResult(problem=f"{problem}: {clause.strip()!r}")
        assert op is not None
        return _two_sided(op, body[: cross.start()], rest, clause, declared)

    lowered = body.lower()
    for phrase, op in _COMPARATORS:
        position = lowered.find(phrase)
        if position < 0:
            continue
        left = body[:position]
        right = body[position + len(phrase) :]
        return _two_sided(op, left, right, clause, declared)

    return _no_comparator(body, clause, declared)


def _two_sided(
    op: ConditionOp,
    left_text: str,
    right_text: str,
    clause: str,
    declared: DeclaredIndicators,
) -> ClauseResult:
    """Read both sides of a comparison, refusing unless both read cleanly."""
    left = read_operand(left_text, declared)
    if not left.ok:
        return ClauseResult(problem=f"left of {op.value!r}: {left.problem}")
    right = read_operand(right_text, declared)
    if not right.ok:
        return ClauseResult(problem=f"right of {op.value!r}: {right.problem}")
    assert left.operand is not None and right.operand is not None
    if isinstance(left.operand, LabelSet) or isinstance(right.operand, LabelSet):
        return ClauseResult(problem=f"a label cannot be compared numerically: {clause.strip()!r}")
    return ClauseResult(
        condition=_leaf(op, left.operand, right.operand),
        assumptions=left.assumptions + right.assumptions,
    )


def _no_comparator(body: str, clause: str, declared: DeclaredIndicators) -> ClauseResult:
    """Read a clause that compares nothing: a pattern or a session name."""
    result = read_operand(body, declared)
    if result.ok and isinstance(result.operand, LabelSet):
        label = result.operand.labels[0]
        op = ConditionOp.SESSION_IS if label in _SESSION_LABELS else ConditionOp.PATTERN_IS
        return ClauseResult(
            condition=LeafCondition(op=op, left=None, right=result.operand),
            assumptions=result.assumptions,
        )
    return ClauseResult(problem=f"no comparison in {clause.strip()!r}")


_SESSION_LABELS = frozenset({"SYDNEY", "TOKYO", "LONDON", "NEWYORK", "LONDON_NY_OVERLAP"})
