"""Turning scraped prose into clauses, without deciding what any of them mean.

Scraped rule text is hard-wrapped at whatever width the page used, so a single
rule regularly arrives as three lines and two rules regularly arrive as one.
Splitting on newlines alone therefore produces fragments no grammar can read
("Price above Ma simple 60" / "period."), and splitting on sentence punctuation
alone leaves rules glued together when the page never punctuated them.

The order here is: normalise the characters, re-join wrapped continuations,
then split what remains on sentence punctuation and top-level conjunctions.
Every step is deliberately conservative in the same direction — when re-flowing
is wrong, two rules end up in one clause and the clause fails to parse, which
:mod:`trading_system.strategies.ingest.rules` reports as a refusal. The failure
mode is a card that does not convert, never a card that converts into a rule
nobody wrote.
"""

import re
import unicodedata

#: Characters that end a clause. Colons are included because the pages use them
#: as list introducers ("Conditions: price above the ema"), not as punctuation
#: inside a rule.
_CLAUSE_ENDINGS = ".;:!?"

#: Conjunctions that separate two independent rules inside one sentence. Split
#: at the top level only, so ``"Stochastic (14, 5, 5)"`` survives intact.
_CONJUNCTIONS = (" and ", " & ", " plus ")

_MARKDOWN_NOISE = re.compile(r"\*+|_{2,}|#+")
_WHITESPACE = re.compile(r"[ \t ]+")
_CONTINUATION = re.compile(r"^[a-z0-9)\-–—,%<>=+]")


def normalise(text: str) -> str:
    """Fold a scraped section into plain, comparable characters.

    Applies NFKC (the pages mix full-width and typographic forms), strips the
    Markdown emphasis the scraper left behind, and unifies the several dash and
    quote characters onto ASCII. Line structure is preserved — it is the input
    to :func:`reflow`, not noise.

    Args:
        text: Raw section text.

    Returns:
        The normalised text.
    """
    folded = unicodedata.normalize("NFKC", text)
    folded = folded.replace(" ", " ")
    for dash in ("‐", "‑", "‒", "–", "—"):
        folded = folded.replace(dash, "-")
    for quote in ("‘", "’", "ʼ"):
        folded = folded.replace(quote, "'")
    for quote in ("“", "”"):
        folded = folded.replace(quote, '"')
    folded = _MARKDOWN_NOISE.sub(" ", folded)
    lines = [_WHITESPACE.sub(" ", line).strip() for line in folded.splitlines()]
    return "\n".join(line for line in lines if line)


def reflow(text: str) -> list[str]:
    """Re-join lines the page wrapped mid-sentence.

    A line continues the one before it when it opens with a lower-case letter,
    a digit, or punctuation — the shapes a sentence never starts with. A line
    opening with a capital is treated as a new rule even when the previous line
    lacked a full stop, because the pages routinely list one rule per line with
    no punctuation at all.

    Args:
        text: Normalised section text.

    Returns:
        One entry per re-flowed line, in order.
    """
    joined: list[str] = []
    for line in text.splitlines():
        if joined and _CONTINUATION.match(line):
            joined[-1] = f"{joined[-1]} {line}"
        else:
            joined.append(line)
    return joined


def _split_top_level(text: str, separators: tuple[str, ...]) -> list[str]:
    """Split ``text`` on ``separators`` that sit outside any bracket.

    Args:
        text: Text to split.
        separators: Literal separators to cut on.

    Returns:
        The pieces, in order, with surrounding whitespace stripped and empties
        dropped.
    """
    pieces: list[str] = []
    depth = 0
    start = 0
    index = 0
    lowered = text.lower()
    while index < len(text):
        char = text[index]
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth = max(0, depth - 1)
        if depth == 0:
            for separator in separators:
                if lowered.startswith(separator, index):
                    pieces.append(text[start:index])
                    index += len(separator)
                    start = index
                    break
            else:
                index += 1
            continue
        index += 1
    pieces.append(text[start:])
    return [piece.strip() for piece in pieces if piece.strip()]


def clauses(text: str) -> tuple[str, ...]:
    """Split a rule section into the smallest pieces that could each be a rule.

    Args:
        text: Raw section text, straight from the card.

    Returns:
        The clauses, in reading order. Bracketed parameter lists are never cut
        into, so ``"Stochastic (14, 5, 5) above 50"`` stays one clause.
    """
    out: list[str] = []
    for line in reflow(normalise(text)):
        for sentence in _split_top_level(line, tuple(_CLAUSE_ENDINGS)):
            out.extend(_split_top_level(sentence, (*_CONJUNCTIONS, ",")))
    return tuple(out)


def contains_any(text: str, phrases: tuple[str, ...]) -> tuple[str, ...]:
    """Which of ``phrases`` occur in ``text``, case-insensitively.

    Args:
        text: Text to search. Need not be normalised.
        phrases: Lower-case phrases to look for. A phrase made only of word
            characters is matched on word boundaries, so ``"ma"`` does not fire
            inside ``"market"``.

    Returns:
        The phrases found, in the order given, without duplicates.
    """
    haystack = normalise(text).lower()
    found: list[str] = []
    for phrase in phrases:
        pattern = (
            rf"\b{re.escape(phrase)}\b" if re.fullmatch(r"[\w %+-]+", phrase) else re.escape(phrase)
        )
        if re.search(pattern, haystack) and phrase not in found:
            found.append(phrase)
    return tuple(found)
