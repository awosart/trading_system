"""Reading one side of a comparison: a feature, a price field, or a constant.

The scanner here is deliberately total. Every token of a fragment must land in
one of the tables in :mod:`trading_system.strategies.ingest.lexicon` — an
indicator, a channel, a price field, a number, a pattern, a session, or a
filler word that changes nothing. One token that lands nowhere fails the whole
fragment and names itself in the reason, because a fragment the scanner only
partly understood is exactly the shape that produces a plausible wrong spec:
"Parabolic Sar dot below the candle price" contains a perfectly readable
"below the candle price" and a subject the system does not have.

Indicator parameters come from the card or not at all. When the mention spells
them out ("Stochastic (14, 5, 5)") they are used as written; when it does not,
the card's own Indicators section is consulted, and it must declare exactly one
such indicator. Falling back on a constructor default would let "EMA" mean
``period=20`` on a page that never said 20.
"""

import re
from dataclasses import dataclass, field

from trading_system.data.sessions import Session
from trading_system.features.registry import INDICATOR_TYPES
from trading_system.strategies.ingest.lexicon import (
    CHANNEL_PHRASES,
    DEFAULT_CHANNELS,
    FILLER_WORDS,
    INDICATOR_PARAMS,
    INDICATOR_PHRASES,
    PATTERN_PHRASES,
    PRICE_PHRASES,
    SESSION_PHRASES,
    ZERO_PHRASES,
    coerce_param,
)
from trading_system.strategies.schema import FeatureRef, LabelSet, Operand

#: Indicator phrases that name a channel by themselves.
IMPLIED_CHANNELS: dict[str, str] = {
    "swing high": "swing_high",
    "swing highs": "swing_high",
    "swing low": "swing_low",
    "swing lows": "swing_low",
}

#: Longest phrase, in tokens, any table holds. Bounds the scanner's lookahead.
_MAX_PHRASE_TOKENS = 4

_TOKEN = re.compile(r"[+\-%]?[a-z][a-z%']*[+\-]?|\d+(?:[.,]\d+)?")

#: Indicators whose constructor takes a ``source`` price field, so that
#: "200 EMA High" is one operand — a moving average of highs — rather than two
#: things named at once.
_SOURCED: frozenset[str] = frozenset({"ema", "sma", "wma", "hma", "vwma", "roc", "stddev"})


def tokenise(fragment: str) -> list[str]:
    """Split a fragment into comparable tokens.

    Args:
        fragment: One side of a comparison, or any short phrase.

    Returns:
        Lower-case tokens: words (keeping the ``%``, ``+`` and ``-`` that make
        ``%k`` and ``+di`` what they are) and numbers, with possessive endings
        dropped so ``"ema's"`` reads as ``"ema"``.
    """
    tokens: list[str] = []
    for raw in _TOKEN.findall(fragment.lower().replace("’", "'")):
        token = raw.removesuffix("'s").removesuffix("'")
        if token:
            tokens.append(token)
    return tokens


@dataclass(frozen=True)
class DeclaredIndicators:
    """Indicators a card lists in its Indicators section, with their parameters.

    A rule saying "RSI above 50" names no period; the card's own indicator list
    usually does. This is that list, and it is consulted only when the mention
    itself carries no numbers.

    Attributes:
        by_key: Registry key to every parameter set declared for it, in the
            order they were declared. More than one entry means the card runs
            two of the same indicator and a bare mention is ambiguous.
    """

    by_key: dict[str, list[dict[str, float | int | str]]] = field(default_factory=dict)

    def sole_params(self, key: str) -> dict[str, float | int | str] | None:
        """The one parameter set declared for ``key``, if there is exactly one.

        Args:
            key: Registry key.

        Returns:
            The parameters, or ``None`` when the card declared none or several.
        """
        declared = self.by_key.get(key, [])
        return dict(declared[0]) if len(declared) == 1 else None


@dataclass(frozen=True)
class TermResult:
    """What a fragment turned out to be, or why it did not turn into anything.

    Attributes:
        operand: The operand read, or ``None`` when the fragment failed.
        problem: Why the fragment failed; ``None`` on success.
        assumptions: Readings the card did not spell out, recorded so a human
            reviewing the converted spec sees them.
    """

    operand: Operand | None
    problem: str | None
    assumptions: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        """Whether a fragment was read."""
        return self.operand is not None


@dataclass
class _Scan:
    """Everything one fragment's tokens turned out to hold."""

    indicators: list[str] = field(default_factory=list)
    channel_phrases: list[str] = field(default_factory=list)
    implied_channels: list[str] = field(default_factory=list)
    prices: list[str] = field(default_factory=list)
    patterns: list[str] = field(default_factory=list)
    sessions: list[Session] = field(default_factory=list)
    numbers: list[float] = field(default_factory=list)
    zero: bool = False
    unknown: list[str] = field(default_factory=list)


#: Every channel phrase any indicator uses, for the scanner's lookup. Which
#: indicator a phrase is legal for is checked after the scan, when the mention
#: is known.
_ALL_CHANNEL_PHRASES: frozenset[str] = frozenset(
    phrase for phrases in CHANNEL_PHRASES.values() for phrase in phrases
)


def _scan(tokens: list[str]) -> _Scan:
    """Consume every token into the category it belongs to.

    Args:
        tokens: Output of :func:`tokenise`.

    Returns:
        The scan. ``unknown`` non-empty means the fragment is unreadable.
    """
    scan = _Scan()
    index = 0
    while index < len(tokens):
        for length in range(min(_MAX_PHRASE_TOKENS, len(tokens) - index), 0, -1):
            phrase = " ".join(tokens[index : index + length])
            if phrase in INDICATOR_PHRASES:
                scan.indicators.append(INDICATOR_PHRASES[phrase])
                if phrase in IMPLIED_CHANNELS:
                    scan.implied_channels.append(IMPLIED_CHANNELS[phrase])
                break
            if phrase in _ALL_CHANNEL_PHRASES:
                scan.channel_phrases.append(phrase)
                break
            if phrase in PATTERN_PHRASES:
                scan.patterns.append(PATTERN_PHRASES[phrase].value)
                break
            if phrase in SESSION_PHRASES:
                scan.sessions.append(SESSION_PHRASES[phrase])
                break
            if phrase in ZERO_PHRASES:
                scan.zero = True
                break
            if phrase in PRICE_PHRASES:
                scan.prices.append(PRICE_PHRASES[phrase])
                break
        else:
            token = tokens[index]
            if re.fullmatch(r"\d+(?:[.,]\d+)?", token):
                scan.numbers.append(float(token.replace(",", ".")))
            elif token not in FILLER_WORDS:
                scan.unknown.append(token)
            index += 1
            continue
        index += length
    return scan


def _resolve_channel(key: str, scan: _Scan) -> tuple[str | None, str | None, tuple[str, ...]]:
    """Pick the channel a mention of ``key`` reads.

    Args:
        key: Registry key of the mentioned indicator.
        scan: The fragment's scan.

    Returns:
        ``(channel, problem, assumptions)``. ``channel`` is ``None`` both for a
        single-output indicator and when ``problem`` is set.
    """
    outputs = INDICATOR_TYPES[key].outputs
    named = scan.implied_channels + [
        channel
        for phrase in scan.channel_phrases
        if (channel := CHANNEL_PHRASES.get(key, {}).get(phrase)) is not None
    ]
    unmatched = [
        phrase for phrase in scan.channel_phrases if phrase not in CHANNEL_PHRASES.get(key, {})
    ]
    if unmatched:
        return None, f"{unmatched[0]!r} is not a line of {key!r}", ()
    if len(named) > 1:
        return None, f"names several lines of {key!r}: {named}", ()
    if len(outputs) == 1:
        if named:
            return None, f"{key!r} has a single output but {named[0]!r} was named", ()
        return None, None, ()
    if named:
        if named[0] not in outputs:
            return None, f"{key!r} has no line {named[0]!r}", ()
        return named[0], None, ()
    default = DEFAULT_CHANNELS.get(key)
    if default is None:
        return None, f"{key!r} has lines {list(outputs)} and the card names none", ()
    return default, None, (f"a bare {key!r} was read as its {default!r} line",)


def _reconcile(
    key: str, mention: dict[str, float | int | str], declared: DeclaredIndicators
) -> tuple[dict[str, float | int | str] | None, str | None, tuple[str, ...]]:
    """Check a spelled-out mention against what the card declared.

    "20 SMA" looks unambiguous until the card's indicator list turns out to
    declare a 20-period SMA of highs and a 20-period SMA of lows. The numbers
    match both, and the rule names neither, so the mention has two readings and
    no reason to prefer one. Where exactly one declaration matches, its extra
    detail — the price it is computed on — is adopted, since that is the
    indicator the card actually runs.

    Args:
        key: Registry key.
        mention: Parameters as the rule spelled them.
        declared: What the card's Indicators section declares.

    Returns:
        ``(params, problem, assumptions)``.
    """
    candidates = [
        entry
        for entry in declared.by_key.get(key, [])
        if all(entry.get(name) == value for name, value in mention.items())
    ]
    if len(candidates) > 1:
        return (
            None,
            f"{key!r} with {mention} matches {len(candidates)} of the card's declared "
            f"{key!r} indicators ({candidates}), and the rule does not say which",
            (),
        )
    if len(candidates) == 1 and candidates[0] != mention:
        adopted = dict(candidates[0])
        return adopted, None, (f"{key!r} matched the card's declared {adopted}",)
    return mention, None, ()


def _resolve_params(
    key: str, scan: _Scan, declared: DeclaredIndicators, source: str = ""
) -> tuple[dict[str, float | int | str] | None, str | None, tuple[str, ...]]:
    """Pick the parameters a mention of ``key`` carries.

    Args:
        key: Registry key of the mentioned indicator.
        scan: The fragment's scan.
        declared: What the card's Indicators section declares.
        source: Price field the mention names the indicator computed on, if any.

    Returns:
        ``(params, problem, assumptions)``.
    """
    expected = INDICATOR_PARAMS[key]
    numbers = scan.numbers
    if numbers:
        if len(numbers) != len(expected):
            return (
                None,
                f"{key!r} takes {len(expected)} parameter(s) {list(expected)} "
                f"but the rule gives {len(numbers)}: {numbers}",
                (),
            )
        built: dict[str, float | int | str] = {
            name: coerce_param(name, value) for name, value in zip(expected, numbers, strict=True)
        }
        if source:
            built["source"] = source
        return _reconcile(key, built, declared)
    if not expected:
        return {}, None, ()
    params = declared.sole_params(key)
    if params is None:
        count = len(declared.by_key.get(key, []))
        detail = "none" if count == 0 else f"{count} different ones"
        return (
            None,
            f"{key!r} is named without parameters and the card's indicator list declares {detail}",
            (),
        )
    return params, None, (f"parameters for {key!r} taken from the card's indicator list: {params}",)


def _imply_indicator(scan: _Scan, declared: DeclaredIndicators) -> tuple[str, ...]:
    """Name the indicator a channel-only fragment must belong to, if only one can.

    "+DI above -DI" names two lines and no indicator. The lines belong to ADX
    and to nothing else, and the card declares an ADX, so the fragment has one
    reading. Where two indicators share the channel spelling — "upper band" is
    Bollinger, Keltner and Donchian alike — the card's declarations have to
    narrow it to one, or the fragment stays unreadable.

    Args:
        scan: The fragment's scan, extended in place when a reading is found.
        declared: What the card's Indicators section declares.

    Returns:
        The assumption recorded, or empty when nothing was implied.
    """
    if scan.indicators or not scan.channel_phrases:
        return ()
    candidates = [
        key
        for key, phrases in CHANNEL_PHRASES.items()
        if all(phrase in phrases for phrase in scan.channel_phrases)
        and (key in declared.by_key or not INDICATOR_PARAMS[key])
    ]
    if len(candidates) != 1:
        return ()
    scan.indicators.append(candidates[0])
    return (f"{scan.channel_phrases[0]!r} read as a line of the declared {candidates[0]!r}",)


def read_operand(fragment: str, declared: DeclaredIndicators) -> TermResult:
    """Read one side of a comparison.

    Args:
        fragment: The text on that side, e.g. ``"the 200 period EMA"``.
        declared: What the card's Indicators section declares, used only when a
            mention carries no parameters of its own.

    Returns:
        The operand, or the reason the fragment could not be read.
    """
    tokens = tokenise(fragment)
    if not tokens:
        return TermResult(None, "empty")
    scan = _scan(tokens)
    if scan.unknown:
        return TermResult(None, f"unknown word {scan.unknown[0]!r} in {fragment.strip()!r}")

    implied_note = _imply_indicator(scan, declared)
    subjects = len(scan.indicators) + len(scan.prices) + len(scan.patterns) + len(scan.sessions)
    if len(scan.indicators) > 1:
        return TermResult(None, f"names several indicators at once: {scan.indicators}")

    source_note: tuple[str, ...] = ()
    if (
        subjects == 2
        and len(scan.indicators) == 1
        and len(scan.prices) == 1
        and scan.indicators[0] in _SOURCED
    ):
        # "200 EMA High" is a moving average computed on highs, which the
        # indicator takes as its ``source``. Only for indicators that have one:
        # elsewhere an indicator and a price in one fragment is two subjects.
        source = scan.prices.pop()
        subjects = 1
        source_note = (f"{scan.indicators[0]!r} read as computed on the {source!r} price",)
    else:
        source = ""
    if subjects > 1:
        return TermResult(None, f"names several things at once in {fragment.strip()!r}")

    if scan.patterns:
        return TermResult(LabelSet(labels=(scan.patterns[0],)), None)
    if scan.sessions:
        return TermResult(LabelSet(labels=(scan.sessions[0].value,)), None)

    if scan.indicators:
        key = scan.indicators[0]
        params, problem, param_notes = _resolve_params(key, scan, declared, source)
        if problem is not None:
            return TermResult(None, problem)
        channel, problem, channel_notes = _resolve_channel(key, scan)
        if problem is not None:
            return TermResult(None, problem)
        assert params is not None
        resolved: dict[str, float | int | str] = dict(params)
        return TermResult(
            FeatureRef(indicator=key, params=resolved, channel=channel),
            None,
            implied_note + param_notes + channel_notes + source_note,
        )

    if scan.prices:
        if scan.numbers:
            return TermResult(None, f"a price with a number attached: {fragment.strip()!r}")
        return TermResult(f"price:{scan.prices[0]}", None)

    if scan.zero and not scan.numbers:
        return TermResult(0.0, None)
    if len(scan.numbers) == 1:
        return TermResult(scan.numbers[0], None)
    if len(scan.numbers) > 1:
        return TermResult(None, f"several bare numbers: {scan.numbers}")
    return TermResult(None, f"nothing to compare in {fragment.strip()!r}")


def declared_indicators(indicators_raw: str | None) -> DeclaredIndicators:
    """Read a card's Indicators section into parameter sets per indicator.

    Each line is scanned on its own. A line naming one known indicator and the
    right number of parameters contributes a declaration; anything else is
    ignored, since this list only ever *supplies* parameters a rule omitted and
    never decides whether a card converts.

    Args:
        indicators_raw: The card's Indicators section, or ``None``.

    Returns:
        The declarations found.
    """
    by_key: dict[str, list[dict[str, float | int | str]]] = {}
    if not indicators_raw:
        return DeclaredIndicators(by_key)
    for line in indicators_raw.splitlines():
        scan = _scan(tokenise(line))
        if len(scan.indicators) != 1:
            continue
        key = scan.indicators[0]
        expected = INDICATOR_PARAMS[key]
        if len(scan.numbers) != len(expected):
            continue
        params: dict[str, float | int | str] = {
            name: coerce_param(name, value)
            for name, value in zip(expected, scan.numbers, strict=True)
        }
        if len(scan.prices) == 1 and key in _SOURCED and scan.prices[0] != "close":
            # "20 SMA High" and "20 SMA Low" are two different indicators. Folding
            # the source away would make a card that declares both look like one
            # that declares a single unambiguous SMA.
            params["source"] = scan.prices[0]
        entries = by_key.setdefault(key, [])
        if params not in entries:
            entries.append(params)
    return DeclaredIndicators(by_key)
