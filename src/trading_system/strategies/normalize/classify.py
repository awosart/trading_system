"""Sorting a normalised card onto two axes: how long it holds, and what it bets on.

The two are independent and both are needed. ``StrategyType`` is a schema field
and says how long a position lives; it decides the exit preset a silent card is
given and the fold geometry the run will need. The family says what the entry
is a bet *on*, and it is the axis a reader browses by — a corpus of 883 pages
holds perhaps seven distinct ideas, and filing them only by holding period
would put a Bollinger fade next to an opening-range breakout.

Neither is stated by the pages. Both are derived from evidence the page does
carry — its site category, its declared indicators, its own words — and every
derivation is returned with the reason, because a classification without one is
indistinguishable from a guess.
"""

import re
from collections.abc import Collection
from enum import StrEnum

from trading_system.core.types import Timeframe
from trading_system.strategies.schema import StrategyType


class Family(StrEnum):
    """What an entry bets on.

    Members are chosen so that two strategies in the same family would show up
    correlated in :mod:`trading_system.analytics.library_report` — that is the
    test of whether a partition of trading ideas is real or decorative.
    """

    TREND = "TREND"
    BREAKOUT = "BREAKOUT"
    MEAN_REVERSION = "MEAN_REVERSION"
    MOMENTUM = "MOMENTUM"
    PATTERN = "PATTERN"
    PIVOT = "PIVOT"
    VOLATILITY = "VOLATILITY"


#: Reason returned when nothing in a card places it. The family that comes back
#: with it is the corpus's largest and is a filing decision, not a reading — so
#: a caller must not build an entry out of it. Compared by identity, because a
#: sentence that happens to match is not the same as this having been the
#: outcome.
UNPLACED_REASON = "nothing in the card places it; filed under the corpus's largest family"

#: Site categories that name a family outright. The page's own filing is the
#: strongest evidence available and is used before anything is inferred from
#: indicators, which are shared across families far more than categories are.
_CATEGORY_FAMILY: dict[str, Family] = {
    "trend-following-forex-strategies": Family.TREND,
    "trend-following-forex-strategies-ii": Family.TREND,
    "breakout-forex-strategies": Family.BREAKOUT,
    "support-and-resistance-forex-strategies": Family.MEAN_REVERSION,
    "patterns-forex-strategies": Family.PATTERN,
    "candlestick-forex-strategies": Family.PATTERN,
    "pivot-forex-strategies": Family.PIVOT,
    "volatility-forex-strategies": Family.VOLATILITY,
}

#: Indicators that, absent a category saying otherwise, place a card. Ordered
#: most specific first: ``donchian`` is a breakout channel wherever it appears,
#: while ``ema`` is in nearly every card of every family and therefore decides
#: nothing on its own.
_INDICATOR_FAMILY: tuple[tuple[str, Family], ...] = (
    ("donchian", Family.BREAKOUT),
    ("pivots", Family.PIVOT),
    ("keltner", Family.VOLATILITY),
    ("supertrend", Family.TREND),
    ("ichimoku", Family.TREND),
    ("bbands", Family.MEAN_REVERSION),
    ("willr", Family.MEAN_REVERSION),
    ("cci", Family.MOMENTUM),
    ("stoch", Family.MEAN_REVERSION),
    ("rsi", Family.MEAN_REVERSION),
    ("mfi", Family.MEAN_REVERSION),
    ("macd", Family.MOMENTUM),
    ("roc", Family.MOMENTUM),
    ("adx", Family.TREND),
    ("hma", Family.TREND),
    ("ema", Family.TREND),
    ("sma", Family.TREND),
    ("wma", Family.TREND),
)

#: Words in a card's own prose that override an indicator-derived family. A
#: page that says "breakout" is describing a breakout whatever it draws it with.
_PROSE_FAMILY: tuple[tuple[re.Pattern[str], Family], ...] = (
    (re.compile(r"\bbreak(?:out|s out|ing out)\b|\bbreach\b", re.I), Family.BREAKOUT),
    (re.compile(r"\bscalp\w*\b", re.I), Family.MOMENTUM),
    (
        re.compile(r"\boverbought\b|\boversold\b|\bfade\b|\brevert\b|\bmean revers", re.I),
        Family.MEAN_REVERSION,
    ),
    (re.compile(r"\bpullback\b|\bretrace\w*\b|\btrend\b", re.I), Family.TREND),
    (re.compile(r"\bsqueeze\b|\bvolatilit\w+\b|\bexpansion\b", re.I), Family.VOLATILITY),
    (re.compile(r"\bpivot\b|\bcamarilla\b|\bfloor trader", re.I), Family.PIVOT),
    (
        re.compile(r"\bengulf\w*\b|\bpin bar\b|\bdoji\b|\binside bar\b|\bcandlestick\b", re.I),
        Family.PATTERN,
    ),
)


def classify_family(category: str, indicators: Collection[str], prose: str) -> tuple[Family, str]:
    """Decide what a card bets on.

    Evidence is ranked, not blended: the site's own category first, the card's
    prose second, its indicators last. The order is the order of how specific
    the evidence is — every card carries indicators, far fewer carry the word
    "breakout", and the category is a librarian's judgement about the page as a
    whole.

    Args:
        category: The page's site section.
        indicators: Registry names the card resolved to.
        prose: Title and description, for the words the page uses about itself.

    Returns:
        ``(family, reason)``.
    """
    known = _CATEGORY_FAMILY.get(category)
    if known is not None:
        return known, f"page category {category!r}"

    for pattern, family in _PROSE_FAMILY:
        found = pattern.search(prose)
        if found is not None:
            return family, f"the page says {found.group(0)!r}"

    for name, family in _INDICATOR_FAMILY:
        if name in indicators:
            return family, f"the card's indicator {name!r}"

    return Family.TREND, UNPLACED_REASON


def classify_type(category: str, timeframe: Timeframe) -> tuple[StrategyType, str]:
    """Decide how long a position lives.

    The bar size decides, with one override: a page filed under scalping is a
    scalp whatever bar size it names, because the category is a statement about
    holding period and the timeframe is only evidence for one.

    Args:
        category: The page's site section.
        timeframe: The bar size the spec will trade.

    Returns:
        ``(type, reason)``.
    """
    if "scalping" in category:
        return StrategyType.SCALP, f"page category {category!r} names the holding period"
    if timeframe in (Timeframe.M1, Timeframe.M5):
        return StrategyType.SCALP, f"bar size {timeframe.value}"
    if timeframe is Timeframe.D1:
        return StrategyType.POSITION, f"bar size {timeframe.value}"
    if timeframe is Timeframe.H4:
        return StrategyType.SWING, f"bar size {timeframe.value}"
    return StrategyType.INTRADAY, f"bar size {timeframe.value}"
