"""Cards built by hand, so no test depends on a scrape being on disk.

The scraped corpus is local data, not a fixture: it is not in the repository
and its contents change with every re-scrape. Every test here therefore builds
the card it needs, which also makes each test say in its own body what shape of
page it is about.
"""

from typing import Any

from trading_system.strategies.ingest.card import ScrapedCard

#: A card whose rules the grammar can read end to end: both sides, plain
#: comparisons against declared indicators, a stop naming a level, a 2R target.
READABLE_CARD: dict[str, Any] = {
    "source": "example.com",
    "source_url": "https://example.com/ema-adx",
    "strategy_id": "ema-and-adx",
    "category": "trend-following-forex-strategies",
    "title": "EMA and ADX Trading System",
    "description": "A trend filter and a strength filter.",
    "instruments_raw": "EUR/USD",
    "timeframe_raw": "15 min or higher.",
    "indicators_raw": "EMA (50)\nADX (14)",
    "sections": {
        "Indicators": "EMA (50)\nADX (14)",
        "Buy": "Price above the EMA (50).\nADX (14) > 25.",
        "Sell": "Price below the EMA (50).\nADX (14) > 25.",
        "Exit position": "Place initial stop loss below/above the previous swing high/low.\n"
        "Profit target ratio 1:2.",
    },
    "download_links": [],
    "parse_warnings": [],
    "parse_confidence": "high",
}


def card(**changes: Any) -> ScrapedCard:
    """A readable card with ``changes`` applied.

    Args:
        **changes: Fields to replace. ``sections`` replaces the whole mapping,
            so a test that wants different rules states them in full.

    Returns:
        The card.
    """
    payload = {**READABLE_CARD, **changes}
    return ScrapedCard.model_validate(payload)


def sections(**parts: str) -> dict[str, str]:
    """Section mapping written as keyword arguments.

    Args:
        **parts: Section text keyed by header with spaces written as
            underscores, e.g. ``Exit_position``.

    Returns:
        The mapping with headers restored.
    """
    return {header.replace("_", " "): text for header, text in parts.items()}
