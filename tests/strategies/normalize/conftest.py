"""Fixtures for the lenient normaliser: cards and a store that are not on disk.

The scraped corpus and the parquet store are both local data that changes; a
test that depended on either would pass or fail for reasons unrelated to the
code. Every card here is built in the test that needs it, and the coverage is
constructed rather than measured.
"""

from datetime import UTC, datetime
from typing import Any

import pytest

from trading_system.core.instruments import InstrumentClass
from trading_system.core.types import Timeframe
from trading_system.strategies.ingest.card import ScrapedCard
from trading_system.strategies.normalize.coverage import MarketCoverage, SeriesCoverage

#: A card whose rules the strict grammar reads end to end.
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
        "Buy": "Price above the EMA (50).",
        "Sell": "Price below the EMA (50).",
        "Exit position": "Place initial stop loss below/above the previous swing high/low.\n"
        "Profit target ratio 1:2.",
    },
    "download_links": [],
    "parse_warnings": [],
    "parse_confidence": "high",
}


def card(**changes: Any) -> ScrapedCard:
    """A readable card with ``changes`` applied."""
    return ScrapedCard.model_validate({**READABLE_CARD, **changes})


def series(
    symbol: str,
    timeframe: Timeframe,
    *,
    median_range_points: float,
    cost_points: float,
    has_volume: bool = True,
    asset_class: InstrumentClass = InstrumentClass.FX,
) -> SeriesCoverage:
    """One measured series, stated rather than measured."""
    return SeriesCoverage(
        symbol=symbol,
        asset_class=asset_class,
        timeframe=timeframe,
        bars=1000,
        start=datetime(2020, 1, 1, tzinfo=UTC),
        end=datetime(2024, 1, 1, tzinfo=UTC),
        median_range_points=median_range_points,
        has_volume=has_volume,
        day_anchor_ok=True,
        cost_points=cost_points,
    )


@pytest.fixture
def coverage() -> MarketCoverage:
    """Three symbols with the three properties that decide a universe.

    ``EURUSD`` is tradeable and carries volume, ``XAUUSD`` is tradeable and does
    not, ``USDCAD`` carries volume and is drowned by its own spread — one
    symbol per branch of
    :meth:`~trading_system.strategies.normalize.coverage.MarketCoverage.admissible`.
    """
    found = {}
    for timeframe in Timeframe:
        found[("EURUSD", timeframe)] = series(
            "EURUSD", timeframe, median_range_points=10.0, cost_points=1.5
        )
        found[("XAUUSD", timeframe)] = series(
            "XAUUSD",
            timeframe,
            median_range_points=200.0,
            cost_points=25.0,
            has_volume=False,
            asset_class=InstrumentClass.COMMODITY,
        )
        found[("USDCAD", timeframe)] = series(
            "USDCAD", timeframe, median_range_points=1.0, cost_points=1.3
        )
    return MarketCoverage(series=found)
