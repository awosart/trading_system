"""The scraped card as it lands on disk, before anything interprets it.

A card is prose that happens to be stored as JSON: a title, a description, and
a handful of free-text sections a human would read as trading rules. Nothing
here parses those rules — this module only makes the file's shape a typed
object so that a missing section is an absent key rather than a ``KeyError``
in the middle of conversion, and so that a scrape whose layout changes fails
at load time rather than silently converting fewer cards.

Section headers are the one place a card's layout leaks in: the same rule is
filed under ``"Buy"``, ``"Long"``, ``"Enter Long:"`` or ``"Rules for a Long
Trade"`` depending on the page. :data:`DIRECTION_HEADERS` maps the ones that
genuinely name a side; anything else stays unclaimed, because a header nobody
recognised must read as "this card has no rules for that side" rather than as
rules quietly taken from the wrong section.
"""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from trading_system.strategies.schema import Direction

#: Section headers that name a trading side, lower-cased and stripped of
#: punctuation for lookup. Long-side and short-side spellings only: a header
#: like ``"Entry"`` or ``"Trading Rules"`` describes both sides at once and is
#: deliberately absent, since splitting such a section into legs is exactly the
#: guess this pipeline refuses to make.
DIRECTION_HEADERS: dict[str, Direction] = {
    "buy": Direction.LONG,
    "long": Direction.LONG,
    "longs": Direction.LONG,
    "long entries": Direction.LONG,
    "long trades": Direction.LONG,
    "long trade": Direction.LONG,
    "enter long": Direction.LONG,
    "enter long position": Direction.LONG,
    "rules for a long trade": Direction.LONG,
    "buy signal": Direction.LONG,
    "buy rules": Direction.LONG,
    "sell": Direction.SHORT,
    "short": Direction.SHORT,
    "shorts": Direction.SHORT,
    "short entries": Direction.SHORT,
    "short trades": Direction.SHORT,
    "short trade": Direction.SHORT,
    "enter short": Direction.SHORT,
    "enter short position": Direction.SHORT,
    "rules for a short trade": Direction.SHORT,
    "sell signal": Direction.SHORT,
    "sell rules": Direction.SHORT,
}

#: Section headers that carry where a trade is given up on. Read in this order:
#: a dedicated stop section outranks the general exit prose.
STOP_HEADERS: tuple[str, ...] = ("stop loss", "stop-loss", "stops", "exit position", "take profit")

#: Section headers that carry how a trade is taken off.
EXIT_HEADERS: tuple[str, ...] = ("exit position", "take profit", "take-profit", "stop loss")

#: Section headers that carry entry rules without naming a side. A card with
#: one of these is describing conditions for both legs at once, and the
#: converter reads none of them: it cannot tell which leg a shared sentence
#: belongs to. What it can do is notice that they exist, so that converting the
#: ``Buy`` section of a card whose ``Trading Rules`` section holds three more
#: conditions is a refusal rather than a spec missing three conditions.
UNSIDED_RULE_HEADERS: tuple[str, ...] = (
    "entry",
    "entries",
    "entry rules",
    "2. entry rules",
    "trading rules",
    "rules",
    "the rules",
    "system rules",
    "trade rules",
    "how to trade",
    "conditions",
    "strategy setup",
    "set up",
    "setup",
    "chart set-up",
    "setting up your charts",
    "filter",
    "long and short entries",
    "long and short triggers",
    "the strategy",
)


def _header_key(header: str) -> str:
    """Normalise a section header for lookup: lower-cased, no trailing punctuation."""
    return header.strip().strip(":*.- ").lower()


class ScrapedCard(BaseModel):
    """One scraped strategy page, exactly as the scraper wrote it.

    Attributes:
        source: Site the card came from.
        source_url: Page the card was scraped from.
        strategy_id: Slug the scraper assigned, unique within one scrape.
        category: Site section the page was filed under.
        title: Page title; frequently empty.
        description: Lead paragraph.
        instruments_raw: Instruments line, unparsed.
        timeframe_raw: Timeframe line, unparsed.
        indicators_raw: Indicator list, unparsed.
        sections: Section header to its raw text.
        download_links: Attachments offered on the page.
        parse_confidence: The scraper's own view of how well it read the page.
        parse_warnings: What the scraper could not find.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: str
    source_url: str
    strategy_id: str
    category: str
    title: str
    description: str
    instruments_raw: str | None
    timeframe_raw: str | None
    indicators_raw: str | None
    sections: dict[str, str]
    download_links: list[str]
    parse_warnings: list[str]
    parse_confidence: str

    def rules_for(self, direction: Direction) -> str | None:
        """The card's own words for how to enter ``direction``.

        Args:
            direction: Side to look up.

        Returns:
            The section text, or ``None`` when the card files no section under
            a header naming that side. Several matching sections are joined,
            since a page occasionally repeats the same side under two headers.
        """
        parts = [
            text
            for header, text in self.sections.items()
            if DIRECTION_HEADERS.get(_header_key(header)) is direction and text.strip()
        ]
        return "\n".join(parts) if parts else None

    def section_text(self, headers: tuple[str, ...]) -> str | None:
        """The first non-empty section whose header matches, in ``headers`` order.

        Args:
            headers: Normalised headers to try, most specific first.

        Returns:
            The section text, or ``None`` when the card carries none of them.
        """
        by_key = {_header_key(header): text for header, text in self.sections.items()}
        for header in headers:
            text = by_key.get(header)
            if text and text.strip():
                return text
        return None

    def unsided_rule_sections(self) -> tuple[tuple[str, str], ...]:
        """Sections carrying entry rules that name no side.

        Returns:
            ``(header, text)`` for each, in the card's own order.
        """
        return tuple(
            (header, text)
            for header, text in self.sections.items()
            if _header_key(header) in UNSIDED_RULE_HEADERS and text.strip()
        )

    @property
    def has_direction_sections(self) -> bool:
        """Whether the card names entry rules for at least one side."""
        return any(
            _header_key(header) in DIRECTION_HEADERS and text.strip()
            for header, text in self.sections.items()
        )


def load_card(path: Path) -> ScrapedCard:
    """Read one scraped card.

    Args:
        path: JSON file the scraper wrote.

    Returns:
        The parsed card.

    Raises:
        pydantic.ValidationError: If the file does not match the scrape's shape.
    """
    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    return ScrapedCard.model_validate(payload)


def load_cards(directory: Path) -> Iterator[tuple[Path, ScrapedCard]]:
    """Read every card in ``directory``, in a stable order.

    Args:
        directory: Directory of scraped card JSON files.

    Yields:
        Each file and the card it holds, ordered by filename so that two runs
        over the same directory produce the same report.
    """
    for path in sorted(directory.glob("*.json")):
        yield path, load_card(path)
