"""Reading scraped strategy pages into strategy specs, or refusing to.

The pipeline is four steps, each of which may only narrow what survives:
:mod:`card` types the scraped file, :mod:`text` cuts its prose into clauses,
:mod:`terms` and :mod:`rules` read a clause into a condition using the closed
vocabulary in :mod:`lexicon`, and :mod:`convert` assembles what was read into a
:class:`~trading_system.strategies.schema.StrategySpec` — or into a list of
stated reasons it could not. :mod:`triage` runs the whole corpus and counts the
reasons.

Nothing here guesses. A word outside the vocabulary, a rule needing another
timeframe, a stop given only as a pip distance: each ends the conversion of
that card with a reason attached. The cost is a low conversion rate; the thing
bought is that a spec produced here says what its page said.
"""

from trading_system.strategies.ingest.card import ScrapedCard, load_card, load_cards
from trading_system.strategies.ingest.convert import (
    Conversion,
    Refusal,
    RefusalCode,
    convert_card,
)
from trading_system.strategies.ingest.overrides import CardOverride, load_overrides
from trading_system.strategies.ingest.triage import CorpusReport, render, triage

__all__ = [
    "CardOverride",
    "Conversion",
    "CorpusReport",
    "Refusal",
    "RefusalCode",
    "ScrapedCard",
    "convert_card",
    "load_card",
    "load_cards",
    "load_overrides",
    "render",
    "triage",
]
