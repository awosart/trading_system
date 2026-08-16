"""Normalising a whole scrape into strategy specs, filling in what pages omit.

The strict reader in :mod:`trading_system.strategies.ingest` answers "what does
this page say" and refuses whenever the answer is incomplete. This package
answers a different question — "what is the most defensible strategy this page
implies, on data we hold" — and therefore supplies missing values instead of
refusing. The two are meant to coexist: a spec from ``ingest`` is evidence
about a page, a spec from here is a testable candidate whose every departure
from its page is recorded on it.

Nothing here grants a strategy anything. A normalised spec lands outside
``strategies/library`` with no bookkeeping, no runs and no verdict, and the
gate on ``APPROVED`` is unchanged.
"""

from trading_system.strategies.normalize.classify import Family, classify_family, classify_type
from trading_system.strategies.normalize.coverage import (
    MarketCoverage,
    SeriesCoverage,
    measure_coverage,
)
from trading_system.strategies.normalize.inference import Inference, InferenceCode
from trading_system.strategies.normalize.normalise import (
    Fidelity,
    Normalisation,
    normalise_card,
)
from trading_system.strategies.normalize.write import NormalisedCorpus, write_corpus

__all__ = [
    "Family",
    "Fidelity",
    "Inference",
    "InferenceCode",
    "MarketCoverage",
    "NormalisedCorpus",
    "Normalisation",
    "SeriesCoverage",
    "classify_family",
    "classify_type",
    "measure_coverage",
    "normalise_card",
    "write_corpus",
]
