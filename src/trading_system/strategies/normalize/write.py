"""Laying a normalised corpus out on disk, sorted, with its own provenance beside it.

Two files answer two different questions and neither answers the other's. The
spec says what would be traded; the manifest entry says where each part of it
came from — which sentence of which page, or which inference in place of one.
Keeping them apart is the same decision the strategy library already makes by
splitting ``{id}.json`` from ``{id}.meta.json``: a digest over a spec has to be
a digest over what is traded, and provenance is not that.

The directory layout is the sort. ``specs/{TYPE}/{FAMILY}/`` puts a holding
period above what the entry bets on, because those are the two axes a reader
browses by and 883 files in one directory are not browsable at all.
"""

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from trading_system.strategies.normalize.normalise import Fidelity, Normalisation

#: Written into the manifest so a reader knows which run produced the tree.
MANIFEST_FILE = "manifest.json"

#: A flat, sortable view of the same corpus, for the questions a manifest is
#: awkward for: how many of each family, which cards are archetypes, what the
#: universe of each is.
INDEX_FILE = "index.csv"


@dataclass(frozen=True)
class NormalisedCorpus:
    """The outcome of normalising a whole scrape.

    Attributes:
        results: Every card's outcome, in input order.
        root: Where the tree was written.
    """

    results: tuple[Normalisation, ...]
    root: Path

    @property
    def normalised(self) -> tuple[Normalisation, ...]:
        """The cards that produced a spec."""
        return tuple(result for result in self.results if result.normalised)

    @property
    def refused(self) -> tuple[Normalisation, ...]:
        """The cards that produced none."""
        return tuple(result for result in self.results if not result.normalised)

    def by_fidelity(self, fidelity: Fidelity) -> tuple[Normalisation, ...]:
        """Every produced spec at one grade of faithfulness to its page."""
        return tuple(result for result in self.normalised if result.fidelity is fidelity)


def _entry(result: Normalisation) -> dict[str, object]:
    """One manifest record for one card."""
    spec = result.spec
    return {
        "card_id": result.card_id,
        "normalised": result.normalised,
        "spec_id": spec.id if spec is not None else None,
        "path": _relative_path(result) if spec is not None else None,
        "type": spec.type.value if spec is not None else None,
        "family": result.family.value,
        "fidelity": result.fidelity.value,
        "timeframe": spec.timeframes.signal_tf.value if spec is not None else None,
        "exit_ref": spec.exit_ref if spec is not None else None,
        "legs": [entry.direction.value for entry in spec.entries] if spec is not None else [],
        "universe": list(result.universe),
        "refused_symbols": dict(result.refused_symbols),
        "inferences": [
            {"code": inference.code.value, "detail": inference.detail}
            for inference in result.inferences
        ],
        "dropped_clauses": list(result.dropped),
        "refusal": result.refusal,
    }


def _relative_path(result: Normalisation) -> str:
    """Where a produced spec sits under ``specs/``."""
    assert result.spec is not None
    return f"specs/{result.spec.type.value}/{result.family.value}/{result.spec.id}.json"


def write_corpus(results: Sequence[Normalisation], root: Path, *, source: str) -> NormalisedCorpus:
    """Write every produced spec, the manifest, and the flat index.

    Args:
        results: What normalisation produced, in input order.
        root: Directory to write under; created if absent.
        source: Where the cards came from, recorded in the manifest so a tree
            can be traced back to the scrape that produced it.

    Returns:
        The corpus, with ``root`` set.

    Raises:
        ValueError: If two cards produced the same spec id, which would make
            one silently overwrite the other.
    """
    seen: dict[str, str] = {}
    for result in results:
        if result.spec is None:
            continue
        clash = seen.get(result.spec.id)
        if clash is not None:
            raise ValueError(
                f"cards {clash!r} and {result.card_id!r} both normalise to spec id "
                f"{result.spec.id!r}"
            )
        seen[result.spec.id] = result.card_id

    root.mkdir(parents=True, exist_ok=True)
    for result in results:
        if result.spec is None:
            continue
        path = root / _relative_path(result)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            result.spec.model_dump_json(indent=2, exclude_none=True) + "\n", encoding="utf-8"
        )

    produced = [result for result in results if result.normalised]
    manifest = {
        "generated": datetime.now(UTC).date().isoformat(),
        "source": source,
        "purpose": (
            "Every card of the scrape rendered as a StrategySpec, with what the page did not "
            "say filled in and recorded. Not library records and not measurements: nothing here "
            "has been run, and a spec graded 'archetype' states a rule its page never wrote."
        ),
        "counts": {
            "cards": len(results),
            "normalised": len(produced),
            "refused": len(results) - len(produced),
            "by_fidelity": {
                fidelity.value: sum(1 for result in produced if result.fidelity is fidelity)
                for fidelity in Fidelity
            },
            "by_type": _tally(result.spec.type.value for result in produced if result.spec),
            "by_family": _tally(result.family.value for result in produced),
        },
        "cards": [_entry(result) for result in results],
    }
    (root / MANIFEST_FILE).write_text(
        json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    header = "card_id,spec_id,type,family,fidelity,timeframe,legs,exit_ref,universe\n"
    rows = [
        ",".join(
            (
                result.card_id,
                result.spec.id,
                result.spec.type.value,
                result.family.value,
                result.fidelity.value,
                result.spec.timeframes.signal_tf.value,
                str(len(result.spec.entries)),
                result.spec.exit_ref,
                " ".join(result.universe),
            )
        )
        for result in produced
        if result.spec is not None
    ]
    (root / INDEX_FILE).write_text(header + "\n".join(rows) + "\n", encoding="utf-8")
    return NormalisedCorpus(results=tuple(results), root=root)


def _tally(values: Iterable[str]) -> dict[str, int]:
    """Count occurrences, most frequent first, as a plain dict."""
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items(), key=lambda pair: (-pair[1], pair[0])))
