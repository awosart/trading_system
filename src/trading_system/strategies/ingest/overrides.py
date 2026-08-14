"""What a human reviewer may add to a card, and what they may not.

A refusal is an invitation to review, not a verdict about the idea: the
converter says "this card never states a timeframe" and a person who has read
the page can answer it. That answer belongs in a file, next to the card, signed
and explained — not in the converter's tables, where it would silently become a
rule about every other card.

The line is drawn by what an override may carry. It may **supply a value the
card leaves missing** — the timeframe, the exit pairing, the level a stop sits
at, the instruments — and it may **dismiss a section** the reviewer has read and
found to restate the sided rules rather than add to them. It may not touch the
entry logic: if a rule sentence does not parse, no override makes it parse, and
the card stays refused. That is deliberate. The value of this pipeline is that
a converted trigger says what its page said, and an override that could rewrite
a trigger would remove exactly that guarantee while keeping the appearance of it.

Every override carries a reviewer and a note, both mandatory. The note is what
makes the override auditable: it is expected to quote the card, and it travels
onto the converted spec's bookkeeping as an assumption, so the fact that a
human supplied the exit is visible wherever the spec is.
"""

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from trading_system.core.types import Timeframe
from trading_system.strategies.schema import Direction, InstrumentScope, Operand


class CardOverride(BaseModel):
    """A reviewer's answer to what one card left unsaid.

    Attributes:
        card_id: The card this answers, matched against
            :attr:`~trading_system.strategies.ingest.card.ScrapedCard.strategy_id`.
        reviewer: Who read the page and takes responsibility for the reading.
        note: Why these values are faithful to the card, quoting it where the
            card is the evidence.
        timeframe: Bar size to use when the card names none, several, or one
            this system has no bar size for.
        exit_ref: Exit preset to pair the entry with. Recorded as the
            reviewer's pairing, never as something the card said.
        invalidation: Price level per side, for a card whose stop names a level
            the grammar cannot reach.
        instruments: Scope to use instead of the default FX-wide one.
        dismiss_sections: Headers of unsided rule sections the reviewer has
            read and found to restate the sided rules. Their text is then
            excluded from the blocker scan too — a section that adds nothing
            cannot be what blocks the card.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    card_id: str = Field(min_length=1)
    reviewer: str = Field(min_length=1)
    note: str = Field(min_length=1)
    timeframe: Timeframe | None = None
    exit_ref: str | None = None
    invalidation: dict[Direction, Operand] = Field(default_factory=dict)
    instruments: InstrumentScope | None = None
    dismiss_sections: tuple[str, ...] = ()

    def provenance(self) -> tuple[str, ...]:
        """One line per value supplied, for the converted spec's assumptions.

        Returns:
            The lines, each naming the reviewer and what they supplied.
        """
        supplied: list[str] = []
        if self.timeframe is not None:
            supplied.append(f"timeframe {self.timeframe.value}")
        if self.exit_ref is not None:
            supplied.append(f"exit_ref {self.exit_ref!r} (a pairing, not something the card said)")
        for direction, level in self.invalidation.items():
            supplied.append(f"{direction.value} invalidation {level!r}")
        if self.instruments is not None:
            supplied.append("instrument scope")
        if self.dismiss_sections:
            supplied.append(
                f"dismissed sections {list(self.dismiss_sections)} as restating the rules"
            )
        if not supplied:
            return ()
        return (
            f"reviewer {self.reviewer} supplied: {'; '.join(supplied)}",
            f"reviewer's reasoning: {self.note}",
        )


def load_overrides(directory: Path | None) -> dict[str, CardOverride]:
    """Read every override in ``directory``, keyed by card id.

    Args:
        directory: Directory of override JSON files, or ``None`` for none.

    Returns:
        Card id to override.

    Raises:
        ValueError: If two files claim the same card, which would make the
            applied reading depend on filesystem order.
        pydantic.ValidationError: If a file does not match the model.
    """
    if directory is None or not directory.is_dir():
        return {}
    overrides: dict[str, CardOverride] = {}
    for path in sorted(directory.glob("*.json")):
        override = CardOverride.model_validate(json.loads(path.read_text(encoding="utf-8")))
        if override.card_id in overrides:
            raise ValueError(f"two overrides claim card {override.card_id!r}: {path} is the second")
        overrides[override.card_id] = override
    return overrides
