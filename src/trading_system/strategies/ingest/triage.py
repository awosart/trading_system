"""Counting what a corpus of scraped cards is, and what stops each one.

The report answers one question — how much of this corpus the current schema
can express — and refuses to answer it with a single number. A card is counted
under the obstacle that stopped it first (:data:`REFUSAL_PRECEDENCE`), and also
under every obstacle it has, because those are different quantities: the first
says what to fix now, the second says what the corpus would still need after
that fix. A card blocked only by a missing exit preset is a different finding
from one blocked by rules written about arrows.
"""

from collections import Counter
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass

from trading_system.strategies.ingest.card import ScrapedCard
from trading_system.strategies.ingest.convert import (
    REFUSAL_GROUPS,
    Conversion,
    RefusalCode,
    convert_card,
)
from trading_system.strategies.ingest.overrides import CardOverride

#: Refusals a human reviewer is allowed to answer, by supplying a value the
#: card never stated or by judging an unsided section to restate the rules.
#: Everything outside this set is about the rules themselves and is not
#: reviewable: see :class:`~trading_system.strategies.ingest.overrides.CardOverride`.
REVIEWABLE_CODES: tuple[RefusalCode, ...] = (
    RefusalCode.NO_TIMEFRAME,
    RefusalCode.NO_INVALIDATION,
    RefusalCode.EXIT_UNSTATED,
    RefusalCode.EXIT_NOT_IN_LIBRARY,
    RefusalCode.UNREAD_SECTION,
)


@dataclass(frozen=True)
class CorpusReport:
    """What a whole scrape turned out to be.

    Attributes:
        conversions: Every card's outcome, in input order.
    """

    conversions: tuple[Conversion, ...]

    @property
    def total(self) -> int:
        """How many cards were examined."""
        return len(self.conversions)

    @property
    def converted(self) -> tuple[Conversion, ...]:
        """The cards that produced a spec."""
        return tuple(c for c in self.conversions if c.converted)

    @property
    def primary_counts(self) -> Counter[RefusalCode]:
        """How many cards each obstacle stopped first."""
        counts: Counter[RefusalCode] = Counter()
        for conversion in self.conversions:
            code = conversion.primary
            if code is not None:
                counts[code] += 1
        return counts

    @property
    def any_counts(self) -> Counter[RefusalCode]:
        """How many cards each obstacle affects at all, first or not."""
        counts: Counter[RefusalCode] = Counter()
        for conversion in self.conversions:
            for code in {refusal.code for refusal in conversion.refusals}:
                counts[code] += 1
        return counts

    def blocked_only_by(self, codes: Sequence[RefusalCode]) -> tuple[Conversion, ...]:
        """Cards whose every obstacle is in ``codes``.

        This is the actionable number: how many cards a single change would
        unlock, as opposed to how many merely mention the obstacle.

        Args:
            codes: The obstacles to test against.

        Returns:
            The matching cards, in input order.
        """
        wanted = set(codes)
        return tuple(
            conversion
            for conversion in self.conversions
            if conversion.refusals and {r.code for r in conversion.refusals} <= wanted
        )

    @property
    def review_shortlist(self) -> tuple[Conversion, ...]:
        """Cards a reviewer could finish, and nobody else can.

        Every rule sentence of these cards already reads; what is missing is
        something an override may legitimately supply — the timeframe, the
        exit pairing, the level a stop sits at, or a judgement that an unsided
        section restates the rules. Cards refused for anything else are not
        here, because no amount of review makes an unreadable sentence
        readable.

        Returns:
            The candidates, in input order.
        """
        return self.blocked_only_by(REVIEWABLE_CODES)

    def examples(self, code: RefusalCode, limit: int = 3) -> tuple[tuple[str, str], ...]:
        """A few cards stopped first by ``code``, with the detail that stopped them.

        Args:
            code: The obstacle.
            limit: How many to return.

        Returns:
            ``(card_id, detail)`` pairs.
        """
        out: list[tuple[str, str]] = []
        if limit <= 0:
            return ()
        for conversion in self.conversions:
            if conversion.primary is not code:
                continue
            detail = next(r.detail for r in conversion.refusals if r.code is code)
            out.append((conversion.card_id, detail))
            if len(out) == limit:
                break
        return tuple(out)


def triage(
    cards: Iterable[ScrapedCard],
    known_exit_ids: Collection[str] | None = None,
    overrides: Mapping[str, CardOverride] | None = None,
) -> CorpusReport:
    """Run every card through the converter and collect the outcomes.

    Args:
        cards: The cards to examine.
        known_exit_ids: Exit preset ids ``exit_ref`` is checked against.
        overrides: Reviewer answers, keyed by card id.

    Returns:
        The report.
    """
    supplied = overrides or {}
    return CorpusReport(
        tuple(convert_card(card, known_exit_ids, supplied.get(card.strategy_id)) for card in cards)
    )


def render(report: CorpusReport, examples: int = 2) -> str:
    """Render a report as plain text.

    Args:
        report: The report to render.
        examples: How many example cards to show per obstacle.

    Returns:
        The rendered report.
    """
    lines: list[str] = []
    converted = len(report.converted)
    lines.append(f"cards examined      {report.total}")
    lines.append(f"converted           {converted}")
    lines.append(f"refused             {report.total - converted}")
    lines.append("")
    primary = report.primary_counts
    any_counts = report.any_counts
    for group, codes in REFUSAL_GROUPS.items():
        total = sum(primary.get(code, 0) for code in codes)
        lines.append(f"[{group}] {total} card(s) stopped here first")
        for code in codes:
            first = primary.get(code, 0)
            anywhere = any_counts.get(code, 0)
            alone = len(report.blocked_only_by([code]))
            lines.append(
                f"  {code.value:24} first {first:4}   affects {anywhere:4}   only reason {alone:4}"
            )
            for card_id, detail in report.examples(code, examples):
                lines.append(f"      {card_id}: {detail[:110]}")
        lines.append("")
    shortlist = report.review_shortlist
    lines.append(f"[review shortlist] {len(shortlist)} card(s) a reviewer could finish")
    for conversion in shortlist:
        missing = sorted({refusal.code.value for refusal in conversion.refusals})
        lines.append(f"  {conversion.card_id}: needs {', '.join(missing)}")
    return "\n".join(lines)
