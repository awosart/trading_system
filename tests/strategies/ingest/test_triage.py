"""Tests for counting a whole corpus."""

from trading_system.exit.library import known_exit_ids
from trading_system.strategies.ingest.card import ScrapedCard
from trading_system.strategies.ingest.convert import RefusalCode
from trading_system.strategies.ingest.triage import render, triage

from .conftest import card

EXIT_IDS = known_exit_ids()


def _corpus() -> tuple[ScrapedCard, ...]:
    """A small corpus with one of each interesting outcome."""
    return (
        card(),
        card(strategy_id="visual", sections={**card().sections, "Buy": "Buy on the blue arrow."}),
        card(strategy_id="silent", timeframe_raw=None),
    )


class TestCounts:
    """Two different questions, two different numbers."""

    def test_a_card_is_counted_once_under_the_obstacle_that_stopped_it_first(self) -> None:
        report = triage(_corpus(), EXIT_IDS)
        assert sum(report.primary_counts.values()) == report.total - len(report.converted)

    def test_affects_counts_every_obstacle_a_card_has(self) -> None:
        # The visual card also fails to read its own Buy sentence; "affects"
        # sees both, "first" sees only the visual rule.
        report = triage(_corpus(), EXIT_IDS)
        assert report.any_counts[RefusalCode.VISUAL_RULE] == 1
        assert report.any_counts[RefusalCode.UNREADABLE_CLAUSE] >= 1
        assert report.primary_counts[RefusalCode.UNREADABLE_CLAUSE] == 0

    def test_blocked_only_by_is_the_number_that_says_what_a_fix_would_buy(self) -> None:
        # The silent card lacks nothing but a timeframe, so supplying one
        # converts it; the visual card would still need its rules rewritten.
        report = triage(_corpus(), EXIT_IDS)
        only_timeframe = report.blocked_only_by([RefusalCode.NO_TIMEFRAME])
        assert [conversion.card_id for conversion in only_timeframe] == ["silent"]


class TestReviewShortlist:
    """Which refusals a human could answer, and which nobody can."""

    def test_lists_the_card_whose_rules_already_read(self) -> None:
        report = triage(_corpus(), EXIT_IDS)
        assert [conversion.card_id for conversion in report.review_shortlist] == ["silent"]

    def test_leaves_out_a_card_refused_for_its_rules(self) -> None:
        report = triage(_corpus(), EXIT_IDS)
        assert "visual" not in {conversion.card_id for conversion in report.review_shortlist}


class TestRender:
    """The report is text a person reads, so it says the three numbers."""

    def test_prints_a_line_per_obstacle_with_all_three_counts(self) -> None:
        text = render(triage(_corpus(), EXIT_IDS), examples=1)
        assert "cards examined      3" in text
        assert "no_timeframe" in text
        assert "review shortlist" in text
