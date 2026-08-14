"""Tests for cutting scraped prose into clauses."""

from trading_system.strategies.ingest.text import clauses, contains_any, normalise, reflow


class TestNormalise:
    """Folds the characters the pages actually contain."""

    def test_strips_the_markdown_the_scraper_left_behind(self) -> None:
        assert normalise("**Price above the EMA**") == "Price above the EMA"

    def test_folds_the_non_breaking_spaces_the_pages_are_full_of(self) -> None:
        # Every second card carries U+00A0 between words; a scanner that treats
        # it as a word character never matches a phrase table entry again.
        assert normalise("Ema's\xa0(12) >\xa050") == "Ema's (12) > 50"

    def test_keeps_line_structure_because_reflow_needs_it(self) -> None:
        assert normalise("one\n\ntwo") == "one\ntwo"


class TestReflow:
    """Re-joins lines the page wrapped mid-sentence."""

    def test_a_lower_case_line_continues_the_one_above(self) -> None:
        assert reflow("Price above Ma simple 60\nperiod.") == ["Price above Ma simple 60 period."]

    def test_a_capitalised_line_starts_a_new_rule_even_without_punctuation(self) -> None:
        # The pages list one rule per line and punctuate none of them; joining
        # on missing full stops would glue two rules into one unreadable clause.
        assert reflow("Candle above 200 EMA\nSupertrend line below the price") == [
            "Candle above 200 EMA",
            "Supertrend line below the price",
        ]

    def test_a_digit_line_continues_because_a_sentence_never_starts_with_one(self) -> None:
        assert reflow("Price above\n14 EMA") == ["Price above 14 EMA"]


class TestClauses:
    """Splits a section into the smallest things that could each be a rule."""

    def test_splits_on_sentence_punctuation_and_conjunctions(self) -> None:
        assert clauses("RSI > 50 and Stochastic > 50. MACD > 0") == (
            "RSI > 50",
            "Stochastic > 50",
            "MACD > 0",
        )

    def test_never_cuts_inside_a_parameter_list(self) -> None:
        # "Stochastic (14, 5, 5)" holds two commas that are not separators, and
        # a splitter that took them would leave "5)" as a clause of its own.
        assert clauses("Stochastic (14, 5, 5) above 50") == ("Stochastic (14, 5, 5) above 50",)

    def test_a_comma_between_two_rules_does_separate_them(self) -> None:
        assert clauses("8SMA>18SMA, MACD>0") == ("8SMA>18SMA", "MACD>0")


class TestContainsAny:
    """Finding marker phrases without firing on words that merely contain them."""

    def test_a_word_marker_does_not_fire_inside_a_longer_word(self) -> None:
        # "ma" inside "market" once made every card look like it named a moving
        # average; markers made of word characters are matched on boundaries.
        assert contains_any("the market is quiet", ("ma",)) == ()

    def test_reports_the_phrases_it_found_in_the_order_given(self) -> None:
        assert contains_any("uses an arrow and a dot", ("dot", "arrow")) == ("dot", "arrow")
