"""Tests for reading one clause into one condition."""

from trading_system.strategies.ingest.rules import is_noise, read_clause
from trading_system.strategies.ingest.terms import declared_indicators
from trading_system.strategies.schema import ConditionOp, LabelSet, LeafCondition

NOTHING = declared_indicators(None)
DECLARED = declared_indicators("EMA (50)\nRSI (14)\nMACD (12, 26, 9)")


def _leaf(clause: str) -> LeafCondition:
    """The leaf a clause reads into, failing the test if it does not read."""
    result = read_clause(clause, DECLARED)
    assert result.condition is not None, result.problem
    assert isinstance(result.condition, LeafCondition)
    return result.condition


class TestComparators:
    """Plain comparisons, and the crossings that contain them."""

    def test_above_is_a_state_and_crosses_above_is_an_event(self) -> None:
        # "closes above" is a level test; "crosses above" is a transition. The
        # crossing forms are searched first because each one contains the other.
        assert _leaf("Price closes above the EMA").op is ConditionOp.GT
        assert _leaf("Price crosses above the EMA").op is ConditionOp.CROSS_ABOVE

    def test_reads_a_symbol_comparison(self) -> None:
        leaf = _leaf("RSI > 50")
        assert leaf.op is ConditionOp.GT
        assert leaf.right == 50.0

    def test_takes_the_side_of_a_crossing_from_a_word_at_the_end(self) -> None:
        # "crosses the middle band upwards" puts the direction after the target,
        # which is how half the corpus writes it.
        assert _leaf("Price crosses the EMA upwards").op is ConditionOp.CROSS_ABOVE

    def test_a_crossing_naming_both_sides_is_refused(self) -> None:
        result = read_clause(
            "the CCI crosses its mid line in either direction up or down", DECLARED
        )
        assert result.condition is None and result.problem is not None

    def test_a_crossing_naming_no_side_is_refused(self) -> None:
        # "MACD crosses" is a crossing of what, in which direction? Both
        # readings change the strategy, so neither is chosen.
        result = read_clause("the MACD crosses", DECLARED)
        assert result.condition is None


class TestLabels:
    """Clauses that name a kind of bar rather than compare two series."""

    def test_a_pattern_name_alone_is_a_pattern_test(self) -> None:
        result = read_clause("a bullish engulfing", NOTHING)
        assert result.condition == LeafCondition(
            op=ConditionOp.PATTERN_IS, left=None, right=LabelSet(labels=("BULLISH_ENGULFING",))
        )

    def test_a_label_cannot_be_compared_numerically(self) -> None:
        result = read_clause("a doji above the hammer", NOTHING)
        assert result.condition is None


class TestNoise:
    """Captions are dropped; sentences that failed to read are not."""

    def test_a_caption_is_dropped_without_comment(self) -> None:
        assert is_noise("In the pictures below")
        assert read_clause("In the pictures below", NOTHING).noise

    def test_a_repeat_of_the_page_title_is_a_caption(self) -> None:
        assert is_noise("Supertrend with EMA Channel", title="Supertrend with EMA Channel")

    def test_a_rule_whose_words_are_all_in_the_title_is_still_a_rule(self) -> None:
        # Regression: a clause carrying a number or a comparison is a candidate
        # rule and must be read or refused, never dropped for resembling the
        # title. "ADX (14) > 25" on a page called "EMA and ADX Trading System"
        # was being dropped, and the spec came out with the filter missing.
        assert not is_noise("ADX (14) > 25", title="EMA and ADX Trading System")

    def test_a_sentence_it_could_not_read_is_a_problem_not_noise(self) -> None:
        result = read_clause("Price bounces off the support zone", NOTHING)
        assert not result.noise
        assert result.problem is not None
