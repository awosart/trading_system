"""Tests for normalising a card by filling in what its page never said.

The strict pipeline's tests assert that nothing is invented. These assert the
opposite discipline: that everything invented is *recorded*, that the reading
chosen is the defensible one, and that a spec produced here still passes the
same validator every other spec in the system passes.
"""

from pathlib import Path

import pytest

from trading_system.core.config import Settings
from trading_system.core.instruments import InstrumentClass, load_instruments
from trading_system.core.types import Timeframe
from trading_system.exit.library import known_exit_ids
from trading_system.strategies.normalize.classify import Family, classify_family, classify_type
from trading_system.strategies.normalize.coverage import MarketCoverage, cost_points
from trading_system.strategies.normalize.inference import (
    InferenceCode,
    family_archetype,
    invalidation_level,
    read_ratio,
    snap_exit_ratio,
)
from trading_system.strategies.normalize.normalise import Fidelity, normalise_card
from trading_system.strategies.normalize.salvage import salvage_clause, sniff_indicators
from trading_system.strategies.normalize.write import write_corpus
from trading_system.strategies.schema import (
    ConditionOp,
    Direction,
    FeatureRef,
    LeafCondition,
    StrategyType,
)
from trading_system.strategies.validator import Severity, validate_spec

from .conftest import card

EXIT_IDS = known_exit_ids()


def _normalise(coverage: MarketCoverage, **changes: object):
    """One card through the normaliser, with the shared exit library."""
    return normalise_card(card(**changes), coverage=coverage, known_exit_ids=EXIT_IDS)


class TestTheUniverseComesFromTheStore:
    """A spec may only name instruments the store can actually serve."""

    def test_a_volume_reading_spec_excludes_a_zero_volume_series(
        self, coverage: MarketCoverage
    ) -> None:
        allowed, refused = coverage.admissible(Timeframe.H1, needs_volume=True)
        assert "XAUUSD" not in allowed
        assert refused["XAUUSD"] == "no_volume"

    def test_the_same_series_is_offered_when_the_spec_reads_no_volume(
        self, coverage: MarketCoverage
    ) -> None:
        allowed, _refused = coverage.admissible(Timeframe.H1, needs_volume=False)
        assert "XAUUSD" in allowed

    def test_a_symbol_whose_cost_is_most_of_its_bar_is_refused(
        self, coverage: MarketCoverage
    ) -> None:
        allowed, refused = coverage.admissible(Timeframe.M1, needs_volume=False)
        assert "USDCAD" not in allowed
        assert refused["USDCAD"] == "spread_dominates"

    def test_a_symbol_the_store_does_not_hold_is_neither_allowed_nor_refused(
        self, coverage: MarketCoverage
    ) -> None:
        allowed, refused = coverage.admissible(Timeframe.H1, needs_volume=False)
        assert "GBPUSD" not in allowed
        assert "GBPUSD" not in refused

    def test_the_cost_of_a_round_turn_counts_commission_as_well_as_spread(self) -> None:
        eurusd = load_instruments(Settings().instruments_path)["EURUSD"]
        # 0.8 points of spread plus 7.00 per lot round turn over a 10.00
        # point value: the commission is not a rounding error on this pair.
        assert cost_points(eurusd) == pytest.approx(1.5, abs=0.01)


class TestClassification:
    """Two axes, each derived from the strongest evidence the page carries."""

    def test_the_pages_own_category_outranks_its_prose(self) -> None:
        family, reason = classify_family("breakout-forex-strategies", {"rsi"}, "an overbought fade")
        assert family is Family.BREAKOUT
        assert "category" in reason

    def test_prose_decides_when_the_category_is_not_one_of_the_known_ones(self) -> None:
        family, reason = classify_family(
            "forex-strategies-based-on-indicators", {"ema"}, "A breakout of the range"
        )
        assert family is Family.BREAKOUT
        assert "the page says" in reason

    def test_a_scalping_page_is_a_scalp_whatever_bar_size_it_names(self) -> None:
        chosen, reason = classify_type("scalping-forex-strategies", Timeframe.D1)
        assert chosen is StrategyType.SCALP
        assert "category" in reason

    def test_otherwise_the_bar_size_decides(self) -> None:
        assert classify_type("breakout-forex-strategies", Timeframe.H4)[0] is StrategyType.SWING
        assert classify_type("breakout-forex-strategies", Timeframe.M5)[0] is StrategyType.SCALP


class TestTheExitIsChosenAndTheDistanceIsStated:
    """Rounding a target onto a preset changes the strategy, so it is named."""

    def test_an_exact_ratio_says_so(self) -> None:
        name, detail = snap_exit_ratio(2.0, EXIT_IDS)
        assert name == "conservative_2r"
        assert "exactly" in detail

    def test_an_inexact_ratio_reports_how_far_it_moved(self) -> None:
        name, detail = snap_exit_ratio(1.3, EXIT_IDS)
        assert name == "rr_1_25r"
        assert "0.05R" in detail
        assert "changes what is traded" in detail

    def test_a_ratio_is_read_written_either_way_round(self) -> None:
        assert read_ratio("target risk to reward 1:1.5 or better") == 1.5
        assert read_ratio("a 3:1 reward to risk") == 3.0
        assert read_ratio("no target given") is None


class TestTheInvalidationIsInferredInTheRightOrder:
    """A trigger's own line beats a generic swing, and both are recorded."""

    def test_a_trigger_line_is_preferred_over_structure(self) -> None:
        level, code, detail = invalidation_level(["supertrend"], Direction.LONG)
        assert code is InferenceCode.INVALIDATION_FROM_TRIGGER_LINE
        assert isinstance(level, FeatureRef)
        assert level.indicator == "supertrend"
        assert "stops holding" in detail

    def test_structure_is_the_fallback_and_says_it_changes_the_strategy(self) -> None:
        level, code, detail = invalidation_level(["rsi"], Direction.LONG)
        assert code is InferenceCode.INVALIDATION_FROM_SWING
        assert isinstance(level, FeatureRef)
        assert level.channel == "swing_low"
        assert "different places" in detail

    def test_the_side_of_the_level_follows_the_leg(self) -> None:
        long_level, _code, _detail = invalidation_level(["bbands"], Direction.LONG)
        short_level, _code, _detail = invalidation_level(["bbands"], Direction.SHORT)
        assert isinstance(long_level, FeatureRef) and long_level.channel == "lower"
        assert isinstance(short_level, FeatureRef) and short_level.channel == "upper"


class TestSalvage:
    """Reading a sentence by comparison alone, and knowing when not to."""

    def test_it_reads_a_sentence_the_strict_grammar_refuses(self) -> None:
        found = salvage_clause("ADX reading >=25", Direction.LONG)
        assert found is not None
        condition, reading = found
        assert isinstance(condition, LeafCondition)
        assert condition.op is ConditionOp.GTE
        assert condition.right == 25.0
        assert "salvaged" in reading

    def test_a_bare_price_level_is_not_a_rule(self) -> None:
        # "price above 1.2500" is a level off someone's chart: it would carry
        # onto another instrument as a condition that is always or never true.
        assert salvage_clause("buy when price is above 1.2500", Direction.LONG) is None

    def test_a_band_takes_the_side_the_comparison_points_at(self) -> None:
        above = salvage_clause("close above the bollinger band", Direction.LONG)
        below = salvage_clause("close below the bollinger band", Direction.SHORT)
        assert above is not None and below is not None
        assert isinstance(above[0], LeafCondition) and isinstance(above[0].right, FeatureRef)
        assert above[0].right.channel == "upper"
        assert isinstance(below[0], LeafCondition) and isinstance(below[0].right, FeatureRef)
        assert below[0].right.channel == "lower"

    def test_a_sentence_with_no_comparison_is_not_salvaged(self) -> None:
        assert salvage_clause("wait for the market to calm down", Direction.LONG) is None

    def test_indicators_are_sniffed_out_of_ordinary_prose(self) -> None:
        assert sniff_indicators("A 50 EMA with the Stochastic and MACD") == [
            "ema",
            "stoch",
            "macd",
        ]


class TestTheArchetypeIsLabelledAsInvention:
    """A rule written here is never presented as something the page said."""

    def test_a_card_with_no_rules_still_produces_a_spec(self, coverage: MarketCoverage) -> None:
        result = _normalise(
            coverage,
            sections={"Indicators": "RSI (14)"},
            description="",
            indicators_raw="RSI (14)",
        )
        assert result.normalised
        assert result.fidelity is Fidelity.ARCHETYPE
        codes = {inference.code for inference in result.inferences}
        assert InferenceCode.ARCHETYPE_FROM_INDICATORS in codes

    def test_the_family_flips_what_an_oscillator_means(self) -> None:
        from trading_system.strategies.normalize.inference import archetype_conditions

        following, _notes = archetype_conditions(["rsi"], Direction.LONG, mean_reverting=False)
        fading, _notes = archetype_conditions(["rsi"], Direction.LONG, mean_reverting=True)
        assert isinstance(following[0], LeafCondition)
        assert following[0].op is ConditionOp.CROSS_ABOVE
        assert isinstance(fading[0], LeafCondition)
        assert fading[0].op is ConditionOp.LT

    def test_a_donchian_archetype_compares_against_the_previous_bars_channel(self) -> None:
        from trading_system.strategies.normalize.inference import archetype_conditions

        conditions, notes = archetype_conditions(["donchian"], Direction.LONG, False)
        assert isinstance(conditions[0], LeafCondition)
        right = conditions[0].right
        assert isinstance(right, FeatureRef)
        # Without the shift the comparison is an identity of the indicator and
        # can never fire: CLAUDE.md records this as the reason shift exists.
        assert right.shift == 1
        assert any("never fire" in note for note in notes)

    def test_a_candlestick_page_with_nothing_else_fires_on_the_label(self) -> None:
        found = family_archetype("PATTERN", Direction.LONG, "Trading the pin bar at support")
        assert found is not None
        conditions, reading = found
        assert isinstance(conditions[0], LeafCondition)
        assert conditions[0].op is ConditionOp.PATTERN_IS
        assert "pin bar" in reading

    def test_a_family_with_no_canonical_form_returns_nothing(self) -> None:
        assert family_archetype("PATTERN", Direction.LONG, "no candle named here") is None


class TestWhatANormalisedSpecGuarantees:
    """Whatever was inferred, the result is a spec like any other."""

    def test_it_passes_the_ordinary_validator(self, coverage: MarketCoverage) -> None:
        result = _normalise(coverage)
        assert result.spec is not None, result.refusal
        errors = [
            issue
            for issue in validate_spec(result.spec, known_exit_ids=EXIT_IDS)
            if issue.severity is Severity.ERROR
        ]
        assert errors == []

    def test_every_departure_from_the_page_carries_a_reason(self, coverage: MarketCoverage) -> None:
        result = _normalise(coverage)
        assert result.inferences
        assert all(inference.detail.strip() for inference in result.inferences)

    def test_the_timeframe_comes_from_the_category_when_the_card_names_none(
        self, coverage: MarketCoverage
    ) -> None:
        result = _normalise(coverage, timeframe_raw=None)
        assert result.spec is not None
        assert result.spec.timeframes.signal_tf is Timeframe.H4
        codes = {inference.code for inference in result.inferences}
        assert InferenceCode.TIMEFRAME_FROM_CATEGORY in codes

    def test_no_higher_timeframe_filter_is_ever_declared(self, coverage: MarketCoverage) -> None:
        # Cross-timeframe conditions are not supported; a declared filter
        # timeframe would only produce a warning and change nothing.
        result = _normalise(coverage)
        assert result.spec is not None
        assert result.spec.timeframes.htf_filter_tf is None

    def test_a_card_with_nothing_at_all_is_refused_rather_than_invented(
        self, coverage: MarketCoverage
    ) -> None:
        result = _normalise(
            coverage,
            sections={},
            description="",
            indicators_raw=None,
            title="",
            category="unknown-category",
        )
        assert not result.normalised
        assert result.refusal is not None
        assert "no rule" in result.refusal

    def test_the_asset_classes_follow_the_symbols_that_survived(
        self, coverage: MarketCoverage
    ) -> None:
        result = _normalise(coverage)
        assert result.spec is not None
        assert InstrumentClass.FX in result.spec.instruments.allowed_classes
        assert set(result.spec.instruments.allowed_symbols) == set(result.universe)


class TestTheTreeOnDisk:
    """The layout is the sort, and the manifest is the provenance."""

    def test_specs_are_filed_under_type_then_family(
        self, coverage: MarketCoverage, tmp_path: Path
    ) -> None:
        result = _normalise(coverage)
        corpus = write_corpus([result], tmp_path, source="test")
        assert result.spec is not None
        expected = (
            tmp_path
            / "specs"
            / result.spec.type.value
            / result.family.value
            / f"{result.spec.id}.json"
        )
        assert expected.is_file()
        assert corpus.root == tmp_path

    def test_two_cards_normalising_to_one_id_is_an_error_not_an_overwrite(
        self, coverage: MarketCoverage, tmp_path: Path
    ) -> None:
        first = _normalise(coverage)
        second = _normalise(coverage, source_url="https://example.com/other")
        with pytest.raises(ValueError, match="both normalise to spec id"):
            write_corpus([first, second], tmp_path, source="test")

    def test_the_manifest_counts_what_the_index_lists(
        self, coverage: MarketCoverage, tmp_path: Path
    ) -> None:
        import json

        result = _normalise(coverage)
        write_corpus([result], tmp_path, source="test")
        manifest = json.loads((tmp_path / "manifest.json").read_text())
        rows = (tmp_path / "index.csv").read_text().strip().splitlines()
        assert manifest["counts"]["normalised"] == len(rows) - 1
        assert manifest["cards"][0]["inferences"]
