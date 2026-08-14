"""Tests for turning a whole card into a spec, or into stated refusals."""

from pathlib import Path

import pytest

from trading_system.core.types import Timeframe
from trading_system.exit.library import known_exit_ids
from trading_system.strategies.ingest.convert import RefusalCode, convert_card
from trading_system.strategies.ingest.overrides import CardOverride, load_overrides
from trading_system.strategies.schema import Direction, FeatureRef, StrategyType
from trading_system.strategies.validator import Severity, validate_spec

from .conftest import card, sections

EXIT_IDS = known_exit_ids()


def _codes(**changes: object) -> set[RefusalCode]:
    """Every refusal a card with ``changes`` produces."""
    conversion = convert_card(card(**changes), EXIT_IDS)
    return {refusal.code for refusal in conversion.refusals}


class TestAConvertedCard:
    """The one path that produces a spec."""

    def test_produces_a_spec_that_passes_the_ordinary_validator(self) -> None:
        conversion = convert_card(card(), EXIT_IDS)
        assert conversion.spec is not None, conversion.refusals
        issues = [
            issue
            for issue in validate_spec(conversion.spec, known_exit_ids=EXIT_IDS)
            if issue.severity is Severity.ERROR
        ]
        assert issues == []

    def test_both_sides_become_legs_of_one_spec(self) -> None:
        spec = convert_card(card(), EXIT_IDS).spec
        assert spec is not None
        assert {entry.direction for entry in spec.entries} == {Direction.LONG, Direction.SHORT}

    def test_the_stop_level_the_card_names_becomes_the_invalidation(self) -> None:
        spec = convert_card(card(), EXIT_IDS).spec
        assert spec is not None
        long_leg = spec.entry_for(Direction.LONG)
        assert long_leg is not None
        level = long_leg.invalidation.price_level
        assert isinstance(level, FeatureRef)
        # "below/above the previous swing high/low" resolves per leg: the long
        # leg is given up on below the swing low.
        assert (level.indicator, level.channel) == ("swing", "swing_low")

    def test_the_timeframe_floor_the_card_states_is_the_one_used(self) -> None:
        spec = convert_card(card(), EXIT_IDS).spec
        assert spec is not None
        assert spec.timeframes.signal_tf is Timeframe.M15
        assert spec.timeframes.entry_tf is Timeframe.M15
        assert spec.timeframes.htf_filter_tf is None

    def test_the_instruments_the_card_names_become_an_allow_list(self) -> None:
        spec = convert_card(card(), EXIT_IDS).spec
        assert spec is not None
        assert spec.instruments.allowed_symbols == ["EURUSD"]

    def test_holding_period_class_is_derived_and_says_so(self) -> None:
        conversion = convert_card(card(), EXIT_IDS)
        assert conversion.spec is not None
        assert conversion.spec.type is StrategyType.INTRADAY
        assert any("holding-period class" in note for note in conversion.assumptions)

    def test_every_reading_the_card_did_not_spell_out_is_recorded(self) -> None:
        # The three encoding decisions this pipeline is allowed to make must be
        # visible on the conversion, not buried in a module constant.
        conversion = convert_card(card(), EXIT_IDS)
        joined = " ".join(conversion.assumptions)
        assert "stop_reference" in joined
        assert "base_quality" in joined
        assert "confirmation" in joined


class TestPartialReadingsNeverHappen:
    """A card is converted whole or refused whole."""

    def test_one_unreadable_sentence_refuses_the_card_it_sits_in(self) -> None:
        # The other sentence of this Buy section reads perfectly. Keeping it
        # and dropping the rest would produce a spec that trades a strategy
        # nobody wrote — the failure this whole pipeline exists to avoid.
        conversion = convert_card(
            card(
                sections=sections(
                    Indicators="EMA (50)",
                    Buy="Price above the EMA (50).\nThe price bounces off the support zone.",
                    Sell="Price below the EMA (50).",
                    Exit_position="Stop loss below/above the previous swing high/low. Ratio 1:2.",
                )
            ),
            EXIT_IDS,
        )
        assert conversion.spec is None
        assert RefusalCode.UNREADABLE_CLAUSE in {r.code for r in conversion.refusals}

    def test_a_section_of_rules_naming_no_side_refuses_the_card(self) -> None:
        # Converting the Buy section of a card whose Trading Rules section holds
        # three more conditions would silently drop three conditions.
        codes = _codes(
            sections={
                **card().sections,
                "Trading Rules": "The CCI (20) must be above 100.",
            }
        )
        assert RefusalCode.UNREAD_SECTION in codes


class TestSchemaLimits:
    """The three limitations CLAUDE.md already records, found by phrase."""

    def test_a_rule_about_another_timeframe_is_refused_as_such(self) -> None:
        assert RefusalCode.CROSS_TIMEFRAME in _codes(
            sections={**card().sections, "Buy": "Price above the EMA (50) on the daily chart."}
        )

    def test_arithmetic_over_bar_fields_is_refused_as_such(self) -> None:
        assert RefusalCode.BAR_ARITHMETIC in _codes(
            sections={**card().sections, "Buy": "The body of the candle is above the EMA (50)."}
        )

    def test_a_regime_gate_is_refused_as_such(self) -> None:
        assert RefusalCode.REGIME in _codes(
            sections={**card().sections, "Buy": "In a trending market, price above the EMA (50)."}
        )

    def test_a_blocker_in_an_unsided_section_still_counts(self) -> None:
        # A card whose Buy section is clean and whose Entry section says "on the
        # 60 min chart" is a cross-timeframe strategy all the same.
        assert RefusalCode.CROSS_TIMEFRAME in _codes(
            sections={**card().sections, "Entry": "5EMA > 100 EMA on the 60 min chart."}
        )


class TestContentThisSystemDoesNotHave:
    """Indicators and chart objects with no implementation here."""

    def test_an_off_registry_indicator_is_named_in_the_refusal(self) -> None:
        conversion = convert_card(
            card(sections={**card().sections, "Buy": "Parabolic SAR below the price."}), EXIT_IDS
        )
        detail = next(
            r.detail for r in conversion.refusals if r.code is RefusalCode.OFF_REGISTRY_INDICATOR
        )
        assert "parabolic sar" in detail

    def test_a_rule_about_something_drawn_on_the_chart_is_refused(self) -> None:
        assert RefusalCode.VISUAL_RULE in _codes(
            sections={**card().sections, "Buy": "Buy on the blue arrow."}
        )


class TestWhatTheCardNeverSaid:
    """Silence about something the schema demands is a refusal too."""

    def test_a_card_with_no_side_sections_is_refused_immediately(self) -> None:
        conversion = convert_card(
            card(sections=sections(Indicators="EMA (50)", Trading_Rules="Buy in an uptrend.")),
            EXIT_IDS,
        )
        assert {r.code for r in conversion.refusals} == {RefusalCode.NO_RULES}

    def test_no_timeframe_at_all_is_refused(self) -> None:
        assert RefusalCode.NO_TIMEFRAME in _codes(timeframe_raw=None)

    def test_two_timeframes_are_refused_rather_than_resolved(self) -> None:
        assert RefusalCode.NO_TIMEFRAME in _codes(timeframe_raw="4H and Daily")

    def test_a_timeframe_this_system_has_no_bar_size_for_is_refused_by_name(self) -> None:
        # M30 is not rounded onto M15 or H1: the card was written for bars this
        # system cannot build.
        conversion = convert_card(card(timeframe_raw="30 min or higher."), EXIT_IDS)
        detail = next(r.detail for r in conversion.refusals if r.code is RefusalCode.NO_TIMEFRAME)
        assert "30 min" in detail

    def test_a_stop_given_only_as_a_distance_is_refused_and_says_why(self) -> None:
        # An invalidation is an absolute level, and there is no operand for
        # "entry price minus thirty pips".
        conversion = convert_card(
            card(
                sections={
                    **card().sections,
                    "Exit position": "Place a stop loss of 30 pips. Profit target ratio 1:2.",
                }
            ),
            EXIT_IDS,
        )
        detail = next(
            r.detail for r in conversion.refusals if r.code is RefusalCode.NO_INVALIDATION
        )
        assert "30 pips" in detail

    def test_a_target_ratio_the_library_has_no_preset_for_is_refused_not_rounded(self) -> None:
        # Rounding 1:1.3 onto the 2R preset would change the strategy: a fixed
        # target and a trail have already been measured to differ materially.
        conversion = convert_card(
            card(
                sections={
                    **card().sections,
                    "Exit position": "Stop loss below/above the previous swing high/low. "
                    "Profit target ratio 1:1.3.",
                }
            ),
            EXIT_IDS,
        )
        assert conversion.spec is None
        detail = next(
            r.detail for r in conversion.refusals if r.code is RefusalCode.EXIT_NOT_IN_LIBRARY
        )
        assert "1:1.3" in detail


class TestRefusalReporting:
    """Every obstacle is kept, and one of them is called the reason."""

    def test_a_card_reports_every_obstacle_not_only_the_first(self) -> None:
        conversion = convert_card(
            card(
                timeframe_raw=None,
                sections={**card().sections, "Buy": "Price above the EMA (50) on the daily chart."},
            ),
            EXIT_IDS,
        )
        codes = {r.code for r in conversion.refusals}
        assert {RefusalCode.CROSS_TIMEFRAME, RefusalCode.NO_TIMEFRAME} <= codes

    def test_the_reason_reported_is_the_limitation_not_the_symptom(self) -> None:
        conversion = convert_card(
            card(
                timeframe_raw=None,
                sections={**card().sections, "Buy": "Price above the EMA (50) on the daily chart."},
            ),
            EXIT_IDS,
        )
        assert conversion.primary is RefusalCode.CROSS_TIMEFRAME


class TestOverrides:
    """What a reviewer may add, and what no reviewer may add."""

    def test_a_reviewer_may_supply_a_timeframe_the_card_never_stated(self) -> None:
        override = CardOverride(
            card_id="ema-and-adx",
            reviewer="tester",
            note="the page's screenshots are all H1",
            timeframe=Timeframe.H1,
        )
        conversion = convert_card(card(timeframe_raw=None), EXIT_IDS, override)
        assert conversion.spec is not None
        assert conversion.spec.timeframes.signal_tf is Timeframe.H1

    def test_a_supplied_value_travels_with_the_spec_as_the_reviewers(self) -> None:
        override = CardOverride(
            card_id="ema-and-adx",
            reviewer="tester",
            note="the card names no exit at all",
            exit_ref="breakeven_runner",
        )
        conversion = convert_card(
            card(
                sections={
                    **card().sections,
                    "Exit position": "Stop loss below/above the previous swing high/low.",
                }
            ),
            EXIT_IDS,
            override,
        )
        assert conversion.spec is not None
        assert conversion.spec.exit_ref == "breakeven_runner"
        joined = " ".join(conversion.assumptions)
        assert "tester" in joined and "a pairing, not something the card said" in joined

    def test_no_override_makes_an_unreadable_rule_readable(self) -> None:
        # This is the line that keeps the pipeline worth having: a converted
        # trigger says what its page said, and nothing a reviewer writes can
        # change a trigger.
        override = CardOverride(
            card_id="ema-and-adx",
            reviewer="tester",
            note="I know what it means",
            timeframe=Timeframe.H1,
        )
        conversion = convert_card(
            card(sections={**card().sections, "Buy": "Price bounces off the support zone."}),
            EXIT_IDS,
            override,
        )
        assert conversion.spec is None

    def test_a_dismissed_section_stops_blocking_and_stops_blocker_scanning(self) -> None:
        # A Trading Rules section that restates the sided rules in prose blocks
        # twice: once as unread rules, once through whatever phrase it uses.
        rules = "Trading only in the direction of the trend. Price above the EMA (50) = up trend."
        without = _codes(sections={**card().sections, "Trading Rules": rules})
        assert {RefusalCode.UNREAD_SECTION, RefusalCode.REGIME} <= without

        override = CardOverride(
            card_id="ema-and-adx",
            reviewer="tester",
            note="the section restates the Buy rule and adds nothing",
            dismiss_sections=("Trading Rules",),
        )
        conversion = convert_card(
            card(sections={**card().sections, "Trading Rules": rules}), EXIT_IDS, override
        )
        assert conversion.spec is not None, conversion.refusals


class TestLoadingOverrides:
    """Overrides come from files, one per card."""

    def test_two_files_claiming_one_card_is_an_error(self, tmp_path: Path) -> None:
        # Otherwise which reading applies depends on filesystem order.
        for name in ("a.json", "b.json"):
            (tmp_path / name).write_text(
                '{"card_id": "x", "reviewer": "t", "note": "n"}', encoding="utf-8"
            )
        with pytest.raises(ValueError, match="two overrides claim"):
            load_overrides(tmp_path)

    def test_a_missing_directory_is_simply_no_overrides(self, tmp_path: Path) -> None:
        assert load_overrides(tmp_path / "nothing") == {}
