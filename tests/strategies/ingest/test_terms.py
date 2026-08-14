"""Tests for reading one side of a comparison out of a fragment."""

from trading_system.strategies.ingest.terms import declared_indicators, read_operand
from trading_system.strategies.schema import FeatureRef, LabelSet

NOTHING = declared_indicators(None)


class TestReadingAFeature:
    """An indicator mention becomes a structural reference or nothing."""

    def test_reads_an_indicator_with_its_parameters_written_out(self) -> None:
        result = read_operand("the RSI (14)", NOTHING)
        assert result.operand == FeatureRef(indicator="rsi", params={"period": 14})

    def test_reads_parameters_in_the_order_the_card_writes_them(self) -> None:
        result = read_operand("Stochastic (14, 5, 5) %K", NOTHING)
        assert result.operand == FeatureRef(
            indicator="stoch",
            params={"k_period": 14, "k_smooth": 5, "d_period": 5},
            channel="k",
        )

    def test_a_mention_without_parameters_borrows_the_cards_declaration(self) -> None:
        declared = declared_indicators("RSI (21) indicator;")
        result = read_operand("RSI", declared)
        assert result.operand == FeatureRef(indicator="rsi", params={"period": 21})
        assert result.assumptions  # the borrowing is recorded, not silent

    def test_a_mention_without_parameters_and_without_a_declaration_is_refused(self) -> None:
        # Falling back on the constructor default would make "RSI" mean
        # period=14 on a page that never said 14.
        result = read_operand("RSI", NOTHING)
        assert not result.ok
        assert result.problem is not None and "declares none" in result.problem

    def test_two_declarations_of_one_indicator_make_a_bare_mention_ambiguous(self) -> None:
        declared = declared_indicators("EMA (12)\nEMA (34)")
        assert not read_operand("the EMA", declared).ok

    def test_a_spelled_out_mention_is_checked_against_the_declarations(self) -> None:
        # "20 SMA" looks unambiguous until the card turns out to run a 20-period
        # SMA of highs and a 20-period SMA of lows. Numbers match both.
        declared = declared_indicators("20 SMA High;\n20 SMA Low;")
        result = read_operand("20SMA", declared)
        assert not result.ok
        assert result.problem is not None and "does not say which" in result.problem

    def test_an_indicator_named_with_a_price_is_that_indicator_of_that_price(self) -> None:
        declared = declared_indicators("200 EMA, High.\n200 EMA, Low.")
        result = read_operand("200 EMA High", declared)
        assert result.operand == FeatureRef(
            indicator="ema", params={"period": 200, "source": "high"}
        )

    def test_a_wrong_parameter_count_is_refused_rather_than_padded(self) -> None:
        assert not read_operand("Bollinger Bands (20) upper band", NOTHING).ok


class TestChannels:
    """Which line of a multi-output indicator a rule means."""

    def test_a_named_line_is_used(self) -> None:
        result = read_operand("the MACD (12, 26, 9) signal line", NOTHING)
        assert result.operand == FeatureRef(
            indicator="macd",
            params={"fast_period": 12, "slow_period": 26, "signal_period": 9},
            channel="signal",
        )

    def test_a_line_the_indicator_does_not_have_is_refused(self) -> None:
        assert not read_operand("RSI (14) upper band", NOTHING).ok

    def test_an_indicator_with_no_default_line_must_be_told_which(self) -> None:
        # "the bollinger bands are rising" says nothing about which of three
        # lines, so there is no reading to prefer.
        result = read_operand("Bollinger Bands (20, 2)", NOTHING)
        assert not result.ok
        assert result.problem is not None and "names none" in result.problem

    def test_a_line_alone_names_the_declared_indicator_it_belongs_to(self) -> None:
        declared = declared_indicators("ADX (13) with +DI and -DI")
        result = read_operand("DI+", declared)
        assert result.operand == FeatureRef(
            indicator="adx", params={"period": 13}, channel="plus_di"
        )

    def test_a_line_two_indicators_share_stays_unreadable(self) -> None:
        # "upper band" is Bollinger, Keltner and Donchian alike; with neither
        # declared there is nothing to narrow it.
        assert not read_operand("the upper band", NOTHING).ok


class TestOtherOperands:
    """Prices, constants and labels."""

    def test_reads_a_price_field(self) -> None:
        assert read_operand("the closing price", NOTHING).operand == "price:close"

    def test_reads_a_bare_number(self) -> None:
        assert read_operand("25", NOTHING).operand == 25.0

    def test_reads_the_zero_line_as_zero(self) -> None:
        assert read_operand("the zero line", NOTHING).operand == 0.0

    def test_reads_a_candlestick_pattern_as_a_label(self) -> None:
        assert read_operand("a bullish engulfing", NOTHING).operand == LabelSet(
            labels=("BULLISH_ENGULFING",)
        )


class TestRefusals:
    """One unknown token fails the whole fragment, and says which."""

    def test_names_the_word_it_did_not_know(self) -> None:
        result = read_operand("Parabolic Sar dot", NOTHING)
        assert not result.ok
        assert result.problem is not None and "parabolic" in result.problem

    def test_two_subjects_in_one_fragment_are_refused(self) -> None:
        assert not read_operand("RSI (14) and the CCI (20)", NOTHING).ok


class TestDeclarations:
    """Reading the card's own indicator list."""

    def test_keeps_the_source_apart_so_two_declarations_stay_two(self) -> None:
        declared = declared_indicators("20 SMA High;\n20 SMA Low;")
        assert declared.by_key["sma"] == [
            {"period": 20, "source": "high"},
            {"period": 20, "source": "low"},
        ]

    def test_ignores_a_line_it_cannot_read_rather_than_guessing_at_it(self) -> None:
        # The list only ever supplies parameters a rule omitted; a line naming
        # some custom indicator must not stop a card converting.
        declared = declared_indicators("Super Passband Filter (120, 140, 50)\nRSI (21)")
        assert set(declared.by_key) == {"rsi"}
