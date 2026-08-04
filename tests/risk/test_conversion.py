"""FX rates: where they come from, and that they are never invented."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import polars as pl
import pytest

from trading_system.core.types import Timeframe
from trading_system.data.models import OHLCVFrame
from trading_system.risk.conversion import (
    BarFxConverter,
    FxRateUnavailableError,
    SameCurrencyConverter,
    StaticFxConverter,
    convert,
)

START = datetime(2024, 1, 2, 9, 0, tzinfo=UTC)


def rate_series(
    closes: list[float],
    *,
    symbol: str = "USDJPY",
    timeframe: Timeframe = Timeframe.H1,
    start: datetime = START,
) -> OHLCVFrame:
    """Build a conversion-pair series with the given closes, one bar each."""
    rows = [
        (start + timeframe.duration * i, close, close, close, close, 0.0)
        for i, close in enumerate(closes)
    ]
    return OHLCVFrame.from_raw(
        pl.DataFrame(
            rows,
            schema=["timestamp", "open", "high", "low", "close", "volume"],
            orient="row",
        ),
        symbol,
        timeframe,
    )


class TestSameCurrencyConverter:
    def test_a_currency_against_itself_is_one(self) -> None:
        assert SameCurrencyConverter().rate(base="USD", quote="USD", at=START) == Decimal(1)

    def test_a_real_pair_is_refused_rather_than_guessed(self) -> None:
        with pytest.raises(FxRateUnavailableError, match="handles no conversions"):
            SameCurrencyConverter().rate(base="JPY", quote="USD", at=START)


class TestStaticFxConverter:
    def test_a_direct_pair_is_returned_as_written(self) -> None:
        converter = StaticFxConverter({("USD", "JPY"): Decimal("150")})
        assert converter.rate(base="USD", quote="JPY", at=START) == Decimal("150")

    def test_the_opposite_direction_is_inverted(self) -> None:
        # So a table need only carry each pair once.
        converter = StaticFxConverter({("USD", "JPY"): Decimal("150")})
        assert converter.rate(base="JPY", quote="USD", at=START) == Decimal(1) / Decimal("150")

    def test_an_absent_pair_is_refused(self) -> None:
        with pytest.raises(FxRateUnavailableError, match="no rate in the static table"):
            StaticFxConverter({}).rate(base="JPY", quote="USD", at=START)

    def test_a_non_positive_rate_is_rejected_at_construction(self) -> None:
        # Not a rate, and inverting it divides by zero.
        with pytest.raises(ValueError, match="must be positive"):
            StaticFxConverter({("USD", "JPY"): Decimal("0")})


class TestBarFxConverterNoLookahead:
    """The rate is the last bar to have CLOSED, never the bar in progress."""

    def test_the_bar_containing_the_instant_is_not_used(self) -> None:
        # Bar 0 opens 09:00 and closes 10:00; bar 1 runs 10:00-11:00. At 10:59
        # bar 1 has not closed, so its close is not knowable and bar 0's is
        # used. This is the same rule that governs signals and exits.
        converter = BarFxConverter({"USDJPY": rate_series([150.0, 151.0, 152.0])})
        at_1059 = datetime(2024, 1, 2, 10, 59, tzinfo=UTC)
        assert converter.rate(base="USD", quote="JPY", at=at_1059) == Decimal("150.0")

    def test_the_instant_a_bar_closes_makes_it_available(self) -> None:
        converter = BarFxConverter({"USDJPY": rate_series([150.0, 151.0, 152.0])})
        at_1100 = datetime(2024, 1, 2, 11, 0, tzinfo=UTC)
        assert converter.rate(base="USD", quote="JPY", at=at_1100) == Decimal("151.0")

    def test_before_the_series_starts_there_is_no_rate(self) -> None:
        converter = BarFxConverter({"USDJPY": rate_series([150.0])})
        with pytest.raises(FxRateUnavailableError, match="no bar had closed"):
            converter.rate(base="USD", quote="JPY", at=START)


class TestBarFxConverterStaleness:
    """Staleness is bar time, which is what makes it usable in a backtest."""

    def test_it_is_measured_from_the_bar_close_not_from_load_time(self) -> None:
        # Historical data loaded today is not stale. Measuring against the wall
        # clock would make every rate in a 2024 backtest infinitely old and
        # refuse the entire run.
        converter = BarFxConverter({"USDJPY": rate_series([150.0])})
        just_after_close = START + Timeframe.H1.duration
        assert converter.rate(base="USD", quote="JPY", at=just_after_close) == Decimal("150.0")

    def test_a_weekend_gap_does_not_reject_monday(self) -> None:
        # The case that would break the whole system on every Monday morning.
        # FX closes around Friday 21:00 UTC and reopens Sunday evening, so a
        # rate requested at Monday 00:00 is 51 hours old through no fault of
        # anyone's. The default bound is three days precisely so that this is
        # a rate and not a gap.
        friday_last_bar_open = datetime(2024, 3, 1, 20, 0, tzinfo=UTC)
        converter = BarFxConverter({"USDJPY": rate_series([150.0], start=friday_last_bar_open)})
        monday = datetime(2024, 3, 4, 0, 0, tzinfo=UTC)
        assert (monday - (friday_last_bar_open + Timeframe.H1.duration)) > timedelta(hours=50)
        assert converter.rate(base="USD", quote="JPY", at=monday) == Decimal("150.0")

    def test_a_series_that_genuinely_stopped_is_caught(self) -> None:
        # Three days clears a weekend; three weeks is a gap in the data.
        converter = BarFxConverter({"USDJPY": rate_series([150.0])})
        with pytest.raises(FxRateUnavailableError, match="staleness bound"):
            converter.rate(base="USD", quote="JPY", at=START + timedelta(days=21))

    def test_the_bound_is_configurable(self) -> None:
        converter = BarFxConverter(
            {"USDJPY": rate_series([150.0])}, max_staleness=timedelta(hours=2)
        )
        with pytest.raises(FxRateUnavailableError, match="staleness bound"):
            converter.rate(base="USD", quote="JPY", at=START + timedelta(hours=4))

    def test_a_non_positive_bound_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            BarFxConverter({}, max_staleness=timedelta(0))


class TestBarFxConverterPairs:
    def test_only_one_direction_of_a_pair_need_be_loaded(self) -> None:
        converter = BarFxConverter({"USDJPY": rate_series([150.0])})
        at = START + Timeframe.H1.duration
        assert converter.rate(base="JPY", quote="USD", at=at) == Decimal(1) / Decimal("150.0")

    def test_a_pair_that_is_not_loaded_is_refused_by_name(self) -> None:
        converter = BarFxConverter({"USDJPY": rate_series([150.0])})
        with pytest.raises(FxRateUnavailableError, match="neither EURGBP nor GBPEUR"):
            converter.rate(base="EUR", quote="GBP", at=START)

    def test_a_currency_against_itself_needs_no_series(self) -> None:
        assert BarFxConverter({}).rate(base="USD", quote="USD", at=START) == Decimal(1)


class TestConvert:
    def test_the_same_currency_is_returned_untouched(self) -> None:
        # And without consulting the converter, so a run whose universe is all
        # account-currency-quoted needs no rate source at all.
        amount = Decimal("123.45")
        assert (
            convert(
                amount,
                from_currency="USD",
                to_currency="USD",
                at=START,
                converter=SameCurrencyConverter(),
            )
            == amount
        )

    def test_a_real_conversion_multiplies_by_the_rate(self) -> None:
        converter = StaticFxConverter({("JPY", "USD"): Decimal("0.006")})
        assert convert(
            Decimal("1000"),
            from_currency="JPY",
            to_currency="USD",
            at=START,
            converter=converter,
        ) == Decimal("6.000")
