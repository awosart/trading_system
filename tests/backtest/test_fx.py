"""Assembling a run's rate source, and the two things it must not get wrong.

The wiring is small; what it has to protect is not. A run that needs no
conversion must keep the converter it always had, because the converter is
folded into ``run_id`` and swapping an equivalent one renumbers stored history.
A run that needs a series the store lacks must stop, not accumulate a refusal
per bar.
"""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import polars as pl
import pytest

from trading_system.backtest.fx import (
    ConversionSeriesMissingError,
    build_converter,
    load_conversion_series,
)
from trading_system.core.instruments import load_instruments
from trading_system.core.types import Timeframe
from trading_system.data.models import OHLCVFrame
from trading_system.risk.conversion import (
    BarFxConverter,
    SameCurrencyConverter,
    conversion_pair_candidates,
    required_conversion_currencies,
)

REGISTRY = load_instruments(Path("configs/instruments.yaml"))


def _frame(symbol: str, close: float, *, bars: int = 30) -> OHLCVFrame:
    """A flat series, enough for a rate lookup to succeed."""
    start = datetime(2024, 1, 8, tzinfo=UTC)
    return OHLCVFrame.from_raw(
        pl.DataFrame(
            {
                "timestamp": [start + timedelta(hours=4 * i) for i in range(bars)],
                "open": [close] * bars,
                "high": [close] * bars,
                "low": [close] * bars,
                "close": [close] * bars,
                "volume": [1.0] * bars,
            }
        ),
        symbol=symbol,
        timeframe=Timeframe.H4,
    )


def _loader(available: dict[str, OHLCVFrame]):
    """A store stand-in: an empty frame means the symbol is not carried."""

    def load(symbol: str, timeframe: Timeframe) -> OHLCVFrame:
        return available.get(symbol) or OHLCVFrame.from_raw(
            pl.DataFrame(
                schema={
                    "timestamp": pl.Datetime(time_unit="us", time_zone="UTC"),
                    "open": pl.Float64,
                    "high": pl.Float64,
                    "low": pl.Float64,
                    "close": pl.Float64,
                    "volume": pl.Float64,
                }
            ),
            symbol=symbol,
            timeframe=timeframe,
        )

    return load


class TestWhichCurrenciesNeedConverting:
    def test_only_the_quote_currency_counts_never_the_base(self) -> None:
        # GBPJPY on a USD account needs JPY/USD. It does not need GBP/USD:
        # one point is worth point_size x contract_size in the QUOTE currency,
        # and the base currency never enters the sizing arithmetic.
        assert required_conversion_currencies([REGISTRY["GBPJPY"]], account_currency="USD") == (
            "JPY",
        )

    def test_a_usd_quoted_instrument_needs_nothing(self) -> None:
        assert required_conversion_currencies([REGISTRY["NAS100"]], account_currency="USD") == ()

    def test_the_result_is_sorted_so_a_converter_is_built_the_same_way_twice(self) -> None:
        specs = [REGISTRY[s] for s in ("GBPJPY", "EURGBP", "USDCHF", "EURUSD")]
        assert required_conversion_currencies(specs, account_currency="USD") == (
            "CHF",
            "GBP",
            "JPY",
        )

    def test_both_spellings_are_offered_because_the_market_publishes_only_one(self) -> None:
        assert conversion_pair_candidates("JPY", "USD") == ("JPYUSD", "USDJPY")
        assert conversion_pair_candidates("USD", "USD") == ()


class TestBuildingTheConverter:
    def test_a_run_needing_no_conversion_keeps_same_currency_converter(self) -> None:
        """Not an empty BarFxConverter: identity must move only when behaviour does.

        RunInputs.components() folds the converter into run_id, so returning a
        different-but-equivalent object here would renumber every EURUSD run
        already stored without changing a single decision.
        """
        converter = build_converter(
            [REGISTRY["EURUSD"], REGISTRY["NAS100"]],
            account_currency="USD",
            timeframe=Timeframe.H4,
            load=_loader({}),
        )
        assert isinstance(converter, SameCurrencyConverter)

    def test_a_jpy_quoted_instrument_gets_a_bar_converter_off_the_inverse_pair(self) -> None:
        converter = build_converter(
            [REGISTRY["USDJPY"]],
            account_currency="USD",
            timeframe=Timeframe.H4,
            load=_loader({"USDJPY": _frame("USDJPY", 150.0)}),
        )
        assert isinstance(converter, BarFxConverter)
        # JPY priced in USD is 1/150: the pair is published as USDJPY and the
        # converter inverts it rather than needing a JPYUSD series to exist.
        rate = converter.rate(base="JPY", quote="USD", at=datetime(2024, 1, 9, tzinfo=UTC))
        assert rate == pytest.approx(Decimal(1) / Decimal("150.0"))

    def test_one_series_covers_every_instrument_sharing_a_quote_currency(self) -> None:
        converter = build_converter(
            [REGISTRY["USDJPY"], REGISTRY["GBPJPY"], REGISTRY["EURJPY"]],
            account_currency="USD",
            timeframe=Timeframe.H4,
            load=_loader({"USDJPY": _frame("USDJPY", 150.0)}),
        )
        assert isinstance(converter, BarFxConverter)

    def test_a_missing_series_stops_the_run_and_names_both_spellings(self) -> None:
        with pytest.raises(ConversionSeriesMissingError, match="CHFUSD nor USDCHF"):
            build_converter(
                [REGISTRY["USDCHF"]],
                account_currency="USD",
                timeframe=Timeframe.H4,
                load=_loader({}),
            )

    def test_a_non_usd_quoted_index_converts_like_any_fx_pair(self) -> None:
        """GER40 is quoted in EUR, so sizing it on a USD account reads EURUSD.

        The converter has no notion of asset class — only of quote currency —
        which is why an index needed no new machinery.
        """
        converter = build_converter(
            [REGISTRY["GER40"]],
            account_currency="USD",
            timeframe=Timeframe.H4,
            load=_loader({"EURUSD": _frame("EURUSD", 1.10)}),
        )
        rate = converter.rate(base="EUR", quote="USD", at=datetime(2024, 1, 9, tzinfo=UTC))
        assert rate == Decimal("1.1")


class TestLoadConversionSeries:
    def test_the_direct_spelling_is_preferred_when_both_exist(self) -> None:
        series = load_conversion_series(
            ["GBP"],
            account_currency="USD",
            timeframe=Timeframe.H4,
            load=_loader({"GBPUSD": _frame("GBPUSD", 1.27), "USDGBP": _frame("USDGBP", 0.79)}),
        )
        assert set(series) == {"GBPUSD"}


class TestTheRegistryCoversTheTradedUniverse:
    """The registry and the converter have to agree on what is tradeable."""

    def test_every_registered_instrument_resolves_to_a_conversion_pair_or_none(self) -> None:
        for symbol in REGISTRY:
            spec = REGISTRY[symbol]
            candidates = conversion_pair_candidates(spec.quote_currency, "USD")
            assert candidates == () or len(candidates) == 2, symbol

    def test_no_registered_pair_needs_chaining_on_a_usd_account(self) -> None:
        """Every conversion on a USD account is one hop, which is why there is no chain.

        Each quote currency is published against USD directly (GBPUSD) or
        inversely (USDJPY). A chain would only be needed for an account
        denominated in something that is not one leg of the pair.
        """
        majors = {"USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD"}
        for symbol in REGISTRY:
            quote = REGISTRY[symbol].quote_currency
            assert quote in majors, f"{symbol} is quoted in {quote}, which has no USD leg here"
