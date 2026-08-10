"""Contract specifications: the arithmetic everything above them depends on.

The numbers asserted here are hand-computed, not read back off the code. A
registry that agrees with itself proves nothing; a registry that agrees with a
number written independently in a test is the check that matters.
"""

from decimal import Decimal
from pathlib import Path

import pytest

from trading_system.core.exceptions import ValidationError
from trading_system.core.instruments import (
    CommissionBasis,
    InstrumentClass,
    InstrumentRegistry,
    InstrumentSpec,
    load_instruments,
)

REGISTRY_PATH = Path(__file__).resolve().parents[1] / "configs" / "instruments.yaml"


def spec(**overrides: object) -> InstrumentSpec:
    """Build a spec with EURUSD-like defaults, overriding one field at a time."""
    base: dict[str, object] = {
        "symbol": "TESTFX",
        "asset_class": InstrumentClass.FX,
        "tick_size": 0.00001,
        "point_size": 0.0001,
        "lot_step": Decimal("0.01"),
        "min_lot": Decimal("0.01"),
        "max_lot": Decimal("100"),
        "contract_size": Decimal("100000"),
        "base_currency": "EUR",
        "quote_currency": "USD",
        "margin_rate": 0.0333,
        "typical_spread_points": 0.8,
        "commission_per_lot": Decimal("7"),
        "commission_basis": CommissionBasis.ROUND_TURN,
        "swap_long": Decimal("-7.5"),
        "swap_short": Decimal("2.1"),
        "min_stop_distance_points": 1.0,
    }
    return InstrumentSpec(**(base | overrides))  # type: ignore[arg-type]


class TestPointsAreNotTicks:
    """The distinction the whole module is built around."""

    def test_a_point_is_a_whole_number_of_ticks(self) -> None:
        assert spec().ticks_per_point == 10

    def test_a_point_that_is_not_a_whole_number_of_ticks_is_a_typo(self) -> None:
        # 0.000015 is one and a half ticks. One of the two fields is wrong and
        # which one is not guessable, so the file does not load.
        with pytest.raises(ValueError, match="whole multiple of tick_size"):
            spec(point_size=0.000015)

    def test_an_unusual_but_whole_ratio_is_accepted(self) -> None:
        # 25 ticks to a point is odd but coherent, and the check is about
        # coherence, not about the ratio being 1 or 10.
        assert spec(point_size=0.00025).ticks_per_point == 25

    def test_a_point_smaller_than_a_tick_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="whole multiple of tick_size"):
            spec(point_size=0.000001)

    def test_fifty_points_is_fifty_pips_not_five(self) -> None:
        # The reason the two fields exist: with point_size collapsed onto
        # tick_size, this would be 0.00050 -- half a pip -- and every stop in
        # the system would be a tenth of its intended distance.
        assert spec().points_to_price(50) == pytest.approx(0.0050)

    def test_price_to_points_inverts_points_to_price(self) -> None:
        instrument = spec()
        assert instrument.price_to_points(instrument.points_to_price(37.5)) == pytest.approx(37.5)


class TestShiftPrice:
    """Moving a price by a signed distance, without ever shortening the move."""

    def test_a_shift_landing_on_the_grid_is_exact(self) -> None:
        # Half a one-point spread is five ticks on a five-digit feed. As floats
        # `1.1 + 0.00005` is 1.1000500000000001, and rounding that away from the
        # start would add a whole tick nobody asked for; the decimal arithmetic
        # inside shift_price is what keeps this exact.
        assert spec().shift_price(1.10000, 0.5) == pytest.approx(1.10005)
        assert spec().shift_price(1.10000, -0.5) == pytest.approx(1.09995)

    def test_an_off_grid_shift_rounds_away_from_the_start(self) -> None:
        # 0.15 points is one and a half ticks, so the move is two ticks, not one.
        assert spec().shift_price(1.10000, 0.15) == pytest.approx(1.10002)
        assert spec().shift_price(1.10000, -0.15) == pytest.approx(1.09998)

    def test_the_shift_is_never_shorter_than_asked_for(self) -> None:
        instrument = spec()
        for points in (0.01, 0.15, 0.5, 1.0, 2.7):
            up = instrument.shift_price(1.10000, points)
            down = instrument.shift_price(1.10000, -points)
            assert up - 1.10000 >= instrument.points_to_price(points) - 1e-12
            assert 1.10000 - down >= instrument.points_to_price(points) - 1e-12

    def test_a_zero_shift_just_snaps_to_the_grid(self) -> None:
        assert spec().shift_price(1.084567, 0.0) == pytest.approx(1.08457)


class TestCommissionBasisIsStated:
    """The per-side figure, and the absence of a default."""

    def test_round_turn_halves_and_per_side_does_not(self) -> None:
        assert spec(commission_basis=CommissionBasis.ROUND_TURN).commission_per_side == Decimal(
            "3.5"
        )
        assert spec(commission_basis=CommissionBasis.PER_SIDE).commission_per_side == Decimal("7")

    def test_the_basis_is_required(self) -> None:
        # "7.00 per lot" is genuinely ambiguous and the readings differ by
        # exactly two, so a registry that omits the basis does not load.
        assert InstrumentSpec.model_fields["commission_basis"].is_required()


class TestRounding:
    def test_price_snaps_to_the_tick_grid(self) -> None:
        assert spec().round_price(1.084567) == pytest.approx(1.08457)

    def test_directional_rounding_brackets_the_nearest(self) -> None:
        instrument = spec()
        assert instrument.round_price_down(1.084567) == pytest.approx(1.08456)
        assert instrument.round_price_up(1.084567) == pytest.approx(1.08457)

    def test_a_price_already_on_the_grid_is_unmoved_in_either_direction(self) -> None:
        instrument = spec()
        assert instrument.round_price_down(1.08456) == pytest.approx(1.08456)
        assert instrument.round_price_up(1.08456) == pytest.approx(1.08456)

    def test_volume_rounds_down_never_to_nearest(self) -> None:
        # 0.019 is nearer 0.02 than 0.01, and still becomes 0.01: rounding a
        # size up risks more money than the sizing method computed.
        assert spec().round_volume(Decimal("0.019")) == Decimal("0.01")

    def test_volume_below_one_step_rounds_to_an_honest_zero(self) -> None:
        # Not min_lot. Whether a zero size is a refused trade is the caller's
        # decision, and inventing a tradable size here would take it away.
        assert spec().round_volume(Decimal("0.004")) == Decimal("0")

    def test_a_lot_step_that_is_not_a_power_of_ten_still_works(self) -> None:
        assert spec(lot_step=Decimal("0.25"), min_lot=Decimal("0.25")).round_volume(
            Decimal("1.7")
        ) == Decimal("1.50")

    def test_a_negative_volume_is_a_caller_error(self) -> None:
        with pytest.raises(ValueError, match="must not be negative"):
            spec().round_volume(Decimal("-1"))


class TestLotBounds:
    def test_max_below_min_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="below min_lot"):
            spec(min_lot=Decimal("1"), max_lot=Decimal("0.5"))

    def test_a_min_lot_off_the_step_grid_is_rejected(self) -> None:
        # min_lot 0.03 on a step of 0.02 means the smallest tradable size is
        # not itself tradable.
        with pytest.raises(ValueError, match="not a whole multiple of lot_step"):
            spec(lot_step=Decimal("0.02"), min_lot=Decimal("0.03"))


class TestValuePerPoint:
    """One formula, ``point_size * contract_size``, for every asset class."""

    @pytest.mark.parametrize(
        ("symbol", "expected", "currency"),
        [
            ("EURUSD", Decimal("10"), "USD"),
            ("GBPJPY", Decimal("1000"), "JPY"),
            ("XAUUSD", Decimal("1"), "USD"),
            ("NAS100", Decimal("20"), "USD"),
            ("US30", Decimal("5"), "USD"),
            ("BTCUSD", Decimal("1"), "USD"),
        ],
    )
    def test_hand_computed_per_instrument(
        self, symbol: str, expected: Decimal, currency: str
    ) -> None:
        instrument = load_instruments(REGISTRY_PATH)[symbol]
        assert instrument.value_per_point_quote == expected
        assert instrument.quote_currency == currency

    def test_the_result_is_decimal_not_float(self) -> None:
        # It is about to be multiplied by a position size.
        assert isinstance(spec().value_per_point_quote, Decimal)


class TestRegistryFile:
    def test_the_bundled_registry_loads(self) -> None:
        registry = load_instruments(REGISTRY_PATH)
        assert set(registry.symbols) == {
            # FX, quoted in USD
            "EURUSD",
            "GBPUSD",
            "AUDUSD",
            "NZDUSD",
            # FX, quoted in JPY
            "USDJPY",
            "EURJPY",
            "GBPJPY",
            "CHFJPY",
            "AUDJPY",
            "NZDJPY",
            # FX, quoted in CHF / CAD / GBP
            "USDCHF",
            "USDCAD",
            "EURGBP",
            "EURCHF",
            "EURCAD",
            "GBPCHF",
            # Metals, indices, crypto
            "XAUUSD",
            "XAGUSD",
            "NAS100",
            "US30",
            "BTCUSD",
        }

    def test_every_non_usd_quote_has_its_conversion_pair_in_the_registry(self) -> None:
        # A USD account cannot size a CHF-quoted pair without a CHF/USD rate,
        # and risk/conversion.py refuses rather than assuming parity. Shipping a
        # cross whose conversion pair is absent is therefore shipping an
        # instrument that cannot be traded, which is worse than not shipping it.
        registry = load_instruments(REGISTRY_PATH)
        carried = set(registry.symbols)
        for symbol in registry:
            quote = registry[symbol].quote_currency
            if quote == "USD":
                continue
            assert f"USD{quote}" in carried or f"{quote}USD" in carried, (
                f"{symbol} is quoted in {quote} but neither USD{quote} nor "
                f"{quote}USD is in the registry to convert it"
            )

    def test_the_registry_spans_four_asset_classes(self) -> None:
        # Deliberately mixed: a sizing bug that only shows up on a non-FX
        # contract size is invisible on an all-FX universe.
        registry = load_instruments(REGISTRY_PATH)
        assert {registry[symbol].asset_class for symbol in registry} == {
            InstrumentClass.FX,
            InstrumentClass.COMMODITY,
            InstrumentClass.INDEX,
            InstrumentClass.CRYPTO,
        }

    def test_a_missing_file_names_itself(self) -> None:
        with pytest.raises(ValidationError, match="cannot be read"):
            load_instruments(Path("/nonexistent/instruments.yaml"))

    def test_a_duplicate_symbol_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "dupes.yaml"
        path.write_text(
            "instruments:\n"
            + 2
            * (
                "  - symbol: EURUSD\n"
                "    asset_class: FX\n"
                "    tick_size: 0.00001\n"
                "    point_size: 0.0001\n"
                '    lot_step: "0.01"\n'
                '    min_lot: "0.01"\n'
                '    max_lot: "100"\n'
                '    contract_size: "100000"\n'
                "    base_currency: EUR\n"
                "    quote_currency: USD\n"
                "    margin_rate: 0.0333\n"
                "    typical_spread_points: 0.8\n"
                '    commission_per_lot: "7"\n'
                "    commission_basis: ROUND_TURN\n"
                '    swap_long: "-7.5"\n'
                '    swap_short: "2.1"\n'
                "    min_stop_distance_points: 1.0\n"
            ),
            encoding="utf-8",
        )
        with pytest.raises(ValidationError, match="declared twice"):
            load_instruments(path)

    def test_an_unknown_field_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "typo.yaml"
        path.write_text("instruments: []\nextra_key: 1\n", encoding="utf-8")
        with pytest.raises(ValidationError):
            load_instruments(path)

    def test_a_file_that_is_not_a_mapping_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "list.yaml"
        path.write_text("- symbol: EURUSD\n", encoding="utf-8")
        with pytest.raises(ValidationError, match="expected a mapping"):
            load_instruments(path)


class TestRegistryLookup:
    def test_an_unknown_symbol_raises_rather_than_substituting(self) -> None:
        # Guessing a contract size is guessing a position size.
        registry = InstrumentRegistry({"EURUSD": spec(symbol="EURUSD")})
        with pytest.raises(KeyError, match="unknown instrument"):
            registry["GBPUSD"]

    def test_get_returns_none_for_an_unknown_symbol(self) -> None:
        registry = InstrumentRegistry({"EURUSD": spec(symbol="EURUSD")})
        assert registry.get("GBPUSD") is None
        assert registry.get("EURUSD") is not None

    def test_membership_and_length(self) -> None:
        registry = InstrumentRegistry({"EURUSD": spec(symbol="EURUSD")})
        assert "EURUSD" in registry
        assert "GBPUSD" not in registry
        assert len(registry) == 1
