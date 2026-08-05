"""Configuration: the double-count guard, and the shipped file."""

from pathlib import Path

import pytest

from trading_system.core.exceptions import ValidationError
from trading_system.execution.config import (
    CostConfig,
    ExecutionRunConfig,
    SpreadConfig,
    SpreadSource,
    load_execution_config,
)

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "execution.yaml"


class TestSpreadIsNotPaidTwice:
    """The guard that exists before the thing it guards.

    P10's ``stop_buffer.spread_multiple`` widens the protective stop by a spread
    because a long is stopped on the bid and a mid-space trigger test does not
    know that. Once a run has real quotes and the trigger test knows, the same
    half spread is charged twice — a stop placed further out than it needs to be,
    *and* a trigger that already accounts for the quote side. The symptom is a
    strategy that stops out less often than it should while sizing smaller than
    it should, which nobody diagnoses as a config interaction.
    """

    def test_quote_aware_with_a_buffered_stop_refuses_to_load(self) -> None:
        """The pair is rejected, with the mechanism written into the message."""
        with pytest.raises(ValueError, match="twice"):
            ExecutionRunConfig(
                costs=CostConfig(spread=SpreadConfig(source=SpreadSource.QUOTE_AWARE)),
                stop_buffer_spread_multiple=1.0,
            )

    def test_quote_aware_with_no_stop_buffer_is_fine(self) -> None:
        """The combination the check is steering runs towards."""
        config = ExecutionRunConfig(
            costs=CostConfig(spread=SpreadConfig(source=SpreadSource.QUOTE_AWARE)),
            stop_buffer_spread_multiple=0.0,
        )
        assert config.costs.spread.source is SpreadSource.QUOTE_AWARE

    @pytest.mark.parametrize("source", [SpreadSource.TYPICAL, SpreadSource.QUOTED])
    def test_a_mid_quoted_source_keeps_the_buffer(self, source: SpreadSource) -> None:
        """The check does not fire today, which is the point of writing it today.

        Both mid-quoted sources need the buffer: the trigger test works in mid
        space, so the bid/ask asymmetry has to be corrected somewhere, and the
        stop placement is where P10 put it.
        """
        config = ExecutionRunConfig(
            costs=CostConfig(spread=SpreadConfig(source=source)),
            stop_buffer_spread_multiple=1.0,
        )
        assert config.stop_buffer_spread_multiple == 1.0

    def test_the_message_names_both_settings(self) -> None:
        """A guard nobody can act on is a guard that gets deleted."""
        with pytest.raises(ValueError) as caught:
            ExecutionRunConfig(
                costs=CostConfig(spread=SpreadConfig(source=SpreadSource.QUOTE_AWARE)),
                stop_buffer_spread_multiple=1.0,
            )
        message = str(caught.value)
        assert "QUOTE_AWARE" in message
        assert "spread_multiple" in message


class TestShippedConfig:
    """The bundled file loads and says what it looks like it says."""

    def test_it_loads(self) -> None:
        """A config that does not parse is a run that does not start."""
        config = load_execution_config(CONFIG_PATH)
        assert config.costs.spread.source is SpreadSource.TYPICAL

    def test_stops_are_configured_to_slip_more_than_limits(self) -> None:
        """Asserted against the shipped numbers, not only against the validator."""
        slippage = load_execution_config(CONFIG_PATH).costs.slippage
        assert slippage.stop.base_points > slippage.limit.base_points
        assert slippage.stop.atr_coefficient > slippage.limit.atr_coefficient

    def test_the_gap_penalty_is_not_zero(self) -> None:
        """A shipped zero would silently restore P07's optimistic reference model."""
        gap = load_execution_config(CONFIG_PATH).costs.gap
        assert gap.penalty_base_points > 0
        assert gap.penalty_fraction > 0

    def test_gapped_limits_get_no_improvement_by_default(self) -> None:
        """The P07 correction, asserted where a future edit would undo it."""
        assert not load_execution_config(CONFIG_PATH).costs.gap.grant_improvement_to_limits

    def test_every_asset_class_has_a_financing_model(self) -> None:
        """An unlisted class accrues nothing, which should be a choice not an omission."""
        swap = load_execution_config(CONFIG_PATH).costs.swap
        assert swap["CRYPTO"].kind == "funding_rate"
        assert swap["FX"].kind == "per_lot_rollover"
        assert swap["FX"].triple_weekday == 2

    def test_a_missing_file_names_itself(self) -> None:
        """The error says which path, not merely that something went wrong."""
        with pytest.raises(ValidationError, match="cannot be read"):
            load_execution_config(CONFIG_PATH.parent / "nope.yaml")

    def test_an_unknown_field_is_rejected(self, tmp_path: Path) -> None:
        """A typo in a config key must not become a silently ignored setting."""
        path = tmp_path / "execution.yaml"
        path.write_text("costs:\n  spred:\n    source: TYPICAL\n", encoding="utf-8")
        with pytest.raises(ValidationError):
            load_execution_config(path)
