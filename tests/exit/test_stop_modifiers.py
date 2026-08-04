"""ATRStop, StructureStop, TrailingStop, BreakevenMove: scenario and ratchet tests.

None of these emit an :class:`~trading_system.exit.base.ExitDecision` — they
propose a stop level, and :class:`~trading_system.exit.rules.protective_stop.
ProtectiveStop` is what turns a touched level into an exit. So every scenario
here pairs the modifier under test with ``ProtectiveStop`` in a plan and reads
the result off ``position.stop`` and, where a level is actually touched, off
the fill's price and reason.
"""

import pytest

from trading_system.core.types import Price
from trading_system.exit.base import ExitReason, StopModifier
from trading_system.exit.context import exit_contexts
from trading_system.exit.plan import ExitPlan
from trading_system.exit.rules import (
    ATRStop,
    AtrTrail,
    BreakevenMove,
    Chandelier,
    MaTrail,
    ProtectiveStop,
    StructureStop,
    SwingTrail,
    TrailingStop,
)
from trading_system.exit.rules._features import atr_key, ma_key, swing_keys

from .conftest import Bar, long_position, series, short_position

ATR_14 = atr_key(14)
SWING_LOW_5, SWING_HIGH_5 = swing_keys(5)
EMA_20 = ma_key(20, "close")


def plan_with(modifier: StopModifier) -> ExitPlan:
    """A plan pairing one stop modifier with the protective stop."""
    return ExitPlan(
        exit_id="modifier-scenario",
        protective_stop=ProtectiveStop(),
        stop_modifiers=[modifier],
    )


class TestATRStop:
    def test_a_long_stop_sits_below_entry_by_the_atr_multiple(self) -> None:
        bars: list[Bar] = [(1.1000, 1.1020, 1.0990, 1.1010)]
        position = long_position(entry=1.1000, stop=1.0800)
        atr = {ATR_14: [0.0050]}
        plan_with(ATRStop(period=14, multiple=2.0)).run(
            position, exit_contexts(series(bars, features=atr))
        )
        # entry - 2*0.0050 = 1.0900, tighter than the initial 1.0800.
        assert position.stop == pytest.approx(1.0900)

    def test_a_short_stop_sits_above_entry_by_the_atr_multiple(self) -> None:
        bars: list[Bar] = [(1.1000, 1.1020, 1.0990, 1.1010)]
        position = short_position(entry=1.1000, stop=1.1200)
        atr = {ATR_14: [0.0050]}
        plan_with(ATRStop(period=14, multiple=2.0)).run(
            position, exit_contexts(series(bars, features=atr))
        )
        assert position.stop == pytest.approx(1.1100)

    def test_recompute_true_tightens_as_atr_shrinks_but_never_widens(self) -> None:
        # ATR shrinks then grows back: the tighter level from bar 1 survives
        # bar 2's looser proposal — the same ratchet as everything else here.
        bars: list[Bar] = [
            (1.1000, 1.1020, 1.0990, 1.1010),
            (1.1010, 1.1030, 1.0995, 1.1020),
            (1.1020, 1.1040, 1.1000, 1.1030),
        ]
        atr = {ATR_14: [0.0100, 0.0040, 0.0090]}
        position = long_position(entry=1.1000, stop=1.0700)
        plan_with(ATRStop(period=14, multiple=1.0, recompute=True)).run(
            position, exit_contexts(series(bars, features=atr))
        )
        # Tightest proposal was bar 1: entry - 0.0040 = 1.0960.
        assert position.stop == pytest.approx(1.0960)

    def test_recompute_false_reads_atr_once_and_keeps_the_same_level(self) -> None:
        bars: list[Bar] = [
            (1.1000, 1.1020, 1.0990, 1.1010),
            (1.1010, 1.1030, 1.0995, 1.1020),
        ]
        atr = {ATR_14: [0.0100, 0.0010]}
        position = long_position(entry=1.1000, stop=1.0700)
        plan_with(ATRStop(period=14, multiple=1.0, recompute=False)).run(
            position, exit_contexts(series(bars, features=atr))
        )
        # A much tighter ATR arrived on bar 2, but recompute=False never reads
        # it: the level stays at the bar-1 reading, entry - 0.0100 = 1.0900.
        assert position.stop == pytest.approx(1.0900)

    def test_the_stop_fires_through_protective_stop_not_atr_stop(self) -> None:
        bars: list[Bar] = [
            (1.1000, 1.1020, 1.0990, 1.1010),
            (1.0900, 1.0910, 1.0700, 1.0750),  # low pierces the tightened stop
        ]
        atr = {ATR_14: [0.0100, 0.0100]}
        position = long_position(entry=1.1000, stop=1.0700)
        result = plan_with(ATRStop(period=14, multiple=1.0)).run(
            position, exit_contexts(series(bars, features=atr))
        )
        assert len(result.fills) == 1
        assert result.fills[0].rule == "protective_stop"
        assert result.fills[0].decision.reason is ExitReason.PROTECTIVE_STOP
        assert result.fills[0].leg.price == pytest.approx(1.0900)

    @pytest.mark.parametrize(("period", "multiple"), [(0, 1.0), (14, 0.0), (14, -1.0)])
    def test_construction_rejects_non_positive_parameters(
        self, period: int, multiple: float
    ) -> None:
        with pytest.raises(ValueError, match="positive"):
            ATRStop(period=period, multiple=multiple)


class TestStructureStop:
    def test_a_long_stop_sits_at_the_swing_low(self) -> None:
        bars: list[Bar] = [(1.1000, 1.1020, 1.0990, 1.1010)]
        swing = {SWING_LOW_5: [1.0850], SWING_HIGH_5: [1.1200]}
        position = long_position(entry=1.1000, stop=1.0700)
        plan_with(StructureStop(lookback=5)).run(
            position, exit_contexts(series(bars, features=swing))
        )
        assert position.stop == pytest.approx(1.0850)

    def test_a_short_stop_sits_at_the_swing_high(self) -> None:
        bars: list[Bar] = [(1.1000, 1.1020, 1.0990, 1.1010)]
        swing = {SWING_LOW_5: [1.0850], SWING_HIGH_5: [1.1200]}
        position = short_position(entry=1.1000, stop=1.1400)
        plan_with(StructureStop(lookback=5)).run(
            position, exit_contexts(series(bars, features=swing))
        )
        assert position.stop == pytest.approx(1.1200)

    def test_a_buffer_widens_the_distance_beyond_the_swing(self) -> None:
        bars: list[Bar] = [(1.1000, 1.1020, 1.0990, 1.1010)]
        columns = {SWING_LOW_5: [1.0850], SWING_HIGH_5: [1.1200], ATR_14: [0.0050]}
        position = long_position(entry=1.1000, stop=1.0700)
        plan_with(StructureStop(lookback=5, buffer_atr_multiple=2.0)).run(
            position, exit_contexts(series(bars, features=columns))
        )
        # 1.0850 - 2*0.0050 = 1.0750.
        assert position.stop == pytest.approx(1.0750)

    def test_active_unconditionally_from_the_first_bar(self) -> None:
        # Unlike TrailingStop's swing source, no activation threshold gates it.
        bars: list[Bar] = [(1.1000, 1.1005, 1.0995, 1.1000)]
        swing = {SWING_LOW_5: [1.0950], SWING_HIGH_5: [1.1200]}
        position = long_position(entry=1.1000, stop=1.0700)
        assert position.r_multiple(Price(1.1005)) < 1.0  # far from any R threshold
        plan_with(StructureStop(lookback=5)).run(
            position, exit_contexts(series(bars, features=swing))
        )
        assert position.stop == pytest.approx(1.0950)

    def test_structure_moving_further_away_does_not_widen_the_stop(self) -> None:
        bars: list[Bar] = [
            (1.1000, 1.1020, 1.0990, 1.1010),
            (1.1010, 1.1030, 1.0995, 1.1020),
        ]
        swing = {SWING_LOW_5: [1.0950, 1.0800], SWING_HIGH_5: [1.1200, 1.1200]}
        position = long_position(entry=1.1000, stop=1.0700)
        plan_with(StructureStop(lookback=5)).run(
            position, exit_contexts(series(bars, features=swing))
        )
        assert position.stop == pytest.approx(1.0950)

    def test_construction_rejects_a_non_positive_lookback(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            StructureStop(lookback=0)

    def test_construction_rejects_a_negative_buffer(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            StructureStop(buffer_atr_multiple=-1.0)


class TestTrailingStopActivation:
    def test_no_level_is_proposed_before_the_activation_threshold(self) -> None:
        bars: list[Bar] = [(1.1000, 1.1050, 1.0990, 1.1040)]  # 0.5R only
        atr = {ATR_14: [0.0050]}
        position = long_position(entry=1.1000, stop=1.0900)
        plan_with(TrailingStop(AtrTrail(period=14, multiple=1.0), activation_r=2.0)).run(
            position, exit_contexts(series(bars, features=atr))
        )
        assert position.stop == pytest.approx(1.0900)

    def test_a_level_is_proposed_on_the_bar_the_threshold_is_reached(self) -> None:
        # High reaches 1.1200, exactly 2R from entry 1.1000 with risk 0.0100.
        bars: list[Bar] = [(1.1150, 1.1200, 1.1140, 1.1180)]
        atr = {ATR_14: [0.0050]}
        position = long_position(entry=1.1000, stop=1.0900)
        plan_with(TrailingStop(AtrTrail(period=14, multiple=1.0), activation_r=2.0)).run(
            position, exit_contexts(series(bars, features=atr))
        )
        assert position.stop == pytest.approx(1.1150)

    def test_activation_latches_and_a_later_pullback_does_not_deactivate_it(self) -> None:
        bars: list[Bar] = [
            (1.1150, 1.1200, 1.1140, 1.1180),  # reaches 2R, activates
            (1.1050, 1.1060, 1.1040, 1.1050),  # pulls well back below 2R
        ]
        atr = {ATR_14: [0.0050, 0.0050]}
        position = long_position(entry=1.1000, stop=1.0900)
        plan_with(TrailingStop(AtrTrail(period=14, multiple=1.0), activation_r=2.0)).run(
            position, exit_contexts(series(bars, features=atr))
        )
        # Bar 2's proposal (1.1060 - 0.0050 = 1.1010) is looser than bar 1's
        # (1.1200 - 0.0050 = 1.1150) and is declined by the ratchet either way.
        assert position.stop == pytest.approx(1.1150)

    def test_construction_rejects_a_non_positive_activation(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            TrailingStop(AtrTrail(), activation_r=0.0)


class TestTrailingStopSources:
    def test_atr_trail_follows_todays_extreme_minus_the_atr_multiple(self) -> None:
        bars: list[Bar] = [
            (1.1150, 1.1200, 1.1140, 1.1180),  # activates at 1.1200
            (1.1180, 1.1300, 1.1170, 1.1290),  # extreme extends to 1.1300
        ]
        atr = {ATR_14: [0.0050, 0.0050]}
        position = long_position(entry=1.1000, stop=1.0900)
        plan_with(TrailingStop(AtrTrail(period=14, multiple=1.0), activation_r=2.0)).run(
            position, exit_contexts(series(bars, features=atr))
        )
        assert position.stop == pytest.approx(1.1250)  # 1.1300 - 0.0050

    def test_atr_trail_ratchets_even_though_it_recomputes_every_bar(self) -> None:
        bars: list[Bar] = [
            (1.1150, 1.1300, 1.1140, 1.1180),  # high 1.1300, activates
            (1.1180, 1.1200, 1.1170, 1.1190),  # high pulls back to 1.1200
        ]
        atr = {ATR_14: [0.0050, 0.0050]}
        position = long_position(entry=1.1000, stop=1.0900)
        plan_with(TrailingStop(AtrTrail(period=14, multiple=1.0), activation_r=2.0)).run(
            position, exit_contexts(series(bars, features=atr))
        )
        # Bar 2 proposes 1.1200-0.0050=1.1150, looser than bar 1's 1.1250.
        assert position.stop == pytest.approx(1.1250)

    def test_chandelier_uses_the_highest_high_of_its_own_window_not_since_entry(self) -> None:
        bars: list[Bar] = [
            (1.1150, 1.1400, 1.1140, 1.1180),  # activates; window high so far 1.1400
            (1.1180, 1.1200, 1.1170, 1.1190),
            (1.1190, 1.1210, 1.1180, 1.1200),
        ]
        atr = {ATR_14: [0.0050, 0.0050, 0.0050]}
        position = long_position(entry=1.1000, stop=1.0900)
        # A 2-bar window: by bar 2 (index 1), 1.1400 has scrolled out of the
        # window (which now covers bars 0-1... wait lookback=2 covers [t, t-1]).
        plan_with(
            TrailingStop(Chandelier(lookback=2, period=14, multiple=1.0), activation_r=2.0)
        ).run(position, exit_contexts(series(bars, features=atr)))
        # Bar 0: window=[1.1400], but lookback=2 needs bar -1 too -> None, no
        # proposal yet on bar 0 even though activation triggers on it (favourable
        # extreme check uses the bar's own high regardless of window depth).
        # Bar 1: window=[1.1200(bar1), 1.1400(bar0)], max=1.1400, level=1.1350.
        # Bar 2: window=[1.1210(bar2), 1.1200(bar1)], max=1.1210, level=1.1160
        #        (looser than 1.1350, declined).
        assert position.stop == pytest.approx(1.1350)

    def test_ma_trail_sits_at_the_moving_average(self) -> None:
        bars: list[Bar] = [
            (1.1150, 1.1200, 1.1140, 1.1180),
            (1.1180, 1.1220, 1.1170, 1.1200),
        ]
        columns = {EMA_20: [1.1050, 1.1080]}
        position = long_position(entry=1.1000, stop=1.0900)
        plan_with(TrailingStop(MaTrail(period=20), activation_r=2.0)).run(
            position, exit_contexts(series(bars, features=columns))
        )
        assert position.stop == pytest.approx(1.1080)

    def test_ma_trail_buffer_pulls_the_level_further_from_price(self) -> None:
        bars: list[Bar] = [(1.1150, 1.1200, 1.1140, 1.1180)]
        columns = {EMA_20: [1.1050], ATR_14: [0.0020]}
        position = long_position(entry=1.1000, stop=1.0900)
        plan_with(
            TrailingStop(
                MaTrail(period=20, buffer_atr_multiple=2.0, atr_period=14), activation_r=2.0
            )
        ).run(position, exit_contexts(series(bars, features=columns)))
        assert position.stop == pytest.approx(1.1010)  # 1.1050 - 2*0.0020

    def test_swing_trail_sits_at_the_swing_low_only_once_activated(self) -> None:
        bars: list[Bar] = [
            (1.1000, 1.1050, 1.0990, 1.1040),  # 0.5R, not activated
            (1.1150, 1.1200, 1.1140, 1.1180),  # 2R, activates
        ]
        swing = {SWING_LOW_5: [1.0950, 1.0975], SWING_HIGH_5: [1.1500, 1.1500]}
        position = long_position(entry=1.1000, stop=1.0900)
        plan_with(TrailingStop(SwingTrail(lookback=5), activation_r=2.0)).run(
            position, exit_contexts(series(bars, features=swing))
        )
        # Bar 0's tighter swing (1.0950) is never proposed since it precedes
        # activation; only bar 1's 1.0975 is.
        assert position.stop == pytest.approx(1.0975)


class TestBreakevenMove:
    def test_no_move_before_the_activation_threshold(self) -> None:
        bars: list[Bar] = [(1.1000, 1.1050, 1.0990, 1.1040)]  # 0.5R
        position = long_position(entry=1.1000, stop=1.0900)
        plan_with(BreakevenMove(activation_r=1.0)).run(position, exit_contexts(series(bars)))
        assert position.stop == pytest.approx(1.0900)

    def test_moves_to_entry_on_the_bar_the_threshold_is_reached(self) -> None:
        bars: list[Bar] = [(1.1000, 1.1100, 1.0990, 1.1080)]  # touches 1R
        position = long_position(entry=1.1000, stop=1.0900)
        plan_with(BreakevenMove(activation_r=1.0)).run(position, exit_contexts(series(bars)))
        assert position.stop == pytest.approx(1.1000)

    def test_spread_offsets_the_breakeven_level(self) -> None:
        bars: list[Bar] = [(1.1000, 1.1100, 1.0990, 1.1080)]
        position = long_position(entry=1.1000, stop=1.0900)
        plan_with(BreakevenMove(activation_r=1.0, spread=0.0002)).run(
            position, exit_contexts(series(bars))
        )
        assert position.stop == pytest.approx(1.1002)

    def test_a_short_moves_to_entry_minus_spread(self) -> None:
        bars: list[Bar] = [(1.1000, 1.1010, 1.0900, 1.0920)]  # touches 1R for a short
        position = short_position(entry=1.1000, stop=1.1100)
        plan_with(BreakevenMove(activation_r=1.0, spread=0.0002)).run(
            position, exit_contexts(series(bars))
        )
        assert position.stop == pytest.approx(1.0998)

    def test_a_later_pullback_does_not_move_the_stop_back(self) -> None:
        bars: list[Bar] = [
            (1.1000, 1.1100, 1.0990, 1.1080),  # touches 1R, moves to BE
            (1.1080, 1.1085, 1.1050, 1.1060),  # pulls back, still above BE
        ]
        position = long_position(entry=1.1000, stop=1.0900)
        plan_with(BreakevenMove(activation_r=1.0)).run(position, exit_contexts(series(bars)))
        assert position.stop == pytest.approx(1.1000)

    def test_construction_rejects_a_non_positive_activation(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            BreakevenMove(activation_r=0.0)

    def test_construction_rejects_a_negative_spread(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            BreakevenMove(activation_r=1.0, spread=-0.0001)
