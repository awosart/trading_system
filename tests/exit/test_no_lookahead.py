"""No-lookahead by the P06 equivalence recipe, run against every stage-2 rule.

The proof P06 established for :class:`~trading_system.entry.context.BarContext`
— a context built at bar ``t`` over the full series must answer exactly as one
built at ``t`` over the series truncated at ``t`` — is re-proved directly for
:class:`~trading_system.exit.context.ExitContext` in ``test_context.py``. What
this file checks is the layer built *on top* of that context: a rule or a whole
plan that only ever calls the proven-safe accessors cannot see forward through
them, but a stateful rule (an activation flag, a bar counter, a fired-rung list)
could in principle still leak information some other way — by branching on
``len(...)`` of something it shouldn't, for instance. The general form of the
proof covers that: running a plan for exactly ``t`` bars, whether those bars
come from a series truncated at ``t`` or as the first ``t`` bars of the full
series, must produce the identical sequence of fills and leave the position in
the identical state. A rule with a hidden forward-reaching path would disagree.
"""

from collections.abc import Iterator
from decimal import Decimal
from itertools import islice

import pytest

from trading_system.core.types import Side
from trading_system.data.sessions import Session
from trading_system.exit.context import ExitContext, exit_contexts
from trading_system.exit.plan import ExitPlan
from trading_system.exit.rules import (
    ATRStop,
    AtrTrail,
    BreakevenMove,
    Chandelier,
    FixedRR,
    MaTrail,
    PartialClose,
    PartialRung,
    ProtectiveStop,
    SignalReverseExit,
    StructureStop,
    SwingTrail,
    TimeExit,
    TimeExitMode,
    TrailingStop,
)
from trading_system.exit.rules._features import atr_key, ma_key, swing_keys

from .conftest import Bar, long_position, series

ATR_14 = atr_key(14)
SWING_LOW_5, SWING_HIGH_5 = swing_keys(5)
EMA_20 = ma_key(20, "close")

#: A long enough, wandering series to give every rule something to react to:
#: a strong run-up (activates trailing/breakeven/partials), a pullback, and a
#: second leg, all inside plausible ATR-scaled ranges.
_CLOSES = [
    1.1000, 1.1015, 1.1040, 1.1080, 1.1130, 1.1190, 1.1160, 1.1120,
    1.1150, 1.1200, 1.1260, 1.1230, 1.1190, 1.1220, 1.1280, 1.1260,
    1.1300, 1.1270, 1.1240, 1.1290,
]  # fmt: skip

BARS: list[Bar] = [(close - 0.0010, close + 0.0025, close - 0.0030, close) for close in _CLOSES]

_ATR_COLUMN = [0.0060] * len(BARS)
_SWING_LOW_COLUMN = [c - 0.0150 for c in _CLOSES]
_SWING_HIGH_COLUMN = [c + 0.0150 for c in _CLOSES]
_EMA_COLUMN = [c - 0.0080 for c in _CLOSES]

FEATURES = {
    ATR_14: _ATR_COLUMN,
    SWING_LOW_5: _SWING_LOW_COLUMN,
    SWING_HIGH_5: _SWING_HIGH_COLUMN,
    EMA_20: _EMA_COLUMN,
}

REVERSE_SIGNALS = {15: Side.SELL}


def plans() -> Iterator[tuple[str, ExitPlan]]:
    """One plan per stage-2 rule (and one combining several), named for -k."""
    yield (
        "atr_stop",
        ExitPlan(
            exit_id="atr_stop",
            protective_stop=ProtectiveStop(),
            stop_modifiers=[ATRStop(period=14, multiple=1.0)],
        ),
    )
    yield (
        "structure_stop",
        ExitPlan(
            exit_id="structure_stop",
            protective_stop=ProtectiveStop(),
            stop_modifiers=[StructureStop(lookback=5)],
        ),
    )
    yield (
        "trailing_atr",
        ExitPlan(
            exit_id="trailing_atr",
            protective_stop=ProtectiveStop(),
            stop_modifiers=[TrailingStop(AtrTrail(period=14, multiple=1.0), activation_r=1.0)],
        ),
    )
    yield (
        "trailing_chandelier",
        ExitPlan(
            exit_id="trailing_chandelier",
            protective_stop=ProtectiveStop(),
            stop_modifiers=[
                TrailingStop(Chandelier(lookback=3, period=14, multiple=1.0), activation_r=1.0)
            ],
        ),
    )
    yield (
        "trailing_ma",
        ExitPlan(
            exit_id="trailing_ma",
            protective_stop=ProtectiveStop(),
            stop_modifiers=[TrailingStop(MaTrail(period=20), activation_r=1.0)],
        ),
    )
    yield (
        "trailing_swing",
        ExitPlan(
            exit_id="trailing_swing",
            protective_stop=ProtectiveStop(),
            stop_modifiers=[TrailingStop(SwingTrail(lookback=5), activation_r=1.0)],
        ),
    )
    yield (
        "breakeven",
        ExitPlan(
            exit_id="breakeven",
            protective_stop=ProtectiveStop(),
            stop_modifiers=[BreakevenMove(activation_r=1.0, spread=0.0002)],
        ),
    )
    yield (
        "partial_close",
        ExitPlan(
            exit_id="partial_close",
            protective_stop=ProtectiveStop(),
            rules=[
                PartialClose(
                    [
                        PartialRung(r_multiple=1.0, fraction=Decimal("0.5")),
                        PartialRung(r_multiple=2.0, fraction=Decimal("0.25")),
                    ]
                )
            ],
        ),
    )
    yield (
        "time_exit_bars",
        ExitPlan(
            exit_id="time_exit_bars",
            protective_stop=ProtectiveStop(),
            rules=[TimeExit(TimeExitMode.MAX_BARS_HELD, max_bars_held=7)],
        ),
    )
    yield (
        "time_exit_session",
        ExitPlan(
            exit_id="time_exit_session",
            protective_stop=ProtectiveStop(),
            rules=[TimeExit(TimeExitMode.SESSION_CLOSE, session=Session.LONDON)],
        ),
    )
    yield (
        "signal_reverse",
        ExitPlan(
            exit_id="signal_reverse", protective_stop=ProtectiveStop(), rules=[SignalReverseExit()]
        ),
    )
    yield (
        "kitchen_sink",
        ExitPlan(
            exit_id="kitchen_sink",
            protective_stop=ProtectiveStop(),
            rules=[
                FixedRR(4.0),
                PartialClose([PartialRung(r_multiple=1.0, fraction=Decimal("0.3"))]),
                TimeExit(TimeExitMode.MAX_BARS_HELD, max_bars_held=10),
                SignalReverseExit(),
            ],
            stop_modifiers=[
                ATRStop(period=14, multiple=1.5),
                TrailingStop(AtrTrail(period=14, multiple=1.0), activation_r=1.0),
                BreakevenMove(activation_r=0.5),
            ],
        ),
    )


def _contexts(length: int, *, truncate: bool) -> Iterator[ExitContext]:
    """The first ``length`` exit contexts, from a truncated or a full series."""
    source = series(BARS[:length], features={key: col[:length] for key, col in FEATURES.items()})
    reverse = {index: side for index, side in REVERSE_SIGNALS.items() if index < length}
    if truncate:
        return exit_contexts(source, reverse_signals=reverse)
    full = series(BARS, features=FEATURES)
    return islice(exit_contexts(full, reverse_signals=REVERSE_SIGNALS), length)


@pytest.mark.parametrize(("plan_name", "plan"), list(plans()))
@pytest.mark.parametrize("length", [1, 3, 5, 8, 12, len(BARS)])
def test_a_plan_run_for_t_bars_agrees_truncated_vs_prefix_of_full(
    plan_name: str,  # noqa: ARG001 — carried only so -k can select one plan
    plan: ExitPlan,
    length: int,
) -> None:
    truncated_position = long_position(entry=BARS[0][0], stop=BARS[0][0] - 0.0300)
    truncated_result = plan.run(truncated_position, _contexts(length, truncate=True))

    prefix_position = long_position(entry=BARS[0][0], stop=BARS[0][0] - 0.0300)
    prefix_result = plan.run(prefix_position, _contexts(length, truncate=False))

    assert truncated_result.bars == prefix_result.bars
    assert truncated_result.closed == prefix_result.closed
    assert len(truncated_result.fills) == len(prefix_result.fills)
    for here, there in zip(truncated_result.fills, prefix_result.fills, strict=True):
        assert here.bar_index == there.bar_index
        assert here.decided_bar_index == there.decided_bar_index
        assert here.decision.reason == there.decision.reason
        assert here.decision.kind == there.decision.kind
        assert here.leg.price == pytest.approx(there.leg.price)
        assert here.leg.fraction == there.leg.fraction

    assert truncated_position.stop == pytest.approx(prefix_position.stop)
    assert truncated_position.remaining_fraction == prefix_position.remaining_fraction
    assert truncated_position.realized_r() == pytest.approx(prefix_position.realized_r())
