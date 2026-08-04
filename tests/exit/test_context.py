"""ExitContext: what it exposes, and the proof that none of it reaches forward.

The equivalence technique is P06's, re-run against the wrapper. Wrapping a
``BarContext`` is only safe if the wrapper cannot smuggle anything past it, and
the one genuinely new surface — ticks — is the easiest place to do exactly that.
"""

import inspect

import pytest

from trading_system.entry.context import PRICE_FIELDS
from trading_system.exit.context import ExitContext, Tick, exit_contexts

from .conftest import Bar, bar_close_ts, bar_open_ts, series, ticks

#: The whole public surface of an exit context: the bar context's, plus ticks.
#: Anything added here is a new route to data and has to be argued for.
EXPECTED_API = frozenset(
    {
        "bars",
        "index",
        "symbol",
        "timeframe",
        "bar_open_ts",
        "bar_close_ts",
        "has_ticks",
        "ticks",
        "price",
        "feature",
        "labels",
        "feature_snapshot",
    }
)

BARS: list[Bar] = [
    (1.1000 + 0.001 * i, 1.1020 + 0.001 * i, 1.0990 + 0.001 * i, 1.1010 + 0.001 * i)
    for i in range(12)
]


class TestNoLookahead:
    def test_public_api_is_exactly_the_history_only_surface(self) -> None:
        public = {name for name in dir(ExitContext) if not name.startswith("_")}
        assert public == EXPECTED_API

    def test_there_is_no_forward_accessor_by_any_conventional_name(self) -> None:
        for name in (
            "next",
            "next_bar",
            "peek",
            "future",
            "forward",
            "bar_at",
            "at",
            "series",
            "frame",
            "df",
            "__getitem__",
            "__iter__",
            "__next__",
        ):
            assert not hasattr(ExitContext, name), f"ExitContext exposes {name!r}"

    @pytest.mark.parametrize("accessor", ["price", "feature", "labels"])
    def test_every_data_accessor_counts_backwards(self, accessor: str) -> None:
        parameters = list(inspect.signature(getattr(ExitContext, accessor)).parameters)
        assert parameters[-1] == "lookback"

    def test_a_negative_lookback_is_rejected_rather_than_wrapping(self) -> None:
        ctx = ExitContext(series(BARS).context(0))
        with pytest.raises(ValueError, match="non-negative"):
            ctx.price("close", -1)

    def test_every_accessor_agrees_with_a_series_truncated_at_the_bar(self) -> None:
        # The real proof, by exhaustion over the API: if any accessor could read
        # past ``t``, deleting everything past ``t`` would change its answer.
        full = series(BARS)
        for index in range(len(BARS)):
            truncated = full.truncated(index + 1)
            bar_ticks = ticks(index, [1.1005, 1.1015])
            here = ExitContext(full.context(index), bar_ticks)
            there = ExitContext(truncated.context(index), bar_ticks)

            assert here.index == there.index
            assert here.symbol == there.symbol
            assert here.timeframe == there.timeframe
            assert here.bar_open_ts == there.bar_open_ts
            assert here.bar_close_ts == there.bar_close_ts
            assert here.has_ticks == there.has_ticks
            assert here.ticks == there.ticks
            assert here.feature_snapshot() == there.feature_snapshot()
            for lookback in range(4):
                for field in PRICE_FIELDS:
                    assert here.price(field, lookback) == there.price(field, lookback)

    def test_context_slots_prevent_attaching_extra_state(self) -> None:
        ctx = ExitContext(series(BARS).context(1))
        with pytest.raises(AttributeError):
            ctx.smuggled = 1  # type: ignore[attr-defined]


class TestTickWindow:
    def test_a_tick_from_the_next_bar_is_refused(self) -> None:
        # The most direct lookahead available: a price that had not printed yet
        # when bar t closed, handed to a rule deciding on bar t.
        bars = series(BARS)
        with pytest.raises(ValueError, match="outside bar"):
            ExitContext(bars.context(3), [Tick(ts=bar_close_ts(3), price=1.1000)])

    def test_a_tick_from_the_previous_bar_is_refused(self) -> None:
        bars = series(BARS)
        with pytest.raises(ValueError, match="outside bar"):
            ExitContext(
                bars.context(3), [Tick(ts=bar_open_ts(3) - bars.timeframe.duration, price=1.1)]
            )

    def test_ticks_must_be_ascending(self) -> None:
        bars = series(BARS)
        out_of_order = list(reversed(ticks(2, [1.1000, 1.1010, 1.1020])))
        with pytest.raises(ValueError, match="ascending"):
            ExitContext(bars.context(2), out_of_order)

    def test_the_bar_open_is_one_timeframe_before_its_close(self) -> None:
        ctx = ExitContext(series(BARS).context(4))
        assert ctx.bar_open_ts == bar_open_ts(4)
        assert ctx.bar_close_ts == bar_close_ts(4)
        assert ctx.bar_close_ts - ctx.bar_open_ts == ctx.timeframe.duration

    def test_a_run_without_tick_data_carries_none(self) -> None:
        contexts = list(exit_contexts(series(BARS)))
        assert len(contexts) == len(BARS)
        assert not any(ctx.has_ticks for ctx in contexts)

    def test_ticks_are_filed_against_the_bar_they_belong_to(self) -> None:
        contexts = list(exit_contexts(series(BARS), {2: ticks(2, [1.1, 1.101])}))
        assert [ctx.index for ctx in contexts if ctx.has_ticks] == [2]
        assert [tick.price for tick in contexts[2].ticks] == [1.1, 1.101]

    def test_ticks_filed_against_the_wrong_bar_are_refused(self) -> None:
        with pytest.raises(ValueError, match="outside bar"):
            list(exit_contexts(series(BARS), {5: ticks(2, [1.1])}))
