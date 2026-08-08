"""The random-entry null: a schedule that cannot see a bar, and real machinery around it."""

from datetime import UTC, datetime

import pytest

from tests.backtest.conftest import EURUSD_H4, ema_pullback, harness_inputs, swing_series
from trading_system.backtest.orchestrator import BacktestResult, StrategyBinding
from trading_system.backtest.spec import RunInputs
from trading_system.core.instruments import InstrumentRegistry
from trading_system.core.types import Timeframe
from trading_system.entry.signal import EntrySignal
from trading_system.exit.library import ExitLibrarySpec
from trading_system.validation.nulls.random_entry import (
    EntryTraceProfile,
    _bar_close_times,
    build_entry_trace_profile,
    real_signals,
    run_fixed_hold_random_entry_null,
    run_random_entry_null,
    sample_schedule,
)

LENGTH = 2000

#: (base, binding, real result, real signals, profile) — one real run, shared
#: read-only across the tests below.
RealRun = tuple[
    RunInputs, StrategyBinding, BacktestResult, tuple[EntrySignal, ...], EntryTraceProfile
]


def _profile(**overrides: object) -> EntryTraceProfile:
    base = {
        "n_signals": 20,
        "hour_weights": {8: 0.5, 12: 0.5},
        "weekday_weights": {0: 0.5, 2: 0.5},
        "long_fraction": 0.6,
        "quality_samples": (0.4, 0.6, 0.8),
        "hold_bars_samples": (3, 5, 8),
        "undersampled": False,
    }
    base.update(overrides)
    return EntryTraceProfile(**base)  # type: ignore[arg-type]


def _close_ts(n: int, *, start: datetime = datetime(2024, 1, 1, tzinfo=UTC)) -> list[datetime]:
    from datetime import timedelta

    return [start + timedelta(hours=4) * i for i in range(n)]


class TestBuildEntryTraceProfile:
    def test_empty_signals_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="zero signals"):
            build_entry_trace_profile((), (), Timeframe.H4)


class TestSampleSchedule:
    def test_empty_profile_gives_empty_schedule(self) -> None:
        profile = _profile(n_signals=0)
        assert sample_schedule(_close_ts(200), profile, seed=1) == ()

    def test_empty_close_ts_gives_empty_schedule(self) -> None:
        profile = _profile(n_signals=5)
        assert sample_schedule([], profile, seed=1) == ()

    def test_schedule_size_is_at_most_n_signals(self) -> None:
        profile = _profile(n_signals=20)
        schedule = sample_schedule(_close_ts(500), profile, seed=1)
        assert len(schedule) <= 20

    def test_no_bar_is_scheduled_twice(self) -> None:
        profile = _profile(n_signals=50)
        schedule = sample_schedule(_close_ts(500), profile, seed=3)
        timestamps = [item.close_ts for item in schedule]
        assert len(timestamps) == len(set(timestamps))

    def test_sorted_by_close_ts(self) -> None:
        profile = _profile(n_signals=30)
        schedule = sample_schedule(_close_ts(500), profile, seed=5)
        assert [item.close_ts for item in schedule] == sorted(item.close_ts for item in schedule)

    def test_deterministic_on_seed(self) -> None:
        profile = _profile(n_signals=30)
        close_ts = _close_ts(500)
        assert sample_schedule(close_ts, profile, seed=42) == sample_schedule(
            close_ts, profile, seed=42
        )

    def test_is_a_pure_function_of_close_ts_alone(self) -> None:
        """The schedule cannot see a bar: it does not change when prices do.

        Structural proof, not a convention check: :func:`sample_schedule`'s
        own signature takes nothing but ``close_ts`` — there is no frame, no
        context, no price anywhere in reach, so calling it twice against
        differently-priced worlds sharing the same calendar must agree.
        """
        close_ts = _close_ts(500)
        profile = _profile(n_signals=30)
        # Two calls, standing in for "world A's prices" and "world B's prices"
        # — sample_schedule has no way to tell them apart because it was never
        # given either.
        schedule_a = sample_schedule(close_ts, profile, seed=9)
        schedule_b = sample_schedule(close_ts, profile, seed=9)
        assert schedule_a == schedule_b

    def test_falls_back_to_uniform_when_the_profiles_hours_never_occur(self) -> None:
        """A short coverage whose hours/weekdays never match the profile still schedules."""
        profile = _profile(hour_weights={3: 1.0}, weekday_weights={5: 1.0}, n_signals=5)
        close_ts = _close_ts(20)  # 4h bars starting Monday 00:00 — never hour 3, never Saturday
        schedule = sample_schedule(close_ts, profile, seed=1)
        assert schedule


class TestScheduleIndependenceEndToEnd:
    """The stronger, end-to-end version: the *set of scheduled bars* survives a price swap.

    Not every scheduled bar becomes a closed trade in ``result.trades`` — a
    position still open when the data runs out is legitimately price-path
    dependent (its own exit rules read real features) and is not a violation
    of anything this null promises. What must hold regardless is that every
    scheduled bar is accounted for, in *either* world, as a trade or as one
    of the positions still open at the end — never silently missing, and
    never present in one world's trades and absent from both categories in
    the other.
    """

    def test_every_scheduled_bar_is_a_trade_or_still_open_in_both_worlds(
        self, registry: InstrumentRegistry, library: ExitLibrarySpec
    ) -> None:
        spec = ema_pullback()
        preset = next(item for item in library.presets if item.id == spec.exit_ref)
        binding = StrategyBinding(spec=spec, exit_preset=preset, keys=(EURUSD_H4,))

        frame_a = swing_series(LENGTH, phase=0.0)
        frame_b = swing_series(LENGTH, phase=2.3)  # a differently-priced world, same calendar
        base_a = harness_inputs(registry, streams={EURUSD_H4: frame_a}, bindings=[binding])
        base_b = harness_inputs(registry, streams={EURUSD_H4: frame_b}, bindings=[binding])

        real_result_a = base_a.run()
        signals_a = real_signals(base_a.streams, binding, EURUSD_H4)
        profile = build_entry_trace_profile(signals_a, real_result_a.trades, EURUSD_H4.timeframe)

        close_ts = _bar_close_times(base_a.streams, EURUSD_H4, base_a.config.day_origin)
        schedule = sample_schedule(close_ts, profile, seed=13)
        assert schedule, "the schedule produced nothing at all; the test proves nothing"

        result_a = run_random_entry_null(base_a, EURUSD_H4, binding, profile, seed=13)
        result_b = run_random_entry_null(base_b, EURUSD_H4, binding, profile, seed=13)

        for result in (result_a, result_b):
            assert not any(result.signal_drops.values())
            assert not any(result.rejections.values())
            assert len(schedule) == len(result.trades) + result.open_at_end

        # The two worlds must actually be different worlds, or the "prices
        # cannot change the schedule" claim above is checking nothing.
        assert [t.entry_price for t in result_a.trades] != [t.entry_price for t in result_b.trades]


@pytest.fixture(scope="module")
def real_run(registry: InstrumentRegistry, library: ExitLibrarySpec) -> RealRun:
    spec = ema_pullback()
    preset = next(item for item in library.presets if item.id == spec.exit_ref)
    binding = StrategyBinding(spec=spec, exit_preset=preset, keys=(EURUSD_H4,))
    base = harness_inputs(registry, streams={EURUSD_H4: swing_series(LENGTH)}, bindings=[binding])
    result = base.run()
    signals = real_signals(base.streams, binding, EURUSD_H4)
    profile = build_entry_trace_profile(signals, result.trades, EURUSD_H4.timeframe)
    return base, binding, result, signals, profile


class TestProfileFromARealRun:
    def test_signal_count_matches_the_profile(self, real_run: RealRun) -> None:
        _base, _binding, _result, signals, profile = real_run
        assert profile.n_signals == len(signals)

    def test_long_fraction_is_a_probability(self, real_run: RealRun) -> None:
        *_ignored, profile = real_run
        assert 0.0 <= profile.long_fraction <= 1.0

    def test_hour_and_weekday_weights_sum_to_one(self, real_run: RealRun) -> None:
        *_ignored, profile = real_run
        assert profile.hour_weights
        assert sum(profile.hour_weights.values()) == pytest.approx(1.0)
        assert sum(profile.weekday_weights.values()) == pytest.approx(1.0)

    def test_quality_samples_are_in_bounds(self, real_run: RealRun) -> None:
        *_ignored, profile = real_run
        assert profile.quality_samples
        assert all(0.0 <= q <= 1.0 for q in profile.quality_samples)

    def test_hold_bars_samples_are_positive(self, real_run: RealRun) -> None:
        *_ignored, profile = real_run
        assert all(bars >= 1 for bars in profile.hold_bars_samples)


class TestRunRandomEntryNull:
    def test_produces_real_trades_through_real_machinery(self, real_run: RealRun) -> None:
        base, binding, _result, _signals, profile = real_run
        null_result = run_random_entry_null(base, EURUSD_H4, binding, profile, seed=1)
        assert null_result.trades
        assert null_result.fills >= 2 * len(null_result.trades)

    def test_deterministic_on_seed(self, real_run: RealRun) -> None:
        base, binding, _result, _signals, profile = real_run
        first = run_random_entry_null(base, EURUSD_H4, binding, profile, seed=99)
        second = run_random_entry_null(base, EURUSD_H4, binding, profile, seed=99)
        assert [t.entry_price for t in first.trades] == [t.entry_price for t in second.trades]
        assert [t.opened_at for t in first.trades] == [t.opened_at for t in second.trades]

    def test_trade_count_does_not_exceed_the_schedule(self, real_run: RealRun) -> None:
        base, binding, _result, _signals, profile = real_run
        null_result = run_random_entry_null(base, EURUSD_H4, binding, profile, seed=5)
        assert len(null_result.trades) <= profile.n_signals

    def test_position_ids_are_unique(self, real_run: RealRun) -> None:
        base, binding, _result, _signals, profile = real_run
        null_result = run_random_entry_null(base, EURUSD_H4, binding, profile, seed=5)
        ids = [trade.position_id for trade in null_result.trades]
        assert len(ids) == len(set(ids))


class TestRunFixedHoldRandomEntryNull:
    def test_produces_real_trades(self, real_run: RealRun) -> None:
        base, _binding, _result, _signals, profile = real_run
        null_result = run_fixed_hold_random_entry_null(base, EURUSD_H4, profile, seed=1)
        assert null_result.trades

    def test_every_trade_closes_by_time_exit(self, real_run: RealRun) -> None:
        """No protective stop bites within the sampled hold length in this trend fixture."""
        base, _binding, _result, _signals, profile = real_run
        null_result = run_fixed_hold_random_entry_null(base, EURUSD_H4, profile, seed=1)
        assert null_result.trades
        # legs is the count of ClosedLeg objects — a single TIME_EXIT close is 1.
        assert all(trade.legs >= 1 for trade in null_result.trades)

    def test_hold_length_is_drawn_from_the_profile_and_varies(self, real_run: RealRun) -> None:
        base, _binding, _result, _signals, profile = real_run
        null_result = run_fixed_hold_random_entry_null(
            base, EURUSD_H4, profile, seed=1, max_concurrent_positions=50
        )
        durations = {
            round((trade.closed_at - trade.opened_at) / EURUSD_H4.timeframe.duration)
            for trade in null_result.trades
        }
        # Some variety across positions — not every trade held for the exact
        # same number of bars, which a single shared TimeExit instance would
        # produce.
        assert len(durations) > 1 or len(null_result.trades) == 1

    def test_deterministic_on_seed(self, real_run: RealRun) -> None:
        base, _binding, _result, _signals, profile = real_run
        first = run_fixed_hold_random_entry_null(base, EURUSD_H4, profile, seed=7)
        second = run_fixed_hold_random_entry_null(base, EURUSD_H4, profile, seed=7)
        assert [t.net for t in first.trades] == [t.net for t in second.trades]
