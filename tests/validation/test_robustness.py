"""Perturbing the tape: valid bars by construction, matched volatility, zero drift."""

import math
import statistics
from datetime import UTC, datetime, timedelta

import pytest

from tests.backtest.conftest import swing_series
from trading_system.backtest.clock import StreamKey
from trading_system.core.types import Timeframe
from trading_system.data.models import OHLCVFrame
from trading_system.data.resample import FX_DAY_ORIGIN
from trading_system.validation.robustness import (
    PeriodConsistency,
    RobustnessRun,
    add_price_noise,
    build_robustness_report,
    daily_sigma,
    period_consistency,
    shift_start,
    summarise_synthetic,
    synthetic_frame,
    synthetic_streams,
)

EURUSD_H4 = StreamKey("EURUSD", Timeframe.H4)


def _ohlc_is_valid(frame: OHLCVFrame) -> bool:
    """Whether every bar satisfies ``low <= min(open, close) <= max(open, close) <= high``."""
    table = frame.df
    for row in table.iter_rows(named=True):
        if not (
            row["low"] <= min(row["open"], row["close"])
            and max(row["open"], row["close"]) <= row["high"]
        ):
            return False
    return True


class TestSyntheticTapesAreValidByConstruction:
    """Not repaired afterwards: the generator cannot emit a malformed bar."""

    def test_every_synthetic_bar_is_valid_ohlc(self) -> None:
        real = swing_series(600)
        built = synthetic_frame(real, day_origin=FX_DAY_ORIGIN, seed=0)
        assert _ohlc_is_valid(built)
        assert len(built) == len(real)

    def test_the_time_grid_is_untouched(self) -> None:
        real = swing_series(400)
        built = synthetic_frame(real, day_origin=FX_DAY_ORIGIN, seed=1)
        assert built.df["timestamp"].to_list() == real.df["timestamp"].to_list()

    def test_volume_is_carried_across_rather_than_invented(self) -> None:
        real = swing_series(400)
        built = synthetic_frame(real, day_origin=FX_DAY_ORIGIN, seed=1)
        assert built.df["volume"].to_list() == real.df["volume"].to_list()

    def test_the_tape_is_reproducible_and_seed_dependent(self) -> None:
        real = swing_series(300)
        first = synthetic_frame(real, day_origin=FX_DAY_ORIGIN, seed=5)
        again = synthetic_frame(real, day_origin=FX_DAY_ORIGIN, seed=5)
        other = synthetic_frame(real, day_origin=FX_DAY_ORIGIN, seed=6)
        assert first.df["close"].to_list() == again.df["close"].to_list()
        assert first.df["close"].to_list() != other.df["close"].to_list()

    def test_shapes_are_generated_not_borrowed_from_the_real_bars(self) -> None:
        """The line against the permutation null, which reattaches real shapes."""
        real = swing_series(500)
        built = synthetic_frame(real, day_origin=FX_DAY_ORIGIN, seed=2)

        def ranges(frame: OHLCVFrame) -> list[float]:
            table = frame.df
            return sorted(
                high / low
                for high, low in zip(table["high"].to_list(), table["low"].to_list(), strict=True)
            )

        assert ranges(built) != pytest.approx(ranges(real)), (
            "a synthetic tape that reproduced the real multiset of bar ranges "
            "would be the permutation null under a second name"
        )


class TestVolatilityIsMatchedAndDriftIsNot:
    def test_the_synthetic_tape_has_comparable_volatility(self) -> None:
        real = swing_series(2000)
        built = synthetic_frame(real, day_origin=FX_DAY_ORIGIN, seed=3)

        def sigma(frame: OHLCVFrame) -> float:
            closes = frame.df["close"].to_list()
            return statistics.stdev(
                math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))
            )

        ratio = sigma(built) / sigma(real)
        assert 0.5 < ratio < 2.0, f"volatility ratio {ratio:.2f} is not comparable"

    def test_drift_is_zero_by_construction_unlike_the_trending_real_series(self) -> None:
        real = swing_series(3000)
        closes = real.df["close"].to_list()
        real_drift = math.log(closes[-1] / closes[0])

        drifts = []
        for seed in range(12):
            built = synthetic_frame(real, day_origin=FX_DAY_ORIGIN, seed=seed)
            values = built.df["close"].to_list()
            drifts.append(math.log(values[-1] / values[0]))

        assert abs(statistics.fmean(drifts)) < abs(real_drift), (
            "the fixture must actually trend for this to mean anything"
        )
        assert min(drifts) < 0 < max(drifts), (
            "a driftless walk must go both ways across seeds; a matched drift "
            "would make buy-and-hold earn on the null"
        )

    def test_the_volatility_schedule_is_cut_per_trading_day(self) -> None:
        real = swing_series(1000)
        per_day = daily_sigma(real, FX_DAY_ORIGIN)
        assert len(per_day) > 1, "a multi-day fixture must produce a multi-day schedule"
        assert all(value >= 0 for value in per_day.values())


class TestCoarseStreamsAreRebuiltNotGeneratedSeparately:
    def test_an_h4_stream_is_resampled_from_its_own_synthetic_h1(self) -> None:
        from trading_system.data.resample import resample

        h1 = StreamKey("EURUSD", Timeframe.H1)
        h4 = StreamKey("EURUSD", Timeframe.H4)
        real_h1 = OHLCVFrame(swing_series(1200).df, symbol="EURUSD", timeframe=Timeframe.H1)
        real_h4 = resample(real_h1, Timeframe.H4, origin=FX_DAY_ORIGIN)
        built = synthetic_streams({h1: real_h1, h4: real_h4}, day_origin=FX_DAY_ORIGIN, seed=0)
        expected = resample(built[h1], Timeframe.H4, origin=FX_DAY_ORIGIN)
        assert built[h4].df["close"].to_list() == expected.df["close"].to_list()


class TestNoiseKeepsBarsValid:
    def test_noise_never_produces_a_malformed_bar(self) -> None:
        real = swing_series(500)
        noisy = add_price_noise(real, relative_sigma=0.002, seed=0)
        assert _ohlc_is_valid(noisy)

    def test_noise_actually_moves_prices(self) -> None:
        real = swing_series(300)
        noisy = add_price_noise(real, relative_sigma=0.001, seed=0)
        assert noisy.df["close"].to_list() != real.df["close"].to_list()

    def test_zero_noise_is_the_identity(self) -> None:
        real = swing_series(200)
        same = add_price_noise(real, relative_sigma=0.0, seed=0)
        assert same.df["close"].to_list() == pytest.approx(real.df["close"].to_list())

    def test_noise_is_reproducible(self) -> None:
        real = swing_series(200)
        first = add_price_noise(real, relative_sigma=0.001, seed=4)
        again = add_price_noise(real, relative_sigma=0.001, seed=4)
        assert first.df["close"].to_list() == again.df["close"].to_list()


class TestStartShiftMovesEveryStreamByTheSameSpan:
    def test_a_coarse_stream_loses_the_same_period_as_the_fine_one(self) -> None:
        from trading_system.data.resample import resample

        h1 = StreamKey("EURUSD", Timeframe.H1)
        h4 = StreamKey("EURUSD", Timeframe.H4)
        real_h1 = OHLCVFrame(swing_series(1200).df, symbol="EURUSD", timeframe=Timeframe.H1)
        real_h4 = resample(real_h1, Timeframe.H4, origin=FX_DAY_ORIGIN)
        shifted = shift_start({h1: real_h1, h4: real_h4}, 24)
        assert shifted[h1].start is not None and shifted[h4].start is not None
        assert real_h1.start is not None
        assert shifted[h1].start >= real_h1.start + timedelta(hours=24)
        assert shifted[h4].start >= real_h1.start + timedelta(hours=24), (
            "a coarse stream must lose the same calendar span, not the same bar count"
        )

    def test_a_zero_shift_is_the_identity(self) -> None:
        h1 = StreamKey("EURUSD", Timeframe.H1)
        real = OHLCVFrame(swing_series(300).df, symbol="EURUSD", timeframe=Timeframe.H1)
        assert len(shift_start({h1: real}, 0)[h1]) == len(real)


class TestPeriodsAreEqualByArithmetic:
    """No boundary is ever chosen; they come from the two ends and a count."""

    def _trades(self, stamps: list[datetime], rs: list[float]) -> list[object]:
        class _Trade:
            def __init__(self, closed_at: datetime, realized_r: float) -> None:
                self.closed_at = closed_at
                self.realized_r = realized_r

        return [_Trade(ts, r) for ts, r in zip(stamps, rs, strict=True)]

    def test_periods_tile_the_span_exactly_and_in_order(self) -> None:
        start = datetime(2021, 1, 1, tzinfo=UTC)
        end = start + timedelta(days=400)
        result = period_consistency([], start=start, end=end, n_periods=4)  # type: ignore[arg-type]
        assert result.periods[0].start == start
        assert result.periods[-1].end == end
        for earlier, later in zip(result.periods, result.periods[1:], strict=False):
            assert earlier.end == later.start

    def test_a_result_concentrated_in_one_period_is_visible(self) -> None:
        start = datetime(2021, 1, 1, tzinfo=UTC)
        end = start + timedelta(days=400)
        stamps = [start + timedelta(days=10 + i) for i in range(20)]
        stamps += [start + timedelta(days=310 + i) for i in range(20)]
        rs = [3.0] * 20 + [-0.5] * 20
        result = period_consistency(
            self._trades(stamps, rs),  # type: ignore[arg-type]
            start=start,
            end=end,
            n_periods=4,
        )
        assert result.n_profitable == 1
        assert result.n_empty == 2
        assert result.dispersion is not None and result.dispersion > 0

    def test_fewer_than_two_periods_is_refused(self) -> None:
        start = datetime(2021, 1, 1, tzinfo=UTC)
        with pytest.raises(ValueError, match="at least 2"):
            period_consistency([], start=start, end=start + timedelta(days=10), n_periods=1)  # type: ignore[arg-type]


class TestSyntheticSummaryPlacesTheRealRun:
    def test_a_real_run_above_every_tape_lands_at_the_top(self) -> None:
        runs = tuple(
            RobustnessRun(f"synthetic_{i}", 20, -0.1 + 0.01 * i, None, None, None)
            for i in range(10)
        )
        summary = summarise_synthetic(runs, real_expectancy_r=1.0)
        assert summary.real_percentile == 100.0

    def test_a_real_run_inside_the_pack_lands_in_the_middle(self) -> None:
        runs = tuple(
            RobustnessRun(f"synthetic_{i}", 20, float(i), None, None, None) for i in range(10)
        )
        summary = summarise_synthetic(runs, real_expectancy_r=4.5)
        assert 40.0 <= summary.real_percentile <= 60.0  # type: ignore[operator]

    def test_a_tape_that_barely_traded_is_visible_rather_than_averaged_away(self) -> None:
        runs = (
            RobustnessRun("synthetic_0", 1, 5.0, None, None, None),
            RobustnessRun("synthetic_1", 40, -0.1, None, None, None),
            RobustnessRun("synthetic_2", 38, -0.2, None, None, None),
        )
        summary = summarise_synthetic(runs, real_expectancy_r=0.1)
        assert summary.median_trade_count == 38.0


class TestReportDerivesTheScalarsAVerdictReads:
    def test_dispersion_and_retention_come_out_of_the_runs(self) -> None:
        shifts = tuple(
            RobustnessRun(f"start_shift_{i}", 50, value, None, None, None)
            for i, value in enumerate((0.20, 0.22, 0.18, 0.21))
        )
        noise = (
            RobustnessRun("noise_0.0002", 50, 0.20, None, None, None),
            RobustnessRun("noise_0.001", 40, 0.10, None, None, None),
        )
        summary = summarise_synthetic(
            (RobustnessRun("synthetic_0", 10, -0.3, None, None, None),), real_expectancy_r=0.2
        )
        empty = PeriodConsistency(
            periods=(), n_periods=4, n_profitable=3, n_empty=0, dispersion=0.0
        )
        report = build_robustness_report(shifts, noise, summary, empty)
        assert report.start_shift_dispersion == pytest.approx(
            statistics.stdev([0.20, 0.22, 0.18, 0.21])
        )
        assert report.noise_retention == pytest.approx(0.5)

    def test_retention_is_undefined_rather_than_infinite_through_zero(self) -> None:
        noise = (
            RobustnessRun("noise_low", 50, 0.0, None, None, None),
            RobustnessRun("noise_high", 40, -0.3, None, None, None),
        )
        summary = summarise_synthetic(
            (RobustnessRun("synthetic_0", 10, -0.3, None, None, None),), real_expectancy_r=0.0
        )
        empty = PeriodConsistency(
            periods=(), n_periods=4, n_profitable=2, n_empty=0, dispersion=0.0
        )
        report = build_robustness_report((), noise, summary, empty)
        assert report.noise_retention is None
