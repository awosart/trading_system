"""Resampling correctness, with lookahead as the headline concern."""

from datetime import UTC, date, datetime, time, timedelta

import pytest

from trading_system.core.exceptions import DataError
from trading_system.core.types import Timeframe
from trading_system.data.models import OHLCVFrame
from trading_system.data.resample import (
    FX_DAY_ORIGIN,
    DayOrigin,
    resample,
    trading_day,
    trading_day_of_close,
)
from trading_system.data.sessions import AssetClass, TradingCalendar

from .conftest import make_frame


def test_minute_to_hour_produces_exact_ohlcv(minute_frame: OHLCVFrame) -> None:
    """1m -> H1 on synthetic data yields precisely the expected aggregation."""
    hourly = resample(minute_frame, Timeframe.H1)

    # 210 minutes from 09:00 covers three whole hours; the fourth is partial.
    assert len(hourly) == 3
    rows = hourly.df.rows(named=True)

    assert rows[0]["timestamp"] == datetime(2024, 1, 1, 9, 0, tzinfo=UTC)
    assert rows[0]["open"] == 0.0  # first bar's open
    assert rows[0]["high"] == 61.0  # max over bars 0..59 -> 59 + 2
    assert rows[0]["low"] == -2.0  # min over bars 0..59 -> 0 - 2
    assert rows[0]["close"] == 60.0  # last bar's close -> 59 + 1
    assert rows[0]["volume"] == 60.0  # sixty 1.0 volumes

    assert rows[1]["timestamp"] == datetime(2024, 1, 1, 10, 0, tzinfo=UTC)
    assert rows[1]["open"] == 60.0
    assert rows[1]["high"] == 121.0
    assert rows[1]["low"] == 58.0
    assert rows[1]["close"] == 120.0
    assert rows[1]["volume"] == 60.0

    assert rows[2]["timestamp"] == datetime(2024, 1, 1, 11, 0, tzinfo=UTC)
    assert rows[2]["close"] == 180.0
    assert rows[2]["volume"] == 60.0


def test_timestamps_label_the_window_open(minute_frame: OHLCVFrame) -> None:
    """label='left': a bar is stamped with its opening time, never its close."""
    hourly = resample(minute_frame, Timeframe.H1)
    assert hourly.df["timestamp"].to_list()[0] == datetime(2024, 1, 1, 9, 0, tzinfo=UTC)


def test_incomplete_trailing_window_is_dropped(minute_frame: OHLCVFrame) -> None:
    """The partial 12:00 hour is withheld: its close is not knowable at 12:29."""
    hourly = resample(minute_frame, Timeframe.H1)
    assert datetime(2024, 1, 1, 12, 0, tzinfo=UTC) not in hourly.df["timestamp"].to_list()

    including_partial = resample(minute_frame, Timeframe.H1, drop_incomplete=False)
    assert datetime(2024, 1, 1, 12, 0, tzinfo=UTC) in including_partial.df["timestamp"].to_list()


def test_h4_bar_closes_only_after_its_final_minute() -> None:
    """An H4 bar appears only once every minute of its window exists.

    Feeding exactly one minute short of the window must yield nothing; adding
    that final minute must publish the bar. This is the no-lookahead guarantee
    stated as a boundary condition.
    """
    start = datetime(2024, 1, 1, 8, 0, tzinfo=UTC)  # aligns to a UTC 4h boundary

    one_minute_short = make_frame(start, 239)
    assert len(resample(one_minute_short, Timeframe.H4)) == 0

    complete = make_frame(start, 240)
    closed = resample(complete, Timeframe.H4)
    assert len(closed) == 1
    assert closed.df["timestamp"].item(0) == start
    assert closed.df["close"].item(0) == 240.0  # bar 239's close
    assert closed.df["volume"].item(0) == 240.0


def test_partial_window_never_leaks_future_extremes() -> None:
    """A dropped partial window cannot contribute its high to an emitted bar."""
    start = datetime(2024, 1, 1, 8, 0, tzinfo=UTC)
    frame = make_frame(start, 250)  # 240 complete + 10 into the next window
    resampled = resample(frame, Timeframe.H4)
    assert len(resampled) == 1
    # Bars 240..249 have highs up to 251; none may appear in the closed bar.
    assert resampled.df["high"].item(0) == 241.0


def test_daily_bars_anchor_to_session_close_across_dst() -> None:
    """A day anchored to 17:00 New York tracks DST rather than a fixed UTC offset."""
    start = datetime(2024, 3, 7, 0, 0, tzinfo=UTC)
    frame = make_frame(start, 24 * 8, timeframe=Timeframe.H1)
    daily = resample(frame, Timeframe.D1, origin=FX_DAY_ORIGIN)

    stamps = daily.df["timestamp"].to_list()
    # US DST began 2024-03-10, so 17:00 New York moves from 22:00 to 21:00 UTC.
    assert datetime(2024, 3, 7, 22, 0, tzinfo=UTC) in stamps
    assert datetime(2024, 3, 8, 22, 0, tzinfo=UTC) in stamps
    assert datetime(2024, 3, 11, 21, 0, tzinfo=UTC) in stamps
    assert datetime(2024, 3, 12, 21, 0, tzinfo=UTC) in stamps


def test_dst_transition_day_is_short() -> None:
    """The spring-forward trading day holds 23 hourly bars, not 24."""
    start = datetime(2024, 3, 7, 0, 0, tzinfo=UTC)
    frame = make_frame(start, 24 * 8, timeframe=Timeframe.H1)
    daily = resample(frame, Timeframe.D1, origin=FX_DAY_ORIGIN)

    row = daily.df.filter(daily.df["timestamp"] == datetime(2024, 3, 9, 22, 0, tzinfo=UTC)).rows(
        named=True
    )[0]
    assert row["volume"] == 23.0


def test_utc_midnight_is_the_default_anchor() -> None:
    """Without an origin, daily windows start at UTC midnight."""
    start = datetime(2024, 6, 3, 0, 0, tzinfo=UTC)
    frame = make_frame(start, 48, timeframe=Timeframe.H1)
    daily = resample(frame, Timeframe.D1)
    assert daily.df["timestamp"].to_list() == [
        datetime(2024, 6, 3, 0, 0, tzinfo=UTC),
        datetime(2024, 6, 4, 0, 0, tzinfo=UTC),
    ]


def test_custom_origin_shifts_the_window() -> None:
    """An explicit origin anchors intraday windows to a local wall-clock time."""
    origin = DayOrigin(tz="UTC", at=time(2, 0))
    frame = make_frame(datetime(2024, 6, 3, 0, 0, tzinfo=UTC), 24 * 60)
    hourly = resample(frame, Timeframe.H4, origin=origin)
    assert hourly.df["timestamp"].item(0) == datetime(2024, 6, 3, 2, 0, tzinfo=UTC)


def test_leading_partial_window_is_dropped() -> None:
    """A window the source joins midway through is withheld.

    With a 2-hour origin the grid runs 02:00, 06:00, ... so data starting at
    00:00 falls inside the previous 22:00 window. That window would be stamped
    22:00 while holding only 00:00-02:00 data, misreporting its open, extremes
    and volume, so it must not be published.
    """
    origin = DayOrigin(tz="UTC", at=time(2, 0))
    frame = make_frame(datetime(2024, 6, 3, 0, 0, tzinfo=UTC), 24 * 60)

    dropped = resample(frame, Timeframe.H4, origin=origin)
    assert datetime(2024, 6, 2, 22, 0, tzinfo=UTC) not in dropped.df["timestamp"].to_list()

    kept = resample(frame, Timeframe.H4, origin=origin, drop_incomplete=False)
    partial = kept.df.rows(named=True)[0]
    assert partial["timestamp"] == datetime(2024, 6, 2, 22, 0, tzinfo=UTC)
    assert partial["volume"] == 120.0  # two hours of minutes, not four


def test_volume_sums_and_extremes_span_the_window() -> None:
    """M1 -> M15 keeps first/max/min/last/sum semantics."""
    frame = make_frame(datetime(2024, 1, 1, 0, 0, tzinfo=UTC), 30)
    quarter = resample(frame, Timeframe.M15)
    rows = quarter.df.rows(named=True)
    assert len(rows) == 2
    assert (rows[0]["open"], rows[0]["close"], rows[0]["volume"]) == (0.0, 15.0, 15.0)
    assert (rows[1]["open"], rows[1]["close"], rows[1]["volume"]) == (15.0, 30.0, 15.0)


def test_empty_frame_resamples_to_empty() -> None:
    from trading_system.data.models import OHLCVFrame

    empty = OHLCVFrame.empty("EURUSD", Timeframe.M1)
    assert resample(empty, Timeframe.H1).is_empty


@pytest.mark.parametrize(
    ("source", "target"),
    [(Timeframe.H1, Timeframe.M1), (Timeframe.H1, Timeframe.H1)],
)
def test_rejects_non_upward_steps(source: Timeframe, target: Timeframe) -> None:
    frame = make_frame(datetime(2024, 1, 1, tzinfo=UTC), 10, timeframe=source)
    with pytest.raises(DataError):
        resample(frame, target)


@pytest.mark.parametrize(
    ("source", "target"),
    [
        (Timeframe.M1, Timeframe.H4),
        (Timeframe.M5, Timeframe.H1),
        (Timeframe.M15, Timeframe.D1),
        (Timeframe.H1, Timeframe.D1),
    ],
)
def test_supported_steps_are_accepted(source: Timeframe, target: Timeframe) -> None:
    """Every currently-defined upward pair divides evenly and must be allowed.

    The divisibility guard in ``_validate_step`` is therefore unreachable today;
    it exists so that adding an awkward timeframe later fails loudly instead of
    silently emitting misaligned bars.
    """
    frame = make_frame(datetime(2024, 1, 1, tzinfo=UTC), 5, timeframe=source)
    resample(frame, target)  # must not raise


def test_resampled_frame_carries_the_target_timeframe(minute_frame: OHLCVFrame) -> None:
    assert resample(minute_frame, Timeframe.H1).timeframe is Timeframe.H1
    assert resample(minute_frame, Timeframe.H1).symbol == minute_frame.symbol


def test_resampling_is_associative_for_nested_windows() -> None:
    """M1->H1->H4 agrees with M1->H4; window nesting must not change the result."""
    frame = make_frame(datetime(2024, 1, 1, 0, 0, tzinfo=UTC), 480)
    direct = resample(frame, Timeframe.H4)
    staged = resample(resample(frame, Timeframe.H1), Timeframe.H4)
    assert direct.df.equals(staged.df)


def test_gap_in_source_does_not_shift_window_boundaries() -> None:
    """A missing minute leaves the hour's boundaries and label untouched."""
    frame = make_frame(datetime(2024, 1, 1, 0, 0, tzinfo=UTC), 120)
    with_hole = frame.with_df(
        frame.df.filter(frame.df["timestamp"] != datetime(2024, 1, 1, 0, 30, tzinfo=UTC))
    )
    hourly = resample(with_hole, Timeframe.H1)
    assert hourly.df["timestamp"].item(0) == datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
    assert hourly.df["volume"].item(0) == 59.0


class TestTradingDayOfClose:
    """A bar's close can land inside a window the market never opened in.

    2024-08-23 is a Friday, 2024-08-22 the Thursday before it. An H4 bar
    opening at 13:00 EDT (17:00 UTC) that Friday — the last one before the
    weekly close — finishes at 2024-08-24T00:00 UTC (20:00 EDT), three hours
    into the ``[Fri 17:00 NY, Sat 17:00 NY)`` window FX never trades in at
    all. ``trading_day`` alone labels it "Friday" anyway; this is the
    instant the fold-back exists for.
    """

    #: The verified spillover instant: naively labelled 2024-08-23 (Friday),
    #: correctly 2024-08-22 (Thursday) once the market being shut is known.
    SPILLOVER = datetime(2024, 8, 24, 0, 0, tzinfo=UTC)

    def test_the_naive_label_is_the_defect_this_function_fixes(self) -> None:
        """Establishes what's being corrected, independent of the fix itself."""
        assert trading_day(self.SPILLOVER, FX_DAY_ORIGIN) == date(2024, 8, 23)

    def test_an_open_market_instant_passes_through_unchanged(self) -> None:
        normal_tuesday = datetime(2024, 8, 6, 14, 0, tzinfo=UTC)
        calendar = TradingCalendar(AssetClass.FX)
        assert calendar.is_open(normal_tuesday)
        assert trading_day_of_close(normal_tuesday, FX_DAY_ORIGIN, calendar) == trading_day(
            normal_tuesday, FX_DAY_ORIGIN
        )

    def test_a_weekend_spillover_folds_back_to_the_thursday_that_produced_it(self) -> None:
        calendar = TradingCalendar(AssetClass.FX)
        assert not calendar.is_open(self.SPILLOVER)
        assert trading_day_of_close(self.SPILLOVER, FX_DAY_ORIGIN, calendar) == date(2024, 8, 22)

    def test_a_crypto_calendar_never_folds_back(self) -> None:
        """The exact same closed-for-FX instant is a normal Saturday for crypto."""
        calendar = TradingCalendar(AssetClass.CRYPTO)
        assert calendar.is_open(self.SPILLOVER)
        assert trading_day_of_close(self.SPILLOVER, FX_DAY_ORIGIN, calendar) == trading_day(
            self.SPILLOVER, FX_DAY_ORIGIN
        )

    def test_an_instant_needing_more_than_one_days_step_still_resolves(self) -> None:
        """Sunday before the reopen: one day back lands on Saturday, still shut."""
        deep_sunday = datetime(2024, 8, 25, 18, 0, tzinfo=UTC)  # Sunday 14:00 EDT
        calendar = TradingCalendar(AssetClass.FX)
        assert not calendar.is_open(deep_sunday)
        assert not calendar.is_open(deep_sunday - timedelta(days=1))
        assert trading_day_of_close(deep_sunday, FX_DAY_ORIGIN, calendar) == date(2024, 8, 22)

    def test_a_closure_longer_than_the_search_limit_is_refused_not_guessed(self) -> None:
        holidays = frozenset(date(2024, 1, day) for day in range(1, 20))
        calendar = TradingCalendar(AssetClass.EQUITY, holidays=holidays)
        with pytest.raises(ValueError, match="market-closed window"):
            trading_day_of_close(datetime(2024, 1, 18, tzinfo=UTC), FX_DAY_ORIGIN, calendar)
