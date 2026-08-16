"""The merged clock: what closes when, and what is visible to what.

The defect this whole module guards against has one shape: a bar keyed on its
``timestamp`` field, which is its OPEN. A daily bar merged that way enters the
stream twenty-four hours before it finishes, and every strategy filtering on it
reads a bar that has not happened.
"""

from datetime import UTC, datetime, timedelta

import pytest

from tests.backtest.conftest import bars
from trading_system.backtest.clock import StreamKey, bar_close_ts, day_close_ts
from trading_system.backtest.engine import BarStore, DataHandler, LookaheadError
from trading_system.core.types import Timeframe
from trading_system.data.resample import FX_DAY_ORIGIN, DayOrigin, trading_day
from trading_system.entry.context import BarSeries

#: The FX day starts at 17:00 New York, which is 21:00 UTC in summer.
SUMMER_D1_OPEN = datetime(2024, 7, 1, 21, 0, tzinfo=UTC)

#: ...and 22:00 UTC in winter.
WINTER_D1_OPEN = datetime(2024, 12, 2, 22, 0, tzinfo=UTC)


def series(frame: object) -> BarSeries:
    """Materialise a frame as the series the handler walks."""
    return BarSeries.from_frame(frame)  # type: ignore[arg-type]


class TestCloseInstants:
    """A bar becomes visible at its close, and the close is not always +duration."""

    def test_an_intraday_bar_closes_one_duration_after_it_opens(self) -> None:
        opening = datetime(2024, 3, 4, 9, 0, tzinfo=UTC)
        assert bar_close_ts(Timeframe.H1, opening, FX_DAY_ORIGIN) == opening + timedelta(hours=1)
        assert bar_close_ts(Timeframe.M15, opening, FX_DAY_ORIGIN) == opening + timedelta(
            minutes=15
        )

    @pytest.mark.parametrize(
        ("opening", "expected"),
        [
            (SUMMER_D1_OPEN, datetime(2024, 7, 2, 21, 0, tzinfo=UTC)),
            (WINTER_D1_OPEN, datetime(2024, 12, 3, 22, 0, tzinfo=UTC)),
        ],
    )
    def test_a_daily_bar_closes_on_the_day_anchor_not_24h_later(
        self, opening: datetime, expected: datetime
    ) -> None:
        """Outside a transition the two agree, which is why the bug hides."""
        assert day_close_ts(opening, FX_DAY_ORIGIN) == expected
        assert day_close_ts(opening, FX_DAY_ORIGIN) == opening + timedelta(days=1)

    def test_across_a_dst_transition_the_daily_close_is_not_24h_after_the_open(self) -> None:
        """US clocks go forward on 2024-03-10, so that trading day is 23 hours.

        ``Timeframe.duration`` says 24 hours and its own docstring says that is
        nominal. This is the bar where believing it puts the daily close an hour
        into the future — and therefore publishes a bar before it exists.
        """
        opening = datetime(2024, 3, 8, 22, 0, tzinfo=UTC)  # Friday 17:00 EST
        assert trading_day(opening, FX_DAY_ORIGIN) == datetime(2024, 3, 8).date()

        closes = day_close_ts(opening, FX_DAY_ORIGIN)
        assert closes == datetime(2024, 3, 9, 22, 0, tzinfo=UTC)

        spring_forward = datetime(2024, 3, 9, 22, 0, tzinfo=UTC)  # Saturday 17:00 EST
        assert day_close_ts(spring_forward, FX_DAY_ORIGIN) == datetime(
            2024, 3, 10, 21, 0, tzinfo=UTC
        )
        assert day_close_ts(spring_forward, FX_DAY_ORIGIN) - spring_forward == timedelta(hours=23)
        assert day_close_ts(spring_forward, FX_DAY_ORIGIN) != spring_forward + Timeframe.D1.duration

    def test_a_daily_bar_off_the_anchor_is_refused_rather_than_guessed(self) -> None:
        """A series cut on a different origin than the run uses is a load error."""
        with pytest.raises(ValueError, match="does not sit on the"):
            day_close_ts(datetime(2024, 7, 1, 9, 30, tzinfo=UTC), FX_DAY_ORIGIN)


class TestMergeOrder:
    """Question (a): who reaches the strategy first at 21:00, the H1 or the D1."""

    def _h1_and_d1(self) -> tuple[StreamKey, StreamKey, DataHandler]:
        """One UTC day of H1 bars alongside the D1 bar they compose.

        The H1 series runs 21:00 to 21:00, so its last bar closes at exactly the
        instant the daily bar does.
        """
        h1_key = StreamKey("EURUSD", Timeframe.H1)
        d1_key = StreamKey("EURUSD", Timeframe.D1)
        h1 = bars(
            [1.10 + 0.001 * i for i in range(24)],
            timeframe=Timeframe.H1,
            start=SUMMER_D1_OPEN,
        )
        d1 = bars([1.12], timeframe=Timeframe.D1, start=SUMMER_D1_OPEN)
        handler = DataHandler({h1_key: series(h1), d1_key: series(d1)}, FX_DAY_ORIGIN)
        return h1_key, d1_key, handler

    def test_the_daily_bar_is_not_published_at_its_own_timestamp(self) -> None:
        """The whole point. Its timestamp is 21:00 on day one; it closes on day two."""
        _h1, d1_key, handler = self._h1_and_d1()
        assert handler.close_times(d1_key)[0] == SUMMER_D1_OPEN + timedelta(days=1)

        first_instant = next(iter(handler.instants()))
        assert first_instant.ts == SUMMER_D1_OPEN + timedelta(hours=1)
        assert [event.key for event in first_instant.bars] == [StreamKey("EURUSD", Timeframe.H1)]

    def test_the_h1_and_the_d1_that_close_together_arrive_in_one_instant(self) -> None:
        h1_key, d1_key, handler = self._h1_and_d1()
        instants = list(handler.instants())

        final = instants[-1]
        assert final.ts == SUMMER_D1_OPEN + timedelta(days=1)
        assert [event.key for event in final.bars] == [h1_key, d1_key]

    def test_the_finer_timeframe_is_published_first_within_an_instant(self) -> None:
        """Reproducibility, not correctness: the aggregate follows its parts."""
        _h1, _d1, handler = self._h1_and_d1()
        final = list(handler.instants())[-1]
        assert [event.key.timeframe for event in final.bars] == [Timeframe.H1, Timeframe.D1]

    def test_the_h1_evaluated_at_2100_sees_the_daily_bar_that_just_closed(self) -> None:
        """Question (a)'s answer, as a property of the store rather than a rule.

        Publication precedes evaluation, so by the time anything reads the 21:00
        H1 bar, the D1 bar closing at the same instant is in the store. That is
        correct because the two carry the same information — the daily bar is an
        aggregate of bars ``<= 21:00`` — and it is what a live run sees.
        """
        h1_key, d1_key, handler = self._h1_and_d1()
        store = BarStore(handler.streams)

        for instant in handler.instants():
            for event in instant.bars:
                store.publish(event)

            if instant.ts == SUMMER_D1_OPEN + timedelta(days=1):
                assert store.has_published(d1_key)
                assert store.cursor(h1_key) == 23
                assert store.context(d1_key).price("close") == pytest.approx(1.12)

    def test_before_that_instant_no_daily_bar_is_readable_at_all(self) -> None:
        """The alternative to seeing today's D1 is not yesterday's — it is none."""
        _h1, d1_key, handler = self._h1_and_d1()
        store = BarStore(handler.streams)
        instants = list(handler.instants())

        for event in instants[0].bars:
            store.publish(event)
        assert not store.has_published(d1_key)
        with pytest.raises(LookaheadError, match="no bar has closed yet"):
            store.context(d1_key)

    def test_every_timeframe_that_exists_can_be_merged(self) -> None:
        """The table of ranks may not be a second list of which timeframes exist.

        It was one, and it went stale: ``M30`` was added to :class:`Timeframe`
        with the third vendor import, and the rank table did not learn about it.
        The store held M30 bars, the schema accepted an M30 spec, and the run
        died on a ``KeyError`` inside the heap merge — a timeframe the rest of
        the system considered real and the engine could not step through.
        Enumerating the enum here means the next member added cannot repeat it,
        and the failure is named rather than arriving from inside ``heapq``.
        """
        for timeframe in Timeframe:
            key = StreamKey("EURUSD", timeframe)
            handler = DataHandler(
                {key: series(bars([1.10, 1.11], timeframe=timeframe, start=SUMMER_D1_OPEN))},
                FX_DAY_ORIGIN,
            )
            assert [event.key for instant in handler.instants() for event in instant.bars] == [
                key,
                key,
            ], f"{timeframe.value} cannot be merged"

    def test_ranks_order_the_timeframes_finest_first(self) -> None:
        """The rank exists to break ties deterministically, so only order matters.

        Deriving it from ``duration`` rather than a written list is what makes
        the previous test's guarantee automatic; this one states the property
        that derivation has to preserve.
        """
        by_rank = sorted(Timeframe, key=lambda tf: StreamKey("EURUSD", tf).rank)
        assert by_rank == sorted(Timeframe, key=lambda tf: tf.duration)

    def test_a_daily_close_off_the_hour_lands_between_two_h1_events(self) -> None:
        """The policy is a consequence of the close instants, not a branch.

        Move the day anchor half an hour and the co-timestamped case simply stops
        arising — the daily event falls between two hourly ones and the 21:00 H1
        sees yesterday's daily. Nothing in the merge had to be told.
        """
        origin = DayOrigin(tz="America/New_York", at=datetime(2024, 1, 1, 17, 30).time())
        opening = datetime(2024, 7, 1, 21, 30, tzinfo=UTC)
        closes_at = day_close_ts(opening, origin)
        assert closes_at == datetime(2024, 7, 2, 21, 30, tzinfo=UTC)
        assert closes_at.minute == 30


class TestForwardOnly:
    """The handler feeds strictly forward and the store refuses everything else."""

    def test_instants_are_strictly_increasing_across_streams(self) -> None:
        eurusd = StreamKey("EURUSD", Timeframe.H1)
        nas = StreamKey("NAS100", Timeframe.M15)
        handler = DataHandler(
            {
                eurusd: series(bars([1.1, 1.2, 1.3], timeframe=Timeframe.H1)),
                nas: series(
                    bars(
                        [100.0] * 12,
                        symbol="NAS100",
                        timeframe=Timeframe.M15,
                        spread=1.0,
                    )
                ),
            },
            FX_DAY_ORIGIN,
        )
        stamps = [instant.ts for instant in handler.instants()]
        assert stamps == sorted(stamps)
        assert len(stamps) == len(set(stamps))

    def test_every_bar_of_every_stream_appears_exactly_once(self) -> None:
        eurusd = StreamKey("EURUSD", Timeframe.H1)
        nas = StreamKey("NAS100", Timeframe.M15)
        handler = DataHandler(
            {
                eurusd: series(bars([1.1, 1.2, 1.3], timeframe=Timeframe.H1)),
                nas: series(
                    bars([100.0] * 12, symbol="NAS100", timeframe=Timeframe.M15, spread=1.0)
                ),
            },
            FX_DAY_ORIGIN,
        )
        seen = [
            (event.key, event.index) for instant in handler.instants() for event in instant.bars
        ]
        assert len(seen) == len(set(seen)) == 15

    def test_reading_past_the_cursor_raises_rather_than_returning_something(self) -> None:
        key = StreamKey("EURUSD", Timeframe.H1)
        store = BarStore({key: series(bars([1.1, 1.2, 1.3]))})
        store.publish(next(iter(DataHandler(store.streams, FX_DAY_ORIGIN).instants())).bars[0])

        assert store.cursor(key) == 0
        with pytest.raises(LookaheadError, match="has not closed yet"):
            store.context(key, 1)

    def test_publishing_out_of_order_is_refused(self) -> None:
        key = StreamKey("EURUSD", Timeframe.H1)
        streams = {key: series(bars([1.1, 1.2, 1.3]))}
        store = BarStore(streams)
        events = [
            event
            for instant in DataHandler(streams, FX_DAY_ORIGIN).instants()
            for event in instant.bars
        ]
        with pytest.raises(ValueError, match="next one is 0"):
            store.publish(events[1])
