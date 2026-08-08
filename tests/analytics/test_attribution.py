"""Attribution: slices, quality against realised R, and excursions.

The excursion tests are built on synthetic bars whose highs and lows are chosen
so the correct MAE and MFE are known by construction — a price path that goes
exactly 3R in favour and exactly 1.5R against has one right answer, and any
sign error, denominator error or off-by-one in the bar window changes it.
"""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import polars as pl
import pytest

from trading_system.analytics.attribution import (
    OUTSIDE_SESSIONS,
    QualityMode,
    attribute_all,
    by_direction,
    by_quality,
    by_session,
    excursions,
    quality_vs_r,
)
from trading_system.backtest.clock import StreamKey
from trading_system.backtest.portfolio import TradeRecord
from trading_system.core.types import Side, Timeframe
from trading_system.data.models import OHLCVFrame

from .conftest import trade

START = datetime(2024, 1, 2, 0, 0, tzinfo=UTC)


def at(hours: int) -> datetime:
    """An instant ``hours`` after the fixture start."""
    return START + timedelta(hours=hours)


def make_trade(
    *,
    position_id: str = "t",
    side: Side = Side.BUY,
    entry: float = 1.1000,
    risk: float = 0.0010,
    quality: float = 0.5,
    realized_r: float = 0.0,
    net: float | None = None,
    opened: int = 0,
    closed: int = 4,
    symbol: str = "EURUSD",
    strategy_id: str = "s",
) -> TradeRecord:
    """A closed trade positioned on the fixture clock."""
    return trade(
        position_id=position_id,
        symbol=symbol,
        strategy_id=strategy_id,
        side=side,
        opened_at=at(opened),
        closed_at=at(closed),
        entry_price=entry,
        quality=quality,
        initial_risk_distance=risk,
        net=realized_r if net is None else net,
        realized_r=realized_r,
    )


def frame_from(bars: list[tuple[datetime, float, float, float, float]]) -> OHLCVFrame:
    """An :class:`OHLCVFrame` from ``(ts, open, high, low, close)`` rows."""
    return OHLCVFrame(
        pl.DataFrame(
            {
                "timestamp": [row[0] for row in bars],
                "open": [row[1] for row in bars],
                "high": [row[2] for row in bars],
                "low": [row[3] for row in bars],
                "close": [row[4] for row in bars],
                "volume": [100.0] * len(bars),
            },
            schema={
                "timestamp": pl.Datetime("us", "UTC"),
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Float64,
            },
        ),
        "EURUSD",
        Timeframe.H1,
    )


class TestSlicesCarryTheirOwnSampleSize:
    """A slice row without its count is a number nobody can weigh."""

    def test_every_row_reports_the_trades_behind_it(self) -> None:
        trades = [
            make_trade(position_id="a", side=Side.BUY, realized_r=1.0),
            make_trade(position_id="b", side=Side.BUY, realized_r=-1.0),
            make_trade(position_id="c", side=Side.SELL, realized_r=2.0),
        ]
        cut = by_direction(trades)
        rows = {row.label: row for row in cut.rows}
        assert rows["BUY"].count == 2
        assert rows["SELL"].count == 1
        assert rows["BUY"].expectancy_r == pytest.approx(0.0)
        assert rows["SELL"].expectancy_r == pytest.approx(2.0)

    def test_a_thin_row_is_flagged_and_still_present(self) -> None:
        trades = [make_trade(position_id=str(i), realized_r=0.5) for i in range(3)]
        (row,) = by_direction(trades).rows
        assert row.count == 3
        assert row.thin is True

    def test_a_partitioning_slice_sums_back_to_the_total(self) -> None:
        trades = [
            make_trade(position_id="a", side=Side.BUY, realized_r=1.0, net=100.0),
            make_trade(position_id="b", side=Side.SELL, realized_r=-1.0, net=-40.0),
        ]
        cut = by_direction(trades)
        assert cut.partitions is True
        assert sum(row.net for row in cut.rows) == Decimal("60.0")
        assert sum(row.count for row in cut.rows) == cut.n_trades


class TestOverlappingSessionsAreDeclaredRatherThanResolved:
    """Sessions overlap; the rows say so instead of inventing a precedence rule."""

    def test_the_session_slice_declares_that_it_does_not_partition(self) -> None:
        cut = by_session([make_trade(realized_r=1.0)])
        assert cut.partitions is False

    def test_a_trade_in_two_sessions_appears_in_both_rows(self) -> None:
        # 14:00 UTC is inside London and New York at once, so the overlap
        # session is active too — one trade, three rows.
        overlapping = make_trade(opened=14, closed=18, realized_r=1.0)
        cut = by_session([overlapping])
        assert cut.n_trades == 1
        assert sum(row.count for row in cut.rows) > cut.n_trades

    def test_a_trade_outside_every_session_gets_a_named_row(self) -> None:
        # Every weekday hour falls inside some session, so the empty case is
        # the weekend: Friday 22:00 UTC, after New York has closed and before
        # Sydney reopens on Sunday evening.
        cut = by_session([make_trade(opened=118, closed=119, realized_r=1.0)])
        labels = [row.label for row in cut.rows]
        assert OUTSIDE_SESSIONS in labels


class TestQualityBranchesOnLevelsNotOnTradeCount:
    """A decile needs ten distinguishable values; no number of trades creates them."""

    def test_constant_quality_returns_no_correlation_and_says_why(self) -> None:
        trades = [
            make_trade(position_id=str(i), quality=0.55, realized_r=float(i % 3) - 1.0)
            for i in range(40)
        ]
        analysis = quality_vs_r(trades)
        assert analysis.mode == QualityMode.CONSTANT
        assert analysis.n_levels == 1
        assert analysis.correlation is None
        assert "constant at 0.55" in analysis.note
        assert analysis.gap is None

    def test_constant_quality_neither_raises_nor_invents_a_number(self) -> None:
        # The failure this guards is a correlation of 0.0 being returned for a
        # sample that cannot support one — indistinguishable, downstream, from
        # a measured absence of relationship.
        analysis = quality_vs_r([make_trade(quality=0.7, realized_r=r) for r in (1.0, -1.0, 2.0)])
        assert analysis.correlation is None
        assert analysis.groups[0].count == 3

    def test_no_trades_is_its_own_mode(self) -> None:
        analysis = quality_vs_r([])
        assert analysis.mode == QualityMode.NO_TRADES
        assert analysis.correlation is None
        assert analysis.groups == ()

    def test_two_levels_group_by_value_rather_than_into_ten_buckets(self) -> None:
        trades = [
            make_trade(position_id=f"lo{i}", quality=0.55, realized_r=-0.5) for i in range(10)
        ] + [make_trade(position_id=f"hi{i}", quality=0.70, realized_r=1.5) for i in range(10)]
        analysis = quality_vs_r(trades)
        assert analysis.mode == QualityMode.LEVELS
        assert analysis.n_levels == 2
        assert len(analysis.groups) == 2
        assert [group.label for group in analysis.groups] == ["0.55", "0.7"]

    def test_the_gap_reports_the_direction_the_score_ranked_in(self) -> None:
        trades = [
            make_trade(position_id=f"lo{i}", quality=0.55, realized_r=-0.5 + 0.1 * i)
            for i in range(10)
        ] + [
            make_trade(position_id=f"hi{i}", quality=0.70, realized_r=1.5 + 0.1 * i)
            for i in range(10)
        ]
        gap = quality_vs_r(trades).gap
        assert gap is not None
        assert gap.difference == pytest.approx(2.0)
        assert gap.n_low == 10
        assert gap.n_high == 10
        assert gap.ci_low is not None and gap.ci_high is not None
        assert gap.ci_low < gap.difference < gap.ci_high

    def test_a_single_trade_level_leaves_the_interval_undefined_not_zero(self) -> None:
        trades = [make_trade(position_id="lo", quality=0.5, realized_r=1.0)] + [
            make_trade(position_id=f"hi{i}", quality=0.9, realized_r=float(i)) for i in range(5)
        ]
        gap = quality_vs_r(trades).gap
        assert gap is not None
        assert gap.ci_low is None
        assert gap.p_value is None

    def test_ten_distinct_values_switch_to_deciles(self) -> None:
        trades = [
            make_trade(position_id=str(i), quality=round(0.05 + i * 0.05, 4), realized_r=float(i))
            for i in range(18)
        ]
        analysis = quality_vs_r(trades)
        assert analysis.mode == QualityMode.DECILES
        assert len(analysis.groups) == 10
        assert analysis.correlation is not None
        assert analysis.correlation > 0.9

    def test_the_quality_slice_orders_by_value_not_by_pnl(self) -> None:
        trades = [
            make_trade(position_id="a", quality=0.7, realized_r=5.0, net=500.0),
            make_trade(position_id="b", quality=0.55, realized_r=-1.0, net=-100.0),
        ]
        assert [row.label for row in by_quality(trades).rows] == ["0.55", "0.7"]


class TestExcursionsOnAKnownPath:
    """MAE/MFE against a path whose extremes were chosen, not observed."""

    def test_a_long_reports_the_exact_adverse_and_favourable_run(self) -> None:
        # Entry 1.1000, risk 0.0010 (10 pips = 1R). The path dips to 1.0985
        # (15 pips against = 1.5R) and peaks at 1.1030 (30 pips for = 3.0R).
        bars = [
            (at(0), 1.1000, 1.1005, 1.0995, 1.1000),  # entry bar, must be excluded
            (at(1), 1.1000, 1.1010, 1.0985, 1.0990),
            (at(2), 1.0990, 1.1030, 1.0990, 1.1025),
            (at(3), 1.1025, 1.1028, 1.1015, 1.1020),
            (at(4), 1.1020, 1.1022, 1.1018, 1.1020),
        ]
        stats = excursions(
            [make_trade(side=Side.BUY, entry=1.1000, risk=0.0010, opened=0, closed=4)],
            {StreamKey("EURUSD", Timeframe.H1): frame_from(bars)},
        )
        (item,) = stats.excursions
        assert item.mae_r == pytest.approx(1.5)
        assert item.mfe_r == pytest.approx(3.0)
        assert item.bars == 4

    def test_a_short_mirrors_it_exactly(self) -> None:
        # Same path, opposite side: a rise is now adverse and a fall favourable.
        bars = [
            (at(0), 1.1000, 1.1005, 1.0995, 1.1000),
            (at(1), 1.1000, 1.1010, 1.0985, 1.0990),
            (at(2), 1.0990, 1.1030, 1.0990, 1.1025),
            (at(3), 1.1025, 1.1028, 1.1015, 1.1020),
            (at(4), 1.1020, 1.1022, 1.1018, 1.1020),
        ]
        stats = excursions(
            [make_trade(side=Side.SELL, entry=1.1000, risk=0.0010, opened=0, closed=4)],
            {StreamKey("EURUSD", Timeframe.H1): frame_from(bars)},
        )
        (item,) = stats.excursions
        assert item.mae_r == pytest.approx(3.0)
        assert item.mfe_r == pytest.approx(1.5)

    def test_the_entry_bar_is_excluded_because_its_range_predates_the_fill(self) -> None:
        # The entry bar alone swings 5R against; every later bar is flat at the
        # entry price. Counting the entry bar would report 5R of adverse
        # excursion the position was never exposed to.
        bars = [
            (at(0), 1.1000, 1.1050, 1.0950, 1.1000),
            (at(1), 1.1000, 1.1000, 1.1000, 1.1000),
            (at(2), 1.1000, 1.1000, 1.1000, 1.1000),
        ]
        stats = excursions(
            [make_trade(side=Side.BUY, entry=1.1000, risk=0.0010, opened=0, closed=2)],
            {StreamKey("EURUSD", Timeframe.H1): frame_from(bars)},
        )
        (item,) = stats.excursions
        assert item.mae_r == pytest.approx(0.0)
        assert item.mfe_r == pytest.approx(0.0)
        assert item.bars == 2

    def test_the_closing_bar_is_included(self) -> None:
        bars = [
            (at(0), 1.1000, 1.1000, 1.1000, 1.1000),
            (at(1), 1.1000, 1.1000, 1.1000, 1.1000),
            (at(2), 1.1000, 1.1020, 1.1000, 1.1020),
        ]
        stats = excursions(
            [make_trade(side=Side.BUY, entry=1.1000, risk=0.0010, opened=0, closed=2)],
            {StreamKey("EURUSD", Timeframe.H1): frame_from(bars)},
        )
        (item,) = stats.excursions
        assert item.mfe_r == pytest.approx(2.0)

    def test_both_excursions_are_non_negative_even_when_the_path_never_turns(self) -> None:
        # A long that only ever goes up has zero adverse excursion, not a
        # negative one — MAE is a magnitude.
        bars = [
            (at(0), 1.1000, 1.1000, 1.1000, 1.1000),
            (at(1), 1.1010, 1.1020, 1.1010, 1.1020),
        ]
        stats = excursions(
            [make_trade(side=Side.BUY, entry=1.1000, risk=0.0010, opened=0, closed=1)],
            {StreamKey("EURUSD", Timeframe.H1): frame_from(bars)},
        )
        (item,) = stats.excursions
        assert item.mae_r == 0.0
        assert item.mfe_r == pytest.approx(2.0)

    def test_a_trade_whose_stream_is_missing_is_counted_not_dropped(self) -> None:
        stats = excursions([make_trade(symbol="GBPUSD")], {})
        assert stats.count == 0
        assert stats.skipped == 1
        assert stats.mean_mae_r is None

    def test_a_loser_that_ran_far_in_favour_is_counted(self) -> None:
        bars = [
            (at(0), 1.1000, 1.1000, 1.1000, 1.1000),
            (at(1), 1.1000, 1.1030, 1.1000, 1.1025),
            (at(2), 1.1025, 1.1025, 1.0990, 1.0990),
        ]
        stats = excursions(
            [
                make_trade(
                    side=Side.BUY, entry=1.1000, risk=0.0010, opened=0, closed=2, realized_r=-1.0
                )
            ],
            {StreamKey("EURUSD", Timeframe.H1): frame_from(bars)},
        )
        assert stats.losers_with_favourable_run == 1
        assert stats.mean_mfe_r == pytest.approx(3.0)


class TestEverySliceRunsOnTheSameTrades:
    """``attribute_all`` must not need the report to know which cuts exist."""

    def test_all_declared_slices_are_produced(self) -> None:
        trades = [make_trade(position_id=str(i), realized_r=float(i)) for i in range(5)]
        cuts = attribute_all(trades)
        assert set(cuts) == {
            "strategy",
            "symbol",
            "direction",
            "session",
            "weekday",
            "hour",
            "quality",
        }
        assert all(cut.n_trades == 5 for cut in cuts.values())

    def test_no_slice_raises_on_an_empty_run(self) -> None:
        cuts = attribute_all([])
        assert all(cut.rows == () for cut in cuts.values())
