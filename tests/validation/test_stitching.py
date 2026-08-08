"""Splicing OOS windows by return: only the trade window counts, and overlap is a hard error."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from trading_system.backtest.portfolio import EquityPoint
from trading_system.validation.splitting import Fold, FoldWindow
from trading_system.validation.stitching import StitchedCurve, stitch

START = datetime(2024, 1, 1, tzinfo=UTC)


def _point(ts: datetime, equity: float) -> EquityPoint:
    return EquityPoint(
        ts=ts,
        day=ts.date(),
        balance=Decimal(str(equity)),
        equity=Decimal(str(equity)),
        realized=Decimal(0),
        unrealized=Decimal(0),
        commission_paid=Decimal(0),
        swap_paid=Decimal(0),
        open_positions=0,
    )


def _fold(index: int, trade_start: datetime, trade_end: datetime) -> Fold:
    is_start = trade_start - timedelta(days=40)
    is_end = trade_start - timedelta(days=5)
    return Fold(
        index=index,
        is_window=FoldWindow(
            data_start=is_start - timedelta(days=10), trade_start=is_start, trade_end=is_end
        ),
        oos_window=FoldWindow(
            data_start=trade_start - timedelta(days=5), trade_start=trade_start, trade_end=trade_end
        ),
        embargo=timedelta(days=5),
    )


def _hourly_curve(window: FoldWindow, equities: list[float]) -> list[EquityPoint]:
    """One point per hour from ``window.trade_start`` to ``trade_end`` inclusive, given values."""
    step = (window.trade_end - window.trade_start) / (len(equities) - 1)
    return [
        _point(window.trade_start + step * index, value) for index, value in enumerate(equities)
    ]


class TestContract:
    def test_length_mismatch_is_refused(self) -> None:
        fold = _fold(0, START, START + timedelta(days=10))
        with pytest.raises(ValueError, match="same length"):
            stitch([fold, fold], [[]])

    def test_non_positive_starting_equity_is_refused(self) -> None:
        fold = _fold(0, START, START + timedelta(days=10))
        with pytest.raises(ValueError, match="positive"):
            stitch([fold], [[]], starting_equity=Decimal(0))

    def test_overlapping_oos_windows_are_refused(self) -> None:
        fold0 = _fold(0, START, START + timedelta(days=20))
        fold1 = _fold(1, START + timedelta(days=10), START + timedelta(days=30))  # overlaps fold0
        curve = _hourly_curve(fold0.oos_window, [100.0, 110.0])
        with pytest.raises(ValueError, match="overlap"):
            stitch([fold0, fold1], [curve, curve])

    def test_back_to_back_windows_are_accepted(self) -> None:
        fold0 = _fold(0, START, START + timedelta(days=20))
        fold1 = _fold(1, START + timedelta(days=20), START + timedelta(days=40))
        curve0 = _hourly_curve(fold0.oos_window, [100.0, 110.0])
        curve1 = _hourly_curve(fold1.oos_window, [50.0, 55.0])
        stitched = stitch([fold0, fold1], [curve0, curve1])
        assert isinstance(stitched, StitchedCurve)


class TestCompounding:
    def test_stitches_by_return_not_by_money(self) -> None:
        """Two folds at wildly different money scales still compound cleanly by return."""
        fold0 = _fold(0, START, START + timedelta(days=20))
        fold1 = _fold(1, START + timedelta(days=20), START + timedelta(days=40))
        # fold0: +10% over its window, on a curve starting at 100.
        curve0 = _hourly_curve(fold0.oos_window, [100.0, 110.0])
        # fold1: +10% again, but the underlying run started from an entirely
        # different base (50) — its own starting balance, irrelevant to the splice.
        curve1 = _hourly_curve(fold1.oos_window, [50.0, 55.0])

        stitched = stitch([fold0, fold1], [curve0, curve1], starting_equity=Decimal(100_000))

        assert stitched.points[-1].equity == Decimal("121000.000")

    def test_order_of_folds_passed_in_does_not_matter_only_index_does(self) -> None:
        fold0 = _fold(0, START, START + timedelta(days=20))
        fold1 = _fold(1, START + timedelta(days=20), START + timedelta(days=40))
        curve0 = _hourly_curve(fold0.oos_window, [100.0, 110.0])
        curve1 = _hourly_curve(fold1.oos_window, [50.0, 55.0])

        forward = stitch([fold0, fold1], [curve0, curve1])
        backward = stitch([fold1, fold0], [curve1, curve0])

        assert forward.points == backward.points


class TestWindowClipping:
    def test_rows_outside_the_trade_window_are_excluded(self) -> None:
        fold = _fold(0, START, START + timedelta(days=10))
        window = fold.oos_window
        curve = [
            _point(window.data_start, 999.0),  # warmup prefix — must not count
            _point(window.trade_start, 100.0),
            _point(window.trade_start + timedelta(days=5), 105.0),
            _point(window.trade_end, 110.0),
            _point(window.trade_end + timedelta(days=5), 5.0),  # drain tail — must not count
        ]
        stitched = stitch([fold], [curve], starting_equity=Decimal(1000))

        assert len(stitched.points) == 3
        assert stitched.points[0].equity == Decimal(1000)
        assert stitched.points[-1].equity == Decimal("1100.0")

    def test_a_fold_with_nothing_inside_its_own_window_contributes_flatly(self) -> None:
        fold0 = _fold(0, START, START + timedelta(days=20))
        fold1 = _fold(1, START + timedelta(days=20), START + timedelta(days=40))
        curve0 = _hourly_curve(fold0.oos_window, [100.0, 121.0])
        # fold1's stored curve happens to carry no rows inside its own window at all.
        curve1: list[EquityPoint] = []

        stitched = stitch([fold0, fold1], [curve0, curve1], starting_equity=Decimal(1000))

        assert stitched.points[-1].equity == Decimal("1210.0")


class TestDayLabels:
    def test_day_labels_come_from_the_source_points(self) -> None:
        fold = _fold(0, START, START + timedelta(days=10))
        window = fold.oos_window
        curve = [_point(window.trade_start, 100.0), _point(window.trade_end, 110.0)]

        stitched = stitch([fold], [curve])

        assert [point.day for point in stitched.points] == [
            window.trade_start.date(),
            window.trade_end.date(),
        ]


class TestNotARun:
    def test_a_stitched_curve_is_not_a_backtest_result(self) -> None:
        """write_run needs curve/trades/rejections/... on the object; a StitchedCurve has none."""
        fold = _fold(0, START, START + timedelta(days=10))
        window = fold.oos_window
        curve = [_point(window.trade_start, 100.0), _point(window.trade_end, 110.0)]
        stitched = stitch([fold], [curve])

        assert not hasattr(stitched, "trades")
        assert not hasattr(stitched, "rejections")
