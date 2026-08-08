"""Folds cannot be built out of order, and the two walk-forward modes cut what they claim to."""

import inspect
from datetime import UTC, datetime, timedelta

import pytest

from trading_system.data.resample import FX_DAY_ORIGIN
from trading_system.validation import splitting
from trading_system.validation.splitting import (
    Fold,
    FoldWindow,
    PurgedFold,
    PurgedKFold,
    WalkForwardMode,
    WalkForwardSplitter,
    _snap_forward,
)

START = datetime(2015, 1, 1, tzinfo=UTC)
END = datetime(2024, 1, 1, tzinfo=UTC)


def _splitter(**overrides: object) -> WalkForwardSplitter:
    params: dict[str, object] = {
        "mode": WalkForwardMode.ROLLING,
        "is_span": timedelta(days=730),
        "oos_span": timedelta(days=180),
        "step": timedelta(days=180),
        "embargo": timedelta(days=5),
        "warmup": timedelta(days=60),
    }
    params.update(overrides)
    return WalkForwardSplitter(**params)  # type: ignore[arg-type]


class TestFoldWindowConstruction:
    def test_naive_bounds_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="tz-aware"):
            FoldWindow(
                data_start=datetime(2020, 1, 1),
                trade_start=START,
                trade_end=START + timedelta(days=10),
            )

    def test_reversed_order_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="data_start < trade_start < trade_end"):
            FoldWindow(data_start=START, trade_start=START + timedelta(days=10), trade_end=START)

    def test_equal_trade_start_and_end_is_rejected(self) -> None:
        same = START + timedelta(days=5)
        with pytest.raises(ValueError, match="data_start < trade_start < trade_end"):
            FoldWindow(data_start=START, trade_start=same, trade_end=same)


class TestFoldConstruction:
    def _window(self, offset_days: int, span_days: int = 30) -> FoldWindow:
        start = START + timedelta(days=offset_days)
        return FoldWindow(
            data_start=start - timedelta(days=10),
            trade_start=start,
            trade_end=start + timedelta(days=span_days),
        )

    def test_oos_starting_before_is_ends_is_rejected(self) -> None:
        is_window = self._window(0)
        oos_window = self._window(10)  # overlaps is_window, which ends at day 30
        with pytest.raises(ValueError, match="OOS window must start no earlier"):
            Fold(index=0, is_window=is_window, oos_window=oos_window, embargo=timedelta(0))

    def test_oos_starting_inside_the_embargo_gap_is_rejected(self) -> None:
        is_window = self._window(0)  # trade_end at day 30
        oos_window = self._window(32)  # only 2 days after is_window ends
        with pytest.raises(ValueError, match="OOS window must start no earlier"):
            Fold(index=0, is_window=is_window, oos_window=oos_window, embargo=timedelta(days=5))

    def test_oos_starting_exactly_at_the_embargo_boundary_is_accepted(self) -> None:
        is_window = self._window(0)  # trade_end at day 30
        oos_window = self._window(35)  # exactly is_window.trade_end + 5d embargo
        fold = Fold(index=0, is_window=is_window, oos_window=oos_window, embargo=timedelta(days=5))
        assert fold.oos_window.trade_start == fold.is_window.trade_end + fold.embargo

    def test_negative_embargo_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            Fold(
                index=0,
                is_window=self._window(0),
                oos_window=self._window(40),
                embargo=timedelta(days=-1),
            )

    def test_negative_index_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            Fold(
                index=-1,
                is_window=self._window(0),
                oos_window=self._window(40),
                embargo=timedelta(0),
            )

    def test_fold_is_frozen(self) -> None:
        fold = Fold(
            index=0, is_window=self._window(0), oos_window=self._window(40), embargo=timedelta(0)
        )
        with pytest.raises(AttributeError):
            fold.index = 1  # type: ignore[misc]


class TestNoRandomSplitter:
    """The API must make a random or shuffled fold sequence impossible to build."""

    def test_no_public_name_mentions_shuffle(self) -> None:
        """No class, function or field anywhere in the module is spelled ``shuffle``.

        Checked against the module's actual names rather than its prose: the
        module docstring is free to explain *why* no shuffling splitter
        exists (and does), which would defeat a check against raw source text.
        """
        names: list[str] = []
        for module_name, obj in vars(splitting).items():
            names.append(module_name)
            if inspect.isclass(obj):
                names.extend(vars(obj).keys())
                names.extend(getattr(obj, "__dataclass_fields__", {}).keys())
        assert not any("shuffle" in name.lower() for name in names)

    def test_the_splitting_module_defines_exactly_one_splitter(self) -> None:
        splitters = [
            name
            for name, obj in vars(splitting).items()
            if inspect.isclass(obj) and name.endswith("Splitter")
        ]
        assert splitters == ["WalkForwardSplitter"]


class TestSnapForward:
    def test_idempotent(self) -> None:
        once = _snap_forward(START, FX_DAY_ORIGIN)
        assert _snap_forward(once, FX_DAY_ORIGIN) == once

    def test_monotone_on_a_sweep(self) -> None:
        previous = _snap_forward(START, FX_DAY_ORIGIN)
        for hours in range(1, 24 * 40, 3):
            current = _snap_forward(START + timedelta(hours=hours), FX_DAY_ORIGIN)
            assert current >= previous
            previous = current

    def test_snaps_forward_not_backward(self) -> None:
        mid_day = START.replace(hour=12, minute=0, second=0, microsecond=0)
        assert _snap_forward(mid_day, FX_DAY_ORIGIN) >= mid_day


class TestSplitterValidation:
    @pytest.mark.parametrize("field", ["is_span", "oos_span", "step"])
    def test_non_positive_span_is_rejected(self, field: str) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            _splitter(**{field: timedelta(0)})

    @pytest.mark.parametrize("field", ["embargo", "warmup"])
    def test_negative_gap_is_rejected(self, field: str) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            _splitter(**{field: timedelta(days=-1)})

    def test_zero_embargo_and_warmup_are_accepted(self) -> None:
        _splitter(embargo=timedelta(0), warmup=timedelta(0))


class TestInsufficientCoverage:
    def test_naive_coverage_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="tz-aware"):
            _splitter().split((datetime(2020, 1, 1), END), day_origin=FX_DAY_ORIGIN)

    def test_inverted_coverage_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="before its end"):
            _splitter().split((END, START), day_origin=FX_DAY_ORIGIN)

    def test_zero_folds_reports_the_shortfall(self) -> None:
        short_end = START + timedelta(days=10)
        with pytest.raises(ValueError, match="short by") as excinfo:
            _splitter().split((START, short_end), day_origin=FX_DAY_ORIGIN)
        assert "975 days" in str(excinfo.value)  # warmup 60 + is 730 + embargo 5 + oos 180

    def test_embargo_eating_the_window_still_reports_zero_folds(self) -> None:
        splitter = _splitter(
            is_span=timedelta(days=30),
            oos_span=timedelta(days=10),
            step=timedelta(days=10),
            embargo=timedelta(days=400),
            warmup=timedelta(days=5),
        )
        with pytest.raises(ValueError, match="short by"):
            splitter.split((START, START + timedelta(days=60)), day_origin=FX_DAY_ORIGIN)


class TestRollingSplit:
    def test_every_is_window_has_the_configured_span(self) -> None:
        splitter = _splitter()
        for fold in splitter.split((START, END), day_origin=FX_DAY_ORIGIN):
            assert fold.is_window.trade_end - fold.is_window.trade_start == splitter.is_span

    def test_every_oos_window_has_the_configured_span(self) -> None:
        splitter = _splitter()
        for fold in splitter.split((START, END), day_origin=FX_DAY_ORIGIN):
            assert fold.oos_window.trade_end - fold.oos_window.trade_start == splitter.oos_span

    def test_is_start_advances_by_step_each_fold(self) -> None:
        splitter = _splitter()
        folds = splitter.split((START, END), day_origin=FX_DAY_ORIGIN)
        for previous, current in zip(folds, folds[1:], strict=False):
            assert current.is_window.trade_start - previous.is_window.trade_start == splitter.step

    def test_embargo_is_exact(self) -> None:
        splitter = _splitter()
        for fold in splitter.split((START, END), day_origin=FX_DAY_ORIGIN):
            assert fold.oos_window.trade_start - fold.is_window.trade_end == splitter.embargo

    def test_indices_are_contiguous_from_zero(self) -> None:
        folds = _splitter().split((START, END), day_origin=FX_DAY_ORIGIN)
        assert [fold.index for fold in folds] == list(range(len(folds)))

    def test_consecutive_oos_windows_are_exactly_back_to_back_when_step_equals_oos_span(
        self,
    ) -> None:
        splitter = _splitter(
            is_span=timedelta(days=200), oos_span=timedelta(days=60), step=timedelta(days=60)
        )
        folds = splitter.split((START, END), day_origin=FX_DAY_ORIGIN)
        assert len(folds) >= 3
        for previous, current in zip(folds, folds[1:], strict=False):
            assert previous.oos_window.trade_end == current.oos_window.trade_start

    def test_no_fold_exceeds_coverage(self) -> None:
        folds = _splitter().split((START, END), day_origin=FX_DAY_ORIGIN)
        assert folds
        for fold in folds:
            assert fold.oos_window.trade_end <= END

    def test_one_more_fold_would_not_fit(self) -> None:
        """The splitter stops as soon as the *next* fold would not fit — not later."""
        splitter = _splitter()
        folds = splitter.split((START, END), day_origin=FX_DAY_ORIGIN)
        last = folds[-1]
        next_oos_end = last.oos_window.trade_end + splitter.step
        assert next_oos_end > END


class TestAnchoredSplit:
    def test_is_start_never_moves(self) -> None:
        splitter = _splitter(mode=WalkForwardMode.ANCHORED)
        folds = splitter.split((START, END), day_origin=FX_DAY_ORIGIN)
        assert len({fold.is_window.trade_start for fold in folds}) == 1

    def test_is_window_grows_by_step_each_fold(self) -> None:
        splitter = _splitter(mode=WalkForwardMode.ANCHORED)
        folds = splitter.split((START, END), day_origin=FX_DAY_ORIGIN)
        for previous, current in zip(folds, folds[1:], strict=False):
            grown = current.is_window.trade_end - previous.is_window.trade_end
            assert grown == splitter.step

    def test_first_fold_is_window_equals_is_span(self) -> None:
        splitter = _splitter(mode=WalkForwardMode.ANCHORED)
        folds = splitter.split((START, END), day_origin=FX_DAY_ORIGIN)
        first = folds[0]
        assert first.is_window.trade_end - first.is_window.trade_start == splitter.is_span

    def test_anchored_and_rolling_agree_on_the_first_fold(self) -> None:
        """The two modes only differ from the second fold onward."""
        rolling = _splitter(mode=WalkForwardMode.ROLLING)
        anchored = _splitter(mode=WalkForwardMode.ANCHORED)
        rolling_folds = rolling.split((START, END), day_origin=FX_DAY_ORIGIN)
        anchored_folds = anchored.split((START, END), day_origin=FX_DAY_ORIGIN)
        assert rolling_folds[0] == anchored_folds[0]

    def test_produces_fewer_or_equal_folds_than_rolling(self) -> None:
        """An expanding IS window eats into the same coverage faster."""
        rolling = _splitter(mode=WalkForwardMode.ROLLING)
        anchored = _splitter(mode=WalkForwardMode.ANCHORED)
        rolling_folds = rolling.split((START, END), day_origin=FX_DAY_ORIGIN)
        anchored_folds = anchored.split((START, END), day_origin=FX_DAY_ORIGIN)
        assert len(anchored_folds) <= len(rolling_folds)


class TestDataStartWarmup:
    def test_is_data_start_is_exactly_warmup_before_trade_start(self) -> None:
        splitter = _splitter()
        for fold in splitter.split((START, END), day_origin=FX_DAY_ORIGIN):
            assert fold.is_window.trade_start - fold.is_window.data_start == splitter.warmup

    def test_oos_data_start_is_exactly_warmup_before_trade_start(self) -> None:
        splitter = _splitter()
        for fold in splitter.split((START, END), day_origin=FX_DAY_ORIGIN):
            assert fold.oos_window.trade_start - fold.oos_window.data_start == splitter.warmup


class TestPurgedKFoldKeepsTrainAwayFromTest:
    """DoD: no train/test overlap once purge and embargo are applied.

    Checked on synthetic labels with a *deliberate* overlap: every position is
    given a holding horizon equal to ``label_span``, so a train segment that
    ended one instant too late would contain a trade still running inside the
    test piece — which is exactly the contamination purging exists to remove,
    and exactly what a construction that merely avoided interval overlap would
    miss.
    """

    def _window(self) -> FoldWindow:
        start = datetime(2021, 1, 4, tzinfo=UTC)
        return FoldWindow(
            data_start=start - timedelta(days=30),
            trade_start=start,
            trade_end=start + timedelta(days=360),
        )

    def test_the_test_pieces_tile_the_window_exactly_and_in_order(self) -> None:
        window = self._window()
        folds = PurgedKFold(k=5, embargo=timedelta(days=2), label_span=timedelta(days=5)).split(
            window
        )
        assert len(folds) == 5
        assert folds[0].test[0] == window.trade_start
        assert folds[-1].test[1] == window.trade_end
        for earlier, later in zip(folds, folds[1:], strict=False):
            assert earlier.test[1] == later.test[0], "pieces must meet exactly, not nearly"

    def test_no_train_segment_overlaps_its_own_test_piece(self) -> None:
        folds = PurgedKFold(k=5, embargo=timedelta(days=2), label_span=timedelta(days=5)).split(
            self._window()
        )
        for fold in folds:
            test_start, test_end = fold.test
            for start, end in fold.train:
                assert end <= test_start or start >= test_end

    def test_the_gap_before_a_test_piece_is_at_least_the_label_span(self) -> None:
        label_span = timedelta(days=5)
        folds = PurgedKFold(k=5, embargo=timedelta(days=2), label_span=label_span).split(
            self._window()
        )
        for fold in folds:
            test_start = fold.test[0]
            for _, end in (segment for segment in fold.train if segment[1] <= test_start):
                assert test_start - end >= label_span

    def test_the_gap_after_a_test_piece_is_at_least_the_embargo(self) -> None:
        embargo = timedelta(days=2)
        folds = PurgedKFold(k=5, embargo=embargo, label_span=timedelta(days=5)).split(
            self._window()
        )
        for fold in folds:
            test_end = fold.test[1]
            for start, _ in (segment for segment in fold.train if segment[0] >= test_end):
                assert start - test_end >= embargo

    def test_no_synthetic_trade_opened_in_train_is_still_running_during_test(self) -> None:
        label_span = timedelta(days=5)
        window = self._window()
        folds = PurgedKFold(k=5, embargo=timedelta(days=2), label_span=label_span).split(window)
        # One "trade" opened every 6 hours across the whole window, each held
        # for the full label span — the maximum overlap the configuration admits.
        opens = [
            window.trade_start + timedelta(hours=6) * step
            for step in range(int((window.trade_end - window.trade_start) / timedelta(hours=6)))
        ]
        for fold in folds:
            test_start, test_end = fold.test
            for segment_start, segment_end in fold.train:
                inside = [ts for ts in opens if segment_start <= ts < segment_end]
                for opened in inside:
                    closed = opened + label_span
                    assert not (opened < test_end and test_start < closed), (
                        f"a trade opened {opened!r} and held {label_span} is still running "
                        f"inside the test piece {test_start!r}..{test_end!r}"
                    )

    def test_a_single_piece_is_not_a_cross_validation(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            PurgedKFold(k=1, embargo=timedelta(days=1), label_span=timedelta(days=1))

    def test_a_window_too_short_for_k_reports_a_reason_instead_of_raising(self) -> None:
        window = FoldWindow(
            data_start=datetime(2021, 1, 1, tzinfo=UTC),
            trade_start=datetime(2021, 1, 4, tzinfo=UTC),
            trade_end=datetime(2021, 2, 4, tzinfo=UTC),
        )
        splitter = PurgedKFold(k=5, embargo=timedelta(days=2), label_span=timedelta(days=5))
        reason = splitter.unusable_reason(window, min_test_span=timedelta(days=30))
        assert reason is not None
        assert "below min_test_span" in reason

    def test_a_purge_wider_than_a_piece_reports_a_reason(self) -> None:
        window = self._window()
        splitter = PurgedKFold(k=10, embargo=timedelta(days=20), label_span=timedelta(days=20))
        reason = splitter.unusable_reason(window, min_test_span=timedelta(days=1))
        assert reason is not None
        assert "every train segment would be cut away" in reason

    def test_a_window_that_fits_reports_no_reason(self) -> None:
        splitter = PurgedKFold(k=4, embargo=timedelta(days=2), label_span=timedelta(days=5))
        assert splitter.unusable_reason(self._window(), min_test_span=timedelta(days=30)) is None

    def test_a_train_segment_overlapping_its_test_piece_cannot_be_constructed(self) -> None:
        start = datetime(2021, 1, 4, tzinfo=UTC)
        with pytest.raises(ValueError, match="overlaps the test piece"):
            PurgedFold(
                index=0,
                test=(start, start + timedelta(days=10)),
                train=((start + timedelta(days=5), start + timedelta(days=20)),),
            )
