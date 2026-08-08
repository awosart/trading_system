"""Slicing history into folds, so that a fold cannot be built out of order.

**Windows in tz-aware UTC, anchored to one snapped instant and built from it by
plain addition.** A fold measured in bars is not well-defined once a run
trades more than one stream — "500 bars" of which series? Time is unambiguous,
and the whole sequence is rooted at
:func:`~trading_system.backtest.clock.day_close_ts` on the run's own
``day_origin`` — the same boundary VWAP resets on, the daily loss limit
measures against and the swap rollover fires on — so a walk-forward's first
window does not open on a partial daily-loss budget.

Only that one root is snapped. Every other boundary — ``is_trade_end``,
``oos_trade_start``, ``oos_trade_end``, the next fold's own ``is_trade_start``
— is the root plus a fixed sum of ``is_span``/``step``/``embargo``/``oos_span``,
never a second call to the snap. Snapping is not additive (the day-boundary
search can move a candidate by anywhere up to a day), so snapping each of
those independently let a fold's OOS window drift by a different amount than
the next fold's, and with ``step == oos_span`` — the ordinary contiguous
walk-forward — two OOS windows meant to sit exactly back to back would
overlap by up to a day instead. Plain addition from one anchor does not have
that failure mode: two boundaries defined by the same arithmetic sum are
identical, not merely close. The week is not snapped to at all: no weekly
boundary function exists elsewhere in this system, and inventing one here for
a second-order effect that :mod:`trading_system.validation.walkforward`'s
``boundary_residual`` already surfaces if it matters would be exactly the kind
of forward-looking abstraction CLAUDE.md rules out.

**Three points per window, not two.** ``data_start < trade_start < trade_end``:
bars publish from ``data_start`` so indicators and the ATR baseline warm up,
but nothing may be *evaluated* before ``trade_start``. Publication and
evaluation are already two separate phases in the event loop
(:mod:`trading_system.backtest.engine`); this reuses that split rather than
inventing a third concept.

**A fold cannot be constructed out of order.** :class:`Fold` is frozen and
validates in its own constructor that the IS window ends, with embargo, at or
before the OOS window starts — the same discipline
:class:`~trading_system.backtest.engine.BarStore` uses against lookahead: the
object with the wrong order simply does not exist, rather than existing and
being checked later by something that might forget to check.

**No shuffling splitter, and no ``shuffle`` parameter anywhere.** A random
k-fold splits observations without regard to time and would let a fold be
trained on next month and tested on last; nothing in this module offers one,
and :class:`PurgedKFold` — added in stage 2, where a parameter choice finally
needs validating — cuts *contiguous* pieces in time order, which is a different
thing entirely. A portfolio backtest is path-dependent besides, so k
non-contiguous test pieces could not be stitched into one curve regardless; see
CLAUDE.md's "Решения P15 этап 1" for the fuller argument.

**What "purged" means here, given that nothing is fitted.** The López de Prado
construction removes from *train* any observation whose label overlaps the test
piece in time. This system has no fitting step at all — a parameter set is not
estimated from train data, it is proposed and then *walked* — so what
cross-validation buys is not an unbiased fit but ``k`` sub-samples of in-sample
performance whose results do not contaminate each other. The contamination is
real and is exactly what purging removes: a position opened near the end of one
piece is still open during the next, so its outcome would be counted as
evidence about a window it was not entered in. :class:`PurgedFold` therefore
reports both its ``test`` piece and the ``train`` segments the standard
construction yields, and
:class:`~trading_system.validation.walkforward.OptimizingSelector` consumes only
``test``. The ``train`` segments are not decoration: they are what makes the
purge and embargo invariant checkable, which is the property that matters.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum

from trading_system.data.resample import DayOrigin, trading_day

#: Subtracted before every snap so that an instant already sitting on a day
#: boundary snaps to itself rather than jumping a full day forward. See
#: :func:`_snap_forward`.
_EPSILON = timedelta(microseconds=1)

#: Upper bound the forward search for the next day boundary starts from. A
#: trading day is 23-25 hours long across a DST transition, so a boundary is
#: never more than 25 hours from the start of the day containing ``instant``,
#: and therefore never more than this much later than ``instant`` itself.
_SEARCH_WINDOW = timedelta(hours=26)


def _snap_forward(instant: datetime, day_origin: DayOrigin) -> datetime:
    """The earliest day-close boundary at or after ``instant``.

    Idempotent and monotone: snapping an already-snapped instant returns it
    unchanged, and snapping a later instant never returns an earlier boundary
    than snapping an earlier one did.

    Unlike :func:`~trading_system.backtest.clock.day_close_ts`, ``instant`` is
    not assumed to already sit near the day anchor — a fold boundary is an
    arbitrary point in a coverage range, not a daily bar's own open, so that
    function's "look within an hour of 24h later" search does not apply here.
    The boundary is still *found*, never constructed from a local wall-clock
    time (which the P10 stage 2 decision this module inherits rules out — such
    a time does not exist on the spring-forward day and is ambiguous on the
    autumn one): this bisects on :func:`~trading_system.data.resample.trading_day`
    itself, the one function every day boundary in this system is cut with.

    Args:
        instant: The instant to snap, tz-aware.
        day_origin: Where the trading day starts.

    Returns:
        The boundary. ``instant`` if it already sits on one.
    """
    label_before = trading_day(instant - _EPSILON, day_origin)
    if trading_day(instant, day_origin) > label_before:
        return instant
    low, high = instant, instant + _SEARCH_WINDOW
    while high - low > _EPSILON:
        mid = low + (high - low) // 2
        if trading_day(mid, day_origin) > label_before:
            high = mid
        else:
            low = mid
    return high


@dataclass(frozen=True)
class FoldWindow:
    """One window: a warmup prefix and the span it may actually be evaluated over.

    Attributes:
        data_start: Bars publish from here. Not evaluated on: this is warmup.
        trade_start: Earliest instant a signal may be recognised.
        trade_end: Instant at or after which no new signal may be recognised.
            Positions already open may still be managed past it — see
            :mod:`trading_system.validation.walkforward` on draining.
    """

    data_start: datetime
    trade_start: datetime
    trade_end: datetime

    def __post_init__(self) -> None:
        """Validate ordering and tz-awareness.

        Raises:
            ValueError: If any bound is tz-naive, or the three points are not
                in strict ``data_start < trade_start < trade_end`` order.
        """
        for name, value in (
            ("data_start", self.data_start),
            ("trade_start", self.trade_start),
            ("trade_end", self.trade_end),
        ):
            if value.tzinfo is None:
                raise ValueError(f"FoldWindow.{name} must be tz-aware")
        if not self.data_start < self.trade_start < self.trade_end:
            raise ValueError(
                f"FoldWindow requires data_start < trade_start < trade_end; got "
                f"{self.data_start!r}, {self.trade_start!r}, {self.trade_end!r}"
            )


@dataclass(frozen=True)
class Fold:
    """One walk-forward fold: an in-sample window and the out-of-sample window after it.

    Validated at construction so that a fold with overlapping or reordered
    windows cannot exist, the same discipline
    :class:`~trading_system.backtest.engine.BarStore` applies against
    lookahead: nothing downstream has to re-check an ordering that the type
    already guarantees.

    Attributes:
        index: Position in the walk-forward sequence, oldest first.
        is_window: The in-sample window.
        oos_window: The out-of-sample window, strictly after ``is_window`` with
            at least ``embargo`` between them.
        embargo: Minimum gap enforced between ``is_window.trade_end`` and
            ``oos_window.trade_start``.
    """

    index: int
    is_window: FoldWindow
    oos_window: FoldWindow
    embargo: timedelta

    def __post_init__(self) -> None:
        """Validate the embargo and the ordering between the two windows.

        Raises:
            ValueError: If ``index`` is negative, ``embargo`` is negative, or
                the OOS window starts before the IS window ends plus embargo.
        """
        if self.index < 0:
            raise ValueError(f"Fold.index must be non-negative, got {self.index}")
        if self.embargo < timedelta(0):
            raise ValueError(f"Fold.embargo must be non-negative, got {self.embargo}")
        if self.is_window.trade_end + self.embargo > self.oos_window.trade_start:
            raise ValueError(
                f"fold {self.index}: IS window ends {self.is_window.trade_end!r}, embargo "
                f"{self.embargo}, but OOS window starts {self.oos_window.trade_start!r} — "
                "the OOS window must start no earlier than the IS window's end plus embargo"
            )


class WalkForwardMode(StrEnum):
    """Whether the in-sample window's start is fixed or slides forward each fold."""

    #: The IS window's start never moves; its end grows by ``step`` each fold
    #: (an expanding window).
    ANCHORED = "anchored"

    #: The IS window keeps a fixed span and slides forward by ``step`` each
    #: fold (a sliding window).
    ROLLING = "rolling"


@dataclass(frozen=True)
class WalkForwardSplitter:
    """Cuts a coverage range into folds, in tz-aware UTC, snapped to day boundaries.

    Attributes:
        mode: :data:`WalkForwardMode.ANCHORED` or
            :data:`WalkForwardMode.ROLLING`.
        is_span: Length of the in-sample window. The whole span for ROLLING;
            the *initial* span for ANCHORED, which then grows by ``step`` each
            fold.
        oos_span: Length of the out-of-sample window, constant across folds in
            both modes.
        step: How far the fold sequence advances each iteration — the IS
            window's start for ROLLING, its end for ANCHORED. Must be
            positive: zero would produce the same fold forever.
        embargo: Minimum gap enforced between each fold's IS end and OOS
            start.
        warmup: Prefix reserved before each window's own ``trade_start`` for
            indicator and ATR-baseline warmup. A caller's own number — this
            module does not import the feature pipeline to compute one, the
            same boundary :mod:`trading_system.execution.market_state` draws
            around :mod:`trading_system.execution`.
    """

    mode: WalkForwardMode
    is_span: timedelta
    oos_span: timedelta
    step: timedelta
    embargo: timedelta
    warmup: timedelta

    def __post_init__(self) -> None:
        """Validate that every span is usable.

        Raises:
            ValueError: If ``is_span``, ``oos_span`` or ``step`` is not
                positive, or ``embargo``/``warmup`` is negative. A
                non-positive ``step`` would never advance the fold sequence
                and the splitter would loop forever.
        """
        for name, value in (
            ("is_span", self.is_span),
            ("oos_span", self.oos_span),
            ("step", self.step),
        ):
            if value <= timedelta(0):
                raise ValueError(f"WalkForwardSplitter.{name} must be positive, got {value}")
        for name, value in (("embargo", self.embargo), ("warmup", self.warmup)):
            if value < timedelta(0):
                raise ValueError(f"WalkForwardSplitter.{name} must be non-negative, got {value}")

    def split(self, coverage: tuple[datetime, datetime], *, day_origin: DayOrigin) -> list[Fold]:
        """Cut ``coverage`` into as many folds as fit.

        Args:
            coverage: ``(start, end)`` of the data actually available, tz-aware
                UTC.
            day_origin: The run's own day boundary — the same one its
                ``BacktestConfig`` cuts the trading day on, so a fold's window
                and the run's own daily-loss and swap boundaries agree. Passed
                per call rather than held on the splitter, so nothing here can
                drift from the run it is cutting for.

        Returns:
            Folds, oldest first, each individually valid by construction and
            collectively non-overlapping between one fold's OOS trade window
            and the next fold's IS trade window (embargo guarantees the
            latter; nothing here claims the OOS windows of two folds cannot
            overlap — that is what ``step`` controls, and
            :mod:`trading_system.validation.stitching` is where an overlap
            becomes a hard error rather than a silent double-count).

        Raises:
            ValueError: If ``coverage`` is not tz-aware or is empty or
                inverted, or if not even one fold fits — reporting how much
                more history the configuration needs.
        """
        start, end = coverage
        if start.tzinfo is None or end.tzinfo is None:
            raise ValueError("coverage bounds must be tz-aware")
        if start >= end:
            raise ValueError(f"coverage start {start!r} must be before its end {end!r}")

        # Snapped exactly once, at the root of the whole sequence — every other
        # boundary is plain timedelta arithmetic from here, never a second call
        # to _snap_forward. Snapping is not additive (day_close_ts's search can
        # move a candidate by anywhere up to ~1 day), so snapping is_trade_end,
        # oos_trade_start and oos_trade_end independently would let each one
        # drift by a different amount; the embargo invariant and — the case
        # that actually surfaced this — exact back-to-back OOS windows when
        # step == oos_span both depend on pure addition, not on every boundary
        # separately sitting on a day close. What every boundary still gets is
        # a fixed offset (in whole days, for any config that expresses its
        # spans that way) from one instant that genuinely is a day close.
        root = _snap_forward(start + self.warmup, day_origin)

        folds: list[Fold] = []
        index = 0
        while True:
            if self.mode is WalkForwardMode.ANCHORED:
                is_trade_start = root
                is_trade_end = is_trade_start + self.is_span + index * self.step
            else:
                is_trade_start = root + index * self.step
                is_trade_end = is_trade_start + self.is_span

            oos_trade_start = is_trade_end + self.embargo
            oos_trade_end = oos_trade_start + self.oos_span

            if oos_trade_end > end:
                break

            is_data_start = is_trade_start - self.warmup
            oos_data_start = oos_trade_start - self.warmup
            folds.append(
                Fold(
                    index=index,
                    is_window=FoldWindow(is_data_start, is_trade_start, is_trade_end),
                    oos_window=FoldWindow(oos_data_start, oos_trade_start, oos_trade_end),
                    embargo=self.embargo,
                )
            )
            index += 1

        if not folds:
            minimum = self.warmup + self.is_span + self.embargo + self.oos_span
            available = end - start
            raise ValueError(
                f"coverage {start!r}..{end!r} ({available}) does not fit a single "
                f"{self.mode.value} fold: needs at least {minimum} "
                f"(warmup {self.warmup} + is_span {self.is_span} + embargo {self.embargo} "
                f"+ oos_span {self.oos_span}), short by {minimum - available}"
            )
        return folds


@dataclass(frozen=True)
class PurgedFold:
    """One cross-validation piece of an in-sample window, with its purged train side.

    Attributes:
        index: Position in the k-fold sequence, earliest test piece first.
        test: ``[start, end)`` of the piece a parameter set is scored on.
        train: The remaining segments, with the purge and embargo already cut
            out. Zero, one or two of them: a middle piece leaves a segment on
            each side, the first and last leave one.
    """

    index: int
    test: tuple[datetime, datetime]
    train: tuple[tuple[datetime, datetime], ...]

    def __post_init__(self) -> None:
        """Validate that no train segment touches the test piece.

        Raises:
            ValueError: If ``index`` is negative, the test piece is not a
                positive interval, or any train segment overlaps it. Checked
                here rather than trusted from the constructor that built it,
                the same discipline :class:`Fold` applies to its own ordering.
        """
        if self.index < 0:
            raise ValueError(f"PurgedFold.index must be non-negative, got {self.index}")
        test_start, test_end = self.test
        if test_start >= test_end:
            raise ValueError(f"PurgedFold.test must be a positive interval, got {self.test!r}")
        for start, end in self.train:
            if start >= end:
                raise ValueError(f"PurgedFold train segment {start!r}..{end!r} is not positive")
            if start < test_end and test_start < end:
                raise ValueError(
                    f"PurgedFold {self.index}: train segment {start!r}..{end!r} overlaps the "
                    f"test piece {test_start!r}..{test_end!r}"
                )


@dataclass(frozen=True)
class PurgedKFold:
    """Cuts one in-sample window into k contiguous test pieces, purged and embargoed.

    Not a replacement for the walk-forward: this runs *inside* a single fold's
    in-sample window, to score one parameter set on several sub-samples rather
    than on one. See this module's docstring on what purging means when nothing
    is fitted.

    Attributes:
        k: How many pieces. At least two — one piece is not a
            cross-validation, and returning the whole window under that name
            would report a validation that did not happen.
        embargo: Gap after each test piece that no train segment may enter.
            Covers the correlation that outlives a position: the bar right
            after a test piece is not independent of the bars inside it.
        label_span: How long a position may stay open. The gap cut *before*
            each test piece, because a trade opened that recently is still
            running when the piece begins, and its outcome is therefore partly
            a fact about the test piece rather than about the train segment it
            was entered in. A caller's own number, for the same reason
            :attr:`WalkForwardSplitter.warmup` is: this module does not import
            an exit preset to guess a holding horizon.
    """

    k: int
    embargo: timedelta
    label_span: timedelta

    def __post_init__(self) -> None:
        """Validate the configuration.

        Raises:
            ValueError: If ``k`` is below two, or either gap is negative.
        """
        if self.k < 2:
            raise ValueError(f"PurgedKFold.k must be at least 2, got {self.k}")
        for name, value in (("embargo", self.embargo), ("label_span", self.label_span)):
            if value < timedelta(0):
                raise ValueError(f"PurgedKFold.{name} must be non-negative, got {value}")

    def unusable_reason(self, window: FoldWindow, *, min_test_span: timedelta) -> str | None:
        """Why this window cannot be cross-validated, or ``None`` if it can.

        Returned as a reason rather than raised, and never silently repaired by
        lowering ``k``: a quiet fallback would leave a report unable to say
        which folds were actually cross-validated and which were not. An
        anchored walk-forward's early folds are legitimately shorter than its
        late ones, so a hard error would kill a whole run over a fold that is
        behaving exactly as configured.

        Args:
            window: The in-sample window to cut.
            min_test_span: Shortest test piece worth scoring.

        Returns:
            The reason, or ``None``.
        """
        span = window.trade_end - window.trade_start
        piece = span / self.k
        if piece < min_test_span:
            return (
                f"in-sample span {span} split {self.k} ways gives {piece} per piece, "
                f"below min_test_span {min_test_span}"
            )
        if self.label_span + self.embargo >= piece:
            return (
                f"purge {self.label_span} plus embargo {self.embargo} is not shorter than the "
                f"{piece} test piece; every train segment would be cut away"
            )
        return None

    def split(self, window: FoldWindow) -> list[PurgedFold]:
        """Cut ``window``'s tradeable span into k purged, embargoed pieces.

        The pieces tile ``[trade_start, trade_end)`` exactly and in order —
        boundaries are computed by proportional division from ``trade_start``,
        the same "one anchor plus plain arithmetic" rule
        :meth:`WalkForwardSplitter.split` uses, so adjacent pieces meet exactly
        rather than nearly.

        Args:
            window: The in-sample window.

        Returns:
            The folds, earliest test piece first.

        Raises:
            ValueError: If a fold cannot be built at all — check
                :meth:`unusable_reason` first if that is a possibility the
                caller wants to handle rather than propagate.
        """
        start, end = window.trade_start, window.trade_end
        span = end - start
        edges = [start + (span * index) // self.k for index in range(self.k)] + [end]

        folds: list[PurgedFold] = []
        for index in range(self.k):
            test_start, test_end = edges[index], edges[index + 1]
            segments: list[tuple[datetime, datetime]] = []
            left_end = test_start - self.label_span
            if left_end > start:
                segments.append((start, left_end))
            right_start = test_end + self.embargo
            if right_start < end:
                segments.append((right_start, end))
            folds.append(
                PurgedFold(index=index, test=(test_start, test_end), train=tuple(segments))
            )
        return folds
