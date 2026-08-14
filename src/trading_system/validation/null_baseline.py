"""Each categorical value's own null, because their zeros are not the same number.

Comparing exits on raw scores compares numbers measured from different origins.

**The measurement this module exists for.** Eight exit presets were run on one
entry (``channel-breakout-h4``, EURUSD H4, 593 recognised signals) and then run
again on permuted tape, 24 shuffles each. Every preset's permutation-null
expectancy came out negative, but not equally so: from ``-0.011R``
(``scalp_quick``) to ``-0.052R`` (``conservative_2r``). A raw comparison across
that axis therefore ranks presets partly by where each one's zero happens to
sit. Re-ranking the same eight by *excess over their own null* agrees with the
raw ranking only at Spearman ``+0.71``, and by percentile against it only at
``+0.53`` — the correction moves the answer, it does not decorate it.

**Why the baseline is per categorical value and not per space.** One number for
all eight presets misstates a cell by 79% of the median excess being measured;
one number per preset misstates it by 21%. That is the whole justification for
the cost this module incurs, and it is a measured ratio rather than a principle.

**Why one calibration is reused across the numeric grid.** The same eight
presets were calibrated at five points spanning the numeric axes (corners and
centre of ``ema_period`` x ``multiple``). Baselines do move — ``conservative_2r``
ran from ``-0.030`` to ``-0.069`` — but a paired test over the shared shuffles
puts every preset's drift inside its own noise (largest ``|t| = 1.70``, 23 df),
and reusing one baseline per preset across the whole grid misstates a cell by
21% of the median excess, against 25% for a fitted model of it. So the
calibration is paid once per fold, at the grid's centre point, not once per grid
point. **The condition under which that stops holding is checkable and cheap:**
the baseline tracks the log of the cell's trade count (Pearson ``+0.81``, slope
``+0.024`` per log unit), so a preset whose null trade count varies by more than
about a factor of two would drift by ``0.017R`` — comparable to the effect.
Across the measured grid the widest within-preset variation was ``1.32x``.
:attr:`NullBaseline.trade_count_spread` reports the shuffle-to-shuffle form of
that quantity per run, which is the precondition rather than the grid drift
itself; see its own docstring for exactly what it does and does not cover.

**Why the seed count is per value rather than one number for the space.** The
baseline is only worth subtracting when it is measured more precisely than the
thing it corrects. At 24 shuffles the eight presets' standard errors spanned
``0.003`` to ``0.031`` — a factor of ten, because a preset that holds longer
takes fewer trades and its null mean is correspondingly noisier. A single seed
count either overspends on the tight presets or leaves the loose ones with a
correction noisier than the correction. So each value draws shuffles until its
own standard error falls under :attr:`BaselineRequest.target_fraction` of its
own excess, and reports where it stopped. Measured on the same eight: 7 to 426
shuffles, 956 runs per fold in total.
"""

import json
import math
import statistics
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from trading_system.backtest.clock import StreamKey
from trading_system.core.logging import get_logger
from trading_system.core.types import Timeframe
from trading_system.data.models import OHLCVFrame
from trading_system.data.resample import DayOrigin
from trading_system.validation.nulls.permutation import PermutationConfig, permute_run_streams
from trading_system.validation.optimization import (
    AxisValue,
    ParamSet,
    SearchSpace,
    TrialRecord,
    TrialRunner,
)

logger = get_logger(__name__)

#: Filename the calibration is written under, inside a fold's own directory.
BASELINES_FILE = "null_baselines.json"


@dataclass(frozen=True)
class BaselineRequest:
    """How hard to work at measuring one categorical value's null.

    Attributes:
        target_fraction: The standard error each value's baseline must reach,
            as a fraction of that value's own excess over it. A quarter by
            default: the correction then carries about a sixteenth of the
            variance of what it corrects, which is small enough to be worth
            applying and large enough not to spend a fold's budget chasing.
        min_seeds: Shuffles drawn before the requirement is first evaluated.
            Eight, because a standard error estimated from fewer is itself too
            noisy to project a sample size from.
        max_seeds: Ceiling. A value whose excess is near zero would otherwise
            demand an unbounded number of shuffles to resolve a correction that
            barely exists; it stops here and says so.
        seed_offset: Added to every shuffle index, so that a calibration can be
            re-run on independent shuffles without colliding with the first.
    """

    target_fraction: float = 0.25
    min_seeds: int = 8
    max_seeds: int = 512
    seed_offset: int = 0

    def __post_init__(self) -> None:
        """Validate the request.

        Raises:
            ValueError: If the fraction is not positive, or the seed bounds are
                not ``0 < min <= max``.
        """
        if self.target_fraction <= 0:
            raise ValueError(f"target_fraction must be positive, got {self.target_fraction}")
        if self.min_seeds < 2:
            raise ValueError(f"min_seeds must be at least 2, got {self.min_seeds}")
        if self.max_seeds < self.min_seeds:
            raise ValueError(f"max_seeds {self.max_seeds} is below min_seeds {self.min_seeds}")


#: The default effort, as a module-level singleton so it is not constructed in a
#: signature's default and so two callers that do not configure it share one.
DEFAULT_BASELINE_REQUEST = BaselineRequest()


@dataclass(frozen=True)
class NullBaseline:
    """One categorical value's null, and how well it is known.

    Attributes:
        value: The axis value this belongs to.
        baseline: Mean objective score over the shuffles drawn.
        sem: Standard error of that mean.
        real_score: The same configuration's score on unpermuted bars.
        excess: ``real_score - baseline``, or ``None`` when the real run could
            not be scored at all — in which case the value has no excess to
            correct and is dropped from the selection rather than given a
            guessed one.
        seeds: How many shuffles were drawn.
        stopped_because: ``"target"`` when the precision requirement was met,
            ``"budget"`` when :attr:`BaselineRequest.max_seeds` was reached
            first, ``"unscoreable"`` when the real run produced no score.
        trade_count_spread: Largest over smallest null trade count **across the
            shuffles drawn at this one configuration**. Not the grid drift the
            module docstring reuses one baseline against — measuring that would
            need a calibration per grid point, which is the cost being avoided.
            What it monitors is the *precondition* underneath: the baseline
            tracks the log of the trade count, so a configuration whose own
            trade count swings from shuffle to shuffle has a baseline averaging
            over a heterogeneous mixture, and the reuse argument — which was
            made on presets whose counts moved by 1.32x — was never measured in
            that regime.
        median_trade_count: Median null trade count over the shuffles drawn.
            Published beside the spread because a ratio of 23 between counts of
            1 and 23 is a statement about a thin sample, not about instability,
            and the two are indistinguishable from the ratio alone.
    """

    value: AxisValue
    baseline: float
    sem: float
    real_score: float | None
    excess: float | None
    seeds: int
    stopped_because: str
    trade_count_spread: float
    median_trade_count: float

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe form."""
        return {
            "value": self.value,
            "baseline": self.baseline,
            "sem": self.sem,
            "real_score": self.real_score,
            "excess": self.excess,
            "seeds": self.seeds,
            "stopped_because": self.stopped_because,
            "trade_count_spread": self.trade_count_spread,
            "median_trade_count": self.median_trade_count,
        }


@dataclass(frozen=True)
class BaselineCalibration:
    """Every categorical value's null for one fold.

    Attributes:
        axis: Name of the categorical axis calibrated.
        at: The numeric coordinates the calibration was measured at, held
            fixed while the categorical value varied. Recorded because reuse
            across the rest of the grid is an assumption with a measured error,
            not an identity.
        baselines: One per value of the axis, in the axis's own order.
        runs: Backtests spent, including the unpermuted run per value.
        seconds: Wall clock spent.
    """

    axis: str
    at: tuple[int, ...]
    baselines: tuple[NullBaseline, ...]
    runs: int
    seconds: float

    def by_value(self) -> dict[AxisValue, NullBaseline]:
        """The baselines keyed by axis value."""
        return {item.value: item for item in self.baselines}

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe form."""
        return {
            "axis": self.axis,
            "at": list(self.at),
            "baselines": [item.to_dict() for item in self.baselines],
            "runs": self.runs,
            "seconds": self.seconds,
        }


def sole_categorical_axis(space: SearchSpace) -> int | None:
    """Index of the space's one categorical axis, or ``None`` if it has none.

    Args:
        space: The space.

    Returns:
        The axis index, or ``None``.

    Raises:
        ValueError: If more than one axis is categorical. Two categorical axes
            would need a baseline per *combination*, and the reuse measurement
            that justifies one baseline per value was made on a single axis —
            extending it to a product without measuring the product would be
            assuming the thing this module is careful about.
    """
    indices = [index for index, axis in enumerate(space.axes) if axis.categorical]
    if not indices:
        return None
    if len(indices) > 1:
        raise ValueError(
            "null baselines are calibrated for one categorical axis; this space has "
            f"{[space.axes[index].name for index in indices]}. A baseline per combination is a "
            "different measurement from the one that justified reusing a baseline per value."
        )
    return indices[0]


def _finest(streams: Mapping[StreamKey, OHLCVFrame]) -> Timeframe:
    """The shortest timeframe present — the one a permutation shuffles at."""
    return min((key.timeframe for key in streams), key=lambda tf: tf.duration)


def _permuted(runner: TrialRunner, seed: int, day_origin: DayOrigin) -> TrialRunner:
    """The same runner over one shuffle of its own bars.

    The shuffle keeps every timestamp where it was — see
    :mod:`trading_system.validation.nulls.permutation` — so the in-sample view
    is still an in-sample view: nothing moves a bar across ``trade_end``, and
    :class:`~trading_system.validation.optimization.ISWindowView`'s own
    constructor would refuse it if anything did.
    """
    streams = permute_run_streams(
        runner.view.streams,
        PermutationConfig(finest=_finest(runner.view.streams), day_origin=day_origin, seed=seed),
    )
    return replace(
        runner,
        view=replace(runner.view, streams=streams),
        template=replace(runner.template, streams=streams),
    )


def _centre(space: SearchSpace, categorical_index: int, value_index: int) -> ParamSet:
    """The grid's middle point on every numeric axis, at one categorical value.

    The middle rather than the template's own values because the space need not
    contain them, and rather than a corner because a corner is where a
    parameter is most likely to be at the edge of what anyone intended.
    """
    coords = [
        value_index if index == categorical_index else len(axis.values) // 2
        for index, axis in enumerate(space.axes)
    ]
    return space.point(coords)


def calibrate_null_baselines(
    runner: TrialRunner,
    space: SearchSpace,
    *,
    day_origin: DayOrigin,
    request: BaselineRequest = DEFAULT_BASELINE_REQUEST,
) -> BaselineCalibration | None:
    """Measure each categorical value's permutation null, to its own precision.

    Each value is scored once on real bars and then repeatedly on shuffled
    bars, drawing more shuffles until the standard error of the null mean falls
    below :attr:`BaselineRequest.target_fraction` of that value's excess. The
    next sample size is *projected* from the one already drawn — the standard
    error falls as ``1/sqrt(n)``, so the count needed is known after the first
    batch rather than approached by doubling.

    Args:
        runner: The fold's trial runner. Its view supplies the bars that get
            shuffled, so the null is drawn from the same window the search
            scores on and never from anything later.
        space: The space being searched.
        day_origin: Where the trading day starts, for re-aggregating a coarser
            stream off the shuffled finest one.
        request: How precisely to measure.

    Returns:
        The calibration, or ``None`` if the space has no categorical axis —
        there is then nothing whose zero could differ from anything else's.

    Raises:
        ValueError: If the space has more than one categorical axis.
    """
    index = sole_categorical_axis(space)
    if index is None:
        return None

    axis = space.axes[index]
    started = time.monotonic()
    runs = 0
    results: list[NullBaseline] = []

    for value_index, value in enumerate(axis.values):
        params = _centre(space, index, value_index)
        real = runner.evaluate(params)
        runs += runner.runs_per_trial
        if real.score is None:
            results.append(
                NullBaseline(
                    value=value,
                    baseline=0.0,
                    sem=0.0,
                    real_score=None,
                    excess=None,
                    seeds=0,
                    stopped_because="unscoreable",
                    trade_count_spread=1.0,
                    median_trade_count=0.0,
                )
            )
            continue

        scores: list[float] = []
        counts: list[int] = []
        drawn = 0
        wanted = request.min_seeds
        stopped = "budget"
        while drawn < wanted:
            for seed in range(drawn, wanted):
                outcome = _permuted(runner, request.seed_offset + seed, day_origin).evaluate(params)
                runs += runner.runs_per_trial
                counts.append(outcome.n_trades)
                if outcome.score is not None:
                    scores.append(outcome.score)
            drawn = wanted
            if len(scores) < 2:
                # Every shuffle so far was unscoreable. More of them is the
                # only thing that could change that, so keep drawing to the
                # ceiling rather than dividing by a standard error that does
                # not exist.
                wanted = min(request.max_seeds, drawn * 2)
                if wanted <= drawn:
                    break
                continue
            sem = statistics.stdev(scores) / math.sqrt(len(scores))
            target = request.target_fraction * abs(real.score - statistics.fmean(scores))
            if sem <= target or drawn >= request.max_seeds:
                stopped = "target" if sem <= target else "budget"
                break
            projected = math.ceil(drawn * (sem / target) ** 2) if target > 0 else request.max_seeds
            wanted = min(request.max_seeds, max(drawn + 1, projected))

        baseline = statistics.fmean(scores) if scores else 0.0
        sem = statistics.stdev(scores) / math.sqrt(len(scores)) if len(scores) > 1 else 0.0
        positive = [count for count in counts if count > 0]
        spread = (max(positive) / min(positive)) if positive else 1.0
        median_count = statistics.median(counts) if counts else 0.0
        results.append(
            NullBaseline(
                value=value,
                baseline=baseline,
                sem=sem,
                real_score=real.score,
                excess=real.score - baseline if scores else None,
                seeds=drawn,
                stopped_because=stopped if scores else "unscoreable",
                trade_count_spread=spread,
                median_trade_count=median_count,
            )
        )
        logger.info(
            "null_baseline.value",
            axis=axis.name,
            value=value,
            baseline=baseline,
            sem=sem,
            seeds=drawn,
            stopped_because=results[-1].stopped_because,
            trade_count_spread=spread,
            median_trade_count=median_count,
        )

    calibration = BaselineCalibration(
        axis=axis.name,
        at=tuple(_centre(space, index, 0).coords),
        baselines=tuple(results),
        runs=runs,
        seconds=time.monotonic() - started,
    )
    wide = [
        item.value
        for item in results
        if item.trade_count_spread > 2.0 and item.median_trade_count >= 10
    ]
    if wide:
        logger.warning(
            "null_baseline.trade_count_spread_wide",
            axis=axis.name,
            values=wide,
            detail=(
                "these values' null trade counts vary by more than 2x from shuffle to shuffle on "
                "a sample thick enough for that to mean something, so their baseline averages "
                "over a mixture the reuse argument was never measured on"
            ),
        )
    return calibration


def adjusted_scores(
    records: Sequence[TrialRecord], calibration: BaselineCalibration | None
) -> list[float | None]:
    """Each trial's score as an excess over its own categorical value's null.

    The raw score stays on the record and in the stored trial table; this is
    what selection ranks by, in the same way :func:`roughness` is computed over
    the table rather than folded into the score a trial reports. A trial whose
    value has no measured baseline scores ``None`` — it is left out of the
    selection rather than compared on a scale nobody measured.

    Args:
        records: The evaluated trials.
        calibration: What :func:`calibrate_null_baselines` produced, or
            ``None`` to leave every score untouched. It names the axis it
            calibrated, so the space itself is not needed here.

    Returns:
        One adjusted score per record, aligned with ``records``.
    """
    if calibration is None:
        return [record.outcome.score for record in records]
    baselines = calibration.by_value()
    out: list[float | None] = []
    for record in records:
        score = record.outcome.score
        if score is None:
            out.append(None)
            continue
        item = baselines.get(record.params.as_dict()[calibration.axis])
        out.append(None if item is None or item.excess is None else score - item.baseline)
    return out


def write_calibration(directory: Path, calibration: BaselineCalibration) -> None:
    """Persist a calibration next to the fold's trial table.

    Args:
        directory: Where to write. Created if absent.
        calibration: What to write.
    """
    directory.mkdir(parents=True, exist_ok=True)
    (directory / BASELINES_FILE).write_text(json.dumps(calibration.to_dict(), indent=2))


def read_calibration(directory: Path) -> BaselineCalibration | None:
    """Read a calibration back, or ``None`` if none was written.

    Args:
        directory: Where the fold's search wrote.

    Returns:
        The calibration, or ``None``.
    """
    path = directory / BASELINES_FILE
    if not path.exists():
        return None
    parsed = json.loads(path.read_text())
    return BaselineCalibration(
        axis=parsed["axis"],
        at=tuple(parsed["at"]),
        baselines=tuple(NullBaseline(**item) for item in parsed["baselines"]),
        runs=parsed["runs"],
        seconds=parsed["seconds"],
    )


__all__ = [
    "BASELINES_FILE",
    "DEFAULT_BASELINE_REQUEST",
    "BaselineCalibration",
    "BaselineRequest",
    "NullBaseline",
    "adjusted_scores",
    "calibrate_null_baselines",
    "read_calibration",
    "sole_categorical_axis",
    "write_calibration",
]
