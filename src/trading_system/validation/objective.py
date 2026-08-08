"""What one number a run is scored by, and how flat the ground around it is.

**The stability penalty is not a term inside :class:`Objective`, and that is a
decision rather than an omission.** Stage 1.5 left the penalty unimplemented
and predicted stage 2 would "compose a penalty into a *new* implementation
without this one changing shape". Building it made the shape question concrete:
``Objective.score`` receives one
:class:`~trading_system.backtest.orchestrator.BacktestResult` and nothing else,
while instability is by definition a property of a *neighbourhood* of parameter
values — a quantity no single run can see. Putting the penalty behind
``score(result)`` would mean handing that method a neighbourhood it has no way
to obtain, so the penalty lives here as a separate function over the trial
table (:func:`roughness`), applied by
:class:`~trading_system.validation.walkforward.OptimizingSelector` after a
search finishes. :class:`Objective` stays a one-method protocol, exactly as
promised.

**Neighbourhood is measured in grid-index space, not parameter units.** A
parameter table mixes periods (tens of bars) with ATR multiples (units of
~2), so a distance in raw parameter units would be dominated by whichever axis
happens to carry the largest numbers, and the "neighbourhood" of a point would
silently mean "its neighbours along the period axis only". Every axis of a
:class:`~trading_system.validation.optimization.SearchSpace` is discrete by
construction, so each point has an integer coordinate per axis, and two points
are neighbours when they differ by at most one step on every axis
(Chebyshev/L-infinity distance ``<= 1``). On a full grid this is the ordinary
``3^d - 1`` neighbourhood; under random or TPE sampling it is whatever
actually got sampled nearby, which is why :class:`PlateauAnalysis` reports how
many points had no neighbours at all rather than letting sparse sampling
quietly produce a zero penalty.

**The plateau tolerance is measured in units of the score's own dispersion,
never as a fraction of the best score.** Scores here are Sortino times a trade
count and are routinely negative — stage 1.5 measured the demo strategy at
-0.359 — so "within 10% of the best" is undefined in exactly the regime this
system keeps landing in. This is the same argument
:mod:`trading_system.validation.report` already makes when it refuses to
publish an IS/OOS ratio. A tolerance of ``t`` sigmas of the scored trials'
standard deviation is well defined for any sign, and degenerates correctly:
when every trial scores identically the dispersion is zero and the whole space
is one plateau, which is the true answer.
"""

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from trading_system.analytics.metrics import daily_curve, sortino_daily
from trading_system.backtest.orchestrator import BacktestResult


class Objective(Protocol):
    """Scores one run. Higher is better; the scale is the implementation's own."""

    def score(self, result: BacktestResult) -> float:
        """Score one run's result.

        Args:
            result: What the run produced.

        Returns:
            The score.

        Raises:
            ValueError: If the result cannot be scored — no trades, or too
                few daily returns for the underlying statistic to be defined.
                Raised rather than returning a sentinel: a caller running many
                iterations (:mod:`trading_system.validation.calibration`)
                decides for itself whether an unscoreable iteration is
                discarded or counted, and a silent ``NaN`` would take that
                decision away from it.
        """
        ...


@dataclass(frozen=True)
class SortinoTimesSqrtTrades:
    """The default objective: annualised Sortino, scaled by trade count.

    Sortino alone rewards a curve that is smooth on the downside regardless of
    how many trades produced it, which a two-trade run can achieve by luck as
    easily as a two-hundred-trade one by design. Multiplying by
    ``sqrt(trade count)`` is the same discount a Sharpe ratio's own standard
    error carries — it does not *decide* how many trades are enough (that
    judgment stays in :mod:`trading_system.analytics.statistical`), it just
    stops a thin sample from scoring identically to a thick one.

    Attributes:
        risk_free_rate: Passed straight through to
            :func:`~trading_system.analytics.metrics.sortino_daily`.
        mar: Minimum acceptable per-period return, same function.
    """

    risk_free_rate: float = 0.0
    mar: float = 0.0

    def score(self, result: BacktestResult) -> float:
        """``sortino_daily(daily_curve(result.curve)).value * sqrt(len(result.trades))``.

        Args:
            result: What the run produced.

        Returns:
            The score.

        Raises:
            ValueError: If there are no trades, fewer than one daily return,
                or the downside deviation is zero — the same conditions
                :func:`~trading_system.analytics.metrics.sortino_daily` itself
                refuses to paper over.
        """
        if not result.trades:
            raise ValueError("cannot score a run with zero trades")
        daily = daily_curve(result.curve)
        sortino = sortino_daily(daily, risk_free_rate=self.risk_free_rate, mar=self.mar)
        return sortino.value * math.sqrt(len(result.trades))


@dataclass(frozen=True)
class ExpectancyR:
    """Mean realised R per trade — the objective both nulls are calibrated on.

    Not the default, and not a rival to :class:`SortinoTimesSqrtTrades`: the
    two answer different questions. The optimiser wants a score that discounts
    a thin sample, because it is choosing between parameter sets on unequal
    trade counts. A null comparison wants the plainest possible statement of
    edge per unit of risk, because the whole question is whether the real tape
    beats a structureless one on the *same* metric — and a trade-count factor
    would let a null that simply traded more outrank the real run.

    It is also the number CLAUDE.md records the P15 nulls against, so a verdict
    computed today is comparable with the one recorded then.
    """

    def score(self, result: BacktestResult) -> float:
        """Mean of ``trade.realized_r``.

        Args:
            result: What the run produced.

        Returns:
            The mean realised R.

        Raises:
            ValueError: If the run closed no trades. Raised rather than
                returning zero: "no trades" and "trades that averaged nothing"
                are different facts, and the caller running many iterations
                decides which to discard.
        """
        if not result.trades:
            raise ValueError("cannot score a run with zero trades")
        return statistics.fmean(trade.realized_r for trade in result.trades)


# ---------------------------------------------------------------------------
# Neighbourhood stability over a finished trial table
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScoredPoint:
    """One evaluated parameter set, reduced to what a neighbourhood needs.

    Deliberately not a :class:`~trading_system.validation.optimization.ParamSet`:
    this module would then have to import the optimiser that imports it. The
    caller converts, and gets back *indices into the sequence it passed*, so no
    parameter values are duplicated into a second representation that could
    drift from the first.

    Attributes:
        coords: The point's grid index on each axis, in the axis order the
            search space declares.
        score: The objective's value here. Only scoreable trials belong in a
            sequence handed to :func:`analyse_plateau`; an unscoreable one has
            no height and therefore no slope.
    """

    coords: tuple[int, ...]
    score: float


def neighbours(points: Sequence[ScoredPoint], index: int) -> tuple[int, ...]:
    """Indices of the points adjacent to ``points[index]``, itself excluded.

    Adjacency is Chebyshev distance ``<= 1`` in grid-index space — see the
    module docstring on why index space and not parameter units.

    Args:
        points: The evaluated points. Every one must have the same number of
            coordinates.
        index: Which point's neighbours to find.

    Returns:
        Indices into ``points``, ascending.

    Raises:
        IndexError: If ``index`` is out of range.
        ValueError: If the points do not all share a coordinate count.
    """
    centre = points[index]
    width = len(centre.coords)
    result: list[int] = []
    for other_index, other in enumerate(points):
        if len(other.coords) != width:
            raise ValueError(
                f"point {other_index} has {len(other.coords)} coordinates, "
                f"point {index} has {width}"
            )
        if other_index == index:
            continue
        if all(abs(a - b) <= 1 for a, b in zip(centre.coords, other.coords, strict=True)):
            result.append(other_index)
    return tuple(result)


def roughness(points: Sequence[ScoredPoint], index: int) -> float | None:
    """How steeply the score falls off around ``points[index]``.

    The mean *drop* to an adjacent point, counting only neighbours that score
    lower::

        roughness = mean over neighbours of max(0, score(here) - score(there))

    Only downward differences count because the quantity being penalised is a
    peak's sharpness, not a slope's steepness: a point partway up a hillside has
    neighbours above it, and that says nothing about whether the summit it
    leads to is a spike or a plateau. A genuine spike — high here, much lower
    on every side — scores a large roughness; the middle of a flat region
    scores near zero however high or low that region sits, which is what
    "the penalty grows with the steepness of the local maximum, not merely with
    its height" requires.

    Args:
        points: The evaluated points.
        index: Which point to measure around.

    Returns:
        The mean drop, or ``None`` if the point has no neighbours at all — a
        real possibility under sparse random or TPE sampling, and reported as
        the absence it is rather than as a zero penalty that would make an
        unmeasured point look maximally stable.
    """
    adjacent = neighbours(points, index)
    if not adjacent:
        return None
    here = points[index].score
    return statistics.fmean(max(0.0, here - points[other].score) for other in adjacent)


@dataclass(frozen=True)
class PlateauAnalysis:
    """The best region of a finished search, not merely its best point.

    Attributes:
        best_index: Index of the highest *penalised* score. It coincides with a
            plain ``argmax`` over raw scores only when ``penalty_weight`` is
            zero; above zero it is already the penalty's choice of peak, and
            the comparison "what would argmax have picked" therefore needs a
            separate run at ``penalty_weight=0``, not this field.
        selected_index: Index of the point actually chosen: the evaluated
            point nearest the plateau's centroid. An evaluated point rather
            than the centroid itself, because the arithmetic mean of
            ``{30, 50, 80}`` is 53.3 — not a member of the space, possibly not
            even a legal value, and above all a point *no run ever measured*,
            whose reported IS score would therefore be a number nobody
            computed.
        penalty_weight: What :attr:`roughness_at_best` was multiplied by
            before the maximum was taken.
        tolerance: Absolute score margin defining plateau membership,
            ``tolerance_sigmas`` times the standard deviation of the scored
            trials.
        tolerance_sigmas: The margin as configured, in standard deviations.
        plateau_size: How many evaluated points are in the best plateau — the
            connected region, under the same adjacency
            :func:`neighbours` uses, of points scoring within
            :attr:`tolerance` of the best and reachable from it.
        plateau_fraction: :attr:`plateau_size` over the number of scored
            trials. A sharp peak and a broad plateau with the same maximum
            differ here and nowhere else in a report.
        axis_extent: Per axis, how many distinct grid indices the plateau
            spans. ``1`` on every axis is a single point; a plateau wide on one
            axis and narrow on another is visible here and invisible in
            :attr:`plateau_size` alone.
        selection_shift: Chebyshev index distance between :attr:`best_index`
            and :attr:`selected_index`. Zero when the peak is its own plateau
            centre.
        score_gap: Penalised score at the best point minus that at the
            selected one — never negative. Large gap with large shift reads as
            "the peak was sharp and we declined it deliberately", which is the
            thing a reader must be able to see.
        roughness_at_best: :func:`roughness` at :attr:`best_index`.
        roughness_at_selected: :func:`roughness` at :attr:`selected_index`.
        n_scored: How many points were scoreable and therefore analysed.
        n_without_neighbours: How many of those had no adjacent point, so
            their roughness could not be measured and was treated as zero for
            ranking. Under a full grid this is zero; under sparse sampling it
            is the honest size of what the penalty could not see.
    """

    best_index: int
    selected_index: int
    penalty_weight: float
    tolerance: float
    tolerance_sigmas: float
    plateau_size: int
    plateau_fraction: float
    axis_extent: tuple[int, ...]
    selection_shift: int
    score_gap: float
    roughness_at_best: float | None
    roughness_at_selected: float | None
    n_scored: int
    n_without_neighbours: int


def _connected_plateau(points: Sequence[ScoredPoint], seed: int, floor: float) -> set[int]:
    """Indices reachable from ``seed`` through points scoring at or above ``floor``.

    Connected rather than merely "every point above the floor": two separate
    high regions on opposite sides of the space are two plateaus, and reporting
    their combined size as one would describe a width that no continuous walk
    through the parameter space actually has.
    """
    region: set[int] = {seed}
    frontier = [seed]
    while frontier:
        current = frontier.pop()
        for candidate in neighbours(points, current):
            if candidate not in region and points[candidate].score >= floor:
                region.add(candidate)
                frontier.append(candidate)
    return region


def analyse_plateau(
    points: Sequence[ScoredPoint], *, tolerance_sigmas: float = 1.0, penalty_weight: float = 0.5
) -> PlateauAnalysis:
    """Find the best plateau and the evaluated point at its centre.

    Ranking uses the *penalised* score, ``score - penalty_weight * roughness``,
    so that the penalty decides which peak wins; the plateau's centre then
    decides where within that peak the selection lands. Both steps are needed:
    the penalty alone would still return a single argmax, and the centring
    alone would still centre on whichever spike happened to be tallest.

    Args:
        points: Every scoreable evaluated point. Order is the caller's; the
            returned indices refer to it.
        tolerance_sigmas: Plateau membership margin, in standard deviations of
            the scores present. See the module docstring on why sigmas and not
            a fraction of the best score.
        penalty_weight: Multiplier on :func:`roughness` when ranking. Zero
            reproduces plain argmax selection, which is what the DoD's
            overfitted-configuration test compares against.

    Returns:
        The analysis.

    Raises:
        ValueError: If ``points`` is empty, ``tolerance_sigmas`` is negative,
            or ``penalty_weight`` is negative.
    """
    if not points:
        raise ValueError("cannot analyse a plateau over zero scored points")
    if tolerance_sigmas < 0:
        raise ValueError(f"tolerance_sigmas must be non-negative, got {tolerance_sigmas}")
    if penalty_weight < 0:
        raise ValueError(f"penalty_weight must be non-negative, got {penalty_weight}")

    measured = [roughness(points, index) for index in range(len(points))]
    # An unmeasurable roughness ranks as zero penalty rather than dropping the
    # point: a lone sample is not evidence of instability, and discarding it
    # would silently shrink the searched space. How many such points there were
    # is reported instead, so the reader sees what the penalty could not see.
    penalised = [
        point.score - penalty_weight * (value if value is not None else 0.0)
        for point, value in zip(points, measured, strict=True)
    ]
    best_index = max(range(len(points)), key=lambda index: penalised[index])

    scores = [point.score for point in points]
    spread = statistics.stdev(scores) if len(scores) > 1 else 0.0
    tolerance = tolerance_sigmas * spread

    ranked = [
        ScoredPoint(coords=point.coords, score=value)
        for point, value in zip(points, penalised, strict=True)
    ]
    region = _connected_plateau(ranked, best_index, penalised[best_index] - tolerance)

    width = len(points[0].coords)
    axis_extent = tuple(
        len({points[index].coords[axis] for index in region}) for axis in range(width)
    )
    centroid = [
        statistics.fmean(float(points[index].coords[axis]) for index in region)
        for axis in range(width)
    ]
    # Nearest by squared Euclidean distance to the centroid, ties broken by the
    # lowest index so that two equidistant candidates resolve identically on
    # every re-run — the determinism invariant this stage owes GridSearch.
    selected_index = min(
        sorted(region),
        key=lambda index: sum(
            (float(points[index].coords[axis]) - centroid[axis]) ** 2 for axis in range(width)
        ),
    )
    shift = max(
        abs(a - b)
        for a, b in zip(points[best_index].coords, points[selected_index].coords, strict=True)
    )
    return PlateauAnalysis(
        best_index=best_index,
        selected_index=selected_index,
        penalty_weight=penalty_weight,
        tolerance=tolerance,
        tolerance_sigmas=tolerance_sigmas,
        plateau_size=len(region),
        plateau_fraction=len(region) / len(points),
        axis_extent=axis_extent,
        selection_shift=shift,
        score_gap=penalised[best_index] - penalised[selected_index],
        roughness_at_best=measured[best_index],
        roughness_at_selected=measured[selected_index],
        n_scored=len(points),
        n_without_neighbours=sum(1 for value in measured if value is None),
    )
