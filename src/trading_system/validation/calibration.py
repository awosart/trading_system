"""Running many iterations of one zero process, and reporting where the real run sits.

**Nothing here is an optimiser or a verdict.** This module answers one
question — "how does the real run's score compare to ``n`` scores produced by
a process that, by construction, has no edge" — and reports a percentile and
a confidence interval. What counts as a good percentile, and what happens
next, are stage 5's business.

**A full run's curve and trade table are not written per iteration.** A
thousand iterations each producing an 8 MB
:class:`~trading_system.backtest.orchestrator.BacktestResult` (P13's own
figure) would be 8 GB on disk for one calibration, almost none of it ever
read back. Only a :class:`CalibrationRecord` — the score, the trade count,
the fill count, the rejection and degradation counters — travels out of a
worker; the whole distribution across ``n`` iterations is one
``records.parquet`` under ``calibration/{id}/``, not ``n`` directories in
``runs/``.

**One seed derivation, shared by every null kind.** :func:`iteration_seed`
hashes ``(calibration_id, iteration)`` with ``blake2b`` — the same discipline
:func:`~trading_system.execution.rng.fill_seed` uses and for the same reason:
:func:`hash` is per-process-randomised for :class:`str` unless
``PYTHONHASHSEED`` is pinned, which is not a reproducibility guarantee at all.
The fill generator inside each iteration's own run is a different concern
and is **not** re-seeded here — it stays the real run's own ``costs.run_seed``,
so a null iteration's costs are drawn by the same process the real run's were,
and the only thing that differs between the two is what this module exists to
isolate (permuted prices, or a random schedule).
"""

import json
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from enum import StrEnum
from hashlib import blake2b
from multiprocessing import get_context
from os import cpu_count
from pathlib import Path
from random import Random
from statistics import median
from typing import Any

import polars as pl

from trading_system.backtest.clock import StreamKey
from trading_system.backtest.orchestrator import BacktestResult, StrategyBinding
from trading_system.backtest.spec import RunInputs
from trading_system.core.types import Timeframe
from trading_system.data.resample import DayOrigin
from trading_system.validation.nulls.permutation import PermutationConfig, permute_run_streams
from trading_system.validation.nulls.random_entry import (
    EntryTraceProfile,
    run_fixed_hold_random_entry_null,
    run_random_entry_null,
)
from trading_system.validation.objective import Objective

#: Filenames under ``calibration/{id}/``.
MANIFEST_FILE = "manifest.json"
RECORDS_FILE = "records.parquet"

#: Iterations the median's bootstrap confidence interval is resampled over.
_BOOTSTRAP_ITERATIONS = 2000

#: Confidence level of the reported interval.
_CONFIDENCE = 0.95

#: Above this relative divergence between a null's median trade count and the
#: real run's, :attr:`CalibrationResult.position_count_divergence` is flagging
#: a different Risk Engine regime, not just stochastic noise in which signals
#: became positions. CLAUDE.md P15 stage 1.5: wide enough that legitimate
#: sizing-path variation does not trip it, narrow enough to catch a schedule
#: that pushes the Risk Engine into systematically more (or fewer) refusals
#: than the real run saw.
POSITION_COUNT_DIVERGENCE_THRESHOLD = 0.2


class NullKind(StrEnum):
    """Which zero process one calibration runs."""

    PERMUTATION = "permutation"
    RANDOM_ENTRY = "random-entry"
    RANDOM_ENTRY_FIXED_HOLD = "random-entry-fixed-hold"


def iteration_seed(calibration_id: str, iteration: int) -> int:
    """Derive one iteration's seed from the calibration's own id and its index.

    Args:
        calibration_id: Identifies the whole calibration.
        iteration: The iteration's index, ``0..n-1``. ``-1`` is reserved for
            the bootstrap CI's own seed, so it never collides with an actual
            iteration.

    Returns:
        A 64-bit seed, depending on nothing else — repeating a calibration
        under the same id reproduces every iteration's seed bit for bit.
    """
    material = f"{calibration_id}\x00{iteration}".encode()
    return int.from_bytes(blake2b(material, digest_size=8).digest(), "big")


@dataclass(frozen=True)
class CalibrationIterationSpec:
    """Everything one worker needs for one iteration, picklable end to end.

    Only the fields a given ``kind`` actually reads are required in practice;
    the others are ``None`` and unused — see :func:`run_one_iteration`. Kept
    as one dataclass rather than three, so :func:`run_one_iteration` stays the
    single picklable, module-level unit of work every null kind's worker uses,
    matching the discipline
    :func:`~trading_system.backtest.parallel.run_one_to_store` already sets.
    """

    kind: NullKind
    base: RunInputs
    objective: Objective
    iteration: int
    seed: int
    finest: Timeframe | None = None
    day_origin: DayOrigin | None = None
    key: StreamKey | None = None
    real_binding: StrategyBinding | None = None
    profile: EntryTraceProfile | None = None
    stop_pips: float = 20.0
    max_concurrent_positions: int = 50


@dataclass(frozen=True)
class CalibrationRecord:
    """One iteration's compact result — no curve, no trade table.

    Attributes:
        iteration: Index within the calibration.
        seed: This iteration's own seed.
        score: ``objective.score(result)``, or ``None`` if the run could not
            be scored (:class:`~trading_system.validation.objective.Objective.score`
            raised) — recorded rather than dropped, so "N iterations produced
            no score" is a visible count, not a shrunken sample nobody
            noticed shrank.
        n_trades: Trades the iteration closed.
        fills: Fills the iteration executed.
        rejections: Risk Engine refusals, every reason present.
        degradations: Risk Engine measurements that fell back to a prior.
    """

    iteration: int
    seed: int
    score: float | None
    n_trades: int
    fills: int
    rejections: Mapping[str, int]
    degradations: Mapping[str, int]


def run_one_iteration(spec: CalibrationIterationSpec) -> CalibrationRecord:
    """Run one null iteration and compact it to a record. The unit of work a worker receives.

    Args:
        spec: Everything needed, including which of the three null kinds to run.

    Returns:
        The record.

    Raises:
        ValueError: If ``spec.kind`` requires a field left ``None``.
    """
    if spec.kind is NullKind.PERMUTATION:
        if spec.finest is None or spec.day_origin is None:
            raise ValueError("permutation iterations require finest and day_origin")
        permuted = permute_run_streams(
            spec.base.streams,
            PermutationConfig(finest=spec.finest, day_origin=spec.day_origin, seed=spec.seed),
        )
        result = spec.base.with_streams(permuted).run()
    elif spec.kind is NullKind.RANDOM_ENTRY:
        if spec.key is None or spec.real_binding is None or spec.profile is None:
            raise ValueError("random-entry iterations require key, real_binding and profile")
        result = run_random_entry_null(
            spec.base,
            spec.key,
            spec.real_binding,
            spec.profile,
            seed=spec.seed,
            stop_pips=spec.stop_pips,
            max_concurrent_positions=spec.max_concurrent_positions,
        )
    else:
        if spec.key is None or spec.profile is None:
            raise ValueError("random-entry-fixed-hold iterations require key and profile")
        result = run_fixed_hold_random_entry_null(
            spec.base,
            spec.key,
            spec.profile,
            seed=spec.seed,
            stop_pips=spec.stop_pips,
            max_concurrent_positions=spec.max_concurrent_positions,
        )

    try:
        score = spec.objective.score(result)
    except ValueError:
        score = None
    return CalibrationRecord(
        iteration=spec.iteration,
        seed=spec.seed,
        score=score,
        n_trades=len(result.trades),
        fills=result.fills,
        rejections=dict(result.rejections),
        degradations=dict(result.degradations),
    )


def _run_batch(
    specs: Sequence[CalibrationIterationSpec],
    *,
    parallel_threshold_seconds: float,
    workers: int | None,
) -> list[CalibrationRecord]:
    """Sequential or parallel, decided by timing the first iteration — see CLAUDE.md."""
    if not specs:
        return []
    started = time.perf_counter()
    first = run_one_iteration(specs[0])
    elapsed = time.perf_counter() - started
    rest = specs[1:]
    if not rest:
        return [first]
    if elapsed < parallel_threshold_seconds:
        return [first, *(run_one_iteration(spec) for spec in rest)]

    processes = min(workers if workers is not None else (cpu_count() or 1), len(rest))
    if processes < 2:
        return [first, *(run_one_iteration(spec) for spec in rest)]
    with ProcessPoolExecutor(max_workers=processes, mp_context=get_context("spawn")) as pool:
        rest_records = list(pool.map(run_one_iteration, rest))
    return [first, *rest_records]


def _percentile_of(value: float, distribution: Sequence[float]) -> float:
    """Share of ``distribution`` at or below ``value``, in ``[0, 100]``."""
    if not distribution:
        raise ValueError("cannot compute a percentile against an empty distribution")
    at_or_below = sum(1 for item in distribution if item <= value)
    return 100.0 * at_or_below / len(distribution)


def _bootstrap_median_ci(values: Sequence[float], *, seed: int) -> tuple[float, float]:
    """A percentile bootstrap confidence interval for the median of ``values``."""
    if len(values) < 2:
        raise ValueError("need at least two scored iterations for a bootstrap CI")
    rng = Random(seed)
    n = len(values)
    medians = sorted(
        median(values[rng.randrange(n)] for _ in range(n)) for _ in range(_BOOTSTRAP_ITERATIONS)
    )
    alpha = (1 - _CONFIDENCE) / 2
    low = medians[max(0, int(alpha * _BOOTSTRAP_ITERATIONS))]
    high = medians[min(_BOOTSTRAP_ITERATIONS - 1, int((1 - alpha) * _BOOTSTRAP_ITERATIONS))]
    return low, high


@dataclass(frozen=True)
class CalibrationResult:
    """A finished calibration: every iteration's record, and where the real run sits.

    Attributes:
        calibration_id: Identifies this calibration.
        kind: Which null was run.
        n: Iterations requested.
        real_score: ``objective.score(real_result)``, or ``None`` if it could
            not be scored.
        real_trade_count: ``len(real_result.trades)`` — what
            :func:`position_count_divergence` compares every iteration's own
            ``n_trades`` against. Matched by *signal* count, per CLAUDE.md
            P15 stage 1.5's answer to "signals or positions": a null's
            schedule is built to reproduce the real run's signal count, and
            how many of those signals actually became positions is something
            the Risk Engine and Prop Guard decide, not something this module
            forces to agree — it only reports how far apart they ended up.
        records: Every iteration's record, in iteration order.
        n_scored: Records with a non-``None`` score.
        percentile: Share of scored iterations at or below ``real_score``, in
            ``[0, 100]`` — ``None`` if ``real_score`` is ``None`` or nothing
            scored.
        median_score: Median of the scored iterations, or ``None``.
        median_ci_low: Lower bound of the median's bootstrap confidence
            interval, or ``None`` with fewer than two scored iterations.
        median_ci_high: Upper bound.
        store_root: Where ``calibration/{calibration_id}/`` lives.
    """

    calibration_id: str
    kind: NullKind
    n: int
    real_score: float | None
    real_trade_count: int
    records: tuple[CalibrationRecord, ...]
    n_scored: int
    percentile: float | None
    median_score: float | None
    median_ci_low: float | None
    median_ci_high: float | None
    store_root: Path

    @property
    def directory(self) -> Path:
        """``calibration/{calibration_id}/``."""
        return self.store_root / "calibration" / self.calibration_id

    @property
    def median_null_trade_count(self) -> float | None:
        """Median ``n_trades`` across every iteration, or ``None`` with no records."""
        counts = [record.n_trades for record in self.records]
        return median(counts) if counts else None

    @property
    def position_count_divergence(self) -> float | None:
        """``|median null trade count - real trade count| / real trade count``.

        ``None`` when there are no records, or ``inf`` when the real run
        itself closed zero trades — dividing by zero would otherwise read as
        "perfect agreement" for the one case that is anything but.
        """
        null_count = self.median_null_trade_count
        if null_count is None:
            return None
        if self.real_trade_count == 0:
            return float("inf")
        return abs(null_count - self.real_trade_count) / self.real_trade_count


def run_calibration(
    kind: NullKind,
    base: RunInputs,
    *,
    objective: Objective,
    real_result: BacktestResult,
    n: int,
    calibration_id: str,
    store_root: Path,
    finest: Timeframe | None = None,
    day_origin: DayOrigin | None = None,
    key: StreamKey | None = None,
    real_binding: StrategyBinding | None = None,
    profile: EntryTraceProfile | None = None,
    stop_pips: float = 20.0,
    max_concurrent_positions: int = 50,
    parallel_threshold_seconds: float = 2.0,
    workers: int | None = None,
) -> CalibrationResult:
    """Run ``n`` iterations of one null and report where the real run sits.

    Idempotent on ``calibration_id``: if ``calibration/{calibration_id}/manifest.json``
    already exists, it is read back and **no iteration is re-run**.

    Args:
        kind: Which null to run.
        base: The real run's own inputs.
        objective: Scores each iteration and the real result.
        real_result: The real run's own output, already computed by the caller.
        n: Iterations to run.
        calibration_id: Identifies this calibration; also seeds every iteration.
        store_root: Where ``calibration/{calibration_id}/`` is written.
        finest: Required for :attr:`NullKind.PERMUTATION`.
        day_origin: Required for :attr:`NullKind.PERMUTATION`.
        key: Required for :attr:`NullKind.RANDOM_ENTRY` and
            :attr:`NullKind.RANDOM_ENTRY_FIXED_HOLD`.
        real_binding: Required for :attr:`NullKind.RANDOM_ENTRY`.
        profile: Required for :attr:`NullKind.RANDOM_ENTRY` and
            :attr:`NullKind.RANDOM_ENTRY_FIXED_HOLD`.
        stop_pips: Passed to the random-entry null variants.
        max_concurrent_positions: Passed to the random-entry null variants.
        parallel_threshold_seconds: Below this many seconds for the first
            iteration, the rest run sequentially — see
            :mod:`trading_system.validation.walkforward` for the same
            reasoning applied to fold batches.
        workers: Worker count for a parallel batch. The machine's CPU count
            by default.

    Returns:
        The result, freshly computed or read back.
    """
    manifest_path = store_root / "calibration" / calibration_id / MANIFEST_FILE
    if manifest_path.exists():
        return _read_calibration(manifest_path, store_root)

    specs = [
        CalibrationIterationSpec(
            kind=kind,
            base=base,
            objective=objective,
            iteration=iteration,
            seed=iteration_seed(calibration_id, iteration),
            finest=finest,
            day_origin=day_origin,
            key=key,
            real_binding=real_binding,
            profile=profile,
            stop_pips=stop_pips,
            max_concurrent_positions=max_concurrent_positions,
        )
        for iteration in range(n)
    ]
    records = _run_batch(
        specs, parallel_threshold_seconds=parallel_threshold_seconds, workers=workers
    )

    try:
        real_score = objective.score(real_result)
    except ValueError:
        real_score = None

    scores = [record.score for record in records if record.score is not None]
    percentile = _percentile_of(real_score, scores) if real_score is not None and scores else None
    median_score = median(scores) if scores else None
    ci_low: float | None
    ci_high: float | None
    if len(scores) >= 2:
        ci_low, ci_high = _bootstrap_median_ci(scores, seed=iteration_seed(calibration_id, -1))
    else:
        ci_low, ci_high = None, None

    result = CalibrationResult(
        calibration_id=calibration_id,
        kind=kind,
        n=n,
        real_score=real_score,
        real_trade_count=len(real_result.trades),
        records=tuple(records),
        n_scored=len(scores),
        percentile=percentile,
        median_score=median_score,
        median_ci_low=ci_low,
        median_ci_high=ci_high,
        store_root=store_root,
    )
    _write_calibration(result, real_run_id=base.manifest().run_id)
    return result


def _write_calibration(result: CalibrationResult, *, real_run_id: str) -> None:
    """Persist a calibration's manifest and its records, in one parquet file."""
    directory = result.directory
    directory.mkdir(parents=True, exist_ok=True)

    rows = [
        {
            "iteration": record.iteration,
            "seed": record.seed,
            "score": record.score,
            "n_trades": record.n_trades,
            "fills": record.fills,
            "rejections": json.dumps(dict(record.rejections)),
            "degradations": json.dumps(dict(record.degradations)),
        }
        for record in result.records
    ]
    frame = pl.DataFrame(
        rows,
        schema={
            "iteration": pl.Int64,
            "seed": pl.UInt64,
            "score": pl.Float64,
            "n_trades": pl.Int64,
            "fills": pl.Int64,
            "rejections": pl.String,
            "degradations": pl.String,
        },
    )
    frame.write_parquet(directory / RECORDS_FILE)

    payload = {
        "calibration_id": result.calibration_id,
        "kind": result.kind.value,
        "n": result.n,
        "real_run_id": real_run_id,
        "real_score": result.real_score,
        "real_trade_count": result.real_trade_count,
        "n_scored": result.n_scored,
        "percentile": result.percentile,
        "median_score": result.median_score,
        "median_ci_low": result.median_ci_low,
        "median_ci_high": result.median_ci_high,
    }
    (directory / MANIFEST_FILE).write_text(json.dumps(payload, indent=2))


def _read_calibration(manifest_path: Path, store_root: Path) -> CalibrationResult:
    """Rebuild a :class:`CalibrationResult` from a previously written manifest."""
    payload: dict[str, Any] = json.loads(manifest_path.read_text())
    frame = pl.read_parquet(manifest_path.parent / RECORDS_FILE)
    records = tuple(
        CalibrationRecord(
            iteration=row["iteration"],
            seed=row["seed"],
            score=row["score"],
            n_trades=row["n_trades"],
            fills=row["fills"],
            rejections=json.loads(row["rejections"]),
            degradations=json.loads(row["degradations"]),
        )
        for row in frame.iter_rows(named=True)
    )
    return CalibrationResult(
        calibration_id=payload["calibration_id"],
        kind=NullKind(payload["kind"]),
        n=payload["n"],
        real_score=payload["real_score"],
        real_trade_count=payload["real_trade_count"],
        records=records,
        n_scored=payload["n_scored"],
        percentile=payload["percentile"],
        median_score=payload["median_score"],
        median_ci_low=payload["median_ci_low"],
        median_ci_high=payload["median_ci_high"],
        store_root=store_root,
    )
