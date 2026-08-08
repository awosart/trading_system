"""Many runs at once — and a statement about which runs those may be.

**Partitions are whole, independent runs. A single portfolio is never split.**
Handing EURUSD to one process and NAS100 to another looks like the obvious
parallelisation and would produce a different backtest, not a faster one: sizing
is a fraction of *equity* both instruments move, portfolio heat is a budget they
share, and a daily loss limit is one limit over both. Those couplings are the
Risk Engine working, so a split that removes them is a different question being
answered quickly. What is genuinely independent — a walk-forward fold, a
parameter sweep, one instrument traded on its own account — is a
:class:`~trading_system.backtest.spec.RunInputs` of its own, and *those* run in
parallel with results identical to running them one after another.

**Identical, not "identical up to floating point".** Nothing in a run reads a
wall clock, a process id or a worker count, and P12 seeds each fill from
``(run_seed, symbol, ts, order_id)`` rather than from a shared generator, so a
fill draws the same number wherever it is computed. That claim is checked in
``tests/backtest/test_parallel.py`` by digesting both sets of results, not
asserted here.

**Spawn, not fork.** Fork copies a process that has already imported polars and
started its thread pool, and a forked child holding locks that were taken in the
parent is the classic way to get a hang that reproduces once a week. Spawn costs
a few hundred milliseconds per worker; a run worth parallelising is worth more
than that, and :func:`run_parallel` says so by falling back to a sequential run
when the pool would be a single worker.
"""

import itertools
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from multiprocessing import get_context
from os import cpu_count
from pathlib import Path
from typing import Any

from trading_system.backtest.orchestrator import BacktestResult
from trading_system.backtest.reproducibility import result_counters, result_digest, write_run
from trading_system.backtest.spec import RunInputs
from trading_system.core.logging import get_logger

logger = get_logger(__name__)


def run_one(inputs: RunInputs) -> BacktestResult:
    """Walk one run. The unit of work a worker receives.

    Module level and public because a process pool pickles this function *by
    name*: a closure or a method would be unpicklable, and the failure arrives
    as an opaque error from inside the pool rather than at the call that made
    the mistake.

    Args:
        inputs: The run to walk.

    Returns:
        What it produced.
    """
    return inputs.run()


def run_sequential(specs: Sequence[RunInputs]) -> list[BacktestResult]:
    """Walk every run, one after another, in the order given.

    Args:
        specs: The runs.

    Returns:
        One result per run, in the same order.
    """
    return [run_one(inputs) for inputs in specs]


def run_parallel(specs: Sequence[RunInputs], *, workers: int | None = None) -> list[BacktestResult]:
    """Walk every run across a process pool, in the order given.

    Args:
        specs: The runs. Each must be independent of the others — see this
            module's docstring on why a single portfolio is not split here.
        workers: How many processes. The machine's CPU count by default, never
            more than there are runs to walk.

    Returns:
        One result per run, **in the order the specs were given**, which is what
        makes the output comparable to :func:`run_sequential`'s and independent
        of which worker happened to finish first.

    Raises:
        ValueError: If ``workers`` is not positive.
    """
    if workers is not None and workers < 1:
        raise ValueError(f"workers must be at least 1, got {workers}")
    processes = min(workers if workers is not None else (cpu_count() or 1), len(specs))
    if processes < 2:
        # One worker is a sequential run with a process spawn in front of it.
        # The results are the same by construction — a run reads nothing that
        # differs between processes — so this is a cost saved, not a behaviour.
        return run_sequential(specs)

    logger.info("backtest.parallel_start", runs=len(specs), workers=processes)
    with ProcessPoolExecutor(max_workers=processes, mp_context=get_context("spawn")) as pool:
        # `map` yields in submission order regardless of completion order, which
        # is the whole reason it is used instead of as_completed: results that
        # arrived in finishing order would make the output of a reproducible run
        # depend on the machine's scheduling.
        return list(pool.map(run_one, specs))


@dataclass(frozen=True)
class StoredRun:
    """A run's identity and counters, with its tables left on disk.

    The counterpart to :class:`~trading_system.backtest.reproducibility.StoredRun`
    that this module's docstring names as the P13 limitation's resolution: that
    one is what you get from *reading a run back*, tables rebuilt in full, and
    is exactly what must not travel through a pool's pipe — a 60k-row curve of
    ``Decimal`` fields pickles to megabytes per run, which is what capped
    :func:`run_parallel`'s speedup at 2x rather than the core count. This one is
    what a worker hands back instead: the id, the digest, where the tables were
    written, and the counters a report reads without touching a
    :class:`~pathlib.Path`.

    Attributes:
        run_id: The run's identity, from its manifest.
        digest: :func:`~trading_system.backtest.reproducibility.result_digest`
            of what it produced.
        path: Directory :func:`~trading_system.backtest.reproducibility.write_run`
            wrote it to — ``curve.parquet``, ``trades.parquet``,
            ``manifest.json``, ``result.json``.
        counters: Every field of :class:`~trading_system.backtest.orchestrator.BacktestResult`
            except the curve and the trade list — rejections, degradations,
            exit and entry drops, signal drops, expired orders, cost
            degradations, ATR ratio coverage, fills, FX fallback marks, and
            open-at-end. The curve and the trades are read from ``path``
            explicitly, by whoever needs them.
    """

    run_id: str
    digest: str
    path: Path
    counters: Mapping[str, Any]


def run_one_to_store(inputs: RunInputs, root: Path) -> StoredRun:
    """Walk one run and store it, returning only its identity and counters.

    The unit of work a worker receives for the store-writing variants, for the
    same picklability reason :func:`run_one` is module level.

    Args:
        inputs: The run to walk.
        root: Store root the result is written under.

    Returns:
        What identifies and counts the run — not what it produced in full.
    """
    result = inputs.run()
    manifest = inputs.manifest()
    path = write_run(root, manifest, result)
    return StoredRun(
        run_id=manifest.run_id,
        digest=result_digest(result),
        path=path,
        counters=result_counters(result),
    )


def run_sequential_to_store(specs: Sequence[RunInputs], root: Path) -> list[StoredRun]:
    """Walk every run, one after another, storing each as it finishes.

    Args:
        specs: The runs.
        root: Store root.

    Returns:
        One :class:`StoredRun` per run, in the same order.
    """
    return [run_one_to_store(inputs, root) for inputs in specs]


def run_parallel_to_store(
    specs: Sequence[RunInputs], root: Path, *, workers: int | None = None
) -> list[StoredRun]:
    """Walk every run across a process pool, storing each in its own worker.

    The pipe this module's docstring flags as the ceiling on :func:`run_parallel`'s
    speedup carries a :class:`StoredRun` back instead of a
    :class:`~trading_system.backtest.orchestrator.BacktestResult` — id, digest and
    a few counters rather than a curve and a trade table serialised through
    :class:`~decimal.Decimal`. The write itself
    (:func:`~trading_system.backtest.reproducibility.write_run`) happens inside
    the worker, where the data already lives, not after the trip back.

    Args:
        specs: The runs. Each must be independent of the others, per this
            module's own docstring.
        root: Store root every worker writes under. Must already exist or be
            creatable by each worker — :func:`~trading_system.backtest.reproducibility.write_run`
            creates the per-run subdirectory itself.
        workers: How many processes. The machine's CPU count by default, never
            more than there are runs to walk.

    Returns:
        One :class:`StoredRun` per run, in the order the specs were given.

    Raises:
        ValueError: If ``workers`` is not positive.
    """
    if workers is not None and workers < 1:
        raise ValueError(f"workers must be at least 1, got {workers}")
    processes = min(workers if workers is not None else (cpu_count() or 1), len(specs))
    if processes < 2:
        return run_sequential_to_store(specs, root)

    logger.info("backtest.parallel_store_start", runs=len(specs), workers=processes)
    with ProcessPoolExecutor(max_workers=processes, mp_context=get_context("spawn")) as pool:
        return list(pool.map(run_one_to_store, specs, itertools.repeat(root)))
