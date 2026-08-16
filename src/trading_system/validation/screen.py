"""Screening a shelf of strategies over the pre-holdout history, cheaply and durably.

What this is for, stated once so no consumer has to infer it: **the screen
produces hypotheses, never evidence.** Every number it emits is in-sample, on
each strategy's own parameters, over one window, with no folds, no parameter
selection and no null. Its output is a shortlist worth the cost of a
walk-forward, and the multiple-comparison arithmetic in
:mod:`trading_system.validation.trials` exists to say how much less than that
even the shortlist means.

Three things about the mechanism come from decisions already paid for elsewhere:

* **The holdout is cut before anything runs, structurally.** A worker builds its
  inputs through :func:`build_screen_inputs`, which slices every series at the
  boundary; the frame the run walks does not contain a holdout bar. Nothing here
  is asked to refrain from reading them.
* **Workers are handed paths and return numbers.** P13 measured a
  ``BacktestResult`` at ~8 MB through a pipe against 1.8 s of work in it, and
  P15 stage 1.5 measured what thousands of run directories cost when each exists
  for one number. A task is a few strings; a row is a few floats plus the
  result's digest, which is what makes the second pass — re-running the
  survivors to store them in full — a verifiable reconstruction rather than a
  new measurement.
* **Rows are appended to a durable log as they arrive.** The first attempt at
  this sweep kept 2 400 results in the parent's memory and lost all of them to
  one interruption. The log is JSON lines because appending a line is the only
  write that survives being killed mid-way; ``rows.parquet`` is materialised
  from it at the end and on demand, so the analysis artefact is still a single
  parquet as the contract asks.
"""

import json
import time
import traceback
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field, fields, replace
from datetime import UTC, datetime
from decimal import Decimal
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import polars as pl

from trading_system.analytics.metrics import (
    daily_curve,
    drawdown_stats,
    sharpe_daily,
    simple_returns,
    sortino_daily,
    total_return,
    trade_stats,
)
from trading_system.backtest.clock import StreamKey
from trading_system.backtest.config import BacktestConfig
from trading_system.backtest.fx import build_converter
from trading_system.backtest.orchestrator import StrategyBinding
from trading_system.backtest.reproducibility import (
    code_version,
    digest,
    result_digest,
    write_run,
)
from trading_system.backtest.spec import RunInputs
from trading_system.core.instruments import load_instruments
from trading_system.core.logging import setup_logging
from trading_system.core.types import Timeframe
from trading_system.data.store import ParquetStore
from trading_system.execution.config import CostConfig
from trading_system.exit.library import ExitLibrarySpec
from trading_system.risk.sizing.methods import FixedFractional
from trading_system.strategies.normalize.coverage import MarketCoverage
from trading_system.strategies.schema import StrategySpec
from trading_system.validation.holdout import (
    DEFAULT_HOLDOUT_FRACTION,
    HoldoutBoundary,
    screen_frame,
)

#: Bars a task runs over, counted back from the holdout boundary. A fixed bar
#: budget rather than a calendar span, so every strategy gets the same
#: statistical weight and the same compute; the span it buys differs by an order
#: of magnitude between D1 and M1, and the report says so.
DEFAULT_BAR_BUDGET = 20_000

#: Where a screen's artefacts live.
SCREEN_ROOT = "sweep"

#: Durable append-only log of rows, written as they arrive.
ROWS_LOG = "rows.jsonl"

#: The same rows as one parquet, materialised from the log.
ROWS_PARQUET = "rows.parquet"

#: Daily returns of the sampled tasks, for the effective-trials estimate.
RETURNS_PARQUET = "returns.parquet"

#: The screen's own description: what was run, where the holdout was cut, and
#: how many independent trials the run is worth.
MANIFEST = "manifest.json"


@dataclass(frozen=True)
class ScreenTask:
    """One strategy on one instrument, over the pre-holdout history.

    Attributes:
        spec_path: Strategy spec to load. A path, not the spec: what crosses
            into a worker should be small.
        symbol: Instrument to trade.
        bar_budget: Bars from the holdout boundary backwards to run over.
        holdout_fraction: Share of the series withheld from the screen.
        risk_pct: Fraction of equity risked per trade.
        run_seed: Seed for the per-fill random streams.
        cost_ratio: Round-turn cost as a share of the median bar's range on this
            series. Carried on the task so the row can state the cost regime it
            was produced in — a row is not comparable with another without it.
        keep_returns: Whether to bring back the daily return vector. True for
            the sample the effective-trials estimate is built from, false for
            everything else, because those vectors are what a compact row exists
            to avoid carrying.
        keep_run_max_dd: Write the whole run when its maximum drawdown is
            below this fraction; ``None`` writes none on that account. Measured
            and **not recommended as a selection**: on the delivered screen the
            rows under 10% were positive at 42.2% against a base rate of 42.0%,
            with a *worse* median expectancy and a third of the trades. The
            threshold selects strategies that barely traded, and it moves with
            ``risk_pct`` — at 0.2% risk three quarters of the shelf would pass.
        keep_run: Write the whole run unconditionally. What the second pass
            uses: by then the selection has already happened on the rows, and
            re-running a chosen task to store it is a reconstruction the row's
            ``result_digest`` can verify.
    """

    spec_path: Path
    symbol: str
    bar_budget: int = DEFAULT_BAR_BUDGET
    holdout_fraction: float = DEFAULT_HOLDOUT_FRACTION
    risk_pct: float = 0.01
    run_seed: int = 0
    cost_ratio: float = 0.0
    keep_returns: bool = False
    keep_run_max_dd: float | None = None
    keep_run: bool = False

    @property
    def key(self) -> str:
        """Identity of this task within a screen, for resuming."""
        return f"{self.spec_path.stem}@{self.symbol}"


@dataclass(frozen=True)
class ScreenRow:
    """What one task produced.

    Attributes:
        key: The task's identity.
        spec_id: The strategy's id.
        symbol: Instrument traded.
        timeframe: Bar size traded.
        bars: Bars actually run over.
        start: First bar's open time.
        end: Last bar's open time — always strictly before the holdout.
        cost_ratio: Round-turn cost over the median bar's range.
        trades: Closed trades.
        fills: Fills, entries and exits together.
        expired_orders: Resting entries that timed out.
        open_at_end: Positions still open when the window ran out.
        expectancy_r: Mean result per trade in R.
        winrate: Fraction of trades closed above zero.
        profit_factor: Gross win over gross loss.
        total_return: Change in equity over the window, as a fraction.
        sharpe: Annualised Sharpe of the daily curve.
        sortino: Annualised Sortino of the daily curve.
        max_drawdown_pct: Deepest peak-to-trough fall, as a fraction.
        dominant_reason: The refusal or drop that fired most.
        dominant_count: How many times it fired.
        kept_run: Whether the whole run was written to the store, which happens
            only when the drawdown filter admits it.
        result_digest: Digest of the full result, so the second pass can prove
            it reconstructed this row rather than measured a new one.
        seconds: Wall time the run took.
        error: The exception that stopped it, or ``None``.
        daily_returns: Daily simple returns, for sampled tasks only.
    """

    key: str
    spec_id: str
    symbol: str
    timeframe: str
    bars: int
    start: str | None
    end: str | None
    cost_ratio: float
    trades: int
    fills: int
    expired_orders: int
    open_at_end: int
    expectancy_r: float | None
    winrate: float | None
    profit_factor: float | None
    total_return: float | None
    sharpe: float | None
    sortino: float | None
    max_drawdown_pct: float | None
    dominant_reason: str
    dominant_count: int
    kept_run: bool
    result_digest: str | None
    seconds: float
    error: str | None = None
    daily_returns: tuple[float, ...] = ()

    @classmethod
    def from_record(cls, payload: Mapping[str, Any]) -> "ScreenRow":
        """Build a row from a stored record, tolerating fields added since.

        ``rows.parquet`` is a derived table, not a run digest: it is rebuilt
        from the log whenever anything reads it, and a column added later must
        not make an earlier screen unreadable. That tolerance is deliberate and
        does **not** extend to stored runs, where
        :func:`~trading_system.backtest.reproducibility.read_run` refuses a
        changed schema outright — there the digest is the promise.

        Args:
            payload: The record as read from parquet or the JSON log.

        Returns:
            The row, with absent fields at their defaults.
        """
        known = {field.name for field in fields(cls)}
        supplied = {name: value for name, value in payload.items() if name in known}
        supplied.setdefault("kept_run", False)
        supplied["daily_returns"] = tuple(supplied.get("daily_returns") or ())
        return cls(**supplied)

    def scalars(self) -> dict[str, object]:
        """The row without its return vector, for the tabular artefact."""
        payload = asdict(self)
        payload.pop("daily_returns")
        return payload


def build_screen_inputs(
    task: ScreenTask, data_dir: Path, instruments_path: Path, exit_library: Path
) -> tuple[RunInputs, HoldoutBoundary]:
    """Assemble what one screen task runs, with the holdout already gone.

    Public and separate from :func:`run_task` for one reason: the guarantee that
    a screen cannot see a holdout bar is a property of this object, and a test
    can only assert it by walking this object. A guarantee that can only be
    inspected by reading the code is not the same guarantee.

    Args:
        task: What to run.
        data_dir: Root of the parquet store.
        instruments_path: Instrument registry file.
        exit_library: Exit preset library the spec's ``exit_ref`` is looked up in.

    Returns:
        ``(inputs, boundary)``. Every frame in ``inputs`` ends strictly before
        ``boundary.boundary``.

    Raises:
        ValueError: If the spec names an exit the library does not hold, or the
            store holds no bars for this series.
    """
    spec = StrategySpec.model_validate_json(task.spec_path.read_text(encoding="utf-8"))
    signal_tf: Timeframe = spec.timeframes.signal_tf

    library = ExitLibrarySpec.model_validate_json(exit_library.read_text(encoding="utf-8"))
    preset = next((item for item in library.presets if item.id == spec.exit_ref), None)
    if preset is None:
        raise ValueError(f"exit_ref {spec.exit_ref!r} is not in {exit_library}")

    store = ParquetStore(data_dir)
    whole = store.get(task.symbol, signal_tf)
    if whole.is_empty:
        raise ValueError(f"no bars stored for {task.symbol} {signal_tf.value}")

    visible, boundary = screen_frame(whole, fraction=task.holdout_fraction)
    if visible.is_empty:
        raise ValueError(f"the holdout leaves no bars for {task.symbol} {signal_tf.value}")
    if len(visible) > task.bar_budget:
        cut = visible.timestamps[len(visible) - task.bar_budget]
        visible = visible.slice(cut, None)

    instruments = load_instruments(instruments_path)
    key = StreamKey(task.symbol, signal_tf)

    # The converter reads its own series through the same slice: a conversion
    # rate taken from a holdout bar would be a holdout bar reaching the screen
    # by the side door.
    def load(symbol: str, timeframe: Timeframe) -> object:
        frame = store.get(symbol, timeframe)
        return frame.slice(None, boundary.boundary)

    inputs = RunInputs(
        config=BacktestConfig(account_currency="USD", starting_balance=Decimal(100_000)),
        streams={key: visible},
        bindings=(StrategyBinding(spec=spec, exit_preset=preset, keys=(key,)),),
        instruments=instruments,
        costs=CostConfig(run_seed=task.run_seed),
        sizing=FixedFractional(risk_pct=task.risk_pct),
        converter=build_converter(
            (instruments[task.symbol],),
            account_currency="USD",
            timeframe=signal_tf,
            load=load,  # type: ignore[arg-type]
        ),
    )
    return inputs, boundary


def _failed(
    task: ScreenTask, spec_id: str, timeframe: str, detail: str, seconds: float
) -> ScreenRow:
    """A row standing for a task that did not complete."""
    return ScreenRow(
        key=task.key,
        spec_id=spec_id,
        symbol=task.symbol,
        timeframe=timeframe,
        bars=0,
        start=None,
        end=None,
        cost_ratio=task.cost_ratio,
        trades=0,
        fills=0,
        expired_orders=0,
        open_at_end=0,
        expectancy_r=None,
        winrate=None,
        profit_factor=None,
        total_return=None,
        sharpe=None,
        sortino=None,
        max_drawdown_pct=None,
        dominant_reason="",
        dominant_count=0,
        kept_run=False,
        result_digest=None,
        seconds=seconds,
        error=detail[:300],
    )


def run_task(
    task: ScreenTask,
    data_dir: Path,
    instruments_path: Path,
    exit_library: Path,
    runs_dir: Path | None = None,
) -> ScreenRow:
    """Run one task and reduce it to a row.

    Every failure is caught and returned rather than raised: one malformed spec
    among thousands must not end the screen, and a row naming its exception is
    more useful than a traceback that stops everything else.

    Args:
        task: What to run.
        data_dir: Root of the parquet store.
        instruments_path: Instrument registry file.
        exit_library: Exit preset library.
        runs_dir: Where a run that passes the drawdown filter is written.
            ``None`` keeps nothing, whatever the task asks for.

    Returns:
        The row.
    """
    started = time.monotonic()
    spec_id = task.spec_path.stem
    timeframe = ""
    try:
        inputs, _boundary = build_screen_inputs(task, data_dir, instruments_path, exit_library)
        binding = inputs.bindings[0]
        spec_id = binding.spec.id
        timeframe = binding.spec.timeframes.signal_tf.value
        frame = next(iter(inputs.streams.values()))
        result = inputs.run()
    except Exception as error:  # noqa: BLE001 - a screen reports failures as data
        detail = f"{type(error).__name__}: {error}"
        if not str(error):
            detail = traceback.format_exc().splitlines()[-1]
        return _failed(task, spec_id, timeframe, detail, time.monotonic() - started)

    stats = trade_stats(result.trades) if result.trades else None
    daily = daily_curve(result.curve) if result.curve else None
    sharpe = sortino = returns = None
    drawdown: float | None = None
    vector: tuple[float, ...] = ()
    if daily is not None and len(daily.days) >= 3:
        try:
            sharpe = sharpe_daily(daily).value
            sortino = sortino_daily(daily).value
        except (ValueError, ZeroDivisionError):
            sharpe = sortino = None
        try:
            drawdown = float(drawdown_stats(daily).max_drawdown_pct)
        except ValueError:
            # A curve that never fell below its own peak has no drawdown to
            # summarise; that is a property of the sample, not a failure.
            drawdown = 0.0
        returns = total_return(daily)
        if task.keep_returns:
            vector = tuple(simple_returns(daily))

    kept = False
    admitted = task.keep_run or (
        task.keep_run_max_dd is not None
        and drawdown is not None
        and abs(drawdown) < task.keep_run_max_dd
    )
    if runs_dir is not None and admitted and result.trades:
        # Written from inside the worker, as P15 stage 1 established: the parent
        # receives a row either way, and a run that survives the filter is on
        # disk before anything else can fail.
        write_run(runs_dir, inputs.manifest(), result)
        kept = True

    counters = {
        **{f"drop:{reason}": count for reason, count in result.signal_drops.items()},
        **{f"reject:{reason}": count for reason, count in result.rejections.items()},
    }
    fired = [(count, name) for name, count in counters.items() if count]
    dominant_count, dominant_reason = max(fired) if fired else (0, "")

    return ScreenRow(
        key=task.key,
        spec_id=spec_id,
        symbol=task.symbol,
        timeframe=timeframe,
        bars=len(frame),
        start=frame.start.isoformat() if frame.start else None,
        end=frame.end.isoformat() if frame.end else None,
        cost_ratio=task.cost_ratio,
        trades=len(result.trades),
        fills=result.fills,
        expired_orders=result.expired_orders,
        open_at_end=result.open_at_end,
        expectancy_r=stats.expectancy_r if stats else None,
        winrate=stats.winrate if stats else None,
        profit_factor=stats.profit_factor if stats else None,
        total_return=returns,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown_pct=drawdown,
        dominant_reason=dominant_reason,
        dominant_count=dominant_count,
        kept_run=kept,
        result_digest=result_digest(result),
        seconds=time.monotonic() - started,
        daily_returns=vector,
    )


def _worker(payload: tuple[ScreenTask, Path, Path, Path, Path | None]) -> ScreenRow:
    """Pool entry point: silence the run's own logging, then run it."""
    setup_logging(level="ERROR")
    task, data_dir, instruments_path, exit_library, runs_dir = payload
    return run_task(task, data_dir, instruments_path, exit_library, runs_dir)


class ScreenStore:
    """The durable side of a screen: an append-only log and the parquet from it.

    Appending one JSON line is the only write that survives the process being
    killed part-way, which is what "an interruption costs one task" requires.
    Parquet cannot be appended to without holding the whole file, so it is
    materialised from the log rather than written incrementally — one artefact
    for analysis, one for durability, and the second is the authority.
    """

    def __init__(self, root: Path) -> None:
        """Open the store at ``root``, creating it if absent."""
        self._root = root
        self._root.mkdir(parents=True, exist_ok=True)
        self._log = root / ROWS_LOG

    @property
    def root(self) -> Path:
        """Where the screen's artefacts live."""
        return self._root

    def completed(self) -> set[str]:
        """Task keys already in the log."""
        if not self._log.exists():
            return set()
        found: set[str] = set()
        for line in self._log.read_text(encoding="utf-8").splitlines():
            if line.strip():
                found.add(str(json.loads(line)["key"]))
        return found

    def append(self, row: ScreenRow) -> None:
        """Add one row to the log, flushed before returning."""
        with self._log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(row), default=str) + "\n")
            handle.flush()

    def rows(self) -> tuple[ScreenRow, ...]:
        """Every row in the log, in the order it was written."""
        if not self._log.exists():
            return ()
        found: list[ScreenRow] = []
        for line in self._log.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            found.append(ScreenRow.from_record(json.loads(line)))
        return tuple(found)

    def materialise(self) -> tuple[Path, Path | None]:
        """Write the log out as parquet; return the paths written.

        Returns:
            ``(rows_parquet, returns_parquet)``. The second is ``None`` when no
            task was asked to keep its returns.
        """
        rows = self.rows()
        # `infer_schema_length=None` reads every row before deciding a dtype.
        # The default looks at the first hundred, and on a screen the first
        # hundred rows all have `error=None` — so the column is typed Null and
        # the first real failure, hundreds of rows later, cannot be appended.
        # Losing a finished screen to a dtype inferred from its own good news is
        # exactly the failure the durable log exists to make survivable.
        table = (
            pl.DataFrame([row.scalars() for row in rows], infer_schema_length=None)
            if rows
            else pl.DataFrame()
        )
        rows_path = self._root / ROWS_PARQUET
        table.write_parquet(rows_path)

        sampled = [row for row in rows if row.daily_returns]
        if not sampled:
            return rows_path, None
        returns = pl.DataFrame(
            {
                "key": [row.key for row in sampled],
                "returns": [list(row.daily_returns) for row in sampled],
            }
        )
        returns_path = self._root / RETURNS_PARQUET
        returns.write_parquet(returns_path)
        return rows_path, returns_path


def screen_id(
    spec_paths: Sequence[Path],
    *,
    universe: Sequence[str],
    bar_budget: int,
    holdout_fraction: float,
    risk_pct: float,
    symbols_per_spec: int,
) -> str:
    """Identity of a screen: the same inputs give the same id.

    Built from the code as well as the inputs, for the reason ``run_id`` is
    (P13 stage 2): a tree with uncommitted edits has the same commit as one
    without, so the authority is the source digest.

    Args:
        spec_paths: The strategies swept.
        universe: Symbols offered to them.
        bar_budget: Bars each run gets.
        holdout_fraction: Share of each series withheld.
        risk_pct: Fraction of equity risked per trade.
        symbols_per_spec: Instruments each strategy is run on.

    Returns:
        A hex id.
    """
    specs = digest(sorted(path.name for path in spec_paths))
    return digest(
        {
            "source_digest": code_version().source_digest,
            "specs": specs,
            "n_specs": len(spec_paths),
            "universe": sorted(universe),
            "bar_budget": bar_budget,
            "holdout_fraction": holdout_fraction,
            "risk_pct": risk_pct,
            "symbols_per_spec": symbols_per_spec,
        }
    )[:16]


def plan_tasks(
    spec_paths: Sequence[Path],
    coverage: MarketCoverage,
    *,
    symbols_per_spec: int = 1,
    bar_budget: int = DEFAULT_BAR_BUDGET,
    holdout_fraction: float = DEFAULT_HOLDOUT_FRACTION,
    risk_pct: float = 0.01,
    returns_sample: int = 0,
    sample_seed: int = 0,
    keep_run_max_dd: float | None = None,
) -> tuple[tuple[ScreenTask, ...], dict[str, str]]:
    """Decide what to run: each spec against the instruments it may trade.

    A spec's ``allowed_symbols`` is already what normalisation judged admissible
    on this store. Which of them to spend a run on is ranked by measured cost:
    the symbol whose round turn is the smallest share of its median bar is where
    a strategy has the best chance of showing whatever it has.

    Args:
        spec_paths: Strategy files to screen.
        coverage: Measured store, for the ranking and to skip absent series.
        symbols_per_spec: Instruments each spec is run on; ``0`` means every
            one its universe names and the store holds.
        bar_budget: Bars each run gets.
        holdout_fraction: Share of each series withheld.
        risk_pct: Fraction of equity risked per trade.
        returns_sample: How many tasks bring back their daily returns, for the
            effective-trials estimate. Chosen deterministically by stride so the
            sample spans the whole shelf rather than its alphabetical head.
        sample_seed: Offset into the stride, so a second screen can sample a
            different subset without changing the stride.
        keep_run_max_dd: Drawdown below which a task's whole run is kept on
            disk. Passed to every task; ``None`` keeps none.

    Returns:
        ``(tasks, skipped)``. ``skipped`` maps a spec id to why it produced no
        task at all, which is a different outcome from producing one that failed.
    """
    tasks: list[ScreenTask] = []
    skipped: dict[str, str] = {}
    for path in spec_paths:
        try:
            spec = StrategySpec.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception as error:  # noqa: BLE001 - reported, not raised
            skipped[path.stem] = f"unreadable spec: {type(error).__name__}"
            continue
        timeframe = spec.timeframes.signal_tf
        candidates = [
            (found.cost_ratio, symbol)
            for symbol in spec.instruments.allowed_symbols
            if (found := coverage.get(symbol, timeframe)) is not None
        ]
        if not candidates:
            skipped[spec.id] = (
                f"none of its {len(spec.instruments.allowed_symbols)} symbol(s) are stored "
                f"at {timeframe.value}"
            )
            continue
        candidates.sort()
        chosen = candidates if symbols_per_spec <= 0 else candidates[:symbols_per_spec]
        for ratio, symbol in chosen:
            tasks.append(
                ScreenTask(
                    spec_path=path,
                    symbol=symbol,
                    bar_budget=bar_budget,
                    holdout_fraction=holdout_fraction,
                    risk_pct=risk_pct,
                    cost_ratio=ratio,
                    keep_run_max_dd=keep_run_max_dd,
                )
            )

    if returns_sample > 0 and tasks:
        stride = max(len(tasks) // returns_sample, 1)
        sampled = set(range(sample_seed % max(stride, 1), len(tasks), stride))
        tasks = [
            replace(task, keep_returns=True) if index in sampled else task
            for index, task in enumerate(tasks)
        ]
    return tuple(tasks), skipped


def run_screen(
    tasks: Sequence[ScreenTask],
    store: ScreenStore,
    *,
    data_dir: Path,
    instruments_path: Path,
    exit_library: Path,
    workers: int = 1,
    runs_dir: Path | None = None,
    on_row: Callable[[ScreenRow], None] | None = None,
) -> tuple[ScreenRow, ...]:
    """Run every task not already in the log, appending each row as it lands.

    Args:
        tasks: What to run.
        store: Where rows are appended.
        data_dir: Root of the parquet store.
        instruments_path: Instrument registry file.
        exit_library: Exit preset library.
        workers: Processes to spread the tasks over. One runs them here, which
            is what a short screen should do: a pool costs about a second to
            spawn and these tasks are frequently shorter.
        runs_dir: Where a run passing the drawdown filter is written, from
            inside the worker that produced it.
        on_row: Called with each row as it arrives, for progress.

    Returns:
        Every row in the log afterwards — the ones already there plus the ones
        this call produced.
    """
    done = store.completed()
    pending = [task for task in tasks if task.key not in done]
    payloads = [
        (task, data_dir, instruments_path, exit_library, runs_dir) for task in pending
    ]

    if workers <= 1:
        for payload in payloads:
            row = _worker(payload)
            store.append(row)
            if on_row is not None:
                on_row(row)
        return store.rows()

    # Spawn, not fork: forking a process that has already started polars' thread
    # pool hangs about once a week (CLAUDE.md, P13 stage 2).
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
        for row in pool.map(_worker, payloads, chunksize=1):
            store.append(row)
            if on_row is not None:
                on_row(row)
    return store.rows()


@dataclass(frozen=True)
class ScreenManifest:
    """What a screen was, written beside its rows.

    Attributes:
        screen_id: Identity of the screen.
        generated: When it finished, ISO-8601.
        source_digest: The code that ran it.
        holdout_fraction: Share of every series withheld.
        boundaries: Where each series was cut.
        n_specs: Strategies offered to it.
        n_tasks: Runs planned.
        bar_budget: Bars each run got.
        symbols_per_spec: Instruments each strategy was run on.
        trials: The effective-trials estimate, as
            :meth:`~trading_system.validation.trials.EffectiveTrials.as_record`.
        skipped: Specs that produced no task, by reason.
    """

    screen_id: str
    generated: str
    source_digest: str
    holdout_fraction: float
    boundaries: tuple[HoldoutBoundary, ...]
    n_specs: int
    n_tasks: int
    bar_budget: int
    symbols_per_spec: int
    trials: dict[str, object] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)

    def write(self, root: Path) -> Path:
        """Write the manifest into ``root``."""
        payload = {
            "screen_id": self.screen_id,
            "generated": self.generated,
            "source_digest": self.source_digest,
            "holdout_fraction": self.holdout_fraction,
            "holdout_boundaries": [item.as_record() for item in self.boundaries],
            "n_specs": self.n_specs,
            "n_tasks": self.n_tasks,
            "bar_budget": self.bar_budget,
            "symbols_per_spec": self.symbols_per_spec,
            "trials": self.trials,
            "skipped": self.skipped,
            "what_this_is": (
                "A screen, not evidence. Every figure is in-sample, on each strategy's own "
                "parameters, over one window, with no folds, no selection and no null. The "
                "holdout named above was cut before any of it ran and must stay untouched "
                "until a candidate is confirmed on it."
            ),
        }
        path = root / MANIFEST
        path.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        return path


def now_iso() -> str:
    """The current instant, ISO-8601, UTC."""
    return datetime.now(UTC).isoformat()


def load_manifest(root: Path) -> dict[str, object] | None:
    """Read a screen's manifest, or ``None`` when it has not finished."""
    path = root / MANIFEST
    if not path.is_file():
        return None
    payload: dict[str, object] = json.loads(path.read_text(encoding="utf-8"))
    return payload


def load_corpus_meta(manifest_path: Path) -> dict[str, dict[str, str]]:
    """Read fidelity, family and type per spec from a normalised corpus manifest.

    A row deliberately does not carry these: a row is what a *run* produced, and
    how faithfully its spec was read is a property of the corpus. Joining them
    at report time keeps each file the authority on its own facts.

    Args:
        manifest_path: ``manifest.json`` of a normalised corpus.

    Returns:
        Spec id to its labels. Empty when the file does not exist, so a screen
        of hand-written specs still renders.
    """
    if not manifest_path.is_file():
        return {}
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    found: dict[str, dict[str, str]] = {}
    for card in payload.get("cards", []):
        spec_id = card.get("spec_id")
        if not spec_id:
            continue
        found[spec_id] = {
            "fidelity": card.get("fidelity", "?"),
            "family": card.get("family", "?"),
            "type": card.get("type", "?"),
        }
    return found


def distinct_signatures(
    spec_paths: Sequence[Path], families: Mapping[str, Mapping[str, str]]
) -> int:
    """How many distinguishable ideas a shelf of specs holds.

    A signature is ``(family, the set of indicators the spec reads)``. Two specs
    with the same signature differ only in thresholds and periods, which is a
    difference of parameters rather than of idea — and the effective-trials
    estimate must not credit the shelf with more independence than it has ideas.

    The walk is over the spec's JSON rather than its parsed model: any
    ``"indicator"`` key anywhere in the tree is one the spec reads, whatever
    nests it, and a walk that has to know the schema would miss whichever branch
    somebody adds next.

    Args:
        spec_paths: Strategy files.
        families: Corpus labels by spec id, for the family half of the
            signature. A spec with no label contributes its indicators alone.

    Returns:
        The count of distinct signatures.
    """
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for path in spec_paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        names: set[str] = set()
        stack: list[object] = [payload]
        while stack:
            node = stack.pop()
            if isinstance(node, dict):
                indicator = node.get("indicator")
                if isinstance(indicator, str):
                    names.add(indicator)
                stack.extend(node.values())
            elif isinstance(node, list):
                stack.extend(node)
        spec_id = str(payload.get("id", path.stem))
        family = families.get(spec_id, {}).get("family", "?")
        seen.add((family, tuple(sorted(names))))
    return len(seen)


def cross_sectional_z(rows: Iterable[ScreenRow], *, statistic: str = "sortino") -> dict[str, float]:
    """Standardise each row against the other strategies on its own instrument.

    The comparison unit for the screen, and the reason is that a raw Sortino on
    XAUUSD and on EURUSD are not the same quantity: tail shape, gap risk and the
    cost regime are instrument-specific, and normalising risk by ATR does not
    remove them. The cross-section of several hundred mostly-edgeless strategies
    on one instrument is an empirical reference for that instrument, and it is
    free — the screen is its own reference.

    **What this does not remove, and the report must say so:** a bias shared by
    every strategy on that instrument. If they are all long-biased on a market
    that rose, the whole cross-section shifts and the z-score keeps none of that
    out. It is a screening normaliser. The confirming comparison is the
    percentile against that instrument's own permutation and synthetic nulls.

    Args:
        rows: Completed rows.
        statistic: Which field to standardise.

    Returns:
        Z by task key. A row whose instrument has fewer than three usable
        peers is absent rather than assigned zero: no reference, no score.
    """
    usable = [row for row in rows if row.error is None and getattr(row, statistic) is not None]
    by_symbol: dict[str, list[ScreenRow]] = {}
    for row in usable:
        by_symbol.setdefault(row.symbol, []).append(row)

    scores: dict[str, float] = {}
    for group in by_symbol.values():
        values = [float(getattr(row, statistic)) for row in group]
        if len(values) < 3:
            continue
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        if variance <= 0:
            continue
        deviation = variance**0.5
        for row, value in zip(group, values, strict=True):
            scores[row.key] = (value - mean) / deviation
    return scores
