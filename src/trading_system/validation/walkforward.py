"""Running the system over a fold sequence: IS, select, OOS — and nothing shared between folds.

**Trading is forbidden outside a window by never evaluating it there, not by
sizing a signal and then dropping it.**
:attr:`~trading_system.backtest.config.BacktestConfig.evaluation_start` and
``evaluation_end`` are read by
:meth:`~trading_system.backtest.orchestrator.Orchestrator.on_recognise` before
it calls a single entry engine — see that module's decision. The consequence
used throughout this module: **every trade an out-of-sample run reports opened
inside that fold's own OOS window**, because nothing outside
``[trade_start, trade_end)`` was ever offered the chance to open one. Attributing
a boundary-crossing trade to "the fold that opened it" therefore costs nothing
extra here — it is what :attr:`~trading_system.backtest.orchestrator.BacktestResult.trades`
already is.

**Draining, not force-closing.** A position opened just before ``trade_end``
does not vanish at the boundary and does not get an invented liquidation fill.
The OOS run's own data extends ``max_drain_bars`` past ``trade_end`` on every
stream, and ``evaluation_end`` only stops *new* entries there — existing
positions keep being managed by their own exit rules on real subsequent bars.
A position that still has not closed once that extension runs out is counted
under ``open_at_end`` on the stored run, which this module's report reads as
"drain truncated".

**``boundary_residual`` is a bookkeeping identity, not an estimate.** For a
window ``[trade_start, trade_end]``::

    boundary_residual = equity(trade_end) - equity(trade_start)
                         - sum(trade.net for trades closed strictly inside the window)

``equity`` moves only through booked legs, commission, financing and
mark-to-market (see :mod:`trading_system.backtest.portfolio`), so this holds
to the cent by construction — it does not need the drain to balance. What the
drain buys is a residual that is *small*: without it, every position still
open at ``trade_end`` would sit in the residual as pure unrealised mark for
the rest of that fold's life; with it, a boundary trade that goes on to close
during the drain settles into the *next* window's accounting the moment it
closes, since ``trades closed strictly inside the window`` only ever means
``trade_start < closed_at <= trade_end`` for THIS window — a trade closing
during the drain is outside every window this fold reports, and its full
``net`` is simply visible in ``result.trades`` for whoever reads the stored
run, same as any other trade this fold opened.

**Equity resets between folds because nothing is shared between folds.** Each
fold's IS and OOS run is an independent :class:`~trading_system.backtest.spec.RunInputs`,
which means an independent :class:`~trading_system.risk.engine.RiskEngine` and
an independent :class:`~trading_system.risk.circuit_breakers.CircuitBreakers` —
a breaker tripped in fold 3 cannot silently mute fold 4, because fold 4 never
touches fold 3's breaker instance. Nothing here has to reset anything; the
independence is what building each fold as its own ``RunInputs`` already gives.
"""

import json
import time
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

from trading_system.backtest.clock import StreamKey
from trading_system.backtest.parallel import (
    StoredRun,
    run_one_to_store,
    run_parallel_to_store,
    run_sequential_to_store,
)
from trading_system.backtest.reproducibility import CodeVersion, code_version, digest
from trading_system.backtest.spec import RunInputs
from trading_system.core.logging import get_logger
from trading_system.data.models import OHLCVFrame
from trading_system.validation.objective import Objective
from trading_system.validation.optimization import (
    FoldOptimization,
    ISWindowView,
    ParameterSearch,
    SearchSpace,
    TrialLedger,
    TrialRunner,
    read_fold_selection,
    summarise,
    write_fold_optimization,
)
from trading_system.validation.splitting import (
    Fold,
    FoldWindow,
    PurgedKFold,
    WalkForwardSplitter,
)

logger = get_logger(__name__)

#: Filename of a walk-forward's own manifest, next to (but distinct from) the
#: per-fold runs it references by id, which continue to live in ``runs/``.
WF_MANIFEST_FILE = "manifest.json"


class ParameterSelector(Protocol):
    """Chooses OOS parameters from a fold and its IS result, and can name itself.

    Stage 1 fixed this protocol with :meth:`select` alone, so that stage 2's
    optimiser would plug in without touching :class:`WalkForwardRunner`. The
    selection half of that held exactly as intended — :class:`OptimizingSelector`
    implements ``select`` and the runner is unchanged. :meth:`key` is the half
    that did not, and it was added because the omission was a real defect rather
    than an aesthetic one.

    **What went wrong.** ``wf_id`` was digested from the base inputs, the
    splitter and the resolved fold list, on the reasoning that those are what a
    walk-forward is. With :class:`IdentitySelector` the only implementation, the
    selector was a constant and left no fingerprint to record. It is not a
    constant any more: an optimising and a non-optimising walk-forward over the
    same history and the same fold geometry produce *different out-of-sample
    runs* and were nevertheless landing on the same ``wf_id``. Because
    :meth:`WalkForwardRunner.run` is idempotent on that id, the second of the
    two silently returned the first one's folds without running anything — and
    a report built from it described the wrong run while looking entirely
    normal. This was caught by running the optimiser and then the stage 1
    baseline back to back and noticing the identical id, not by reasoning about
    it.

    This is the same class of mistake
    :meth:`~trading_system.backtest.spec.RunInputs.components` exists to prevent
    one level down — a field that varies the result but not the identity — and
    it is fixed the same way: identity is assembled from a method the varying
    object owns, so adding a selector means answering "what names you?" rather
    than remembering to edit a digest somewhere else.
    """

    def key(self) -> str:
        """Identify this selector's configuration for ``wf_id``.

        Returns:
            A string that differs whenever this selector would produce
            different out-of-sample parameters. Two selectors that would choose
            identically may share one.
        """
        ...

    def select(self, fold: Fold, is_result: StoredRun) -> RunInputs:
        """Choose the parameters an OOS run uses.

        Args:
            fold: The fold being processed.
            is_result: The in-sample run's identity and counters. Its curve
                and trades, if a selector needs them, are read explicitly from
                ``is_result.path`` — the same discipline
                :func:`~trading_system.backtest.parallel.run_parallel_to_store`
                imposes on every other caller.

        Returns:
            The inputs to run out-of-sample. Its own ``streams`` and
            ``config.evaluation_start``/``evaluation_end`` are overwritten by
            :class:`WalkForwardRunner` regardless of what is returned here —
            a selector chooses *parameters*, not the window.
        """
        ...


@dataclass(frozen=True)
class IdentitySelector:
    """The only selector this stage has: the base parameters, unchanged.

    Attributes:
        base: What every fold's OOS run uses, ignoring the IS result entirely.
            No optimisation exists yet to do otherwise.
    """

    base: RunInputs

    def key(self) -> str:
        """``"identity"`` — this selector has no configuration to vary."""
        return "identity"

    def select(self, fold: Fold, is_result: StoredRun) -> RunInputs:  # noqa: ARG002
        """Return :attr:`base`, regardless of ``fold`` or ``is_result``."""
        return self.base


@dataclass(frozen=True)
class OptimizingSelector:
    """Searches a parameter space on each fold's in-sample window, and only there.

    **The protocol did not have to change to accommodate this**, which is worth
    recording: stage 1 fixed :class:`ParameterSelector` as "called once per fold
    with the fold and the IS run, returns the inputs to run out-of-sample", and
    an optimiser fits that signature without amendment. The stage 1 note that a
    forced signature change would signal an incompletely designed protocol
    therefore resolves the other way.

    **How out-of-sample invisibility is actually obtained.** This object does
    hold :attr:`base` with full-coverage streams — it must, because the runner
    slices the OOS window out of whatever is returned. What the *search* gets is
    never that. :meth:`select` builds an
    :class:`~trading_system.validation.optimization.ISWindowView`, whose
    constructor refuses to hold a bar at or after the in-sample ``trade_end``,
    and a :class:`~trading_system.validation.optimization.TrialRunner`, whose
    constructor refuses a run template carrying any streams but the view's. Two
    types declining to exist, rather than a rule someone has to keep. The OOS
    window's *datetime* stays visible — the runner needs it — and no claim is
    made otherwise; what is structurally absent is any bar from it.

    Attributes:
        base: The run every fold varies from. Its bindings supply the strategy
            the space is written against; its streams are never handed to a
            search.
        space: The parameters that may vary.
        search: How the space is visited.
        objective: How one walked run is scored.
        trial_budget: Parameter sets each fold may evaluate. **Per fold**, and
            never carried forward: a fold that spends less does not lend its
            remainder to the next, which would give later folds more attempts
            at the same problem for no reason connected to the data.
        store_root: Where per-fold trial tables and selections are written,
            under ``optimization/{key}/fold_{i}/``.
        cv: Cross-validation inside each in-sample window, or ``None`` to score
            each trial over the whole window at once.
        min_cv_test_span: Shortest cross-validation piece worth scoring. A fold
            whose window cannot yield ``k`` pieces this long is optimised
            directly over its whole in-sample window and says so in its record
            — never silently at a reduced ``k``.
        tolerance_sigmas: Plateau membership margin, in standard deviations of
            the trial scores.
        penalty_weight: How hard neighbourhood instability is penalised when
            ranking. See
            :func:`~trading_system.validation.objective.analyse_plateau`.
        ledger: Trial and run counts per fold. Mutable, and written to disk with
            every fold's selection so that a crashed walk-forward resumes with
            its history rather than restarting a count.
    """

    base: RunInputs
    space: SearchSpace
    search: ParameterSearch
    objective: Objective
    trial_budget: int
    store_root: Path
    cv: PurgedKFold | None = None
    min_cv_test_span: timedelta = timedelta(days=30)
    tolerance_sigmas: float = 1.0
    penalty_weight: float = 0.5
    ledger: TrialLedger = field(default_factory=TrialLedger)
    _outcomes: dict[int, FoldOptimization] = field(default_factory=dict, repr=False)
    _key: list[str] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        """Validate the budget.

        Raises:
            ValueError: If ``trial_budget`` is not positive.
        """
        if self.trial_budget < 1:
            raise ValueError(f"trial_budget must be at least 1, got {self.trial_budget}")

    @property
    def outcomes(self) -> Mapping[int, FoldOptimization]:
        """Each fold's optimisation record, by fold index, for folds run in this process."""
        return dict(self._outcomes)

    def key(self) -> str:
        """``"optimize:"`` plus :meth:`search_key` — what distinguishes this selector's choices."""
        return f"optimize:{self.search_key()}"

    def search_key(self) -> str:
        """Identifies this optimiser's configuration, naming the directory it writes under.

        Digested from the base run's own manifest and every knob that changes
        which points get visited, so that two different searches over the same
        history cannot land in one directory and read back each other's
        selections.
        """
        if not self._key:
            self._key.append(
                digest(
                    {
                        "base": self.base.manifest().to_dict(),
                        "space": self.space.model_dump(mode="json"),
                        "method": self.search.name,
                        "search": repr(self.search),
                        "budget": self.trial_budget,
                        "cv": None
                        if self.cv is None
                        else {
                            "k": self.cv.k,
                            "embargo_s": self.cv.embargo.total_seconds(),
                            "label_span_s": self.cv.label_span.total_seconds(),
                        },
                        "min_cv_test_span_s": self.min_cv_test_span.total_seconds(),
                        "tolerance_sigmas": self.tolerance_sigmas,
                        "penalty_weight": self.penalty_weight,
                    }
                )
            )
        return self._key[0]

    def fold_dir(self, fold_index: int) -> Path:
        """Where one fold's trial table and selection live."""
        return self.store_root / "optimization" / self.search_key() / f"fold_{fold_index}"

    def _pieces(
        self, fold: Fold
    ) -> tuple[tuple[tuple[datetime, datetime], ...] | None, str | None]:
        """The cross-validation test pieces for a fold, or ``None`` with the reason why not."""
        if self.cv is None:
            return None, "cross-validation not configured"
        reason = self.cv.unusable_reason(fold.is_window, min_test_span=self.min_cv_test_span)
        if reason is not None:
            return None, reason
        return tuple(piece.test for piece in self.cv.split(fold.is_window)), None

    def select(self, fold: Fold, is_result: StoredRun) -> RunInputs:  # noqa: ARG002
        """Search this fold's in-sample window and return the winning parameters.

        Idempotent per fold: a fold whose selection has already been written is
        read back rather than re-searched, so a walk-forward interrupted partway
        resumes instead of respending thousands of backtests. The same trust
        :meth:`WalkForwardRunner.run` places in a ``wf_id`` it already holds.

        Args:
            fold: The fold being processed.
            is_result: The in-sample run of the *base* parameters — a natural
                baseline to compare the search against, and not read here.

        Returns:
            :attr:`base` with the selected parameters written into its single
            strategy binding. Full-coverage streams, because
            :meth:`WalkForwardRunner._oos_inputs` slices the OOS window out of
            what this returns.
        """
        directory = self.fold_dir(fold.index)
        pieces, skip_reason = self._pieces(fold)

        stored = read_fold_selection(directory)
        if stored is not None:
            params = self.space.point(stored["selected_coords"])
            self.ledger.record(
                fold.index, trials=int(stored["n_trials"]), runs=int(stored["n_runs"])
            )
            logger.info(
                "optimize.fold_cached", fold=fold.index, selected=str(params), path=str(directory)
            )
            return self._with_params(params)

        view = ISWindowView.build(self.base.streams, fold.is_window)
        runner = TrialRunner(
            view=view,
            template=replace(self.base, streams=view.streams),
            space=self.space,
            objective=self.objective,
            pieces=pieces,
        )
        records = self.search.run(self.space, runner.evaluate, self.trial_budget)
        outcome = summarise(
            fold.index,
            self.search,
            records,
            budget=self.trial_budget,
            runs_per_trial=runner.runs_per_trial,
            cv_applied=pieces is not None,
            cv_skip_reason=skip_reason,
            cv_k=None if pieces is None else len(pieces),
            directory=directory,
            tolerance_sigmas=self.tolerance_sigmas,
            penalty_weight=self.penalty_weight,
        )
        write_fold_optimization(directory, outcome, records)
        self.ledger.record(fold.index, trials=outcome.n_trials, runs=outcome.n_runs)
        self._outcomes[fold.index] = outcome
        logger.info(
            "optimize.fold_done",
            fold=fold.index,
            method=outcome.method,
            trials=outcome.n_trials,
            runs=outcome.n_runs,
            selected=str(outcome.selected),
            selected_score=outcome.selected_score,
            best=str(outcome.best),
            best_score=outcome.best_score,
            plateau_size=None if outcome.plateau is None else outcome.plateau.plateau_size,
            cv_applied=outcome.cv_applied,
        )
        return self._with_params(outcome.selected)

    def _with_params(self, params: Any) -> RunInputs:
        """:attr:`base` with ``params`` written into its single strategy binding."""
        binding = self.base.bindings[0]
        varied = replace(binding, spec=self.space.apply(binding.spec, params))
        return replace(self.base, bindings=(varied,))


@dataclass(frozen=True)
class FoldRun:
    """One fold's two runs, stored.

    Attributes:
        fold: The fold.
        is_run: The in-sample run's identity and counters.
        oos_run: The out-of-sample run's identity and counters — its data
            extends past ``fold.oos_window.trade_end`` for the drain; see the
            module docstring.
    """

    fold: Fold
    is_run: StoredRun
    oos_run: StoredRun


@dataclass(frozen=True)
class WalkForwardResult:
    """A finished walk-forward: its identity and every fold's two runs.

    Attributes:
        wf_id: Identifies this walk-forward — the base inputs, the splitter's
            configuration and the resolved fold list, digested together. A
            second ``run()`` with the same id is a no-op over the same folds.
        folds: Each fold's runs, oldest first.
        store_root: Where the per-fold runs and this walk-forward's own
            manifest live.
    """

    wf_id: str
    folds: tuple[FoldRun, ...]
    store_root: Path

    @property
    def manifest_path(self) -> Path:
        """Where this walk-forward's own manifest was written."""
        return self.store_root / "walkforward" / self.wf_id / WF_MANIFEST_FILE


def _coverage(streams: Mapping[StreamKey, OHLCVFrame]) -> tuple[datetime, datetime]:
    """The time range every stream in a run actually covers.

    Args:
        streams: Bars per stream.

    Returns:
        ``(latest start, earliest end)`` across every stream — the
        intersection, so that no fold is ever built over a range one of the
        streams has no bars for at all.

    Raises:
        ValueError: If ``streams`` is empty, or any stream is empty.
    """
    if not streams:
        raise ValueError("cannot derive coverage from zero streams")
    starts: list[datetime] = []
    ends: list[datetime] = []
    for key, frame in streams.items():
        if frame.start is None or frame.end is None:
            raise ValueError(f"stream {key} has no bars; cannot contribute to coverage")
        starts.append(frame.start)
        ends.append(frame.end)
    return max(starts), min(ends)


def _slice_streams(
    streams: Mapping[StreamKey, OHLCVFrame], start: datetime, end: datetime
) -> dict[StreamKey, OHLCVFrame]:
    """Every stream, restricted to one shared ``[start, end)`` window."""
    return {key: frame.slice(start, end) for key, frame in streams.items()}


def _drained_end(
    streams: Mapping[StreamKey, OHLCVFrame], trade_end: datetime, max_drain_bars: int
) -> dict[StreamKey, OHLCVFrame]:
    """Slice every stream from nothing up to ``trade_end`` plus its own drain allowance.

    Each stream's own timeframe decides how much calendar time
    ``max_drain_bars`` actually buys it — a coarser stream gets a longer
    extension for the same bar count, which is the point: the drain is a bar
    budget, not a calendar one.
    """
    return {
        key: frame.slice(None, trade_end + key.timeframe.duration * max_drain_bars)
        for key, frame in streams.items()
    }


@dataclass(frozen=True)
class WalkForwardRunner:
    """Runs a walk-forward: split, then IS, select, OOS per fold, stored idempotently.

    Attributes:
        base: The run every fold varies from — its streams stand for the whole
            coverage; :meth:`run` slices them per fold and per window.
        splitter: Cuts the base streams' coverage into folds.
        selector: Chooses each fold's OOS parameters from its IS result.
        store_root: Where per-fold runs and this walk-forward's own manifest
            are written.
        max_drain_bars: How many bars past each OOS window's ``trade_end`` a
            position already open may still be managed on before it is
            counted as drain-truncated. An explicit number rather than one
            derived from an exit preset's own time-stop: a run can bind
            several strategies with several presets, and a run wide number
            beats a fragile guess at which preset's stop should govern the
            fold boundary.
        parallel_threshold_seconds: Below this many seconds per run, folds run
            sequentially — a process spawn plus P13's own measured pipe cost
            is not worth paying for a batch that finishes before a worker
            pool would even be up. Decided from the *first* run of each batch,
            timed: with an unknown strategy and an unknown data volume, a
            static estimate would either always guess parallel (paying spawn
            cost on a batch of fast folds) or always guess sequential (giving
            up the same 2x P13 measured on a batch of slow ones).
        workers: Worker count for a parallel batch. The machine's CPU count by
            default.
    """

    base: RunInputs
    splitter: WalkForwardSplitter
    selector: ParameterSelector
    store_root: Path
    max_drain_bars: int
    parallel_threshold_seconds: float = 2.0
    workers: int | None = None

    def __post_init__(self) -> None:
        """Validate the drain allowance.

        Raises:
            ValueError: If ``max_drain_bars`` is not positive — zero would
                make every boundary position drain-truncated immediately,
                which is indistinguishable from not draining at all and would
                silently defeat the point of this field.
        """
        if self.max_drain_bars <= 0:
            raise ValueError(f"max_drain_bars must be positive, got {self.max_drain_bars}")

    def run(self) -> WalkForwardResult:
        """Split the base run's coverage and walk every fold: IS, select, OOS.

        Idempotent on ``wf_id``: if this walk-forward has already been stored
        under the id these inputs and this fold sequence produce, the stored
        manifest is read back and **no fold is re-run** — not even to confirm
        it reproduces, the same trust :func:`~trading_system.backtest.reproducibility.write_run`
        places in an id it already holds.

        Returns:
            The result, freshly computed or read back.
        """
        code = code_version()
        coverage = _coverage(self.base.streams)
        folds = self.splitter.split(coverage, day_origin=self.base.config.day_origin)
        wf_id = digest(
            _manifest_payload(self.base, self.splitter, folds, code, self.selector.key())
        )
        wf_dir = self.store_root / "walkforward" / wf_id
        manifest_path = wf_dir / WF_MANIFEST_FILE
        if manifest_path.exists():
            return read_result(manifest_path, wf_id, self.store_root)

        is_specs = [self._is_inputs(fold) for fold in folds]
        is_runs = self._run_batch(is_specs)

        oos_specs = [
            self._oos_inputs(fold, self.selector.select(fold, is_run))
            for fold, is_run in zip(folds, is_runs, strict=True)
        ]
        oos_runs = self._run_batch(oos_specs)

        fold_runs = tuple(
            FoldRun(fold=fold, is_run=is_run, oos_run=oos_run)
            for fold, is_run, oos_run in zip(folds, is_runs, oos_runs, strict=True)
        )
        result = WalkForwardResult(wf_id=wf_id, folds=fold_runs, store_root=self.store_root)
        _write_manifest(manifest_path, result)
        return result

    def _is_inputs(self, fold: Fold) -> RunInputs:
        """The in-sample run: sliced to the fold's own IS window, no drain."""
        window = fold.is_window
        config = self.base.config.model_copy(
            update={"evaluation_start": window.trade_start, "evaluation_end": None}
        )
        streams = _slice_streams(self.base.streams, window.data_start, window.trade_end)
        return replace(self.base, config=config, streams=streams)

    def _oos_inputs(self, fold: Fold, selected: RunInputs) -> RunInputs:
        """The out-of-sample run: selected parameters, sliced with the drain allowance.

        The upper bound comes from :func:`_drained_end` (``trade_end`` plus
        each stream's own drain allowance); only the lower bound is applied
        here, on top of it, so the two slices compose rather than one
        undoing the other.
        """
        window = fold.oos_window
        config = selected.config.model_copy(
            update={"evaluation_start": window.trade_start, "evaluation_end": window.trade_end}
        )
        drained = _drained_end(selected.streams, window.trade_end, self.max_drain_bars)
        streams = {key: frame.slice(window.data_start, None) for key, frame in drained.items()}
        return replace(selected, config=config, streams=streams)

    def _run_batch(self, specs: list[RunInputs]) -> list[StoredRun]:
        """Run a batch of independent inputs, sequentially or in parallel.

        Times the first run and decides the rest from
        :attr:`parallel_threshold_seconds` — see the class docstring.
        """
        if not specs:
            return []
        started = time.perf_counter()
        first = run_one_to_store(specs[0], self.store_root)
        elapsed = time.perf_counter() - started
        rest = specs[1:]
        if not rest:
            return [first]
        if elapsed >= self.parallel_threshold_seconds:
            rest_runs = run_parallel_to_store(rest, self.store_root, workers=self.workers)
        else:
            rest_runs = run_sequential_to_store(rest, self.store_root)
        return [first, *rest_runs]


def _manifest_payload(
    base: RunInputs,
    splitter: WalkForwardSplitter,
    folds: list[Fold],
    code: CodeVersion,
    selector_key: str,
) -> dict[str, Any]:
    """JSON-safe description of everything ``wf_id`` must depend on.

    Built by hand rather than handed to :func:`~trading_system.backtest.reproducibility.canonical`
    directly: that function has no case for :class:`~datetime.timedelta` and
    would describe every one alike through its generic object fallback,
    silently making two splitters with different spans hash the same.

    ``selector_key`` is here because leaving it out was a defect, not a
    simplification — see :class:`ParameterSelector` on the two walk-forwards
    that shared an id and therefore shared a result.
    """
    return {
        "base": base.manifest(code=code).to_dict(),
        "selector": selector_key,
        "splitter": {
            "mode": splitter.mode.value,
            "is_span_s": splitter.is_span.total_seconds(),
            "oos_span_s": splitter.oos_span.total_seconds(),
            "step_s": splitter.step.total_seconds(),
            "embargo_s": splitter.embargo.total_seconds(),
            "warmup_s": splitter.warmup.total_seconds(),
        },
        "folds": [_fold_payload(fold) for fold in folds],
        "source_digest": code.source_digest,
    }


def _fold_payload(fold: Fold) -> dict[str, Any]:
    """JSON-safe description of one fold's windows."""

    def window(w: FoldWindow) -> dict[str, str]:
        return {
            "data_start": w.data_start.isoformat(),
            "trade_start": w.trade_start.isoformat(),
            "trade_end": w.trade_end.isoformat(),
        }

    return {
        "index": fold.index,
        "is_window": window(fold.is_window),
        "oos_window": window(fold.oos_window),
        "embargo_s": fold.embargo.total_seconds(),
    }


def _write_manifest(path: Path, result: WalkForwardResult) -> None:
    """Persist a walk-forward's own manifest, referencing folds by run id."""
    payload = {
        "wf_id": result.wf_id,
        "folds": [
            {
                "index": fold_run.fold.index,
                "is_window": _fold_payload(fold_run.fold)["is_window"],
                "oos_window": _fold_payload(fold_run.fold)["oos_window"],
                "embargo_s": fold_run.fold.embargo.total_seconds(),
                "is_run": _stored_run_payload(fold_run.is_run),
                "oos_run": _stored_run_payload(fold_run.oos_run),
            }
            for fold_run in result.folds
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))


def _stored_run_payload(run: StoredRun) -> dict[str, Any]:
    """JSON-safe description of one :class:`~trading_system.backtest.parallel.StoredRun`."""
    return {
        "run_id": run.run_id,
        "digest": run.digest,
        "path": str(run.path),
        "counters": run.counters,
    }


def read_result(path: Path, wf_id: str, store_root: Path) -> WalkForwardResult:
    """Rebuild a :class:`WalkForwardResult` from a previously written manifest.

    Public because grading happens after the fact: the command that computes a
    verdict is not the one that ran the folds, and re-running them to grade
    them would defeat the idempotence :meth:`WalkForwardRunner.run` exists for.

    Args:
        path: The walk-forward's ``manifest.json``.
        wf_id: The id that manifest belongs to.
        store_root: Where the individual fold runs are stored.

    Returns:
        The result, with every fold's stored run attached.
    """
    payload = json.loads(path.read_text())

    def window(raw: Mapping[str, str]) -> FoldWindow:
        return FoldWindow(
            data_start=datetime.fromisoformat(raw["data_start"]),
            trade_start=datetime.fromisoformat(raw["trade_start"]),
            trade_end=datetime.fromisoformat(raw["trade_end"]),
        )

    def stored_run(raw: Mapping[str, Any]) -> StoredRun:
        return StoredRun(
            run_id=raw["run_id"],
            digest=raw["digest"],
            path=Path(raw["path"]),
            counters=raw["counters"],
        )

    folds = tuple(
        FoldRun(
            fold=Fold(
                index=raw["index"],
                is_window=window(raw["is_window"]),
                oos_window=window(raw["oos_window"]),
                embargo=timedelta(seconds=raw["embargo_s"]),
            ),
            is_run=stored_run(raw["is_run"]),
            oos_run=stored_run(raw["oos_run"]),
        )
        for raw in payload["folds"]
    )
    return WalkForwardResult(wf_id=wf_id, folds=folds, store_root=store_root)
