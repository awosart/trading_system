"""Choosing which screened rows are worth a walk-forward, and running it for them.

The screen ranks; this module decides. Two things about the deciding are load
bearing and neither is a matter of taste.

**The thresholds are frozen before the first walk-forward runs.** They go into a
manifest that the runner reads back rather than recomputes, so a threshold
cannot be nudged after the results are visible. That is not a hypothetical
temptation: the whole reason a screen needs a multiple-comparison discount is
that choosing after looking is what inflates a best-of-many result, and choosing
the *threshold* after looking is the same act one level up.

**The ranking statistic is per-trade, and then the exposure confound is
regressed out.** Measured on the delivered screen: the share of a strategy's
signals lost to ``max_concurrent_positions`` correlates with its cross-sectional
z at **+0.63** on Sortino and **+0.28** on expectancy — and the correlation is
not explained by trade count (partial correlation +0.63 against a raw +0.63).
Sortino is a property of the equity curve and therefore of how much exposure ran
at once; expectancy per trade is not. Ranking on expectancy removes most of it,
and taking the residual against the drop share removes the linear remainder by
construction. What neither removes is stated where it belongs — in the report.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from trading_system.backtest.clock import StreamKey
from trading_system.backtest.config import BacktestConfig
from trading_system.backtest.fx import build_converter
from trading_system.backtest.orchestrator import StrategyBinding
from trading_system.backtest.spec import RunInputs
from trading_system.core.instruments import load_instruments
from trading_system.core.types import Timeframe
from trading_system.data.models import OHLCVFrame
from trading_system.data.store import ParquetStore
from trading_system.execution.config import CostConfig
from trading_system.exit.library import ExitLibrarySpec
from trading_system.risk.sizing.methods import FixedFractional
from trading_system.strategies.schema import StrategySpec
from trading_system.validation.holdout import DEFAULT_HOLDOUT_FRACTION, screen_frame
from trading_system.validation.optimization import (
    Evaluate,
    ParameterSearch,
    ParamSet,
    SearchSpace,
    TrialOutcome,
    TrialRecord,
)
from trading_system.validation.screen import ScreenRow

#: Timeframes excluded from the shortlist by default, and why: on the delivered
#: screen the median expectancy by bar size ran H4 +0.029, H1 -0.008, M15
#: -0.011, M5 -0.041, M1 -0.078 — monotone in the round turn's share of the
#: median bar's range. Spending walk-forwards where the cost model dominates the
#: expectation is a price with no hypothesis behind it.
EXCLUDED_TIMEFRAMES: tuple[str, ...] = ("M1", "M5")

#: Axes struck out of every generated search space before optimisation. Both are
#: risk throttles: they change how much exposure is realised rather than what the
#: entry says, and check 1 of this stage measured the consequence — the share of
#: signals lost to ``max_concurrent_positions`` correlates with cross-sectional z
#: at +0.63 on Sortino. Letting a search tune them would be letting it optimise
#: that confound directly, and it would find the same thing: cap exposure, report
#: a better ratio. The generator emits every candidate axis on purpose (P05 stage
#: 2: the tool enumerates, the human strikes out); with hundreds of candidates
#: there is no human in the loop, so the striking out is written down here.
THROTTLE_AXES: tuple[str, ...] = ("max_concurrent_positions", "cooldown_bars_after_loss")

#: Trades one fold's search may simulate before it stops proposing points. The
#: cap is on *trades*, not on wall time and not on the trial count, and each of
#: those choices is forced:
#:
#: * wall time is not reproducible — a candidate truncated at ten minutes on one
#:   machine and not on another produces different parameters, which breaks the
#:   only guarantee this system has about a stored run;
#: * a trial-count cap is what was already in place (48) and it is precisely
#:   what failed: cost per trial varies about sixtyfold between candidates;
#: * a trade cap tracks the actual cost driver. Profiling established the cycle
#:   is linear in trades, so equal trades is equal compute.
#:
#: It does **not** exclude frequent strategies: a strategy that trades often
#: still gets its folds, its OOS runs and its place in the report — it gets a
#: coarser search, and the coarseness is recorded on the result as a truncation
#: rather than left to be inferred from a small ``n_trials``.
TRADE_BUDGET_PER_FOLD = 15_000

#: Trials a fold always gets, whatever the trade budget says. Truncating to one
#: or two points would leave the selector choosing from a sample too small to
#: have a plateau at all, which is a different failure from an expensive search.
MIN_TRIALS_PER_FOLD = 8

#: In-sample trades a row needs before its ratios are worth optimising against.
#: The verdict's own gate is 100 trades over a whole walk-forward; a row that
#: cannot manage that over the screen's window will not manage it over folds.
MIN_TRADES = 100


@dataclass(frozen=True)
class Candidate:
    """One screened row promoted to a walk-forward.

    Attributes:
        key: The screen row's key.
        spec_id: Strategy id.
        spec_path: Where the spec lives.
        symbol: Instrument.
        timeframe: Bar size.
        trades: In-sample trades the screen saw.
        expectancy_r: In-sample expectancy the screen saw.
        z: Cross-sectional z of expectancy within this instrument.
        drop_share: Share of signals lost to the position limit.
        residual_z: ``z`` with the drop share regressed out — the ranking key.
    """

    key: str
    spec_id: str
    spec_path: str
    symbol: str
    timeframe: str
    trades: int
    expectancy_r: float
    z: float
    drop_share: float
    residual_z: float


@dataclass(frozen=True)
class FoldOutcome:
    """One fold of one candidate, compactly.

    Attributes:
        index: Fold number.
        oos_trades: Trades the OOS run closed.
        oos_expectancy_r: OOS expectancy.
        n_trials: Parameter sets the selector evaluated on this fold's IS.
        chosen: The parameters it chose.
        plateau_size: Points inside the chosen point's plateau.
        selection_shift: Distance between the argmax and the chosen point.
        score_gap: Score given up by centring rather than taking the argmax.
    """

    index: int
    oos_trades: int
    oos_expectancy_r: float | None
    n_trials: int
    chosen: dict[str, float | int]
    plateau_size: int | None
    selection_shift: int | None
    score_gap: float | None


@dataclass(frozen=True)
class CandidateResult:
    """What a walk-forward said about one candidate.

    Attributes:
        key: The candidate's key.
        wf_id: Identity of the walk-forward.
        folds: Per-fold outcomes.
        stitched_trades: OOS trades across all folds.
        stitched_expectancy_r: OOS expectancy across all folds.
        n_trials_total: Parameter sets evaluated across all folds — what this
            candidate adds to the screen's own trial count.
        truncated_folds: Folds whose search stopped on the trade budget rather
            than on the trial budget. Recorded rather than inferred from a small
            ``n_trials``: a coarse search is a fact about this measurement, and
            a reader comparing two candidates has to be able to see that one of
            them was searched less.
        seconds: Wall time.
        error: Why there is no result, when there is none.
    """

    key: str
    wf_id: str | None
    folds: tuple[FoldOutcome, ...]
    stitched_trades: int
    stitched_expectancy_r: float | None
    n_trials_total: int
    seconds: float
    truncated_folds: tuple[int, ...] = ()
    error: str | None = None


def drop_share(row: ScreenRow) -> float:
    """Share of a row's signals that the position limit refused.

    Args:
        row: A screen row.

    Returns:
        ``dropped / (dropped + trades)`` when the position limit was the
        dominant counter, else 0. The row carries only its dominant counter, so
        this understates the share for rows blocked mainly by something else —
        which is the safe direction: it never invents a confound.
    """
    if row.dominant_reason != "drop:position_limit":
        return 0.0
    total = row.dominant_count + row.trades
    return row.dominant_count / total if total else 0.0


def residual_z(scores: Sequence[float], shares: Sequence[float]) -> list[float]:
    """``scores`` with the linear dependence on ``shares`` removed.

    One simple regression, not a model: the point is only that the ranking key
    has zero linear correlation with the confound, so a candidate cannot be
    promoted for having been thinned. The non-linear remainder is not removed
    and the report says so.

    Args:
        scores: The statistic being ranked on.
        shares: The confound.

    Returns:
        Residuals, in input order. ``scores`` unchanged when the confound has no
        variance to regress against.
    """
    size = len(scores)
    if size < 3:
        return list(scores)
    mean_x = sum(shares) / size
    mean_y = sum(scores) / size
    var_x = sum((value - mean_x) ** 2 for value in shares)
    if var_x <= 0:
        return list(scores)
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(shares, scores, strict=True))
    slope = cov / var_x
    intercept = mean_y - slope * mean_x
    return [y - (intercept + slope * x) for x, y in zip(shares, scores, strict=True)]


def select(
    rows: Sequence[ScreenRow],
    z_scores: Mapping[str, float],
    spec_paths: Mapping[str, Path],
    *,
    limit: int,
    min_trades: int = MIN_TRADES,
    excluded_timeframes: Sequence[str] = EXCLUDED_TIMEFRAMES,
) -> tuple[Candidate, ...]:
    """Promote the best rows, ranked on the confound-adjusted statistic.

    Args:
        rows: Every screened row.
        z_scores: Cross-sectional z of ``expectancy_r`` by row key.
        spec_paths: Where each spec id's file lives.
        limit: How many candidates to promote.
        min_trades: In-sample trades a row needs.
        excluded_timeframes: Bar sizes not worth a walk-forward.

    Returns:
        The candidates, best first.
    """
    eligible = [
        row
        for row in rows
        if row.error is None
        and row.timeframe not in excluded_timeframes
        and row.trades >= min_trades
        and row.expectancy_r is not None
        and row.key in z_scores
        and row.spec_id in spec_paths
    ]
    if not eligible:
        return ()
    shares = [drop_share(row) for row in eligible]
    adjusted = residual_z([z_scores[row.key] for row in eligible], shares)
    candidates = [
        Candidate(
            key=row.key,
            spec_id=row.spec_id,
            spec_path=str(spec_paths[row.spec_id]),
            symbol=row.symbol,
            timeframe=row.timeframe,
            trades=row.trades,
            expectancy_r=float(row.expectancy_r or 0.0),
            z=z_scores[row.key],
            drop_share=share,
            residual_z=value,
        )
        for row, share, value in zip(eligible, shares, adjusted, strict=True)
    ]
    candidates.sort(key=lambda item: -item.residual_z)
    return tuple(candidates[:limit])


@dataclass(frozen=True)
class ShortlistManifest:
    """The shortlist, frozen before a single walk-forward has run.

    Attributes:
        screen_id: The screen the candidates came from.
        generated: When the shortlist was fixed.
        holdout_fraction: Share of every series still withheld.
        thresholds: The rules that produced it, stated rather than implied.
        confound: What the exposure confound measured, before adjustment.
        candidates: The shortlist, best first.
    """

    screen_id: str
    generated: str
    holdout_fraction: float
    thresholds: dict[str, object]
    confound: dict[str, float]
    candidates: tuple[Candidate, ...]

    def write(self, path: Path) -> Path:
        """Write the manifest, refusing to overwrite one that exists.

        Raises:
            FileExistsError: If a shortlist is already frozen at ``path``. A
                second selection over the same screen would be a threshold
                chosen after seeing results.
        """
        if path.exists():
            raise FileExistsError(
                f"a shortlist is already frozen at {path}; re-selecting after results "
                "exist is choosing a threshold with the answers visible"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "screen_id": self.screen_id,
            "generated": self.generated,
            "holdout_fraction": self.holdout_fraction,
            "thresholds": self.thresholds,
            "confound": self.confound,
            "candidates": [asdict(item) for item in self.candidates],
            "what_this_is": (
                "A shortlist frozen before any walk-forward ran. The holdout is untouched: "
                "nothing here is confirmation, and a candidate that survives the walk-forward "
                "has still only been measured on bars the screen already selected it on."
            ),
        }
        path.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        return path

    @staticmethod
    def read(path: Path) -> "ShortlistManifest | None":
        """Read a frozen shortlist, or ``None`` when none exists."""
        if not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        return ShortlistManifest(
            screen_id=payload["screen_id"],
            generated=payload["generated"],
            holdout_fraction=payload["holdout_fraction"],
            thresholds=payload["thresholds"],
            confound=payload["confound"],
            candidates=tuple(Candidate(**item) for item in payload["candidates"]),
        )


def build_walkforward_inputs(
    candidate: Candidate,
    data_dir: Path,
    instruments_path: Path,
    exit_library: Path,
    *,
    holdout_fraction: float = DEFAULT_HOLDOUT_FRACTION,
) -> tuple[RunInputs, StrategySpec]:
    """Assemble the base run a candidate's walk-forward varies from.

    The whole pre-holdout series rather than the screen's bar budget: folds need
    the span. The holdout is cut exactly as the screen cut it, by the same
    function, so stage two cannot see a bar stage one could not.

    Args:
        candidate: What to run.
        data_dir: Root of the parquet store.
        instruments_path: Instrument registry file.
        exit_library: Exit preset library.
        holdout_fraction: Share of the series withheld.

    Returns:
        ``(inputs, spec)``.

    Raises:
        ValueError: If the exit preset is missing or the store holds no bars.
    """
    spec = StrategySpec.model_validate_json(Path(candidate.spec_path).read_text(encoding="utf-8"))
    timeframe: Timeframe = spec.timeframes.signal_tf
    library = ExitLibrarySpec.model_validate_json(exit_library.read_text(encoding="utf-8"))
    preset = next((item for item in library.presets if item.id == spec.exit_ref), None)
    if preset is None:
        raise ValueError(f"exit_ref {spec.exit_ref!r} is not in {exit_library}")

    store = ParquetStore(data_dir)
    whole = store.get(candidate.symbol, timeframe)
    if whole.is_empty:
        raise ValueError(f"no bars stored for {candidate.symbol} {timeframe.value}")
    visible, boundary = screen_frame(whole, fraction=holdout_fraction)

    def load(symbol: str, bars: Timeframe) -> OHLCVFrame:
        return store.get(symbol, bars).slice(None, boundary.boundary)

    instruments = load_instruments(instruments_path)
    key = StreamKey(candidate.symbol, timeframe)
    inputs = RunInputs(
        config=BacktestConfig(account_currency="USD", starting_balance=Decimal(100_000)),
        streams={key: visible},
        bindings=(StrategyBinding(spec=spec, exit_preset=preset, keys=(key,)),),
        instruments=instruments,
        costs=CostConfig(run_seed=0),
        sizing=FixedFractional(risk_pct=0.01),
        converter=build_converter(
            (instruments[candidate.symbol],),
            account_currency="USD",
            timeframe=timeframe,
            load=load,
        ),
    )
    return inputs, spec


@dataclass(frozen=True)
class StageTwoManifest:
    """What stage two ran and what it added to the trial count.

    Attributes:
        screen_id: The screen the shortlist came from.
        generated: When the stage finished.
        results: One record per candidate.
        n_trials_screen: Comparisons the screen made.
        n_trials_effective_screen: Independent comparisons it was worth.
        n_trials_optimisation: Parameter sets stage two evaluated in total.
        n_trials_effective_total: The two added, not replaced.
        alpha: Significance the honest threshold is derived from.
    """

    screen_id: str
    generated: str
    results: tuple[CandidateResult, ...]
    n_trials_screen: int
    n_trials_effective_screen: float
    n_trials_optimisation: int
    n_trials_effective_total: float
    alpha: float = 0.05
    notes: dict[str, object] = field(default_factory=dict)

    @property
    def honest_percentile(self) -> float:
        """The percentile a single candidate must beat, given every trial made.

        ``100 x (1 - alpha / N_eff)``. The number exists to be looked at: at a
        few hundred effective trials it is past what a twenty-iteration
        calibration can resolve at all, which is the fact that decides whether a
        stage can confirm anything.
        """
        return 100.0 * (1.0 - self.alpha / max(self.n_trials_effective_total, 1.0))

    def write(self, path: Path) -> Path:
        """Write the manifest."""
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "screen_id": self.screen_id,
            "generated": self.generated,
            "n_trials_screen": self.n_trials_screen,
            "n_trials_effective_screen": round(self.n_trials_effective_screen, 2),
            "n_trials_optimisation": self.n_trials_optimisation,
            "n_trials_effective_total": round(self.n_trials_effective_total, 2),
            "alpha": self.alpha,
            "honest_percentile": round(self.honest_percentile, 4),
            "results": [asdict(item) for item in self.results],
            "notes": self.notes,
            "what_this_is": (
                "A walk-forward over the same pre-holdout bars the screen already selected on. "
                "It is not confirmation: the holdout has not been touched, and the trial count "
                "the screen contributed does not go away because a second stage followed it."
            ),
        }
        path.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        return path


def now_iso() -> str:
    """The current instant, ISO-8601, UTC."""
    return datetime.now(UTC).isoformat()


def without_throttles(
    document: Mapping[str, Any],
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """A generated search space with the risk throttles removed.

    Args:
        document: What
            :func:`~trading_system.validation.space_builder.build_space_document`
            produced.

    Returns:
        ``(document, removed)``. The document is a copy; ``removed`` names the
        axes struck out so the manifest can record them.
    """
    axes: list[dict[str, Any]] = list(document.get("axes", []))
    kept = [axis for axis in axes if axis.get("name") not in THROTTLE_AXES]
    removed = tuple(str(axis.get("name")) for axis in axes if axis.get("name") in THROTTLE_AXES)
    names = {axis.get("name") for axis in kept}
    constraints: list[dict[str, Any]] = [
        item
        for item in document.get("constraints", [])
        if item.get("less") in names and item.get("greater") in names
    ]
    return {**dict(document), "axes": kept, "constraints": constraints}, removed


class _BudgetExhaustedError(Exception):
    """Raised inside a wrapped evaluate to end a fold's search early."""


@dataclass
class TradeBudgetedSearch:
    """A search that stops once its fold has simulated enough trades.

    Wraps any :class:`~trading_system.validation.optimization.ParameterSearch`
    and counts what each trial actually cost, in the one unit that predicts cost
    and is reproducible: closed trades. When the budget runs out the inner
    search is interrupted and the trials already evaluated are returned, so the
    fold still selects — from fewer points, and saying so.

    Prediction was tried first and does not work. The screen measures a
    strategy's trade count on its *file's* parameters, and a search samples
    other parameters: the candidate that ran sixty times slower than its
    neighbour had **fewer** screened trades (487 against 760) and a search space
    three thousand times larger. Cost is a property of what the search samples,
    which nothing outside the search can know in advance.

    Attributes:
        inner: The search actually proposing points.
        trade_budget: Trades a fold may simulate before proposals stop.
        min_trials: Trials a fold gets regardless.
        truncated_folds: Indices of folds that hit the budget, in order.
        spent: Trades simulated on the most recent fold.
    """

    inner: ParameterSearch
    trade_budget: int = TRADE_BUDGET_PER_FOLD
    min_trials: int = MIN_TRIALS_PER_FOLD
    truncated_folds: list[int] = field(default_factory=list)
    spent: int = 0
    _fold: int = 0

    @property
    def name(self) -> str:
        """Identifies the method in reports and stored selections."""
        return f"{self.inner.name}+trades<={self.trade_budget}"

    def run(self, space: "SearchSpace", evaluate: "Evaluate", budget: int) -> list["TrialRecord"]:
        """Spend up to ``budget`` trials, or the trade budget, whichever binds first."""
        collected: list[TrialRecord] = []
        spent = 0

        def counting(params: "ParamSet") -> "TrialOutcome":
            nonlocal spent
            outcome = evaluate(params)
            collected.append(TrialRecord(params=params, outcome=outcome))
            spent += outcome.n_trades
            if spent >= self.trade_budget and len(collected) >= self.min_trials:
                raise _BudgetExhaustedError
            return outcome

        try:
            records = self.inner.run(space, counting, budget)
        except _BudgetExhaustedError:
            self.truncated_folds.append(self._fold)
            records = collected
        self.spent = spent
        self._fold += 1
        return records
