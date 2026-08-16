"""Confirming candidates on the holdout: the one measurement that is not a selection.

Everything before this stage ranked. The screen ranked eight hundred strategies
over the first eighty per cent of every series; stage two ranked thirty of them
by walking folds cut from those same bars — and the overlap was measured at
**100% for 21 of 29 candidates**, so what stage two called out-of-sample was
data the candidate had already been chosen on. Neither stage measured anything
about an edge, and the numbers they produced are an ordering, not evidence.

The holdout is the last fifth of every series, cut before the screen ran and
untouched since. It is the only data in the project that took no part in
choosing what to test on it, and **it can be spent exactly once**: reading a
result and then adjusting anything — a threshold, a candidate list, a
parameter — turns it into one more selection set, and no later measurement can
undo that. So the criterion is frozen in a manifest before the first run, in a
file that refuses to be overwritten, and the parameters come in frozen from
stage two.

**The honest significance threshold is not reachable, and this module says so
rather than quietly using a weaker one.** The trial count accumulated by the two
ranking stages is ``N_eff`` between 928 and 1191, which puts a Bonferroni-style
threshold at the 99.995th percentile. Resolving that by calibration needs on the
order of twenty thousand null iterations per candidate; the budget here is two
hundred, which resolves half a percentile point. A candidate that passes has
therefore beaten *the strongest test this data can support*, which is roughly a
correction for ten trials rather than for a thousand — and
:attr:`ConfirmCriterion.claim_strength` states that in the manifest so no reader
has to reconstruct it.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from trading_system.backtest.clock import StreamKey
from trading_system.backtest.config import BacktestConfig
from trading_system.backtest.fx import build_converter
from trading_system.backtest.orchestrator import BacktestResult, StrategyBinding
from trading_system.backtest.spec import RunInputs
from trading_system.core.instruments import load_instruments
from trading_system.core.types import Timeframe
from trading_system.data.models import OHLCVFrame
from trading_system.data.store import ParquetStore
from trading_system.execution.config import CostConfig
from trading_system.exit.library import ExitLibrarySpec
from trading_system.risk.sizing.methods import FixedFractional
from trading_system.strategies.schema import StrategySpec
from trading_system.validation.holdout import DEFAULT_HOLDOUT_FRACTION, holdout_boundary
from trading_system.validation.optimization import ParamSet, SearchSpace
from trading_system.validation.shortlist import Candidate

#: Trades a candidate needs on the holdout before its expectancy is read at all.
#: The same gate the verdict already applies, and the stage-two post-mortem
#: measured why: at 12-25 trades a per-window expectancy of +-0.4R is
#: indistinguishable from zero, and two of the three best-looking candidates sat
#: below that on every fold.
MIN_TRADES = 100

#: Equal-length slices the holdout is cut into for reading, not for deciding.
#: Four is enough to expose the failure stage two hid — an aggregate carried by
#: one or two windows — and few enough that each still holds trades.
PERIODS = 4

#: Share of total R that a single period may contribute before the result is
#: called concentrated. Stage two's best candidates had two folds of fifteen
#: carrying 83% and one fold of twelve carrying 99%.
MAX_PERIOD_SHARE = 0.5

#: Percentile of a candidate's own null it must beat. The highest that two
#: hundred iterations resolve with margin; see the module docstring for what it
#: is and is not worth.
NULL_PERCENTILE = 99.0

#: Null iterations per candidate. Fixed before the run, like everything else.
NULL_ITERATIONS = 200


@dataclass(frozen=True)
class ConfirmCriterion:
    """What counts as confirmation, written down before anything is run.

    Attributes:
        min_trades: Trades on the holdout below which no verdict is issued.
        periods: Equal slices the holdout is read in.
        max_period_share: Share of total R one period may contribute.
        null_percentile: Percentile of its own null a candidate must beat.
        null_iterations: Iterations that null is estimated from.
        statistic: What is compared, named so it cannot drift.
        parameter_source: Where each candidate's frozen parameters come from.
        honest_percentile: The threshold the accumulated trial count would
            require.
        claim_strength: What passing this criterion does and does not license.
    """

    min_trades: int = MIN_TRADES
    periods: int = PERIODS
    max_period_share: float = MAX_PERIOD_SHARE
    null_percentile: float = NULL_PERCENTILE
    null_iterations: int = NULL_ITERATIONS
    statistic: str = (
        "expectancy_r over the whole holdout, with the per-period breakdown "
        "published beside it and a concentration guard applied to it"
    )
    parameter_source: str = (
        "the parameters stage two's last fold selected — what the walk-forward "
        "procedure would actually hand over at the moment the holdout begins. "
        "Nothing is tuned here."
    )
    honest_percentile: float = 99.995
    claim_strength: str = (
        "Passing means the candidate beat the strongest test these data support, "
        "not that it is significant after correcting for the 928-1191 effective "
        "trials the two ranking stages spent. Resolving that correction needs "
        "~20 000 null iterations per candidate against the 200 budgeted here, so "
        "a pass is worth roughly a correction for ten trials, not a thousand."
    )


@dataclass(frozen=True)
class PeriodOutcome:
    """One slice of the holdout, for reading concentration.

    Attributes:
        index: Slice number, oldest first.
        start: First instant in the slice.
        end: Last instant in the slice.
        trades: Trades closed inside it.
        expectancy_r: Mean R of those trades, or ``None`` with none.
        total_r: Sum of R, which is what concentration is measured on.
    """

    index: int
    start: str
    end: str
    trades: int
    expectancy_r: float | None
    total_r: float


@dataclass(frozen=True)
class ConfirmOutcome:
    """What the holdout said about one candidate.

    Attributes:
        key: The candidate's key.
        spec_id: Strategy id.
        symbol: Instrument.
        timeframe: Bar size.
        params: The frozen parameters it ran on.
        bars: Holdout bars walked.
        start: First holdout bar.
        end: Last holdout bar.
        trades: Trades closed.
        expectancy_r: Expectancy over the whole holdout.
        median_period_expectancy_r: Median over the periods that traded.
        periods: The per-period breakdown.
        top_period_share: Share of total R from the single best period.
        null_percentile: Where the run sits in its own null's distribution.
        null_median: Median of that null.
        passed: Whether it met every clause of the criterion.
        failed_clauses: Which clauses it missed, named.
        seconds: Wall time.
        error: Why there is no outcome, when there is none.
    """

    key: str
    spec_id: str
    symbol: str
    timeframe: str
    params: dict[str, float | int]
    bars: int
    start: str | None
    end: str | None
    trades: int
    expectancy_r: float | None
    median_period_expectancy_r: float | None
    periods: tuple[PeriodOutcome, ...]
    top_period_share: float | None
    null_percentile: float | None
    null_median: float | None
    passed: bool
    failed_clauses: tuple[str, ...]
    seconds: float
    error: str | None = None


def holdout_frame(
    frame: OHLCVFrame, *, fraction: float = DEFAULT_HOLDOUT_FRACTION
) -> OHLCVFrame:
    """The withheld tail of a series — the complement of what the screen saw.

    Args:
        frame: The whole series.
        fraction: Share withheld, the same figure the screen was cut with.

    Returns:
        Every bar at or after the boundary. Cut with the same function, so the
        two sides tile exactly and no bar belongs to both.
    """
    return frame.slice(holdout_boundary(frame, fraction=fraction), None)


def params_from(space: SearchSpace, chosen: Mapping[str, float | int]) -> ParamSet:
    """Rebuild a point from the values stage two recorded for it.

    Args:
        space: The candidate's search space.
        chosen: Axis name to value, as stored in the fold's selection.

    Returns:
        The point.

    Raises:
        ValueError: If a value is not one the axis offers — which would mean the
            recorded selection and the space have drifted apart, and running
            something adjacent to what stage two chose would make this stage
            measure a strategy nobody selected.
    """
    coords: list[int] = []
    for axis in space.axes:
        if axis.name not in chosen:
            raise ValueError(f"the recorded selection has no value for axis {axis.name!r}")
        value = chosen[axis.name]
        if value not in axis.values:
            raise ValueError(
                f"axis {axis.name!r} does not offer {value!r}; the recorded selection and "
                "the space have drifted apart"
            )
        coords.append(list(axis.values).index(value))
    return space.point(tuple(coords))


def build_holdout_inputs(
    spec: StrategySpec,
    symbol: str,
    data_dir: Path,
    instruments_path: Path,
    exit_library: Path,
    *,
    fraction: float = DEFAULT_HOLDOUT_FRACTION,
    risk_pct: float = 0.01,
) -> RunInputs:
    """Assemble a run over the holdout alone, on already-frozen parameters.

    The mirror image of
    :func:`~trading_system.validation.screen.build_screen_inputs`: that one
    keeps everything before the boundary, this one everything after, and both
    compute the boundary with the same function so the two never overlap.

    Args:
        spec: The strategy, with its frozen parameters already written in.
        symbol: Instrument to trade.
        data_dir: Root of the parquet store.
        instruments_path: Instrument registry file.
        exit_library: Exit preset library.
        fraction: Share of the series withheld.
        risk_pct: Fraction of equity risked per trade.

    Returns:
        The run.

    Raises:
        ValueError: If the exit preset is missing or the store holds no bars.
    """
    timeframe: Timeframe = spec.timeframes.signal_tf
    library = ExitLibrarySpec.model_validate_json(exit_library.read_text(encoding="utf-8"))
    preset = next((item for item in library.presets if item.id == spec.exit_ref), None)
    if preset is None:
        raise ValueError(f"exit_ref {spec.exit_ref!r} is not in {exit_library}")

    store = ParquetStore(data_dir)
    whole = store.get(symbol, timeframe)
    if whole.is_empty:
        raise ValueError(f"no bars stored for {symbol} {timeframe.value}")
    tail = holdout_frame(whole, fraction=fraction)
    if tail.is_empty:
        raise ValueError(f"the holdout of {symbol} {timeframe.value} is empty")

    boundary = holdout_boundary(whole, fraction=fraction)

    def load(other: str, bars: Timeframe) -> OHLCVFrame:
        return store.get(other, bars).slice(boundary, None)

    instruments = load_instruments(instruments_path)
    key = StreamKey(symbol, timeframe)
    return RunInputs(
        config=BacktestConfig(account_currency="USD", starting_balance=Decimal(100_000)),
        streams={key: tail},
        bindings=(StrategyBinding(spec=spec, exit_preset=preset, keys=(key,)),),
        instruments=instruments,
        costs=CostConfig(run_seed=0),
        sizing=FixedFractional(risk_pct=risk_pct),
        converter=build_converter(
            (instruments[symbol],),
            account_currency="USD",
            timeframe=timeframe,
            load=load,
        ),
    )


def split_periods(
    result: BacktestResult, frame: OHLCVFrame, periods: int
) -> tuple[PeriodOutcome, ...]:
    """Cut a run's trades into equal calendar slices, for reading concentration.

    Equal in *time*, not in trades: slices of equal trade count would hide
    exactly the thing being looked for, a burst of result inside one stretch of
    market.

    Args:
        result: The finished run.
        frame: The bars it walked, for the slice boundaries.
        periods: How many slices.

    Returns:
        One outcome per slice, oldest first.
    """
    if frame.start is None or frame.end is None:
        return ()
    span = (frame.end - frame.start) / periods
    found: list[PeriodOutcome] = []
    for index in range(periods):
        start = frame.start + span * index
        end = frame.start + span * (index + 1) if index < periods - 1 else frame.end
        inside = [
            trade
            for trade in result.trades
            if start <= trade.closed_at <= end
        ]
        total = sum(trade.realized_r for trade in inside)
        found.append(
            PeriodOutcome(
                index=index,
                start=start.isoformat(),
                end=end.isoformat(),
                trades=len(inside),
                expectancy_r=(total / len(inside)) if inside else None,
                total_r=float(total),
            )
        )
    return tuple(found)


def judge(
    *,
    trades: int,
    expectancy_r: float | None,
    periods: Sequence[PeriodOutcome],
    null_percentile: float | None,
    criterion: ConfirmCriterion,
) -> tuple[bool, tuple[str, ...], float | None]:
    """Apply the frozen criterion, naming every clause that failed.

    Args:
        trades: Trades on the holdout.
        expectancy_r: Expectancy over the whole holdout.
        periods: The per-period breakdown.
        null_percentile: Where the run sits against its own null.
        criterion: The frozen rules.

    Returns:
        ``(passed, failed_clauses, top_period_share)``. Every clause is checked
        even after one fails, because "which of them it missed" is the useful
        output and stopping at the first would throw it away.
    """
    failed: list[str] = []
    if trades < criterion.min_trades:
        failed.append(f"trades {trades} < {criterion.min_trades}")
    if expectancy_r is None or expectancy_r <= 0:
        failed.append(f"expectancy_r {expectancy_r} is not above zero")

    positive = [period.total_r for period in periods if period.total_r > 0]
    total = sum(period.total_r for period in periods)
    share: float | None = None
    if total > 0 and positive:
        share = max(positive) / total
        if share > criterion.max_period_share:
            failed.append(
                f"one period contributes {share:.0%} of total R, above "
                f"{criterion.max_period_share:.0%}"
            )
    if null_percentile is None:
        failed.append("no null calibration")
    elif null_percentile < criterion.null_percentile:
        failed.append(
            f"null percentile {null_percentile:.1f} < {criterion.null_percentile:.1f}"
        )
    return not failed, tuple(failed), share


@dataclass(frozen=True)
class ConfirmManifest:
    """The frozen procedure, written before the first holdout run.

    Attributes:
        screen_id: The screen the candidates came from.
        generated: When the procedure was fixed.
        criterion: What counts as confirmation.
        candidates: Who is being tested, in order.
        holdout_fraction: Share of every series withheld.
        note: What this stage is and what spending it costs.
    """

    screen_id: str
    generated: str
    criterion: ConfirmCriterion
    candidates: tuple[Candidate, ...]
    holdout_fraction: float
    note: str = (
        "The holdout is spent by this run. Reading these results and then changing a "
        "threshold, a candidate list or a parameter would make it one more selection set, "
        "and nothing measured afterwards on these bars would be evidence again."
    )

    def write(self, path: Path) -> Path:
        """Write the manifest, refusing to overwrite one that exists.

        Raises:
            FileExistsError: If a procedure is already frozen at ``path``.
        """
        if path.exists():
            raise FileExistsError(
                f"a confirmation procedure is already frozen at {path}; rewriting it after "
                "results exist is choosing the criterion with the answers visible"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "screen_id": self.screen_id,
            "generated": self.generated,
            "holdout_fraction": self.holdout_fraction,
            "criterion": asdict(self.criterion),
            "candidates": [asdict(item) for item in self.candidates],
            "note": self.note,
        }
        path.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        return path

    @staticmethod
    def read(path: Path) -> "ConfirmManifest | None":
        """Read a frozen procedure, or ``None`` when none exists."""
        if not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        return ConfirmManifest(
            screen_id=payload["screen_id"],
            generated=payload["generated"],
            criterion=ConfirmCriterion(**payload["criterion"]),
            candidates=tuple(Candidate(**item) for item in payload["candidates"]),
            holdout_fraction=payload["holdout_fraction"],
            note=payload.get("note", ""),
        )


def now_iso() -> str:
    """The current instant, ISO-8601, UTC."""
    return datetime.now(UTC).isoformat()


@dataclass
class ConfirmLog:
    """Append-only log of outcomes, for the same reason the screen has one."""

    path: Path

    def completed(self) -> set[str]:
        """Keys already written."""
        if not self.path.exists():
            return set()
        return {
            str(json.loads(line)["key"])
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }

    def append(self, outcome: ConfirmOutcome) -> None:
        """Add one outcome, flushed before returning."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(outcome), default=str) + "\n")
            handle.flush()

    def outcomes(self) -> tuple[ConfirmOutcome, ...]:
        """Everything written so far."""
        if not self.path.exists():
            return ()
        found: list[ConfirmOutcome] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            payload["periods"] = tuple(PeriodOutcome(**item) for item in payload["periods"])
            payload["failed_clauses"] = tuple(payload["failed_clauses"])
            found.append(ConfirmOutcome(**payload))
        return tuple(found)
