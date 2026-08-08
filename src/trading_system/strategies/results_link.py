"""Measured results, bound to the exact strategy and the exact data that produced them.

**The join key is a digest, never a version string.** A result belongs to the
spec it actually ran — :attr:`ResultRecord.spec_digest`, the same number
:class:`~trading_system.backtest.reproducibility.RunManifest` records for that
strategy. The version is carried alongside as provenance for a human. That makes
"is this old result still valid for the current spec?" a comparison rather than a
rule about which semver bumps invalidate what, and no bookkeeping edit can
falsify it, because bookkeeping no longer lives in the spec.

**What a walk-forward binds to is the template, plus the selector.** An
optimising run overwrites the spec's parameter values on every fold, so its
result is evidence about the *procedure* — the pair (template spec, selector) —
and never about the parameter values the template happens to carry.
:attr:`ResultRecord.selector_key` is therefore required, not optional: a result
that cannot name its selector cannot say what it is evidence for.

**A dataset hash is an identity; coverage is what orders.** Runs over 2024-2025
and 2024-2026 hash differently and are not comparable, yet they overlap and are
not independent either. Neither fact is recoverable from two opaque hashes, so
this module stores the hash *and* the coverage, answers ``best`` with one row per
dataset rather than a single winner, and computes overlap on request while
leaving what to do about it to the caller.

Money never appears here. Every stored metric is a ratio or an R-multiple, which
is what makes results from different account sizes comparable in the first place.
"""

import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import duckdb
import polars as pl
from pydantic import BaseModel, ConfigDict, Field, model_validator

from trading_system.backtest.clock import StreamKey
from trading_system.backtest.reproducibility import RunManifest, digest
from trading_system.core.exceptions import ValidationError
from trading_system.data.models import OHLCVFrame
from trading_system.strategies.schema import StrategySpec

if TYPE_CHECKING:
    from trading_system.strategies.repository import StrategyRecord, StrategyRepository

#: File the result log is appended to, under the repository root.
RESULTS_FILE = "results.parquet"

#: What produced a result, distinguishing a single backtest from a walk-forward.
RUN_KIND_FLAT = "run"
RUN_KIND_WALKFORWARD = "walkforward"


class StreamCoverage(BaseModel):
    """What one stream of a run covered.

    Attributes:
        stream: ``"SYMBOL@TF"``.
        symbol: Instrument.
        timeframe: Bar size.
        start: First bar open, UTC.
        end: Last bar open, UTC.
        n_bars: How many bars.
        digest: The stream's own digest, from the run manifest.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    stream: str
    symbol: str
    timeframe: str
    start: datetime
    end: datetime
    n_bars: int
    digest: str


class ResultRecord(BaseModel):
    """One measured evaluation of one strategy on one dataset.

    Attributes:
        strategy_id: Which strategy.
        version: The version label the spec carried when it ran. Provenance
            only — never a join key.
        spec_digest: The spec that actually ran, digested exactly as
            :func:`trading_system.strategies.repository.spec_digest` does it —
            which is what lets the library ask "is this result still valid for
            what I hold?" as a comparison. For a walk-forward this is the
            template handed to the runner, because the folds each ran a
            different materialised spec and no single digest describes them.
        binding_digest: The manifest's own digest for this strategy, over the
            spec *and* its exit preset *and* the streams it traded. Stored
            beside ``spec_digest`` rather than instead of it: they answer
            different questions, and collapsing them would make a changed
            ``exit_ref`` read as "same spec".
        run_id: The run id, or the ``wf_id`` of a walk-forward.
        run_kind: ``"run"`` or ``"walkforward"``.
        selector_key: The parameter selector, e.g. ``"identity"`` or
            ``"optimize:..."``. Required: it names what the result is evidence
            about.
        dataset_hash: Digest over every stream's bars. Identity, not an order.
        coverage: Per-stream extent, which is what makes two datasets
            comparable or not.
        source_digest: The code that ran.
        metrics: Measured quantities, all ratios or R-multiples.
        verdict: The P15 verdict, when one was computed.
        created_at: When the row was written, UTC.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    strategy_id: str
    version: str
    spec_digest: str
    binding_digest: str
    run_id: str
    run_kind: str
    selector_key: str = Field(min_length=1)
    dataset_hash: str
    coverage: tuple[StreamCoverage, ...] = Field(min_length=1)
    source_digest: str
    metrics: Mapping[str, float] = Field(default_factory=dict)
    verdict: str | None = None
    created_at: datetime

    @model_validator(mode="after")
    def _kind_is_known(self) -> "ResultRecord":
        """Reject a run kind nothing downstream would recognise."""
        if self.run_kind not in (RUN_KIND_FLAT, RUN_KIND_WALKFORWARD):
            raise ValueError(
                f"run_kind must be {RUN_KIND_FLAT!r} or {RUN_KIND_WALKFORWARD!r}, "
                f"got {self.run_kind!r}"
            )
        return self

    @property
    def symbols(self) -> tuple[str, ...]:
        """Instruments this result covered, sorted."""
        return tuple(sorted({item.symbol for item in self.coverage}))

    @property
    def period_start(self) -> datetime:
        """Earliest bar across every stream."""
        return min(item.start for item in self.coverage)

    @property
    def period_end(self) -> datetime:
        """Latest bar across every stream."""
        return max(item.end for item in self.coverage)

    @property
    def n_bars(self) -> int:
        """Bars across every stream."""
        return sum(item.n_bars for item in self.coverage)

    def metric(self, name: str) -> float | None:
        """One metric, or ``None`` when this run did not report it.

        Args:
            name: Metric name.

        Returns:
            The value, or ``None``. Never a default: a metric that was not
            measured must not read as zero.
        """
        return self.metrics.get(name)

    def comparable_to(self, other: "ResultRecord") -> bool:
        """Whether two results were measured on identical bars.

        Args:
            other: The result to compare against.

        Returns:
            True when the dataset hashes match, which is the only condition
            under which two numbers may be ranked directly.
        """
        return self.dataset_hash == other.dataset_hash

    def to_row(self) -> dict[str, Any]:
        """Flatten to one storable row.

        Returns:
            JSON-able data, with the nested parts serialised as JSON text so
            the log stays a flat table a query layer can read without unnesting.
        """
        return {
            "strategy_id": self.strategy_id,
            "version": self.version,
            "spec_digest": self.spec_digest,
            "binding_digest": self.binding_digest,
            "run_id": self.run_id,
            "run_kind": self.run_kind,
            "selector_key": self.selector_key,
            "dataset_hash": self.dataset_hash,
            "coverage": json.dumps([item.model_dump(mode="json") for item in self.coverage]),
            "source_digest": self.source_digest,
            "metrics": json.dumps(dict(self.metrics), sort_keys=True),
            "verdict": self.verdict,
            "created_at": self.created_at,
            "symbols": json.dumps(list(self.symbols)),
            "period_start": self.period_start,
            "period_end": self.period_end,
            "n_bars": self.n_bars,
        }

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "ResultRecord":
        """Rebuild a record from a stored row.

        Args:
            row: One row as written by :meth:`to_row`.

        Returns:
            The record. The derived columns (``symbols``, ``period_start``,
            ``period_end``, ``n_bars``) are ignored on the way back in: they
            exist so a SQL reader need not unnest, and recomputing them from
            :attr:`coverage` is what keeps them from drifting.
        """
        return cls(
            strategy_id=row["strategy_id"],
            version=row["version"],
            spec_digest=row["spec_digest"],
            binding_digest=row["binding_digest"],
            run_id=row["run_id"],
            run_kind=row["run_kind"],
            selector_key=row["selector_key"],
            dataset_hash=row["dataset_hash"],
            coverage=tuple(
                StreamCoverage.model_validate(item) for item in json.loads(row["coverage"])
            ),
            source_digest=row["source_digest"],
            metrics=json.loads(row["metrics"]),
            verdict=row["verdict"],
            created_at=_as_utc(row["created_at"]),
        )

    def content_digest(self) -> str:
        """Digest of everything this row asserts except when it was written.

        Returns:
            A hex digest. Two records sharing a ``run_id`` must share this, or
            one of them is wrong — which is the check
            :meth:`ResultsLink.record` performs.
        """
        return digest(
            {
                "strategy_id": self.strategy_id,
                "version": self.version,
                "spec_digest": self.spec_digest,
                "binding_digest": self.binding_digest,
                "run_id": self.run_id,
                "run_kind": self.run_kind,
                "selector_key": self.selector_key,
                "dataset_hash": self.dataset_hash,
                "coverage": [item.model_dump(mode="json") for item in self.coverage],
                "source_digest": self.source_digest,
                "metrics": dict(sorted(self.metrics.items())),
                "verdict": self.verdict,
            }
        )


def dataset_hash(manifest: RunManifest) -> str:
    """The scalar identity of a run's bars.

    Derived from :attr:`RunManifest.data` rather than hashed afresh: a second
    hashing scheme over the same bars is a second answer to "same data?" waiting
    to disagree with the first. Distinct from ``run_id``, which also folds in
    code and configs and therefore cannot answer that question at all.

    Args:
        manifest: The run's manifest.

    Returns:
        Hex digest over the per-stream digests.
    """
    return digest(dict(manifest.data))


def coverage_of(
    streams: Mapping[StreamKey, OHLCVFrame], manifest: RunManifest
) -> tuple[StreamCoverage, ...]:
    """Describe what each stream covered.

    Args:
        streams: The bars the run was given.
        manifest: The run's manifest, for the per-stream digests.

    Returns:
        One entry per stream, sorted by stream name.

    Raises:
        ValidationError: If a stream is empty. A run over no bars has no
            coverage, and a record claiming one would be a claim about nothing.
    """
    out = []
    for key, frame in sorted(streams.items(), key=str):
        if frame.is_empty or frame.start is None or frame.end is None:
            raise ValidationError(f"stream {key} is empty; a result cannot cover no bars")
        out.append(
            StreamCoverage(
                stream=str(key),
                symbol=key.symbol,
                timeframe=key.timeframe.value,
                start=frame.start,
                end=frame.end,
                n_bars=len(frame),
                digest=manifest.data[str(key)],
            )
        )
    return tuple(out)


def overlap_fraction(left: ResultRecord, right: ResultRecord) -> float:
    """How much of the shorter run's period the two share.

    Answers the question two hashes cannot: results over 2024-2025 and
    2024-2026 are not the same dataset and not independent evidence either.
    What to conclude from the number stays with the caller, because that depends
    on the question being asked.

    Args:
        left: One result.
        right: The other.

    Returns:
        Overlapping span divided by the shorter of the two spans, in ``[0, 1]``.
        Zero when they do not overlap; zero-length spans give zero.
    """
    start = max(left.period_start, right.period_start)
    end = min(left.period_end, right.period_end)
    if end <= start:
        return 0.0
    shared = (end - start).total_seconds()
    shortest = min(
        (left.period_end - left.period_start).total_seconds(),
        (right.period_end - right.period_start).total_seconds(),
    )
    if shortest <= 0:
        return 0.0
    return min(1.0, shared / shortest)


class ResultsLink:
    """The append-only log of measured results, and the queries over it.

    Attributes:
        root: Directory holding ``results.parquet``.
    """

    def __init__(self, root: Path) -> None:
        """Open the result log rooted at ``root``, creating the directory.

        Args:
            root: Where ``results.parquet`` lives or should live.
        """
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._path = self._root / RESULTS_FILE

    @property
    def root(self) -> Path:
        """The repository root."""
        return self._root

    @property
    def path(self) -> Path:
        """Where the log is stored."""
        return self._path

    def records(self) -> list[ResultRecord]:
        """Every stored result.

        Returns:
            All records, newest last.
        """
        if not self._path.exists():
            return []
        frame = pl.read_parquet(self._path)
        return [ResultRecord.from_row(row) for row in frame.to_dicts()]

    def get(self, run_id: str) -> ResultRecord | None:
        """The result stored under ``run_id``.

        Args:
            run_id: Run or walk-forward id.

        Returns:
            The record, or ``None``.
        """
        for record in self.records():
            if record.run_id == run_id:
                return record
        return None

    def record(self, result: ResultRecord) -> ResultRecord:
        """Append a result, or confirm it repeats one already stored.

        Idempotent on ``run_id``, and deliberately unforgiving about
        disagreement: a run id is a promise that the same inputs produce the
        same output, so two records sharing one and differing in what they
        measured is a broken promise, not a row to overwrite. This mirrors
        :func:`trading_system.backtest.reproducibility.write_run`, which
        refuses the same way and likewise keeps what it already had.

        Args:
            result: The record to store.

        Returns:
            The stored record — the existing one when this repeats it.

        Raises:
            ValidationError: If a different result is already stored under this
                run id.
        """
        existing = self.get(result.run_id)
        if existing is not None:
            if existing.content_digest() != result.content_digest():
                raise ValidationError(
                    f"run {result.run_id} already holds a different result: stored "
                    f"{existing.content_digest()}, offered {result.content_digest()}. "
                    "A shared run id promises identical inputs and therefore identical "
                    "metrics; the stored row is kept."
                )
            return existing

        row = result.to_row()
        frame = pl.DataFrame([row])
        if self._path.exists():
            frame = pl.concat([pl.read_parquet(self._path), frame], how="vertical_relaxed")
        frame.write_parquet(self._path)
        return result

    def for_strategy(
        self, strategy_id: str, *, spec_digest: str | None = None
    ) -> list[ResultRecord]:
        """Every run of one strategy.

        Args:
            strategy_id: The strategy.
            spec_digest: Keep only runs of this exact spec. This is how a
                caller asks for results that are still valid for what the
                library currently holds — the version string cannot answer it.

        Returns:
            Matching records, oldest first.
        """
        out = [record for record in self.records() if record.strategy_id == strategy_id]
        if spec_digest is not None:
            out = [record for record in out if record.spec_digest == spec_digest]
        return sorted(out, key=lambda record: record.created_at)

    def best(
        self,
        strategy_id: str,
        *,
        metric: str = "sharpe",
        run_kind: str | None = None,
    ) -> list[ResultRecord]:
        """The best run *per dataset*, not the best run.

        A single winner across datasets would reward whichever run drew the
        kindest period, and two overlapping runs are not independent evidence
        anyway. Ranking therefore happens only inside a group of runs measured
        on identical bars; comparing across groups is a decision this module
        refuses to make silently. Same reasoning as the walk-forward report
        publishing ``(is, oos)`` pairs instead of their ratio.

        Args:
            strategy_id: The strategy.
            metric: Which metric to rank on.
            run_kind: Restrict to ``"run"`` or ``"walkforward"``.

        Returns:
            One record per dataset hash — the best on ``metric`` — ordered
            best first. Runs that did not report the metric are skipped;
            datasets where nothing reported it do not appear.
        """
        best_by_dataset: dict[str, ResultRecord] = {}
        for record in self.for_strategy(strategy_id):
            if run_kind is not None and record.run_kind != run_kind:
                continue
            value = record.metric(metric)
            if value is None:
                continue
            current = best_by_dataset.get(record.dataset_hash)
            if current is None or value > (current.metric(metric) or float("-inf")):
                best_by_dataset[record.dataset_hash] = record
        return sorted(
            best_by_dataset.values(),
            key=lambda record: record.metric(metric) or float("-inf"),
            reverse=True,
        )

    def find(
        self,
        *,
        metric: str = "sharpe",
        minimum: float | None = None,
        run_kind: str | None = None,
        verdict: str | None = None,
        strategy_ids: Sequence[str] | None = None,
    ) -> list[ResultRecord]:
        """Results clearing a threshold — "strategies with Sharpe above 1".

        Args:
            metric: Which metric to test.
            minimum: Lowest acceptable value. ``None`` keeps every run that
                reported the metric at all.
            run_kind: Restrict to ``"run"`` or ``"walkforward"``. Passing
                ``"walkforward"`` is how "on OOS" is expressed, because a
                walk-forward's stitched metrics *are* its out-of-sample ones.
            verdict: Keep only runs carrying this verdict.
            strategy_ids: Restrict to these strategies.

        Returns:
            Matching records, best first on ``metric``.
        """
        wanted = set(strategy_ids) if strategy_ids is not None else None
        out = []
        for record in self.records():
            if wanted is not None and record.strategy_id not in wanted:
                continue
            if run_kind is not None and record.run_kind != run_kind:
                continue
            if verdict is not None and record.verdict != verdict:
                continue
            value = record.metric(metric)
            if value is None:
                continue
            if minimum is not None and value < minimum:
                continue
            out.append(record)
        return sorted(out, key=lambda record: record.metric(metric) or float("-inf"), reverse=True)

    def index(
        self, connection: "duckdb.DuckDBPyConnection | None" = None
    ) -> "duckdb.DuckDBPyConnection":
        """Expose the log as a DuckDB table for ad-hoc SQL.

        Args:
            connection: Connection to build into. A fresh in-memory one by
                default.

        Returns:
            A connection holding a ``backtest_results`` table. Empty when
            nothing has been recorded — an absent log is no rows, not an error.
        """
        connection = connection if connection is not None else duckdb.connect(":memory:")
        connection.execute("DROP TABLE IF EXISTS backtest_results")
        if self._path.exists():
            # Read the parquet directly rather than handing DuckDB a polars
            # frame: the replacement scan converts through pyarrow, which this
            # project deliberately does not depend on (P02). Same reason
            # ``data/store.py`` queries files rather than frames.
            connection.execute(
                "CREATE TABLE backtest_results AS SELECT * FROM read_parquet(?)",
                [str(self._path)],
            )
        else:
            connection.execute(
                """
                CREATE TABLE backtest_results (
                    strategy_id VARCHAR, version VARCHAR, spec_digest VARCHAR,
                    run_id VARCHAR, run_kind VARCHAR, selector_key VARCHAR,
                    dataset_hash VARCHAR, coverage VARCHAR, source_digest VARCHAR,
                    metrics VARCHAR, verdict VARCHAR, created_at TIMESTAMPTZ,
                    symbols VARCHAR, period_start TIMESTAMPTZ, period_end TIMESTAMPTZ,
                    n_bars BIGINT
                )
                """
            )
        return connection


def build_record(
    *,
    spec: StrategySpec,
    manifest: RunManifest,
    streams: Mapping[StreamKey, OHLCVFrame],
    run_id: str,
    run_kind: str,
    selector_key: str,
    metrics: Mapping[str, float],
    verdict: str | None = None,
    created_at: datetime | None = None,
) -> ResultRecord:
    """Assemble a result record from a run's own manifest.

    Taking the manifest rather than loose fields is what keeps the record's
    ``binding_digest`` and ``dataset_hash`` equal to the run's own numbers
    instead of re-derived look-alikes; taking the spec is what makes
    ``spec_digest`` the same number the library computes, so the two can be
    compared at all.

    Args:
        spec: The strategy that ran — the template, for a walk-forward.
        manifest: The run's manifest.
        streams: The bars the run was given, for coverage.
        run_id: Run id, or ``wf_id`` for a walk-forward.
        run_kind: ``"run"`` or ``"walkforward"``.
        selector_key: The parameter selector's key.
        metrics: Measured quantities.
        verdict: The verdict, when one was computed.
        created_at: Write time; now by default.

    Returns:
        The record.

    Raises:
        ValidationError: If the manifest holds no binding for this strategy.
    """
    from trading_system.strategies.repository import spec_digest as digest_spec

    strategy_id = spec.id
    if strategy_id not in manifest.strategies:
        raise ValidationError(
            f"run {run_id} has no binding for strategy {strategy_id!r}; "
            f"it bound {sorted(manifest.strategies)}"
        )
    return ResultRecord(
        strategy_id=strategy_id,
        version=spec.version,
        spec_digest=digest_spec(spec),
        binding_digest=manifest.strategies[strategy_id],
        run_id=run_id,
        run_kind=run_kind,
        selector_key=selector_key,
        dataset_hash=dataset_hash(manifest),
        coverage=coverage_of(streams, manifest),
        source_digest=manifest.code.source_digest,
        metrics=dict(metrics),
        verdict=verdict,
        created_at=created_at if created_at is not None else datetime.now(UTC),
    )


def approve_from_result(
    repository: "StrategyRepository", link: ResultsLink, run_id: str
) -> "StrategyRecord":
    """Approve a strategy on the strength of a stored result.

    The only supported route to :attr:`Status.APPROVED`, because it is the only
    one that cannot invent its own evidence: the verdict, the selector and the
    strategy are all read from the recorded run rather than passed in by
    whoever wants the approval.

    Args:
        repository: The library holding the strategy.
        link: The result log holding the run.
        run_id: The run that justifies the approval.

    Returns:
        The approved record.

    Raises:
        KeyError: If no such run, or no such strategy.
        ValidationError: If the run carries no verdict, if that verdict is not
            ``ROBUST``, or if the spec has since changed — approving a spec
            the run did not evaluate would be an approval of something
            unmeasured.
    """
    from trading_system.strategies.repository import ROBUST_VERDICT

    result = link.get(run_id)
    if result is None:
        raise KeyError(f"no result recorded for run {run_id!r}")
    if result.verdict is None:
        raise ValidationError(
            f"run {run_id} carries no verdict; approval requires {ROBUST_VERDICT}"
        )
    if result.verdict != ROBUST_VERDICT:
        raise ValidationError(
            f"run {run_id} is {result.verdict}, not {ROBUST_VERDICT}; approval refused"
        )
    current = repository.get(result.strategy_id)
    if current.digest != result.spec_digest:
        raise ValidationError(
            f"strategy {result.strategy_id!r} has changed since run {run_id} "
            f"(spec {result.spec_digest} evaluated, {current.digest} held); "
            "re-run before approving"
        )
    return repository.approve(
        result.strategy_id,
        run_id=run_id,
        selector_key=result.selector_key,
        verdict=result.verdict,
    )


def _as_utc(moment: datetime) -> datetime:
    """Force a stored timestamp back to tz-aware UTC.

    Args:
        moment: The value read back from parquet.

    Returns:
        The same instant, tz-aware.
    """
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=UTC)
