"""The strategy library: specs on disk, bookkeeping beside them, an index over both.

**Two files per strategy, never one.** ``library/{type}/{id}.json`` holds the
:class:`~trading_system.strategies.schema.StrategySpec` — what is traded — and
``library/{type}/{id}.meta.json`` holds :class:`StrategyMeta` — what the entry is
called, who owns it, and what stage it has reached. The split is not filing
tidiness. The spec is digested into a run id
(:class:`~trading_system.backtest.reproducibility.RunManifest`), so any field
living inside it moves that id; a name, an author or a lifecycle stage inside the
spec would make renaming an author invalidate a verdict. Sections of one file
would not do either: the digest would then be a function of *part* of a file
rather than of the file, and "which part" is exactly the kind of rule that decays.

**Lifecycle is the log, not a field.** :attr:`StrategyMeta.status` is derived
from the last :class:`LifecycleEvent`, so a status cannot exist without the
record that produced it. ``APPROVED`` additionally refuses to be constructed
without a run id, a selector key and a ``ROBUST`` verdict — provenance is
structural, the same discipline that makes ``LEVEL_TOUCH`` require a price. The
repository has no method that shortens either log.

**The index is derived and rebuilt, never repaired.** :class:`StrategyRepository`
scans the library into an in-memory DuckDB table on demand. A stale index is
therefore not a state this module can be in, which is worth more than the scan it
costs at this corpus size.

**A missing meta file is not an error.** The spec alone is enough to trade, so
the record reads back with :attr:`Status.DRAFT` and empty bookkeeping. Losing a
meta file loses an approval rather than granting one — but the record still says
:attr:`StrategyRecord.meta_present` is false, because "lost" must stay
distinguishable from "never filled in".
"""

import json
from collections.abc import Collection, Iterator, Sequence
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import duckdb
from pydantic import BaseModel, ConfigDict, Field, model_validator

from trading_system.core.exceptions import ValidationError
from trading_system.core.instruments import InstrumentClass
from trading_system.strategies.schema import Regime, StrategySpec, StrategyType

#: Directory under the repository root that holds the specs.
LIBRARY_DIR = "library"

#: Suffix distinguishing the bookkeeping file from the spec beside it.
META_SUFFIX = ".meta.json"

#: Subdirectory, per strategy, holding superseded spec versions.
HISTORY_DIR_SUFFIX = ".history"

#: The verdict that :meth:`StrategyRepository.approve` demands. Spelled here as
#: a string rather than imported from :mod:`trading_system.validation.report`
#: because the strategy library must not depend on the validation stack to load.
ROBUST_VERDICT = "ROBUST"


class Status(StrEnum):
    """Lifecycle stage of a library entry.

    Repository state, deliberately not a :class:`StrategySpec` field: a stage
    moves *against* what is traded (retiring a strategy changes no rule), where
    ``version`` moves *with* it.
    """

    DRAFT = "DRAFT"
    TESTING = "TESTING"
    APPROVED = "APPROVED"
    RETIRED = "RETIRED"


class LifecycleEvent(BaseModel):
    """One stage transition, with the evidence that justified it.

    Attributes:
        status: The stage entered.
        at: When, UTC.
        spec_digest: The spec this applied to, so a stage can be told from the
            stage of a later edit.
        reason: Why. Required for :attr:`Status.RETIRED`.
        run_id: The run or walk-forward that justified an approval.
        selector_key: Which parameter selector that run used. Required for an
            approval because an optimising run endorses the pair (spec,
            selector) — the spec's own parameter values are overwritten per fold
            and were never the thing evaluated.
        verdict: The verdict recorded, ``"ROBUST"`` for an approval.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: Status
    at: datetime
    spec_digest: str
    reason: str | None = None
    run_id: str | None = None
    selector_key: str | None = None
    verdict: str | None = None

    @model_validator(mode="after")
    def _approval_carries_its_provenance(self) -> "LifecycleEvent":
        """Refuse an approval that cannot say what approved it."""
        if self.status is not Status.APPROVED:
            return self
        missing = [
            field for field in ("run_id", "selector_key", "verdict") if getattr(self, field) is None
        ]
        if missing:
            raise ValueError(f"an APPROVED event requires {missing}")
        if self.verdict != ROBUST_VERDICT:
            raise ValueError(f"approval requires verdict {ROBUST_VERDICT}, got {self.verdict!r}")
        return self

    @model_validator(mode="after")
    def _retirement_states_a_reason(self) -> "LifecycleEvent":
        """Refuse a retirement with no reason: the reason is the whole record."""
        if self.status is Status.RETIRED and not (self.reason and self.reason.strip()):
            raise ValueError("a RETIRED event requires a reason")
        return self


class VersionRecord(BaseModel):
    """One published version of the spec.

    Attributes:
        version: The semver the spec carried.
        spec_digest: What that spec digested to — the key results link on.
        at: When it was published, UTC.
        note: What changed, for a reader of the log.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: str
    spec_digest: str
    at: datetime
    note: str | None = None


class StrategyMeta(BaseModel):
    """Everything about a library entry that is not what it trades.

    Both logs are append-only by construction: :attr:`status` is read off the
    end of :attr:`lifecycle` rather than stored, so there is no way to record a
    stage without recording how it was reached.

    Attributes:
        id: The spec's id, repeated so the file names itself.
        name: Human-readable name.
        author: Who owns the entry.
        source: Where the idea came from (URL, book, paper).
        tags: Free-form labels the index filters on.
        notes: Prose about the entry.
        versions: Every published version, oldest first.
        lifecycle: Every stage transition, oldest first.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    name: str = Field(min_length=1)
    author: str = Field(min_length=1)
    source: str | None = None
    tags: tuple[str, ...] = ()
    notes: str | None = None
    versions: tuple[VersionRecord, ...] = ()
    lifecycle: tuple[LifecycleEvent, ...] = ()

    @property
    def status(self) -> Status:
        """The current stage.

        Returns:
            The last recorded stage, or :attr:`Status.DRAFT` when nothing has
            been recorded — a strategy nobody has graded is a draft.
        """
        return self.lifecycle[-1].status if self.lifecycle else Status.DRAFT

    @property
    def approval(self) -> LifecycleEvent | None:
        """The event that put this entry in :attr:`Status.APPROVED`, if it is.

        Returns:
            The last lifecycle event when it is an approval, else ``None``. A
            later transition supersedes an approval rather than amending it, so
            an entry retired after approval reports ``None`` here and keeps the
            approval visible in :attr:`lifecycle`.
        """
        if self.lifecycle and self.lifecycle[-1].status is Status.APPROVED:
            return self.lifecycle[-1]
        return None

    def appending(self, event: LifecycleEvent) -> "StrategyMeta":
        """This meta with ``event`` on the end of the lifecycle log.

        Args:
            event: The transition to record.

        Returns:
            A new meta. The original is untouched — the log only grows.
        """
        return self.model_copy(update={"lifecycle": (*self.lifecycle, event)})

    def publishing(self, record: VersionRecord) -> "StrategyMeta":
        """This meta with ``record`` on the end of the version log.

        Args:
            record: The version to record.

        Returns:
            A new meta.
        """
        return self.model_copy(update={"versions": (*self.versions, record)})


def spec_digest(spec: StrategySpec) -> str:
    """Digest a spec the same way a run id does.

    The one hashing rule in the system is
    :func:`trading_system.backtest.reproducibility.digest`; calling it here
    rather than hashing the JSON text means a result's ``spec_digest`` and the
    ``strategies`` entry of its :class:`RunManifest` are the same number,
    computed by the same code, and cannot drift apart.

    Args:
        spec: The strategy.

    Returns:
        Hex digest over every field the spec carries.
    """
    from trading_system.backtest.reproducibility import digest

    return digest(spec)


class StrategyRecord(BaseModel):
    """A library entry as the repository hands it out: spec, bookkeeping, digest.

    Attributes:
        spec: What is traded.
        meta: What the entry is called and what stage it reached.
        digest: :func:`spec_digest` of :attr:`spec`, precomputed because every
            caller that links a result needs it.
        meta_present: Whether a meta file was found. False means the
            bookkeeping shown is a default, not a record — a distinction that
            keeps a deleted approval from reading as an honest draft.
        path: Where the spec lives.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    spec: StrategySpec
    meta: StrategyMeta
    digest: str
    meta_present: bool
    path: Path

    @property
    def id(self) -> str:
        """The strategy id."""
        return self.spec.id

    @property
    def status(self) -> Status:
        """Current lifecycle stage."""
        return self.meta.status


def _default_meta(spec: StrategySpec) -> StrategyMeta:
    """Bookkeeping for a spec whose meta file is missing.

    Args:
        spec: The strategy found on disk.

    Returns:
        A draft entry named after its own id. Deliberately not an error: the
        spec alone is enough to trade, and refusing to read it would make a
        lost bookkeeping file break backtesting.
    """
    return StrategyMeta(id=spec.id, name=spec.id, author="unknown")


class StrategyRepository:
    """Specs and their bookkeeping on disk, with a rebuilt-on-demand index.

    Attributes:
        root: Directory holding ``library/``.
    """

    def __init__(self, root: Path) -> None:
        """Open a repository rooted at ``root``, creating the library directory.

        Args:
            root: Where ``library/`` lives or should live.
        """
        self._root = Path(root)
        self._library = self._root / LIBRARY_DIR
        self._library.mkdir(parents=True, exist_ok=True)

    @property
    def root(self) -> Path:
        """The repository root."""
        return self._root

    @property
    def library(self) -> Path:
        """The directory holding ``{type}/{id}.json``."""
        return self._library

    # -- paths -------------------------------------------------------------

    def spec_path(self, spec_type: StrategyType, strategy_id: str) -> Path:
        """Where a strategy's current spec lives.

        Args:
            spec_type: Holding-period class, which is the directory level.
            strategy_id: The strategy id.

        Returns:
            ``library/{type}/{id}.json``.
        """
        return self._library / spec_type.value / f"{strategy_id}.json"

    def meta_path(self, spec_type: StrategyType, strategy_id: str) -> Path:
        """Where a strategy's bookkeeping lives.

        Args:
            spec_type: Holding-period class.
            strategy_id: The strategy id.

        Returns:
            ``library/{type}/{id}.meta.json``.
        """
        return self._library / spec_type.value / f"{strategy_id}{META_SUFFIX}"

    def history_dir(self, spec_type: StrategyType, strategy_id: str) -> Path:
        """Where superseded versions of a strategy's spec are archived.

        Args:
            spec_type: Holding-period class.
            strategy_id: The strategy id.

        Returns:
            ``library/{type}/{id}.history/``.
        """
        return self._library / spec_type.value / f"{strategy_id}{HISTORY_DIR_SUFFIX}"

    def _find_spec_path(self, strategy_id: str) -> Path | None:
        """Locate a strategy's spec without knowing its type.

        Args:
            strategy_id: The strategy id.

        Returns:
            The path, or ``None`` when no entry carries that id.
        """
        for spec_type in StrategyType:
            candidate = self.spec_path(spec_type, strategy_id)
            if candidate.exists():
                return candidate
        return None

    # -- reading -----------------------------------------------------------

    def _read_record(self, path: Path) -> StrategyRecord:
        """Load one entry from its spec path.

        Args:
            path: The spec file.

        Returns:
            The record, with default bookkeeping when no meta file sits beside it.

        Raises:
            ValidationError: If the spec or its meta does not parse.
        """
        try:
            spec = StrategySpec.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception as error:  # noqa: BLE001 - re-raised with the path named
            raise ValidationError(f"{path}: {error}") from error

        meta_file = path.with_name(f"{path.stem}{META_SUFFIX}")
        present = meta_file.exists()
        if present:
            try:
                meta = StrategyMeta.model_validate_json(meta_file.read_text(encoding="utf-8"))
            except Exception as error:  # noqa: BLE001 - re-raised with the path named
                raise ValidationError(f"{meta_file}: {error}") from error
            if meta.id != spec.id:
                raise ValidationError(
                    f"{meta_file}: meta id {meta.id!r} does not match spec id {spec.id!r}"
                )
        else:
            meta = _default_meta(spec)

        return StrategyRecord(
            spec=spec,
            meta=meta,
            digest=spec_digest(spec),
            meta_present=present,
            path=path,
        )

    def _iter_spec_paths(self) -> Iterator[Path]:
        """Every current spec file in the library.

        Yields:
            Spec paths, sorted, excluding meta files and archived versions.
        """
        for spec_type in StrategyType:
            directory = self._library / spec_type.value
            if not directory.is_dir():
                continue
            for path in sorted(directory.glob("*.json")):
                if path.name.endswith(META_SUFFIX):
                    continue
                yield path

    def records(self) -> list[StrategyRecord]:
        """Every entry in the library.

        Returns:
            All records, ordered by id.
        """
        found = [self._read_record(path) for path in self._iter_spec_paths()]
        return sorted(found, key=lambda record: record.id)

    def get(self, strategy_id: str, version: str | None = None) -> StrategyRecord:
        """One entry, current or historical.

        Args:
            strategy_id: The strategy id.
            version: Which version. ``None`` is the current one; any other
                value is read from the archive, which is what makes an update
                non-destructive.

        Returns:
            The record. A historical read carries the *current* bookkeeping,
            because bookkeeping is not versioned — its own logs already say
            what happened when.

        Raises:
            KeyError: If no such strategy, or no such archived version.
        """
        path = self._find_spec_path(strategy_id)
        if path is None:
            raise KeyError(f"no strategy {strategy_id!r} in {self._library}")
        current = self._read_record(path)
        if version is None or version == current.spec.version:
            return current

        archived = self.history_dir(current.spec.type, strategy_id) / f"{version}.json"
        if not archived.exists():
            known = [record.version for record in current.meta.versions]
            raise KeyError(f"strategy {strategy_id!r} has no version {version!r}; have {known}")
        historical = self._read_record(archived)
        return historical.model_copy(
            update={"meta": current.meta, "meta_present": current.meta_present}
        )

    # -- writing -----------------------------------------------------------

    def add(
        self,
        spec: StrategySpec,
        *,
        name: str,
        author: str,
        source: str | None = None,
        tags: Sequence[str] = (),
        notes: str | None = None,
    ) -> StrategyRecord:
        """Put a new strategy in the library.

        Args:
            spec: The strategy.
            name: Human-readable name.
            author: Who owns it.
            source: Where the idea came from.
            tags: Labels the index filters on.
            notes: Prose about the entry.

        Returns:
            The stored record, at :attr:`Status.DRAFT` — a new entry is
            ungraded by definition, and no argument moves it, because every
            other stage needs evidence this method has not been given.

        Raises:
            ValidationError: If a strategy with this id already exists.
        """
        if self._find_spec_path(spec.id) is not None:
            raise ValidationError(f"strategy {spec.id!r} already in the library; use update()")

        digest = spec_digest(spec)
        meta = StrategyMeta(
            id=spec.id,
            name=name,
            author=author,
            source=source,
            tags=tuple(tags),
            notes=notes,
        ).publishing(
            VersionRecord(version=spec.version, spec_digest=digest, at=_now(), note="added")
        )
        self._write(spec, meta)
        return self._read_record(self.spec_path(spec.type, spec.id))

    def update(self, spec: StrategySpec, *, note: str | None = None) -> StrategyRecord:
        """Publish a new version of an existing strategy, archiving the old one.

        Args:
            spec: The new spec. Its version must differ from the stored one and
                must not have been published before.
            note: What changed.

        Returns:
            The stored record.

        Raises:
            KeyError: If the strategy is not in the library.
            ValidationError: If the version was not bumped, if it repeats a
                published version, or if the type changed — a different holding
                class is a different strategy, not a new version of this one.
        """
        current = self.get(spec.id)
        if spec.type is not current.spec.type:
            raise ValidationError(
                f"strategy {spec.id!r} is {current.spec.type.value}; a change to "
                f"{spec.type.value} is a different strategy, not a new version"
            )
        if spec.version == current.spec.version:
            raise ValidationError(
                f"strategy {spec.id!r} is already at version {spec.version}; bump it to update"
            )
        published = {record.version for record in current.meta.versions}
        if spec.version in published:
            raise ValidationError(
                f"strategy {spec.id!r} already published version {spec.version}; "
                "history is append-only"
            )

        history = self.history_dir(current.spec.type, spec.id)
        history.mkdir(parents=True, exist_ok=True)
        archived = history / f"{current.spec.version}.json"
        archived.write_text(
            current.spec.model_dump_json(indent=2, exclude_none=True) + "\n", encoding="utf-8"
        )

        meta = current.meta.publishing(
            VersionRecord(version=spec.version, spec_digest=spec_digest(spec), at=_now(), note=note)
        )
        self._write(spec, meta)
        return self._read_record(self.spec_path(spec.type, spec.id))

    def set_status(
        self,
        strategy_id: str,
        status: Status,
        *,
        reason: str | None = None,
        run_id: str | None = None,
        selector_key: str | None = None,
        verdict: str | None = None,
    ) -> StrategyRecord:
        """Record a stage transition.

        Args:
            strategy_id: The strategy.
            status: The stage to enter.
            reason: Why. Required to retire.
            run_id: Run or walk-forward that justified an approval.
            selector_key: Parameter selector that run used.
            verdict: Verdict recorded; must be ``ROBUST`` to approve.

        Returns:
            The stored record.

        Raises:
            KeyError: If the strategy is not in the library.
            ValidationError: If the transition lacks the evidence it requires —
                raised by :class:`LifecycleEvent` itself, so no caller can
                assemble an approval without provenance.
        """
        current = self.get(strategy_id)
        try:
            event = LifecycleEvent(
                status=status,
                at=_now(),
                spec_digest=current.digest,
                reason=reason,
                run_id=run_id,
                selector_key=selector_key,
                verdict=verdict,
            )
        except Exception as error:  # noqa: BLE001 - re-raised as the project's own type
            raise ValidationError(f"strategy {strategy_id!r}: {error}") from error
        self._write(current.spec, current.meta.appending(event))
        return self.get(strategy_id)

    def approve(
        self, strategy_id: str, *, run_id: str, selector_key: str, verdict: str
    ) -> StrategyRecord:
        """Move a strategy to :attr:`Status.APPROVED`.

        Args:
            strategy_id: The strategy.
            run_id: The walk-forward (or run) that produced the verdict.
            selector_key: The parameter selector that run used — an optimising
                run endorses the pair (spec, selector), never the spec's own
                parameter values, which it overwrote per fold.
            verdict: The verdict. Anything but ``ROBUST`` is refused.

        Returns:
            The stored record.

        Raises:
            ValidationError: If the verdict is not ``ROBUST``.
        """
        return self.set_status(
            strategy_id,
            Status.APPROVED,
            run_id=run_id,
            selector_key=selector_key,
            verdict=verdict,
        )

    def retire(self, strategy_id: str, reason: str) -> StrategyRecord:
        """Move a strategy to :attr:`Status.RETIRED`.

        Args:
            strategy_id: The strategy.
            reason: Why it is being retired. Required.

        Returns:
            The stored record.
        """
        return self.set_status(strategy_id, Status.RETIRED, reason=reason)

    def _write(self, spec: StrategySpec, meta: StrategyMeta) -> None:
        """Write a spec and its bookkeeping side by side.

        Args:
            spec: The strategy.
            meta: Its bookkeeping.
        """
        directory = self._library / spec.type.value
        directory.mkdir(parents=True, exist_ok=True)
        self.spec_path(spec.type, spec.id).write_text(
            spec.model_dump_json(indent=2, exclude_none=True) + "\n", encoding="utf-8"
        )
        self.meta_path(spec.type, spec.id).write_text(
            meta.model_dump_json(indent=2, exclude_none=True) + "\n", encoding="utf-8"
        )

    # -- querying ----------------------------------------------------------

    def index(
        self, connection: "duckdb.DuckDBPyConnection | None" = None
    ) -> "duckdb.DuckDBPyConnection":
        """Build the DuckDB index over the library.

        Rebuilt from the files rather than maintained, so it cannot be stale.

        Args:
            connection: Connection to build into. A fresh in-memory one by
                default.

        Returns:
            A connection holding a ``strategies`` table.
        """
        connection = connection if connection is not None else duckdb.connect(":memory:")
        connection.execute("DROP TABLE IF EXISTS strategies")
        connection.execute(
            """
            CREATE TABLE strategies (
                id VARCHAR,
                version VARCHAR,
                spec_digest VARCHAR,
                type VARCHAR,
                status VARCHAR,
                name VARCHAR,
                author VARCHAR,
                tags VARCHAR[],
                regimes VARCHAR[],
                instrument_classes VARCHAR[],
                symbols VARCHAR[],
                denied_symbols VARCHAR[],
                exit_ref VARCHAR,
                meta_present BOOLEAN,
                path VARCHAR
            )
            """
        )
        rows = [
            (
                record.id,
                record.spec.version,
                record.digest,
                record.spec.type.value,
                record.status.value,
                record.meta.name,
                record.meta.author,
                list(record.meta.tags),
                [regime.value for regime in record.spec.market_regimes],
                [cls.value for cls in record.spec.instruments.allowed_classes],
                list(record.spec.instruments.allowed_symbols),
                list(record.spec.instruments.denied_symbols),
                record.spec.exit_ref,
                record.meta_present,
                str(record.path),
            )
            for record in self.records()
        ]
        if rows:
            connection.executemany(
                "INSERT INTO strategies VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows
            )
        return connection

    def list(
        self,
        *,
        type: StrategyType | None = None,
        status: Status | Collection[Status] | None = None,
        regime: Regime | None = None,
        instrument: str | None = None,
        instrument_class: InstrumentClass | None = None,
        tag: str | None = None,
        author: str | None = None,
    ) -> list[StrategyRecord]:
        """Entries matching every filter given.

        Args:
            type: Holding-period class.
            status: One stage or several.
            regime: Keep entries permitted in this regime. An entry with no
                declared regimes is unrestricted and therefore matches, which
                is what the empty list means in the spec.
            instrument: Keep entries that may trade this symbol — allowed
                explicitly or by an unrestricted allow-list, and not denied.
            instrument_class: Keep entries permitted on this asset class.
            tag: Keep entries carrying this tag.
            author: Keep entries owned by this author.

        Returns:
            Matching records, ordered by id. Filtering on ``Sharpe`` and other
            measured quantities lives in
            :mod:`trading_system.strategies.results_link`, which is where
            results are: a strategy filter that silently needed a result store
            would be a filter that answers differently depending on what has
            been run.
        """
        wanted = (
            {status} if isinstance(status, Status) else set(status) if status is not None else None
        )
        out = []
        for record in self.records():
            if type is not None and record.spec.type is not type:
                continue
            if wanted is not None and record.status not in wanted:
                continue
            # An empty market_regimes means unrestricted in the spec, so it must
            # match every regime here rather than none.
            if (
                regime is not None
                and record.spec.market_regimes
                and regime not in record.spec.market_regimes
            ):
                continue
            if (
                instrument_class is not None
                and instrument_class not in record.spec.instruments.allowed_classes
            ):
                continue
            if instrument is not None and not _may_trade(record.spec, instrument):
                continue
            if tag is not None and tag not in record.meta.tags:
                continue
            if author is not None and record.meta.author != author:
                continue
            out.append(record)
        return out

    def diff(self, strategy_id: str, left: str, right: str) -> str:
        """Human-readable difference between two versions of a spec.

        Compares the parsed specs field by field rather than the JSON text, so
        reordered keys and formatting do not show up as changes and a nested
        edit is named by its path.

        Args:
            strategy_id: The strategy.
            left: Version on the left.
            right: Version on the right.

        Returns:
            One line per differing path, plus a header. A note that nothing
            differs when the two specs are identical.

        Raises:
            KeyError: If either version is unknown.
        """
        before = self.get(strategy_id, left).spec
        after = self.get(strategy_id, right).spec
        lines = [f"{strategy_id}: {left} -> {right}"]
        changes = list(
            _diff_values(
                before.model_dump(mode="json"),
                after.model_dump(mode="json"),
                path="",
            )
        )
        if not changes:
            lines.append("  (specs are identical)")
            return "\n".join(lines)
        lines.extend(f"  {line}" for line in changes)
        return "\n".join(lines)


def _now() -> datetime:
    """The current instant, UTC and tz-aware.

    Returns:
        Now.
    """
    return datetime.now(UTC)


def _may_trade(spec: StrategySpec, symbol: str) -> bool:
    """Whether a spec's instrument scope permits ``symbol``.

    Args:
        spec: The strategy.
        symbol: Instrument identifier.

    Returns:
        True when the symbol is not denied and is either named explicitly or
        left to the class-level allowance.
    """
    scope = spec.instruments
    if symbol in scope.denied_symbols:
        return False
    if scope.allowed_symbols:
        return symbol in scope.allowed_symbols
    return True


def _diff_values(before: Any, after: Any, *, path: str) -> Iterator[str]:
    """Walk two JSON-able structures, naming every leaf that differs.

    Args:
        before: Left value.
        after: Right value.
        path: Dotted path to this point, ``""`` at the root.

    Yields:
        One line per difference, deepest path named.
    """
    if isinstance(before, dict) and isinstance(after, dict):
        for key in sorted(set(before) | set(after)):
            here = f"{path}.{key}" if path else key
            if key not in before:
                yield f"+ {here}: {json.dumps(after[key], ensure_ascii=False)}"
            elif key not in after:
                yield f"- {here}: {json.dumps(before[key], ensure_ascii=False)}"
            else:
                yield from _diff_values(before[key], after[key], path=here)
        return
    if isinstance(before, list) and isinstance(after, list):
        for index in range(max(len(before), len(after))):
            here = f"{path}[{index}]"
            if index >= len(before):
                yield f"+ {here}: {json.dumps(after[index], ensure_ascii=False)}"
            elif index >= len(after):
                yield f"- {here}: {json.dumps(before[index], ensure_ascii=False)}"
            else:
                yield from _diff_values(before[index], after[index], path=here)
        return
    if before != after:
        yield (
            f"~ {path or '(root)'}: "
            f"{json.dumps(before, ensure_ascii=False)} -> {json.dumps(after, ensure_ascii=False)}"
        )
