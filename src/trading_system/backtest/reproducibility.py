"""Run identity: what a run was, and proof that it produced what it produced.

Two digests, and they answer different questions.

:attr:`RunManifest.run_id` digests the **inputs** — every bar of every stream,
the code that walked them, the configs, the strategies, the exit presets, the
instrument specifications and the seed. Two runs sharing a ``run_id`` were asked
the same question. :func:`result_digest` digests the **output** — the equity
curve, the trades and every counter the run kept. Two runs sharing a ``run_id``
and disagreeing on ``result_digest`` are the finding this module exists to
surface, and :func:`write_run` raises rather than storing the second one.

**The code is part of the input, and the commit alone is not the code.** A
working tree with uncommitted edits has the same commit as the tree without
them, so the authority here is ``source_digest`` — a hash of every ``.py`` file
under the package — and the commit is recorded alongside it as provenance a
human can look up. A run id that only tracked the commit would collide across
precisely the edits somebody is most likely to be in the middle of.

**Bars are digested by their values, not by a serialisation format.** Parquet and
Arrow IPC both embed writer metadata and both are free to change layout between
versions, which would silently change every run id on an unrelated dependency
bump. Rows are packed here explicitly — an int64 of epoch microseconds and five
IEEE doubles — so the digest depends on the numbers and on nothing else.

**Floats are digested through :meth:`float.hex`.** ``0.1`` and the double nearest
to it print identically at repr precision on some paths and not others; the hex
form is exact and platform-stable, and a run id is worth nothing if it collides
across two configurations that differ in the last bit of a coefficient.

**A component the digest cannot describe is an error, never a fallback.**
:func:`canonical` accepts pydantic models, dataclasses, containers and objects
digested by their own attribute state; anything else raises. The alternative —
falling back to :func:`repr` or to the type name — is a run id that is stable
across a change it does not see, which is worse than no run id at all.
"""

import json
import struct
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from datetime import date, datetime, time
from decimal import Decimal
from enum import Enum
from hashlib import blake2b
from pathlib import Path
from types import BuiltinFunctionType, FunctionType, MethodType, ModuleType
from typing import Any

import polars as pl
from pydantic import BaseModel

from trading_system.backtest.clock import StreamKey
from trading_system.backtest.orchestrator import (
    AtrRatioCoverage,
    BacktestResult,
    SignalDrop,
    StrategyBinding,
)
from trading_system.backtest.portfolio import EquityPoint, TradeRecord
from trading_system.core.exceptions import TradingSystemError
from trading_system.core.instruments import InstrumentRegistry
from trading_system.core.types import Price, Side, Timeframe
from trading_system.data.models import OHLCVFrame
from trading_system.exit.base import ExitDropReason

#: Length of every digest this module produces, in bytes. Sixteen is far past
#: any collision risk for a run store and short enough to be a directory name.
_DIGEST_BYTES = 16

#: Package whose sources are hashed into the code version.
_PACKAGE_ROOT = Path(__file__).resolve().parent.parent

#: Struct layout of one bar: epoch microseconds, then OHLCV as doubles. Little
#: endian and standard sizes are named explicitly so the digest does not depend
#: on the machine it is computed on.
_BAR_LAYOUT = struct.Struct("<q5d")

#: Filenames inside a stored run.
MANIFEST_FILE = "manifest.json"
RESULT_FILE = "result.json"
CURVE_FILE = "curve.parquet"
TRADES_FILE = "trades.parquet"


class ReproducibilityError(TradingSystemError):
    """A run id was reused by a run that produced something else.

    Raised rather than resolved. Overwriting would destroy the evidence, and
    keeping both under one id would make the id meaningless — the whole point of
    a run id is that it names one result.
    """


def _blake(payload: bytes) -> str:
    """Hash bytes into this module's fixed-width hex digest."""
    return blake2b(payload, digest_size=_DIGEST_BYTES).hexdigest()


def canonical(value: object) -> Any:
    """Describe a value as JSON-able data that captures everything it is.

    Args:
        value: The thing to describe.

    Returns:
        Data built from ``dict``, ``list``, ``str``, ``int``, ``bool`` and
        ``None`` alone, in which every float is its exact hexadecimal form and
        every type is named, so two differently-typed values carrying the same
        payload do not describe alike.

    Raises:
        TypeError: If the value cannot be described. Deliberately fatal: a
            fallback description is one that fails to notice a change.
    """
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        # Exact and platform-stable. repr() round-trips in CPython but is not a
        # guarantee this module wants to depend on for run identity.
        return {"__float__": float(value).hex()}
    if isinstance(value, Decimal):
        return {"__decimal__": str(value)}
    if isinstance(value, datetime | date | time):
        return {"__time__": value.isoformat()}
    if isinstance(value, Enum):
        return {"__enum__": f"{type(value).__name__}.{value.name}", "value": canonical(value.value)}
    if isinstance(value, Path):
        return {"__path__": str(value)}
    if isinstance(value, BaseModel):
        return {"__model__": type(value).__name__, "fields": canonical(dict(value))}
    if isinstance(value, OHLCVFrame):
        return {"__frame__": digest_frame(value)}
    if isinstance(value, Mapping):
        items = [(canonical(key), canonical(item)) for key, item in value.items()]
        return {"__map__": sorted(items, key=lambda pair: json.dumps(pair[0], sort_keys=True))}
    if isinstance(value, frozenset | set):
        described = [canonical(item) for item in value]
        return {"__set__": sorted(described, key=lambda item: json.dumps(item, sort_keys=True))}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return [canonical(item) for item in value]
    if is_dataclass(value) and not isinstance(value, type):
        by_field = {field.name: canonical(getattr(value, field.name)) for field in fields(value)}
        return {"__dataclass__": type(value).__name__, "fields": by_field}
    return _canonical_object(value)


def _canonical_object(value: object) -> Any:
    """Describe a plain object by its own attribute state.

    Complete by construction, which :func:`repr` is not: every attribute the
    object holds is in the description, so a parameter somebody forgot to print
    is still a parameter that changes the run id.

    Args:
        value: The object.

    Returns:
        Its qualified type name, and its state when it has any.

    Raises:
        TypeError: If the value is a function, a method, a module or a class —
            things whose behaviour is not in their attributes, so a description
            built from those attributes would call two different ones the same.
    """
    if isinstance(value, FunctionType | BuiltinFunctionType | MethodType | ModuleType | type):
        raise TypeError(
            f"{value!r} cannot be described for a run id: its behaviour is not in its "
            "attributes. Pass a config object or a plain description of what it does."
        )
    name = f"{type(value).__module__}.{type(value).__qualname__}"
    state: dict[str, Any] = {}
    for attribute in _state_attributes(type(value)):
        if hasattr(value, attribute):
            state[attribute] = canonical(getattr(value, attribute))
    # __dict__ as well as __slots__: a class can declare slots and still carry a
    # dict, and an attribute in either is state this description must not miss.
    instance_dict = getattr(value, "__dict__", None)
    if instance_dict is not None:
        state.update({key: canonical(item) for key, item in instance_dict.items()})
    if not state:
        return {"__object__": name}
    return {"__object__": name, "state": dict(sorted(state.items()))}


def _state_attributes(cls: type) -> tuple[str, ...]:
    """Every ``__slots__`` name declared by a class or its bases."""
    names: list[str] = []
    for base in cls.__mro__:
        slots = getattr(base, "__slots__", ())
        if isinstance(slots, str):
            names.append(slots)
        else:
            names.extend(slots)
    return tuple(names)


def digest(value: object) -> str:
    """Digest anything :func:`canonical` can describe.

    Args:
        value: The thing to digest.

    Returns:
        A hex digest that changes whenever the description does.
    """
    payload = json.dumps(canonical(value), sort_keys=True, separators=(",", ":"))
    return _blake(payload.encode("utf-8"))


def digest_frame(frame: OHLCVFrame) -> str:
    """Digest one stream's bars by their values.

    Args:
        frame: The bars.

    Returns:
        A hex digest over the symbol, the timeframe and every row — epoch
        microseconds and the five OHLCV doubles — independent of how polars or
        parquet happen to lay the data out.
    """
    hasher = blake2b(digest_size=_DIGEST_BYTES)
    hasher.update(f"{frame.symbol}\x00{frame.timeframe.value}\x00".encode())
    table = frame.df
    stamps = table["timestamp"].dt.epoch("us").to_list()
    columns = [table[name].to_list() for name in ("open", "high", "low", "close", "volume")]
    for index, stamp in enumerate(stamps):
        hasher.update(_BAR_LAYOUT.pack(stamp, *(column[index] for column in columns)))
    return hasher.hexdigest()


@dataclass(frozen=True)
class CodeVersion:
    """Which code walked the bars.

    Attributes:
        source_digest: Hash of every ``.py`` file under the package. The
            authority — it is what actually ran.
        commit: Git commit the tree is at, or ``None`` outside a repository.
            Provenance for a human, not identity: an edited tree has the commit
            of the tree it was edited from.
        dirty: Whether the tree had uncommitted changes when the run was built.
    """

    source_digest: str
    commit: str | None
    dirty: bool


def code_version(root: Path | None = None) -> CodeVersion:
    """Fingerprint the code as it is on disk right now.

    Args:
        root: Package directory to hash. The installed package by default.

    Returns:
        The version. ``commit`` is ``None`` when git is unavailable or the tree
        is not a repository — recorded honestly rather than substituted.
    """
    package = root if root is not None else _PACKAGE_ROOT
    hasher = blake2b(digest_size=_DIGEST_BYTES)
    for path in sorted(package.rglob("*.py")):
        hasher.update(str(path.relative_to(package)).encode("utf-8"))
        hasher.update(b"\x00")
        hasher.update(path.read_bytes())
    commit, dirty = _git_state(package)
    return CodeVersion(source_digest=hasher.hexdigest(), commit=commit, dirty=dirty)


def _git_state(package: Path) -> tuple[str | None, bool]:
    """The commit and dirtiness of the tree containing ``package``.

    Failures are answered with ``(None, False)`` rather than raised: a run in an
    exported tree with no git available is a legitimate run, and its identity
    rests on ``source_digest`` regardless.
    """

    def ask(*args: str) -> str | None:
        try:
            done = subprocess.run(  # noqa: S603 - fixed argv, no shell
                ["git", *args],
                cwd=package,
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return done.stdout if done.returncode == 0 else None

    commit = ask("rev-parse", "HEAD")
    if commit is None:
        return None, False
    status = ask("status", "--porcelain")
    return commit.strip(), bool(status and status.strip())


@dataclass(frozen=True)
class RunManifest:
    """Everything a run was given, digested.

    Attributes:
        code: The code that ran.
        seed: The run seed the per-fill random streams are derived from.
        config: Digest of the :class:`~trading_system.backtest.config.BacktestConfig`.
        data: Digest per stream, keyed by ``"SYMBOL@TF"``.
        strategies: Digest per binding — the strategy spec, its exit preset and
            the streams it trades — keyed by strategy id.
        instruments: Digest of the instrument specifications the run priced with.
        components: Digest per named component the caller declared: the cost
            config, the sizing method, the converter, the risk limits. Named by
            the caller because only the caller knows what it wired together; a
            component left out is a component two runs can differ in while
            sharing an id, which is why :meth:`build` takes the mapping
            explicitly rather than defaulting it to empty.
    """

    code: CodeVersion
    seed: int
    config: str
    data: Mapping[str, str]
    strategies: Mapping[str, str]
    instruments: str
    components: Mapping[str, str]

    @classmethod
    def build(
        cls,
        *,
        config: object,
        streams: Mapping[StreamKey, OHLCVFrame],
        bindings: Sequence[StrategyBinding],
        instruments: InstrumentRegistry,
        seed: int,
        components: Mapping[str, object],
        code: CodeVersion | None = None,
    ) -> "RunManifest":
        """Digest one run's inputs.

        Args:
            config: The run config.
            streams: Bars per stream.
            bindings: Strategies, exit presets and the streams they trade.
            instruments: Contract specifications.
            seed: The run seed.
            components: Everything else the run was wired with, by name — cost
                config, sizing method, converter, risk limits. Required, with no
                default: an empty mapping must be a statement, not an omission.
            code: Code version. Measured from disk when not given.

        Returns:
            The manifest.
        """
        return cls(
            code=code if code is not None else code_version(),
            seed=seed,
            config=digest(config),
            data={str(key): digest_frame(frame) for key, frame in sorted(streams.items())},
            strategies={
                binding.spec.id: digest(
                    {
                        "spec": binding.spec,
                        "exit_preset": binding.exit_preset,
                        "keys": [str(key) for key in binding.keys],
                    }
                )
                for binding in bindings
            },
            instruments=digest(
                {symbol: instruments[symbol] for symbol in sorted(instruments.symbols)}
            ),
            components={name: digest(item) for name, item in sorted(components.items())},
        )

    @property
    def run_id(self) -> str:
        """Identifier of the question this run asked.

        Returns:
            A hex digest over every field. Two runs sharing it were given the
            same bars, the same code, the same configs and the same seed — and
            are therefore required to produce the same result.
        """
        return digest(self)

    def to_dict(self) -> dict[str, Any]:
        """Serialise, with the run id alongside so a stored run names itself.

        Returns:
            JSON-able data.
        """
        return {
            "run_id": self.run_id,
            "code": {
                "source_digest": self.code.source_digest,
                "commit": self.code.commit,
                "dirty": self.code.dirty,
            },
            "seed": self.seed,
            "config": self.config,
            "data": dict(self.data),
            "strategies": dict(self.strategies),
            "instruments": self.instruments,
            "components": dict(self.components),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RunManifest":
        """Rebuild a manifest from :meth:`to_dict` output.

        Args:
            payload: The stored mapping.

        Returns:
            The manifest. The stored ``run_id`` is not read back — it is
            recomputed from the fields, so a hand-edited file is caught rather
            than believed.
        """
        code = payload["code"]
        return cls(
            code=CodeVersion(
                source_digest=code["source_digest"],
                commit=code["commit"],
                dirty=bool(code["dirty"]),
            ),
            seed=int(payload["seed"]),
            config=payload["config"],
            data=dict(payload["data"]),
            strategies=dict(payload["strategies"]),
            instruments=payload["instruments"],
            components=dict(payload["components"]),
        )


def result_digest(result: BacktestResult) -> str:
    """Digest everything a run produced.

    Args:
        result: The run's output.

    Returns:
        A hex digest over the equity curve, the trades and **every counter** —
        the drops, the degradations and the ATR coverage included. A digest over
        the curve alone would call two runs identical when one of them silently
        discarded half its signals.
    """
    return digest(result_payload(result))


def result_payload(result: BacktestResult) -> dict[str, Any]:
    """The whole result as JSON-able data, curve and trades included.

    Args:
        result: The run's output.

    Returns:
        Data covering every field of :class:`~trading_system.backtest.orchestrator.BacktestResult`.
    """
    return {
        "curve": [_curve_row(point) for point in result.curve],
        "trades": [_trade_row(trade) for trade in result.trades],
        **result_counters(result),
    }


def result_counters(result: BacktestResult) -> dict[str, Any]:
    """Every field of a result except the curve and the trade list.

    Split out because these are what a human reads and what the run store keeps
    in JSON, while the two tables go to parquet.

    Args:
        result: The run's output.

    Returns:
        The counters, JSON-able.
    """
    return {
        "instants": result.instants,
        "rejections": dict(result.rejections),
        "degradations": dict(result.degradations),
        "exit_drops": {reason.value: count for reason, count in result.exit_drops.items()},
        "entry_drops": dict(result.entry_drops),
        "signal_drops": {reason.value: count for reason, count in result.signal_drops.items()},
        "expired_orders": result.expired_orders,
        "cost_degradations": {
            reason: float(value) for reason, value in result.cost_degradations.items()
        },
        "atr_ratio_coverage": {
            str(key): {"bars": coverage.bars, "unavailable": coverage.unavailable}
            for key, coverage in result.atr_ratio_coverage.items()
        },
        "fills": result.fills,
        "fx_fallback_marks": result.fx_fallback_marks,
        "open_at_end": result.open_at_end,
    }


def _curve_row(point: EquityPoint) -> dict[str, Any]:
    """One equity row as storable values: money as exact decimal strings."""
    return {
        "ts": point.ts,
        "day": point.day,
        "balance": str(point.balance),
        "equity": str(point.equity),
        "realized": str(point.realized),
        "unrealized": str(point.unrealized),
        "commission_paid": str(point.commission_paid),
        "swap_paid": str(point.swap_paid),
        "open_positions": point.open_positions,
    }


def _trade_row(trade: TradeRecord) -> dict[str, Any]:
    """One trade as storable values.

    Money is written as its exact decimal string, never as a float: a run store
    that round-trips ``Decimal`` through binary floating point stops being able
    to prove that two runs agree exactly, which is the only thing it is for.
    ``entry_price`` and ``realized_r`` are prices and ratios, and stay floats.
    """
    return {
        "position_id": trade.position_id,
        "symbol": trade.symbol,
        "strategy_id": trade.strategy_id,
        "side": trade.side.value,
        "size": str(trade.size),
        "opened_at": trade.opened_at,
        "closed_at": trade.closed_at,
        "entry_price": float(trade.entry_price),
        "gross": str(trade.gross),
        "commission": str(trade.commission),
        "swap": str(trade.swap),
        "net": str(trade.net),
        "realized_r": trade.realized_r,
        "legs": trade.legs,
    }


@dataclass(frozen=True)
class StoredRun:
    """A run read back off disk.

    Attributes:
        manifest: What the run was given.
        result: What it produced, rebuilt exactly.
        digest: Digest of the result as it was stored.
    """

    manifest: RunManifest
    result: BacktestResult
    digest: str


def run_directory(root: Path, run_id: str) -> Path:
    """Where one run's artefacts live.

    Args:
        root: Store root, conventionally ``runs/``.
        run_id: The run's id.

    Returns:
        ``root / run_id``.
    """
    return root / run_id


def write_run(root: Path, manifest: RunManifest, result: BacktestResult) -> Path:
    """Store a run's artefacts under its own id.

    Idempotent for a reproducible run: writing the same result under the same id
    twice is a no-op. Writing a *different* result under an id that already
    exists is the failure this module exists to catch, and it raises — the
    stored evidence is not overwritten, because it is the half of the
    contradiction that already survived.

    Args:
        root: Store root, conventionally ``runs/``.
        manifest: What the run was given.
        result: What it produced.

    Returns:
        The run's directory.

    Raises:
        ReproducibilityError: If the id already holds a different result.
    """
    directory = run_directory(root, manifest.run_id)
    produced = result_digest(result)
    stored = directory / RESULT_FILE
    if stored.exists():
        previous = json.loads(stored.read_text())
        if previous["digest"] != produced:
            raise ReproducibilityError(
                f"run {manifest.run_id} already holds result {previous['digest']}, and this "
                f"run produced {produced}. Identical inputs produced different output: the "
                "run is not reproducible, and the stored result has been kept."
            )
        return directory

    directory.mkdir(parents=True, exist_ok=True)
    (directory / MANIFEST_FILE).write_text(json.dumps(manifest.to_dict(), indent=2, default=str))
    _curve_frame(result).write_parquet(directory / CURVE_FILE)
    _trades_frame(result).write_parquet(directory / TRADES_FILE)
    stored.write_text(
        json.dumps({"digest": produced, **result_counters(result)}, indent=2, default=str)
    )
    return directory


def read_run(directory: Path) -> StoredRun:
    """Read a stored run back.

    Args:
        directory: The run's directory.

    Returns:
        The manifest, the result rebuilt field for field, and the digest as it
        was stored.

    Raises:
        ReproducibilityError: If the rebuilt result does not hash to the stored
            digest — the artefacts disagree with each other and neither can be
            trusted over the other.
    """
    manifest = RunManifest.from_dict(json.loads((directory / MANIFEST_FILE).read_text()))
    counters = json.loads((directory / RESULT_FILE).read_text())
    result = BacktestResult(
        curve=[
            _curve_point(row)
            for row in pl.read_parquet(directory / CURVE_FILE).iter_rows(named=True)
        ],
        trades=[
            _trade_record(row)
            for row in pl.read_parquet(directory / TRADES_FILE).iter_rows(named=True)
        ],
        instants=int(counters["instants"]),
        rejections=dict(counters["rejections"]),
        degradations=dict(counters["degradations"]),
        exit_drops={
            ExitDropReason(name): int(count) for name, count in counters["exit_drops"].items()
        },
        entry_drops=dict(counters["entry_drops"]),
        signal_drops={
            SignalDrop(name): int(count) for name, count in counters["signal_drops"].items()
        },
        expired_orders=int(counters["expired_orders"]),
        cost_degradations={
            name: float(value) for name, value in counters["cost_degradations"].items()
        },
        atr_ratio_coverage={
            _stream_key(name): AtrRatioCoverage(
                bars=int(entry["bars"]), unavailable=int(entry["unavailable"])
            )
            for name, entry in counters["atr_ratio_coverage"].items()
        },
        fills=int(counters["fills"]),
        fx_fallback_marks=int(counters["fx_fallback_marks"]),
        open_at_end=int(counters["open_at_end"]),
    )
    rebuilt = result_digest(result)
    if rebuilt != counters["digest"]:
        raise ReproducibilityError(
            f"{directory}: the stored tables hash to {rebuilt}, but the stored digest is "
            f"{counters['digest']}. The artefacts of this run disagree with each other."
        )
    return StoredRun(manifest=manifest, result=result, digest=counters["digest"])


def _stream_key(name: str) -> StreamKey:
    """Parse ``"EURUSD@H1"`` back into a stream key."""
    symbol, _, timeframe = name.partition("@")
    return StreamKey(symbol, Timeframe(timeframe))


def _curve_frame(result: BacktestResult) -> pl.DataFrame:
    """The equity curve as a table, with an explicit schema for an empty run."""
    rows = [_curve_row(point) for point in result.curve]
    return pl.DataFrame(
        rows,
        schema={
            "ts": pl.Datetime("us", "UTC"),
            "day": pl.Date,
            "balance": pl.String,
            "equity": pl.String,
            "realized": pl.String,
            "unrealized": pl.String,
            "commission_paid": pl.String,
            "swap_paid": pl.String,
            "open_positions": pl.Int64,
        },
    )


def _trades_frame(result: BacktestResult) -> pl.DataFrame:
    """The trade list as a table, with an explicit schema for a run with none."""
    rows = [_trade_row(trade) for trade in result.trades]
    return pl.DataFrame(
        rows,
        schema={
            "position_id": pl.String,
            "symbol": pl.String,
            "strategy_id": pl.String,
            "side": pl.String,
            "size": pl.String,
            "opened_at": pl.Datetime("us", "UTC"),
            "closed_at": pl.Datetime("us", "UTC"),
            "entry_price": pl.Float64,
            "gross": pl.String,
            "commission": pl.String,
            "swap": pl.String,
            "net": pl.String,
            "realized_r": pl.Float64,
            "legs": pl.Int64,
        },
    )


def _curve_point(row: Mapping[str, Any]) -> EquityPoint:
    """Rebuild one equity row from storage."""
    return EquityPoint(
        ts=row["ts"],
        day=row["day"],
        balance=Decimal(row["balance"]),
        equity=Decimal(row["equity"]),
        realized=Decimal(row["realized"]),
        unrealized=Decimal(row["unrealized"]),
        commission_paid=Decimal(row["commission_paid"]),
        swap_paid=Decimal(row["swap_paid"]),
        open_positions=int(row["open_positions"]),
    )


def _trade_record(row: Mapping[str, Any]) -> TradeRecord:
    """Rebuild one trade from storage."""
    return TradeRecord(
        position_id=row["position_id"],
        symbol=row["symbol"],
        strategy_id=row["strategy_id"],
        side=Side(row["side"]),
        size=Decimal(row["size"]),
        opened_at=row["opened_at"],
        closed_at=row["closed_at"],
        entry_price=Price(row["entry_price"]),
        gross=Decimal(row["gross"]),
        commission=Decimal(row["commission"]),
        swap=Decimal(row["swap"]),
        net=Decimal(row["net"]),
        realized_r=float(row["realized_r"]),
        legs=int(row["legs"]),
    )
