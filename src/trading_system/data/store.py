"""Local market-data store: Parquet on disk, DuckDB for querying.

On-disk layout::

    {root}/ohlcv/{symbol}/{timeframe}/{year}.parquet

Year partitioning keeps a re-download of one period from rewriting a decade of
history, and lets range queries skip files by name before DuckDB opens them.

Writes are idempotent: :meth:`ParquetStore.upsert` merges incoming bars with what
is already stored, keyed on timestamp, so re-importing an overlapping period
corrects the affected bars instead of duplicating them.
"""

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import polars as pl

from trading_system.core.exceptions import DataError
from trading_system.core.types import Timeframe
from trading_system.data.models import TIMESTAMP_COLUMN, OHLCVFrame

_SAFE_SYMBOL = re.compile(r"^[A-Za-z0-9._=-]+$")


@dataclass(frozen=True)
class Coverage:
    """What the store holds for one symbol and timeframe.

    Attributes:
        symbol: Instrument identifier.
        timeframe: Bar size.
        start: Open time of the earliest stored bar.
        end: Open time of the latest stored bar.
        bars: Total number of stored bars.
    """

    symbol: str
    timeframe: Timeframe
    start: datetime
    end: datetime
    bars: int


class ParquetStore:
    """A directory of Parquet partitions queried through DuckDB."""

    def __init__(self, root: Path) -> None:
        """Bind the store to a root directory.

        Args:
            root: Directory holding the ``ohlcv`` tree. Created on first write.
        """
        self._root = root

    @property
    def root(self) -> Path:
        """Root directory of the store."""
        return self._root

    def directory_for(self, symbol: str, timeframe: Timeframe) -> Path:
        """Return the partition directory for a symbol and timeframe.

        Args:
            symbol: Instrument identifier.
            timeframe: Bar size.

        Returns:
            Path to the directory, which may not exist yet.

        Raises:
            DataError: If ``symbol`` contains characters that are unsafe in a
                path, which would let a crafted name escape the store root.
        """
        if not _SAFE_SYMBOL.match(symbol):
            raise DataError(
                f"unsafe symbol {symbol!r}: expected only letters, digits, and '.', '_', '=', '-'"
            )
        return self._root / "ohlcv" / symbol / str(timeframe)

    def partition_path(self, symbol: str, timeframe: Timeframe, year: int) -> Path:
        """Return the Parquet file holding one year of data."""
        return self.directory_for(symbol, timeframe) / f"{year}.parquet"

    def partitions(self, symbol: str, timeframe: Timeframe) -> list[Path]:
        """List existing partition files, oldest year first."""
        directory = self.directory_for(symbol, timeframe)
        if not directory.is_dir():
            return []
        return sorted(directory.glob("*.parquet"))

    def symbols(self) -> list[str]:
        """List symbols present in the store."""
        ohlcv = self._root / "ohlcv"
        if not ohlcv.is_dir():
            return []
        return sorted(entry.name for entry in ohlcv.iterdir() if entry.is_dir())

    def timeframes(self, symbol: str) -> list[Timeframe]:
        """List timeframes stored for ``symbol``."""
        directory = self._root / "ohlcv" / symbol
        if not directory.is_dir():
            return []
        stored = []
        for entry in sorted(directory.iterdir()):
            if entry.is_dir() and entry.name in Timeframe.__members__.values():
                stored.append(Timeframe(entry.name))
        return stored

    def upsert(self, frame: OHLCVFrame) -> int:
        """Merge ``frame`` into the store, overwriting bars at colliding timestamps.

        Incoming bars win on collision: a vendor re-issuing a corrected bar should
        replace the original rather than be discarded.

        Args:
            frame: Bars to store. Its symbol and timeframe select the partitions.

        Returns:
            Number of bars written across all touched partitions.
        """
        if frame.is_empty:
            return 0

        directory = self.directory_for(frame.symbol, frame.timeframe)
        directory.mkdir(parents=True, exist_ok=True)

        written = 0
        incoming = frame.df.with_columns(pl.col(TIMESTAMP_COLUMN).dt.year().alias("_year"))
        for (year,), chunk in incoming.group_by(["_year"], maintain_order=True):
            year_value = int(year)
            path = self.partition_path(frame.symbol, frame.timeframe, year_value)
            new_rows = chunk.drop("_year")
            if path.exists():
                existing = pl.read_parquet(path)
                merged = pl.concat([existing, new_rows], how="vertical")
            else:
                merged = new_rows
            merged = merged.sort(TIMESTAMP_COLUMN).unique(
                subset=[TIMESTAMP_COLUMN], keep="last", maintain_order=True
            )
            merged.write_parquet(path)
            written += merged.height
        return written

    def get(
        self,
        symbol: str,
        timeframe: Timeframe,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> OHLCVFrame:
        """Read a half-open ``[start, end)`` range out of the store.

        Args:
            symbol: Instrument identifier.
            timeframe: Bar size.
            start: Inclusive lower bound on bar open time, or ``None``.
            end: Exclusive upper bound on bar open time, or ``None``.

        Returns:
            The matching bars, empty when nothing is stored.

        Raises:
            ValueError: If a bound is tz-naive.
        """
        if start is not None and start.tzinfo is None:
            raise ValueError("start must be tz-aware")
        if end is not None and end.tzinfo is None:
            raise ValueError("end must be tz-aware")

        paths = self._relevant_partitions(symbol, timeframe, start, end)
        if not paths:
            return OHLCVFrame.empty(symbol, timeframe)

        # Row retrieval goes through polars rather than DuckDB: it reads Parquet
        # natively with predicate pushdown and returns a polars frame directly,
        # whereas DuckDB would need pyarrow purely as a format bridge. DuckDB is
        # retained for aggregate work in coverage() and query(), where SQL earns
        # its place.
        scan = pl.scan_parquet(paths)
        if start is not None:
            scan = scan.filter(pl.col(TIMESTAMP_COLUMN) >= start)
        if end is not None:
            scan = scan.filter(pl.col(TIMESTAMP_COLUMN) < end)
        result = scan.sort(TIMESTAMP_COLUMN).collect()

        if result.is_empty():
            return OHLCVFrame.empty(symbol, timeframe)
        return OHLCVFrame.from_raw(result, symbol, timeframe)

    def coverage(self, symbol: str, timeframe: Timeframe) -> Coverage | None:
        """Summarise what is stored for a symbol and timeframe.

        Args:
            symbol: Instrument identifier.
            timeframe: Bar size.

        Returns:
            Coverage summary, or ``None`` when nothing is stored.
        """
        paths = self.partitions(symbol, timeframe)
        if not paths:
            return None
        query = (
            f"SELECT CAST(min({TIMESTAMP_COLUMN}) AS TIMESTAMP), "
            f"CAST(max({TIMESTAMP_COLUMN}) AS TIMESTAMP), count(*) "
            "FROM read_parquet(?)"
        )
        with self._connect() as connection:
            row = connection.execute(query, [[str(path) for path in paths]]).fetchone()
        if row is None or row[2] == 0:
            return None
        return Coverage(
            symbol=symbol,
            timeframe=timeframe,
            start=row[0].replace(tzinfo=UTC),
            end=row[1].replace(tzinfo=UTC),
            bars=int(row[2]),
        )

    def query(self, sql: str, symbol: str, timeframe: Timeframe) -> list[tuple[object, ...]]:
        """Run an arbitrary SQL query against a symbol's partitions.

        The partition file list is bound to the first placeholder, so the query
        should read from ``read_parquet(?)``. Timestamps are returned as UTC.

        Args:
            sql: A DuckDB SQL statement containing one ``?`` placeholder.
            symbol: Instrument whose partitions the query runs over.
            timeframe: Bar size whose partitions the query runs over.

        Returns:
            Result rows as tuples.
        """
        paths = self.partitions(symbol, timeframe)
        with self._connect() as connection:
            return connection.execute(sql, [[str(path) for path in paths]]).fetchall()

    def _relevant_partitions(
        self,
        symbol: str,
        timeframe: Timeframe,
        start: datetime | None,
        end: datetime | None,
    ) -> list[Path]:
        """Select partitions whose year could intersect the requested range."""
        selected = []
        for path in self.partitions(symbol, timeframe):
            try:
                year = int(path.stem)
            except ValueError:
                continue  # not a year partition; ignore stray files
            if start is not None and year < start.year:
                continue
            if end is not None and year > end.year:
                continue
            selected.append(path)
        return selected

    def _connect(self) -> "duckdb.DuckDBPyConnection":
        """Open an in-memory DuckDB connection pinned to UTC.

        Pinning the session timezone keeps ``TIMESTAMP WITH TIME ZONE`` values
        round-tripping as UTC rather than the host's local zone.
        """
        connection = duckdb.connect(":memory:")
        connection.execute("SET TimeZone='UTC'")
        return connection
