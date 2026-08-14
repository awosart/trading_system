"""CSV provider with flexible column mapping.

Vendor CSVs disagree on nearly everything: column names, column order, date
format, whether date and time are one field or two, and which timezone the
timestamps are in. This provider takes an explicit mapping rather than guessing,
because a silently mis-parsed timestamp column is the kind of defect that
survives all the way into a backtest's results.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import polars as pl

from trading_system.core.exceptions import DataError
from trading_system.core.types import Timeframe
from trading_system.data.models import (
    OHLCV_COLUMNS,
    REQUIRED_COLUMNS,
    TIMESTAMP_COLUMN,
    OHLCVFrame,
)

DEFAULT_COLUMN_MAP: dict[str, str] = {
    "timestamp": TIMESTAMP_COLUMN,
    "date": TIMESTAMP_COLUMN,
    "datetime": TIMESTAMP_COLUMN,
    "time": TIMESTAMP_COLUMN,
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "volume": "volume",
    "vol": "volume",
}


@dataclass(frozen=True)
class CSVSchema:
    """How to read one family of CSV files.

    Attributes:
        column_map: Source column name (lowercased) to canonical OHLCV name.
            Defaults cover the common spellings.
        timestamp_format: ``chrono`` format string for parsing the timestamp
            column. ``None`` lets polars infer an ISO-8601 layout.
        source_tz: IANA zone the file's timestamps are recorded in. Required when
            the timestamps carry no offset of their own.
        date_column: Name of a separate date column, when date and time are split.
        time_column: Name of a separate time column, when date and time are split.
        separator: Field delimiter.
        has_header: Whether the first line names the columns.
        drop_unnamed_fields: Whether rows may carry more fields than the header
            names. ``False`` — the default — makes such a file an error, because
            a row wider than its header usually means the separator is wrong and
            every column is off by one. ``True`` states that the trailing fields
            were looked at and are not OHLCV: they are discarded, not stored.
            Some broker exports append a per-bar spread this way, and the frame
            has nowhere to put it.
        positional_columns: Canonical name of each field, in file order, with
            ``None`` for a field that is present and deliberately not stored.
            Set it when the file names no columns at all, or when it names them
            wrongly — a header calling its sixth field ``Volume`` when the field
            counts minutes is worse than no header, because the default map
            believes it. Takes precedence over ``column_map``; the length must
            equal the field count, so a file that grew a column is an error
            rather than a silent shift.
        absent_volume: Whether the file carries no volume at all. ``True`` writes
            the column as ``0.0``, which is the one value ``data.quality``
            reports (``zero_volume``) on every bar. Borrowing a neighbouring
            column instead — a minute count, a constant — would sail past every
            check and reach VWAP, MFI, OBV and RelativeVolume as a plausible
            weight. Contradicting it by also naming a volume column is an error.
    """

    column_map: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_COLUMN_MAP))
    timestamp_format: str | None = None
    source_tz: str = "UTC"
    date_column: str | None = None
    time_column: str | None = None
    separator: str = ","
    has_header: bool = True
    drop_unnamed_fields: bool = False
    positional_columns: tuple[str | None, ...] | None = None
    absent_volume: bool = False

    def __post_init__(self) -> None:
        """Reject a schema that names volume and denies it in the same breath."""
        if self.positional_columns is None:
            return
        named = [name for name in self.positional_columns if name is not None]
        unknown = sorted(set(named) - set(REQUIRED_COLUMNS))
        if unknown:
            raise DataError(
                f"positional_columns names {unknown}, which are not canonical; "
                f"use {list(REQUIRED_COLUMNS)} or None to skip a field"
            )
        duplicated = sorted({name for name in named if named.count(name) > 1})
        if duplicated:
            raise DataError(f"positional_columns names {duplicated} more than once")
        if self.absent_volume and "volume" in named:
            raise DataError(
                "positional_columns names a volume column while absent_volume says "
                "the file has none; one of the two is wrong"
            )


class CSVProvider:
    """Reads OHLCV bars from local CSV files."""

    def __init__(self, path: Path, schema: CSVSchema | None = None) -> None:
        """Bind the provider to a file or directory.

        Args:
            path: A CSV file, or a directory containing ``{symbol}.csv`` files.
            schema: How to interpret the files. Defaults handle ISO timestamps in
                UTC with conventionally named columns.
        """
        self._path = path
        self._schema = schema or CSVSchema()

    def fetch(
        self,
        symbol: str,
        timeframe: Timeframe,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> OHLCVFrame:
        """Read and normalise bars for ``symbol``.

        Args:
            symbol: Instrument identifier; also selects ``{symbol}.csv`` when this
                provider points at a directory.
            timeframe: Bar size the file is known to contain. Not verified against
                the data, since a file may legitimately have gaps.
            start: Inclusive lower bound on bar open time.
            end: Exclusive upper bound on bar open time.

        Returns:
            Validated bars within the requested range.

        Raises:
            DataError: If the file is missing, or its columns cannot be mapped.
        """
        path = self._resolve(symbol)
        try:
            raw = pl.read_csv(
                path,
                separator=self._schema.separator,
                has_header=self._schema.has_header,
                try_parse_dates=False,
                truncate_ragged_lines=self._schema.drop_unnamed_fields,
                # Every field arrives as text and is cast once, by OHLCVFrame.
                # Inferring from the head of the file lets the first hundred
                # rows decide the type of the whole column, and an index that
                # traded at whole points for its first day is then an integer
                # column that the first fractional price cannot be parsed into.
                infer_schema_length=0,
            )
        except pl.exceptions.ComputeError as error:
            raise DataError(
                f"{path}: could not be read as {self._schema.separator!r}-separated CSV "
                f"({error}). If its rows carry trailing fields the header does not "
                "name, say so with drop_unnamed_fields"
            ) from error
        renamed = self._apply_column_map(raw, path)
        with_volume = self._fill_absent_volume(renamed, path)
        with_timestamp = self._build_timestamp(with_volume, path)
        frame = OHLCVFrame.from_raw(
            with_timestamp,
            symbol,
            timeframe,
            assume_tz=self._schema.source_tz,
        )
        return frame.slice(start, end)

    def _resolve(self, symbol: str) -> Path:
        """Locate the CSV backing ``symbol``."""
        path = self._path / f"{symbol}.csv" if self._path.is_dir() else self._path
        if not path.is_file():
            raise DataError(f"no CSV for {symbol} at {path}")
        return path

    def _apply_column_map(self, df: pl.DataFrame, path: Path) -> pl.DataFrame:
        """Rename source columns onto canonical names."""
        renamed = (
            self._apply_positions(df, path)
            if self._schema.positional_columns is not None
            else df.rename(
                {
                    column: self._schema.column_map[column.strip().lower()]
                    for column in df.columns
                    if column.strip().lower() in self._schema.column_map
                }
            )
        )
        expected = [
            column
            for column in OHLCV_COLUMNS
            if not (column == "volume" and self._schema.absent_volume)
        ]
        missing = [column for column in expected if column not in renamed.columns]
        if missing:
            raise DataError(
                f"{path}: could not map columns {missing}; source columns were {df.columns}"
            )
        return renamed

    def _apply_positions(self, df: pl.DataFrame, path: Path) -> pl.DataFrame:
        """Name the fields by position, dropping the ones the caller skipped."""
        positions = self._schema.positional_columns
        assert positions is not None  # guarded by the caller
        if len(positions) != df.width:
            raise DataError(
                f"{path}: positional_columns describes {len(positions)} fields but the "
                f"file has {df.width}; naming them by position would shift every column"
            )
        keep = {
            source: name
            for source, name in zip(df.columns, positions, strict=True)
            if name is not None
        }
        return df.select(list(keep)).rename(keep)

    def _fill_absent_volume(self, df: pl.DataFrame, path: Path) -> pl.DataFrame:
        """Write the zero volume of a tape that reports none."""
        if not self._schema.absent_volume:
            return df
        if "volume" in df.columns:
            raise DataError(
                f"{path}: absent_volume says the file has no volume, but a volume "
                "column was mapped from it"
            )
        return df.with_columns(pl.lit(0.0).alias("volume"))

    def _build_timestamp(self, df: pl.DataFrame, path: Path) -> pl.DataFrame:
        """Produce a single parsed timestamp column."""
        schema = self._schema
        if schema.date_column is not None:
            if schema.time_column is None:
                raise DataError(f"{path}: date_column given without time_column")
            if schema.date_column not in df.columns or schema.time_column not in df.columns:
                raise DataError(
                    f"{path}: expected columns {schema.date_column!r} and "
                    f"{schema.time_column!r}, got {df.columns}"
                )
            combined = pl.concat_str(
                [pl.col(schema.date_column), pl.col(schema.time_column)], separator=" "
            )
        elif TIMESTAMP_COLUMN in df.columns:
            combined = pl.col(TIMESTAMP_COLUMN).cast(pl.String)
        else:
            raise DataError(f"{path}: no timestamp column found among {df.columns}")

        parsed = combined.str.strptime(
            pl.Datetime(time_unit="us"), format=schema.timestamp_format, strict=True
        )
        return df.with_columns(parsed.alias(TIMESTAMP_COLUMN))
