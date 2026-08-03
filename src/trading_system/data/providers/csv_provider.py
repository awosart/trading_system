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
from trading_system.data.models import OHLCV_COLUMNS, TIMESTAMP_COLUMN, OHLCVFrame

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
    """

    column_map: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_COLUMN_MAP))
    timestamp_format: str | None = None
    source_tz: str = "UTC"
    date_column: str | None = None
    time_column: str | None = None
    separator: str = ","
    has_header: bool = True


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
        raw = pl.read_csv(
            path,
            separator=self._schema.separator,
            has_header=self._schema.has_header,
            try_parse_dates=False,
        )
        renamed = self._apply_column_map(raw, path)
        with_timestamp = self._build_timestamp(renamed, path)
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
        mapping = {
            column: self._schema.column_map[column.strip().lower()]
            for column in df.columns
            if column.strip().lower() in self._schema.column_map
        }
        renamed = df.rename(mapping)
        missing = [column for column in OHLCV_COLUMNS if column not in renamed.columns]
        if missing:
            raise DataError(
                f"{path}: could not map columns {missing}; source columns were {df.columns}"
            )
        return renamed

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
