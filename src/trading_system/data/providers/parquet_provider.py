"""Parquet provider: reads and writes the store's native format.

Parquet already carries dtypes and timezone metadata, so unlike CSV there is
nothing to guess. This provider exists so that Parquet files produced elsewhere
can enter the system through the same validated boundary as any other source.
"""

from datetime import datetime
from pathlib import Path

import polars as pl

from trading_system.core.exceptions import DataError
from trading_system.core.types import Timeframe
from trading_system.data.models import OHLCVFrame


class ParquetProvider:
    """Reads OHLCV bars from Parquet files."""

    def __init__(self, path: Path) -> None:
        """Bind the provider to a file or directory.

        Args:
            path: A Parquet file, or a directory holding ``{symbol}.parquet``
                files or a partitioned tree of them.
        """
        self._path = path

    def fetch(
        self,
        symbol: str,
        timeframe: Timeframe,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> OHLCVFrame:
        """Read and normalise bars for ``symbol``.

        Args:
            symbol: Instrument identifier; selects ``{symbol}.parquet`` when this
                provider points at a directory.
            timeframe: Bar size the file contains.
            start: Inclusive lower bound on bar open time.
            end: Exclusive upper bound on bar open time.

        Returns:
            Validated bars within the requested range.

        Raises:
            DataError: If no Parquet data exists for ``symbol``.
        """
        sources = self._resolve(symbol)
        raw = pl.concat([pl.read_parquet(source) for source in sources], how="vertical")
        frame = OHLCVFrame.from_raw(raw, symbol, timeframe)
        return frame.slice(start, end)

    def write(self, frame: OHLCVFrame, path: Path | None = None) -> Path:
        """Write ``frame`` to a Parquet file.

        Args:
            frame: Bars to write.
            path: Destination file. Defaults to ``{symbol}.parquet`` under this
                provider's directory.

        Returns:
            The path written to.

        Raises:
            DataError: If no destination can be determined.
        """
        destination = path or self._default_destination(frame.symbol)
        destination.parent.mkdir(parents=True, exist_ok=True)
        frame.df.write_parquet(destination)
        return destination

    def _default_destination(self, symbol: str) -> Path:
        """Choose where ``symbol`` should be written when no path is given."""
        if self._path.suffix == ".parquet":
            return self._path
        return self._path / f"{symbol}.parquet"

    def _resolve(self, symbol: str) -> list[Path]:
        """Locate the Parquet files backing ``symbol``."""
        if self._path.is_file():
            return [self._path]
        if self._path.is_dir():
            direct = self._path / f"{symbol}.parquet"
            if direct.is_file():
                return [direct]
            nested = sorted((self._path / symbol).glob("**/*.parquet"))
            if nested:
                return nested
            flat = sorted(self._path.glob("*.parquet"))
            if flat:
                return flat
        raise DataError(f"no Parquet data for {symbol} at {self._path}")
