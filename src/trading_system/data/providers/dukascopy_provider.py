"""Dukascopy provider: real bid/ask quotes turned into a mid-price OHLCVFrame.

Dukascopy is the priority source (over HistData and yfinance) precisely because
it hands back bid and ask as separate series. Everything else this system reads
from a bar is a mid price (P12 decision: "bars are mid; the model crosses the
spread once, at the fill"), so this provider's whole job is to fetch both sides
and average them, rather than mislabelling a single-sided quote as mid.

The historical feed itself is not reimplemented here (raw tick ``.bi5`` files
are LZMA-compressed binary and Dukascopy's own aggregation rules for turning
ticks into candles are undocumented). Instead this shells out to the
``dukascopy-node`` CLI, a maintained library that already does that
reconstruction, and treats its CSV output as an untrusted external format —
the same boundary role :class:`~trading_system.data.providers.csv_provider.CSVProvider`
plays for vendor CSVs. The subprocess call is injected (``runner``), so tests
never touch the network.
"""

import subprocess
import tempfile
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from trading_system.core.exceptions import DataError
from trading_system.core.logging import get_logger
from trading_system.core.types import Timeframe
from trading_system.data.models import OHLCVFrame

logger = get_logger(__name__)

#: Project timeframe to the CLI's ``-t/--timeframe`` value.
_TIMEFRAME_ARG: dict[Timeframe, str] = {
    Timeframe.M1: "m1",
    Timeframe.M5: "m5",
    Timeframe.M15: "m15",
    Timeframe.H1: "h1",
    Timeframe.H4: "h4",
    Timeframe.D1: "d1",
}

DEFAULT_NODE_COMMAND: tuple[str, ...] = ("npx", "dukascopy-node")

Runner = Callable[[Sequence[str]], "subprocess.CompletedProcess[str]"]


def _default_runner(argv: Sequence[str]) -> "subprocess.CompletedProcess[str]":
    """Invoke the CLI as a real subprocess. The production default for ``runner``."""
    return subprocess.run(list(argv), capture_output=True, text=True, timeout=900, check=False)


class DukascopyProvider:
    """Fetches bid and ask bars from Dukascopy and merges them into mid OHLCV.

    Two ``dukascopy-node`` invocations per :meth:`fetch` call — one per price
    side — because the CLI has no notion of mid; it only knows bid and ask.
    """

    def __init__(
        self,
        *,
        node_command: Sequence[str] = DEFAULT_NODE_COMMAND,
        runner: Runner = _default_runner,
        volume_units: str = "units",
        retries: int = 3,
    ) -> None:
        """Configure how the underlying CLI is invoked.

        Args:
            node_command: Executable and leading arguments used to invoke
                ``dukascopy-node``. Defaults to running it through ``npx``, which
                works without a prior global install but does a registry check on
                first use.
            runner: Callable that runs a subprocess and returns a
                ``CompletedProcess``. Overridden in tests to avoid the network;
                production code should leave the default.
            volume_units: Passed through to the CLI's ``-vu`` flag. Dukascopy's
                volume is a vendor-defined tick-derived quantity, not a count of
                real contracts; this only selects its scale.
            retries: Passed through to the CLI's ``-r`` flag for transient
                download failures of individual daily/hourly artifacts.
        """
        self._node_command = tuple(node_command)
        self._runner = runner
        self._volume_units = volume_units
        self._retries = retries

    def fetch(
        self,
        symbol: str,
        timeframe: Timeframe,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> OHLCVFrame:
        """Retrieve mid-price bars for a half-open ``[start, end)`` range.

        Args:
            symbol: Instrument identifier. Lower-cased to build the Dukascopy
                instrument code (``EURUSD`` -> ``eurusd``), which matches its
                convention for FX majors. Symbols that need a different code are
                not supported by this simple mapping.
            timeframe: Bar size requested.
            start: Inclusive lower bound on bar open time. Required: Dukascopy's
                earliest available date differs per instrument, so treating
                ``None`` as "from the beginning" would mean silently guessing a
                start date instead of stating one.
            end: Exclusive upper bound on bar open time. ``None`` fetches up to
                the current moment.

        Returns:
            Validated mid-price bars within the requested range.

        Raises:
            DataError: If ``start`` is omitted, the CLI invocation fails, or its
                output cannot be read.
        """
        if start is None:
            raise DataError(
                f"{symbol}: DukascopyProvider requires an explicit start date; "
                "Dukascopy's earliest coverage differs per instrument"
            )
        end = end if end is not None else datetime.now(UTC)
        if start.tzinfo is None or end.tzinfo is None:
            raise DataError(f"{symbol}: start and end must be tz-aware")

        timeframe_arg = _TIMEFRAME_ARG[timeframe]
        with tempfile.TemporaryDirectory(prefix="dukascopy_") as tmp:
            out_dir = Path(tmp)
            bid = self._fetch_side(symbol, timeframe_arg, start, end, "bid", out_dir)
            ask = self._fetch_side(symbol, timeframe_arg, start, end, "ask", out_dir)

        merged, corrected = _merge_mid(bid, ask)
        if merged.is_empty():
            return OHLCVFrame.empty(symbol, timeframe)
        if corrected:
            logger.warning(
                "dukascopy.mid_ohlc_corrected",
                symbol=symbol,
                timeframe=str(timeframe),
                corrected=corrected,
                total=merged.height,
            )
        frame = OHLCVFrame(merged, symbol, timeframe)
        return frame.slice(start, end)

    def _fetch_side(
        self,
        symbol: str,
        timeframe_arg: str,
        start: datetime,
        end: datetime,
        price_type: str,
        out_dir: Path,
    ) -> pl.DataFrame:
        """Run one CLI invocation for one price side and parse its CSV output."""
        file_name = f"{symbol.lower()}_{timeframe_arg}_{price_type}"
        argv = [
            *self._node_command,
            "-i",
            symbol.lower(),
            "-from",
            start.date().isoformat(),
            "-to",
            _to_date_arg(end),
            "-t",
            timeframe_arg,
            "-p",
            price_type,
            "-f",
            "csv",
            "-dir",
            str(out_dir),
            "-fn",
            file_name,
            "-v",
            "-vu",
            self._volume_units,
            "-r",
            str(self._retries),
        ]
        result = self._runner(argv)
        if result.returncode != 0:
            raise DataError(
                f"{symbol}: dukascopy-node failed fetching {price_type} "
                f"(exit {result.returncode}): {result.stderr.strip()}"
            )

        path = out_dir / f"{file_name}.csv"
        if not path.is_file():
            raise DataError(
                f"{symbol}: dukascopy-node reported success but wrote no output "
                f"for {price_type} at {path}"
            )
        return _read_dukascopy_csv(path, symbol)


def _to_date_arg(end: datetime) -> str:
    """Render the CLI's ``-to`` bound so the fetched range covers ``end``.

    The CLI's date bound is day-granular and, per observed behaviour, excludes
    the given day itself (``-to 2026-08-05`` yields bars through 2026-08-04). A
    caller asking for an ``end`` with a nonzero time-of-day still needs that
    day's bars available to slice down to the exact instant, so the day is
    pushed forward whenever ``end`` is not exactly midnight.
    """
    day = end.date()
    if end.time() != datetime.min.time():
        day = day + timedelta(days=1)
    return day.isoformat()


def _read_dukascopy_csv(path: Path, symbol: str) -> pl.DataFrame:
    """Parse one side's CSV into a typed frame with a UTC timestamp column."""
    try:
        raw = pl.read_csv(
            path,
            schema_overrides={
                "timestamp": pl.Int64,
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Float64,
            },
        )
    except Exception as error:
        raise DataError(f"{symbol}: could not parse {path}: {error}") from error

    expected = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = expected - set(raw.columns)
    if missing:
        raise DataError(f"{symbol}: {path} is missing columns {sorted(missing)}")

    return raw.with_columns(
        pl.from_epoch(pl.col("timestamp"), time_unit="ms")
        .dt.replace_time_zone("UTC")
        .dt.cast_time_unit("us")
        .alias("timestamp")
    )


def _merge_mid(bid: pl.DataFrame, ask: pl.DataFrame) -> tuple[pl.DataFrame, int]:
    """Average matching bid/ask bars into mid OHLCV, then repair the merge artifact.

    Joined on timestamp rather than concatenated: bid and ask are fetched as two
    independent CLI runs and can, at the edges of a range, disagree on which
    instants have data. A bar only one side reports is not a mid bar, so it is
    dropped rather than fabricated from a single side.

    Averaging two already-aggregated OHLC series column-by-column does not
    guarantee ``low <= open, close <= high``: bid's intra-hour extreme and
    ask's intra-hour extreme can land at different moments, so their means
    need not bracket the means of open and close. This is a correction of
    that artifact, not new information — the true mid extreme within the bar
    was never tighter than what the averaged open/close already show, so
    ``high = max(high, open, close)`` and ``low = min(low, open, close)``
    restores a consistent bar without inventing a level beyond what open/close
    already imply.

    Returns:
        The mid frame (columns ``timestamp, open, high, low, close, volume``,
        sorted, bracket-consistent) and the number of bars whose ``high`` or
        ``low`` needed widening. This should stay tiny (single digits) across
        a multi-year pull. If it climbs into the hundreds, the two sides are
        diverging enough that averaging their OHLC — rather than raw ticks —
        is no longer a reasonable approximation of mid.
    """
    if bid.is_empty() or ask.is_empty():
        return bid.clear(), 0

    joined = bid.join(ask, on="timestamp", how="inner", suffix="_ask")
    mid = joined.select(
        pl.col("timestamp"),
        ((pl.col("open") + pl.col("open_ask")) / 2).alias("open"),
        ((pl.col("high") + pl.col("high_ask")) / 2).alias("high"),
        ((pl.col("low") + pl.col("low_ask")) / 2).alias("low"),
        ((pl.col("close") + pl.col("close_ask")) / 2).alias("close"),
        ((pl.col("volume") + pl.col("volume_ask")) / 2).alias("volume"),
    ).sort("timestamp")

    bracket_high = pl.max_horizontal("open", "high", "close")
    bracket_low = pl.min_horizontal("open", "low", "close")
    needs_widening = (pl.col("high") < bracket_high) | (pl.col("low") > bracket_low)
    corrected = int(mid.select(needs_widening.sum()).item() or 0)

    fixed = mid.with_columns(bracket_high.alias("high"), bracket_low.alias("low"))
    return fixed, corrected
