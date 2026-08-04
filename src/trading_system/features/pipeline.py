"""Declarative feature computation over a frame of bars.

A pipeline is a list of ``(name, kind, params)`` specifications resolved into
indicators once, at construction. A misspelled indicator or an impossible period
raises there and not four hours into a walk-forward run.

Warmup is the load-bearing concept. Every indicator declares how many leading
bars it cannot value; the pipeline reports the largest, and those rows carry
nulls in every column. They are **never** forward-filled. A forward-filled
warmup produces a feature matrix with no gaps and no meaning: the first hundred
rows all claim to know a 200-period average, a strategy trades them, and the
backtest reports an edge that exists only in the fill.
"""

import hashlib
import io
from collections import OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from trading_system.core.exceptions import ValidationError
from trading_system.core.types import Timeframe
from trading_system.data.models import TIMESTAMP_COLUMN, OHLCVFrame
from trading_system.features.base import BaseIndicator
from trading_system.features.registry import build_indicator


class FeatureSpec(BaseModel):
    """One entry of a pipeline: what to compute and what to call it.

    Attributes:
        name: Column name for a single-valued indicator, or the prefix for a
            multi-channel one, which publishes ``{name}_{channel}``.
        kind: Registry key, e.g. ``"rsi"``.
        params: Constructor arguments for the indicator.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    params: dict[str, Any] = Field(default_factory=dict)


class FeaturePipelineConfig(BaseModel):
    """A whole pipeline, as it appears in a config file.

    Attributes:
        features: Specifications to compute, in order.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    features: list[FeatureSpec] = Field(default_factory=list)


class FeatureCache(Protocol):
    """Somewhere to keep computed feature frames between calls."""

    def get(self, key: str) -> pl.DataFrame | None:
        """Return the frame stored under ``key``, or ``None``."""

    def put(self, key: str, value: pl.DataFrame) -> None:
        """Store ``value`` under ``key``."""


class InMemoryFeatureCache:
    """A bounded, least-recently-used cache of feature frames.

    Bounded on purpose: a research loop over a decade of minute bars will
    otherwise pin every intermediate feature matrix in memory until the process
    dies of it.
    """

    def __init__(self, max_entries: int = 8) -> None:
        """Create an empty cache.

        Args:
            max_entries: How many frames to retain before evicting the least
                recently used.

        Raises:
            ValueError: If ``max_entries`` is not positive.
        """
        if max_entries < 1:
            raise ValueError(f"max_entries must be positive, got {max_entries}")
        self._max_entries = max_entries
        self._entries: OrderedDict[str, pl.DataFrame] = OrderedDict()

    def __len__(self) -> int:
        """Number of frames currently held."""
        return len(self._entries)

    def get(self, key: str) -> pl.DataFrame | None:
        """Return the frame stored under ``key``, marking it most recent."""
        value = self._entries.get(key)
        if value is not None:
            self._entries.move_to_end(key)
        return value

    def put(self, key: str, value: pl.DataFrame) -> None:
        """Store ``value``, evicting the least recently used entry if needed."""
        self._entries[key] = value
        self._entries.move_to_end(key)
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)


@dataclass(frozen=True)
class FeatureSet:
    """Features computed over one frame, aligned to its bars.

    Attributes:
        symbol: Instrument the features describe.
        timeframe: Bar size they were computed at.
        df: ``timestamp`` plus one column per feature channel, one row per bar.
            Each column keeps its *own* indicator's warmup, so a fast RSI carries
            values on rows a slow MACD does not. Research work gets the longest
            series each indicator can honestly produce; trading goes through
            :meth:`valid`, which applies the pipeline-wide boundary.
        warmup_bars: Leading rows at least one indicator cannot value, and
            therefore the first row on which the whole feature vector exists.
        feature_columns: Feature column names, excluding ``timestamp``.
    """

    symbol: str
    timeframe: Timeframe
    df: pl.DataFrame
    warmup_bars: int
    feature_columns: tuple[str, ...]

    def __len__(self) -> int:
        """Number of bars, warmup included."""
        return self.df.height

    @property
    def timestamps(self) -> pl.Series:
        """Bar open times, ascending."""
        return self.df[TIMESTAMP_COLUMN]

    def valid_mask(self) -> pl.Series:
        """Which rows are safe to trade on.

        A row qualifies when it is past the warmup *and* carries no null in any
        feature column. The second test matters for indicators whose validity
        depends on the data rather than on a bar count — an anchored VWAP is
        null until its anchor, whatever the pipeline's warmup says.

        Returns:
            A Boolean series aligned with :attr:`df`.
        """
        if not self.feature_columns:
            return pl.Series("valid", [True] * self.df.height, dtype=pl.Boolean)
        past_warmup = pl.int_range(0, pl.len(), dtype=pl.Int64) >= self.warmup_bars
        complete = pl.all_horizontal(
            pl.col(column).is_not_null() for column in self.feature_columns
        )
        return self.df.select((past_warmup & complete).alias("valid"))["valid"]

    def valid(self) -> pl.DataFrame:
        """The rows a strategy may act on.

        Returns:
            :attr:`df` filtered by :meth:`valid_mask`. Warmup rows are dropped,
            never filled.
        """
        return self.df.filter(self.valid_mask())


class FeaturePipeline:
    """Computes a fixed set of features over frames of bars."""

    def __init__(
        self,
        specs: Iterable[FeatureSpec] | FeaturePipelineConfig,
        *,
        cache: FeatureCache | None = None,
    ) -> None:
        """Resolve every specification into an indicator.

        Args:
            specs: The features to compute, or a whole config object.
            cache: Where to memoise results. ``None`` disables caching; the
                dependency is explicit rather than a module-level default so two
                pipelines cannot silently share one.

        Raises:
            ValidationError: If a name is duplicated, or a specification names an
                unknown indicator or invalid parameters.
        """
        resolved = list(specs.features if isinstance(specs, FeaturePipelineConfig) else specs)
        names = [spec.name for spec in resolved]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValidationError(f"duplicate feature names: {duplicates}")

        self._specs = tuple(resolved)
        self._indicators = tuple(build_indicator(spec.kind, spec.params) for spec in resolved)
        self._cache = cache
        self._columns = tuple(
            column
            for spec, indicator in zip(self._specs, self._indicators, strict=True)
            for column in _column_names(spec.name, indicator)
        )
        collisions = sorted({name for name in self._columns if self._columns.count(name) > 1})
        if collisions:
            raise ValidationError(f"feature specs produce colliding columns: {collisions}")

    @property
    def specs(self) -> tuple[FeatureSpec, ...]:
        """The specifications this pipeline was built from."""
        return self._specs

    @property
    def indicators(self) -> tuple[BaseIndicator[Any], ...]:
        """The resolved indicators, aligned with :attr:`specs`."""
        return self._indicators

    @property
    def feature_columns(self) -> tuple[str, ...]:
        """Every column the pipeline emits, excluding ``timestamp``."""
        return self._columns

    @property
    def warmup_bars(self) -> int:
        """Leading bars no indicator can value; 0 for an empty pipeline."""
        return max((indicator.warmup for indicator in self._indicators), default=0)

    def signature(self) -> str:
        """Stable description of what this pipeline computes.

        Returns:
            A string covering every spec name and the resolved indicator
            identity, which encodes its parameters.
        """
        return ";".join(
            f"{spec.name}={indicator.name}"
            for spec, indicator in zip(self._specs, self._indicators, strict=True)
        )

    def cache_key(self, frame: OHLCVFrame) -> str:
        """Key identifying this pipeline's output for ``frame``.

        Covers the symbol, the timeframe, the pipeline signature, and a digest of
        the bar values themselves. Hashing the data rather than the range means a
        vendor correcting a single bar in the middle of a stored year invalidates
        the entry, which a start/end key would not.

        Args:
            frame: Bars the features would be computed over.

        Returns:
            A hexadecimal digest.
        """
        digest = hashlib.blake2b(digest_size=16)
        digest.update(frame.symbol.encode())
        digest.update(str(frame.timeframe).encode())
        digest.update(self.signature().encode())
        digest.update(_frame_fingerprint(frame))
        return digest.hexdigest()

    def compute(self, frame: OHLCVFrame) -> FeatureSet:
        """Compute every feature over ``frame``.

        Args:
            frame: Bars to evaluate over.

        Returns:
            The features, aligned to the frame's bars.

        Raises:
            ValidationError: If an indicator returns a frame of the wrong shape.
        """
        key = self.cache_key(frame) if self._cache is not None else ""
        if self._cache is not None:
            cached = self._cache.get(key)
            if cached is not None:
                return self._build_set(frame, cached)

        columns = [frame.df[TIMESTAMP_COLUMN]]
        for spec, indicator in zip(self._specs, self._indicators, strict=True):
            computed = indicator.compute_frame(frame)
            names = _column_names(spec.name, indicator)
            columns.extend(
                computed[channel].rename(name)
                for channel, name in zip(indicator.outputs, names, strict=True)
            )
        result = pl.DataFrame(columns)

        if self._cache is not None:
            self._cache.put(key, result)
        return self._build_set(frame, result)

    def _build_set(self, frame: OHLCVFrame, df: pl.DataFrame) -> FeatureSet:
        """Wrap a computed frame in its metadata."""
        return FeatureSet(
            symbol=frame.symbol,
            timeframe=frame.timeframe,
            df=df,
            warmup_bars=self.warmup_bars,
            feature_columns=self._columns,
        )


def pipeline_from_specs(
    specs: Sequence[Mapping[str, Any]], *, cache: FeatureCache | None = None
) -> FeaturePipeline:
    """Build a pipeline from plain mappings, as parsed from YAML.

    Args:
        specs: One mapping per feature, with ``name``, ``kind`` and optional
            ``params`` keys.
        cache: Where to memoise results.

    Returns:
        The constructed pipeline.

    Raises:
        ValidationError: If a mapping does not describe a valid specification.
    """
    try:
        parsed = [FeatureSpec.model_validate(spec) for spec in specs]
    except Exception as error:  # pydantic raises its own ValidationError type
        raise ValidationError(f"invalid feature specification: {error}") from error
    return FeaturePipeline(parsed, cache=cache)


def _column_names(alias: str, indicator: BaseIndicator[Any]) -> tuple[str, ...]:
    """Column names a single indicator contributes under ``alias``."""
    if len(indicator.outputs) == 1:
        return (alias,)
    return tuple(f"{alias}_{channel}" for channel in indicator.outputs)


def _frame_fingerprint(frame: OHLCVFrame) -> bytes:
    """Digest the frame's contents.

    ``hash_rows`` reduces each bar to a 64-bit value inside polars, and the
    resulting column is serialised through Arrow IPC — both Rust-side, so a
    million bars cost milliseconds rather than the seconds a Python-level
    traversal would.
    """
    buffer = io.BytesIO()
    frame.df.hash_rows().to_frame().write_ipc(buffer, compression="uncompressed")
    return buffer.getvalue()
