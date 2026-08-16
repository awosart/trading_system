"""What the local store actually holds, measured rather than declared.

A normalised spec has to name instruments, and the only defensible list is the
one the store can serve. Two properties beyond "the series exists" decide
whether a symbol belongs in a spec's universe, and both are measured here
rather than assumed:

* whether the series carries volume at all — five of the eleven symbols in this
  store were imported with ``volume`` written as a literal zero (the vendor's
  sixth column was a count of minutes with a tick, not a size), so every
  volume-reading indicator on them is arithmetic over a constant;
* how the declared cost compares to the size of a bar — on a minute bar of a
  major FX pair the spread plus commission is most of the bar's range, which
  decides nothing about a strategy and everything about whether the pair is
  tradeable at that bar size.

Both are per ``(symbol, timeframe)`` and neither is stable across imports, so
this module measures and does not cache to disk.
"""

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime

import polars as pl

from trading_system.core.instruments import (
    InstrumentClass,
    InstrumentRegistry,
    InstrumentSpec,
)
from trading_system.core.types import Timeframe
from trading_system.data.models import OHLCVFrame

#: Cost-to-range ratio above which a symbol is not offered at a timeframe.
#: Half the median bar is not a threshold with a theory behind it; it is the
#: point past which the round turn is the dominant term in every trade the
#: strategy can take, so what the backtest measures is the cost model rather
#: than the entry. Callers may pass their own.
DEFAULT_MAX_COST_RATIO = 0.5


@dataclass(frozen=True)
class SeriesCoverage:
    """One ``(symbol, timeframe)`` series as the store holds it.

    Attributes:
        symbol: Instrument symbol.
        asset_class: What kind of instrument it is, from the registry.
        timeframe: Bar size.
        bars: Rows stored.
        start: First bar's open time.
        end: Last bar's open time.
        median_range_points: Median ``high - low`` in instrument points.
        has_volume: Whether any bar carries a non-zero volume.
        cost_points: Round-turn cost in points: the declared typical spread
            plus commission expressed in the same unit.
    """

    symbol: str
    asset_class: InstrumentClass
    timeframe: Timeframe
    bars: int
    start: datetime
    end: datetime
    median_range_points: float
    has_volume: bool
    cost_points: float

    @property
    def cost_ratio(self) -> float:
        """Round-turn cost as a fraction of the median bar's range.

        Returns:
            The ratio; ``inf`` when bars have no range at all, which is a
            degenerate series rather than a free one.
        """
        if self.median_range_points <= 0:
            return float("inf")
        return self.cost_points / self.median_range_points


@dataclass(frozen=True)
class MarketCoverage:
    """Every series the store was asked about.

    Attributes:
        series: Coverage by ``(symbol, timeframe)``.
    """

    series: Mapping[tuple[str, Timeframe], SeriesCoverage]

    def get(self, symbol: str, timeframe: Timeframe) -> SeriesCoverage | None:
        """The coverage of one series, or ``None`` when it is not stored."""
        return self.series.get((symbol, timeframe))

    def symbols(self, timeframe: Timeframe) -> tuple[str, ...]:
        """Every symbol stored at ``timeframe``, in alphabetical order."""
        return tuple(sorted(sym for sym, tf in self.series if tf is timeframe))

    def timeframes(self, symbol: str) -> tuple[Timeframe, ...]:
        """Every timeframe stored for ``symbol``, coarsest last."""
        found = [tf for sym, tf in self.series if sym == symbol]
        return tuple(sorted(found, key=lambda tf: tf.duration))

    def admissible(
        self,
        timeframe: Timeframe,
        *,
        needs_volume: bool,
        max_cost_ratio: float = DEFAULT_MAX_COST_RATIO,
    ) -> tuple[tuple[str, ...], dict[str, str]]:
        """Which symbols a spec may be offered at this bar size, and why not the rest.

        Args:
            timeframe: The bar size the spec trades.
            needs_volume: Whether the spec reads volume anywhere.
            max_cost_ratio: Ratio of round-turn cost to median bar range above
                which a symbol is refused.

        Returns:
            ``(allowed, refused)``. ``refused`` maps each rejected symbol to a
            refusal code: ``no_volume`` or ``spread_dominates``. A symbol the
            store does not hold at this timeframe appears in neither — it was
            never a candidate, which is a different fact from being refused.
        """
        allowed: list[str] = []
        refused: dict[str, str] = {}
        for symbol in self.symbols(timeframe):
            found = self.series[(symbol, timeframe)]
            if needs_volume and not found.has_volume:
                refused[symbol] = "no_volume"
            elif found.cost_ratio >= max_cost_ratio:
                refused[symbol] = "spread_dominates"
            else:
                allowed.append(symbol)
        return tuple(allowed), refused


def cost_points(instrument: InstrumentSpec) -> float:
    """Round-turn cost of one lot, expressed in the instrument's points.

    The two halves are declared in different units — the spread already in
    points, the commission in account money per lot — and comparing either one
    alone with a bar's range understates the cost by roughly half on a major
    FX pair.

    Args:
        instrument: The instrument, carrying both figures.

    Returns:
        Spread plus round-turn commission, in points.
    """
    point_value = float(instrument.point_size) * float(instrument.contract_size)
    if point_value <= 0:
        return float(instrument.typical_spread_points)
    commission = float(instrument.commission_per_side) * 2.0
    return float(instrument.typical_spread_points) + commission / point_value


def measure_coverage(
    load: Callable[[str, Timeframe], OHLCVFrame],
    instruments: InstrumentRegistry,
    symbols: Iterable[str],
    timeframes: Sequence[Timeframe],
) -> MarketCoverage:
    """Measure every stored series among ``symbols`` × ``timeframes``.

    Args:
        load: Reads one series whole; a series the store does not hold must
            come back empty rather than raise.
        instruments: Registry the point size and declared costs come from.
        symbols: Symbols to look for.
        timeframes: Bar sizes to look for.

    Returns:
        The coverage. A series that is absent, empty, or names an instrument
        the registry does not carry is simply not in the result: there is
        nothing to measure and nothing to refuse.
    """
    found: dict[tuple[str, Timeframe], SeriesCoverage] = {}
    for symbol in symbols:
        instrument = instruments.get(symbol)
        if instrument is None:
            continue
        point = float(instrument.point_size)
        for timeframe in timeframes:
            frame = load(symbol, timeframe)
            if frame.is_empty or frame.start is None or frame.end is None:
                continue
            stats = frame.df.select(
                ((pl.col("high") - pl.col("low")) / point).median().alias("range"),
                (pl.col("volume") != 0).any().alias("volume"),
            )
            median_range = stats["range"][0]
            found[(symbol, timeframe)] = SeriesCoverage(
                symbol=symbol,
                asset_class=instrument.asset_class,
                timeframe=timeframe,
                bars=len(frame),
                start=frame.start,
                end=frame.end,
                median_range_points=float(median_range) if median_range is not None else 0.0,
                has_volume=bool(stats["volume"][0]),
                cost_points=cost_points(instrument),
            )
    return MarketCoverage(series=found)
