"""The zero for the fold sequence itself: shuffle bars, not returns, and never move the clock.

**The unit of permutation is the whole bar, described multiplicatively.**
Whether a stop was touched is decided by high/low, not by close-to-close
return; permuting doughnut-hole close returns and reconstructing OHLC around
them would make every stop-touch outcome a function of the reconstruction
rule rather than of anything the market actually printed. Each bar after the
first is instead described relative to what surrounds it — a gap ratio off
the *previous* close, a body ratio off its *own* open, and two extreme ratios
off its own open/close range — and a shuffled bar is re-anchored onto its new
predecessor by undoing exactly that description:

    open[i]  = close[i-1] * gap[source]
    close[i] = open[i] * body[source]
    high[i]  = max(open[i], close[i]) * high_ratio[source]
    low[i]   = min(open[i], close[i]) * low_ratio[source]

``high_ratio`` and ``low_ratio`` are always ``>= 1`` and ``<= 1`` respectively
(a bar's own high/low never sit inside its open/close range), so every
reconstructed bar is valid OHLC regardless of which original bar's shape
landed where — the permutation cannot produce a malformed bar by construction.

**The time grid never moves — only its contents do.** ``timestamp`` is left
untouched; only open/high/low/close/volume are reassigned. Moving timestamps
would desynchronise the session multiplier on spread, the tripled Wednesday
swap, the weekend gap and ``trading_day`` itself from the bars they are
supposed to describe, and the difference between a permuted run and the real
one would then include a difference in *costs*, not only a difference in
predictability — exactly the confound this null exists to rule out.

**One shuffle order, shared across every symbol at the finest timeframe.**
Shuffling each symbol independently would kill cross-instrument correlation,
which is a real, legitimate coupling the Risk Engine prices (portfolio heat,
cluster limits) — the same argument P13's permutation harness makes for why
its own check runs with those couplings deliberately switched off rather than
calling their presence a leak. Here the couplings stay on, so the shuffle
must not remove what they depend on. A coarser stream of the same symbol is
never shuffled on its own — it is re-aggregated from the permuted finest
series with :func:`~trading_system.data.resample.resample`, the same function
every other coarser stream in this system is built with, so a permuted H4
never disagrees with its own permuted H1.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from random import Random

import polars as pl

from trading_system.backtest.clock import StreamKey
from trading_system.core.types import Timeframe
from trading_system.data.models import OHLCVFrame
from trading_system.data.resample import DayOrigin, resample


def permutation_order(n: int, *, seed: int) -> list[int]:
    """A Fisher-Yates shuffle of bar indices ``1..n-1``.

    Bar ``0`` is never part of the shuffle — it anchors the reconstruction
    (:func:`permute_frame` keeps its open/high/low/close/volume exactly as
    given), so there is nothing before it to describe a gap or a body from.

    Args:
        n: Number of bars in the series being permuted.
        seed: Seed for the shuffle.

    Returns:
        A permutation of ``list(range(1, n))``: ``order[i - 1]`` names which
        original bar's shape is reattached at position ``i``.

    Raises:
        ValueError: If ``n`` is fewer than 2 — a single bar has no gap or
            body to describe at all.
    """
    if n < 2:
        raise ValueError(f"permutation_order needs at least 2 bars, got {n}")
    order = list(range(1, n))
    Random(seed).shuffle(order)
    return order


def permute_frame(frame: OHLCVFrame, order: Sequence[int]) -> OHLCVFrame:
    """Reattach ``frame``'s own bar shapes onto its own time grid in ``order``.

    Args:
        frame: The source series. Bar ``0``'s own OHLCV is the reconstruction
            anchor and is carried through unchanged.
        order: A permutation of ``1..len(frame) - 1``, from :func:`permutation_order`.

    Returns:
        A new frame, same symbol, same timeframe, same timestamps — the
        multiset of gaps, bodies, extreme ratios and volumes is exactly the
        source's own, just reassigned to different bars.

    Raises:
        ValueError: If ``order`` is not exactly a permutation of
            ``1..len(frame) - 1``.
    """
    n = len(frame)
    if sorted(order) != list(range(1, n)):
        raise ValueError(
            f"order must be a permutation of 1..{n - 1} ({n - 1} values), got {len(order)} values"
        )

    table = frame.df
    opens = table["open"].to_list()
    highs = table["high"].to_list()
    lows = table["low"].to_list()
    closes = table["close"].to_list()
    volumes = table["volume"].to_list()

    # Index k (0-based) of these lists describes original bar (k + 1).
    gaps = [opens[i] / closes[i - 1] for i in range(1, n)]
    bodies = [closes[i] / opens[i] for i in range(1, n)]
    high_ratios = [highs[i] / max(opens[i], closes[i]) for i in range(1, n)]
    low_ratios = [lows[i] / min(opens[i], closes[i]) for i in range(1, n)]
    vols = volumes[1:]

    new_open = [opens[0]]
    new_high = [highs[0]]
    new_low = [lows[0]]
    new_close = [closes[0]]
    new_volume = [volumes[0]]
    for source in order:
        k = source - 1
        opening = new_close[-1] * gaps[k]
        closing = opening * bodies[k]
        new_open.append(opening)
        new_close.append(closing)
        new_high.append(max(opening, closing) * high_ratios[k])
        new_low.append(min(opening, closing) * low_ratios[k])
        new_volume.append(vols[k])

    permuted = table.with_columns(
        pl.Series("open", new_open),
        pl.Series("high", new_high),
        pl.Series("low", new_low),
        pl.Series("close", new_close),
        pl.Series("volume", new_volume),
    )
    return frame.with_df(permuted)


@dataclass(frozen=True)
class PermutationConfig:
    """What one permutation null needs beyond the streams it is given.

    Attributes:
        finest: The timeframe the shuffle is actually performed at. Every
            symbol must carry a stream at this timeframe in the mapping
            passed to :func:`permute_run_streams`; a coarser stream of the
            same symbol is re-derived from it rather than shuffled on its own.
        day_origin: Anchor a ``D1`` coarser stream is re-aggregated on. Not
            used for any other target timeframe — :func:`~trading_system.data.resample.resample`
            aligns those to UTC.
        seed: Seed for the shared shuffle order.
    """

    finest: Timeframe
    day_origin: DayOrigin
    seed: int


def permute_run_streams(
    streams: Mapping[StreamKey, OHLCVFrame], config: PermutationConfig
) -> dict[StreamKey, OHLCVFrame]:
    """Permute a whole run's streams: one order, shared across symbols, at ``config.finest``.

    Args:
        streams: Every stream the run trades, keyed as
            :class:`~trading_system.backtest.clock.StreamKey`.
        config: The shuffle's own parameters.

    Returns:
        A new mapping, same keys as ``streams``. Streams at ``config.finest``
        are permuted directly; any coarser stream is re-aggregated from its
        own symbol's permuted finest series.

    Raises:
        ValueError: If a symbol has no stream at ``config.finest`` to shuffle
            or to re-aggregate a coarser stream of the same symbol from, or if
            the finest streams do not all share one bar count — sharing one
            shuffle order across symbols requires one shared length to share
            it over.
    """
    finest_keys = [key for key in streams if key.timeframe is config.finest]
    if not finest_keys:
        raise ValueError(f"no stream at {config.finest.value} to permute")
    lengths = {len(streams[key]) for key in finest_keys}
    if len(lengths) != 1:
        raise ValueError(
            f"streams at {config.finest.value} disagree on bar count "
            f"({ {str(key): len(streams[key]) for key in finest_keys} }); "
            "one shuffle order cannot be shared across them"
        )
    (n,) = lengths
    order = permutation_order(n, seed=config.seed)

    permuted: dict[StreamKey, OHLCVFrame] = {}
    permuted_finest_by_symbol: dict[str, OHLCVFrame] = {}
    for key in finest_keys:
        result = permute_frame(streams[key], order)
        permuted[key] = result
        permuted_finest_by_symbol[key.symbol] = result

    for key in streams:
        if key.timeframe is config.finest:
            continue
        base = permuted_finest_by_symbol.get(key.symbol)
        if base is None:
            raise ValueError(
                f"{key}: no {config.finest.value} stream for {key.symbol} to re-aggregate it from"
            )
        origin = config.day_origin if key.timeframe is Timeframe.D1 else None
        permuted[key] = resample(base, key.timeframe, origin=origin)
    return permuted
