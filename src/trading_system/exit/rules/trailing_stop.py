"""A stop that follows the trade once it has earned the right to, one way of four.

:class:`TrailingStop` needs no state of its own to trail. Each source below
proposes a level computed fresh from the current bar — this bar's extreme for
ATR-trail, a fixed lookback window for chandelier, a moving-average reading, or
the nearest confirmed swing — and it is
:meth:`~trading_system.exit.position.ManagedPosition.tighten_stop` that turns
"recompute every bar" into "only ever move toward the entry": a proposal that
would loosen the stop is declined by the ratchet, so a bar-by-bar re-read of
today's high already behaves like tracking the peak since entry, without this
class remembering the peak itself. The one piece of memory this class does
keep is whether the activation threshold has ever been reached — a single
``bool``, since the underlying prices already keep the rest.
"""

from dataclasses import dataclass

from trading_system.core.types import Price, Side
from trading_system.exit.context import ExitContext
from trading_system.exit.position import ManagedPosition
from trading_system.exit.rules._features import atr_key, ma_key, swing_level


@dataclass(frozen=True)
class AtrTrail:
    """Trail behind this bar's favourable extreme, by a multiple of ATR.

    Attributes:
        period: ATR smoothing period.
        multiple: Multiple of ATR the trail sits behind the extreme.
    """

    period: int = 14
    multiple: float = 3.0

    def __post_init__(self) -> None:
        """Reject a non-positive period or multiple."""
        if self.period < 1:
            raise ValueError(f"period must be positive, got {self.period}")
        if self.multiple <= 0:
            raise ValueError(f"multiple must be positive, got {self.multiple}")


@dataclass(frozen=True)
class Chandelier:
    """Trail behind the highest high (or lowest low) of a fixed lookback window.

    The classic Chandelier Exit: unlike :class:`AtrTrail`, the anchor is a
    window of bars rather than the running extreme since entry, so it forgets a
    spike once it scrolls out of the window.

    Attributes:
        lookback: Bars in the window, this bar included.
        period: ATR smoothing period.
        multiple: Multiple of ATR the trail sits behind the window's extreme.
    """

    lookback: int = 22
    period: int = 22
    multiple: float = 3.0

    def __post_init__(self) -> None:
        """Reject a non-positive lookback, period or multiple."""
        if self.lookback < 1:
            raise ValueError(f"lookback must be positive, got {self.lookback}")
        if self.period < 1:
            raise ValueError(f"period must be positive, got {self.period}")
        if self.multiple <= 0:
            raise ValueError(f"multiple must be positive, got {self.multiple}")


@dataclass(frozen=True)
class MaTrail:
    """Trail at (or just past) a moving average.

    Attributes:
        period: Moving-average period.
        source: Price series the average is computed from.
        buffer_atr_multiple: Extra ATR-scaled distance past the average, in the
            direction away from price. Zero trails exactly at the average.
        atr_period: ATR period the buffer is scaled by, read only when
            ``buffer_atr_multiple`` is positive.
    """

    period: int = 20
    source: str = "close"
    buffer_atr_multiple: float = 0.0
    atr_period: int = 14

    def __post_init__(self) -> None:
        """Reject a non-positive period or a negative buffer."""
        if self.period < 1:
            raise ValueError(f"period must be positive, got {self.period}")
        if self.buffer_atr_multiple < 0:
            raise ValueError(
                f"buffer_atr_multiple must be non-negative, got {self.buffer_atr_multiple}"
            )


@dataclass(frozen=True)
class SwingTrail:
    """Trail just beyond the nearest confirmed swing high or low.

    The same computation :class:`~trading_system.exit.rules.structure_stop.
    StructureStop` uses, gated instead behind this modifier's activation
    threshold — active only once the trade has earned it, rather than from the
    first bar.

    Attributes:
        lookback: Bars required on each side of a pivot.
        buffer_atr_multiple: Extra ATR-scaled distance beyond the swing.
        atr_period: ATR period the buffer is scaled by, read only when
            ``buffer_atr_multiple`` is positive.
    """

    lookback: int = 5
    buffer_atr_multiple: float = 0.0
    atr_period: int = 14

    def __post_init__(self) -> None:
        """Reject a non-positive lookback or a negative buffer."""
        if self.lookback < 1:
            raise ValueError(f"lookback must be positive, got {self.lookback}")
        if self.buffer_atr_multiple < 0:
            raise ValueError(
                f"buffer_atr_multiple must be non-negative, got {self.buffer_atr_multiple}"
            )


#: Where a trail's level comes from.
TrailingSource = AtrTrail | Chandelier | MaTrail | SwingTrail


class TrailingStop:
    """Trail the stop once the trade has reached ``activation_r``, never before.

    Below the activation threshold this proposes nothing, leaving the stop
    wherever the position's other modifiers left it. Once the threshold has
    ever been reached — checked against the bar's own favourable extreme, so an
    intrabar touch counts even if the close pulled back — activation latches
    and does not un-latch; only :meth:`reset` clears it.
    """

    def __init__(self, source: TrailingSource, *, activation_r: float) -> None:
        """Configure the trailing source and the profit it takes to activate.

        Args:
            source: Which of the four ways to compute the trailing level.
            activation_r: Multiple of the initial risk the trade must reach,
                by its favourable extreme, before this starts proposing levels.

        Raises:
            ValueError: If ``activation_r`` is not positive.
        """
        if activation_r <= 0:
            raise ValueError(f"activation_r must be positive, got {activation_r}")
        self._source = source
        self._activation_r = activation_r
        self._activated = False

    @property
    def name(self) -> str:
        """Stable identifier naming the source and the activation threshold."""
        return f"trailing_stop_{self._source_label()}_{self._activation_r:g}r"

    def on_bar(self, position: ManagedPosition, ctx: ExitContext) -> Price | None:
        """Propose this bar's trailing level, once activated.

        Args:
            position: Position being managed.
            ctx: View of the closed bar ``t``.

        Returns:
            The proposed level, or ``None`` before activation or while the
            source's feature has not warmed up.
        """
        favourable = ctx.price("high") if position.side is Side.BUY else ctx.price("low")
        if favourable is None:
            return None
        if not self._activated:
            if position.r_multiple(Price(favourable)) < self._activation_r:
                return None
            self._activated = True
        return self._level(position, ctx, favourable)

    def reset(self) -> None:
        """Un-latch activation for a fresh run."""
        self._activated = False

    def _source_label(self) -> str:
        """A short name for the configured source, used in :attr:`name`."""
        source = self._source
        if isinstance(source, AtrTrail):
            return f"atr_{source.period}_{source.multiple:g}"
        if isinstance(source, Chandelier):
            return f"chandelier_{source.lookback}_{source.multiple:g}"
        if isinstance(source, MaTrail):
            return f"ma_{source.period}"
        return f"swing_{source.lookback}"

    def _level(
        self, position: ManagedPosition, ctx: ExitContext, favourable: float
    ) -> Price | None:
        """Compute this bar's proposed level from the configured source."""
        source = self._source
        long = position.side is Side.BUY

        if isinstance(source, AtrTrail):
            atr_value = ctx.feature(atr_key(source.period))
            if atr_value is None:
                return None
            distance = source.multiple * atr_value
            return Price(favourable - distance) if long else Price(favourable + distance)

        if isinstance(source, Chandelier):
            field = "high" if long else "low"
            window = [ctx.price(field, lookback) for lookback in range(source.lookback)]
            values = [value for value in window if value is not None]
            if len(values) != len(window):
                return None
            atr_value = ctx.feature(atr_key(source.period))
            if atr_value is None:
                return None
            distance = source.multiple * atr_value
            extreme = max(values) if long else min(values)
            return Price(extreme - distance) if long else Price(extreme + distance)

        if isinstance(source, MaTrail):
            ma_value = ctx.feature(ma_key(source.period, source.source))
            if ma_value is None:
                return None
            buffer = 0.0
            if source.buffer_atr_multiple > 0:
                atr_value = ctx.feature(atr_key(source.atr_period))
                if atr_value is None:
                    return None
                buffer = source.buffer_atr_multiple * atr_value
            return Price(ma_value - buffer) if long else Price(ma_value + buffer)

        buffer = 0.0
        if source.buffer_atr_multiple > 0:
            atr_value = ctx.feature(atr_key(source.atr_period))
            if atr_value is None:
                return None
            buffer = source.buffer_atr_multiple * atr_value
        return swing_level(ctx, source.lookback, position.side, buffer)
