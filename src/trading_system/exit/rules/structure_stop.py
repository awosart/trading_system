"""Stop placed just beyond the nearest confirmed swing structure."""

from trading_system.core.types import Price
from trading_system.exit.context import ExitContext
from trading_system.exit.position import ManagedPosition
from trading_system.exit.rules._features import atr_key, swing_level


class StructureStop:
    """Propose a level just beyond the nearest confirmed swing high or low.

    Unconditional — active from the first bar, unlike
    :class:`~trading_system.exit.rules.trailing_stop.TrailingStop`'s swing
    source, which is gated behind an activation threshold in R. This one is a
    :class:`~trading_system.exit.base.StopModifier`: as the swing feature's
    "most recent confirmed pivot" is displaced by a newer one, the proposed
    level moves with it, but only ever through
    :meth:`~trading_system.exit.position.ManagedPosition.tighten_stop`, so a
    structure shift that would loosen the stop is declined like any other.
    """

    def __init__(
        self, *, lookback: int = 5, buffer_atr_multiple: float = 0.0, atr_period: int = 14
    ) -> None:
        """Configure which swing to read and how much room to leave beyond it.

        Args:
            lookback: Bars required on each side of a pivot, passed to the
                swing indicator.
            buffer_atr_multiple: Extra ATR-scaled distance beyond the swing, to
                absorb noise around it. Zero means the level sits exactly at
                the swing.
            atr_period: ATR period the buffer is scaled by, read only when
                ``buffer_atr_multiple`` is positive.

        Raises:
            ValueError: If ``lookback`` is not positive or
                ``buffer_atr_multiple`` is negative.
        """
        if lookback < 1:
            raise ValueError(f"lookback must be positive, got {lookback}")
        if buffer_atr_multiple < 0:
            raise ValueError(f"buffer_atr_multiple must be non-negative, got {buffer_atr_multiple}")
        self._lookback = lookback
        self._buffer_multiple = buffer_atr_multiple
        self._atr_period = atr_period

    @property
    def name(self) -> str:
        """Stable identifier, e.g. ``structure_stop_5``."""
        return f"structure_stop_{self._lookback}"

    def on_bar(self, position: ManagedPosition, ctx: ExitContext) -> Price | None:
        """Propose the swing-based level for this bar.

        Args:
            position: Position being managed.
            ctx: View of the closed bar ``t``.

        Returns:
            The proposed level, or ``None`` if the swing (or, with a buffer,
            the ATR) has not warmed up.
        """
        buffer = 0.0
        if self._buffer_multiple > 0:
            atr_value = ctx.feature(atr_key(self._atr_period))
            if atr_value is None:
                return None
            buffer = self._buffer_multiple * atr_value
        return swing_level(ctx, self._lookback, position.side, buffer)

    def reset(self) -> None:
        """No state to clear; every proposal is read fresh from the series."""
