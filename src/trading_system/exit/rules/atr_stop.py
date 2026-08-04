"""Stop anchored at entry, k×ATR away, with an optional volatility recompute."""

from trading_system.core.types import Price, Side
from trading_system.exit.context import ExitContext
from trading_system.exit.position import ManagedPosition
from trading_system.exit.rules._features import atr_key


class ATRStop:
    """Propose ``entry_price ∓ multiple × ATR`` as the stop.

    Anchored at the entry price, never at the running price — that is what
    distinguishes this from :class:`~trading_system.exit.rules.trailing_stop.
    TrailingStop`'s ATR-trail source, which anchors on the favourable extreme
    reached since entry and so moves with the trade. This one answers "how far
    should the stop from entry be, given how the instrument has been moving",
    which only needs re-answering when volatility itself changes.

    A :class:`~trading_system.exit.base.StopModifier`, not an
    :class:`~trading_system.exit.base.ExitRule`: it proposes a level and lets
    :meth:`~trading_system.exit.position.ManagedPosition.tighten_stop` decide
    whether the proposal is accepted. With ``recompute=True`` a widening ATR
    reading is exactly the kind of proposal the ratchet is for — refused, not
    specially guarded against here, since the ratchet is the one enforcement
    site for every modifier in this library, this one included.
    """

    def __init__(self, *, period: int = 14, multiple: float = 1.5, recompute: bool = True) -> None:
        """Configure the distance and whether it re-reads ATR every bar.

        Args:
            period: ATR smoothing period.
            multiple: Multiple of ATR used as the distance from entry.
            recompute: If ``True``, re-reads ATR every bar; the proposed level
                still only moves through the position's ratchet. If ``False``,
                ATR is read once, on the first bar this modifier sees, and the
                same level is proposed for the rest of the trade.

        Raises:
            ValueError: If ``period`` or ``multiple`` is not positive.
        """
        if period < 1:
            raise ValueError(f"period must be positive, got {period}")
        if multiple <= 0:
            raise ValueError(f"multiple must be positive, got {multiple}")
        self._period = period
        self._multiple = multiple
        self._recompute = recompute
        self._fixed_atr: float | None = None

    @property
    def name(self) -> str:
        """Stable identifier, e.g. ``atr_stop_14_1.5``."""
        return f"atr_stop_{self._period}_{self._multiple:g}"

    def on_bar(self, position: ManagedPosition, ctx: ExitContext) -> Price | None:
        """Propose the ATR-distance level for this bar.

        Args:
            position: Position being managed.
            ctx: View of the closed bar ``t``.

        Returns:
            The proposed level, or ``None`` if ATR has not warmed up.
        """
        atr_value = ctx.feature(atr_key(self._period))
        if atr_value is None:
            return None
        if not self._recompute:
            if self._fixed_atr is None:
                self._fixed_atr = atr_value
            atr_value = self._fixed_atr

        distance = self._multiple * atr_value
        if position.side is Side.BUY:
            return Price(position.entry_price - distance)
        return Price(position.entry_price + distance)

    def reset(self) -> None:
        """Forget the cached ATR reading, for ``recompute=False``."""
        self._fixed_atr = None
