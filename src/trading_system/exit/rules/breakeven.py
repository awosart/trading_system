"""Move the stop to breakeven once the trade has earned it."""

from trading_system.core.types import Price, Side
from trading_system.exit.context import ExitContext
from trading_system.exit.position import ManagedPosition


class BreakevenMove:
    """Propose ``entry_price ± spread`` once the trade reaches ``activation_r``.

    Stateless by construction, not merely by convention: activation is checked
    fresh every bar against the bar's own favourable extreme, and the constant
    level this proposes never depends on *when* the threshold was first
    reached. Once reached, the position's stop is at breakeven for good — not
    because this modifier remembers reaching the threshold, but because
    :meth:`~trading_system.exit.position.ManagedPosition.tighten_stop` already
    refuses to move it back. A later bar that pulls back below the threshold
    simply gets a declined re-proposal, which is a no-op, not a retreat.
    """

    def __init__(self, *, activation_r: float, spread: float = 0.0) -> None:
        """Configure the activation threshold and the breakeven offset.

        Args:
            activation_r: Multiple of the initial risk the trade must reach,
                by its favourable extreme, before this proposes anything.
            spread: Extra distance past the entry, in price units, to clear the
                bid/ask spread rather than land exactly on it. Zero means
                exactly at entry.

        Raises:
            ValueError: If ``activation_r`` is not positive or ``spread`` is
                negative.
        """
        if activation_r <= 0:
            raise ValueError(f"activation_r must be positive, got {activation_r}")
        if spread < 0:
            raise ValueError(f"spread must be non-negative, got {spread}")
        self._activation_r = activation_r
        self._spread = spread

    @property
    def name(self) -> str:
        """Stable identifier, e.g. ``breakeven_1r``."""
        return f"breakeven_{self._activation_r:g}r"

    def on_bar(self, position: ManagedPosition, ctx: ExitContext) -> Price | None:
        """Propose breakeven if this bar's extreme has reached the threshold.

        Args:
            position: Position being managed.
            ctx: View of the closed bar ``t``.

        Returns:
            ``entry_price ± spread``, once the threshold is reached; ``None``
            before that.
        """
        favourable = ctx.price("high") if position.side is Side.BUY else ctx.price("low")
        if favourable is None:
            return None
        if position.r_multiple(Price(favourable)) < self._activation_r:
            return None
        if position.side is Side.BUY:
            return Price(position.entry_price + self._spread)
        return Price(position.entry_price - self._spread)

    def reset(self) -> None:
        """No state to clear; activation is re-checked from scratch every bar."""
