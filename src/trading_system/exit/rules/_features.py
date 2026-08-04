"""Feature-key helpers shared by rules that read P03 indicators.

Exit does not import Entry's ``feature_key`` — pulling it in would import
``trading_system.entry``, which is exactly the dependency the two libraries are
built not to have on each other. The naming convention it implements
(``indicator.name`` for a single-output indicator, ``f"{indicator.name}_{channel}"``
for a multi-output one) is defined once, in P03's pipeline
(:func:`~trading_system.features.pipeline._column_names`), and both Entry's
``feature_key`` and this module read it off the indicator objects themselves
rather than each other. Reusing the indicator classes keeps that one definition
authoritative; this module does not re-derive the naming rule, only applies it.

A rule that reads one of these keys still depends on the series it runs over
actually carrying that column — building the right :class:`FeaturePipeline` for
a composed :class:`~trading_system.exit.plan.ExitPlan` is a P07 stage 3
concern, once ``library.json`` exists to declare it.
"""

from trading_system.core.types import Price, Side
from trading_system.exit.context import ExitContext
from trading_system.features.indicators.structure import SwingPoints
from trading_system.features.indicators.trend import EMA
from trading_system.features.indicators.volatility import ATR


def atr_key(period: int) -> str:
    """The column name carrying an ATR reading of ``period``.

    Args:
        period: ATR smoothing period.

    Returns:
        The feature key, e.g. ``"atr_14"``.
    """
    return ATR(period=period).name


def ma_key(period: int, source: str) -> str:
    """The column name carrying an EMA reading of ``period`` over ``source``.

    Args:
        period: Smoothing period.
        source: Price series the average is computed from, e.g. ``"close"``.

    Returns:
        The feature key, e.g. ``"ema_20"``.
    """
    return EMA(period=period, source=source).name


def swing_keys(lookback: int) -> tuple[str, str]:
    """The column names carrying the confirmed swing low and high.

    Args:
        lookback: Bars required on each side of a pivot.

    Returns:
        ``(low_key, high_key)``.
    """
    swing = SwingPoints(lookback=lookback)
    return f"{swing.name}_swing_low", f"{swing.name}_swing_high"


def swing_level(ctx: ExitContext, lookback: int, side: Side, buffer: float) -> Price | None:
    """The stop level implied by the nearest confirmed swing, plus a buffer.

    Below the swing low for a long, above the swing high for a short — the
    buffer widens the distance in both cases, never narrows it.

    Args:
        ctx: View of the closed bar.
        lookback: Bars required on each side of a pivot.
        side: Side of the position being stopped out.
        buffer: Extra distance beyond the swing, in price units, non-negative.

    Returns:
        The level, or ``None`` if the swing feature is not carried by this
        series or has not warmed up on this bar.
    """
    low_key, high_key = swing_keys(lookback)
    level = ctx.feature(low_key if side is Side.BUY else high_key)
    if level is None:
        return None
    return Price(level - buffer) if side is Side.BUY else Price(level + buffer)
