"""Registry mapping configuration names to indicator classes.

Two things depend on this table. Configuration files name indicators by string,
and the registry is what turns ``kind: "rsi"`` into an :class:`RSI` instance —
failing loudly at construction time rather than deep inside a backtest. The test
suite iterates the same table, so an indicator added here is automatically held
to the parity, no-lookahead and warmup contracts without anyone remembering to
add a test.
"""

from collections.abc import Mapping
from typing import Any

from trading_system.core.exceptions import ValidationError
from trading_system.features.base import BaseIndicator
from trading_system.features.indicators.momentum import (
    CCI,
    MFI,
    ROC,
    RSI,
    Stochastic,
    WilliamsR,
)
from trading_system.features.indicators.structure import (
    MarketStructure,
    PivotPoints,
    RangeState,
    SwingPoints,
)
from trading_system.features.indicators.trend import (
    ADX,
    EMA,
    HMA,
    MACD,
    SMA,
    VWMA,
    WMA,
    Donchian,
    Ichimoku,
    Supertrend,
)
from trading_system.features.indicators.volatility import (
    ATR,
    BollingerBands,
    Choppiness,
    Keltner,
    StdDev,
)
from trading_system.features.indicators.volume import (
    OBV,
    AnchoredVwap,
    RelativeVolume,
    SessionVwap,
    VolumeMA,
)

#: Configuration name to indicator class. Names are lower-case and stable;
#: renaming one breaks existing configs, so treat them as public API.
INDICATOR_TYPES: dict[str, type[BaseIndicator[Any]]] = {
    # Trend
    "sma": SMA,
    "ema": EMA,
    "wma": WMA,
    "hma": HMA,
    "vwma": VWMA,
    "macd": MACD,
    "adx": ADX,
    "supertrend": Supertrend,
    "ichimoku": Ichimoku,
    "donchian": Donchian,
    # Volatility
    "atr": ATR,
    "stddev": StdDev,
    "bbands": BollingerBands,
    "keltner": Keltner,
    "chop": Choppiness,
    # Momentum
    "rsi": RSI,
    "stoch": Stochastic,
    "cci": CCI,
    "willr": WilliamsR,
    "roc": ROC,
    "mfi": MFI,
    # Volume
    "obv": OBV,
    "vwap_session": SessionVwap,
    "vwap_anchored": AnchoredVwap,
    "volume_ma": VolumeMA,
    "rvol": RelativeVolume,
    # Structure
    "swing": SwingPoints,
    "pivots": PivotPoints,
    "structure": MarketStructure,
    "range": RangeState,
}


def build_indicator(kind: str, params: Mapping[str, Any] | None = None) -> BaseIndicator[Any]:
    """Construct an indicator by configuration name.

    Args:
        kind: A key of :data:`INDICATOR_TYPES`.
        params: Constructor arguments. Every indicator's parameters have
            defaults, so an empty mapping yields the conventional settings.

    Returns:
        The constructed indicator.

    Raises:
        ValidationError: If the name is unknown, or the parameters do not match
            the indicator's signature or fail its own validation. Raised here so
            a bad config fails at startup rather than mid-backtest.
    """
    indicator_type = INDICATOR_TYPES.get(kind)
    if indicator_type is None:
        raise ValidationError(f"unknown indicator {kind!r}; known kinds: {sorted(INDICATOR_TYPES)}")
    try:
        return indicator_type(**dict(params or {}))
    except (TypeError, ValueError) as error:
        raise ValidationError(f"cannot build indicator {kind!r}: {error}") from error


def default_indicators() -> list[BaseIndicator[Any]]:
    """One instance of every registered indicator, at its default parameters.

    Returns:
        Indicators in registry order. ``vwap_anchored`` is built without an
        explicit anchor so that it anchors to the first bar and is valid
        everywhere; with an anchor mid-frame its leading rows are null by
        design, which is not a warmup.
    """
    return [build_indicator(kind) for kind in INDICATOR_TYPES]
