"""The provider interface.

A provider's only job is to turn some external representation of market data
into a validated :class:`~trading_system.data.models.OHLCVFrame`. Everything
downstream depends on that guarantee, so providers never hand back a raw
DataFrame — normalisation happens at the boundary, once.
"""

from datetime import datetime
from typing import Protocol, runtime_checkable

from trading_system.core.types import Timeframe
from trading_system.data.models import OHLCVFrame


@runtime_checkable
class DataProvider(Protocol):
    """Source of OHLCV bars for a symbol."""

    def fetch(
        self,
        symbol: str,
        timeframe: Timeframe,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> OHLCVFrame:
        """Retrieve bars for a half-open ``[start, end)`` range.

        Args:
            symbol: Instrument identifier, in whatever form this provider expects.
            timeframe: Bar size requested.
            start: Inclusive lower bound on bar open time. ``None`` means from the
                earliest data available.
            end: Exclusive upper bound on bar open time. ``None`` means up to the
                latest data available.

        Returns:
            Validated bars, possibly empty.

        Raises:
            DataError: If the source is unreadable or its contents cannot be
                mapped onto the OHLCV schema.
        """
        ...
