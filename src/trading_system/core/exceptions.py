"""Exception hierarchy for the trading system.

Every failure raised by the system derives from :class:`TradingSystemError`, so
a caller can catch the whole family without also swallowing library errors.
"""


class TradingSystemError(Exception):
    """Base class for every error raised by this package."""


class DataError(TradingSystemError):
    """Market data is missing, malformed, or fails an integrity check."""


class ValidationError(TradingSystemError):
    """A value violates a domain invariant."""


class RiskLimitError(TradingSystemError):
    """A risk limit would be breached by the requested action."""


class ExecutionError(TradingSystemError):
    """An order could not be placed, modified, or cancelled."""
