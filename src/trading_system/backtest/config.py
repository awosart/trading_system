"""Settings of a run that are the orchestrator's own decisions.

Everything here is a number some other module needs and cannot derive: how the
volatility ratio the cost model prices with is measured, where the trading day is
cut, and how long an entry order that names no lifetime of its own rests.
"""

from datetime import datetime, time
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from trading_system.data.resample import DayOrigin
from trading_system.exit.base import IntrabarPolicy


class BacktestConfig(BaseModel):
    """One run's parameters.

    Attributes:
        account_currency: Denomination of the account. Every money figure the run
            reports is in it.
        starting_balance: Opening balance.
        day_origin_tz: IANA zone the trading day is anchored in. Never a fixed
            UTC offset — 17:00 New York is 22:00 UTC in winter and 21:00 in
            summer, and a stored offset drifts twice a year.
        day_origin_at: Local wall-clock time the trading day starts at.
        atr_period: Period of the ATR the cost model's volatility ratio is built
            from, and of the ATR handed to the Risk Engine for stop and sizing
            methods that need one.
        atr_baseline_bars: Rolling window the ATR is divided by to get
            ``MarketState.atr_ratio``. The window has to be long relative to
            ``atr_period`` so the ratio measures "now against normal" rather than
            "now against recently", and it is a **bar count rather than a
            calendar span** because P12's slippage coefficient ``k`` is
            calibrated on the ratio's distribution, which must not change shape
            between timeframes.

            .. warning::

               The warmup is ``atr_period + atr_baseline_bars - 1`` bars, during
               which ``atr_ratio`` is ``None`` and the cost model prices on its
               prior. On five years of H1 (~31 000 bars) the default costs 1.7%
               of the run. **On five years of D1 (~1 300 bars) it costs 40%**, and
               the number must come down. This is not left to be guessed:
               :attr:`~trading_system.execution.costs.CostStats.fraction` of
               ``ATR_UNAVAILABLE`` is reported by every run, and
               ``atr_unavailable_report_threshold`` decides when the run says so
               out loud.
        atr_unavailable_report_threshold: Fraction of fills priced without a
            volatility measurement above which the run logs a warning naming the
            figure. A run that quietly prices half its fills on a prior is the
            defect this threshold exists to surface.
        default_entry_order_ttl_bars: How many bars an entry order rests when its
            :class:`~trading_system.strategies.schema.EntryOrderSpec` names no
            ``expire_after_bars``. A **fallback only** — the spec's field is the
            authority, and the name says so, because two lifetimes with different
            values and equal standing is exactly how they drift apart.
        intrabar_policy: How several exit levels touched in one bar are resolved.
            Pessimistic unless deliberately changed. **Authoritative over any
            exit preset's own** :attr:`~trading_system.exit.library.ExitPresetSpec.intrabar_policy`
            — the run decides what it is willing to assume about intrabar
            order, not any one exit's design, so this field wins whenever a
            plan is built inside a run. A preset's own field is only the
            fallback for building a plan with no run behind it at all (an
            :class:`~trading_system.exit.library.ExitLibrary`, a standalone
            test).
        evaluation_start: Earliest bar-close instant at which entry signals may
            be recognised at all. ``None`` (the default) evaluates from the
            first bar, exactly as before this field existed. Set by
            :mod:`trading_system.validation.walkforward` so a fold's warmup
            prefix (``data_start .. trade_start``) publishes bars — feature
            pipelines and the ATR baseline still see them — without the
            strategy ever being asked to open a position on one. Enforced
            where entry recognition itself happens
            (:meth:`~trading_system.backtest.orchestrator.Orchestrator.on_recognise`),
            not by dropping a signal after the fact: a signal that is never
            evaluated cannot leave a trace in any drop counter to be confused
            with a strategy that looked and declined.
        evaluation_end: Instant at or after which entry signals stop being
            recognised. ``None`` (the default) evaluates through the last bar.
            Positions already open keep being managed by their own exit rules
            past this instant — this field only stops *new* ones — which is
            what lets a walk-forward fold's out-of-sample run drain positions
            still open at its own ``trade_end`` on real subsequent bars
            instead of inventing a forced-liquidation fill.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    account_currency: str = Field(default="USD", min_length=1)
    starting_balance: Decimal = Field(default=Decimal(100_000), gt=0)

    day_origin_tz: str = Field(default="America/New_York", min_length=1)
    day_origin_at: time = time(17, 0)

    atr_period: int = Field(default=14, gt=0)
    atr_baseline_bars: int = Field(default=500, gt=1)
    atr_unavailable_report_threshold: float = Field(default=0.05, ge=0, le=1)

    default_entry_order_ttl_bars: int = Field(default=1, gt=0)

    intrabar_policy: IntrabarPolicy = IntrabarPolicy.PESSIMISTIC

    evaluation_start: datetime | None = None
    evaluation_end: datetime | None = None

    @property
    def day_origin(self) -> DayOrigin:
        """Where the trading day starts, as the data layer expresses it."""
        return DayOrigin(tz=self.day_origin_tz, at=self.day_origin_at)

    @property
    def atr_warmup_bars(self) -> int:
        """Bars before ``atr_ratio`` becomes available on a stream."""
        return self.atr_period + self.atr_baseline_bars - 1

    @model_validator(mode="after")
    def _check_evaluation_window(self) -> "BacktestConfig":
        """Reject a naive or inverted evaluation window.

        Returns:
            The validated config.

        Raises:
            ValueError: If either bound is tz-naive, or both are given with
                the start at or after the end — an empty window silently
                evaluates nothing, which must be visible at load time rather
                than discovered as a walk-forward fold with zero trades.
        """
        for name, value in (
            ("evaluation_start", self.evaluation_start),
            ("evaluation_end", self.evaluation_end),
        ):
            if value is not None and value.tzinfo is None:
                raise ValueError(f"{name} must be tz-aware")
        if (
            self.evaluation_start is not None
            and self.evaluation_end is not None
            and self.evaluation_start >= self.evaluation_end
        ):
            raise ValueError(
                f"evaluation_start {self.evaluation_start} must be before "
                f"evaluation_end {self.evaluation_end}"
            )
        return self
