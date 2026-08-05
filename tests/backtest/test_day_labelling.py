"""EquityPoint.day, calendar-aware: the single-bar-Friday defect and its fix.

The last H4 bar before a weekly close finishes a few hours into the
``[Fri 17:00 NY, Sat 17:00 NY)`` window FX never trades in at all —
``trading_day`` labels it "Friday" anyway, minting a trading day with exactly
one bar in it (see ``data/resample.py::trading_day_of_close`` for the fix and
the mechanism in full). This module proves the fix at the level a strategy or
a risk limit would actually see it: a full ``Orchestrator`` run over a
realistic, weekend-gapped H4 series.

**These tests are written to fail without the fix.** Reverting
``derive_run_calendar``'s call sites (``Orchestrator.__init__`` and
``RunInputs.orchestrator``) back to not passing a calendar reproduces the
defect and turns ``test_a_multi_week_h4_series_has_no_single_bar_trading_days``
red — verified by hand while writing this file, not merely asserted here.
"""

from collections import Counter
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import polars as pl

from tests.backtest.conftest import REGISTRY_PATH, harness_inputs, strategy
from trading_system.backtest.clock import StreamKey
from trading_system.backtest.orchestrator import StrategyBinding, derive_run_calendar
from trading_system.backtest.portfolio import Portfolio
from trading_system.core.instruments import InstrumentClass, load_instruments
from trading_system.core.types import Timeframe
from trading_system.data.models import OHLCVFrame
from trading_system.data.resample import FX_DAY_ORIGIN
from trading_system.data.sessions import AssetClass, TradingCalendar
from trading_system.exit.library import ExitPresetSpec
from trading_system.risk.circuit_breakers import CircuitBreakerConfig
from trading_system.risk.conversion import SameCurrencyConverter

EURUSD_H4 = StreamKey("EURUSD", Timeframe.H4)

#: A real Sunday 17:00 New York open. August sits in EDT (UTC-4), so 17:00 NY
#: is 21:00 UTC — the exact instant the FX week begins.
WEEK_START = datetime(2024, 8, 4, 21, 0, tzinfo=UTC)


def fx_gapped_h4_series(weeks: int, *, symbol: str = "EURUSD", base: float = 1.10) -> OHLCVFrame:
    """``weeks`` full FX trading weeks of H4 bars, gapped over the weekly close.

    Built from :class:`~trading_system.data.sessions.TradingCalendar` directly
    — a bar's open is included only when the market is open at that instant —
    rather than the flat, gapless striding :func:`~tests.backtest.conftest.bars`
    and :func:`~tests.backtest.conftest.swing_series` use elsewhere in this
    suite. Those helpers produce a continuous series with no weekend at all,
    which cannot reproduce a defect that is specifically about the gap.

    Args:
        weeks: How many full Sunday-open-to-Sunday-open weeks to generate.
        symbol: Instrument the frame is for.
        base: Price the series oscillates near.

    Returns:
        The frame. Every trading day in it, including the last one of each
        week (the one whose final bar spills into the closed weekend), has a
        full complement of bars once day-labelling is fixed — that uniformity
        is what makes a single-bar day easy to spot as a defect rather than
        requiring a magic count.
    """
    calendar = TradingCalendar(AssetClass.FX)
    end = WEEK_START + timedelta(weeks=weeks)
    rows: list[dict[str, object]] = []
    ts = WEEK_START
    previous = base
    index = 0
    while ts < end:
        if calendar.is_open(ts):
            close = previous + 0.0001 * ((index % 5) - 2)
            rows.append(
                {
                    "timestamp": ts,
                    "open": previous,
                    "high": max(previous, close) + 0.0003,
                    "low": min(previous, close) - 0.0003,
                    "close": close,
                    "volume": 1000.0,
                }
            )
            previous = close
            index += 1
        ts += timedelta(hours=4)
    return OHLCVFrame(pl.DataFrame(rows), symbol=symbol, timeframe=Timeframe.H4)


def _never_fires_binding(preset: ExitPresetSpec) -> StrategyBinding:
    """A bound strategy whose trigger cannot ever be true.

    Isolates day-labelling from trading logic entirely: zero signals, zero
    fills, zero trades — only the equity curve's own day labels are under
    test here.
    """
    spec = strategy(
        trigger={"type": "leaf", "op": "gt", "left": "price:close", "right": 999_999.0},
        signal_tf=Timeframe.H4,
    )
    return StrategyBinding(spec=spec, exit_preset=preset, keys=(EURUSD_H4,))


class TestNoSingleBarTradingDays:
    """DoD: an H4 series through a Friday close produces no single-bar days."""

    def test_a_multi_week_h4_series_has_no_single_bar_trading_days(
        self, preset: ExitPresetSpec
    ) -> None:
        registry = load_instruments(REGISTRY_PATH)
        series = fx_gapped_h4_series(weeks=3)
        inputs = harness_inputs(
            registry,
            streams={EURUSD_H4: series},
            bindings=[_never_fires_binding(preset)],
        )
        result = inputs.run()
        assert len(result.trades) == 0  # trading logic is not what this test is about

        per_day = Counter(point.day for point in result.curve)
        singleton_days = sorted(day for day, count in per_day.items() if count == 1)
        assert singleton_days == [], (
            f"{len(singleton_days)} single-bar trading day(s) found: {singleton_days}. "
            "Each is a week's last H4 bar, whose close spills into the closed weekend "
            "window and mints itself a trading day of its own."
        )

    def test_every_weekday_is_represented_five_times_per_week(self, preset: ExitPresetSpec) -> None:
        """A sharper form of the same check: exactly 5 day-labels/week, not 6."""
        registry = load_instruments(REGISTRY_PATH)
        series = fx_gapped_h4_series(weeks=3)
        inputs = harness_inputs(
            registry,
            streams={EURUSD_H4: series},
            bindings=[_never_fires_binding(preset)],
        )
        result = inputs.run()
        distinct_days = {point.day for point in result.curve}
        assert len(distinct_days) == 15  # 5 trading days * 3 weeks


class TestOrchestratorSuppliesARealCalendar:
    """DoD: Orchestrator/RunInputs pass a real calendar, never a silent None."""

    def test_derive_run_calendar_is_unambiguous_for_a_single_asset_class(self) -> None:
        registry = load_instruments(REGISTRY_PATH)
        assert registry["EURUSD"].asset_class is InstrumentClass.FX
        calendar = derive_run_calendar({EURUSD_H4: fx_gapped_h4_series(weeks=1)}, registry)
        assert calendar == TradingCalendar(AssetClass.FX)

    def test_derive_run_calendar_refuses_to_guess_across_mixed_asset_classes(self) -> None:
        registry = load_instruments(REGISTRY_PATH)
        xauusd = StreamKey("XAUUSD", Timeframe.H4)
        assert registry["XAUUSD"].asset_class is not InstrumentClass.FX
        calendar = derive_run_calendar(
            {
                EURUSD_H4: fx_gapped_h4_series(weeks=1),
                xauusd: fx_gapped_h4_series(weeks=1, symbol="XAUUSD"),
            },
            registry,
        )
        assert calendar is None

    def test_derive_run_calendar_is_none_for_no_streams(self) -> None:
        registry = load_instruments(REGISTRY_PATH)
        assert derive_run_calendar({}, registry) is None

    def test_run_inputs_orchestrator_wires_a_real_calendar_into_circuit_breakers(
        self, preset: ExitPresetSpec
    ) -> None:
        """White-box: CircuitBreakerConfig.calendar has no other externally visible seam.

        A breaker trip is the only public effect of this wiring, and forcing
        one deterministically on the exact spillover bar is not worth the
        fragility it would add to this test — reaching one attribute into
        the assembled engine is the more honest check.
        """
        registry = load_instruments(REGISTRY_PATH)
        series = fx_gapped_h4_series(weeks=1)
        inputs = harness_inputs(
            registry,
            streams={EURUSD_H4: series},
            bindings=[_never_fires_binding(preset)],
        )
        orchestrator = inputs.orchestrator()
        assert orchestrator._risk.breakers.config.calendar == TradingCalendar(AssetClass.FX)  # noqa: SLF001

    def test_an_explicit_calendar_on_circuit_breaker_config_is_not_overridden(
        self, preset: ExitPresetSpec
    ) -> None:
        registry = load_instruments(REGISTRY_PATH)
        series = fx_gapped_h4_series(weeks=1)
        chosen = TradingCalendar(AssetClass.CRYPTO)
        inputs = harness_inputs(
            registry,
            streams={EURUSD_H4: series},
            bindings=[_never_fires_binding(preset)],
            breakers=CircuitBreakerConfig(calendar=chosen),
        )
        orchestrator = inputs.orchestrator()
        assert orchestrator._risk.breakers.config.calendar == chosen  # noqa: SLF001

    def test_a_portfolio_or_breakers_config_built_directly_still_defaults_to_none(self) -> None:
        """The isolation path DoD asked to keep: no Orchestrator, no calendar."""
        registry = load_instruments(REGISTRY_PATH)
        portfolio = Portfolio(
            currency="USD",
            starting_balance=Decimal(10_000),
            instruments=registry,
            converter=SameCurrencyConverter(),
            day_origin=FX_DAY_ORIGIN,
        )
        assert portfolio._calendar is None  # noqa: SLF001
        assert CircuitBreakerConfig().calendar is None
