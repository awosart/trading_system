"""Where decisions become fills: the two-bar delay, and how long an order rests."""

from datetime import UTC, datetime, timedelta
from typing import Any

import polars as pl
import pytest
from structlog.testing import capture_logs

from tests.backtest.conftest import (
    EURUSD_H1,
    bars,
    costless_registry,
    flat_costs,
    orchestrator,
    strategy,
)
from trading_system.backtest.config import BacktestConfig
from trading_system.backtest.orchestrator import BacktestResult, SignalDrop, StrategyBinding
from trading_system.core.instruments import InstrumentRegistry
from trading_system.core.types import Timeframe
from trading_system.data.models import OHLCVFrame
from trading_system.exit.base import IntrabarPolicy
from trading_system.exit.library import ExitLibrarySpec, ExitPresetSpec

START = datetime(2024, 3, 4, 18, 0, tzinfo=UTC)


def run(
    registry: InstrumentRegistry,
    preset: ExitPresetSpec,
    frame: OHLCVFrame,
    *,
    config: BacktestConfig | None = None,
    **strategy_kwargs: Any,
) -> BacktestResult:
    """One run of a price-only strategy over ``frame``."""
    spec = strategy(**strategy_kwargs)
    return orchestrator(
        registry=registry,
        streams={EURUSD_H1: frame},
        bindings=[StrategyBinding(spec=spec, exit_preset=preset, keys=(EURUSD_H1,))],
        config=config,
        costs=flat_costs(),
    ).run()


class TestBarCloseFillsAtTheNextOpen:
    """A ``BAR_CLOSE`` exit fills at ``open(t+1)``, never at ``close(t)``.

    The two prices are equal in a gapless series, which is why every test here
    runs over one that gaps: with ``open(t+1) == close(t)`` an implementation
    that fills on the close passes, and the defect ships.
    """

    #: How far bar 11 opens above bar 10's close. Large enough that filling at
    #: the wrong one of the two is a difference no rounding can explain.
    GAP = 0.0100

    @pytest.fixture
    def gapped(self) -> OHLCVFrame:
        """Rising hourly bars where bar 11 opens a full point above bar 10's close.

        The exit is decided on bar 10's close and fills on bar 11's open, so the
        gap sits exactly across the handover this test is about.
        """
        closes = [1.1000 + 0.0010 * i for i in range(1, 13)]
        opens = [1.1000, *closes[:-1]]
        opens[11] += self.GAP
        rows: list[dict[str, Any]] = []
        for index, (opening, close) in enumerate(zip(opens, closes, strict=True)):
            top = max(opening, close)
            bottom = min(opening, close)
            rows.append(
                {
                    "timestamp": START + timedelta(hours=index),
                    "open": opening,
                    "high": top + 0.0005,
                    "low": bottom - 0.0005,
                    "close": close,
                    "volume": 1000.0,
                }
            )
        return OHLCVFrame(pl.DataFrame(rows), symbol="EURUSD", timeframe=Timeframe.H1)

    @pytest.fixture
    def time_boxed(self, library: ExitLibrarySpec) -> ExitPresetSpec:
        """Close after ten bars — a ``BAR_CLOSE`` rule, which names no price."""
        spec = next(item for item in library.presets if item.id == "time_boxed")
        return spec.model_copy(
            update={"rules": [rule.model_copy(update={"max_bars_held": 10}) for rule in spec.rules]}
        )

    def test_the_exit_fills_at_the_next_bars_open_and_not_at_the_decision_close(
        self, registry: InstrumentRegistry, time_boxed: ExitPresetSpec, gapped: OHLCVFrame
    ) -> None:
        result = run(costless_registry(registry), time_boxed, gapped, invalidation=1.05)
        trade = result.trades[0]

        prices = gapped.df
        close_of_decision_bar = prices["close"][10]
        open_of_fill_bar = prices["open"][11]
        assert open_of_fill_bar == pytest.approx(close_of_decision_bar + self.GAP)

        # With costs removed a fill is its mid price exactly, so this is a
        # direct reading of which of the two prices the exit was booked at.
        exit_price = float(trade.gross) / float(trade.size) / 100_000 + trade.entry_price
        assert exit_price == pytest.approx(open_of_fill_bar, abs=1e-9)
        assert exit_price != pytest.approx(close_of_decision_bar, abs=1e-6)

    def test_the_fill_is_stamped_at_the_next_bars_open_time(
        self, registry: InstrumentRegistry, time_boxed: ExitPresetSpec, gapped: OHLCVFrame
    ) -> None:
        """The instant, not only the price: bar 11 opens one hour after bar 10."""
        result = run(costless_registry(registry), time_boxed, gapped, invalidation=1.05)
        trade = result.trades[0]
        assert trade.closed_at == START + timedelta(hours=11)

    def test_the_gap_is_carried_by_the_trade_rather_than_smoothed_away(
        self, registry: InstrumentRegistry, time_boxed: ExitPresetSpec, gapped: OHLCVFrame
    ) -> None:
        """A run that filled on the close would report a materially smaller profit."""
        result = run(costless_registry(registry), time_boxed, gapped, invalidation=1.05)
        trade = result.trades[0]
        gap_value = (
            float(trade.size) * 100_000 * self.GAP
        )  # what the gap alone is worth on this size
        assert float(trade.gross) > gap_value


class TestEntryOrderLifetime:
    """``EntryOrderSpec.expire_after_bars`` is the authority; the config is a fallback.

    P06 stage 1 left the field unread, on the grounds that an order's lifetime is
    Execution's business. This is the layer that reads it, and these tests pin
    which of the two numbers wins — the point of naming the config field
    ``default_entry_order_ttl_bars`` rather than ``entry_order_ttl_bars``.
    """

    #: A limit 30 points below the trigger bar's close. Bars 1 and 2 stay above
    #: it; bar 3 trades through it.
    OFFSET = 0.0030

    @pytest.fixture
    def late_touch(self) -> OHLCVFrame:
        """Exactly one trigger, and the limit only reachable three bars later.

        Bar 0 closes up, so the setup triggers there and the order is placed for
        bar 1 onwards, resting at ``1.1000 - 0.0030``. Bars 1 and 2 close *down*,
        which matters twice over: their lows stay clear of the level, and — the
        part a first draft of this fixture got wrong — they raise no new setup.
        With them rising, the order placed on bar 2 would reach the level on bar
        3 under any lifetime at all, and "the order expired" would be
        indistinguishable from "a later order filled instead".

        Bar 3 trades through the level; bar 4 closes down, so nothing new is
        triggered before the data ends.
        """
        return bars(
            [1.1000, 1.0998, 1.0996, 1.0950, 1.0949],
            start=START,
            first_open=1.0990,
            spread=0.0005,
        )

    def _limit_strategy(self, **kwargs: Any) -> dict[str, Any]:
        """A strategy placing a limit order at ``close - OFFSET``."""
        return {
            "order": {"type": "LIMIT", "offset": self.OFFSET},
            "invalidation": 1.0900,
            **kwargs,
        }

    def test_an_order_with_expire_after_bars_three_rests_long_enough_to_fill(
        self, registry: InstrumentRegistry, preset: ExitPresetSpec, late_touch: OHLCVFrame
    ) -> None:
        result = run(
            costless_registry(registry),
            preset,
            late_touch,
            **self._limit_strategy(expire_after_bars=3),
        )
        assert result.expired_orders == 0
        assert result.fills >= 1
        assert result.open_at_end + len(result.trades) == 1

    def test_an_order_with_no_lifetime_of_its_own_rests_one_bar_and_expires(
        self, registry: InstrumentRegistry, preset: ExitPresetSpec, late_touch: OHLCVFrame
    ) -> None:
        """The config's fallback applies, and it is one bar by default."""
        result = run(
            costless_registry(registry),
            preset,
            late_touch,
            **self._limit_strategy(expire_after_bars=None),
        )
        assert result.expired_orders >= 1
        assert result.trades == []
        assert result.open_at_end == 0

    def test_the_specs_lifetime_beats_the_configs_fallback(
        self, registry: InstrumentRegistry, preset: ExitPresetSpec, late_touch: OHLCVFrame
    ) -> None:
        """Set the fallback to something else entirely; the spec still decides.

        With the fallback at 1 and the spec at 3 the two agree on "it fills", so
        that pairing cannot tell precedence from coincidence. Here the fallback
        would keep the order alive well past bar 3 and the spec cuts it short at
        1 — so a run that filled would be reading the wrong number.
        """
        config = BacktestConfig(atr_baseline_bars=5, atr_period=3, default_entry_order_ttl_bars=9)
        result = run(
            costless_registry(registry),
            preset,
            late_touch,
            config=config,
            **self._limit_strategy(expire_after_bars=1),
        )
        assert result.expired_orders >= 1
        assert result.trades == []

    def test_the_same_fallback_governs_when_the_spec_stays_silent(
        self, registry: InstrumentRegistry, preset: ExitPresetSpec, late_touch: OHLCVFrame
    ) -> None:
        """The mirror of the previous test: silence defers to the config."""
        config = BacktestConfig(atr_baseline_bars=5, atr_period=3, default_entry_order_ttl_bars=9)
        result = run(
            costless_registry(registry),
            preset,
            late_touch,
            config=config,
            **self._limit_strategy(expire_after_bars=None),
        )
        assert result.expired_orders == 0
        assert result.fills >= 1


class TestSilentDropsAreCounted:
    """A run that produced few trades must be able to say why."""

    def test_signals_blocked_by_the_position_limit_are_counted_not_discarded(
        self, registry: InstrumentRegistry, preset: ExitPresetSpec
    ) -> None:
        """Otherwise this is indistinguishable from a selective strategy.

        The same reasoning P06 applied to dropped entry signals: the defect
        surfaces downstream as "not many trades", which is exactly what a
        working, choosy strategy looks like.
        """
        rising = bars([1.1000 + 0.0010 * i for i in range(1, 30)], first_open=1.0990)
        result = run(
            costless_registry(registry),
            preset,
            rising,
            invalidation=1.05,
            max_concurrent_positions=1,
        )
        assert result.signal_drops[SignalDrop.POSITION_LIMIT] > 0

    def test_every_reason_is_present_even_at_zero(
        self, registry: InstrumentRegistry, preset: ExitPresetSpec
    ) -> None:
        """A missing key is indistinguishable from a counter somebody forgot.

        So every reason is present from the start, and a run in which the limit
        never bound records that as a fact rather than as an absence.
        """
        flat = bars([1.1000] * 5)
        result = run(costless_registry(registry), preset, flat, invalidation=1.05)
        assert set(result.signal_drops) == set(SignalDrop)


class TestIntrabarPolicyPrecedence:
    """``BacktestConfig.intrabar_policy`` beats the preset's own field.

    Resolving several exit levels touched in one bar is a property of what the
    run is willing to assume about intrabar order, not of any one exit's
    design — a run comparing the same preset under both policies would
    otherwise have to duplicate the preset with only that field changed. The
    preset's own field remains the fallback for building a plan with no run
    behind it at all (``ExitLibrary`` construction, ``tests/exit/test_library.py``).

    Both directions are tested, not just one: a config and a preset that happen
    to agree cannot tell precedence from coincidence, the same reasoning
    ``TestEntryOrderLifetime`` above already applies to ``expire_after_bars``.
    """

    def _open_position_plan(
        self,
        registry: InstrumentRegistry,
        preset: ExitPresetSpec,
        config: BacktestConfig,
    ) -> IntrabarPolicy:
        """Open one position under ``config`` and read its plan's actual policy.

        The stop and target are both far outside the tiny, near-flat series that
        follows the trigger bar, so the position is still open when the run
        ends — closed positions' plans are not retained anywhere a test can
        reach them, only :attr:`~trading_system.backtest.portfolio.Portfolio.open_positions`
        survives past :meth:`~trading_system.backtest.orchestrator.Orchestrator.run`.
        """
        spec = strategy(invalidation=1.05, stop_pips=200.0)
        frame = bars([1.1000, 1.0995, 1.0995, 1.0995], first_open=1.0980)
        instance = orchestrator(
            registry=costless_registry(registry),
            streams={EURUSD_H1: frame},
            bindings=[StrategyBinding(spec=spec, exit_preset=preset, keys=(EURUSD_H1,))],
            config=config,
            costs=flat_costs(),
        )
        instance.run()
        open_positions = instance.portfolio.open_positions
        assert len(open_positions) == 1, "fixture must leave exactly one position open"
        return open_positions[0].plan.intrabar_policy

    def test_a_pessimistic_config_overrides_an_optimistic_preset(
        self, registry: InstrumentRegistry, preset: ExitPresetSpec
    ) -> None:
        optimistic_preset = preset.model_copy(update={"intrabar_policy": IntrabarPolicy.OPTIMISTIC})
        config = BacktestConfig(
            atr_baseline_bars=5, atr_period=3, intrabar_policy=IntrabarPolicy.PESSIMISTIC
        )
        policy = self._open_position_plan(registry, optimistic_preset, config)
        assert policy is IntrabarPolicy.PESSIMISTIC

    def test_an_optimistic_config_overrides_a_pessimistic_preset(
        self, registry: InstrumentRegistry, preset: ExitPresetSpec
    ) -> None:
        assert preset.intrabar_policy is IntrabarPolicy.PESSIMISTIC, "fixture assumes the default"
        config = BacktestConfig(
            atr_baseline_bars=5, atr_period=3, intrabar_policy=IntrabarPolicy.OPTIMISTIC
        )
        policy = self._open_position_plan(registry, preset, config)
        assert policy is IntrabarPolicy.OPTIMISTIC


class TestAtrRatioCoverage:
    """The bar-denominated ATR_UNAVAILABLE counterpart to CostStats' fill-denominated one.

    ``cost_degradations``'s ``ATR_UNAVAILABLE`` divides by fills; a low-frequency
    strategy can take a handful of fills across thousands of bars, and that
    fraction then says nothing about the rest of the data. ``atr_ratio_coverage``
    divides by bars instead, so warmup shows up honestly regardless of how many
    fills a run happened to produce.
    """

    def test_warmup_is_counted_against_every_bar_of_the_stream(
        self, registry: InstrumentRegistry
    ) -> None:
        frame = bars([1.1000 + 0.0001 * i for i in range(10)])
        config = BacktestConfig(atr_period=3, atr_baseline_bars=5)
        result = orchestrator(
            registry=costless_registry(registry),
            streams={EURUSD_H1: frame},
            bindings=[],
            config=config,
            costs=flat_costs(),
        ).run()
        coverage = result.atr_ratio_coverage[EURUSD_H1]
        assert coverage.bars == 10
        # atr_ratio warms up over atr_period + atr_baseline_bars - 1 bars.
        assert coverage.unavailable == 3 + 5 - 1
        assert coverage.fraction == pytest.approx(0.7)

    def test_the_fraction_is_zero_for_an_empty_stream_not_undefined(self) -> None:
        from trading_system.backtest.orchestrator import AtrRatioCoverage

        assert AtrRatioCoverage(bars=0, unavailable=0).fraction == 0.0


class TestSignalTimeframeMustMatchTheStream:
    """A mismatched stream timeframe changes what every bar-count field means.

    A strategy trades at ``timeframes.signal_tf``; binding it to a stream at a
    different timeframe silently changes what ``confirmation_window_bars``,
    ``expire_after_bars`` and ``cooldown_bars_after_loss`` all mean.
    """

    def test_a_mismatched_stream_timeframe_is_rejected_at_construction(
        self, registry: InstrumentRegistry, preset: ExitPresetSpec
    ) -> None:
        spec = strategy(signal_tf=Timeframe.H4)
        frame = bars([1.1000] * 5)  # EURUSD H1, from the conftest default
        with pytest.raises(ValueError, match="signal_tf"):
            orchestrator(
                registry=costless_registry(registry),
                streams={EURUSD_H1: frame},
                bindings=[StrategyBinding(spec=spec, exit_preset=preset, keys=(EURUSD_H1,))],
                costs=flat_costs(),
            )

    def test_a_matching_stream_timeframe_is_accepted(
        self, registry: InstrumentRegistry, preset: ExitPresetSpec
    ) -> None:
        spec = strategy(signal_tf=Timeframe.H1)
        frame = bars([1.1000] * 5)
        instance = orchestrator(
            registry=costless_registry(registry),
            streams={EURUSD_H1: frame},
            bindings=[StrategyBinding(spec=spec, exit_preset=preset, keys=(EURUSD_H1,))],
            costs=flat_costs(),
        )
        assert instance is not None


class TestAtrWarmupIsReportedNotAutoScaled:
    """Excess ``atr_ratio`` warmup is a warning at construction, never a silent rescale.

    CLAUDE.md rules out config magic: ``atr_baseline_bars`` is one number a run
    applies to every stream it trades, and scaling it to fit whatever data
    happened to be passed in would turn a config choice into a moving target
    nobody could reason about from the config alone. This reports the measured
    fraction and leaves the decision — a smaller ``atr_baseline_bars``, a finer
    timeframe, or accepting the warmup — to whoever reads the log.
    """

    def test_a_warmup_exceeding_the_threshold_is_reported_with_the_fraction(
        self, registry: InstrumentRegistry
    ) -> None:
        frame = bars([1.1000 + 0.0001 * i for i in range(10)])
        config = BacktestConfig(atr_period=3, atr_baseline_bars=5)
        with capture_logs() as logs:
            orchestrator(
                registry=costless_registry(registry),
                streams={EURUSD_H1: frame},
                bindings=[],
                config=config,
                costs=flat_costs(),
            )
        warnings = [
            entry
            for entry in logs
            if entry["event"] == "backtest.atr_ratio_warmup_exceeds_threshold"
        ]
        assert len(warnings) == 1
        warning = warnings[0]
        assert warning["log_level"] == "warning"
        assert warning["stream"] == str(EURUSD_H1)
        # warmup = atr_period + atr_baseline_bars - 1 = 3 + 5 - 1 = 7, of 10 bars.
        assert warning["unavailable_bars"] == 7
        assert warning["total_bars"] == 10
        assert warning["fraction"] == pytest.approx(0.7)

    def test_a_warmup_within_the_threshold_is_silent(self, registry: InstrumentRegistry) -> None:
        frame = bars([1.1000 + 0.0001 * i for i in range(1000)])
        config = BacktestConfig(atr_period=3, atr_baseline_bars=5)
        with capture_logs() as logs:
            orchestrator(
                registry=costless_registry(registry),
                streams={EURUSD_H1: frame},
                bindings=[],
                config=config,
                costs=flat_costs(),
            )
        warnings = [
            entry
            for entry in logs
            if entry["event"] == "backtest.atr_ratio_warmup_exceeds_threshold"
        ]
        assert warnings == []
