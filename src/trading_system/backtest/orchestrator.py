"""The layer every previous block referred to, and the only one that knows them all.

Six couplings were deliberately left dangling by P06, P07, P10 and P12, each with
a note saying "whoever holds both sides fills this in". This is that holder:

* :class:`~trading_system.execution.market_state.MarketState` arrives as plain
  scalars so that :mod:`trading_system.execution` never imports the feature
  pipeline. The ATR and its baseline are computed here, once per stream, before
  the loop.
* :meth:`~trading_system.exit.plan.ExitPlan.smallest_closing_fraction` is read
  here and handed to the Risk Engine as a bare ``Decimal``, so that
  :mod:`trading_system.risk` never imports the Exit Engine.
* :attr:`~trading_system.execution.orders.Fill.slippage_points` is fed to
  :meth:`~trading_system.risk.circuit_breakers.CircuitBreakers.record_fill`,
  which P10 stage 2 created as "a shape the execution layer will fill".
* :attr:`~trading_system.exit.context.ExitContext.reverse_signal_side` is filled
  from the entry engine, so that ``SignalReverseExit`` never imports Entry.
* :meth:`~trading_system.risk.models.AccountState.with_opened` is called on every
  approval, which is the contract that makes the portfolio guarantee hold across
  signals arriving at one instant — for free margin and exposure as well as for
  risk. The reservation is self-healing: the account is rebuilt from live
  positions at the next snapshot, so an approved order that never fills stops
  consuming margin without anything having to release it.
* ``InstrumentSpec.margin_rate`` finally has a reader. It reaches the engine
  through :class:`~trading_system.backtest.portfolio.Portfolio.used_margin`,
  which scales each position's frozen per-lot figure by what is left of it.
* ``EntryOrderSpec.expire_after_bars`` is finally read. P06 stage 1 left it
  untouched on the grounds that an order's lifetime is Execution's business, not
  Entry's. The config's ``default_entry_order_ttl_bars`` is a **fallback for a
  spec that names none**, and its name says so — two lifetimes of equal standing
  is how they end up disagreeing.

**Each position gets its own exit plan instance.** P07 stage 3 established that a
loaded ``ExitLibrary`` hands out live plans reusable across positions *in
sequence*, not concurrently — rules carry state (a trailing activation, which
rungs of a ladder have fired). Two positions open at once on one shared plan
would read each other's. So a binding holds the preset *spec* and builds a fresh
plan per position.

**Entry recognition is split from sizing.** Recognition is pure — it reads the
closed bar and produces signals, touching no money — so it runs before the exits,
which lets ``SignalReverseExit`` see on *this* bar that an opposing signal was
recognised. Sizing runs after the equity snapshot, against the account as the
instant left it. The full phase order is in :mod:`trading_system.backtest.engine`.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum

import polars as pl

from trading_system.backtest.clock import StreamKey
from trading_system.backtest.config import BacktestConfig
from trading_system.backtest.engine import BacktestEngine, BarEvent, BarStore, DataHandler
from trading_system.backtest.portfolio import EquityPoint, OpenPosition, Portfolio, TradeRecord
from trading_system.core.instruments import InstrumentClass, InstrumentRegistry, InstrumentSpec
from trading_system.core.logging import get_logger
from trading_system.core.types import Bar, OrderType, Price, Side
from trading_system.data.models import OHLCVFrame
from trading_system.data.sessions import AssetClass, TradingCalendar
from trading_system.entry.compiler import EntryEngine, compile_entry
from trading_system.entry.context import BarSeries
from trading_system.entry.features import (
    FeatureRegistry,
    iter_feature_refs,
    required_label_categories,
)
from trading_system.entry.labels import LabelCategory
from trading_system.entry.signal import EntrySignal
from trading_system.execution.costs import CostModel, accrue_swap
from trading_system.execution.fill_model import (
    GapFill,
    LimitTouch,
    MarketFillModel,
    NextBarOpen,
    RestingOrderModel,
    order_from_fill,
)
from trading_system.execution.market_state import MarketState
from trading_system.execution.orders import ExecutionOrder, Fill
from trading_system.exit.base import ExitDecision, ExitDropReason, ExitKind, empty_drop_counts
from trading_system.exit.context import ExitContext
from trading_system.exit.library import (
    ATRStopSpec,
    AtrTrailSpec,
    ChandelierSpec,
    ExitPresetSpec,
    MaTrailSpec,
    StructureStopSpec,
    TrailingSourceSpec,
    TrailingStopSpec,
    build_plan,
)
from trading_system.exit.plan import DeferredExit
from trading_system.exit.position import ManagedPosition
from trading_system.exit.pricing import decision_level
from trading_system.features.indicators.structure import SwingPoints
from trading_system.features.indicators.trend import EMA
from trading_system.features.indicators.volatility import ATR
from trading_system.features.pipeline import FeaturePipeline, FeatureSpec
from trading_system.prop.guard import (
    GuardDecision,
    PropAccountState,
    PropGuard,
    ProposedOrder,
)
from trading_system.prop.rules import day_origin_divergence
from trading_system.risk.circuit_breakers import SlippageReport
from trading_system.risk.conversion import FxConverter
from trading_system.risk.engine import RiskEngine
from trading_system.risk.models import AccountState, RiskDecision
from trading_system.strategies.schema import (
    Direction,
    EntryOrder,
    LimitOrder,
    MarketOrder,
    StrategySpec,
)

#: Direction to trading side. The same mapping ``compile_entry`` uses, written
#: against the enum rather than against its rendered text: a string comparison
#: here would be the defect P04 removed from ``FeatureRef``, where a name that
#: merely looked right resolved and then behaved as something else.
_SIDES: dict[Direction, Side] = {Direction.LONG: Side.BUY, Direction.SHORT: Side.SELL}

logger = get_logger(__name__)


@dataclass(frozen=True)
class StrategyBinding:
    """One strategy attached to the streams it trades.

    Attributes:
        spec: The strategy.
        exit_preset: Spec of the exit it pairs with. The *spec*, not a built
            plan: a fresh plan is built per position, because exit rules carry
            per-position state.
        keys: Streams this strategy runs on. One compiled evaluator per stream —
            an :class:`~trading_system.entry.compiler.EntryEvaluator` tracks one
            pending setup and is therefore bound to one bar stream.
    """

    spec: StrategySpec
    exit_preset: ExitPresetSpec
    keys: tuple[StreamKey, ...]


#: Only the instrument classes with an unambiguous weekly open/closed calendar.
#: Deliberately partial: FUTURES/INDEX/COMMODITY don't have one settlement-wide
#: convention the way FX, equities and crypto do, and guessing one would be
#: exactly the kind of silent wrong answer this module's day-labelling fix
#: exists to avoid, not a place to reproduce it.
_CALENDAR_ASSET_CLASS: dict[InstrumentClass, AssetClass] = {
    InstrumentClass.FX: AssetClass.FX,
    InstrumentClass.CRYPTO: AssetClass.CRYPTO,
    InstrumentClass.EQUITY: AssetClass.EQUITY,
}


def derive_run_calendar(
    streams: Mapping[StreamKey, OHLCVFrame], instruments: InstrumentRegistry
) -> TradingCalendar | None:
    """The one calendar a run's instruments make unambiguous, or ``None``.

    A :class:`~trading_system.data.sessions.TradingCalendar` needs a single
    asset class, and a portfolio does not always have one — FX and crypto
    traded together have no shared weekly closure, since crypto never closes
    at all. This measures what is actually being traded rather than assuming
    it: when every stream's symbol resolves to the same, calendar-mappable
    :class:`~trading_system.core.instruments.InstrumentClass`, that class's
    calendar is unambiguous and is returned. Otherwise — no streams, more
    than one asset class among them, or one this module has no calendar
    mapping for — ``None``, so the day-of-close fold-back is simply not
    applied rather than applied on a guess.

    Args:
        streams: Bars per instrument and timeframe, as a run is built from.
        instruments: Contract specifications the streams' symbols resolve
            against.

    Returns:
        The run's calendar, or ``None`` when none is unambiguous.
    """
    classes = {instruments[key.symbol].asset_class for key in streams}
    if len(classes) != 1:
        return None
    (only,) = classes
    mapped = _CALENDAR_ASSET_CLASS.get(only)
    return TradingCalendar(mapped) if mapped is not None else None


@dataclass
class PendingEntry:
    """An approved entry order awaiting its fill.

    Attributes:
        order_id: Unique within the run, and derived from the signal's own
            identity rather than from a counter — it seeds the fill's random
            stream, so anything else in the run being able to shift it would
            make this order's fills depend on that. See :meth:`Orchestrator._order_id`.
        key: Stream the order rests on.
        binding: Strategy that placed it.
        signal: The signal it came from.
        decision: The size and stop the Risk Engine settled on, at the signal's
            own instant. Frozen there per P10: the FX rate, the stop and the size
            are not revisited at fill time.
        order: How it reaches the market.
        decided_index: Bar of ``key`` the decision was taken on.
        last_fillable_index: Last bar it may fill on. Derived from
            ``EntryOrderSpec.expire_after_bars`` where the spec names one, and
            from the config's fallback where it does not.
    """

    order_id: str
    key: StreamKey
    binding: StrategyBinding
    signal: EntrySignal
    decision: RiskDecision
    order: EntryOrder
    decided_index: int
    last_fillable_index: int


class SignalDrop(StrEnum):
    """Where this layer discarded a signal the Entry and Risk Engines both allowed.

    Counted, never merely skipped. Both of these look identical from outside — a
    run that produced few trades — and identical to a strategy that is simply
    selective, which is the confusion P06 introduced its own drop counters to
    end. Every reason is present with a count of zero, so "the position limit
    never bound" is a recorded fact rather than an absent key.
    """

    #: The strategy already held (or had queued) as many positions as
    #: ``max_concurrent_positions`` allows. Not a Risk Engine refusal: the engine
    #: is never asked, because the limit belongs to the strategy spec.
    POSITION_LIMIT = "position_limit"

    #: The entry filled past the stop the size had been computed against, so the
    #: position was not opened. Entering with the stop already behind price would
    #: leave the risk denominator meaningless and every R figure derived from it
    #: a fiction.
    FILLED_THROUGH_STOP = "filled_through_stop"

    #: A configured :class:`~trading_system.prop.guard.PropGuard` refused the
    #: order outright. Counted here rather than as a Risk Engine rejection: the
    #: firm's rulebook and the engine's own sizing refusals are different
    #: authorities, and folding one into the other would make "the sizing said
    #: no" indistinguishable from "the firm said no" in aggregate.
    PROP_GUARD = "prop_guard"

    #: A configured guard permitted the order at a smaller size than the Risk
    #: Engine sized. Not a drop — the trade happened — but counted alongside
    #: them because a run whose sizes were quietly cut is not the run whose
    #: sizes were computed, and the difference is otherwise invisible.
    PROP_REDUCED = "prop_reduced"


def empty_signal_drops() -> dict[SignalDrop, int]:
    """A zeroed count for every reason a signal can be dropped here.

    Returns:
        Every :class:`SignalDrop` mapped to zero.
    """
    return dict.fromkeys(SignalDrop, 0)


@dataclass(frozen=True)
class AtrRatioCoverage:
    """How many bars of one stream had no ``atr_ratio`` measurement.

    Distinct from :attr:`BacktestResult.cost_degradations`'s ``ATR_UNAVAILABLE``
    entry, which is a share of *fills* — the cost model's own denominator,
    correct for what it answers ("how much of what was actually priced used a
    fallback"). It is the wrong denominator for "how much of the data itself
    had no volatility measurement": a low-frequency strategy can take a
    handful of fills across a run of thousands of bars, and the fills-based
    figure then says nothing about the other bars a stop, a filter or any
    future consumer of ``atr_ratio`` would have read. Bars is the only
    denominator that answers that question, and this exists because ``atr_ratio``
    warms up over ``atr_period + atr_baseline_bars - 1`` bars — visibly longer
    than the raw ATR's own ``atr_period - 1`` — so the two counts read very
    differently on the same run.

    Attributes:
        bars: Total bars in the stream.
        unavailable: Bars where ``atr_ratio`` was ``None`` — still in warmup.
    """

    bars: int
    unavailable: int

    @property
    def fraction(self) -> float:
        """Share of bars with no ``atr_ratio``, in ``[0, 1]``.

        Returns:
            The share, or zero for an empty stream — no bars means nothing was
            unmeasured, not that everything was.
        """
        if self.bars == 0:
            return 0.0
        return self.unavailable / self.bars


@dataclass(frozen=True)
class BacktestResult:
    """Everything a run produced.

    Attributes:
        curve: The equity curve, one row per instant of the merged stream.
        trades: Completed trades, decomposed into gross, commission and swap.
        instants: How many instants were walked.
        rejections: Risk Engine refusals by reason, every reason present.
        degradations: Risk Engine measurements that fell back to a prior.
        exit_drops: Exit instructions that could not be carried out as stated.
        entry_drops: Entry signals discarded as defective, by reason.
        signal_drops: Signals this layer discarded, by reason. Distinct from
            ``rejections``: these never reached the Risk Engine.
        expired_orders: Entry orders that rested to their expiry unfilled.
        cost_degradations: Fraction of fills priced without each measurement,
            from :class:`~trading_system.execution.costs.CostStats`. Filled
            denominator, not a bar denominator — see :class:`AtrRatioCoverage`.
        atr_ratio_coverage: Per stream, how many bars had no ``atr_ratio`` —
            the bar-denominated counterpart to ``cost_degradations``'s
            fill-denominated ``ATR_UNAVAILABLE``. See :class:`AtrRatioCoverage`.
        fills: How many fills the run executed.
        fx_fallback_marks: Marks priced at a stale rate.
        open_at_end: Positions still open when the data ran out. Not trades, and
            not counted as such — the run ended, the position did not.
    """

    curve: Sequence[EquityPoint]
    trades: Sequence[TradeRecord]
    instants: int
    rejections: Mapping[str, int]
    degradations: Mapping[str, int]
    exit_drops: Mapping[ExitDropReason, int]
    entry_drops: Mapping[str, int]
    signal_drops: Mapping[SignalDrop, int]
    expired_orders: int
    cost_degradations: Mapping[str, float]
    atr_ratio_coverage: Mapping[StreamKey, AtrRatioCoverage]
    fills: int
    fx_fallback_marks: int
    open_at_end: int


@dataclass
class _BarPricer:
    """Prices the levels touched on one bar, through the P12 execution stack.

    Built fresh per bar and passed into :meth:`~trading_system.exit.plan.ExitPlan.on_bar`
    rather than stored on the plan: the price depends on the bar and the instant,
    and a plan outlives both.

    Fills are recorded against the *identity* of the decision they priced, which
    is sound because the decision objects the plan hands back in
    :attr:`~trading_system.exit.plan.BarOutcome.applied` are the same instances
    that were priced. Holding the decision alongside the fill keeps that identity
    valid for the duration.
    """

    orchestrator: "Orchestrator"
    held: OpenPosition
    bar: Bar
    market_state: MarketState
    instrument: InstrumentSpec
    fills: dict[int, tuple[ExitDecision, Fill]] = field(default_factory=dict)
    sequence: int = 0

    def price(
        self, decision: ExitDecision, *, position: ManagedPosition, ctx: ExitContext
    ) -> Price | None:
        """Resolve one touched level to an executable price, or decline it.

        ``order_id`` is named on ``ctx.bar_close_ts``, not ``ctx.index`` —
        the same fold-invariance reason :meth:`Orchestrator._order_id` gives:
        a bar's position within a sliced stream is not portable across two
        differently-sliced runs of the same history, and the calendar
        instant already uniquely names the bar.
        """
        self.sequence += 1
        order_id = f"{self.held.position_id}:x{ctx.bar_close_ts.isoformat()}:{self.sequence}"
        resting = self.orchestrator.resting_fill.fill(
            decision_level(decision),
            bar=self.bar,
            side=position.exit_side,
            order_type=decision.order_type,
            instrument=self.instrument,
            order_id=order_id,
        )
        if resting is None:
            return None

        size = _closing_size(decision, position)
        if self.instrument.round_volume(size) <= 0:
            self.orchestrator.count_exit_drop(ExitDropReason.PARTIAL_QUANTIZED_TO_ZERO)
            return None

        order = order_from_fill(
            resting,
            order_id=order_id,
            symbol=position.symbol,
            side=position.exit_side,
            order_type=decision.order_type,
            size=size,
        )
        fill = self.orchestrator.execute(order, self.market_state)
        self.fills[id(decision)] = (decision, fill)
        return fill.price


class Orchestrator:
    """Wires the engines together and implements every phase of the loop."""

    def __init__(
        self,
        *,
        config: BacktestConfig,
        streams: Mapping[StreamKey, OHLCVFrame],
        bindings: Sequence[StrategyBinding],
        instruments: InstrumentRegistry,
        risk_engine: RiskEngine,
        cost_model: CostModel,
        converter: FxConverter,
        market_fill: MarketFillModel | None = None,
        run_seed: int = 0,
        prop_guard: PropGuard | None = None,
    ) -> None:
        """Assemble a run.

        Args:
            config: The run's own parameters.
            streams: Bars per instrument and timeframe.
            bindings: Strategies, their exit presets and the streams they trade.
            instruments: Contract specifications.
            risk_engine: Sizing, portfolio limits and circuit breakers.
            cost_model: Spread, slippage, commission.
            converter: FX rates, for the multi-currency account.
            market_fill: When a close-decided order executes. ``NextBarOpen`` by
                default, which is the only timing bar data honestly supports.
            run_seed: Seed for the per-fill random streams.
            prop_guard: The firm's account rules, applied after sizing and
                before an order is queued. ``None`` is a run on nobody's prop
                account, which is every run in this repository until one is
                asked for — not a disabled guard but an absent firm.

        Raises:
            ValueError: If a binding names a stream that was not supplied, or
                binds a strategy to a stream whose timeframe does not match its
                own ``timeframes.signal_tf``.
        """
        self._config = config
        self._instruments = instruments
        self._risk = risk_engine
        self._costs = cost_model
        self._converter = converter
        self._market_fill = market_fill if market_fill is not None else NextBarOpen()
        self.resting_fill = RestingOrderModel(
            limit=LimitTouch(cost_model.config.limit_fill, cost_model.config.gap, run_seed),
            stop=GapFill(),
        )

        self._bindings = tuple(bindings)
        for binding in self._bindings:
            missing = [key for key in binding.keys if key not in streams]
            if missing:
                raise ValueError(
                    f"strategy {binding.spec.id} names streams not supplied: "
                    f"{', '.join(str(key) for key in missing)}"
                )
            signal_tf = binding.spec.timeframes.signal_tf
            mistimed = [key for key in binding.keys if key.timeframe is not signal_tf]
            if mistimed:
                raise ValueError(
                    f"strategy {binding.spec.id} has timeframes.signal_tf={signal_tf.value}, "
                    f"but is bound to stream(s) at a different timeframe: "
                    f"{', '.join(str(key) for key in mistimed)}. Every window and bar-count "
                    "field on the spec (confirmation_window_bars, expire_after_bars, "
                    "cooldown_bars_after_loss) means bars of signal_tf; trading it on another "
                    "timeframe silently changes what all of them mean. Bind it to a stream at "
                    f"{signal_tf.value}, or change the spec's signal_tf."
                )

        # One FeatureRegistry per STREAM, not per strategy — built from the union
        # of every binding that trades it. FeatureRegistry already dedups by the
        # resolved indicator's own name (entry/features.py), so two strategies
        # both asking for ema(period=50) collapse onto one FeatureSpec and the
        # pipeline built from it computes that column once. Passing the merged,
        # superset registry into `compile_entry` for each individual strategy is
        # safe: a registry only has to CONTAIN a strategy's own references, and
        # extra ones it never resolves are never read.
        #
        # The registry alone is not the whole story: a position's ExitPlan reads
        # its own columns — an ATR-based stop, a swing-based trail — by name,
        # through `exit/rules/_features.py`, never through a FeatureRef the
        # Entry-side registry could see. The FeaturePipeline actually computed
        # for a stream is therefore the union of the entry registry's specs AND
        # every exit preset's own feature needs, not the registry alone.
        self._series: dict[StreamKey, BarSeries] = {}
        self._atr: dict[StreamKey, list[float | None]] = {}
        self._atr_ratio: dict[StreamKey, list[float | None]] = {}
        self._atr_ratio_coverage: dict[StreamKey, AtrRatioCoverage] = {}
        self._registries: dict[StreamKey, FeatureRegistry] = {}
        for key, frame in streams.items():
            on_stream = [binding for binding in self._bindings if key in binding.keys]
            specs_on_stream = [binding.spec for binding in on_stream]
            registry = _merged_registry(specs_on_stream)
            self._registries[key] = registry
            feature_specs = _merged_feature_specs(
                registry.specs, [binding.exit_preset for binding in on_stream]
            )
            categories = _merged_label_categories(specs_on_stream)
            series, atr, ratio = self._prepare(key, frame, feature_specs, categories)
            self._series[key] = series
            self._atr[key] = atr
            self._atr_ratio[key] = ratio
            self._atr_ratio_coverage[key] = AtrRatioCoverage(
                bars=len(ratio), unavailable=sum(1 for value in ratio if value is None)
            )
        self._report_atr_warmup()

        # One compiled engine per (strategy, stream): an EntryEvaluator tracks a
        # single pending setup, so it is bound to one bar stream.
        self._entries: dict[tuple[str, StreamKey], EntryEngine] = {
            (binding.spec.id, key): compile_entry(binding.spec, self._registries[key])
            for binding in self._bindings
            for key in binding.keys
        }
        self._smallest_fraction: dict[str, Decimal] = {
            binding.spec.id: build_plan(
                binding.exit_preset, intrabar_policy=config.intrabar_policy
            ).smallest_closing_fraction()
            for binding in self._bindings
        }

        self._data = DataHandler(self._series, config.day_origin)
        self._store = BarStore(self._series)
        self._portfolio = Portfolio(
            currency=config.account_currency,
            starting_balance=config.starting_balance,
            instruments=instruments,
            converter=converter,
            day_origin=config.day_origin,
            calendar=derive_run_calendar(streams, instruments),
        )

        self._pending_orders: list[PendingEntry] = []
        self._recognised: dict[StreamKey, list[tuple[StrategyBinding, EntrySignal]]] = {}
        self._account: AccountState | None = None
        self._exit_drops = empty_drop_counts()
        self._signal_drops = empty_signal_drops()
        self._expired_orders = 0
        self._issued_ids: set[str] = set()
        self._instant_index = -1

        self._prop = prop_guard
        # Day-start figures are recomputed from the curve, never accumulated:
        # the same reasoning circuit_breakers.py gives for holding no "blocked"
        # flag. Cached by the firm's own day label rather than kept as a single
        # mutable "today" slot — a single refreshed slot is exactly the leak
        # CorrelationProvider avoids, where a later instant's value is read by
        # an earlier decision.
        self._prop_day_start: dict[date, tuple[Decimal, Decimal]] = {}
        self._prop_high_water = config.starting_balance
        if prop_guard is not None:
            divergence = day_origin_divergence(prop_guard.rules, config.day_origin)
            if divergence is not None:
                logger.warning(
                    "prop.day_origin_divergence",
                    firm=prop_guard.rules.name,
                    firm_tz=prop_guard.rules.daily_reset_tz,
                    firm_at=str(prop_guard.rules.daily_reset_time),
                    run_tz=config.day_origin.tz,
                    run_at=str(config.day_origin.at),
                    detail=divergence,
                )

    def __repr__(self) -> str:
        """Compact description naming the streams and strategies."""
        return (
            f"Orchestrator(streams={len(self._series)}, "
            f"strategies={[binding.spec.id for binding in self._bindings]})"
        )

    @property
    def portfolio(self) -> Portfolio:
        """The account this run trades."""
        return self._portfolio

    @property
    def store(self) -> BarStore:
        """The published-bar view every phase reads through."""
        return self._store

    def count_exit_drop(self, reason: ExitDropReason) -> None:
        """Record an exit instruction this layer could not carry out.

        P07 files ``PARTIAL_QUANTIZED_TO_ZERO`` under the executor that owns the
        instrument metadata, which is this one; the counts are reported together
        with the plans' own.

        Args:
            reason: What could not be done.
        """
        self._exit_drops[reason] += 1

    def execute(self, order: ExecutionOrder, market_state: MarketState) -> Fill:
        """Cost one order and report its slippage to the circuit breakers.

        The breaker is fed the deviation alone, without the half spread: a spread
        is deterministic and expected, and a breaker that fired on a session
        multiplier would be tripping on noise.

        Args:
            order: An order a fill model has resolved to a mid price.
            market_state: Conditions at the fill instant.

        Returns:
            The fill.
        """
        fill = self._costs.apply(order, market_state)
        self._risk.breakers.record_fill(
            SlippageReport(at=fill.ts, slippage_points=fill.slippage_points, symbol=fill.symbol)
        )
        return fill

    def run(self) -> BacktestResult:
        """Walk the whole run and report what it produced.

        Returns:
            The equity curve, the trades and every counter the run kept.
        """
        engine = BacktestEngine(self._data, self._store, self)
        instants = engine.run()
        self._report_cost_coverage()
        stats = self._costs.stats
        return BacktestResult(
            curve=self._portfolio.curve,
            trades=self._portfolio.trades,
            instants=instants,
            rejections={reason.value: count for reason, count in self._risk.rejections.items()},
            degradations={reason.value: count for reason, count in self._risk.degradations.items()},
            exit_drops=self._exit_drops_total(),
            entry_drops=self._entry_drops(),
            signal_drops=dict(self._signal_drops),
            expired_orders=self._expired_orders,
            cost_degradations={reason.value: value for reason, value in stats.fractions.items()},
            atr_ratio_coverage=dict(self._atr_ratio_coverage),
            fills=stats.fills,
            fx_fallback_marks=self._portfolio.fx_fallback_marks,
            open_at_end=len(self._portfolio.open_positions),
        )

    # ------------------------------------------------------------------
    # Phases, in the order the engine calls them.
    # ------------------------------------------------------------------

    def on_financing(self, ts: datetime) -> None:
        """Charge financing over ``(last accrual, ts]`` for every open position.

        Charged before anything can close at this instant, so a position going
        flat here still pays the carry it actually incurred. Accrual is
        incremental and chained — each interval starts where the previous ended —
        which sums to exactly the same charge as one call over the whole holding
        period, because
        :func:`~trading_system.execution.costs.accrue_swap` counts trading-day
        labels entered strictly after the opening one.
        """
        self._instant_index += 1
        self._risk.breakers.note_bar(self._instant_index)
        for held in self._portfolio.open_positions:
            instrument = self._instruments.get(held.symbol)
            if instrument is None:
                continue
            model = self._costs.config.swap.get(instrument.asset_class.value)
            if model is None or ts <= held.last_financing_ts:
                continue
            mark = self._store.last_close(held.key)
            accrual = accrue_swap(
                instrument,
                side=held.side,
                size=held.position.remaining_size,
                held_from=held.last_financing_ts,
                held_to=ts,
                model=model,
                account_currency=self._config.account_currency,
                mark_price=None if mark is None else Price(mark),
            )
            amount = accrual.amount
            if amount and accrual.currency != self._config.account_currency:
                amount *= self._converter.rate(
                    base=accrual.currency, quote=self._config.account_currency, at=ts
                )
            self._portfolio.charge_financing(held, amount, ts)

    def on_deferred_exits(self, event: BarEvent) -> None:
        """Fill exits decided on the previous bar's close, at this bar's open.

        A ``BAR_CLOSE`` decision names no price, because the Exit Engine cannot
        see the next open and must not appear to. It is priced here, by the
        market fill model, at the open of the bar that has just arrived — which
        is also the earliest instant this layer *could* price it, since that bar
        did not exist before now.
        """
        index = event.index
        if index == 0:
            return
        for held in self._portfolio.positions_on(event.key):
            deferred = held.deferred_exit
            if deferred is None:
                continue
            if deferred.decided_bar_index != index - 1:
                raise ValueError(
                    f"{held.position_id}: exit deferred on bar {deferred.decided_bar_index} "
                    f"was not filled on bar {deferred.decided_bar_index + 1}; it is now "
                    f"bar {index}. A deferred decision must be consumed at its own next bar."
                )
            held.deferred_exit = None
            self._fill_deferred(held, deferred, event)

    def on_pending_entries(self, event: BarEvent) -> None:
        """Fill, or expire, entry orders resting on this stream."""
        surviving: list[PendingEntry] = []
        for pending in self._pending_orders:
            if pending.key != event.key:
                surviving.append(pending)
                continue
            if event.index <= pending.decided_index:
                surviving.append(pending)
                continue
            if self._try_fill_entry(pending, event):
                continue
            if event.index >= pending.last_fillable_index:
                self._expired_orders += 1
                logger.info(
                    "backtest.entry_order_expired",
                    strategy=pending.binding.spec.id,
                    symbol=pending.key.symbol,
                    order_id=pending.order_id,
                    rested_bars=event.index - pending.decided_index,
                )
                continue
            surviving.append(pending)
        self._pending_orders = surviving

    def on_recognise(self, event: BarEvent) -> None:
        """Recognise entry signals on this bar's close. Nothing is sized or spent.

        Outside ``[config.evaluation_start, config.evaluation_end)`` no entry
        engine is even asked: this is a walk-forward fold's warmup prefix or
        drain tail, and a signal that is never evaluated leaves no trace in
        any drop counter — the distinction P15 draws against dropping a
        recognised signal after the fact.
        """
        start = self._config.evaluation_start
        end = self._config.evaluation_end
        if (start is not None and event.close_ts < start) or (
            end is not None and event.close_ts >= end
        ):
            self._recognised[event.key] = []
            return
        ctx = self._store.context(event.key, event.index)
        found: list[tuple[StrategyBinding, EntrySignal]] = []
        for binding in self._bindings:
            if event.key not in binding.keys:
                continue
            engine = self._entries[(binding.spec.id, event.key)]
            for signal in engine.evaluate(ctx):
                found.append((binding, signal))
        self._recognised[event.key] = found

    def on_exits(self, event: BarEvent) -> None:
        """Run exit rules then stop modifiers over this bar, for every position on it."""
        held_positions = self._portfolio.positions_on(event.key)
        if not held_positions:
            return
        bar_ctx = self._store.context(event.key, event.index)
        instrument = self._instruments.get(event.key.symbol)
        if instrument is None:
            return
        bar = self._store.bar(event.key, event.index)
        state = self._market_state(event.key, event.index, event.close_ts)

        for held in held_positions:
            reverse = self._reverse_side(event.key, held.side)
            ctx = ExitContext(bar_ctx, reverse_signal_side=reverse)
            pricer = _BarPricer(
                orchestrator=self,
                held=held,
                bar=bar,
                market_state=state,
                instrument=instrument,
            )
            outcome = held.plan.on_bar(held.position, ctx, pricer=pricer)
            for applied in outcome.applied:
                record = pricer.fills.get(id(applied.decision))
                if record is None:  # pragma: no cover - defensive
                    raise ValueError(
                        f"{held.position_id}: a leg was booked without a fill behind it"
                    )
                self._portfolio.book_leg(held, applied.leg, record[1])
            if outcome.deferred is not None:
                held.deferred_exit = outcome.deferred
            self._settle(held, event.close_ts)

    def on_mark(self, ts: datetime) -> None:
        """Mark open positions to their own streams' latest closes, and restate risk."""
        marks: dict[str, float] = {}
        for held in self._portfolio.open_positions:
            close = self._store.last_close(held.key)
            if close is not None:
                marks[held.symbol] = close
            self._portfolio.restate_risk(held)
        self._portfolio.mark(marks, at=ts)

    def on_snapshot(self, ts: datetime) -> None:
        """Write the instant's single equity row and freeze the account for sizing."""
        self._portfolio.snapshot(ts)
        self._account = self._portfolio.account_state(ts)

    def on_sizing(self, event: BarEvent) -> None:
        """Size the signals recognised on this bar and queue what is approved."""
        signals = self._recognised.pop(event.key, [])
        if not signals:
            return
        account = self._account
        if account is None:  # pragma: no cover - the engine always snapshots first
            raise ValueError("sizing ran before the instant was snapshotted")

        atr_price = self._atr[event.key][event.index]
        trades = self._portfolio.closed_trades
        for binding, signal in signals:
            if self._at_position_limit(binding):
                self._signal_drops[SignalDrop.POSITION_LIMIT] += 1
                continue
            decision = self._risk.evaluate(
                signal,
                account=account,
                stop_reference=binding.spec.risk_profile.stop_reference,
                smallest_exit_fraction=self._smallest_fraction[binding.spec.id],
                bar_index=self._instant_index,
                trades=trades,
                atr_price=atr_price,
            )
            if not decision.approved:
                continue
            permitted = self._prop_check(decision, signal, event)
            if permitted is None:
                continue
            decision = permitted
            # The engine keeps no memory of what it approved, so the approval is
            # fed back here. Two signals at one instant would otherwise both be
            # measured against headroom the first has already taken.
            account = account.with_opened(
                signal.symbol,
                binding.spec.id,
                signal.side,
                decision.risk_amount,
                margin=decision.margin_amount,
                notional=decision.notional_amount,
            )
            self._account = account
            self._queue_entry(binding, signal, decision, event)

    # ------------------------------------------------------------------
    # Internals.
    # ------------------------------------------------------------------

    def _prop_check(
        self, decision: RiskDecision, signal: EntrySignal, event: BarEvent
    ) -> RiskDecision | None:
        """Put a sized order past the firm's rulebook, the last veto before queuing.

        Runs after the Risk Engine and before the order is queued — the
        position CLAUDE.md's architecture line already draws. After sizing
        because a ``REDUCE`` verdict has to name a size, and no size exists
        before then.

        A refusal returns ``None`` and is counted in
        :class:`SignalDrop`, this layer's own tally, rather than being dressed
        up as a :class:`~trading_system.risk.models.RiskDecision` carrying some
        borrowed :class:`~trading_system.risk.models.RiskReason`. The firm's
        rulebook and the Risk Engine's own refusals are different authorities
        and must stay countable apart — which is the same reason
        :class:`~trading_system.prop.guard.PropReason` is a separate enum.

        Args:
            decision: The Risk Engine's approval.
            signal: The signal it approved.
            event: The bar being sized on.

        Returns:
            The decision unchanged when the firm permits it, a proportionally
            reduced one when the firm permits less, or ``None`` when it permits
            nothing. A reduction rescales ``risk_amount`` with the size rather
            than keeping the original figure: what the decision reports has to
            stay what the account actually stands to lose, the same invariant
            the Risk Engine maintains when it recomputes after lot
            quantisation.
        """
        guard = self._prop
        if guard is None:
            return decision
        instrument = self._instruments.get(signal.symbol)
        if instrument is None:  # pragma: no cover - the Risk Engine refuses these
            return decision

        verdict = guard.check(
            ProposedOrder(
                symbol=signal.symbol,
                size=decision.size,
                risk_amount=decision.risk_amount,
                instrument=instrument,
            ),
            self._prop_account(event.close_ts),
        )
        if verdict.decision is GuardDecision.ALLOW:
            return decision
        if verdict.decision is GuardDecision.REJECT:
            self._signal_drops[SignalDrop.PROP_GUARD] += 1
            logger.debug(
                "prop.rejected",
                strategy=signal.strategy_id,
                symbol=signal.symbol,
                reason=verdict.reason.value,
                detail=verdict.detail,
            )
            return None

        self._signal_drops[SignalDrop.PROP_REDUCED] += 1
        scale = verdict.allowed_size / decision.size
        return replace(
            decision,
            size=verdict.allowed_size,
            risk_amount=decision.risk_amount * scale,
            risk_pct=decision.risk_pct * float(scale),
            reasons=(*decision.reasons, f"{verdict.reason.value}: {verdict.detail}"),
        )

    def _prop_account(self, at: datetime) -> PropAccountState:
        """The account as the firm's rules measure it, at this instant.

        The day-start figures are recomputed from the equity curve and cached
        by the **firm's** day label. Cached because a linear scan of the curve
        on every bar would make a 45 000-bar run quadratic; keyed by the label
        rather than held in one "today" slot because a single refreshed slot is
        how a later instant's value ends up read by an earlier decision — the
        leak :class:`~trading_system.risk.correlation.CorrelationProvider`
        keys its own cache by trading day to avoid.
        """
        guard = self._prop
        assert guard is not None  # only called with a guard configured
        day = guard.trading_day_of(at)
        portfolio = self._portfolio
        cached = self._prop_day_start.get(day)
        if cached is None:
            opening = self._day_opening(day, guard)
            self._prop_day_start[day] = opening
            cached = opening
        equity = portfolio.equity
        self._prop_high_water = max(self._prop_high_water, equity)
        return PropAccountState(
            at=at,
            equity=equity,
            balance=portfolio.balance,
            day_start_balance=cached[0],
            day_start_equity=cached[1],
            high_water_mark=self._prop_high_water,
            day=day,
        )

    def _day_opening(self, day: date, guard: PropGuard) -> tuple[Decimal, Decimal]:
        """``(balance, equity)`` as they stood when the firm's day opened.

        Read off the last curve row belonging to an **earlier** firm-day: that
        row is the account as the previous day left it, which is what the new
        day's allowance is measured from. With no earlier row — the first day
        of the run — the run's own starting balance stands in, since that is
        literally what the account opened at.
        """
        previous: EquityPoint | None = None
        for point in reversed(self._portfolio.curve):
            if guard.trading_day_of(point.ts) < day:
                previous = point
                break
        if previous is None:
            start = self._config.starting_balance
            return start, start
        return previous.balance, previous.equity

    def _prepare(
        self,
        key: StreamKey,
        frame: OHLCVFrame,
        feature_specs: Sequence[FeatureSpec],
        categories: Sequence[LabelCategory],
    ) -> tuple[BarSeries, list[float | None], list[float | None]]:
        """Materialise one stream's bars, its features and its volatility columns.

        Features are computed vectorised, once per stream, from every column any
        strategy's entry or any of its exit's rules needs on that stream — not
        once per strategy and not once per position. Computed before the loop
        and never inside it, alongside the ATR columns, which is what keeps a
        five-year three-instrument run inside its time budget. This is not
        lookahead: P03 indicators are causal by construction, and
        ``BaseIndicator.verify_parity`` proves the vectorised and incremental
        forms agree to 1e-9 over the whole registry.
        """
        if frame.symbol != key.symbol or frame.timeframe is not key.timeframe:
            raise ValueError(f"stream keyed {key} holds {frame.symbol}@{frame.timeframe.value}")
        features = FeaturePipeline(feature_specs).compute(frame) if feature_specs else None
        series = BarSeries.from_frame(frame, features=features, categories=categories)
        atr_column = ATR(period=self._config.atr_period).compute(frame)
        baseline = atr_column.rolling_mean(
            window_size=self._config.atr_baseline_bars,
            min_samples=self._config.atr_baseline_bars,
        )
        ratio = pl.when(baseline > 0).then(atr_column / baseline).otherwise(None)
        ratio_column = pl.select(ratio).to_series()
        return series, atr_column.to_list(), ratio_column.to_list()

    def _market_state(self, key: StreamKey, index: int, ts: datetime) -> MarketState:
        """Conditions at one instant, from the last bar that had closed by then.

        Args:
            key: Stream being priced.
            index: Index of the newest bar closed at or before ``ts``.
            ts: The fill instant.

        Returns:
            The state. ``quoted_spread_points`` is ``None`` — a run without quote
            data — and ``news_severity`` is ``NONE`` until P16 supplies a
            calendar. ``session`` is not passed at all: it is derived from ``ts``.
        """
        ratio = self._atr_ratio[key][index] if 0 <= index < len(self._atr_ratio[key]) else None
        return MarketState(ts=ts, atr_ratio=ratio)

    def _reverse_side(self, key: StreamKey, held_side: Side) -> Side | None:
        """The side of an opposing signal recognised on this bar, if any.

        A flat fact about the bar, not a reference to the Entry Engine: this is
        how ``SignalReverseExit`` learns about a reversal without importing
        anything from :mod:`trading_system.entry`.
        """
        opposite = Side.SELL if held_side is Side.BUY else Side.BUY
        for _binding, signal in self._recognised.get(key, ()):
            if signal.side is opposite:
                return opposite
        return None

    def _at_position_limit(self, binding: StrategyBinding) -> bool:
        """Whether this strategy already holds all the positions it may.

        ``max_concurrent_positions`` lives on
        :class:`~trading_system.strategies.schema.RiskProfileSpec`, so the limit
        is per strategy and counted across every stream it trades — not per
        instrument. Orders still resting count against it: an approved order is
        a position the strategy has committed to, and letting the queue exceed
        the limit would just move the breach one bar later.
        """
        limit = binding.spec.risk_profile.max_concurrent_positions
        held = sum(
            1
            for position in self._portfolio.open_positions
            if position.strategy_id == binding.spec.id
        )
        queued = sum(
            1 for pending in self._pending_orders if pending.binding.spec.id == binding.spec.id
        )
        return held + queued >= limit

    def _queue_entry(
        self,
        binding: StrategyBinding,
        signal: EntrySignal,
        decision: RiskDecision,
        event: BarEvent,
    ) -> None:
        """Place an approved order, with the lifetime its own spec names."""
        entry_spec = next(
            (spec for spec in binding.spec.entries if _SIDES[spec.direction] is signal.side),
            None,
        )
        if entry_spec is None:  # pragma: no cover - compile_entry guarantees a leg per side
            raise ValueError(f"{binding.spec.id}: no entry leg for {signal.side.value}")
        # The spec's own field is the authority; the config is the fallback for a
        # spec that names no lifetime, which is what its name says.
        ttl = entry_spec.entry_order.expire_after_bars
        if ttl is None:
            ttl = self._config.default_entry_order_ttl_bars
        self._pending_orders.append(
            PendingEntry(
                order_id=self._order_id(binding, event, signal.side),
                key=event.key,
                binding=binding,
                signal=signal,
                decision=decision,
                order=entry_spec.entry_order.order,
                decided_index=event.index,
                last_fillable_index=event.index + ttl,
            )
        )

    def _order_id(self, binding: StrategyBinding, event: BarEvent, side: Side) -> str:
        """The identifier of the one order this signal can be, derived from itself.

        Composed of what already identifies the signal — the strategy, the
        stream, the bar and the side — and of **nothing else**. Never a running
        counter, and the difference is not cosmetic:
        :func:`~trading_system.execution.rng.fill_seed` keys a fill's random
        stream on ``(run_seed, symbol, ts, order_id)`` precisely so that a fill
        draws the same number no matter what else the run contains. A shared
        counter puts the position in the queue back into the id, so adding a
        second instrument renumbers the first's orders and silently redraws
        every slippage jitter on it — the exact artefact
        :mod:`trading_system.execution.rng` exists to remove, reintroduced one
        layer up. The no-lookahead harness caught this; the id is now
        independent by construction.

        Named on ``event.close_ts``, not ``event.index``. P15's walk-forward
        surfaced the reason the harness could not: a bar's *position* in a
        stream is a property of where that run's own slice happens to start,
        so the same calendar bar carries a different ``index`` in an
        in-sample run than in the out-of-sample run built from a
        differently-bounded slice of the same history. An id built from
        ``index`` would draw a different slippage jitter for what is, in
        every way that matters to the strategy, the identical bar — the
        exact "measuring the generator instead of the strategy" artefact
        CLAUDE.md's P15 stage 1 decisions rule out. ``close_ts`` already
        uniquely names the bar within a stream (:class:`DataHandler`
        requires strictly increasing bar opens), so nothing about
        uniqueness is lost by naming it instead of counting to it.

        Unique by the same argument the compiler makes: an
        :class:`~trading_system.entry.compiler.EntryEvaluator` tracks one pending
        setup per leg, so one strategy on one stream can produce at most one
        signal per side per bar. The set is kept anyway, because "the id scheme
        turned out not to be unique" must be a raised error rather than two
        positions quietly sharing an identity.

        Args:
            binding: Strategy the order comes from.
            event: Bar the decision was taken on.
            side: Direction the order takes.

        Returns:
            The identifier, carried by the order, its fills and the position.

        Raises:
            ValueError: If the same identifier has already been issued.
        """
        order_id = f"{binding.spec.id}@{event.key}@{event.close_ts.isoformat()}{side.value[0]}"
        if order_id in self._issued_ids:
            raise ValueError(
                f"order id {order_id} was issued twice; one strategy on one stream may "
                "produce at most one signal per side per bar"
            )
        self._issued_ids.add(order_id)
        return order_id

    def _try_fill_entry(self, pending: PendingEntry, event: BarEvent) -> bool:
        """Attempt one entry order against the bar that has just closed.

        Returns:
            ``True`` if it filled, ``False`` if it is still resting.
        """
        instrument = self._instruments.get(pending.key.symbol)
        if instrument is None:  # pragma: no cover - the Risk Engine refuses these
            return False
        bar = self._store.bar(event.key, event.index)
        side = pending.signal.side

        if isinstance(pending.order, MarketOrder):
            decision_bar = self._store.bar(event.key, pending.decided_index)
            mid = self._market_fill.reference_price(decision_bar, bar)
            if mid is None:
                return False
            # The fill happens at the bar's open, so that is the instant it is
            # priced at, and the volatility it is priced with is the last bar
            # that had closed by then — the decision bar.
            state = self._market_state(event.key, event.index - 1, bar.timestamp)
            order = ExecutionOrder(
                order_id=pending.order_id,
                symbol=pending.key.symbol,
                side=side,
                order_type=OrderType.MARKET,
                size=pending.decision.size,
                mid_price=mid,
            )
            fill = self.execute(order, state)
            self._open_position(pending, fill, event)
            return True

        order_type = OrderType.LIMIT if isinstance(pending.order, LimitOrder) else OrderType.STOP
        resting = self.resting_fill.fill(
            pending.signal.reference_price,
            bar=bar,
            side=side,
            order_type=order_type,
            instrument=instrument,
            order_id=pending.order_id,
        )
        if resting is None:
            return False
        state = self._market_state(event.key, event.index, event.close_ts)
        order = order_from_fill(
            resting,
            order_id=pending.order_id,
            symbol=pending.key.symbol,
            side=side,
            order_type=order_type,
            size=pending.decision.size,
        )
        fill = self.execute(order, state)
        self._open_position(pending, fill, event)
        return True

    def _open_position(self, pending: PendingEntry, fill: Fill, event: BarEvent) -> None:
        """Turn a filled entry into a managed position with its own exit plan."""
        stop = pending.decision.stop_price
        wrong_side = stop >= fill.price if pending.signal.side is Side.BUY else stop <= fill.price
        if wrong_side:
            # The fill slipped past the stop the size was computed against. The
            # trade is not opened: entering with the stop already behind price
            # would make the position's risk denominator meaningless, and every
            # R figure derived from it a fiction.
            self._signal_drops[SignalDrop.FILLED_THROUGH_STOP] += 1
            logger.warning(
                "backtest.entry_filled_through_stop",
                strategy=pending.binding.spec.id,
                symbol=pending.key.symbol,
                fill=fill.price,
                stop=stop,
            )
            return
        plan = build_plan(pending.binding.exit_preset, intrabar_policy=self._config.intrabar_policy)
        plan.reset()
        position = ManagedPosition(
            symbol=pending.key.symbol,
            side=pending.signal.side,
            entry_price=fill.price,
            size=pending.decision.size,
            initial_stop=stop,
            opened_at=fill.ts,
            strategy_id=pending.binding.spec.id,
        )
        held = OpenPosition(
            position_id=pending.order_id,
            key=pending.key,
            position=position,
            plan=plan,
            strategy_id=pending.binding.spec.id,
            entry_fill=fill,
            entry_bar_index=event.index,
            entry_fx_rate=pending.decision.fx_rate,
            risk_amount=pending.decision.risk_amount,
            entry_quality=pending.signal.quality,
            margin_per_lot=pending.decision.margin_per_lot,
            notional_per_lot=pending.decision.notional_per_lot,
        )
        self._portfolio.open(held)
        self._portfolio.restate_risk(held)

    def _fill_deferred(self, held: OpenPosition, deferred: DeferredExit, event: BarEvent) -> None:
        """Fill one ``BAR_CLOSE`` decision at this bar's open.

        ``order_id`` is named on ``event.close_ts``, not ``event.index`` — see
        :meth:`Orchestrator._order_id` on why a bar's local position in a
        stream is not what identifies it across two differently-sliced runs.
        """
        instrument = self._instruments.get(held.symbol)
        if instrument is None:  # pragma: no cover - a position implies an instrument
            return
        decision_bar = self._store.bar(event.key, deferred.decided_bar_index)
        bar = self._store.bar(event.key, event.index)
        mid = self._market_fill.reference_price(decision_bar, bar)
        if mid is None:  # pragma: no cover - the next bar exists by construction here
            return
        size = _closing_size(deferred.decision, held.position)
        if instrument.round_volume(size) <= 0:
            self.count_exit_drop(ExitDropReason.PARTIAL_QUANTIZED_TO_ZERO)
            return
        order = ExecutionOrder(
            order_id=f"{held.position_id}:d{event.close_ts.isoformat()}",
            symbol=held.symbol,
            side=held.position.exit_side,
            order_type=OrderType.MARKET,
            size=size,
            mid_price=mid,
        )
        state = self._market_state(event.key, deferred.decided_bar_index, bar.timestamp)
        fill = self.execute(order, state)
        if deferred.decision.kind is ExitKind.FULL or (
            deferred.decision.fraction is not None
            and deferred.decision.fraction >= held.position.remaining_fraction
        ):
            leg = held.position.close_all(
                price=fill.price, ts=fill.ts, reason=deferred.decision.reason
            )
        else:
            assert deferred.decision.fraction is not None
            leg = held.position.close(
                deferred.decision.fraction,
                price=fill.price,
                ts=fill.ts,
                reason=deferred.decision.reason,
            )
        self._portfolio.book_leg(held, leg, fill)
        self._settle(held, fill.ts)

    def _settle(self, held: OpenPosition, ts: datetime) -> None:
        """Retire a position that has gone flat, or restate what it still risks."""
        if held.position.is_open:
            self._portfolio.restate_risk(held)
            return
        last_leg = held.position.legs[-1] if held.position.legs else None
        self._portfolio.close_out(held, last_leg.ts if last_leg is not None else ts)

    def _exit_drops_total(self) -> dict[ExitDropReason, int]:
        """This layer's drop counts plus every live plan's."""
        total = dict(self._exit_drops)
        for held in self._portfolio.open_positions:
            for reason, count in held.plan.drops.items():
                total[reason] += count
        return total

    def _entry_drops(self) -> dict[str, int]:
        """Entry signals discarded as defective, summed over every compiled engine."""
        total: dict[str, int] = {}
        for engine in self._entries.values():
            for reason, count in engine.drops.items():
                total[reason.value] = total.get(reason.value, 0) + count
        return total

    def _report_cost_coverage(self) -> None:
        """Say out loud when too many fills were priced without a measurement.

        ``atr_baseline_bars`` costs a warmup, and on a coarse timeframe it can
        cost most of the run. A run that quietly prices half its fills on a prior
        is the defect this exists to surface, so it is reported rather than left
        to be inferred from the config.
        """
        stats = self._costs.stats
        if stats.fills == 0:
            return
        for reason, fraction in stats.fractions.items():
            if fraction > self._config.atr_unavailable_report_threshold:
                logger.warning(
                    "backtest.cost_measurement_missing",
                    reason=reason.value,
                    fraction=round(fraction, 4),
                    fills=stats.fills,
                    atr_baseline_bars=self._config.atr_baseline_bars,
                    detail=(
                        "fills were priced on the model's prior rather than a measurement; "
                        "lower atr_baseline_bars for this timeframe"
                    ),
                )

    def _report_atr_warmup(self) -> None:
        """Say out loud when a stream's own bar count makes ``atr_ratio`` warmup too large.

        Checked at construction, against the stream itself — before a single
        bar has been walked, and independent of whether the run goes on to
        produce any fills at all, which is what :meth:`_report_cost_coverage`'s
        fill-denominated check depends on and can never see for a strategy that
        trades rarely. ``atr_baseline_bars`` is one number a run applies to
        every stream it trades; the warmup it costs is a fixed bar count, so
        the coarser a stream's timeframe, the larger a share of it that count
        eats — the same number that is 1.7% of five years of H1 is 40% of five
        years of D1.

        Deliberately not auto-corrected. Scaling ``atr_baseline_bars`` to the
        data would be exactly the config magic CLAUDE.md rules out: a config
        error must fail loudly at load time, not get silently repaired into a
        different, unstated configuration. This reports the measured fraction
        and leaves the choice — a smaller ``atr_baseline_bars``, a finer
        timeframe, or accepting the run as it is — to whoever reads the log.
        """
        for key, coverage in self._atr_ratio_coverage.items():
            if coverage.fraction > self._config.atr_unavailable_report_threshold:
                logger.warning(
                    "backtest.atr_ratio_warmup_exceeds_threshold",
                    stream=str(key),
                    unavailable_bars=coverage.unavailable,
                    total_bars=coverage.bars,
                    fraction=round(coverage.fraction, 4),
                    threshold=self._config.atr_unavailable_report_threshold,
                    atr_period=self._config.atr_period,
                    atr_baseline_bars=self._config.atr_baseline_bars,
                    detail=(
                        "atr_ratio is None for this share of the stream's own bars; lower "
                        "atr_baseline_bars for this timeframe, or accept the warmup"
                    ),
                )


def _merged_registry(specs: Sequence[StrategySpec]) -> FeatureRegistry:
    """One registry covering every feature any of ``specs`` references.

    The union, not a list of per-strategy registries:
    :class:`~trading_system.entry.features.FeatureRegistry` already collapses a
    reference onto the resolved indicator's own name — the same key ``ema_50``
    for two strategies that both ask for
    ``FeatureRef(indicator="ema", params={"period": 50})``, however differently
    each spelled it. Building one registry from the combined reference stream is
    what makes that collapse happen once, for the whole stream, rather than once
    per strategy with the pipeline computing the same column twice.

    Args:
        specs: Every strategy trading one stream.

    Returns:
        A registry resolving every reference any of them makes.
    """
    refs = [ref for spec in specs for ref in iter_feature_refs(spec)]
    return FeatureRegistry(refs)


def _merged_feature_specs(
    entry_specs: Sequence[FeatureSpec], presets: Sequence[ExitPresetSpec]
) -> tuple[FeatureSpec, ...]:
    """Every column a stream's FeaturePipeline must compute.

    The entry registry's own specs are only half of it. A position's
    :class:`~trading_system.exit.plan.ExitPlan` reads its own columns — an
    ATR-based stop, a swing-based trail — directly by name, through
    :mod:`trading_system.exit.rules._features`, which is never reached by
    :class:`~trading_system.entry.features.FeatureRegistry` at all: Exit
    deliberately keeps its own naming convention rather than importing Entry's,
    per P07's import discipline. This is the counterpart the module's own
    docstring names as future work — "building the right FeaturePipeline for a
    composed ExitPlan is a P07 stage 3 concern, once library.json exists to
    declare it" — now that it does.

    Deduplication is by column name, the same invariant
    :class:`~trading_system.entry.features.FeatureRegistry` relies on: two specs
    that would produce the same name are, by construction, the same indicator
    resolved the same way, so the first one found wins rather than being
    computed twice.

    Args:
        entry_specs: A stream's merged entry registry's specs.
        presets: Every exit preset a position opened on this stream might use.

    Returns:
        The full set of specs to build one
        :class:`~trading_system.features.pipeline.FeaturePipeline` from.
    """
    merged: dict[str, FeatureSpec] = {spec.name: spec for spec in entry_specs}
    for preset in presets:
        for spec in _exit_feature_specs(preset):
            merged.setdefault(spec.name, spec)
    return tuple(merged.values())


def _exit_feature_specs(preset: ExitPresetSpec) -> list[FeatureSpec]:
    """Every P03 column one exit preset's rules and modifiers read directly.

    Mirrors exactly the fields the built rule objects gate their own
    ``ctx.feature`` calls on — an ``ATRStop`` always reads its ATR, a
    ``StructureStop`` always reads its swing but only reads an ATR when its
    ``buffer_atr_multiple`` is positive — so a column this function omits is one
    no rule in the preset would actually have asked for either. The five rules
    and stop modifiers that read no feature at all (``ProtectiveStop``,
    ``FixedRR``, ``PartialClose``, ``TimeExit``, ``SignalReverseExit``,
    ``BreakevenMove``) contribute nothing, which they do simply by not matching
    any branch below.

    Args:
        preset: The exit spec a position's plan will be built from.

    Returns:
        One :class:`~trading_system.features.pipeline.FeatureSpec` per column
        some rule or modifier in the preset reads, before dedup against the
        stream's other requirements.
    """
    specs: list[FeatureSpec] = []
    for modifier in preset.stop_modifiers:
        if isinstance(modifier, ATRStopSpec):
            specs.append(_atr_feature(modifier.period))
        elif isinstance(modifier, StructureStopSpec):
            # swing_level() is called unconditionally; the ATR buffer only when
            # buffer_atr_multiple is positive — the same two calls
            # StructureStop.on_bar makes.
            specs.append(_swing_feature(modifier.lookback))
            if modifier.buffer_atr_multiple > 0:
                specs.append(_atr_feature(modifier.atr_period))
        elif isinstance(modifier, TrailingStopSpec):
            specs.extend(_trailing_source_features(modifier.source))
    return specs


def _trailing_source_features(source: TrailingSourceSpec) -> list[FeatureSpec]:
    """Feature columns one :class:`TrailingStopSpec` source reads.

    Args:
        source: Which of the four trailing sources is configured.

    Returns:
        The columns that source's branch of
        :meth:`~trading_system.exit.rules.trailing_stop.TrailingStop._level`
        reads.
    """
    if isinstance(source, AtrTrailSpec):
        return [_atr_feature(source.period)]
    if isinstance(source, ChandelierSpec):
        # The rolling high/low extreme comes from raw OHLC via ctx.price, not a
        # feature column; only the ATR distance is one.
        return [_atr_feature(source.period)]
    if isinstance(source, MaTrailSpec):
        specs = [_ma_feature(source.period, source.source)]
        if source.buffer_atr_multiple > 0:
            specs.append(_atr_feature(source.atr_period))
        return specs
    # The remaining variant is SwingTrailSpec: swing_level() is called
    # unconditionally, the ATR buffer only when configured.
    specs = [_swing_feature(source.lookback)]
    if source.buffer_atr_multiple > 0:
        specs.append(_atr_feature(source.atr_period))
    return specs


def _atr_feature(period: int) -> FeatureSpec:
    """The column :func:`~trading_system.exit.rules._features.atr_key` names."""
    return FeatureSpec(name=ATR(period=period).name, kind="atr", params={"period": period})


def _ma_feature(period: int, source: str) -> FeatureSpec:
    """The column :func:`~trading_system.exit.rules._features.ma_key` names."""
    indicator = EMA(period=period, source=source)
    return FeatureSpec(name=indicator.name, kind="ema", params={"period": period, "source": source})


def _swing_feature(lookback: int) -> FeatureSpec:
    """The columns :func:`~trading_system.exit.rules._features.swing_keys` names."""
    return FeatureSpec(
        name=SwingPoints(lookback=lookback).name, kind="swing", params={"lookback": lookback}
    )


def _merged_label_categories(specs: Sequence[StrategySpec]) -> tuple[LabelCategory, ...]:
    """Every label category any of ``specs`` asks about, in enum order.

    Classifying candlesticks walks every bar in Python, so a stream none of whose
    strategies mention ``pattern_is`` should not pay for it — but a stream where
    even one of several strategies does must build the column, since it is one
    shared :class:`~trading_system.entry.context.BarSeries` per stream, not one
    per strategy.

    Args:
        specs: Every strategy trading one stream.

    Returns:
        The categories to classify, deduplicated and in
        :class:`~trading_system.entry.labels.LabelCategory` order.
    """
    found: set[LabelCategory] = set()
    for spec in specs:
        found.update(required_label_categories(spec))
    return tuple(category for category in LabelCategory if category in found)


def _closing_size(decision: ExitDecision, position: ManagedPosition) -> Decimal:
    """Size in lots that one exit decision closes.

    The **exact** fractional size the position books, not a lot-quantised one.
    ``ManagedPosition`` tracks fractions of the original size in exact decimal,
    which is what makes "the position is closed" an equality rather than a
    comparison against a tolerance; charging commission on a quantised size
    instead would break the identity that a ladder's commissions sum to the
    commission on the whole. Quantisation is used only to detect the size that
    rounds away to nothing, which is the case P10 guards at position open.
    """
    if decision.kind is ExitKind.FULL or decision.fraction is None:
        return position.remaining_size
    return min(position.size * decision.fraction, position.remaining_size)
