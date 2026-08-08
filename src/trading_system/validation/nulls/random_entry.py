"""The zero for whether these entries are better than random ones with the same profile.

**The schedule is a pure function of ``close_ts``, never of a bar.**
:func:`sample_schedule` takes a plain ``Sequence[datetime]`` — no
:class:`~trading_system.entry.context.BarContext`, no
:class:`~trading_system.data.models.OHLCVFrame` reaches it, structurally, the
same way :class:`~trading_system.backtest.engine.BarStore` prevents lookahead
by not holding a bar it has not published rather than by a rule someone has to
remember. :class:`RandomEntryEngine` — the thing actually wired into the
event loop — is handed the finished schedule and does exactly one price read
per hit, to price a signal's ``reference_price`` the same way the real
compiler prices a ``MARKET`` order (P06: ``reference = close(t)``); that read
happens *after* the scheduling decision, never as part of it.

**Everything else about the null's trades is a real trade, through the real
machinery.** A hit builds a real
:class:`~trading_system.entry.signal.EntrySignal`, which
:class:`~trading_system.backtest.orchestrator.Orchestrator` passes through the
unmodified Risk Engine, Prop Guard, execution and portfolio — nothing here
invents a fill or skips sizing. The only substitution is
``Orchestrator._entries[(strategy_id, key)]``, mutated after construction
(the dict is a plain instance attribute the orchestrator never re-derives) —
:class:`RandomEntryEngine` duck-types the two methods
:meth:`~trading_system.backtest.orchestrator.Orchestrator.on_recognise`
actually calls (``evaluate``, ``drops``), and nothing about
:class:`~trading_system.backtest.spec.RunInputs` or ``Orchestrator.__init__``
needs to know a substitute exists.

**A synthetic ``StrategySpec`` is still required, and most of it still
matters.** ``entries[].trigger`` is inert — compiled once at construction so
:func:`~trading_system.entry.compiler.compile_entry` does not raise, never
evaluated once the substitute engine is in place — but ``entries[].entry_order``
is read directly off the spec by
:meth:`~trading_system.backtest.orchestrator.Orchestrator._queue_entry`, and
``risk_profile.stop_reference`` is read unconditionally by
:meth:`~trading_system.risk.engine.RiskEngine.evaluate`. Both have to be real.

**``FixedHoldRandomEntryNull`` isolates the exit's own contribution.** The
plain random-entry null still exits through the real strategy's own rules, so
a system whose entire edge lives in the exit (a well-tuned trail, a
volatility stop) can score near the median on entries alone and look like
noise — the difference between the two null variants *is* that contribution,
not a second opinion on the same question.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from random import Random
from typing import Any

from trading_system.backtest.clock import StreamKey, bar_close_ts
from trading_system.backtest.engine import BarEvent
from trading_system.backtest.orchestrator import (
    BacktestResult,
    Orchestrator,
    PendingEntry,
    StrategyBinding,
    _merged_feature_specs,
    _merged_label_categories,
    _merged_registry,
    derive_run_calendar,
)
from trading_system.backtest.portfolio import TradeRecord
from trading_system.backtest.spec import RunInputs
from trading_system.core.types import Price, Side, Timeframe
from trading_system.data.models import OHLCVFrame
from trading_system.data.resample import DayOrigin
from trading_system.entry.compiler import compile_entry
from trading_system.entry.context import BarContext, BarSeries
from trading_system.entry.signal import EntrySignal
from trading_system.execution.costs import CostModel
from trading_system.execution.orders import Fill
from trading_system.exit.library import (
    ExitPresetSpec,
    ProtectiveStopSpec,
    TimeExitSpec,
    build_plan,
)
from trading_system.exit.rules.time_exit import TimeExitMode
from trading_system.features.pipeline import FeaturePipeline
from trading_system.risk.circuit_breakers import CircuitBreakers
from trading_system.risk.engine import RiskEngine
from trading_system.risk.portfolio_risk import PortfolioRisk
from trading_system.strategies.schema import StrategySpec

#: Below this many real signals, hour/weekday marginals are estimated from too
#: few observations to mean much — reported as a flag, not withheld.
HOUR_WEEKDAY_MIN_SAMPLES = 30


@dataclass(frozen=True)
class EntryTraceProfile:
    """The empirical trace a null's schedule is matched against.

    Attributes:
        n_signals: How many real signals the schedule tries to reproduce.
        hour_weights: Marginal distribution over ``bar_close_ts.hour`` (0-23),
            summing to 1 over the hours actually observed.
        weekday_weights: Marginal distribution over ``bar_close_ts.weekday()``
            (0=Monday..6=Sunday), summing to 1.
        long_fraction: Share of real signals with ``side is Side.BUY``.
        quality_samples: Every real signal's ``quality``, sampled from
            independently of which bar a null signal lands on.
        hold_bars_samples: Real trades' holding time in bars
            (``round((closed_at - opened_at) / timeframe.duration)``, floored
            at 1) — read by :class:`FixedHoldRandomEntryNull` only.
        undersampled: Whether ``n_signals`` is below :data:`HOUR_WEEKDAY_MIN_SAMPLES`
            — the marginals above are still used, but are a rough estimate.
    """

    n_signals: int
    hour_weights: Mapping[int, float]
    weekday_weights: Mapping[int, float]
    long_fraction: float
    quality_samples: tuple[float, ...]
    hold_bars_samples: tuple[int, ...]
    undersampled: bool


def real_signals(
    frame_streams: Mapping[StreamKey, OHLCVFrame], binding: StrategyBinding, key: StreamKey
) -> tuple[EntrySignal, ...]:
    """Recognise the real strategy's signals on one stream, without sizing or execution.

    The same recognition path :class:`~trading_system.backtest.orchestrator.Orchestrator`
    builds per stream (registry, features, labels), run once through
    :meth:`~trading_system.entry.compiler.EntryEngine.run` rather than the
    bar-by-bar loop — nothing here spends money, so there is no need for the
    rest of the event loop to get a signal trace to calibrate a null against.

    Args:
        frame_streams: Bars per stream — the real run's own ``RunInputs.streams``.
        binding: The real strategy and the exit preset it pairs with.
        key: Which stream to recognise signals on.

    Returns:
        Every signal the real strategy would recognise on this stream, oldest
        first.
    """
    frame = frame_streams[key]
    registry = _merged_registry([binding.spec])
    feature_specs = _merged_feature_specs(registry.specs, [binding.exit_preset])
    categories = _merged_label_categories([binding.spec])
    features = FeaturePipeline(feature_specs).compute(frame) if feature_specs else None
    series = BarSeries.from_frame(frame, features=features, categories=categories)
    engine = compile_entry(binding.spec, registry)
    return tuple(engine.run(series).signals)


def build_entry_trace_profile(
    signals: Sequence[EntrySignal], trades: Sequence[TradeRecord], timeframe: Timeframe
) -> EntryTraceProfile:
    """Summarise a real run's signals and trades into what a null's schedule matches.

    Args:
        signals: The real strategy's recognised signals — see :func:`real_signals`.
        trades: The real run's closed trades
            (:class:`~trading_system.backtest.portfolio.TradeRecord`), for the
            holding-time distribution.
        timeframe: The stream's bar size, to convert holding time into bars.

    Returns:
        The profile.

    Raises:
        ValueError: If ``signals`` is empty — a null calibrated against zero
            signals answers no question.
    """
    if not signals:
        raise ValueError("cannot build an entry trace profile from zero signals")

    hour_counts: dict[int, int] = {}
    weekday_counts: dict[int, int] = {}
    long_count = 0
    for signal in signals:
        hour_counts[signal.bar_close_ts.hour] = hour_counts.get(signal.bar_close_ts.hour, 0) + 1
        weekday = signal.bar_close_ts.weekday()
        weekday_counts[weekday] = weekday_counts.get(weekday, 0) + 1
        if signal.side is Side.BUY:
            long_count += 1

    n = len(signals)
    hold_bars = tuple(
        max(1, round((trade.closed_at - trade.opened_at) / timeframe.duration)) for trade in trades
    )
    return EntryTraceProfile(
        n_signals=n,
        hour_weights={hour: count / n for hour, count in hour_counts.items()},
        weekday_weights={day: count / n for day, count in weekday_counts.items()},
        long_fraction=long_count / n,
        quality_samples=tuple(signal.quality for signal in signals),
        hold_bars_samples=hold_bars,
        undersampled=n < HOUR_WEEKDAY_MIN_SAMPLES,
    )


@dataclass(frozen=True)
class ScheduledSignal:
    """One entry the null schedule places, decided from calendar alone.

    Attributes:
        close_ts: Bar this lands on, by close time.
        side: Direction.
        quality: Sampled from the real run's own signals, independent of
            ``close_ts``.
    """

    close_ts: datetime
    side: Side
    quality: float


def sample_schedule(
    close_ts: Sequence[datetime], profile: EntryTraceProfile, *, seed: int
) -> tuple[ScheduledSignal, ...]:
    """Sample a null's entry schedule from calendar features alone.

    Args:
        close_ts: Every bar close time the schedule may land on — nothing
            else about the bar (see the module docstring on why this
            function's own signature is the structural guarantee, not a
            convention).
        profile: What the schedule is matched to — signal count, hour and
            weekday marginals, long/short balance, quality distribution.
        seed: Seed for the sampler.

    Returns:
        Up to ``profile.n_signals`` scheduled entries (fewer if ``close_ts``
        cannot support that many without duplicating a bar), sorted by
        ``close_ts``. Empty if ``profile.n_signals`` is zero.
    """
    if profile.n_signals == 0 or not close_ts:
        return ()

    rng = Random(seed)
    buckets: dict[tuple[int, int], list[datetime]] = {}
    for ts in close_ts:
        buckets.setdefault((ts.weekday(), ts.hour), []).append(ts)

    weight_by_key: dict[tuple[int, int], float] = {}
    for key in buckets:
        weekday, hour = key
        weight = profile.weekday_weights.get(weekday, 0.0) * profile.hour_weights.get(hour, 0.0)
        if weight > 0:
            weight_by_key[key] = weight
    if not weight_by_key:
        # The real profile's observed hours/weekdays never occur in this
        # coverage at all (a short synthetic fixture, most likely) — fall
        # back to uniform over whatever bars exist rather than scheduling
        # nothing.
        weight_by_key = {key: float(len(members)) for key, members in buckets.items()}

    available = {key: list(members) for key, members in buckets.items()}
    scheduled: list[ScheduledSignal] = []
    target = min(profile.n_signals, len(close_ts))
    for _ in range(target):
        live_keys = [key for key in weight_by_key if available.get(key)]
        if not live_keys:
            break
        live_weights = [weight_by_key[key] for key in live_keys]
        chosen = rng.choices(live_keys, weights=live_weights, k=1)[0]
        pool = available[chosen]
        ts = pool.pop(rng.randrange(len(pool)))
        side = Side.BUY if rng.random() < profile.long_fraction else Side.SELL
        quality = rng.choice(profile.quality_samples) if profile.quality_samples else 0.5
        scheduled.append(ScheduledSignal(close_ts=ts, side=side, quality=quality))

    return tuple(sorted(scheduled, key=lambda item: item.close_ts))


class RandomEntryEngine:
    """Duck-typed substitute for :class:`~trading_system.entry.compiler.EntryEngine`.

    Consults a precomputed schedule by the bar's own close time — the only
    price read is the one needed to price an already-decided signal's
    ``reference_price``, never to decide whether to fire at all.
    """

    def __init__(
        self,
        schedule: Mapping[datetime, ScheduledSignal],
        *,
        strategy_id: str,
        symbol: str,
        invalidation_distance: float,
    ) -> None:
        """Bind the engine to a fixed schedule.

        Args:
            schedule: Bar close time to the signal scheduled there.
            strategy_id: Carried onto every emitted signal.
            symbol: Carried onto every emitted signal.
            invalidation_distance: Absolute price distance from
                ``reference_price`` to ``invalidation_price`` — one number,
                the same for every signal, so the stop this null trades
                against is exactly what its own ``FIXED_PIPS`` risk profile
                already names, and the two never disagree.
        """
        self._schedule = schedule
        self._strategy_id = strategy_id
        self._symbol = symbol
        self._invalidation_distance = invalidation_distance

    @property
    def drops(self) -> Mapping[str, int]:
        """Nothing is ever dropped here — a scheduled hit always emits."""
        return {}

    def reset(self) -> None:
        """No state to clear: the schedule is immutable for the run's life."""

    def evaluate(self, ctx: BarContext) -> tuple[EntrySignal, ...]:
        """Emit the scheduled signal for this bar, if any.

        Args:
            ctx: The published bar.

        Returns:
            Zero or one signal — the schedule places at most one entry per
            bar by construction (:func:`sample_schedule` draws each
            ``close_ts`` at most once).
        """
        scheduled = self._schedule.get(ctx.bar_close_ts)
        if scheduled is None:
            return ()
        close = ctx.price("close")
        if close is None:
            return ()
        reference = Price(close)
        offset = self._invalidation_distance
        if scheduled.side is Side.BUY:
            invalidation = Price(reference - offset)
        else:
            invalidation = Price(reference + offset)
        return (
            EntrySignal(
                strategy_id=self._strategy_id,
                symbol=self._symbol,
                bar_close_ts=ctx.bar_close_ts,
                side=scheduled.side,
                reference_price=reference,
                invalidation_price=invalidation,
                quality=scheduled.quality,
            ),
        )


def _null_strategy_spec(
    strategy_id: str,
    *,
    symbol: str,
    signal_tf: Timeframe,
    instrument_class: str,
    stop_pips: float,
    max_concurrent_positions: int,
) -> StrategySpec:
    """A real, schema-valid ``StrategySpec`` whose trigger is inert.

    ``entries[].trigger`` compiles (``compile_entry`` runs unconditionally at
    :class:`~trading_system.backtest.orchestrator.Orchestrator` construction —
    see the module docstring) but is a constant comparison that never fires
    and is never asked to: :class:`RandomEntryEngine` replaces the compiled
    evaluator entirely. ``entry_order`` and ``risk_profile`` are not inert —
    both are read by the real orchestrator regardless of which engine
    produced the signal.
    """
    never_true = {"type": "leaf", "op": "gt", "left": 0.0, "right": 1.0}
    entries = [
        {
            "direction": direction,
            "trigger": never_true,
            "invalidation": {"price_level": 0.0},
            "entry_order": {"order": {"type": "MARKET"}},
        }
        for direction in ("LONG", "SHORT")
    ]
    return StrategySpec.model_validate(
        {
            "id": strategy_id,
            "name": "Random Entry Null",
            "version": "1.0.0",
            "author": "validation.nulls",
            "type": "INTRADAY",
            "timeframes": {"signal_tf": signal_tf.value, "entry_tf": signal_tf.value},
            "instruments": {"allowed_classes": [instrument_class], "allowed_symbols": [symbol]},
            "entries": entries,
            "exit_ref": "random-entry-null",
            "risk_profile": {
                "base_quality": 0.5,
                "stop_reference": {"kind": "FIXED_PIPS", "pips": stop_pips},
                "max_concurrent_positions": max_concurrent_positions,
            },
        }
    )


def _bar_close_times(
    frame_streams: Mapping[StreamKey, OHLCVFrame], key: StreamKey, day_origin: DayOrigin
) -> list[datetime]:
    """Every bar's close time on one stream — what the schedule is drawn over."""
    frame = frame_streams[key]
    return [
        bar_close_ts(key.timeframe, opened, day_origin) for opened in frame.timestamps.to_list()
    ]


def _build_orchestrator(
    inputs: RunInputs, *, cls: type[Orchestrator] = Orchestrator, **extra: Any
) -> Orchestrator:
    """Assemble an orchestrator from ``inputs``, optionally as a subclass.

    Mirrors :meth:`~trading_system.backtest.spec.RunInputs.orchestrator`
    field for field. Duplicated rather than extending that method with a
    ``cls`` parameter it would carry for exactly one caller
    (:class:`FixedHoldRandomEntryNull`) — the kind of hook stage 1 rules out
    adding "for later".
    """
    breakers = inputs.breakers
    if breakers.calendar is None:
        breakers = breakers.model_copy(
            update={"calendar": derive_run_calendar(inputs.streams, inputs.instruments)}
        )

    return cls(
        config=inputs.config,
        streams=inputs.streams,
        bindings=inputs.bindings,
        instruments=inputs.instruments,
        risk_engine=RiskEngine(
            instruments=inputs.instruments,
            sizing=inputs.sizing,
            converter=inputs.converter,
            config=inputs.risk,
            portfolio=PortfolioRisk(inputs.limits),
            breakers=CircuitBreakers(breakers),
        ),
        cost_model=CostModel(
            {key.symbol: inputs.instruments[key.symbol] for key in inputs.streams}, inputs.costs
        ),
        converter=inputs.converter,
        run_seed=inputs.costs.run_seed,
        **extra,
    )


def _schedule_engine(
    inputs: RunInputs,
    key: StreamKey,
    profile: EntryTraceProfile,
    *,
    strategy_id: str,
    seed: int,
    stop_pips: float,
) -> RandomEntryEngine:
    """The scheduled engine both null variants inject, built the same way."""
    close_times = _bar_close_times(inputs.streams, key, inputs.config.day_origin)
    schedule = {item.close_ts: item for item in sample_schedule(close_times, profile, seed=seed)}
    instrument = inputs.instruments[key.symbol]
    return RandomEntryEngine(
        schedule,
        strategy_id=strategy_id,
        symbol=key.symbol,
        invalidation_distance=instrument.points_to_price(stop_pips),
    )


def run_random_entry_null(
    base: RunInputs,
    key: StreamKey,
    real_binding: StrategyBinding,
    profile: EntryTraceProfile,
    *,
    seed: int,
    stop_pips: float = 20.0,
    max_concurrent_positions: int = 50,
) -> BacktestResult:
    """Run the plain random-entry null: scheduled entries, the real strategy's own exits.

    Args:
        base: The real run's own inputs — same streams, instruments, costs,
            sizing, limits and breakers.
        key: Stream the null trades.
        real_binding: The real strategy's binding, for its exit preset —
            reused unchanged, per this null's own point: only entries differ.
        profile: The schedule is matched to this.
        seed: Seed for the schedule sampler. Not the run's own ``costs.run_seed``,
            which stays the base run's — see CLAUDE.md P15 stage 1.5 on why
            the fill generator must be the same process as the real run's.
        stop_pips: Distance from ``reference_price`` to ``invalidation_price``,
            and the null's own ``FIXED_PIPS`` risk profile — the two always
            agree, by construction.
        max_concurrent_positions: Cap for the null's own strategy. Generous by
            default so the null's own position limit rarely binds where the
            real strategy's did not.

    Returns:
        What the null produced.
    """
    strategy_id = f"random-entry-null-{key.symbol.lower()}"
    instrument_class = base.instruments[key.symbol].asset_class.value
    spec = _null_strategy_spec(
        strategy_id,
        symbol=key.symbol,
        signal_tf=key.timeframe,
        instrument_class=instrument_class,
        stop_pips=stop_pips,
        max_concurrent_positions=max_concurrent_positions,
    )
    binding = StrategyBinding(spec=spec, exit_preset=real_binding.exit_preset, keys=(key,))
    inputs = replace(base, bindings=(binding,))
    engine = _schedule_engine(
        inputs, key, profile, strategy_id=strategy_id, seed=seed, stop_pips=stop_pips
    )

    orch = _build_orchestrator(inputs)
    # RandomEntryEngine duck-types EntryEngine's two call sites
    # (.evaluate, .drops) rather than subclassing it — see the module
    # docstring — so this assignment is a deliberate type mismatch, not
    # an oversight.
    orch._entries[(strategy_id, key)] = engine  # type: ignore[assignment]
    return orch.run()


def _fixed_hold_preset(max_bars_held: int) -> ExitPresetSpec:
    """A minimal preset: protective stop plus a fixed-bar-count time exit, no partials."""
    return ExitPresetSpec(
        id=f"random-entry-null-fixed-hold-{max_bars_held}",
        name="Random entry null: fixed hold",
        protective_stop=ProtectiveStopSpec(),
        rules=[TimeExitSpec(mode=TimeExitMode.MAX_BARS_HELD, max_bars_held=max_bars_held)],
        stop_modifiers=[],
    )


class _FixedHoldOrchestrator(Orchestrator):
    """Swaps a per-position, empirically-sampled hold length in for the static preset.

    Necessary because :meth:`~trading_system.backtest.orchestrator.Orchestrator._open_position`
    builds every position's plan from one static
    :class:`~trading_system.exit.library.ExitPresetSpec` per strategy
    (:func:`~trading_system.exit.library.build_plan` takes no per-call
    parameters), and this null wants a *different* hold length per position,
    drawn from the real run's own holding-time distribution. Overriding
    ``_open_position`` — not name-mangled, confirmed a legitimate seam — swaps
    the plan immediately after the real construction, before the instant's
    exit phase ever reads it.
    """

    def __init__(self, *, hold_sampler: Callable[[], int], **kwargs: Any) -> None:
        """Wire the sampler alongside everything ``Orchestrator`` already takes."""
        super().__init__(**kwargs)
        self._hold_sampler = hold_sampler

    def _open_position(self, pending: PendingEntry, fill: Fill, event: BarEvent) -> None:
        """Open as usual, then replace the freshly-built plan with a sampled-hold one."""
        super()._open_position(pending, fill, event)
        for held in self._portfolio.open_positions:
            if held.position_id == pending.order_id:
                n = self._hold_sampler()
                plan = build_plan(
                    _fixed_hold_preset(n), intrabar_policy=self._config.intrabar_policy
                )
                plan.reset()
                held.plan = plan
                break


def run_fixed_hold_random_entry_null(
    base: RunInputs,
    key: StreamKey,
    profile: EntryTraceProfile,
    *,
    seed: int,
    stop_pips: float = 20.0,
    max_concurrent_positions: int = 50,
) -> BacktestResult:
    """Run the fixed-hold random-entry null: scheduled entries, exit by sampled hold length.

    Isolates the entries' own contribution from the real strategy's exit
    rules — see the module docstring.

    Args:
        base: The real run's own inputs.
        key: Stream the null trades.
        profile: The schedule and the hold-length distribution both come from
            this.
        seed: Seed for the schedule sampler. The hold-length sampler is seeded
            independently (derived from this same seed, offset so re-seeding
            one does not silently reseed the other) so the two draws do not
            share a stream.
        stop_pips: Same role as in :func:`run_random_entry_null`.
        max_concurrent_positions: Same role as in :func:`run_random_entry_null`.

    Returns:
        What the null produced.
    """
    strategy_id = f"random-entry-null-fixed-hold-{key.symbol.lower()}"
    instrument_class = base.instruments[key.symbol].asset_class.value
    spec = _null_strategy_spec(
        strategy_id,
        symbol=key.symbol,
        signal_tf=key.timeframe,
        instrument_class=instrument_class,
        stop_pips=stop_pips,
        max_concurrent_positions=max_concurrent_positions,
    )
    placeholder_hold = profile.hold_bars_samples[0] if profile.hold_bars_samples else 1
    binding = StrategyBinding(
        spec=spec, exit_preset=_fixed_hold_preset(placeholder_hold), keys=(key,)
    )
    inputs = replace(base, bindings=(binding,))
    engine = _schedule_engine(
        inputs, key, profile, strategy_id=strategy_id, seed=seed, stop_pips=stop_pips
    )

    hold_rng = Random(seed ^ 0x686F6C64)  # "hold" — a distinct stream from the schedule's own
    hold_samples = profile.hold_bars_samples or (1,)

    def sample_hold() -> int:
        return hold_rng.choice(hold_samples)

    orch = _build_orchestrator(inputs, cls=_FixedHoldOrchestrator, hold_sampler=sample_hold)
    # RandomEntryEngine duck-types EntryEngine's two call sites
    # (.evaluate, .drops) rather than subclassing it — see the module
    # docstring — so this assignment is a deliberate type mismatch, not
    # an oversight.
    orch._entries[(strategy_id, key)] = engine  # type: ignore[assignment]
    return orch.run()
