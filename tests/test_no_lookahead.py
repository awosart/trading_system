"""The harness. Everything else in this project is honest; this is what makes it proven.

Four checks, each catching a class of leak the others cannot see — the three the
brief names, and a fourth that closes the blind spot the first two turned out to
share once they were made to fail on purpose. They run
the shipped ``ema_pullback`` strategy — not a hand-written stand-in — over a
synthetic H4 series that trends, turns over and produces trades that open *and*
close, because a check that compares two runs producing nothing compares
nothing.

**Truncation.** Run to bar ``T``; run to bar ``T + N``. The shorter run's equity
curve must equal the longer run's first rows exactly, and the trades that closed
inside the prefix must be the *same trades* — same entry price, same exit, same
money, same ids. Compared by value, never by count: two runs producing three
trades each is not evidence that they are the same three. What this catches is a
decision on bar ``t`` that read bar ``t + k``, at the seam where the extra bars
either exist or do not.

**Poisoning.** Replace every bar after ``T`` with garbage — NaN, zeros, absurd
prices — and the result up to ``T`` must not move by one byte. This is the check
truncation cannot make: a statistic fitted once over the whole dataset (a
normalisation, a mean, a baseline) is *present* in both truncated runs and
differs between them only in ways truncation reads as "the data really was
different". Poisoning holds the prefix fixed and changes only the future, so any
prefix difference at all is the future leaking backwards.

**Permutation.** Run one instrument; run it alongside a second. The first
instrument's trades must not change. This catches state shared between streams —
a cache, a counter, an evaluator, a random stream — and it is deliberately run
with the *legitimate* couplings switched off: flat-amount sizing so equity does
not feed back, portfolio ceilings out of the way, circuit breakers disabled. Two
instruments in one account really do affect each other through equity and heat;
that is economics, not leakage. What must not affect the first instrument is the
mere presence of the second.

**Each check has been made to fail, and what failed is recorded here rather than
claimed.** A harness that has never failed proves nothing. Three leaks were
introduced into the source, run, and reverted; the results are not symmetric,
and the asymmetry is the most useful thing in this file:

* ``features/pipeline.py`` shifted every computed column back by one bar, so bar
  ``t`` carried bar ``t+1``'s value. **The fourth check below caught it**
  (``TestFeaturesAreCausalBarByBar``). Truncation and poisoning did **not** fail
  their equality assertions — the strategy stopped trading altogether under a
  misaligned RSI, and it was the emptiness guards that fired. Shifting a single
  column by thirty bars, which left the strategy trading, kept every end-to-end
  check green. That is the seam-locality property, not a fluke: see the fourth
  check's own docstring.
* ``backtest/orchestrator.py`` replaced the rolling ATR baseline with the mean
  of the whole series — the textbook whole-dataset statistic. **Poisoning
  failed** on three flavours of four. Truncation did **not**: the mean over
  1300 bars and over 2000 differs by a percent or so, the slippage term it
  feeds moves by thousandths of a point, and tick rounding absorbs it. Only the
  poison, which moves that mean by orders of magnitude, makes the leak bigger
  than the grid. The NaN flavour also passed, for a reason worth knowing: NaN
  bars produce *null* ATR, and a mean skips nulls, so the poisoned mean stayed
  close to the clean one. Four flavours rather than one, for exactly this.
* ``backtest/orchestrator.py`` numbered entry orders from one run-wide counter.
  **Permutation failed**, alone and immediately. This one was never
  hypothetical: it is what the code did before this file existed. Entry ids fed
  :func:`~trading_system.execution.rng.fill_seed`, so a second instrument
  renumbered the first's orders and redrew its slippage — the artefact
  :mod:`trading_system.execution.rng` was written to prevent, reintroduced one
  layer above it. :meth:`~trading_system.backtest.orchestrator.Orchestrator._order_id`
  now derives an id from the signal alone.

Runtime is a few hundredths of a second per run, so the checks sweep several cut
points and several poison flavours rather than picking one and hoping it lands
somewhere interesting.
"""

from collections.abc import Mapping, Sequence
from datetime import datetime
from decimal import Decimal

import polars as pl
import pytest

from tests.backtest.conftest import (
    EURUSD_H4,
    LIBRARY_PATH,
    REGISTRY_PATH,
    XAUUSD_H4,
    RunInputs,
    ema_pullback,
    harness_inputs,
    noisy_costs,
    swing_series,
)
from trading_system.backtest.clock import StreamKey
from trading_system.backtest.orchestrator import (
    BacktestResult,
    StrategyBinding,
    _merged_feature_specs,
    _merged_registry,
)
from trading_system.backtest.portfolio import TradeRecord
from trading_system.backtest.reproducibility import digest
from trading_system.core.instruments import InstrumentRegistry, load_instruments
from trading_system.data.models import OHLCVFrame
from trading_system.exit.library import ExitLibrarySpec
from trading_system.features.pipeline import FeaturePipeline
from trading_system.risk.circuit_breakers import CircuitBreakerConfig
from trading_system.risk.portfolio_risk import PortfolioLimitsConfig
from trading_system.risk.sizing.methods import FixedAmount

#: Bars in the harness series. Long enough that the strategy takes a couple of
#: dozen trades — a check comparing one trade would be comparing an anecdote.
LENGTH = 2000

#: Where the truncated runs stop, as fractions of the series. Several rather than
#: one: a leak of horizon ``k`` bars is only visible at a seam that lands within
#: ``k`` bars of a decision, so one cut point tests one seam and proves little.
CUTS = (0.35, 0.5, 0.65, 0.8, 0.95)

#: Phase offset of the second instrument's series in the permutation check. Not
#: zero, so gold is its own series rather than a copy of EURUSD at another price
#: level, and checked to produce trades of its own — see
#: ``test_the_second_instrument_actually_traded``.
GOLD_PHASE = 1.6

#: Where the poison starts, as a fraction of the series.
POISON_AT = 0.55

#: What the future is replaced by. Each is a different way for a number to be
#: wrong: not a number at all, a price of nothing, a price from another universe,
#: and a price on the wrong side of zero.
POISONS: dict[str, float] = {
    "nan": float("nan"),
    "zero": 0.0,
    "absurd": 987_654.0,
    "negative": -5.0,
}

#: OHLCV columns the poison overwrites — every column but the timestamp, which
#: must stay intact so both runs walk the same instants and the comparison is
#: between two runs of the same clock rather than two different clocks.
_POISONED_COLUMNS = ("open", "high", "low", "close", "volume")


@pytest.fixture(scope="session")
def registry() -> InstrumentRegistry:
    """The bundled instrument registry.

    Declared here rather than inherited: the backtest package's conftest is not
    visible from this directory, and the harness deliberately lives at the top
    of the tree, next to nothing, because it is a claim about the whole system
    rather than about one package.
    """
    return load_instruments(REGISTRY_PATH)


@pytest.fixture(scope="session")
def library() -> ExitLibrarySpec:
    """The bundled exit library, as specs."""
    return ExitLibrarySpec.model_validate_json(LIBRARY_PATH.read_text())


@pytest.fixture
def harness(registry: InstrumentRegistry, library: ExitLibrarySpec) -> RunInputs:
    """One EURUSD H4 run of the shipped ``ema_pullback``, with costs turned on.

    Costs deliberately noisy: ``jitter_points`` is what makes a fill draw from
    :func:`~trading_system.execution.rng.fill_seed` at all, and
    ``atr_coefficient`` is what makes its price depend on a rolling statistic.
    With both at zero, two of these three checks would be looking at code paths
    that never execute.
    """
    spec = ema_pullback()
    preset = next(item for item in library.presets if item.id == spec.exit_ref)
    return harness_inputs(
        registry,
        streams={EURUSD_H4: swing_series(LENGTH)},
        bindings=[StrategyBinding(spec=spec, exit_preset=preset, keys=(EURUSD_H4,))],
    )


def truncated(frame: OHLCVFrame, bars: int) -> OHLCVFrame:
    """The first ``bars`` bars of a frame, untouched."""
    return frame.with_df(frame.df.head(bars))


def poisoned(frame: OHLCVFrame, *, after: int, value: float) -> OHLCVFrame:
    """A frame whose bars after ``after`` are all ``value``, timestamps kept.

    Args:
        frame: The clean series.
        after: How many leading bars survive untouched.
        value: What every OHLCV field of every later bar becomes.

    Returns:
        The poisoned frame, still structurally valid — the timestamps are the
        originals, so the poisoned run walks exactly the instants the clean run
        walked and any difference is about values rather than about the clock.
    """
    table = frame.df
    tail = table.slice(after)
    replacement = [
        pl.Series(name, [value] * tail.height, dtype=pl.Float64) for name in _POISONED_COLUMNS
    ]
    return frame.with_df(pl.concat([table.head(after), tail.with_columns(replacement)]))


def trades_closed_by(result: BacktestResult, moment: datetime) -> list[TradeRecord]:
    """Trades of a run that had gone flat at or before ``moment``.

    A position still open at ``moment`` is not a trade, in either run: in the
    shorter one the data simply ran out, and in the longer one the trade closed
    on bars the shorter run never saw.
    """
    return [trade for trade in result.trades if trade.closed_at <= moment]


def assert_same_prefix(short: BacktestResult, long_run: BacktestResult, *, rows: int) -> None:
    """Assert two runs agree exactly over the first ``rows`` instants.

    Compares the equity rows and the completed trades **by value** and then by
    digest. The digest is not redundant: value equality is what says which field
    differs, and the digest is what says "not one byte", including the fields a
    reader of this test has not thought to name.
    """
    horizon = long_run.curve[rows - 1].ts
    short_curve = list(short.curve)[:rows]
    long_curve = list(long_run.curve)[:rows]
    assert short_curve == long_curve
    short_trades = trades_closed_by(short, horizon)
    long_trades = trades_closed_by(long_run, horizon)
    assert short_trades == long_trades
    assert digest([short_curve, short_trades]) == digest([long_curve, long_trades])


class TestTheHarnessHasSomethingToCompare:
    """Every check below is vacuous on a run that does nothing. This is the guard."""

    def test_the_strategy_trades_and_nothing_silently_suppressed_it(
        self, harness: RunInputs
    ) -> None:
        """The base run must produce trades, and produce them for the stated reason."""
        result = harness.run()

        assert len(result.trades) >= 10
        assert result.fills >= 2 * len(result.trades)
        # Nothing was refused or discarded, so the checks below compare a
        # strategy's decisions rather than a limit's.
        assert not [reason for reason, count in result.rejections.items() if count]
        assert not [reason for reason, count in result.signal_drops.items() if count]

    def test_fills_are_priced_through_the_paths_a_leak_would_hide_in(
        self, harness: RunInputs
    ) -> None:
        """Randomness and volatility-dependence must both be live in this configuration.

        With ``jitter_points`` at zero no fill consults its random stream, and
        the permutation check could not see a shared one. With
        ``atr_coefficient`` at zero no fill price depends on a rolling
        statistic, and the poisoning check could not see a whole-series one.
        """
        slippage = harness.costs.slippage

        assert slippage.market.jitter_points > 0
        assert slippage.stop.jitter_points > 0
        assert slippage.market.atr_coefficient > 0
        assert slippage.stop.atr_coefficient > 0


class TestTruncation:
    """A run given fewer bars must reproduce the longer run's prefix exactly."""

    @pytest.mark.parametrize("cut", CUTS)
    def test_a_shorter_run_reproduces_the_longer_runs_prefix(
        self, harness: RunInputs, cut: float
    ) -> None:
        """Same instants, same trades, same money — by value, not by count."""
        bars = int(LENGTH * cut)
        full = harness.run()
        short = harness.with_streams({EURUSD_H4: truncated(harness.streams[EURUSD_H4], bars)}).run()

        assert short.instants == bars
        assert_same_prefix(short, full, rows=bars)

    def test_the_cut_points_between_them_cover_completed_trades(self, harness: RunInputs) -> None:
        """The prefixes compared above are not empty ones.

        Without this, every truncation assertion could be passing on two runs
        that had not yet traded — true, and evidence of nothing.
        """
        full = harness.run()
        covered = [
            len(trades_closed_by(full, full.curve[int(LENGTH * cut) - 1].ts)) for cut in CUTS
        ]

        assert covered[0] >= 1
        assert covered == sorted(covered)
        assert covered[-1] >= 10


class TestPoisoning:
    """Replacing the future with garbage must not move the past."""

    @pytest.mark.parametrize("flavour", sorted(POISONS))
    def test_the_prefix_survives_a_poisoned_future(self, harness: RunInputs, flavour: str) -> None:
        """Every result up to the poison line is identical to the clean run's."""
        bars = int(LENGTH * POISON_AT)
        clean = harness.run()
        spoiled = harness.with_streams(
            {EURUSD_H4: poisoned(harness.streams[EURUSD_H4], after=bars, value=POISONS[flavour])}
        ).run()

        assert spoiled.instants == clean.instants
        assert_same_prefix(spoiled, clean, rows=bars)

    @pytest.mark.parametrize("flavour", sorted(POISONS))
    def test_the_poison_reached_the_run_at_all(self, harness: RunInputs, flavour: str) -> None:
        """After the line, the poisoned run must differ — otherwise it proved nothing.

        A poison the run ignores would make the check above pass for the wrong
        reason: not "the future cannot reach the past", but "the future was never
        different".
        """
        bars = int(LENGTH * POISON_AT)
        clean = harness.run()
        spoiled = harness.with_streams(
            {EURUSD_H4: poisoned(harness.streams[EURUSD_H4], after=bars, value=POISONS[flavour])}
        ).run()

        assert list(spoiled.curve)[bars:] != list(clean.curve)[bars:]


class TestFeaturesAreCausalBarByBar:
    """Truncation again, at the one place it has a blind spot end to end.

    A leak of horizon **one bar** is invisible to the two checks above, and the
    reason is structural rather than a weakness of this data. A decision taken
    on the last bar of a prefix cannot execute before the next bar's open, which
    is the first bar the prefix does not contain — so the *consequence* of the
    leak falls outside everything there is to compare. Both checks catch a leak
    from two bars out, where the fill lands inside the prefix; neither catches
    one bar.

    Inside the loop that gap is closed by construction:
    :class:`~trading_system.backtest.engine.BarStore` raises on any read past
    the cursor, so nothing evaluated bar by bar can reach the next bar at all.
    The exposure is the **vectorised** feature layer, which computes whole
    columns up front — where a stray ``shift(-1)`` is one character and no
    exception can be raised, because the value really is in the frame.

    So the same truncation argument is applied there directly, and at every bar
    rather than at five cut points: the features computed over the first ``t+1``
    bars must agree, in their last row, with the features computed over the
    whole series at bar ``t``. Every indicator, every channel. This is the P06
    equivalence test — a context on ``t`` over the full series must equal one on
    ``t`` over the series cut at ``t`` — applied one layer up, to the pipeline
    that P13 assembles per stream.
    """

    def test_a_bars_features_do_not_depend_on_bars_after_it(self, library: ExitLibrarySpec) -> None:
        """Recompute on a growing prefix and demand the newest row never moves."""
        spec = ema_pullback()
        preset = next(item for item in library.presets if item.id == spec.exit_ref)
        specs = _merged_feature_specs(_merged_registry([spec]).specs, [preset])
        frame = swing_series(LENGTH)
        pipeline = FeaturePipeline(specs)
        whole = pipeline.compute(frame).df

        # The columns are the run's own, entry references and exit rules
        # together, so this checks what is actually computed rather than a
        # list re-derived here and free to drift from it.
        assert len(specs) >= 4
        assert whole.height == LENGTH

        for index in range(200, LENGTH, 97):
            prefix = pipeline.compute(frame.with_df(frame.df.head(index + 1))).df
            assert prefix.height == index + 1
            assert prefix.slice(index, 1).equals(whole.slice(index, 1))

    def test_the_registry_this_checks_is_the_one_the_run_computes(
        self, library: ExitLibrarySpec
    ) -> None:
        """The pipeline above must cover the exit's columns as well as the entry's.

        ``ema_pullback`` pairs with ``structure_trail``, whose swing level and
        ATR buffer are read by name through
        :mod:`trading_system.exit.rules._features` and are invisible to the
        entry-side registry. A check that only covered entry references would
        leave every exit column unexamined.
        """
        spec = ema_pullback()
        preset = next(item for item in library.presets if item.id == spec.exit_ref)
        entry_only = {item.name for item in _merged_registry([spec]).specs}
        merged = {
            item.name for item in _merged_feature_specs(_merged_registry([spec]).specs, [preset])
        }

        assert merged - entry_only


class TestPermutation:
    """A second instrument in the same run must not change the first one's trades."""

    @pytest.fixture
    def two_instruments(
        self, registry: InstrumentRegistry, library: ExitLibrarySpec
    ) -> tuple[RunInputs, RunInputs]:
        """One EURUSD run, and the same run with gold alongside it.

        The two legitimate couplings are switched off deliberately, and this is
        the whole design of the check. Equity-linked sizing is replaced by a flat
        stake, the portfolio ceilings are lifted and the circuit breakers are
        disabled — because two instruments in one account genuinely do compete
        for heat and genuinely do share a drawdown, and a check that called that
        leakage would be wrong. What is left is presence alone: gold's bars
        existing must not change what EURUSD does.

        Gold rather than a second FX pair because it is quoted in the account
        currency, so no FX rate enters and the comparison stays about the
        streams. Its own series is phase-shifted rather than copied, so the two
        instruments are not accidentally the same instrument twice.
        """
        eurusd = ema_pullback()
        gold = ema_pullback(
            strategy_id="ema-pullback-gold", asset_class="COMMODITY", symbol="XAUUSD"
        )
        preset = next(item for item in library.presets if item.id == eurusd.exit_ref)
        alone = _independent_run(
            registry,
            streams={EURUSD_H4: swing_series(LENGTH)},
            bindings=[StrategyBinding(spec=eurusd, exit_preset=preset, keys=(EURUSD_H4,))],
        )
        together = _independent_run(
            registry,
            streams={
                EURUSD_H4: swing_series(LENGTH),
                XAUUSD_H4: swing_series(
                    LENGTH, symbol="XAUUSD", base=2000.0, scale=2000.0, phase=GOLD_PHASE
                ),
            },
            bindings=[
                StrategyBinding(spec=eurusd, exit_preset=preset, keys=(EURUSD_H4,)),
                StrategyBinding(spec=gold, exit_preset=preset, keys=(XAUUSD_H4,)),
            ],
        )
        return alone, together

    def test_the_first_instruments_trades_are_untouched_by_the_second(
        self, two_instruments: tuple[RunInputs, RunInputs]
    ) -> None:
        """Same entries, same exits, same money, same ids."""
        alone, together = two_instruments
        one = alone.run()
        both = together.run()

        assert _trades_of(both, "EURUSD") == list(one.trades)
        assert digest(_trades_of(both, "EURUSD")) == digest(list(one.trades))

    def test_the_second_instrument_actually_traded(
        self, two_instruments: tuple[RunInputs, RunInputs]
    ) -> None:
        """A silent second stream would make the check above prove nothing."""
        _alone, together = two_instruments
        both = together.run()

        assert _trades_of(both, "XAUUSD")

    def test_neither_run_had_a_signal_refused_or_dropped(
        self, two_instruments: tuple[RunInputs, RunInputs]
    ) -> None:
        """The couplings this check switches off must really be switched off.

        If a portfolio ceiling or a breaker bound in the two-instrument run, the
        equality above would be asserting something false — and would be being
        rescued by the limits happening not to bite on this data, which is not a
        property anybody wants to depend on.
        """
        alone, together = two_instruments
        for result in (alone.run(), together.run()):
            assert not [reason for reason, count in result.rejections.items() if count]
            assert not [reason for reason, count in result.degradations.items() if count]
            assert not [reason for reason, count in result.signal_drops.items() if count]


def _trades_of(result: BacktestResult, symbol: str) -> Sequence[TradeRecord]:
    """One instrument's trades, in the order the run closed them."""
    return [trade for trade in result.trades if trade.symbol == symbol]


def _independent_run(
    registry: InstrumentRegistry,
    *,
    streams: Mapping[StreamKey, OHLCVFrame],
    bindings: Sequence[StrategyBinding],
) -> RunInputs:
    """A run whose instruments compete for nothing.

    Flat-stake sizing so equity does not feed back into position size, ceilings
    lifted so heat taken by one instrument is not headroom denied to another,
    breakers off so one instrument's losing day does not stop the other's
    trading. Every one of those is a real coupling in a real portfolio; the
    permutation check is about what remains when they are gone.

    Args:
        registry: Instrument specifications.
        streams: Bars per stream.
        bindings: Strategies and the streams they trade.

    Returns:
        The inputs.
    """
    return harness_inputs(
        registry,
        streams=streams,
        bindings=bindings,
        costs=noisy_costs(),
        sizing=FixedAmount(Decimal(500)),
        limits=PortfolioLimitsConfig(
            max_portfolio_risk_pct=1.0,
            max_instrument_risk_pct=1.0,
            max_cluster_risk_pct=1.0,
            max_strategy_risk_pct=1.0,
        ),
        breakers=CircuitBreakerConfig(
            max_daily_loss_pct=None, max_weekly_loss_pct=None, max_monthly_loss_pct=None
        ),
    )
