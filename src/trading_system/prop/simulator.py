"""What are the odds this strategy passes the firm's account, and blows it.

Replays a finished strategy's trades through one firm's rulebook, many times,
resampling the order of outcomes. It answers three questions a single backtest
cannot: how often the account passes, how often it is lost, and how long either
takes.

**The resampling unit is the block, exactly as in P15.** Trades are shuffled
*within* a fold and the order of folds is preserved, for the reason
:mod:`trading_system.validation.monte_carlo` gives at length: folds share
neither parameters (the optimiser chose different ones) nor market regime, so a
pooled shuffle would produce sequences the system could not have generated. The
literal :class:`~trading_system.validation.monte_carlo.BlockPermutation` cannot
be reused here because
:class:`~trading_system.validation.monte_carlo.TradeSample` carries R multiples
alone, and every rule in this module needs a *date* — but the unit is the same
one, and ``test_simulator.py`` asserts the two produce the same per-fold
multiset rather than leaving that as a claim in prose.

**R multiples are shuffled onto fixed date slots.** The dates are the schedule
— when this strategy would have been in a position, how many trades landed on
one day, which days it sat out — and the R values are what lands on them. That
keeps the day structure the rules depend on while randomising the outcomes,
which is precisely the "path, not multiset" question a permutation exists to
ask. Permuting ``(date, r)`` pairs together instead would put the dates out of
order and dissolve the day grouping the daily limit is defined on.

**A permutation cannot change the final equity, only the path.** P15 records
this as arithmetic — compounding is a product and multiplication commutes — and
it matters more here than there. Every question this module asks is a question
about the *path*: does equity touch the target before the floor, does any one
day contribute too much of the profit, how many days pass on the way. That is
exactly what a permutation varies, and exactly what a single backtest reports
one arbitrary draw of.

**The stopping rule is a modelling decision, and ``min_trading_days`` is what
makes it one.** An episode ends when the account passes, when it is lost, or
when the trades run out. Passing needs *all* of: equity at or above the target,
at least :attr:`~trading_system.prop.rules.PropRules.min_trading_days` days
with a closed trade, and the consistency rule satisfied. A strategy that
reaches the target on day two of a four-day minimum has not passed and has not
failed — it keeps trading, at risk, which is what a real account would do.
Stopping at the target regardless would overstate the odds by counting a pass
the firm would not have granted.

**Ruin is rarer here than it is in life, and the reason is worth knowing before
reading a zero.** An episode only ever loses the account by *trading* into the
floor, and the buffered allowance
(:func:`~trading_system.prop.guard.PropGuard.max_allowed_risk_now`) stops
opening trades before the floor is reachable — so the guard converts most
blow-ups into stagnation rather than preventing them, and ``P(ruin)`` falls
towards zero while ``P(exhausted)`` absorbs the difference. Measured: removing
the buffer and the total-floor headroom check takes ``ema-pullback-h1`` from
``P(ruin) = 0%`` to ``100%``. What this does **not** model is the way a real
account actually dies with a control like this in place: a gap through the stop
on a position that was already open, which no amount of declining to open new
ones prevents. Trades here are a sequence of closed outcomes, so a position's
own path between open and close does not exist to gap. Read ``P(ruin) = 0`` as
"this strategy does not trade itself to death", never as "this account is
safe".

**Consistency is inside the pass predicate, not applied afterwards.** A
strategy that makes everything in one day is not a passing strategy that later
turns out to be disqualified; it is a failing strategy, and the objective in
:mod:`trading_system.prop.objective` has to see that while it is searching
rather than learn it after. The consequence to keep in view: because the
permutation moves outcomes between days, *which* day is the best day varies per
iteration, so the reported figure is the share of iterations that satisfied the
rule rather than a property of the one observed ordering. The observed
ordering's own consistency is reported beside it, unresampled, as
:attr:`PropSimulation.observed_day_share`.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum
from random import Random

from trading_system.backtest.portfolio import TradeRecord
from trading_system.backtest.reproducibility import read_run
from trading_system.data.resample import trading_day
from trading_system.prop.guard import DEFAULT_BUFFER
from trading_system.prop.rules import DailyLossBasis, PropRules, TotalLossBasis
from trading_system.validation.walkforward import WalkForwardResult

#: Iterations a search-time estimate uses. Enough to rank candidates — the
#: standard error of a proportion at this N is at most ``0.5/sqrt(200)`` ≈ 3.5
#: percentage points, which is finer than the differences an optimiser is
#: choosing between and far coarser than a number worth publishing.
SEARCH_ITERATIONS = 200

#: Iterations a reported estimate uses. At most 0.5 percentage points of
#: standard error, which is below the resolution anyone reads a probability at.
REPORT_ITERATIONS = 10_000


@dataclass(frozen=True)
class PropFold:
    """One fold's trades, with the dates the rules need.

    Attributes:
        fold_index: Which fold produced them.
        closed_at: When each trade closed, chronological. The schedule the
            shuffled outcomes are dealt onto.
        r_multiples: ``TradeRecord.realized_r`` per trade, aligned to
            ``closed_at``.
    """

    fold_index: int
    closed_at: tuple[datetime, ...]
    r_multiples: tuple[float, ...]

    def __post_init__(self) -> None:
        """Check the two columns describe the same trades.

        Raises:
            ValueError: If the lengths disagree — a schedule and a set of
                outcomes of different sizes is not a fold.
        """
        if len(self.closed_at) != len(self.r_multiples):
            raise ValueError(
                f"fold {self.fold_index}: {len(self.closed_at)} close times against "
                f"{len(self.r_multiples)} outcomes"
            )

    def __len__(self) -> int:
        """How many trades the fold closed."""
        return len(self.r_multiples)


@dataclass(frozen=True)
class PropSample:
    """Every trade of a walk-forward, grouped by fold, with dates.

    Attributes:
        folds: Per-fold trades, in fold order.
        risk_pct: Equity fraction risked per trade — the run's own sizing
            parameter, not a choice made here. What turns an R multiple back
            into a move in equity.
    """

    folds: tuple[PropFold, ...]
    risk_pct: float

    def __post_init__(self) -> None:
        """Validate the sizing fraction.

        Raises:
            ValueError: If ``risk_pct`` is outside ``(0, 1]``.
        """
        if not 0 < self.risk_pct <= 1:
            raise ValueError(f"risk_pct must be in (0, 1], got {self.risk_pct}")

    @property
    def n_trades(self) -> int:
        """Trades across every fold."""
        return sum(len(fold) for fold in self.folds)

    def observed(self) -> list[tuple[datetime, float]]:
        """The trades in the order they actually happened, folds in fold order."""
        return [
            (when, r)
            for fold in self.folds
            for when, r in zip(fold.closed_at, fold.r_multiples, strict=True)
        ]

    def draw(self, rng: Random) -> list[tuple[datetime, float]]:
        """One block-permuted sequence: outcomes shuffled onto the fixed schedule.

        Args:
            rng: The random source.

        Returns:
            ``(closed_at, r)`` pairs, folds in fold order, dates untouched.
        """
        drawn: list[tuple[datetime, float]] = []
        for fold in self.folds:
            outcomes = list(fold.r_multiples)
            rng.shuffle(outcomes)
            drawn.extend(zip(fold.closed_at, outcomes, strict=True))
        return drawn


def sample_from_walkforward(result: WalkForwardResult, *, risk_pct: float) -> PropSample:
    """Read every fold's out-of-sample trades off disk into a :class:`PropSample`.

    Only out-of-sample trades, for the reason
    :func:`~trading_system.validation.monte_carlo.sample_from_walkforward`
    gives: an in-sample trade is one the parameters were chosen on.

    Args:
        result: A finished walk-forward.
        risk_pct: The run's own per-trade equity fraction.

    Returns:
        The sample.
    """
    folds = []
    for fold_run in result.folds:
        stored = read_run(fold_run.oos_run.path)
        trades = sorted(stored.result.trades, key=lambda trade: trade.closed_at)
        folds.append(
            PropFold(
                fold_index=fold_run.fold.index,
                closed_at=tuple(trade.closed_at for trade in trades),
                r_multiples=tuple(trade.realized_r for trade in trades),
            )
        )
    return PropSample(folds=tuple(folds), risk_pct=risk_pct)


def sample_from_trades(trades: Sequence[TradeRecord], *, risk_pct: float) -> PropSample:
    """A one-fold sample from a flat trade list, for a run that is not a walk-forward."""
    ordered = sorted(trades, key=lambda trade: trade.closed_at)
    return PropSample(
        folds=(
            PropFold(
                fold_index=0,
                closed_at=tuple(trade.closed_at for trade in ordered),
                r_multiples=tuple(trade.realized_r for trade in ordered),
            ),
        ),
        risk_pct=risk_pct,
    )


class Outcome(StrEnum):
    """How one simulated attempt at the account ended."""

    #: Target reached, minimum days served, consistency satisfied.
    PASSED = "passed"

    #: Equity fell to the account floor.
    RUINED = "ruined"

    #: Neither: the trades ran out with the account alive but unpassed.
    EXHAUSTED = "exhausted"


@dataclass(frozen=True)
class Episode:
    """One simulated attempt at the account.

    Attributes:
        outcome: How it ended.
        final_equity: Equity at the end, as a multiple of the starting balance.
        max_drawdown: Deepest peak-to-trough fall along the way, as a fraction.
        trading_days: Days with at least one closed trade.
        calendar_days: Days from the first trade to the last, inclusive.
        best_day_share: Largest share of total profit one day contributed, or
            ``None`` when the attempt never turned a profit at all — a share of
            a non-positive total is not a number this reports as zero.
        blocked_trades: Trades skipped because the daily limit had already been
            hit that day. What the daily rule actually cost in participation.
    """

    outcome: Outcome
    final_equity: float
    max_drawdown: float
    trading_days: int
    calendar_days: int
    best_day_share: float | None
    blocked_trades: int


def run_episode(
    sequence: Sequence[tuple[datetime, float]],
    rules: PropRules,
    *,
    risk_pct: float,
    buffer: float = DEFAULT_BUFFER,
) -> Episode:
    """Walk one sequence of dated outcomes through the firm's rules.

    Equity is carried as a multiple of the starting balance, so the account size
    cancels out of everything except the reported figures — the rules are all
    percentages of a basis, and a percentage of a percentage needs no currency.

    Args:
        sequence: ``(closed_at, r)`` pairs, chronological within each fold.
        rules: The firm's plan.
        risk_pct: Equity fraction risked per trade.
        buffer: Share of the remaining allowance a trade may use, matching
            :class:`~trading_system.prop.guard.PropGuard`. A trade whose risk
            does not fit is **skipped** rather than shrunk: the guard would
            reduce it, but a reduced trade's R is not the R that was recorded,
            and inventing one would put a number in the distribution that no
            backtest produced.

    Returns:
        The episode.
    """
    origin = rules.day_origin
    equity = 1.0
    peak = 1.0
    day_start_equity = 1.0
    day_start_balance = 1.0
    current_day: date | None = None
    traded_days: set[date] = set()
    day_profit: dict[date, float] = {}
    blocked = 0
    worst_drawdown = 0.0
    target = 1.0 + rules.profit_target_pct
    daily_pct = rules.max_daily_loss_pct
    total_pct = rules.max_total_loss_pct
    outcome = Outcome.EXHAUSTED
    first_close = sequence[0][0] if sequence else None
    last_close = first_close

    for closed_at, r in sequence:
        last_close = closed_at
        day = trading_day(closed_at, origin)
        if day != current_day:
            current_day = day
            # A closed-only basis ignores whatever was floating at the reset;
            # an equity basis does not. With no open positions modelled here
            # the two coincide, and the branch is kept because the *rule* is
            # what differs between firms, not this simulator's convenience.
            day_start_equity = equity
            day_start_balance = equity

        basis = (
            day_start_balance
            if rules.daily_loss_basis is DailyLossBasis.BALANCE_AT_DAY_START
            else day_start_equity
        )
        daily_floor = basis * (1 - daily_pct)
        total_basis = 1.0 if rules.total_loss_basis is TotalLossBasis.STATIC else peak
        total_floor = total_basis * (1 - total_pct)

        if equity <= daily_floor or equity <= total_floor:
            blocked += 1
            continue

        headroom = min(equity - daily_floor, equity - total_floor)
        if risk_pct > headroom * buffer:
            blocked += 1
            continue

        before = equity
        equity = max(0.0, equity * (1 + risk_pct * r))
        traded_days.add(day)
        day_profit[day] = day_profit.get(day, 0.0) + (equity - before)
        peak = max(peak, equity)
        if peak > 0:
            worst_drawdown = max(worst_drawdown, (peak - equity) / peak)

        recomputed_total_basis = 1.0 if rules.total_loss_basis is TotalLossBasis.STATIC else peak
        if equity <= recomputed_total_basis * (1 - total_pct):
            outcome = Outcome.RUINED
            break

        if equity >= target and len(traded_days) >= rules.min_trading_days:
            profit = equity - 1.0
            share = max(day_profit.values()) / profit if profit > 0 else None
            limit = rules.max_single_day_profit_share
            if limit is None or share is None or share <= limit:
                outcome = Outcome.PASSED
                break

    profit = equity - 1.0
    best_share = (max(day_profit.values()) / profit) if profit > 0 and day_profit else None
    span = (
        (last_close.date() - first_close.date()).days + 1
        if first_close is not None and last_close is not None
        else 0
    )
    return Episode(
        outcome=outcome,
        final_equity=equity,
        max_drawdown=worst_drawdown,
        trading_days=len(traded_days),
        calendar_days=span,
        best_day_share=best_share,
        blocked_trades=blocked,
    )


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile of an already-sorted sequence."""
    if not sorted_values:
        raise ValueError("cannot take a percentile of nothing")
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * q / 100.0
    low = int(position)
    high = min(low + 1, len(sorted_values) - 1)
    weight = position - low
    return sorted_values[low] * (1 - weight) + sorted_values[high] * weight


#: Below this, :meth:`PropSimulation.summary` says the strategy fails rather
#: than reporting a number. Thirty per cent because a challenge costs money and
#: an attempt with worse than one-in-three odds is a purchase, not a strategy.
UNLIKELY_THRESHOLD = 0.30


@dataclass(frozen=True)
class PropSimulation:
    """What many simulated attempts at one firm's account came to.

    Attributes:
        rules_name: Which plan was applied.
        iterations: How many attempts were simulated.
        p_pass: Share that passed.
        p_ruin: Share that lost the account.
        p_exhausted: Share that ended alive and unpassed.
        se_pass: Standard error of :attr:`p_pass`, ``sqrt(p(1-p)/n)``. Carried
            beside the estimate because a probability from a simulation is a
            measurement with a width, and a plateau analysis run against a
            tolerance finer than this width is measuring the simulator rather
            than the strategy.
        mean_trading_days: Mean days with a closed trade, across attempts.
        mean_calendar_days_to_pass: Mean calendar span of the attempts that
            passed, or ``None`` when none did.
        drawdown_median: Median of the per-attempt maximum drawdowns.
        drawdown_p95: 95th percentile of the same — the bad-but-not-worst case.
        drawdown_worst: The deepest seen.
        observed_day_share: Largest share of total profit one day contributed
            **in the unresampled, as-it-happened ordering**, or ``None`` when
            the observed run never profited. Reported beside the resampled
            figures because the permutation moves outcomes between days, so the
            simulated consistency is a distribution and this is the one draw
            that actually occurred.
        blocked_trades_median: Median trades skipped per attempt because the
            daily limit had already bound. What the rule cost in participation.
        n_trades: Trades in the sample the attempts were drawn from.
    """

    rules_name: str
    iterations: int
    p_pass: float
    p_ruin: float
    p_exhausted: float
    se_pass: float
    mean_trading_days: float
    mean_calendar_days_to_pass: float | None
    drawdown_median: float
    drawdown_p95: float
    drawdown_worst: float
    observed_day_share: float | None
    blocked_trades_median: float
    n_trades: int

    @property
    def unlikely(self) -> bool:
        """Whether the odds of passing are below the threshold worth saying plainly."""
        return self.p_pass < UNLIKELY_THRESHOLD

    def summary(self) -> str:
        """One line, stating a poor result as a poor result.

        Returns:
            The sentence. Below :data:`UNLIKELY_THRESHOLD` it says so outright
            rather than reporting the number and leaving the reader to notice —
            a simulation whose whole purpose is to answer "would this survive"
            should not need interpreting when the answer is no.
        """
        head = (
            f"{self.rules_name}: P(pass) {self.p_pass:.1%} (±{self.se_pass:.1%}), "
            f"P(ruin) {self.p_ruin:.1%} over {self.iterations} attempts on {self.n_trades} trades"
        )
        if self.unlikely:
            return (
                f"{head} — below {UNLIKELY_THRESHOLD:.0%}. This strategy does not pass this "
                "account. Not marginal, not promising: it fails."
            )
        return head


def simulate(
    sample: PropSample,
    rules: PropRules,
    *,
    iterations: int = REPORT_ITERATIONS,
    seed: int = 0,
    buffer: float = DEFAULT_BUFFER,
) -> PropSimulation:
    """Run many block-permuted attempts at one firm's account.

    Args:
        sample: The strategy's out-of-sample trades, grouped by fold.
        rules: The firm's plan.
        iterations: How many attempts to simulate.
        seed: Seed for the permutation stream. Callers optimising over
            parameters should hold this **fixed across every candidate** — see
            :mod:`trading_system.prop.objective` on common random numbers.
        buffer: Share of the remaining allowance a trade may use.

    Returns:
        The simulation.

    Raises:
        ValueError: If the sample holds no trades, or ``iterations`` is not
            positive.
    """
    if sample.n_trades == 0:
        raise ValueError("cannot simulate an account from a sample with no trades")
    if iterations <= 0:
        raise ValueError(f"iterations must be positive, got {iterations}")

    rng = Random(seed)
    episodes = [
        run_episode(sample.draw(rng), rules, risk_pct=sample.risk_pct, buffer=buffer)
        for _ in range(iterations)
    ]

    passed = [episode for episode in episodes if episode.outcome is Outcome.PASSED]
    ruined = sum(1 for episode in episodes if episode.outcome is Outcome.RUINED)
    p_pass = len(passed) / iterations
    drawdowns = sorted(episode.max_drawdown for episode in episodes)
    blocked = sorted(float(episode.blocked_trades) for episode in episodes)

    observed = run_episode(sample.observed(), rules, risk_pct=sample.risk_pct, buffer=buffer)

    return PropSimulation(
        rules_name=rules.name,
        iterations=iterations,
        p_pass=p_pass,
        p_ruin=ruined / iterations,
        p_exhausted=1.0 - p_pass - ruined / iterations,
        se_pass=(p_pass * (1 - p_pass) / iterations) ** 0.5,
        mean_trading_days=sum(episode.trading_days for episode in episodes) / iterations,
        mean_calendar_days_to_pass=(
            sum(episode.calendar_days for episode in passed) / len(passed) if passed else None
        ),
        drawdown_median=_percentile(drawdowns, 50),
        drawdown_p95=_percentile(drawdowns, 95),
        drawdown_worst=drawdowns[-1],
        observed_day_share=observed.best_day_share,
        blocked_trades_median=_percentile(blocked, 50),
        n_trades=sample.n_trades,
    )


def simulate_all(
    sample: PropSample,
    library: Mapping[str, PropRules],
    *,
    iterations: int = REPORT_ITERATIONS,
    seed: int = 0,
) -> dict[str, PropSimulation]:
    """One simulation per rule set, over the same sample and the same seed.

    The shared seed is deliberate: two plans compared on different permutation
    draws would differ by simulation noise as well as by their rules, and the
    whole point of running several is to see what the rules alone change.

    Args:
        sample: The strategy's trades.
        library: Rule sets by name.
        iterations: How many attempts per plan.
        seed: Shared permutation seed.

    Returns:
        One simulation per plan, keyed by name.
    """
    return {
        name: simulate(sample, rules, iterations=iterations, seed=seed)
        for name, rules in library.items()
    }
