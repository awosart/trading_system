"""Scoring a parameter set for a prop account: pass the challenge, do not blow it.

The ordinary objective (:class:`~trading_system.validation.objective.SortinoTimesSqrtTrades`)
maximises a risk-adjusted return. A prop account asks something else entirely —
the probability of reaching a target before hitting a floor — and it is not the
same question with a penalty added, because the two disagree about what a good
result is: a smooth, slowly compounding curve scores well on Sortino and may
never touch a +10% target inside the trades available.

**Three levels, lexicographic, never a weighted sum.** A sum
``P(pass) − λ·P(ruin)`` needs ``λ``: an exchange rate between the chance of
passing and the chance of losing the account, which nobody can derive and
everybody would tune. The ranked form has no such number. Its parameters are
two ceilings, and a ceiling is a policy someone states rather than a knob
someone turns:

===========================  ===================================  ==============
band                         score                                range
===========================  ===================================  ==============
feasible                     ``P(pass)``                          ``[0, 1]``
fails the edge test          ``EDGE_BASE − (thr − pct)/100``      ``[-1.95, -1)``
fails the ruin ceiling       ``RUIN_BASE − (P(ruin) − ceiling)``  ``[-4, -3)``
===========================  ===================================  ==============

Every feasible point outranks every infeasible one, and among infeasible ones
the ranking is by how badly the constraint is missed, so a search still has a
gradient back into the feasible region rather than a flat plateau of equal
failures. Ruin sits below the edge failure because losing the account is a
worse outcome than not having an edge, and because a search should climb out of
the ruin region first.

**Why the edge constraint is here and not a convention.** Measured on this
repository's own four strategies: ``volume-thrust-h4`` scores ``P(pass) = 0.96``
under FTMO swing rules at 1% risk while its verdict is ``OVERFIT`` — 60th
percentile against the permutation null, which is to say indistinguishable from
a structureless tape. And ``channel-breakout-h4``'s ``P(pass)`` rises from 3.0%
at 1% risk to 19.1% at 2%, while its ``P(ruin)`` rises from 0.4% to 80.9%. Both
are the same defect: with no edge, the way to touch a target before a floor is
to be *more volatile*, and an unconstrained ``P(pass)`` rewards exactly that.
The fix could have been a rule that this objective is only ever applied to
strategies that already cleared their nulls — but a rule is an agreement, and
an agreement is forgotten. A constraint inside the function cannot be, and it
removes an ordering dependency (verdict first, then search) that would
otherwise have to hold and be remembered.

**The threshold is the verdict's own 95, not a second number.**
:attr:`~trading_system.validation.report.VerdictThresholds.null_percentile` is
imported rather than restated: a second authority on "does this have an edge"
is free to disagree with the first, and this project has spent several stages
removing exactly that. It also lands cleanly on the measurement grid —
:func:`~trading_system.validation.calibration._percentile_of` returns
``100 × count/N``, so at ``N=20`` the resolution is 5 points and 95 is
``19/20`` exactly, requiring no finer resolution than the estimate has. The
consequence is worth stating plainly rather than engineering around: no
strategy currently in the library clears it, so this objective currently rates
every one of them infeasible. That is the correct answer, not a problem.

**The null percentile is computed once per spec, never per trial.** It costs
``N`` full backtests — at ``N=20`` and the 0.104 s per trial P15 measured, 2.08 s
against a trial's own 0.104 s, a **20x** increase that would turn a 4400-trial
grid from 7.6 minutes into about 2.7 hours. Per-trial percentiles would also be
resolving differences below their own 5-point step. So the percentile is a
property of the *template* spec, measured before the search and constant across
its trials, and the edge band therefore admits or rejects a whole fold's search
rather than ranking inside it. That is what a gate is, and it is stated here so
nobody reads the flat band as a bug.

**Simulation noise is killed with common random numbers, not averaged away.**
Two calls with different seeds give different ``P(pass)``, and an objective that
moves under its own input makes :func:`~trading_system.validation.objective.roughness`
and the plateau analysis meaningless — both are built on *differences* between
neighbouring points. :attr:`PropObjective.seed` is fixed across every candidate
of a fold, so every trial is scored against the same permutation schedule and
the Monte Carlo error is common-mode, cancelling in exactly those differences.
The same discipline :mod:`trading_system.execution.rng` applies to fills: a
value must not depend on how many draws came before it. The residue this leaves
is named rather than hidden — trials have different trade counts, so their
schedules cannot be identical index for index, and the cancellation is
substantial rather than total. :attr:`~trading_system.prop.simulator.PropSimulation.se_pass`
travels with every estimate so a plateau tolerance finer than the simulator's
own width can be spotted instead of believed.
"""

import math
from dataclasses import dataclass, field

from trading_system.backtest.orchestrator import BacktestResult
from trading_system.prop.rules import PropRules
from trading_system.prop.simulator import (
    SEARCH_ITERATIONS,
    PropSimulation,
    sample_from_trades,
    simulate,
)
from trading_system.validation.report import VerdictThresholds

#: Top of the band holding points that fail the edge test. Scores land in
#: ``[EDGE_BASE - 0.95, EDGE_BASE)``, strictly below every feasible score
#: (``P(pass) >= 0``) and strictly above every ruin failure.
EDGE_BASE = -1.0

#: Top of the band holding points that breach the ruin ceiling. Scores land in
#: ``[RUIN_BASE - 1, RUIN_BASE)``, below the whole edge band, whose floor is
#: ``EDGE_BASE - 0.95 = -1.95``.
RUIN_BASE = -3.0

#: Largest probability of losing the account a candidate may carry and still be
#: considered at all. Five per cent: a challenge fee is a purchase, and one
#: attempt in twenty ending the account is the most a search should be allowed
#: to trade away for a better chance of passing.
DEFAULT_RUIN_CEILING = 0.05


@dataclass(frozen=True)
class PropScore:
    """One candidate's score, with the band it landed in and why.

    Attributes:
        value: The lexicographic score. Higher is better.
        feasible: Whether both constraints were met.
        p_pass: Probability of passing the account.
        p_ruin: Probability of losing it.
        se_pass: Standard error of ``p_pass``.
        null_percentile: The spec's percentile against the permutation null,
            as supplied.
        detail: What decided the band, in plain language.
    """

    value: float
    feasible: bool
    p_pass: float
    p_ruin: float
    se_pass: float
    null_percentile: float | None
    detail: str


@dataclass(frozen=True)
class PropObjective:
    """Probability of passing a prop account, gated on ruin and on edge.

    Attributes:
        rules: The firm's plan.
        risk_pct: Equity fraction risked per trade — the run's own sizing
            parameter, needed to turn R multiples back into equity moves.
        null_percentile: The template spec's percentile against the permutation
            null, measured once before the search. ``None`` means it was never
            measured, which is treated as *failing* the edge test rather than
            passing it: an unmeasured edge is not an edge, and defaulting the
            other way would let the whole constraint be skipped by omission.
        ruin_ceiling: Largest tolerable probability of losing the account.
        edge_threshold: Percentile the null comparison must reach. The
            verdict's own figure by default, never a second number.
        iterations: Simulated attempts per score.
        seed: Permutation seed, **held fixed across every candidate** — see the
            module docstring on common random numbers.
    """

    rules: PropRules
    risk_pct: float
    null_percentile: float | None
    ruin_ceiling: float = DEFAULT_RUIN_CEILING
    edge_threshold: float = field(default_factory=lambda: VerdictThresholds().null_percentile)
    iterations: int = SEARCH_ITERATIONS
    seed: int = 0

    def __post_init__(self) -> None:
        """Validate the ceilings.

        Raises:
            ValueError: If the ruin ceiling is outside ``[0, 1]`` or the edge
                threshold outside ``[0, 100]``.
        """
        if not 0 <= self.ruin_ceiling <= 1:
            raise ValueError(f"ruin_ceiling must be in [0, 1], got {self.ruin_ceiling}")
        if not 0 <= self.edge_threshold <= 100:
            raise ValueError(f"edge_threshold must be in [0, 100], got {self.edge_threshold}")

    def score(self, result: BacktestResult) -> float:
        """Score one run, lexicographically across the three bands.

        Args:
            result: What the run produced.

        Returns:
            The score. Higher is better.

        Raises:
            ValueError: If the run closed no trades — the same refusal
                :class:`~trading_system.validation.objective.SortinoTimesSqrtTrades`
                makes, and for the same reason: a caller running many
                iterations decides for itself what an unscoreable one means.
        """
        return self.evaluate(result).value

    def evaluate(self, result: BacktestResult) -> PropScore:
        """Score one run and say which band it fell in.

        Args:
            result: What the run produced.

        Returns:
            The score and its reasoning.

        Raises:
            ValueError: If the run closed no trades.
        """
        if not result.trades:
            raise ValueError("cannot score a run with zero trades")
        simulation = self.simulate(result)
        return self._band(simulation)

    def simulate(self, result: BacktestResult) -> PropSimulation:
        """Run this objective's own simulation over a result's trades.

        Exposed so a caller can report the distribution behind a score rather
        than only the score.

        Args:
            result: What the run produced.

        Returns:
            The simulation.
        """
        sample = sample_from_trades(result.trades, risk_pct=self.risk_pct)
        return simulate(sample, self.rules, iterations=self.iterations, seed=self.seed)

    def _band(self, simulation: PropSimulation) -> PropScore:
        """Place a simulation in the ranked bands."""
        if simulation.p_ruin > self.ruin_ceiling:
            excess = simulation.p_ruin - self.ruin_ceiling
            return PropScore(
                value=RUIN_BASE - excess,
                feasible=False,
                p_pass=simulation.p_pass,
                p_ruin=simulation.p_ruin,
                se_pass=simulation.se_pass,
                null_percentile=self.null_percentile,
                detail=(
                    f"P(ruin) {simulation.p_ruin:.1%} exceeds the {self.ruin_ceiling:.1%} "
                    f"ceiling by {excess:.1%}; P(pass) {simulation.p_pass:.1%} is not "
                    "considered while the account is this likely to be lost"
                ),
            )

        percentile = self.null_percentile
        if percentile is None or percentile < self.edge_threshold:
            measured = 0.0 if percentile is None else percentile
            shortfall = (self.edge_threshold - measured) / 100.0
            unmeasured = (
                " (never measured, which counts as failing rather than passing)"
                if percentile is None
                else ""
            )
            return PropScore(
                value=EDGE_BASE - shortfall,
                feasible=False,
                p_pass=simulation.p_pass,
                p_ruin=simulation.p_ruin,
                se_pass=simulation.se_pass,
                null_percentile=percentile,
                detail=(
                    f"percentile {measured:.1f} against the permutation null is below the "
                    f"{self.edge_threshold:.1f} required{unmeasured}; P(pass) "
                    f"{simulation.p_pass:.1%} is a statement about path shape, not about edge, "
                    "and is not considered"
                ),
            )

        return PropScore(
            value=simulation.p_pass,
            feasible=True,
            p_pass=simulation.p_pass,
            p_ruin=simulation.p_ruin,
            se_pass=simulation.se_pass,
            null_percentile=percentile,
            detail=(
                f"P(pass) {simulation.p_pass:.1%} (±{simulation.se_pass:.1%}) with P(ruin) "
                f"{simulation.p_ruin:.1%} inside the {self.ruin_ceiling:.1%} ceiling and "
                f"percentile {percentile:.1f} clearing {self.edge_threshold:.1f}"
            ),
        )


def band_of(score: float) -> str:
    """Which band a score fell in, for a reader looking at a number alone.

    Args:
        score: A value :meth:`PropObjective.score` returned.

    Returns:
        ``"feasible"``, ``"edge"`` or ``"ruin"``.
    """
    if score >= 0.0:
        return "feasible"
    if score > RUIN_BASE:
        return "edge"
    return "ruin"


def minimum_tolerance(simulation: PropSimulation, *, sigmas: float = 3.0) -> float:
    """Smallest plateau tolerance this simulation can actually resolve.

    A plateau declared with a tolerance below the simulator's own standard
    error is measuring the permutation draw rather than the strategy. This is
    what a caller compares its configured tolerance against.

    Args:
        simulation: The simulation behind a score.
        sigmas: How many standard errors to demand. Three, so that neighbouring
            points called equal really are within noise rather than one draw
            apart.

    Returns:
        The floor, in score units.
    """
    return sigmas * simulation.se_pass


def iterations_for(target_se: float) -> int:
    """How many iterations reach a given standard error, worst case.

    The standard error of a proportion peaks at ``p = 0.5``, where it is
    ``0.5/sqrt(n)`` — so this sizes for the hardest case rather than the one
    that happened to occur.

    Args:
        target_se: Desired standard error.

    Returns:
        Iterations needed.

    Raises:
        ValueError: If ``target_se`` is not positive.
    """
    if target_se <= 0:
        raise ValueError(f"target_se must be positive, got {target_se}")
    return math.ceil(0.25 / target_se**2)
