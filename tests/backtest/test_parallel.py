"""Parallel and sequential must be the same runs, and the list order must not matter.

Both claims are about determinism rather than speed. The first says a process
pool changes nothing; the second says the *position of a run in the list*
changes nothing either — the same statement the harness makes about a second
instrument in a portfolio, at the level above it, and the one that would break
first if anything in a run consulted a shared generator or a wall clock.

Results are compared through :func:`~trading_system.backtest.reproducibility.result_digest`,
so "the same" means every equity row, every trade and every counter, not a
matching trade count.
"""

from decimal import Decimal

import pytest

from tests.backtest.conftest import (
    EURUSD_H4,
    LIBRARY_PATH,
    XAUUSD_H4,
    ema_pullback,
    harness_inputs,
    noisy_costs,
    swing_series,
)
from trading_system.backtest.orchestrator import StrategyBinding
from trading_system.backtest.parallel import run_parallel, run_sequential
from trading_system.backtest.reproducibility import result_digest
from trading_system.backtest.spec import RunInputs
from trading_system.core.instruments import InstrumentRegistry
from trading_system.exit.library import ExitLibrarySpec
from trading_system.risk.sizing.methods import FixedFractional

#: Bars per run. Short: this module is about identity, not throughput, and the
#: speed comparison belongs in a benchmark rather than in a correctness test
#: whose result would then depend on how busy the machine is.
LENGTH = 700

#: How many independent runs. Four so that a four-way pool has one each and the
#: interleaving is real rather than nominal.
RUNS = 4


@pytest.fixture(scope="module")
def specs(registry: InstrumentRegistry) -> list[RunInputs]:
    """Four runs that differ from each other in a way the result can show.

    Different phases of the series, different risk fractions and — for one of
    them — a different instrument, so that comparing the four to each other is
    not comparing four copies. If they were identical, every assertion below
    would hold for a runner that ignored its input and returned the first
    result four times.
    """
    library = ExitLibrarySpec.model_validate_json(LIBRARY_PATH.read_text())
    eurusd = ema_pullback()
    gold = ema_pullback(strategy_id="ema-pullback-gold", asset_class="COMMODITY", symbol="XAUUSD")
    preset = next(item for item in library.presets if item.id == eurusd.exit_ref)
    built: list[RunInputs] = []
    for index in range(RUNS - 1):
        built.append(
            harness_inputs(
                registry,
                streams={EURUSD_H4: swing_series(LENGTH, phase=0.4 * index)},
                bindings=[StrategyBinding(spec=eurusd, exit_preset=preset, keys=(EURUSD_H4,))],
                costs=noisy_costs(seed=11 + index),
                sizing=FixedFractional(risk_pct=0.005 + 0.002 * index),
            )
        )
    built.append(
        harness_inputs(
            registry,
            streams={
                XAUUSD_H4: swing_series(
                    LENGTH, symbol="XAUUSD", base=2000.0, scale=2000.0, phase=1.6
                )
            },
            bindings=[StrategyBinding(spec=gold, exit_preset=preset, keys=(XAUUSD_H4,))],
        )
    )
    return built


@pytest.fixture(scope="module")
def expected(specs: list[RunInputs]) -> list[str]:
    """The digest of each run, computed one run at a time in this process."""
    return [result_digest(result) for result in run_sequential(specs)]


class TestParallelMatchesSequential:
    """A process pool must not be able to change what a run produced."""

    def test_every_run_hashes_to_what_it_hashed_to_sequentially(
        self, specs: list[RunInputs], expected: list[str]
    ) -> None:
        """Digest for digest, in order."""
        results = run_parallel(specs, workers=RUNS)

        assert [result_digest(result) for result in results] == expected

    def test_the_number_of_workers_is_not_an_input_to_the_result(
        self, specs: list[RunInputs], expected: list[str]
    ) -> None:
        """Two workers and four must produce the same four runs.

        A run whose output depended on the pool size would be one holding state
        somewhere outside itself — the class of defect that cannot be found by
        rerunning the same configuration.
        """
        results = run_parallel(specs, workers=2)

        assert [result_digest(result) for result in results] == expected

    def test_the_runs_differ_from_one_another(self, expected: list[str]) -> None:
        """Otherwise every comparison above would hold for a runner that lied."""
        assert len(set(expected)) == len(expected)

    def test_the_runs_are_not_empty(self, specs: list[RunInputs]) -> None:
        """Four runs that traded nothing would agree about nothing."""
        for result in run_sequential(specs):
            assert result.trades
            assert result.fills


class TestOrderIsNotAnInput:
    """Where a run sits in the list must not change what the run produces."""

    def test_reversing_the_list_leaves_each_runs_result_alone(
        self, specs: list[RunInputs], expected: list[str]
    ) -> None:
        """Same runs, opposite order, same digests once paired back up."""
        results = run_parallel(list(reversed(specs)), workers=RUNS)

        assert [result_digest(result) for result in results] == list(reversed(expected))

    def test_a_run_on_its_own_matches_the_same_run_in_a_batch(
        self, specs: list[RunInputs], expected: list[str]
    ) -> None:
        """The strongest form: a batch of one against a batch of four.

        A single-spec call takes :func:`run_parallel`'s sequential shortcut, so
        this also pins that the shortcut is the same computation and not a
        second implementation of it.
        """
        for index, inputs in enumerate(specs):
            assert result_digest(run_parallel([inputs])[0]) == expected[index]


class TestTheContract:
    """Small refusals and the empty case."""

    def test_no_specs_is_no_results(self) -> None:
        """An empty batch is a legitimate batch, not an error."""
        assert run_parallel([]) == []

    def test_zero_workers_is_refused(self, specs: list[RunInputs]) -> None:
        """A pool of nothing would silently mean "as many as you like"."""
        with pytest.raises(ValueError, match="at least 1"):
            run_parallel(specs, workers=0)

    def test_a_spec_survives_the_trip_to_a_worker(self, specs: list[RunInputs]) -> None:
        """Every field of a run must be picklable, including the sizing method.

        Checked through the pool rather than by calling ``pickle.dumps`` here:
        what matters is that the object arriving in the worker is the one that
        was sent, which a spawn-based pool proves and a local round trip does
        not.
        """
        results = run_parallel(specs[:2], workers=2)

        assert len(results) == 2
        assert all(result.curve for result in results)
        assert results[0].curve[0].balance == Decimal(100_000)
