"""The firm's day boundary across DST, and the episode machinery over it.

The reset tests are the reason ``daily_reset_tz`` is an IANA zone rather than a
stored offset: 00:00 Prague is 23:00 UTC in winter and 22:00 UTC in summer, and
a fixed offset is right for half the year. Europe and Israel change clocks on
different dates, which is why both shipped zones are exercised rather than one
standing in for the other.
"""

from datetime import UTC, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path
from random import Random

import pytest

from trading_system.data.resample import trading_day
from trading_system.prop.guard import PropGuard
from trading_system.prop.rules import DailyLossBasis, PropRules, TotalLossBasis
from trading_system.prop.simulator import (
    Outcome,
    PropFold,
    PropSample,
    run_episode,
    sample_from_trades,
    simulate,
)
from trading_system.validation.monte_carlo import BlockPermutation, FoldTrades, TradeSample

RULES_PATH = Path(__file__).resolve().parents[2] / "configs" / "prop_rules.yaml"


def rules(**overrides: object) -> PropRules:
    """A rule set with FTMO-like defaults, overridable field by field."""
    base: dict[str, object] = {
        "name": "test-plan",
        "prop_profile": "ftmo_swing",
        "source": "test",
        "account_size": Decimal("100000"),
        "profit_target_pct": 0.10,
        "max_daily_loss_pct": 0.05,
        "daily_loss_basis": DailyLossBasis.BALANCE_AT_DAY_START,
        "daily_reset_time": time(0, 0),
        "daily_reset_tz": "Europe/Prague",
        "max_total_loss_pct": 0.10,
        "total_loss_basis": TotalLossBasis.STATIC,
        "max_single_day_profit_share": 0.5,
        "min_trading_days": 0,
    }
    base.update(overrides)
    return PropRules.model_validate(base)


class TestTheFirmDayCrossesDST:
    """The boundary moves in UTC twice a year, and the label stays right."""

    def test_prague_midnight_is_2300_utc_in_winter(self) -> None:
        guard = PropGuard(rules(daily_reset_tz="Europe/Prague"))
        # 22:59 UTC on 15 January is still 23:59 Prague on the 15th.
        assert guard.trading_day_of(datetime(2024, 1, 15, 22, 59, tzinfo=UTC)).day == 15
        # 23:00 UTC is 00:00 Prague on the 16th — a new firm-day.
        assert guard.trading_day_of(datetime(2024, 1, 15, 23, 0, tzinfo=UTC)).day == 16

    def test_prague_midnight_is_2200_utc_in_summer(self) -> None:
        guard = PropGuard(rules(daily_reset_tz="Europe/Prague"))
        # The same wall-clock reset is an hour earlier in UTC once CEST starts.
        assert guard.trading_day_of(datetime(2024, 7, 15, 21, 59, tzinfo=UTC)).day == 15
        assert guard.trading_day_of(datetime(2024, 7, 15, 22, 0, tzinfo=UTC)).day == 16

    def test_a_fixed_offset_would_have_been_wrong_for_half_the_year(self) -> None:
        """The concrete instant a stored offset mislabels.

        22:30 UTC is 00:30 Prague in summer (already the next firm-day) and
        23:30 Prague in winter (still the current one). One stored offset
        cannot label both correctly; an IANA zone labels both.
        """
        guard = PropGuard(rules(daily_reset_tz="Europe/Prague"))
        summer = guard.trading_day_of(datetime(2024, 7, 10, 22, 30, tzinfo=UTC))
        winter = guard.trading_day_of(datetime(2024, 1, 10, 22, 30, tzinfo=UTC))
        assert summer.day == 11
        assert winter.day == 10

    def test_israel_changes_clocks_on_its_own_dates(self) -> None:
        """The5ers' zone is not Europe's, and the difference is observable.

        Israel left DST on 2024-10-27 and central Europe on 2024-10-27 too, but
        Israel *entered* it on 2024-03-29 against Europe's 2024-03-31. On the
        30th of March the two zones are an hour apart in their reset instants.
        """
        israel = PropGuard(rules(daily_reset_tz="Asia/Jerusalem"))
        prague = PropGuard(rules(daily_reset_tz="Europe/Prague"))
        instant = datetime(2024, 3, 30, 22, 30, tzinfo=UTC)
        # 22:30 UTC is 01:30 Israel (already the 31st) and 23:30 Prague (still
        # the 30th).
        assert israel.trading_day_of(instant).day == 31
        assert prague.trading_day_of(instant).day == 30

    def test_the_spring_forward_day_is_23_hours_and_nothing_breaks(self) -> None:
        """02:00-03:00 does not exist locally, and the labelling never builds it."""
        guard = PropGuard(rules(daily_reset_tz="Europe/Prague", daily_reset_time=time(2, 30)))
        # Across the whole transition day every instant still maps to a day.
        start = datetime(2024, 3, 31, 0, 0, tzinfo=UTC)
        labels = {
            guard.trading_day_of(start + timedelta(minutes=15 * step)) for step in range(4 * 26)
        }
        assert labels, "every instant on a spring-forward day must still get a label"

    def test_the_guard_uses_the_one_shared_day_function(self) -> None:
        """Not a second definition of 'what day is it' — the same one, another origin."""
        firm = rules(daily_reset_tz="Asia/Jerusalem", daily_reset_time=time(0, 0))
        guard = PropGuard(firm)
        instant = datetime(2024, 6, 11, 3, 0, tzinfo=UTC)
        assert guard.trading_day_of(instant) == trading_day(instant, firm.day_origin)


def _sequence(days_and_rs: list[tuple[int, float]]) -> list[tuple[datetime, float]]:
    """``(day-of-June-2024, r)`` pairs as dated outcomes, safely inside one zone's day."""
    return [(datetime(2024, 6, day, 12, 0, tzinfo=UTC), r) for day, r in days_and_rs]


class TestEpisodeRules:
    def test_reaching_the_target_passes(self) -> None:
        # 1% risk, +10 R spread over four days: +10% before compounding.
        outcome = run_episode(
            _sequence([(3, 3.0), (4, 3.0), (5, 3.0), (6, 3.0)]),
            rules(min_trading_days=0, max_single_day_profit_share=None),
            risk_pct=0.01,
        )
        assert outcome.outcome is Outcome.PASSED

    def test_min_trading_days_delays_the_pass_rather_than_denying_it(self) -> None:
        """DoD (a): the rule modifies the stopping condition, it is not a veto."""
        # The whole +12% arrives on one day. With no day minimum that passes.
        one_day = _sequence([(3, 12.0)])
        assert (
            run_episode(
                one_day, rules(min_trading_days=0, max_single_day_profit_share=None), risk_pct=0.01
            ).outcome
            is Outcome.PASSED
        )
        # With a four-day minimum the same sequence does not, and the account
        # is neither passed nor lost — it simply runs out of trades.
        assert (
            run_episode(
                one_day, rules(min_trading_days=4, max_single_day_profit_share=None), risk_pct=0.01
            ).outcome
            is Outcome.EXHAUSTED
        )

    def test_consistency_denies_a_profitable_sequence(self) -> None:
        """DoD: a strategy that makes everything in one day does not pass."""
        # +12% on day 3, then a scratch on three more days to satisfy the day
        # count. One day still holds ~100% of the profit.
        sequence = _sequence([(3, 12.0), (4, 0.0), (5, 0.0), (6, 0.0)])
        strict = rules(min_trading_days=4, max_single_day_profit_share=0.5)
        assert run_episode(sequence, strict, risk_pct=0.01).outcome is Outcome.EXHAUSTED
        relaxed = rules(min_trading_days=4, max_single_day_profit_share=None)
        assert run_episode(sequence, relaxed, risk_pct=0.01).outcome is Outcome.PASSED

    def test_spreading_the_same_profit_over_days_passes_the_consistency_rule(self) -> None:
        sequence = _sequence([(3, 3.0), (4, 3.0), (5, 3.0), (6, 3.0)])
        strict = rules(min_trading_days=4, max_single_day_profit_share=0.5)
        result = run_episode(sequence, strict, risk_pct=0.01)
        assert result.outcome is Outcome.PASSED
        assert result.best_day_share is not None
        assert result.best_day_share <= 0.5

    def test_the_daily_limit_blocks_the_rest_of_the_day_and_the_next_day_resumes(self) -> None:
        # Two big losses on day 3 take equity through the 5% daily floor; the
        # third trade that day is blocked, and day 4 trades again.
        sequence = _sequence([(3, -3.0), (3, -3.0), (3, -3.0), (4, 1.0)])
        result = run_episode(sequence, rules(min_trading_days=0), risk_pct=0.01, buffer=1.0)
        assert result.blocked_trades >= 1
        assert result.trading_days == 2

    def test_one_outcome_large_enough_to_jump_the_floor_is_ruin(self) -> None:
        """Ruin needs a jump, not a slide — which is the documented caveat.

        A single -12 R outcome at 1% risk takes equity from 1.00 to 0.88 in one
        step, past a floor at 0.90 that nothing had a chance to stop short of.
        This is the only shape of ruin the simulator can produce, and it is the
        shape a real account dies in: a gap through the stop on a position that
        was already open. Declining to open the *next* trade never prevents it.
        """
        result = run_episode(_sequence([(3, -12.0)]), rules(min_trading_days=0), risk_pct=0.01)
        assert result.outcome is Outcome.RUINED

    def test_a_slide_towards_the_floor_is_stopped_short_of_it_instead(self) -> None:
        """The measured difference between the prototype and the real guard.

        Twelve separate 1% losses would compound to 0.886, through the same
        0.90 floor — but the buffered allowance refuses the last trades once
        the remaining headroom no longer covers the risk, so the account
        stagnates at 0.91 rather than being lost. Removing this check is what
        took ``ema-pullback-h1`` from P(ruin) 0% to 100% in the prototype.
        """
        sequence = _sequence([(3 + i, -1.0) for i in range(12)])
        result = run_episode(sequence, rules(min_trading_days=0), risk_pct=0.01)
        assert result.outcome is Outcome.EXHAUSTED
        assert result.blocked_trades > 0
        assert result.final_equity > 0.90


class TestResamplingUnitMatchesP15:
    def test_a_draw_is_a_within_fold_permutation_of_the_same_multiset(self) -> None:
        """The unit is P15's block, asserted rather than claimed in prose."""
        folds = [
            PropFold(
                fold_index=index,
                closed_at=tuple(
                    datetime(2024, 6, 1, 12, tzinfo=UTC) + timedelta(days=index * 10 + step)
                    for step in range(5)
                ),
                r_multiples=tuple(float(index * 10 + step) for step in range(5)),
            )
            for index in range(3)
        ]
        prop_sample = PropSample(folds=tuple(folds), risk_pct=0.01)
        mc_sample = TradeSample(
            folds=tuple(
                FoldTrades(fold_index=fold.fold_index, r_multiples=fold.r_multiples)
                for fold in folds
            ),
            risk_pct=0.01,
        )

        drawn = [r for _, r in prop_sample.draw(Random(7))]
        mc_drawn = list(BlockPermutation().draw(mc_sample, Random(7)))

        offset = 0
        for fold in folds:
            width = len(fold)
            assert sorted(drawn[offset : offset + width]) == sorted(fold.r_multiples)
            assert sorted(mc_drawn[offset : offset + width]) == sorted(fold.r_multiples)
            offset += width

    def test_the_dates_never_move(self) -> None:
        fold = PropFold(
            fold_index=0,
            closed_at=tuple(
                datetime(2024, 6, 1, 12, tzinfo=UTC) + timedelta(days=step) for step in range(9)
            ),
            r_multiples=tuple(float(step) for step in range(9)),
        )
        sample = PropSample(folds=(fold,), risk_pct=0.01)
        for seed in range(5):
            assert [when for when, _ in sample.draw(Random(seed))] == list(fold.closed_at)

    def test_the_same_seed_gives_the_same_draw(self) -> None:
        fold = PropFold(
            fold_index=0,
            closed_at=tuple(
                datetime(2024, 6, 1, 12, tzinfo=UTC) + timedelta(days=step) for step in range(9)
            ),
            r_multiples=tuple(float(step) for step in range(9)),
        )
        sample = PropSample(folds=(fold,), risk_pct=0.01)
        assert sample.draw(Random(3)) == sample.draw(Random(3))


class TestSimulate:
    def test_the_same_seed_reproduces_the_whole_simulation(self) -> None:
        sample = _demo_sample()
        first = simulate(sample, rules(), iterations=200, seed=11)
        second = simulate(sample, rules(), iterations=200, seed=11)
        assert first == second

    def test_probabilities_sum_to_one(self) -> None:
        result = simulate(_demo_sample(), rules(), iterations=300, seed=1)
        assert result.p_pass + result.p_ruin + result.p_exhausted == pytest.approx(1.0)

    def test_a_hopeless_strategy_says_so_plainly(self) -> None:
        losing = PropSample(
            folds=(
                PropFold(
                    fold_index=0,
                    closed_at=tuple(
                        datetime(2024, 6, 1, 12, tzinfo=UTC) + timedelta(days=step)
                        for step in range(40)
                    ),
                    r_multiples=tuple(-1.0 for _ in range(40)),
                ),
            ),
            risk_pct=0.01,
        )
        result = simulate(losing, rules(), iterations=100, seed=0)
        assert result.p_pass == 0.0
        assert result.unlikely
        assert "it fails" in result.summary()

    def test_an_empty_sample_is_refused_rather_than_scored_as_zero(self) -> None:
        empty = PropSample(
            folds=(PropFold(fold_index=0, closed_at=(), r_multiples=()),), risk_pct=0.01
        )
        with pytest.raises(ValueError, match="no trades"):
            simulate(empty, rules(), iterations=10)

    def test_standard_error_shrinks_with_iterations(self) -> None:
        sample = _demo_sample()
        few = simulate(sample, rules(), iterations=100, seed=2)
        many = simulate(sample, rules(), iterations=2000, seed=2)
        if 0 < few.p_pass < 1:
            assert many.se_pass < few.se_pass


def _demo_sample() -> PropSample:
    """A mixed win/loss sample spread over enough days to reach a target."""
    outcomes = [2.0, -1.0, 1.5, -1.0, 3.0, -1.0, 2.5, -1.0, 1.0, 2.0] * 6
    return PropSample(
        folds=(
            PropFold(
                fold_index=0,
                closed_at=tuple(
                    datetime(2024, 6, 1, 12, tzinfo=UTC) + timedelta(days=step)
                    for step in range(len(outcomes))
                ),
                r_multiples=tuple(outcomes),
            ),
        ),
        risk_pct=0.01,
    )


class TestSampleFromTrades:
    def test_trades_are_ordered_by_close_time(self) -> None:
        from tests.analytics.conftest import trade

        base = datetime(2024, 6, 1, 12, tzinfo=UTC)
        out_of_order = [
            trade(opened_at=base, closed_at=base + timedelta(days=3), net=1.0, realized_r=1.0),
            trade(opened_at=base, closed_at=base + timedelta(days=1), net=-1.0, realized_r=-1.0),
        ]
        sample = sample_from_trades(out_of_order, risk_pct=0.01)
        assert sample.folds[0].closed_at == (base + timedelta(days=1), base + timedelta(days=3))
        assert sample.folds[0].r_multiples == (-1.0, 1.0)
