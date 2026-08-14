"""Per-value null baselines: adaptive effort, honest stopping, and what they adjust.

The calibration is driven through a stub :class:`TrialRunner` rather than real
bars. What is under test is the stopping rule and the arithmetic of the
correction; that a permutation of real bars is a null is
:mod:`trading_system.validation.nulls.permutation`'s own business, tested there.
"""

import random
from dataclasses import dataclass
from typing import Any

import pytest

from trading_system.data.resample import FX_DAY_ORIGIN
from trading_system.validation.null_baseline import (
    BaselineCalibration,
    BaselineRequest,
    NullBaseline,
    adjusted_scores,
    calibrate_null_baselines,
    read_calibration,
    sole_categorical_axis,
    write_calibration,
)
from trading_system.validation.optimization import (
    AxisTarget,
    ParameterAxis,
    SearchSpace,
    TrialOutcome,
    TrialRecord,
)


def _space(*, numeric: bool = True) -> SearchSpace:
    axes: list[ParameterAxis] = [
        ParameterAxis(name="exit", target=AxisTarget.EXIT_PRESET, values=("tight", "loose", "dead"))
    ]
    if numeric:
        axes.append(ParameterAxis(name="period", paths=("/a",), values=(10, 20, 30)))
    return SearchSpace(axes=tuple(axes))


@dataclass
class _StubRunner:
    """Stands in for a TrialRunner: real score per value, noisy null per shuffle.

    ``spreads`` is what makes the adaptive rule observable — a value whose null
    scatters widely must draw more shuffles than one that barely moves, which is
    exactly the ten-to-one range the eight real presets showed.
    """

    reals: dict[str, float | None]
    spreads: dict[str, float]
    trades: dict[str, int]
    seed: int = 0
    calls: int = 0
    runs_per_trial: int = 1

    def _value(self, params: Any) -> str:
        return str(params.as_dict()["exit"])

    def evaluate(self, params: Any) -> TrialOutcome:
        self.calls += 1
        name = self._value(params)
        score = self.reals[name]
        return TrialOutcome(
            score=score,
            piece_scores=(score,),
            dispersion=None,
            n_trades=self.trades[name],
            returns=(),
            unscoreable=None if score is not None else "no trades",
        )

    def permuted(self, seed: int) -> "_StubRunner":
        clone = _StubRunner(reals=self.reals, spreads=self.spreads, trades=self.trades, seed=seed)
        clone.calls = 0
        return clone


def _install(monkeypatch: pytest.MonkeyPatch, base: _StubRunner) -> dict[str, int]:
    """Replace the permutation step with a deterministic pseudo-null.

    Counts shuffles per value so the adaptive rule can be asserted on.
    """
    drawn: dict[str, int] = {}

    class _Null(_StubRunner):
        def evaluate(self, params: Any) -> TrialOutcome:
            name = self._value(params)
            drawn[name] = drawn.get(name, 0) + 1
            rng = random.Random((self.seed, name).__hash__())
            spread = self.spreads[name]
            return TrialOutcome(
                score=rng.gauss(0.0, spread),
                piece_scores=(0.0,),
                dispersion=None,
                n_trades=self.trades[name],
                returns=(),
                unscoreable=None,
            )

    def fake_permuted(_runner: Any, seed: int, _day_origin: Any) -> Any:
        return _Null(reals=base.reals, spreads=base.spreads, trades=base.trades, seed=seed)

    monkeypatch.setattr("trading_system.validation.null_baseline._permuted", fake_permuted)
    return drawn


class TestOnlyOneCategoricalAxisIsCalibrated:
    """A baseline per combination is a different measurement from the one that was made."""

    def test_a_space_with_none_calibrates_nothing(self) -> None:
        space = SearchSpace(axes=(ParameterAxis(name="p", paths=("/a",), values=(1, 2)),))
        assert sole_categorical_axis(space) is None

    def test_two_categorical_axes_are_refused(self) -> None:
        space = SearchSpace(
            axes=(
                ParameterAxis(name="a", paths=("/a",), values=("x", "y")),
                ParameterAxis(name="b", paths=("/b",), values=("p", "q")),
            )
        )
        with pytest.raises(ValueError, match="one categorical axis"):
            sole_categorical_axis(space)

    def test_calibration_returns_none_when_there_is_nothing_to_calibrate(self) -> None:
        space = SearchSpace(axes=(ParameterAxis(name="p", paths=("/a",), values=(1, 2)),))
        runner = _StubRunner(reals={}, spreads={}, trades={})
        assert (
            calibrate_null_baselines(runner, space, day_origin=FX_DAY_ORIGIN) is None  # type: ignore[arg-type]
        )


class TestEffortFollowsEachValuesOwnNoise:
    """One seed count either overspends on the tight values or underspends on the loose."""

    def test_a_noisy_value_draws_more_shuffles_than_a_quiet_one(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        base = _StubRunner(
            reals={"tight": 1.0, "loose": 1.0, "dead": 1.0},
            spreads={"tight": 0.01, "loose": 0.5, "dead": 0.05},
            trades={"tight": 400, "loose": 90, "dead": 200},
        )
        drawn = _install(monkeypatch, base)
        calibration = calibrate_null_baselines(
            base,  # type: ignore[arg-type]
            _space(),
            day_origin=FX_DAY_ORIGIN,
            request=BaselineRequest(target_fraction=0.05, min_seeds=8, max_seeds=400),
        )
        assert calibration is not None
        assert drawn["loose"] > drawn["tight"]
        by_value = calibration.by_value()
        assert by_value["tight"].seeds < by_value["loose"].seeds

    def test_reaching_the_target_is_recorded_separately_from_running_out_of_budget(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        base = _StubRunner(
            reals={"tight": 1.0, "loose": 1.0, "dead": 1.0},
            spreads={"tight": 0.01, "loose": 5.0, "dead": 0.05},
            trades={"tight": 400, "loose": 90, "dead": 200},
        )
        _install(monkeypatch, base)
        calibration = calibrate_null_baselines(
            base,  # type: ignore[arg-type]
            _space(),
            day_origin=FX_DAY_ORIGIN,
            request=BaselineRequest(min_seeds=8, max_seeds=24),
        )
        assert calibration is not None
        by_value = calibration.by_value()
        assert by_value["tight"].stopped_because == "target"
        assert by_value["loose"].stopped_because == "budget"
        assert by_value["loose"].seeds == 24

    def test_an_unscoreable_configuration_says_so_rather_than_getting_a_zero(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        base = _StubRunner(
            reals={"tight": 1.0, "loose": 1.0, "dead": None},
            spreads={"tight": 0.01, "loose": 0.2, "dead": 0.05},
            trades={"tight": 400, "loose": 90, "dead": 0},
        )
        _install(monkeypatch, base)
        calibration = calibrate_null_baselines(
            base,  # type: ignore[arg-type]
            _space(),
            day_origin=FX_DAY_ORIGIN,
            request=BaselineRequest(min_seeds=4, max_seeds=32),
        )
        assert calibration is not None
        dead = calibration.by_value()["dead"]
        assert dead.stopped_because == "unscoreable"
        assert dead.excess is None
        assert dead.seeds == 0

    def test_the_request_validates_its_own_bounds(self) -> None:
        with pytest.raises(ValueError, match="target_fraction"):
            BaselineRequest(target_fraction=0.0)
        with pytest.raises(ValueError, match="min_seeds"):
            BaselineRequest(min_seeds=1)
        with pytest.raises(ValueError, match="below min_seeds"):
            BaselineRequest(min_seeds=8, max_seeds=4)


class TestTheCorrectionSubtractsEachValuesOwnZero:
    """The whole point: two exits are not comparable on a scale whose origins differ."""

    def _records(self) -> list[TrialRecord]:
        space = _space()
        records = []
        for point in space.enumerate():
            records.append(
                TrialRecord(
                    params=point,
                    outcome=TrialOutcome(
                        score=1.0,
                        piece_scores=(1.0,),
                        dispersion=None,
                        n_trades=10,
                        returns=(),
                        unscoreable=None,
                    ),
                )
            )
        return records

    def test_equal_raw_scores_become_unequal_once_their_zeros_differ(self) -> None:
        calibration = BaselineCalibration(
            axis="exit",
            at=(0, 1),
            baselines=(
                NullBaseline("tight", -0.1, 0.01, 1.0, 1.1, 20, "target", 1.1, 120.0),
                NullBaseline("loose", -0.5, 0.02, 1.0, 1.5, 40, "target", 1.2, 60.0),
                NullBaseline("dead", 0.0, 0.0, None, None, 0, "unscoreable", 1.0, 0.0),
            ),
            runs=100,
            seconds=1.0,
        )
        records = self._records()
        adjusted = adjusted_scores(records, calibration)
        by_value = {
            str(record.params.as_dict()["exit"]): value
            for record, value in zip(records, adjusted, strict=True)
        }
        assert by_value["tight"] == pytest.approx(1.1)
        assert by_value["loose"] == pytest.approx(1.5)

    def test_a_value_with_no_measured_baseline_is_dropped_rather_than_guessed(self) -> None:
        calibration = BaselineCalibration(
            axis="exit",
            at=(0, 1),
            baselines=(
                NullBaseline("tight", -0.1, 0.01, 1.0, 1.1, 20, "target", 1.1, 120.0),
                NullBaseline("dead", 0.0, 0.0, None, None, 0, "unscoreable", 1.0, 0.0),
            ),
            runs=10,
            seconds=1.0,
        )
        records = self._records()
        adjusted = adjusted_scores(records, calibration)
        dropped = {
            str(record.params.as_dict()["exit"])
            for record, value in zip(records, adjusted, strict=True)
            if value is None
        }
        assert dropped == {"loose", "dead"}

    def test_no_calibration_leaves_every_score_untouched(self) -> None:
        records = self._records()
        assert adjusted_scores(records, None) == [record.outcome.score for record in records]


class TestACalibrationSurvivesARoundTrip:
    """A fold's calibration is written beside its trial table and read back verbatim."""

    def test_write_then_read_returns_the_same_numbers(self, tmp_path) -> None:
        calibration = BaselineCalibration(
            axis="exit",
            at=(0, 1),
            baselines=(NullBaseline("tight", -0.1, 0.01, 1.0, 1.1, 20, "target", 1.4, 90.0),),
            runs=21,
            seconds=3.5,
        )
        write_calibration(tmp_path, calibration)
        back = read_calibration(tmp_path)
        assert back == calibration

    def test_a_directory_without_one_reads_as_none(self, tmp_path) -> None:
        assert read_calibration(tmp_path) is None
