"""A run id names one question, and one question has one answer.

Two claims, and they are separate. That the same inputs produce the same
``run_id`` is arithmetic over the manifest. That the same ``run_id`` produces the
same *result* is a property of the whole system, and it is the one that fails
first when a dict iteration order, a wall clock or a shared random stream sneaks
in — so it is asserted by running the run twice and hashing everything both times.

The store enforces the second claim on its own: writing a different result under
an id that already holds one raises rather than overwriting, because the stored
half of a contradiction is evidence.
"""

import json
from decimal import Decimal
from pathlib import Path

import polars as pl
import pytest

from tests.backtest.conftest import (
    EURUSD_H4,
    LIBRARY_PATH,
    RunInputs,
    ema_pullback,
    harness_inputs,
    swing_series,
)
from trading_system.backtest.orchestrator import StrategyBinding
from trading_system.backtest.reproducibility import (
    CURVE_FILE,
    MANIFEST_FILE,
    RESULT_FILE,
    TRADES_FILE,
    CodeVersion,
    ReproducibilityError,
    RunManifest,
    canonical,
    code_version,
    digest,
    digest_frame,
    read_run,
    result_digest,
    write_run,
)
from trading_system.core.instruments import InstrumentRegistry
from trading_system.exit.library import ExitLibrarySpec
from trading_system.risk.sizing.methods import FixedFractional

#: Short enough to keep the module quick, long enough that the run trades.
LENGTH = 900


@pytest.fixture
def inputs(registry: InstrumentRegistry) -> RunInputs:
    """One ``ema_pullback`` run over a synthetic H4 series."""
    library = ExitLibrarySpec.model_validate_json(LIBRARY_PATH.read_text())
    spec = ema_pullback()
    preset = next(item for item in library.presets if item.id == spec.exit_ref)
    return harness_inputs(
        registry,
        streams={EURUSD_H4: swing_series(LENGTH)},
        bindings=[StrategyBinding(spec=spec, exit_preset=preset, keys=(EURUSD_H4,))],
    )


def manifest_of(inputs: RunInputs) -> RunManifest:
    """The manifest of a run described by ``inputs``.

    The code version is measured once and pinned, so tests comparing two
    manifests are comparing the inputs rather than racing an edit to the tree.
    """
    return RunManifest.build(
        config=inputs.config,
        streams=inputs.streams,
        bindings=inputs.bindings,
        instruments=inputs.instruments,
        seed=inputs.costs.run_seed,
        components=inputs.components(),
        code=CodeVersion(source_digest="fixed", commit="fixed", dirty=False),
    )


class TestTheSameRunTwice:
    """The DoD's own sentence: two runs at one id must hash alike."""

    def test_two_runs_of_one_id_produce_one_result_digest(self, inputs: RunInputs) -> None:
        """Same inputs, same id, same output — hashed, not eyeballed."""
        first, second = inputs.run(), inputs.run()

        assert manifest_of(inputs).run_id == manifest_of(inputs).run_id
        assert result_digest(first) == result_digest(second)
        assert first.trades == second.trades
        assert first.curve == second.curve

    def test_the_digest_covers_the_counters_and_not_only_the_money(self, inputs: RunInputs) -> None:
        """A result differing only in a drop count must not hash the same.

        Two runs agreeing on the curve while disagreeing on how many signals
        were discarded are two different runs, and a digest that could not tell
        them apart would certify the wrong thing.
        """
        result = inputs.run()
        doctored = type(result)(
            **{
                **{
                    field: getattr(result, field)
                    for field in result.__dataclass_fields__
                    if field != "expired_orders"
                },
                "expired_orders": result.expired_orders + 1,
            }
        )

        assert result_digest(doctored) != result_digest(result)


class TestWhatChangesTheId:
    """Every input is in the id, and nothing else is."""

    def test_one_different_bar_changes_the_id(self, inputs: RunInputs) -> None:
        """A single close moved by one tick is a different question."""
        frame = inputs.streams[EURUSD_H4]
        bump = pl.Series("bump", [0.0] * (LENGTH - 1) + [0.00001])
        table = frame.df.with_columns((frame.df["close"] + bump).alias("close"))
        moved = inputs.with_streams({EURUSD_H4: frame.with_df(table)})

        assert manifest_of(moved).run_id != manifest_of(inputs).run_id
        assert digest_frame(moved.streams[EURUSD_H4]) != digest_frame(frame)

    def test_a_different_seed_changes_the_id(self, inputs: RunInputs) -> None:
        """The seed is an input; two seeds are two runs, not one run twice."""
        reseeded = RunInputs(
            **{
                **{field: getattr(inputs, field) for field in inputs.__dataclass_fields__},
                "costs": inputs.costs.model_copy(update={"run_seed": inputs.costs.run_seed + 1}),
            }
        )

        assert manifest_of(reseeded).run_id != manifest_of(inputs).run_id

    def test_a_different_sizing_method_changes_the_id(self, inputs: RunInputs) -> None:
        """Components are digested by their state, so a parameter cannot hide.

        ``FixedFractional`` keeps its fraction in a private slot and prints it in
        no format this module reads. Digesting by attribute state rather than by
        repr is what makes two risk fractions two ids.
        """
        bolder = RunInputs(
            **{
                **{field: getattr(inputs, field) for field in inputs.__dataclass_fields__},
                "sizing": FixedFractional(risk_pct=0.02),
            }
        )

        assert manifest_of(bolder).run_id != manifest_of(inputs).run_id

    def test_the_code_is_part_of_the_id(self, inputs: RunInputs) -> None:
        """An edited working tree is not the tree it was edited from."""
        base = manifest_of(inputs)
        edited = RunManifest(
            code=CodeVersion(source_digest="something-else", commit="fixed", dirty=True),
            seed=base.seed,
            config=base.config,
            data=base.data,
            strategies=base.strategies,
            instruments=base.instruments,
            components=base.components,
        )

        assert edited.run_id != base.run_id

    def test_the_measured_code_version_names_the_source_and_not_only_the_commit(self) -> None:
        """``source_digest`` is measured from the files; the commit is provenance."""
        version = code_version()

        assert len(version.source_digest) == 32
        assert version.commit is None or len(version.commit) == 40


class TestTheStore:
    """Artefacts round-trip exactly, and a contradiction is refused."""

    def test_a_stored_run_reads_back_field_for_field(
        self, inputs: RunInputs, tmp_path: Path
    ) -> None:
        """Decimals, timestamps, enums, drop counts and coverage all survive."""
        result = inputs.run()
        manifest = manifest_of(inputs)
        directory = write_run(tmp_path, manifest, result)
        stored = read_run(directory)

        assert stored.result == result
        assert stored.manifest == manifest
        assert stored.digest == result_digest(result)

    def test_the_artefacts_are_the_ones_the_dod_names(
        self, inputs: RunInputs, tmp_path: Path
    ) -> None:
        """``runs/{run_id}/`` holds parquet, json and the metadata beside them."""
        manifest = manifest_of(inputs)
        directory = write_run(tmp_path, manifest, inputs.run())

        assert directory == tmp_path / manifest.run_id
        assert {path.name for path in directory.iterdir()} == {
            MANIFEST_FILE,
            RESULT_FILE,
            CURVE_FILE,
            TRADES_FILE,
        }
        assert json.loads((directory / MANIFEST_FILE).read_text())["run_id"] == manifest.run_id

    def test_writing_the_same_result_twice_is_a_no_op(
        self, inputs: RunInputs, tmp_path: Path
    ) -> None:
        """A reproducible run may be stored again without complaint."""
        result = inputs.run()
        manifest = manifest_of(inputs)
        first = write_run(tmp_path, manifest, result)
        second = write_run(tmp_path, manifest, inputs.run())

        assert first == second

    def test_a_different_result_under_one_id_is_refused(
        self, inputs: RunInputs, tmp_path: Path
    ) -> None:
        """The store is where irreproducibility becomes visible, loudly."""
        result = inputs.run()
        manifest = manifest_of(inputs)
        write_run(tmp_path, manifest, result)
        doctored = type(result)(
            **{
                **{
                    field: getattr(result, field)
                    for field in result.__dataclass_fields__
                    if field != "fills"
                },
                "fills": result.fills + 1,
            }
        )

        with pytest.raises(ReproducibilityError, match="not reproducible"):
            write_run(tmp_path, manifest, doctored)

    def test_a_tampered_result_file_does_not_read_back_as_valid(
        self, inputs: RunInputs, tmp_path: Path
    ) -> None:
        """The stored digest is recomputed on read, never trusted."""
        directory = write_run(tmp_path, manifest_of(inputs), inputs.run())
        payload = json.loads((directory / RESULT_FILE).read_text())
        payload["fills"] += 1
        (directory / RESULT_FILE).write_text(json.dumps(payload))

        with pytest.raises(ReproducibilityError, match="disagree"):
            read_run(directory)


class TestTheDescriptionRefusesWhatItCannotSee:
    """A digest that quietly accepts anything is a digest that certifies nothing."""

    def test_a_float_is_described_exactly_rather_than_at_print_precision(self) -> None:
        """Two doubles differing in the last bit must not describe alike."""
        assert canonical(0.1) != canonical(0.1 + 5e-17)
        assert digest(0.1) != digest(0.1 + 5e-17)

    def test_a_decimal_and_a_float_of_the_same_value_are_different_descriptions(self) -> None:
        """Money and prices are different types on purpose; the digest keeps them so."""
        assert digest(Decimal("1.5")) != digest(1.5)

    def test_mapping_order_does_not_change_a_digest(self) -> None:
        """Two dicts built in different orders are one input, not two."""
        assert digest({"a": 1, "b": 2}) == digest({"b": 2, "a": 1})

    def test_a_function_cannot_be_described(self) -> None:
        """Behaviour that is not in the attributes cannot be hashed from them."""
        with pytest.raises(TypeError, match="cannot be described"):
            digest(lambda value: value)
