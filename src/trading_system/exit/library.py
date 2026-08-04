"""The Exit DB: named, pre-composed :class:`~trading_system.exit.plan.ExitPlan` instances.

Every preset is described declaratively — the same split P04 and P06 already
use: a pydantic model is the syntax (structurally valid, no unknown fields,
right types), :func:`build_plan` is the semantics (turns a validated spec into
the real rule and modifier objects, which raise their own domain errors — a
partial ladder summing past 100%, a non-positive multiple — the same way they
would if constructed by hand). A preset that fails either layer fails to load,
with the preset's own id in the message, rather than producing a plan nobody
asked for.

**A preset without a protective stop does not parse.** ``protective_stop`` is a
required field of :class:`ExitPresetSpec`, spelled out the same way every other
component is (``{"kind": "PROTECTIVE_STOP"}``), not inferred or defaulted. This
is JSON's version of the same rule :class:`~trading_system.exit.plan.ExitPlan`
already enforces in code: an unprotected position is not constructible, in a
config file any more than in Python.

Fractions in a partial-close ladder are written as JSON **strings**
(``"0.5"``), not numbers. A JSON number is parsed as a float before pydantic
ever sees it; a fraction is money-adjacent and has to land on the exact
:class:`~decimal.Decimal` written, not its nearest float neighbour.

Loading this library once and handing the same :class:`ExitPlan` objects out
for every position is intentional, not an optimisation to be wary of:
:meth:`~trading_system.exit.plan.ExitPlan.run` calls
:meth:`~trading_system.exit.plan.ExitPlan.reset` on entry, which is exactly the
contract that makes reuse across walk-forward folds and combinatorial runs
safe — sequential reuse, never interleaved.
"""

from collections.abc import Iterator, Mapping
from decimal import Decimal
from pathlib import Path
from typing import Annotated, Literal

import pydantic
from pydantic import BaseModel, ConfigDict, Field

from trading_system.core.exceptions import ValidationError
from trading_system.data.sessions import AssetClass, Session
from trading_system.exit.base import IntrabarPolicy
from trading_system.exit.plan import ExitPlan
from trading_system.exit.rules import (
    ATRStop,
    AtrTrail,
    BreakevenMove,
    Chandelier,
    FixedRR,
    MaTrail,
    PartialClose,
    PartialRung,
    ProtectiveStop,
    SignalReverseExit,
    StructureStop,
    SwingTrail,
    TimeExit,
    TimeExitMode,
    TrailingSource,
    TrailingStop,
)

#: Location of the bundled preset library, alongside this module.
DEFAULT_LIBRARY_PATH = Path(__file__).parent / "library.json"


class ProtectiveStopSpec(BaseModel):
    """The one required, parameterless component every preset must declare."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["PROTECTIVE_STOP"] = "PROTECTIVE_STOP"


class FixedRRSpec(BaseModel):
    """Config for :class:`~trading_system.exit.rules.fixed_rr.FixedRR`.

    Attributes:
        r_multiple: Multiple of the initial risk to take profit at.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["FIXED_RR"] = "FIXED_RR"
    r_multiple: float = Field(gt=0)


class PartialRungSpec(BaseModel):
    """One rung of a partial-close ladder.

    Attributes:
        r_multiple: Multiple of the initial risk this rung's target sits at.
        fraction: Share of the ORIGINAL position size this rung closes, written
            as a JSON string so it lands on the exact ``Decimal`` intended.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    r_multiple: float = Field(gt=0)
    fraction: Decimal = Field(gt=Decimal(0), lt=Decimal(1))


class PartialCloseSpec(BaseModel):
    """Config for :class:`~trading_system.exit.rules.partial_close.PartialClose`.

    Attributes:
        rungs: The ladder, in any order — :class:`PartialClose` sorts it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["PARTIAL_CLOSE"] = "PARTIAL_CLOSE"
    rungs: list[PartialRungSpec] = Field(min_length=1)


class TimeExitSpec(BaseModel):
    """Config for :class:`~trading_system.exit.rules.time_exit.TimeExit`.

    Attributes:
        mode: Which time-based condition closes the position.
        max_bars_held: Required by, and only read under, ``MAX_BARS_HELD``.
        session: Required by, and only read under, ``SESSION_CLOSE``.
        asset_class: Read only by ``BEFORE_WEEKEND``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["TIME_EXIT"] = "TIME_EXIT"
    mode: TimeExitMode
    max_bars_held: int | None = Field(default=None, gt=0)
    session: Session | None = None
    asset_class: AssetClass = AssetClass.FX


class SignalReverseExitSpec(BaseModel):
    """Config for :class:`~trading_system.exit.rules.signal_reverse.SignalReverseExit`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["SIGNAL_REVERSE_EXIT"] = "SIGNAL_REVERSE_EXIT"


#: An :class:`~trading_system.exit.base.ExitRule` component of a preset.
RuleSpec = Annotated[
    FixedRRSpec | PartialCloseSpec | TimeExitSpec | SignalReverseExitSpec,
    Field(discriminator="kind"),
]


class ATRStopSpec(BaseModel):
    """Config for :class:`~trading_system.exit.rules.atr_stop.ATRStop`.

    Attributes:
        period: ATR smoothing period.
        multiple: Multiple of ATR used as the distance from entry.
        recompute: Whether ATR is re-read every bar.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["ATR_STOP"] = "ATR_STOP"
    period: int = Field(default=14, gt=0)
    multiple: float = Field(default=1.5, gt=0)
    recompute: bool = True


class StructureStopSpec(BaseModel):
    """Config for :class:`~trading_system.exit.rules.structure_stop.StructureStop`.

    Attributes:
        lookback: Bars required on each side of a pivot.
        buffer_atr_multiple: Extra ATR-scaled distance beyond the swing.
        atr_period: ATR period the buffer is scaled by.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["STRUCTURE_STOP"] = "STRUCTURE_STOP"
    lookback: int = Field(default=5, gt=0)
    buffer_atr_multiple: float = Field(default=0.0, ge=0)
    atr_period: int = Field(default=14, gt=0)


class AtrTrailSpec(BaseModel):
    """Config for :class:`~trading_system.exit.rules.trailing_stop.AtrTrail`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["ATR_TRAIL"] = "ATR_TRAIL"
    period: int = Field(default=14, gt=0)
    multiple: float = Field(default=3.0, gt=0)


class ChandelierSpec(BaseModel):
    """Config for :class:`~trading_system.exit.rules.trailing_stop.Chandelier`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["CHANDELIER"] = "CHANDELIER"
    lookback: int = Field(default=22, gt=0)
    period: int = Field(default=22, gt=0)
    multiple: float = Field(default=3.0, gt=0)


class MaTrailSpec(BaseModel):
    """Config for :class:`~trading_system.exit.rules.trailing_stop.MaTrail`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["MA_TRAIL"] = "MA_TRAIL"
    period: int = Field(default=20, gt=0)
    source: str = "close"
    buffer_atr_multiple: float = Field(default=0.0, ge=0)
    atr_period: int = Field(default=14, gt=0)


class SwingTrailSpec(BaseModel):
    """Config for :class:`~trading_system.exit.rules.trailing_stop.SwingTrail`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["SWING_TRAIL"] = "SWING_TRAIL"
    lookback: int = Field(default=5, gt=0)
    buffer_atr_multiple: float = Field(default=0.0, ge=0)
    atr_period: int = Field(default=14, gt=0)


#: Where a :class:`TrailingStopSpec`'s level comes from.
TrailingSourceSpec = Annotated[
    AtrTrailSpec | ChandelierSpec | MaTrailSpec | SwingTrailSpec,
    Field(discriminator="kind"),
]


class TrailingStopSpec(BaseModel):
    """Config for :class:`~trading_system.exit.rules.trailing_stop.TrailingStop`.

    Attributes:
        source: Which of the four ways to compute the trailing level.
        activation_r: Multiple of the initial risk required before this starts
            proposing levels.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["TRAILING_STOP"] = "TRAILING_STOP"
    source: TrailingSourceSpec
    activation_r: float = Field(gt=0)


class BreakevenMoveSpec(BaseModel):
    """Config for :class:`~trading_system.exit.rules.breakeven.BreakevenMove`.

    Attributes:
        activation_r: Multiple of the initial risk required before this
            proposes anything.
        spread: Extra distance past the entry, to clear the bid/ask spread.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["BREAKEVEN_MOVE"] = "BREAKEVEN_MOVE"
    activation_r: float = Field(gt=0)
    spread: float = Field(default=0.0, ge=0)


#: A :class:`~trading_system.exit.base.StopModifier` component of a preset.
ModifierSpec = Annotated[
    ATRStopSpec | StructureStopSpec | TrailingStopSpec | BreakevenMoveSpec,
    Field(discriminator="kind"),
]


class ExitPresetSpec(BaseModel):
    """One named, ready-to-compose exit.

    Attributes:
        id: Stable identifier a strategy's ``exit_ref`` resolves to.
        name: Human-readable name.
        description: What the preset is for and when to reach for it.
        protective_stop: Mandatory and explicit — see the module docstring.
        rules: Everything else that may close the position.
        stop_modifiers: Rules that move the stop without closing anything,
            applied in the order listed.
        intrabar_policy: How to resolve several levels touched in one bar.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    description: str = ""
    protective_stop: ProtectiveStopSpec
    rules: list[RuleSpec] = Field(default_factory=list)
    stop_modifiers: list[ModifierSpec] = Field(default_factory=list)
    intrabar_policy: IntrabarPolicy = IntrabarPolicy.PESSIMISTIC


class ExitLibrarySpec(BaseModel):
    """The whole contents of ``exit/library.json``.

    Attributes:
        presets: Every preset, in file order.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    presets: list[ExitPresetSpec] = Field(min_length=1)


def _build_trailing_source(spec: TrailingSourceSpec) -> TrailingSource:
    """Resolve one :class:`TrailingSourceSpec` variant into its runtime object."""
    if isinstance(spec, AtrTrailSpec):
        return AtrTrail(period=spec.period, multiple=spec.multiple)
    if isinstance(spec, ChandelierSpec):
        return Chandelier(lookback=spec.lookback, period=spec.period, multiple=spec.multiple)
    if isinstance(spec, MaTrailSpec):
        return MaTrail(
            period=spec.period,
            source=spec.source,
            buffer_atr_multiple=spec.buffer_atr_multiple,
            atr_period=spec.atr_period,
        )
    return SwingTrail(
        lookback=spec.lookback,
        buffer_atr_multiple=spec.buffer_atr_multiple,
        atr_period=spec.atr_period,
    )


def _build_modifier(spec: ModifierSpec) -> ATRStop | StructureStop | TrailingStop | BreakevenMove:
    """Resolve one :class:`ModifierSpec` variant into its runtime object."""
    if isinstance(spec, ATRStopSpec):
        return ATRStop(period=spec.period, multiple=spec.multiple, recompute=spec.recompute)
    if isinstance(spec, StructureStopSpec):
        return StructureStop(
            lookback=spec.lookback,
            buffer_atr_multiple=spec.buffer_atr_multiple,
            atr_period=spec.atr_period,
        )
    if isinstance(spec, TrailingStopSpec):
        return TrailingStop(_build_trailing_source(spec.source), activation_r=spec.activation_r)
    return BreakevenMove(activation_r=spec.activation_r, spread=spec.spread)


def _build_rule(spec: RuleSpec) -> FixedRR | PartialClose | TimeExit | SignalReverseExit:
    """Resolve one :class:`RuleSpec` variant into its runtime object."""
    if isinstance(spec, FixedRRSpec):
        return FixedRR(spec.r_multiple)
    if isinstance(spec, PartialCloseSpec):
        return PartialClose(
            [PartialRung(r_multiple=rung.r_multiple, fraction=rung.fraction) for rung in spec.rungs]
        )
    if isinstance(spec, TimeExitSpec):
        return TimeExit(
            spec.mode,
            max_bars_held=spec.max_bars_held,
            session=spec.session,
            asset_class=spec.asset_class,
        )
    return SignalReverseExit()


def build_plan(spec: ExitPresetSpec) -> ExitPlan:
    """Compose one validated preset into a runnable :class:`ExitPlan`.

    Args:
        spec: A single preset, already syntactically valid.

    Returns:
        The composed plan.

    Raises:
        ValidationError: If a component is syntactically valid on its own but
            semantically broken together — a partial ladder summing past 100%,
            a ``TIME_EXIT`` in ``MAX_BARS_HELD`` mode missing its bar count,
            and so on. The rule constructors raise a mix of ``ValueError`` and
            :class:`ValidationError` for these, neither of which names the
            preset on its own; this always does.
    """
    try:
        return ExitPlan(
            exit_id=spec.id,
            protective_stop=ProtectiveStop(),
            rules=[_build_rule(rule) for rule in spec.rules],
            stop_modifiers=[_build_modifier(modifier) for modifier in spec.stop_modifiers],
            intrabar_policy=spec.intrabar_policy,
        )
    except (ValueError, ValidationError) as error:
        raise ValidationError(f"exit preset {spec.id!r}: {error}") from error


class ExitLibrary:
    """A loaded set of named exit plans, ready to hand to a backtest.

    Iterating, indexing and membership all operate on preset ids — the same
    ids a :class:`~trading_system.strategies.schema.StrategySpec`'s
    ``exit_ref`` names.
    """

    def __init__(self, plans: Mapping[str, ExitPlan]) -> None:
        """Wrap an already-built id-to-plan mapping.

        Args:
            plans: One plan per preset id. Not validated here — construct
                through :func:`load_library` unless you are building a test
                double.
        """
        self._plans = dict(plans)

    def __repr__(self) -> str:
        """Compact description naming how many presets are loaded."""
        return f"ExitLibrary({sorted(self._plans)})"

    def __contains__(self, exit_id: str) -> bool:
        """Whether ``exit_id`` names a loaded preset."""
        return exit_id in self._plans

    def __getitem__(self, exit_id: str) -> ExitPlan:
        """The plan registered under ``exit_id``.

        Raises:
            ValidationError: If no preset carries that id.
        """
        try:
            return self._plans[exit_id]
        except KeyError:
            raise ValidationError(
                f"unknown exit preset {exit_id!r}; known: {sorted(self._plans)}"
            ) from None

    def __iter__(self) -> Iterator[str]:
        """Preset ids, in load order."""
        return iter(self._plans)

    def __len__(self) -> int:
        """Number of loaded presets."""
        return len(self._plans)

    @property
    def ids(self) -> frozenset[str]:
        """Every loaded preset id — what a caller passes as ``known_exit_ids``."""
        return frozenset(self._plans)

    def get(self, exit_id: str) -> ExitPlan | None:
        """The plan registered under ``exit_id``, or ``None`` if there is none.

        Args:
            exit_id: Preset id to look up.

        Returns:
            The plan, or ``None``.
        """
        return self._plans.get(exit_id)


def load_library(path: Path = DEFAULT_LIBRARY_PATH) -> ExitLibrary:
    """Read, validate and compose every preset in an exit library file.

    Args:
        path: JSON file to load, defaulting to the bundled library.

    Returns:
        The loaded library.

    Raises:
        ValidationError: If the file cannot be read, is not valid JSON against
            :class:`ExitLibrarySpec`, declares a duplicate id, or a preset's
            components are individually valid but semantically incompatible
            (see :func:`build_plan`). Every case names the file and, where
            applicable, the offending preset.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as error:
        raise ValidationError(f"cannot read exit library {path}: {error}") from error

    try:
        parsed = ExitLibrarySpec.model_validate_json(raw)
    except pydantic.ValidationError as error:
        details = "; ".join(
            f"{'.'.join(str(part) for part in item['loc'])}: {item['msg']}"
            for item in error.errors()
        )
        raise ValidationError(f"malformed exit library {path}: {details}") from error

    ids = [preset.id for preset in parsed.presets]
    duplicates = sorted({preset_id for preset_id in ids if ids.count(preset_id) > 1})
    if duplicates:
        raise ValidationError(f"exit library {path} has duplicate preset ids: {duplicates}")

    return ExitLibrary({preset.id: build_plan(preset) for preset in parsed.presets})


def known_exit_ids(path: Path = DEFAULT_LIBRARY_PATH) -> frozenset[str]:
    """The preset ids a library file declares.

    For :func:`~trading_system.strategies.validator.validate_paths`'s
    ``known_exit_ids`` argument. A thin convenience over :func:`load_library`
    — it still builds every plan, since a preset's id is only known-good once
    its components are known to compose, and a partially-broken library
    should fail loudly rather than validate strategy specs against an
    incomplete registry.

    Args:
        path: JSON file to load, defaulting to the bundled library.

    Returns:
        Every loaded preset id.
    """
    return load_library(path).ids
