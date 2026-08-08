"""Fixtures for the Entry Engine tests.

Specs are built in code rather than loaded from the example JSON files: a test
that asserts "this trigger fires on bar 7" has to own the numbers it depends on,
and the examples are free to change without any test noticing.
"""

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

import polars as pl
import pytest

from trading_system.core.instruments import InstrumentClass
from trading_system.core.types import Timeframe
from trading_system.data.models import OHLCVFrame
from trading_system.entry.context import PRICE_FIELDS, BarSeries
from trading_system.strategies.schema import (
    Condition,
    EntryOrderSpec,
    EntrySpec,
    FeatureRef,
    InstrumentScope,
    Invalidation,
    LabelSet,
    LeafCondition,
    MarketOrder,
    Operand,
    RiskProfileSpec,
    StopReference,
    StrategySpec,
    StrategyType,
    StructureStop,
    TimeframeSpec,
)

SYMBOL = "TESTFX"
START = datetime(2024, 1, 2, 9, 0, tzinfo=UTC)
TIMEFRAME = Timeframe.M15


def frame_from_closes(
    closes: Sequence[float],
    *,
    lows: Sequence[float] | None = None,
    highs: Sequence[float] | None = None,
    symbol: str = SYMBOL,
) -> OHLCVFrame:
    """Build a frame whose bars sit flat at their closes unless widened.

    Args:
        closes: One close per bar; also the open of the same bar.
        lows: Per-bar low. Defaults to the close, i.e. a zero-range bar.
        highs: Per-bar high. Defaults to the close.
        symbol: Instrument identifier.

    Returns:
        A validated frame.
    """
    low_values = list(lows) if lows is not None else list(closes)
    high_values = list(highs) if highs is not None else list(closes)
    return OHLCVFrame.from_raw(
        pl.DataFrame(
            {
                "timestamp": [START + TIMEFRAME.duration * i for i in range(len(closes))],
                "open": list(closes),
                "high": high_values,
                "low": low_values,
                "close": list(closes),
                "volume": [1_000.0] * len(closes),
            }
        ),
        symbol,
        TIMEFRAME,
    )


def series_from_features(
    closes: Sequence[float],
    features: Mapping[str, Sequence[float | None]] | None = None,
    *,
    labels: Mapping[str, Sequence[frozenset[str] | None]] | None = None,
    lows: Sequence[float] | None = None,
    highs: Sequence[float] | None = None,
) -> BarSeries:
    """Build a series with hand-written feature columns.

    Feature values are supplied directly rather than computed, so an operator
    test can state the exact series it means — including the nulls that a real
    indicator's warmup would produce.

    Args:
        closes: One close per bar.
        features: Feature key to column, aligned with ``closes``.
        labels: Label category to column, aligned with ``closes``.
        lows: Per-bar low, defaulting to the close.
        highs: Per-bar high, defaulting to the close.

    Returns:
        The series.
    """
    frame = frame_from_closes(closes, lows=lows, highs=highs)
    return BarSeries(
        symbol=frame.symbol,
        timeframe=frame.timeframe,
        timestamps=frame.df["timestamp"].to_list(),
        prices={field: frame.df[field].to_list() for field in PRICE_FIELDS},
        features=features or {},
        labels=labels or {},
    )


def leaf(op: str, left: Operand | None = None, right: Any = None) -> LeafCondition:
    """Build one leaf condition.

    Args:
        op: Operator name as it appears in a strategy file.
        left: Left operand.
        right: Right operand or range bounds.

    Returns:
        The leaf.
    """
    return LeafCondition.model_validate({"type": "leaf", "op": op, "left": left, "right": right})


def ref(indicator: str, channel: str | None = None, **params: Any) -> FeatureRef:
    """Build a feature reference.

    Args:
        indicator: Registry key, e.g. ``"rsi"``.
        channel: Output channel for a multi-output indicator.
        **params: Indicator constructor arguments.

    Returns:
        The reference.
    """
    return FeatureRef(indicator=indicator, params=params, channel=channel)


def entry_spec(
    *,
    trigger: Condition,
    direction: str = "LONG",
    confirmation: Sequence[Condition] = (),
    confirmation_window_bars: int = 0,
    invalidation: Invalidation | None = None,
    order: Any = None,
) -> EntrySpec:
    """Build one leg of an entry.

    Args:
        trigger: Trigger condition.
        direction: ``"LONG"`` or ``"SHORT"``.
        confirmation: Confirmation conditions.
        confirmation_window_bars: Bars of opportunity, trigger bar included.
        invalidation: Invalidation rule; defaults to a level below every close
            used in these tests, so it never fires unless a test wants it to.
        order: Entry order; defaults to a market order.

    Returns:
        The entry.
    """
    return EntrySpec(
        direction=direction,  # type: ignore[arg-type]
        trigger=trigger,
        confirmation=list(confirmation),
        confirmation_window_bars=confirmation_window_bars,
        invalidation=invalidation if invalidation is not None else Invalidation(price_level=1.0),
        entry_order=EntryOrderSpec(order=order if order is not None else MarketOrder()),
    )


def labels(*names: str) -> LabelSet:
    """Build a categorical label-set operand.

    Args:
        *names: Pattern, regime or session names.

    Returns:
        The label set.
    """
    return LabelSet(labels=names)


def strategy_spec(
    *,
    trigger: Condition | None = None,
    entries: Sequence[EntrySpec] = (),
    direction: str = "LONG",
    confirmation: Sequence[Condition] = (),
    confirmation_window_bars: int = 0,
    invalidation: Invalidation | None = None,
    order: Any = None,
    base_quality: float = 0.5,
    quality_modifiers: Sequence[Any] = (),
    stop_reference: StopReference | None = None,
    strategy_id: str = "test-entry",
) -> StrategySpec:
    """Assemble a minimal but complete spec.

    Either pass ``entries`` directly, or pass the single-leg arguments and get a
    one-entry spec — most tests care about one direction and should not have to
    say so twice.

    Args:
        trigger: Trigger for the implied single leg.
        entries: Explicit legs, for the two-directional cases.
        direction: Direction of the implied single leg.
        confirmation: Confirmations of the implied single leg.
        confirmation_window_bars: Window of the implied single leg.
        invalidation: Invalidation of the implied single leg.
        order: Entry order of the implied single leg.
        base_quality: Quality before modifiers, shared by every leg.
        quality_modifiers: Condition-gated adjustments, shared by every leg.
        stop_reference: Risk Engine stop derivation; irrelevant to entry.
        strategy_id: Spec id.

    Returns:
        The spec.
    """
    if not entries:
        assert trigger is not None, "pass either a trigger or explicit entries"
        entries = [
            entry_spec(
                trigger=trigger,
                direction=direction,
                confirmation=confirmation,
                confirmation_window_bars=confirmation_window_bars,
                invalidation=invalidation,
                order=order,
            )
        ]
    return StrategySpec(
        id=strategy_id,
        version="1.0.0",
        type=StrategyType.INTRADAY,
        timeframes=TimeframeSpec(signal_tf=TIMEFRAME, entry_tf=TIMEFRAME),
        instruments=InstrumentScope(allowed_classes=[InstrumentClass.FX]),
        entries=list(entries),
        exit_ref="test-exit",
        risk_profile=RiskProfileSpec(
            base_quality=base_quality,
            quality_modifiers=list(quality_modifiers),
            stop_reference=stop_reference if stop_reference is not None else StructureStop(),
        ),
    )


@pytest.fixture
def rising_closes() -> list[float]:
    """Twelve closes stepping up by one tick each bar."""
    return [1.1000 + 0.0001 * index for index in range(12)]
