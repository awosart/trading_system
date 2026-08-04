"""The indicator base machinery: warmup masking, shape validation, parity checking."""

from dataclasses import dataclass
from datetime import datetime

import polars as pl
import pytest

from trading_system.core.exceptions import ValidationError
from trading_system.data.models import OHLCVFrame
from trading_system.features.base import (
    BaseStreaming,
    ScalarIndicator,
    as_row,
    iter_bars,
    run_streaming,
)
from trading_system.features.indicators.trend import MACD, SMA


@dataclass(frozen=True)
class Doubler(ScalarIndicator):
    """Twice the close, declared valid from bar 2 onwards."""

    @property
    def name(self) -> str:
        return "doubler"

    @property
    def warmup(self) -> int:
        return 2

    def _expression(self) -> pl.Expr:
        return pl.col("close") * 2

    def streaming(self) -> "DoublerStream":
        return DoublerStream(self)


class DoublerStream(BaseStreaming[Doubler, float]):
    def reset(self) -> None:
        self._seen = 0

    def step(
        self,
        _timestamp: datetime,
        _open_price: float,
        _high: float,
        _low: float,
        close: float,
        _volume: float,
        /,
    ) -> float | None:
        self._seen += 1
        if self._seen <= self._indicator.warmup:
            return None
        return close * 2


@dataclass(frozen=True)
class Liar(Doubler):
    """Vectorised path says ``close * 2``; the state machine says ``close * 3``."""

    @property
    def name(self) -> str:
        return "liar"

    def streaming(self) -> "LiarStream":
        return LiarStream(self)


class LiarStream(DoublerStream):
    def step(
        self,
        timestamp: datetime,
        open_price: float,
        high: float,
        low: float,
        close: float,
        volume: float,
        /,
    ) -> float | None:
        value = super().step(timestamp, open_price, high, low, close, volume)
        return None if value is None else value * 1.5


@dataclass(frozen=True)
class WrongShape(Doubler):
    """Returns a column nobody asked for."""

    @property
    def name(self) -> str:
        return "wrong_shape"

    def _evaluate(self, frame: OHLCVFrame, /) -> pl.DataFrame:
        return frame.df.select((pl.col("close") * 2).alias("unexpected"))


@dataclass(frozen=True)
class NanMaker(Doubler):
    """Produces a NaN the base class is expected to normalise into a null."""

    @property
    def name(self) -> str:
        return "nan_maker"

    def _expression(self) -> pl.Expr:
        return pl.col("close") * 0.0 / 0.0


def test_compute_frame_nulls_the_warmup_prefix(reference_frame: OHLCVFrame) -> None:
    values = Doubler().compute_frame(reference_frame)["value"].to_list()
    assert values[:2] == [None, None]
    assert values[2] == pytest.approx(reference_frame.df["close"].item(2) * 2)


def test_compute_names_the_series_after_the_indicator(reference_frame: OHLCVFrame) -> None:
    assert SMA(10).compute(reference_frame).name == "sma_10"


def test_nan_is_normalised_to_null(reference_frame: OHLCVFrame) -> None:
    """One representation of "no value", so ``is_null`` is the only check callers need."""
    values = NanMaker().compute_frame(reference_frame)["value"]
    assert values.null_count() == len(reference_frame)
    assert not values.is_nan().any()


def test_wrong_columns_are_rejected(reference_frame: OHLCVFrame) -> None:
    with pytest.raises(ValidationError, match="expected"):
        WrongShape().compute_frame(reference_frame)


def test_verify_parity_reports_the_first_disagreement(reference_frame: OHLCVFrame) -> None:
    with pytest.raises(ValidationError, match=r"liar\.value\[2\]"):
        Liar().verify_parity(reference_frame)


def test_verify_parity_passes_when_paths_agree(reference_frame: OHLCVFrame) -> None:
    Doubler().verify_parity(reference_frame)


def test_multi_output_indicators_expose_every_channel(reference_frame: OHLCVFrame) -> None:
    computed = MACD().compute_frame(reference_frame)
    assert tuple(computed.columns) == ("macd", "signal", "histogram")
    assert computed.height == len(reference_frame)


def test_iter_bars_round_trips_the_frame(reference_frame: OHLCVFrame) -> None:
    bars = list(iter_bars(reference_frame))
    assert len(bars) == len(reference_frame)
    first = bars[0]
    row = reference_frame.df.row(0, named=True)
    assert first.timestamp == row["timestamp"]
    assert (first.open, first.high, first.low, first.close) == (
        row["open"],
        row["high"],
        row["low"],
        row["close"],
    )


def test_run_streaming_produces_the_declared_channels(reference_frame: OHLCVFrame) -> None:
    streamed = run_streaming(MACD(), reference_frame)
    assert tuple(streamed.columns) == ("macd", "signal", "histogram")
    assert streamed.height == len(reference_frame)


def test_as_row_normalises_both_value_shapes() -> None:
    assert as_row(1.5) == (1.5,)
    assert as_row((1.0, 2.0)) == (1.0, 2.0)


def test_streaming_reads_parameters_from_its_indicator() -> None:
    indicator = SMA(7)
    state = indicator.streaming()
    assert state.indicator is indicator
    assert state.warmup == 6
    assert state.outputs == ("value",)
