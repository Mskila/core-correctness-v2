"""Deterministic, human-readable OHLC fixtures for the PA formula tests."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from pa_core import BarSeries, PABar


BASE_TS = 1_700_000_000
STEP_SECONDS = 900


def make_series(
    rows: Sequence[tuple[float, float, float, float]],
    *,
    symbol: str = "XAUUSD",
    timeframe: str = "M15",
    tick: float = 0.01,
    segment_id: str = "fixture-0",
) -> BarSeries:
    bars = tuple(
        PABar(
            timestamp=BASE_TS + index * STEP_SECONDS,
            open=open_,
            high=high,
            low=low,
            close=close,
            segment_id=segment_id,
        )
        for index, (open_, high, low, close) in enumerate(rows)
    )
    return BarSeries(symbol=symbol, timeframe=timeframe, bars=bars, tick=tick)


def trend_rows(count: int, *, direction: str = "bullish") -> list[tuple[float, ...]]:
    rows: list[tuple[float, ...]] = []
    for index in range(count):
        if direction == "bullish":
            open_ = 100.0 + index * 0.8
            rows.append((open_, open_ + 0.8, open_ - 0.1, open_ + 0.7))
        elif direction == "bearish":
            open_ = 130.0 - index * 0.8
            rows.append((open_, open_ + 0.1, open_ - 0.8, open_ - 0.7))
        else:
            raise ValueError(f"unsupported direction: {direction}")
    return rows


def append_rows(
    prefix: Iterable[tuple[float, float, float, float]],
    suffix: Iterable[tuple[float, float, float, float]],
) -> list[tuple[float, float, float, float]]:
    return [*prefix, *suffix]
