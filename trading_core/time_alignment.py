"""Closed-bar alignment and XAUUSD PA continuity policy."""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable

from .models import (
    AlignedDecisionFrameV1,
    AlignedTimeframeV1,
    ClosedBarV1,
    TimeframeSeriesV1,
)


TIMEFRAME_SECONDS: dict[str, int] = {"H1": 3_600, "M15": 900, "M5": 300}
MAX_EXPECTED_CLOSURE_SECONDS = 90 * 60


def bar_close_timestamp(bar: ClosedBarV1, timeframe: str) -> int:
    try:
        duration = TIMEFRAME_SECONDS[timeframe]
    except KeyError as exc:
        raise ValueError(f"unsupported timeframe: {timeframe}") from exc
    return bar.timestamp + duration


def assign_pa_continuity(
    bars: Iterable[ClosedBarV1],
    *,
    timeframe: str,
    max_expected_closure_seconds: int = MAX_EXPECTED_CLOSURE_SECONDS,
) -> tuple[ClosedBarV1, ...]:
    """Assign PA segments without filling bars.

    A missing-time span up to 90 minutes is treated as the known XAUUSD
    maintenance closure.  Longer discontinuities start a fresh PA segment.
    Every bridged short closure is explicitly marked on its first later bar.
    """

    if timeframe not in TIMEFRAME_SECONDS:
        raise ValueError(f"unsupported timeframe: {timeframe}")
    if type(max_expected_closure_seconds) is not int or max_expected_closure_seconds < 0:
        raise ValueError("max_expected_closure_seconds must be a non-negative integer")
    values = tuple(bars)
    if any(type(bar) is not ClosedBarV1 for bar in values):
        raise TypeError("bars must contain exact ClosedBarV1 values")
    if any(
        values[index].timestamp <= values[index - 1].timestamp
        for index in range(1, len(values))
    ):
        raise ValueError("bar timestamps must be strictly increasing")

    nominal = TIMEFRAME_SECONDS[timeframe]
    segment = 0
    output: list[ClosedBarV1] = []
    for index, bar in enumerate(values):
        bridged = False
        if index:
            delta = bar.timestamp - values[index - 1].timestamp
            if delta < nominal:
                raise ValueError(f"bar spacing is shorter than timeframe {timeframe}")
            if delta > nominal:
                missing_duration = delta - nominal
                if missing_duration <= max_expected_closure_seconds:
                    bridged = True
                else:
                    segment += 1
        output.append(
            replace(
                bar,
                segment_id=str(segment),
                bridged_short_gap=bridged,
            )
        )
    return tuple(output)


def _aligned(series: TimeframeSeriesV1, decision_close_timestamp: int) -> AlignedTimeframeV1:
    eligible = tuple(
        bar
        for bar in series.bars
        if bar_close_timestamp(bar, series.timeframe) <= decision_close_timestamp
    )
    if not eligible:
        raise ValueError(f"{series.timeframe} has no closed bars at decision close")
    return AlignedTimeframeV1(
        timeframe=series.timeframe,
        bars=eligible,
        last_close_timestamp=bar_close_timestamp(eligible[-1], series.timeframe),
    )


def align_decision_frame(
    *,
    symbol: str,
    decision_close_timestamp: int,
    h1: TimeframeSeriesV1,
    m15: TimeframeSeriesV1,
    m5: TimeframeSeriesV1,
) -> AlignedDecisionFrameV1:
    """Create one no-future M15 decision frame from three closed series."""

    if type(decision_close_timestamp) is not int:
        raise ValueError("decision_close_timestamp must be an integer Unix second")
    expected = ((h1, "H1"), (m15, "M15"), (m5, "M5"))
    for series, timeframe in expected:
        if series.symbol != symbol:
            raise ValueError(f"{timeframe} symbol does not match decision symbol")
        if series.timeframe != timeframe:
            raise ValueError(f"expected {timeframe} series")

    aligned_h1 = _aligned(h1, decision_close_timestamp)
    aligned_m15 = _aligned(m15, decision_close_timestamp)
    aligned_m5 = _aligned(m5, decision_close_timestamp)
    if aligned_m15.last_close_timestamp != decision_close_timestamp:
        raise ValueError("M15 does not contain the exact decision close")
    if aligned_m5.last_close_timestamp != decision_close_timestamp:
        raise ValueError("M5 does not contain the exact decision close")
    return AlignedDecisionFrameV1(
        symbol=symbol,
        decision_close_timestamp=decision_close_timestamp,
        h1=aligned_h1,
        m15=aligned_m15,
        m5=aligned_m5,
    )
