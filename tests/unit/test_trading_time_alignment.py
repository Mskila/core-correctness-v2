from __future__ import annotations

from dataclasses import replace

import pytest

from trading_core import (
    ClosedBarV1,
    TimeframeSeriesV1,
    align_decision_frame,
    assign_pa_continuity,
)


def _bar(timestamp: int, close: float = 100.0) -> ClosedBarV1:
    return ClosedBarV1(
        timestamp=timestamp,
        open=close - 0.25,
        high=close + 0.5,
        low=close - 0.5,
        close=close,
        volume=10.0,
    )


def _series(timeframe: str, *timestamps: int) -> TimeframeSeriesV1:
    return TimeframeSeriesV1(
        symbol="XAUUSD",
        timeframe=timeframe,
        bars=tuple(_bar(timestamp, 100.0 + index) for index, timestamp in enumerate(timestamps)),
        tick=0.01,
    )


def test_alignment_uses_bar_close_time_and_excludes_future_tail() -> None:
    decision_close = 4_500
    h1 = _series("H1", 0, 3_600)
    m15 = _series("M15", 2_700, 3_600, 4_500)
    m5 = _series("M5", 3_900, 4_200, 4_500)

    frame = align_decision_frame(
        symbol="XAUUSD",
        decision_close_timestamp=decision_close,
        h1=h1,
        m15=m15,
        m5=m5,
    )

    assert [bar.timestamp for bar in frame.h1.bars] == [0]
    assert [bar.timestamp for bar in frame.m15.bars] == [2_700, 3_600]
    assert [bar.timestamp for bar in frame.m5.bars] == [3_900, 4_200]
    assert frame.h1.last_close_timestamp == 3_600
    assert frame.m15.last_close_timestamp == decision_close
    assert frame.m5.last_close_timestamp == decision_close

    extended = align_decision_frame(
        symbol="XAUUSD",
        decision_close_timestamp=decision_close,
        h1=replace(h1, bars=(*h1.bars, _bar(7_200))),
        m15=replace(m15, bars=(*m15.bars, _bar(5_400))),
        m5=replace(m5, bars=(*m5.bars, _bar(4_800))),
    )
    assert extended == frame


def test_alignment_requires_exact_closed_m15_and_m5_boundary() -> None:
    with pytest.raises(ValueError, match="M15.*decision close"):
        align_decision_frame(
            symbol="XAUUSD",
            decision_close_timestamp=4_500,
            h1=_series("H1", 0),
            m15=_series("M15", 2_700),
            m5=_series("M5", 4_200),
        )

    with pytest.raises(ValueError, match="M5.*decision close"):
        align_decision_frame(
            symbol="XAUUSD",
            decision_close_timestamp=4_500,
            h1=_series("H1", 0),
            m15=_series("M15", 3_600),
            m5=_series("M5", 3_900),
        )


def test_pa_continuity_bridges_only_recorded_short_market_closure() -> None:
    bars = (_bar(0), _bar(300), _bar(4_800), _bar(5_100), _bar(20_000))
    assigned = assign_pa_continuity(bars, timeframe="M5")

    assert len(assigned) == len(bars)
    assert [bar.timestamp for bar in assigned] == [bar.timestamp for bar in bars]
    assert [bar.segment_id for bar in assigned] == ["0", "0", "0", "0", "1"]
    assert assigned[2].bridged_short_gap is True
    assert assigned[4].bridged_short_gap is False


def test_closed_bar_rejects_forming_or_invalid_ohlc() -> None:
    with pytest.raises(ValueError, match="closed"):
        replace(_bar(0), closed=False)
    with pytest.raises(ValueError, match="OHLC"):
        replace(_bar(0), high=99.0)
