from __future__ import annotations

from dataclasses import replace

import pytest

from trading_core import (
    AlphaObservationV1,
    CandidateFreshnessStateV1,
    ClosedBarV1,
    FusionInputsV1,
    TimeframeSeriesV1,
    align_decision_frame,
    build_pa_observation,
    fuse_signals,
)


DECISION_CLOSE = 2_000_000
TIMEFRAME_SECONDS = {"H1": 3_600, "M15": 900, "M5": 300}


def _series(timeframe: str, seed: int) -> TimeframeSeriesV1:
    step = TIMEFRAME_SECONDS[timeframe]
    count = 48
    start = DECISION_CLOSE - count * step
    close = 2_000.0 + seed
    bars: list[ClosedBarV1] = []
    for index in range(count):
        drift = (((index + seed) * 7) % 9 - 4) * 0.07 + 0.05
        open_ = close
        close = open_ + drift
        bars.append(
            ClosedBarV1(
                timestamp=start + index * step,
                open=open_,
                high=max(open_, close) + 0.35,
                low=min(open_, close) - 0.30,
                close=close,
                volume=100.0 + index,
            )
        )
    return TimeframeSeriesV1(
        symbol="XAUUSD",
        timeframe=timeframe,
        bars=tuple(bars),
        tick=0.01,
    )


def _append_adversarial_future(series: TimeframeSeriesV1) -> TimeframeSeriesV1:
    step = TIMEFRAME_SECONDS[series.timeframe]
    future: list[ClosedBarV1] = []
    for offset in range(1, 7):
        price = 4_000.0 + 100.0 * offset
        future.append(
            ClosedBarV1(
                timestamp=series.bars[-1].timestamp + offset * step,
                open=price,
                high=price + 50.0,
                low=price - 50.0,
                close=price + (-25.0 if offset % 2 else 25.0),
                volume=10_000.0,
            )
        )
    return replace(series, bars=(*series.bars, *future))


def _alpha(observation, position: float) -> AlphaObservationV1:
    return AlphaObservationV1(
        symbol="XAUUSD",
        timeframe=observation.timeframe,
        bar_close_timestamp=observation.bar_close_timestamp,
        strategy_fingerprint=f"fixture-{observation.timeframe}",
        position=position,
        strength=abs(position),
        factor_value=position,
        bars_used=48,
    )


@pytest.mark.parametrize("seed", [0, 3, 11])
def test_appended_future_bars_cannot_change_multitimeframe_decision(seed: int) -> None:
    original = {
        timeframe: _series(timeframe, seed)
        for timeframe in ("H1", "M15", "M5")
    }
    extended = {
        timeframe: _append_adversarial_future(series)
        for timeframe, series in original.items()
    }

    original_frame = align_decision_frame(
        symbol="XAUUSD",
        decision_close_timestamp=DECISION_CLOSE,
        h1=original["H1"],
        m15=original["M15"],
        m5=original["M5"],
    )
    extended_frame = align_decision_frame(
        symbol="XAUUSD",
        decision_close_timestamp=DECISION_CLOSE,
        h1=extended["H1"],
        m15=extended["M15"],
        m5=extended["M5"],
    )
    assert extended_frame == original_frame

    original_pa = {}
    extended_pa = {}
    for timeframe in ("H1", "M15", "M5"):
        original_pa[timeframe], _ = build_pa_observation(
            original[timeframe],
            decision_close_timestamp=DECISION_CLOSE,
            previous=CandidateFreshnessStateV1(),
        )
        extended_pa[timeframe], _ = build_pa_observation(
            extended[timeframe],
            decision_close_timestamp=DECISION_CLOSE,
            previous=CandidateFreshnessStateV1(),
        )
    assert extended_pa == original_pa

    def decision(pa):
        return fuse_signals(
            FusionInputsV1(
                symbol="XAUUSD",
                decision_close_timestamp=DECISION_CLOSE,
                h1_alpha=_alpha(pa["H1"], 0.8),
                h1_pa=pa["H1"],
                m15_alpha=_alpha(pa["M15"], 0.8),
                m15_pa=pa["M15"],
                m5_pa=pa["M5"],
            )
        )

    assert decision(extended_pa) == decision(original_pa)
