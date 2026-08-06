from __future__ import annotations

from dataclasses import replace

import pytest

from tests.support.pa_fixtures import BASE_TS, STEP_SECONDS, trend_rows
from trading_core import (
    CandidateFreshnessStateV1,
    ClosedBarV1,
    TimeframeSeriesV1,
    build_pa_observation,
    evaluate_pa_observations_at_indices,
)


def _series(count: int, *, direction: str = "bullish") -> TimeframeSeriesV1:
    rows = trend_rows(count, direction=direction)
    return TimeframeSeriesV1(
        symbol="XAUUSD",
        timeframe="M15",
        tick=0.01,
        bars=tuple(
            ClosedBarV1(
                timestamp=BASE_TS + index * STEP_SECONDS,
                open=row[0],
                high=row[1],
                low=row[2],
                close=row[3],
                volume=100.0 + index,
            )
            for index, row in enumerate(rows)
        ),
    )


def test_pa_observation_uses_valid_stage1_snapshot_and_close_availability() -> None:
    series = _series(25)
    decision_close = series.bars[-1].timestamp + STEP_SECONDS
    observation, state = build_pa_observation(
        series,
        decision_close_timestamp=decision_close,
        previous=CandidateFreshnessStateV1(),
    )

    assert observation.symbol == "XAUUSD"
    assert observation.timeframe == "M15"
    assert observation.bar_close_timestamp == decision_close
    assert observation.direction_score > 0.0
    assert observation.atr > 0.0
    assert observation.current_close == series.bars[-1].close
    assert state.seen_keys == frozenset(
        "|".join(
            (
                occurrence.candidate.family,
                occurrence.candidate.direction,
                occurrence.candidate.setup_type,
                str(occurrence.candidate.trigger_at),
                occurrence.candidate.source_policy,
            )
        )
        for occurrence in observation.candidates
    )


def test_pa_observation_is_invariant_to_appended_future_bars() -> None:
    prefix = _series(25)
    decision_close = prefix.bars[-1].timestamp + STEP_SECONDS
    first, _ = build_pa_observation(
        prefix,
        decision_close_timestamp=decision_close,
        previous=CandidateFreshnessStateV1(),
        emit_new=False,
    )
    future = tuple(
        replace(
            bar,
            timestamp=prefix.bars[-1].timestamp + (index + 1) * STEP_SECONDS,
            open=80.0 - index,
            high=81.0 - index,
            low=75.0 - index,
            close=76.0 - index,
        )
        for index, bar in enumerate(prefix.bars[:5])
    )
    extended = replace(prefix, bars=(*prefix.bars, *future))
    second, _ = build_pa_observation(
        extended,
        decision_close_timestamp=decision_close,
        previous=CandidateFreshnessStateV1(),
        emit_new=False,
    )

    assert second == first


def test_pa_observation_fails_closed_before_indicator_warmup() -> None:
    series = _series(19)
    with pytest.raises(ValueError, match="insufficient.*PA.*M15"):
        build_pa_observation(
            series,
            decision_close_timestamp=series.bars[-1].timestamp + STEP_SECONDS,
            previous=CandidateFreshnessStateV1(),
        )


def test_cached_pa_sequence_matches_individual_prefix_evaluation() -> None:
    series = _series(28)
    indices = (19, 24, 27)
    batched, _ = evaluate_pa_observations_at_indices(
        series,
        end_indices=indices,
        previous=CandidateFreshnessStateV1(),
        emit_new=False,
    )
    individual = tuple(
        build_pa_observation(
            series,
            decision_close_timestamp=series.bars[index].timestamp + STEP_SECONDS,
            previous=CandidateFreshnessStateV1(),
            emit_new=False,
        )[0]
        for index in indices
    )

    assert batched == individual
