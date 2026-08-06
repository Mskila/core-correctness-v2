# SPDX-License-Identifier: AGPL-3.0-or-later
"""Adapter from closed OHLCV bars to the deterministic PA formula layer.

The underlying formula semantics are derived from PA_Agent at commit
d92ecd827fe671a589b7fdfdbba41e5e98081d87 (AGPL-3.0-or-later), Copyright
(C) 2026 PA Agent Contributors.  This adapter adds only AlphaMaster's
closed-time alignment, continuity policy and candidate freshness metadata.
"""

from __future__ import annotations

from functools import lru_cache

from pa_core import (
    BarFormulaEngine,
    BarSeries,
    PABar,
    PAStrategyDetector,
    StructureFormulaEngine,
)

from .fusion import classify_candidate_freshness
from .models import (
    CandidateFreshnessStateV1,
    ClosedBarV1,
    PAObservationV1,
    TimeframeSeriesV1,
)
from .time_alignment import assign_pa_continuity, bar_close_timestamp


_REQUIRED_RESULTS = (
    "structure.direction",
    "structure.barbwire",
    "structure.patterns",
)


class _CachedBarFormulaEngine(BarFormulaEngine):
    @lru_cache(maxsize=None)
    def evaluate(self, index: int, *, as_of_index: int | None = None):
        return super().evaluate(index, as_of_index=as_of_index)


class _CachedStructureFormulaEngine(StructureFormulaEngine):
    """Per-segment cache; outputs stay identical to the Stage-1 engine."""

    def __init__(self, series: BarSeries):
        super().__init__(series)
        self.bar_engine = _CachedBarFormulaEngine(series)

    @lru_cache(maxsize=None)
    def pivots(self, radius: int, *, as_of_index: int | None = None):
        return super().pivots(radius, as_of_index=as_of_index)

    @lru_cache(maxsize=None)
    def swing_structure(self, *, radius: int = 2, as_of_index: int | None = None):
        return super().swing_structure(radius=radius, as_of_index=as_of_index)

    @lru_cache(maxsize=None)
    def support_resistance(self, *, as_of_index: int | None = None):
        return super().support_resistance(as_of_index=as_of_index)

    @lru_cache(maxsize=None)
    def direction(self, *, as_of_index: int | None = None):
        return super().direction(as_of_index=as_of_index)

    @lru_cache(maxsize=None)
    def always_in(self, *, as_of_index: int | None = None):
        return super().always_in(as_of_index=as_of_index)

    @lru_cache(maxsize=None)
    def channel(self, *, as_of_index: int | None = None):
        return super().channel(as_of_index=as_of_index)

    @lru_cache(maxsize=None)
    def range_state(self, *, as_of_index: int | None = None):
        return super().range_state(as_of_index=as_of_index)

    @lru_cache(maxsize=None)
    def barbwire(self, *, as_of_index: int | None = None):
        return super().barbwire(as_of_index=as_of_index)

    @lru_cache(maxsize=None)
    def breakout_events(self, *, as_of_index: int | None = None):
        return super().breakout_events(as_of_index=as_of_index)

    @lru_cache(maxsize=None)
    def hl_state(
        self,
        *,
        as_of_index: int | None = None,
        background: str | None = None,
    ):
        return super().hl_state(as_of_index=as_of_index, background=background)

    @lru_cache(maxsize=None)
    def patterns(self, *, as_of_index: int | None = None):
        return super().patterns(as_of_index=as_of_index)

    @lru_cache(maxsize=None)
    def environment(self, *, as_of_index: int | None = None):
        return super().environment(as_of_index=as_of_index)

    @lru_cache(maxsize=None)
    def measured_moves(self, *, as_of_index: int | None = None):
        return super().measured_moves(as_of_index=as_of_index)


def _pa_series(series: TimeframeSeriesV1, segment) -> BarSeries:
    return BarSeries(
        symbol=series.symbol,
        timeframe=series.timeframe,
        tick=series.tick,
        bars=tuple(
            PABar(
                timestamp=bar.timestamp,
                open=bar.open,
                high=bar.high,
                low=bar.low,
                close=bar.close,
                segment_id=bar.segment_id,
            )
            for bar in segment
        ),
    )


def _observation_from_engine(
    *,
    series: TimeframeSeriesV1,
    segment,
    local_index: int,
    decision_close_timestamp: int,
    previous: CandidateFreshnessStateV1,
    emit_new: bool,
    engine: StructureFormulaEngine,
) -> tuple[PAObservationV1 | None, CandidateFreshnessStateV1]:
    snapshot = engine.snapshot(as_of_index=local_index)
    invalid = tuple(
        result.formula_id
        for result in (snapshot.get(formula_id) for formula_id in _REQUIRED_RESULTS)
        if not result.valid
    )
    atr = engine.bar_engine.atr14[local_index]
    if invalid or atr is None or atr <= 0.0:
        return None, previous
    candidates = PAStrategyDetector(snapshot).detect()
    occurrences, state = classify_candidate_freshness(
        candidates,
        observed_at=decision_close_timestamp,
        previous=previous,
        emit_new=emit_new,
    )
    current = segment[local_index]
    return (
        PAObservationV1(
            symbol=series.symbol,
            timeframe=series.timeframe,
            bar_close_timestamp=bar_close_timestamp(current, series.timeframe),
            direction_score=snapshot.direction.score / 5.0,
            barbwire=snapshot.barbwire.candidate,
            extreme_range=snapshot.range_state.extreme,
            atr=float(atr),
            current_open=current.open,
            current_high=current.high,
            current_low=current.low,
            current_close=current.close,
            tick=series.tick,
            supports=snapshot.support_resistance.supports,
            resistances=snapshot.support_resistance.resistances,
            candidates=occurrences,
        ),
        state,
    )


def build_pa_observation(
    series: TimeframeSeriesV1,
    *,
    decision_close_timestamp: int,
    previous: CandidateFreshnessStateV1,
    emit_new: bool = True,
) -> tuple[PAObservationV1, CandidateFreshnessStateV1]:
    """Evaluate the latest complete PA segment at one decision timestamp."""

    eligible = tuple(
        bar
        for bar in series.bars
        if bar_close_timestamp(bar, series.timeframe) <= decision_close_timestamp
    )
    if not eligible:
        raise ValueError(f"insufficient PA history for {series.timeframe}: 0 bars")
    assigned = assign_pa_continuity(eligible, timeframe=series.timeframe)
    current_segment = assigned[-1].segment_id
    segment = tuple(bar for bar in assigned if bar.segment_id == current_segment)
    engine = _CachedStructureFormulaEngine(_pa_series(series, segment))
    observation, state = _observation_from_engine(
        series=series,
        segment=segment,
        local_index=len(segment) - 1,
        decision_close_timestamp=decision_close_timestamp,
        previous=previous,
        emit_new=emit_new,
        engine=engine,
    )
    if observation is None:
        raise ValueError(
            f"insufficient stable PA history for {series.timeframe}: indicator_warmup"
        )
    return observation, state


def evaluate_pa_observations_at_indices(
    series: TimeframeSeriesV1,
    *,
    end_indices: tuple[int, ...],
    previous: CandidateFreshnessStateV1,
    emit_new: bool = True,
) -> tuple[tuple[PAObservationV1 | None, ...], CandidateFreshnessStateV1]:
    """Evaluate an ordered PA sequence while caching pure per-segment formulas."""

    if type(end_indices) is not tuple or any(type(index) is not int for index in end_indices):
        raise TypeError("end_indices must be an exact tuple of integers")
    if any(index < 0 or index >= len(series.bars) for index in end_indices):
        raise ValueError("PA end index is outside the series")
    if any(end_indices[index] <= end_indices[index - 1] for index in range(1, len(end_indices))):
        raise ValueError("PA end indices must be strictly increasing")
    assigned = assign_pa_continuity(series.bars, timeframe=series.timeframe)
    segment_bounds: dict[str, tuple[int, int]] = {}
    for index, bar in enumerate(assigned):
        start, _ = segment_bounds.get(bar.segment_id, (index, index))
        segment_bounds[bar.segment_id] = (start, index + 1)

    active_segment_id: str | None = None
    active_segment: tuple[ClosedBarV1, ...] = ()
    active_engine: _CachedStructureFormulaEngine | None = None
    state = previous
    output: list[PAObservationV1 | None] = []
    for global_index in end_indices:
        segment_id = assigned[global_index].segment_id
        start, end = segment_bounds[segment_id]
        if segment_id != active_segment_id:
            active_segment_id = segment_id
            active_segment = assigned[start:end]
            active_engine = _CachedStructureFormulaEngine(
                _pa_series(series, active_segment)
            )
        assert active_engine is not None
        local_index = global_index - start
        observation, state = _observation_from_engine(
            series=series,
            segment=active_segment,
            local_index=local_index,
            decision_close_timestamp=bar_close_timestamp(
                assigned[global_index], series.timeframe
            ),
            previous=state,
            emit_new=emit_new,
            engine=active_engine,
        )
        output.append(observation)
    return tuple(output), state
