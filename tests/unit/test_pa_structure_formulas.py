from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from pa_core import (
    Pivot,
    StructureFormulaEngine,
    classify_channel_metrics,
    classify_range_flags,
    completed_pullback_ratio,
    detect_wedge_candidates,
    evaluate_final_flag,
    evaluate_mtr,
    project_measured_move,
    score_barbwire,
)
from tests.support.pa_fixtures import BASE_TS, STEP_SECONDS, make_series, trend_rows


def test_radius_pivots_publish_only_after_confirmation_bar() -> None:
    rows = [
        (9.0, 10.0, 8.0, 9.0),
        (10.0, 11.0, 9.0, 10.0),
        (12.0, 15.0, 10.0, 14.0),
        (11.0, 12.0, 9.0, 10.0),
        (12.0, 13.0, 8.0, 11.0),
        (11.0, 12.5, 9.0, 10.5),
    ]
    engine = StructureFormulaEngine(make_series(rows))

    assert not any(p.index == 2 for p in engine.pivots(2, as_of_index=3))
    confirmed = next(p for p in engine.pivots(2, as_of_index=4) if p.index == 2)
    assert confirmed.kind == "high"
    assert confirmed.price == 15.0
    assert confirmed.confirmed_at == BASE_TS + 4 * STEP_SECONDS

    later = engine.pivots(2, as_of_index=5)
    assert next(p for p in later if p.index == 2) == confirmed


def test_direction_vote_and_always_in_on_clean_bull_trend() -> None:
    engine = StructureFormulaEngine(make_series(trend_rows(40)))
    direction = engine.direction()
    always_in = engine.always_in()

    assert direction.direction == "bullish"
    assert direction.score >= 3
    assert direction.strength == pytest.approx(abs(direction.score) / 5)
    assert always_in.state == "AIL"
    assert always_in.source_window == "near"
    assert always_in.above_ratio >= 0.65


def test_structure_results_keep_common_provenance_and_are_immutable() -> None:
    snapshot = StructureFormulaEngine(make_series(trend_rows(40))).snapshot()
    direction = snapshot.get("structure.direction")

    assert direction.valid is True
    assert direction.as_of == snapshot.as_of
    assert direction.source_start <= direction.source_end == direction.confirmed_at
    assert direction.provenance_class == "D0"
    with pytest.raises(FrozenInstanceError):
        setattr(snapshot.direction, "score", 0)


def test_pullback_correction_uses_latest_completed_leg_not_oldest_pivot() -> None:
    pivots = (
        Pivot(0, BASE_TS, "low", 90.0, 1, BASE_TS + STEP_SECONDS),
        Pivot(1, BASE_TS + STEP_SECONDS, "high", 110.0, 1, BASE_TS + 2 * STEP_SECONDS),
        Pivot(2, BASE_TS + 2 * STEP_SECONDS, "low", 100.0, 1, BASE_TS + 3 * STEP_SECONDS),
        Pivot(3, BASE_TS + 3 * STEP_SECONDS, "high", 120.0, 1, BASE_TS + 4 * STEP_SECONDS),
        Pivot(4, BASE_TS + 4 * STEP_SECONDS, "low", 117.0, 1, BASE_TS + 5 * STEP_SECONDS),
    )
    upstream_oldest_pivot_result = (110.0 - 100.0) / (110.0 - 90.0)
    corrected = completed_pullback_ratio(pivots, direction="bullish", tick=0.01)

    assert upstream_oldest_pivot_result == pytest.approx(0.50)
    assert corrected == pytest.approx(0.15)


@pytest.mark.parametrize(
    ("groups", "pullback", "parallel", "residual", "expected"),
    [
        (2, 0.20, 0.10, 0.20, "trending_tr"),
        (3, 0.2999, 0.10, 0.20, "tight_channel"),
        (3, 0.30, 0.10, 0.20, "normal_channel"),
        (3, 0.50, 0.10, 0.20, "normal_channel"),
        (3, 0.5001, 0.10, 0.20, "broad_channel"),
        (3, 0.786, 0.10, 0.20, "broad_channel"),
        (3, 0.7861, 0.10, 0.20, "trending_tr"),
        (3, 0.20, 0.3501, 0.20, "trending_tr"),
        (3, 0.20, 0.10, 0.7501, "trending_tr"),
    ],
)
def test_channel_boundaries_are_frozen(
    groups: int,
    pullback: float,
    parallel: float,
    residual: float,
    expected: str,
) -> None:
    state = classify_channel_metrics(
        group_count=groups,
        pullback_ratio=pullback,
        parallel_error=parallel,
        max_residual_atr=residual,
    )
    assert state.label == expected


def test_barbwire_component_sum_and_threshold() -> None:
    below = score_barbwire(
        overlap_mean_10=0.64,
        doji_inside_ratio_10=0.39,
        range_width_atr=3.01,
        width_to_avg_range=0.31,
    )
    boundary = score_barbwire(
        overlap_mean_10=0.65,
        doji_inside_ratio_10=0.40,
        range_width_atr=3.01,
        width_to_avg_range=0.31,
    )
    assert below.score == 0.0
    assert below.candidate is False
    assert boundary.score == pytest.approx(0.6)
    assert boundary.candidate is True


def test_range_and_extreme_range_exact_boundaries() -> None:
    ordinary, extreme = classify_range_flags(
        high_tests=2,
        low_tests=2,
        direction="neutral",
        swing_structure="mixed",
        ema_slope_atr=0.05,
        overlap_mean=0.70,
        direction_score=1,
    )
    assert ordinary is True
    assert extreme is True

    assert classify_range_flags(
        high_tests=1,
        low_tests=2,
        direction="neutral",
        swing_structure="mixed",
        ema_slope_atr=0.0501,
        overlap_mean=0.6999,
        direction_score=2,
    ) == (False, False)


def test_environment_separates_spike_climax_and_micro_channel() -> None:
    spike_rows = [(100.0, 101.0, 99.0, 100.0)] * 20 + trend_rows(4)
    spike = StructureFormulaEngine(make_series(spike_rows)).environment()
    assert spike.spike is True
    assert "standard_spike" in spike.labels
    assert spike.micro_channel is False

    climax = StructureFormulaEngine(make_series(trend_rows(30))).environment()
    assert climax.spike is False
    assert "climax_warning" in climax.labels
    assert climax.micro_channel is False


def test_breakout_failure_retest_and_failure_of_failure_are_causal() -> None:
    rows = [(100.0, 101.0, 99.0, 100.0)] * 14
    rows.extend(
        [
            (100.0, 103.0, 100.0, 102.0),
            (102.0, 102.5, 101.2, 101.5),
            (101.5, 102.0, 100.0, 100.5),
            (100.5, 102.5, 100.4, 102.0),
            (102.0, 103.0, 101.5, 102.6),
        ]
    )
    engine = StructureFormulaEngine(make_series(rows))

    before_follow = engine.breakout_events(as_of_index=17)
    event_before = next(
        event for event in before_follow if event.breakout_at == BASE_TS + 14 * STEP_SECONDS
    )
    assert event_before.failure_at == BASE_TS + 16 * STEP_SECONDS
    assert event_before.retest_at == BASE_TS + 15 * STEP_SECONDS
    assert event_before.failed_failure_at is None

    after_follow = engine.breakout_events(as_of_index=18)
    event_after = next(
        event for event in after_follow if event.breakout_at == BASE_TS + 14 * STEP_SECONDS
    )
    assert event_after.failed_failure_at == BASE_TS + 18 * STEP_SECONDS


def test_breakout_follow_uses_breakout_direction_not_candle_colour() -> None:
    rows = [(100.0, 101.0, 99.0, 100.0)] * 14
    rows.extend(
        [
            (103.0, 104.0, 101.5, 102.0),
            (102.0, 103.5, 101.8, 103.0),
        ]
    )
    events = StructureFormulaEngine(make_series(rows)).breakout_events()
    event = next(item for item in events if item.breakout_at == BASE_TS + 14 * STEP_SECONDS)

    assert rows[14][3] < rows[14][0]
    assert event.direction == "up"
    assert event.follow_through is True


def test_true_h1_h2_requires_second_pullback_leg() -> None:
    rows = [
        (100.0, 101.0, 99.0, 100.8),
        (101.0, 102.0, 100.0, 101.8),
        (102.0, 103.0, 101.0, 102.8),
        (103.0, 105.0, 102.0, 104.5),
        (104.0, 104.2, 101.0, 102.0),
        (102.0, 104.5, 100.0, 104.0),
        (101.0, 102.5, 99.5, 100.0),
        (100.0, 103.0, 100.0, 102.5),
    ]
    state = StructureFormulaEngine(make_series(rows)).hl_state(
        as_of_index=7,
        background="bullish",
    )

    assert [trigger.label for trigger in state.triggers] == ["H1", "H2"]
    assert state.triggers[0].trigger_at == BASE_TS + 5 * STEP_SECONDS
    assert state.triggers[1].trigger_at == BASE_TS + 7 * STEP_SECONDS
    assert state.triggers[1].pullback_pivot_at == BASE_TS + 6 * STEP_SECONDS


def test_prior_bar_extreme_breaks_are_not_mislabeled_as_h1_h2() -> None:
    state = StructureFormulaEngine(make_series(trend_rows(30))).hl_state(background="bullish")
    assert state.triggers == ()
    assert state.candidate == "none"


def test_wedge_mtr_and_final_flag_d1_boundaries() -> None:
    pivots = [
        (0, "low", 100.0),
        (5, "high", 110.0),
        (10, "low", 108.0),
        (15, "high", 118.0),
        (20, "low", 117.0),
        (25, "high", 124.0),
    ]
    candidates = detect_wedge_candidates(
        pivots,
        atr=2.0,
        tick=0.01,
        base_timestamp=BASE_TS,
        step_seconds=STEP_SECONDS,
        trend_direction="bullish",
    )
    assert [candidate.pattern for candidate in candidates] == ["wedge_reversal"]

    assert (
        evaluate_mtr(
            direction="bullish",
            swing_structure="HH_HL",
            trendline_break=True,
            recovery_failed=True,
            extreme_retest_failed=True,
            confirmed_at=BASE_TS,
        ).pattern
        == "mtr"
    )
    assert (
        evaluate_mtr(
            direction="bullish",
            swing_structure="HH_HL",
            trendline_break=True,
            recovery_failed=False,
            extreme_retest_failed=True,
            confirmed_at=BASE_TS,
        ).pattern
        == "reversal_attempt"
    )

    final_flag = evaluate_final_flag(
        direction_score=3,
        had_always_in_or_spike=True,
        consolidation_bars=10,
        ema_slope_atr=0.10,
        overlap_mean=0.50,
        doji_inside_ratio=0.40,
        target_distance_atr=0.50,
        breakout_observed=True,
        breakout_follow=False,
        returned_to_range=True,
        confirmed_at=BASE_TS,
    )
    assert final_flag is not None
    assert final_flag.pattern == "failed_final_flag"
    assert (
        evaluate_final_flag(
            direction_score=3,
            had_always_in_or_spike=True,
            consolidation_bars=10,
            ema_slope_atr=0.1001,
            overlap_mean=0.50,
            doji_inside_ratio=0.40,
            target_distance_atr=0.50,
            breakout_observed=True,
            breakout_follow=False,
            returned_to_range=True,
            confirmed_at=BASE_TS,
        )
        is None
    )
    assert (
        evaluate_final_flag(
            direction_score=3,
            had_always_in_or_spike=True,
            consolidation_bars=10,
            ema_slope_atr=0.10,
            overlap_mean=0.50,
            doji_inside_ratio=0.40,
            target_distance_atr=0.50,
            breakout_observed=True,
            breakout_follow=True,
            returned_to_range=False,
            confirmed_at=BASE_TS,
        )
        is None
    )


def test_double_top_and_measured_moves_from_completed_structure() -> None:
    rows = [(100.0, 101.0, 99.0, 100.0)] * 14
    rows.extend(
        [
            (102.0, 105.0, 101.0, 104.0),
            (103.0, 104.0, 100.0, 101.0),
            (101.0, 102.0, 98.0, 99.0),
            (99.0, 103.0, 99.0, 102.0),
            (103.0, 105.05, 102.0, 104.0),
            (104.0, 104.5, 100.0, 100.5),
        ]
    )
    engine = StructureFormulaEngine(make_series(rows))
    patterns = engine.patterns()
    moves = engine.measured_moves()

    assert any(candidate.pattern == "double_top" for candidate in patterns)
    assert {move.kind for move in moves} >= {"range_up", "range_down"}
    assert all(move.completed for move in moves if move.kind.startswith("leg_"))
    leg_moves = [move for move in moves if move.kind.startswith("leg_")]
    assert len(leg_moves) <= 1
    assert all(move.source_start < move.source_end for move in leg_moves)

    wedge_projection = project_measured_move(
        kind="wedge_down",
        height=10.0,
        anchor_price=120.0,
        direction="down",
        source_start=BASE_TS,
        source_end=BASE_TS + 10 * STEP_SECONDS,
    )
    assert wedge_projection.target_price == 110.0
    with pytest.raises(ValueError, match="time ordered"):
        project_measured_move(
            kind="leg_up",
            height=10.0,
            anchor_price=120.0,
            direction="up",
            source_start=BASE_TS + STEP_SECONDS,
            source_end=BASE_TS,
        )


def test_double_top_is_invalidated_by_confirmed_extreme_break() -> None:
    rows = [(100.0, 101.0, 99.0, 100.0)] * 14
    rows.extend(
        [
            (102.0, 105.0, 101.0, 104.0),
            (103.0, 104.0, 100.0, 101.0),
            (101.0, 102.0, 98.0, 99.0),
            (99.0, 103.0, 99.0, 102.0),
            (103.0, 105.05, 102.0, 104.0),
            (104.0, 104.5, 100.0, 100.5),
            (100.5, 106.0, 100.0, 105.5),
            (105.5, 106.5, 105.0, 106.0),
        ]
    )
    patterns = StructureFormulaEngine(make_series(rows)).patterns()
    assert not any(candidate.pattern == "double_top" for candidate in patterns)
