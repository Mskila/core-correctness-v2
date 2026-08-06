from __future__ import annotations

from dataclasses import replace

import pytest

from pa_core import StrategyCandidate
from trading_core import (
    FusionSignalV1,
    InstrumentConstraintsV1,
    PACandidateOccurrenceV1,
    PAObservationV1,
    generate_order_plans,
)


DECISION_CLOSE = 10_800


def _candidate(
    *,
    direction: str = "long",
    family: str = "trend_continuation",
    chase_forbidden: bool = False,
    target_anchors: tuple[float, ...] | None = None,
) -> StrategyCandidate:
    long = direction == "long"
    return StrategyCandidate(
        family=family,
        direction=direction,
        setup_type="H2" if long else "L2",
        trigger_at=9_900,
        evidence=("fixture",),
        invalidation_anchor=99.0 if long else 101.0,
        target_anchors=(103.0, 106.0) if target_anchors is None else target_anchors,
        source_policy="diagnostic_only" if family == "reversal" else "formula_candidate_only",
        chase_forbidden=chase_forbidden,
    )


def _pa(candidate: StrategyCandidate, *, low_rr_geometry: bool = False) -> PAObservationV1:
    if candidate.direction == "long":
        supports = (99.0,) if low_rr_geometry else (99.4, 99.0)
        resistances = (101.0, 103.0, 106.0)
    else:
        supports = (99.0, 97.0, 94.0)
        resistances = (100.6, 101.0)
    return PAObservationV1(
        symbol="XAUUSD",
        timeframe="M15",
        bar_close_timestamp=DECISION_CLOSE,
        direction_score=0.8 if candidate.direction == "long" else -0.8,
        barbwire=False,
        extreme_range=False,
        atr=2.0,
        current_open=99.8 if low_rr_geometry else 100.0,
        current_high=100.0 if low_rr_geometry else 100.5,
        current_low=99.5,
        current_close=100.0,
        tick=0.01,
        supports=supports,
        resistances=resistances,
        candidates=(
            PACandidateOccurrenceV1(
                candidate=candidate,
                observed_at=DECISION_CLOSE,
                is_new=True,
            ),
        ),
    )


def _signal(candidate: StrategyCandidate) -> FusionSignalV1:
    return FusionSignalV1(
        accepted=True,
        side=candidate.direction,
        direction_score=0.8 if candidate.direction == "long" else -0.8,
        entry_score=0.9,
        reject_reasons=(),
        direction_weights=(),
        entry_weights=(),
        selected_m15_candidates=(candidate,),
    )


def test_long_candidate_generates_at_most_three_non_executable_price_plans() -> None:
    candidate = _candidate()
    result = generate_order_plans(_signal(candidate), _pa(candidate))

    assert result.reject_reasons == ()
    assert [plan.style for plan in result.plans] == [
        "pa_primary",
        "confirmation_breakout",
        "pullback",
    ]
    assert [plan.order_type for plan in result.plans] == ["market", "stop", "limit"]
    assert all(plan.executable is False for plan in result.plans)
    assert all(not hasattr(plan, "volume") for plan in result.plans)

    primary, confirmation, pullback = result.plans
    assert primary.entry_price == pytest.approx(100.0)
    assert confirmation.entry_price == pytest.approx(100.51)
    assert confirmation.trigger_price == pytest.approx(100.51)
    assert pullback.entry_price == pytest.approx(99.4)
    assert pullback.limit_price == pytest.approx(99.4)
    assert [plan.stop_loss for plan in result.plans] == pytest.approx([98.8, 98.8, 98.8])
    assert primary.take_profit_1 == pytest.approx(103.0)
    assert primary.take_profit_2 == pytest.approx(106.0)
    assert confirmation.take_profit_1 == pytest.approx(106.0)
    assert confirmation.take_profit_2 is None
    assert len({plan.plan_id for plan in result.plans}) == 3


def test_stop_limit_uses_same_deterministic_atr_buffer_as_price_cap() -> None:
    candidate = _candidate()
    result = generate_order_plans(
        _signal(candidate),
        _pa(candidate),
        constraints=InstrumentConstraintsV1(
            tick=0.01,
            supports_stop_limit=True,
        ),
    )
    confirmation = next(plan for plan in result.plans if plan.style == "confirmation_breakout")

    assert confirmation.order_type == "stop_limit"
    assert confirmation.trigger_price == pytest.approx(100.51)
    assert confirmation.limit_price == pytest.approx(100.71)


def test_chase_forbidden_candidate_only_keeps_confirmed_pullback() -> None:
    candidate = _candidate(chase_forbidden=True)
    result = generate_order_plans(_signal(candidate), _pa(candidate))

    assert [plan.style for plan in result.plans] == ["pullback"]
    assert result.plans[0].order_type == "limit"


def test_existing_profit_side_targets_below_minimum_r_are_rejected() -> None:
    candidate = _candidate(target_anchors=(100.5,))
    result = generate_order_plans(
        _signal(candidate),
        _pa(candidate, low_rr_geometry=True),
    )

    assert result.plans == ()
    assert "low_reward_risk" in result.reject_reasons


def test_reversal_uses_1_2r_floor_and_keeps_policy_origin() -> None:
    candidate = _candidate(family="reversal", target_anchors=())
    result = generate_order_plans(_signal(candidate), _pa(candidate))
    primary = next(plan for plan in result.plans if plan.style == "pa_primary")

    assert primary.take_profit_1_r == pytest.approx(1.2)
    assert primary.policy_origin == "alphamaster_stage2_reversal"
    assert primary.source_policy == "diagnostic_only"


def test_short_geometry_is_the_exact_directional_mirror() -> None:
    candidate = _candidate(
        direction="short",
        target_anchors=(97.0, 94.0),
    )
    result = generate_order_plans(_signal(candidate), _pa(candidate))

    assert [plan.order_type for plan in result.plans] == ["market", "stop", "limit"]
    primary, confirmation, pullback = result.plans
    assert primary.entry_price == pytest.approx(100.0)
    assert confirmation.entry_price == pytest.approx(99.49)
    assert pullback.entry_price == pytest.approx(100.6)
    assert [plan.stop_loss for plan in result.plans] == pytest.approx([101.2, 101.2, 101.2])
    assert primary.take_profit_1 == pytest.approx(97.0)
    assert primary.take_profit_2 == pytest.approx(94.0)


def test_broker_minimum_distance_can_only_widen_structure_stop() -> None:
    candidate = _candidate()
    result = generate_order_plans(
        _signal(candidate),
        _pa(candidate),
        constraints=InstrumentConstraintsV1(tick=0.01, min_stop_distance=2.0),
    )
    primary = next(plan for plan in result.plans if plan.style == "pa_primary")
    assert primary.stop_loss == pytest.approx(98.0)


def test_rejected_fusion_never_produces_a_price_plan() -> None:
    candidate = _candidate()
    signal = replace(_signal(candidate), accepted=False, reject_reasons=("barbwire:M15",))
    result = generate_order_plans(signal, _pa(candidate))

    assert result.plans == ()
    assert result.reject_reasons == ("fusion_rejected",)
