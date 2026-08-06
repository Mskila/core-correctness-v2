from __future__ import annotations

import json
from pathlib import Path

from pa_core import StrategyCandidate
from trading_core import (
    FusionSignalV1,
    PACandidateOccurrenceV1,
    PAObservationV1,
    default_module_selection,
    generate_order_plans,
    resolve_fusion_weights,
)


FIXTURE = Path(__file__).parents[1] / "fixtures" / "trading_stage2_golden.json"


def _candidate() -> StrategyCandidate:
    return StrategyCandidate(
        family="trend_continuation",
        direction="long",
        setup_type="H2",
        trigger_at=9_900,
        evidence=("golden_fixture",),
        invalidation_anchor=99.0,
        target_anchors=(103.0, 106.0),
        source_policy="formula_candidate_only",
        chase_forbidden=False,
    )


def _golden_output() -> dict[str, object]:
    def rounded(value: float | None) -> float | None:
        return None if value is None else round(value, 8)

    candidate = _candidate()
    signal = FusionSignalV1(
        accepted=True,
        side="long",
        direction_score=0.84,
        entry_score=0.95,
        reject_reasons=(),
        direction_weights=(),
        entry_weights=(),
        selected_m15_candidates=(candidate,),
    )
    observation = PAObservationV1(
        symbol="XAUUSD",
        timeframe="M15",
        bar_close_timestamp=10_800,
        direction_score=0.8,
        barbwire=False,
        extreme_range=False,
        atr=2.0,
        current_open=100.0,
        current_high=100.5,
        current_low=99.5,
        current_close=100.0,
        tick=0.01,
        supports=(99.4, 99.0),
        resistances=(101.0, 103.0, 106.0),
        candidates=(
            PACandidateOccurrenceV1(
                candidate=candidate,
                observed_at=10_800,
                is_new=True,
            ),
        ),
    )
    weights = resolve_fusion_weights(default_module_selection())
    plans = generate_order_plans(signal, observation).plans
    return {
        "direction_weights": dict(weights.direction),
        "entry_weights": dict(weights.entry),
        "plans": [
            {
                "plan_id": plan.plan_id,
                "style": plan.style,
                "order_type": plan.order_type,
                "entry_price": rounded(plan.entry_price),
                "stop_loss": rounded(plan.stop_loss),
                "take_profit_1": rounded(plan.take_profit_1),
                "take_profit_2": rounded(plan.take_profit_2),
                "take_profit_1_r": rounded(plan.take_profit_1_r),
                "take_profit_2_r": rounded(plan.take_profit_2_r),
                "executable": plan.executable,
            }
            for plan in plans
        ],
    }


def test_stage2_weights_and_order_geometry_match_golden_fixture() -> None:
    expected = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert _golden_output() == expected
