from __future__ import annotations

from dataclasses import replace

import pytest

from trading_core import (
    DecisionReviewV1,
    OrderPlanCandidateV1,
    TradeDecisionV1,
    TradingConfigV1,
)


def _plan() -> OrderPlanCandidateV1:
    return OrderPlanCandidateV1(
        plan_id="plan-1",
        style="pa_primary",
        side="long",
        family="trend_continuation",
        setup_type="H2",
        order_type="market",
        entry_price=2400.0,
        trigger_price=None,
        limit_price=None,
        stop_loss=2398.0,
        take_profit_1=2403.0,
        take_profit_2=2404.0,
        risk_distance=2.0,
        take_profit_1_r=1.5,
        take_profit_2_r=2.0,
        source_trigger_at=900,
        source_policy="formula_candidate_only",
        policy_origin="pa_formula_candidate",
    )


def test_default_config_has_canonical_hash_and_m5_alpha_disabled() -> None:
    config = TradingConfigV1.default()
    reordered = TradingConfigV1(
        mode=config.mode,
        direction_module_ids=tuple(reversed(config.direction_module_ids)),
        entry_module_ids=tuple(reversed(config.entry_module_ids)),
        pa_family_ids=tuple(reversed(config.pa_family_ids)),
        tp1_lots=config.tp1_lots,
        tp2_lots=config.tp2_lots,
    )

    assert "alpha_m5" not in config.entry_module_ids
    assert reordered == config
    assert reordered.config_hash == config.config_hash
    assert config.to_payload()["config_hash"] == config.config_hash
    assert len(config.config_hash) == 64


@pytest.mark.parametrize(
    "changes",
    [
        {"tp1_lots": 0.0},
        {"tp2_lots": -0.01},
        {"direction_module_ids": ()},
        {"entry_module_ids": ()},
        {"pa_family_ids": ()},
        {"symbol": "EURUSD"},
    ],
)
def test_config_rejects_invalid_main_flow_values(changes: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        replace(TradingConfigV1.default(), **changes)


def test_config_payload_rejects_a_mismatched_identity_hash() -> None:
    payload = TradingConfigV1.default().to_payload()
    payload["config_hash"] = "0" * 64

    with pytest.raises(ValueError, match="hash"):
        TradingConfigV1.from_payload(payload)


def test_stage4_decision_exposes_lots_but_remains_non_executable() -> None:
    config = replace(TradingConfigV1.default(), tp1_lots=0.02, tp2_lots=0.01)
    review = DecisionReviewV1(
        decision_id="decision-1",
        verdict="approve",
        selected_plan_id="plan-1",
        reason_code="approved_selected_plan",
        summary_zh="规则条件满足，采用主方案。",
    )
    decision = TradeDecisionV1(
        decision_id="decision-1",
        symbol="XAUUSD",
        decision_close_timestamp=1800,
        h1_close_timestamp=0,
        m15_close_timestamp=1800,
        m5_close_timestamp=1800,
        config=config,
        alpha_scores=(("H1", 0.8), ("M15", 0.7), ("M5", None)),
        pa_scores=(("H1", 0.7), ("M15", 0.8), ("M5", 0.9)),
        pa_evidence=("M15:trend_continuation:H2:long:new",),
        input_warnings=(),
        direction_score=0.76,
        entry_score=0.81,
        side="long",
        fusion_accepted=True,
        fusion_reject_reasons=(),
        plan_reject_reasons=(),
        plans=(_plan(),),
        review=review,
        final_action="preview_approved",
        created_at=1.0,
    )

    payload = decision.to_payload()

    assert payload["plans"][0]["tp1_lots"] == 0.02
    assert payload["plans"][0]["tp2_lots"] == 0.01
    assert payload["plans"][0]["broker_validated"] is False
    assert payload["plans"][0]["executable"] is False
    assert payload["execution"] == {
        "enabled": False,
        "ticket": None,
        "retcode": None,
    }
