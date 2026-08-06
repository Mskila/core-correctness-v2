from __future__ import annotations

import asyncio
import json
from dataclasses import replace

import pytest

from trading_core import (
    AlphaObservationV1,
    CodexDecisionReviewer,
    CodexModelConfigurationError,
    CodexNotLoggedInError,
    CodexReviewerUnavailableError,
    CodexRuntimeResponseV1,
    DecisionReviewer,
    FusionInputsV1,
    FusionSignalV1,
    MechanicalDecisionReviewer,
    OrderPlanCandidateV1,
    OrderPlanGenerationV1,
    PAObservationV1,
    build_decision_review_request,
    decision_review_output_schema,
    decision_review_request_payload,
)


DECISION_CLOSE = 10_800


def _alpha(timeframe: str, position: float) -> AlphaObservationV1:
    return AlphaObservationV1(
        symbol="XAUUSD",
        timeframe=timeframe,
        bar_close_timestamp=DECISION_CLOSE,
        strategy_fingerprint=f"{timeframe.lower()}-strategy",
        position=position,
        strength=abs(position),
        factor_value=1.25 * position,
        bars_used=120,
    )


def _pa(timeframe: str, direction: float) -> PAObservationV1:
    return PAObservationV1(
        symbol="XAUUSD",
        timeframe=timeframe,
        bar_close_timestamp=DECISION_CLOSE,
        direction_score=direction,
        barbwire=False,
        extreme_range=False,
        atr=2.0,
        current_open=100.0,
        current_high=101.0,
        current_low=99.0,
        current_close=100.5,
        tick=0.01,
        supports=(99.0,),
        resistances=(103.0,),
        candidates=(),
    )


def _plan(
    plan_id: str,
    style: str,
    *,
    order_type: str = "market",
    entry_price: float = 100.5,
) -> OrderPlanCandidateV1:
    trigger = entry_price if order_type in {"stop", "stop_limit"} else None
    limit = entry_price if order_type in {"limit", "stop_limit"} else None
    return OrderPlanCandidateV1(
        plan_id=plan_id,
        style=style,
        side="long",
        family="trend_continuation",
        setup_type="H2",
        order_type=order_type,
        entry_price=entry_price,
        trigger_price=trigger,
        limit_price=limit,
        stop_loss=98.5,
        take_profit_1=103.5,
        take_profit_2=106.0,
        risk_distance=2.0,
        take_profit_1_r=1.5,
        take_profit_2_r=2.75,
        source_trigger_at=9_900,
        source_policy="formula_candidate_only",
        policy_origin="pa_formula_candidate",
    )


def _request(
    plans: tuple[OrderPlanCandidateV1, ...] | None = None,
    *,
    accepted: bool = True,
    plan_reject_reasons: tuple[str, ...] = (),
):
    actual_plans = (
        (
            _plan("plan-pullback", "pullback", order_type="limit", entry_price=99.5),
            _plan("plan-primary", "pa_primary"),
        )
        if plans is None
        else plans
    )
    inputs = FusionInputsV1(
        symbol="XAUUSD",
        decision_close_timestamp=DECISION_CLOSE,
        h1_alpha=_alpha("H1", 0.8),
        h1_pa=_pa("H1", 0.7),
        m15_alpha=_alpha("M15", 0.7),
        m15_pa=_pa("M15", 0.8),
        m5_pa=_pa("M5", 0.9),
    )
    signal = FusionSignalV1(
        accepted=accepted,
        side="long" if accepted else None,
        direction_score=0.76,
        entry_score=0.81,
        reject_reasons=() if accepted else ("direction_below_threshold",),
        direction_weights=(("alpha_h1", 0.4), ("pa_h1", 0.6)),
        entry_weights=(("pa_m15_pattern", 0.6), ("pa_m5_timing", 0.4)),
        selected_m15_candidates=(),
    )
    return build_decision_review_request(
        inputs,
        signal,
        OrderPlanGenerationV1(
            plans=actual_plans if accepted else (),
            reject_reasons=(
                plan_reject_reasons
                if accepted
                else ("fusion_rejected",)
            ),
        ),
    )


def test_request_is_canonical_bound_and_contains_no_position_size() -> None:
    first = _request()
    second = _request()
    changed_plan = replace(first.plans[0], entry_price=99.25, limit_price=99.25)
    changed = _request((changed_plan, first.plans[1]))

    assert first.decision_id == second.decision_id
    assert first.decision_id != changed.decision_id

    payload = decision_review_request_payload(first)
    assert set(payload) == {
        "decision_id",
        "market",
        "fusion",
        "alpha_evidence",
        "pa_evidence",
        "order_plans",
    }
    encoded = json.dumps(payload, sort_keys=True)
    assert "entry_price" in encoded
    assert "volume" not in encoded
    assert "lots" not in encoded
    assert "broker_validated" not in encoded
    assert "executable" not in encoded


def test_output_schema_is_exact_and_binds_decision_and_plan_ids() -> None:
    request = _request()
    schema = decision_review_output_schema(request)

    assert schema["additionalProperties"] is False
    assert schema["required"] == [
        "decision_id",
        "verdict",
        "selected_plan_id",
        "reason_code",
        "summary_zh",
    ]
    assert schema["properties"]["decision_id"]["const"] == request.decision_id
    selected = schema["properties"]["selected_plan_id"]
    assert selected["enum"] == [None, "plan-pullback", "plan-primary"]
    assert schema["properties"]["summary_zh"]["maxLength"] == 80


def test_mechanical_reviewer_selects_primary_plan_deterministically() -> None:
    reviewer: DecisionReviewer = MechanicalDecisionReviewer()

    result = asyncio.run(reviewer.review(_request()))

    assert result.verdict == "approve"
    assert result.selected_plan_id == "plan-primary"
    assert result.reason_code == "approved_selected_plan"
    assert len(result.summary_zh) <= 80


def test_mechanical_reviewer_preserves_order_generation_reward_risk_rejection() -> None:
    request = _request(plans=(), plan_reject_reasons=("low_reward_risk",))

    result = asyncio.run(MechanicalDecisionReviewer().review(request))

    assert result.verdict == "reject"
    assert result.reason_code == "reject_reward_risk"


class _FakeRuntime:
    def __init__(
        self,
        raw_output: str | None = None,
        *,
        error: Exception | None = None,
        delay: float = 0.0,
        used_tools: bool = False,
        rerouted: bool = False,
    ) -> None:
        self.raw_output = raw_output
        self.error = error
        self.delay = delay
        self.used_tools = used_tools
        self.rerouted = rerouted
        self.calls: list[tuple[str, dict[str, object]]] = []

    async def evaluate(
        self,
        request_json: str,
        output_schema: dict[str, object],
    ) -> CodexRuntimeResponseV1:
        self.calls.append((request_json, output_schema))
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error is not None:
            raise self.error
        assert self.raw_output is not None
        return CodexRuntimeResponseV1(
            raw_output=self.raw_output,
            used_tools=self.used_tools,
            rerouted=self.rerouted,
        )


def _raw_result(request, **changes: object) -> str:
    data: dict[str, object] = {
        "decision_id": request.decision_id,
        "verdict": "approve",
        "selected_plan_id": "plan-primary",
        "reason_code": "approved_selected_plan",
        "summary_zh": "多周期方向一致，采用给定主方案。",
    }
    data.update(changes)
    return json.dumps(data, ensure_ascii=False)


def test_codex_reviewer_accepts_only_a_precomputed_plan() -> None:
    request = _request()
    runtime = _FakeRuntime(_raw_result(request))

    result = asyncio.run(CodexDecisionReviewer(runtime).review(request))

    assert result.verdict == "approve"
    assert result.selected_plan_id == "plan-primary"
    assert result.reason_code == "approved_selected_plan"
    assert len(runtime.calls) == 1
    sent_payload = json.loads(runtime.calls[0][0])
    assert sent_payload == decision_review_request_payload(request)
    assert "volume" not in runtime.calls[0][0]


def test_codex_reviewer_accepts_a_schema_valid_no_trade_decision() -> None:
    request = _request()
    runtime = _FakeRuntime(
        _raw_result(
            request,
            verdict="reject",
            selected_plan_id=None,
            reason_code="reject_pa_ambiguity",
            summary_zh="形态证据存在歧义，本周期不下单。",
        )
    )

    result = asyncio.run(CodexDecisionReviewer(runtime).review(request))

    assert result.verdict == "reject"
    assert result.selected_plan_id is None
    assert result.reason_code == "reject_pa_ambiguity"


@pytest.mark.parametrize(
    "mutate",
    [
        {"decision_id": "another-decision"},
        {"selected_plan_id": "invented-plan"},
        {"summary_zh": "过" * 81},
        {"summary_zh": "No suitable plan for this decision."},
        {"entry_price": 123.45},
        {"verdict": "reject", "selected_plan_id": "plan-primary"},
        {"reason_code": "invented_reason"},
    ],
)
def test_codex_reviewer_fails_closed_on_invalid_or_expansive_output(
    mutate: dict[str, object],
) -> None:
    request = _request()
    runtime = _FakeRuntime(_raw_result(request, **mutate))

    result = asyncio.run(CodexDecisionReviewer(runtime).review(request))

    assert result.verdict == "reject"
    assert result.selected_plan_id is None
    assert result.reason_code == "reviewer_invalid_output"


@pytest.mark.parametrize(
    ("error", "reason_code"),
    [
        (CodexNotLoggedInError("not logged in"), "reviewer_not_logged_in"),
        (CodexModelConfigurationError("wrong model"), "reviewer_model_mismatch"),
        (CodexReviewerUnavailableError("unavailable"), "reviewer_unavailable"),
    ],
)
def test_codex_reviewer_maps_runtime_failures_to_no_trade(
    error: Exception,
    reason_code: str,
) -> None:
    result = asyncio.run(
        CodexDecisionReviewer(_FakeRuntime(error=error)).review(_request())
    )

    assert result.verdict == "reject"
    assert result.selected_plan_id is None
    assert result.reason_code == reason_code


def test_codex_reviewer_enforces_timeout_and_rejects_tool_use_or_reroute() -> None:
    request = _request()
    timed_out = asyncio.run(
        CodexDecisionReviewer(
            _FakeRuntime(_raw_result(request), delay=0.05),
            timeout_seconds=0.001,
        ).review(request)
    )
    tool_use = asyncio.run(
        CodexDecisionReviewer(
            _FakeRuntime(_raw_result(request), used_tools=True)
        ).review(request)
    )
    rerouted = asyncio.run(
        CodexDecisionReviewer(
            _FakeRuntime(_raw_result(request), rerouted=True)
        ).review(request)
    )

    assert timed_out.reason_code == "reviewer_timeout"
    assert tool_use.reason_code == "reviewer_invalid_output"
    assert rerouted.reason_code == "reviewer_model_mismatch"
    assert all(result.verdict == "reject" for result in (timed_out, tool_use, rerouted))


def test_codex_reviewer_does_not_call_model_without_an_accepted_plan() -> None:
    runtime = _FakeRuntime("unused")

    result = asyncio.run(CodexDecisionReviewer(runtime).review(_request(accepted=False)))

    assert result.verdict == "reject"
    assert result.reason_code == "reject_entry_quality"
    assert runtime.calls == []
