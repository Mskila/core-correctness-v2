"""Closed-world review of immutable Stage-2 order candidates.

The reviewer boundary cannot create or modify an order plan.  It can only
select one exact ``plan_id`` from the supplied Stage-2 candidates or reject
the decision.  Broker execution and position sizing remain later stages.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from pa_core import StrategyCandidate

from .models import (
    AlphaObservationV1,
    FusionInputsV1,
    FusionSignalV1,
    OrderPlanCandidateV1,
    OrderPlanGenerationV1,
    PAObservationV1,
)


APPROVED_SELECTED_PLAN = "approved_selected_plan"
REJECT_CONTEXT_CONFLICT = "reject_context_conflict"
REJECT_ENTRY_QUALITY = "reject_entry_quality"
REJECT_PA_AMBIGUITY = "reject_pa_ambiguity"
REJECT_REWARD_RISK = "reject_reward_risk"
REJECT_NO_SUITABLE_PLAN = "reject_no_suitable_plan"
REVIEWER_TIMEOUT = "reviewer_timeout"
REVIEWER_UNAVAILABLE = "reviewer_unavailable"
REVIEWER_NOT_LOGGED_IN = "reviewer_not_logged_in"
REVIEWER_MODEL_MISMATCH = "reviewer_model_mismatch"
REVIEWER_INVALID_OUTPUT = "reviewer_invalid_output"

MODEL_REASON_CODES = (
    APPROVED_SELECTED_PLAN,
    REJECT_CONTEXT_CONFLICT,
    REJECT_ENTRY_QUALITY,
    REJECT_PA_AMBIGUITY,
    REJECT_REWARD_RISK,
    REJECT_NO_SUITABLE_PLAN,
)
FAILURE_REASON_CODES = (
    REVIEWER_TIMEOUT,
    REVIEWER_UNAVAILABLE,
    REVIEWER_NOT_LOGGED_IN,
    REVIEWER_MODEL_MISMATCH,
    REVIEWER_INVALID_OUTPUT,
)
ALL_REASON_CODES = frozenset((*MODEL_REASON_CODES, *FAILURE_REASON_CODES))
OUTPUT_FIELDS = (
    "decision_id",
    "verdict",
    "selected_plan_id",
    "reason_code",
    "summary_zh",
)


class CodexReviewerUnavailableError(RuntimeError):
    """The optional Codex runtime cannot complete a review."""


class CodexNotLoggedInError(CodexReviewerUnavailableError):
    """The SDK is not using the required local ChatGPT login."""


class CodexModelConfigurationError(CodexReviewerUnavailableError):
    """The exact model, reasoning effort or service tier is unavailable."""


@dataclass(frozen=True, slots=True)
class DecisionReviewRequestV1:
    decision_id: str
    inputs: FusionInputsV1
    signal: FusionSignalV1
    plans: tuple[OrderPlanCandidateV1, ...]
    plan_reject_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.decision_id) is not str or not self.decision_id:
            raise ValueError("decision_id must be a non-empty string")
        if type(self.inputs) is not FusionInputsV1:
            raise TypeError("inputs must be an exact FusionInputsV1")
        if type(self.signal) is not FusionSignalV1:
            raise TypeError("signal must be an exact FusionSignalV1")
        if type(self.plans) is not tuple or any(
            type(plan) is not OrderPlanCandidateV1 for plan in self.plans
        ):
            raise TypeError("plans must be an exact tuple of OrderPlanCandidateV1 values")
        if len(self.plans) > 3:
            raise ValueError("a review request accepts no more than three plans")
        if type(self.plan_reject_reasons) is not tuple or any(
            type(reason) is not str for reason in self.plan_reject_reasons
        ):
            raise TypeError("plan_reject_reasons must be an exact tuple of strings")
        plan_ids = tuple(plan.plan_id for plan in self.plans)
        if len(set(plan_ids)) != len(plan_ids):
            raise ValueError("plan_id values must be unique")
        if any(plan.executable or plan.broker_validated for plan in self.plans):
            raise ValueError("review accepts theoretical, non-executable plans only")
        expected = _decision_id(
            self.inputs,
            self.signal,
            self.plans,
            self.plan_reject_reasons,
        )
        if self.decision_id != expected:
            raise ValueError("decision_id does not match the immutable request payload")


@dataclass(frozen=True, slots=True)
class DecisionReviewV1:
    decision_id: str
    verdict: str
    selected_plan_id: str | None
    reason_code: str
    summary_zh: str

    def __post_init__(self) -> None:
        if type(self.decision_id) is not str or not self.decision_id:
            raise ValueError("decision_id must be a non-empty string")
        if type(self.verdict) is not str or self.verdict not in {"approve", "reject"}:
            raise ValueError("verdict must be approve or reject")
        if type(self.reason_code) is not str or self.reason_code not in ALL_REASON_CODES:
            raise ValueError("reason_code is not part of the fixed vocabulary")
        if type(self.summary_zh) is not str or not self.summary_zh.strip():
            raise ValueError("summary_zh must be a non-empty string")
        if len(self.summary_zh) > 80:
            raise ValueError("summary_zh must contain no more than 80 characters")
        if not any("\u4e00" <= character <= "\u9fff" for character in self.summary_zh):
            raise ValueError("summary_zh must contain Chinese text")
        if self.verdict == "approve":
            if type(self.selected_plan_id) is not str or not self.selected_plan_id:
                raise ValueError("approve requires selected_plan_id")
            if self.reason_code != APPROVED_SELECTED_PLAN:
                raise ValueError("approve requires approved_selected_plan")
        else:
            if self.selected_plan_id is not None:
                raise ValueError("reject requires selected_plan_id=null")
            if self.reason_code == APPROVED_SELECTED_PLAN:
                raise ValueError("reject cannot use the approval reason code")


@dataclass(frozen=True, slots=True)
class CodexRuntimeResponseV1:
    raw_output: str
    used_tools: bool = False
    rerouted: bool = False

    def __post_init__(self) -> None:
        if type(self.raw_output) is not str:
            raise TypeError("raw_output must be a string")
        if type(self.used_tools) is not bool or type(self.rerouted) is not bool:
            raise TypeError("runtime flags must be bool")


@runtime_checkable
class DecisionReviewRuntime(Protocol):
    async def evaluate(
        self,
        request_json: str,
        output_schema: dict[str, object],
    ) -> CodexRuntimeResponseV1: ...


@runtime_checkable
class DecisionReviewer(Protocol):
    async def review(self, request: DecisionReviewRequestV1) -> DecisionReviewV1: ...


def _strategy_candidate_payload(candidate: StrategyCandidate) -> dict[str, object]:
    return {
        "family": candidate.family,
        "direction": candidate.direction,
        "setup_type": candidate.setup_type,
        "trigger_at": candidate.trigger_at,
        "evidence": list(candidate.evidence),
        "invalidation_anchor": candidate.invalidation_anchor,
        "target_anchors": list(candidate.target_anchors),
        "source_policy": candidate.source_policy,
        "chase_forbidden": candidate.chase_forbidden,
    }


def _alpha_payload(observation: AlphaObservationV1) -> dict[str, object]:
    return {
        "timeframe": observation.timeframe,
        "bar_close_timestamp": observation.bar_close_timestamp,
        "strategy_fingerprint": observation.strategy_fingerprint,
        "position": observation.position,
        "strength": observation.strength,
        "factor_value": observation.factor_value,
        "bars_used": observation.bars_used,
    }


def _pa_payload(observation: PAObservationV1) -> dict[str, object]:
    return {
        "timeframe": observation.timeframe,
        "bar_close_timestamp": observation.bar_close_timestamp,
        "direction_score": observation.direction_score,
        "barbwire": observation.barbwire,
        "extreme_range": observation.extreme_range,
        "atr": observation.atr,
        "closed_bar": {
            "open": observation.current_open,
            "high": observation.current_high,
            "low": observation.current_low,
            "close": observation.current_close,
        },
        "tick": observation.tick,
        "supports": list(observation.supports),
        "resistances": list(observation.resistances),
        "candidates": [
            {
                **_strategy_candidate_payload(occurrence.candidate),
                "observed_at": occurrence.observed_at,
                "is_new": occurrence.is_new,
            }
            for occurrence in observation.candidates
        ],
    }


def _plan_payload(plan: OrderPlanCandidateV1) -> dict[str, object]:
    return {
        "plan_id": plan.plan_id,
        "style": plan.style,
        "side": plan.side,
        "family": plan.family,
        "setup_type": plan.setup_type,
        "order_type": plan.order_type,
        "entry_price": plan.entry_price,
        "trigger_price": plan.trigger_price,
        "limit_price": plan.limit_price,
        "stop_loss": plan.stop_loss,
        "take_profit_1": plan.take_profit_1,
        "take_profit_2": plan.take_profit_2,
        "risk_distance": plan.risk_distance,
        "take_profit_1_r": plan.take_profit_1_r,
        "take_profit_2_r": plan.take_profit_2_r,
        "source_trigger_at": plan.source_trigger_at,
        "source_policy": plan.source_policy,
        "policy_origin": plan.policy_origin,
    }


def _request_payload_core(
    inputs: FusionInputsV1,
    signal: FusionSignalV1,
    plans: tuple[OrderPlanCandidateV1, ...],
    plan_reject_reasons: tuple[str, ...],
) -> dict[str, object]:
    alpha = tuple(
        observation
        for observation in (inputs.h1_alpha, inputs.m15_alpha, inputs.m5_alpha)
        if observation is not None
    )
    pa = tuple(
        observation
        for observation in (inputs.h1_pa, inputs.m15_pa, inputs.m5_pa)
        if observation is not None
    )
    close_times = {
        observation.timeframe: observation.bar_close_timestamp for observation in alpha
    }
    close_times.update(
        {observation.timeframe: observation.bar_close_timestamp for observation in pa}
    )
    return {
        "market": {
            "symbol": inputs.symbol,
            "decision_close_timestamp": inputs.decision_close_timestamp,
            "closed_bar_timestamps": close_times,
        },
        "fusion": {
            "accepted": signal.accepted,
            "side": signal.side,
            "direction_score": signal.direction_score,
            "entry_score": signal.entry_score,
            "reject_reasons": list(signal.reject_reasons),
            "direction_weights": [
                {"module_id": name, "weight": weight}
                for name, weight in signal.direction_weights
            ],
            "entry_weights": [
                {"module_id": name, "weight": weight}
                for name, weight in signal.entry_weights
            ],
            "order_plan_reject_reasons": list(plan_reject_reasons),
        },
        "alpha_evidence": [_alpha_payload(observation) for observation in alpha],
        "pa_evidence": [_pa_payload(observation) for observation in pa],
        "order_plans": [_plan_payload(plan) for plan in plans],
    }


def _canonical_json(payload: dict[str, object]) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _decision_id(
    inputs: FusionInputsV1,
    signal: FusionSignalV1,
    plans: tuple[OrderPlanCandidateV1, ...],
    plan_reject_reasons: tuple[str, ...],
) -> str:
    digest = hashlib.sha256(
        _canonical_json(
            _request_payload_core(inputs, signal, plans, plan_reject_reasons)
        ).encode("utf-8")
    ).hexdigest()
    return f"decision-v1-{digest}"


def build_decision_review_request(
    inputs: FusionInputsV1,
    signal: FusionSignalV1,
    generation: OrderPlanGenerationV1,
) -> DecisionReviewRequestV1:
    """Bind a review to the exact immutable Stage-2 inputs and plans."""

    if type(inputs) is not FusionInputsV1:
        raise TypeError("inputs must be an exact FusionInputsV1")
    if type(signal) is not FusionSignalV1:
        raise TypeError("signal must be an exact FusionSignalV1")
    if type(generation) is not OrderPlanGenerationV1:
        raise TypeError("generation must be an exact OrderPlanGenerationV1")
    plans = generation.plans
    return DecisionReviewRequestV1(
        decision_id=_decision_id(inputs, signal, plans, generation.reject_reasons),
        inputs=inputs,
        signal=signal,
        plans=plans,
        plan_reject_reasons=generation.reject_reasons,
    )


def decision_review_request_payload(request: DecisionReviewRequestV1) -> dict[str, object]:
    """Return the only market data sent across the reviewer boundary."""

    if type(request) is not DecisionReviewRequestV1:
        raise TypeError("request must be an exact DecisionReviewRequestV1")
    return {
        "decision_id": request.decision_id,
        **_request_payload_core(
            request.inputs,
            request.signal,
            request.plans,
            request.plan_reject_reasons,
        ),
    }


def decision_review_output_schema(request: DecisionReviewRequestV1) -> dict[str, object]:
    """Build the strict, decision-bound JSON Schema used by Codex."""

    plan_ids = [plan.plan_id for plan in request.plans]
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(OUTPUT_FIELDS),
        "properties": {
            "decision_id": {"type": "string", "const": request.decision_id},
            "verdict": {"type": "string", "enum": ["approve", "reject"]},
            "selected_plan_id": {
                "type": ["string", "null"],
                "enum": [None, *plan_ids],
            },
            "reason_code": {"type": "string", "enum": list(MODEL_REASON_CODES)},
            "summary_zh": {"type": "string", "minLength": 1, "maxLength": 80},
        },
    }


def _rejection_reason(request: DecisionReviewRequestV1) -> str:
    reasons = (*request.signal.reject_reasons, *request.plan_reject_reasons)
    if any("conflict" in reason for reason in reasons):
        return REJECT_CONTEXT_CONFLICT
    if any(
        marker in reason
        for reason in reasons
        for marker in ("barbwire", "ambiguous_pa", "extreme_range")
    ):
        return REJECT_PA_AMBIGUITY
    if any("reward_risk" in reason for reason in reasons):
        return REJECT_REWARD_RISK
    if not request.signal.accepted:
        return REJECT_ENTRY_QUALITY
    return REJECT_NO_SUITABLE_PLAN


def _closed_rejection(
    request: DecisionReviewRequestV1,
    reason_code: str,
    summary: str,
) -> DecisionReviewV1:
    return DecisionReviewV1(
        decision_id=request.decision_id,
        verdict="reject",
        selected_plan_id=None,
        reason_code=reason_code,
        summary_zh=summary,
    )


class MechanicalDecisionReviewer:
    """Deterministically choose the established Stage-2 style priority."""

    _STYLE_PRIORITY = {
        "pa_primary": 0,
        "confirmation_breakout": 1,
        "pullback": 2,
    }

    async def review(self, request: DecisionReviewRequestV1) -> DecisionReviewV1:
        if not request.signal.accepted or not request.plans:
            reason = _rejection_reason(request)
            return _closed_rejection(request, reason, "规则条件不足，本周期不生成交易。")
        selected = min(
            request.plans,
            key=lambda plan: (self._STYLE_PRIORITY[plan.style], plan.plan_id),
        )
        return DecisionReviewV1(
            decision_id=request.decision_id,
            verdict="approve",
            selected_plan_id=selected.plan_id,
            reason_code=APPROVED_SELECTED_PLAN,
            summary_zh="规则条件满足，采用既定优先级最高的候选方案。",
        )


def _parse_model_review(
    request: DecisionReviewRequestV1,
    raw_output: str,
) -> DecisionReviewV1:
    try:
        payload = json.loads(raw_output)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("review output is not valid JSON") from error
    if type(payload) is not dict or set(payload) != set(OUTPUT_FIELDS):
        raise ValueError("review output fields do not match the strict schema")
    if payload["decision_id"] != request.decision_id:
        raise ValueError("review output is bound to a different decision")
    verdict = payload["verdict"]
    selected_plan_id = payload["selected_plan_id"]
    reason_code = payload["reason_code"]
    summary = payload["summary_zh"]
    if type(verdict) is not str or type(reason_code) is not str or type(summary) is not str:
        raise ValueError("review output scalar types are invalid")
    if selected_plan_id is not None and type(selected_plan_id) is not str:
        raise ValueError("selected_plan_id must be a string or null")
    if reason_code not in MODEL_REASON_CODES:
        raise ValueError("reason_code is not part of the model vocabulary")
    known_plan_ids = {plan.plan_id for plan in request.plans}
    if verdict == "approve":
        if selected_plan_id not in known_plan_ids:
            raise ValueError("review selected an unknown plan")
        if reason_code != APPROVED_SELECTED_PLAN:
            raise ValueError("approve has an invalid reason code")
    elif verdict == "reject":
        if selected_plan_id is not None:
            raise ValueError("reject must not select a plan")
        if reason_code == APPROVED_SELECTED_PLAN:
            raise ValueError("reject has an invalid reason code")
    else:
        raise ValueError("verdict is invalid")
    return DecisionReviewV1(
        decision_id=request.decision_id,
        verdict=verdict,
        selected_plan_id=selected_plan_id,
        reason_code=reason_code,
        summary_zh=summary,
    )


class CodexDecisionReviewer:
    """Run one isolated Codex review and fail closed on every anomaly."""

    def __init__(
        self,
        runtime: DecisionReviewRuntime,
        *,
        timeout_seconds: float = 90.0,
    ) -> None:
        if not isinstance(runtime, DecisionReviewRuntime):
            raise TypeError("runtime must implement DecisionReviewRuntime")
        if type(timeout_seconds) not in {int, float}:
            raise ValueError("timeout_seconds must be a finite positive number")
        timeout = float(timeout_seconds)
        if not math.isfinite(timeout) or timeout <= 0.0:
            raise ValueError("timeout_seconds must be a finite positive number")
        self._runtime = runtime
        self.timeout_seconds = timeout

    async def review(self, request: DecisionReviewRequestV1) -> DecisionReviewV1:
        if not request.signal.accepted or not request.plans:
            reason = _rejection_reason(request)
            return _closed_rejection(request, reason, "规则条件不足，本周期不调用 Codex。")

        request_json = _canonical_json(decision_review_request_payload(request))
        schema = decision_review_output_schema(request)
        try:
            runtime_result = await asyncio.wait_for(
                self._runtime.evaluate(request_json, schema),
                timeout=self.timeout_seconds,
            )
        except asyncio.TimeoutError:
            return _closed_rejection(request, REVIEWER_TIMEOUT, "Codex 审查超时，本周期不下单。")
        except CodexNotLoggedInError:
            return _closed_rejection(
                request,
                REVIEWER_NOT_LOGGED_IN,
                "Codex 未使用本机 ChatGPT 登录，本周期不下单。",
            )
        except CodexModelConfigurationError:
            return _closed_rejection(
                request,
                REVIEWER_MODEL_MISMATCH,
                "Codex 模型配置不符，本周期不下单。",
            )
        except CodexReviewerUnavailableError:
            return _closed_rejection(
                request,
                REVIEWER_UNAVAILABLE,
                "Codex 当前不可用，本周期不下单。",
            )
        except Exception:
            return _closed_rejection(
                request,
                REVIEWER_UNAVAILABLE,
                "Codex 审查异常，本周期不下单。",
            )

        if runtime_result.rerouted:
            return _closed_rejection(
                request,
                REVIEWER_MODEL_MISMATCH,
                "Codex 发生模型改道，本周期不下单。",
            )
        if runtime_result.used_tools:
            return _closed_rejection(
                request,
                REVIEWER_INVALID_OUTPUT,
                "Codex 执行了不允许的动作，本周期不下单。",
            )
        try:
            return _parse_model_review(request, runtime_result.raw_output)
        except (TypeError, ValueError):
            return _closed_rejection(
                request,
                REVIEWER_INVALID_OUTPUT,
                "Codex 返回内容不符合固定格式，本周期不下单。",
            )
