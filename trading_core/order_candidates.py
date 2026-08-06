"""Construct theoretical order-price plans from an accepted fusion signal."""

from __future__ import annotations

import hashlib
import json
import math

from pa_core import StrategyCandidate

from .models import (
    FusionSignalV1,
    InstrumentConstraintsV1,
    OrderPlanCandidateV1,
    OrderPlanGenerationV1,
    PAObservationV1,
)


_FAMILY_MIN_R = {
    "trend_continuation": 1.5,
    "breakout": 1.5,
    "reversal": 1.2,
    "range": 1.2,
}
_FAMILY_PRIORITY = {
    "breakout": 0,
    "trend_continuation": 1,
    "range": 2,
    "reversal": 3,
}


def _candidate_sort_key(candidate: StrategyCandidate) -> tuple[int, int, int, str]:
    return (
        -int(candidate.trigger_at),
        0 if candidate.source_policy == "formula_candidate_only" else 1,
        _FAMILY_PRIORITY.get(candidate.family, 99),
        candidate.setup_type,
    )


def _append_once(reasons: list[str], reason: str) -> None:
    if reason not in reasons:
        reasons.append(reason)


def _stop_loss(
    *,
    side: str,
    entry: float,
    invalidation: float,
    buffer: float,
    minimum_distance: float,
) -> float | None:
    if side == "long":
        if invalidation >= entry:
            return None
        return min(invalidation - buffer, entry - minimum_distance)
    if invalidation <= entry:
        return None
    return max(invalidation + buffer, entry + minimum_distance)


def _profit_targets(
    *,
    side: str,
    entry: float,
    stop_loss: float,
    anchors: tuple[float, ...],
    minimum_r: float,
) -> tuple[float, float | None, float, float | None] | None:
    risk = abs(entry - stop_loss)
    if not math.isfinite(risk) or risk <= 0.0:
        return None
    if side == "long":
        directional = sorted({float(value) for value in anchors if float(value) > entry})
        ratio = lambda target: (target - entry) / risk
        fallback = lambda multiple: entry + multiple * risk
        farther = lambda target, tp1: target > tp1
    else:
        directional = sorted(
            {float(value) for value in anchors if float(value) < entry}, reverse=True
        )
        ratio = lambda target: (entry - target) / risk
        fallback = lambda multiple: entry - multiple * risk
        farther = lambda target, tp1: target < tp1

    if directional:
        eligible = [target for target in directional if ratio(target) >= minimum_r]
        if not eligible:
            return None
        tp1 = eligible[0]
    else:
        tp1 = fallback(minimum_r)
    tp1_r = ratio(tp1)

    structural_tp2 = next((target for target in directional if farther(target, tp1)), None)
    fallback_tp2 = fallback(2.0)
    tp2 = structural_tp2
    if tp2 is None and farther(fallback_tp2, tp1):
        tp2 = fallback_tp2
    tp2_r = ratio(tp2) if tp2 is not None else None
    return tp1, tp2, tp1_r, tp2_r


def _plan_id(
    *,
    decision_close_timestamp: int,
    candidate: StrategyCandidate,
    style: str,
    order_type: str,
    entry: float,
    stop_loss: float,
    tp1: float,
    tp2: float | None,
) -> str:
    payload = {
        "decision_close_timestamp": decision_close_timestamp,
        "family": candidate.family,
        "setup_type": candidate.setup_type,
        "source_trigger_at": candidate.trigger_at,
        "style": style,
        "order_type": order_type,
        "entry": entry,
        "stop_loss": stop_loss,
        "tp1": tp1,
        "tp2": tp2,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    return f"stage2-{digest}"


def _make_plan(
    *,
    observation: PAObservationV1,
    candidate: StrategyCandidate,
    constraints: InstrumentConstraintsV1,
    style: str,
    order_type: str,
    entry: float,
    trigger: float | None,
    limit: float | None,
    buffer: float,
) -> OrderPlanCandidateV1 | None:
    stop_loss = _stop_loss(
        side=candidate.direction,
        entry=entry,
        invalidation=float(candidate.invalidation_anchor),
        buffer=buffer,
        minimum_distance=constraints.min_stop_distance,
    )
    if stop_loss is None:
        return None
    minimum_r = _FAMILY_MIN_R.get(candidate.family)
    if minimum_r is None:
        return None
    targets = _profit_targets(
        side=candidate.direction,
        entry=entry,
        stop_loss=stop_loss,
        anchors=candidate.target_anchors,
        minimum_r=minimum_r,
    )
    if targets is None:
        return None
    tp1, tp2, tp1_r, tp2_r = targets
    policy_origin = (
        "alphamaster_stage2_reversal"
        if candidate.family == "reversal"
        else "pa_formula_candidate"
    )
    return OrderPlanCandidateV1(
        plan_id=_plan_id(
            decision_close_timestamp=observation.bar_close_timestamp,
            candidate=candidate,
            style=style,
            order_type=order_type,
            entry=entry,
            stop_loss=stop_loss,
            tp1=tp1,
            tp2=tp2,
        ),
        style=style,
        side=candidate.direction,
        family=candidate.family,
        setup_type=candidate.setup_type,
        order_type=order_type,
        entry_price=entry,
        trigger_price=trigger,
        limit_price=limit,
        stop_loss=stop_loss,
        take_profit_1=tp1,
        take_profit_2=tp2,
        risk_distance=abs(entry - stop_loss),
        take_profit_1_r=tp1_r,
        take_profit_2_r=tp2_r,
        source_trigger_at=candidate.trigger_at,
        source_policy=candidate.source_policy,
        policy_origin=policy_origin,
        broker_validated=False,
        executable=False,
    )


def generate_order_plans(
    signal: FusionSignalV1,
    m15_pa: PAObservationV1,
    *,
    constraints: InstrumentConstraintsV1 | None = None,
) -> OrderPlanGenerationV1:
    """Generate no more than three theoretical price plans.

    The function has no volume, broker or MT5 dependency.  Broker limits may
    be supplied as immutable numerical constraints, but the output remains
    explicitly non-executable until the Stage-5 adapter revalidates it.
    """

    if not signal.accepted:
        return OrderPlanGenerationV1(plans=(), reject_reasons=("fusion_rejected",))
    if signal.side not in {"long", "short"}:
        return OrderPlanGenerationV1(plans=(), reject_reasons=("missing_direction",))
    candidates = tuple(
        candidate
        for candidate in signal.selected_m15_candidates
        if candidate.direction == signal.side
    )
    if not candidates:
        return OrderPlanGenerationV1(
            plans=(), reject_reasons=("no_fresh_m15_pa_candidate",)
        )
    candidate = sorted(candidates, key=_candidate_sort_key)[0]
    rules = constraints or InstrumentConstraintsV1(tick=m15_pa.tick)
    if not math.isclose(rules.tick, m15_pa.tick, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("instrument tick does not match PA observation")
    buffer = max(2.0 * rules.tick, 0.10 * m15_pa.atr)
    failures: list[str] = []
    plans: list[OrderPlanCandidateV1] = []

    def append_plan(plan: OrderPlanCandidateV1 | None, failure: str) -> None:
        if plan is None:
            _append_once(failures, failure)
        else:
            plans.append(plan)

    if not candidate.chase_forbidden:
        append_plan(
            _make_plan(
                observation=m15_pa,
                candidate=candidate,
                constraints=rules,
                style="pa_primary",
                order_type="market",
                entry=m15_pa.current_close,
                trigger=None,
                limit=None,
                buffer=buffer,
            ),
            "low_reward_risk",
        )

        if candidate.direction == "long":
            trigger = max(
                m15_pa.current_high + rules.tick,
                m15_pa.current_close + rules.min_pending_distance,
            )
            cap = trigger + buffer
        else:
            trigger = min(
                m15_pa.current_low - rules.tick,
                m15_pa.current_close - rules.min_pending_distance,
            )
            cap = trigger - buffer
        order_type = "stop_limit" if rules.supports_stop_limit else "stop"
        append_plan(
            _make_plan(
                observation=m15_pa,
                candidate=candidate,
                constraints=rules,
                style="confirmation_breakout",
                order_type=order_type,
                entry=trigger,
                trigger=trigger,
                limit=cap if rules.supports_stop_limit else None,
                buffer=buffer,
            ),
            "low_reward_risk",
        )

    if candidate.direction == "long":
        pullbacks = tuple(
            level
            for level in m15_pa.supports
            if candidate.invalidation_anchor < level < m15_pa.current_close
            and m15_pa.current_close - level >= rules.min_pending_distance
        )
        pullback = max(pullbacks) if pullbacks else None
    else:
        pullbacks = tuple(
            level
            for level in m15_pa.resistances
            if m15_pa.current_close < level < candidate.invalidation_anchor
            and level - m15_pa.current_close >= rules.min_pending_distance
        )
        pullback = min(pullbacks) if pullbacks else None
    if pullback is None:
        _append_once(failures, "no_pullback_anchor")
    else:
        append_plan(
            _make_plan(
                observation=m15_pa,
                candidate=candidate,
                constraints=rules,
                style="pullback",
                order_type="limit",
                entry=float(pullback),
                trigger=None,
                limit=float(pullback),
                buffer=buffer,
            ),
            "low_reward_risk",
        )

    return OrderPlanGenerationV1(
        plans=tuple(plans[:3]),
        reject_reasons=() if plans else tuple(failures or ("no_legal_order_plan",)),
    )
