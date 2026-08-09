"""Immutable Stage-4 trading configuration and read-only decision records."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, cast

from .fusion import (
    ALPHA_H1,
    ALPHA_M15,
    M5_ALPHA,
    PA_H1,
    PA_M15_CONTEXT,
    PA_M15_PATTERN,
    PA_M5_TIMING,
    default_module_selection,
)
from .models import ModuleSelectionV1, OrderPlanCandidateV1
from .decision_reviewer import DecisionReviewV1
from .time_alignment import TIMEFRAME_SECONDS


TRADING_CONFIG_VERSION = "trading-config-v1"
TRADE_DECISION_VERSION = "trade-decision-v1"
TRADING_SYMBOL = "XAUUSD"
TRADING_MODES = ("rules", "rules_codex")
DIRECTION_MODULE_IDS = (ALPHA_H1, PA_H1, ALPHA_M15, PA_M15_CONTEXT)
ENTRY_MODULE_IDS = (PA_M15_PATTERN, PA_M5_TIMING, ALPHA_M15, M5_ALPHA)
PA_FAMILY_IDS = ("trend_continuation", "breakout", "reversal", "range")


def _finite_number(value: object, *, field: str) -> float:
    if type(value) not in {int, float}:
        raise ValueError(f"{field} must be a finite number")
    number = float(cast(int | float, value))
    if not math.isfinite(number):
        raise ValueError(f"{field} must be a finite number")
    return number


def _validate_ids(
    values: tuple[str, ...],
    *,
    allowed: tuple[str, ...],
    field: str,
) -> tuple[str, ...]:
    if type(values) is not tuple or any(type(value) is not str for value in values):
        raise TypeError(f"{field} must be an exact tuple of strings")
    if not values:
        raise ValueError(f"{field} must not be empty")
    if len(set(values)) != len(values):
        raise ValueError(f"{field} must not contain duplicates")
    unknown = set(values).difference(allowed)
    if unknown:
        raise ValueError(f"unknown {field}: {sorted(unknown)}")
    return tuple(value for value in allowed if value in values)


@dataclass(frozen=True, slots=True)
class TradingConfigV1:
    """One immutable UI configuration snapshot applied at an M15 boundary."""

    mode: str
    direction_module_ids: tuple[str, ...]
    entry_module_ids: tuple[str, ...]
    pa_family_ids: tuple[str, ...]
    tp1_lots: float
    tp2_lots: float
    symbol: str = TRADING_SYMBOL
    config_version: str = TRADING_CONFIG_VERSION

    def __post_init__(self) -> None:
        if self.config_version != TRADING_CONFIG_VERSION:
            raise ValueError("unsupported trading config version")
        if self.symbol != TRADING_SYMBOL:
            raise ValueError(f"Stage 4 supports only {TRADING_SYMBOL}")
        if self.mode not in TRADING_MODES:
            raise ValueError(f"mode must be one of {TRADING_MODES}")
        direction = _validate_ids(
            self.direction_module_ids,
            allowed=DIRECTION_MODULE_IDS,
            field="direction_module_ids",
        )
        entry = _validate_ids(
            self.entry_module_ids,
            allowed=ENTRY_MODULE_IDS,
            field="entry_module_ids",
        )
        families = _validate_ids(
            self.pa_family_ids,
            allowed=PA_FAMILY_IDS,
            field="pa_family_ids",
        )
        tp1 = _finite_number(self.tp1_lots, field="tp1_lots")
        tp2 = _finite_number(self.tp2_lots, field="tp2_lots")
        if tp1 <= 0.0:
            raise ValueError("tp1_lots must be greater than zero")
        if tp2 < 0.0:
            raise ValueError("tp2_lots must be non-negative")
        object.__setattr__(self, "direction_module_ids", direction)
        object.__setattr__(self, "entry_module_ids", entry)
        object.__setattr__(self, "pa_family_ids", families)
        object.__setattr__(self, "tp1_lots", tp1)
        object.__setattr__(self, "tp2_lots", tp2)

    @classmethod
    def default(cls) -> "TradingConfigV1":
        selection = default_module_selection()
        return cls(
            mode="rules",
            direction_module_ids=tuple(
                module for module in DIRECTION_MODULE_IDS
                if module in selection.direction_modules
            ),
            entry_module_ids=tuple(
                module for module in ENTRY_MODULE_IDS
                if module in selection.entry_modules
            ),
            pa_family_ids=PA_FAMILY_IDS,
            tp1_lots=0.01,
            tp2_lots=0.0,
        )

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "TradingConfigV1":
        if type(payload) is not dict:
            raise TypeError("trading config payload must be an exact dictionary")
        config = cls(
            mode=payload.get("mode", ""),
            direction_module_ids=tuple(payload.get("direction_module_ids") or ()),
            entry_module_ids=tuple(payload.get("entry_module_ids") or ()),
            pa_family_ids=tuple(payload.get("pa_family_ids") or ()),
            tp1_lots=_finite_number(payload.get("tp1_lots"), field="tp1_lots"),
            tp2_lots=_finite_number(payload.get("tp2_lots"), field="tp2_lots"),
            symbol=payload.get("symbol", TRADING_SYMBOL),
            config_version=payload.get("config_version", TRADING_CONFIG_VERSION),
        )
        supplied_hash = payload.get("config_hash")
        if supplied_hash is not None and supplied_hash != config.config_hash:
            raise ValueError("trading config hash does not match its contents")
        return config

    def module_selection(self) -> ModuleSelectionV1:
        return ModuleSelectionV1(
            direction_modules=frozenset(self.direction_module_ids),
            entry_modules=frozenset(self.entry_module_ids),
        )

    def identity_payload(self) -> dict[str, Any]:
        return {
            "config_version": self.config_version,
            "symbol": self.symbol,
            "mode": self.mode,
            "direction_module_ids": list(self.direction_module_ids),
            "entry_module_ids": list(self.entry_module_ids),
            "pa_family_ids": list(self.pa_family_ids),
            "tp1_lots": self.tp1_lots,
            "tp2_lots": self.tp2_lots,
        }

    @property
    def config_hash(self) -> str:
        encoded = json.dumps(
            self.identity_payload(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def to_payload(self) -> dict[str, Any]:
        return {**self.identity_payload(), "config_hash": self.config_hash}


def _plan_payload(
    plan: OrderPlanCandidateV1,
    *,
    tp1_lots: float,
    tp2_lots: float,
) -> dict[str, Any]:
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
        "take_profit_1_r": plan.take_profit_1_r,
        "take_profit_2_r": plan.take_profit_2_r,
        "tp1_lots": tp1_lots,
        "tp2_lots": tp2_lots,
        "broker_validated": False,
        "executable": False,
    }


@dataclass(frozen=True, slots=True)
class TradeDecisionV1:
    """Read-only Stage-4 decision preview; execution fields are always empty."""

    decision_id: str
    symbol: str
    decision_close_timestamp: int
    h1_close_timestamp: int
    m15_close_timestamp: int
    m5_close_timestamp: int
    config: TradingConfigV1
    alpha_scores: tuple[tuple[str, float | None], ...]
    pa_scores: tuple[tuple[str, float | None], ...]
    pa_evidence: tuple[str, ...]
    input_warnings: tuple[str, ...]
    direction_score: float | None
    entry_score: float | None
    side: str | None
    fusion_accepted: bool
    fusion_reject_reasons: tuple[str, ...]
    plan_reject_reasons: tuple[str, ...]
    plans: tuple[OrderPlanCandidateV1, ...]
    review: DecisionReviewV1
    final_action: str
    created_at: float
    schema_version: str = TRADE_DECISION_VERSION
    mt5_ticket: int | None = None
    mt5_retcode: int | None = None

    def __post_init__(self) -> None:
        if self.schema_version != TRADE_DECISION_VERSION:
            raise ValueError("unsupported trade decision version")
        if self.symbol != TRADING_SYMBOL or self.config.symbol != self.symbol:
            raise ValueError("trade decision symbol does not match its configuration")
        if type(self.decision_close_timestamp) is not int:
            raise ValueError("decision close must be an integer Unix second")
        if self.m15_close_timestamp != self.decision_close_timestamp:
            raise ValueError("M15 close must equal the decision close")
        m15_interval_start = (
            self.decision_close_timestamp - TIMEFRAME_SECONDS["M15"]
        )
        if not (
            m15_interval_start
            < self.m5_close_timestamp
            <= self.decision_close_timestamp
        ):
            raise ValueError("M5 close must be inside the current M15 interval")
        if self.h1_close_timestamp > self.decision_close_timestamp:
            raise ValueError("H1 close cannot be later than the decision close")
        if self.review.decision_id != self.decision_id:
            raise ValueError("review does not belong to this decision")
        if self.final_action not in {"preview_approved", "preview_rejected"}:
            raise ValueError("invalid Stage-4 final action")
        if self.mt5_ticket is not None or self.mt5_retcode is not None:
            raise ValueError("Stage-4 previews cannot contain MT5 execution results")
        _finite_number(self.created_at, field="created_at")

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "decision_id": self.decision_id,
            "symbol": self.symbol,
            "decision_close_timestamp": self.decision_close_timestamp,
            "close_times": {
                "H1": self.h1_close_timestamp,
                "M15": self.m15_close_timestamp,
                "M5": self.m5_close_timestamp,
            },
            "config_version": self.config.config_version,
            "config_hash": self.config.config_hash,
            "mode": self.config.mode,
            "module_selection": {
                "direction": list(self.config.direction_module_ids),
                "entry": list(self.config.entry_module_ids),
                "pa_families": list(self.config.pa_family_ids),
            },
            "scores": {
                "direction": self.direction_score,
                "entry": self.entry_score,
                "alpha": dict(self.alpha_scores),
                "pa": dict(self.pa_scores),
            },
            "pa_evidence": list(self.pa_evidence),
            "input_warnings": list(self.input_warnings),
            "fusion": {
                "accepted": self.fusion_accepted,
                "side": self.side,
                "reject_reasons": list(self.fusion_reject_reasons),
            },
            "plan_reject_reasons": list(self.plan_reject_reasons),
            "plans": [
                _plan_payload(
                    plan,
                    tp1_lots=self.config.tp1_lots,
                    tp2_lots=self.config.tp2_lots,
                )
                for plan in self.plans
            ],
            "review": {
                "verdict": self.review.verdict,
                "selected_plan_id": self.review.selected_plan_id,
                "reason_code": self.review.reason_code,
                "summary_zh": self.review.summary_zh,
            },
            "final_action": self.final_action,
            "created_at": self.created_at,
            "execution": {
                "enabled": False,
                "ticket": None,
                "retcode": None,
            },
        }
