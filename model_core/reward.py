"""Pure, sign-safe reward gates and timeframe reward helpers."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math

import torch

from data_pipeline.validation import DataValidationError, normalize_timeframe_name


logger = logging.getLogger(__name__)

_BARS_PER_DAY = {
    "M1": 1_440.0,
    "M5": 288.0,
    "M15": 96.0,
    "M30": 48.0,
    "H1": 24.0,
    "H4": 6.0,
    "D1": 1.0,
}


@dataclass(frozen=True)
class FloatGateResult:
    base: float
    adjustment: float
    final: float


@dataclass(frozen=True)
class TensorGateResult:
    base: torch.Tensor
    adjustment: torch.Tensor
    final: torch.Tensor


def _finite_number(value: object, *, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise DataValidationError(f"{name} must be a finite number")
    return float(value)


def apply_oos_gate(
    base_score: float,
    oos_sortino: float,
    *,
    scale: float = 0.5,
) -> FloatGateResult:
    """Apply a bounded additive OOS adjustment independent of base sign."""
    base = _finite_number(base_score, name="base_score")
    sortino = _finite_number(oos_sortino, name="oos_sortino")
    gate_scale = _finite_number(scale, name="oos_gate_scale")
    if gate_scale < 0.0:
        raise DataValidationError("oos_gate_scale must be non-negative")
    adjustment = gate_scale * math.tanh(sortino)
    final = base + adjustment
    if not math.isfinite(final):
        raise DataValidationError("OOS gate final score must be finite")
    logger.debug(
        "OOS gate base=%s adjustment=%s final=%s sortino=%s",
        base,
        adjustment,
        final,
        sortino,
    )
    return FloatGateResult(base=base, adjustment=adjustment, final=final)


def apply_ic_gate(
    reward: torch.Tensor,
    ic_mean: float | torch.Tensor,
    *,
    threshold: float = 0.01,
    positive_multiplier: float = 1.15,
    negative_multiplier: float = 0.75,
    scale_floor: float = 0.0,
) -> TensorGateResult:
    """Apply an additive IC adjustment whose sign follows IC, not reward."""
    if type(reward) is not torch.Tensor or reward.shape != torch.Size([]):
        raise DataValidationError("reward must be a scalar tensor")
    if not reward.is_floating_point() or not bool(torch.isfinite(reward).item()):
        raise DataValidationError("reward must be a finite floating scalar tensor")
    ic_value = (
        float(ic_mean.item())
        if type(ic_mean) is torch.Tensor and ic_mean.shape == torch.Size([])
        else _finite_number(ic_mean, name="ic_mean")
    )
    if not math.isfinite(ic_value):
        raise DataValidationError("ic_mean must be finite")
    gate_threshold = _finite_number(threshold, name="ic_gate_threshold")
    positive = _finite_number(
        positive_multiplier, name="ic_gate_positive_multiplier"
    )
    negative = _finite_number(
        negative_multiplier, name="ic_gate_negative_multiplier"
    )
    floor = _finite_number(scale_floor, name="ic_gate_scale_floor")
    if gate_threshold < 0.0 or positive < 1.0 or not 0.0 <= negative <= 1.0:
        raise DataValidationError("IC gate strengths are outside their valid ranges")
    if floor < 0.0:
        raise DataValidationError("ic_gate_scale_floor must be non-negative")

    scale = torch.clamp(reward.abs(), min=floor)
    if ic_value > gate_threshold:
        adjustment = scale * (positive - 1.0)
    elif ic_value < -gate_threshold:
        adjustment = -scale * (1.0 - negative)
    else:
        adjustment = torch.zeros_like(reward)
    final = reward + adjustment
    if not bool(torch.isfinite(final).item()):
        raise DataValidationError("IC gate final score must be finite")
    logger.debug(
        "IC gate base=%s adjustment=%s final=%s ic=%s",
        float(reward.detach().cpu()),
        float(adjustment.detach().cpu()),
        float(final.detach().cpu()),
        ic_value,
    )
    return TensorGateResult(base=reward, adjustment=adjustment, final=final)


def target_bars_per_trade(timeframe: str | int, trades_per_day: float) -> float:
    """Convert one daily activity target into bars per target trade."""
    canonical = normalize_timeframe_name(timeframe)
    target = _finite_number(trades_per_day, name="target_trades_per_day")
    if target <= 0.0:
        raise DataValidationError("target_trades_per_day must be positive")
    try:
        bars_per_day = _BARS_PER_DAY[canonical]
    except KeyError as exc:
        raise DataValidationError(
            f"timeframe reward target is unsupported: {canonical}"
        ) from exc
    return bars_per_day / target
