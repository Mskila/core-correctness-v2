"""Shared causal tensor operations used by features, operators, and the VM."""

import math

import torch


EMA_TAIL_WEIGHT_THRESHOLD = 1e-6
CAUSAL_ZSCORE_STD_THRESHOLD = 1e-6


def ema_effective_window(span: int) -> int:
    """Return the finite EMA kernel implied by the V2 tail cutoff."""
    if isinstance(span, bool) or not isinstance(span, int) or span < 1:
        raise ValueError("EMA span must be a positive integer")
    alpha = 2.0 / (span + 1.0)
    if alpha >= 1.0:
        return 1
    return max(
        1,
        math.ceil(
            -math.log(EMA_TAIL_WEIGHT_THRESHOLD)
            / (-math.log(1.0 - alpha))
        ),
    )


def causal_rolling_zscore(x: torch.Tensor, window: int = 200) -> torch.Tensor:
    """Return causal population z-scores for a floating ``[N, T]`` tensor.

    Warm-up windows contain only values that have actually appeared. Left-side
    padding is used solely to form fixed-width windows and is excluded from all
    statistics. Prefixes whose population standard deviation is not strictly
    greater than ``1e-6`` produce zero.
    """
    if (
        not isinstance(x, torch.Tensor)
        or x.ndim != 2
        or x.shape[1] == 0
        or isinstance(window, bool)
        or not isinstance(window, int)
        or window < 1
    ):
        raise ValueError(
            "causal_rolling_zscore expects [N,T] and integer window >= 1"
        )
    if not torch.is_floating_point(x):
        raise ValueError("causal_rolling_zscore expects a floating [N,T] tensor")
    if not torch.isfinite(x).all():
        raise FloatingPointError(
            "causal_rolling_zscore requires finite input values"
        )

    # Float64 work values keep every finite float32 square representable while
    # preserving the threshold in the input's original units. Outputs are cast
    # back only after the rolling statistics have been validated.
    work_dtype = (
        torch.float64
        if x.dtype in (torch.float16, torch.bfloat16, torch.float32)
        else x.dtype
    )
    work = x.to(work_dtype)
    n_rows, time_steps = work.shape

    padding = torch.zeros(
        (n_rows, window - 1), dtype=work.dtype, device=work.device
    )
    padded = torch.cat((padding, work), dim=1)
    windows = padded.unfold(dimension=1, size=window, step=1).contiguous()

    invalid = torch.zeros(
        (n_rows, window - 1), dtype=torch.bool, device=work.device
    )
    observed = torch.ones(
        (n_rows, time_steps), dtype=torch.bool, device=work.device
    )
    valid = (
        torch.cat((invalid, observed), dim=1)
        .unfold(1, window, 1)
        .contiguous()
    )
    weights = valid.to(work.dtype)
    counts = weights.sum(dim=-1).clamp_min(1.0)

    means = (windows * weights).sum(dim=-1) / counts
    if not torch.isfinite(means).all():
        raise FloatingPointError(
            "causal_rolling_zscore rolling mean is not finite"
        )
    centered = (windows - means.unsqueeze(-1)) * weights
    if not torch.isfinite(centered).all():
        raise FloatingPointError(
            "causal_rolling_zscore centered values are not finite"
        )
    variances = centered.square().sum(dim=-1) / counts
    if not torch.isfinite(variances).all():
        raise FloatingPointError(
            "causal_rolling_zscore rolling variance is not finite"
        )

    # Avoid sqrt(0) in the graph: torch.where selects values in forward, but
    # an unsafe inactive expression can still poison backward with 0 * inf.
    positive_variance = variances > 0
    sqrt_input = torch.where(
        positive_variance, variances, torch.ones_like(variances)
    )
    standard_deviations = sqrt_input.sqrt()
    active = positive_variance & (
        standard_deviations > CAUSAL_ZSCORE_STD_THRESHOLD
    )
    denominators = torch.where(
        active, standard_deviations, torch.ones_like(standard_deviations)
    )
    normalized = (work - means) / denominators
    result = torch.where(active, normalized, torch.zeros_like(normalized))

    if not torch.isfinite(result).all():
        raise FloatingPointError(
            "causal_rolling_zscore produced a non-finite result"
        )
    result = result.to(x.dtype)
    if not torch.isfinite(result).all():
        raise FloatingPointError(
            "causal_rolling_zscore result is not representable in input dtype"
        )
    return result
