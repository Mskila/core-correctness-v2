import math

import torch


EMA_TAIL_WEIGHT_THRESHOLD = 1e-6


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
    """Return a causal rolling z-score for a two-dimensional ``[N, T]`` tensor.

    Warm-up windows contain only values that have actually appeared. Left-side
    padding exists solely to form windows and is excluded from the statistics.
    """
    if (
        x.ndim != 2
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

    original_dtype = x.dtype
    # Keep hooks on a private work tensor: for float64 input, ``to`` alone may
    # return the caller's tensor and would otherwise affect unrelated branches.
    work = x.to(torch.float64).clone()
    if work.requires_grad:
        # Subnormal inputs can imply a derivative larger than their dtype can
        # represent. Saturate only at that dtype's representability boundary.
        gradient_limit = torch.finfo(original_dtype).max

        def _keep_gradient_representable(gradient: torch.Tensor) -> torch.Tensor:
            if torch.isnan(gradient).any():
                raise FloatingPointError(
                    "causal_rolling_zscore produced a NaN gradient"
                )
            return gradient.clamp(-gradient_limit, gradient_limit)

        work.register_hook(_keep_gradient_representable)
    pad = torch.zeros(
        x.shape[0], window - 1, dtype=work.dtype, device=x.device
    )
    valid_pad = torch.zeros(
        x.shape[0], window - 1, dtype=torch.bool, device=x.device
    )
    windows = torch.cat([pad, work], dim=1).unfold(1, window, 1)
    valid = torch.cat(
        [valid_pad, torch.ones_like(x, dtype=torch.bool)], dim=1
    ).unfold(1, window, 1)
    valid_work = valid.to(work.dtype)
    count = valid.sum(dim=-1).clamp_min(1).to(work.dtype)

    # Per-window relative scaling prevents both square overflow and loss of
    # representable small-scale variation. Only a truly all-zero prefix uses
    # the neutral divisor; there is no absolute variance threshold.
    scale = (windows.abs() * valid_work).amax(dim=-1)
    safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale)).detach()
    scaled = windows / safe_scale.unsqueeze(-1)
    mean = (scaled * valid_work).sum(dim=-1) / count
    centered = (scaled - mean.unsqueeze(-1)) * valid_work
    variance = centered.square().sum(dim=-1) / count
    has_variance = variance > 0
    safe_variance = torch.where(
        has_variance, variance, torch.ones_like(variance)
    )
    std = safe_variance.sqrt()
    zscore = torch.where(
        has_variance,
        (work / safe_scale - mean) / std,
        torch.zeros_like(work),
    )
    if not torch.isfinite(zscore).all():
        raise FloatingPointError(
            "causal_rolling_zscore produced a non-finite result"
        )
    result = zscore.to(original_dtype)
    if not torch.isfinite(result).all():
        raise FloatingPointError(
            "causal_rolling_zscore result is not representable in input dtype"
        )
    return result
