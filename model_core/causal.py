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

    # CPU reductions for half types are both more stable and more broadly
    # supported in float32. Float32/float64 retain their native semantics.
    work_dtype = (
        torch.float32
        if x.dtype in (torch.float16, torch.bfloat16)
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

    # A native variance reduction is safe when every value in the window is
    # below half of sqrt(finfo.max / window): even the conservative bound
    # ``window * (2 * max_abs) ** 2`` then remains representable. Build the
    # risk mask causally from detached magnitudes so ordinary float32 keeps the
    # original graph, while a large future value can only switch its own and
    # later windows to the high-precision fallback.
    use_high_precision_fallback = False
    unsafe_windows = None
    if work.dtype == torch.float32:
        native_limit = 0.5 * math.sqrt(
            torch.finfo(work.dtype).max / window
        )
        risky_values = work.detach().abs() >= native_limit
        risk_prefix = risky_values.to(torch.int64).cumsum(dim=1)
        if window >= time_steps:
            prior_risk = torch.zeros_like(risk_prefix)
        else:
            prior_risk = torch.cat(
                (
                    torch.zeros(
                        (n_rows, window),
                        dtype=risk_prefix.dtype,
                        device=work.device,
                    ),
                    risk_prefix[:, :-window],
                ),
                dim=1,
            )
        unsafe_windows = (risk_prefix - prior_risk) > 0
        use_high_precision_fallback = bool(unsafe_windows.any())

    statistics_windows = (
        torch.where(
            (~unsafe_windows).unsqueeze(-1),
            windows,
            torch.zeros_like(windows),
        )
        if use_high_precision_fallback
        else windows
    )
    means = (statistics_windows * weights).sum(dim=-1) / counts
    if work.dtype == torch.float64 and not torch.isfinite(means).all():
        raise FloatingPointError(
            "causal_rolling_zscore rolling mean is not finite"
        )
    centered = (statistics_windows - means.unsqueeze(-1)) * weights
    if use_high_precision_fallback:
        safe_windows = ~unsafe_windows
    else:
        safe_windows = None
    if work.dtype == torch.float64 and not torch.isfinite(centered).all():
        raise FloatingPointError(
            "causal_rolling_zscore centered values are not finite"
        )
    variances = centered.square().sum(dim=-1) / counts
    if work.dtype == torch.float64 and not torch.isfinite(variances).all():
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
    if use_high_precision_fallback:
        active = active & safe_windows
    denominators = torch.where(
        active, standard_deviations, torch.ones_like(standard_deviations)
    )
    native_means = (
        torch.where(safe_windows, means, work)
        if use_high_precision_fallback
        else means
    )
    normalized = (work - native_means) / denominators
    result = torch.where(active, normalized, torch.zeros_like(normalized))

    if use_high_precision_fallback:
        fallback_windows = windows[unsafe_windows].to(torch.float64)
        fallback_weights = weights[unsafe_windows].to(torch.float64)
        fallback_counts = counts[unsafe_windows].to(torch.float64)
        fallback_means = (
            (fallback_windows * fallback_weights).sum(dim=-1)
            / fallback_counts
        )
        fallback_centered = (
            fallback_windows - fallback_means.unsqueeze(-1)
        ) * fallback_weights
        fallback_variances = (
            fallback_centered.square().sum(dim=-1) / fallback_counts
        )
        positive_fallback_variance = fallback_variances > 0
        fallback_sqrt_input = torch.where(
            positive_fallback_variance,
            fallback_variances,
            torch.ones_like(fallback_variances),
        )
        fallback_standard_deviations = fallback_sqrt_input.sqrt()
        fallback_active = positive_fallback_variance & (
            fallback_standard_deviations > CAUSAL_ZSCORE_STD_THRESHOLD
        )
        fallback_denominators = torch.where(
            fallback_active,
            fallback_standard_deviations,
            torch.ones_like(fallback_standard_deviations),
        )
        fallback_current = work[unsafe_windows].to(torch.float64)
        fallback_normalized = (
            fallback_current - fallback_means
        ) / fallback_denominators
        fallback_result = torch.where(
            fallback_active,
            fallback_normalized,
            torch.zeros_like(fallback_normalized),
        ).to(result.dtype)
        result = result + torch.zeros_like(result).masked_scatter(
            unsafe_windows, fallback_result
        )

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
