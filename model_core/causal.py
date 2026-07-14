import math

import torch


EMA_TAIL_WEIGHT_THRESHOLD = 1e-6


class _CausalRollingZScore(torch.autograd.Function):
    """Numerically stable rolling z-score with an analytic backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, window: int) -> torch.Tensor:
        original_dtype = x.dtype
        work = x.to(torch.float64)
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

        # Relative scaling is exact for a z-score and keeps both very large and
        # subnormal, but non-zero, prefixes representable during the reduction.
        scale = (windows.abs() * valid_work).amax(dim=-1)
        safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))
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

        ctx.input_dtype = original_dtype
        ctx.window = window
        ctx.save_for_backward(
            centered,
            valid_work,
            count,
            std,
            zscore,
            safe_scale,
            has_variance,
        )
        return zscore.to(original_dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (
            centered,
            valid_work,
            count,
            std,
            zscore,
            safe_scale,
            has_variance,
        ) = ctx.saved_tensors
        grad = grad_output.to(torch.float64)
        if not torch.isfinite(grad).all():
            raise FloatingPointError(
                "causal_rolling_zscore requires finite output gradients"
            )

        window = ctx.window
        n_rows, time_steps, _ = centered.shape
        normalized_window = centered / std.unsqueeze(-1)
        numerator = (
            torch.nn.functional.one_hot(
                torch.full(
                    (time_steps,),
                    window - 1,
                    dtype=torch.long,
                    device=grad.device,
                ),
                num_classes=window,
            )
            .to(grad.dtype)
            .unsqueeze(0)
            - valid_work / count.unsqueeze(-1)
            - zscore.unsqueeze(-1)
            * normalized_window
            / count.unsqueeze(-1)
        )
        active = (
            valid_work.bool()
            & has_variance.unsqueeze(-1)
            & (numerator != 0)
            & (grad.unsqueeze(-1) != 0)
        )

        # Each output contributes numerator / (scale * std) to its causal
        # window. Form ordinary, representable rows directly so their gradient
        # remains bit-for-bit governed by the analytic formula. For subnormal
        # scales, combine signs and magnitudes in log space before saturating a
        # genuinely unrepresentable final gradient at the input dtype boundary.
        sign = grad.unsqueeze(-1).sign() * numerator.sign()
        log_magnitude = (
            grad.unsqueeze(-1).abs().log()
            + numerator.abs().log()
            - std.unsqueeze(-1).log()
            - safe_scale.unsqueeze(-1).log()
        )
        log_magnitude = torch.where(
            active,
            log_magnitude,
            torch.full_like(log_magnitude, -torch.inf),
        )

        gradient_limit = torch.tensor(
            torch.finfo(ctx.input_dtype).max,
            dtype=grad.dtype,
            device=grad.device,
        )
        log_limit = gradient_limit.log()
        per_contribution_limit = log_limit - math.log(window + 1)
        direct_numerator = grad.unsqueeze(-1) * numerator
        direct_divisor = safe_scale.unsqueeze(-1) * std.unsqueeze(-1)
        direct_arithmetic_safe = (
            torch.isfinite(direct_numerator)
            & (direct_numerator != 0)
            & torch.isfinite(direct_divisor)
            & (direct_divisor > 0)
        )
        row_is_direct = (
            (~active)
            | (
                (log_magnitude <= per_contribution_limit)
                & direct_arithmetic_safe
            )
        ).all(dim=-1)
        direct_mask = active & row_is_direct.unsqueeze(-1)
        divisor = torch.where(
            direct_mask,
            direct_divisor,
            torch.ones_like(log_magnitude),
        )
        direct_contribution = torch.where(
            direct_mask,
            direct_numerator / divisor,
            torch.zeros_like(log_magnitude),
        )

        padded_steps = time_steps + window - 1
        indices = (
            torch.arange(time_steps, device=grad.device).unsqueeze(-1)
            + torch.arange(window, device=grad.device).unsqueeze(0)
        )
        flat_indices = indices.reshape(1, -1).expand(n_rows, -1)
        direct_padded = torch.zeros(
            n_rows, padded_steps, dtype=grad.dtype, device=grad.device
        )
        direct_padded.scatter_add_(
            1, flat_indices, direct_contribution.reshape(n_rows, -1)
        )

        log_padded = torch.full(
            (n_rows, padded_steps),
            -torch.inf,
            dtype=grad.dtype,
            device=grad.device,
        )
        log_padded.scatter_reduce_(
            1,
            flat_indices,
            log_magnitude.reshape(n_rows, -1),
            reduce="amax",
            include_self=True,
        )
        gathered_max = log_padded.gather(1, flat_indices).reshape_as(
            log_magnitude
        )
        finite_max = torch.where(
            torch.isfinite(gathered_max),
            gathered_max,
            torch.zeros_like(gathered_max),
        )
        scaled_signed = torch.where(
            active,
            sign * torch.exp(log_magnitude - finite_max),
            torch.zeros_like(log_magnitude),
        )
        signed_padded = torch.zeros_like(log_padded)
        signed_padded.scatter_add_(
            1, flat_indices, scaled_signed.reshape(n_rows, -1)
        )
        unsafe_contribution = (
            active & ~row_is_direct.unsqueeze(-1)
        ).to(grad.dtype)
        unsafe_padded = torch.zeros_like(log_padded)
        unsafe_padded.scatter_add_(
            1, flat_indices, unsafe_contribution.reshape(n_rows, -1)
        )
        total_log_magnitude = log_padded + signed_padded.abs().log()
        log_result = (
            signed_padded.sign()
            * torch.minimum(
                total_log_magnitude.exp(),
                gradient_limit,
            )
        )

        start = window - 1
        direct_result = direct_padded[:, start : start + time_steps]
        log_result = log_result[:, start : start + time_steps]
        input_is_direct = (
            unsafe_padded[:, start : start + time_steps] == 0
        )
        input_gradient = torch.where(
            input_is_direct,
            direct_result,
            log_result,
        ).clamp(-gradient_limit, gradient_limit)
        if not torch.isfinite(input_gradient).all():
            raise FloatingPointError(
                "causal_rolling_zscore produced a non-finite gradient"
            )
        return input_gradient.to(ctx.input_dtype), None


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
    zscore = _CausalRollingZScore.apply(x, window)
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
