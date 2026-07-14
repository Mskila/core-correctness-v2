import math

import torch


EMA_TAIL_WEIGHT_THRESHOLD = 1e-6


def _signed_log_add(
    left_sign: torch.Tensor,
    left_log: torch.Tensor,
    right_sign: torch.Tensor,
    right_log: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Add signed log-magnitudes without materializing unsafe values."""
    left_active = left_sign != 0
    right_active = right_sign != 0
    both_active = left_active & right_active
    same_sign = both_active & (left_sign == right_sign)
    opposite_sign = both_active & ~same_sign

    result_sign = torch.where(left_active, left_sign, right_sign)
    result_log = torch.where(left_active, left_log, right_log)
    result_log = torch.where(
        same_sign,
        torch.logaddexp(left_log, right_log),
        result_log,
    )

    left_is_larger = left_log > right_log
    larger_log = torch.where(left_is_larger, left_log, right_log)
    smaller_log = torch.where(left_is_larger, right_log, left_log)
    larger_sign = torch.where(left_is_larger, left_sign, right_sign)
    unequal_opposites = opposite_sign & (left_log != right_log)
    safe_larger_log = torch.where(
        unequal_opposites, larger_log, torch.zeros_like(larger_log)
    )
    log_ratio = torch.where(
        unequal_opposites,
        smaller_log - larger_log,
        -torch.ones_like(smaller_log),
    )
    difference_log = safe_larger_log + torch.log(
        -torch.expm1(log_ratio)
    )
    result_sign = torch.where(unequal_opposites, larger_sign, result_sign)
    result_log = torch.where(unequal_opposites, difference_log, result_log)

    exact_cancellation = opposite_sign & (left_log == right_log)
    result_sign = torch.where(
        exact_cancellation,
        torch.zeros_like(result_sign),
        result_sign,
    )
    result_log = torch.where(
        exact_cancellation,
        torch.full_like(result_log, -torch.inf),
        result_log,
    )
    return result_sign, result_log


def _sum_signed_logs(
    signs: torch.Tensor,
    log_magnitudes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sum causal-window contributions while retaining an absorbed residual."""
    high_sign = torch.zeros_like(log_magnitudes[..., 0])
    high_log = torch.full_like(high_sign, -torch.inf)
    residual_sign = torch.zeros_like(high_sign)
    residual_log = torch.full_like(high_sign, -torch.inf)

    for index in range(log_magnitudes.shape[-1]):
        offset = log_magnitudes.shape[-1] - 1 - index
        if offset >= log_magnitudes.shape[1]:
            term_log = torch.full_like(high_log, -torch.inf)
            term_sign = torch.zeros_like(high_sign)
        elif offset:
            term_log = torch.nn.functional.pad(
                log_magnitudes[:, offset:, index],
                (0, offset),
                value=-torch.inf,
            )
            term_sign = torch.nn.functional.pad(
                signs[:, offset:, index],
                (0, offset),
                value=0,
            )
        else:
            term_log = log_magnitudes[..., index]
            term_sign = signs[..., index]
        high_active = high_sign != 0
        term_active = torch.isfinite(term_log) & (term_sign != 0)
        both_active = high_active & term_active
        same_magnitude = both_active & (high_log == term_log)
        exact_opposites = same_magnitude & (high_sign == -term_sign)
        exact_same_sign = same_magnitude & (high_sign == term_sign)

        merged_sign, merged_log = _signed_log_add(
            high_sign,
            high_log,
            term_sign,
            term_log,
        )
        term_was_absorbed = (
            both_active
            & ~same_magnitude
            & (merged_sign == high_sign)
            & (merged_log == high_log)
        )
        high_was_absorbed = (
            both_active
            & ~same_magnitude
            & (merged_sign == term_sign)
            & (merged_log == term_log)
        )
        defer_term = exact_same_sign | term_was_absorbed
        deferred_sign = torch.where(
            defer_term,
            term_sign,
            torch.where(
                high_was_absorbed,
                high_sign,
                torch.zeros_like(high_sign),
            ),
        )
        deferred_log = torch.where(
            defer_term,
            term_log,
            torch.where(
                high_was_absorbed,
                high_log,
                torch.full_like(high_log, -torch.inf),
            ),
        )
        next_residual_sign, next_residual_log = _signed_log_add(
            residual_sign,
            residual_log,
            deferred_sign,
            deferred_log,
        )

        next_high_sign = torch.where(
            exact_same_sign | term_was_absorbed,
            high_sign,
            torch.where(high_was_absorbed, term_sign, merged_sign),
        )
        next_high_log = torch.where(
            exact_same_sign | term_was_absorbed,
            high_log,
            torch.where(high_was_absorbed, term_log, merged_log),
        )
        next_high_sign = torch.where(
            exact_opposites,
            residual_sign,
            next_high_sign,
        )
        next_high_log = torch.where(
            exact_opposites,
            residual_log,
            next_high_log,
        )
        next_residual_sign = torch.where(
            exact_opposites,
            torch.zeros_like(next_residual_sign),
            next_residual_sign,
        )
        next_residual_log = torch.where(
            exact_opposites,
            torch.full_like(next_residual_log, -torch.inf),
            next_residual_log,
        )

        residual_is_larger = (next_residual_sign != 0) & (
            (next_high_sign == 0) | (next_residual_log > next_high_log)
        )
        high_sign = torch.where(
            residual_is_larger, next_residual_sign, next_high_sign
        )
        high_log = torch.where(
            residual_is_larger, next_residual_log, next_high_log
        )
        residual_sign = torch.where(
            residual_is_larger, next_high_sign, next_residual_sign
        )
        residual_log = torch.where(
            residual_is_larger, next_high_log, next_residual_log
        )

    return _signed_log_add(
        high_sign,
        high_log,
        residual_sign,
        residual_log,
    )


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
        # For a non-current element k, the standardized-current Jacobian is
        #   -sum_i((x_i-x_t)(x_i-x_k)) / (n * sum_i((x_i-mean)^2)).
        # Writing the numerator with pairwise differences makes repeated-value
        # structural zeros exact instead of amplifying a rounded mean residual.
        relative_to_current = (
            centered - centered[..., -1:]
        ) * valid_work
        relative_sum = relative_to_current.sum(dim=-1, keepdim=True)
        relative_square_sum = relative_to_current.square().sum(
            dim=-1, keepdim=True
        )
        pairwise_numerator = (
            relative_square_sum - relative_to_current * relative_sum
        )
        centered_square_sum = centered.square().sum(dim=-1, keepdim=True)
        denominator = count.unsqueeze(-1) * centered_square_sum
        safe_denominator = torch.where(
            has_variance.unsqueeze(-1),
            denominator,
            torch.ones_like(denominator),
        )
        noncurrent_mask = valid_work.bool() & (
            torch.arange(window, device=grad.device) != window - 1
        ).reshape(1, 1, -1)
        noncurrent_numerator = torch.where(
            noncurrent_mask & has_variance.unsqueeze(-1),
            -pairwise_numerator / safe_denominator,
            torch.zeros_like(pairwise_numerator),
        )
        current_numerator = -noncurrent_numerator.sum(dim=-1, keepdim=True)
        numerator = noncurrent_numerator + current_numerator * (
            torch.arange(window, device=grad.device) == window - 1
        ).reshape(1, 1, -1)
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
        product_arithmetic_safe = (
            torch.isfinite(direct_numerator)
            & (direct_numerator != 0)
            & torch.isfinite(direct_divisor)
            & (direct_divisor > 0)
        )
        factorized_contribution = (
            grad.unsqueeze(-1) / safe_scale.unsqueeze(-1)
        ) * (numerator / std.unsqueeze(-1))
        factorized_arithmetic_safe = (
            torch.isfinite(factorized_contribution)
            & (factorized_contribution != 0)
        )
        direct_arithmetic_safe = (
            product_arithmetic_safe | factorized_arithmetic_safe
        )
        row_is_direct = (
            (~active)
            | (
                (log_magnitude <= per_contribution_limit)
                & direct_arithmetic_safe
            )
        ).all(dim=-1)
        direct_mask = active & row_is_direct.unsqueeze(-1)
        product_divisor = torch.where(
            direct_mask & product_arithmetic_safe,
            direct_divisor,
            torch.ones_like(log_magnitude),
        )
        product_contribution = direct_numerator / product_divisor
        direct_contribution = torch.where(
            direct_mask,
            torch.where(
                product_arithmetic_safe,
                product_contribution,
                factorized_contribution,
            ),
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

        unsafe_mask = active & ~row_is_direct.unsqueeze(-1)
        unsafe_logs = torch.where(
            unsafe_mask,
            log_magnitude,
            torch.full_like(log_magnitude, -torch.inf),
        )
        unsafe_signs = torch.where(
            unsafe_mask,
            sign,
            torch.zeros_like(sign),
        )
        start = window - 1
        aggregate_sign, aggregate_log = _sum_signed_logs(
            unsafe_signs,
            unsafe_logs,
        )
        unsafe_contribution = unsafe_mask.to(grad.dtype)
        unsafe_padded = torch.zeros(
            n_rows, padded_steps, dtype=grad.dtype, device=grad.device
        )
        unsafe_padded.scatter_add_(
            1, flat_indices, unsafe_contribution.reshape(n_rows, -1)
        )
        unsafe_result = (
            aggregate_sign
            * torch.exp(torch.minimum(aggregate_log, log_limit))
        )

        direct_result = direct_padded[:, start : start + time_steps]
        input_has_unsafe = (
            unsafe_padded[:, start : start + time_steps] != 0
        )
        direct_sign = direct_result.sign()
        direct_log = torch.where(
            direct_sign != 0,
            direct_result.abs().log(),
            torch.full_like(direct_result, -torch.inf),
        )
        combined_sign, combined_log = _signed_log_add(
            aggregate_sign,
            aggregate_log,
            direct_sign,
            direct_log,
        )
        combined_result = combined_sign * torch.exp(
            torch.minimum(combined_log, log_limit)
        )
        combined_result = torch.where(
            aggregate_sign == 0,
            direct_result,
            torch.where(direct_sign == 0, unsafe_result, combined_result),
        )
        input_gradient = torch.where(
            input_has_unsafe,
            combined_result,
            direct_result,
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
