import math

import torch


EMA_TAIL_WEIGHT_THRESHOLD = 1e-6
_LOG_ACCUMULATOR_RADIX_BITS = 30
_LOG_ACCUMULATOR_LIMBS = 256
_LOG_ACCUMULATOR_SIGNIFICAND_BITS = 52


def _sum_signed_logs(
    signs: torch.Tensor,
    log_magnitudes: torch.Tensor,
    direct_contributions: torch.Tensor,
    direct_mask: torch.Tensor,
    gradient_limit: torch.Tensor,
) -> torch.Tensor:
    """Exactly reduce aligned extreme-scale terms with a fixed superaccumulator."""
    n_rows, time_steps, window = log_magnitudes.shape
    padded_steps = time_steps + window - 1
    target_indices = (
        torch.arange(time_steps, device=signs.device).unsqueeze(-1)
        + torch.arange(window, device=signs.device).unsqueeze(0)
    )
    row_offsets = (
        torch.arange(n_rows, device=signs.device) * padded_steps
    ).reshape(-1, 1, 1)
    flat_targets = (
        target_indices.reshape(1, time_steps, window) + row_offsets
    ).reshape(-1)

    symbolic_active = (
        ~direct_mask
        & (signs != 0)
        & torch.isfinite(log_magnitudes)
    ).reshape(-1)
    safe_logs = torch.where(
        symbolic_active,
        log_magnitudes.reshape(-1),
        torch.zeros_like(log_magnitudes.reshape(-1)),
    )
    log2 = math.log(2.0)
    symbolic_exponents = torch.floor(safe_logs / log2).to(torch.int64)
    remainders = (
        safe_logs - symbolic_exponents.to(safe_logs.dtype) * log2
    )
    symbolic_significands = torch.round(
        torch.exp(remainders)
        * (1 << _LOG_ACCUMULATOR_SIGNIFICAND_BITS)
    ).to(torch.int64)
    below_unit = symbolic_significands < (
        1 << _LOG_ACCUMULATOR_SIGNIFICAND_BITS
    )
    symbolic_significands = torch.where(
        below_unit,
        symbolic_significands * 2,
        symbolic_significands,
    )
    symbolic_exponents = torch.where(
        below_unit,
        symbolic_exponents - 1,
        symbolic_exponents,
    )
    rounded_up = symbolic_significands >= (
        1 << (_LOG_ACCUMULATOR_SIGNIFICAND_BITS + 1)
    )
    symbolic_significands = torch.where(
        rounded_up,
        torch.div(symbolic_significands, 2, rounding_mode="floor"),
        symbolic_significands,
    )
    symbolic_exponents = torch.where(
        rounded_up,
        symbolic_exponents + 1,
        symbolic_exponents,
    )

    flat_direct = direct_contributions.reshape(-1)
    direct_active = direct_mask.reshape(-1) & (flat_direct != 0)
    direct_mantissas, direct_powers = torch.frexp(flat_direct.abs())
    direct_significands = torch.round(
        direct_mantissas
        * (1 << (_LOG_ACCUMULATOR_SIGNIFICAND_BITS + 1))
    ).to(torch.int64)
    direct_exponents = direct_powers.to(torch.int64) - 1

    active = symbolic_active | direct_active
    term_signs = torch.where(
        direct_active,
        flat_direct.sign(),
        signs.reshape(-1),
    ).to(torch.int64)
    significands = torch.where(
        direct_active,
        direct_significands,
        symbolic_significands,
    )
    exponents = torch.where(
        direct_active,
        direct_exponents,
        symbolic_exponents,
    )

    target_count = n_rows * padded_steps
    exponent_sentinel = torch.iinfo(torch.int64).min
    top_exponents = torch.full(
        (target_count,),
        exponent_sentinel,
        dtype=torch.int64,
        device=signs.device,
    )
    top_exponents.scatter_reduce_(
        0,
        flat_targets,
        torch.where(
            active,
            exponents,
            torch.full_like(exponents, exponent_sentinel),
        ),
        reduce="amax",
        include_self=True,
    )
    target_active = top_exponents != exponent_sentinel
    safe_top_exponents = torch.where(
        target_active,
        top_exponents,
        torch.zeros_like(top_exponents),
    )
    base_exponents = safe_top_exponents - (
        _LOG_ACCUMULATOR_RADIX_BITS * (_LOG_ACCUMULATOR_LIMBS - 1)
    )
    term_base_exponents = base_exponents.gather(0, flat_targets)
    bit_offsets = (
        exponents
        - _LOG_ACCUMULATOR_SIGNIFICAND_BITS
        - term_base_exponents
    )
    low_limbs = torch.div(
        bit_offsets,
        _LOG_ACCUMULATOR_RADIX_BITS,
        rounding_mode="floor",
    )
    shifts = bit_offsets - low_limbs * _LOG_ACCUMULATOR_RADIX_BITS
    in_range = (
        ~active
        | ((low_limbs >= 0) & (low_limbs + 2 < _LOG_ACCUMULATOR_LIMBS))
    )
    if not in_range.all():
        raise FloatingPointError(
            "causal_rolling_zscore gradient exponent span is unsupported"
        )
    low_limbs = torch.where(active, low_limbs, torch.zeros_like(low_limbs))

    radix = 1 << _LOG_ACCUMULATOR_RADIX_BITS
    radix_mask = radix - 1
    low_significands = significands & radix_mask
    high_significands = torch.bitwise_right_shift(
        significands,
        _LOG_ACCUMULATOR_RADIX_BITS,
    )
    shifted_low = torch.bitwise_left_shift(low_significands, shifts)
    chunk_0 = shifted_low & radix_mask
    carry_0 = torch.bitwise_right_shift(
        shifted_low,
        _LOG_ACCUMULATOR_RADIX_BITS,
    )
    shifted_high = (
        torch.bitwise_left_shift(high_significands, shifts) + carry_0
    )
    chunk_1 = shifted_high & radix_mask
    chunk_2 = torch.bitwise_right_shift(
        shifted_high,
        _LOG_ACCUMULATOR_RADIX_BITS,
    )

    accumulator = torch.zeros(
        target_count * _LOG_ACCUMULATOR_LIMBS,
        dtype=torch.int64,
        device=signs.device,
    )
    for limb_delta, chunk in enumerate((chunk_0, chunk_1, chunk_2)):
        accumulator_indices = (
            flat_targets * _LOG_ACCUMULATOR_LIMBS
            + low_limbs
            + limb_delta
        )
        accumulator.scatter_add_(
            0,
            accumulator_indices,
            torch.where(
                active,
                term_signs * chunk,
                torch.zeros_like(chunk),
            ),
        )
    accumulator = accumulator.reshape(
        target_count,
        _LOG_ACCUMULATOR_LIMBS,
    )

    for limb in range(_LOG_ACCUMULATOR_LIMBS - 1):
        carry = torch.div(
            accumulator[:, limb],
            radix,
            rounding_mode="floor",
        )
        accumulator[:, limb] -= carry * radix
        accumulator[:, limb + 1] += carry

    negative = accumulator[:, -1] < 0
    lower = accumulator[:, :-1]
    lower_indices = torch.arange(
        _LOG_ACCUMULATOR_LIMBS - 1,
        dtype=torch.int64,
        device=signs.device,
    ).reshape(1, -1)
    first_lower_nonzero = torch.where(
        lower != 0,
        lower_indices,
        torch.full_like(lower_indices, _LOG_ACCUMULATOR_LIMBS - 1),
    ).amin(dim=-1)
    has_lower_nonzero = first_lower_nonzero < (
        _LOG_ACCUMULATOR_LIMBS - 1
    )
    negative_lower = torch.where(
        lower_indices < first_lower_nonzero.unsqueeze(-1),
        torch.zeros_like(lower),
        torch.where(
            lower_indices == first_lower_nonzero.unsqueeze(-1),
            radix - lower,
            radix - 1 - lower,
        ),
    )
    negative_top = (
        -accumulator[:, -1] - has_lower_nonzero.to(torch.int64)
    ).unsqueeze(-1)
    negative_magnitude = torch.cat([negative_lower, negative_top], dim=-1)
    magnitude = torch.where(
        negative.unsqueeze(-1),
        negative_magnitude,
        accumulator,
    )

    limb_indices = torch.arange(
        _LOG_ACCUMULATOR_LIMBS,
        dtype=torch.int64,
        device=signs.device,
    ).reshape(1, -1)
    highest_nonzero = torch.where(
        magnitude != 0,
        limb_indices,
        torch.full_like(limb_indices, -1),
    ).amax(dim=-1)
    is_zero = highest_nonzero < 0
    safe_highest = highest_nonzero.clamp_min(0)
    top_limb = magnitude.gather(1, safe_highest.unsqueeze(-1)).squeeze(-1)
    next_index = (safe_highest - 1).clamp_min(0)
    next_limb = magnitude.gather(1, next_index.unsqueeze(-1)).squeeze(-1)
    third_index = (safe_highest - 2).clamp_min(0)
    third_limb = magnitude.gather(1, third_index.unsqueeze(-1)).squeeze(-1)
    scaled_magnitude = top_limb.to(torch.float64)
    scaled_magnitude = scaled_magnitude + torch.where(
        safe_highest > 0,
        next_limb.to(torch.float64) / radix,
        torch.zeros_like(scaled_magnitude),
    )
    scaled_magnitude = scaled_magnitude + torch.where(
        safe_highest > 1,
        third_limb.to(torch.float64) / (radix * radix),
        torch.zeros_like(scaled_magnitude),
    )
    normalized_magnitude, exponent_adjustment = torch.frexp(
        scaled_magnitude
    )
    result_exponents = (
        base_exponents
        + safe_highest * _LOG_ACCUMULATOR_RADIX_BITS
        + exponent_adjustment.to(torch.int64)
    )
    limit_mantissa, limit_exponent = torch.frexp(gradient_limit)
    saturated = (
        (result_exponents > limit_exponent)
        | (
            (result_exponents == limit_exponent)
            & (normalized_magnitude >= limit_mantissa)
        )
    ) & ~is_zero
    safe_result_exponents = torch.where(
        saturated | is_zero,
        torch.zeros_like(result_exponents),
        result_exponents,
    )
    result_magnitude = torch.ldexp(
        normalized_magnitude,
        safe_result_exponents,
    )
    result_magnitude = torch.where(
        saturated,
        gradient_limit,
        torch.where(is_zero, torch.zeros_like(result_magnitude), result_magnitude),
    )
    result_sign = torch.where(
        is_zero,
        torch.zeros_like(result_magnitude),
        torch.where(
            negative,
            -torch.ones_like(result_magnitude),
            torch.ones_like(result_magnitude),
        ),
    )
    return (result_sign * result_magnitude).reshape(n_rows, padded_steps)


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
        work_smallest_normal = torch.finfo(grad.dtype).smallest_normal
        product_is_normally_represented = (
            direct_numerator.abs() >= work_smallest_normal
        ) & (direct_divisor >= work_smallest_normal)
        use_product_path = product_arithmetic_safe & (
            product_is_normally_represented
            | ~factorized_arithmetic_safe
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
            direct_mask & use_product_path,
            direct_divisor,
            torch.ones_like(log_magnitude),
        )
        product_contribution = direct_numerator / product_divisor
        direct_contribution = torch.where(
            direct_mask,
            torch.where(
                use_product_path,
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
        unsafe_contribution = unsafe_mask.to(grad.dtype)
        unsafe_padded = torch.zeros(
            n_rows, padded_steps, dtype=grad.dtype, device=grad.device
        )
        unsafe_padded.scatter_add_(
            1, flat_indices, unsafe_contribution.reshape(n_rows, -1)
        )
        start = window - 1
        direct_result = direct_padded[:, start : start + time_steps]
        input_has_unsafe = (
            unsafe_padded[:, start : start + time_steps] != 0
        )
        if unsafe_mask.any():
            exact_padded = _sum_signed_logs(
                unsafe_signs,
                unsafe_logs,
                direct_contribution,
                direct_mask,
                gradient_limit,
            )
            exact_result = exact_padded[:, start : start + time_steps]
            input_gradient = torch.where(
                input_has_unsafe,
                exact_result,
                direct_result,
            )
        else:
            input_gradient = direct_result
        input_gradient = input_gradient.clamp(
            -gradient_limit,
            gradient_limit,
        )
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
