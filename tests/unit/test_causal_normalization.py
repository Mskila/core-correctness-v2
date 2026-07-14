import importlib
import importlib.util

import pytest
import torch

from model_core.causal import causal_rolling_zscore
from model_core.features import MT5FeatureEngineer
from model_core.ops import _op_jump, _ts_zscore


def test_shared_causal_normalization_module_exists() -> None:
    assert importlib.util.find_spec("model_core.causal") is not None
    module = importlib.import_module("model_core.causal")
    assert callable(getattr(module, "causal_rolling_zscore", None))


def test_warmup_uses_only_observed_prefix_values() -> None:
    x = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    actual = causal_rolling_zscore(x, window=200)
    expected = torch.tensor([[0.0, 1.0, 1.2247449]], dtype=torch.float32)
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)


def test_constant_prefix_normalizes_to_zero() -> None:
    x = torch.full((2, 32), 7.0, dtype=torch.float32)
    torch.testing.assert_close(
        causal_rolling_zscore(x, window=200),
        torch.zeros_like(x),
        rtol=0,
        atol=0,
    )


def test_future_changes_do_not_change_historical_zscores() -> None:
    torch.manual_seed(7)
    x = torch.randn(2, 64, dtype=torch.float32)
    changed = x.clone()
    changed[:, 33:] = changed[:, 33:] * 11.0 + 5.0
    left = causal_rolling_zscore(x, window=20)
    right = causal_rolling_zscore(changed, window=20)
    torch.testing.assert_close(left[:, :33], right[:, :33], rtol=0, atol=0)


@pytest.mark.parametrize("window", [0, -1, True, 1.5])
def test_invalid_window_fails_clearly(window) -> None:
    with pytest.raises(ValueError, match=r"\[N,T\].*window >= 1"):
        causal_rolling_zscore(torch.ones(1, 3), window=window)


@pytest.mark.parametrize("shape", [(3,), (1, 2, 3)])
def test_invalid_shape_fails_clearly(shape) -> None:
    with pytest.raises(ValueError, match=r"\[N,T\].*window >= 1"):
        causal_rolling_zscore(torch.ones(shape), window=2)


def test_feature_and_operator_normalizers_share_causal_semantics() -> None:
    x = torch.tensor([[1.0, 2.0, 3.0, 3.0]], dtype=torch.float32)
    expected_200 = causal_rolling_zscore(x, window=200)
    expected_3 = causal_rolling_zscore(x, window=3)
    torch.testing.assert_close(
        MT5FeatureEngineer._robust_norm(x, 200),
        expected_200.clamp(-5.0, 5.0),
    )
    torch.testing.assert_close(_ts_zscore(x, 3), expected_3)
    torch.testing.assert_close(_op_jump(x), torch.tanh(expected_200 - 1.5))


def test_ordinary_finite_input_has_finite_forward_and_backward() -> None:
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]], requires_grad=True)
    out = causal_rolling_zscore(x, window=200)
    assert torch.isfinite(out).all()
    out.square().sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize(
    ("dtype", "amplitude"),
    [
        (torch.float16, 60_000.0),
        (torch.float32, 1.0e20),
        (torch.float64, 1.0e200),
    ],
)
def test_extreme_finite_input_is_not_masked_and_has_finite_gradients(
    dtype, amplitude
) -> None:
    x = torch.tensor(
        [[amplitude, -amplitude, amplitude, -amplitude]],
        dtype=dtype,
        requires_grad=True,
    )
    out = causal_rolling_zscore(x, window=200)
    assert out.dtype == dtype
    assert torch.isfinite(out).all()
    assert out[:, 1:].abs().max() > 0
    out.to(torch.float64).square().sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize(
    ("dtype", "scale", "atol"),
    [
        (
            torch.float16,
            torch.finfo(torch.float16).smallest_normal * torch.finfo(torch.float16).eps,
            2e-3,
        ),
        (
            torch.bfloat16,
            torch.finfo(torch.bfloat16).smallest_normal * torch.finfo(torch.bfloat16).eps,
            1e-2,
        ),
        (
            torch.float32,
            torch.finfo(torch.float32).smallest_normal * torch.finfo(torch.float32).eps,
            1e-6,
        ),
        (torch.float64, 1.0e-200, 1e-12),
    ],
)
def test_nonzero_small_scale_is_preserved_with_finite_gradients(
    dtype, scale, atol
) -> None:
    reference = torch.tensor([[0.0, 1.0, 2.0]], dtype=dtype)
    tiny = (reference * scale).requires_grad_()

    expected = causal_rolling_zscore(reference, window=200)
    actual = causal_rolling_zscore(tiny, window=200)

    torch.testing.assert_close(actual, expected, rtol=0, atol=atol)
    actual.to(torch.float64).sum().backward()
    assert tiny.grad is not None
    assert torch.isfinite(tiny.grad).all()


def test_amihud_illiq_does_not_collapse_representable_variation() -> None:
    close = torch.tensor(
        [[1.0, 1.0001, 0.9998, 1.0003, 0.9997, 1.0002]],
        dtype=torch.float32,
    )
    raw = {
        "close": close,
        "volume": torch.full_like(close, 1.0e8),
    }

    actual = MT5FeatureEngineer._c_amihud_illiq(raw)

    assert torch.isfinite(actual).all()
    assert torch.count_nonzero(actual).item() > 0


_FLOAT64_MIN_SUBNORMAL = torch.nextafter(
    torch.tensor(0.0, dtype=torch.float64),
    torch.tensor(1.0, dtype=torch.float64),
).item()
_FLOAT64_SCALE_CASES = [
    _FLOAT64_MIN_SUBNORMAL,
    1.0e-320,
    1.0e-310,
    1.0e-308,
    torch.finfo(torch.float64).smallest_normal,
    3.0e-308,
    1.0e-200,
    1.0,
    1.0e200,
]
_SUBNORMAL_PATTERN = torch.tensor(
    [[0.0, 1.0, 2.0, -1.0]], dtype=torch.float64
)
_ASYMMETRIC_WEIGHTS = torch.tensor(
    [[1.0, -2.0, 3.0, 0.5]], dtype=torch.float64
)


def _apply_loss(output: torch.Tensor, loss_name: str) -> torch.Tensor:
    if loss_name == "sum":
        return output.sum()
    if loss_name == "square_sum":
        return output.square().sum()
    if loss_name == "weighted_sum":
        return (output * _ASYMMETRIC_WEIGHTS.to(output)).sum()
    raise AssertionError(f"unknown test loss: {loss_name}")


def _direct_prefix_zscore(x: torch.Tensor, window: int) -> torch.Tensor:
    """Small ordinary-scale reference independent of production scaling."""
    outputs = []
    for end in range(x.shape[1]):
        prefix = x[:, max(0, end - window + 1):end + 1]
        mean = prefix.mean(dim=1)
        variance = (prefix - mean.unsqueeze(1)).square().mean(dim=1)
        safe_variance = torch.where(
            variance > 0, variance, torch.ones_like(variance)
        )
        value = (x[:, end] - mean) / safe_variance.sqrt()
        outputs.append(torch.where(variance > 0, value, torch.zeros_like(value)))
    return torch.stack(outputs, dim=1)


def test_float64_min_subnormal_square_loss_backward_is_finite() -> None:
    x = (_SUBNORMAL_PATTERN * _FLOAT64_MIN_SUBNORMAL).requires_grad_()

    output = causal_rolling_zscore(x, window=200)

    expected = torch.tensor(
        [[0.0, 1.0, 1.224744871391589, -1.3416407864998738]],
        dtype=torch.float64,
    )
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    output.square().sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize(
    "scale",
    _FLOAT64_SCALE_CASES,
    ids=[
        "min-subnormal",
        "1e-320",
        "1e-310",
        "1e-308",
        "min-normal",
        "3e-308",
        "1e-200",
        "unit",
        "1e200",
    ],
)
@pytest.mark.parametrize("loss_name", ["sum", "square_sum", "weighted_sum"])
def test_float64_scale_matrix_preserves_forward_and_gradient_direction(
    scale: float, loss_name: str
) -> None:
    unit = _SUBNORMAL_PATTERN.clone().requires_grad_()
    scaled = (_SUBNORMAL_PATTERN * scale).requires_grad_()

    unit_output = causal_rolling_zscore(unit, window=200)
    scaled_output = causal_rolling_zscore(scaled, window=200)
    torch.testing.assert_close(scaled_output, unit_output, rtol=0, atol=1e-12)

    _apply_loss(unit_output, loss_name).backward()
    _apply_loss(scaled_output, loss_name).backward()
    assert unit.grad is not None and scaled.grad is not None
    assert torch.isfinite(scaled.grad).all()
    assert torch.count_nonzero(scaled.grad).item() > 0

    active = unit.grad != 0
    assert torch.equal(
        torch.sign(scaled.grad[active]),
        torch.sign(unit.grad[active]),
    )


@pytest.mark.parametrize("loss_name", ["sum", "square_sum", "weighted_sum"])
def test_ordinary_scale_gradient_matches_direct_prefix_reference(
    loss_name: str,
) -> None:
    values = torch.tensor(
        [[0.25, 1.0, 2.0, -1.0]], dtype=torch.float64
    )
    actual_input = values.clone().requires_grad_()
    reference_input = values.clone().requires_grad_()

    actual = causal_rolling_zscore(actual_input, window=200)
    reference = _direct_prefix_zscore(reference_input, window=200)
    torch.testing.assert_close(actual, reference, rtol=0, atol=1e-12)
    _apply_loss(actual, loss_name).backward()
    _apply_loss(reference, loss_name).backward()

    torch.testing.assert_close(
        actual_input.grad,
        reference_input.grad,
        rtol=1e-12,
        atol=1e-12,
    )


def test_signed_zero_and_true_zero_variance_have_finite_zero_gradients() -> None:
    x = torch.tensor(
        [[0.0, -0.0, 0.0], [7.0, 7.0, 7.0]],
        dtype=torch.float64,
        requires_grad=True,
    )

    output = causal_rolling_zscore(x, window=200)
    output.square().sum().backward()

    torch.testing.assert_close(output, torch.zeros_like(output), rtol=0, atol=0)
    torch.testing.assert_close(x.grad, torch.zeros_like(x), rtol=0, atol=0)


@pytest.mark.parametrize("window", [1, 200, 500])
def test_noncontiguous_backward_is_finite_for_short_and_long_windows(
    window: int,
) -> None:
    leaf = torch.arange(16.0, dtype=torch.float64, requires_grad=True)
    x = leaf.reshape(2, 8)[:, ::2]
    assert not x.is_contiguous()

    output = causal_rolling_zscore(x, window=window)
    (output * torch.tensor([[1.0, -2.0, 3.0, 0.5]])).sum().backward()

    assert output.shape == x.shape
    assert leaf.grad is not None
    assert torch.isfinite(leaf.grad).all()


def test_rows_are_isolated_in_forward_and_backward() -> None:
    left = torch.tensor(
        [[0.25, 1.0, 2.0, -1.0], [3.0, -4.0, 5.0, 6.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    changed = left.detach().clone()
    changed[1] = changed[1] * -1000.0 + 17.0
    changed.requires_grad_()

    left_output = causal_rolling_zscore(left, window=3)
    changed_output = causal_rolling_zscore(changed, window=3)
    left_output[0].square().sum().backward()
    changed_output[0].square().sum().backward()

    torch.testing.assert_close(
        left_output[0], changed_output[0], rtol=0, atol=0
    )
    torch.testing.assert_close(left.grad[0], changed.grad[0], rtol=0, atol=0)


def test_backward_does_not_pollute_unrelated_caller_branch() -> None:
    values = torch.tensor(
        [[0.25, 1.0, 2.0, -1.0]], dtype=torch.float64
    )
    branch_weights = torch.tensor(
        [[0.125, -0.25, 0.5, 1.0]], dtype=torch.float64
    )
    combined_input = values.clone().requires_grad_()
    zscore_only_input = values.clone().requires_grad_()

    combined = causal_rolling_zscore(combined_input, window=200)
    zscore_only = causal_rolling_zscore(zscore_only_input, window=200)
    (_apply_loss(combined, "weighted_sum") +
     (combined_input * branch_weights).sum()).backward()
    _apply_loss(zscore_only, "weighted_sum").backward()

    torch.testing.assert_close(
        combined_input.grad - zscore_only_input.grad,
        branch_weights,
        rtol=0,
        atol=1e-15,
    )


def test_window_one_keeps_unrelated_subnormal_branch_gradient() -> None:
    x = (_SUBNORMAL_PATTERN * _FLOAT64_MIN_SUBNORMAL).requires_grad_()
    branch_weights = torch.tensor(
        [[0.125, -0.25, 0.5, 1.0]], dtype=torch.float64
    )

    output = causal_rolling_zscore(x, window=1)
    (output.sum() + (x * branch_weights).sum()).backward()

    torch.testing.assert_close(output, torch.zeros_like(output), rtol=0, atol=0)
    torch.testing.assert_close(x.grad, branch_weights, rtol=0, atol=0)


def _ordinary_gradient_from_upstream(
    pattern: torch.Tensor,
    window: int,
    upstream: torch.Tensor,
) -> torch.Tensor:
    reference_input = pattern.clone().requires_grad_()
    reference = _direct_prefix_zscore(reference_input, window)
    reference.backward(upstream)
    assert reference_input.grad is not None
    return reference_input.grad


def test_signed_log_aggregation_preserves_exact_cancellation_residual() -> None:
    scale = torch.nextafter(
        torch.tensor(0.0, dtype=torch.float64),
        torch.tensor(1.0, dtype=torch.float64),
    )
    pattern = torch.tensor(
        [[-2.0, -2.0, -2.0, -2.0, -2.0, -1.0, -2.0, -2.0, -2.0]],
        dtype=torch.float64,
    )
    upstream = torch.zeros_like(pattern)
    upstream[0, 6] = scale
    upstream[0, 7] = 1.0e308
    upstream[0, 8] = -1.0e308
    x = (pattern * scale).requires_grad_()

    causal_rolling_zscore(x, window=4).backward(upstream)

    assert x.grad is not None
    assert x.grad[0, 6].item() == pytest.approx(
        1.539600717839002,
        rel=0,
        abs=1.0e-15,
    )


@pytest.mark.parametrize(
    (
        "pattern",
        "window",
        "target",
        "small_upstream",
        "large_outputs",
        "large_signs",
    ),
    [
        (
            [-2.0, -2.0, -2.0, -2.0, -2.0, -1.0, -2.0, -2.0, -2.0],
            4,
            6,
            {6: 1.0},
            (7, 8),
            (-1.0, 1.0),
        ),
        (
            [-2.0, -2.0, -2.0, -2.0, -2.0, -1.0, -2.0, -2.0, -2.0],
            4,
            6,
            {6: -1.0},
            (7, 8),
            (-1.0, 1.0),
        ),
        (
            [-2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -1.0, -2.0, -2.0, -2.0],
            4,
            7,
            {7: 1.0},
            (8, 9),
            (1.0, -1.0),
        ),
        (
            [-2.0, -2.0, -2.0, -2.0, -2.0, -1.0, -2.0, -2.0, -2.0, -2.0],
            5,
            6,
            {6: 1.0, 7: 2.0},
            (8, 9),
            (1.0, -1.0),
        ),
        (
            [-2.0, -2.0, -2.0, -2.0, -2.0, -1.0, -2.0, -2.0, -2.0, -2.0],
            5,
            6,
            {8: 1.0},
            (7, 9),
            (1.0, -1.0),
        ),
        (
            [-2.0, -2.0, -2.0, -2.0, -2.0, -1.0, -2.0, -2.0, -2.0, -2.0],
            5,
            6,
            {9: 1.0},
            (7, 8),
            (1.0, -1.0),
        ),
        (
            [
                -2.0,
                -2.0,
                -2.0,
                -2.0,
                -2.0,
                -1.0,
                -2.0,
                -2.0,
                -2.0,
                17.0,
                -31.0,
            ],
            4,
            6,
            {6: 1.0},
            (7, 8),
            (1.0, -1.0),
        ),
    ],
)
def test_signed_log_cancellation_matrix_is_order_and_cut_invariant(
    pattern,
    window,
    target,
    small_upstream,
    large_outputs,
    large_signs,
) -> None:
    scale = torch.nextafter(
        torch.tensor(0.0, dtype=torch.float64),
        torch.tensor(1.0, dtype=torch.float64),
    )
    values = torch.tensor([pattern], dtype=torch.float64)
    small = torch.zeros_like(values)
    actual_upstream = torch.zeros_like(values)
    for index, weight in small_upstream.items():
        small[0, index] = weight
        actual_upstream[0, index] = weight * scale
    for index, sign in zip(large_outputs, large_signs):
        actual_upstream[0, index] = sign * 1.0e308
    expected = _ordinary_gradient_from_upstream(values, window, small)[0, target]
    x = (values * scale).requires_grad_()

    causal_rolling_zscore(x, window).backward(actual_upstream)

    assert x.grad is not None
    torch.testing.assert_close(x.grad[0, target], expected, rtol=0, atol=1.0e-15)


def test_repeated_value_structural_zero_jacobian_is_exact() -> None:
    scale = torch.nextafter(
        torch.tensor(0.0, dtype=torch.float64),
        torch.tensor(1.0, dtype=torch.float64),
    )
    pattern = torch.tensor(
        [[-2.0, -2.0, -2.0, -2.0, -1.0, -2.0, -2.0]],
        dtype=torch.float64,
    )
    x = (pattern * scale).requires_grad_()

    causal_rolling_zscore(x, window=3)[0, 6].backward()

    assert x.grad is not None
    assert x.grad[0, 4].item() == 0.0


@pytest.mark.parametrize(
    ("tail", "window", "target_offset"),
    [
        ([-1.0, -2.0, -2.0], 3, 0),
        ([1.0, 2.0, 2.0], 3, 0),
        ([1.0, 0.0, 0.0], 3, 0),
        ([-2.0, -1.0, -2.0], 3, 1),
        ([-1.0, -2.0, -2.0, -2.0], 4, 0),
        ([-2.0, -1.0, -2.0, -2.0], 4, 1),
        ([-1.0, -2.0, -2.0, -2.0, -2.0], 5, 0),
    ],
)
def test_repeated_value_structural_zero_matrix(
    tail,
    window,
    target_offset,
) -> None:
    scale = torch.nextafter(
        torch.tensor(0.0, dtype=torch.float64),
        torch.tensor(1.0, dtype=torch.float64),
    )
    prefix = [-7.0] * 5
    pattern = torch.tensor([prefix + tail], dtype=torch.float64)
    target = len(prefix) + target_offset
    x = (pattern * scale).requires_grad_()

    causal_rolling_zscore(x, window)[0, -1].backward()

    assert x.grad is not None
    assert x.grad[0, target].item() == 0.0


@pytest.mark.parametrize(
    "tail",
    [
        [-1.0, -2.0, -3.0],
        [-1.0, -3.0, -2.0],
        [1.0, -2.0, 3.0],
    ],
)
def test_nearby_true_subnormal_jacobian_is_not_erased(tail) -> None:
    scale = torch.nextafter(
        torch.tensor(0.0, dtype=torch.float64),
        torch.tensor(1.0, dtype=torch.float64),
    )
    pattern = torch.tensor([[-7.0] * 5 + tail], dtype=torch.float64)
    ordinary = pattern.clone().requires_grad_()
    _direct_prefix_zscore(ordinary, 3)[0, -1].backward()
    x = (pattern * scale).requires_grad_()

    causal_rolling_zscore(x, window=3)[0, -1].backward()

    assert ordinary.grad is not None and x.grad is not None
    expected_sign = torch.sign(ordinary.grad[0, -3])
    assert expected_sign != 0
    assert torch.isfinite(x.grad).all()
    assert x.grad[0, -3] != 0
    assert torch.sign(x.grad[0, -3]) == expected_sign
