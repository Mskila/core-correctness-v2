import importlib
import importlib.util

import pytest
import torch

import model_core.vm as vm_module
from model_core.causal import causal_rolling_zscore
from model_core.features import MT5FeatureEngineer
from model_core.ops import _op_jump, _ts_zscore


def _prefix_reference(x: torch.Tensor, window: int) -> torch.Tensor:
    """Direct public-contract reference for ordinary finite inputs."""
    work = x.to(torch.float64) if x.dtype != torch.float64 else x
    values = []
    for end in range(work.shape[1]):
        prefix = work[:, max(0, end - window + 1):end + 1]
        mean = prefix.mean(dim=1)
        variance = (prefix - mean[:, None]).square().mean(dim=1)
        positive = variance > 0
        safe_variance = torch.where(
            positive, variance, torch.ones_like(variance)
        )
        std = safe_variance.sqrt()
        active = positive & (std > 1.0e-6)
        safe_std = torch.where(active, std, torch.ones_like(std))
        zscore = (work[:, end] - mean) / safe_std
        values.append(torch.where(active, zscore, torch.zeros_like(zscore)))
    return torch.stack(values, dim=1).to(x.dtype)


def test_shared_causal_normalization_module_exists() -> None:
    assert importlib.util.find_spec("model_core.causal") is not None
    module = importlib.import_module("model_core.causal")
    assert callable(getattr(module, "causal_rolling_zscore", None))


def test_warmup_uses_only_observed_prefix_values() -> None:
    x = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    expected = torch.tensor([[0.0, 1.0, 1.2247449]], dtype=torch.float32)
    torch.testing.assert_close(
        causal_rolling_zscore(x, window=200), expected, rtol=0, atol=1e-6
    )


def test_constant_prefix_normalizes_to_zero_with_finite_gradient() -> None:
    x = torch.full((2, 32), 7.0, dtype=torch.float32, requires_grad=True)
    output = causal_rolling_zscore(x, window=200)
    output.sum().backward()
    torch.testing.assert_close(output, torch.zeros_like(output), rtol=0, atol=0)
    assert x.grad is not None
    torch.testing.assert_close(x.grad, torch.zeros_like(x), rtol=0, atol=0)


@pytest.mark.parametrize(
    ("standard_deviation", "expected"),
    [(0.5e-6, 0.0), (1.0e-6, 0.0), (1.0001e-6, 1.0)],
)
def test_standard_deviation_threshold_is_strictly_greater_than_one_e_minus_six(
    standard_deviation: float,
    expected: float,
) -> None:
    x = torch.tensor([[0.0, 2.0 * standard_deviation]], dtype=torch.float64)
    actual = causal_rolling_zscore(x, window=2)
    assert actual[0, 1].item() == pytest.approx(expected, rel=0, abs=1e-12)


def test_inactive_standard_deviation_branch_has_safe_backward() -> None:
    x = torch.tensor(
        [[4.0, 4.0, 4.0], [0.0, 1.0e-6, 0.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    output = causal_rolling_zscore(x, window=3)
    output.square().sum().backward()
    torch.testing.assert_close(output, torch.zeros_like(output), rtol=0, atol=0)
    assert x.grad is not None and torch.isfinite(x.grad).all()
    torch.testing.assert_close(x.grad, torch.zeros_like(x), rtol=0, atol=0)


def test_future_changes_do_not_change_historical_zscores() -> None:
    generator = torch.Generator().manual_seed(7)
    x = torch.randn(2, 64, dtype=torch.float32, generator=generator)
    changed = x.clone()
    changed[:, 33:] = changed[:, 33:] * 11.0 + 5.0
    left = causal_rolling_zscore(x, window=20)
    right = causal_rolling_zscore(changed, window=20)
    torch.testing.assert_close(left[:, :33], right[:, :33], rtol=0, atol=0)


@pytest.mark.parametrize("window", [0, -1, True, 1.5, "3"])
def test_invalid_window_fails_clearly(window) -> None:
    with pytest.raises(ValueError, match=r"\[N,T\].*window >= 1"):
        causal_rolling_zscore(torch.ones(1, 3), window=window)


@pytest.mark.parametrize("value", [torch.ones(3), torch.ones(1, 2, 3), torch.empty(1, 0), [1.0]])
def test_invalid_shape_or_input_type_fails_clearly(value) -> None:
    with pytest.raises(ValueError, match=r"\[N,T\].*window >= 1"):
        causal_rolling_zscore(value, window=2)


@pytest.mark.parametrize(
    "x",
    [torch.ones(1, 3, dtype=torch.int64), torch.ones(1, 3, dtype=torch.bool)],
)
def test_non_floating_input_fails_clearly(x: torch.Tensor) -> None:
    with pytest.raises(ValueError, match="floating"):
        causal_rolling_zscore(x, window=2)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_input_fails_closed(bad_value: float) -> None:
    x = torch.tensor([[1.0, bad_value, 2.0]], dtype=torch.float32)
    with pytest.raises(FloatingPointError, match="finite input"):
        causal_rolling_zscore(x, window=2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_output_preserves_shape_device_and_dtype(dtype: torch.dtype) -> None:
    x = torch.tensor([[1.0, 2.0, 4.0], [3.0, 1.0, -2.0]], dtype=dtype)
    output = causal_rolling_zscore(x, window=3)
    assert output.shape == x.shape
    assert output.device == x.device
    assert output.dtype == x.dtype
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("window", [1, 2, 7, 200, 201, 500])
def test_ordinary_float32_matches_per_prefix_population_reference(window: int) -> None:
    x = torch.tensor(
        [[0.25, 1.0, 2.0, -1.0, 3.0], [4.0, 3.0, 5.0, 8.0, 2.0]],
        dtype=torch.float32,
    )
    torch.testing.assert_close(
        causal_rolling_zscore(x, window),
        _prefix_reference(x, window),
        rtol=0,
        atol=2e-7,
    )


def test_ordinary_float32_forward_and_backward_are_finite() -> None:
    x = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0], [0.5, -1.0, 2.0, 1.5]],
        dtype=torch.float32,
        requires_grad=True,
    )
    output = causal_rolling_zscore(x, window=3)
    output.square().sum().backward()
    assert torch.isfinite(output).all()
    assert x.grad is not None and torch.isfinite(x.grad).all()


@pytest.mark.parametrize("amplitude", [1.0e20, 3.0e38])
def test_large_finite_float32_prefix_has_nonzero_output_and_finite_backward(
    amplitude: float,
) -> None:
    x = torch.tensor(
        [[amplitude, -amplitude]], dtype=torch.float32, requires_grad=True
    )

    output = causal_rolling_zscore(x, window=2)
    output.square().sum().backward()

    torch.testing.assert_close(
        output,
        torch.tensor([[0.0, -1.0]], dtype=torch.float32),
        rtol=0,
        atol=1.0e-6,
    )
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_large_finite_ohlcv_keeps_macd_hist_normalization_informative() -> None:
    index = torch.arange(256, dtype=torch.float32).unsqueeze(0)
    close = 100.0 + 0.08 * index + torch.sin(index / 4.0)
    open_ = close + 0.1 * torch.cos(index / 5.0)
    spread = 0.5 + 0.05 * torch.sin(index / 7.0).abs()
    raw = {
        "open": open_,
        "high": torch.maximum(open_, close) + spread,
        "low": torch.minimum(open_, close) - spread,
        "close": close,
        "volume": 1000.0 + 3.0 * index + 25.0 * torch.cos(index / 6.0),
    }
    scaled = {name: value * 1.0e20 for name, value in raw.items()}

    ordinary = MT5FeatureEngineer._c_macd_hist(raw)
    large = MT5FeatureEngineer._c_macd_hist(scaled)

    assert torch.isfinite(large).all()
    assert torch.count_nonzero(ordinary).item() == 255
    assert torch.count_nonzero(large).item() >= 250


def test_unsupported_finite_float64_variance_fails_closed() -> None:
    x = torch.tensor([[1.0e200, -1.0e200]], dtype=torch.float64)

    with pytest.raises(FloatingPointError, match="variance"):
        causal_rolling_zscore(x, window=2)


def test_ordinary_float64_gradcheck_and_gradgradcheck() -> None:
    x = torch.tensor(
        [[0.25, 1.0, 2.0, -1.0, 3.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    function = lambda value: causal_rolling_zscore(value, window=4)
    assert torch.autograd.gradcheck(function, (x,), eps=1e-6, atol=1e-5, rtol=1e-4)
    assert torch.autograd.gradgradcheck(
        function, (x,), eps=1e-6, atol=2e-5, rtol=2e-4
    )


def test_feature_and_operator_normalizers_share_causal_semantics() -> None:
    x = torch.tensor([[1.0, 2.0, 3.0, 3.0]], dtype=torch.float32)
    expected_200 = causal_rolling_zscore(x, window=200)
    expected_3 = causal_rolling_zscore(x, window=3)
    torch.testing.assert_close(
        MT5FeatureEngineer._robust_norm(x, 200), expected_200.clamp(-5.0, 5.0)
    )
    torch.testing.assert_close(_ts_zscore(x, 3), expected_3)
    torch.testing.assert_close(_op_jump(x), torch.tanh(expected_200 - 1.5))


def test_vm_normalizer_delegates_to_shared_causal_function(monkeypatch) -> None:
    calls = []

    def fake_normalizer(value: torch.Tensor, window: int) -> torch.Tensor:
        calls.append((value, window))
        return torch.full_like(value, 2.0)

    monkeypatch.setattr(vm_module, "causal_rolling_zscore", fake_normalizer)
    x = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    output = vm_module.StackVM._normalize_output(x)
    assert calls == [(x, 200)]
    torch.testing.assert_close(output, torch.full_like(x, 2.0), rtol=0, atol=0)


def test_rows_are_isolated_in_forward_and_backward() -> None:
    original = torch.tensor(
        [[0.25, 1.0, 2.0, -1.0], [3.0, -4.0, 5.0, 6.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    changed = original.detach().clone()
    changed[1] = changed[1] * -1000.0 + 17.0
    changed.requires_grad_()
    left = causal_rolling_zscore(original, window=3)
    right = causal_rolling_zscore(changed, window=3)
    left[0].square().sum().backward()
    right[0].square().sum().backward()
    torch.testing.assert_close(left[0], right[0], rtol=0, atol=0)
    assert original.grad is not None and changed.grad is not None
    torch.testing.assert_close(original.grad[0], changed.grad[0], rtol=0, atol=0)
    torch.testing.assert_close(original.grad[1], torch.zeros(4, dtype=torch.float64))


@pytest.mark.parametrize("window", [1, 3, 200])
def test_noncontiguous_input_has_stable_shape_and_backward(window: int) -> None:
    leaf = torch.arange(16.0, dtype=torch.float64, requires_grad=True)
    x = leaf.reshape(2, 8)[:, ::2]
    assert not x.is_contiguous()
    output = causal_rolling_zscore(x, window=window)
    output.square().sum().backward()
    assert output.shape == x.shape
    assert leaf.grad is not None and torch.isfinite(leaf.grad).all()


def test_backward_does_not_pollute_unrelated_caller_branch() -> None:
    values = torch.tensor([[0.25, 1.0, 2.0, -1.0]], dtype=torch.float64)
    branch_weights = torch.tensor([[0.125, -0.25, 0.5, 1.0]], dtype=torch.float64)
    combined_input = values.clone().requires_grad_()
    isolated_input = values.clone().requires_grad_()
    combined = causal_rolling_zscore(combined_input, window=3)
    isolated = causal_rolling_zscore(isolated_input, window=3)
    (combined.square().sum() + (combined_input * branch_weights).sum()).backward()
    isolated.square().sum().backward()
    assert combined_input.grad is not None and isolated_input.grad is not None
    torch.testing.assert_close(
        combined_input.grad - isolated_input.grad,
        branch_weights,
        rtol=0,
        atol=1e-15,
    )
