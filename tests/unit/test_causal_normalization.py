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
