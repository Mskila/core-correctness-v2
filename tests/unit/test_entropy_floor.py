"""Autograd contract for the policy entropy floor."""

import torch
from torch.distributions import Categorical
import pytest

from model_core.engine import AlphaEngine


def test_low_entropy_floor_preserves_nonzero_logits_gradient() -> None:
    logits = torch.tensor([[4.0, -4.0, -4.0]], requires_grad=True)
    mean_entropy = Categorical(logits=logits).entropy().mean()
    penalty = AlphaEngine._entropy_floor_loss(mean_entropy)

    assert penalty.item() > 0.0
    penalty.backward()
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad).item() > 0


def test_entropy_above_floor_has_zero_penalty_and_gradient() -> None:
    logits = torch.zeros((1, 4), requires_grad=True)
    mean_entropy = Categorical(logits=logits).entropy().mean()
    penalty = AlphaEngine._entropy_floor_loss(mean_entropy)

    assert penalty.item() == 0.0
    penalty.backward()
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits))


@pytest.mark.parametrize(
    "values",
    [
        [100.0, -100.0, -100.0],
        [3.0e38, -3.0e38, -3.0e38],
    ],
)
def test_extreme_finite_float32_entropy_floor_has_useful_directional_gradient(
    values,
) -> None:
    logits = torch.tensor([values], dtype=torch.float32, requires_grad=True)
    mean_entropy = AlphaEngine._stable_categorical_entropy(logits).mean()
    penalty = AlphaEngine._entropy_floor_loss(mean_entropy)

    assert torch.isfinite(mean_entropy)
    assert penalty.item() > 0.0
    penalty.backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert torch.count_nonzero(logits.grad).item() == logits.numel()
    assert logits.grad[0, 0] > 0
    assert logits.grad[0, 1] < 0
    assert logits.grad[0, 2] < 0


def test_stable_entropy_preserves_ordinary_and_uniform_forward_semantics() -> None:
    logits = torch.tensor(
        [[0.25, -0.5, 0.75], [0.0, 0.0, 0.0]],
        dtype=torch.float32,
        requires_grad=True,
    )
    reference_logits = logits.detach().clone().requires_grad_(True)
    expected = Categorical(logits=reference_logits).entropy()
    actual = AlphaEngine._stable_categorical_entropy(logits)

    torch.testing.assert_close(actual, expected)
    actual[0].backward(retain_graph=True)
    expected[0].backward()
    torch.testing.assert_close(logits.grad[0], reference_logits.grad[0])
    logits.grad.zero_()
    penalty = AlphaEngine._entropy_floor_loss(actual[1])
    assert penalty.item() == 0.0
    penalty.backward()
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits))
