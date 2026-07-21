import pytest
import torch
from hypothesis import given, settings, strategies as st

import model_core.features as features_module
import model_core.ops as ops_module
from model_core.features import FEATURE_REGISTRY, MT5FeatureEngineer
from model_core.ops import OPERATOR_REGISTRY
from model_core.vm import StackVM
from model_core.vocab import FORMULA_VOCAB


def make_ohlcv(t: int = 64) -> dict[str, torch.Tensor]:
    index = torch.arange(t, dtype=torch.float32).unsqueeze(0)
    close = 100.0 + 0.08 * index + torch.sin(index / 4.0)
    open_ = close + 0.1 * torch.cos(index / 5.0)
    spread = 0.5 + 0.05 * torch.sin(index / 7.0).abs()
    high = torch.maximum(open_, close) + spread
    low = torch.minimum(open_, close) - spread
    volume = 1000.0 + 3.0 * index + 25.0 * torch.cos(index / 6.0)
    return {
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }


def mutate_future(
    raw: dict[str, torch.Tensor], cut: int
) -> dict[str, torch.Tensor]:
    changed = {name: value.clone() for name, value in raw.items()}
    for name in ("open", "high", "low", "close", "volume"):
        changed[name][:, cut + 1:] *= 1.37
    return changed


def assert_prefix_equal(
    left: torch.Tensor, right: torch.Tensor, cut: int
) -> None:
    torch.testing.assert_close(
        left[..., :cut + 1],
        right[..., :cut + 1],
        rtol=0,
        atol=1e-6,
    )


def test_ema_helpers_are_invariant_to_appended_future_and_large_scale() -> None:
    index = torch.arange(250, dtype=torch.float64).unsqueeze(0)
    long = 1.0e12 + 1.0e9 * torch.sin(index / 3.0)
    short = long[:, :20].clone()

    for helper in (features_module.MT5FeatureEngineer._ema_simple,
                   ops_module._ema_simple):
        for span in (5, 20):
            expected = helper(short, span)
            actual = helper(long, span)[:, :short.shape[1]]
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_ema_helpers_preserve_dtype_device_shape_and_autograd() -> None:
    for helper in (features_module.MT5FeatureEngineer._ema_simple,
                   ops_module._ema_simple):
        x = torch.linspace(1.0, 2.0, 73, dtype=torch.float64).unsqueeze(0)
        x.requires_grad_()
        out = helper(x, 20)
        assert out.shape == x.shape
        assert out.dtype == x.dtype
        assert out.device == x.device
        out.square().sum().backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()


def test_all_ema_backed_features_are_invariant_to_appended_future() -> None:
    names = {
        "MACD_HIST",
        "EMA_RATIO_12_26",
        "TRIX_15",
        "PPO",
        "TRIX_SIGNAL",
        "KELTNER_POS_20",
        "SAR_DIST",
    }
    long = make_ohlcv(250)
    long = {name: value * 100_000.0 for name, value in long.items()}
    short = {name: value[:, :20].clone() for name, value in long.items()}

    visited = set()
    for spec in FEATURE_REGISTRY.feature_specs:
        if spec.name not in names:
            continue
        visited.add(spec.name)
        torch.testing.assert_close(
            spec.compute(long)[:, :20],
            spec.compute(short),
            rtol=0,
            atol=0,
        )
    assert visited == names


@settings(max_examples=25, deadline=None)
@given(cut=st.integers(min_value=20, max_value=48))
def test_all_registered_features_are_prefix_invariant(cut: int) -> None:
    raw = make_ohlcv()
    changed = mutate_future(raw, cut)
    visited = set()
    for spec in FEATURE_REGISTRY.feature_specs:
        visited.add(spec.name)
        assert_prefix_equal(spec.compute(raw), spec.compute(changed), cut)
    assert visited == set(FEATURE_REGISTRY.feature_names)


def make_operator_inputs(arity: int, t: int = 64) -> list[torch.Tensor]:
    index = torch.arange(t, dtype=torch.float32).unsqueeze(0)
    return [
        torch.sin(index / (3.0 + operand))
        + 0.02 * index
        + 0.25 * (operand + 1)
        for operand in range(arity)
    ]


def make_true_append_operator_inputs(
    arity: int,
    t: int,
    seed: int,
    noncontiguous: bool,
) -> list[torch.Tensor]:
    operands = []
    for operand in range(arity):
        generator = torch.Generator().manual_seed(seed + 101 * operand)
        storage_t = t * 2 if noncontiguous else t
        storage = torch.randn(
            2, storage_t, dtype=torch.float32, generator=generator
        )
        operands.append(storage[:, ::2] if noncontiguous else storage)
    return operands


@pytest.mark.parametrize(
    ("short_t", "long_t", "seed", "noncontiguous"),
    [
        pytest.param(63, 64, 20260715, False, id="adjacent-lengths"),
        pytest.param(600, 700, 20260715, False, id="reviewer-shape"),
        pytest.param(600, 700, 20260715, True, id="noncontiguous"),
        pytest.param(31, 96, 19, False, id="long-tail"),
    ],
)
def test_all_registered_operators_are_invariant_to_true_append(
    short_t: int,
    long_t: int,
    seed: int,
    noncontiguous: bool,
) -> None:
    visited = set()
    for spec in OPERATOR_REGISTRY.operator_specs:
        visited.add(spec.name)
        long_operands = make_true_append_operator_inputs(
            spec.arity, long_t, seed, noncontiguous
        )
        if spec.name == "PRODUCT_5":
            long_operands = [torch.tanh(value) * 0.5 for value in long_operands]
        short_operands = [
            value[:, :short_t]
            if noncontiguous
            else value[:, :short_t].clone()
            for value in long_operands
        ]

        short_result = spec.transform(*short_operands)
        long_prefix = spec.transform(*long_operands)[..., :short_t]
        try:
            torch.testing.assert_close(
                short_result,
                long_prefix,
                rtol=0,
                atol=1e-6,
            )
        except AssertionError as error:
            raise AssertionError(
                f"{spec.name} changed under true append "
                f"(short_t={short_t}, long_t={long_t}, "
                f"noncontiguous={noncontiguous})"
            ) from error

    assert visited == set(OPERATOR_REGISTRY.operator_names)


@settings(max_examples=25, deadline=None)
@given(cut=st.integers(min_value=20, max_value=48))
def test_all_registered_operators_are_prefix_invariant(cut: int) -> None:
    visited = set()
    for spec in OPERATOR_REGISTRY.operator_specs:
        visited.add(spec.name)
        inputs = make_operator_inputs(spec.arity)
        changed = [value.clone() for value in inputs]
        for operand, value in enumerate(changed):
            value[:, cut + 1:] = value[:, cut + 1:] * 1.37 + 0.1 * (operand + 1)
        assert_prefix_equal(
            spec.transform(*inputs),
            spec.transform(*changed),
            cut,
        )
    assert visited == set(OPERATOR_REGISTRY.operator_names)


def token_id(name: str) -> int:
    return FORMULA_VOCAB.token_names.index(name)


@settings(max_examples=25, deadline=None)
@given(cut=st.integers(min_value=20, max_value=48))
def test_vm_formulas_are_prefix_invariant(cut: int) -> None:
    raw = make_ohlcv()
    changed = mutate_future(raw, cut)
    features = MT5FeatureEngineer.compute_features(raw)
    changed_features = MT5FeatureEngineer.compute_features(changed)
    vm = StackVM()
    formulas = [
        [token_id("RET"), token_id("JUMP")],
        [
            token_id("RET5"),
            token_id("TS_MEAN_5"),
            token_id("TS_ZSCORE_10"),
        ],
    ]
    for formula in formulas:
        left = vm.execute(formula, features)
        right = vm.execute(formula, changed_features)
        assert left is not None and right is not None
        assert_prefix_equal(left, right, cut)
