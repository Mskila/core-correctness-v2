import torch
from hypothesis import given, settings, strategies as st

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
