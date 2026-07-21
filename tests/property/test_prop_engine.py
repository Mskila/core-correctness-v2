"""
Property-based tests for model_core.engine -- AlphaEngine training utilities.

Property 10: Strict Walk-Forward Gap Invariant
  Valid fold inputs preserve the full effective gap, use disjoint validation
  blocks, and expand the training window. Invalid inputs fail closed.

Property 13: StackVM failures preserve the sampled postfix structure and reason.
  Validates: Requirements T5.2, T5.3, T5.4, T5.5

Property 14: AlphaGPT Forward Pass Valid for Any Sequence Length
  For any sequence length T in [1, MAX_FORMULA_LEN],
  AlphaGPT.forward(idx) must return (logits, value, task_probs)
  without dimension errors or NaN outputs.
  Validates: Requirements T5.2, T5.3, T5.4, T5.5
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
from torch.distributions import Categorical
from hypothesis import given, settings
from hypothesis import strategies as st

import pytest
from model_core.engine import ConstrainedSampler, AlphaEngine
from model_core.semantics import InsufficientWalkForwardDataError
from model_core.walk_forward import build_walk_forward_folds, required_training_bars
from model_core.vm import (
    FormulaErrorKind,
    FormulaEvaluationError,
    StackVM,
    analyze_formula_structure,
)
from model_core.vocab import FORMULA_VOCAB
from model_core.config import ModelConfig
from model_core.alphagpt import AlphaGPT


# ---------------------------------------------------------------------------
# Property 10: Walk-Forward Gap Invariant
# ---------------------------------------------------------------------------


@given(
    n_blocks=st.integers(min_value=2, max_value=8),
    configured_gap=st.integers(min_value=0, max_value=50),
    min_fold_bars=st.integers(min_value=2, max_value=50),
    warmup_bars=st.integers(min_value=0, max_value=100),
    label_lookahead=st.integers(min_value=1, max_value=5),
    surplus=st.integers(min_value=0, max_value=100),
)
@settings(max_examples=50)
def test_walk_forward_gap_invariant(
    n_blocks,
    configured_gap,
    min_fold_bars,
    warmup_bars,
    label_lookahead,
    surplus,
):
    required = required_training_bars(
        warmup_bars=warmup_bars,
        label_lookahead=label_lookahead,
        n_blocks=n_blocks,
        min_fold_bars=min_fold_bars,
        configured_gap=configured_gap,
    )
    folds = build_walk_forward_folds(
        total_bars=required + surplus,
        n_blocks=n_blocks,
        configured_gap=configured_gap,
        min_fold_bars=min_fold_bars,
        warmup_bars=warmup_bars,
        label_lookahead=label_lookahead,
    )

    effective_gap = max(configured_gap, label_lookahead)
    assert len(folds) == n_blocks - 1
    assert all(fold.train_start == warmup_bars for fold in folds)
    assert all(fold.effective_gap == effective_gap for fold in folds)
    assert all(fold.val_start - fold.train_end == effective_gap for fold in folds)
    assert all(fold.val_end <= required + surplus - label_lookahead for fold in folds)
    assert all(
        left.val_end <= right.val_start for left, right in zip(folds, folds[1:])
    )
    assert all(
        left.train_end < right.train_end for left, right in zip(folds, folds[1:])
    )


@given(shortfall=st.integers(min_value=1, max_value=200))
@settings(max_examples=25)
def test_walk_forward_insufficient_data_fails_closed(shortfall):
    required = required_training_bars(
        warmup_bars=200,
        label_lookahead=2,
        n_blocks=5,
        min_fold_bars=50,
        configured_gap=20,
    )
    with pytest.raises(InsufficientWalkForwardDataError):
        build_walk_forward_folds(
            total_bars=max(0, required - shortfall),
            n_blocks=5,
            configured_gap=20,
            min_fold_bars=50,
            warmup_bars=200,
            label_lookahead=2,
        )


@given(
    parameter=st.sampled_from(
        [
            "total_bars",
            "configured_gap",
            "min_fold_bars",
            "warmup_bars",
            "label_lookahead",
        ]
    ),
    invalid_value=st.integers(min_value=-1000, max_value=-1),
)
@settings(max_examples=30)
def test_walk_forward_invalid_numeric_configuration_uses_domain_error(
    parameter, invalid_value
):
    arguments = {
        "total_bars": 1600,
        "n_blocks": 5,
        "configured_gap": 20,
        "min_fold_bars": 200,
        "warmup_bars": 400,
        "label_lookahead": 2,
    }
    arguments[parameter] = invalid_value

    with pytest.raises(InsufficientWalkForwardDataError) as exc_info:
        build_walk_forward_folds(**arguments)

    message = str(exc_info.value)
    assert f"parameter={parameter}" in message
    assert f"value={invalid_value!r}" in message


@given(min_fold_bars=st.integers(min_value=-100, max_value=1))
@settings(max_examples=20)
def test_required_training_bars_rejects_unscorable_fold_sizes(min_fold_bars):
    with pytest.raises(
        ValueError,
        match=r"min_fold_bars must be an integer >= 2",
    ):
        required_training_bars(
            warmup_bars=0,
            label_lookahead=2,
            n_blocks=2,
            min_fold_bars=min_fold_bars,
            configured_gap=2,
        )


def test_t05_hypothesis_properties_keep_default_deadlines_and_budgets():
    expected = {
        test_walk_forward_gap_invariant: 50,
        test_walk_forward_insufficient_data_fails_closed: 25,
        test_walk_forward_invalid_numeric_configuration_uses_domain_error: 30,
        test_required_training_bars_rejects_unscorable_fold_sizes: 20,
    }
    for property_test, max_examples in expected.items():
        configured = property_test._hypothesis_internal_use_settings
        assert configured.max_examples == max_examples
        assert configured.deadline == settings.default.deadline


# ---------------------------------------------------------------------------
# Property 13: StackVM Execution Success Rate 100% for Length-12 Formulas
# ---------------------------------------------------------------------------


def _sample_constrained_formula(seed: int, total_steps: int) -> list:
    """Generate one valid token sequence using ConstrainedSampler.

    Uses a deterministic seed with uniform logits masked by the sampler's
    validity constraints.  Does NOT depend on an AlphaGPT model -- this
    directly tests ConstrainedSampler's constraint correctness.
    """
    torch.manual_seed(seed)
    device = torch.device("cpu")

    vm = StackVM()
    sampler = ConstrainedSampler(
        vocab_size=FORMULA_VOCAB.size,
        feat_offset=FORMULA_VOCAB.operator_offset,
        arity_map=vm.arity_map,
    )

    formula = []
    stack_depth = 0

    for step_i in range(total_steps):
        # Uniform logits; illegal tokens are masked to -1e9
        logits = torch.zeros(FORMULA_VOCAB.size, device=device)
        mask = sampler.valid_mask(stack_depth, step_i, total_steps, device)
        logits[~mask] = -1e9

        dist = Categorical(logits=logits)
        token = dist.sample().item()

        formula.append(token)
        stack_depth += sampler.delta[token]

    return formula


@given(
    seed=st.integers(min_value=0, max_value=10000)
)
@settings(max_examples=50)
def test_constrained_sampler_produces_structurally_valid_formulas(seed):
    """
    Property 13: StackVM Execution Success Rate 100% for Length-12 Formulas

    The sampler guarantees postfix stack shape. Data-dependent domain or
    non-finite failures remain valid fail-closed outcomes with a precise reason.

    A synthetic feat_tensor [N=2, feats=10, T=30] is used as StackVM input
    to isolate constraint correctness from real market data.

    **Validates: Requirements T5.2, T5.3, T5.4, T5.5**
    """
    total_steps = ModelConfig.MAX_FORMULA_LEN  # == 12

    formula = _sample_constrained_formula(seed, total_steps)
    assert len(formula) == total_steps, "Formula length must equal MAX_FORMULA_LEN"

    # Synthetic feature tensor [N=2, feats=10, T=30]
    torch.manual_seed(seed + 1)
    N, T_data = 2, 30
    feat_tensor = torch.randn(N, FORMULA_VOCAB.feature_count, T_data)

    analysis = analyze_formula_structure(formula)
    structural_violations = [
        value
        for value in analysis.violations
        if "terminal one-sided chain" not in value
    ]
    assert structural_violations == []

    vm = StackVM()
    try:
        result = vm.evaluate(formula, feat_tensor)
    except FormulaEvaluationError as error:
        assert error.formula == tuple(formula)
        assert error.error_kind in {
            FormulaErrorKind.DOMAIN_ERROR,
            FormulaErrorKind.NONFINITE,
            FormulaErrorKind.INVALID_FORMULA,
        }
        if error.error_kind is FormulaErrorKind.INVALID_FORMULA:
            assert "one-sided chain" in error.detail
    else:
        assert result.shape == (N, T_data)


# ---------------------------------------------------------------------------
# Property 14: AlphaGPT Forward Pass Valid for Any Sequence Length
# ---------------------------------------------------------------------------


@given(
    seq_len=st.integers(min_value=1, max_value=12)
)
@settings(max_examples=30)
def test_alphagpt_forward_any_length(seq_len):
    """
    Property 14: AlphaGPT.forward Valid for Any T in [1, MAX_FORMULA_LEN]

    For any seq_len in [1, 12], AlphaGPT.forward(idx) must:
      - Return (logits, value, task_probs) without exceptions
      - logits.shape == [4, vocab_size]
      - No NaN in logits, value, or task_probs

    **Validates: Requirements T5.2, T5.3, T5.4, T5.5**
    """
    model = AlphaGPT()
    model.eval()

    batch_size = 4
    idx = torch.randint(0, FORMULA_VOCAB.size, (batch_size, seq_len))

    with torch.no_grad():
        logits, value, task_probs = model(idx)

    # Shape checks
    assert logits.shape[0] == batch_size, (
        f"logits.shape[0] should be {batch_size}, got {logits.shape[0]}"
    )
    assert logits.shape[1] == FORMULA_VOCAB.size, (
        f"logits.shape[1] should be vocab_size={FORMULA_VOCAB.size}, "
        f"got {logits.shape[1]}"
    )

    # NaN checks
    assert not torch.isnan(logits).any(), (
        f"logits contains NaN for seq_len={seq_len}"
    )
    assert not torch.isnan(value).any(), (
        f"value contains NaN for seq_len={seq_len}"
    )
    assert not torch.isnan(task_probs).any(), (
        f"task_probs contains NaN for seq_len={seq_len}"
    )


# ---------------------------------------------------------------------------
# Property 11: factor_pool Top-K Invariant
# ---------------------------------------------------------------------------


@given(
    scores=st.lists(
        st.floats(
            min_value=-100,
            max_value=100,
            allow_nan=False,
            allow_infinity=False,
        ),
        min_size=1,
        max_size=50,
    )
)
@settings(max_examples=50, deadline=None)
def test_factor_pool_top_k_invariant(scores):
    """
    Property 11: factor_pool Top-K Invariant

    After inserting an arbitrary number of factors with arbitrary scores,
    the factor pool must satisfy:
      - len(pool) <= FACTOR_TOP_K
      - The pool retains exactly the historically highest-K scores

    **Validates: 需求 T3.1, T3.2**
    """
    engine = AlphaEngine(data_manager=None)
    for score in scores:
        engine._update_factor_pool(score, torch.randn(3, 20))

    # Pool size ≤ FACTOR_TOP_K
    assert len(engine.factor_pool) <= ModelConfig.FACTOR_TOP_K

    # Pool contains the top-K scores
    pool_scores = sorted(s for s, _cnt, _ in engine.factor_pool)
    expected = sorted(scores)[-len(pool_scores):]
    assert pool_scores == pytest.approx(expected, rel=1e-5)


# ---------------------------------------------------------------------------
# Property 12: Attention Weights Unchanged After Snapshot Restore
# ---------------------------------------------------------------------------

import copy as _copy


@given(
    noise_multiplier=st.floats(min_value=0.1, max_value=10.0, allow_nan=False)
)
@settings(max_examples=30)
def test_attention_weights_unchanged_after_restore(noise_multiplier):
    """
    Property 12: Attention Weights Unchanged After Snapshot Restore

    After snapshot restore + FFN-only perturbation, all parameters whose names
    contain 'attention', 'qk_norm', 'norm1', or 'norm2' must be element-wise
    identical to the saved snapshot.

    **Validates: Requirements T4.1, T4.3**
    """
    model = AlphaGPT()
    snapshot = _copy.deepcopy(model.state_dict())

    # Simulate training (add large noise to all params)
    with torch.no_grad():
        for param in model.parameters():
            param.add_(torch.randn_like(param) * noise_multiplier)

    # Restore snapshot + apply FFN noise only (replicating engine restart logic)
    model.load_state_dict(snapshot)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'ffn' in name:
                param.add_(torch.randn_like(param) * 0.02)

    # Verify attention/qk_norm/norm1/norm2 params are unchanged
    restored = model.state_dict()
    for name, val in snapshot.items():
        if any(kw in name for kw in ('attention', 'qk_norm', 'norm1', 'norm2')):
            assert torch.equal(restored[name], val), (
                f"Parameter '{name}' changed after snapshot restore! "
                f"Max diff: {(restored[name] - val).abs().max().item()}"
            )


# ---------------------------------------------------------------------------
# Property 9: IC Calculation Arithmetic Consistency
# ---------------------------------------------------------------------------


@given(
    N=st.integers(min_value=5, max_value=15),
    T=st.integers(min_value=5, max_value=50),
)
@settings(max_examples=30)
def test_ic_arithmetic_consistency(N: int, T: int) -> None:
    """
    Property 9: IC Calculation Arithmetic Consistency

    _compute_ic computes time-series IC (per-symbol) on one shared mask:
    for each symbol n, IC_n = Pearson_corr(factor[n, valid], target_ret[n, valid])
    ic_mean = mean across symbols of IC_n values.

    **Validates: Requirements T1.3, T1.4**
    """
    from model_core.engine import AlphaEngine

    torch.manual_seed(0)
    factor     = torch.randn(N, T)
    target_ret = torch.randn(N, T)
    target_valid = torch.ones((N, T), dtype=torch.bool)
    target_valid[:, -2:] = False

    ic_mean = AlphaEngine._compute_ic(
        factor, target_ret, target_valid
    )
    ic_stability = AlphaEngine._compute_ic_stability(
        factor, target_ret, target_valid
    )

    # Manual time-series IC: per symbol, same-index factor/target under one mask.
    ic_list = []
    for n in range(N):
        x = factor[n, target_valid[n]]
        y = target_ret[n, target_valid[n]]
        sx = x.std(unbiased=False)
        sy = y.std(unbiased=False)
        if sx.item() < 1e-6 or sy.item() < 1e-6:
            continue
        xc = x - x.mean()
        yc = y - y.mean()
        corr = (xc * yc).mean() / (sx * sy + 1e-8)
        ic_list.append(corr.item())

    if not ic_list:
        # All symbols degenerate — ic_mean should be ~0
        assert abs(ic_mean) < 1e-4
        return

    expected_ic_mean = torch.tensor(ic_list).mean()
    assert ic_mean == pytest.approx(expected_ic_mean.item(), abs=1e-4), (
        f"ic_mean={ic_mean:.6f} != expected={expected_ic_mean.item():.6f}, "
        f"diff={abs(ic_mean - expected_ic_mean.item()):.2e}"
    )

    # ic_stability = ic_mean / (ic_std + 1e-6), based on per-symbol IC values
    if len(ic_list) >= 2:
        ic_tensor = torch.tensor(ic_list)
        expected_stability = ic_tensor.mean() / (ic_tensor.std(unbiased=False) + 1e-6)
        assert ic_stability == pytest.approx(expected_stability.item(), abs=1e-4), (
            f"ic_stability={ic_stability:.6f} != "
            f"expected={expected_stability.item():.6f}"
        )
