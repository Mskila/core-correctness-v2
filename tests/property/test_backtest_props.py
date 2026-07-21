"""Properties of mask-aligned shared-execution backtest scoring."""

import pytest
import torch
from hypothesis import assume, given, settings, strategies as st

from model_core.backtest import MT5Backtest
from model_core.execution import performance_metrics
from model_core.walk_forward import build_walk_forward_folds, required_training_bars


def inputs(length: int, valid_bars: int):
    factors = torch.linspace(-1.0, 1.0, length).unsqueeze(0)
    target_ret = torch.cos(torch.arange(length, dtype=torch.float32)).unsqueeze(0) * 0.01
    target_valid = torch.arange(length).unsqueeze(0) < valid_bars
    bar_time_ns = (
        torch.arange(length, dtype=torch.int64).unsqueeze(0)
        * 3_600
        * 1_000_000_000
    )
    return factors, target_ret, target_valid, bar_time_ns


@pytest.fixture(scope="module", autouse=True)
def _warm_backtest_runtime() -> None:
    values = inputs(8, 6)
    factors, target_ret, target_valid, bar_time_ns = values
    baseline = MT5Backtest().evaluate_segment(*values)
    factors_changed = factors.clone()
    target_changed = target_ret.clone()
    factors_changed[~target_valid] = 1.0e10
    target_changed[~target_valid] = -1.0e10
    changed = MT5Backtest().evaluate_segment(
        factors_changed, target_changed, target_valid, bar_time_ns
    )
    torch.testing.assert_close(changed, baseline)


@given(
    length=st.integers(min_value=8, max_value=80),
    invalid_tail=st.integers(min_value=2, max_value=5),
)
@settings(max_examples=100)
def test_masked_tail_is_irrelevant_to_segment_score(length, invalid_tail):
    invalid_tail = min(invalid_tail, length - 3)
    values = inputs(length, length - invalid_tail)
    factors, target_ret, target_valid, bar_time_ns = values
    baseline = MT5Backtest().evaluate_segment(*values)

    factors_changed = factors.clone()
    target_changed = target_ret.clone()
    factors_changed[~target_valid] = 1.0e10
    target_changed[~target_valid] = -1.0e10
    changed = MT5Backtest().evaluate_segment(
        factors_changed, target_changed, target_valid, bar_time_ns
    )
    torch.testing.assert_close(changed, baseline)


@given(
    length=st.integers(min_value=8, max_value=80),
    base_cost=st.floats(
        min_value=1.0e-6,
        max_value=0.005,
        allow_nan=False,
        allow_infinity=False,
    ),
)
@settings(max_examples=100)
def test_cost_stress_rerun_cannot_improve_shared_net_pnl(length, base_cost):
    values = inputs(length, length - 2)
    bt = MT5Backtest(cost_rate=base_cost)
    normal = bt._run_execution(*values, cost_rate=base_cost)
    stressed = bt._run_execution(*values, cost_rate=base_cost * 2.0)
    assert stressed.net_pnl.sum().item() <= normal.net_pnl.sum().item() + 1e-8


@given(
    length=st.integers(min_value=8, max_value=80),
    base_cost=st.floats(
        min_value=1.0e-6,
        max_value=0.001,
        allow_nan=False,
        allow_infinity=False,
    ),
)
@settings(max_examples=30)
def test_public_cost_stress_matches_shared_execution_at_double_cost(length, base_cost):
    values = inputs(length, length - 2)
    factors, target_ret, target_valid, bar_time_ns = values
    bt = MT5Backtest(cost_rate=base_cost)
    stressed = bt._run_execution(*values, cost_rate=base_cost * 2.0)
    expected = max(-5.0, min(5.0, performance_metrics(stressed).sortino))

    assert bt._cost_stress(
        factors, target_ret, target_valid, bar_time_ns
    ) == pytest.approx(expected)


@given(
    length=st.integers(min_value=12, max_value=80),
    bounds=st.tuples(*(st.integers(min_value=-2, max_value=82) for _ in range(4))),
)
@settings(max_examples=60)
def test_evaluate_fold_rejects_every_invalid_integer_boundary_tuple(length, bounds):
    train_start, train_end, val_start, val_end = bounds
    is_valid = (
        0 <= train_start < train_end < val_start < val_end
        and train_end - train_start >= 2
        and val_end - val_start >= 2
        and train_end + 2 <= length
        and val_end + 2 <= length
    )
    assume(not is_valid)
    with pytest.raises(
        ValueError,
        match=r"field=(fold_bounds|train_bars|val_bars|train_end|val_end)",
    ):
        MT5Backtest().evaluate_fold(
            *inputs(length, length - 2),
            train_start=train_start,
            train_end=train_end,
            val_start=val_start,
            val_end=val_end,
        )


@given(
    train_bars=st.integers(min_value=8, max_value=12),
    gap=st.integers(min_value=1, max_value=8),
    val_bars=st.integers(min_value=8, max_value=12),
)
@settings(max_examples=30)
def test_evaluate_fold_accepts_valid_disjoint_ranges(train_bars, gap, val_bars):
    val_start = train_bars + gap
    val_end = val_start + val_bars
    length = val_end + 2
    train_score, val_score = MT5Backtest().evaluate_fold(
        *inputs(length, length - 2),
        train_start=0,
        train_end=train_bars,
        val_start=val_start,
        val_end=val_end,
    )
    assert torch.isfinite(train_score)
    assert torch.isfinite(val_score)


@given(
    field=st.sampled_from(["train_start", "train_end", "val_start", "val_end"]),
    exponent=st.integers(min_value=1000, max_value=5000),
    sign=st.sampled_from([-1, 1]),
)
@settings(max_examples=30)
def test_excessive_exact_fold_boundaries_have_bounded_field_diagnostics(
    field, exponent, sign
):
    bounds = dict(train_start=0, train_end=8, val_start=10, val_end=18)
    bounds[field] = sign * (10**exponent)

    with pytest.raises(ValueError) as caught:
        MT5Backtest().evaluate_fold(
            *inputs(20, 18),
            **bounds,
        )

    message = str(caught.value)
    assert len(message) <= 1024
    assert f"field={field}" in message
    assert "actual=out of operational bounds" in message


@given(
    warmup_bars=st.integers(min_value=0, max_value=10),
    configured_gap=st.integers(min_value=0, max_value=10),
)
@settings(max_examples=30)
def test_smallest_builder_fold_is_always_scorable(warmup_bars, configured_gap):
    total_bars = required_training_bars(
        warmup_bars=warmup_bars,
        label_lookahead=2,
        n_blocks=2,
        min_fold_bars=200,
        configured_gap=configured_gap,
    )
    fold = build_walk_forward_folds(
        total_bars=total_bars,
        n_blocks=2,
        configured_gap=configured_gap,
        min_fold_bars=200,
        warmup_bars=warmup_bars,
        label_lookahead=2,
    )[0]
    factors = torch.zeros((1, total_bars), dtype=torch.float32)
    target_ret = torch.zeros_like(factors)
    for start, end in (
        (fold.train_start, fold.train_end),
        (fold.val_start, fold.val_end),
    ):
        factors[:, start:end] = torch.where(
            torch.arange(end - start).unsqueeze(0) % 2 == 0,
            1.0,
            -1.0,
        )
        target_ret[:, start:end] = 0.01
    target_valid = torch.arange(total_bars).unsqueeze(0) < total_bars - 2
    bar_time_ns = (
        torch.arange(total_bars, dtype=torch.int64).unsqueeze(0)
        * 3_600
        * 1_000_000_000
    )

    train_score, val_score = MT5Backtest().evaluate_fold(
        factors=factors,
        target_ret=target_ret,
        target_valid=target_valid,
        bar_time_ns=bar_time_ns,
        train_start=fold.train_start,
        train_end=fold.train_end,
        val_start=fold.val_start,
        val_end=fold.val_end,
    )
    assert torch.isfinite(train_score)
    assert torch.isfinite(val_score)


@given(
    dtype=st.sampled_from([torch.float16, torch.float32, torch.float64]),
    scale=st.sampled_from([0.5, 1.0, 2.0, 8.0]),
)
@settings(max_examples=30)
def test_public_ic_is_finite_and_scale_equivalent(dtype, scale):
    from model_core.backtest import compute_ic_metrics

    base = torch.tensor([[1.0, 2.0, 4.0, 8.0]], dtype=dtype)
    target = torch.tensor([[1.0, 3.0, 2.0, 5.0]], dtype=dtype)
    valid = torch.ones_like(base, dtype=torch.bool)

    baseline, baseline_stability = compute_ic_metrics(base, target, valid)
    scaled, scaled_stability = compute_ic_metrics(base * scale, target, valid)

    assert torch.isfinite(scaled)
    assert torch.isfinite(scaled_stability)
    torch.testing.assert_close(scaled, baseline, rtol=0, atol=2.0e-3)
    torch.testing.assert_close(
        scaled_stability, baseline_stability, rtol=0, atol=2.0e-3
    )
