"""Tests for V2 training scores backed by shared execution."""

import inspect
from numbers import Integral

import pytest
import torch

import model_core.backtest as backtest_module
from model_core.backtest import MT5Backtest, compute_ic_metrics
from model_core.execution import ExecutionResult, performance_metrics
from model_core.semantics import DataValidationError
from model_core.walk_forward import build_walk_forward_folds


def segment_inputs(length: int = 12):
    factors = torch.linspace(-1.0, 1.0, length).unsqueeze(0)
    target_ret = torch.sin(torch.arange(length, dtype=torch.float32)).unsqueeze(0) * 0.01
    target_valid = torch.ones_like(factors, dtype=torch.bool)
    target_valid[:, -2:] = False
    bar_time_ns = (
        torch.arange(length, dtype=torch.int64).unsqueeze(0)
        * 3_600
        * 1_000_000_000
    )
    return factors, target_ret, target_valid, bar_time_ns


def test_constructor_has_no_fixed_periods_per_year() -> None:
    assert "periods_per_year" not in inspect.signature(MT5Backtest).parameters
    with pytest.raises(TypeError):
        MT5Backtest(periods_per_year=6240)


def test_evaluate_segment_consumes_factor_label_mask_and_timestamps() -> None:
    score = MT5Backtest().evaluate_segment(*segment_inputs())
    assert isinstance(score, torch.Tensor)
    assert score.shape == torch.Size([])
    assert torch.isfinite(score)


def test_invalid_tail_extremes_do_not_change_score_or_ic() -> None:
    factors, target_ret, target_valid, bar_time_ns = segment_inputs()
    baseline = MT5Backtest().evaluate_segment(
        factors, target_ret, target_valid, bar_time_ns
    )
    changed_factors = factors.clone()
    changed_target = target_ret.clone()
    changed_factors[:, -2:] = torch.tensor([[-1.0e20, 1.0e20]])
    changed_target[:, -2:] = torch.tensor([[1.0e20, -1.0e20]])
    changed = MT5Backtest().evaluate_segment(
        changed_factors, changed_target, target_valid, bar_time_ns
    )

    torch.testing.assert_close(changed, baseline)
    assert MT5Backtest()._ts_ic_stability(
        changed_factors, changed_target, target_valid
    ) == pytest.approx(
        MT5Backtest()._ts_ic_stability(factors, target_ret, target_valid)
    )


@pytest.mark.parametrize("scale", [1.0, 1.0e-4, 1.0e-6, 1.0e-7])
@pytest.mark.parametrize("direction", [1.0, -1.0])
def test_shared_ic_is_scale_invariant_for_finite_proportional_series(
    scale, direction
) -> None:
    factors = torch.tensor([[1.0, 2.0, 4.0, 8.0]]) * scale
    target = factors * direction
    valid = torch.ones_like(factors, dtype=torch.bool)

    mean_ic, stability = compute_ic_metrics(factors, target, valid)

    assert mean_ic.item() == pytest.approx(direction, abs=1.0e-6)
    assert stability.item() == pytest.approx(direction, abs=1.0e-6)
    assert MT5Backtest()._ts_ic_stability(
        factors, target, valid
    ) == pytest.approx(direction, abs=1.0e-6)


def test_shared_ic_treats_only_constant_series_as_unscorable() -> None:
    factors = torch.tensor([[3.0, 3.0, 3.0, 3.0]])
    target = torch.tensor([[1.0, 2.0, 4.0, 8.0]])
    valid = torch.ones_like(factors, dtype=torch.bool)

    mean_ic, stability = compute_ic_metrics(factors, target, valid)

    assert mean_ic.item() == 0.0
    assert stability.item() == 0.0


def test_shared_ic_multi_symbol_zero_dispersion_uses_mean_ic() -> None:
    factors = torch.tensor(
        [[1.0, 2.0, 4.0, 8.0], [2.0, 3.0, 5.0, 9.0]]
    )
    target = factors * torch.tensor([[2.0], [0.25]])
    valid = torch.ones_like(factors, dtype=torch.bool)

    mean_ic, stability = compute_ic_metrics(factors, target, valid)

    assert mean_ic.item() == pytest.approx(1.0)
    assert stability.item() == pytest.approx(1.0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
@pytest.mark.parametrize("magnitude", ["minimum_subnormal", "maximum_finite"])
def test_ic_and_public_score_have_finite_nonzero_directional_gradients_at_finite_extremes(
    dtype: torch.dtype,
    magnitude: str,
) -> None:
    info = torch.finfo(dtype)
    if magnitude == "minimum_subnormal":
        base = torch.nextafter(
            torch.tensor(0.0, dtype=dtype), torch.tensor(1.0, dtype=dtype)
        )
    else:
        base = torch.tensor(info.max / 8.0, dtype=dtype)
    values = base * torch.tensor([1.0, 2.0, 4.0, 8.0], dtype=dtype)
    target_values = torch.tensor([1.0, 3.0, 2.0, 5.0], dtype=dtype)
    valid = torch.ones((1, 4), dtype=torch.bool)

    direct_factors = values.unsqueeze(0).clone().requires_grad_(True)
    direct_ic, direct_stability = compute_ic_metrics(
        direct_factors, target_values.unsqueeze(0), valid
    )
    direct_ic.backward()

    assert torch.isfinite(direct_ic)
    assert torch.isfinite(direct_stability)
    assert direct_ic.item() == pytest.approx(0.8669214469, abs=2.0e-3)
    assert direct_factors.grad is not None
    assert torch.isfinite(direct_factors.grad).all()
    assert torch.count_nonzero(direct_factors.grad) == direct_factors.numel()
    assert torch.dot(
        direct_factors.grad.reshape(-1).to(torch.float64),
        target_values.to(torch.float64),
    ).item() > 0.0

    public_factors = torch.cat(
        [values, values[:2]], dim=0
    ).unsqueeze(0).clone().requires_grad_(True)
    public_target = torch.tensor(
        [[0.01, -0.03, 0.02, -0.05, 0.0, 0.0]], dtype=dtype
    )
    public_valid = torch.tensor([[True, True, True, True, False, False]])
    times = (
        torch.arange(6, dtype=torch.int64).unsqueeze(0)
        * 3_600
        * 1_000_000_000
    )
    score = MT5Backtest(cost_rate=0.0).evaluate_segment(
        public_factors, public_target, public_valid, times
    )
    score.backward()

    assert torch.isfinite(score)
    assert public_factors.grad is not None
    scored_gradient = public_factors.grad[0, :4]
    assert torch.isfinite(scored_gradient).all()
    assert torch.count_nonzero(scored_gradient) == scored_gradient.numel()
    assert torch.dot(
        scored_gradient.to(torch.float64), public_target[0, :4].to(torch.float64)
    ).item() > 0.0


def test_extreme_finite_float64_ic_is_finite_scale_invariant_and_directional() -> None:
    maximum = torch.finfo(torch.float64).max
    factors = torch.tensor(
        [[maximum, maximum, -maximum, -maximum]],
        dtype=torch.float64,
        requires_grad=True,
    )
    target = factors.detach().clone()
    valid = torch.ones_like(factors, dtype=torch.bool)

    mean_ic, stability = compute_ic_metrics(factors, target, valid)

    assert torch.isfinite(mean_ic)
    assert torch.isfinite(stability)
    assert mean_ic.item() == pytest.approx(1.0)
    assert stability.item() == pytest.approx(1.0)

    perturbed = torch.tensor(
        [[maximum, maximum * 0.5, -maximum * 0.25, -maximum * 0.75]],
        dtype=torch.float64,
        requires_grad=True,
    )
    directional_target = torch.tensor(
        [[maximum * 0.8, maximum * 0.6, -maximum * 0.5, -maximum * 0.9]],
        dtype=torch.float64,
    )
    directional_ic, _ = compute_ic_metrics(perturbed, directional_target, valid)
    directional_ic.backward()
    assert torch.isfinite(directional_ic)
    assert perturbed.grad is not None
    assert torch.isfinite(perturbed.grad).all()
    assert perturbed.grad.abs().sum().item() > 0.0
    assert torch.dot(
        perturbed.grad.reshape(-1), directional_target.reshape(-1)
    ).item() > 0.0


def test_public_segment_score_is_finite_for_extreme_finite_float64_factors() -> None:
    maximum = torch.finfo(torch.float64).max
    factors = torch.tensor(
        [[maximum, maximum, -maximum, -maximum, maximum, -maximum]],
        dtype=torch.float64,
        requires_grad=True,
    )
    target = torch.tensor(
        [[0.01, -0.02, 0.01, -0.02, 0.005, -0.005]], dtype=torch.float64
    )
    valid = torch.ones_like(factors, dtype=torch.bool)
    valid[:, -2:] = False
    times = (
        torch.arange(factors.shape[1], dtype=torch.int64).unsqueeze(0)
        * 3_600
        * 1_000_000_000
    )

    score = MT5Backtest(cost_rate=0.0).evaluate_segment(
        factors, target, valid, times
    )
    score.backward()

    assert torch.isfinite(score)
    assert factors.grad is not None
    assert torch.isfinite(factors.grad).all()


def test_finite_float32_annualized_return_is_not_narrowed_to_infinity() -> None:
    factors = torch.tensor(
        [[0.2, 0.4, 0.6, 0.8, 0.0, 0.0]],
        dtype=torch.float32,
        requires_grad=True,
    )
    target = torch.tensor(
        [[0.05, -0.01, 0.05, 0.03, 0.0, 0.0]], dtype=torch.float32
    )
    valid = torch.tensor([[True, True, True, True, False, False]])
    times = (
        torch.arange(6, dtype=torch.int64).unsqueeze(0)
        * 3_600
        * 1_000_000_000
    )

    score = MT5Backtest(cost_rate=0.0).evaluate_segment(
        factors, target, valid, times
    )
    score.backward()

    assert score.dtype == torch.float64
    assert torch.isfinite(score)
    assert score.item() > torch.finfo(torch.float32).max
    assert factors.grad is not None
    assert torch.isfinite(factors.grad).all()
    assert torch.count_nonzero(factors.grad[0, :4]) == 4


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
@pytest.mark.parametrize(
    "magnitude",
    ["moderate", "maximum_finite", "minimum_normal", "minimum_subnormal"],
)
def test_public_score_uses_a_finite_promoted_contract_across_factor_dtypes(
    dtype: torch.dtype,
    magnitude: str,
) -> None:
    info = torch.finfo(dtype)
    if magnitude == "moderate":
        base = torch.tensor(0.125, dtype=dtype)
    elif magnitude == "maximum_finite":
        base = torch.tensor(info.max / 16.0, dtype=dtype)
    elif magnitude == "minimum_normal":
        base = torch.tensor(info.tiny, dtype=dtype)
    else:
        base = torch.nextafter(
            torch.tensor(0.0, dtype=dtype), torch.tensor(1.0, dtype=dtype)
        )
    values = base * torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=dtype)
    factors = torch.cat([values, values[:2]]).unsqueeze(0).requires_grad_(True)
    target = torch.tensor(
        [[0.01, -0.02, 0.03, -0.01, 0.0, 0.0]], dtype=dtype
    )
    valid = torch.tensor([[True, True, True, True, False, False]])
    times = (
        torch.arange(6, dtype=torch.int64).unsqueeze(0)
        * 3_600
        * 1_000_000_000
    )

    score = MT5Backtest(cost_rate=0.0).evaluate_segment(
        factors, target, valid, times
    )
    score.backward()

    assert score.dtype == torch.float64
    assert torch.isfinite(score)
    assert factors.grad is not None
    assert torch.isfinite(factors.grad).all()
    assert torch.count_nonzero(factors.grad[0, :4]) == 4


def test_cost_stress_delegates_once_at_exactly_double_cost_and_is_strictly_worse(
    monkeypatch,
) -> None:
    factors = torch.tensor([[0.2, -0.2, 0.3, -0.3, 0.4, -0.4]])
    target = torch.tensor(
        [[0.0002, -0.0002, 0.0002, -0.0002, 0.0, 0.0]],
        dtype=factors.dtype,
    )
    valid = torch.ones_like(factors, dtype=torch.bool)
    valid[:, -2:] = False
    times = (
        torch.arange(factors.shape[1], dtype=torch.int64).unsqueeze(0)
        * 3_600
        * 1_000_000_000
    )
    backtest = MT5Backtest(cost_rate=1.0e-4)
    real_run = backtest._run_execution
    observed_costs = []

    def spy(*args, **kwargs):
        observed_costs.append(kwargs["cost_rate"])
        return real_run(*args, **kwargs)

    monkeypatch.setattr(backtest, "_run_execution", spy)
    stressed_score = backtest._cost_stress(factors, target, valid, times)
    normal = real_run(factors, target, valid, times, cost_rate=backtest.cost_rate)
    normal_score = performance_metrics(normal).sortino

    assert observed_costs == [backtest.cost_rate * 2.0]
    assert stressed_score < normal_score
    assert real_run(
        factors,
        target,
        valid,
        times,
        cost_rate=backtest.cost_rate * 2.0,
    ).net_pnl.sum().item() < normal.net_pnl.sum().item()


def _activity_result(position, turnover, valid=None) -> ExecutionResult:
    position = torch.tensor([position], dtype=torch.float32)
    turnover = torch.tensor([turnover], dtype=torch.float32)
    if valid is None:
        valid = [True] * position.shape[1]
    target_valid = torch.tensor([valid], dtype=torch.bool)
    zeros = torch.zeros_like(position)
    times = torch.arange(position.shape[1], dtype=torch.int64).unsqueeze(0)
    return ExecutionResult(
        position=position,
        turnover=turnover,
        gross_pnl=zeros,
        cost=zeros,
        net_pnl=zeros,
        target_valid=target_valid,
        bar_time_ns=times,
        final_liquidation_cost=torch.zeros(1),
    )


@pytest.mark.parametrize(
    ("position", "turnover", "expected_events", "expected_runs"),
    [
        ([0.0] * 6, [0.0] * 6, 0, []),
        ([0.5] * 6, [0.5] + [0.0] * 5, 1, [6]),
        ([0.5] * 3 + [-0.5] * 3, [0.5, 0.0, 0.0, 1.0, 0.0, 0.0], 2, [3, 3]),
        ([0.5, -0.5] * 3, [0.5] + [1.0] * 5, 6, [1] * 6),
    ],
)
def test_turnover_activity_counts_entries_and_reversals_from_execution_result(
    position, turnover, expected_events, expected_runs
) -> None:
    result = _activity_result(position, turnover)
    total_bars, event_counts, hold_runs = MT5Backtest._turnover_activity(result)

    assert total_bars == len(position)
    assert event_counts == [expected_events]
    assert hold_runs == expected_runs


def test_turnover_activity_uses_result_turnover_and_ignores_invalid_tail() -> None:
    result = _activity_result(
        [0.5, -0.5, 0.5, -0.5],
        [0.0, 0.0, 1.0, 1.0],
        [True, True, False, False],
    )

    total_bars, event_counts, hold_runs = MT5Backtest._turnover_activity(result)

    assert total_bars == 2
    assert event_counts == [0]
    assert hold_runs == [1, 1]

def test_half_consistency_splits_time_before_selecting_symbols() -> None:
    pnl = torch.tensor([[0.02] * 20, [-0.01] * 20])
    target_valid = torch.ones_like(pnl, dtype=torch.bool)
    backtest = MT5Backtest()

    assert backtest._half_consistency_bonus(pnl, target_valid, 1.0) == 0.5
    assert backtest._half_consistency_bonus(
        pnl.flip(0), target_valid.flip(0), 1.0
    ) == 0.5


def _sparse_temporal_witness(length: int) -> tuple[torch.Tensor, torch.Tensor]:
    midpoint = length // 2
    early = list(range(0, 10, 2))
    bridge = list(range(10, midpoint, 2))
    late = list(range(midpoint, length, 3))[:5]
    pnl = torch.zeros((2, length))
    target_valid = torch.zeros_like(pnl, dtype=torch.bool)
    for indices, positive_total, negative_total in (
        (early, 150.0, -50.0),
        (bridge, 20.0, -80.0),
        (late, 100.0, -90.0),
    ):
        pnl[0, indices] = positive_total / len(indices)
        pnl[1, indices] = negative_total / len(indices)
        target_valid[:, indices] = True
    return pnl, target_valid


@pytest.mark.parametrize("length", [28, 31])
def test_half_consistency_uses_dynamic_temporal_midpoint_with_sparse_symbols(
    length,
) -> None:
    pnl, target_valid = _sparse_temporal_witness(length)

    assert int(target_valid.sum()) >= 20
    assert MT5Backtest()._half_consistency_bonus(
        pnl, target_valid, 1.0
    ) == 0.5


def test_half_consistency_returns_zero_when_a_temporal_half_is_empty(
    monkeypatch,
) -> None:
    pnl = torch.tensor([[0.02] * 20, [-0.01] * 20])
    target_valid = torch.zeros_like(pnl, dtype=torch.bool)
    target_valid[:, 10:] = True
    backtest = MT5Backtest()
    sortino_inputs = []
    real_sortino = backtest._sortino

    def reject_empty_sortino(values, periods_per_year, eps=1e-8):
        assert values.numel() > 0
        sortino_inputs.append(values.clone())
        return real_sortino(values, periods_per_year, eps)

    monkeypatch.setattr(backtest, "_sortino", reject_empty_sortino)

    assert int(target_valid.sum()) >= 20
    assert backtest._half_consistency_bonus(
        pnl, target_valid, 1.0
    ) == 0.0
    assert sortino_inputs == []


def test_half_consistency_returns_zero_below_twenty_valid_observations() -> None:
    pnl = torch.ones((2, 10))
    target_valid = torch.ones_like(pnl, dtype=torch.bool)
    target_valid[0, 0] = False

    assert MT5Backtest()._half_consistency_bonus(
        pnl, target_valid, 1.0
    ) == 0.0


def test_evaluate_fold_executes_train_and_validation_slices_independently(
    monkeypatch,
) -> None:
    factors, target_ret, target_valid, bar_time_ns = segment_inputs(20)
    calls = []
    real_run_execution = backtest_module.run_execution

    def capture_execution(**kwargs):
        result = real_run_execution(**kwargs)
        calls.append(result)
        return result

    monkeypatch.setattr(backtest_module, "run_execution", capture_execution)
    train_score, val_score = MT5Backtest(cost_rate=0.01).evaluate_fold(
        factors=factors,
        target_ret=target_ret,
        target_valid=target_valid,
        bar_time_ns=bar_time_ns,
        train_start=0,
        train_end=8,
        val_start=10,
        val_end=18,
    )

    assert len(calls) == 2
    assert calls[0].position.shape == (1, 10)
    assert calls[1].position.shape == (1, 10)
    for result in calls:
        first_position = result.position[:, 0].abs()
        torch.testing.assert_close(result.turnover[:, 0], first_position)
        valid_count = int(result.target_valid.sum().item())
        torch.testing.assert_close(
            result.final_liquidation_cost,
            result.position[:, valid_count - 1].abs() * 0.01,
        )
    assert torch.isfinite(train_score)
    assert torch.isfinite(val_score)


def _evaluate_valid_fold(**overrides):
    names = ("factors", "target_ret", "target_valid", "bar_time_ns")
    kwargs = dict(zip(names, segment_inputs(20)))
    kwargs.update(
        train_start=0,
        train_end=8,
        val_start=10,
        val_end=18,
    )
    kwargs.update(overrides)
    return MT5Backtest().evaluate_fold(**kwargs)


class _IntSubclass(int):
    pass


class _ProtocolProbe:
    def __init__(self, calls: dict[str, int], *, hostile: bool = False) -> None:
        self.calls = calls
        self.hostile = hostile

    def _called(self, name: str, result=None):
        self.calls[name] = self.calls.get(name, 0) + 1
        if self.hostile:
            raise RuntimeError("hostile protocol executed")
        return result

    def __repr__(self) -> str:
        return self._called("repr", "protocol-probe")

    def __str__(self) -> str:
        return self._called("str", "protocol-probe")

    def __format__(self, spec: str) -> str:
        return self._called("format", "protocol-probe")

    def __int__(self) -> int:
        return self._called("int", 0)

    def __index__(self) -> int:
        return self._called("index", 0)

    def __eq__(self, other) -> bool:
        return self._called("equality", False)

    def __hash__(self) -> int:
        return self._called("hash", 0)


class _BenignIntegralProbe(_ProtocolProbe):
    pass


Integral.register(_BenignIntegralProbe)


_INVALID_FOLD_BOUNDARY_CASES = [
    pytest.param(lambda calls: True, "bool", id="bool"),
    pytest.param(lambda calls: 1.5, "float", id="float"),
    pytest.param(lambda calls: _IntSubclass(0), "_IntSubclass", id="int-subclass"),
    pytest.param(
        lambda calls: _BenignIntegralProbe(calls),
        "_BenignIntegralProbe",
        id="benign-integral-subclass",
    ),
    pytest.param(
        lambda calls: _ProtocolProbe(calls, hostile=True),
        "_ProtocolProbe",
        id="hostile-protocol-object",
    ),
    pytest.param(lambda calls: 10**1000, None, id="positive-10pow1000"),
    pytest.param(lambda calls: -(10**1000), None, id="negative-10pow1000"),
    pytest.param(lambda calls: 10**5000, None, id="positive-10pow5000"),
    pytest.param(lambda calls: -(10**5000), None, id="negative-10pow5000"),
]


@pytest.mark.parametrize(
    "field", ["train_start", "train_end", "val_start", "val_end"]
)
@pytest.mark.parametrize(("factory", "actual_type"), _INVALID_FOLD_BOUNDARY_CASES)
def test_evaluate_fold_boundary_preflight_is_exact_safe_and_side_effect_free(
    monkeypatch, field, factory, actual_type
) -> None:
    calls: dict[str, int] = {}
    invalid = factory(calls)
    backtest = MT5Backtest()

    def unexpected_side_effect(*args, **kwargs):
        raise AssertionError("fold slicing or scoring ran before boundary rejection")

    monkeypatch.setattr(backtest, "_fold_segment", unexpected_side_effect)
    monkeypatch.setattr(backtest, "_run_execution", unexpected_side_effect)
    monkeypatch.setattr(backtest, "_multi_objective", unexpected_side_effect)

    with pytest.raises(ValueError) as caught:
        values = dict(zip(
            ("factors", "target_ret", "target_valid", "bar_time_ns"),
            segment_inputs(20),
        ))
        values.update(train_start=0, train_end=8, val_start=10, val_end=18)
        values[field] = invalid
        backtest.evaluate_fold(**values)

    message = str(caught.value)
    assert len(message) <= 1024
    assert f"field={field}" in message
    if actual_type is None:
        assert "expected=exact built-in int within operational bounds" in message
        assert "actual=out of operational bounds" in message
    else:
        assert "expected=exact built-in int" in message
        assert f"actual_type={actual_type}" in message
    assert calls == {}


@pytest.mark.parametrize(
    ("field", "replacement", "expected", "actual"),
    [
        ("factors", lambda value: value.tolist(), "torch.Tensor", "list"),
        ("target_ret", lambda value: value.tolist(), "torch.Tensor", "list"),
        ("target_valid", lambda value: value.tolist(), "torch.Tensor", "list"),
        ("bar_time_ns", lambda value: value.tolist(), "torch.Tensor", "list"),
        ("factors", lambda value: value[0], "rank 2", "rank 1"),
        ("target_ret", lambda value: value[0], "rank 2", "rank 1"),
        ("target_valid", lambda value: value.unsqueeze(0), "rank 2", "rank 3"),
        ("bar_time_ns", lambda value: value[0], "rank 2", "rank 1"),
    ],
)
def test_evaluate_fold_rejects_non_tensor_and_rank_mismatches(
    field, replacement, expected, actual
) -> None:
    values = dict(zip(
        ("factors", "target_ret", "target_valid", "bar_time_ns"),
        segment_inputs(20),
    ))
    with pytest.raises(
        DataValidationError,
        match=rf"field={field}.*expected={expected}.*actual={actual}",
    ):
        _evaluate_valid_fold(**{field: replacement(values[field])})


@pytest.mark.parametrize(
    "field",
    ["factors", "target_ret", "target_valid", "bar_time_ns"],
)
def test_evaluate_fold_rejects_full_shape_mismatch_before_slicing(field) -> None:
    values = dict(zip(
        ("factors", "target_ret", "target_valid", "bar_time_ns"),
        segment_inputs(20),
    ))
    replacement = values[field][:, :-1]
    with pytest.raises(
        DataValidationError,
        match=rf"field={field}.*expected=shape \(1, 20\).*actual=shape \(1, 19\)",
    ):
        _evaluate_valid_fold(**{field: replacement})


@pytest.mark.parametrize(
    ("field", "replacement", "expected", "actual"),
    [
        ("factors", lambda value: value.to(torch.int64), "floating", "torch.int64"),
        ("target_ret", lambda value: value.to(torch.int64), "floating", "torch.int64"),
        (
            "target_valid",
            lambda value: value.to(torch.float32),
            "torch.bool",
            "torch.float32",
        ),
        (
            "bar_time_ns",
            lambda value: value.to(torch.int32),
            "torch.int64",
            "torch.int32",
        ),
        (
            "target_ret",
            lambda value: value.to(torch.float64),
            "torch.float32",
            "torch.float64",
        ),
    ],
)
def test_evaluate_fold_rejects_dtype_mismatches(
    field, replacement, expected, actual
) -> None:
    values = dict(zip(
        ("factors", "target_ret", "target_valid", "bar_time_ns"),
        segment_inputs(20),
    ))
    with pytest.raises(
        DataValidationError,
        match=rf"field={field}.*expected={expected}.*actual={actual}",
    ):
        _evaluate_valid_fold(**{field: replacement(values[field])})


def test_evaluate_fold_rejects_device_mismatch_before_slicing() -> None:
    _, target_ret, _, _ = segment_inputs(20)
    with pytest.raises(
        DataValidationError,
        match=r"field=target_ret.*expected=device cpu.*actual=device meta",
    ):
        _evaluate_valid_fold(target_ret=target_ret.to("meta"))


@pytest.mark.parametrize(
    "field", ["train_start", "train_end", "val_start", "val_end"]
)
@pytest.mark.parametrize("invalid", [True, 1.5])
def test_evaluate_fold_rejects_non_integer_fold_indices(field, invalid) -> None:
    with pytest.raises(
        ValueError,
        match=rf"field={field}.*expected=exact built-in int.*actual_type=",
    ):
        _evaluate_valid_fold(**{field: invalid})


@pytest.mark.parametrize(
    "bounds",
    [
        (-1, 8, 10, 18),
        (0, 0, 10, 18),
        (8, 4, 10, 18),
        (0, 8, 8, 18),
        (0, 8, 7, 18),
        (0, 8, 10, 10),
        (0, 8, 18, 10),
    ],
)
def test_evaluate_fold_rejects_invalid_boundary_relationships(bounds) -> None:
    with pytest.raises(
        ValueError,
        match=(
            r"field=fold_bounds.*expected=0 <= train_start < train_end < "
            r"val_start < val_end.*actual="
        ),
    ):
        _evaluate_valid_fold(**dict(zip(
            ("train_start", "train_end", "val_start", "val_end"),
            bounds,
        )))


def test_evaluate_fold_rejects_segment_without_two_exit_timestamps() -> None:
    with pytest.raises(
        ValueError,
        match=r"field=val_end.*expected=val_end \+ 2 <= common length 20.*actual=19",
    ):
        _evaluate_valid_fold(val_end=19)


@pytest.mark.parametrize(
    ("bounds", "field"),
    [
        ((0, 1, 3, 8), "train_bars"),
        ((0, 8, 10, 11), "val_bars"),
    ],
)
def test_evaluate_fold_rejects_segments_below_two_observations(
    bounds, field
) -> None:
    with pytest.raises(
        ValueError,
        match=rf"field={field}.*expected=at least 2 observations.*actual=1",
    ):
        _evaluate_valid_fold(**dict(zip(
            ("train_start", "train_end", "val_start", "val_end"),
            bounds,
        )))


def test_fold_segment_returns_independent_tensors_for_noncontiguous_inputs() -> None:
    base = torch.arange(24).reshape(2, 12)
    factors = base.to(torch.float32)[:, ::2]
    target_ret = (base.to(torch.float32) / 100.0)[:, ::2]
    target_valid = (base % 3 != 0)[:, ::2]
    bar_time_ns = (base.to(torch.int64) * 1_000_000_000)[:, ::2]
    inputs = (factors, target_ret, target_valid, bar_time_ns)
    snapshots = tuple(value.clone() for value in inputs)

    assert all(not value.is_contiguous() for value in inputs)
    segment = MT5Backtest._fold_segment(*inputs, start=1, end=3)
    expected = (
        torch.cat([factors[:, 1:3], torch.zeros_like(factors[:, :2])], dim=1),
        torch.cat(
            [target_ret[:, 1:3], torch.zeros_like(target_ret[:, :2])], dim=1
        ),
        torch.cat(
            [target_valid[:, 1:3], torch.zeros_like(target_valid[:, :2])],
            dim=1,
        ),
        bar_time_ns[:, 1:5],
    )

    assert [value.shape for value in segment] == [torch.Size([2, 4])] * 4
    for actual, wanted in zip(segment, expected):
        torch.testing.assert_close(actual, wanted)

    for value, replacement in zip(segment, (999.0, -999.0, False, -1)):
        value.fill_(replacement)
    for actual, original in zip(inputs, snapshots):
        torch.testing.assert_close(actual, original)


def test_smallest_builder_fold_is_scorable(monkeypatch) -> None:
    folds = build_walk_forward_folds(
        total_bars=8,
        n_blocks=2,
        configured_gap=2,
        min_fold_bars=2,
        warmup_bars=0,
        label_lookahead=2,
    )
    assert len(folds) == 1
    fold = folds[0]
    assert (fold.train_start, fold.train_end, fold.val_start, fold.val_end) == (
        0,
        2,
        4,
        6,
    )

    factors = torch.tensor(
        [[101.0, -102.0, 901.0, 902.0, 201.0, -202.0, 903.0, 904.0]]
    )
    target_ret = torch.tensor(
        [[0.01, 0.02, 9.01, 9.02, 0.03, 0.04, 9.03, 9.04]]
    )
    target_valid = torch.arange(8).unsqueeze(0) < 6
    bar_time_ns = (
        torch.arange(8, dtype=torch.int64).unsqueeze(0)
        * 3_600
        * 1_000_000_000
    )

    backtest = MT5Backtest()
    calls = []
    real_run_execution = backtest._run_execution

    def capture_execution(*args, **kwargs):
        calls.append(tuple(value.detach().clone() for value in args))
        return real_run_execution(*args, **kwargs)

    monkeypatch.setattr(backtest, "_run_execution", capture_execution)
    train_score, val_score = backtest.evaluate_fold(
        factors=factors,
        target_ret=target_ret,
        target_valid=target_valid,
        bar_time_ns=bar_time_ns,
        train_start=fold.train_start,
        train_end=fold.train_end,
        val_start=fold.val_start,
        val_end=fold.val_end,
    )

    assert len(calls) == 2
    expected_valid = torch.tensor([[True, True, False, False]])
    expected = (
        (
            torch.tensor([[101.0, -102.0, 0.0, 0.0]]),
            torch.tensor([[0.01, 0.02, 0.0, 0.0]]),
            expected_valid,
            bar_time_ns[:, 0:4],
        ),
        (
            torch.tensor([[201.0, -202.0, 0.0, 0.0]]),
            torch.tensor([[0.03, 0.04, 0.0, 0.0]]),
            expected_valid,
            bar_time_ns[:, 4:8],
        ),
    )
    excluded_values = factors[:, [2, 3, 6, 7]]
    excluded_targets = target_ret[:, [2, 3, 6, 7]]
    for actual, wanted in zip(calls, expected):
        assert [value.shape for value in actual] == [torch.Size([1, 4])] * 4
        for actual_value, wanted_value in zip(actual, wanted):
            torch.testing.assert_close(actual_value, wanted_value)
        assert not torch.isin(actual[0], excluded_values).any()
        assert not torch.isin(actual[1], excluded_targets).any()

    assert torch.isfinite(train_score)
    assert torch.isfinite(val_score)


def test_timestamp_spacing_changes_annualized_score() -> None:
    factors, target_ret, target_valid, hourly = segment_inputs()
    daily = hourly * 24
    hourly_score = MT5Backtest().evaluate_segment(
        factors, target_ret, target_valid, hourly
    )
    daily_score = MT5Backtest().evaluate_segment(
        factors, target_ret, target_valid, daily
    )
    assert hourly_score != daily_score
