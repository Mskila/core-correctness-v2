from dataclasses import FrozenInstanceError, fields, replace
import math
import pickle

import pytest
import torch

from model_core.execution import (
    ExecutionResult,
    LedgerEntry,
    PerformanceMetrics,
    build_execution_ledger,
    derive_periods_per_year,
    factor_to_position,
    performance_metrics,
    run_execution,
)
from model_core.semantics import DataValidationError


_EXECUTION_TENSOR_FIELDS = (
    "position",
    "turnover",
    "gross_pnl",
    "cost",
    "net_pnl",
    "target_valid",
    "bar_time_ns",
    "final_liquidation_cost",
)


def test_factor_to_position_uses_tanh_and_neutral_band() -> None:
    factors = torch.tensor([[-2.0, -0.01, 0.0, 0.01, 2.0]])

    actual = factor_to_position(factors, min_exposure=0.05)

    expected = torch.tanh(factors)
    expected = torch.where(
        expected.abs() < 0.05,
        torch.zeros_like(expected),
        expected,
    )
    torch.testing.assert_close(actual, expected)


def test_factor_to_position_preserves_shape_and_gradient() -> None:
    factors = torch.tensor(
        [[-1.0, -0.01, 0.4], [0.0, 0.01, 1.0]],
        requires_grad=True,
    )

    positions = factor_to_position(factors, min_exposure=0.05)
    positions.sum().backward()

    assert positions.shape == factors.shape
    assert factors.grad is not None
    assert torch.count_nonzero(factors.grad[positions.abs() >= 0.05]) > 0


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_factor_to_position_rejects_non_finite_input(invalid: float) -> None:
    factors = torch.tensor([[0.0, invalid]])

    with pytest.raises(DataValidationError, match="factors.*finite"):
        factor_to_position(factors, min_exposure=0.05)


def test_factor_to_position_rejects_integer_input() -> None:
    with pytest.raises(DataValidationError, match="floating-point"):
        factor_to_position(
            torch.tensor([[1, 0]], dtype=torch.int64),
            min_exposure=0.05,
        )


def test_factor_to_position_rejects_meta_input() -> None:
    with pytest.raises(DataValidationError, match="CPU or CUDA"):
        factor_to_position(
            torch.zeros((1, 2), device="meta"),
            min_exposure=0.05,
        )


@pytest.mark.parametrize(
    "dtype",
    [torch.float8_e4m3fn, torch.float8_e5m2],
    ids=["float8-e4m3fn", "float8-e5m2"],
)
def test_factor_to_position_rejects_unsupported_float_dtype(
    dtype: torch.dtype,
) -> None:
    with pytest.raises(DataValidationError, match="supported floating-point dtype"):
        factor_to_position(
            torch.zeros((1, 2), dtype=dtype),
            min_exposure=0.0,
        )


@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.bfloat16, torch.float32, torch.float64],
    ids=["float16", "bfloat16", "float32", "float64"],
)
def test_factor_and_execution_support_explicit_float_dtypes(
    dtype: torch.dtype,
) -> None:
    factors = torch.tensor([[0.2, -0.3, 0.0, 0.0]], dtype=dtype)

    positions = factor_to_position(factors, min_exposure=0.0)
    result = run_execution(
        factors=factors,
        target_ret=torch.tensor([[0.1, 0.2, 0.0, 0.0]], dtype=dtype),
        target_valid=torch.tensor([[True, True, False, False]]),
        bar_time_ns=_hourly_times(4),
        cost_rate=0.01,
        min_exposure=0.0,
    )

    assert positions.dtype is dtype
    assert result.position.dtype is dtype
    assert result.cost.dtype is dtype
    assert bool(torch.isfinite(result.net_pnl).all())


@pytest.mark.parametrize("field_name", ["factors", "target_ret"])
@pytest.mark.parametrize(
    "dtype",
    [torch.float8_e4m3fn, torch.float8_e5m2],
    ids=["float8-e4m3fn", "float8-e5m2"],
)
def test_execution_rejects_unsupported_float_dtype_before_backend(
    field_name: str,
    dtype: torch.dtype,
) -> None:
    inputs = {
        "factors": torch.zeros((1, 4), dtype=torch.float32),
        "target_ret": torch.zeros((1, 4), dtype=torch.float32),
    }
    inputs[field_name] = torch.zeros((1, 4), dtype=dtype)

    with pytest.raises(DataValidationError, match="supported floating-point dtype"):
        run_execution(
            **inputs,
            target_valid=torch.tensor([[True, True, False, False]]),
            bar_time_ns=_hourly_times(4),
            cost_rate=0.0,
            min_exposure=0.0,
        )


def _hourly_times(length: int) -> torch.Tensor:
    return (
        torch.arange(length, dtype=torch.int64).unsqueeze(0)
        * 3_600
        * 1_000_000_000
    )


def test_execution_charges_entry_reversal_and_final_liquidation() -> None:
    factors = torch.atanh(torch.tensor([[0.5, -0.5, 0.0, 0.0]]))
    returns = torch.tensor([[0.1, 0.2, 99.0, 99.0]])
    valid = torch.tensor([[True, True, False, False]])

    result = run_execution(
        factors=factors,
        target_ret=returns,
        target_valid=valid,
        bar_time_ns=_hourly_times(4),
        cost_rate=0.1,
        min_exposure=0.05,
    )

    torch.testing.assert_close(
        result.position,
        torch.tensor([[0.5, -0.5, 0.0, 0.0]]),
    )
    torch.testing.assert_close(
        result.turnover,
        torch.tensor([[0.5, 1.0, 0.0, 0.0]]),
    )
    torch.testing.assert_close(
        result.gross_pnl,
        torch.tensor([[0.05, -0.1, 0.0, 0.0]]),
    )
    torch.testing.assert_close(
        result.cost,
        torch.tensor([[0.05, 0.15, 0.0, 0.0]]),
    )
    torch.testing.assert_close(
        result.net_pnl,
        torch.tensor([[0.0, -0.25, 0.0, 0.0]]),
    )
    assert result.final_liquidation_cost.item() == pytest.approx(0.05)


def test_execution_all_flat_positions_have_no_cost() -> None:
    result = run_execution(
        factors=torch.zeros((1, 4)),
        target_ret=torch.ones((1, 4)),
        target_valid=torch.tensor([[True, True, False, False]]),
        bar_time_ns=_hourly_times(4),
        cost_rate=0.1,
        min_exposure=0.05,
    )

    torch.testing.assert_close(result.turnover, torch.zeros((1, 4)))
    torch.testing.assert_close(result.cost, torch.zeros((1, 4)))
    torch.testing.assert_close(result.net_pnl, torch.zeros((1, 4)))
    torch.testing.assert_close(result.final_liquidation_cost, torch.zeros(1))


def test_execution_charges_position_delta_for_same_direction_add() -> None:
    factors = torch.atanh(torch.tensor([[0.25, 0.75, 0.0, 0.0]]))

    result = run_execution(
        factors=factors,
        target_ret=torch.zeros((1, 4)),
        target_valid=torch.tensor([[True, True, False, False]]),
        bar_time_ns=_hourly_times(4),
        cost_rate=0.1,
        min_exposure=0.05,
    )

    torch.testing.assert_close(
        result.turnover,
        torch.tensor([[0.25, 0.5, 0.0, 0.0]]),
    )
    torch.testing.assert_close(
        result.cost,
        torch.tensor([[0.025, 0.125, 0.0, 0.0]]),
    )
    assert result.final_liquidation_cost.item() == pytest.approx(0.075)


def test_execution_zeroes_every_output_at_invalid_positions() -> None:
    factors = torch.atanh(torch.tensor([[0.5, -0.5, 0.75, -0.75]]))
    valid = torch.tensor([[True, True, False, False]])

    result = run_execution(
        factors=factors,
        target_ret=torch.tensor([[0.1, 0.2, 99.0, 99.0]]),
        target_valid=valid,
        bar_time_ns=_hourly_times(4),
        cost_rate=0.1,
        min_exposure=0.05,
    )

    for value in (
        result.position,
        result.turnover,
        result.gross_pnl,
        result.cost,
        result.net_pnl,
    ):
        torch.testing.assert_close(value[~valid], torch.zeros(2))


def test_execution_rejects_no_valid_labels() -> None:
    with pytest.raises(DataValidationError, match="valid label"):
        run_execution(
            factors=torch.zeros((1, 4)),
            target_ret=torch.zeros((1, 4)),
            target_valid=torch.zeros((1, 4), dtype=torch.bool),
            bar_time_ns=_hourly_times(4),
            cost_rate=0.1,
            min_exposure=0.05,
        )


def test_execution_rejects_empty_symbol_batch() -> None:
    with pytest.raises(DataValidationError, match="at least one symbol"):
        run_execution(
            factors=torch.empty((0, 4)),
            target_ret=torch.empty((0, 4)),
            target_valid=torch.empty((0, 4), dtype=torch.bool),
            bar_time_ns=torch.empty((0, 4), dtype=torch.int64),
            cost_rate=0.0,
            min_exposure=0.05,
        )


def test_execution_rejects_non_int64_bar_time() -> None:
    with pytest.raises(DataValidationError, match="bar_time_ns.*int64"):
        run_execution(
            factors=torch.zeros((1, 4)),
            target_ret=torch.zeros((1, 4)),
            target_valid=torch.tensor([[True, True, False, False]]),
            bar_time_ns=torch.tensor(
                [[0.75, 10.75, 20.75, 30.75]],
                dtype=torch.float64,
            ),
            cost_rate=0.0,
            min_exposure=0.05,
        )


def test_execution_rejects_mixed_factor_and_return_dtypes() -> None:
    with pytest.raises(DataValidationError, match="factors and target_ret.*same dtype"):
        run_execution(
            factors=torch.zeros((1, 4), dtype=torch.float32),
            target_ret=torch.zeros((1, 4), dtype=torch.float64),
            target_valid=torch.tensor([[True, True, False, False]]),
            bar_time_ns=_hourly_times(4),
            cost_rate=0.0,
            min_exposure=0.0,
        )


def test_execution_rejects_mismatched_tensor_devices() -> None:
    with pytest.raises(DataValidationError, match="same device"):
        run_execution(
            factors=torch.zeros((1, 4)),
            target_ret=torch.zeros((1, 4), device="meta"),
            target_valid=torch.tensor([[True, True, False, False]]),
            bar_time_ns=_hourly_times(4),
            cost_rate=0.0,
            min_exposure=0.05,
        )


def test_execution_rejects_non_prefix_valid_mask() -> None:
    with pytest.raises(DataValidationError, match="continuous prefix"):
        run_execution(
            factors=torch.zeros((1, 4)),
            target_ret=torch.zeros((1, 4)),
            target_valid=torch.tensor([[True, False, True, False]]),
            bar_time_ns=_hourly_times(4),
            cost_rate=0.1,
            min_exposure=0.05,
        )


def test_execution_rejects_unrepresentable_finite_float16_result() -> None:
    with pytest.raises(DataValidationError, match="execution result.*finite"):
        run_execution(
            factors=torch.tensor([[10.0, 0.0, 0.0]], dtype=torch.float16),
            target_ret=torch.zeros((1, 3), dtype=torch.float16),
            target_valid=torch.tensor([[True, False, False]]),
            bar_time_ns=_hourly_times(3),
            cost_rate=40_000.0,
            min_exposure=0.0,
        )


def test_execution_rejects_nonzero_float16_cost_underflow() -> None:
    with pytest.raises(DataValidationError, match="representable.*cost"):
        run_execution(
            factors=torch.tensor([[10.0, 0.0, 0.0]], dtype=torch.float16),
            target_ret=torch.zeros((1, 3), dtype=torch.float16),
            target_valid=torch.tensor([[True, False, False]]),
            bar_time_ns=_hourly_times(3),
            cost_rate=1e-8,
            min_exposure=0.0,
        )


def test_execution_rejects_float64_working_cost_product_underflow() -> None:
    with pytest.raises(DataValidationError, match="cost component.*underflow"):
        run_execution(
            factors=torch.atanh(
                torch.tensor([[0.5, 0.0, 0.0]], dtype=torch.float64)
            ),
            target_ret=torch.zeros((1, 3), dtype=torch.float64),
            target_valid=torch.tensor([[True, False, False]]),
            bar_time_ns=_hourly_times(3),
            cost_rate=math.ulp(0.0),
            min_exposure=0.0,
        )


@pytest.mark.parametrize(
    "factors",
    [
        torch.tensor(
            [[math.atanh(0.5), 1.0, 0.0, 0.0]],
            dtype=torch.float64,
        ),
        torch.tensor(
            [[1.0, math.atanh(-0.5), 0.0, 0.0]],
            dtype=torch.float64,
        ),
    ],
    ids=["entry-turnover", "final-liquidation"],
)
def test_execution_rejects_each_float64_cost_component_underflow(
    factors: torch.Tensor,
) -> None:
    with pytest.raises(DataValidationError, match="cost component.*underflow"):
        run_execution(
            factors=factors,
            target_ret=torch.zeros((1, 4), dtype=torch.float64),
            target_valid=torch.tensor([[True, True, False, False]]),
            bar_time_ns=_hourly_times(4),
            cost_rate=math.ulp(0.0),
            min_exposure=0.0,
        )


@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.bfloat16],
    ids=["float16", "bfloat16"],
)
def test_execution_rejects_low_precision_working_cost_product_underflow(
    dtype: torch.dtype,
) -> None:
    with pytest.raises(DataValidationError, match="cost component.*underflow"):
        run_execution(
            factors=torch.atanh(torch.tensor([[0.5, 0.0, 0.0]], dtype=dtype)),
            target_ret=torch.zeros((1, 3), dtype=dtype),
            target_valid=torch.tensor([[True, False, False]]),
            bar_time_ns=_hourly_times(3),
            cost_rate=1e-46,
            min_exposure=0.0,
        )


def test_execution_accepts_representable_float64_subnormal_cost() -> None:
    result = run_execution(
        factors=torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64),
        target_ret=torch.zeros((1, 3), dtype=torch.float64),
        target_valid=torch.tensor([[True, False, False]]),
        bar_time_ns=_hourly_times(3),
        cost_rate=math.ulp(0.0),
        min_exposure=0.0,
    )

    assert result.cost[0, 0] == 2 * math.ulp(0.0)
    assert result.final_liquidation_cost[0] == math.ulp(0.0)


def test_execution_accepts_float32_cost_rate_below_property_old_floor() -> None:
    result = run_execution(
        factors=torch.atanh(torch.tensor([[0.5, 0.0, 0.0]], dtype=torch.float32)),
        target_ret=torch.zeros((1, 3), dtype=torch.float32),
        target_valid=torch.tensor([[True, False, False]]),
        bar_time_ns=_hourly_times(3),
        cost_rate=1e-8,
        min_exposure=0.0,
    )

    assert result.cost[0, 0] > 0
    assert result.final_liquidation_cost[0] > 0


@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.bfloat16, torch.float32, torch.float64],
    ids=["float16", "bfloat16", "float32", "float64"],
)
def test_execution_preserves_representable_cost_dtype_and_gradient(
    dtype: torch.dtype,
) -> None:
    factors = torch.tensor(
        [[0.5, 0.25, 0.0, 0.0]],
        dtype=dtype,
        requires_grad=True,
    )
    result = run_execution(
        factors=factors,
        target_ret=torch.zeros((1, 4), dtype=dtype),
        target_valid=torch.tensor([[True, True, False, False]]),
        bar_time_ns=_hourly_times(4),
        cost_rate=0.125,
        min_exposure=0.0,
    )

    assert result.cost.dtype is dtype
    assert result.final_liquidation_cost.dtype is dtype
    assert bool((result.cost[result.target_valid] > 0).all())
    result.net_pnl.sum().backward()
    assert factors.grad is not None
    assert bool((factors.grad.abs() > 0).any())


def test_execution_snapshots_target_valid_for_repeatable_ledger() -> None:
    target_valid = torch.tensor([[True, True, False, False]])
    expected_target_valid = target_valid.clone()
    result = run_execution(
        factors=torch.tensor([[0.2, 0.3, 0.0, 0.0]]),
        target_ret=torch.tensor([[0.1, 0.2, 0.0, 0.0]]),
        target_valid=target_valid,
        bar_time_ns=_hourly_times(4),
        cost_rate=0.01,
        min_exposure=0.0,
    )
    ledger_before = build_execution_ledger(result, ["EURUSD"])

    assert result.target_valid.data_ptr() != target_valid.data_ptr()
    target_valid[0, 1] = False

    torch.testing.assert_close(result.target_valid, expected_target_valid)
    ledger_after = build_execution_ledger(result, ["EURUSD"])
    assert ledger_after == ledger_before
    assert sum(row.net_pnl for row in ledger_after) == pytest.approx(
        result.net_pnl[result.target_valid].sum().item(),
        abs=1e-8,
    )


def test_execution_snapshots_bar_time_for_repeatable_ledger() -> None:
    bar_time_ns = _hourly_times(4)
    expected_bar_time_ns = bar_time_ns.clone()
    result = run_execution(
        factors=torch.tensor([[0.2, 0.3, 0.0, 0.0]]),
        target_ret=torch.tensor([[0.1, 0.2, 0.0, 0.0]]),
        target_valid=torch.tensor([[True, True, False, False]]),
        bar_time_ns=bar_time_ns,
        cost_rate=0.01,
        min_exposure=0.0,
    )
    ledger_before = build_execution_ledger(result, ["EURUSD"])

    assert result.bar_time_ns.data_ptr() != bar_time_ns.data_ptr()
    bar_time_ns[0, 1] += 1

    torch.testing.assert_close(result.bar_time_ns, expected_bar_time_ns)
    ledger_after = build_execution_ledger(result, ["EURUSD"])
    assert ledger_after == ledger_before
    assert sum(row.net_pnl for row in ledger_after) == pytest.approx(
        result.net_pnl[result.target_valid].sum().item(),
        abs=1e-8,
    )


def _snapshot_result() -> ExecutionResult:
    return run_execution(
        factors=torch.tensor([[0.2, -0.3, 0.0, 0.0, 0.0]]),
        target_ret=torch.tensor([[0.1, 0.2, 0.0, 0.0, 0.0]]),
        target_valid=torch.tensor([[True, True, False, False, False]]),
        bar_time_ns=_hourly_times(5),
        cost_rate=0.01,
        min_exposure=0.0,
    )


@pytest.mark.parametrize("field_name", _EXECUTION_TENSOR_FIELDS)
def test_execution_result_tensor_access_has_isolated_storage(
    field_name: str,
) -> None:
    result = _snapshot_result()

    first = getattr(result, field_name)
    second = getattr(result, field_name)

    assert first.data_ptr() != second.data_ptr()
    torch.testing.assert_close(first, second)


def _mutate_exposed_tensor(value: torch.Tensor, mutation: str) -> None:
    if mutation == "getitem":
        flat = value.reshape(-1)
        if value.dtype is torch.bool:
            flat[0] = ~flat[0]
        else:
            flat[0] += 1
    elif mutation == "storage_view":
        storage_alias = value.view(torch.uint8).reshape(-1)
        storage_alias[0].bitwise_xor_(1)
    else:
        value.resize_(0)


@pytest.mark.parametrize("mutation", ["getitem", "storage_view", "resize"])
def test_execution_result_public_tensor_mutations_do_not_change_snapshot(
    mutation: str,
) -> None:
    for field_name in _EXECUTION_TENSOR_FIELDS:
        result = _snapshot_result()
        expected = getattr(result, field_name).clone()
        exposed = getattr(result, field_name)

        _mutate_exposed_tensor(exposed, mutation)

        torch.testing.assert_close(getattr(result, field_name), expected)


def test_execution_result_snapshots_constructor_tensor_inputs() -> None:
    float_matrix = torch.zeros((1, 4))
    inputs = {
        "position": float_matrix.clone(),
        "turnover": float_matrix.clone(),
        "gross_pnl": float_matrix.clone(),
        "cost": float_matrix.clone(),
        "net_pnl": float_matrix.clone(),
        "target_valid": torch.tensor([[True, True, False, False]]),
        "bar_time_ns": _hourly_times(4),
        "final_liquidation_cost": torch.zeros(1),
    }
    result = ExecutionResult(**inputs)
    expected = {
        field_name: getattr(result, field_name).clone()
        for field_name in _EXECUTION_TENSOR_FIELDS
    }

    for value in inputs.values():
        _mutate_exposed_tensor(value, "getitem")

    for field_name in _EXECUTION_TENSOR_FIELDS:
        torch.testing.assert_close(getattr(result, field_name), expected[field_name])


def test_execution_result_stays_frozen_replaceable_and_serializable() -> None:
    result = _snapshot_result()

    with pytest.raises(FrozenInstanceError):
        result.position = torch.zeros_like(result.position)  # type: ignore[misc]

    replacement = replace(result, position=torch.ones_like(result.position))
    torch.testing.assert_close(
        replacement.position,
        torch.ones_like(replacement.position),
    )
    restored = pickle.loads(pickle.dumps(result))
    for field_name in _EXECUTION_TENSOR_FIELDS:
        torch.testing.assert_close(
            getattr(restored, field_name),
            getattr(result, field_name),
        )
    assert "ExecutionResult" in repr(result)


def test_execution_result_defensive_access_keeps_consumers_and_backward() -> None:
    result = _snapshot_result()
    expected_ledger = build_execution_ledger(result, ["EURUSD"])
    expected_metrics = performance_metrics(result)

    for field_name in _EXECUTION_TENSOR_FIELDS:
        _mutate_exposed_tensor(getattr(result, field_name), "getitem")

    assert build_execution_ledger(result, ["EURUSD"]) == expected_ledger
    assert performance_metrics(result) == expected_metrics

    factors = torch.tensor(
        [[0.2, 0.4, 0.0, 0.0]],
        requires_grad=True,
    )
    differentiable = run_execution(
        factors=factors,
        target_ret=torch.zeros((1, 4)),
        target_valid=torch.tensor([[True, True, False, False]]),
        bar_time_ns=_hourly_times(4),
        cost_rate=0.1,
        min_exposure=0.0,
    )
    first_access = differentiable.net_pnl
    second_access = differentiable.net_pnl
    assert first_access.data_ptr() != second_access.data_ptr()
    second_access.sum().backward()
    assert factors.grad is not None
    assert bool((factors.grad.abs() > 0).any())


def test_execution_keeps_nonzero_factor_gradient() -> None:
    factors = torch.tensor([[0.2, -0.4, 0.0, 0.0]], requires_grad=True)
    result = run_execution(
        factors=factors,
        target_ret=torch.tensor([[0.1, -0.2, 0.0, 0.0]]),
        target_valid=torch.tensor([[True, True, False, False]]),
        bar_time_ns=_hourly_times(4),
        cost_rate=0.0,
        min_exposure=0.05,
    )

    result.net_pnl.sum().backward()

    assert factors.grad is not None
    assert torch.count_nonzero(factors.grad) > 0


@pytest.mark.parametrize(
    ("interval_seconds", "expected"),
    [
        (3_600, 8_765.82),
        (4 * 3_600, 2_191.455),
        (24 * 3_600, 365.2425),
    ],
)
def test_derive_periods_per_year_uses_bar_timestamps(
    interval_seconds: int,
    expected: float,
) -> None:
    times = (
        torch.arange(4, dtype=torch.int64).unsqueeze(0)
        * interval_seconds
        * 1_000_000_000
    )

    actual = derive_periods_per_year(
        times,
        torch.tensor([[True, True, False, False]]),
    )

    assert actual == pytest.approx(expected, rel=1e-8)


def _multi_symbol_cadence_case(
    *,
    interval_ns: int,
    valid_counts: tuple[int, ...],
    entry_starts_ns: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    time_length = max(valid_counts) + 2
    time_rows: list[list[int]] = []
    valid_rows: list[list[bool]] = []
    for valid_count, entry_start_ns in zip(
        valid_counts,
        entry_starts_ns,
        strict=True,
    ):
        relevant_times = [
            entry_start_ns - interval_ns,
            *[
                entry_start_ns + step * interval_ns
                for step in range(valid_count + 1)
            ],
        ]
        tail_length = time_length - len(relevant_times)
        time_rows.append(
            relevant_times
            + [relevant_times[-1] + interval_ns] * tail_length
        )
        valid_rows.append(
            [True] * valid_count + [False] * (time_length - valid_count)
        )
    return (
        torch.tensor(time_rows, dtype=torch.int64),
        torch.tensor(valid_rows, dtype=torch.bool),
    )


def test_derive_periods_per_year_rejects_h1_h4_mixed_cadence() -> None:
    hour_ns = 3_600 * 1_000_000_000
    times = torch.tensor(
        [
            [0, hour_ns, 2 * hour_ns, 3 * hour_ns],
            [0, 4 * hour_ns, 8 * hour_ns, 12 * hour_ns],
        ],
        dtype=torch.int64,
    )
    valid = torch.tensor(
        [
            [True, True, False, False],
            [True, True, False, False],
        ]
    )

    with pytest.raises(DataValidationError, match="cadence"):
        derive_periods_per_year(times, valid)


@pytest.mark.parametrize(
    "intervals_hours",
    [(1, 4), (1, 24), (4, 24)],
    ids=["h1-h4", "h1-d1", "h4-d1"],
)
def test_derive_periods_per_year_rejects_heterogeneous_cadence_matrix(
    intervals_hours: tuple[int, int],
) -> None:
    hour_ns = 3_600 * 1_000_000_000
    rows = [
        [0, interval * hour_ns, 2 * interval * hour_ns, 3 * interval * hour_ns]
        for interval in intervals_hours
    ]
    valid = torch.tensor(
        [
            [True, True, False, False],
            [True, True, False, False],
        ]
    )

    for order in ([0, 1], [1, 0]):
        with pytest.raises(DataValidationError, match="cadence"):
            derive_periods_per_year(
                torch.tensor([rows[index] for index in order], dtype=torch.int64),
                valid,
            )


@pytest.mark.parametrize(
    ("interval_ns", "expected_periods_per_year"),
    [
        (3_600 * 1_000_000_000, 8_765.82),
        (4 * 3_600 * 1_000_000_000, 2_191.455),
        (24 * 3_600 * 1_000_000_000, 365.2425),
    ],
    ids=["h1", "h4", "d1"],
)
def test_derive_periods_per_year_accepts_equal_multi_symbol_cadence(
    interval_ns: int,
    expected_periods_per_year: float,
) -> None:
    times, valid = _multi_symbol_cadence_case(
        interval_ns=interval_ns,
        valid_counts=(2, 4),
        entry_starts_ns=(17 * interval_ns, 101 * interval_ns + 123),
    )
    order = torch.tensor([1, 0])

    actual = derive_periods_per_year(times, valid)
    permuted = derive_periods_per_year(times[order], valid[order])

    assert actual == pytest.approx(expected_periods_per_year, rel=1e-12)
    assert permuted == pytest.approx(actual, rel=0.0, abs=0.0)


def test_derive_periods_per_year_compares_large_int64_cadence_exactly() -> None:
    interval_ns = 1_000_000_007
    times, valid = _multi_symbol_cadence_case(
        interval_ns=interval_ns,
        valid_counts=(2, 4),
        entry_starts_ns=(
            8_000_000_000_000_000_000,
            -8_000_000_000_000_000_000,
        ),
    )

    actual = derive_periods_per_year(times, valid)

    expected = 365.2425 * 24 * 3_600 * 1_000_000_000 / interval_ns
    assert actual == pytest.approx(expected, rel=1e-15)


def test_derive_rejects_cadences_even_when_aggregate_matches_h1() -> None:
    hour_ns = 3_600 * 1_000_000_000
    spans = (hour_ns, 2 * hour_ns, 3 * hour_ns)
    times = torch.tensor(
        [
            [-1, 0, span // 2, span]
            for span in spans
        ],
        dtype=torch.int64,
    )
    valid = torch.tensor(
        [[True, True, False, False]] * 3,
        dtype=torch.bool,
    )

    with pytest.raises(DataValidationError, match="cadence"):
        derive_periods_per_year(times, valid)


def test_mixed_cadence_is_a_metrics_boundary_not_a_ledger_boundary() -> None:
    hour_ns = 3_600 * 1_000_000_000
    times = torch.tensor(
        [
            [0, hour_ns, 2 * hour_ns, 3 * hour_ns],
            [0, 4 * hour_ns, 8 * hour_ns, 12 * hour_ns],
        ],
        dtype=torch.int64,
    )
    valid = torch.tensor(
        [
            [True, True, False, False],
            [True, True, False, False],
        ]
    )
    result = run_execution(
        factors=torch.zeros((2, 4)),
        target_ret=torch.zeros((2, 4)),
        target_valid=valid,
        bar_time_ns=times,
        cost_rate=0.0,
        min_exposure=0.0,
    )

    ledger = build_execution_ledger(result, ["H1", "H4"])

    assert len(ledger) == 4
    assert [row.symbol for row in ledger] == ["H1", "H1", "H4", "H4"]
    with pytest.raises(DataValidationError, match="cadence"):
        derive_periods_per_year(result.bar_time_ns, result.target_valid)
    with pytest.raises(DataValidationError, match="cadence"):
        performance_metrics(result)


def test_derive_periods_per_year_rejects_fewer_than_two_observations() -> None:
    with pytest.raises(DataValidationError, match="at least two"):
        derive_periods_per_year(
            _hourly_times(3),
            torch.tensor([[True, False, False]]),
        )


def test_derive_periods_per_year_rejects_missing_final_exit_timestamp() -> None:
    with pytest.raises(DataValidationError, match="final exit timestamp"):
        derive_periods_per_year(
            _hourly_times(3),
            torch.tensor([[True, True, False]]),
        )


def test_derive_periods_per_year_rejects_non_positive_span() -> None:
    times = torch.tensor([[0, 3_600, 7_200, 3_600]], dtype=torch.int64)
    times = times * 1_000_000_000

    with pytest.raises(DataValidationError, match="positive"):
        derive_periods_per_year(
            times,
            torch.tensor([[True, True, False, False]]),
        )


def test_performance_metrics_uses_only_valid_net_log_returns() -> None:
    net_pnl = torch.tensor(
        [[torch.log(torch.tensor(1.10)), torch.log(torch.tensor(0.95)), 99.0, 99.0]]
    )
    valid = torch.tensor([[True, True, False, False]])
    zeros = torch.zeros_like(net_pnl)
    result = ExecutionResult(
        position=zeros,
        turnover=zeros,
        gross_pnl=zeros,
        cost=zeros,
        net_pnl=net_pnl,
        target_valid=valid,
        bar_time_ns=_hourly_times(4),
        final_liquidation_cost=torch.zeros(1),
    )

    metrics = performance_metrics(result)

    assert isinstance(metrics, PerformanceMetrics)
    assert metrics.observations == 2
    assert metrics.periods_per_year == pytest.approx(8_765.82)
    assert metrics.total_return == pytest.approx(0.045, abs=1e-7)
    assert metrics.max_drawdown == pytest.approx(0.05, abs=1e-7)
    assert metrics.win_rate == pytest.approx(0.5)
    assert metrics.elapsed_years == pytest.approx(2 / 8_765.82)
    assert metrics.annualized_return > 0.0


def test_performance_metrics_is_invariant_to_symbol_row_permutation() -> None:
    day_ns = 86_400 * 1_000_000_000
    valid = torch.tensor(
        [
            [True, True, False, False],
            [True, True, False, False],
        ]
    )
    net_pnl = torch.tensor(
        [
            [-0.10, 0.12, 0.0, 0.0],
            [0.10, -0.08, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )
    zeros = torch.zeros_like(net_pnl)
    result = ExecutionResult(
        position=zeros,
        turnover=zeros,
        gross_pnl=net_pnl,
        cost=zeros,
        net_pnl=net_pnl,
        target_valid=valid,
        bar_time_ns=torch.tensor(
            [[0, day_ns, 2 * day_ns, 3 * day_ns]] * 2,
            dtype=torch.int64,
        ),
        final_liquidation_cost=torch.zeros(2, dtype=torch.float64),
    )
    symbols = ["ALPHA", "BETA"]
    order = torch.tensor([1, 0])
    permuted = replace(
        result,
        position=result.position[order],
        turnover=result.turnover[order],
        gross_pnl=result.gross_pnl[order],
        cost=result.cost[order],
        net_pnl=result.net_pnl[order],
        target_valid=result.target_valid[order],
        bar_time_ns=result.bar_time_ns[order],
        final_liquidation_cost=result.final_liquidation_cost[order],
    )
    permuted_symbols = [symbols[index] for index in order.tolist()]

    actual = performance_metrics(permuted)
    expected = performance_metrics(result)
    ledger = build_execution_ledger(permuted, permuted_symbols)

    actual_metrics = {
        field.name: getattr(actual, field.name) for field in fields(PerformanceMetrics)
    }
    expected_metrics = {
        field.name: getattr(expected, field.name) for field in fields(PerformanceMetrics)
    }
    assert actual_metrics == pytest.approx(expected_metrics, rel=1e-12, abs=1e-12)
    assert [row.symbol for row in ledger] == ["BETA", "BETA", "ALPHA", "ALPHA"]


@pytest.mark.parametrize(
    ("valid_net_pnl", "expected_max_drawdown"),
    [
        ([-0.1, 0.0], 1.0 - math.exp(-0.1)),
        ([-0.1, -0.2], 1.0 - math.exp(-0.3)),
    ],
)
def test_performance_metrics_includes_initial_equity_in_drawdown(
    valid_net_pnl: list[float],
    expected_max_drawdown: float,
) -> None:
    net_pnl = torch.tensor([[*valid_net_pnl, 0.0, 0.0]])
    valid = torch.tensor([[True, True, False, False]])
    zeros = torch.zeros_like(net_pnl)
    result = ExecutionResult(
        position=zeros,
        turnover=zeros,
        gross_pnl=zeros,
        cost=zeros,
        net_pnl=net_pnl,
        target_valid=valid,
        bar_time_ns=_hourly_times(4) * 24,
        final_liquidation_cost=torch.zeros(1),
    )

    metrics = performance_metrics(result)

    assert metrics.max_drawdown == pytest.approx(
        expected_max_drawdown,
        abs=1e-8,
    )


def test_performance_metrics_rejects_non_finite_derived_values() -> None:
    result = run_execution(
        factors=torch.atanh(torch.tensor([[0.5, 0.5, 0.0, 0.0]])),
        target_ret=torch.tensor([[1000.0, 1000.0, 0.0, 0.0]]),
        target_valid=torch.tensor([[True, True, False, False]]),
        bar_time_ns=_hourly_times(4),
        cost_rate=0.0,
        min_exposure=0.05,
    )
    assert bool(torch.isfinite(result.net_pnl).all())

    with pytest.raises(DataValidationError, match="finite performance metrics"):
        performance_metrics(result)


def _metric_result(
    net_pnl: torch.Tensor,
    target_valid: torch.Tensor,
) -> ExecutionResult:
    zeros = torch.zeros_like(net_pnl)
    time_length = net_pnl.shape[1]
    return ExecutionResult(
        position=zeros,
        turnover=zeros,
        gross_pnl=net_pnl,
        cost=zeros,
        net_pnl=net_pnl,
        target_valid=target_valid,
        bar_time_ns=_hourly_times(time_length).expand(net_pnl.shape[0], -1).clone(),
        final_liquidation_cost=torch.zeros(
            net_pnl.shape[0],
            dtype=net_pnl.dtype,
        ),
    )


@pytest.mark.parametrize(
    ("net_pnl", "target_valid"),
    [
        (
            torch.tensor([[0.01, 0.01, 0.0, 0.0]], dtype=torch.float64),
            torch.tensor([[True, True, False, False]]),
        ),
        (
            torch.tensor([[-0.01, -0.01, 0.0, 0.0]], dtype=torch.float64),
            torch.tensor([[True, True, False, False]]),
        ),
        (
            torch.tensor(
                [[0.01, 0.0, 0.0], [0.01, 0.0, 0.0]],
                dtype=torch.float64,
            ),
            torch.tensor(
                [[True, False, False], [True, False, False]],
            ),
        ),
    ],
    ids=["constant-positive", "constant-negative", "single-period-per-symbol"],
)
def test_performance_metrics_rejects_nonzero_return_with_zero_risk_denominator(
    net_pnl: torch.Tensor,
    target_valid: torch.Tensor,
) -> None:
    with pytest.raises(DataValidationError, match="non-zero return.*zero risk"):
        performance_metrics(_metric_result(net_pnl, target_valid))


def test_performance_metrics_all_zero_returns_have_zero_risk_ratios() -> None:
    metrics = performance_metrics(
        _metric_result(
            torch.zeros((1, 4), dtype=torch.float64),
            torch.tensor([[True, True, False, False]]),
        )
    )

    assert metrics.total_return == 0.0
    assert metrics.annualized_return == 0.0
    assert metrics.volatility == 0.0
    assert metrics.sharpe == 0.0
    assert metrics.sortino == 0.0
    assert metrics.max_drawdown == 0.0
    assert metrics.calmar == 0.0


def test_performance_metrics_keeps_tiny_nonzero_risk_denominators() -> None:
    metrics = performance_metrics(
        _metric_result(
            torch.tensor(
                [[-1e-12, 2e-12, 0.0, 0.0]],
                dtype=torch.float64,
            ),
            torch.tensor([[True, True, False, False]]),
        )
    )

    assert metrics.volatility > 0.0
    assert metrics.max_drawdown > 0.0
    assert all(
        math.isfinite(value)
        for value in (metrics.sharpe, metrics.sortino, metrics.calmar)
    )


def test_performance_metrics_result_is_frozen() -> None:
    metrics = PerformanceMetrics(
        observations=2,
        elapsed_years=1.0,
        periods_per_year=2.0,
        total_return=0.0,
        annualized_return=0.0,
        volatility=0.0,
        sharpe=0.0,
        sortino=0.0,
        max_drawdown=0.0,
        calmar=0.0,
        win_rate=0.0,
    )

    with pytest.raises(FrozenInstanceError):
        metrics.total_return = 1.0  # type: ignore[misc]


def test_execution_ledger_has_fixed_fields_and_copies_shared_result() -> None:
    factors = torch.atanh(torch.tensor([[0.5, -0.5, 0.0, 0.0]]))
    result = run_execution(
        factors=factors,
        target_ret=torch.tensor([[0.1, 0.2, 99.0, 99.0]]),
        target_valid=torch.tensor([[True, True, False, False]]),
        bar_time_ns=_hourly_times(4),
        cost_rate=0.1,
        min_exposure=0.05,
    )

    ledger = build_execution_ledger(result, ["EURUSD"])

    assert [field.name for field in fields(LedgerEntry)] == [
        "symbol",
        "signal_time_ns",
        "entry_time_ns",
        "exit_time_ns",
        "position",
        "gross_pnl",
        "cost",
        "net_pnl",
        "is_final_liquidation",
    ]
    assert len(ledger) == 2
    assert ledger[0] == LedgerEntry(
        symbol="EURUSD",
        signal_time_ns=0,
        entry_time_ns=3_600_000_000_000,
        exit_time_ns=7_200_000_000_000,
        position=pytest.approx(0.5),
        gross_pnl=pytest.approx(0.05),
        cost=pytest.approx(0.05),
        net_pnl=pytest.approx(0.0),
        is_final_liquidation=False,
    )
    assert ledger[1].signal_time_ns == 3_600_000_000_000
    assert ledger[1].entry_time_ns == 7_200_000_000_000
    assert ledger[1].exit_time_ns == 10_800_000_000_000
    assert ledger[1].position == pytest.approx(-0.5)
    assert ledger[1].gross_pnl == pytest.approx(-0.1)
    assert ledger[1].cost == pytest.approx(0.15)
    assert ledger[1].net_pnl == pytest.approx(-0.25)
    assert ledger[1].is_final_liquidation is True
    assert sum(row.net_pnl for row in ledger) == pytest.approx(
        result.net_pnl.sum().item(),
        abs=1e-8,
    )


def test_execution_ledger_rejects_symbol_count_mismatch() -> None:
    result = run_execution(
        factors=torch.zeros((1, 4)),
        target_ret=torch.zeros((1, 4)),
        target_valid=torch.tensor([[True, True, False, False]]),
        bar_time_ns=_hourly_times(4),
        cost_rate=0.1,
        min_exposure=0.05,
    )

    with pytest.raises(DataValidationError, match="symbols"):
        build_execution_ledger(result, [])


def test_execution_ledger_rejects_non_int64_bar_time() -> None:
    result = run_execution(
        factors=torch.zeros((1, 4)),
        target_ret=torch.zeros((1, 4)),
        target_valid=torch.tensor([[True, True, False, False]]),
        bar_time_ns=_hourly_times(4),
        cost_rate=0.0,
        min_exposure=0.05,
    )
    invalid_result = replace(
        result,
        bar_time_ns=result.bar_time_ns.to(torch.float64) + 0.75,
    )

    with pytest.raises(DataValidationError, match="bar_time_ns.*int64"):
        build_execution_ledger(invalid_result, ["EURUSD"])


def _consumer_result() -> ExecutionResult:
    return run_execution(
        factors=torch.zeros((1, 5)),
        target_ret=torch.zeros((1, 5)),
        target_valid=torch.tensor([[True, True, False, False, False]]),
        bar_time_ns=_hourly_times(5),
        cost_rate=0.0,
        min_exposure=0.05,
    )


@pytest.mark.parametrize("field_name", ["gross_pnl", "cost", "net_pnl"])
def test_execution_ledger_rejects_inconsistent_pnl_fields(
    field_name: str,
) -> None:
    result = _consumer_result()
    inconsistent_value = getattr(result, field_name).clone()
    inconsistent_value[0, 0] += 1.0
    result = replace(result, **{field_name: inconsistent_value})

    with pytest.raises(
        DataValidationError,
        match="net_pnl.*gross_pnl.*cost",
    ):
        build_execution_ledger(result, ["EURUSD"])


def test_execution_ledger_rejects_large_cancellation_residuals() -> None:
    result = replace(
        _consumer_result(),
        gross_pnl=torch.tensor(
            [[1e8, 1e8, 0.0, 0.0, 0.0]],
            dtype=torch.float32,
        ),
        cost=torch.tensor(
            [[1e8, 1e8, 0.0, 0.0, 0.0]],
            dtype=torch.float32,
        ),
        net_pnl=torch.tensor(
            [[10.0, -10.0, 0.0, 0.0, 0.0]],
            dtype=torch.float32,
        ),
    )

    with pytest.raises(
        DataValidationError,
        match="net_pnl.*gross_pnl.*cost",
    ):
        build_execution_ledger(result, ["EURUSD"])


def test_execution_ledger_rejects_mixed_read_field_dtypes() -> None:
    result = replace(
        _consumer_result(),
        position=torch.zeros((1, 5), dtype=torch.float64),
        gross_pnl=torch.tensor(
            [[1.0, 1.0, 0.0, 0.0, 0.0]],
            dtype=torch.float64,
        ),
        cost=torch.zeros((1, 5), dtype=torch.float16),
        net_pnl=torch.tensor(
            [[1.001, 1.0, 0.0, 0.0, 0.0]],
            dtype=torch.float64,
        ),
    )

    with pytest.raises(DataValidationError, match="same dtype"):
        build_execution_ledger(result, ["EURUSD"])


@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.float32, torch.float64],
    ids=["float16", "float32", "float64"],
)
def test_execution_ledger_uses_consistent_four_ulp_boundary(
    dtype: torch.dtype,
) -> None:
    zero = torch.tensor(0.0, dtype=dtype)
    smallest_subnormal = torch.nextafter(
        zero,
        torch.tensor(float("inf"), dtype=dtype),
    )
    reference_values = (zero, smallest_subnormal, torch.tensor(1.0, dtype=dtype))

    for reference in reference_values:
        four_ulp = reference.clone()
        for _ in range(4):
            four_ulp = torch.nextafter(
                four_ulp,
                torch.tensor(float("inf"), dtype=dtype),
            )
        five_ulp = torch.nextafter(
            four_ulp,
            torch.tensor(float("inf"), dtype=dtype),
        )
        gross_pnl = torch.zeros((1, 5), dtype=dtype)
        gross_pnl[0, :2] = reference
        cost = torch.zeros_like(gross_pnl)
        accepted_net = gross_pnl.clone()
        accepted_net[0, :2] = four_ulp
        accepted = replace(
            _consumer_result(),
            position=torch.zeros_like(gross_pnl),
            gross_pnl=gross_pnl,
            cost=cost,
            net_pnl=accepted_net,
        )

        assert len(build_execution_ledger(accepted, ["EURUSD"])) == 2

        rejected_net = accepted_net.clone()
        rejected_net[0, :2] = five_ulp
        rejected = replace(accepted, net_pnl=rejected_net)
        with pytest.raises(
            DataValidationError,
            match="net_pnl.*gross_pnl.*cost",
        ):
            build_execution_ledger(rejected, ["EURUSD"])


def test_execution_ledger_uses_net_ulp_for_cancellation_tolerance() -> None:
    gross_pnl = torch.tensor(
        [[100_000_008.0, 100_000_000.0, 0.0, 0.0, 0.0]],
        dtype=torch.float32,
    )
    cost = torch.tensor(
        [[100_000_000.0, 100_000_008.0, 0.0, 0.0, 0.0]],
        dtype=torch.float32,
    )
    expected_net = gross_pnl - cost
    one_ulp_net = expected_net.clone()
    one_ulp_net[0, :2] = torch.nextafter(
        expected_net[0, :2],
        torch.tensor([float("inf"), -float("inf")]),
    )
    near_result = replace(
        _consumer_result(),
        gross_pnl=gross_pnl,
        cost=cost,
        net_pnl=one_ulp_net,
    )

    assert len(build_execution_ledger(near_result, ["EURUSD"])) == 2

    invalid_net = expected_net.clone()
    invalid_net[0, :2] = torch.tensor([8.5, -8.5])
    invalid_result = replace(near_result, net_pnl=invalid_net)
    with pytest.raises(
        DataValidationError,
        match="net_pnl.*gross_pnl.*cost",
    ):
        build_execution_ledger(invalid_result, ["EURUSD"])


@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.float32, torch.float64],
    ids=["float16", "float32", "float64"],
)
def test_execution_ledger_accepts_legal_execution_pnl_by_dtype(
    dtype: torch.dtype,
) -> None:
    result = run_execution(
        factors=torch.tensor([[0.2, -0.3, 0.0, 0.0]], dtype=dtype),
        target_ret=torch.tensor([[1.0, -0.5, 0.0, 0.0]], dtype=dtype),
        target_valid=torch.tensor([[True, True, False, False]]),
        bar_time_ns=_hourly_times(4),
        cost_rate=0.125,
        min_exposure=0.0,
    )

    ledger = build_execution_ledger(result, ["EURUSD"])

    assert len(ledger) == 2
    assert sum(row.net_pnl for row in ledger) == pytest.approx(
        result.net_pnl[result.target_valid].sum().item(),
        abs=2e-3 if dtype is torch.float16 else 1e-8,
    )


@pytest.mark.parametrize("invalid_symbol", [None, 7, ""])
def test_execution_ledger_rejects_invalid_symbol(invalid_symbol: object) -> None:
    with pytest.raises(DataValidationError, match="non-empty strings"):
        build_execution_ledger(_consumer_result(), [invalid_symbol])  # type: ignore[list-item]


@pytest.mark.parametrize(
    "bar_time_ns",
    [
        torch.tensor([[0, 10, 5, 30, 40]], dtype=torch.int64),
        torch.tensor([[0, 10, 10, 30, 40]], dtype=torch.int64),
    ],
    ids=["descending", "equal"],
)
@pytest.mark.parametrize("consumer", ["performance", "ledger"])
def test_execution_consumers_reject_non_increasing_relevant_timestamps(
    bar_time_ns: torch.Tensor,
    consumer: str,
) -> None:
    result = replace(_consumer_result(), bar_time_ns=bar_time_ns)

    with pytest.raises(DataValidationError, match="strictly increasing"):
        if consumer == "performance":
            performance_metrics(result)
        else:
            build_execution_ledger(result, ["EURUSD"])


def test_execution_consumers_validate_only_timestamps_they_read() -> None:
    result = replace(
        _consumer_result(),
        bar_time_ns=torch.tensor([[15, 10, 20, 30, 40]], dtype=torch.int64),
    )

    assert performance_metrics(result).observations == 2
    with pytest.raises(DataValidationError, match="strictly increasing"):
        build_execution_ledger(result, ["EURUSD"])


@pytest.mark.parametrize("consumer", ["performance", "ledger"])
def test_execution_consumers_ignore_non_increasing_unread_timestamp_tail(
    consumer: str,
) -> None:
    result = replace(
        _consumer_result(),
        bar_time_ns=torch.tensor([[0, 10, 20, 30, 5]], dtype=torch.int64),
    )

    if consumer == "performance":
        assert performance_metrics(result).observations == 2
    else:
        ledger = build_execution_ledger(result, ["EURUSD"])
        assert [(row.entry_time_ns, row.exit_time_ns) for row in ledger] == [
            (10, 20),
            (20, 30),
        ]


def test_execution_ledger_rejects_non_prefix_result_mask() -> None:
    result = replace(
        _consumer_result(),
        target_valid=torch.tensor([[True, False, True, False, False]]),
        net_pnl=torch.tensor([[1.0, 100.0, 2.0, 0.0, 0.0]]),
    )

    with pytest.raises(DataValidationError, match="continuous prefix"):
        build_execution_ledger(result, ["EURUSD"])


def test_performance_metrics_rejects_result_shape_mismatch_before_indexing() -> None:
    result = replace(
        _consumer_result(),
        net_pnl=torch.tensor([[1.0, 1.0]]),
    )

    with pytest.raises(DataValidationError, match="net_pnl.*shape"):
        performance_metrics(result)


def test_performance_metrics_rejects_non_floating_net_pnl() -> None:
    result = replace(
        _consumer_result(),
        net_pnl=torch.zeros((1, 5), dtype=torch.int64),
    )

    with pytest.raises(DataValidationError, match="net_pnl.*floating-point"):
        performance_metrics(result)


def test_performance_metrics_rejects_result_device_mismatch() -> None:
    result = replace(
        _consumer_result(),
        net_pnl=torch.zeros((1, 5), device="meta"),
    )

    with pytest.raises(DataValidationError, match="same device"):
        performance_metrics(result)


@pytest.mark.parametrize(
    "field_name",
    ["position", "gross_pnl", "cost", "net_pnl"],
)
def test_execution_ledger_rejects_result_field_shape_mismatch(
    field_name: str,
) -> None:
    result = replace(
        _consumer_result(),
        **{field_name: torch.zeros((1, 2))},
    )

    with pytest.raises(DataValidationError, match=rf"{field_name}.*shape"):
        build_execution_ledger(result, ["EURUSD"])


@pytest.mark.parametrize(
    "field_name",
    ["position", "gross_pnl", "cost", "net_pnl"],
)
def test_execution_ledger_rejects_non_floating_result_field(
    field_name: str,
) -> None:
    result = replace(
        _consumer_result(),
        **{field_name: torch.zeros((1, 5), dtype=torch.int64)},
    )

    with pytest.raises(DataValidationError, match=rf"{field_name}.*floating-point"):
        build_execution_ledger(result, ["EURUSD"])


@pytest.mark.parametrize(
    "field_name",
    ["position", "gross_pnl", "cost", "net_pnl"],
)
def test_execution_ledger_rejects_result_field_device_mismatch(
    field_name: str,
) -> None:
    result = replace(
        _consumer_result(),
        **{field_name: torch.zeros((1, 5), device="meta")},
    )

    with pytest.raises(DataValidationError, match=rf"{field_name}.*same device"):
        build_execution_ledger(result, ["EURUSD"])


@pytest.mark.parametrize("field_name", ["position", "gross_pnl", "cost", "net_pnl"])
def test_execution_ledger_rejects_non_finite_read_field(field_name: str) -> None:
    result = _consumer_result()
    invalid_value = getattr(result, field_name).clone()
    invalid_value[0, 0] = float("nan")
    result = replace(result, **{field_name: invalid_value})

    with pytest.raises(DataValidationError, match=rf"valid {field_name}.*finite"):
        build_execution_ledger(result, ["EURUSD"])


@pytest.mark.parametrize("corruption", ["non_finite", "shape", "dtype"])
def test_execution_ledger_ignores_unread_turnover_field(corruption: str) -> None:
    result = _consumer_result()
    if corruption == "non_finite":
        invalid_turnover = torch.full_like(result.turnover, float("nan"))
    elif corruption == "shape":
        invalid_turnover = torch.zeros((1, 2))
    else:
        invalid_turnover = torch.zeros((1, 5), dtype=torch.int64)
    result = replace(result, turnover=invalid_turnover)

    ledger = build_execution_ledger(result, ["EURUSD"])

    assert len(ledger) == 2
    assert sum(row.net_pnl for row in ledger) == pytest.approx(
        result.net_pnl[result.target_valid].sum().item(),
        abs=1e-8,
    )


@pytest.mark.parametrize("corruption", ["non_finite", "shape", "dtype"])
def test_execution_ledger_ignores_unread_final_liquidation_cost(
    corruption: str,
) -> None:
    result = _consumer_result()
    if corruption == "non_finite":
        invalid_liquidation_cost = torch.tensor([float("nan")])
    elif corruption == "shape":
        invalid_liquidation_cost = torch.zeros((2, 3))
    else:
        invalid_liquidation_cost = torch.zeros(1, dtype=torch.int64)
    result = replace(
        result,
        final_liquidation_cost=invalid_liquidation_cost,
    )

    ledger = build_execution_ledger(result, ["EURUSD"])

    assert len(ledger) == 2
    assert sum(row.net_pnl for row in ledger) == pytest.approx(
        result.net_pnl[result.target_valid].sum().item(),
        abs=1e-8,
    )


def test_performance_metrics_rejects_unrepresentable_equity_path() -> None:
    seconds_per_year = 365.2425 * 24 * 3_600
    two_years_ns = round(2 * seconds_per_year * 1_000_000_000)
    result = replace(
        _consumer_result(),
        net_pnl=torch.tensor(
            [[-8e307, -8e307, 0.0, 0.0]],
            dtype=torch.float64,
        ),
        target_valid=torch.tensor([[True, True, False, False]]),
        bar_time_ns=torch.tensor(
            [[0, 0, two_years_ns // 2, two_years_ns]],
            dtype=torch.int64,
        ),
    )

    with pytest.raises(DataValidationError, match="representable equity"):
        performance_metrics(result)


def test_derive_periods_per_year_rejects_int64_wraparound_span() -> None:
    times = torch.tensor(
        [[0, torch.iinfo(torch.int64).max, 0, torch.iinfo(torch.int64).min]],
        dtype=torch.int64,
    )

    with pytest.raises(DataValidationError, match="positive"):
        derive_periods_per_year(
            times,
            torch.tensor([[True, True, False, False]]),
        )


def test_turnover_cost_keeps_factor_gradient_when_returns_are_zero() -> None:
    factors = torch.tensor([[0.2, 0.4, 0.0, 0.0]], requires_grad=True)
    result = run_execution(
        factors=factors,
        target_ret=torch.zeros((1, 4)),
        target_valid=torch.tensor([[True, True, False, False]]),
        bar_time_ns=_hourly_times(4),
        cost_rate=0.1,
        min_exposure=0.0,
    )

    result.net_pnl[0, 0].backward()

    assert factors.grad is not None
    assert factors.grad[0, 0].abs() > 0


def test_final_liquidation_cost_keeps_factor_gradient_when_returns_are_zero() -> None:
    factors = torch.tensor([[0.2, 0.2, 0.0, 0.0]], requires_grad=True)
    result = run_execution(
        factors=factors,
        target_ret=torch.zeros((1, 4)),
        target_valid=torch.tensor([[True, True, False, False]]),
        bar_time_ns=_hourly_times(4),
        cost_rate=0.1,
        min_exposure=0.0,
    )

    result.net_pnl[0, 1].backward()

    assert factors.grad is not None
    assert factors.grad[0, 1].abs() > 0


def test_execution_rejects_all_meta_inputs() -> None:
    with pytest.raises(DataValidationError, match="CPU or CUDA"):
        run_execution(
            factors=torch.zeros((1, 4), device="meta"),
            target_ret=torch.zeros((1, 4), device="meta"),
            target_valid=torch.tensor(
                [[True, True, False, False]],
                device="meta",
            ),
            bar_time_ns=torch.arange(4, dtype=torch.int64, device="meta")
            .unsqueeze(0),
            cost_rate=0.0,
            min_exposure=0.05,
        )


def test_multi_symbol_prefixes_reconcile_ledger_and_metrics() -> None:
    factors = torch.tensor(
        [
            [0.2, -0.3, 0.0, 0.0, 0.0, 0.0],
            [0.1, 0.4, -0.2, 0.0, 0.0, 0.0],
        ]
    )
    target_valid = torch.tensor(
        [
            [True, True, False, False, False, False],
            [True, True, True, False, False, False],
        ]
    )
    result = run_execution(
        factors=factors,
        target_ret=torch.tensor(
            [
                [0.01, -0.02, 0.0, 0.0, 0.0, 0.0],
                [0.03, -0.01, 0.02, 0.0, 0.0, 0.0],
            ]
        ),
        target_valid=target_valid,
        bar_time_ns=_hourly_times(6).expand(2, -1).clone(),
        cost_rate=0.001,
        min_exposure=0.05,
    )

    ledger = build_execution_ledger(result, ["EURUSD", "GBPUSD"])
    metrics = performance_metrics(result)

    assert len(ledger) == 5
    assert sum(row.net_pnl for row in ledger) == pytest.approx(
        result.net_pnl[target_valid].sum().item(),
        abs=1e-8,
    )
    assert metrics.observations == 5
