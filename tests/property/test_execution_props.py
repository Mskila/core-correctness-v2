import pytest
import torch
from hypothesis import find, given, settings, strategies as st

from model_core.execution import (
    ExecutionResult,
    build_execution_ledger,
    run_execution,
)
from model_core.semantics import DataValidationError


_FINITE_FLOATS = st.one_of(
    st.just(0.0),
    st.floats(
        min_value=-3.0,
        max_value=-0.0009765625,
        allow_nan=False,
        allow_infinity=False,
        width=32,
    ),
    st.floats(
        min_value=0.0009765625,
        max_value=3.0,
        allow_nan=False,
        allow_infinity=False,
        width=32,
    ),
)
_FINITE_RETURNS = st.floats(
    min_value=-0.5,
    max_value=0.5,
    allow_nan=False,
    allow_infinity=False,
    width=32,
)
_FLOAT32_COST_MAX = float(torch.tensor(0.05, dtype=torch.float32))
_COST_RATES = st.one_of(
    st.just(0.0),
    st.just(1e-8),
    st.floats(
        min_value=0.0,
        max_value=_FLOAT32_COST_MAX,
        allow_nan=False,
        allow_infinity=False,
        allow_subnormal=True,
        exclude_min=True,
        width=32,
    ),
)


def test_cost_rate_strategy_covers_small_float32_rates() -> None:
    search_settings = settings(max_examples=2_000, database=None, deadline=None)

    assert find(
        _COST_RATES,
        lambda rate: rate == 1e-8,
        settings=search_settings,
    ) == pytest.approx(1e-8, rel=0.0, abs=0.0)
    assert 0.0 < find(
        _COST_RATES,
        lambda rate: 0.0 < rate < 1e-9,
        settings=search_settings,
    ) < 1e-9


def _run_cost_case_or_assert_fail_closed(
    *,
    factors: torch.Tensor,
    returns: torch.Tensor,
    valid: torch.Tensor,
    times: torch.Tensor,
    cost_rate: float,
) -> ExecutionResult | None:
    try:
        result = run_execution(
            factors=factors,
            target_ret=returns,
            target_valid=valid,
            bar_time_ns=times,
            cost_rate=cost_rate,
            min_exposure=0.05,
        )
    except DataValidationError as exc:
        message = str(exc)
        assert cost_rate > 0.0
        assert "cost" in message
        assert "underflow" in message or "representable" in message
        return None

    if cost_rate > 0.0:
        valid_turnover = result.turnover[valid]
        valid_cost = result.cost[valid]
        nonzero_turnover = valid_turnover > 0
        assert bool((valid_cost[nonzero_turnover] > 0).all())

        last_indices = valid.sum(dim=1) - 1
        last_position = result.position.gather(1, last_indices[:, None]).squeeze(1)
        nonzero_last_position = last_position != 0
        assert bool(
            (result.final_liquidation_cost[nonzero_last_position] > 0).all()
        )
    return result


def _assert_execution_properties(
    result: ExecutionResult,
    *,
    valid_count: int,
    cost_rate: float,
) -> None:
    assert bool((result.cost >= 0).all())
    torch.testing.assert_close(
        result.position[:, valid_count:],
        torch.zeros_like(result.position[:, valid_count:]),
    )
    torch.testing.assert_close(
        result.final_liquidation_cost,
        result.position[:, valid_count - 1].abs() * cost_rate,
    )

    ledger = build_execution_ledger(result, ["EURUSD"])
    assert ledger[-1].is_final_liquidation is True
    assert sum(row.net_pnl for row in ledger) == pytest.approx(
        result.net_pnl.sum().item(),
        abs=1e-6,
    )


@settings(deadline=None, max_examples=75)
@given(
    factors=st.lists(_FINITE_FLOATS, min_size=6, max_size=6),
    returns=st.lists(_FINITE_RETURNS, min_size=6, max_size=6),
    valid_count=st.integers(min_value=1, max_value=4),
    base_cost_rate=_COST_RATES,
    extra_cost_rate=_COST_RATES,
)
def test_execution_cost_liquidation_ledger_and_cost_monotonicity(
    factors: list[float],
    returns: list[float],
    valid_count: int,
    base_cost_rate: float,
    extra_cost_rate: float,
) -> None:
    factor_tensor = torch.tensor([factors], dtype=torch.float32)
    return_tensor = torch.tensor([returns], dtype=torch.float32)
    valid = torch.arange(6).unsqueeze(0) < valid_count
    times = (
        torch.arange(6, dtype=torch.int64).unsqueeze(0)
        * 3_600
        * 1_000_000_000
    )

    base = _run_cost_case_or_assert_fail_closed(
        factors=factor_tensor,
        returns=return_tensor,
        valid=valid,
        times=times,
        cost_rate=base_cost_rate,
    )
    higher_cost = _run_cost_case_or_assert_fail_closed(
        factors=factor_tensor,
        returns=return_tensor,
        valid=valid,
        times=times,
        cost_rate=base_cost_rate + extra_cost_rate,
    )
    if base is not None:
        _assert_execution_properties(
            base,
            valid_count=valid_count,
            cost_rate=base_cost_rate,
        )
    if higher_cost is not None:
        _assert_execution_properties(
            higher_cost,
            valid_count=valid_count,
            cost_rate=base_cost_rate + extra_cost_rate,
        )
    if base is None or higher_cost is None:
        return

    assert (
        higher_cost.net_pnl.sum().item()
        <= base.net_pnl.sum().item() + 1e-6
    )
