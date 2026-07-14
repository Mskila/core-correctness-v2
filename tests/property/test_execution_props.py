import pytest
import torch
from hypothesis import given, settings, strategies as st

from model_core.execution import build_execution_ledger, run_execution


_FINITE_FLOATS = st.floats(
    min_value=-3.0,
    max_value=3.0,
    allow_nan=False,
    allow_infinity=False,
    width=32,
)
_FINITE_RETURNS = st.floats(
    min_value=-0.5,
    max_value=0.5,
    allow_nan=False,
    allow_infinity=False,
    width=32,
)
_COST_RATES = st.floats(
    min_value=0.0,
    max_value=0.05,
    allow_nan=False,
    allow_infinity=False,
    width=64,
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

    base = run_execution(
        factors=factor_tensor,
        target_ret=return_tensor,
        target_valid=valid,
        bar_time_ns=times,
        cost_rate=base_cost_rate,
        min_exposure=0.05,
    )
    higher_cost = run_execution(
        factors=factor_tensor,
        target_ret=return_tensor,
        target_valid=valid,
        bar_time_ns=times,
        cost_rate=base_cost_rate + extra_cost_rate,
        min_exposure=0.05,
    )

    assert bool((base.cost >= 0).all())
    torch.testing.assert_close(
        base.position[:, valid_count:],
        torch.zeros_like(base.position[:, valid_count:]),
    )
    torch.testing.assert_close(
        base.final_liquidation_cost,
        base.position[:, valid_count - 1].abs() * base_cost_rate,
    )

    ledger = build_execution_ledger(base, ["EURUSD"])
    assert ledger[-1].is_final_liquidation is True
    assert sum(row.net_pnl for row in ledger) == pytest.approx(
        base.net_pnl.sum().item(),
        abs=1e-6,
    )
    assert (
        higher_cost.net_pnl.sum().item()
        <= base.net_pnl.sum().item() + 1e-6
    )
