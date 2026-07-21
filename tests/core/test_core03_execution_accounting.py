from __future__ import annotations

import math
from datetime import datetime, timezone

import pytest
import torch

from data_pipeline.validation import DataValidationError
from model_core.execution import (
    ExecutionResult,
    build_execution_ledger,
    classify_position_events,
    derive_periods_per_year,
    net_log_return_from_position,
    performance_metrics,
    run_execution,
)
from model_core.semantics import EXECUTION_SEMANTICS_VERSION


@pytest.mark.core
@pytest.mark.parametrize(
    ("position", "asset_ratio", "expected"),
    [
        (0.0, 2.0, 0.0),
        (1.0, 2.0, math.log(2.0)),
        (-1.0, 0.5, math.log(1.5)),
        (0.5, 2.0, math.log(1.5)),
    ],
)
def test_position_closed_forms_use_simple_return_domain(
    position: float,
    asset_ratio: float,
    expected: float,
) -> None:
    result = net_log_return_from_position(
        torch.tensor([position], dtype=torch.float64),
        torch.tensor([math.log(asset_ratio)], dtype=torch.float64),
        torch.tensor([0.0], dtype=torch.float64),
    )
    assert result.asset_simple_return.item() == pytest.approx(asset_ratio - 1.0)
    assert result.net_log_return.item() == pytest.approx(expected)


@pytest.mark.core
def test_cost_is_equity_fraction_and_insolvency_fails_closed() -> None:
    charged = net_log_return_from_position(
        torch.tensor([0.5], dtype=torch.float64),
        torch.tensor([0.0], dtype=torch.float64),
        torch.tensor([0.1], dtype=torch.float64),
    )
    assert charged.portfolio_simple_return.item() == pytest.approx(-0.1)
    assert charged.net_log_return.item() == pytest.approx(math.log(0.9))

    with pytest.raises(DataValidationError, match="insolven"):
        net_log_return_from_position(
            torch.tensor([-1.0], dtype=torch.float64),
            torch.tensor([math.log(2.0)], dtype=torch.float64),
            torch.tensor([0.0], dtype=torch.float64),
        )


@pytest.mark.core
def test_multisymbol_metrics_form_equal_weight_time_portfolio() -> None:
    log_returns = torch.tensor(
        [
            [math.log(1.1), math.log(0.95), math.log(1.01), 0.0, 0.0],
            [math.log(0.9), math.log(1.05), math.log(1.01), 0.0, 0.0],
        ],
        dtype=torch.float64,
    )
    zeros = torch.zeros_like(log_returns)
    valid = torch.tensor(
        [[True, True, True, False, False], [True, True, True, False, False]]
    )
    day_ns = 86_400_000_000_000
    times = torch.tensor(
        [[0, day_ns, 2 * day_ns, 3 * day_ns, 4 * day_ns]] * 2,
        dtype=torch.int64,
    )
    result = ExecutionResult(
        position=zeros,
        turnover=zeros,
        gross_pnl=torch.expm1(log_returns),
        cost=zeros,
        net_pnl=log_returns,
        target_valid=valid,
        bar_time_ns=times,
        final_liquidation_cost=torch.zeros(2, dtype=torch.float64),
    )

    metrics = performance_metrics(result)
    assert metrics.observations == 3
    assert metrics.total_return == pytest.approx(0.01, abs=1e-12)


@pytest.mark.core
def test_position_event_taxonomy_separates_turnover_and_trades() -> None:
    events = classify_position_events(
        torch.tensor([0.0, 0.4, 0.8, 0.3, -0.2, 0.0], dtype=torch.float64)
    )
    assert events.turnover_events == 5
    assert events.entries == 1
    assert events.exits == 1
    assert events.reversals == 1
    assert events.liquidation_events == 0
    assert events.display_trades == 2

    open_at_end = classify_position_events(
        torch.tensor([0.0, 0.4, 0.8], dtype=torch.float64)
    )
    assert open_at_end.exits == 1
    assert open_at_end.liquidation_events == 1
    assert open_at_end.display_trades == 1


@pytest.mark.core
def test_execution_ledger_reconciles_the_shared_net_log_return() -> None:
    result = run_execution(
        factors=torch.atanh(
            torch.tensor([[0.5, -0.25, 0.0, 0.0]], dtype=torch.float64)
        ),
        target_ret=torch.tensor(
            [[math.log(2.0), math.log(0.8), 0.0, 0.0]], dtype=torch.float64
        ),
        target_valid=torch.tensor([[True, True, False, False]]),
        bar_time_ns=torch.tensor([[0, 1, 2, 3]], dtype=torch.int64) * 3_600_000_000_000,
        cost_rate=0.01,
        min_exposure=0.0,
    )
    ledger = build_execution_ledger(result, ["X"])
    assert sum(row.net_pnl for row in ledger) == pytest.approx(
        result.net_pnl[result.target_valid].sum().item()
    )
    for row in ledger:
        assert math.expm1(row.net_pnl) == pytest.approx(row.gross_pnl - row.cost)


@pytest.mark.core
def test_irregular_real_month_lengths_drive_annualization() -> None:
    timestamps = [
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        datetime(2024, 1, 31, tzinfo=timezone.utc),
        datetime(2024, 2, 29, tzinfo=timezone.utc),
        datetime(2024, 3, 31, tzinfo=timezone.utc),
    ]
    times_ns = torch.tensor(
        [[int(value.timestamp() * 1_000_000_000) for value in timestamps]],
        dtype=torch.int64,
    )
    periods = derive_periods_per_year(
        times_ns,
        torch.tensor([[True, True, False, False]]),
    )
    assert periods == pytest.approx(2 * 365.2425 / 60.0)


@pytest.mark.core
def test_core03_bumps_execution_semantics_version() -> None:
    assert EXECUTION_SEMANTICS_VERSION == "tanh-threshold-cost-liquidate-v4"
