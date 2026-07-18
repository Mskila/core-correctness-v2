from __future__ import annotations

import math

import pytest
import torch

import backtest_viz.engine as viz_module
from backtest_viz.engine import BacktestEngine
from data_pipeline.data_manager import compute_forward_open_returns
from model_core.execution import (
    LedgerEntry,
    build_execution_ledger,
    performance_metrics,
    run_execution,
)


def _inputs() -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    opens = torch.tensor([[100.0, 101.0, 99.0, 102.0, 100.0, 103.0, 101.0, 104.0]])
    times = torch.arange(8, dtype=torch.int64)[None, :] * 4 * 3_600 * 1_000_000_000
    raw = {
        "open": opens,
        "high": opens + 1.0,
        "low": opens - 1.0,
        "close": opens + 0.25,
        "volume": torch.arange(8, dtype=torch.float32)[None, :] + 100.0,
        "time": times,
    }
    factors = torch.atanh(torch.tensor([[0.5, -0.3, 0.7, -0.2, 0.4, -0.6, 0.0, 0.0]]))
    features = torch.zeros((1, 1, 8), dtype=torch.float32)
    return raw, factors, features


def test_visual_engine_is_a_single_shared_execution_adapter(monkeypatch) -> None:
    raw, factors, features = _inputs()
    engine = BacktestEngine(formula=[0], cost_rate=0.002, min_exposure=0.05)
    monkeypatch.setattr(engine.vm, "execute", lambda formula, feat: factors)
    calls = 0
    original = viz_module.run_execution

    def counted(**kwargs):
        nonlocal calls
        calls += 1
        return original(**kwargs)

    monkeypatch.setattr(viz_module, "run_execution", counted)
    result = engine.run(raw, features, ["EURUSD"])[0]

    target_ret, target_valid = compute_forward_open_returns(raw["open"])
    shared = original(
        factors=factors,
        target_ret=target_ret,
        target_valid=target_valid,
        bar_time_ns=raw["time"],
        cost_rate=0.002,
        min_exposure=0.05,
    )
    assert calls == 1
    torch.testing.assert_close(result.execution.position, shared.position)
    torch.testing.assert_close(result.execution.gross_pnl, shared.gross_pnl)
    torch.testing.assert_close(result.execution.cost, shared.cost)
    torch.testing.assert_close(result.execution.net_pnl, shared.net_pnl)
    assert result.metrics == performance_metrics(shared)
    assert result.ledger == build_execution_ledger(shared, ["EURUSD"])
    assert result.cum_pnl[-1] == pytest.approx(shared.net_pnl.sum().item())


def test_visual_trades_include_final_cost_and_reconcile(monkeypatch) -> None:
    raw, factors, features = _inputs()
    engine = BacktestEngine(formula=[0], cost_rate=0.01, min_exposure=0.05)
    monkeypatch.setattr(engine.vm, "execute", lambda formula, feat: factors)
    result = engine.run(raw, features, ["EURUSD"])[0]

    assert result.ledger[-1].is_final_liquidation
    assert result.ledger[-1].cost >= result.execution.final_liquidation_cost.item()
    assert sum(row.net_pnl for row in result.ledger) == pytest.approx(
        result.execution.net_pnl.sum().item(), abs=1e-8
    )
    assert sum(trade.net_pnl for trade in result.trades) == pytest.approx(
        result.execution.net_pnl.sum().item(), abs=1e-8
    )


def test_h4_metrics_are_derived_from_raw_nanosecond_timestamps(monkeypatch) -> None:
    raw, factors, features = _inputs()
    engine = BacktestEngine(formula=[0], cost_rate=0.001, min_exposure=0.05)
    monkeypatch.setattr(engine.vm, "execute", lambda formula, feat: factors)
    result = engine.run(raw, features, ["EURUSD"])[0]

    assert result.metrics.periods_per_year == pytest.approx(365.2425 * 6, rel=1e-6)
    assert not math.isclose(result.metrics.periods_per_year, 6240.0)


def test_public_metric_aliases_never_recompute_from_grouped_display_trades(
    monkeypatch,
) -> None:
    opens = torch.tensor([[99.0, 100.0, 110.0, 121.0, 115.0]])
    times = torch.arange(5, dtype=torch.int64)[None, :] * 4 * 3_600 * 1_000_000_000
    raw = {
        "open": opens,
        "high": opens + 1.0,
        "low": opens - 1.0,
        "close": opens + 0.25,
        "volume": torch.arange(5, dtype=torch.float32)[None, :] + 100.0,
        "time": times,
    }
    factors = torch.atanh(torch.tensor([[0.5, 0.5, 0.5, 0.0, 0.0]]))
    features = torch.zeros((1, 1, 5), dtype=torch.float32)
    engine = BacktestEngine(formula=[0], cost_rate=0.0, min_exposure=0.05)
    monkeypatch.setattr(engine.vm, "execute", lambda formula, feat: factors)

    result = engine.run(raw, features, ["EURUSD"])[0]

    assert len(result.trades) == 1
    assert result.trades[0].net_pnl > 0
    assert result.metrics.win_rate == pytest.approx(2 / 3)
    assert result.win_rate == result.metrics.win_rate
    assert result.profit_loss_ratio is None


def _ledger_row(bar: int, position: float, gross: float, cost: float) -> LedgerEntry:
    return LedgerEntry(
        symbol="EURUSD",
        signal_time_ns=bar,
        entry_time_ns=bar + 1,
        exit_time_ns=bar + 2,
        position=position,
        gross_pnl=gross,
        cost=cost,
        net_pnl=gross - cost,
        is_final_liquidation=False,
    )


def test_display_trades_close_on_first_flat_row_and_keep_exit_cost() -> None:
    ledger = [
        _ledger_row(0, 0.5, 1.0, 0.10),
        _ledger_row(1, 0.0, 0.0, 0.25),
        _ledger_row(2, 0.0, 0.0, 0.00),
        _ledger_row(3, 0.5, 2.0, 0.20),
        _ledger_row(4, 0.0, 0.0, 0.35),
    ]

    trades = viz_module._group_display_trades(
        ledger, open_prices=torch.arange(10, dtype=torch.float64).numpy()
    )

    assert [(trade.entry_bar, trade.exit_bar) for trade in trades] == [(0, 1), (3, 4)]
    assert [trade.cost for trade in trades] == pytest.approx([0.35, 0.55])
    assert [trade.net_pnl for trade in trades] == pytest.approx([0.65, 1.45])


def test_display_trades_split_reversals_and_ignore_consecutive_flats() -> None:
    ledger = [
        _ledger_row(0, 0.5, 0.7, 0.1),
        _ledger_row(1, -0.5, -0.2, 0.3),
        _ledger_row(2, 0.0, 0.0, 0.4),
        _ledger_row(3, 0.0, 0.0, 0.0),
        _ledger_row(4, -0.25, 0.8, 0.2),
        _ledger_row(5, 0.0, 0.0, 0.5),
    ]

    trades = viz_module._group_display_trades(
        ledger, open_prices=torch.arange(10, dtype=torch.float64).numpy()
    )

    assert [(trade.direction, trade.entry_bar, trade.exit_bar) for trade in trades] == [
        (1, 0, 0),
        (-1, 1, 2),
        (-1, 4, 5),
    ]
    assert sum(trade.cost for trade in trades) == pytest.approx(
        sum(row.cost for row in ledger)
    )
    assert sum(trade.net_pnl for trade in trades) == pytest.approx(
        sum(row.net_pnl for row in ledger)
    )
