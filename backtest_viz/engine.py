"""Visualization adapter over the shared V2 execution model."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from data_pipeline.data_manager import compute_forward_open_returns
from model_core.execution import (
    ExecutionResult,
    LedgerEntry,
    PerformanceMetrics,
    build_execution_ledger,
    performance_metrics,
    run_execution,
)
from model_core.vm import StackVM


@dataclass(frozen=True)
class Trade:
    """Display-only grouping of consecutive shared ledger rows."""

    symbol: str
    direction: int
    entry_bar: int
    entry_time: int
    entry_price: float
    exit_bar: int
    exit_time: int
    exit_price: float
    gross_pnl: float
    cost: float
    net_pnl: float
    cum_pnl: float

    @property
    def pnl(self) -> float:
        return self.net_pnl


@dataclass
class SymbolResult:
    """One symbol's raw display arrays plus authoritative shared results."""

    symbol: str
    times: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    factor: np.ndarray
    execution: ExecutionResult
    metrics: PerformanceMetrics
    ledger: list[LedgerEntry]
    signal: np.ndarray
    position: np.ndarray
    gross_pnl: np.ndarray
    cost: np.ndarray
    net_pnl: np.ndarray
    pnl: np.ndarray
    cum_pnl: np.ndarray
    trades: list[Trade] = field(default_factory=list)
    sortino: float = 0.0
    sharpe: float = 0.0
    total_return: float = 0.0
    n_trades: int = 0
    win_rate: float = 0.0
    max_drawdown: float = 0.0
    avg_hold_bars: float = 0.0
    profit_loss_ratio: float | None = None


def _numpy_1d(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy().copy()


def _group_display_trades(
    ledger: list[LedgerEntry],
    *,
    open_prices: np.ndarray,
) -> list[Trade]:
    """Group ledger rows without deriving any financial value again."""
    if not ledger:
        return []

    groups: list[list[tuple[int, LedgerEntry]]] = []
    current: list[tuple[int, LedgerEntry]] = []
    current_direction = 0
    for bar, row in enumerate(ledger):
        direction = 1 if row.position > 0 else -1 if row.position < 0 else 0
        if direction and current and direction != current_direction:
            groups.append(current)
            current = []
        if direction:
            current_direction = direction
            current.append((bar, row))
        elif current:
            current.append((bar, row))
            groups.append(current)
            current = []
            current_direction = 0
    if current:
        groups.append(current)

    trades: list[Trade] = []
    cumulative = 0.0
    for group in groups:
        first_bar, first = group[0]
        last_bar, last = group[-1]
        gross = sum(row.gross_pnl for _, row in group)
        cost = sum(row.cost for _, row in group)
        net = sum(row.net_pnl for _, row in group)
        cumulative += net
        entry_index = min(first_bar + 1, len(open_prices) - 1)
        exit_index = min(last_bar + 2, len(open_prices) - 1)
        trades.append(
            Trade(
                symbol=first.symbol,
                direction=1 if first.position > 0 else -1,
                entry_bar=first_bar,
                entry_time=first.entry_time_ns,
                entry_price=float(open_prices[entry_index]),
                exit_bar=last_bar,
                exit_time=last.exit_time_ns,
                exit_price=float(open_prices[exit_index]),
                gross_pnl=gross,
                cost=cost,
                net_pnl=net,
                cum_pnl=cumulative,
            )
        )
    return trades


class BacktestEngine:
    """Execute a formula once and adapt the shared result for visualization."""

    def __init__(
        self,
        formula: list[int] | tuple[int, ...],
        cost_rate: float = 0.0001,
        min_exposure: float = 0.05,
    ) -> None:
        self.formula = list(formula)
        self.cost_rate = cost_rate
        self.min_exposure = min_exposure
        self.vm = StackVM()

    def run(
        self,
        raw_dict: dict[str, torch.Tensor],
        feat_tensor: torch.Tensor,
        symbols: list[str],
    ) -> list[SymbolResult]:
        factors = self.vm.execute(self.formula, feat_tensor)
        if factors is None:
            raise RuntimeError(f"StackVM cannot execute formula {self.formula}")
        target_ret, target_valid = compute_forward_open_returns(raw_dict["open"])
        execution = run_execution(
            factors=factors,
            target_ret=target_ret,
            target_valid=target_valid,
            bar_time_ns=raw_dict["time"].to(dtype=torch.int64),
            cost_rate=self.cost_rate,
            min_exposure=self.min_exposure,
        )
        ledger = build_execution_ledger(execution, symbols)

        results: list[SymbolResult] = []
        for index, symbol in enumerate(symbols):
            symbol_execution = ExecutionResult(
                position=execution.position[index : index + 1],
                turnover=execution.turnover[index : index + 1],
                gross_pnl=execution.gross_pnl[index : index + 1],
                cost=execution.cost[index : index + 1],
                net_pnl=execution.net_pnl[index : index + 1],
                target_valid=execution.target_valid[index : index + 1],
                bar_time_ns=execution.bar_time_ns[index : index + 1],
                final_liquidation_cost=execution.final_liquidation_cost[index : index + 1],
            )
            symbol_metrics = performance_metrics(symbol_execution)
            symbol_ledger = [row for row in ledger if row.symbol == symbol]
            position = _numpy_1d(symbol_execution.position[0])
            gross = _numpy_1d(symbol_execution.gross_pnl[0])
            cost = _numpy_1d(symbol_execution.cost[0])
            net = _numpy_1d(symbol_execution.net_pnl[0])
            open_prices = _numpy_1d(raw_dict["open"][index])
            trades = _group_display_trades(symbol_ledger, open_prices=open_prices)
            results.append(
                SymbolResult(
                    symbol=symbol,
                    times=_numpy_1d(raw_dict["time"][index]),
                    open=open_prices,
                    high=_numpy_1d(raw_dict["high"][index]),
                    low=_numpy_1d(raw_dict["low"][index]),
                    close=_numpy_1d(raw_dict["close"][index]),
                    volume=_numpy_1d(raw_dict["volume"][index]),
                    factor=_numpy_1d(factors[index]),
                    execution=symbol_execution,
                    metrics=symbol_metrics,
                    ledger=symbol_ledger,
                    signal=position,
                    position=position,
                    gross_pnl=gross,
                    cost=cost,
                    net_pnl=net,
                    pnl=net,
                    cum_pnl=np.cumsum(net),
                    trades=trades,
                    sortino=symbol_metrics.sortino,
                    sharpe=symbol_metrics.sharpe,
                    total_return=symbol_metrics.total_return,
                    n_trades=len(trades),
                    win_rate=symbol_metrics.win_rate,
                    max_drawdown=symbol_metrics.max_drawdown,
                    avg_hold_bars=(
                        sum(t.exit_bar - t.entry_bar + 1 for t in trades) / len(trades)
                        if trades
                        else 0.0
                    ),
                    profit_loss_ratio=None,
                )
            )
        return results
