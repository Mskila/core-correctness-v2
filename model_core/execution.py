from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor

from .semantics import DataValidationError


@dataclass(frozen=True)
class ExecutionResult:
    position: Tensor
    turnover: Tensor
    gross_pnl: Tensor
    cost: Tensor
    net_pnl: Tensor
    target_valid: Tensor
    bar_time_ns: Tensor
    final_liquidation_cost: Tensor


@dataclass(frozen=True)
class PerformanceMetrics:
    observations: int
    elapsed_years: float
    periods_per_year: float
    total_return: float
    annualized_return: float
    volatility: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    win_rate: float


@dataclass(frozen=True)
class LedgerEntry:
    symbol: str
    signal_time_ns: int
    entry_time_ns: int
    exit_time_ns: int
    position: float
    gross_pnl: float
    cost: float
    net_pnl: float
    is_final_liquidation: bool


_SECONDS_PER_YEAR = 365.2425 * 24 * 3_600
_NANOSECONDS_PER_SECOND = 1_000_000_000


def factor_to_position(factors: Tensor, *, min_exposure: float) -> Tensor:
    """Convert finite factors to continuous positions with a neutral band."""
    if not math.isfinite(min_exposure) or min_exposure < 0.0:
        raise DataValidationError("min_exposure must be finite and non-negative")
    if not bool(torch.isfinite(factors).all()):
        raise DataValidationError("factors must contain only finite values")

    raw_position = torch.tanh(factors)
    return torch.where(
        raw_position.abs() < min_exposure,
        torch.zeros_like(raw_position),
        raw_position,
    )


def _validate_execution_inputs(
    *,
    factors: Tensor,
    target_ret: Tensor,
    target_valid: Tensor,
    bar_time_ns: Tensor,
    cost_rate: float,
) -> None:
    if factors.ndim != 2:
        raise DataValidationError("factors must have shape [symbols, time]")
    if target_ret.shape != factors.shape:
        raise DataValidationError("target_ret shape must match factors")
    if target_valid.shape != factors.shape or target_valid.dtype is not torch.bool:
        raise DataValidationError("target_valid must be a boolean mask matching factors")
    if bar_time_ns.shape != factors.shape:
        raise DataValidationError("bar_time_ns shape must match factors")
    if not factors.is_floating_point() or not target_ret.is_floating_point():
        raise DataValidationError("factors and target_ret must be floating-point tensors")
    if not math.isfinite(cost_rate) or cost_rate < 0.0:
        raise DataValidationError("cost_rate must be finite and non-negative")
    if bool((target_valid.sum(dim=1) == 0).any()):
        raise DataValidationError("each symbol must have at least one valid label")

    invalid_seen = (~target_valid).cumsum(dim=1) > 0
    if bool((invalid_seen & target_valid).any()):
        raise DataValidationError(
            "target_valid must be one continuous prefix per symbol"
        )
    if not bool(torch.isfinite(target_ret[target_valid]).all()):
        raise DataValidationError("valid target_ret values must be finite")


def run_execution(
    *,
    factors: Tensor,
    target_ret: Tensor,
    target_valid: Tensor,
    bar_time_ns: Tensor,
    cost_rate: float,
    min_exposure: float,
) -> ExecutionResult:
    """Run the shared differentiable execution and cost model."""
    _validate_execution_inputs(
        factors=factors,
        target_ret=target_ret,
        target_valid=target_valid,
        bar_time_ns=bar_time_ns,
        cost_rate=cost_rate,
    )

    raw_position = factor_to_position(factors, min_exposure=min_exposure)
    position = torch.where(
        target_valid,
        raw_position,
        torch.zeros_like(raw_position),
    )
    previous_position = torch.cat(
        [torch.zeros_like(position[:, :1]), position[:, :-1]],
        dim=1,
    )
    turnover = torch.where(
        target_valid,
        (position - previous_position).abs(),
        torch.zeros_like(position),
    )

    next_valid = torch.cat(
        [target_valid[:, 1:], torch.zeros_like(target_valid[:, :1])],
        dim=1,
    )
    final_valid = target_valid & ~next_valid
    liquidation_cost_by_time = torch.where(
        final_valid,
        position.abs() * cost_rate,
        torch.zeros_like(position),
    )
    final_liquidation_cost = liquidation_cost_by_time.sum(dim=1)

    cost = torch.where(
        target_valid,
        turnover * cost_rate + liquidation_cost_by_time,
        torch.zeros_like(position),
    )
    valid_target_ret = torch.where(
        target_valid,
        target_ret,
        torch.zeros_like(target_ret),
    )
    gross_pnl = torch.where(
        target_valid,
        position * valid_target_ret,
        torch.zeros_like(position),
    )
    net_pnl = torch.where(
        target_valid,
        gross_pnl - cost,
        torch.zeros_like(position),
    )

    return ExecutionResult(
        position=position,
        turnover=turnover,
        gross_pnl=gross_pnl,
        cost=cost,
        net_pnl=net_pnl,
        target_valid=target_valid,
        bar_time_ns=bar_time_ns,
        final_liquidation_cost=final_liquidation_cost,
    )


def _validate_time_mask(bar_time_ns: Tensor, target_valid: Tensor) -> Tensor:
    if bar_time_ns.ndim != 2 or target_valid.shape != bar_time_ns.shape:
        raise DataValidationError(
            "bar_time_ns and target_valid must have matching [symbols, time] shape"
        )
    if bar_time_ns.dtype is not torch.int64:
        raise DataValidationError("bar_time_ns must contain int64 nanosecond timestamps")
    if target_valid.dtype is not torch.bool:
        raise DataValidationError("target_valid must be a boolean mask")

    valid_counts = target_valid.sum(dim=1)
    if bool((valid_counts == 0).any()):
        raise DataValidationError("each symbol must have at least one valid label")
    if int(valid_counts.sum()) < 2:
        raise DataValidationError("at least two valid observations are required")

    invalid_seen = (~target_valid).cumsum(dim=1) > 0
    if bool((invalid_seen & target_valid).any()):
        raise DataValidationError(
            "target_valid must be one continuous prefix per symbol"
        )
    return valid_counts


def derive_periods_per_year(bar_time_ns: Tensor, target_valid: Tensor) -> float:
    """Derive annualization from each symbol's first entry and final exit."""
    valid_counts = _validate_time_mask(bar_time_ns, target_valid)
    time_length = bar_time_ns.shape[1]
    if time_length <= 1:
        raise DataValidationError("first entry timestamp is missing")

    final_exit_indices = valid_counts + 1
    if bool((final_exit_indices >= time_length).any()):
        raise DataValidationError("final exit timestamp is missing")

    first_entry_ns = bar_time_ns[:, 1]
    final_exit_ns = bar_time_ns.gather(1, final_exit_indices[:, None]).squeeze(1)
    elapsed_ns = final_exit_ns - first_entry_ns
    if bool((elapsed_ns <= 0).any()):
        raise DataValidationError(
            "each symbol must have a positive entry-to-exit timestamp span"
        )

    observations = valid_counts.sum().to(torch.float64)
    elapsed_seconds = elapsed_ns.to(torch.float64).sum() / _NANOSECONDS_PER_SECOND
    periods = observations * _SECONDS_PER_YEAR / elapsed_seconds
    return float(periods.cpu())


def performance_metrics(result: ExecutionResult) -> PerformanceMetrics:
    """Compute timestamp-derived metrics from valid net log returns."""
    net_pnl = result.net_pnl[result.target_valid]
    periods_per_year = derive_periods_per_year(
        result.bar_time_ns,
        result.target_valid,
    )
    observations = net_pnl.numel()
    elapsed_years = observations / periods_per_year

    cumulative_log_return = torch.cumsum(net_pnl, dim=0)
    equity = torch.exp(cumulative_log_return)
    running_peak = torch.cummax(equity, dim=0).values
    drawdown = 1.0 - equity / running_peak

    total_return_tensor = equity[-1] - 1.0
    annualized_return_tensor = torch.exp(
        cumulative_log_return[-1] / elapsed_years
    ) - 1.0
    mean_return = net_pnl.mean()
    return_std = net_pnl.std(unbiased=False)
    annualization_scale = math.sqrt(periods_per_year)
    sharpe_tensor = torch.where(
        return_std > 0,
        mean_return / return_std * annualization_scale,
        torch.zeros_like(mean_return),
    )
    downside_deviation = torch.sqrt(torch.mean(torch.minimum(
        net_pnl,
        torch.zeros_like(net_pnl),
    ).square()))
    sortino_tensor = torch.where(
        downside_deviation > 0,
        mean_return / downside_deviation * annualization_scale,
        torch.zeros_like(mean_return),
    )
    max_drawdown_tensor = drawdown.max()
    calmar_tensor = torch.where(
        max_drawdown_tensor > 0,
        annualized_return_tensor / max_drawdown_tensor,
        torch.zeros_like(annualized_return_tensor),
    )

    return PerformanceMetrics(
        observations=observations,
        elapsed_years=elapsed_years,
        periods_per_year=periods_per_year,
        total_return=float(total_return_tensor.detach().cpu()),
        annualized_return=float(annualized_return_tensor.detach().cpu()),
        volatility=float((return_std * annualization_scale).detach().cpu()),
        sharpe=float(sharpe_tensor.detach().cpu()),
        sortino=float(sortino_tensor.detach().cpu()),
        max_drawdown=float(max_drawdown_tensor.detach().cpu()),
        calmar=float(calmar_tensor.detach().cpu()),
        win_rate=float((net_pnl > 0).to(torch.float64).mean().detach().cpu()),
    )


def build_execution_ledger(
    result: ExecutionResult,
    symbols: list[str] | tuple[str, ...],
) -> list[LedgerEntry]:
    """Copy each valid shared execution row into an auditable ledger."""
    symbol_count, time_length = result.target_valid.shape
    if len(symbols) != symbol_count:
        raise DataValidationError(
            f"symbols length must match execution rows: "
            f"expected {symbol_count}, actual {len(symbols)}"
        )

    valid_counts = result.target_valid.sum(dim=1)
    if bool((valid_counts + 1 >= time_length).any()):
        raise DataValidationError("final exit timestamp is missing for ledger")

    ledger: list[LedgerEntry] = []
    for symbol_index, symbol in enumerate(symbols):
        valid_count = int(valid_counts[symbol_index].detach().cpu())
        for time_index in range(valid_count):
            ledger.append(
                LedgerEntry(
                    symbol=symbol,
                    signal_time_ns=int(
                        result.bar_time_ns[symbol_index, time_index].detach().cpu()
                    ),
                    entry_time_ns=int(
                        result.bar_time_ns[symbol_index, time_index + 1]
                        .detach()
                        .cpu()
                    ),
                    exit_time_ns=int(
                        result.bar_time_ns[symbol_index, time_index + 2]
                        .detach()
                        .cpu()
                    ),
                    position=float(
                        result.position[symbol_index, time_index].detach().cpu()
                    ),
                    gross_pnl=float(
                        result.gross_pnl[symbol_index, time_index].detach().cpu()
                    ),
                    cost=float(result.cost[symbol_index, time_index].detach().cpu()),
                    net_pnl=float(
                        result.net_pnl[symbol_index, time_index].detach().cpu()
                    ),
                    is_final_liquidation=time_index == valid_count - 1,
                )
            )
    return ledger
