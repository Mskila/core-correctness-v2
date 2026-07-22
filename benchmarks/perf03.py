"""PERF-03 segmented portfolio aggregation benchmark (informational)."""
from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

from model_core.execution import (
    ExecutionResult,
    _equal_weight_portfolio_log_returns,
)


def _result(symbols: int, bars: int) -> ExecutionResult:
    zeros = torch.zeros((symbols, bars), dtype=torch.float32)
    target_valid = torch.ones((symbols, bars), dtype=torch.bool)
    target_valid[:, -2:] = False
    target_valid[:, bars // 3 : bars // 3 + 10] = False
    if symbols > 1:
        target_valid[1::2, bars // 2 : bars // 2 + 10] = False
    bar_time_ns = (
        torch.arange(bars, dtype=torch.int64) * 900_000_000_000
    ).unsqueeze(0).expand(symbols, -1).clone()
    return ExecutionResult(
        position=zeros,
        turnover=zeros,
        gross_pnl=zeros,
        cost=zeros,
        net_pnl=zeros,
        target_valid=target_valid,
        bar_time_ns=bar_time_ns,
        final_liquidation_cost=torch.zeros(symbols),
    )


def _reference(result: ExecutionResult) -> torch.Tensor:
    net_pnl = result._borrow_tensor("net_pnl")
    valid = result._borrow_tensor("target_valid")
    returns: list[torch.Tensor] = []
    for time_index in torch.nonzero(valid.any(dim=0)).flatten().tolist():
        returns.append(
            torch.log1p(
                torch.expm1(net_pnl[valid[:, time_index], time_index].double()).mean()
            )
        )
    return torch.stack(returns)


def run_benchmark(*, bars: int = 50_000, iterations: int = 5) -> dict:
    cases = []
    for symbols in (1, 5):
        result = _result(symbols, bars)
        started = time.perf_counter_ns()
        reference = _reference(result)
        reference_ns = time.perf_counter_ns() - started

        optimized_ns = []
        optimized = None
        for _ in range(iterations):
            started = time.perf_counter_ns()
            optimized, _, _ = _equal_weight_portfolio_log_returns(result)
            optimized_ns.append(time.perf_counter_ns() - started)
        assert optimized is not None
        if not torch.equal(reference, optimized):
            raise AssertionError("vectorized portfolio path diverged from reference")
        median_ns = int(statistics.median(optimized_ns))
        cases.append(
            {
                "symbols": symbols,
                "bars": bars,
                "observations": optimized.numel(),
                "reference_ns": reference_ns,
                "optimized_median_ns": median_ns,
                "speedup": reference_ns / median_ns,
                "torch_equal": True,
            }
        )
    return {"schema_version": "perf03-v1", "required_ci": False, "cases": cases}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars", type=int, default=50_000)
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()
    print(json.dumps(run_benchmark(bars=args.bars, iterations=args.iterations), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
