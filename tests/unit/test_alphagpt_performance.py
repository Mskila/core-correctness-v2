from __future__ import annotations

import torch

from benchmarks.perf01 import PERF01_BENCHMARK_SCHEMA_VERSION, run_benchmark
from model_core.alphagpt import AlphaGPT
from model_core.backtest import MT5Backtest
from model_core.execution import run_execution


def test_causal_mask_is_cached_per_sequence_device_and_dtype(monkeypatch) -> None:
    model = AlphaGPT().eval()
    calls = []
    original = torch.nn.Transformer.generate_square_subsequent_mask

    def counted(size, *args, **kwargs):
        calls.append((size, kwargs.get("device"), kwargs.get("dtype")))
        return original(size, *args, **kwargs)

    monkeypatch.setattr(
        torch.nn.Transformer,
        "generate_square_subsequent_mask",
        counted,
    )
    short = torch.zeros((1, 3), dtype=torch.long)
    longer = torch.zeros((1, 4), dtype=torch.long)

    with torch.no_grad():
        first = model(short)
        second = model(short)
        model(longer)

    assert len(calls) == 2
    for left, right in zip(first, second):
        assert torch.equal(left, right)
    assert set(model._causal_mask_cache) == {
        (3, short.device.type, short.device.index, model.token_emb.weight.dtype),
        (4, longer.device.type, longer.device.index, model.token_emb.weight.dtype),
    }


def test_execution_result_internal_borrow_avoids_clone_but_public_access_is_defensive() -> None:
    result = run_execution(
        factors=torch.tensor([[1.0, -1.0, 0.5, 0.0]]),
        target_ret=torch.tensor([[0.01, -0.02, 0.03, 0.0]]),
        target_valid=torch.tensor([[True, True, False, False]]),
        bar_time_ns=torch.tensor([[0, 1, 2, 3]], dtype=torch.int64),
        cost_rate=0.001,
        min_exposure=0.0,
    )

    internal = result._borrow_tensor("net_pnl")
    assert result._borrow_tensor("net_pnl") is internal
    public = result.net_pnl
    assert public is not internal
    public.zero_()
    assert torch.equal(result._borrow_tensor("net_pnl"), internal)
    explicit_copy = result.copy_tensor("net_pnl")
    assert explicit_copy is not internal
    assert torch.equal(explicit_copy, internal)


def test_prepared_fold_is_exactly_equal_to_reference_evaluation() -> None:
    backtest = MT5Backtest(cost_rate=0.001)
    factors = torch.linspace(-1.0, 1.0, 14).unsqueeze(0)
    target = torch.tensor(
        [[0.01, -0.02, 0.015, 0.005, -0.01, 0.02, -0.01,
          0.01, 0.005, -0.005, 0.01, -0.01, 0.0, 0.0]]
    )
    valid = torch.ones_like(target, dtype=torch.bool)
    valid[:, -2:] = False
    timestamps = torch.arange(14, dtype=torch.int64).unsqueeze(0) * 3_600_000_000_000
    boundaries = dict(train_start=0, train_end=5, val_start=7, val_end=12)

    reference = backtest.evaluate_fold(
        factors=factors,
        target_ret=target,
        target_valid=valid,
        bar_time_ns=timestamps,
        **boundaries,
    )
    prepared = backtest.prepare_fold(
        target_ret=target,
        target_valid=valid,
        bar_time_ns=timestamps,
        **boundaries,
    )
    optimized = backtest.evaluate_prepared_fold(factors, prepared)

    assert all(torch.equal(left, right) for left, right in zip(reference, optimized))


def test_perf01_benchmark_reports_exact_parity_and_required_metrics(tmp_path) -> None:
    output = tmp_path / "perf01.json"
    report = run_benchmark(iterations=1, output_path=output)

    assert output.exists()
    assert report["schema_version"] == PERF01_BENCHMARK_SCHEMA_VERSION
    assert report["required_ci"] is False
    assert report["parity"]["torch_equal"] is True
    assert report["parity"]["reference_vm_evaluations_per_formula"] == 2
    assert report["parity"]["optimized_vm_evaluations_per_formula"] == 1
    assert report["throughput"]["optimized_formulas_per_second"] > 0
    assert report["step_ns"]["median"] > 0
    assert report["step_ns"]["p95"] > 0
    assert set(report["stage_ns"]) == {"vm", "fold", "transaction"}
    assert report["memory_peak_bytes"]["retained_factors"] > 0
