"""PERF-02 exact-parity persistent CPU evaluator benchmark."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import time
from pathlib import Path

import torch

from model_core.formula_evaluation import RawEvaluationContext, ReferenceCpuEvaluator
from model_core.parallel_evaluator import ParallelCpuEvaluator
from model_core.walk_forward import WalkForwardFold
from tests.support.core00 import (
    CORE00_FIXTURE_SHA256,
    load_formula_corpus,
    load_ohlcv_fixture,
    resolve_formula_tokens,
)


PERF02_BENCHMARK_SCHEMA_VERSION = "perf02-benchmark-v1"


def _context() -> tuple[RawEvaluationContext, list[list[int]]]:
    fixture = load_ohlcv_fixture()
    features = fixture.feature_tensor()
    target = torch.tensor(
        [[0.01, -0.02, 0.015, -0.01, 0.005] * 8],
        dtype=features.dtype,
    )
    valid = torch.ones_like(target, dtype=torch.bool)
    valid[:, -1] = False
    formulas = [
        resolve_formula_tokens(row["tokens"])
        for row in load_formula_corpus()
        if row["classification"] == "valid"
    ] * 8
    return RawEvaluationContext(
        features=features,
        target_ret=target,
        target_valid=valid,
        bar_time_ns=fixture.tensors["time_ns"],
        folds=(WalkForwardFold(0, 0, 16, 18, 34, 2),),
        selection_index=torch.arange(0, 34, dtype=torch.long),
        cost_rate=0.001,
        timeframe="H1",
        target_trades_per_day=2.0,
        oos_gate_scale=0.5,
    ), formulas


def _digest(results) -> str:
    digest = hashlib.sha256()
    for result in results:
        digest.update(str(result.formula_index).encode("ascii"))
        digest.update(json.dumps(result.formula).encode("ascii"))
        digest.update(str(result.constant).encode("ascii"))
        if result.factor is not None:
            digest.update(result.factor.detach().contiguous().numpy().tobytes())
        for score in result.fold_train_scores + result.fold_val_scores:
            digest.update(score.detach().contiguous().numpy().tobytes())
        digest.update(repr(result.fold_ics).encode("ascii"))
        digest.update(repr(result.selection_ic).encode("ascii"))
        digest.update(repr(result.selection_ic_stability).encode("ascii"))
        digest.update(repr(result.exposure).encode("ascii"))
        digest.update(repr(result.error).encode("utf-8"))
    return digest.hexdigest()


def _p95(values: list[int]) -> int:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * 0.95))]


def run_benchmark(
    *,
    worker_counts: tuple[int, ...] = (1, 2, 4, 8),
    iterations: int = 3,
    torch_threads: int = 1,
    chunk_size: int = 1,
    output_path: Path | None = None,
) -> dict:
    if type(iterations) is not int or iterations < 1:
        raise ValueError("iterations must be a positive built-in int")
    if (
        type(worker_counts) is not tuple
        or not worker_counts
        or any(type(value) is not int or value < 1 for value in worker_counts)
        or len(set(worker_counts)) != len(worker_counts)
    ):
        raise ValueError("worker_counts must be unique positive built-in ints")
    if type(torch_threads) is not int or torch_threads < 1:
        raise ValueError("torch_threads must be a positive built-in int")
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("chunk_size must be a positive built-in int")
    context, formulas = _context()
    reference = ReferenceCpuEvaluator(context)
    expected = reference.evaluate_batch(formulas, step=0)
    reference.close()
    expected_digest = _digest(expected)
    variants = {}

    for workers in worker_counts:
        started = time.perf_counter_ns()
        evaluator = (
            ReferenceCpuEvaluator(context)
            if workers == 1
            else ParallelCpuEvaluator(
                context,
                workers=workers,
                timeout_seconds=120.0,
                torch_threads=torch_threads,
                chunk_size=chunk_size,
            )
        )
        executor_construction_ns = time.perf_counter_ns() - started
        samples = []
        try:
            started = time.perf_counter_ns()
            warmup = evaluator.evaluate_batch(formulas, step=-1)
            cold_start_step_ns = time.perf_counter_ns() - started
            if _digest(warmup) != expected_digest:
                raise AssertionError(
                    f"workers={workers} diverged during cold-start evaluation"
                )
            for iteration in range(iterations):
                started = time.perf_counter_ns()
                actual = evaluator.evaluate_batch(formulas, step=iteration)
                samples.append(time.perf_counter_ns() - started)
                if _digest(actual) != expected_digest:
                    raise AssertionError(
                        f"workers={workers} diverged from reference raw evaluation"
                    )
        finally:
            evaluator.close()
        median_ns = int(statistics.median(samples))
        variants[str(workers)] = {
            "exact_parity": True,
            "executor_construction_ns": executor_construction_ns,
            "cold_start_step_ns": cold_start_step_ns,
            "step_ns": {"median": median_ns, "p95": _p95(samples)},
            "formulas_per_second": len(formulas) * 1e9 / median_ns,
        }

    recommended = min(
        worker_counts,
        key=lambda workers: variants[str(workers)]["step_ns"]["median"],
    )
    report = {
        "schema_version": PERF02_BENCHMARK_SCHEMA_VERSION,
        "required_ci": False,
        "metadata": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "os": platform.platform(),
            "cpu_threads": torch.get_num_threads(),
            "dataset_fingerprint": CORE00_FIXTURE_SHA256,
            "formula_count": len(formulas),
            "iterations": iterations,
            "worker_torch_threads": torch_threads,
            "chunk_size": chunk_size,
        },
        "parity": {"exact": True, "raw_digest": expected_digest},
        "variants": variants,
        "recommended_workers": recommended,
        "recommendation_candidates": list(worker_counts),
    }
    if output_path is not None:
        output_path.write_text(
            json.dumps(report, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_benchmark(
        worker_counts=tuple(args.workers),
        iterations=args.iterations,
        torch_threads=args.torch_threads,
        chunk_size=args.chunk_size,
        output_path=args.output,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
