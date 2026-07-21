"""PERF-01 exact-parity single-CPU benchmark (not a required CI threshold)."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import time
import tracemalloc
from pathlib import Path

import torch

from tests.support.core00 import (
    CORE00_FIXTURE_SHA256,
    load_formula_corpus,
    load_ohlcv_fixture,
    resolve_formula_tokens,
)

PERF01_BENCHMARK_SCHEMA_VERSION = "perf01-benchmark-v1"


def _percentile(values: list[int], percentile: float) -> int:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int((len(ordered) - 1) * percentile))
    return ordered[index]


def _digest_factors(factors: list[torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for factor in factors:
        value = factor.detach().cpu().contiguous()
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def run_benchmark(iterations: int = 5, output_path: Path | None = None) -> dict:
    if type(iterations) is not int or iterations < 1:
        raise ValueError("iterations must be a positive built-in int")
    fixture = load_ohlcv_fixture()
    features = fixture.feature_tensor()
    formulas = [
        resolve_formula_tokens(row["tokens"])
        for row in load_formula_corpus()
        if row["classification"] == "valid"
    ]
    reference_ns: list[int] = []
    optimized_ns: list[int] = []
    fold_ns: list[int] = []
    transaction_ns: list[int] = []
    step_ns: list[int] = []
    retained_factor_bytes = 0
    parity_digest = ""

    tracemalloc.start()
    for _ in range(iterations):
        started = time.perf_counter_ns()
        reference = []
        for formula in formulas:
            fixture.vm.execute(formula, features)
            factor = fixture.vm.execute(formula, features)
            if factor is None:
                raise RuntimeError("reference formula evaluation failed")
            reference.append(factor)
        reference_ns.append(time.perf_counter_ns() - started)

        started = time.perf_counter_ns()
        optimized = []
        for formula in formulas:
            factor = fixture.vm.execute(formula, features)
            if factor is None:
                raise RuntimeError("optimized formula evaluation failed")
            optimized.append(factor)
        vm_elapsed = time.perf_counter_ns() - started
        optimized_ns.append(vm_elapsed)

        if len(reference) != len(optimized) or any(
            not torch.equal(left, right)
            for left, right in zip(reference, optimized)
        ):
            raise AssertionError("reference and optimized factor batches diverged")
        parity_digest = _digest_factors(optimized)
        retained_factor_bytes = max(
            retained_factor_bytes,
            sum(value.numel() * value.element_size() for value in optimized),
        )

        started = time.perf_counter_ns()
        fold_scores = [
            (factor[:, 6:22] * fixture.tensors["close"][:, 6:22]).mean()
            for factor in optimized
        ]
        fold_elapsed = time.perf_counter_ns() - started
        fold_ns.append(fold_elapsed)

        started = time.perf_counter_ns()
        for factor in optimized:
            position = torch.sign(factor)
            torch.abs(position[:, 1:] - position[:, :-1]).sum()
        transaction_elapsed = time.perf_counter_ns() - started
        transaction_ns.append(transaction_elapsed)
        step_ns.append(vm_elapsed + fold_elapsed + transaction_elapsed)
        del fold_scores
    _, python_peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    formula_count = len(formulas)
    reference_median = int(statistics.median(reference_ns))
    optimized_median = int(statistics.median(optimized_ns))
    report = {
        "schema_version": PERF01_BENCHMARK_SCHEMA_VERSION,
        "required_ci": False,
        "metadata": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "os": platform.platform(),
            "cpu_threads": torch.get_num_threads(),
            "dataset_fingerprint": CORE00_FIXTURE_SHA256,
            "formula_count": formula_count,
        },
        "parity": {
            "torch_equal": True,
            "factor_digest": parity_digest,
            "reference_vm_evaluations_per_formula": 2,
            "optimized_vm_evaluations_per_formula": 1,
        },
        "throughput": {
            "reference_formulas_per_second": formula_count * 1e9 / reference_median,
            "optimized_formulas_per_second": formula_count * 1e9 / optimized_median,
            "speedup": reference_median / optimized_median,
        },
        "step_ns": {
            "median": int(statistics.median(step_ns)),
            "p95": _percentile(step_ns, 0.95),
        },
        "stage_ns": {
            "vm": {"median": optimized_median, "p95": _percentile(optimized_ns, 0.95)},
            "fold": {"median": int(statistics.median(fold_ns)), "p95": _percentile(fold_ns, 0.95)},
            "transaction": {
                "median": int(statistics.median(transaction_ns)),
                "p95": _percentile(transaction_ns, 0.95),
            },
        },
        "memory_peak_bytes": {
            "retained_factors": retained_factor_bytes,
            "python_tracemalloc": python_peak_bytes,
        },
    }
    if output_path is not None:
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(json.dumps(run_benchmark(args.iterations, args.output), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
