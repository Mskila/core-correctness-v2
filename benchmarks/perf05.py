"""Two-stage machine-local CPU worker tuner with real-data confirmation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from benchmarks.perf02 import run_benchmark as run_quick_benchmark
from benchmarks.perf04 import run_benchmark as run_real_benchmark

PERF05_BENCHMARK_SCHEMA_VERSION = "cpu-worker-autotune-v1"


def select_real_data_finalists(
    candidates: tuple[int, ...], quick_report: dict[str, Any], current_workers: int
) -> tuple[int, ...]:
    ordered = sorted(candidates)
    best = int(quick_report["recommended_workers"])
    best_index = ordered.index(best)
    priority = [best]
    if current_workers in ordered:
        priority.append(current_workers)
    for offset in (-1, 1, 2, 3):
        index = best_index + offset
        if 0 <= index < len(ordered):
            priority.append(ordered[index])
    ranked = sorted(
        ordered,
        key=lambda workers: quick_report["variants"][str(workers)]["step_ns"][
            "median"
        ],
    )
    priority.extend(ranked)
    finalists: list[int] = []
    for workers in priority:
        if workers not in finalists:
            finalists.append(workers)
        if len(finalists) == min(5, len(ordered)):
            break
    return tuple(sorted(finalists))


def run_benchmark(
    *,
    worker_counts: tuple[int, ...],
    iterations: int,
    current_workers: int,
    data_file: Path | None = None,
    numeric_time_unit: str = "s",
    output_path: Path | None = None,
) -> dict[str, Any]:
    quick = run_quick_benchmark(
        worker_counts=worker_counts,
        iterations=iterations,
        torch_threads=1,
        chunk_size=1,
    )
    mode = "fixture_screen"
    variants: dict[str, dict[str, Any]] = {}
    tested_candidates = worker_counts
    exact = bool(quick["parity"]["exact"])

    if data_file is not None:
        data_file = data_file.resolve(strict=True)
        mode = "real_training_data"
        tested_candidates = select_real_data_finalists(
            worker_counts, quick, current_workers
        )
        expected_digest: str | None = None
        for workers in tested_candidates:
            result = run_real_benchmark(
                data_file=data_file,
                workers=workers,
                torch_threads=1,
                chunk_size=1,
                numeric_time_unit=numeric_time_unit,
            )
            digest = str(result["training_digest"])
            if expected_digest is None:
                expected_digest = digest
            variant_exact = digest == expected_digest
            exact = exact and variant_exact
            variants[str(workers)] = {
                "exact_parity": variant_exact,
                "training_step_seconds": result["stage_seconds"]["compute_total"],
                "raw_evaluation_seconds": result["stage_seconds"]["raw_evaluation"],
                "wall_seconds": result["wall_seconds"],
                "training_digest": digest,
            }
        recommended = min(
            tested_candidates,
            key=lambda workers: (
                variants[str(workers)]["training_step_seconds"], workers
            ),
        )
    else:
        recommended = int(quick["recommended_workers"])
        for workers in tested_candidates:
            row = quick["variants"][str(workers)]
            variants[str(workers)] = {
                "exact_parity": row["exact_parity"],
                "training_step_seconds": row["step_ns"]["median"] / 1e9,
                "raw_evaluation_seconds": None,
                "wall_seconds": row["cold_start_step_ns"] / 1e9,
                "formulas_per_second": row["formulas_per_second"],
            }

    report = {
        "schema_version": PERF05_BENCHMARK_SCHEMA_VERSION,
        "required_ci": False,
        "mode": mode,
        "data_file": None if data_file is None else str(data_file),
        "numeric_time_unit": numeric_time_unit,
        "screen_candidates": list(worker_counts),
        "tested_candidates": list(tested_candidates),
        "quick_recommended_workers": quick["recommended_workers"],
        "recommended_workers": recommended,
        "parity": {"exact": exact},
        "variants": variants,
        "quick_screen": quick,
    }
    if output_path is not None:
        output_path.write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--workers", type=int, nargs="+", required=True)
    parser.add_argument("--current-workers", type=int, required=True)
    parser.add_argument("--data-file", type=Path)
    parser.add_argument(
        "--numeric-time-unit", choices=("s", "ms", "us", "ns"), default="s"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_benchmark(
        worker_counts=tuple(args.workers),
        iterations=args.iterations,
        current_workers=args.current_workers,
        data_file=args.data_file,
        numeric_time_unit=args.numeric_time_unit,
        output_path=args.output,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
