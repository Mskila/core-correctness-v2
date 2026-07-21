"""Non-thresholded CORE-00 stage benchmark.

Run manually with::

    python -m benchmarks.core00 --iterations 3 --output core00-benchmark.json
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import time
from typing import Callable

import torch

from tests.support.core00 import (
    CORE00_FIXTURE_SHA256,
    load_formula_corpus,
    load_ohlcv_fixture,
    resolve_formula_tokens,
)


BENCHMARK_SCHEMA_VERSION = "core00-benchmark-v1"
_STAGES = (
    "sampler",
    "vm",
    "fold_scoring",
    "decision",
    "backward",
    "transaction",
    "logging",
    "checkpoint",
)


def _commit_sha(repo_root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _measure(action: Callable[[], object]) -> int:
    started = time.perf_counter_ns()
    action()
    return time.perf_counter_ns() - started


def run_benchmark(
    iterations: int = 3,
    output_path: Path | None = None,
    repo_root: Path | None = None,
) -> dict[str, object]:
    if iterations < 1:
        raise ValueError("iterations must be at least one")
    root = (repo_root or Path(__file__).resolve().parents[1]).resolve()
    fixture = load_ohlcv_fixture()
    features = fixture.feature_tensor()
    entries = [item for item in load_formula_corpus() if item["classification"] == "valid"]
    formula = resolve_formula_tokens(entries[2]["tokens"])
    factor = fixture.vm.execute(formula, features)
    if factor is None:
        raise RuntimeError("CORE-00 benchmark formula failed to execute")
    close = fixture.tensors["close"]
    logits = torch.zeros(len(entries), dtype=torch.float64, requires_grad=True)
    timings: dict[str, list[int]] = {stage: [] for stage in _STAGES}

    def sample() -> torch.Tensor:
        generator = torch.Generator().manual_seed(20260721)
        return torch.multinomial(torch.softmax(logits.detach(), dim=0), 2, generator=generator)

    def score() -> torch.Tensor:
        target = torch.zeros_like(close)
        target[:, :-2] = torch.log(close[:, 2:] / close[:, 1:-1])
        return torch.stack(
            [(factor[:, 6:22] * target[:, 6:22]).mean(), (factor[:, 22:38] * target[:, 22:38]).mean()]
        )

    def backward() -> None:
        loss = -(torch.log_softmax(logits, dim=0)[:2] * score().mean()).sum()
        loss.backward()
        logits.grad = None

    def transaction() -> torch.Tensor:
        decision = torch.sign(factor)
        turnover = torch.abs(decision[:, 1:] - decision[:, :-1])
        return (decision[:, :-1] * torch.diff(close) - turnover * 0.0001).sum()

    def logging() -> str:
        return json.dumps({"reward": float(score().mean()), "formula": formula}, sort_keys=True)

    def checkpoint() -> bytes:
        return json.dumps(
            {"logits": logits.detach().tolist(), "rng": torch.get_rng_state().tolist()},
            separators=(",", ":"),
        ).encode("utf-8")

    actions: dict[str, Callable[[], object]] = {
        "sampler": sample,
        "vm": lambda: fixture.vm.execute(formula, features),
        "fold_scoring": score,
        "decision": lambda: torch.sign(factor),
        "backward": backward,
        "transaction": transaction,
        "logging": logging,
        "checkpoint": checkpoint,
    }
    for _ in range(iterations):
        for stage in _STAGES:
            timings[stage].append(_measure(actions[stage]))

    report: dict[str, object] = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "required_ci": False,
        "metadata": {
            "machine": platform.node(),
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "os": platform.platform(),
            "cpu_threads": torch.get_num_threads(),
            "dataset_fingerprint": CORE00_FIXTURE_SHA256,
            "commit_sha": _commit_sha(root),
        },
        "timings_ns": timings,
    }
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + f".{os.getpid()}.tmp")
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(destination)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_benchmark(iterations=args.iterations, output_path=args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
