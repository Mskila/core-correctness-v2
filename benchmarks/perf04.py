"""PERF-04 real-data one-step CPU training benchmark (informational)."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import re
import tempfile
import time

import torch

from model_core.config import ModelConfig
from train_file import train_from_file


_TIMING_PATTERN = re.compile(
    r"性能: 采样=(?P<sampling>[0-9.]+)s "
    r"原始评估=(?P<raw_evaluation>[0-9.]+)s "
    r"串行决策=(?P<serial_decision>[0-9.]+)s "
    r"更新=(?P<update>[0-9.]+)s "
    r"发布=(?P<publication>[0-9.]+)s "
    r"其他=(?P<other>[0-9.]+)s "
    r"计算合计=(?P<compute_total>[0-9.]+)s"
)


def _training_digest(engine) -> str:
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            engine.training_history,
            sort_keys=True,
            separators=(",", ":"),
            default=repr,
        ).encode("utf-8")
    )
    digest.update(repr((engine.best_score, engine.best_formula)).encode("utf-8"))
    digest.update(repr(engine._elite_pool).encode("utf-8"))
    for name, value in sorted(engine.model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def run_benchmark(
    *,
    data_file: Path,
    workers: int,
    torch_threads: int,
    chunk_size: int,
    numeric_time_unit: str = "s",
) -> dict:
    data_file = data_file.resolve(strict=True)
    ModelConfig.TRAIN_STEPS = 1
    ModelConfig.TRAIN_LOG_INTERVAL = 1
    ModelConfig.EVALUATION_WORKERS = workers
    ModelConfig.EVALUATION_TORCH_THREADS = torch_threads
    ModelConfig.EVALUATION_CHUNK_SIZE = chunk_size

    output = io.StringIO()
    original_cwd = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="alphamaster-perf04-") as temporary:
        os.chdir(temporary)
        try:
            started = time.perf_counter()
            with contextlib.redirect_stdout(output):
                engine = train_from_file(
                    str(data_file),
                    from_scratch=True,
                    numeric_time_unit=numeric_time_unit,
                )
            wall_seconds = time.perf_counter() - started
        finally:
            os.chdir(original_cwd)

    match = _TIMING_PATTERN.search(output.getvalue())
    if match is None:
        raise RuntimeError("training stage timing line was not emitted")
    stage_seconds = {
        name: float(value) for name, value in match.groupdict().items()
    }
    return {
        "schema_version": "perf04-v1",
        "required_ci": False,
        "config": {
            "workers": workers,
            "torch_threads": torch_threads,
            "chunk_size": chunk_size,
            "numeric_time_unit": numeric_time_unit,
        },
        "stage_seconds": stage_seconds,
        "wall_seconds": wall_seconds,
        "training_digest": _training_digest(engine),
        "torch_version": torch.__version__,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=1)
    parser.add_argument(
        "--numeric-time-unit", choices=("s", "ms", "us", "ns"), default="s"
    )
    args = parser.parse_args()
    print(json.dumps(run_benchmark(
        data_file=args.data_file,
        workers=args.workers,
        torch_threads=args.torch_threads,
        chunk_size=args.chunk_size,
        numeric_time_unit=args.numeric_time_unit,
    ), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
