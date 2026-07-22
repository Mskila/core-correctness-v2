"""Machine-local CPU worker auto-tuning for the training web console."""
from __future__ import annotations

import json
import math
import os
import platform
import subprocess
import sys
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from web.process_lifecycle import popen_group_options
from web.settings import load_settings, save_settings

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
CPU_TUNER_SCHEMA_VERSION = "cpu-worker-autotune-v1"


def _total_memory_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.virtual_memory().total)
    except (ImportError, AttributeError, OSError, ValueError):
        pass
    if os.name == "nt":
        try:
            import ctypes

            class _MemoryStatusEx(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = _MemoryStatusEx()
            status.dwLength = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.ullTotalPhys)
        except (AttributeError, OSError, TypeError, ValueError):
            pass
    return None


def default_worker_candidates(
    *,
    logical_processors: int | None = None,
    total_memory_bytes: int | None = None,
) -> list[int]:
    """Return a bounded set that scales to the current CPU without overcommitting RAM."""
    logical = logical_processors if logical_processors is not None else os.cpu_count()
    logical = max(1, min(64, int(logical or 1)))
    memory = total_memory_bytes if total_memory_bytes is not None else _total_memory_bytes()
    if memory is None:
        memory_cap = logical
    else:
        total_gib = memory / (1024**3)
        # Keep roughly 45% of RAM for Windows, the web process and benchmark parent.
        memory_cap = max(1, int(max(0.0, total_gib * 0.55 - 1.5) / 0.45))
    cap = max(1, min(logical, memory_cap, 64))
    standard = (1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64)
    candidates = [value for value in standard if value <= cap]
    if cap not in candidates:
        candidates.append(cap)
    return sorted(set(candidates))


def hardware_snapshot() -> dict[str, Any]:
    memory = _total_memory_bytes()
    cpu_name = platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "")
    return {
        "cpu": cpu_name,
        "logical_processors": os.cpu_count() or 1,
        "total_memory_gib": None if memory is None else round(memory / (1024**3), 1),
        "platform": platform.platform(),
    }


@dataclass
class CpuTuningJob:
    candidates: list[int]
    data_file: str | None = None
    numeric_time_unit: str = "s"
    benchmark_mode: str | None = None
    state: str = "running"
    pid: int | None = None
    started_at: str = ""
    finished_at: str | None = None
    exit_code: int | None = None
    log_path: str = ""
    report_path: str = ""
    recommended_workers: int | None = None
    results: list[dict[str, Any]] | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidates": self.candidates,
            "data_file": self.data_file,
            "numeric_time_unit": self.numeric_time_unit,
            "benchmark_mode": self.benchmark_mode,
            "state": self.state,
            "pid": self.pid,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "exit_code": self.exit_code,
            "log_path": self.log_path,
            "report_path": self.report_path,
            "recommended_workers": self.recommended_workers,
            "results": self.results,
            "error": self.error,
        }


class CpuTuningManager:
    def __init__(
        self,
        *,
        project_root: Path = PROJECT_ROOT,
        settings_saver: Callable[[dict], dict] = save_settings,
        settings_loader: Callable[[], dict] = load_settings,
    ) -> None:
        self._project_root = project_root
        self._settings_saver = settings_saver
        self._settings_loader = settings_loader
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._job: CpuTuningJob | None = None
        self._log_fp = None
        self._hardware = hardware_snapshot()

    def configuration(self) -> dict[str, Any]:
        return {
            "hardware": dict(self._hardware),
            "candidates": default_worker_candidates(
                logical_processors=int(self._hardware["logical_processors"]),
                total_memory_bytes=(
                    None
                    if self._hardware["total_memory_gib"] is None
                    else int(float(self._hardware["total_memory_gib"]) * 1024**3)
                ),
            ),
        }

    def status(self) -> dict[str, Any]:
        with self._lock:
            self._refresh_locked()
            active = self._job is not None and self._job.state == "running"
            return {
                "active": active,
                "job": None if self._job is None else self._job.to_dict(),
                "configured_workers": int(
                    self._settings_loader().get("evaluation_workers", 8)
                ),
                **self.configuration(),
            }

    def start(
        self,
        *,
        candidates: list[int] | None = None,
        iterations: int = 3,
        data_file: str | None = None,
        numeric_time_unit: str = "s",
        current_workers: int = 8,
    ) -> CpuTuningJob:
        with self._lock:
            self._refresh_locked()
            if self._proc is not None and self._proc.poll() is None:
                raise RuntimeError("CPU 自动调优正在运行")
            chosen = self._validate_candidates(
                candidates if candidates is not None else self.configuration()["candidates"]
            )
            if type(iterations) is not int or not 1 <= iterations <= 10:
                raise ValueError("iterations must be between 1 and 10")
            if numeric_time_unit not in {"s", "ms", "us", "ns"}:
                raise ValueError("unsupported numeric time unit")
            if type(current_workers) is not int or not 1 <= current_workers <= 64:
                raise ValueError("current_workers must be between 1 and 64")

            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
            log_path = LOG_DIR / f"cpu_tuning_{ts}.log"
            report_path = LOG_DIR / f"cpu_tuning_{ts}.json"
            cmd = [
                sys.executable,
                "-u",
                "-m",
                "benchmarks.perf05",
                "--iterations",
                str(iterations),
                "--workers",
                *[str(value) for value in chosen],
                "--current-workers",
                str(current_workers),
                "--numeric-time-unit",
                numeric_time_unit,
                "--output",
                str(report_path),
            ]
            if data_file:
                cmd.extend(["--data-file", str(Path(data_file).resolve())])
            self._log_fp = open(log_path, "w", encoding="utf-8", buffering=1)
            try:
                self._proc = subprocess.Popen(
                    cmd,
                    cwd=self._project_root,
                    stdout=self._log_fp,
                    stderr=subprocess.STDOUT,
                    env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONUTF8": "1"},
                    **popen_group_options(),
                )
            except Exception:
                self._log_fp.close()
                self._log_fp = None
                raise
            self._job = CpuTuningJob(
                candidates=chosen,
                data_file=data_file,
                numeric_time_unit=numeric_time_unit,
                pid=self._proc.pid,
                started_at=datetime.now(timezone.utc).isoformat(),
                log_path=self._relative(log_path),
                report_path=self._relative(report_path),
            )
            return self._job

    def tail_log(self, lines: int = 80) -> list[str]:
        with self._lock:
            if self._job is None or not self._job.log_path:
                return []
            path = self._project_root / self._job.log_path
            if not path.exists():
                return []
            try:
                return path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]
            except OSError:
                return []

    def _relative(self, path: Path) -> str:
        try:
            return str(path.relative_to(self._project_root)).replace("\\", "/")
        except ValueError:
            return str(path)

    @staticmethod
    def _validate_candidates(values: list[int]) -> list[int]:
        if not isinstance(values, list) or not values:
            raise ValueError("candidates must be a non-empty list")
        if any(type(value) is not int or not 1 <= value <= 64 for value in values):
            raise ValueError("candidate workers must be integers between 1 and 64")
        return sorted(set(values))

    def _refresh_locked(self) -> None:
        if self._proc is None or self._job is None:
            return
        code = self._proc.poll()
        if code is None:
            return
        self._job.exit_code = code
        self._job.finished_at = datetime.now(timezone.utc).isoformat()
        if self._log_fp is not None:
            self._log_fp.flush()
            self._log_fp.close()
            self._log_fp = None
        if code != 0:
            self._job.state = "failed"
            self._job.error = f"自动调优进程异常退出 (exit_code={code})"
            self._proc = None
            return
        try:
            report = json.loads(
                (self._project_root / self._job.report_path).read_text(encoding="utf-8")
            )
            recommended, results = self._validate_report(report, self._job.candidates)
            self._settings_saver({"evaluation_workers": recommended})
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            self._job.state = "failed"
            self._job.error = f"自动调优结果无效: {exc}"
        else:
            self._job.state = "completed"
            self._job.benchmark_mode = str(report["mode"])
            self._job.recommended_workers = recommended
            self._job.results = results
        self._proc = None

    @staticmethod
    def _validate_report(
        report: dict[str, Any], candidates: list[int]
    ) -> tuple[int, list[dict[str, Any]]]:
        if report.get("schema_version") != CPU_TUNER_SCHEMA_VERSION:
            raise ValueError("benchmark schema version mismatch")
        if (report.get("parity") or {}).get("exact") is not True:
            raise ValueError("benchmark did not preserve exact result parity")
        if report.get("mode") not in {"fixture_screen", "real_training_data"}:
            raise ValueError("benchmark mode is invalid")
        if report.get("screen_candidates") != candidates:
            raise ValueError("benchmark screen candidates do not match the job")
        tested = report.get("tested_candidates")
        if (
            not isinstance(tested, list)
            or not tested
            or any(type(value) is not int or value not in candidates for value in tested)
        ):
            raise ValueError("benchmark tested candidates are invalid")
        recommended = report.get("recommended_workers")
        if type(recommended) is not int or recommended not in tested:
            raise ValueError("recommended worker count is not a tested candidate")
        variants = report.get("variants")
        if not isinstance(variants, dict):
            raise ValueError("benchmark variants are missing")
        results: list[dict[str, Any]] = []
        for workers in tested:
            row = variants.get(str(workers))
            if not isinstance(row, dict) or row.get("exact_parity") is not True:
                raise ValueError(f"workers={workers} did not preserve exact parity")
            step_seconds = row.get("training_step_seconds")
            raw_seconds = row.get("raw_evaluation_seconds")
            wall_seconds = row.get("wall_seconds")
            throughput = row.get("formulas_per_second")
            if (
                not isinstance(step_seconds, (int, float))
                or not math.isfinite(float(step_seconds))
                or step_seconds <= 0
                or not isinstance(wall_seconds, (int, float))
                or not math.isfinite(float(wall_seconds))
                or wall_seconds <= 0
                or (
                    raw_seconds is not None
                    and (
                        not isinstance(raw_seconds, (int, float))
                        or not math.isfinite(float(raw_seconds))
                        or raw_seconds <= 0
                    )
                )
                or (
                    throughput is not None
                    and (
                        not isinstance(throughput, (int, float))
                        or not math.isfinite(float(throughput))
                        or throughput <= 0
                    )
                )
            ):
                raise ValueError(f"workers={workers} has invalid timing data")
            results.append(
                {
                    "workers": workers,
                    "median_seconds": round(float(step_seconds), 4),
                    "raw_evaluation_seconds": (
                        None if raw_seconds is None else round(float(raw_seconds), 4)
                    ),
                    "wall_seconds": round(float(wall_seconds), 4),
                    "formulas_per_second": (
                        None if throughput is None else round(float(throughput), 2)
                    ),
                    "recommended": workers == recommended,
                }
            )
        return recommended, results


cpu_tuning_manager = CpuTuningManager()
