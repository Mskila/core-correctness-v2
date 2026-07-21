"""Subprocess manager for run_backtest.py jobs.

镜像 training_manager 的设计：用子进程运行 run_backtest.py，
把 stdout 写入 logs/backtest_*.log，前端通过轮询读取尾部日志与阶段进度。
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.train_logging import strip_ansi
from model_core.error_telemetry import ErrorTelemetry
from web.process_lifecycle import (
    ProcessLifecycleError,
    popen_group_options,
    stop_process_tree,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

# 回测阶段：用日志关键字推断当前进行到哪一步，用于前端进度展示
BACKTEST_PHASES: list[tuple[str, str]] = [
    ("init", "初始化"),
    ("cost", "交易成本"),
    ("strategy", "加载策略"),
    ("data", "加载行情数据"),
    ("compute", "回测计算"),
    ("chart", "生成图表"),
    ("done", "完成"),
]
_PHASE_KEYS = [p[0] for p in BACKTEST_PHASES]


class JobState(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


@dataclass
class BacktestJob:
    strategy_file: str
    data_file: str
    mode: str
    symbol: str
    commission_pct: float = 0.02
    slippage_pct: float = 0.01
    state: JobState = JobState.RUNNING
    pid: int | None = None
    log_path: str = ""
    started_at: str = ""
    finished_at: str | None = None
    exit_code: int | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy_file": self.strategy_file,
            "data_file": self.data_file,
            "mode": self.mode,
            "symbol": self.symbol,
            "commission_pct": self.commission_pct,
            "slippage_pct": self.slippage_pct,
            "state": self.state.value,
            "pid": self.pid,
            "log_path": self.log_path,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "exit_code": self.exit_code,
            "error": self.error,
        }


class BacktestManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._job: BacktestJob | None = None
        self._log_fp = None
        self._stopped_by_user = False
        self._errors = ErrorTelemetry()

    def status(self) -> dict[str, Any]:
        with self._lock:
            self._refresh_state()
            job_dict = self._job.to_dict() if self._job else None
        phase_key, phase_label, phase_idx = self._current_phase()
        return {
            "active": self._job is not None and self._job.state == JobState.RUNNING,
            "job": job_dict,
            "phase": phase_key,
            "phase_label": phase_label,
            "phase_index": phase_idx,
            "phase_total": len(BACKTEST_PHASES),
            "errors": self._errors.snapshot(),
        }

    def start(
        self,
        strategy_file: str,
        data_file: str,
        mode: str,
        commission_pct: float = 0.02,
        slippage_pct: float = 0.01,
    ) -> BacktestJob:
        with self._lock:
            self._refresh_state()
            if self._proc is not None and self._proc.poll() is None:
                raise RuntimeError("已有回测任务在运行")

            from web.strategy_file import inspect_strategy_file

            info = inspect_strategy_file(strategy_file)
            symbol = info.get("symbol") or ""

            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            log_path = LOG_DIR / f"backtest_{ts}.log"

            cmd = [
                sys.executable,
                "-u",
                "run_backtest.py",
                "--strategy-file",
                strategy_file,
                "--data-file", data_file,
                "--mode", mode,
                "--commission",
                str(commission_pct),
                "--slippage",
                str(slippage_pct),
            ]
            self._log_fp = open(log_path, "w", encoding="utf-8", buffering=1)
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            env["PYTHONIOENCODING"] = "utf-8"
            env["PYTHONUTF8"] = "1"
            env["LOGURU_COLORIZE"] = "0"

            self._stopped_by_user = False
            self._proc = subprocess.Popen(
                cmd,
                cwd=PROJECT_ROOT,
                stdout=self._log_fp,
                stderr=subprocess.STDOUT,
                env=env,
                **popen_group_options(),
            )
            self._job = BacktestJob(
                strategy_file=strategy_file,
                data_file=data_file,
                mode=mode,
                symbol=symbol,
                commission_pct=float(commission_pct),
                slippage_pct=float(slippage_pct),
                pid=self._proc.pid,
                log_path=str(log_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                started_at=datetime.now(timezone.utc).isoformat(),
            )
            return self._job

    def stop(self, *, timeout: float = 5.0) -> bool:
        with self._lock:
            if self._proc is None or self._proc.poll() is not None:
                return False
            self._stopped_by_user = True
            try:
                stop_process_tree(
                    self._proc,
                    timeout=timeout,
                    telemetry=self._errors,
                )
            except ProcessLifecycleError as failure:
                if self._job is not None:
                    self._job.error = str(failure)
                raise
            self._refresh_state()
            return True

    def _current_phase(self) -> tuple[str, str, int]:
        """根据日志内容推断当前回测阶段。"""
        lines = self.tail_log(200)
        if not lines:
            return ("init", "初始化", 0)
        text = "\n".join(lines)
        # 从后往前匹配最靠后的阶段关键字
        detected = "init"
        if "交易成本" in text or "手续费=" in text:
            detected = "cost"
        if "加载各品种策略" in text or re.search(r"score=", text) or "模式:" in text:
            detected = "strategy"
        if "正在加载数据" in text:
            detected = "data"
        if re.search(r"品种:\s*\[", text) or "多因子回测报告" in text:
            detected = "compute"
        if "生成 K 线图" in text or "张缩放图" in text:
            detected = "chart"
        if "完成。" in text or "JSON 报告已保存" in text:
            detected = "done"
        idx = _PHASE_KEYS.index(detected) if detected in _PHASE_KEYS else 0
        label = BACKTEST_PHASES[idx][1]
        return (detected, label, idx)

    def tail_log(self, lines: int = 200) -> list[str]:
        with self._lock:
            if not self._job or not self._job.log_path:
                return []
            path = PROJECT_ROOT / self._job.log_path
            if not path.exists():
                return []
            try:
                content = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                self._errors.record("log.read", "backtest log read failed")
                return []
            return [strip_ansi(line) for line in content.splitlines()[-lines:]]

    def _refresh_state(self) -> None:
        if self._proc is None or self._job is None:
            return
        code = self._proc.poll()
        if code is None:
            return
        self._job.exit_code = code
        self._job.finished_at = datetime.now(timezone.utc).isoformat()
        if self._job.state == JobState.RUNNING:
            if self._stopped_by_user:
                self._job.state = JobState.STOPPED
            elif code == 0:
                self._job.state = JobState.COMPLETED
            elif code < 0:
                self._job.state = JobState.STOPPED
            else:
                self._job.state = JobState.FAILED
        if self._job.state == JobState.FAILED and self._job.error is None:
            self._job.error = f"回测进程异常退出 (exit_code={code})"
            try:
                if self._job.log_path:
                    path = PROJECT_ROOT / self._job.log_path
                    with path.open("a", encoding="utf-8") as fp:
                        fp.write(f"\n[Web] 回测进程已结束，退出码: {code}\n")
            except OSError:
                self._errors.record("log.append_exit", "backtest exit log append failed")
        if self._log_fp:
            try:
                self._log_fp.flush()
            except Exception as failure:
                self._errors.record("log.flush", failure)
            try:
                self._log_fp.close()
            except Exception as failure:
                self._errors.record("log.close", failure)
            self._log_fp = None
        self._proc = None


backtest_manager = BacktestManager()
