"""Deterministic subprocess-tree shutdown shared by Web job managers."""
from __future__ import annotations

import os
import signal
import subprocess
import sys
from dataclasses import dataclass

from model_core.error_telemetry import ErrorTelemetry


class ProcessLifecycleError(RuntimeError):
    """Raised when a process cannot be driven to a terminal state."""


@dataclass(frozen=True)
class ProcessStopResult:
    exit_code: int
    forced: bool


def popen_group_options() -> dict[str, object]:
    """Return platform options that make the spawned process tree addressable."""
    if sys.platform == "win32":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def _terminate_process_tree(process: subprocess.Popen) -> None:
    if sys.platform == "win32":
        completed = subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T"],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode not in (0, 128):
            process.terminate()
        return
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        process.terminate()


def _kill_process_tree(process: subprocess.Popen) -> None:
    if sys.platform == "win32":
        completed = subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode not in (0, 128):
            process.kill()
        return
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        process.kill()


def stop_process_tree(
    process: subprocess.Popen,
    *,
    timeout: float = 5.0,
    telemetry: ErrorTelemetry | None = None,
) -> ProcessStopResult:
    """Perform terminate -> wait -> kill -> wait and return the final exit code."""
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    current = process.poll()
    if current is not None:
        return ProcessStopResult(exit_code=int(current), forced=False)
    try:
        _terminate_process_tree(process)
    except Exception as failure:
        if telemetry is not None:
            telemetry.record("process.terminate", failure)
        raise ProcessLifecycleError(
            f"process terminate failed: {type(failure).__name__}"
        ) from failure
    try:
        return ProcessStopResult(exit_code=int(process.wait(timeout=timeout)), forced=False)
    except subprocess.TimeoutExpired:
        pass
    except Exception as failure:
        if telemetry is not None:
            telemetry.record("process.wait", failure)
        raise ProcessLifecycleError(
            f"process wait after terminate failed: {type(failure).__name__}"
        ) from failure
    try:
        _kill_process_tree(process)
    except Exception as failure:
        if telemetry is not None:
            telemetry.record("process.kill", failure)
        raise ProcessLifecycleError(
            f"process kill failed: {type(failure).__name__}"
        ) from failure
    try:
        return ProcessStopResult(exit_code=int(process.wait(timeout=timeout)), forced=True)
    except Exception as failure:
        if telemetry is not None:
            telemetry.record("process.wait_after_kill", failure)
        raise ProcessLifecycleError(
            f"process wait after kill failed: {type(failure).__name__}"
        ) from failure
