from __future__ import annotations

import subprocess
import sys
import time
from types import SimpleNamespace

import pytest


class _TimeoutThenExitProcess:
    pid = 8123

    def __init__(self) -> None:
        self.calls: list[object] = []
        self.returncode: int | None = None

    def poll(self) -> int | None:
        self.calls.append("poll")
        return self.returncode

    def terminate(self) -> None:
        self.calls.append("terminate")

    def kill(self) -> None:
        self.calls.append("kill")
        self.returncode = 137

    def wait(self, timeout: float | None = None) -> int:
        self.calls.append(("wait", timeout))
        if self.returncode is None:
            raise subprocess.TimeoutExpired("training", timeout)
        return self.returncode


class _GracefulProcess(_TimeoutThenExitProcess):
    def terminate(self) -> None:
        self.calls.append("terminate")
        self.returncode = 0


def test_process_stop_escalates_terminate_wait_kill_wait(monkeypatch) -> None:
    import web.process_lifecycle as lifecycle

    process = _TimeoutThenExitProcess()
    monkeypatch.setattr(lifecycle, "_terminate_process_tree", lambda proc: proc.terminate())
    monkeypatch.setattr(lifecycle, "_kill_process_tree", lambda proc: proc.kill())

    result = lifecycle.stop_process_tree(process, timeout=0.25)

    assert process.calls == [
        "poll",
        "terminate",
        ("wait", 0.25),
        "kill",
        ("wait", 0.25),
    ]
    assert result.exit_code == 137
    assert result.forced is True


def test_process_stop_does_not_kill_after_graceful_exit(monkeypatch) -> None:
    import web.process_lifecycle as lifecycle

    process = _GracefulProcess()
    monkeypatch.setattr(lifecycle, "_terminate_process_tree", lambda proc: proc.terminate())
    monkeypatch.setattr(
        lifecycle,
        "_kill_process_tree",
        lambda proc: (_ for _ in ()).throw(AssertionError("kill must not run")),
    )

    result = lifecycle.stop_process_tree(process, timeout=0.25)

    assert process.calls == ["poll", "terminate", ("wait", 0.25)]
    assert result.exit_code == 0
    assert result.forced is False


def test_process_stop_failure_is_structured_and_fail_closed(monkeypatch) -> None:
    import web.process_lifecycle as lifecycle

    process = _TimeoutThenExitProcess()
    monkeypatch.setattr(
        lifecycle,
        "_terminate_process_tree",
        lambda proc: (_ for _ in ()).throw(PermissionError("terminate denied")),
    )

    telemetry = lifecycle.ErrorTelemetry(message_limit=2, message_chars=32)
    with pytest.raises(lifecycle.ProcessLifecycleError, match="terminate"):
        lifecycle.stop_process_tree(process, timeout=0.25, telemetry=telemetry)

    snapshot = telemetry.snapshot()
    assert snapshot["counts"] == {"process.terminate": 1}
    assert len(snapshot["messages"]) == 1
    assert len(snapshot["messages"][0]["message"]) <= 32


def test_training_manager_stop_reaches_terminal_state_and_closes_log(monkeypatch) -> None:
    import web.training_manager as manager_module

    process = _TimeoutThenExitProcess()
    log = SimpleNamespace(flushed=False, closed=False)
    log.flush = lambda: setattr(log, "flushed", True)
    log.close = lambda: setattr(log, "closed", True)
    manager = manager_module.TrainingManager()
    manager._proc = process
    manager._log_fp = log
    manager._job = manager_module.TrainingJob(
        data_file="fixture.parquet",
        symbol="EURUSD",
        timeframe="H1",
        mode="ftmo",
    )
    monkeypatch.setattr(
        manager_module,
        "stop_process_tree",
        lambda proc, **kwargs: (
            setattr(proc, "returncode", 137)
            or SimpleNamespace(exit_code=137, forced=True)
        ),
    )
    monkeypatch.setattr(manager, "_record_session_time", lambda: None)

    assert manager.stop(timeout=0.25) is True
    status = manager.status()
    assert status["active"] is False
    assert status["job"]["state"] == "stopped"
    assert status["job"]["exit_code"] == 137
    assert log.flushed is True
    assert log.closed is True


def test_training_manager_closes_log_even_when_flush_fails(monkeypatch) -> None:
    import web.training_manager as manager_module

    process = _TimeoutThenExitProcess()
    process.returncode = 143

    class _FailingLog:
        closed = False

        def flush(self) -> None:
            raise OSError("flush failed")

        def close(self) -> None:
            self.closed = True

    log = _FailingLog()
    manager = manager_module.TrainingManager()
    manager._proc = process
    manager._log_fp = log
    manager._stopped_by_user = True
    manager._job = manager_module.TrainingJob(
        data_file="fixture.parquet",
        symbol="EURUSD",
        timeframe="H1",
        mode="ftmo",
    )
    monkeypatch.setattr(manager, "_record_session_time", lambda: None)

    manager.status()

    assert log.closed is True
    assert manager._errors.snapshot()["counts"] == {"log.flush": 1}


def test_error_telemetry_counts_exactly_and_bounds_messages() -> None:
    from model_core.error_telemetry import ErrorTelemetry

    telemetry = ErrorTelemetry(message_limit=2, message_chars=12)
    telemetry.record("formula.non_finite", ValueError("first-message-is-long"))
    telemetry.record("formula.non_finite", ValueError("second"))
    telemetry.record("process.wait", RuntimeError("third"))

    snapshot = telemetry.snapshot()
    assert snapshot["counts"] == {
        "formula.non_finite": 2,
        "process.wait": 1,
    }
    assert len(snapshot["messages"]) == 2
    assert all(len(row["message"]) <= 12 for row in snapshot["messages"])


def _pid_exists(pid: int) -> bool:
    if sys.platform == "win32":
        import ctypes

        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        ctypes.windll.kernel32.CloseHandle(handle)
        return True
    try:
        import os

        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    stat_path = f"/proc/{pid}/stat"
    if os.path.exists(stat_path):
        try:
            with open(stat_path, encoding="ascii") as stat_file:
                if stat_file.read().split()[2] == "Z":
                    return False
        except (OSError, IndexError):
            pass
    return True


def test_real_process_tree_stop_reaps_parent_and_child(tmp_path) -> None:
    from web.process_lifecycle import popen_group_options, stop_process_tree

    child_pid_file = tmp_path / "child.pid"
    parent = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import pathlib,subprocess,sys,time;"
                "child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']);"
                "pathlib.Path(sys.argv[1]).write_text(str(child.pid));"
                "time.sleep(60)"
            ),
            str(child_pid_file),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        **popen_group_options(),
    )
    child_pid = None
    try:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not child_pid_file.exists():
            time.sleep(0.02)
        assert child_pid_file.exists()
        child_pid = int(child_pid_file.read_text(encoding="utf-8"))
        assert _pid_exists(child_pid)

        stop_process_tree(parent, timeout=2.0)
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and _pid_exists(child_pid):
            time.sleep(0.02)
        assert parent.poll() is not None
        assert not _pid_exists(child_pid)
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait(timeout=2.0)
        if child_pid is not None and _pid_exists(child_pid):
            if sys.platform == "win32":
                subprocess.run(
                    ["taskkill", "/PID", str(child_pid), "/F"],
                    check=False,
                    capture_output=True,
                )


def test_core_responsibility_boundaries_are_independently_importable() -> None:
    from model_core import artifact_publication
    from model_core import batch_transaction
    from model_core import checkpoint_codec
    from model_core import formula_evaluation
    from model_core import sampling
    from model_core import serial_decision
    from backtest_viz import output_publication

    assert callable(sampling.ConstrainedSampler)
    assert callable(formula_evaluation.evaluate_training_formula)
    assert callable(serial_decision.commit_pending_actions)
    assert callable(batch_transaction.begin_batch_transaction)
    assert callable(artifact_publication.publish_training_history)
    assert callable(checkpoint_codec.save_checkpoint)
    assert callable(output_publication.publish_output_set)
