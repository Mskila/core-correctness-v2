from __future__ import annotations

import json

import pytest
from fastapi import HTTPException

import web.app as web_app
import web.cpu_tuner as cpu_tuner
import benchmarks.perf05 as perf05
from benchmarks.perf05 import select_real_data_finalists


def _report(candidates: list[int], recommended: int, *, exact: bool = True) -> dict:
    return {
        "schema_version": cpu_tuner.CPU_TUNER_SCHEMA_VERSION,
        "mode": "fixture_screen",
        "parity": {"exact": exact},
        "screen_candidates": candidates,
        "tested_candidates": candidates,
        "recommended_workers": recommended,
        "variants": {
            str(workers): {
                "exact_parity": exact,
                "training_step_seconds": workers / 1000,
                "raw_evaluation_seconds": None,
                "wall_seconds": workers / 500,
                "formulas_per_second": 1000.0 / workers,
            }
            for workers in candidates
        },
    }


def test_default_worker_candidates_respect_cpu_and_memory_caps() -> None:
    gib = 1024**3
    assert cpu_tuner.default_worker_candidates(
        logical_processors=32, total_memory_bytes=32 * gib
    ) == [1, 2, 4, 6, 8, 12, 16, 24, 32]
    assert cpu_tuner.default_worker_candidates(
        logical_processors=32, total_memory_bytes=16 * gib
    ) == [1, 2, 4, 6, 8, 12, 16]


def test_real_data_finalists_cover_current_and_faster_neighbor_range() -> None:
    candidates = (1, 2, 4, 6, 8, 12, 16, 24)
    quick = {
        "recommended_workers": 4,
        "variants": {
            str(workers): {"step_ns": {"median": abs(workers - 4) + 1}}
            for workers in candidates
        },
    }
    assert select_real_data_finalists(candidates, quick, 8) == (2, 4, 6, 8, 12)


def test_two_stage_benchmark_recommends_real_training_winner(
    monkeypatch, tmp_path
) -> None:
    candidates = (1, 2, 4, 6, 8, 12, 16, 24)
    quick = {
        "parity": {"exact": True},
        "recommended_workers": 4,
        "variants": {
            str(workers): {
                "exact_parity": True,
                "step_ns": {"median": (abs(workers - 4) + 1) * 1_000_000},
            }
            for workers in candidates
        },
    }
    monkeypatch.setattr(perf05, "run_quick_benchmark", lambda **kwargs: quick)
    step_seconds = {2: 20.5, 4: 19.4, 6: 18.1, 8: 16.9, 12: 20.2}
    monkeypatch.setattr(
        perf05,
        "run_real_benchmark",
        lambda **kwargs: {
            "training_digest": "same-digest",
            "stage_seconds": {
                "compute_total": step_seconds[kwargs["workers"]],
                "raw_evaluation": step_seconds[kwargs["workers"]] - 4,
            },
            "wall_seconds": step_seconds[kwargs["workers"]] + 6,
        },
    )
    data_file = tmp_path / "XAUUSD_M15.parquet"
    data_file.write_bytes(b"fixture")
    report = perf05.run_benchmark(
        worker_counts=candidates,
        iterations=3,
        current_workers=8,
        data_file=data_file,
    )
    assert report["mode"] == "real_training_data"
    assert report["tested_candidates"] == [2, 4, 6, 8, 12]
    assert report["recommended_workers"] == 8
    assert report["parity"] == {"exact": True}


def test_completed_tuning_persists_recommendation(monkeypatch, tmp_path) -> None:
    commands = []
    saved = []

    class Process:
        pid = 2468

        def poll(self):
            return 0

    def popen(command, **kwargs):
        commands.append((list(command), kwargs))
        return Process()

    monkeypatch.setattr(cpu_tuner, "LOG_DIR", tmp_path)
    monkeypatch.setattr(cpu_tuner.subprocess, "Popen", popen)
    monkeypatch.setattr(cpu_tuner, "popen_group_options", lambda: {})
    manager = cpu_tuner.CpuTuningManager(
        project_root=tmp_path,
        settings_saver=lambda payload: saved.append(payload) or payload,
        settings_loader=lambda: {"evaluation_workers": saved[-1]["evaluation_workers"] if saved else 8},
    )
    job = manager.start(candidates=[1, 2, 4], iterations=1)
    command, kwargs = commands[0]
    assert command[command.index("--workers") + 1 : command.index("--current-workers")] == [
        "1", "2", "4"
    ]
    assert kwargs["cwd"] == tmp_path
    report_path = tmp_path / job.report_path
    report_path.write_text(json.dumps(_report([1, 2, 4], 2)), encoding="utf-8")

    status = manager.status()
    assert status["active"] is False
    assert status["job"]["state"] == "completed"
    assert status["job"]["benchmark_mode"] == "fixture_screen"
    assert status["job"]["recommended_workers"] == 2
    assert status["configured_workers"] == 2
    assert status["job"]["results"][1]["recommended"] is True
    assert saved == [{"evaluation_workers": 2}]


def test_tuning_rejects_non_parity_report(monkeypatch, tmp_path) -> None:
    class Process:
        pid = 2468

        def poll(self):
            return 0

    monkeypatch.setattr(cpu_tuner, "LOG_DIR", tmp_path)
    monkeypatch.setattr(cpu_tuner.subprocess, "Popen", lambda *args, **kwargs: Process())
    monkeypatch.setattr(cpu_tuner, "popen_group_options", lambda: {})
    saved = []
    manager = cpu_tuner.CpuTuningManager(
        project_root=tmp_path,
        settings_saver=lambda payload: saved.append(payload) or payload,
        settings_loader=lambda: {"evaluation_workers": 8},
    )
    job = manager.start(candidates=[1, 2], iterations=1)
    (tmp_path / job.report_path).write_text(
        json.dumps(_report([1, 2], 2, exact=False)), encoding="utf-8"
    )

    status = manager.status()
    assert status["job"]["state"] == "failed"
    assert "完全一致" in status["job"]["error"] or "exact" in status["job"]["error"]
    assert saved == []


def test_api_starts_tuning_only_when_other_workloads_are_idle(monkeypatch) -> None:
    monkeypatch.setattr(web_app.training_manager, "status", lambda: {"active": False})
    monkeypatch.setattr(web_app.backtest_manager, "status", lambda: {"active": False})
    monkeypatch.setattr(web_app.realtime_manager, "status", lambda: {"running": False})
    monkeypatch.setattr(
        web_app,
        "load_settings",
        lambda: {
            "last_data_file": "",
            "numeric_time_unit": "s",
            "evaluation_workers": 8,
        },
    )
    started = {}

    class Job:
        def to_dict(self):
            return {"state": "running", "candidates": [1, 2, 4]}

    monkeypatch.setattr(
        web_app.cpu_tuning_manager,
        "start",
        lambda **kwargs: started.update(kwargs) or Job(),
    )
    assert web_app.api_cpu_tuning_start() == {
        "ok": True,
        "job": {"state": "running", "candidates": [1, 2, 4]},
    }
    assert started == {
        "data_file": None,
        "numeric_time_unit": "s",
        "current_workers": 8,
    }

    monkeypatch.setattr(web_app.training_manager, "status", lambda: {"active": True})
    with pytest.raises(HTTPException, match="训练进行中") as failure:
        web_app.api_cpu_tuning_start()
    assert failure.value.status_code == 409
