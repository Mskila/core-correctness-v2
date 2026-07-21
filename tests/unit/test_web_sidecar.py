import io
import json

import torch
from fastapi.testclient import TestClient

from model_core.artifacts import StrategyArtifact
from tests.unit.test_artifacts import strategy_artifact
from tests.unit.test_web_progress_v2 import _history_value


def _checkpoint(path, artifact: StrategyArtifact, *, step: int, history_steps) -> None:
    history = _history_value(artifact, history_steps, include_identity=False)
    stream = io.BytesIO()
    torch.save(
        {
            "checkpoint_schema_version": "checkpoint-v3",
            "run_identity": artifact.run_identity.to_dict(),
            "step": step,
            "best_score": artifact.best_score,
            "best_formula": list(artifact.formula_tokens),
            "training_history": history,
            "rank_monitor_history": history["stable_rank"],
        },
        stream,
    )
    path.write_bytes(stream.getvalue())


def _layout(monkeypatch, tmp_path):
    import web.sidecar_progress as sidecar_progress

    checkpoints = tmp_path / "checkpoints"
    strategies = tmp_path / "strategies"
    checkpoints.mkdir()
    strategies.mkdir()
    monkeypatch.setattr(sidecar_progress, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(sidecar_progress, "CHECKPOINT_DIR", checkpoints)
    monkeypatch.setattr(sidecar_progress, "STRATEGIES_DIR", strategies)
    monkeypatch.setattr(sidecar_progress.ModelConfig, "TRAIN_STEPS", 20)
    sidecar_progress.invalidate_checkpoint_cache()
    return sidecar_progress, checkpoints


def test_sidecar_accepts_completed_step_checkpoint_and_ignores_identityless_history(
    monkeypatch, tmp_path
) -> None:
    sidecar_progress, checkpoints = _layout(monkeypatch, tmp_path)
    artifact = strategy_artifact()
    checkpoint = checkpoints / artifact.run_identity.checkpoint_filename(2)
    _checkpoint(checkpoint, artifact, step=2, history_steps=[0, 1, 2])

    standalone = tmp_path / artifact.run_identity.history_filename()
    standalone.write_text(
        json.dumps(_history_value(artifact, [0, 1, 2], include_identity=False)),
        encoding="utf-8",
    )
    before = standalone.read_bytes()

    result = sidecar_progress.get_symbol_progress("EURUSD")

    assert result.status == "in_progress"
    assert result.current_step == 3
    assert result.history is not None
    assert result.history["step"] == [0, 1, 2]
    assert result.incompatible_reasons == ()
    assert standalone.read_bytes() == before


def test_sidecar_accepts_next_step_checkpoint_format(monkeypatch, tmp_path) -> None:
    sidecar_progress, checkpoints = _layout(monkeypatch, tmp_path)
    artifact = strategy_artifact()
    checkpoint = checkpoints / artifact.run_identity.checkpoint_filename(2)
    _checkpoint(checkpoint, artifact, step=2, history_steps=[0, 1])

    result = sidecar_progress.get_symbol_progress("EURUSD")

    assert result.status == "in_progress"
    assert result.current_step == 2
    assert result.history is not None
    assert result.history["step"] == [0, 1]


def test_sidecar_rejects_unrelated_checkpoint_history(monkeypatch, tmp_path) -> None:
    sidecar_progress, checkpoints = _layout(monkeypatch, tmp_path)
    artifact = strategy_artifact()
    checkpoint = checkpoints / artifact.run_identity.checkpoint_filename(4)
    _checkpoint(checkpoint, artifact, step=4, history_steps=[0, 1])

    result = sidecar_progress.get_symbol_progress("EURUSD")

    assert result.status == "incompatible"
    assert result.current_step == 0
    assert result.history is None


def test_sidecar_mirrors_origin_status_and_blocks_mutations(monkeypatch) -> None:
    import web.sidecar_app as sidecar_app

    calls = []

    def fake_origin_get(path_and_query: str):
        calls.append(path_and_query)
        payload = {
            "active": True,
            "job": {"symbol": "XAUUSD", "pid": 4321, "state": "running"},
            "log_tail": ["[31/9000] training"],
        }
        return 200, {"content-type": "application/json"}, json.dumps(payload).encode()

    monkeypatch.setattr(sidecar_app, "_origin_get", fake_origin_get)
    client = TestClient(sidecar_app.create_app())

    mirrored = client.get("/api/training/status")
    assert mirrored.status_code == 200
    assert mirrored.json()["job"]["pid"] == 4321
    assert calls == ["/api/training/status"]

    assert client.post("/api/training/stop").status_code == 405
    assert client.put("/api/settings", json={}).status_code == 405
    assert client.delete("/api/example").status_code == 405
    assert client.get("/api/data-file/browse").status_code == 405
    assert calls == ["/api/training/status"]
