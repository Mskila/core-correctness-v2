import json
import hashlib
import re
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

import web.app as web_app
import web.progress as progress
import web.strategy_file as strategy_file
from model_core.artifacts import (
    ArtifactIdentity,
    StrategyArtifact,
    TrainingRunIdentity,
    sha256_json,
)
from model_core.reward import target_bars_per_trade
from model_core.vocab import FORMULA_VOCAB
from tests.unit.test_artifacts import strategy_artifact
from web.strategy_file import inspect_strategy_file


def test_sync_best_reports_expected_empty_state_without_404(monkeypatch) -> None:
    monkeypatch.setattr(web_app, "_sync_and_persist_best_strategy", lambda _symbol: None)
    response = TestClient(web_app.app).post(
        "/api/strategy-file/sync-best?symbol=XAUUSD"
    )
    assert response.status_code == 200
    assert response.json() == {
        "ok": True,
        "available": False,
        "symbol": "XAUUSD",
        "strategy_file": None,
    }


def test_sync_best_frontend_failure_path_does_not_recurse() -> None:
    source = (web_app.STATIC_DIR / "app.js").read_text(encoding="utf-8")
    match = re.search(
        r"async function applyBestStrategyForBacktest\(.*?\n}\n\n"
        r"async function loadBacktestStrategyContext",
        source,
        flags=re.DOTALL,
    )
    assert match is not None
    body = match.group(0).split("async function loadBacktestStrategyContext", 1)[0]
    assert "loadBacktestStrategyContext" not in body


@pytest.mark.parametrize("payload", [[0], {"formula": [0]}, {"schema_version": "strategy-v2"}])
def test_strategy_inspection_rejects_legacy_or_identityless(tmp_path, payload) -> None:
    path = tmp_path / "best_EURUSD.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="不兼容的 V2 strategy artifact"):
        inspect_strategy_file(str(path))


def _timeframe_artifact(
    timeframe: str, generated_at: str, token: int = 0, run_id: str | None = None
) -> StrategyArtifact:
    template = strategy_artifact()
    source = template.run_identity.artifact_identity
    dataset = replace(source.training_dataset, timeframe=timeframe)
    config = source.to_dict()["training_config"]
    config["timeframe_reward"] = {
        "timeframe": timeframe,
        "target_trades_per_day": 2.0,
        "target_bars_per_trade": target_bars_per_trade(timeframe, 2.0),
    }
    identity = ArtifactIdentity(
        core_semantics_version=source.core_semantics_version,
        vocab_version=source.vocab_version,
        label_semantics_version=source.label_semantics_version,
        execution_semantics_version=source.execution_semantics_version,
        symbol=source.symbol,
        timeframe=timeframe,
        training_dataset=dataset,
        training_config=config,
        training_config_hash=sha256_json(config),
    )
    run = (
        TrainingRunIdentity.create(identity)
        if run_id is None
        else TrainingRunIdentity(run_id=run_id, artifact_identity=identity)
    )
    return StrategyArtifact.create(
        run_identity=run,
        formula_tokens=[token],
        decoded_formula=FORMULA_VOCAB.token_names[token],
        best_score=template.best_score,
        fold_evidence=template.fold_evidence,
        generated_at=generated_at,
        candidate_evaluation_count=1,
    )


def _write_artifact(directory, artifact):
    path = directory / artifact.run_identity.strategy_filename()
    path.write_text(json.dumps(artifact.to_dict()), encoding="utf-8")
    return path


def _strategy_export_response(monkeypatch, tmp_path):
    strategies = tmp_path / "strategies"
    checkpoints = tmp_path / "checkpoints"
    strategies.mkdir()
    checkpoints.mkdir()
    earlier = _timeframe_artifact(
        "H1", "2026-07-16T00:00:00.0000001Z", 0, "f" * 32
    )
    selected = _timeframe_artifact(
        "H1", "2026-07-16T00:00:00.0000002Z", 1, "0" * 32
    )
    _write_artifact(strategies, earlier)
    selected_path = _write_artifact(strategies, selected)

    legacy = strategies / "best_EURUSD.json"
    legacy.write_bytes(b"LEGACY-USER-BYTES")
    noncanonical = strategies / "best_v2_EURUSD_H1_user-copy.json"
    noncanonical.write_bytes(selected_path.read_bytes())
    tampered_artifact = _timeframe_artifact(
        "H1", "2026-07-15T00:00:00Z", 0, "1" * 32
    )
    tampered_payload = tampered_artifact.to_dict()
    tampered_payload["formula_tokens"] = [1]
    tampered = strategies / tampered_artifact.run_identity.strategy_filename()
    tampered.write_text(json.dumps(tampered_payload), encoding="utf-8")
    before = {path.name: path.read_bytes() for path in strategies.iterdir()}

    monkeypatch.setattr(progress, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(progress, "CHECKPOINT_DIR", checkpoints)
    monkeypatch.setattr(progress, "STRATEGIES_DIR", strategies)
    monkeypatch.setattr(strategy_file, "STRATEGIES_DIR", strategies)
    progress.invalidate_checkpoint_cache()
    response = TestClient(web_app.app).get("/api/strategies/EURUSD/export")
    return response, selected_path, selected, before, strategies


def test_strategy_export_route_returns_exact_selected_artifact_bytes(
    monkeypatch, tmp_path
) -> None:
    response, selected_path, _, before, strategies = _strategy_export_response(
        monkeypatch, tmp_path
    )
    source = selected_path.read_bytes()
    assert response.status_code == 200
    assert response.content == source, (
        f"source_sha={hashlib.sha256(source).hexdigest()} "
        f"export_sha={hashlib.sha256(response.content).hexdigest()}"
    )
    assert {path.name: path.read_bytes() for path in strategies.iterdir()} == before


def test_strategy_export_route_uses_canonical_content_disposition(
    monkeypatch, tmp_path
) -> None:
    response, selected_path, _, _, _ = _strategy_export_response(monkeypatch, tmp_path)
    assert response.headers["content-disposition"] == (
        f'attachment; filename="{selected_path.name}"'
    )
    assert not response.headers["content-disposition"].startswith(
        'attachment; filename="strategy_EURUSD_step'
    )


def test_strategy_export_route_preserves_full_artifact_fields(monkeypatch, tmp_path) -> None:
    response, selected_path, _, _, _ = _strategy_export_response(monkeypatch, tmp_path)
    exported = json.loads(response.content)
    source = json.loads(selected_path.read_bytes())
    assert set(exported) == set(source)
    assert set(exported) != {
        "artifact_fingerprint", "best_score", "formula", "formula_decoded",
        "run_identity", "strategy_fingerprint",
    }


def test_strategy_export_route_round_trips_strict_strategy_artifact(
    monkeypatch, tmp_path
) -> None:
    response, _, selected, _, _ = _strategy_export_response(monkeypatch, tmp_path)
    loaded = StrategyArtifact.from_dict(json.loads(response.content))
    assert loaded == selected
    assert loaded.run_identity == selected.run_identity
    assert loaded.fingerprint == selected.fingerprint
    assert loaded.formula_tokens == selected.formula_tokens
    assert loaded.generated_at == selected.generated_at


def test_strategy_resolution_is_exact_per_symbol_and_timeframe(monkeypatch, tmp_path) -> None:
    strategies = tmp_path / "strategies"
    strategies.mkdir()
    h1_old = _write_artifact(
        strategies, _timeframe_artifact("H1", "2026-07-15T00:00:00Z")
    )
    h1_new = _write_artifact(
        strategies, _timeframe_artifact("H1", "2026-07-16T00:00:00Z")
    )
    h4 = _write_artifact(
        strategies, _timeframe_artifact("H4", "2026-07-17T00:00:00Z")
    )
    monkeypatch.setattr(strategy_file, "STRATEGIES_DIR", strategies)
    assert strategy_file.strategy_path_for_symbol("EURUSD", "H1") == h1_new
    assert strategy_file.strategy_path_for_symbol("EURUSD", "H4") == h4
    assert strategy_file.strategy_path_for_symbol("EURUSD", "H1") != h1_old


def test_strategy_resolution_orders_generated_at_as_utc_instant(monkeypatch, tmp_path) -> None:
    strategies = tmp_path / "strategies"
    strategies.mkdir()
    earlier = _write_artifact(
        strategies, _timeframe_artifact("H1", "2026-07-16T00:00:00Z", 0)
    )
    later = _write_artifact(
        strategies, _timeframe_artifact("H1", "2026-07-16T00:00:00.500000Z", 1)
    )
    before = {path.name: path.read_bytes() for path in strategies.iterdir()}
    monkeypatch.setattr(strategy_file, "STRATEGIES_DIR", strategies)
    selected = strategy_file.strategy_path_for_symbol("EURUSD", "H1")
    assert selected == later
    assert selected != earlier
    assert json.loads(selected.read_text(encoding="utf-8"))["formula_tokens"] == [1]
    assert {path.name: path.read_bytes() for path in strategies.iterdir()} == before


def test_equal_utc_instants_use_filename_not_timestamp_spelling(monkeypatch, tmp_path) -> None:
    strategies = tmp_path / "strategies"
    strategies.mkdir()
    z_path = _write_artifact(
        strategies,
        _timeframe_artifact(
            "H1", "2026-07-16T00:00:00Z", 0, "0" * 32
        ),
    )
    offset_path = _write_artifact(
        strategies,
        _timeframe_artifact(
            "H1", "2026-07-16T00:00:00+00:00", 1, "f" * 32
        ),
    )
    monkeypatch.setattr(strategy_file, "STRATEGIES_DIR", strategies)
    assert offset_path.name > z_path.name
    assert strategy_file.strategy_path_for_symbol("EURUSD", "H1") == offset_path


def test_strategy_resolution_preserves_submicrosecond_order(monkeypatch, tmp_path) -> None:
    strategies = tmp_path / "strategies"
    strategies.mkdir()
    earlier = _write_artifact(
        strategies,
        _timeframe_artifact(
            "H1", "2026-07-16T00:00:00.0000001Z", 0, "f" * 32
        ),
    )
    later = _write_artifact(
        strategies,
        _timeframe_artifact(
            "H1", "2026-07-16T00:00:00.0000002Z", 1, "0" * 32
        ),
    )
    assert earlier.name > later.name
    before = {path.name: path.read_bytes() for path in strategies.iterdir()}
    monkeypatch.setattr(strategy_file, "STRATEGIES_DIR", strategies)
    assert strategy_file.strategy_path_for_symbol("EURUSD", "H1") == later
    assert {path.name: path.read_bytes() for path in strategies.iterdir()} == before


def test_numerically_equal_arbitrary_fractions_use_filename_tie(
    monkeypatch, tmp_path
) -> None:
    strategies = tmp_path / "strategies"
    strategies.mkdir()
    short = _write_artifact(
        strategies,
        _timeframe_artifact("H1", "2026-07-16T00:00:00.1Z", 0, "0" * 32),
    )
    padded = _write_artifact(
        strategies,
        _timeframe_artifact(
            "H1", "2026-07-16T00:00:00.1000000+00:00", 1, "f" * 32
        ),
    )
    monkeypatch.setattr(strategy_file, "STRATEGIES_DIR", strategies)
    assert padded.name > short.name
    assert strategy_file.strategy_path_for_symbol("EURUSD", "H1") == padded
