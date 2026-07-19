import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

import web.app as web_app
import web.backtest_manager as manager_module
from tests.unit.test_artifacts import strategy_artifact


def test_request_requires_explicit_data_and_exact_mode() -> None:
    with pytest.raises(ValidationError):
        web_app.StartBacktestRequest(strategy_file="s.json")
    with pytest.raises(ValidationError):
        web_app.StartBacktestRequest(strategy_file="s.json", data_file="d.parquet", mode="oos")
    req = web_app.StartBacktestRequest(
        strategy_file="s.json", data_file="d.parquet", mode="in_sample_replay"
    )
    assert req.data_file == "d.parquet"
    assert req.mode == "in_sample_replay"


def test_manager_forwards_explicit_mode_data_and_costs(monkeypatch, tmp_path) -> None:
    artifact = strategy_artifact()
    strategy = tmp_path / artifact.run_identity.strategy_filename()
    strategy.write_text(json.dumps(artifact.to_dict()), encoding="utf-8")
    data = tmp_path / "data.parquet"
    data.write_bytes(b"data")
    captured = {}

    class _Process:
        pid = 42
        def poll(self): return None

    def _popen(command, **kwargs):
        captured["command"] = command
        return _Process()

    monkeypatch.setattr(manager_module, "LOG_DIR", tmp_path)
    monkeypatch.setattr(manager_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(manager_module.subprocess, "Popen", _popen)
    manager = manager_module.BacktestManager()
    job = manager.start(
        strategy_file=str(strategy), data_file=str(data),
        mode="out_of_sample_backtest", commission_pct=0.12, slippage_pct=0.34,
    )
    try:
        assert job.data_file == str(data)
        assert job.mode == "out_of_sample_backtest"
        command = captured["command"]
        assert command[command.index("--data-file") + 1] == str(data)
        assert command[command.index("--mode") + 1] == "out_of_sample_backtest"
        assert command[command.index("--commission") + 1] == "0.12"
        assert command[command.index("--slippage") + 1] == "0.34"
    finally:
        manager._log_fp.close()


def test_api_uses_only_request_dataset_and_mode(monkeypatch, tmp_path) -> None:
    explicit = str(tmp_path / "explicit.parquet")
    captured = {}
    monkeypatch.setattr(web_app, "_inspect_strategy_or_http", lambda path: {
        "strategy_file": path, "symbol": "EURUSD", "source_path": "forbidden.parquet"
    })
    monkeypatch.setattr(web_app, "load_settings", lambda: {"last_data_file": "forbidden-last.parquet"})
    monkeypatch.setattr(web_app, "save_settings", lambda value: None)
    monkeypatch.setattr(web_app, "inspect_parquet_file", lambda path: {
        "valid": True, "data_file": path
    })
    monkeypatch.setattr(web_app.backtest_manager, "start", lambda **kwargs: (
        captured.update(kwargs) or SimpleNamespace(to_dict=lambda: kwargs)
    ))
    request = web_app.StartBacktestRequest(
        strategy_file="strategy.json", data_file=explicit,
        mode="out_of_sample_backtest", commission_pct=0.1, slippage_pct=0.2,
    )
    web_app.api_backtest_start(request)
    assert captured["data_file"] == explicit
    assert captured["mode"] == "out_of_sample_backtest"
