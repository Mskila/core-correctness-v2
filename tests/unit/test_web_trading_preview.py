from __future__ import annotations

from fastapi.testclient import TestClient

import web.app as web_app
from trading_core import TradingConfigV1


class _FakePreviewManager:
    def __init__(self) -> None:
        self.config = TradingConfigV1.default()
        self.running = False

    def load_persisted(self) -> None:
        pass

    def config_state(self):
        return {
            "config": self.config.to_payload(),
            "active_config_hash": None,
            "pending_next_m15": True,
        }

    def update_config(self, config):
        self.config = config
        return self.config_state()

    def status(self):
        return {
            "stage": 5,
            "read_only": False,
            "execution_enabled": False,
            "running": self.running,
            "latest_decision": None,
        }

    def decisions(self, *, limit):
        return [{"decision_id": "decision-1", "limit": limit}]

    def start(self):
        self.running = True
        return self.status()

    def stop(self):
        self.running = False
        return self.status()


class _FakeAccountReader:
    def account_snapshot(self):
        return {
            "available": True,
            "connected": True,
            "read_only": True,
            "execution_enabled": False,
            "login": 123456,
            "server": "Demo-Server",
            "account_kind": "demo",
            "position_mode": "hedging",
            "trade_allowed": True,
        }


class _FakeExecutionController:
    def __init__(self) -> None:
        self.enabled = False

    def status(self):
        return {
            "execution_enabled": self.enabled,
            "management_enabled": True,
            "management_running": True,
            "blocker": "" if self.enabled else "manual_enable_required",
            "last_error": "",
            "account": None,
            "state": {
                "schema_version": "trading-execution-state-v1",
                "last_processed_m15": None,
                "config_hash": None,
                "cooldown_until_m15": None,
                "managed_trade": None,
                "receipt_count": 0,
            },
            "recent_receipts": [],
        }

    def enable(self, config):
        assert config.symbol == "XAUUSD"
        self.enabled = True
        return self.status()

    def disable(self):
        self.enabled = False
        return self.status()

    def start_management(self):
        return self.status()

    def stop_management(self):
        return self.status()


def _client(monkeypatch):
    manager = _FakePreviewManager()
    execution = _FakeExecutionController()
    monkeypatch.setattr(web_app, "trading_preview_manager", manager)
    monkeypatch.setattr(web_app, "mt5_read_only_market", _FakeAccountReader())
    monkeypatch.setattr(web_app, "trading_execution_controller", execution)
    return TestClient(web_app.app), manager, execution


def test_trading_account_config_status_and_decisions_expose_stage5_state(monkeypatch) -> None:
    client, manager, _ = _client(monkeypatch)

    account = client.get("/api/trading/account")
    config = client.get("/api/trading/config")
    status = client.get("/api/trading/status")
    decisions = client.get("/api/trading/decisions?limit=7")

    assert account.status_code == 200
    assert account.json()["execution_enabled"] is False
    assert account.json()["management_enabled"] is True
    assert config.status_code == 200
    assert config.json()["config"]["config_hash"] == manager.config.config_hash
    assert {row["id"] for row in config.json()["catalog"]["pa_families"]} == {
        "trend_continuation",
        "breakout",
        "reversal",
        "range",
    }
    assert status.json()["stage"] == 5
    assert status.json()["read_only"] is False
    assert status.json()["execution_enabled"] is False
    assert status.json()["execution"]["blocker"] == "manual_enable_required"
    assert decisions.json() == {"decisions": [{"decision_id": "decision-1", "limit": 7}]}


def test_config_update_and_preview_start_stop_main_flow(monkeypatch) -> None:
    client, manager, _ = _client(monkeypatch)
    payload = manager.config.to_payload()
    payload.pop("config_hash")
    payload["mode"] = "rules_codex"
    payload["tp1_lots"] = 0.02
    payload["tp2_lots"] = 0.01

    saved = client.put("/api/trading/config", json=payload)
    started = client.post("/api/trading/preview/start")
    stopped = client.post("/api/trading/preview/stop")

    assert saved.status_code == 200
    assert saved.json()["config"]["mode"] == "rules_codex"
    assert saved.json()["config"]["tp1_lots"] == 0.02
    assert saved.json()["pending_next_m15"] is True
    assert started.json()["running"] is True
    assert started.json()["execution_enabled"] is False
    assert stopped.json()["running"] is False


def test_manual_enable_starts_m15_loop_and_disable_only_blocks_new_entries(monkeypatch) -> None:
    client, manager, execution = _client(monkeypatch)

    enabled = client.post("/api/trading/enable")
    disabled = client.post("/api/trading/disable")

    assert enabled.status_code == 200
    assert enabled.json()["execution_enabled"] is True
    assert manager.running is True
    assert disabled.status_code == 200
    assert disabled.json()["execution_enabled"] is False
    assert disabled.json()["execution"]["management_enabled"] is True
    assert execution.enabled is False


def test_stopping_preview_also_disables_new_entries(monkeypatch) -> None:
    client, manager, execution = _client(monkeypatch)
    client.post("/api/trading/enable")

    stopped = client.post("/api/trading/preview/stop")

    assert stopped.status_code == 200
    assert stopped.json()["running"] is False
    assert stopped.json()["execution_enabled"] is False
    assert execution.enabled is False


def test_config_rejects_zero_tp1_lots(monkeypatch) -> None:
    client, manager, _ = _client(monkeypatch)
    payload = manager.config.identity_payload()
    payload["tp1_lots"] = 0.0

    response = client.put("/api/trading/config", json=payload)

    assert response.status_code == 400
    assert "tp1_lots" in response.json()["detail"]
