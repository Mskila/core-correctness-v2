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
            "stage": 4,
            "read_only": True,
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


def _client(monkeypatch):
    manager = _FakePreviewManager()
    monkeypatch.setattr(web_app, "trading_preview_manager", manager)
    monkeypatch.setattr(web_app, "mt5_read_only_market", _FakeAccountReader())
    return TestClient(web_app.app), manager


def test_trading_account_config_status_and_decisions_are_read_only(monkeypatch) -> None:
    client, manager = _client(monkeypatch)

    account = client.get("/api/trading/account")
    config = client.get("/api/trading/config")
    status = client.get("/api/trading/status")
    decisions = client.get("/api/trading/decisions?limit=7")

    assert account.status_code == 200
    assert account.json()["execution_enabled"] is False
    assert config.status_code == 200
    assert config.json()["config"]["config_hash"] == manager.config.config_hash
    assert {row["id"] for row in config.json()["catalog"]["pa_families"]} == {
        "trend_continuation",
        "breakout",
        "reversal",
        "range",
    }
    assert status.json()["read_only"] is True
    assert status.json()["execution_enabled"] is False
    assert decisions.json() == {"decisions": [{"decision_id": "decision-1", "limit": 7}]}


def test_config_update_and_preview_start_stop_main_flow(monkeypatch) -> None:
    client, manager = _client(monkeypatch)
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


def test_config_rejects_zero_tp1_lots(monkeypatch) -> None:
    client, manager = _client(monkeypatch)
    payload = manager.config.identity_payload()
    payload["tp1_lots"] = 0.0

    response = client.put("/api/trading/config", json=payload)

    assert response.status_code == 400
    assert "tp1_lots" in response.json()["detail"]
