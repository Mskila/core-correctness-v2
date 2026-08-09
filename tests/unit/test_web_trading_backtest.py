from __future__ import annotations

from fastapi.testclient import TestClient

from web.app import app
import web.app as app_module


class _Manager:
    def __init__(self) -> None:
        self.started = False

    def start(self, config):
        self.started = True
        return {"active": True, "state": "running"}

    def status(self):
        return {"active": self.started, "state": "running" if self.started else "idle"}

    def stop(self):
        self.started = False
        return {"active": False, "state": "stopping"}

    def report(self):
        return {"report_path": "report.json", "modes": {}}


def test_trading_backtest_api_start_status_report_stop(monkeypatch) -> None:
    manager = _Manager()
    monkeypatch.setattr(app_module, "trading_backtest_manager", manager)
    monkeypatch.setattr(
        app_module.trading_preview_manager,
        "config_state",
        lambda: {"config": app_module.TradingConfigV1.default().to_payload()},
    )
    client = TestClient(app)
    assert client.post("/api/trading/backtest/start").status_code == 200
    assert client.get("/api/trading/backtest/status").json()["active"] is True
    assert client.get("/api/trading/backtest/report").json()["report_path"] == "report.json"
    assert client.post("/api/trading/backtest/stop").status_code == 200
