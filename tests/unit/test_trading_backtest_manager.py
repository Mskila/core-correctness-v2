from __future__ import annotations

import time

import pytest

from trading_core import TradingConfigV1
from web.trading_backtest import TradingBacktestManager


class _Runner:
    def run(self, config, *, modes, stop_event, progress):
        assert config == TradingConfigV1.default()
        assert modes == ("rules", "rules_codex")
        progress("rules", 1, 2, 900)
        return {"report_path": "backtest_output/trading/latest.json", "modes": {}}


class _BlockingRunner:
    def run(self, config, *, modes, stop_event, progress):
        while not stop_event.wait(0.01):
            progress("rules", 0, 1, None)
        return None


class _FailingRunner:
    def run(self, config, *, modes, stop_event, progress):
        raise RuntimeError("insufficient historical warmup bars: H1 12/2137")


def _wait(manager: TradingBacktestManager) -> dict:
    for _ in range(100):
        status = manager.status()
        if not status["active"]:
            return status
        time.sleep(0.01)
    raise AssertionError("manager did not finish")


def test_manager_runs_both_modes_and_exposes_report() -> None:
    manager = TradingBacktestManager(runner=_Runner())
    assert manager.start(TradingConfigV1.default())["active"] is True
    status = _wait(manager)
    assert status["state"] == "completed"
    assert manager.report()["report_path"].endswith("latest.json")


def test_manager_rejects_duplicate_and_cooperatively_stops() -> None:
    manager = TradingBacktestManager(runner=_BlockingRunner())
    manager.start(TradingConfigV1.default())
    with pytest.raises(RuntimeError, match="active"):
        manager.start(TradingConfigV1.default())
    manager.stop()
    assert _wait(manager)["state"] == "stopped"


def test_manager_surfaces_insufficient_warmup_as_failed() -> None:
    manager = TradingBacktestManager(runner=_FailingRunner())
    manager.start(TradingConfigV1.default())
    status = _wait(manager)
    assert status["state"] == "failed"
    assert "insufficient historical warmup" in status["error"]
