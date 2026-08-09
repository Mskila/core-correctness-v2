from __future__ import annotations

import subprocess
from html.parser import HTMLParser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class _Ids(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: set[str] = set()
        self.text: list[str] = []

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if "id" in values:
            self.ids.add(values["id"])

    def handle_data(self, data):
        if data.strip():
            self.text.append(data.strip())


def test_realtime_page_exposes_stage5_account_config_preview_and_manual_execution() -> None:
    parser = _Ids()
    parser.feed((ROOT / "web/static/index.html").read_text(encoding="utf-8"))

    assert {
        "tradingPreviewPanel",
        "tradingAccountGrid",
        "tradingModeSelect",
        "tradingTp1Lots",
        "tradingTp2Lots",
        "tradingDirectionModules",
        "tradingEntryModules",
        "tradingPaFamilies",
        "tradingSaveConfigBtn",
        "tradingStartPreviewBtn",
        "tradingStopPreviewBtn",
        "tradingEnableBtn",
        "tradingDisableBtn",
        "tradingPreviewStatus",
        "tradingExecutionStatus",
        "tradingDecisionPanel",
    }.issubset(parser.ids)
    content = " ".join(parser.text)
    assert "MT5 决策与执行" in content
    assert "新开仓在每次启动后默认关闭" in content
    assert "手动启用" in content


def test_stage5_javascript_requires_manual_enable_and_exposes_disable() -> None:
    source = (ROOT / "web/static/app.js").read_text(encoding="utf-8")

    assert 'fetchJSON("/api/trading/account"' in source
    assert 'fetchJSON("/api/trading/config"' in source
    assert 'fetchJSON("/api/trading/status"' in source
    assert 'fetchJSON("/api/trading/preview/start"' in source
    assert 'fetchJSON("/api/trading/preview/stop"' in source
    assert 'fetchJSON("/api/trading/enable"' in source
    assert 'fetchJSON("/api/trading/disable"' in source
    assert "window.confirm" in source
    assert "order_send" not in source


def test_stage5_decision_card_renders_its_execution_confirmation_and_receipts() -> None:
    source = (ROOT / "web/static/app.js").read_text(encoding="utf-8")

    assert "decision.execution" in source
    assert "执行动作" in source
    assert "查询确认" in source
    assert "订单票据" in source
    assert "执行回执" in source


def test_stage5_javascript_is_syntax_valid() -> None:
    completed = subprocess.run(
        ["node", "--check", "web/static/app.js"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
