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


def test_realtime_page_exposes_stage4_account_config_and_preview_controls() -> None:
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
        "tradingPreviewStatus",
        "tradingDecisionPanel",
    }.issubset(parser.ids)
    content = " ".join(parser.text)
    assert "只读决策预览" in content
    assert "不会发送真实订单" in content


def test_stage4_javascript_uses_only_readonly_preview_routes() -> None:
    source = (ROOT / "web/static/app.js").read_text(encoding="utf-8")

    assert 'fetchJSON("/api/trading/account"' in source
    assert 'fetchJSON("/api/trading/config"' in source
    assert 'fetchJSON("/api/trading/status"' in source
    assert 'fetchJSON("/api/trading/preview/start"' in source
    assert 'fetchJSON("/api/trading/preview/stop"' in source
    assert "/api/trading/enable" not in source
    assert "order_send" not in source


def test_stage4_javascript_is_syntax_valid() -> None:
    completed = subprocess.run(
        ["node", "--check", "web/static/app.js"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
