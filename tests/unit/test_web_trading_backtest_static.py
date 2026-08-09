from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_recent_seven_day_backtest_ui_main_flow() -> None:
    html = (ROOT / "web/static/index.html").read_text(encoding="utf-8")
    js = (ROOT / "web/static/app.js").read_text(encoding="utf-8")
    assert 'id="tradingBacktestStartBtn"' in html
    assert 'id="tradingBacktestStopBtn"' in html
    assert 'id="tradingBacktestStatus"' in html
    assert "/api/trading/backtest/start" in js
    assert "/api/trading/backtest/status" in js
    assert "研究模拟" in html
    assert "候选" in html and "Codex" in html


def test_app_js_node_syntax() -> None:
    completed = subprocess.run(
        ["node", "--check", str(ROOT / "web/static/app.js")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
