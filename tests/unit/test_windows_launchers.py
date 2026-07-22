from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_start_launcher_bootstraps_and_records_scoped_pid() -> None:
    text = (ROOT / "start_alphamaster.bat").read_text(encoding="utf-8")
    assert ".venv\\Scripts\\python.exe" in text
    assert "requirements.txt" in text
    assert "constraints-core.txt" in text
    assert ".alphamaster_web.pid" in text
    assert "run_web.py" in text
    assert "http://127.0.0.1:8765/api/health" in text
    assert "Start-Process" in text
    assert "netstat.exe" in text
    assert "Get-NetTCPConnection" not in text
    assert "ALPHAMASTER_PYTHON" in text
    assert "ALPHAMASTER_NO_BROWSER" in text
    assert "import fastapi, multipart" in text
    assert "错误日志末尾" in text
    assert "ALPHAMASTER_NO_PAUSE" in text
    assert (ROOT / "start_alphamaster.bat").read_bytes().count(b"\r\n") > 20


def test_stop_launcher_never_kills_python_by_image_name() -> None:
    text = (ROOT / "stop_alphamaster.bat").read_text(encoding="utf-8")
    lowered = text.lower()
    assert ".alphamaster_web.pid" in text
    assert "expectedScript" in text
    assert "--port\\s+8765" in text
    assert "Stop-Process -Id" in text
    assert "taskkill.exe /im python" not in lowered
    assert "stop-process -name python" not in lowered
    assert "netstat.exe" in text
    assert "Get-NetTCPConnection" not in text
    assert "ALPHAMASTER_NO_PAUSE" in text
    assert (ROOT / "stop_alphamaster.bat").read_bytes().count(b"\r\n") > 10
