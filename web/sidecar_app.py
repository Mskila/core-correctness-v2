"""Read-only Web sidecar that mirrors a running AlphaMaster service."""
from __future__ import annotations

import json
import os
import re
import time
import asyncio
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import unquote
from urllib.request import Request as UrlRequest
from urllib.request import urlopen

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from data_pipeline.parquet_manager import parse_parquet_filename
from model_core.config import ModelConfig
from web.progress import list_strategies
from web.settings import load_settings
from web.sidecar_progress import get_symbol_progress


ORIGIN_URL = os.environ.get(
    "ALPHAMASTER_SIDECAR_ORIGIN", "http://127.0.0.1:8765"
).rstrip("/")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = Path(__file__).resolve().parent / "static"
LOG_DIR = PROJECT_ROOT / "logs"

_SIDE_EFFECT_GET_PATHS = {
    "/api/data-file/browse",
    "/api/strategy-file/browse",
    "/api/strategy-file/sync-best",
}
_FORWARDED_HEADERS = {
    "cache-control", "content-disposition", "content-type", "etag", "last-modified"
}


def _origin_get(path_and_query: str) -> tuple[int, dict[str, str], bytes]:
    request = UrlRequest(
        f"{ORIGIN_URL}{path_and_query}",
        headers={"Accept-Encoding": "identity"},
        method="GET",
    )
    try:
        with urlopen(request, timeout=5) as response:
            return response.status, dict(response.headers.items()), response.read()
    except HTTPError as exc:
        return exc.code, dict(exc.headers.items()), exc.read()
    except URLError as exc:
        raise ConnectionError(f"cannot reach AlphaMaster origin {ORIGIN_URL}: {exc}") from exc


def _proxy_response(status: int, headers: dict[str, str], body: bytes) -> Response:
    forwarded = {
        key: value for key, value in headers.items()
        if key.lower() in _FORWARDED_HEADERS
    }
    return Response(content=body, status_code=status, headers=forwarded)


def _progress_payload(symbol: str) -> dict[str, Any]:
    progress = get_symbol_progress(symbol)
    return {
        "symbol": progress.symbol,
        "status": progress.status,
        "current_step": progress.current_step,
        "train_steps": progress.train_steps,
        "progress_pct": round(progress.progress_pct, 1),
        "best_score": progress.best_score,
        "best_formula": progress.best_formula,
        "formula_decoded": progress.formula_decoded,
        "has_checkpoint": bool(progress.checkpoint_path),
        "has_strategy": progress.has_strategy,
        "strategy_score": progress.strategy_score,
        "checkpoint_path": progress.checkpoint_path,
        "history": progress.history,
    }


def _local_file_info(data_file: str) -> dict[str, Any] | None:
    if not data_file:
        return None
    path = Path(data_file)
    try:
        symbol, timeframe = parse_parquet_filename(path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {path}")
        return {
            "data_file": str(path.resolve()),
            "filename": path.name,
            "symbol": symbol,
            "timeframe": timeframe,
            "valid": True,
            "message": "",
        }
    except Exception as exc:
        return {"data_file": data_file, "valid": False, "message": str(exc)}


def _latest_training_log(symbol: str) -> Path | None:
    safe_symbol = symbol.replace(".", "_")
    return max(
        LOG_DIR.glob(f"train_{safe_symbol}_*.log"),
        key=lambda path: path.stat().st_mtime_ns,
        default=None,
    )


def _read_log_tail(path: Path | None, lines: int = 120) -> list[str]:
    if path is None:
        return []
    try:
        with path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            stream.seek(max(0, size - 262_144))
            text = stream.read().decode("utf-8", errors="replace")
        return text.splitlines()[-lines:]
    except OSError:
        return []


def _local_snapshot() -> tuple[dict[str, Any], dict[str, Any]]:
    settings = load_settings()
    data_file = str(settings.get("last_data_file") or "")
    file_info = _local_file_info(data_file)
    symbol = str((file_info or {}).get("symbol") or "")
    progress = _progress_payload(symbol) if symbol else None
    log_path = _latest_training_log(symbol) if symbol else None
    log_tail = _read_log_tail(log_path)
    live_step = None
    for line in reversed(log_tail):
        match = re.search(r"\[(\d+)/\d+\]", line)
        if match:
            live_step = int(match.group(1))
            break
    active = bool(
        log_path
        and time.time() - log_path.stat().st_mtime < 120
        and progress
        and progress["current_step"] < progress["train_steps"]
    )
    if progress and live_step is not None:
        progress["current_step"] = max(progress["current_step"], live_step)
        progress["progress_pct"] = round(
            min(100.0, 100.0 * progress["current_step"] / progress["train_steps"]),
            1,
        )
    if progress and active:
        progress["status"] = "running_job"
    job = None
    if symbol and log_path:
        job = {
            "data_file": data_file,
            "symbol": symbol,
            "timeframe": (file_info or {}).get("timeframe"),
            "state": "running" if active else "completed",
            "pid": None,
            "log_path": str(log_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        }
    training = {
        "active": active,
        "job": job,
        "log_tail": log_tail,
        "errors": {"counts": {}, "messages": []},
    }
    overview = {"data_file": file_info, "progress": progress, "training": training}
    return overview, training


def _patch_overview(body: bytes) -> bytes:
    payload = json.loads(body.decode("utf-8"))
    training = payload.get("training") or {}
    job = training.get("job") or {}
    old_progress = payload.get("progress") or {}
    data_file = payload.get("data_file") or {}
    symbol = job.get("symbol") or old_progress.get("symbol") or data_file.get("symbol")
    if not symbol:
        return body

    recovered = _progress_payload(str(symbol))
    if training.get("active"):
        recovered["status"] = "running_job"
    old_step = old_progress.get("current_step")
    if type(old_step) is int:
        recovered["current_step"] = max(recovered["current_step"], old_step)
        total = recovered["train_steps"]
        recovered["progress_pct"] = (
            round(min(100.0, 100.0 * recovered["current_step"] / total), 1)
            if total > 0 else 0.0
        )
    for field in ("session_seconds", "history_total_seconds"):
        if field in old_progress:
            recovered[field] = old_progress[field]
    payload["progress"] = recovered
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def create_app() -> FastAPI:
    app = FastAPI(title="AlphaMaster Read-only Training Sidecar", version="1.0.0")
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.api_route(
        "/{path:path}",
        methods=["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    )
    async def sidecar(request: Request, path: str) -> Response:
        if request.method not in {"GET", "HEAD"}:
            return JSONResponse(
                status_code=405,
                content={"detail": "read-only sidecar does not allow this request"},
            )

        request_path = "/" + path
        if request_path in _SIDE_EFFECT_GET_PATHS:
            return JSONResponse(
                status_code=405,
                content={"detail": "read-only sidecar blocks this GET endpoint"},
            )

        if request_path.startswith("/api/symbols/"):
            symbol = unquote(request_path.removeprefix("/api/symbols/"))
            if not symbol or "/" in symbol:
                return JSONResponse(status_code=404, content={"detail": "not found"})
            response = JSONResponse(_progress_payload(symbol))
            return Response(status_code=response.status_code) if request.method == "HEAD" else response

        if request_path == "/api/overview":
            overview, _ = _local_snapshot()
            response = JSONResponse(overview)
            return Response(status_code=response.status_code) if request.method == "HEAD" else response
        if request_path == "/api/health":
            return JSONResponse({"status": "ok", "version": "1.2.0-sidecar"})
        if request_path == "/api/config":
            settings = load_settings()
            overview, _ = _local_snapshot()
            workers = int(settings.get("evaluation_workers") or ModelConfig.EVALUATION_WORKERS)
            return JSONResponse({
                "train_steps": ModelConfig.TRAIN_STEPS,
                "batch_size": ModelConfig.BATCH_SIZE,
                "reward_mode": ModelConfig.REWARD_MODE,
                "max_formula_len": ModelConfig.MAX_FORMULA_LEN,
                "device": str(ModelConfig.DEVICE),
                "last_data_file": settings.get("last_data_file", ""),
                "numeric_time_unit": settings.get("numeric_time_unit", "s"),
                "evaluation_workers": workers,
                "worker_candidates": [workers],
                "cpu_hardware": {"logical_processors": os.cpu_count()},
                "data_file": overview["data_file"],
                "last_strategy_file": settings.get("last_strategy_file", ""),
                "strategy_file": None,
                "debug_mode": False,
                "ai_provider": settings.get("ai_provider", "deepseek"),
                "ai_api_key": "",
                "bt_commission_pct": settings.get("bt_commission_pct", 0.02),
                "bt_slippage_pct": settings.get("bt_slippage_pct", 0.01),
                "server_log": "",
                "error_log": "",
            })
        if request_path == "/api/ai/providers":
            return JSONResponse({"providers": [], "selected": "deepseek", "has_api_key": False})
        if request_path == "/api/training/status":
            _, training = _local_snapshot()
            response = JSONResponse(training)
            return Response(status_code=response.status_code) if request.method == "HEAD" else response
        if request_path == "/api/strategies":
            response = JSONResponse({"strategies": list_strategies()})
            return Response(status_code=response.status_code) if request.method == "HEAD" else response
        if request_path == "/api/cpu-tuning/status":
            return JSONResponse({"active": False, "job": None})
        if request_path.startswith("/api/debug/logs"):
            return JSONResponse({
                "server_tail": [], "error_tail": [],
                "server_log": "", "error_log": "",
            })

        target = request_path
        if request.url.query:
            target += "?" + request.url.query
        try:
            status, headers, body = await asyncio.to_thread(_origin_get, target)
            if request.method == "HEAD":
                body = b""
            return _proxy_response(status, headers, body)
        except (ConnectionError, ValueError, json.JSONDecodeError) as exc:
            return JSONResponse(status_code=502, content={"detail": str(exc)})

    return app


app = create_app()
