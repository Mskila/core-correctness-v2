"""Read-only Web sidecar that mirrors a running AlphaMaster service."""
from __future__ import annotations

import json
import os
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import unquote
from urllib.request import Request as UrlRequest
from urllib.request import urlopen

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from web.sidecar_progress import get_symbol_progress


ORIGIN_URL = os.environ.get(
    "ALPHAMASTER_SIDECAR_ORIGIN", "http://127.0.0.1:8765"
).rstrip("/")

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

        target = request_path
        if request.url.query:
            target += "?" + request.url.query
        try:
            status, headers, body = _origin_get(target)
            if request_path == "/api/overview" and 200 <= status < 300:
                body = _patch_overview(body)
                headers = dict(headers)
                headers["content-type"] = "application/json; charset=utf-8"
            if request.method == "HEAD":
                body = b""
            return _proxy_response(status, headers, body)
        except (ConnectionError, ValueError, json.JSONDecodeError) as exc:
            return JSONResponse(status_code=502, content={"detail": str(exc)})

    return app


app = create_app()
