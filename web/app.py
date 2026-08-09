"""FastAPI application for AlphaMaster training UI."""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any
from typing import Literal

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_pipeline.parquet_manager import inspect_parquet_file
from model_core.config import ModelConfig
from model_core.semantics import DataValidationError
from web.file_dialog import pick_parquet_file, pick_strategy_file
from web.progress import (
    generated_at_utc,
    get_symbol_progress,
    get_strategy_for_export,
    invalidate_checkpoint_cache,
    list_strategies,
)
from web.server_log import (
    debug_snapshot,
    get_logger,
    is_debug_mode,
    log_error,
    set_debug_mode,
    setup_logging,
)
from web.settings import load_settings, save_settings
from web.strategy_file import (
    inspect_strategy_file,
    resolve_strategy_file,
    strategy_path_for_symbol,
    sync_best_strategy_for_symbol,
)
from web.training_manager import training_manager
from web.cpu_tuner import cpu_tuning_manager
from web.training_time import get_training_time_summary
from web.training_package import build_training_export_zip, import_training_package
from web.backtest_manager import backtest_manager
from web.realtime_manager import realtime_manager
from web.data_sources.factory import list_sources
from web.trading_mt5 import mt5_read_only_market
from web.trading_execution import trading_execution_controller
from web.trading_preview import trading_preview_manager
from strategy_manager.live_signal import min_exposure
from trading_core import (
    DIRECTION_MODULE_IDS,
    ENTRY_MODULE_IDS,
    PA_FAMILY_IDS,
    TRADING_MODES,
    TradingConfigV1,
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
BACKTEST_OUTPUT_DIR = ROOT / "backtest_output"
_data_file_info_cache: dict[tuple[str, int, int, str], dict[str, Any]] = {}


def _inspect_parquet_cached(
    path: str, *, numeric_time_unit: str
) -> dict[str, Any]:
    resolved = Path(path).resolve()
    stat = resolved.stat()
    key = (str(resolved), stat.st_mtime_ns, stat.st_size, numeric_time_unit)
    cached = _data_file_info_cache.get(key)
    if cached is not None:
        return dict(cached)
    inspected = inspect_parquet_file(
        resolved, numeric_time_unit=numeric_time_unit
    )
    _data_file_info_cache.clear()
    _data_file_info_cache[key] = dict(inspected)
    return dict(inspected)

setup_logging()
logger = get_logger()

app = FastAPI(title="AlphaMaster Training", version="1.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class StartTrainingRequest(BaseModel):
    data_file: str
    from_scratch: bool = False
    numeric_time_unit: Literal["s", "ms", "us", "ns"] = "s"
    evaluation_workers: int | None = Field(default=None, ge=1, le=64)


class CpuTuningRequest(BaseModel):
    data_file: str | None = None
    numeric_time_unit: Literal["s", "ms", "us", "ns"] | None = None


class ClientLogRequest(BaseModel):
    level: str = "error"
    message: str
    context: dict[str, Any] | None = None


class SettingsRequest(BaseModel):
    last_data_file: str | None = None
    last_strategy_file: str | None = None
    numeric_time_unit: Literal["s", "ms", "us", "ns"] | None = None
    evaluation_workers: int | None = Field(default=None, ge=1, le=64)
    debug_mode: bool | None = None
    ai_provider: str | None = None
    ai_api_key: str | None = None
    bt_commission_pct: float | None = None
    bt_slippage_pct: float | None = None


class AnalyzeTrainingRequest(BaseModel):
    provider: str | None = None
    api_key: str | None = None
    symbol: str | None = None


class StartBacktestRequest(BaseModel):
    strategy_file: str
    data_file: str
    mode: Literal["in_sample_replay", "out_of_sample_backtest"]
    commission_pct: float | None = None
    slippage_pct: float | None = None
    numeric_time_unit: Literal["s", "ms", "us", "ns"] = "s"


class AddWatchRequest(BaseModel):
    source: str
    symbol: str
    timeframe: str
    strategy_file: str


class RemoveWatchRequest(BaseModel):
    id: str


class FeishuSettingsRequest(BaseModel):
    enabled: bool | None = None
    webhook_url: str | None = None
    secret: str | None = None


class FeishuTestRequest(BaseModel):
    webhook_url: str | None = None
    secret: str | None = None


class TradingConfigRequest(BaseModel):
    mode: str
    direction_module_ids: list[str]
    entry_module_ids: list[str]
    pa_family_ids: list[str]
    tp1_lots: float
    tp2_lots: float
    symbol: str = "XAUUSD"
    config_version: str = "trading-config-v1"


@app.middleware("http")
async def log_requests(request: Request, call_next):
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception as exc:
        log_error(f"{request.method} {request.url.path} unhandled", exc)
        raise
    elapsed_ms = (time.perf_counter() - started) * 1000
    if is_debug_mode():
        logger.info(
            "%s %s -> %s (%.1fms)",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
    if response.status_code >= 400:
        log_error(f"{request.method} {request.url.path} -> HTTP {response.status_code}")
    return response


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    log_error(f"{request.method} {request.url.path} HTTP {exc.status_code}: {exc.detail}")
    detail = exc.detail
    if not isinstance(detail, str):
        detail = str(detail)
    return JSONResponse(status_code=exc.status_code, content={"detail": detail})


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    log_error(f"{request.method} {request.url.path} crashed", exc)
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc), "traceback": traceback.format_exc()},
    )


def _inspect_or_http(path: str, numeric_time_unit: str) -> dict[str, Any]:
    try:
        return inspect_parquet_file(path, numeric_time_unit=numeric_time_unit)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e)) from e
    except (ValueError, DataValidationError) as e:
        raise HTTPException(400, str(e)) from e


def _browse_data_file(numeric_time_unit: str = "s") -> dict[str, Any]:
    if is_debug_mode():
        logger.info("Opening native file picker")
    try:
        path = pick_parquet_file()
    except Exception as exc:
        log_error("File picker failed", exc)
        raise HTTPException(500, f"文件选择失败: {exc}") from exc

    if not path:
        if is_debug_mode():
            logger.info("File picker cancelled")
        return {"ok": False, "cancelled": True}

    if is_debug_mode():
        logger.info("Selected file: %s", path)
    info = _inspect_or_http(path, numeric_time_unit)
    save_settings({
        "last_data_file": info["data_file"],
        "numeric_time_unit": numeric_time_unit,
    })
    return {"ok": True, "cancelled": False, **info}


def _strategy_context() -> dict[str, Any]:
    settings = load_settings()
    data_file = settings.get("last_data_file") or ""
    train_symbol = None
    if data_file:
        try:
            train_symbol = inspect_parquet_file(
                data_file, numeric_time_unit=settings.get("numeric_time_unit", "s")
            ).get("symbol")
        except Exception:
            pass

    resolved = resolve_strategy_file(
        settings.get("last_strategy_file") or "",
        train_symbol,
    )
    strategy_info = None
    if resolved:
        try:
            strategy_info = inspect_strategy_file(resolved)
        except Exception as e:
            strategy_info = {
                "strategy_file": resolved,
                "valid": False,
                "message": str(e),
            }
    return {
        "last_strategy_file": resolved,
        "strategy_file": strategy_info,
        "train_symbol": train_symbol,
    }


def _browse_strategy_file() -> dict[str, Any]:
    if is_debug_mode():
        logger.info("Opening strategy file picker")
    try:
        path = pick_strategy_file()
    except Exception as exc:
        log_error("Strategy file picker failed", exc)
        raise HTTPException(500, f"文件选择失败: {exc}") from exc

    if not path:
        if is_debug_mode():
            logger.info("Strategy file picker cancelled")
        return {"ok": False, "cancelled": True}

    if is_debug_mode():
        logger.info("Selected strategy: %s", path)
    info = _inspect_strategy_or_http(path)
    save_settings({"last_strategy_file": info["strategy_file"]})
    return {"ok": True, "cancelled": False, **info}


def _inspect_strategy_or_http(path: str) -> dict[str, Any]:
    try:
        return inspect_strategy_file(path)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e)) from e
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


def _resolve_train_symbol(symbol: str | None = None) -> str | None:
    if symbol:
        return symbol.strip() or None
    settings = load_settings()
    data_file = settings.get("last_data_file") or ""
    if not data_file:
        return None
    try:
        return inspect_parquet_file(
            data_file, numeric_time_unit=settings.get("numeric_time_unit", "s")
        ).get("symbol")
    except Exception:
        return None


def _wait_training_idle(timeout_s: float = 5.0) -> None:
    import time

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        training_manager.status()
        if not training_manager.status().get("active"):
            return
        time.sleep(0.2)


def _sync_and_persist_best_strategy(symbol: str) -> dict[str, Any] | None:
    invalidate_checkpoint_cache()
    info = sync_best_strategy_for_symbol(symbol)
    if info:
        save_settings({"last_strategy_file": info["strategy_file"]})
    return info


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": app.version}


@app.get("/api/routes")
def api_routes() -> dict[str, Any]:
    routes = []
    for route in app.routes:
        methods = getattr(route, "methods", None)
        path = getattr(route, "path", None)
        if path and methods:
            routes.append({"path": path, "methods": sorted(methods)})
    return {"routes": sorted(routes, key=lambda r: r["path"])}


@app.get("/api/debug/logs")
def api_debug_logs(lines: int = 200) -> dict[str, Any]:
    return debug_snapshot(lines)


@app.post("/api/debug/client-log")
def api_client_log(req: ClientLogRequest) -> dict[str, bool]:
    msg = req.message
    if req.context:
        msg = f"{msg} | context={req.context}"
    if req.level == "error":
        log_error(f"[client] {msg}")
    elif is_debug_mode():
        logger.info("[client] %s", msg)
    return {"ok": True}


@app.get("/api/settings")
def api_get_settings() -> dict[str, Any]:
    return load_settings()


@app.put("/api/settings")
def api_put_settings(req: SettingsRequest) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if req.last_data_file is not None:
        payload["last_data_file"] = req.last_data_file
    if req.last_strategy_file is not None:
        payload["last_strategy_file"] = req.last_strategy_file
    if req.numeric_time_unit is not None:
        payload["numeric_time_unit"] = req.numeric_time_unit
    if req.evaluation_workers is not None:
        payload["evaluation_workers"] = req.evaluation_workers
    if req.debug_mode is not None:
        payload["debug_mode"] = req.debug_mode
    if req.ai_provider is not None:
        payload["ai_provider"] = req.ai_provider
    if req.ai_api_key is not None:
        payload["ai_api_key"] = req.ai_api_key
    if req.bt_commission_pct is not None:
        payload["bt_commission_pct"] = req.bt_commission_pct
    if req.bt_slippage_pct is not None:
        payload["bt_slippage_pct"] = req.bt_slippage_pct
    saved = save_settings(payload)
    if req.debug_mode is not None:
        set_debug_mode(req.debug_mode)
    return {"ok": True, **saved}


@app.get("/api/config")
def api_config() -> dict[str, Any]:
    settings = load_settings()
    cpu_config = cpu_tuning_manager.configuration()
    data_file = settings.get("last_data_file") or ""
    file_info = None
    if data_file:
        try:
            file_info = _inspect_parquet_cached(
                data_file, numeric_time_unit=settings.get("numeric_time_unit", "s")
            )
        except Exception as e:
            file_info = {
                "data_file": data_file,
                "valid": False,
                "message": str(e),
            }
    snap = debug_snapshot(1)
    strat_ctx = _strategy_context()
    return {
        "train_steps": ModelConfig.TRAIN_STEPS,
        "batch_size": ModelConfig.BATCH_SIZE,
        "reward_mode": ModelConfig.REWARD_MODE,
        "max_formula_len": ModelConfig.MAX_FORMULA_LEN,
        "device": str(ModelConfig.DEVICE),
        "last_data_file": data_file,
        "numeric_time_unit": settings.get("numeric_time_unit", "s"),
        "evaluation_workers": settings.get(
            "evaluation_workers", ModelConfig.EVALUATION_WORKERS
        ),
        "worker_candidates": cpu_config["candidates"],
        "cpu_hardware": cpu_config["hardware"],
        "data_file": file_info,
        "last_strategy_file": strat_ctx["last_strategy_file"],
        "strategy_file": strat_ctx["strategy_file"],
        "debug_mode": load_settings().get("debug_mode", False),
        "ai_provider": load_settings().get("ai_provider", "deepseek"),
        "ai_api_key": load_settings().get("ai_api_key", ""),
        "bt_commission_pct": settings.get("bt_commission_pct", 0.02),
        "bt_slippage_pct": settings.get("bt_slippage_pct", 0.01),
        "server_log": snap["server_log"],
        "error_log": snap["error_log"],
    }


@app.get("/api/ai/providers")
def api_ai_providers() -> dict[str, Any]:
    from web.ai_providers import provider_status

    status = provider_status()
    settings = load_settings()
    status["selected"] = settings.get("ai_provider", "deepseek")
    status["has_api_key"] = bool(settings.get("ai_api_key"))
    return status


@app.post("/api/ai/analyze-training")
def api_ai_analyze_training(req: AnalyzeTrainingRequest):
    from fastapi.responses import StreamingResponse

    from web.ai_analyze import analyze_training_stream

    settings = load_settings()
    raw_key = req.api_key if req.api_key is not None else settings.get("ai_api_key") or ""
    key_lower = str(raw_key).strip().lower()

    # openclaw_wb 必须先于 openclaw 判断
    if key_lower in ("openclaw_wb",) or key_lower.startswith("openclaw_wb/"):
        provider = "openclaw_wb"
    elif key_lower in ("openclaw",) or key_lower.startswith("openclaw/"):
        provider = "openclaw"
    else:
        provider = (req.provider or settings.get("ai_provider") or "deepseek").strip()

    save_settings({
        "ai_provider": provider,
        "ai_api_key": str(raw_key).strip(),
    })

    def event_gen():
        try:
            for event in analyze_training_stream(
                provider=provider,
                api_key=str(raw_key).strip() or None,
                symbol=req.symbol,
            ):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/data-file/browse")
@app.get("/api/data-file/browse")
def api_browse_data_file(
    numeric_time_unit: Literal["s", "ms", "us", "ns"] = Query("s"),
) -> dict[str, Any]:
    return _browse_data_file(numeric_time_unit)


@app.post("/api/strategy-file/browse")
@app.get("/api/strategy-file/browse")
def api_browse_strategy_file() -> dict[str, Any]:
    return _browse_strategy_file()


@app.post("/api/strategy-file/sync-best")
@app.get("/api/strategy-file/sync-best")
def api_sync_best_strategy(symbol: str | None = None) -> dict[str, Any]:
    sym = _resolve_train_symbol(symbol)
    if not sym:
        raise HTTPException(400, "请先选择训练数据文件或指定品种")
    info = _sync_and_persist_best_strategy(sym)
    if not info:
        return {
            "ok": True,
            "available": False,
            "symbol": sym,
            "strategy_file": None,
        }
    return {"ok": True, "available": True, **info}


def _progress_with_live_step(symbol: str, active: bool) -> dict[str, Any]:
    p = get_symbol_progress(symbol)
    current_step = p.current_step
    if active:
        live = training_manager.parse_step_from_log()
        if live is not None:
            current_step = max(current_step, live)
    train_steps = p.train_steps
    progress_pct = min(100.0, 100.0 * current_step / train_steps) if train_steps > 0 else 0.0
    val_score = None
    hist = p.history or {}
    vals = hist.get("val_score") or []
    if vals:
        try:
            val_score = float(vals[-1])
        except (TypeError, ValueError):
            val_score = None
    return {
        "symbol": p.symbol,
        "current_step": current_step,
        "train_steps": train_steps,
        "progress_pct": round(progress_pct, 1),
        "best_score": p.best_score,
        "val_score": val_score,
        "formula_decoded": p.formula_decoded,
        "status": p.status,
        "history": p.history,
        "has_checkpoint": bool(p.checkpoint_path),
        "has_strategy": p.has_strategy,
    }


def _attach_training_time(
    row: dict[str, Any] | None,
    *,
    symbol: str | None,
    job: dict[str, Any] | None,
    active: bool,
) -> dict[str, Any] | None:
    if not row or not symbol:
        return row
    summary = get_training_time_summary(symbol, job=job, active=active)
    row = dict(row)
    row["session_seconds"] = summary.session_seconds
    row["history_total_seconds"] = summary.history_total_seconds
    return row


@app.get("/api/overview")
def api_overview() -> dict[str, Any]:
    settings = load_settings()
    data_file = settings.get("last_data_file") or ""
    file_info = None
    progress = None

    training = training_manager.status()
    job = training.get("job")
    active = bool(training.get("active"))
    selected_symbol = None

    if data_file:
        try:
            file_info = _inspect_parquet_cached(
                data_file, numeric_time_unit=settings.get("numeric_time_unit", "s")
            )
            selected_symbol = file_info.get("symbol")
        except Exception as e:
            file_info = {"data_file": data_file, "valid": False, "message": str(e)}

    active_symbol = job.get("symbol") if job and active else None
    progress_symbol = active_symbol or selected_symbol
    if progress_symbol:
        is_active_symbol = bool(active_symbol == progress_symbol)
        row = _progress_with_live_step(progress_symbol, active=is_active_symbol)
        progress = {
            "symbol": row["symbol"],
            "status": "running_job" if is_active_symbol else row["status"],
            "current_step": row["current_step"],
            "train_steps": row["train_steps"],
            "progress_pct": row["progress_pct"],
            "best_score": row["best_score"],
            "val_score": row.get("val_score"),
            "formula_decoded": row["formula_decoded"],
            "has_checkpoint": row.get("has_checkpoint", False),
            "has_strategy": row.get("has_strategy", False),
        }
        progress = _attach_training_time(
            progress,
            symbol=progress_symbol,
            job=job,
            active=is_active_symbol,
        )

    return {
        "data_file": file_info,
        "progress": progress,
        "training": training,
    }


@app.get("/api/symbols/{symbol}")
def api_symbol(symbol: str) -> dict[str, Any]:
    p = get_symbol_progress(symbol)
    return {
        "symbol": p.symbol,
        "status": p.status,
        "current_step": p.current_step,
        "train_steps": p.train_steps,
        "progress_pct": round(p.progress_pct, 1),
        "best_score": p.best_score,
        "best_formula": p.best_formula,
        "formula_decoded": p.formula_decoded,
        "has_strategy": p.has_strategy,
        "strategy_score": p.strategy_score,
        "checkpoint_path": p.checkpoint_path,
        "history": p.history,
    }


@app.get("/api/strategies")
def api_strategies() -> dict[str, Any]:
    return {"strategies": list_strategies()}


@app.get("/api/strategies/{symbol}/export")
def api_export_strategy(symbol: str):
    from fastapi.responses import Response

    try:
        body, filename = get_strategy_for_export(symbol)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return Response(
        content=body,
        media_type="application/json; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
        },
    )


@app.get("/api/training/{symbol}/export")
def api_export_training(symbol: str):
    from fastapi.responses import Response

    try:
        body, zip_name = build_training_export_zip(symbol)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return Response(
        content=body,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{zip_name}"'},
    )


@app.post("/api/training/import")
async def api_import_training(
    file: UploadFile = File(...),
    symbol: str | None = Query(None, description="当前选择的品种，用于校验导入包是否一致"),
) -> dict[str, Any]:
    if training_manager.status().get("active"):
        raise HTTPException(409, "训练进行中，请先停止再导入")

    raw = await file.read()
    if not raw:
        raise HTTPException(400, "上传文件为空")

    try:
        return import_training_package(
            raw,
            file.filename or "upload.zip",
            expected_symbol=symbol or None,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/api/training/status")
def api_training_status() -> dict[str, Any]:
    status = training_manager.status()
    status["log_tail"] = training_manager.tail_log(150)
    return status


@app.get("/api/cpu-tuning/status")
def api_cpu_tuning_status() -> dict[str, Any]:
    status = cpu_tuning_manager.status()
    status["log_tail"] = cpu_tuning_manager.tail_log(80)
    return status


@app.post("/api/cpu-tuning/start")
def api_cpu_tuning_start(req: CpuTuningRequest | None = None) -> dict[str, Any]:
    if training_manager.status().get("active"):
        raise HTTPException(409, "训练进行中，请先停止再自动调优")
    if backtest_manager.status().get("active"):
        raise HTTPException(409, "回测进行中，请先停止再自动调优")
    if realtime_manager.status().get("running"):
        raise HTTPException(409, "实时分析进行中，请先停止再自动调优")
    settings = load_settings()
    data_file = (req.data_file if req else None) or settings.get("last_data_file") or None
    numeric_time_unit = (
        (req.numeric_time_unit if req else None)
        or settings.get("numeric_time_unit")
        or "s"
    )
    if data_file:
        data_file = _inspect_or_http(data_file, numeric_time_unit)["data_file"]
    current_workers = int(
        settings.get("evaluation_workers", ModelConfig.EVALUATION_WORKERS)
    )
    try:
        job = cpu_tuning_manager.start(
            data_file=data_file,
            numeric_time_unit=numeric_time_unit,
            current_workers=current_workers,
        )
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(409, str(exc)) from exc
    return {"ok": True, "job": job.to_dict()}


@app.post("/api/training/start")
def api_training_start(req: StartTrainingRequest) -> dict[str, Any]:
    if cpu_tuning_manager.status().get("active"):
        raise HTTPException(409, "CPU 自动调优进行中，请等待完成后再训练")
    info = _inspect_or_http(req.data_file, req.numeric_time_unit)
    settings = load_settings()
    evaluation_workers = (
        req.evaluation_workers
        if req.evaluation_workers is not None
        else int(settings.get("evaluation_workers", ModelConfig.EVALUATION_WORKERS))
    )
    save_settings({
        "last_data_file": info["data_file"],
        "numeric_time_unit": req.numeric_time_unit,
        "evaluation_workers": evaluation_workers,
    })
    try:
        job = training_manager.start(
            data_file=info["data_file"],
            symbol=info["symbol"],
            timeframe=info["timeframe"],
            mode="ftmo",
            from_scratch=bool(req.from_scratch),
            numeric_time_unit=req.numeric_time_unit,
            evaluation_workers=evaluation_workers,
        )
    except RuntimeError as e:
        raise HTTPException(409, str(e)) from e
    if req.from_scratch:
        invalidate_checkpoint_cache()
    return {
        "ok": True,
        "job": job.to_dict(),
        "data_file": info,
        "from_scratch": bool(req.from_scratch),
    }


@app.post("/api/training/stop")
def api_training_stop() -> dict[str, Any]:
    job = training_manager.status().get("job") or {}
    symbol = job.get("symbol")
    stopped = training_manager.stop()
    strategy_file = None
    if symbol:
        _wait_training_idle()
        strategy_file = _sync_and_persist_best_strategy(symbol)
    return {
        "ok": stopped,
        "training": training_manager.status(),
        "strategy_file": strategy_file,
    }


# ─────────────────────────────────────────────────────────────────────
# 回测 API
# ─────────────────────────────────────────────────────────────────────

_METRIC_KEYS = (
    "total_return", "sharpe", "sortino", "profit_loss_ratio",
    "n_trades", "win_rate", "avg_hold_bars",
)


def _load_backtest_report() -> dict[str, Any] | None:
    import json

    report_path = BACKTEST_OUTPUT_DIR / "multi_factor_report.json"
    if not report_path.exists():
        return None
    try:
        return json.loads(report_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _backtest_focus_symbol(symbol: str | None = None) -> str | None:
    """Resolve the symbol used to filter backtest charts/report for the web UI."""
    if symbol:
        return symbol.strip() or None

    job = backtest_manager.status().get("job") or {}
    if job.get("symbol"):
        return str(job["symbol"])

    strat = _strategy_context().get("strategy_file") or {}
    if strat.get("symbol"):
        return str(strat["symbol"])

    report = _load_backtest_report()
    if report:
        keys = list((report.get("symbols") or {}).keys())
        if len(keys) == 1:
            return keys[0]
    return None


def _filter_report_for_symbol(report: dict[str, Any], symbol: str) -> dict[str, Any]:
    symbols = report.get("symbols") or {}
    if symbol not in symbols:
        return report

    sym_data = symbols[symbol]
    return {
        **report,
        "focus_symbol": symbol,
        "symbols": {symbol: sym_data},
        "portfolio": {
            "total_return": sym_data.get("total_return"),
            "sharpe": sym_data.get("sharpe"),
            "sortino": sym_data.get("sortino"),
            "profit_loss_ratio": sym_data.get("profit_loss_ratio"),
            "n_trades": sym_data.get("n_trades"),
            "win_rate": sym_data.get("win_rate"),
        },
    }


def _list_backtest_charts(symbol: str | None = None) -> list[dict[str, str]]:
    """列出回测输出目录下的图表；单品种模式只返回该品种相关文件。"""
    if not BACKTEST_OUTPUT_DIR.exists():
        return []

    if symbol:
        charts: list[dict[str, str]] = []
        equity = BACKTEST_OUTPUT_DIR / "portfolio_equity.png"
        if equity.exists():
            charts.append(
                {"name": equity.name, "label": f"{symbol} 资金曲线", "kind": "equity"}
            )
        return charts

    charts = []
    portfolio = BACKTEST_OUTPUT_DIR / "portfolio_equity.png"
    if portfolio.exists():
        charts.append({"name": "portfolio_equity.png", "label": "组合资金曲线", "kind": "portfolio"})
    for path in sorted(BACKTEST_OUTPUT_DIR.glob("equity_*.png")):
        sym = path.stem.replace("equity_", "", 1)
        charts.append({"name": path.name, "label": f"{sym} 资金曲线", "kind": "symbol"})
    return charts


@app.get("/api/backtest/status")
def api_backtest_status() -> dict[str, Any]:
    status = backtest_manager.status()
    status["log_tail"] = backtest_manager.tail_log(200)
    return status


@app.post("/api/backtest/start")
def api_backtest_start(req: StartBacktestRequest) -> dict[str, Any]:
    if cpu_tuning_manager.status().get("active"):
        raise HTTPException(409, "CPU 自动调优进行中，请等待完成后再回测")
    info = _inspect_strategy_or_http(req.strategy_file)
    settings = load_settings()
    commission = (
        float(req.commission_pct)
        if req.commission_pct is not None
        else float(settings.get("bt_commission_pct", 0.02))
    )
    slippage = (
        float(req.slippage_pct)
        if req.slippage_pct is not None
        else float(settings.get("bt_slippage_pct", 0.01))
    )
    if commission < 0 or slippage < 0:
        raise HTTPException(400, "手续费和滑点不能为负数")

    save_settings({
        "last_strategy_file": info["strategy_file"],
        "numeric_time_unit": req.numeric_time_unit,
        "bt_commission_pct": commission,
        "bt_slippage_pct": slippage,
    })

    try:
        pf = inspect_parquet_file(
            req.data_file, numeric_time_unit=req.numeric_time_unit
        )
        if pf.get("valid") is False:
            raise HTTPException(400, pf.get("message") or "回测数据文件无效")
        data_file = pf["data_file"]
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(400, f"回测数据文件无法加载: {exc}") from exc

    try:
        job = backtest_manager.start(
            strategy_file=info["strategy_file"],
            data_file=data_file,
            mode=req.mode,
            commission_pct=commission,
            slippage_pct=slippage,
            numeric_time_unit=req.numeric_time_unit,
        )
    except RuntimeError as e:
        raise HTTPException(409, str(e)) from e
    return {"ok": True, "job": job.to_dict(), "strategy_file": info, "data_file": data_file}


@app.post("/api/backtest/stop")
def api_backtest_stop() -> dict[str, Any]:
    stopped = backtest_manager.stop()
    return {"ok": stopped, "backtest": backtest_manager.status()}


@app.get("/api/backtest/report")
def api_backtest_report(symbol: str | None = None) -> dict[str, Any]:
    report = _load_backtest_report()
    focus = _backtest_focus_symbol(symbol)
    if report and focus:
        report = _filter_report_for_symbol(report, focus)
    return {
        "available": report is not None,
        "report": report,
        "charts": _list_backtest_charts(focus),
        "focus_symbol": focus,
    }


@app.get("/api/backtest/equity")
def api_backtest_equity(symbol: str | None = None) -> dict[str, Any]:
    """资金曲线原始数据（供前端渲染交互式 HTML 图表）。"""
    import json

    path = BACKTEST_OUTPUT_DIR / "equity_curve.json"
    focus = _backtest_focus_symbol(symbol)
    if not path.exists():
        return {"available": False, "focus_symbol": focus, "data": None}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"available": False, "focus_symbol": focus, "data": None}

    # 单品种模式：只保留聚焦品种，去掉无关序列
    if focus and isinstance(data.get("symbols"), dict) and focus in data["symbols"]:
        data = {
            **data,
            "symbols": {focus: data["symbols"][focus]},
        }
        data.pop("portfolio", None)

    return {"available": True, "focus_symbol": focus, "data": data}


@app.get("/api/backtest/chart/{name}")
def api_backtest_chart(name: str):
    # 防止路径穿越：仅允许输出目录内的 png 文件
    if "/" in name or "\\" in name or ".." in name or not name.lower().endswith(".png"):
        raise HTTPException(400, "非法文件名")
    path = (BACKTEST_OUTPUT_DIR / name).resolve()
    try:
        path.relative_to(BACKTEST_OUTPUT_DIR.resolve())
    except ValueError:
        raise HTTPException(400, "非法路径") from None
    if not path.exists():
        raise HTTPException(404, "图表不存在")
    return FileResponse(path, media_type="image/png")


# ─────────────────────────────────────────────────────────────────────
# 实时行情分析 API
# ─────────────────────────────────────────────────────────────────────


@app.on_event("startup")
def _startup_realtime() -> None:
    try:
        realtime_manager.load_persisted()
    except Exception as exc:  # noqa: BLE001
        log_error("realtime load_persisted failed", exc)
    try:
        trading_preview_manager.load_persisted()
    except Exception as exc:  # noqa: BLE001
        log_error("trading preview load_persisted failed", exc)
    try:
        trading_execution_controller.start_management()
    except Exception as exc:  # noqa: BLE001
        log_error("trading execution management startup failed", exc)


@app.on_event("shutdown")
def _shutdown_trading_preview() -> None:
    trading_execution_controller.disable()
    trading_execution_controller.stop_management()
    trading_preview_manager.stop()


@app.get("/api/realtime/sources")
def api_realtime_sources() -> dict[str, Any]:
    return {"sources": list_sources(), "min_exposure": min_exposure()}


@app.get("/api/realtime/strategies")
def api_realtime_strategies() -> dict[str, Any]:
    """已保存的 identity-valid immutable V2 策略，供因子来源下拉。"""
    selected: dict[tuple[str, str], dict[str, Any]] = {}
    for s in list_strategies():
        sym = s.get("symbol")
        timeframe = s.get("timeframe")
        filename = s.get("file")
        if not sym or not timeframe or not filename:
            continue
        try:
            path = strategy_path_for_symbol(
                sym, timeframe, filename=filename
            )
            inspected = inspect_strategy_file(str(path))
        except Exception:
            continue
        row = {
            "symbol": inspected["symbol"],
            "timeframe": inspected["timeframe"],
            "best_score": inspected["best_score"],
            "formula_decoded": inspected["formula_decoded"],
            "strategy_file": inspected["strategy_file"],
            "generated_at": inspected["generated_at"],
            "filename": inspected["filename"],
        }
        key = (row["symbol"], row["timeframe"])
        previous = selected.get(key)
        if previous is None or (
            generated_at_utc(row["generated_at"]), row["filename"]
        ) > (generated_at_utc(previous["generated_at"]), previous["filename"]):
            selected[key] = row
    rows = [selected[key] for key in sorted(selected)]
    return {"strategies": rows}


@app.get("/api/realtime/status")
def api_realtime_status() -> dict[str, Any]:
    return realtime_manager.status()


@app.post("/api/realtime/watch")
def api_realtime_watch(req: AddWatchRequest) -> dict[str, Any]:
    try:
        watch = realtime_manager.add_watch(
            req.source, req.symbol, req.timeframe, req.strategy_file
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"ok": True, "watch": watch}


@app.post("/api/realtime/unwatch")
def api_realtime_unwatch(req: RemoveWatchRequest) -> dict[str, Any]:
    removed = realtime_manager.remove_watch(req.id)
    return {"ok": removed}


@app.post("/api/realtime/start")
def api_realtime_start() -> dict[str, Any]:
    if cpu_tuning_manager.status().get("active"):
        raise HTTPException(409, "CPU 自动调优进行中，请等待完成后再启动实时分析")
    realtime_manager.start()
    return {"ok": True, **realtime_manager.status()}


@app.post("/api/realtime/stop")
def api_realtime_stop() -> dict[str, Any]:
    realtime_manager.stop()
    return {"ok": True, "running": False}


# ─────────────────────────────────────────────────────────────────────
# PA + Alpha 只读交易预览 API（阶段 4：绝不发送订单）
# ─────────────────────────────────────────────────────────────────────


_TRADING_MODULE_LABELS = {
    "alpha_h1": "H1 Alpha 因子",
    "pa_h1": "H1 PA 结构",
    "alpha_m15": "M15 Alpha 因子",
    "pa_m15_context": "M15 PA 环境",
    "pa_m15_pattern": "M15 PA 形态",
    "pa_m5_timing": "M5 PA 时机",
    "alpha_m5": "M5 Alpha 因子（默认关闭）",
}
_TRADING_FAMILY_LABELS = {
    "trend_continuation": "趋势延续",
    "breakout": "突破/回踩",
    "reversal": "反转",
    "range": "区间交易",
}


def _trading_config_response() -> dict[str, Any]:
    state = trading_preview_manager.config_state()
    return {
        **state,
        "catalog": {
            "modes": [
                {"id": "rules", "label": "规则模式"},
                {"id": "rules_codex", "label": "规则 + Codex 审查"},
            ],
            "direction_modules": [
                {"id": module, "label": _TRADING_MODULE_LABELS[module]}
                for module in DIRECTION_MODULE_IDS
            ],
            "entry_modules": [
                {"id": module, "label": _TRADING_MODULE_LABELS[module]}
                for module in ENTRY_MODULE_IDS
            ],
            "pa_families": [
                {"id": family, "label": _TRADING_FAMILY_LABELS[family]}
                for family in PA_FAMILY_IDS
            ],
            "supported_modes": list(TRADING_MODES),
        },
    }


@app.get("/api/trading/account")
def api_trading_account() -> dict[str, Any]:
    snapshot = mt5_read_only_market.account_snapshot()
    execution = trading_execution_controller.status()
    original_message = str(snapshot.get("message") or "")
    execution_message = (
        "实盘新开仓已手动启用"
        if execution["execution_enabled"]
        else "实盘新开仓默认关闭；已有 AlphaMaster 仓位仍会继续管理"
    )
    return {
        **snapshot,
        "read_only": False,
        "execution_enabled": execution["execution_enabled"],
        "management_enabled": execution["management_enabled"],
        "message": (
            execution_message
            if snapshot.get("connected")
            else f"{original_message}；{execution_message}"
        ),
    }


@app.get("/api/trading/config")
def api_trading_config_get() -> dict[str, Any]:
    return _trading_config_response()


@app.put("/api/trading/config")
def api_trading_config_put(req: TradingConfigRequest) -> dict[str, Any]:
    try:
        config = TradingConfigV1(
            mode=req.mode,
            direction_module_ids=tuple(req.direction_module_ids),
            entry_module_ids=tuple(req.entry_module_ids),
            pa_family_ids=tuple(req.pa_family_ids),
            tp1_lots=req.tp1_lots,
            tp2_lots=req.tp2_lots,
            symbol=req.symbol,
            config_version=req.config_version,
        )
        trading_preview_manager.update_config(config)
    except (TypeError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"ok": True, **_trading_config_response()}


@app.get("/api/trading/status")
def api_trading_status() -> dict[str, Any]:
    preview = trading_preview_manager.status()
    execution = trading_execution_controller.status()
    return {
        **preview,
        "stage": 5,
        "read_only": False,
        "execution_enabled": execution["execution_enabled"],
        "execution": execution,
    }


@app.get("/api/trading/decisions")
def api_trading_decisions(limit: int = Query(default=20, ge=1, le=100)) -> dict[str, Any]:
    return {"decisions": trading_preview_manager.decisions(limit=limit)}


@app.post("/api/trading/preview/start")
def api_trading_preview_start() -> dict[str, Any]:
    trading_preview_manager.start()
    return {"ok": True, **api_trading_status()}


@app.post("/api/trading/preview/stop")
def api_trading_preview_stop() -> dict[str, Any]:
    # Always run the disable path: entries may already be off while persisted
    # AlphaMaster pending orders still need an explicit cancellation attempt.
    trading_execution_controller.disable()
    trading_preview_manager.stop()
    return {"ok": True, **api_trading_status()}


@app.post("/api/trading/enable")
def api_trading_enable() -> dict[str, Any]:
    config = TradingConfigV1.from_payload(
        trading_preview_manager.config_state()["config"]
    )
    execution = trading_execution_controller.enable(config)
    if not execution["execution_enabled"]:
        raise HTTPException(
            409,
            execution["last_error"] or execution["blocker"] or "实盘启用失败",
        )
    trading_preview_manager.start()
    return {"ok": True, **api_trading_status()}


@app.post("/api/trading/disable")
def api_trading_disable() -> dict[str, Any]:
    trading_execution_controller.disable()
    return {"ok": True, **api_trading_status()}


@app.get("/api/realtime/feishu")
def api_realtime_feishu_get() -> dict[str, Any]:
    s = load_settings()
    return {
        "enabled": bool(s.get("feishu_enabled")),
        "webhook_url": s.get("feishu_webhook_url") or "",
        "secret": s.get("feishu_secret") or "",
    }


@app.put("/api/realtime/feishu")
def api_realtime_feishu_put(req: FeishuSettingsRequest) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if req.enabled is not None:
        payload["feishu_enabled"] = bool(req.enabled)
    if req.webhook_url is not None:
        payload["feishu_webhook_url"] = req.webhook_url
    if req.secret is not None:
        payload["feishu_secret"] = req.secret
    saved = save_settings(payload)
    return {
        "ok": True,
        "enabled": bool(saved.get("feishu_enabled")),
        "webhook_url": saved.get("feishu_webhook_url") or "",
        "secret": saved.get("feishu_secret") or "",
    }


@app.post("/api/realtime/feishu/test")
def api_realtime_feishu_test(req: FeishuTestRequest) -> dict[str, Any]:
    from web.feishu_notify import send_text

    url = (req.webhook_url or "").strip()
    if not url:
        url = (load_settings().get("feishu_webhook_url") or "").strip()
    if not url:
        raise HTTPException(400, "请先填写 Webhook URL")
    secret = req.secret
    if secret is None:
        secret = load_settings().get("feishu_secret") or ""
    ok, msg = send_text(
        "✅ AlphaMaster 飞书通知测试：配置正常。信号方向转折时会推送提醒。",
        webhook_url=url,
        secret=secret or "",
    )
    if not ok:
        raise HTTPException(400, msg)
    return {"ok": True, "message": msg}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
