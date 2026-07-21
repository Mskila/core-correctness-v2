"""Strict V2 strategy inspection for Web entrypoints."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from model_core.artifacts import StrategyArtifact
from model_core.semantics import STRATEGY_SCHEMA_VERSION
from data_pipeline.validation import normalize_timeframe_name
from web.progress import STRATEGIES_DIR, generated_at_utc


def _load(path: Path) -> StrategyArtifact:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法解析 V2 strategy artifact: {exc}") from exc
    try:
        artifact = StrategyArtifact.from_dict(value)
        if artifact.schema_version != STRATEGY_SCHEMA_VERSION:
            raise ValueError(
                "pre-core-fix/incompatible strategy artifact: "
                f"expected={STRATEGY_SCHEMA_VERSION!r} actual={artifact.schema_version!r}"
            )
        return artifact
    except Exception as exc:
        raise ValueError(f"不兼容的 V2 strategy artifact: {exc}") from exc


def inspect_strategy_file(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"文件不存在: {p}")
    artifact = _load(p)
    if p.name != artifact.run_identity.strategy_filename():
        raise ValueError("noncanonical immutable V2 strategy filename")
    identity = artifact.run_identity.artifact_identity
    dataset = identity.training_dataset
    return {
        "strategy_file": str(p.resolve()),
        "filename": p.name,
        "symbol": identity.symbol,
        "timeframe": identity.timeframe,
        "run_id": artifact.run_identity.run_id,
        "artifact_fingerprint": identity.fingerprint,
        "strategy_fingerprint": artifact.fingerprint,
        "data_hash": dataset.data_fingerprint,
        "data_start_time_ns": dataset.start_time_ns,
        "data_end_time_ns": dataset.end_time_ns,
        "core_semantics_version": identity.core_semantics_version,
        "vocab_version": identity.vocab_version,
        "formula_decoded": artifact.decoded_formula,
        "best_score": artifact.best_score,
        "generated_at": artifact.generated_at,
        "source_path": None,
        "valid": True,
        "message": "",
    }


def resolve_strategy_file(saved_path: str, train_symbol: str | None = None) -> str:
    if saved_path and Path(saved_path).exists():
        inspect_strategy_file(saved_path)
        return str(Path(saved_path).resolve())
    return saved_path or ""


def strategy_path_for_symbol(
    symbol: str,
    timeframe: str,
    *,
    filename: str | None = None,
) -> Path:
    expected_timeframe = normalize_timeframe_name(timeframe)
    if filename is not None:
        path = STRATEGIES_DIR / Path(filename).name
        row = inspect_strategy_file(str(path))
        if row["symbol"] != symbol or row["timeframe"] != expected_timeframe:
            raise ValueError("strategy row identity mismatch")
        return path
    candidates = [
        path for path in STRATEGIES_DIR.glob("best_v3_*.json")
        if _candidate(path, symbol, expected_timeframe)
    ]
    if not candidates:
        raise FileNotFoundError(
            f"no identity-valid V2 strategy for {symbol}/{expected_timeframe}"
        )
    return max(candidates, key=_sort_key)


def _candidate(path: Path, symbol: str, timeframe: str | None = None) -> bool:
    try:
        row = inspect_strategy_file(str(path))
        return row["symbol"] == symbol and (
            timeframe is None or row["timeframe"] == timeframe
        )
    except Exception:
        return False


def _sort_key(path: Path) -> tuple[object, str]:
    artifact = _load(path)
    return (generated_at_utc(artifact.generated_at), path.name)


def sync_best_strategy_for_symbol(symbol: str) -> dict[str, Any] | None:
    """Return the deterministic latest valid immutable V2 artifact; never write."""
    candidates = [p for p in STRATEGIES_DIR.glob("best_v3_*.json") if _candidate(p, symbol)]
    if not candidates:
        return None
    return inspect_strategy_file(str(max(candidates, key=_sort_key)))
