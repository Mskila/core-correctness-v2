"""Read-only progress recovery for a running V2 training service.

This module is intentionally separate from :mod:`web.progress`.  It accepts
both checkpoint step conventions that have existed in V2 artifacts while
keeping the normal Web/export validators strict.
"""
from __future__ import annotations

import copy
import io
import json
from pathlib import Path
from typing import Any

import torch

from model_core.artifacts import StrategyArtifact, TrainingRunIdentity
from model_core.config import ModelConfig
from model_core.vocab import FORMULA_VOCAB
from web.progress import SymbolProgress, _safe_symbol_tag, _validate_history


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
STRATEGIES_DIR = PROJECT_ROOT / "strategies"

_checkpoint_cache: dict[str, tuple[int, int, dict[str, Any]]] = {}


def invalidate_checkpoint_cache() -> None:
    _checkpoint_cache.clear()


def _checkpoint_paths(symbol: str) -> list[Path]:
    return sorted(
        CHECKPOINT_DIR.glob(
            f"ckpt_v2_{_safe_symbol_tag(symbol)}_*_run_*_step_*.pt"
        ),
        key=lambda path: (path.stat().st_mtime_ns, path.name),
    )


def _completed_step_count(history: dict[str, Any], checkpoint_step: int) -> int:
    """Return a display count for either persisted checkpoint convention."""
    steps = history["step"]
    if not steps:
        if checkpoint_step == 0:
            return 0
        raise ValueError("checkpoint has no persisted history for its step")

    last_history_step = steps[-1]
    if last_history_step + 1 == checkpoint_step:
        # The checkpoint stores the next step to execute.
        return checkpoint_step
    if last_history_step == checkpoint_step:
        # The checkpoint stores the most recently completed zero-based step.
        return checkpoint_step + 1
    raise ValueError(
        "history last persisted step matches neither checkpoint step convention"
    )


def _load_checkpoint(path: Path) -> dict[str, Any]:
    before = path.stat()
    cache_key = str(path.resolve())
    cached = _checkpoint_cache.get(cache_key)
    if cached and cached[0] == before.st_mtime_ns and cached[1] == before.st_size:
        return copy.deepcopy(cached[2])

    raw = path.read_bytes()
    after = path.stat()
    if (
        before.st_mtime_ns != after.st_mtime_ns
        or before.st_size != after.st_size
    ):
        raise ValueError("checkpoint changed while being read")
    payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
    if type(payload) is not dict:
        raise ValueError("checkpoint payload must be an object")
    if payload.get("checkpoint_schema_version") != "checkpoint-v2":
        raise ValueError("incompatible checkpoint schema")

    run = TrainingRunIdentity.from_dict(payload["run_identity"])
    step = payload["step"]
    if type(step) is not int:
        raise ValueError("checkpoint step must be an exact built-in integer")
    if step < 0 or step > ModelConfig.TRAIN_STEPS:
        raise ValueError("checkpoint step is outside configured training bounds")
    if path.name != run.checkpoint_filename(step):
        raise ValueError("checkpoint filename and embedded identity/step disagree")

    history = payload["training_history"]
    _validate_history(history, run, require_run_identity=False)
    completed_steps = _completed_step_count(history, step)
    rank_monitor = payload["rank_monitor_history"]
    if type(rank_monitor) is not list or rank_monitor != history["stable_rank"]:
        raise ValueError(
            "checkpoint rank_monitor_history does not match embedded history"
        )

    result = {
        "run": run,
        "raw_step": step,
        "completed_steps": min(completed_steps, ModelConfig.TRAIN_STEPS),
        "best_score": payload.get("best_score"),
        "best_formula": payload.get("best_formula"),
        "training_history": history,
        "mtime": after.st_mtime,
        "mtime_ns": after.st_mtime_ns,
        "path": path,
    }
    _checkpoint_cache[cache_key] = (after.st_mtime_ns, after.st_size, result)
    return copy.deepcopy(result)


def _strategy_rows(symbol: str) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    prefix = f"best_v2_{_safe_symbol_tag(symbol)}_"
    for path in STRATEGIES_DIR.glob(f"{prefix}*.json"):
        try:
            artifact = StrategyArtifact.from_dict(
                json.loads(path.read_text(encoding="utf-8"))
            )
            run = artifact.run_identity
            if run.artifact_identity.symbol != symbol:
                continue
            if path.name != run.strategy_filename():
                raise ValueError("noncanonical strategy filename")
            rows.append({"run": run, "artifact": artifact, "path": path})
        except Exception as exc:
            errors.append(f"{path.name}: {exc}")
    return rows, errors


def _decode_formula(tokens: list[int] | None) -> str | None:
    if not tokens:
        return None
    try:
        return " → ".join(FORMULA_VOCAB.token_names[token] for token in tokens)
    except (IndexError, TypeError):
        return str(tokens)


def get_symbol_progress(symbol: str) -> SymbolProgress:
    """Build progress only from identity-bearing checkpoints and strategies.

    Standalone history JSON files are deliberately not opened.  Their bytes are
    neither trusted nor changed; the checkpoint's identity-validated embedded
    history is the sole curve source.
    """
    checkpoint_rows: list[dict[str, Any]] = []
    reasons: list[str] = []
    for path in _checkpoint_paths(symbol):
        try:
            row = _load_checkpoint(path)
            if row["run"].artifact_identity.symbol != symbol:
                raise ValueError("checkpoint symbol mismatch")
            checkpoint_rows.append(row)
        except Exception as exc:
            reasons.append(f"{path.name}: {exc}")

    strategy_rows, strategy_errors = _strategy_rows(symbol)
    reasons.extend(strategy_errors)
    checkpoint = max(
        checkpoint_rows,
        key=lambda row: (
            row["completed_steps"], row["mtime_ns"], row["path"].name
        ),
        default=None,
    )
    anchor = checkpoint["run"] if checkpoint else None

    if anchor is None and strategy_rows:
        strategy = max(
            strategy_rows,
            key=lambda row: (row["path"].stat().st_mtime_ns, row["path"].name),
        )
        anchor = strategy["run"]
    else:
        matching = [row for row in strategy_rows if row["run"] == anchor]
        strategy = max(
            matching,
            key=lambda row: (row["path"].stat().st_mtime_ns, row["path"].name),
            default=None,
        )

    if anchor is None:
        return SymbolProgress(
            symbol=symbol,
            train_steps=ModelConfig.TRAIN_STEPS,
            current_step=0,
            best_score=None,
            best_formula=None,
            formula_decoded=None,
            has_strategy=False,
            strategy_score=None,
            checkpoint_path=None,
            checkpoint_mtime=None,
            history=None,
            incompatible_reasons=tuple(reasons),
        )

    if strategy_rows and strategy is None:
        reasons.append("strategy belongs to a different run/artifact fingerprint")
    best_score = checkpoint.get("best_score") if checkpoint else None
    best_formula = checkpoint.get("best_formula") if checkpoint else None
    if strategy is not None:
        artifact = strategy["artifact"]
        if best_score is None:
            best_score = artifact.best_score
        if best_formula is None:
            best_formula = list(artifact.formula_tokens)

    return SymbolProgress(
        symbol=symbol,
        train_steps=ModelConfig.TRAIN_STEPS,
        current_step=(0 if reasons else (
            checkpoint["completed_steps"] if checkpoint else 0
        )),
        best_score=best_score,
        best_formula=best_formula,
        formula_decoded=_decode_formula(best_formula),
        has_strategy=strategy is not None,
        strategy_score=(strategy["artifact"].best_score if strategy else None),
        checkpoint_path=(
            str(checkpoint["path"].relative_to(PROJECT_ROOT)).replace("\\", "/")
            if checkpoint else None
        ),
        checkpoint_mtime=(checkpoint["mtime"] if checkpoint else None),
        history=(checkpoint["training_history"] if checkpoint and not reasons else None),
        run_id=anchor.run_id,
        artifact_fingerprint=anchor.artifact_identity.fingerprint,
        incompatible_reasons=tuple(reasons),
    )
