"""Identity-isolated V2 training progress for Web entrypoints."""
from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any

import torch

from model_core.artifacts import StrategyArtifact, TrainingRunIdentity
from model_core.config import ModelConfig
from model_core.semantics import CHECKPOINT_SCHEMA_VERSION, STRATEGY_SCHEMA_VERSION
from model_core.vocab import FORMULA_VOCAB
from model_core.vm import FormulaErrorKind

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
STRATEGIES_DIR = PROJECT_ROOT / "strategies"


def generated_at_utc(value: str) -> tuple[datetime, Fraction]:
    if type(value) is not str:
        raise ValueError("generated_at ordering requires an exact string")
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError("generated_at ordering requires a valid UTC timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValueError("generated_at ordering requires an aware UTC timestamp")
    body = value[:-1] if value.endswith("Z") else value
    if not value.endswith("Z"):
        zone_start = max(body.rfind("+", 10), body.rfind("-", 10))
        if zone_start < 0:
            raise ValueError("generated_at ordering requires an explicit UTC offset")
        body = body[:zone_start]
    separator = max(body.rfind("."), body.rfind(","))
    fraction = Fraction(0)
    if separator >= 0:
        digits = body[separator + 1 :]
        if not digits or not digits.isdigit():
            raise ValueError("generated_at ordering requires decimal fractional digits")
        fraction = Fraction(int(digits), 10 ** len(digits))
    whole_second = parsed.astimezone(timezone.utc).replace(microsecond=0)
    return whole_second, fraction


def _safe_symbol_tag(symbol: str) -> str:
    return symbol.replace(".", "_")


def checkpoint_glob(symbol: str) -> list[Path]:
    return sorted(
        CHECKPOINT_DIR.glob(f"ckpt_v3_{_safe_symbol_tag(symbol)}_*_run_*_step_*.pt"),
        key=lambda path: (path.stat().st_mtime_ns, path.name),
    )


def _step_from_name(path: Path) -> int:
    match = re.search(r"_step_(\d+)\.pt$", path.name)
    return int(match.group(1)) if match else 0


@dataclass
class SymbolProgress:
    symbol: str
    train_steps: int
    current_step: int
    best_score: float | None
    best_formula: list[int] | None
    formula_decoded: str | None
    has_strategy: bool
    strategy_score: float | None
    checkpoint_path: str | None
    checkpoint_mtime: float | None
    history: dict[str, Any] | None
    run_id: str | None = None
    artifact_fingerprint: str | None = None
    incompatible_reasons: tuple[str, ...] = ()

    @property
    def progress_pct(self) -> float:
        return 0.0 if self.train_steps <= 0 else min(100.0, 100.0 * self.current_step / self.train_steps)

    @property
    def status(self) -> str:
        if self.incompatible_reasons:
            return "incompatible"
        if self.current_step >= self.train_steps and self.has_strategy:
            return "completed"
        if self.current_step > 0:
            return "in_progress"
        if self.has_strategy:
            return "strategy_only"
        return "idle"


_ckpt_cache: dict[str, tuple[int, str, dict[str, Any]]] = {}

_HISTORY_METRICS = (
    "avg_reward", "best_score", "val_score", "entropy", "ic_mean",
    "ic_stability", "sortino", "elite_pool_size", "init_entropy",
    "kl_uniform", "kl_prev", "top1_prob", "eff_vocab",
    "batch_uniq_tokens", "batch_uniq_fmls", "batch_fml_div",
)
_HISTORY_STRUCTURED_METRICS = ("formula_error_counts", "formula_error_samples")


def invalidate_checkpoint_cache() -> None:
    _ckpt_cache.clear()


def _load_checkpoint_meta(path: Path) -> dict[str, Any]:
    stat_before = path.stat()
    checkpoint_bytes = path.read_bytes()
    payload = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=False)
    stat_after = path.stat()
    if (
        stat_before.st_mtime_ns != stat_after.st_mtime_ns
        or stat_before.st_size != stat_after.st_size
    ):
        raise ValueError("checkpoint changed while being validated")
    mtime = stat_after.st_mtime
    mtime_ns = stat_after.st_mtime_ns
    if type(payload) is not dict:
        raise ValueError("checkpoint payload must be an object")
    if payload.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("pre-core-fix/incompatible checkpoint schema")
    run = TrainingRunIdentity.from_dict(payload["run_identity"])
    step = payload["step"]
    if type(step) is not int:
        raise ValueError("checkpoint step must be an exact built-in integer")
    if step < 0 or step > ModelConfig.TRAIN_STEPS:
        raise ValueError("checkpoint step is outside configured training bounds")
    if path.name != run.checkpoint_filename(step):
        raise ValueError(
            f"noncanonical checkpoint filename: expected={run.checkpoint_filename(step)} actual={path.name}"
        )
    history = payload["training_history"]
    _validate_history(history, run, require_run_identity=False)
    _validate_history_checkpoint_relationship(history, step)
    rank_monitor = payload["rank_monitor_history"]
    if type(rank_monitor) is not list or rank_monitor != history["stable_rank"]:
        raise ValueError("checkpoint rank_monitor_history does not match training_history stable_rank")
    key = f"{path.resolve()}:{run.artifact_identity.fingerprint}"
    content_hash = hashlib.sha256(checkpoint_bytes).hexdigest()
    cached = _ckpt_cache.get(key)
    if cached and cached[0] == mtime_ns and cached[1] == content_hash:
        return copy.deepcopy(cached[2])
    meta = {
        "run": run, "step": step, "best_score": payload.get("best_score"),
        "best_formula": payload.get("best_formula"),
        "training_history": history,
        "mtime": mtime, "mtime_ns": mtime_ns, "path": path,
    }
    _ckpt_cache[key] = (mtime_ns, content_hash, meta)
    return copy.deepcopy(meta)


def _decode_formula(tokens: list[int] | None) -> str | None:
    if not tokens:
        return None
    try:
        return " → ".join(FORMULA_VOCAB.token_names[token] for token in tokens)
    except (IndexError, TypeError):
        return str(tokens)


def _strategy_rows(symbol: str) -> tuple[list[dict[str, Any]], list[str]]:
    rows, errors = [], []
    prefix = f"best_v3_{_safe_symbol_tag(symbol)}_"
    for path in STRATEGIES_DIR.glob(f"{prefix}*.json"):
        try:
            raw = path.read_bytes()
            artifact = StrategyArtifact.from_dict(json.loads(raw.decode("utf-8")))
            if artifact.schema_version != STRATEGY_SCHEMA_VERSION:
                raise ValueError(
                    "pre-core-fix/incompatible strategy artifact: "
                    f"expected={STRATEGY_SCHEMA_VERSION!r} actual={artifact.schema_version!r}"
                )
            run = artifact.run_identity
            if run.artifact_identity.symbol != symbol:
                continue
            if path.name != run.strategy_filename():
                raise ValueError("noncanonical strategy filename")
            rows.append({"run": run, "artifact": artifact, "path": path, "raw": raw})
        except Exception as exc:
            errors.append(f"{path.name}: {exc}")
    return rows, errors


def _validate_history(
    value: Any,
    run: TrainingRunIdentity,
    *,
    require_run_identity: bool,
) -> None:
    if type(value) is not dict:
        raise ValueError("history must be an object")
    expected_fields = (
        set(_HISTORY_METRICS)
        | set(_HISTORY_STRUCTURED_METRICS)
        | {"step", "stable_rank"}
    )
    if require_run_identity:
        expected_fields.add("run_identity")
    actual_fields = set(value)
    if actual_fields != expected_fields:
        missing = sorted(expected_fields - actual_fields)
        unknown = sorted(actual_fields - expected_fields)
        raise ValueError(f"history fields are invalid: missing={missing} unknown={unknown}")
    if require_run_identity:
        history_run = TrainingRunIdentity.from_dict(value["run_identity"])
        if history_run != run:
            raise ValueError("history belongs to a different run/artifact fingerprint")
    steps = value["step"]
    if type(steps) is not list:
        raise ValueError("history step must be a list")
    if any(type(step) is not int for step in steps):
        raise ValueError("history steps must be exact integers")
    if any(step < 0 or step >= ModelConfig.TRAIN_STEPS for step in steps):
        raise ValueError("history step is outside configured training bounds")
    if any(current <= previous for previous, current in zip(steps, steps[1:])):
        raise ValueError("history steps must be strictly increasing")
    for name in _HISTORY_METRICS:
        values = value[name]
        if type(values) is not list:
            raise ValueError(f"history field {name} must be a list")
        if len(values) != len(steps):
            raise ValueError(f"history field {name} length does not match step")
        for item in values:
            if type(item) not in (int, float):
                raise ValueError(f"history field {name} must contain numeric values")
            if not math.isfinite(float(item)):
                raise ValueError(f"history field {name} contains a non-finite value")
    error_counts = value["formula_error_counts"]
    error_samples = value["formula_error_samples"]
    if type(error_counts) is not list or len(error_counts) != len(steps):
        raise ValueError("history field formula_error_counts length does not match step")
    if type(error_samples) is not list or len(error_samples) != len(steps):
        raise ValueError("history field formula_error_samples length does not match step")
    expected_error_kinds = {kind.value for kind in FormulaErrorKind}
    for counts in error_counts:
        if type(counts) is not dict or set(counts) != expected_error_kinds:
            raise ValueError("history formula_error_counts keys are invalid")
        if any(type(count) is not int or count < 0 for count in counts.values()):
            raise ValueError("history formula_error_counts values are invalid")
    for samples in error_samples:
        if type(samples) is not list or len(samples) > 5:
            raise ValueError("history formula_error_samples entry is invalid")
        if any(type(sample) is not str or len(sample) > 500 for sample in samples):
            raise ValueError("history formula_error_samples values are invalid")
    stable_rank = value["stable_rank"]
    if type(stable_rank) is not list:
        raise ValueError("history stable_rank must be a list")
    for item in stable_rank:
        if type(item) not in (int, float) or not math.isfinite(float(item)) or item < 0:
            raise ValueError("history stable_rank must contain finite nonnegative exact numbers")
    lord = run.artifact_identity.training_config["lord"]
    use_lord = lord["use_lord_regularization"]
    expected_count = sum(step % 10 == 0 for step in steps) if use_lord else 0
    if len(stable_rank) != expected_count:
        raise ValueError(
            "history stable_rank cardinality does not match persisted LORD cadence"
        )


def _validate_history_checkpoint_relationship(value: dict[str, Any], step: int) -> None:
    steps = value["step"]
    if not steps:
        if step == 0:
            return
        raise ValueError("checkpoint with positive step requires persisted history")
    if steps[-1] + 1 == step:
        return
    if steps[-1] == step:
        return
    raise ValueError(
        "history last persisted step matches neither supported checkpoint convention"
    )


def _history_body(value: dict[str, Any]) -> dict[str, Any]:
    return {name: item for name, item in value.items() if name != "run_identity"}


def _history_rows(symbol: str) -> tuple[list[dict[str, Any]], list[str]]:
    rows, errors = [], []
    for path in PROJECT_ROOT.glob(f"training_history_v3_{_safe_symbol_tag(symbol)}_*.json"):
        try:
            stat_before = path.stat()
            value = json.loads(path.read_text(encoding="utf-8"))
            stat_after = path.stat()
            if (
                stat_before.st_mtime_ns != stat_after.st_mtime_ns
                or stat_before.st_size != stat_after.st_size
            ):
                raise ValueError("history changed while being validated")
            run = TrainingRunIdentity.from_dict(value["run_identity"])
            if run.artifact_identity.symbol != symbol:
                continue
            if path.name != run.history_filename():
                raise ValueError("noncanonical history filename")
            _validate_history(value, run, require_run_identity=True)
            rows.append({
                "run": run, "history": value, "path": path,
                "mtime": stat_after.st_mtime, "mtime_ns": stat_after.st_mtime_ns,
            })
        except Exception as exc:
            errors.append(f"{path.name}: {exc}")
    return rows, errors


def _load_strategy(symbol: str) -> dict[str, Any] | None:
    rows, _ = _strategy_rows(symbol)
    if not rows:
        return None
    row = max(
        rows,
        key=lambda item: (
            generated_at_utc(item["artifact"].generated_at), item["path"].name
        ),
    )
    artifact = row["artifact"]
    return {
        "formula": list(artifact.formula_tokens), "best_score": artifact.best_score,
        "formula_decoded": artifact.decoded_formula, "run_identity": artifact.run_identity.to_dict(),
        "artifact_fingerprint": artifact.run_identity.artifact_identity.fingerprint,
        "strategy_fingerprint": artifact.fingerprint,
    }


def get_symbol_progress(symbol: str) -> SymbolProgress:
    checkpoint_rows, reasons = [], []
    legacy_history = PROJECT_ROOT / f"training_history_{symbol}.json"
    for path in checkpoint_glob(symbol):
        try:
            meta = _load_checkpoint_meta(path)
            if meta["run"].artifact_identity.symbol != symbol:
                raise ValueError("checkpoint symbol mismatch")
            checkpoint_rows.append(meta)
        except Exception as exc:
            reasons.append(f"{path.name}: {exc}")
    strategy_rows, strategy_errors = _strategy_rows(symbol)
    history_rows, history_errors = _history_rows(symbol)
    reasons.extend(strategy_errors)
    reasons.extend(history_errors)

    anchor = None
    if checkpoint_rows:
        anchor = max(
            checkpoint_rows,
            key=lambda row: (row["step"], row["mtime_ns"], row["path"].name),
        )["run"]
    elif strategy_rows:
        chosen = max(
            strategy_rows,
            key=lambda row: (
                generated_at_utc(row["artifact"].generated_at), row["path"].name
            ),
        )
        anchor = chosen["run"]
    elif history_rows:
        anchor = max(
            history_rows, key=lambda row: (row["mtime_ns"], row["path"].name)
        )["run"]

    if anchor is None:
        if legacy_history.exists():
            reasons.append(f"{legacy_history.name}: incompatible legacy history")
        return SymbolProgress(symbol, ModelConfig.TRAIN_STEPS, 0, None, None, None,
                              False, None, None, None, None,
                              incompatible_reasons=tuple(reasons))

    matching_ckpts = [row for row in checkpoint_rows if row["run"] == anchor]
    matching_strategies = [row for row in strategy_rows if row["run"] == anchor]
    matching_histories = [row for row in history_rows if row["run"] == anchor]
    if strategy_rows and not matching_strategies:
        reasons.append("strategy belongs to a different run/artifact fingerprint")
    if history_rows and not matching_histories:
        reasons.append("history belongs to a different run/artifact fingerprint")

    checkpoint = max(
        matching_ckpts,
        key=lambda row: (row["step"], row["mtime_ns"], row["path"].name),
        default=None,
    )
    strategy = max(
        matching_strategies,
        key=lambda row: (
            generated_at_utc(row["artifact"].generated_at), row["path"].name
        ),
        default=None,
    )
    history_row = max(
        matching_histories,
        key=lambda row: (row["mtime_ns"], row["path"].name),
        default=None,
    )
    if legacy_history.exists() and checkpoint is None:
        reasons.append(f"{legacy_history.name}: incompatible legacy history")
    if checkpoint and history_row:
        try:
            _validate_history_checkpoint_relationship(
                history_row["history"], checkpoint["step"]
            )
            if _history_body(history_row["history"]) != checkpoint["training_history"]:
                raise ValueError("standalone and checkpoint histories conflict")
        except Exception as exc:
            reasons.append(str(exc))
    current_step = checkpoint["step"] if checkpoint else 0
    best_score = checkpoint.get("best_score") if checkpoint else None
    best_formula = checkpoint.get("best_formula") if checkpoint else None
    history = history_row["history"] if history_row else (checkpoint.get("training_history") if checkpoint else None)
    if history_row:
        steps = history["step"]
        if steps:
            current_step = max(current_step, steps[-1] + 1)
        scores = history["best_score"]
        if scores:
            best_score = float(scores[-1])
    if strategy:
        artifact = strategy["artifact"]
        best_score = artifact.best_score if best_score is None else best_score
        best_formula = list(artifact.formula_tokens) if best_formula is None else best_formula
    return SymbolProgress(
        symbol=symbol, train_steps=ModelConfig.TRAIN_STEPS, current_step=(0 if reasons else current_step),
        best_score=best_score, best_formula=best_formula, formula_decoded=_decode_formula(best_formula),
        has_strategy=strategy is not None, strategy_score=(strategy["artifact"].best_score if strategy else None),
        checkpoint_path=(str(checkpoint["path"].relative_to(PROJECT_ROOT)).replace("\\", "/") if checkpoint else None),
        checkpoint_mtime=(checkpoint["mtime"] if checkpoint else None), history=history,
        run_id=anchor.run_id, artifact_fingerprint=anchor.artifact_identity.fingerprint,
        incompatible_reasons=tuple(reasons),
    )


def get_strategy_for_export(symbol: str) -> tuple[bytes, str]:
    rows, _ = _strategy_rows(symbol)
    if not rows:
        raise FileNotFoundError(f"未找到 {symbol} 的 V2 strategy")
    row = max(
        rows,
        key=lambda item: (
            generated_at_utc(item["artifact"].generated_at), item["path"].name
        ),
    )
    return row["raw"], row["path"].name


def list_strategies() -> list[dict[str, Any]]:
    rows = []
    for row in _strategy_rows("")[0]:
        artifact = row["artifact"]
        identity = artifact.run_identity.artifact_identity
        rows.append({"file": row["path"].name, "symbol": identity.symbol,
                     "timeframe": identity.timeframe, "best_score": artifact.best_score,
                     "formula_decoded": artifact.decoded_formula,
                     "run_id": artifact.run_identity.run_id, "fingerprint": artifact.fingerprint})
    if rows:
        return rows
    # Empty-symbol scan without weakening per-symbol validation.
    for path in sorted(STRATEGIES_DIR.glob("best_v3_*.json")):
        try:
            artifact = StrategyArtifact.from_dict(json.loads(path.read_text(encoding="utf-8")))
            if artifact.schema_version != STRATEGY_SCHEMA_VERSION:
                continue
            if path.name != artifact.run_identity.strategy_filename():
                continue
            identity = artifact.run_identity.artifact_identity
            rows.append({"file": path.name, "symbol": identity.symbol, "timeframe": identity.timeframe,
                         "best_score": artifact.best_score, "formula_decoded": artifact.decoded_formula,
                         "run_id": artifact.run_identity.run_id, "fingerprint": artifact.fingerprint})
        except Exception:
            continue
    return rows
