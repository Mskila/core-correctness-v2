"""Strict identity-aware training service for one V2 symbol."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
import pathlib
import random
import tempfile

import numpy as np
import torch

from data_pipeline.validation import assert_minimum_bars
from model_core.artifacts import (
    ArtifactIdentity,
    FoldEvidence,
    StrategyArtifact,
    TrainingRunIdentity,
    _safe_component,
    sha256_json,
    verify_artifact_identity,
)
from model_core.config import ModelConfig
from model_core.engine import AlphaEngine
from model_core.semantics import (
    CHECKPOINT_SCHEMA_VERSION,
    CORE_SEMANTICS_VERSION,
    EXECUTION_SEMANTICS_VERSION,
    LABEL_LOOKAHEAD_BARS,
    LABEL_SEMANTICS_VERSION,
    ArtifactCompatibilityError,
)
from model_core.vocab import VOCAB_VERSION
from model_core.validation_protocol import ProtocolConsumer, require_search_layer
from model_core.walk_forward import formula_warmup_bars, required_training_bars


CHECKPOINT_DIR = pathlib.Path("checkpoints")
STRATEGY_DIR = pathlib.Path("strategies")


def _training_config(
    random_seed: int,
    *,
    timeframe: str = "H1",
) -> dict[str, object]:
    return ModelConfig.training_config_snapshot(random_seed, timeframe=timeframe)


def _artifact_identity(data_manager, random_seed: int) -> ArtifactIdentity:
    symbols = list(data_manager.symbols)
    if len(symbols) != 1:
        raise ArtifactCompatibilityError(
            "V2 training requires exactly one normalized single symbol: "
            f"expected=1 actual={len(symbols)}"
        )
    identities = tuple(data_manager.data_identities)
    if len(identities) != 1:
        raise ArtifactCompatibilityError(
            "single-symbol dataset identity count mismatch: "
            f"expected=1 actual={len(identities)}"
        )
    dataset = identities[0]
    manager_symbol = symbols[0]
    if type(manager_symbol) is not str or not manager_symbol.strip():
        raise ArtifactCompatibilityError(
            "manager_dataset_symbol mismatch: "
            "expected=non-empty-normalized-str actual=invalid-type-or-empty"
        )
    manager_normalized = manager_symbol.strip().upper()
    dataset_normalized = dataset.symbol.strip().upper()
    if manager_normalized != dataset_normalized:
        raise ArtifactCompatibilityError(
            "manager_dataset_symbol mismatch: "
            f"expected={dataset_normalized[:80]} "
            f"actual={manager_normalized[:80]}"
        )
    config = _training_config(random_seed, timeframe=dataset.timeframe)
    return ArtifactIdentity(
        core_semantics_version=CORE_SEMANTICS_VERSION,
        vocab_version=VOCAB_VERSION,
        label_semantics_version=LABEL_SEMANTICS_VERSION,
        execution_semantics_version=EXECUTION_SEMANTICS_VERSION,
        symbol=dataset.symbol,
        timeframe=dataset.timeframe,
        training_dataset=dataset,
        training_config=config,
        training_config_hash=sha256_json(config),
    )


def _load_candidate(path: pathlib.Path) -> tuple[TrainingRunIdentity, int]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise ArtifactCompatibilityError(
            "checkpoint deserialize mismatch: "
            f"expected=readable {CHECKPOINT_SCHEMA_VERSION} actual=corrupt"
        ) from exc
    actual_schema = (
        payload.get("checkpoint_schema_version")
        if type(payload) is dict
        else "non-dict"
    )
    if actual_schema != CHECKPOINT_SCHEMA_VERSION:
        raise ArtifactCompatibilityError(
            "pre-core-fix/incompatible checkpoint schema: "
            f"expected={CHECKPOINT_SCHEMA_VERSION!r} actual={actual_schema!r}"
        )
    try:
        run = TrainingRunIdentity.from_dict(payload["run_identity"])
        step = payload["step"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ArtifactCompatibilityError(
            "checkpoint identity fields mismatch: expected=complete actual=incomplete"
        ) from exc
    if type(step) is not int or step < 0:
        raise ArtifactCompatibilityError(
            f"checkpoint step mismatch: expected=non-negative-int actual={step!r}"
        )
    return run, step


def _select_resume(identity: ArtifactIdentity) -> tuple[pathlib.Path, TrainingRunIdentity, int] | None:
    symbol_component = _safe_component(identity.symbol)
    schema_tag = CHECKPOINT_SCHEMA_VERSION.removeprefix("checkpoint-")
    symbol_prefix = f"ckpt_{schema_tag}_{symbol_component}_"
    candidates = sorted(
        path
        for path in CHECKPOINT_DIR.glob(f"ckpt_{schema_tag}_*.pt")
        if path.name.startswith(symbol_prefix)
    )
    if len(candidates) > 10_000:
        raise ArtifactCompatibilityError(
            "checkpoint candidate count mismatch: expected=at-most-10000 actual=too-many"
        )
    compatible: list[tuple[int, pathlib.Path, TrainingRunIdentity]] = []
    failures: list[str] = []
    first_failure: ArtifactCompatibilityError | None = None
    for path in candidates:
        try:
            run, step = _load_candidate(path)
            verify_artifact_identity(identity, run.artifact_identity)
            compatible.append((step, path, run))
        except ArtifactCompatibilityError as exc:
            if first_failure is None:
                first_failure = exc
            failures.append(
                f"{path.name}: incompatible {CHECKPOINT_SCHEMA_VERSION}"
            )
    if compatible:
        step, path, run = max(compatible, key=lambda item: (item[0], item[1].name))
        return path, run, step
    if candidates:
        detail = failures[0] if failures else "no exact internal identity"
        error = ArtifactCompatibilityError(
            "checkpoint candidates exist but none has exact internal identity; "
            f"expected={identity.fingerprint!r} actual={detail!r}; use --from-scratch"
        )
        if first_failure is not None:
            raise error from first_failure
        raise error
    return None


def _save_strategy(engine: AlphaEngine) -> pathlib.Path | None:
    if engine.best_formula is None:
        return None
    run_identity = engine.run_identity
    if type(run_identity) is not TrainingRunIdentity:
        raise ArtifactCompatibilityError(
            "run_identity required for strategy: "
            "expected=exact TrainingRunIdentity actual=invalid"
        )

    def revalidate_identity(stage: str) -> None:
        current = object.__getattribute__(engine, "__dict__").get(
            "run_identity", None
        )
        if current is not run_identity:
            object.__setattr__(engine, "run_identity", run_identity)
            raise ArtifactCompatibilityError(
                "run_identity lifetime mismatch after strategy callback: "
                f"expected=exact-entry-object actual=changed stage={stage}"
            )

    metrics = engine.best_metrics or {}
    raw_folds = metrics.get("fold_evidence")
    if type(raw_folds) is not list:
        raise ArtifactCompatibilityError(
            "best fold evidence mismatch: expected=actual walk-forward list actual=missing"
        )
    folds = []
    for item in raw_folds:
        try:
            fold = FoldEvidence.from_dict(item)
        except Exception as exc:
            try:
                revalidate_identity("fold-evidence-validation")
            except ArtifactCompatibilityError as identity_failure:
                raise identity_failure from exc
            raise
        except BaseException:
            object.__setattr__(engine, "run_identity", run_identity)
            raise
        revalidate_identity("fold-evidence-validation")
        folds.append(fold)
    decoded_formula = engine._decode_formula(engine.best_formula)
    revalidate_identity("formula-decoding")

    def read_existing() -> StrategyArtifact:
        try:
            serialized = target.read_text(encoding="utf-8")
            revalidate_identity("existing-artifact-read")
            payload = json.loads(serialized)
            revalidate_identity("existing-artifact-json-parse")
            existing = StrategyArtifact.from_dict(payload)
        except Exception as exc:
            try:
                revalidate_identity("existing-artifact-validation")
            except ArtifactCompatibilityError as identity_failure:
                raise identity_failure from exc
            raise ArtifactCompatibilityError(
                "immutable strategy publication conflict: "
                "expected=valid identical StrategyArtifact actual=corrupt"
            ) from exc
        revalidate_identity("existing-artifact-validation")
        return existing

    def candidate_with_timestamp(generated_at: str) -> StrategyArtifact:
        history = getattr(engine, "training_history", {})
        completed_steps = history.get("step", []) if isinstance(history, dict) else []
        candidate_evaluation_count = max(
            1,
            len(completed_steps) * ModelConfig.BATCH_SIZE,
        )
        try:
            candidate = StrategyArtifact.create(
                run_identity=run_identity,
                formula_tokens=engine.best_formula,
                decoded_formula=decoded_formula,
                best_score=float(engine.best_score),
                fold_evidence=folds,
                generated_at=generated_at,
                candidate_evaluation_count=candidate_evaluation_count,
            )
        except Exception as exc:
            try:
                revalidate_identity("strategy-artifact-creation")
            except ArtifactCompatibilityError as identity_failure:
                raise identity_failure from exc
            raise
        except BaseException:
            object.__setattr__(engine, "run_identity", run_identity)
            raise
        revalidate_identity("strategy-artifact-creation")
        return candidate

    artifact = candidate_with_timestamp(datetime.now(timezone.utc).isoformat())
    filename = run_identity.strategy_filename()
    revalidate_identity("strategy-filename")
    target = STRATEGY_DIR / filename
    revalidate_identity("strategy-path-construction")
    target_exists = target.exists()
    revalidate_identity("strategy-existence-check")
    if target_exists:
        existing = read_existing()
        candidate = candidate_with_timestamp(existing.generated_at)
        if existing.fingerprint == candidate.fingerprint:
            return target
        raise ArtifactCompatibilityError(
            "immutable strategy content mismatch: "
            f"expected={existing.fingerprint!r} actual={candidate.fingerprint!r}"
        )
    artifact_payload = artifact.to_dict()
    revalidate_identity("strategy-artifact-serialization")
    STRATEGY_DIR.mkdir(parents=True, exist_ok=True)
    revalidate_identity("strategy-directory-creation")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = pathlib.Path(temporary_name)
    try:
        revalidate_identity("strategy-temporary-creation")
    except BaseException as failure:
        try:
            os.close(descriptor)
        except BaseException:
            BaseException.add_note(
                failure, "strategy temporary descriptor cleanup failed"
            )
        try:
            temporary.unlink(missing_ok=True)
        except BaseException:
            BaseException.add_note(
                failure, "strategy temporary path cleanup failed"
            )
        raise
    primary: BaseException | None = None
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as fp:
            revalidate_identity("strategy-descriptor-open")
            json.dump(artifact_payload, fp, indent=2, ensure_ascii=False)
            revalidate_identity("strategy-json-write")
            fp.flush()
            revalidate_identity("strategy-flush")
            os.fsync(fp.fileno())
            revalidate_identity("strategy-fsync")
        revalidate_identity("strategy-descriptor-close")
        try:
            os.link(temporary, target)
        except FileExistsError as exc:
            revalidate_identity("strategy-link-conflict")
            existing = read_existing()
            candidate = candidate_with_timestamp(existing.generated_at)
            if existing.fingerprint == candidate.fingerprint:
                return target
            raise ArtifactCompatibilityError(
                "immutable strategy publication conflict: "
                f"expected={existing.fingerprint!r} actual={candidate.fingerprint!r}"
            ) from exc
        except Exception as exc:
            revalidate_identity("strategy-link-failure")
            if target.exists():
                existing = read_existing()
                candidate = candidate_with_timestamp(existing.generated_at)
                if existing.fingerprint == candidate.fingerprint:
                    return target
                raise ArtifactCompatibilityError(
                    "immutable strategy publication conflict: "
                    "expected=uncontended-or-identical actual=foreign"
                ) from exc
            raise ArtifactCompatibilityError(
                "immutable strategy publication failed: "
                "expected=atomic-no-replace actual=ordinary-error"
            ) from exc
        try:
            revalidate_identity("strategy-link-publication")
        except ArtifactCompatibilityError as identity_failure:
            try:
                temporary_stat = os.stat(temporary, follow_symlinks=False)
                target_stat = os.stat(target, follow_symlinks=False)
                if (
                    temporary_stat.st_dev == target_stat.st_dev
                    and temporary_stat.st_ino == target_stat.st_ino
                ):
                    target.unlink()
            except BaseException:
                BaseException.add_note(
                    identity_failure,
                    "owned strategy publication cleanup failed",
                )
            raise
        return target
    except ArtifactCompatibilityError as failure:
        primary = failure
        raise
    except Exception as exc:
        failure = ArtifactCompatibilityError(
            "immutable strategy publication failed: "
            "expected=complete-atomic-StrategyArtifact actual=ordinary-error"
        )
        primary = failure
        raise failure from exc
    except BaseException as failure:
        primary = failure
        raise
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except BaseException as cleanup_failure:
            if primary is None:
                if not isinstance(cleanup_failure, Exception):
                    raise
                raise ArtifactCompatibilityError(
                    "immutable strategy publication cleanup failed: "
                    "expected=temporary-removed actual=ordinary-error"
                ) from cleanup_failure
            try:
                BaseException.add_note(
                    primary,
                    "strategy publication cleanup failure category=base-exception",
                )
            except BaseException:
                pass
        try:
            revalidate_identity("strategy-temporary-cleanup")
        except ArtifactCompatibilityError as identity_failure:
            if primary is None:
                raise
            try:
                BaseException.add_note(
                    primary,
                    "strategy cleanup changed run identity",
                )
            except BaseException:
                pass


def run_training_session(
    data_manager,
    *,
    source_path: str | pathlib.Path | None,
    from_scratch: bool,
    random_seed: int,
) -> AlphaEngine:
    """Train or exactly resume one normalized symbol; domain errors propagate."""
    require_search_layer(data_manager, ProtocolConsumer.TRAINING)
    identity = _artifact_identity(data_manager, random_seed)
    required = required_training_bars(
        warmup_bars=formula_warmup_bars(ModelConfig.MAX_FORMULA_LEN),
        label_lookahead=LABEL_LOOKAHEAD_BARS,
        n_blocks=ModelConfig.WF_N_BLOCKS,
        min_fold_bars=ModelConfig.WF_MIN_FOLD_BARS,
        configured_gap=ModelConfig.WF_GAP,
    )
    assert_minimum_bars(
        identity.training_dataset.bars,
        required,
        context=f"V2 training dataset {identity.symbol} {identity.timeframe}",
    )

    selected = None if from_scratch else _select_resume(identity)
    run_identity = (
        TrainingRunIdentity.create(identity) if selected is None else selected[1]
    )
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
    engine = AlphaEngine(
        data_manager=data_manager,
        target_symbol=identity.symbol,
        run_identity=run_identity,
        use_lord_regularization=ModelConfig.USE_LORD_REGULARIZATION,
        lord_decay_rate=ModelConfig.LORD_DECAY_RATE,
        lord_num_iterations=ModelConfig.LORD_NUM_ITERATIONS,
    )
    engine.source_path = None if source_path is None else str(source_path)
    start_step = 0
    if selected is not None:
        start_step = engine.load_checkpoint(str(selected[0]))
    if start_step < ModelConfig.TRAIN_STEPS:
        engine.train(start_step=start_step)
    _save_strategy(engine)
    return engine
