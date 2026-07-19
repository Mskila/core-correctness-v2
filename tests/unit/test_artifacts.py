from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace
import json
import math
from types import MappingProxyType

import pytest

from data_pipeline.validation import DatasetIdentity
import model_core.artifacts as artifacts_module
from model_core.config import ModelConfig
from model_core.semantics import (
    CORE_SEMANTICS_VERSION,
    EXECUTION_SEMANTICS_VERSION,
    LABEL_LOOKAHEAD_BARS,
    LABEL_SEMANTICS_VERSION,
    ArtifactCompatibilityError,
    DataValidationError,
)
from model_core.vocab import FORMULA_VOCAB, VOCAB_VERSION

from model_core.artifacts import (
    ArtifactIdentity,
    FoldEvidence,
    StrategyArtifact,
    TrainingRunIdentity,
    canonical_json_bytes,
    load_artifact_json,
    sha256_json,
    verify_artifact_identity,
)


def dataset_identity(
    *,
    symbol: str = "EURUSD",
    timeframe: str = "H1",
    start: int = 1_000,
    end: int = 9_000,
    data_fingerprint: str = "a" * 64,
    time_fingerprint: str = "b" * 64,
) -> DatasetIdentity:
    return DatasetIdentity(
        schema_version="ohlcv-v2",
        symbol=symbol,
        timeframe=timeframe,
        start_time_ns=start,
        end_time_ns=end,
        bars=9,
        data_fingerprint=data_fingerprint,
        time_fingerprint=time_fingerprint,
    )


DATASET_FIELDS = (
    "schema_version",
    "symbol",
    "timeframe",
    "start_time_ns",
    "end_time_ns",
    "bars",
    "data_fingerprint",
    "time_fingerprint",
)


def bypass_dataset_identity(
    source: DatasetIdentity | None = None,
    **changes: object,
) -> DatasetIdentity:
    """Build an isolated hostile identity without invoking T01 validation."""
    valid_source = dataset_identity() if source is None else source
    unknown = set(changes).difference(DATASET_FIELDS)
    assert not unknown, f"unknown DatasetIdentity fields: {sorted(unknown)}"
    hostile = object.__new__(DatasetIdentity)
    for field in DATASET_FIELDS:
        value = changes[field] if field in changes else getattr(valid_source, field)
        object.__setattr__(hostile, field, value)
    return hostile


def instrumented_dataset_identity(
    *,
    failing_field: str | None = None,
    failure: Exception | None = None,
) -> tuple[DatasetIdentity, dict[str, int]]:
    reads = {field: 0 for field in DATASET_FIELDS}

    class InstrumentedDatasetIdentity(DatasetIdentity):
        def __getattribute__(self, name: str) -> object:
            if name in reads:
                reads[name] += 1
                if name == failing_field:
                    assert failure is not None
                    raise failure
            return super().__getattribute__(name)

    return (
        InstrumentedDatasetIdentity(
            schema_version="ohlcv-v2",
            symbol="EURUSD",
            timeframe="H1",
            start_time_ns=1_000,
            end_time_ns=9_000,
            bars=9,
            data_fingerprint="a" * 64,
            time_fingerprint="b" * 64,
        ),
        reads,
    )


def training_config(*, seed: int = 42) -> dict[str, object]:
    return {
        "model": {"input_dim": 60, "hidden_dim": 128, "num_layers": 2},
        "batch_size": 192,
        "train_steps": 9_000,
        "max_formula_len": 8,
        "reward": {
            "ic": 1.0,
            "return": 0.5,
            "alpha": 1.0,
            "mode": "ftmo",
            "ic_gate_thresh": 0.01,
            "ic_gate_mult": 1.15,
            "ic_neg_mult": 0.75,
            "ema_baseline": True,
            "ema_decay": 0.95,
            "ema_warmup": 10,
            "factor_top_k": 25,
            "corr_threshold": 0.85,
            "corr_penalty": 0.8,
            "beta_neutral_penalty": True,
            "half_consistency_bonus": True,
            "beta_neutral_thresh": 0.85,
            "beta_neutral_light_thresh": 0.70,
        },
        "entropy": actual_entropy_config(),
        "elite": {
            "size": 20,
            "replay_frac": 0.25,
            "reward_scale": 1.2,
            "decay": True,
            "decay_half_life": 300,
        },
        "restart": {
            "max_restarts": 55,
            "restart_noise": 0.25,
            "stagnation_window": 500,
            "full_reset_every": 3,
            "partial_reset": True,
            "partial_reset_layers": ["ln_f", "blocks"],
        },
        "noise": {
            "initial": 0.1,
            "boost": 2.0,
            "adaptive": True,
            "min": 0.15,
            "max": 0.60,
            "boost_factor": 2.0,
        },
        "lord": {
            "use_lord_regularization": False,
            "lord_decay_rate": 1.0e-3,
            "lord_num_iterations": 5,
        },
        "walk_forward": {
            "blocks": 5,
            "gap": LABEL_LOOKAHEAD_BARS,
            "min_fold_bars": 200,
            "warmup_bars": 200,
            "label_lookahead": LABEL_LOOKAHEAD_BARS,
        },
        "cost_rate": 0.0001,
        "neutral_band": 0.05,
        "random_seed": seed,
    }


def _training_config_with_lord(**changes: object) -> dict[str, object]:
    config = training_config()
    config["lord"] = {
        "use_lord_regularization": True,
        "lord_decay_rate": 1.0e-3,
        "lord_num_iterations": 5,
        **changes,
    }
    return config


def test_training_identity_requires_exact_lord_behavior_schema() -> None:
    config = _training_config_with_lord()
    identity = artifact_identity_with_config(config)

    assert identity.training_config["lord"] == config["lord"]

    missing = training_config()
    missing.pop("lord")
    with pytest.raises(ArtifactCompatibilityError, match=r"missing=.*lord"):
        artifact_identity_with_config(missing)

    extra = _training_config_with_lord(unexpected=True)
    with pytest.raises(ArtifactCompatibilityError, match=r"lord.*unknown"):
        artifact_identity_with_config(extra)


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("use_lord_regularization", 1, "boolean"),
        ("lord_decay_rate", True, "finite number"),
        ("lord_decay_rate", -0.01, "finite number >= 0"),
        ("lord_num_iterations", 0, "integer >= 1"),
        ("lord_num_iterations", 1.5, "integer >= 1"),
    ],
)
def test_training_identity_rejects_invalid_lord_values(
    field: str, value: object, expected: str
) -> None:
    with pytest.raises(
        ArtifactCompatibilityError,
        match=rf"training_config\.lord\.{field}.*expected={expected}",
    ):
        artifact_identity_with_config(_training_config_with_lord(**{field: value}))


def actual_entropy_config() -> dict[str, object]:
    """Training identity keys map one-to-one to ModelConfig behavior controls."""
    return {
        "coeff_max": ModelConfig.ENTROPY_COEFF_MAX,
        "coeff_power": ModelConfig.ENTROPY_COEFF_POWER,
        "collapse_thresh": ModelConfig.ENTROPY_COLLAPSE_THRESH,
        "collapse_steps": ModelConfig.ENTROPY_COLLAPSE_STEPS,
        "floor_enabled": ModelConfig.ENTROPY_FLOOR,
        "floor_thresh": ModelConfig.ENTROPY_FLOOR_THRESH,
        "floor_lambda": ModelConfig.ENTROPY_FLOOR_LAMBDA,
    }


NESTED_BEHAVIOR_CONFIG_PATHS = (
    *(("model", field) for field in ("input_dim", "hidden_dim", "num_layers")),
    *(("reward", field) for field in (
        "ic",
        "return",
        "alpha",
        "mode",
        "ic_gate_thresh",
        "ic_gate_mult",
        "ic_neg_mult",
        "ema_baseline",
        "ema_decay",
        "ema_warmup",
        "factor_top_k",
        "corr_threshold",
        "corr_penalty",
        "beta_neutral_penalty",
        "half_consistency_bonus",
        "beta_neutral_thresh",
        "beta_neutral_light_thresh",
    )),
    *(("entropy", field) for field in (
        "coeff_max",
        "coeff_power",
        "collapse_thresh",
        "collapse_steps",
        "floor_enabled",
        "floor_thresh",
        "floor_lambda",
    )),
    *(("elite", field) for field in (
        "size",
        "replay_frac",
        "reward_scale",
        "decay",
        "decay_half_life",
    )),
    *(("restart", field) for field in (
        "max_restarts",
        "restart_noise",
        "stagnation_window",
        "full_reset_every",
        "partial_reset",
        "partial_reset_layers",
    )),
    *(("noise", field) for field in (
        "initial",
        "boost",
        "adaptive",
        "min",
        "max",
        "boost_factor",
    )),
    *(("lord", field) for field in (
        "use_lord_regularization",
        "lord_decay_rate",
        "lord_num_iterations",
    )),
    *(("walk_forward", field) for field in (
        "blocks",
        "gap",
        "min_fold_bars",
        "warmup_bars",
        "label_lookahead",
    )),
)


def artifact_identity(*, seed: int = 42, blocks: int = 5) -> ArtifactIdentity:
    config = training_config(seed=seed)
    config["walk_forward"]["blocks"] = blocks  # type: ignore[index]
    return artifact_identity_with_config(config)


def artifact_identity_with_config(
    config: dict[str, object],
    *,
    config_hash: str | None = None,
) -> ArtifactIdentity:
    return ArtifactIdentity(
        core_semantics_version=CORE_SEMANTICS_VERSION,
        vocab_version=VOCAB_VERSION,
        label_semantics_version=LABEL_SEMANTICS_VERSION,
        execution_semantics_version=EXECUTION_SEMANTICS_VERSION,
        symbol="EURUSD",
        timeframe="H1",
        training_dataset=dataset_identity(),
        training_config=config,
        training_config_hash=(
            sha256_json(config) if config_hash is None else config_hash
        ),
    )


IDENTITY_MUTATIONS = (
    ("training_config_hash", "training_config_hash"),
    ("training_config_missing", "reward.*alpha"),
    ("training_config_hash_mismatch", "training_config_hash"),
    ("training_config_environment", "device"),
    ("core_semantics_version", "core_semantics_version"),
    ("vocab_version", "vocab_version"),
    ("label_semantics_version", "label_semantics_version"),
    ("execution_semantics_version", "execution_semantics_version"),
    ("symbol", "symbol"),
    ("timeframe", "timeframe"),
    ("dataset_data_fingerprint", "data_fingerprint"),
    ("dataset_time_fingerprint", "time_fingerprint"),
    ("dataset_range", "range"),
    ("dataset_bars", "bars"),
)


def mutate_artifact_identity(identity: ArtifactIdentity, mutation: str) -> None:
    if mutation == "training_config_hash":
        object.__setattr__(identity, "training_config_hash", "0" * 64)
    elif mutation == "training_config_missing":
        config = training_config()
        del config["reward"]["alpha"]  # type: ignore[index]
        object.__setattr__(identity, "training_config", config)
        object.__setattr__(identity, "training_config_hash", sha256_json(config))
    elif mutation == "training_config_hash_mismatch":
        object.__setattr__(identity, "training_config", training_config(seed=43))
    elif mutation == "training_config_environment":
        config = training_config()
        config["device"] = "cpu"
        object.__setattr__(identity, "training_config", config)
        object.__setattr__(identity, "training_config_hash", sha256_json(config))
    elif mutation in {
        "core_semantics_version",
        "vocab_version",
        "label_semantics_version",
        "execution_semantics_version",
    }:
        object.__setattr__(identity, mutation, "legacy")
    elif mutation == "symbol":
        object.__setattr__(identity, "symbol", "GBPUSD")
    elif mutation == "timeframe":
        object.__setattr__(identity, "timeframe", "H4")
    elif mutation == "dataset_data_fingerprint":
        object.__setattr__(
            identity,
            "training_dataset",
            bypass_dataset_identity(
                identity.training_dataset,
                data_fingerprint="short",
            ),
        )
    elif mutation == "dataset_time_fingerprint":
        object.__setattr__(
            identity,
            "training_dataset",
            bypass_dataset_identity(
                identity.training_dataset,
                time_fingerprint="G" * 64,
            ),
        )
    elif mutation == "dataset_range":
        object.__setattr__(
            identity,
            "training_dataset",
            bypass_dataset_identity(identity.training_dataset, start_time_ns=10_000),
        )
    elif mutation == "dataset_bars":
        object.__setattr__(
            identity,
            "training_dataset",
            bypass_dataset_identity(identity.training_dataset, bars=True),
        )
    else:
        raise AssertionError(f"unknown identity mutation: {mutation}")


def strategy_artifact() -> StrategyArtifact:
    run = TrainingRunIdentity.create(artifact_identity())
    return StrategyArtifact.create(
        run_identity=run,
        formula_tokens=[0],
        decoded_formula=FORMULA_VOCAB.token_names[0],
        best_score=1.25,
        fold_evidence=[fold_evidence_with_index(index) for index in range(4)],
        generated_at="2026-07-15T00:00:00Z",
    )


def direct_strategy_artifact(
    *,
    run_identity: TrainingRunIdentity,
    fold_evidence: tuple[FoldEvidence, ...],
) -> StrategyArtifact:
    payload = {
        "schema_version": "strategy-v2",
        "run_identity": run_identity.to_dict(),
        "formula_tokens": [0],
        "decoded_formula": FORMULA_VOCAB.token_names[0],
        "best_score": 1.0,
        "fold_evidence": [fold.to_dict() for fold in fold_evidence],
        "generated_at": "2026-07-15T00:00:00Z",
    }
    return StrategyArtifact(
        schema_version="strategy-v2",
        run_identity=run_identity,
        formula_tokens=(0,),
        decoded_formula=FORMULA_VOCAB.token_names[0],
        best_score=1.0,
        fold_evidence=fold_evidence,
        generated_at="2026-07-15T00:00:00Z",
        fingerprint=sha256_json(payload),
    )


def construct_strategy_with_fold(
    construction: str,
    fold: FoldEvidence,
) -> StrategyArtifact:
    return construct_strategy_with_folds(construction, (fold,))


def fold_evidence_with_index(index: int) -> FoldEvidence:
    train_end = 2_000 + index * 1_500
    return FoldEvidence(
        fold_index=index,
        train_start_time_ns=1_000,
        train_end_time_ns=train_end,
        val_start_time_ns=train_end + 500,
        val_end_time_ns=train_end + 1_000,
        effective_gap=LABEL_LOOKAHEAD_BARS,
        validation_metrics={"ic": 0.1 + index * 0.01, "net_return": 0.02},
    )


def construct_strategy_with_folds(
    construction: str,
    folds: tuple[FoldEvidence, ...],
) -> StrategyArtifact:
    padded = list(folds)
    while padded and len(padded) < 4:
        previous = padded[-1]
        train_end = max(previous.val_end_time_ns, previous.train_end_time_ns + 1)
        padded.append(
            FoldEvidence(
                fold_index=len(padded),
                train_start_time_ns=previous.train_start_time_ns,
                train_end_time_ns=train_end,
                val_start_time_ns=train_end + 1,
                val_end_time_ns=train_end + 2,
                effective_gap=previous.effective_gap,
                validation_metrics={"ic": 0.1},
            )
        )
    folds = tuple(padded)
    run = TrainingRunIdentity.create(artifact_identity(blocks=5))
    if construction == "create":
        return StrategyArtifact.create(
            run_identity=run,
            formula_tokens=[0],
            decoded_formula=FORMULA_VOCAB.token_names[0],
            best_score=1.0,
            fold_evidence=folds,
            generated_at="2026-07-15T00:00:00Z",
        )
    if construction == "direct":
        return direct_strategy_artifact(
            run_identity=run,
            fold_evidence=folds,
        )
    payload = {
        "schema_version": "strategy-v2",
        "run_identity": run.to_dict(),
        "formula_tokens": [0],
        "decoded_formula": FORMULA_VOCAB.token_names[0],
        "best_score": 1.0,
        "fold_evidence": [fold.to_dict() for fold in folds],
        "generated_at": "2026-07-15T00:00:00Z",
    }
    return StrategyArtifact.from_dict(
        {**payload, "fingerprint": sha256_json(payload)}
    )


def construct_strategy_for_blocks(
    boundary: str,
    blocks: int,
    fold_count: int,
) -> StrategyArtifact:
    run = TrainingRunIdentity.create(artifact_identity(blocks=blocks))
    folds = tuple(fold_evidence_with_index(index) for index in range(fold_count))
    if boundary == "create":
        return StrategyArtifact.create(
            run_identity=run,
            formula_tokens=[0],
            decoded_formula=FORMULA_VOCAB.token_names[0],
            best_score=1.0,
            fold_evidence=folds,
            generated_at="2026-07-15T00:00:00Z",
        )
    if boundary == "direct":
        return direct_strategy_artifact(run_identity=run, fold_evidence=folds)
    payload = {
        "schema_version": "strategy-v2",
        "run_identity": run.to_dict(),
        "formula_tokens": [0],
        "decoded_formula": FORMULA_VOCAB.token_names[0],
        "best_score": 1.0,
        "fold_evidence": [fold.to_dict() for fold in folds],
        "generated_at": "2026-07-15T00:00:00Z",
    }
    serialized = {**payload, "fingerprint": sha256_json(payload)}
    if boundary == "load":
        serialized = json.loads(json.dumps(serialized))
    return StrategyArtifact.from_dict(serialized)


@pytest.mark.parametrize("boundary", ["create", "direct", "from_dict", "load"])
@pytest.mark.parametrize("fold_count", [0, 1, 2, 3, 5])
def test_five_block_strategy_rejects_wrong_fold_cardinality_at_every_boundary(
    boundary: str,
    fold_count: int,
) -> None:
    with pytest.raises(
        ArtifactCompatibilityError,
        match=rf"fold_evidence count.*expected=4.*actual={fold_count}",
    ):
        construct_strategy_for_blocks(boundary, 5, fold_count)


@pytest.mark.parametrize("boundary", ["create", "direct", "from_dict", "load"])
def test_three_block_strategy_accepts_two_ordered_folds_and_round_trips(
    boundary: str,
) -> None:
    strategy = construct_strategy_for_blocks(boundary, 3, 2)

    assert [fold.fold_index for fold in strategy.fold_evidence] == [0, 1]
    assert StrategyArtifact.from_dict(strategy.to_dict()) == strategy


@pytest.mark.parametrize("boundary", ["create", "direct", "from_dict", "load"])
@pytest.mark.parametrize("fold_count", [0, 1, 3, 4])
def test_three_block_strategy_rejects_non_two_fold_cardinality(
    boundary: str,
    fold_count: int,
) -> None:
    with pytest.raises(
        ArtifactCompatibilityError,
        match=rf"fold_evidence count.*expected=2.*actual={fold_count}",
    ):
        construct_strategy_for_blocks(boundary, 3, fold_count)


@pytest.mark.parametrize("boundary", ["create", "direct", "from_dict", "load"])
def test_default_five_block_strategy_still_requires_four_folds(boundary: str) -> None:
    strategy = construct_strategy_for_blocks(boundary, 5, 4)

    assert len(strategy.fold_evidence) == 4
    assert StrategyArtifact.from_dict(strategy.to_dict()) == strategy


STRATEGY_PUBLIC_MUTATIONS = (
    ("schema_version", "schema"),
    ("run_identity", "run_identity"),
    ("formula_tokens", "formula token"),
    ("decoded_formula", "decoded_formula"),
    ("best_score", "best_score"),
    ("fold_evidence", "validation_metrics"),
    ("generated_at", "generated_at"),
    ("fingerprint", "fingerprint"),
)


def mutate_strategy_public_state(
    strategy: StrategyArtifact,
    mutation: str,
) -> None:
    original_payload = strategy._payload_without_fingerprint()
    recompute = True
    if mutation == "schema_version":
        object.__setattr__(strategy, "schema_version", "strategy-v1")
    elif mutation == "run_identity":
        object.__setattr__(strategy, "run_identity", "not-a-run-identity")
        recompute = False
    elif mutation == "formula_tokens":
        object.__setattr__(strategy, "formula_tokens", (True,))
    elif mutation == "decoded_formula":
        object.__setattr__(strategy, "decoded_formula", "DIFFERENT_FORMULA")
    elif mutation == "best_score":
        object.__setattr__(strategy, "best_score", True)
    elif mutation == "fold_evidence":
        object.__setattr__(
            strategy.fold_evidence[0],
            "validation_metrics",
            {"ic": "not-a-number"},
        )
    elif mutation == "generated_at":
        object.__setattr__(strategy, "generated_at", "2026-07-15T00:00:00")
    elif mutation == "fingerprint":
        object.__setattr__(strategy, "fingerprint", "0" * 64)
        recompute = False
    else:
        raise AssertionError(f"unknown strategy mutation: {mutation}")
    if recompute:
        if mutation == "fold_evidence":
            original_payload["fold_evidence"][0]["validation_metrics"] = {  # type: ignore[index]
                "ic": "not-a-number"
            }
            fingerprint_payload = original_payload
        else:
            fingerprint_payload = strategy._payload_without_fingerprint()
        object.__setattr__(
            strategy,
            "fingerprint",
            sha256_json(fingerprint_payload),
        )


def test_canonical_json_and_hash_are_order_independent_utf8_and_compact() -> None:
    left = {"z": "策略", "a": {"y": 2, "x": 1}}
    right = {"a": {"x": 1, "y": 2}, "z": "策略"}
    assert canonical_json_bytes(left) == b'{"a":{"x":1,"y":2},"z":"\xe7\xad\x96\xe7\x95\xa5"}'
    assert sha256_json(left) == sha256_json(right)


@pytest.mark.parametrize(
    "value",
    [
        math.nan,
        math.inf,
        -math.inf,
        {1, 2},
        object(),
        (1, 2),
        MappingProxyType({"a": 1}),
    ],
)
def test_canonical_json_rejects_non_json_or_non_finite_values(value: object) -> None:
    with pytest.raises(ArtifactCompatibilityError):
        canonical_json_bytes({"value": value})


def _nested_json(container: str, depth: int) -> object:
    value: object = "leaf"
    for _ in range(depth):
        value = [value] if container == "list" else {"child": value}
    return value


@pytest.mark.parametrize("boundary", ["canonical_json_bytes", "artifact_identity"])
@pytest.mark.parametrize("cycle_kind", ["self", "mutual"])
def test_json_cycles_fail_closed_with_path_context(
    boundary: str,
    cycle_kind: str,
) -> None:
    first: dict[str, object] = {}
    if cycle_kind == "self":
        first["cycle"] = first
    else:
        second: dict[str, object] = {"back": first}
        first["next"] = second

    with pytest.raises(ArtifactCompatibilityError, match=r"cycle.*\$"):
        if boundary == "canonical_json_bytes":
            canonical_json_bytes(first)
        else:
            config = training_config()
            config["cycle"] = first
            artifact_identity_with_config(config, config_hash="0" * 64)


@pytest.mark.parametrize("boundary", ["canonical_json_bytes", "artifact_identity"])
@pytest.mark.parametrize("container", ["list", "mapping"])
def test_excessive_json_depth_fails_closed_with_depth_and_path(
    boundary: str,
    container: str,
) -> None:
    value = _nested_json(container, artifacts_module._MAX_JSON_NESTING_DEPTH + 1)

    with pytest.raises(
        ArtifactCompatibilityError,
        match=rf"nesting depth.*{artifacts_module._MAX_JSON_NESTING_DEPTH}",
    ) as exc_info:
        if boundary == "canonical_json_bytes":
            canonical_json_bytes(value)
        else:
            config = training_config()
            config["deep"] = value
            artifact_identity_with_config(config, config_hash="0" * 64)
    assert "$" in str(exc_info.value)


@pytest.mark.parametrize("container", ["list", "mapping"])
def test_json_depth_boundary_and_flat_scale_controls_remain_valid(
    container: str,
) -> None:
    nested = _nested_json(container, artifacts_module._MAX_JSON_NESTING_DEPTH - 1)
    assert canonical_json_bytes(nested)

    flat = list(range(100_000))
    encoded = canonical_json_bytes(flat)
    assert encoded.startswith(b"[0,1,2")
    assert encoded.endswith(b"99999]")


@pytest.mark.parametrize("boundary", [canonical_json_bytes, sha256_json])
@pytest.mark.parametrize("container", ["list", "mapping"])
def test_public_json_depth_uses_fixed_100_accepted_101_rejected_oracle(
    boundary,
    container: str,
) -> None:
    assert boundary(_nested_json(container, 100))
    with pytest.raises(ArtifactCompatibilityError, match=r"maximum=100 actual=101"):
        boundary(_nested_json(container, 101))


@pytest.mark.parametrize("container", ["list", "mapping", "tuple"])
def test_internal_freeze_graph_depth_uses_fixed_100_accepted_101_rejected_oracle(
    container: str,
) -> None:
    def nested(depth: int) -> object:
        value: object = "leaf"
        for _ in range(depth):
            if container == "list":
                value = [value]
            elif container == "mapping":
                value = {"child": value}
            else:
                value = (value,)
        return value

    artifacts_module._validate_json_graph(nested(100), allow_internal=True)
    with pytest.raises(ArtifactCompatibilityError, match=r"maximum=100 actual=101"):
        artifacts_module._validate_json_graph(nested(101), allow_internal=True)


@pytest.mark.parametrize("container", ["list", "mapping"])
def test_training_config_depth_guard_uses_fixed_100_101_oracle(container: str) -> None:
    at_limit = training_config()
    # The training_config mapping is depth 1, so 99 nested containers below it
    # place the deepest container at the fixed public limit of 100.
    at_limit["deep"] = _nested_json(container, 99)
    with pytest.raises(ArtifactCompatibilityError) as at_limit_error:
        artifact_identity_with_config(at_limit, config_hash="0" * 64)
    assert "nesting depth" not in str(at_limit_error.value)

    excessive = training_config()
    excessive["deep"] = _nested_json(container, 100)
    with pytest.raises(ArtifactCompatibilityError, match=r"maximum=100 actual=101"):
        artifact_identity_with_config(excessive, config_hash="0" * 64)


def test_acyclic_shared_json_references_remain_valid() -> None:
    shared = {"values": [1, 2, 3]}
    value = {"left": shared, "right": shared}
    assert canonical_json_bytes(value) == (
        b'{"left":{"values":[1,2,3]},"right":{"values":[1,2,3]}}'
    )


def test_training_config_order_does_not_change_hash_but_value_does() -> None:
    left = training_config()
    right = dict(reversed(list(left.items())))
    changed = training_config(seed=43)
    assert sha256_json(left) == sha256_json(right)
    assert sha256_json(left) != sha256_json(changed)


def test_training_identity_accepts_exact_model_config_entropy_controls() -> None:
    config = training_config()
    config["entropy"] = actual_entropy_config()

    identity = artifact_identity_with_config(config)

    assert identity.training_config["entropy"]["floor_enabled"] is ModelConfig.ENTROPY_FLOOR  # type: ignore[index]


def test_training_identity_rejects_synthetic_entropy_floor_and_weight() -> None:
    config = training_config()
    config["entropy"] = {
        **actual_entropy_config(),
        "floor": 0.2,
        "weight": 0.01,
    }

    with pytest.raises(
        ArtifactCompatibilityError,
        match=r"training_config\.entropy.*floor.*weight",
    ):
        artifact_identity_with_config(config)


@pytest.mark.parametrize("field", tuple(actual_entropy_config()))
def test_every_model_config_entropy_control_changes_and_restores_identity(
    field: str,
) -> None:
    config = training_config()
    config["entropy"] = actual_entropy_config()
    baseline = artifact_identity_with_config(config)
    changed = deepcopy(config)
    original = changed["entropy"][field]  # type: ignore[index]
    if type(original) is bool:
        changed["entropy"][field] = not original  # type: ignore[index]
    elif type(original) is int:
        changed["entropy"][field] = original + 1  # type: ignore[index]
    else:
        changed["entropy"][field] = float(original) + 0.125  # type: ignore[index]

    mutated = artifact_identity_with_config(changed)
    restored = deepcopy(changed)
    restored["entropy"][field] = original  # type: ignore[index]
    restored_identity = artifact_identity_with_config(restored)

    assert mutated.training_config_hash != baseline.training_config_hash
    assert mutated.fingerprint != baseline.fingerprint
    assert restored_identity.training_config_hash == baseline.training_config_hash
    assert restored_identity.fingerprint == baseline.fingerprint

    run_id = "1" * 32
    baseline_run = TrainingRunIdentity(run_id, baseline)
    mutated_run = TrainingRunIdentity(run_id, mutated)
    restored_run = TrainingRunIdentity(run_id, restored_identity)
    folds = tuple(fold_evidence_with_index(index) for index in range(4))
    baseline_strategy = direct_strategy_artifact(
        run_identity=baseline_run,
        fold_evidence=folds,
    )
    mutated_strategy = direct_strategy_artifact(
        run_identity=mutated_run,
        fold_evidence=folds,
    )
    restored_strategy = direct_strategy_artifact(
        run_identity=restored_run,
        fold_evidence=folds,
    )
    assert sha256_json(mutated_run.to_dict()) != sha256_json(baseline_run.to_dict())
    assert sha256_json(restored_run.to_dict()) == sha256_json(baseline_run.to_dict())
    assert mutated_strategy.fingerprint != baseline_strategy.fingerprint
    assert restored_strategy.fingerprint == baseline_strategy.fingerprint


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("coeff_max", True),
        ("coeff_power", float("nan")),
        ("collapse_thresh", float("inf")),
        ("collapse_steps", True),
        ("floor_enabled", 1),
        ("floor_enabled", "true"),
        ("floor_thresh", float("-inf")),
        ("floor_lambda", False),
    ],
)
def test_entropy_controls_reject_bool_as_number_nonfinite_and_coercion(
    field: str,
    value: object,
) -> None:
    config = training_config()
    config["entropy"][field] = value  # type: ignore[index]
    with pytest.raises(
        ArtifactCompatibilityError,
        match=field,
    ):
        artifact_identity_with_config(config)


def test_entropy_identity_is_deeply_immutable_after_construction() -> None:
    identity = artifact_identity()
    with pytest.raises(TypeError):
        identity.training_config["entropy"]["floor_enabled"] = False  # type: ignore[index]


def test_three_block_artifact_identity_succeeds_at_all_serialization_boundaries() -> None:
    direct = artifact_identity(blocks=3)
    from_dict = ArtifactIdentity.from_dict(direct.to_dict())
    round_tripped = ArtifactIdentity.from_dict(from_dict.to_dict())

    assert direct.training_config["walk_forward"]["blocks"] == 3  # type: ignore[index]
    assert from_dict == direct
    assert round_tripped == direct


def test_walk_forward_block_count_remains_part_of_exact_artifact_identity() -> None:
    three_blocks = artifact_identity(blocks=3)
    five_blocks = artifact_identity(blocks=5)

    assert three_blocks.training_config_hash != five_blocks.training_config_hash
    assert three_blocks.fingerprint != five_blocks.fingerprint
    with pytest.raises(
        ArtifactCompatibilityError,
        match=r"training_config|training_config_hash",
    ):
        verify_artifact_identity(three_blocks, five_blocks)


@pytest.mark.parametrize("boundary", ["direct", "from_dict"])
@pytest.mark.parametrize(
    ("container", "leaf"),
    NESTED_BEHAVIOR_CONFIG_PATHS,
    ids=lambda value: str(value),
)
def test_training_config_requires_every_declared_nested_behavior_leaf(
    boundary: str,
    container: str,
    leaf: str,
) -> None:
    config = deepcopy(training_config())
    del config[container][leaf]  # type: ignore[index]
    with pytest.raises(
        ArtifactCompatibilityError,
        match=rf"training_config\.{container}.*missing=.*{leaf}.*expected=.*actual=",
    ):
        if boundary == "direct":
            artifact_identity_with_config(config)
        else:
            payload = artifact_identity().to_dict()
            payload["training_config"] = config
            payload["training_config_hash"] = sha256_json(config)
            ArtifactIdentity.from_dict(payload)


def test_full_nested_training_config_is_order_independent_and_round_trips() -> None:
    config = training_config()
    assert len(NESTED_BEHAVIOR_CONFIG_PATHS) == 52
    reordered = {
        key: dict(reversed(list(value.items()))) if isinstance(value, dict) else value
        for key, value in reversed(list(config.items()))
    }
    left = artifact_identity_with_config(config)
    right = artifact_identity_with_config(reordered)
    assert left == right
    assert left.training_config_hash == right.training_config_hash
    assert ArtifactIdentity.from_dict(left.to_dict()) == left


@pytest.mark.parametrize(
    "missing",
    [
        "model",
        "batch_size",
        "train_steps",
        "max_formula_len",
        "reward",
        "entropy",
        "elite",
        "restart",
        "noise",
        "walk_forward",
        "cost_rate",
        "neutral_band",
        "random_seed",
    ],
)
def test_training_config_requires_every_v2_behavior_category(missing: str) -> None:
    config = training_config()
    del config[missing]
    with pytest.raises(ArtifactCompatibilityError, match=missing):
        artifact_identity_with_config(config)


def test_training_config_rejects_empty_summary() -> None:
    with pytest.raises(ArtifactCompatibilityError, match="missing"):
        artifact_identity_with_config({})


@pytest.mark.parametrize(
    "container",
    ["model", "reward", "entropy", "elite", "restart", "noise", "walk_forward"],
)
def test_training_config_rejects_empty_required_container(container: str) -> None:
    config = training_config()
    config[container] = {}
    with pytest.raises(ArtifactCompatibilityError, match=container):
        artifact_identity_with_config(config)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("model",), []),
        (("model", "input_dim"), False),
        (("model", "hidden_dim"), 0),
        (("batch_size",), False),
        (("batch_size",), 0),
        (("train_steps",), "9000"),
        (("max_formula_len",), -1),
        (("reward",), []),
        (("reward", "ic"), False),
        (("reward", "return"), -0.1),
        (("entropy", "floor_enabled"), 1),
        (("elite", "size"), 0),
        (("restart", "stagnation_window"), False),
        (("noise", "initial"), -0.1),
        (("walk_forward", "blocks"), 0),
        (("walk_forward", "gap"), -1),
        (("walk_forward", "min_fold_bars"), 0),
        (("cost_rate",), -0.0001),
        (("neutral_band",), 1.1),
        (("random_seed",), "not-an-int"),
    ],
)
def test_training_config_rejects_wrong_type_or_obvious_range(
    path: tuple[str, ...],
    value: object,
) -> None:
    config = deepcopy(training_config())
    if len(path) == 1:
        config[path[0]] = value
    else:
        config[path[0]][path[1]] = value  # type: ignore[index]
    with pytest.raises(ArtifactCompatibilityError, match=path[-1]):
        artifact_identity_with_config(config)


class RewardModeLike(str):
    pass


INVALID_REWARD_MODES = (
    pytest.param("", "''", id="empty"),
    pytest.param(" ", "' '", id="whitespace"),
    pytest.param("totally-unknown-mode", "'totally-unknown-mode'", id="unknown"),
    pytest.param("STANDARD", "'STANDARD'", id="uppercase"),
    pytest.param("FtMo", "'FtMo'", id="mixed-case"),
    pytest.param(" standard", "' standard'", id="leading-space"),
    pytest.param("standard ", "'standard '", id="trailing-space"),
    pytest.param(1, "1", id="int"),
    pytest.param(True, "True", id="bool"),
    pytest.param(None, "None", id="none"),
    pytest.param([], "<list length=0>", id="list"),
    pytest.param(RewardModeLike("standard"), "<RewardModeLike>", id="str-like"),
)
REWARD_MODE_ERROR = (
    "training_config.reward.mode is invalid: "
    "expected=exactly one of ['standard', 'ftmo', 'forex'] actual="
)


def config_with_reward_mode(mode: object) -> dict[str, object]:
    config = deepcopy(training_config())
    config["reward"]["mode"] = mode  # type: ignore[index]
    return config


def bypass_reward_mode(identity: ArtifactIdentity, mode: object) -> None:
    config = config_with_reward_mode(mode)
    object.__setattr__(identity, "training_config", config)
    object.__setattr__(identity, "training_config_hash", "0" * 64)


@pytest.mark.parametrize("mode", ["standard", "ftmo", "forex"])
@pytest.mark.parametrize("boundary", ["direct", "from_dict", "to_dict"])
def test_current_reward_modes_round_trip_at_artifact_identity_boundaries(
    boundary: str,
    mode: str,
) -> None:
    config = config_with_reward_mode(mode)
    if boundary == "direct":
        identity = artifact_identity_with_config(config)
    elif boundary == "from_dict":
        payload = artifact_identity().to_dict()
        payload["training_config"] = config
        payload["training_config_hash"] = sha256_json(config)
        identity = ArtifactIdentity.from_dict(payload)
    else:
        identity = artifact_identity_with_config(config)
        identity = ArtifactIdentity.from_dict(identity.to_dict())
    assert identity.training_config["reward"]["mode"] == mode  # type: ignore[index]
    assert ArtifactIdentity.from_dict(identity.to_dict()) == identity


@pytest.mark.parametrize("mode,actual", INVALID_REWARD_MODES)
@pytest.mark.parametrize("boundary", ["direct", "from_dict", "to_dict"])
def test_artifact_identity_reward_mode_is_a_closed_set_at_every_boundary(
    boundary: str,
    mode: object,
    actual: str,
) -> None:
    config = config_with_reward_mode(mode)
    with pytest.raises(ArtifactCompatibilityError) as exc_info:
        if boundary == "direct":
            artifact_identity_with_config(config, config_hash="0" * 64)
        elif boundary == "from_dict":
            payload = artifact_identity().to_dict()
            payload["training_config"] = config
            payload["training_config_hash"] = "0" * 64
            ArtifactIdentity.from_dict(payload)
        else:
            identity = artifact_identity()
            bypass_reward_mode(identity, mode)
            identity.to_dict()
    if isinstance(mode, str) and type(mode) is not str:
        assert "exact built-in string" in str(exc_info.value)
    else:
        assert REWARD_MODE_ERROR + actual in str(exc_info.value)


@pytest.mark.parametrize(
    "boundary",
    ["create", "direct", "from_dict", "to_dict"],
)
def test_training_run_revalidates_fingerprint_valid_unknown_reward_mode(
    boundary: str,
) -> None:
    mode = "totally-unknown-mode"
    if boundary == "from_dict":
        run_payload = TrainingRunIdentity.create(artifact_identity()).to_dict()
        identity_payload = run_payload["artifact_identity"]
        config = config_with_reward_mode(mode)
        identity_payload["training_config"] = config  # type: ignore[index]
        identity_payload["training_config_hash"] = sha256_json(config)  # type: ignore[index]
        action = lambda: TrainingRunIdentity.from_dict(run_payload)
    else:
        identity = artifact_identity()
        config = config_with_reward_mode(mode)
        object.__setattr__(identity, "training_config", config)
        object.__setattr__(identity, "training_config_hash", sha256_json(config))
        if boundary == "create":
            action = lambda: TrainingRunIdentity.create(identity)
        elif boundary == "direct":
            action = lambda: TrainingRunIdentity(
                run_id="1" * 32,
                artifact_identity=identity,
            )
        else:
            run = TrainingRunIdentity.create(artifact_identity())
            object.__setattr__(run, "artifact_identity", identity)
            action = run.to_dict
    with pytest.raises(
        ArtifactCompatibilityError,
        match=r"training_config\.reward\.mode.*standard.*ftmo.*forex",
    ):
        action()


@pytest.mark.parametrize(
    ("path", "actual", "expected"),
    [
        (("model", "input_dim"), FORMULA_VOCAB.feature_count + 1, FORMULA_VOCAB.feature_count),
        (("walk_forward", "label_lookahead"), LABEL_LOOKAHEAD_BARS + 1, LABEL_LOOKAHEAD_BARS),
    ],
)
@pytest.mark.parametrize("boundary", ["direct", "from_dict", "to_dict"])
def test_artifact_identity_rejects_non_authoritative_semantics_at_every_boundary(
    boundary: str,
    path: tuple[str, str],
    actual: int,
    expected: int,
) -> None:
    config = deepcopy(training_config())
    config[path[0]][path[1]] = actual  # type: ignore[index]
    pattern = rf"training_config\.{path[0]}\.{path[1]}.*expected={expected}.*actual={actual}"

    if boundary == "direct":
        with pytest.raises(ArtifactCompatibilityError, match=pattern):
            artifact_identity_with_config(config)
    elif boundary == "from_dict":
        payload = artifact_identity().to_dict()
        payload["training_config"] = config
        payload["training_config_hash"] = sha256_json(config)
        with pytest.raises(ArtifactCompatibilityError, match=pattern):
            ArtifactIdentity.from_dict(payload)
    else:
        identity = artifact_identity()
        object.__setattr__(identity, "training_config", config)
        object.__setattr__(identity, "training_config_hash", sha256_json(config))
        with pytest.raises(ArtifactCompatibilityError, match=pattern):
            identity.to_dict()


@pytest.mark.parametrize(
    ("path", "actual", "expected"),
    [
        (("model", "input_dim"), FORMULA_VOCAB.feature_count + 1, FORMULA_VOCAB.feature_count),
        (("walk_forward", "label_lookahead"), LABEL_LOOKAHEAD_BARS + 1, LABEL_LOOKAHEAD_BARS),
    ],
)
@pytest.mark.parametrize("boundary", ["create", "direct", "from_dict", "to_dict"])
def test_strategy_boundaries_revalidate_authoritative_training_semantics(
    boundary: str,
    path: tuple[str, str],
    actual: int,
    expected: int,
) -> None:
    pattern = rf"training_config\.{path[0]}\.{path[1]}.*expected={expected}.*actual={actual}"
    if boundary == "from_dict":
        payload = strategy_artifact().to_dict()
        config = deepcopy(training_config())
        config[path[0]][path[1]] = actual  # type: ignore[index]
        artifact_payload = payload["run_identity"]["artifact_identity"]  # type: ignore[index]
        artifact_payload["training_config"] = config
        artifact_payload["training_config_hash"] = sha256_json(config)
        payload["fingerprint"] = sha256_json(
            {key: value for key, value in payload.items() if key != "fingerprint"}
        )
        with pytest.raises(ArtifactCompatibilityError, match=pattern):
            StrategyArtifact.from_dict(payload)
        return

    current = strategy_artifact()
    config = deepcopy(training_config())
    config[path[0]][path[1]] = actual  # type: ignore[index]
    identity = current.run_identity.artifact_identity
    object.__setattr__(identity, "training_config", config)
    object.__setattr__(identity, "training_config_hash", sha256_json(config))
    with pytest.raises(ArtifactCompatibilityError, match=pattern):
        if boundary == "create":
            StrategyArtifact.create(
                run_identity=current.run_identity,
                formula_tokens=[0],
                decoded_formula=FORMULA_VOCAB.token_names[0],
                best_score=1.0,
                fold_evidence=[],
                generated_at="2026-07-15T00:00:00Z",
            )
        elif boundary == "direct":
            StrategyArtifact(
                schema_version="strategy-v2",
                run_identity=current.run_identity,
                formula_tokens=(0,),
                decoded_formula=FORMULA_VOCAB.token_names[0],
                best_score=1.0,
                fold_evidence=(),
                generated_at="2026-07-15T00:00:00Z",
                fingerprint="0" * 64,
            )
        else:
            current.to_dict()


def test_training_config_rejects_non_finite_behavior_number() -> None:
    config = deepcopy(training_config())
    config["noise"]["initial"] = math.inf  # type: ignore[index]
    with pytest.raises(ArtifactCompatibilityError, match="initial"):
        artifact_identity_with_config(config, config_hash="0" * 64)


@pytest.mark.parametrize("alias", ["steps", "formula_length", "seed"])
def test_training_config_rejects_top_level_alias_representation(alias: str) -> None:
    config = training_config()
    config[alias] = config[
        {"steps": "train_steps", "formula_length": "max_formula_len", "seed": "random_seed"}[alias]
    ]
    with pytest.raises(ArtifactCompatibilityError, match=alias):
        artifact_identity_with_config(config)


def test_training_config_preserves_typed_behavior_strings_booleans_and_lists() -> None:
    config = deepcopy(training_config())
    config["reward"].update(  # type: ignore[union-attr]
        {"mode": "ftmo", "ema_baseline": True}
    )
    config["elite"]["decay"] = True  # type: ignore[index]
    config["restart"].update(  # type: ignore[union-attr]
        {"partial_reset": True, "partial_reset_layers": ["ln_f", "blocks"]}
    )
    config["noise"]["adaptive"] = True  # type: ignore[index]
    identity = artifact_identity_with_config(config)
    assert ArtifactIdentity.from_dict(identity.to_dict()) == identity


@pytest.mark.parametrize("forbidden", ["device", "data_path", "checkpoint_dir", "source_path"])
def test_training_identity_rejects_environment_specific_config(forbidden: str) -> None:
    config = training_config()
    config[forbidden] = "C:/local-only"
    with pytest.raises(ArtifactCompatibilityError, match=forbidden):
        ArtifactIdentity(
            core_semantics_version=CORE_SEMANTICS_VERSION,
            vocab_version=VOCAB_VERSION,
            label_semantics_version=LABEL_SEMANTICS_VERSION,
            execution_semantics_version=EXECUTION_SEMANTICS_VERSION,
            symbol="EURUSD",
            timeframe="H1",
            training_dataset=dataset_identity(),
            training_config=config,
            training_config_hash=sha256_json(config),
        )


def test_training_identity_rejects_obvious_non_behavior_config() -> None:
    config = training_config()
    config["notes"] = "analyst comment"
    with pytest.raises(ArtifactCompatibilityError, match="notes"):
        ArtifactIdentity(
            core_semantics_version=CORE_SEMANTICS_VERSION,
            vocab_version=VOCAB_VERSION,
            label_semantics_version=LABEL_SEMANTICS_VERSION,
            execution_semantics_version=EXECUTION_SEMANTICS_VERSION,
            symbol="EURUSD",
            timeframe="H1",
            training_dataset=dataset_identity(),
            training_config=config,
            training_config_hash=sha256_json(config),
        )


def test_artifact_identity_is_immutable_and_round_trips() -> None:
    identity = artifact_identity()
    with pytest.raises(TypeError):
        identity.training_config["random_seed"] = 99  # type: ignore[index]
    assert ArtifactIdentity.from_dict(identity.to_dict()) == identity
    assert ArtifactIdentity.from_dict(dict(reversed(list(identity.to_dict().items())))).fingerprint == identity.fingerprint


def test_artifact_identity_snapshots_each_dataset_field_exactly_once() -> None:
    source, reads = instrumented_dataset_identity()
    config = training_config()
    identity = ArtifactIdentity(
        core_semantics_version=CORE_SEMANTICS_VERSION,
        vocab_version=VOCAB_VERSION,
        label_semantics_version=LABEL_SEMANTICS_VERSION,
        execution_semantics_version=EXECUTION_SEMANTICS_VERSION,
        symbol="EURUSD",
        timeframe="H1",
        training_dataset=source,
        training_config=config,
        training_config_hash=sha256_json(config),
    )
    assert identity.training_dataset is not source
    assert reads == {field: 1 for field in DATASET_FIELDS}


@pytest.mark.parametrize("field", DATASET_FIELDS)
def test_artifact_identity_wraps_each_dataset_field_access_failure_with_cause(
    field: str,
) -> None:
    failure = RuntimeError(f"getter failed for {field}")
    source, _ = instrumented_dataset_identity(
        failing_field=field,
        failure=failure,
    )
    config = training_config()
    with pytest.raises(ArtifactCompatibilityError) as exc_info:
        ArtifactIdentity(
            core_semantics_version=CORE_SEMANTICS_VERSION,
            vocab_version=VOCAB_VERSION,
            label_semantics_version=LABEL_SEMANTICS_VERSION,
            execution_semantics_version=EXECUTION_SEMANTICS_VERSION,
            symbol="EURUSD",
            timeframe="H1",
            training_dataset=source,
            training_config=config,
            training_config_hash=sha256_json(config),
        )
    message = str(exc_info.value)
    assert "training_dataset" in message
    assert f"field={field}" in message
    assert "RuntimeError" in message
    assert exc_info.value.__cause__ is failure


@pytest.mark.parametrize("consumer", ["to_dict", "verify", "training_run"])
@pytest.mark.parametrize("field", DATASET_FIELDS)
def test_identity_consumers_wrap_hostile_dataset_getters(
    consumer: str,
    field: str,
) -> None:
    failure = RuntimeError(f"getter failed for {field}")
    source, _ = instrumented_dataset_identity(
        failing_field=field,
        failure=failure,
    )
    identity = artifact_identity()
    object.__setattr__(identity, "training_dataset", source)
    with pytest.raises(ArtifactCompatibilityError) as exc_info:
        if consumer == "to_dict":
            identity.to_dict()
        elif consumer == "verify":
            verify_artifact_identity(identity, artifact_identity())
        else:
            TrainingRunIdentity.create(identity)
    assert f"field={field}" in str(exc_info.value)
    assert exc_info.value.__cause__ is failure


@pytest.mark.parametrize("boundary", ["direct", "from_dict", "to_dict", "verify"])
@pytest.mark.parametrize("error_type", [AssertionError, RuntimeError])
def test_unexpected_timeframe_helper_exceptions_propagate(
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
    error_type: type[Exception],
) -> None:
    failure = error_type(f"unexpected helper failure at {boundary}")

    def fail_normalization(value: object) -> str:
        raise failure

    identity = artifact_identity()
    payload = identity.to_dict()
    monkeypatch.setattr(artifacts_module, "normalize_timeframe_name", fail_normalization)
    with pytest.raises(error_type) as exc_info:
        if boundary == "direct":
            config = training_config()
            ArtifactIdentity(
                core_semantics_version=CORE_SEMANTICS_VERSION,
                vocab_version=VOCAB_VERSION,
                label_semantics_version=LABEL_SEMANTICS_VERSION,
                execution_semantics_version=EXECUTION_SEMANTICS_VERSION,
                symbol="EURUSD",
                timeframe="H1",
                training_dataset=dataset_identity(),
                training_config=config,
                training_config_hash=sha256_json(config),
            )
        elif boundary == "from_dict":
            ArtifactIdentity.from_dict(payload)
        elif boundary == "to_dict":
            identity.to_dict()
        else:
            verify_artifact_identity(identity, artifact_identity())
    assert exc_info.value is failure


@pytest.mark.parametrize("boundary", ["direct", "from_dict", "to_dict", "verify"])
def test_documented_timeframe_validation_error_is_translated_with_exact_cause(
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
) -> None:
    failure = DataValidationError(f"invalid timeframe at {boundary}")

    def fail_normalization(value: object) -> str:
        raise failure

    identity = artifact_identity()
    payload = identity.to_dict()
    monkeypatch.setattr(artifacts_module, "normalize_timeframe_name", fail_normalization)
    with pytest.raises(ArtifactCompatibilityError) as exc_info:
        if boundary == "direct":
            config = training_config()
            ArtifactIdentity(
                core_semantics_version=CORE_SEMANTICS_VERSION,
                vocab_version=VOCAB_VERSION,
                label_semantics_version=LABEL_SEMANTICS_VERSION,
                execution_semantics_version=EXECUTION_SEMANTICS_VERSION,
                symbol="EURUSD",
                timeframe="H1",
                training_dataset=dataset_identity(),
                training_config=config,
                training_config_hash=sha256_json(config),
            )
        elif boundary == "from_dict":
            ArtifactIdentity.from_dict(payload)
        elif boundary == "to_dict":
            identity.to_dict()
        else:
            verify_artifact_identity(identity, artifact_identity())
    assert "timeframe" in str(exc_info.value)
    assert exc_info.value.__cause__ is failure


@pytest.mark.parametrize("render_mode", ["oversized", "throwing"])
def test_artifact_diagnostics_do_not_render_unknown_objects_and_are_bounded(
    render_mode: str,
) -> None:
    calls = {"repr": 0, "str": 0}

    class HostileDiagnostic:
        def __repr__(self) -> str:
            calls["repr"] += 1
            if render_mode == "throwing":
                raise RuntimeError("hostile repr")
            return "R" * 2_000_000

        def __str__(self) -> str:
            calls["str"] += 1
            if render_mode == "throwing":
                raise RuntimeError("hostile str")
            return "S" * 2_000_000

    source = dataset_identity()
    object.__setattr__(source, "data_fingerprint", HostileDiagnostic())
    config = training_config()
    with pytest.raises(ArtifactCompatibilityError) as exc_info:
        ArtifactIdentity(
            core_semantics_version=CORE_SEMANTICS_VERSION,
            vocab_version=VOCAB_VERSION,
            label_semantics_version=LABEL_SEMANTICS_VERSION,
            execution_semantics_version=EXECUTION_SEMANTICS_VERSION,
            symbol="EURUSD",
            timeframe="H1",
            training_dataset=source,
            training_config=config,
            training_config_hash=sha256_json(config),
        )
    message = str(exc_info.value)
    assert "data_fingerprint" in message
    assert len(message) <= 1_024
    assert calls == {"repr": 0, "str": 0}


def test_oversized_primitive_string_diagnostic_is_bounded() -> None:
    source = bypass_dataset_identity(data_fingerprint="x" * 2_000_000)
    config = training_config()
    with pytest.raises(ArtifactCompatibilityError) as exc_info:
        ArtifactIdentity(
            core_semantics_version=CORE_SEMANTICS_VERSION,
            vocab_version=VOCAB_VERSION,
            label_semantics_version=LABEL_SEMANTICS_VERSION,
            execution_semantics_version=EXECUTION_SEMANTICS_VERSION,
            symbol="EURUSD",
            timeframe="H1",
            training_dataset=source,
            training_config=config,
            training_config_hash=sha256_json(config),
        )
    assert len(str(exc_info.value)) <= 1_024


def test_artifact_identity_rejects_wrong_config_hash() -> None:
    payload = artifact_identity().to_dict()
    payload["training_config_hash"] = "0" * 64
    with pytest.raises(ArtifactCompatibilityError, match="training_config_hash.*expected=.*actual="):
        ArtifactIdentity.from_dict(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", "ohlcv-v1"),
        ("symbol", ""),
        ("timeframe", "h1"),
        ("start_time_ns", True),
        ("end_time_ns", False),
        ("bars", True),
        ("bars", -1),
        ("data_fingerprint", 123),
        ("data_fingerprint", "a" * 12),
        ("data_fingerprint", "G" * 64),
        ("time_fingerprint", "b" * 63),
    ],
)
def test_artifact_identity_strictly_rejects_malformed_nested_dataset(
    field: str,
    value: object,
) -> None:
    payload = artifact_identity().to_dict()
    payload["training_dataset"][field] = value  # type: ignore[index]
    with pytest.raises(ArtifactCompatibilityError, match=field):
        ArtifactIdentity.from_dict(payload)


def test_artifact_identity_rejects_unknown_nested_dataset_field() -> None:
    payload = artifact_identity().to_dict()
    payload["training_dataset"]["legacy_path"] = "C:/data.parquet"  # type: ignore[index]
    with pytest.raises(ArtifactCompatibilityError, match="unknown"):
        ArtifactIdentity.from_dict(payload)


def test_artifact_identity_rejects_reversed_nested_dataset_range() -> None:
    payload = artifact_identity().to_dict()
    payload["training_dataset"]["start_time_ns"] = 10_000  # type: ignore[index]
    payload["training_dataset"]["end_time_ns"] = 1_000  # type: ignore[index]
    with pytest.raises(ArtifactCompatibilityError, match="range"):
        ArtifactIdentity.from_dict(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("core_semantics_version", "1"),
        ("vocab_version", "v-old"),
        ("label_semantics_version", "label-old"),
        ("execution_semantics_version", "execution-old"),
        ("symbol", "GBPUSD"),
        ("timeframe", "H4"),
        ("training_config_hash", "f" * 64),
    ],
)
def test_verify_artifact_identity_reports_each_mismatch(field: str, value: str) -> None:
    expected = artifact_identity()
    if field in {
        "core_semantics_version",
        "vocab_version",
        "label_semantics_version",
        "execution_semantics_version",
    }:
        actual = artifact_identity()
        object.__setattr__(actual, field, value)
        with pytest.raises(ArtifactCompatibilityError) as exc_info:
            verify_artifact_identity(expected, actual)
        message = str(exc_info.value)
        assert field in message
        assert "expected=" in message
        assert "actual=" in message
        return
    changes: dict[str, object] = {field: value}
    if field == "symbol":
        changes["training_dataset"] = replace(expected.training_dataset, symbol=value)
    elif field == "timeframe":
        changes["training_dataset"] = replace(expected.training_dataset, timeframe=value)
    elif field == "training_config_hash":
        changed_config = training_config(seed=43)
        changes["training_config"] = changed_config
        changes["training_config_hash"] = sha256_json(changed_config)
    actual = replace(expected, **changes)
    with pytest.raises(ArtifactCompatibilityError) as exc_info:
        verify_artifact_identity(expected, actual)
    message = str(exc_info.value)
    assert field in message
    assert "expected=" in message
    assert "actual=" in message


@pytest.mark.parametrize("expected", [None, {}, "identity"])
@pytest.mark.parametrize("actual", [None, {}, "identity"])
def test_verify_artifact_identity_rejects_non_identity_types(
    expected: object,
    actual: object,
) -> None:
    with pytest.raises(ArtifactCompatibilityError, match="ArtifactIdentity"):
        verify_artifact_identity(expected, actual)  # type: ignore[arg-type]


@pytest.mark.parametrize("consumer", ["to_dict", "fingerprint"])
@pytest.mark.parametrize(
    ("mutation", "error_pattern"),
    IDENTITY_MUTATIONS,
)
def test_artifact_identity_public_consumers_revalidate_bypass_mutations(
    consumer: str,
    mutation: str,
    error_pattern: str,
) -> None:
    identity = artifact_identity()
    mutate_artifact_identity(identity, mutation)
    with pytest.raises(ArtifactCompatibilityError, match=error_pattern):
        identity.to_dict() if consumer == "to_dict" else identity.fingerprint


@pytest.mark.parametrize("side", ["expected", "actual", "same"])
@pytest.mark.parametrize(
    ("mutation", "error_pattern"),
    IDENTITY_MUTATIONS,
)
def test_verify_artifact_identity_revalidates_both_operands(
    side: str,
    mutation: str,
    error_pattern: str,
) -> None:
    expected = artifact_identity()
    actual = artifact_identity()
    if side == "same":
        mutate_artifact_identity(expected, mutation)
        actual = expected
    elif side == "expected":
        mutate_artifact_identity(expected, mutation)
    else:
        mutate_artifact_identity(actual, mutation)
    with pytest.raises(ArtifactCompatibilityError, match=error_pattern):
        verify_artifact_identity(expected, actual)


@pytest.mark.parametrize("constructor", ["create", "direct"])
@pytest.mark.parametrize(
    ("mutation", "error_pattern"),
    IDENTITY_MUTATIONS,
)
def test_training_run_construction_revalidates_artifact_identity(
    constructor: str,
    mutation: str,
    error_pattern: str,
) -> None:
    identity = artifact_identity()
    mutate_artifact_identity(identity, mutation)
    with pytest.raises(ArtifactCompatibilityError, match=error_pattern):
        if constructor == "create":
            TrainingRunIdentity.create(identity)
        else:
            TrainingRunIdentity(run_id="1" * 32, artifact_identity=identity)


@pytest.mark.parametrize(
    "consumer",
    ["to_dict", "checkpoint_filename", "strategy_filename", "history_filename"],
)
@pytest.mark.parametrize(
    ("mutation", "error_pattern"),
    IDENTITY_MUTATIONS,
)
def test_training_run_public_consumers_revalidate_bypass_mutations(
    consumer: str,
    mutation: str,
    error_pattern: str,
) -> None:
    run = TrainingRunIdentity.create(artifact_identity())
    mutate_artifact_identity(run.artifact_identity, mutation)
    with pytest.raises(ArtifactCompatibilityError, match=error_pattern):
        if consumer == "to_dict":
            run.to_dict()
        elif consumer == "checkpoint_filename":
            run.checkpoint_filename(1)
        elif consumer == "strategy_filename":
            run.strategy_filename()
        else:
            run.history_filename()


@pytest.mark.parametrize("constructor", ["create", "direct"])
@pytest.mark.parametrize("mutation", [item[0] for item in IDENTITY_MUTATIONS])
def test_training_run_snapshots_source_identity_before_later_bypass_mutation(
    constructor: str,
    mutation: str,
) -> None:
    source = artifact_identity()
    if constructor == "create":
        run = TrainingRunIdentity.create(source)
    else:
        run = TrainingRunIdentity(run_id="1" * 32, artifact_identity=source)
    before = run.to_dict()
    mutate_artifact_identity(source, mutation)
    assert run.artifact_identity is not source
    assert run.to_dict() == before
    assert run.checkpoint_filename(1).startswith("ckpt_v2_")
    assert run.strategy_filename().startswith("best_v2_")
    assert run.history_filename().startswith("training_history_v2_")


def test_training_run_uuid_and_immutable_filenames_include_identity() -> None:
    run = TrainingRunIdentity.create(artifact_identity())
    assert len(run.run_id) == 32
    int(run.run_id, 16)
    assert run.checkpoint_filename(17) == f"ckpt_v2_EURUSD_H1_{'a' * 12}_run_{run.run_id[:8]}_step_17.pt"
    assert run.strategy_filename() == f"best_v2_EURUSD_H1_{'a' * 12}_run_{run.run_id[:8]}.json"
    assert run.history_filename() == f"training_history_v2_EURUSD_H1_{'a' * 12}_run_{run.run_id[:8]}.json"


@pytest.mark.parametrize("run_id", ["", "not-a-uuid", "a" * 31, "g" * 32])
def test_training_run_rejects_non_uuid_hex(run_id: str) -> None:
    with pytest.raises(ArtifactCompatibilityError, match="run_id"):
        TrainingRunIdentity(run_id=run_id, artifact_identity=artifact_identity())


def test_filename_revalidates_internal_dataset_fingerprint() -> None:
    run = TrainingRunIdentity.create(artifact_identity())
    object.__setattr__(
        run.artifact_identity,
        "training_dataset",
        bypass_dataset_identity(
            run.artifact_identity.training_dataset,
            data_fingerprint="short",
        ),
    )
    with pytest.raises(ArtifactCompatibilityError, match="data_fingerprint"):
        run.checkpoint_filename(1)


def test_strategy_round_trip_and_fingerprint_cover_full_payload() -> None:
    strategy = strategy_artifact()
    assert StrategyArtifact.from_dict(strategy.to_dict()) == strategy
    reordered = dict(reversed(list(strategy.to_dict().items())))
    assert StrategyArtifact.from_dict(reordered).fingerprint == strategy.fingerprint

    changed = strategy.to_dict()
    changed["best_score"] = 99.0
    with pytest.raises(ArtifactCompatibilityError, match="fingerprint"):
        StrategyArtifact.from_dict(changed)


@pytest.mark.parametrize(
    ("mutation", "message"),
    STRATEGY_PUBLIC_MUTATIONS,
)
def test_strategy_to_dict_revalidates_every_public_field_after_bypass_mutation(
    mutation: str,
    message: str,
) -> None:
    strategy = strategy_artifact()
    mutate_strategy_public_state(strategy, mutation)
    with pytest.raises(ArtifactCompatibilityError, match=message):
        strategy.to_dict()


def test_strategy_to_dict_rejects_bool_token_even_with_recomputed_fingerprint() -> None:
    strategy = strategy_artifact()
    object.__setattr__(strategy, "formula_tokens", (True,))
    object.__setattr__(
        strategy,
        "fingerprint",
        sha256_json(strategy._payload_without_fingerprint()),
    )
    with pytest.raises(ArtifactCompatibilityError, match="formula token"):
        strategy.to_dict()


def test_strategy_direct_construction_rejects_well_shaped_wrong_fingerprint() -> None:
    strategy = strategy_artifact()
    with pytest.raises(
        ArtifactCompatibilityError,
        match="strategy fingerprint mismatch.*expected=.*actual=",
    ):
        replace(strategy, fingerprint="0" * 64)


@pytest.mark.parametrize("construction", ["direct", "create", "from_dict"])
def test_strategy_construction_with_matching_fingerprint_round_trips(
    construction: str,
) -> None:
    created = strategy_artifact()
    if construction == "direct":
        strategy = direct_strategy_artifact(
            run_identity=created.run_identity,
            fold_evidence=created.fold_evidence,
        )
    elif construction == "create":
        strategy = created
    else:
        strategy = StrategyArtifact.from_dict(created.to_dict())

    assert StrategyArtifact.from_dict(strategy.to_dict()) == strategy


@pytest.mark.parametrize(
    "metrics",
    [
        {},
        {"ic": "not-a-number"},
        {"ic": True},
        {"ic": {"nested": 1.0}},
        {"ic": [1.0]},
        {"ic": None},
        {"ic": float("nan")},
        {"ic": float("inf")},
        {"ic": float("-inf")},
        {"": 1.0},
    ],
    ids=[
        "empty",
        "string",
        "bool",
        "nested",
        "list",
        "null",
        "nan",
        "positive-inf",
        "negative-inf",
        "empty-key",
    ],
)
@pytest.mark.parametrize("construction", ["direct", "from_dict"])
def test_fold_evidence_requires_nonempty_finite_numeric_validation_metrics(
    metrics: dict[str, object],
    construction: str,
) -> None:
    payload = {
        "fold_index": 0,
        "train_start_time_ns": 1_000,
        "train_end_time_ns": 2_000,
        "val_start_time_ns": 2_500,
        "val_end_time_ns": 3_000,
        "effective_gap": LABEL_LOOKAHEAD_BARS,
        "validation_metrics": metrics,
    }
    with pytest.raises(ArtifactCompatibilityError, match="validation_metrics"):
        if construction == "direct":
            FoldEvidence(**payload)  # type: ignore[arg-type]
        else:
            FoldEvidence.from_dict(payload)


@pytest.mark.parametrize(
    "indices",
    [(0, 0), (1, 0), (0, 2), (1,)],
    ids=["duplicate", "out-of-order", "gapped", "nonzero-start"],
)
@pytest.mark.parametrize("construction", ["create", "direct", "from_dict"])
def test_strategy_requires_zero_based_contiguous_fold_indices(
    indices: tuple[int, ...],
    construction: str,
) -> None:
    folds_list = []
    for position, fold_index in enumerate(indices):
        fold = fold_evidence_with_index(position)
        object.__setattr__(fold, "fold_index", fold_index)
        folds_list.append(fold)
    folds = tuple(folds_list)
    with pytest.raises(
        ArtifactCompatibilityError,
        match="fold_index.*expected=.*actual=",
    ):
        construct_strategy_with_folds(construction, folds)


@pytest.mark.parametrize("indices", [(0, 1, 2, 3)])
@pytest.mark.parametrize("construction", ["create", "direct", "from_dict"])
def test_strategy_accepts_contiguous_fold_indices_and_round_trips(
    indices: tuple[int, ...],
    construction: str,
) -> None:
    folds = tuple(fold_evidence_with_index(index) for index in indices)
    strategy = construct_strategy_with_folds(construction, folds)
    assert [fold.fold_index for fold in strategy.fold_evidence] == list(indices)
    assert StrategyArtifact.from_dict(strategy.to_dict()) == strategy


@pytest.mark.parametrize("construction", ["create", "direct", "from_dict"])
@pytest.mark.parametrize(
    ("mutation", "field", "actual"),
    [
        ("moving-train-start", "train_start_time_ns", 1_001),
        ("shrinking-train-end", "train_end_time_ns", 1_999),
        ("omits-prior-validation-history", "train_end_time_ns", 2_900),
        ("duplicate-validation", "val_start_time_ns", 2_500),
        ("earlier-validation", "val_start_time_ns", 2_400),
    ],
)
def test_strategy_rejects_non_expanding_cross_fold_lineage_even_when_rehashed(
    construction: str,
    mutation: str,
    field: str,
    actual: int,
) -> None:
    first = fold_evidence_with_index(0)
    second = fold_evidence_with_index(1)
    if mutation == "moving-train-start":
        object.__setattr__(second, "train_start_time_ns", actual)
    elif mutation == "shrinking-train-end":
        object.__setattr__(second, "train_end_time_ns", actual)
    elif mutation == "omits-prior-validation-history":
        object.__setattr__(second, "train_end_time_ns", actual)
    elif mutation == "duplicate-validation":
        object.__setattr__(second, "train_end_time_ns", 2_400)
        object.__setattr__(second, "val_start_time_ns", actual)
        object.__setattr__(second, "val_end_time_ns", 3_000)
    elif mutation == "earlier-validation":
        object.__setattr__(second, "train_end_time_ns", 2_300)
        object.__setattr__(second, "val_start_time_ns", actual)
        object.__setattr__(second, "val_end_time_ns", 2_900)
    else:
        raise AssertionError(f"unknown mutation: {mutation}")

    with pytest.raises(
        ArtifactCompatibilityError,
        match=rf"fold_evidence\[1\]\.{field}.*expected=.*actual={actual}",
    ):
        construct_strategy_with_folds(construction, (first, second))


@pytest.mark.parametrize("construction", ["create", "direct", "from_dict"])
def test_strategy_accepts_expanding_multi_fold_lineage_and_round_trips(
    construction: str,
) -> None:
    folds = (
        FoldEvidence(0, 1_000, 2_000, 2_500, 3_000, 2, {"ic": 0.10}),
        FoldEvidence(1, 1_000, 3_200, 3_500, 4_000, 2, {"ic": 0.11}),
        FoldEvidence(2, 1_000, 4_250, 4_500, 5_000, 2, {"ic": 0.12}),
    )
    artifact = construct_strategy_with_folds(construction, folds)
    assert StrategyArtifact.from_dict(artifact.to_dict()) == artifact


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("fold_index", True, "fold_index"),
        ("train_start_time_ns", "bad", "train_start_time_ns"),
        ("train_start_time_ns", 5_000, "training time range"),
        ("val_end_time_ns", 5_000, "validation time range"),
        ("effective_gap", 0, "effective_gap"),
        ("validation_metrics", {"ic": True}, "validation_metrics.ic"),
    ],
)
def test_fold_evidence_to_dict_revalidates_bypass_mutation_matrix(
    field: str,
    value: object,
    message: str,
) -> None:
    fold = FoldEvidence(
        fold_index=0,
        train_start_time_ns=1_000,
        train_end_time_ns=4_000,
        val_start_time_ns=6_000,
        val_end_time_ns=9_000,
        effective_gap=LABEL_LOOKAHEAD_BARS,
        validation_metrics={"ic": 0.1},
    )
    object.__setattr__(fold, field, value)
    with pytest.raises(ArtifactCompatibilityError, match=message):
        fold.to_dict()


def test_strategy_create_snapshots_fold_metrics_and_index_from_source() -> None:
    source = fold_evidence_with_index(0)
    strategy = construct_strategy_with_folds("create", (source,))
    before = strategy.to_dict()
    object.__setattr__(source, "fold_index", 9)
    object.__setattr__(source, "validation_metrics", {"ic": "mutated"})
    assert strategy.to_dict() == before


@pytest.mark.parametrize("val_start_time_ns", [180, 170])
def test_fold_evidence_rejects_equal_or_overlapping_validation_start(
    val_start_time_ns: int,
) -> None:
    with pytest.raises(ArtifactCompatibilityError) as exc_info:
        FoldEvidence(
            fold_index=0,
            train_start_time_ns=100,
            train_end_time_ns=180,
            val_start_time_ns=val_start_time_ns,
            val_end_time_ns=220,
            effective_gap=2,
            validation_metrics={"ic": 0.1},
        )
    message = str(exc_info.value)
    assert "train_end_time_ns" in message
    assert "val_start_time_ns" in message
    assert "expected=" in message
    assert "actual=" in message


@pytest.mark.parametrize("effective_gap", [0, 1, True])
def test_fold_evidence_requires_semantic_label_lookahead_gap(
    effective_gap: object,
) -> None:
    with pytest.raises(
        ArtifactCompatibilityError,
        match="effective_gap.*expected=.*actual=",
    ):
        FoldEvidence(
            fold_index=0,
            train_start_time_ns=1_000,
            train_end_time_ns=4_000,
            val_start_time_ns=6_000,
            val_end_time_ns=9_000,
            effective_gap=effective_gap,  # type: ignore[arg-type]
            validation_metrics={"ic": 0.1},
        )


def test_fold_evidence_accepts_semantic_label_lookahead_gap() -> None:
    fold = FoldEvidence(
        fold_index=0,
        train_start_time_ns=1_000,
        train_end_time_ns=4_000,
        val_start_time_ns=6_000,
        val_end_time_ns=9_000,
        effective_gap=LABEL_LOOKAHEAD_BARS,
        validation_metrics={"ic": 0.1},
    )
    assert FoldEvidence.from_dict(fold.to_dict()) == fold


@pytest.mark.parametrize("construction", ["create", "direct", "from_dict"])
@pytest.mark.parametrize(
    ("times", "escaped_field"),
    [
        ((0, 500, 600, 950), "train_start_time_ns"),
        ((999, 4_000, 6_000, 9_000), "train_start_time_ns"),
        ((1_000, 4_000, 6_000, 9_001), "val_end_time_ns"),
        ((9_100, 9_200, 9_300, 9_400), "train_start_time_ns"),
    ],
)
def test_strategy_construction_rejects_fold_outside_training_dataset(
    construction: str,
    times: tuple[int, int, int, int],
    escaped_field: str,
) -> None:
    fold = FoldEvidence(
        fold_index=0,
        train_start_time_ns=times[0],
        train_end_time_ns=times[1],
        val_start_time_ns=times[2],
        val_end_time_ns=times[3],
        effective_gap=LABEL_LOOKAHEAD_BARS,
        validation_metrics={"ic": 0.1},
    )
    with pytest.raises(
        ArtifactCompatibilityError,
        match=rf"fold_evidence\[0\]\.{escaped_field}.*expected=.*actual=",
    ):
        construct_strategy_with_fold(construction, fold)


@pytest.mark.parametrize("construction", ["create", "direct", "from_dict"])
def test_strategy_construction_accepts_fold_equal_to_dataset_boundaries(
    construction: str,
) -> None:
    folds = tuple(fold_evidence_with_index(index) for index in range(4))
    object.__setattr__(folds[-1], "val_end_time_ns", 9_000)
    strategy = construct_strategy_with_folds(construction, folds)
    assert StrategyArtifact.from_dict(strategy.to_dict()) == strategy


def configured_gap_identity(gap: int) -> ArtifactIdentity:
    config = deepcopy(training_config())
    config["walk_forward"]["gap"] = gap  # type: ignore[index]
    return artifact_identity_with_config(config)


def fold_with_effective_gap(effective_gap: int) -> FoldEvidence:
    return FoldEvidence(
        fold_index=0,
        train_start_time_ns=1_000,
        train_end_time_ns=4_000,
        val_start_time_ns=6_000,
        val_end_time_ns=9_000,
        effective_gap=effective_gap,
        validation_metrics={"ic": 0.1},
    )


def four_folds_with_effective_gap(effective_gap: int) -> tuple[FoldEvidence, ...]:
    folds = tuple(fold_evidence_with_index(index) for index in range(4))
    for fold in folds:
        object.__setattr__(fold, "effective_gap", effective_gap)
    return folds


@pytest.mark.parametrize("construction", ["create", "direct", "from_dict"])
def test_strategy_construction_rejects_fold_gap_that_contradicts_training_config(
    construction: str,
) -> None:
    run = TrainingRunIdentity.create(configured_gap_identity(20))
    folds = four_folds_with_effective_gap(LABEL_LOOKAHEAD_BARS)
    pattern = r"fold_evidence\[0\]\.effective_gap.*expected=20.*actual=2"
    with pytest.raises(ArtifactCompatibilityError, match=pattern):
        if construction == "create":
            StrategyArtifact.create(
                run_identity=run,
                formula_tokens=[0],
                decoded_formula=FORMULA_VOCAB.token_names[0],
                best_score=1.0,
                fold_evidence=folds,
                generated_at="2026-07-15T00:00:00Z",
            )
        elif construction == "direct":
            direct_strategy_artifact(run_identity=run, fold_evidence=folds)
        else:
            payload = {
                "schema_version": "strategy-v2",
                "run_identity": run.to_dict(),
                "formula_tokens": [0],
                "decoded_formula": FORMULA_VOCAB.token_names[0],
                "best_score": 1.0,
                "fold_evidence": [item.to_dict() for item in folds],
                "generated_at": "2026-07-15T00:00:00Z",
            }
            StrategyArtifact.from_dict({**payload, "fingerprint": sha256_json(payload)})


def test_strategy_serialization_revalidates_configured_effective_gap() -> None:
    strategy = StrategyArtifact.create(
        run_identity=TrainingRunIdentity.create(configured_gap_identity(20)),
        formula_tokens=[0],
        decoded_formula=FORMULA_VOCAB.token_names[0],
        best_score=1.0,
        fold_evidence=four_folds_with_effective_gap(20),
        generated_at="2026-07-15T00:00:00Z",
    )
    object.__setattr__(strategy.fold_evidence[0], "effective_gap", LABEL_LOOKAHEAD_BARS)
    with pytest.raises(
        ArtifactCompatibilityError,
        match=r"fold_evidence\[0\]\.effective_gap.*expected=20.*actual=2",
    ):
        strategy.to_dict()


def test_strategy_accepts_configured_gap_and_rejects_missing_fold_evidence() -> None:
    run = TrainingRunIdentity.create(configured_gap_identity(20))
    with_fold = StrategyArtifact.create(
        run_identity=run,
        formula_tokens=[0],
        decoded_formula=FORMULA_VOCAB.token_names[0],
        best_score=1.0,
        fold_evidence=four_folds_with_effective_gap(20),
        generated_at="2026-07-15T00:00:00Z",
    )
    assert StrategyArtifact.from_dict(with_fold.to_dict()) == with_fold
    with pytest.raises(
        ArtifactCompatibilityError,
        match=r"fold_evidence count.*expected=4.*actual=0",
    ):
        StrategyArtifact.create(
            run_identity=run,
            formula_tokens=[0],
            decoded_formula=FORMULA_VOCAB.token_names[0],
            best_score=1.0,
            fold_evidence=[],
            generated_at="2026-07-15T00:00:00Z",
        )


def test_strategy_effective_gap_uses_label_when_configured_gap_is_smaller() -> None:
    strategy = StrategyArtifact.create(
        run_identity=TrainingRunIdentity.create(configured_gap_identity(0)),
        formula_tokens=[0],
        decoded_formula=FORMULA_VOCAB.token_names[0],
        best_score=1.0,
        fold_evidence=four_folds_with_effective_gap(LABEL_LOOKAHEAD_BARS),
        generated_at="2026-07-15T00:00:00Z",
    )
    assert StrategyArtifact.from_dict(strategy.to_dict()) == strategy


def test_single_bar_train_and_validation_ranges_round_trip_when_strictly_ordered() -> None:
    folds = tuple(
        FoldEvidence(
            fold_index=index,
            train_start_time_ns=1_000,
            train_end_time_ns=1_000 + index,
            val_start_time_ns=1_001 + index,
            val_end_time_ns=1_001 + index,
            effective_gap=LABEL_LOOKAHEAD_BARS,
            validation_metrics={"ic": 0.1},
        )
        for index in range(4)
    )
    artifact = StrategyArtifact.create(
        run_identity=TrainingRunIdentity.create(artifact_identity()),
        formula_tokens=[0],
        decoded_formula=FORMULA_VOCAB.token_names[0],
        best_score=1.0,
        fold_evidence=folds,
        generated_at="2026-07-15T00:00:00Z",
    )
    assert StrategyArtifact.from_dict(artifact.to_dict()) == artifact


@pytest.mark.parametrize("val_start_time_ns", [2_000, 1_999])
def test_strategy_load_rejects_rehashed_equal_or_overlapping_fold_lineage(
    val_start_time_ns: int,
) -> None:
    payload = strategy_artifact().to_dict()
    payload["fold_evidence"][0]["val_start_time_ns"] = val_start_time_ns  # type: ignore[index]
    payload["fingerprint"] = sha256_json(
        {key: item for key, item in payload.items() if key != "fingerprint"}
    )
    with pytest.raises(ArtifactCompatibilityError, match="train_end_time_ns"):
        StrategyArtifact.from_dict(payload)


@pytest.mark.parametrize("construction", ["create", "direct"])
def test_strategy_construction_revalidates_mutated_fold_before_fingerprint(
    construction: str,
) -> None:
    fold = FoldEvidence(
        fold_index=0,
        train_start_time_ns=1_000,
        train_end_time_ns=4_000,
        val_start_time_ns=6_000,
        val_end_time_ns=9_000,
        effective_gap=2,
        validation_metrics={"ic": 0.1},
    )
    object.__setattr__(fold, "val_start_time_ns", 3_999)
    with pytest.raises(ArtifactCompatibilityError, match="train_end_time_ns"):
        if construction == "create":
            StrategyArtifact.create(
                run_identity=TrainingRunIdentity.create(artifact_identity()),
                formula_tokens=[0],
                decoded_formula=FORMULA_VOCAB.token_names[0],
                best_score=1.0,
                fold_evidence=[fold],
                generated_at="2026-07-15T00:00:00Z",
            )
        else:
            direct_strategy_artifact(
                run_identity=TrainingRunIdentity.create(artifact_identity()),
                fold_evidence=(fold,),
            )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("core_semantics_version", "1"),
        ("vocab_version", "v-old"),
        ("label_semantics_version", "label-old"),
        ("execution_semantics_version", "execution-old"),
    ],
)
@pytest.mark.parametrize("construction", ["create", "direct"])
def test_strategy_construction_rejects_non_current_run_semantics(
    field: str,
    value: str,
    construction: str,
) -> None:
    run = TrainingRunIdentity.create(artifact_identity())
    object.__setattr__(run.artifact_identity, field, value)
    with pytest.raises(ArtifactCompatibilityError, match=field):
        if construction == "create":
            StrategyArtifact.create(
                run_identity=run,
                formula_tokens=[0],
                decoded_formula=FORMULA_VOCAB.token_names[0],
                best_score=1.0,
                fold_evidence=[],
                generated_at="2026-07-15T00:00:00Z",
            )
        else:
            StrategyArtifact(
                schema_version="strategy-v2",
                run_identity=run,
                formula_tokens=(0,),
                decoded_formula=FORMULA_VOCAB.token_names[0],
                best_score=1.0,
                fold_evidence=(),
                generated_at="2026-07-15T00:00:00Z",
                fingerprint="0" * 64,
            )


def test_strategy_create_snapshots_valid_supplied_fold_and_round_trips() -> None:
    fold = fold_evidence_with_index(0)
    artifact = construct_strategy_with_folds("create", (fold,))
    object.__setattr__(fold, "val_start_time_ns", 3_999)
    assert artifact.fold_evidence[0].val_start_time_ns == 2_500
    assert StrategyArtifact.from_dict(artifact.to_dict()) == artifact


@pytest.mark.parametrize("error_type", [AssertionError, RuntimeError])
@pytest.mark.parametrize(
    "boundary",
    ["create", "direct", "to_dict", "from_dict"],
)
def test_unexpected_fold_parser_exceptions_propagate_unchanged(
    monkeypatch: pytest.MonkeyPatch,
    error_type: type[Exception],
    boundary: str,
) -> None:
    failure = error_type("internal-fold-bug")
    fold = fold_evidence_with_index(0)
    current = strategy_artifact()
    payload = current.to_dict()

    def fail_fold_parse(
        cls: type[FoldEvidence],
        value: object,
    ) -> FoldEvidence:
        raise failure

    monkeypatch.setattr(
        FoldEvidence,
        "from_dict",
        classmethod(fail_fold_parse),
    )
    with pytest.raises(error_type, match="^internal-fold-bug$") as exc_info:
        if boundary == "create":
            construct_strategy_with_fold("create", fold)
        elif boundary == "direct":
            construct_strategy_with_fold("direct", fold)
        elif boundary == "to_dict":
            current.to_dict()
        else:
            StrategyArtifact.from_dict(payload)

    assert type(exc_info.value) is error_type
    assert exc_info.value is failure


@pytest.mark.parametrize("boundary", ["create", "direct", "to_dict", "from_dict"])
def test_fold_exception_boundary_still_rejects_documented_malformed_data(
    boundary: str,
) -> None:
    fold = fold_evidence_with_index(0)
    current = strategy_artifact()
    payload = current.to_dict()
    if boundary == "from_dict":
        payload["fold_evidence"][0]["validation_metrics"] = {"ic": "bad"}  # type: ignore[index]
        payload["fingerprint"] = sha256_json(
            {key: item for key, item in payload.items() if key != "fingerprint"}
        )
    elif boundary == "to_dict":
        object.__setattr__(
            current.fold_evidence[0],
            "validation_metrics",
            {"ic": "bad"},
        )
    else:
        object.__setattr__(fold, "validation_metrics", {"ic": "bad"})

    with pytest.raises(ArtifactCompatibilityError, match="validation_metrics"):
        if boundary == "create":
            construct_strategy_with_fold("create", fold)
        elif boundary == "direct":
            construct_strategy_with_fold("direct", fold)
        elif boundary == "to_dict":
            current.to_dict()
        else:
            StrategyArtifact.from_dict(payload)


@pytest.mark.parametrize(
    "payload",
    [
        [0],
        {"formula_tokens": [0]},
        {**strategy_artifact().to_dict(), "schema_version": "strategy-v1"},
        {**strategy_artifact().to_dict(), "formula_tokens": [True]},
        {**strategy_artifact().to_dict(), "formula_tokens": [FORMULA_VOCAB.size]},
    ],
)
def test_strategy_rejects_legacy_missing_schema_or_invalid_tokens(payload: object) -> None:
    with pytest.raises(ArtifactCompatibilityError):
        StrategyArtifact.from_dict(payload)  # type: ignore[arg-type]


def test_strategy_rejects_formula_structure_violation() -> None:
    names = FORMULA_VOCAB.token_names
    positive = FORMULA_VOCAB.operator_offset + FORMULA_VOCAB.operator_names.index("ABS")
    propagating = FORMULA_VOCAB.operator_offset + FORMULA_VOCAB.operator_names.index("TS_SUM_5")
    payload = strategy_artifact().to_dict()
    payload["formula_tokens"] = [0, positive, propagating]
    payload["fingerprint"] = sha256_json({key: value for key, value in payload.items() if key != "fingerprint"})
    with pytest.raises(ArtifactCompatibilityError, match="formula structure"):
        StrategyArtifact.from_dict(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [("core_semantics_version", "1"), ("vocab_version", "v-old")],
)
def test_strategy_rejects_current_version_mismatch(field: str, value: str) -> None:
    payload = strategy_artifact().to_dict()
    payload["run_identity"]["artifact_identity"][field] = value  # type: ignore[index]
    payload["fingerprint"] = sha256_json({key: item for key, item in payload.items() if key != "fingerprint"})
    with pytest.raises(ArtifactCompatibilityError, match=field):
        StrategyArtifact.from_dict(payload)


def test_strategy_rejects_malformed_nested_dataset_before_fingerprint() -> None:
    payload = strategy_artifact().to_dict()
    payload["run_identity"]["artifact_identity"]["training_dataset"]["bars"] = True  # type: ignore[index]
    payload["fingerprint"] = sha256_json(
        {key: item for key, item in payload.items() if key != "fingerprint"}
    )
    with pytest.raises(ArtifactCompatibilityError, match="bars"):
        StrategyArtifact.from_dict(payload)


def test_strategy_create_rejects_decoded_formula_that_disagrees_with_tokens() -> None:
    base = strategy_artifact()
    with pytest.raises(ArtifactCompatibilityError, match="decoded_formula"):
        StrategyArtifact.create(
            run_identity=base.run_identity,
            formula_tokens=[0, 1],
            decoded_formula="DIFFERENT_FORMULA",
            best_score=1.0,
            fold_evidence=base.fold_evidence,
            generated_at="2026-07-15T00:00:00Z",
        )


def test_strategy_load_rejects_rehashed_decoded_formula_tamper() -> None:
    payload = strategy_artifact().to_dict()
    payload["decoded_formula"] = "DIFFERENT_FORMULA"
    payload["fingerprint"] = sha256_json(
        {key: item for key, item in payload.items() if key != "fingerprint"}
    )
    with pytest.raises(ArtifactCompatibilityError, match="decoded_formula"):
        StrategyArtifact.from_dict(payload)


@pytest.mark.parametrize(
    ("container", "field"),
    [
        ("model", "notes"),
        ("reward", "comment"),
        ("entropy", "label"),
        ("elite", "created_at"),
        ("restart", "operator_name"),
        ("noise", "analyst_metadata"),
        ("walk_forward", "local_metadata"),
    ],
)
def test_training_config_recursively_rejects_non_behavior_fields(
    container: str,
    field: str,
) -> None:
    config = training_config()
    config[container][field] = "not training behavior"  # type: ignore[index]
    with pytest.raises(ArtifactCompatibilityError, match=field):
        ArtifactIdentity(
            core_semantics_version=CORE_SEMANTICS_VERSION,
            vocab_version=VOCAB_VERSION,
            label_semantics_version=LABEL_SEMANTICS_VERSION,
            execution_semantics_version=EXECUTION_SEMANTICS_VERSION,
            symbol="EURUSD",
            timeframe="H1",
            training_dataset=dataset_identity(),
            training_config=config,
            training_config_hash=sha256_json(config),
        )


def test_training_config_behavior_leaf_rejects_metadata_hidden_in_array() -> None:
    config = training_config()
    config["restart"]["partial_reset_layers"] = [  # type: ignore[index]
        {"notes": "hidden metadata"}
    ]
    with pytest.raises(ArtifactCompatibilityError, match="notes"):
        ArtifactIdentity(
            core_semantics_version=CORE_SEMANTICS_VERSION,
            vocab_version=VOCAB_VERSION,
            label_semantics_version=LABEL_SEMANTICS_VERSION,
            execution_semantics_version=EXECUTION_SEMANTICS_VERSION,
            symbol="EURUSD",
            timeframe="H1",
            training_dataset=dataset_identity(),
            training_config=config,
            training_config_hash=sha256_json(config),
        )


@pytest.mark.parametrize(
    "loader_payload",
    [
        lambda: (ArtifactIdentity.from_dict, artifact_identity().to_dict()),
        lambda: (TrainingRunIdentity.from_dict, TrainingRunIdentity.create(artifact_identity()).to_dict()),
        lambda: (
            FoldEvidence.from_dict,
            FoldEvidence(
                fold_index=0,
                train_start_time_ns=1,
                train_end_time_ns=2,
                val_start_time_ns=3,
                val_end_time_ns=4,
                effective_gap=LABEL_LOOKAHEAD_BARS,
                validation_metrics={"ic": 0.1},
            ).to_dict(),
        ),
        lambda: (StrategyArtifact.from_dict, strategy_artifact().to_dict()),
    ],
)
def test_all_from_dict_loaders_wrap_mixed_type_top_level_keys(loader_payload) -> None:
    loader, payload = loader_payload()
    payload[1] = "non-string"
    payload["unknown"] = "mixed-sort-trigger"
    with pytest.raises(ArtifactCompatibilityError, match="key"):
        loader(payload)


class HostileExactMapping(Mapping[str, object]):
    def __init__(
        self,
        values: Mapping[str, object],
        *,
        failure: BaseException,
        fail_iteration: bool = False,
        failing_key: str | None = None,
    ) -> None:
        self._values = dict(values)
        self.failure = failure
        self.fail_iteration = fail_iteration
        self.failing_key = failing_key
        self.iterations = 0
        self.item_reads: dict[str, int] = {}
        self.render_calls = {"repr": 0, "str": 0}

    def __getitem__(self, key: str) -> object:
        self.item_reads[key] = self.item_reads.get(key, 0) + 1
        if key == self.failing_key:
            raise self.failure
        return self._values[key]

    def __iter__(self):
        self.iterations += 1
        if self.fail_iteration:
            raise self.failure
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __repr__(self) -> str:
        self.render_calls["repr"] += 1
        raise AssertionError("hostile mapping repr must not be called")

    def __str__(self) -> str:
        self.render_calls["str"] += 1
        raise AssertionError("hostile mapping str must not be called")


class HostileList(list[object]):
    def __init__(self, values: list[object], failure: Exception) -> None:
        list.__init__(self, values)
        self.failure = failure
        self.protocol_calls = {
            "iter": 0,
            "len": 0,
            "getitem": 0,
            "repr": 0,
            "str": 0,
        }

    def __iter__(self):
        self.protocol_calls["iter"] += 1
        raise self.failure

    def __len__(self) -> int:
        self.protocol_calls["len"] += 1
        raise self.failure

    def __getitem__(self, key: object) -> object:
        self.protocol_calls["getitem"] += 1
        raise self.failure

    def __repr__(self) -> str:
        self.protocol_calls["repr"] += 1
        raise self.failure

    def __str__(self) -> str:
        self.protocol_calls["str"] += 1
        raise self.failure


class HostileTuple(tuple[object, ...]):
    def __new__(cls, values: tuple[object, ...], failure: Exception):
        instance = tuple.__new__(cls, values)
        instance.failure = failure
        instance.protocol_calls = {
            "iter": 0,
            "len": 0,
            "getitem": 0,
            "repr": 0,
            "str": 0,
        }
        return instance

    def __iter__(self):
        self.protocol_calls["iter"] += 1
        raise self.failure

    def __len__(self) -> int:
        self.protocol_calls["len"] += 1
        raise self.failure

    def __getitem__(self, key: object) -> object:
        self.protocol_calls["getitem"] += 1
        raise self.failure

    def __repr__(self) -> str:
        self.protocol_calls["repr"] += 1
        raise self.failure

    def __str__(self) -> str:
        self.protocol_calls["str"] += 1
        raise self.failure


class StatefulMetrics(Mapping[str, object]):
    def __init__(self) -> None:
        self.iterations = 0

    def __iter__(self):
        self.iterations += 1
        return iter(("ic",))

    def __len__(self) -> int:
        return 1

    def __getitem__(self, key: str) -> object:
        if key != "ic":
            raise KeyError(key)
        return 0.1 if self.iterations == 1 else 0.2


def assert_no_sequence_protocol_calls(value: HostileList | HostileTuple) -> None:
    assert value.protocol_calls == {
        "iter": 0,
        "len": 0,
        "getitem": 0,
        "repr": 0,
        "str": 0,
    }


@pytest.mark.parametrize("sequence_type", [HostileList, HostileTuple])
@pytest.mark.parametrize("loader_index", range(3))
def test_nested_sequence_subclasses_fail_closed_at_all_mapping_loaders(
    sequence_type,
    loader_index: int,
) -> None:
    failure = RuntimeError("hostile-sequence-" + "X" * 5_000)
    hostile = sequence_type(["layer"] if sequence_type is HostileList else ("layer",), failure)
    identity_payload = artifact_identity().to_dict()
    identity_payload["training_config"]["restart"]["partial_reset_layers"] = hostile  # type: ignore[index]
    cases = [
        (ArtifactIdentity.from_dict, identity_payload),
        (
            TrainingRunIdentity.from_dict,
            {"run_id": "1" * 32, "artifact_identity": identity_payload},
        ),
        (
            FoldEvidence.from_dict,
            {
                **fold_evidence_with_index(0).to_dict(),
                "validation_metrics": {"nested": hostile},
            },
        ),
    ]
    loader, payload = cases[loader_index]
    with pytest.raises(ArtifactCompatibilityError, match="unsupported JSON array type") as exc_info:
        loader(payload)
    assert len(str(exc_info.value)) <= 1_024
    assert_no_sequence_protocol_calls(hostile)


def test_strategy_formula_list_subclass_fails_closed_before_protocol_access() -> None:
    failure = RuntimeError("formula-" + "Y" * 5_000)
    hostile = HostileList([0], failure)
    payload = strategy_artifact().to_dict()
    payload["formula_tokens"] = hostile
    with pytest.raises(ArtifactCompatibilityError, match="unsupported JSON array type") as exc_info:
        StrategyArtifact.from_dict(payload)
    assert len(str(exc_info.value)) <= 1_024
    assert_no_sequence_protocol_calls(hostile)


def test_strategy_fold_evidence_list_subclass_fails_closed_before_protocol_access() -> None:
    failure = RuntimeError("folds-" + "F" * 5_000)
    hostile = HostileList([fold_evidence_with_index(0).to_dict()], failure)
    payload = strategy_artifact().to_dict()
    payload["fold_evidence"] = hostile
    with pytest.raises(ArtifactCompatibilityError, match="unsupported JSON array type"):
        StrategyArtifact.from_dict(payload)
    assert_no_sequence_protocol_calls(hostile)


@pytest.mark.parametrize("boundary", ["direct", "create", "to_dict", "verify"])
def test_sequence_subclasses_fail_closed_at_non_loader_boundaries(boundary: str) -> None:
    failure = RuntimeError("boundary-" + "Z" * 5_000)
    hostile = HostileList(["layer"], failure)
    if boundary == "create":
        hostile = HostileList([0], failure)
        action = lambda: StrategyArtifact.create(
            run_identity=TrainingRunIdentity.create(artifact_identity()),
            formula_tokens=hostile,
            decoded_formula=FORMULA_VOCAB.token_names[0],
            best_score=1.0,
            fold_evidence=(),
            generated_at="2026-07-15T00:00:00Z",
        )
    else:
        config = training_config()
        config["restart"]["partial_reset_layers"] = hostile  # type: ignore[index]
        if boundary == "direct":
            action = lambda: artifact_identity_with_config(config, config_hash="0" * 64)
        else:
            identity = artifact_identity()
            object.__setattr__(identity, "training_config", config)
            action = (
                identity.to_dict
                if boundary == "to_dict"
                else lambda: verify_artifact_identity(identity, artifact_identity())
            )
    with pytest.raises(ArtifactCompatibilityError):
        action()
    assert_no_sequence_protocol_calls(hostile)


def test_shared_stateful_mapping_alias_is_snapshotted_once_per_public_load() -> None:
    folds = (fold_evidence_with_index(0), fold_evidence_with_index(1))
    payload = construct_strategy_with_folds("from_dict", folds).to_dict()
    for fold in payload["fold_evidence"]:  # type: ignore[union-attr]
        fold["validation_metrics"] = {"ic": 0.1}
    payload["fingerprint"] = sha256_json(
        {key: value for key, value in payload.items() if key != "fingerprint"}
    )
    shared = StatefulMetrics()
    for fold in payload["fold_evidence"]:  # type: ignore[union-attr]
        fold["validation_metrics"] = shared

    loaded = StrategyArtifact.from_dict(payload)

    assert shared.iterations == 1
    assert [fold.validation_metrics["ic"] for fold in loaded.fold_evidence] == [0.1] * 4
    with pytest.raises(TypeError):
        loaded.fold_evidence[0].validation_metrics["ic"] = 9.0  # type: ignore[index]


def test_shared_100k_aliases_reuse_one_completed_snapshot_identity() -> None:
    shared = StatefulMetrics()
    snapshot = artifacts_module._require_mapping(
        {"aliases": [shared] * 100_000},
        context="shared-100k",
    )
    aliases = snapshot["aliases"]
    assert isinstance(aliases, list)
    assert shared.iterations == 1
    assert len({id(item) for item in aliases}) == 1


def test_repeated_builtin_list_alias_reuses_completed_snapshot() -> None:
    shared = [{"ic": 0.1}]
    snapshot = artifacts_module._require_mapping(
        {"left": shared, "right": shared},
        context="shared-list",
    )
    assert snapshot["left"] is snapshot["right"]


def hostile_loader_cases():
    return (
        (ArtifactIdentity.from_dict, artifact_identity().to_dict()),
        (
            TrainingRunIdentity.from_dict,
            TrainingRunIdentity.create(artifact_identity()).to_dict(),
        ),
        (
            FoldEvidence.from_dict,
            FoldEvidence(
                fold_index=0,
                train_start_time_ns=1,
                train_end_time_ns=2,
                val_start_time_ns=3,
                val_end_time_ns=4,
                effective_gap=LABEL_LOOKAHEAD_BARS,
                validation_metrics={"ic": 0.1},
            ).to_dict(),
        ),
        (StrategyArtifact.from_dict, strategy_artifact().to_dict()),
    )


@pytest.mark.parametrize("loader_index", range(5))
def test_all_from_dict_loaders_snapshot_hostile_iteration_once(loader_index: int) -> None:
    cases = list(hostile_loader_cases())
    config = training_config()
    cases.append((None, config))
    loader, payload = cases[loader_index]
    failure = RuntimeError("iter-" + "X" * 5_000)
    hostile = HostileExactMapping(
        payload,
        failure=failure,
        fail_iteration=True,
    )
    with pytest.raises(ArtifactCompatibilityError) as exc_info:
        if loader is not None:
            loader(hostile)
        else:
            ArtifactIdentity(
                core_semantics_version=CORE_SEMANTICS_VERSION,
                vocab_version=VOCAB_VERSION,
                label_semantics_version=LABEL_SEMANTICS_VERSION,
                execution_semantics_version=EXECUTION_SEMANTICS_VERSION,
                symbol="EURUSD",
                timeframe="H1",
                training_dataset=dataset_identity(),
                training_config=hostile,
                training_config_hash=sha256_json(config),
            )
    assert exc_info.value.__cause__ is failure
    assert len(str(exc_info.value)) <= 1_024
    assert "mapping access failed" in str(exc_info.value)
    assert hostile.iterations == 1
    assert hostile.item_reads == {}
    assert hostile.render_calls == {"repr": 0, "str": 0}


@pytest.mark.parametrize("failure_type", [AssertionError, KeyboardInterrupt, SystemExit])
def test_unexpected_mapping_base_exceptions_propagate_unchanged(
    failure_type: type[BaseException],
) -> None:
    loader, payload = hostile_loader_cases()[0]
    failure = failure_type("internal mapping failure")
    hostile = HostileExactMapping(
        payload,
        failure=failure,
        fail_iteration=True,
    )
    with pytest.raises(failure_type) as exc_info:
        loader(hostile)
    assert exc_info.value is failure


@pytest.mark.parametrize("failure_type", [RuntimeError, KeyError, TypeError, ValueError])
@pytest.mark.parametrize("boundary_index", range(5))
def test_public_mapping_boundaries_wrap_exact_key_item_failures_once(
    failure_type: type[Exception],
    boundary_index: int,
) -> None:
    cases = list(hostile_loader_cases())
    config = training_config()
    cases.append((None, config))
    loader, payload = cases[boundary_index]
    failing_key = next(iter(payload))
    failure = failure_type("getitem-" + "Y" * 5_000)
    hostile = HostileExactMapping(
        payload,
        failure=failure,
        failing_key=failing_key,
    )
    with pytest.raises(ArtifactCompatibilityError) as exc_info:
        if loader is not None:
            loader(hostile)
        else:
            ArtifactIdentity(
                core_semantics_version=CORE_SEMANTICS_VERSION,
                vocab_version=VOCAB_VERSION,
                label_semantics_version=LABEL_SEMANTICS_VERSION,
                execution_semantics_version=EXECUTION_SEMANTICS_VERSION,
                symbol="EURUSD",
                timeframe="H1",
                training_dataset=dataset_identity(),
                training_config=hostile,
                training_config_hash=sha256_json(config),
            )
    assert exc_info.value.__cause__ is failure
    assert len(str(exc_info.value)) <= 1_024
    assert "mapping access failed" in str(exc_info.value)
    assert hostile.iterations == 1
    assert max(hostile.item_reads.values()) == 1
    assert hostile.render_calls == {"repr": 0, "str": 0}


@pytest.mark.parametrize("loader_index", range(4))
def test_all_from_dict_loaders_accept_mapping_proxy_snapshots(loader_index: int) -> None:
    loader, payload = hostile_loader_cases()[loader_index]
    assert loader(MappingProxyType(payload)) == loader(payload)


def test_direct_artifact_identity_accepts_mapping_proxy_training_config() -> None:
    config = training_config()
    expected = artifact_identity_with_config(config)
    actual = artifact_identity_with_config(
        MappingProxyType(config),
        config_hash=sha256_json(config),
    )
    assert actual == expected


@pytest.mark.parametrize("boundary", ["direct", "from_dict"])
def test_nested_training_config_mapping_protocol_failure_is_wrapped(
    boundary: str,
) -> None:
    config = training_config()
    failure = RuntimeError("nested-getitem-" + "Z" * 5_000)
    hostile_reward = HostileExactMapping(
        config["reward"],  # type: ignore[arg-type]
        failure=failure,
        failing_key="mode",
    )
    config["reward"] = hostile_reward
    with pytest.raises(ArtifactCompatibilityError) as exc_info:
        if boundary == "direct":
            ArtifactIdentity(
                core_semantics_version=CORE_SEMANTICS_VERSION,
                vocab_version=VOCAB_VERSION,
                label_semantics_version=LABEL_SEMANTICS_VERSION,
                execution_semantics_version=EXECUTION_SEMANTICS_VERSION,
                symbol="EURUSD",
                timeframe="H1",
                training_dataset=dataset_identity(),
                training_config=config,
                training_config_hash="0" * 64,
            )
        else:
            payload = artifact_identity().to_dict()
            payload["training_config"] = config
            ArtifactIdentity.from_dict(payload)
    assert exc_info.value.__cause__ is failure
    assert len(str(exc_info.value)) <= 1_024
    assert "training_config.reward" in str(exc_info.value)
    assert hostile_reward.iterations == 1
    assert max(hostile_reward.item_reads.values()) == 1
    assert hostile_reward.render_calls == {"repr": 0, "str": 0}


@pytest.mark.parametrize("consumer", ["direct", "to_dict", "verify", "training_run"])
@pytest.mark.parametrize("field", DATASET_FIELDS)
def test_dataset_identity_assertion_getter_propagates_unchanged(
    consumer: str,
    field: str,
) -> None:
    failure = AssertionError(f"internal assertion for {field}")
    source, _ = instrumented_dataset_identity(
        failing_field=field,
        failure=failure,
    )
    with pytest.raises(AssertionError) as exc_info:
        if consumer == "direct":
            config = training_config()
            ArtifactIdentity(
                core_semantics_version=CORE_SEMANTICS_VERSION,
                vocab_version=VOCAB_VERSION,
                label_semantics_version=LABEL_SEMANTICS_VERSION,
                execution_semantics_version=EXECUTION_SEMANTICS_VERSION,
                symbol="EURUSD",
                timeframe="H1",
                training_dataset=source,
                training_config=config,
                training_config_hash=sha256_json(config),
            )
        else:
            identity = artifact_identity()
            object.__setattr__(identity, "training_dataset", source)
            if consumer == "to_dict":
                identity.to_dict()
            elif consumer == "verify":
                verify_artifact_identity(identity, artifact_identity())
            else:
                TrainingRunIdentity.create(identity)
    assert exc_info.value is failure


class ExactKeyString(str):
    pass


class HostileKeyString(str):
    def __repr__(self) -> str:
        raise AssertionError("str-subclass repr must not run")

    def __str__(self) -> str:
        raise AssertionError("str-subclass str must not run")

    def lower(self) -> str:
        raise AssertionError("str-subclass lower must not run")

    def strip(self, *args: object, **kwargs: object) -> str:
        raise AssertionError("str-subclass strip must not run")


class DuplicateItemsMapping(Mapping[str, object]):
    """Mapping whose raw item stream can contain keys a dict would collapse."""

    def __init__(self, pairs: list[tuple[str, object]]) -> None:
        self._pairs = pairs
        self._lookup = dict(pairs)

    def __getitem__(self, key: str) -> object:
        return self._lookup[key]

    def __iter__(self):
        return iter(self._lookup)

    def __len__(self) -> int:
        return len(self._lookup)

    def items(self):
        return iter(self._pairs)


def duplicate_items(
    payload: Mapping[str, object],
    key: str,
    second_value: object,
) -> DuplicateItemsMapping:
    return DuplicateItemsMapping([*payload.items(), (key, second_value)])


class ExactInteger(int):
    pass


class HostileInteger(int):
    def __repr__(self) -> str:
        raise AssertionError("int-subclass repr must not run")

    def __int__(self) -> int:
        raise AssertionError("int-subclass coercion must not run")

    def __lt__(self, other: object) -> bool:
        raise AssertionError("int-subclass comparison must not run")


class ProtocolTrackingInteger(int):
    def __new__(cls, value: int) -> "ProtocolTrackingInteger":
        instance = super().__new__(cls, value)
        instance.protocol_calls = {"lt": 0, "ge": 0, "repr": 0}
        return instance

    def __lt__(self, other: object) -> bool:
        self.protocol_calls["lt"] += 1
        raise AssertionError("HOSTILE_TOKEN_LT")

    def __ge__(self, other: object) -> bool:
        self.protocol_calls["ge"] += 1
        raise AssertionError("HOSTILE_TOKEN_GE")

    def __repr__(self) -> str:
        self.protocol_calls["repr"] += 1
        raise AssertionError("HOSTILE_TOKEN_REPR")


HOSTILE_INTEGER_META_CALLS = {"name": 0}


class HostileIntegerMeta(type):
    def __getattribute__(cls, name: str) -> object:
        if name == "__name__":
            HOSTILE_INTEGER_META_CALLS["name"] += 1
            raise AssertionError("HOSTILE_CLASS_NAME")
        return super().__getattribute__(name)


class HostileMetaInteger(int, metaclass=HostileIntegerMeta):
    pass


HOSTILE_NUMERIC_PROTOCOL_CALLS = {
    "name": 0,
    "repr": 0,
    "str": 0,
    "int": 0,
    "index": 0,
    "eq": 0,
    "hash": 0,
    "float": 0,
}


class HostileNumericMeta(type):
    def __getattribute__(cls, name: str) -> object:
        if name == "__name__":
            HOSTILE_NUMERIC_PROTOCOL_CALLS["name"] += 1
            raise AssertionError("HOSTILE_CLASS_NAME")
        return super().__getattribute__(name)


class HostileBoundaryInteger(int, metaclass=HostileNumericMeta):
    def __repr__(self) -> str:
        HOSTILE_NUMERIC_PROTOCOL_CALLS["repr"] += 1
        raise AssertionError("HOSTILE_REPR")

    def __str__(self) -> str:
        HOSTILE_NUMERIC_PROTOCOL_CALLS["str"] += 1
        raise AssertionError("HOSTILE_STR")

    def __int__(self) -> int:
        HOSTILE_NUMERIC_PROTOCOL_CALLS["int"] += 1
        raise AssertionError("HOSTILE_INT")

    def __index__(self) -> int:
        HOSTILE_NUMERIC_PROTOCOL_CALLS["index"] += 1
        raise AssertionError("HOSTILE_INDEX")

    def __eq__(self, other: object) -> bool:
        HOSTILE_NUMERIC_PROTOCOL_CALLS["eq"] += 1
        raise AssertionError("HOSTILE_EQ")

    def __hash__(self) -> int:
        HOSTILE_NUMERIC_PROTOCOL_CALLS["hash"] += 1
        raise AssertionError("HOSTILE_HASH")


class BenignFloat(float):
    pass


class HostileBoundaryFloat(float, metaclass=HostileNumericMeta):
    def __repr__(self) -> str:
        HOSTILE_NUMERIC_PROTOCOL_CALLS["repr"] += 1
        raise AssertionError("HOSTILE_REPR")

    def __str__(self) -> str:
        HOSTILE_NUMERIC_PROTOCOL_CALLS["str"] += 1
        raise AssertionError("HOSTILE_STR")

    def __float__(self) -> float:
        HOSTILE_NUMERIC_PROTOCOL_CALLS["float"] += 1
        raise AssertionError("HOSTILE_FLOAT_COERCION")

    def __eq__(self, other: object) -> bool:
        HOSTILE_NUMERIC_PROTOCOL_CALLS["eq"] += 1
        raise AssertionError("HOSTILE_EQ")

    def __hash__(self) -> int:
        HOSTILE_NUMERIC_PROTOCOL_CALLS["hash"] += 1
        raise AssertionError("HOSTILE_HASH")


def _reset_hostile_numeric_protocol_calls() -> None:
    for key in HOSTILE_NUMERIC_PROTOCOL_CALLS:
        HOSTILE_NUMERIC_PROTOCOL_CALLS[key] = 0


def _assert_no_hostile_numeric_protocol_calls() -> None:
    assert HOSTILE_NUMERIC_PROTOCOL_CALLS == {
        "name": 0,
        "repr": 0,
        "str": 0,
        "int": 0,
        "index": 0,
        "eq": 0,
        "hash": 0,
        "float": 0,
    }


def _subclass_keys(payload: dict[str, object], key_type: type[str]) -> dict[str, object]:
    return {key_type(key): value for key, value in payload.items()}


def _fold_payload_with_integer_type(integer_type: type[int]) -> dict[str, object]:
    return {
        "fold_index": integer_type(0),
        "train_start_time_ns": integer_type(1_000),
        "train_end_time_ns": integer_type(2_000),
        "val_start_time_ns": integer_type(2_500),
        "val_end_time_ns": integer_type(3_000),
        "effective_gap": integer_type(2),
        "validation_metrics": {"ic": 0.1},
    }


def test_canonical_json_rejects_str_subclass_key_instead_of_colliding() -> None:
    exact = canonical_json_bytes({"a": 1})
    with pytest.raises(ArtifactCompatibilityError, match="key.*exact built-in string"):
        subclass = canonical_json_bytes({ExactKeyString("a"): 1})
        assert subclass != exact


@pytest.mark.parametrize(
    "loader_name",
    ["artifact", "run", "fold", "strategy", "nested_dataset"],
)
@pytest.mark.parametrize("key_type", [ExactKeyString, HostileKeyString])
def test_public_mapping_loaders_reject_str_subclass_keys(
    loader_name: str,
    key_type: type[str],
) -> None:
    if loader_name == "artifact":
        loader = ArtifactIdentity.from_dict
        payload = artifact_identity().to_dict()
    elif loader_name == "run":
        loader = TrainingRunIdentity.from_dict
        payload = TrainingRunIdentity.create(artifact_identity()).to_dict()
    elif loader_name == "fold":
        loader = FoldEvidence.from_dict
        payload = FoldEvidence.from_dict(_fold_payload_with_integer_type(int)).to_dict()
    elif loader_name == "strategy":
        loader = StrategyArtifact.from_dict
        payload = strategy_artifact().to_dict()
    else:
        loader = ArtifactIdentity.from_dict
        payload = artifact_identity().to_dict()
        payload["training_dataset"] = _subclass_keys(
            payload["training_dataset"],  # type: ignore[arg-type]
            key_type,
        )
        with pytest.raises(ArtifactCompatibilityError, match="key.*exact built-in string"):
            loader(payload)
        return

    with pytest.raises(ArtifactCompatibilityError, match="key.*exact built-in string"):
        loader(_subclass_keys(payload, key_type))


@pytest.mark.parametrize("nested", ["training_config", "validation_metrics"])
def test_nested_public_mappings_reject_hostile_str_subclass_keys(nested: str) -> None:
    if nested == "training_config":
        payload = artifact_identity().to_dict()
        payload["training_config"] = _subclass_keys(
            payload["training_config"],  # type: ignore[arg-type]
            HostileKeyString,
        )
        action = lambda: ArtifactIdentity.from_dict(payload)
    else:
        payload = _fold_payload_with_integer_type(int)
        payload["validation_metrics"] = {HostileKeyString("ic"): 0.1}
        action = lambda: FoldEvidence.from_dict(payload)
    with pytest.raises(ArtifactCompatibilityError, match="key.*exact built-in string"):
        action()


@pytest.mark.parametrize("boundary", ["direct", "verify"])
def test_artifact_identity_rejects_str_subclass_training_config_keys(
    boundary: str,
) -> None:
    exact_config = training_config()
    hostile_config = _subclass_keys(exact_config, HostileKeyString)
    if boundary == "direct":
        base = artifact_identity()
        action = lambda: ArtifactIdentity(
            core_semantics_version=base.core_semantics_version,
            vocab_version=base.vocab_version,
            label_semantics_version=base.label_semantics_version,
            execution_semantics_version=base.execution_semantics_version,
            symbol=base.symbol,
            timeframe=base.timeframe,
            training_dataset=base.training_dataset,
            training_config=hostile_config,
            training_config_hash=sha256_json(exact_config),
        )
    else:
        candidate = artifact_identity()
        object.__setattr__(candidate, "training_config", hostile_config)
        action = lambda: verify_artifact_identity(candidate, artifact_identity())
    with pytest.raises(ArtifactCompatibilityError, match="key.*exact built-in string"):
        action()


def test_fold_direct_construction_rejects_str_subclass_metric_key() -> None:
    payload = _fold_payload_with_integer_type(int)
    payload["validation_metrics"] = {HostileKeyString("ic"): 0.1}
    with pytest.raises(ArtifactCompatibilityError, match="key.*exact built-in string"):
        FoldEvidence(**payload)  # type: ignore[arg-type]


@pytest.mark.parametrize("same_value", [True, False])
@pytest.mark.parametrize(
    "boundary",
    ["artifact", "run", "fold", "strategy", "dataset", "config", "config_nested", "metrics"],
)
def test_public_mapping_boundaries_reject_duplicate_raw_items(
    boundary: str,
    same_value: bool,
) -> None:
    if boundary == "artifact":
        payload = artifact_identity().to_dict()
        key = "core_semantics_version"
        payload = duplicate_items(payload, key, payload[key] if same_value else "legacy")
        action = lambda: ArtifactIdentity.from_dict(payload)
    elif boundary == "run":
        payload = TrainingRunIdentity.create(artifact_identity()).to_dict()
        key = "run_id"
        payload = duplicate_items(payload, key, payload[key] if same_value else "2" * 32)
        action = lambda: TrainingRunIdentity.from_dict(payload)
    elif boundary == "fold":
        payload = fold_evidence_with_index(0).to_dict()
        key = "fold_index"
        payload = duplicate_items(payload, key, payload[key] if same_value else 1)
        action = lambda: FoldEvidence.from_dict(payload)
    elif boundary == "strategy":
        payload = strategy_artifact().to_dict()
        key = "schema_version"
        payload = duplicate_items(payload, key, payload[key] if same_value else "strategy-v1")
        action = lambda: StrategyArtifact.from_dict(payload)
    elif boundary == "dataset":
        payload = artifact_identity().to_dict()
        dataset = payload["training_dataset"]
        assert isinstance(dataset, Mapping)
        payload["training_dataset"] = duplicate_items(
            dataset,
            "symbol",
            dataset["symbol"] if same_value else "GBPUSD",
        )
        action = lambda: ArtifactIdentity.from_dict(payload)
    elif boundary in {"config", "config_nested"}:
        payload = artifact_identity().to_dict()
        config = payload["training_config"]
        assert isinstance(config, dict)
        if boundary == "config":
            payload["training_config"] = duplicate_items(
                config,
                "random_seed",
                config["random_seed"] if same_value else 43,
            )
        else:
            reward = config["reward"]
            assert isinstance(reward, Mapping)
            config["reward"] = duplicate_items(
                reward,
                "mode",
                reward["mode"] if same_value else "standard",
            )
        action = lambda: ArtifactIdentity.from_dict(payload)
    else:
        payload = fold_evidence_with_index(0).to_dict()
        metrics = payload["validation_metrics"]
        assert isinstance(metrics, Mapping)
        payload["validation_metrics"] = duplicate_items(
            metrics,
            "ic",
            metrics["ic"] if same_value else 0.9,
        )
        action = lambda: FoldEvidence.from_dict(payload)
    with pytest.raises(ArtifactCompatibilityError, match="duplicate.*key"):
        action()


@pytest.mark.parametrize(
    "blocks",
    [
        pytest.param(0, id="zero"),
        pytest.param(1, id="one"),
        pytest.param(True, id="bool"),
        pytest.param(3.0, id="float"),
        pytest.param("3", id="string"),
        pytest.param(10**1000, id="non-operational-huge-int"),
    ],
)
@pytest.mark.parametrize("boundary", ["direct", "from_dict", "round_trip"])
def test_artifact_identity_rejects_invalid_walk_forward_block_domain(
    blocks: object,
    boundary: str,
) -> None:
    config = training_config()
    config["walk_forward"]["blocks"] = blocks  # type: ignore[index]
    with pytest.raises(
        ArtifactCompatibilityError,
        match=(
            r"training_config\.walk_forward\.blocks.*expected=integer >= 2"
            r"|operational finite-real range.*walk_forward\.blocks"
        ),
    ):
        if boundary == "direct":
            artifact_identity_with_config(config)
        else:
            payload = artifact_identity().to_dict()
            payload["training_config"] = config
            payload["training_config_hash"] = sha256_json(config)
            loaded = ArtifactIdentity.from_dict(payload)
            if boundary == "round_trip":
                ArtifactIdentity.from_dict(loaded.to_dict())


@pytest.mark.parametrize(
    "field_path",
    [
        ("core_semantics_version",),
        ("vocab_version",),
        ("label_semantics_version",),
        ("execution_semantics_version",),
        ("symbol",),
        ("timeframe",),
        ("training_config_hash",),
        ("training_dataset", "schema_version"),
        ("training_dataset", "symbol"),
        ("training_dataset", "timeframe"),
        ("training_dataset", "data_fingerprint"),
        ("training_dataset", "time_fingerprint"),
        ("training_config", "reward", "mode"),
        ("training_config", "restart", "partial_reset_layers", 0),
    ],
)
@pytest.mark.parametrize("string_type", [ExactKeyString, HostileKeyString])
def test_artifact_identity_rejects_scalar_string_subclasses_at_load_boundary(
    field_path: tuple[object, ...],
    string_type: type[str],
) -> None:
    payload = artifact_identity().to_dict()
    target: object = payload
    for part in field_path[:-1]:
        target = target[part]  # type: ignore[index]
    leaf = field_path[-1]
    original = target[leaf]  # type: ignore[index]
    target[leaf] = string_type(original)  # type: ignore[index,call-overload]
    with pytest.raises(ArtifactCompatibilityError, match="exact built-in string"):
        ArtifactIdentity.from_dict(payload)


@pytest.mark.parametrize(
    "field_path",
    [
        ("schema_version",),
        ("decoded_formula",),
        ("generated_at",),
        ("fingerprint",),
        ("run_identity", "run_id"),
    ],
)
def test_strategy_and_nested_metrics_reject_scalar_string_subclasses(
    field_path: tuple[object, ...],
) -> None:
    payload = strategy_artifact().to_dict()
    target: object = payload
    for part in field_path[:-1]:
        target = target[part]  # type: ignore[index]
    leaf = field_path[-1]
    target[leaf] = ExactKeyString(target[leaf])  # type: ignore[index,call-overload]
    with pytest.raises(ArtifactCompatibilityError, match="exact built-in string"):
        StrategyArtifact.from_dict(payload)


@pytest.mark.parametrize("boundary", ["direct", "verify", "training_run"])
@pytest.mark.parametrize("nested_dataset", [False, True])
def test_identity_non_loader_boundaries_reject_scalar_string_subclasses(
    boundary: str,
    nested_dataset: bool,
) -> None:
    candidate = artifact_identity()
    if nested_dataset:
        object.__setattr__(
            candidate,
            "training_dataset",
            bypass_dataset_identity(
                candidate.training_dataset,
                symbol=ExactKeyString("EURUSD"),
            ),
        )
    else:
        object.__setattr__(candidate, "symbol", ExactKeyString("EURUSD"))
    with pytest.raises(ArtifactCompatibilityError, match="exact built-in string"):
        if boundary == "direct":
            config = training_config()
            ArtifactIdentity(
                core_semantics_version=CORE_SEMANTICS_VERSION,
                vocab_version=VOCAB_VERSION,
                label_semantics_version=LABEL_SEMANTICS_VERSION,
                execution_semantics_version=EXECUTION_SEMANTICS_VERSION,
                symbol=ExactKeyString("EURUSD") if not nested_dataset else "EURUSD",
                timeframe="H1",
                training_dataset=(
                    bypass_dataset_identity(symbol=ExactKeyString("EURUSD"))
                    if nested_dataset
                    else dataset_identity()
                ),
                training_config=config,
                training_config_hash=sha256_json(config),
            )
        elif boundary == "verify":
            verify_artifact_identity(candidate, artifact_identity())
        else:
            TrainingRunIdentity.create(candidate)


@pytest.mark.parametrize("field", ["decoded_formula", "generated_at"])
def test_strategy_create_rejects_scalar_string_subclasses(field: str) -> None:
    kwargs = {
        "run_identity": TrainingRunIdentity.create(artifact_identity()),
        "formula_tokens": [0],
        "decoded_formula": FORMULA_VOCAB.token_names[0],
        "best_score": 1.0,
        "fold_evidence": [fold_evidence_with_index(index) for index in range(4)],
        "generated_at": "2026-07-15T00:00:00Z",
    }
    kwargs[field] = ExactKeyString(kwargs[field])
    with pytest.raises(ArtifactCompatibilityError, match="exact built-in string"):
        StrategyArtifact.create(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("construction", ["direct", "from_dict", "round_trip"])
@pytest.mark.parametrize("integer_type", [ExactInteger, HostileInteger])
def test_fold_evidence_rejects_integer_subclasses_at_every_public_boundary(
    construction: str,
    integer_type: type[int],
) -> None:
    payload = _fold_payload_with_integer_type(integer_type)
    with pytest.raises(ArtifactCompatibilityError, match="exact built-in integer"):
        if construction == "direct":
            FoldEvidence(**payload)  # type: ignore[arg-type]
        elif construction == "from_dict":
            FoldEvidence.from_dict(payload)
        else:
            fold = FoldEvidence(**payload)  # type: ignore[arg-type]
            reloaded = FoldEvidence.from_dict(fold.to_dict())
            assert all(
                type(getattr(reloaded, field)) is int
                for field in (
                    "fold_index",
                    "train_start_time_ns",
                    "train_end_time_ns",
                    "val_start_time_ns",
                    "val_end_time_ns",
                    "effective_gap",
                )
            )


@pytest.mark.parametrize("boundary", ["create", "direct", "from_dict", "to_dict"])
@pytest.mark.parametrize("integer_type", [ExactInteger, ProtocolTrackingInteger])
def test_formula_tokens_require_exact_built_in_integers_at_artifact_boundaries(
    boundary: str,
    integer_type: type[int],
) -> None:
    token = integer_type(0)
    base = strategy_artifact()

    if boundary == "create":
        action = lambda: StrategyArtifact.create(
            run_identity=base.run_identity,
            formula_tokens=[token],
            decoded_formula=FORMULA_VOCAB.token_names[0],
            best_score=base.best_score,
            fold_evidence=base.fold_evidence,
            generated_at=base.generated_at,
        )
    elif boundary == "direct":
        action = lambda: StrategyArtifact(
            schema_version=base.schema_version,
            run_identity=base.run_identity,
            formula_tokens=(token,),
            decoded_formula=base.decoded_formula,
            best_score=base.best_score,
            fold_evidence=base.fold_evidence,
            generated_at=base.generated_at,
            fingerprint=base.fingerprint,
        )
    elif boundary == "from_dict":
        payload = base.to_dict()
        payload["formula_tokens"] = [token]
        action = lambda: StrategyArtifact.from_dict(payload)
    else:
        object.__setattr__(base, "formula_tokens", (token,))
        action = base.to_dict

    with pytest.raises(ArtifactCompatibilityError, match="exact built-in integer") as exc_info:
        action()
    assert len(str(exc_info.value)) <= 1_024
    if isinstance(token, ProtocolTrackingInteger):
        assert token.protocol_calls == {"lt": 0, "ge": 0, "repr": 0}


@pytest.mark.parametrize("integer_type", [ExactInteger, ProtocolTrackingInteger])
def test_checkpoint_filename_requires_exact_built_in_step(integer_type: type[int]) -> None:
    step = integer_type(7)
    run = TrainingRunIdentity.create(artifact_identity())

    with pytest.raises(ArtifactCompatibilityError, match="exact built-in integer") as exc_info:
        run.checkpoint_filename(step)

    message = str(exc_info.value)
    assert "expected=" in message
    assert "actual_category=non-exact-integer" in message
    assert len(message) <= 1_024
    if isinstance(step, ProtocolTrackingInteger):
        assert step.protocol_calls == {"lt": 0, "ge": 0, "repr": 0}


def test_exact_built_in_formula_token_and_checkpoint_steps_remain_accepted() -> None:
    artifact = strategy_artifact()
    run = artifact.run_identity

    assert artifact.formula_tokens == (0,)
    assert run.checkpoint_filename(0).endswith("_step_0.pt")
    assert run.checkpoint_filename(17).endswith("_step_17.pt")


@pytest.mark.parametrize("boundary", ["create", "direct", "from_dict", "to_dict"])
def test_formula_token_rejection_never_reads_hostile_metaclass_name(
    boundary: str,
) -> None:
    HOSTILE_INTEGER_META_CALLS["name"] = 0
    token = HostileMetaInteger(0)
    base = strategy_artifact()

    if boundary == "create":
        action = lambda: StrategyArtifact.create(
            run_identity=base.run_identity,
            formula_tokens=[token],
            decoded_formula=FORMULA_VOCAB.token_names[0],
            best_score=base.best_score,
            fold_evidence=base.fold_evidence,
            generated_at=base.generated_at,
        )
    elif boundary == "direct":
        action = lambda: StrategyArtifact(
            schema_version=base.schema_version,
            run_identity=base.run_identity,
            formula_tokens=(token,),
            decoded_formula=base.decoded_formula,
            best_score=base.best_score,
            fold_evidence=base.fold_evidence,
            generated_at=base.generated_at,
            fingerprint=base.fingerprint,
        )
    elif boundary == "from_dict":
        payload = base.to_dict()
        payload["formula_tokens"] = [token]
        action = lambda: StrategyArtifact.from_dict(payload)
    else:
        object.__setattr__(base, "formula_tokens", (token,))
        action = base.to_dict

    with pytest.raises(
        ArtifactCompatibilityError,
        match="actual_category=non-exact-integer",
    ) as exc_info:
        action()
    assert len(str(exc_info.value)) <= 1_024
    assert HOSTILE_INTEGER_META_CALLS == {"name": 0}


def test_checkpoint_rejection_never_reads_hostile_metaclass_name() -> None:
    HOSTILE_INTEGER_META_CALLS["name"] = 0
    run = TrainingRunIdentity.create(artifact_identity())

    with pytest.raises(
        ArtifactCompatibilityError,
        match="actual_category=non-exact-integer",
    ) as exc_info:
        run.checkpoint_filename(HostileMetaInteger(7))

    assert len(str(exc_info.value)) <= 1_024
    assert HOSTILE_INTEGER_META_CALLS == {"name": 0}


@pytest.mark.parametrize(
    "integer_type",
    [ExactInteger, HostileBoundaryInteger],
    ids=["benign-subclass", "hostile-subclass"],
)
@pytest.mark.parametrize("boundary", ["direct", "from_dict"])
def test_fold_non_token_integers_reject_subclasses_without_protocols(
    integer_type: type[int],
    boundary: str,
) -> None:
    _reset_hostile_numeric_protocol_calls()
    payload = _fold_payload_with_integer_type(integer_type)
    action = (
        (lambda: FoldEvidence(**payload))
        if boundary == "direct"
        else (lambda: FoldEvidence.from_dict(payload))
    )

    with pytest.raises(ArtifactCompatibilityError, match="exact built-in integer") as exc_info:
        action()

    assert len(str(exc_info.value)) <= 1_024
    if integer_type is HostileBoundaryInteger:
        _assert_no_hostile_numeric_protocol_calls()


@pytest.mark.parametrize("boundary", ["direct", "from_dict", "to_dict", "verify"])
@pytest.mark.parametrize("location", ["training_dataset.start_time_ns", "training_config.batch_size"])
def test_identity_non_token_integers_reject_hostile_metaclass_without_protocols(
    boundary: str,
    location: str,
) -> None:
    _reset_hostile_numeric_protocol_calls()
    hostile = HostileBoundaryInteger(1_000 if location.endswith("start_time_ns") else 192)
    base = artifact_identity()

    if boundary in {"direct", "from_dict"}:
        payload = base.to_dict()
        if location.startswith("training_dataset"):
            payload["training_dataset"]["start_time_ns"] = hostile  # type: ignore[index]
        else:
            payload["training_config"]["batch_size"] = hostile  # type: ignore[index]
        if boundary == "direct":
            config = payload["training_config"]
            assert isinstance(config, Mapping)
            action = lambda: ArtifactIdentity(
                core_semantics_version=base.core_semantics_version,
                vocab_version=base.vocab_version,
                label_semantics_version=base.label_semantics_version,
                execution_semantics_version=base.execution_semantics_version,
                symbol=base.symbol,
                timeframe=base.timeframe,
                training_dataset=(
                    bypass_dataset_identity(
                        base.training_dataset,
                        start_time_ns=hostile,
                    )
                    if location.startswith("training_dataset")
                    else base.training_dataset
                ),
                training_config=config,
                training_config_hash=base.training_config_hash,
            )
        else:
            action = lambda: ArtifactIdentity.from_dict(payload)
    else:
        candidate = artifact_identity()
        if location.startswith("training_dataset"):
            object.__setattr__(
                candidate,
                "training_dataset",
                bypass_dataset_identity(
                    candidate.training_dataset,
                    start_time_ns=hostile,
                ),
            )
        else:
            config = candidate.to_dict()["training_config"]
            assert isinstance(config, dict)
            config["batch_size"] = hostile
            object.__setattr__(candidate, "training_config", config)
        action = candidate.to_dict if boundary == "to_dict" else lambda: verify_artifact_identity(candidate, base)

    with pytest.raises(ArtifactCompatibilityError) as exc_info:
        action()

    assert location.split(".")[-1] in str(exc_info.value)
    assert len(str(exc_info.value)) <= 1_024
    _assert_no_hostile_numeric_protocol_calls()


@pytest.mark.parametrize(
    "numeric_type",
    [ExactInteger, BenignFloat, HostileBoundaryFloat],
    ids=["benign-int-subclass", "benign-float-subclass", "hostile-float-subclass"],
)
@pytest.mark.parametrize("boundary", ["direct", "create", "from_dict", "to_dict"])
def test_best_score_accepts_only_exact_builtin_numbers_without_coercion(
    numeric_type: type[int] | type[float],
    boundary: str,
) -> None:
    _reset_hostile_numeric_protocol_calls()
    score = numeric_type(1)
    base = strategy_artifact()

    if boundary == "direct":
        action = lambda: StrategyArtifact(
            schema_version=base.schema_version,
            run_identity=base.run_identity,
            formula_tokens=base.formula_tokens,
            decoded_formula=base.decoded_formula,
            best_score=score,
            fold_evidence=base.fold_evidence,
            generated_at=base.generated_at,
            fingerprint=base.fingerprint,
        )
    elif boundary == "create":
        action = lambda: StrategyArtifact.create(
            run_identity=base.run_identity,
            formula_tokens=list(base.formula_tokens),
            decoded_formula=base.decoded_formula,
            best_score=score,  # type: ignore[arg-type]
            fold_evidence=list(base.fold_evidence),
            generated_at=base.generated_at,
        )
    elif boundary == "from_dict":
        payload = base.to_dict()
        payload["best_score"] = score
        action = lambda: StrategyArtifact.from_dict(payload)
    else:
        object.__setattr__(base, "best_score", score)
        action = base.to_dict

    with pytest.raises(ArtifactCompatibilityError, match="best_score|JSON") as exc_info:
        action()

    assert len(str(exc_info.value)) <= 1_024
    if numeric_type is HostileBoundaryFloat:
        _assert_no_hostile_numeric_protocol_calls()


@pytest.mark.parametrize("score", [0, 1, -1, 0.0, 1.25, -2.5])
def test_best_score_preserves_exact_builtin_finite_numeric_domain(score: int | float) -> None:
    artifact = StrategyArtifact.create(
        run_identity=TrainingRunIdentity.create(artifact_identity()),
        formula_tokens=[0],
        decoded_formula=FORMULA_VOCAB.token_names[0],
        best_score=score,
        fold_evidence=[fold_evidence_with_index(index) for index in range(4)],
        generated_at="2026-07-15T00:00:00Z",
    )

    assert type(artifact.best_score) is type(score)
    assert StrategyArtifact.from_dict(artifact.to_dict()) == artifact


def test_best_score_exact_type_guard_is_mutation_sensitive(monkeypatch: pytest.MonkeyPatch) -> None:
    base = strategy_artifact()
    original_sha256_json = artifacts_module.sha256_json
    monkeypatch.setattr(
        artifacts_module,
        "sha256_json",
        lambda value: (
            base.fingerprint
            if type(value) is dict and value.get("schema_version") == "strategy-v2"
            else original_sha256_json(value)
        ),
    )

    with pytest.raises(ArtifactCompatibilityError, match="best_score"):
        StrategyArtifact(
            schema_version=base.schema_version,
            run_identity=base.run_identity,
            formula_tokens=base.formula_tokens,
            decoded_formula=base.decoded_formula,
            best_score=ExactInteger(1),
            fold_evidence=base.fold_evidence,
            generated_at=base.generated_at,
            fingerprint=base.fingerprint,
        )


HOSTILE_CLASS_METADATA_CALLS = {"name": 0, "repr": 0, "str": 0}


class HostileClassMetadata(type):
    def __getattribute__(cls, name: str) -> object:
        if name == "__name__":
            HOSTILE_CLASS_METADATA_CALLS["name"] += 1
            raise AssertionError("HOSTILE_STR_CLASS_NAME")
        return super().__getattribute__(name)


class HostileNameString(str, metaclass=HostileClassMetadata):
    def __repr__(self) -> str:
        HOSTILE_CLASS_METADATA_CALLS["repr"] += 1
        raise AssertionError("HOSTILE_STR_REPR")

    def __str__(self) -> str:
        HOSTILE_CLASS_METADATA_CALLS["str"] += 1
        raise AssertionError("HOSTILE_STR_STR")


class HostileNameList(list[object], metaclass=HostileClassMetadata):
    pass


class HostileCauseMeta(type):
    def __getattribute__(cls, name: str) -> object:
        if name == "__name__":
            raise AssertionError("CAUSE_CLASS_NAME")
        return super().__getattribute__(name)


class HostileCause(RuntimeError, metaclass=HostileCauseMeta):
    pass


def _reset_hostile_class_metadata_calls() -> None:
    for key in HOSTILE_CLASS_METADATA_CALLS:
        HOSTILE_CLASS_METADATA_CALLS[key] = 0


@pytest.mark.parametrize(
    "action",
    [
        lambda: ArtifactIdentity(
            core_semantics_version=CORE_SEMANTICS_VERSION,
            vocab_version=VOCAB_VERSION,
            label_semantics_version=LABEL_SEMANTICS_VERSION,
            execution_semantics_version=EXECUTION_SEMANTICS_VERSION,
            symbol=HostileNameString("EURUSD"),
            timeframe="H1",
            training_dataset=dataset_identity(),
            training_config=training_config(),
            training_config_hash=sha256_json(training_config()),
        ),
        lambda: ArtifactIdentity.from_dict(
            {HostileNameString(key): value for key, value in artifact_identity().to_dict().items()}
        ),
        lambda: StrategyArtifact.create(
            run_identity=TrainingRunIdentity.create(artifact_identity()),
            formula_tokens=HostileNameList([0]),
            decoded_formula=FORMULA_VOCAB.token_names[0],
            best_score=1.0,
            fold_evidence=[fold_evidence_with_index(index) for index in range(4)],
            generated_at="2026-07-15T00:00:00Z",
        ),
    ],
    ids=["scalar", "mapping-key", "sequence"],
)
def test_hostile_dynamic_class_metadata_is_never_read(action) -> None:
    _reset_hostile_class_metadata_calls()
    with pytest.raises(ArtifactCompatibilityError) as exc_info:
        action()
    assert len(str(exc_info.value)) <= 1_024
    assert HOSTILE_CLASS_METADATA_CALLS == {"name": 0, "repr": 0, "str": 0}


def test_hostile_exception_class_metadata_is_never_read() -> None:
    payload = HostileExactMapping(
        artifact_identity().to_dict(),
        failure=HostileCause("bounded cause"),
        fail_iteration=True,
    )
    with pytest.raises(ArtifactCompatibilityError) as exc_info:
        ArtifactIdentity.from_dict(payload)
    assert "mapping access failed" in str(exc_info.value)
    assert len(str(exc_info.value)) <= 1_024


HUGE_EXACT_INTEGERS = (10**1000, -(10**1000), 10**5000, -(10**5000))
HUGE_EXACT_INTEGER_IDS = ("pos-1000", "neg-1000", "pos-5000", "neg-5000")


@pytest.mark.parametrize("huge", HUGE_EXACT_INTEGERS, ids=HUGE_EXACT_INTEGER_IDS)
@pytest.mark.parametrize("boundary", ["direct", "from_dict", "to_dict", "verify"])
def test_training_config_huge_exact_integer_fails_closed(huge: int, boundary: str) -> None:
    base = artifact_identity()
    if boundary in {"direct", "from_dict"}:
        payload = base.to_dict()
        payload["training_config"]["cost_rate"] = huge  # type: ignore[index]
        if boundary == "direct":
            action = lambda: artifact_identity_with_config(payload["training_config"])  # type: ignore[arg-type]
        else:
            action = lambda: ArtifactIdentity.from_dict(payload)
    else:
        config = base.to_dict()["training_config"]
        config["cost_rate"] = huge  # type: ignore[index]
        object.__setattr__(base, "training_config", config)
        action = base.to_dict if boundary == "to_dict" else lambda: verify_artifact_identity(base, artifact_identity())
    with pytest.raises(ArtifactCompatibilityError) as exc_info:
        action()
    assert "cost_rate" in str(exc_info.value) or "JSON" in str(exc_info.value)
    assert len(str(exc_info.value)) <= 1_024


@pytest.mark.parametrize("huge", HUGE_EXACT_INTEGERS, ids=HUGE_EXACT_INTEGER_IDS)
def test_canonical_json_rejects_huge_exact_integer_without_raw_conversion_error(huge: int) -> None:
    with pytest.raises(ArtifactCompatibilityError) as exc_info:
        canonical_json_bytes({"value": huge})
    assert "operational finite-real range" in str(exc_info.value)
    assert len(str(exc_info.value)) <= 1_024


@pytest.mark.parametrize("huge", [10**5000, -(10**5000)], ids=["pos-5000", "neg-5000"])
@pytest.mark.parametrize("location", ["checkpoint", "dataset", "fold", "formula_token"])
def test_structural_huge_exact_integer_fails_closed(huge: int, location: str) -> None:
    if location == "checkpoint":
        action = lambda: TrainingRunIdentity.create(artifact_identity()).checkpoint_filename(huge)
    elif location == "dataset":
        payload = artifact_identity().to_dict()
        payload["training_dataset"]["start_time_ns"] = huge  # type: ignore[index]
        action = lambda: ArtifactIdentity.from_dict(payload)
    elif location == "fold":
        payload = fold_evidence_with_index(0).to_dict()
        payload["fold_index"] = huge
        action = lambda: FoldEvidence.from_dict(payload)
    else:
        base = strategy_artifact()
        action = lambda: StrategyArtifact.create(
            run_identity=base.run_identity,
            formula_tokens=[huge],
            decoded_formula=base.decoded_formula,
            best_score=base.best_score,
            fold_evidence=list(base.fold_evidence),
            generated_at=base.generated_at,
        )
    with pytest.raises(ArtifactCompatibilityError) as exc_info:
        action()
    assert len(str(exc_info.value)) <= 1_024


@pytest.mark.parametrize("huge", HUGE_EXACT_INTEGERS, ids=HUGE_EXACT_INTEGER_IDS)
@pytest.mark.parametrize("boundary", ["direct", "from_dict", "to_dict"])
def test_fold_metric_huge_exact_integer_fails_closed(huge: int, boundary: str) -> None:
    payload = fold_evidence_with_index(0).to_dict()
    payload["validation_metrics"]["ic"] = huge  # type: ignore[index]
    if boundary == "direct":
        action = lambda: FoldEvidence(**payload)  # type: ignore[arg-type]
    elif boundary == "from_dict":
        action = lambda: FoldEvidence.from_dict(payload)
    else:
        fold = fold_evidence_with_index(0)
        object.__setattr__(fold, "validation_metrics", {"ic": huge})
        action = fold.to_dict
    with pytest.raises(ArtifactCompatibilityError) as exc_info:
        action()
    assert "validation_metrics" in str(exc_info.value) or "JSON" in str(exc_info.value)
    assert len(str(exc_info.value)) <= 1_024


@pytest.mark.parametrize("huge", HUGE_EXACT_INTEGERS, ids=HUGE_EXACT_INTEGER_IDS)
@pytest.mark.parametrize("boundary", ["direct", "create", "from_dict", "to_dict"])
def test_strategy_best_score_huge_exact_integer_fails_closed(huge: int, boundary: str) -> None:
    base = strategy_artifact()
    if boundary == "direct":
        action = lambda: StrategyArtifact(
            schema_version=base.schema_version,
            run_identity=base.run_identity,
            formula_tokens=base.formula_tokens,
            decoded_formula=base.decoded_formula,
            best_score=huge,
            fold_evidence=base.fold_evidence,
            generated_at=base.generated_at,
            fingerprint=base.fingerprint,
        )
    elif boundary == "create":
        action = lambda: StrategyArtifact.create(
            run_identity=base.run_identity,
            formula_tokens=list(base.formula_tokens),
            decoded_formula=base.decoded_formula,
            best_score=huge,
            fold_evidence=list(base.fold_evidence),
            generated_at=base.generated_at,
        )
    elif boundary == "from_dict":
        payload = base.to_dict()
        payload["best_score"] = huge
        action = lambda: StrategyArtifact.from_dict(payload)
    else:
        object.__setattr__(base, "best_score", huge)
        action = base.to_dict
    with pytest.raises(ArtifactCompatibilityError) as exc_info:
        action()
    assert "best_score" in str(exc_info.value) or "JSON" in str(exc_info.value)
    assert len(str(exc_info.value)) <= 1_024


@pytest.mark.parametrize(
    "value",
    [0, 1, -1, 0.0, 1.25, -2.5, float.fromhex("0x1.fffffffffffffp+1023"), float.fromhex("0x0.0000000000001p-1022")],
)
def test_exact_operational_reals_preserve_type_round_trip_and_fingerprint(value: int | float) -> None:
    artifact = StrategyArtifact.create(
        run_identity=TrainingRunIdentity.create(artifact_identity()),
        formula_tokens=[0],
        decoded_formula=FORMULA_VOCAB.token_names[0],
        best_score=value,
        fold_evidence=[fold_evidence_with_index(index) for index in range(4)],
        generated_at="2026-07-15T00:00:00Z",
    )
    restored = StrategyArtifact.from_dict(artifact.to_dict())
    assert type(restored.best_score) is type(value)
    assert restored.best_score == value
    assert restored.fingerprint == artifact.fingerprint


def _compact_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _duplicate_json_member(
    raw: str,
    *,
    key: str,
    value: object,
    earlier_value: object,
) -> str:
    member = f"{_compact_json(key)}:{_compact_json(value)}"
    assert member in raw
    duplicate = f"{_compact_json(key)}:{_compact_json(earlier_value)},{member}"
    return raw.replace(member, duplicate, 1)


@pytest.mark.parametrize("earlier_score", [1.25, 999])
def test_raw_json_load_rejects_same_and_conflicting_root_duplicates(
    earlier_score: float,
) -> None:
    raw = _compact_json(strategy_artifact().to_dict())
    duplicated = _duplicate_json_member(
        raw,
        key="best_score",
        value=1.25,
        earlier_value=earlier_score,
    )

    with pytest.raises(ArtifactCompatibilityError, match="duplicate JSON object member") as exc_info:
        load_artifact_json(duplicated, StrategyArtifact)

    assert len(str(exc_info.value)) <= 512


@pytest.mark.parametrize(
    ("key", "value", "earlier_value"),
    [
        ("run_id", None, "0" * 32),
        ("core_semantics_version", "2", "1"),
        ("schema_version", "ohlcv-v2", "legacy-dataset"),
        ("batch_size", 192, 1),
        ("coeff_max", ModelConfig.ENTROPY_COEFF_MAX, -1.0),
        ("fold_index", 0, 99),
        ("ic", 0.1, -99.0),
        ("formula_tokens", [0], [1]),
    ],
)
def test_raw_json_load_rejects_duplicates_at_every_nested_artifact_layer(
    key: str,
    value: object,
    earlier_value: object,
) -> None:
    artifact = strategy_artifact()
    if key == "run_id":
        value = artifact.run_identity.run_id
    raw = _compact_json(artifact.to_dict())
    duplicated = _duplicate_json_member(
        raw,
        key=key,
        value=value,
        earlier_value=earlier_value,
    )

    with pytest.raises(ArtifactCompatibilityError, match="duplicate JSON object member"):
        load_artifact_json(duplicated, StrategyArtifact)


@pytest.mark.parametrize(
    ("artifact_type", "value"),
    [
        (DatasetIdentity, dataset_identity()),
        (ArtifactIdentity, artifact_identity()),
        (TrainingRunIdentity, TrainingRunIdentity.create(artifact_identity())),
        (FoldEvidence, fold_evidence_with_index(0)),
        (StrategyArtifact, strategy_artifact()),
    ],
)
def test_raw_json_load_round_trips_every_supported_artifact_type(
    artifact_type: type[object],
    value: object,
) -> None:
    raw = _compact_json(value.to_dict())

    restored = load_artifact_json(raw, artifact_type)

    assert type(restored) is artifact_type
    assert restored == value
    if hasattr(value, "fingerprint"):
        assert restored.fingerprint == value.fingerprint


@pytest.mark.parametrize(
    "raw",
    ["{", "{} trailing", "[]", "null", "1", '"object-like"'],
)
def test_raw_json_load_maps_invalid_trailing_and_non_object_roots(raw: str) -> None:
    with pytest.raises(ArtifactCompatibilityError) as exc_info:
        load_artifact_json(raw, StrategyArtifact)

    assert len(str(exc_info.value)) <= 512
    assert not isinstance(exc_info.value, json.JSONDecodeError)


def test_raw_json_load_rejects_excessive_depth_and_size_with_bounded_errors() -> None:
    excessive_depth = '{"a":' * 300 + "{}" + "}" * 300
    excessive_size = '{"padding":"' + ("x" * 1_100_000) + '"}'

    for raw in (excessive_depth, excessive_size):
        with pytest.raises(ArtifactCompatibilityError) as exc_info:
            load_artifact_json(raw, StrategyArtifact)
        assert len(str(exc_info.value)) <= 512


def test_raw_json_load_rejects_excessive_container_count_with_bounded_error() -> None:
    excessive_count = '{"items":[' + ("[]," * 10_001) + "]}"

    with pytest.raises(ArtifactCompatibilityError, match="container count") as exc_info:
        load_artifact_json(excessive_count, StrategyArtifact)

    assert len(str(exc_info.value)) <= 512


def test_raw_json_load_rejects_unsupported_target_with_bounded_domain_error() -> None:
    with pytest.raises(ArtifactCompatibilityError, match="target type is unsupported") as exc_info:
        load_artifact_json("{}", dict)

    assert len(str(exc_info.value)) <= 1_024


class HostileJsonString(str):
    calls: list[str] = []

    def __str__(self) -> str:
        self.calls.append("str")
        raise AssertionError("hostile str protocol executed")

    def __repr__(self) -> str:
        self.calls.append("repr")
        raise AssertionError("hostile repr protocol executed")

    def __hash__(self) -> int:
        self.calls.append("hash")
        raise AssertionError("hostile hash protocol executed")

    def __eq__(self, other: object) -> bool:
        self.calls.append("eq")
        raise AssertionError("hostile equality protocol executed")


class HostileJsonBytes(bytes):
    calls: list[str] = []

    def __bytes__(self) -> bytes:
        self.calls.append("bytes")
        raise AssertionError("hostile bytes protocol executed")

    def __repr__(self) -> str:
        self.calls.append("repr")
        raise AssertionError("hostile repr protocol executed")

    def __hash__(self) -> int:
        self.calls.append("hash")
        raise AssertionError("hostile hash protocol executed")

    def __eq__(self, other: object) -> bool:
        self.calls.append("eq")
        raise AssertionError("hostile equality protocol executed")


class HostileTextLike:
    calls: list[str] = []

    def __str__(self) -> str:
        self.calls.append("str")
        raise AssertionError("hostile str protocol executed")

    def __repr__(self) -> str:
        self.calls.append("repr")
        raise AssertionError("hostile repr protocol executed")

    def __bytes__(self) -> bytes:
        self.calls.append("bytes")
        raise AssertionError("hostile bytes protocol executed")


@pytest.mark.parametrize(
    "raw",
    [HostileJsonString("{}"), HostileJsonBytes(b"{}"), HostileTextLike()],
)
def test_raw_json_load_rejects_hostile_text_inputs_without_protocol_calls(raw: object) -> None:
    type(raw).calls.clear()

    with pytest.raises(ArtifactCompatibilityError, match="exact built-in str or bytes"):
        load_artifact_json(raw, StrategyArtifact)

    assert type(raw).calls == []


@pytest.mark.parametrize(
    "failure",
    [
        json.JSONDecodeError("bad", "{", 0),
        OverflowError("overflow"),
        RecursionError("recursion"),
        MemoryError("memory"),
    ],
)
def test_raw_json_load_maps_parser_failures_to_bounded_domain_errors(
    monkeypatch: pytest.MonkeyPatch,
    failure: Exception,
) -> None:
    def fail_parse(*args: object, **kwargs: object) -> object:
        raise failure

    monkeypatch.setattr(artifacts_module.json, "loads", fail_parse)

    with pytest.raises(ArtifactCompatibilityError, match="invalid artifact JSON") as exc_info:
        load_artifact_json("{}", StrategyArtifact)

    assert len(str(exc_info.value)) <= 512


@pytest.mark.parametrize("token_count", [100, 5_000])
def test_raw_json_load_bounds_long_formula_structure_errors(token_count: int) -> None:
    payload = strategy_artifact().to_dict()
    abs_token = FORMULA_VOCAB.token_names.index("ABS")
    decay_token = FORMULA_VOCAB.token_names.index("DECAY")
    payload["formula_tokens"] = [abs_token] + [decay_token] * token_count
    raw = _compact_json(payload)

    messages: list[str] = []
    for _ in range(2):
        with pytest.raises(ArtifactCompatibilityError) as exc_info:
            load_artifact_json(raw, StrategyArtifact)
        messages.append(str(exc_info.value))

    assert messages[0] == messages[1]
    assert messages[0].startswith(
        "artifact JSON validation failed: cause=formula structure is incompatible:"
    )
    assert "...<truncated length=" in messages[0]
    assert len(messages[0]) <= 1_024


@pytest.mark.parametrize(
    ("artifact_type", "value"),
    [
        (DatasetIdentity, dataset_identity()),
        (ArtifactIdentity, artifact_identity()),
        (TrainingRunIdentity, TrainingRunIdentity.create(artifact_identity())),
        (FoldEvidence, fold_evidence_with_index(0)),
        (StrategyArtifact, strategy_artifact()),
    ],
)
def test_raw_json_load_bounds_every_supported_downstream_domain_error(
    monkeypatch: pytest.MonkeyPatch,
    artifact_type: type[object],
    value: object,
) -> None:
    failure = ArtifactCompatibilityError("downstream schema failure: " + "x" * 10_000)

    if artifact_type is DatasetIdentity:
        def fail_dataset(*args: object, **kwargs: object) -> object:
            raise failure

        monkeypatch.setattr(
            artifacts_module,
            "_strict_dataset_identity_from_dict",
            fail_dataset,
        )
    else:
        def fail_from_dict(cls: type[object], payload: object) -> object:
            raise failure

        monkeypatch.setattr(artifact_type, "from_dict", classmethod(fail_from_dict))

    with pytest.raises(ArtifactCompatibilityError) as exc_info:
        load_artifact_json(_compact_json(value.to_dict()), artifact_type)

    message = str(exc_info.value)
    assert message.startswith(
        "artifact JSON validation failed: cause=downstream schema failure:"
    )
    assert "...<truncated length=" in message
    assert len(message) <= 1_024
    assert exc_info.value.__cause__ is failure


def test_raw_json_load_does_not_render_hostile_downstream_error_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hostile = HostileTextLike()
    HostileTextLike.calls.clear()
    failure = ArtifactCompatibilityError(hostile)

    def fail_from_dict(cls: type[object], payload: object) -> object:
        raise failure

    monkeypatch.setattr(StrategyArtifact, "from_dict", classmethod(fail_from_dict))

    with pytest.raises(ArtifactCompatibilityError) as exc_info:
        load_artifact_json(_compact_json(strategy_artifact().to_dict()), StrategyArtifact)

    assert exc_info.value.args == (
        "artifact JSON validation failed: cause=<exception>",
    )
    assert exc_info.value.__cause__ is failure
    assert HostileTextLike.calls == []


@pytest.mark.parametrize(
    "failure_case",
    [
        "nested_config",
        "nested_dataset",
        "nested_fold",
        "nested_entropy",
        "unknown_fields",
        "missing_fields",
        "fingerprint",
        "legacy",
    ],
)
def test_raw_json_load_bounds_real_post_parse_validation_matrix(
    failure_case: str,
) -> None:
    payload = strategy_artifact().to_dict()
    run_identity = payload["run_identity"]
    assert type(run_identity) is dict
    identity = run_identity["artifact_identity"]
    assert type(identity) is dict
    config = identity["training_config"]
    assert type(config) is dict

    if failure_case == "nested_config":
        config["restart"] = {"unknown_" + "x" * 5_000: True}
    elif failure_case == "nested_dataset":
        dataset = identity["training_dataset"]
        assert type(dataset) is dict
        dataset["data_fingerprint"] = "x" * 5_000
    elif failure_case == "nested_fold":
        folds = payload["fold_evidence"]
        assert type(folds) is list and type(folds[0]) is dict
        folds[0]["validation_metrics"] = {"metric_" + "x" * 5_000: "invalid"}
    elif failure_case == "nested_entropy":
        entropy = config["entropy"]
        assert type(entropy) is dict
        entropy["unknown_" + "x" * 5_000] = 1
    elif failure_case == "unknown_fields":
        for index in range(100):
            payload[f"unknown_{index}_" + "x" * 100] = index
    elif failure_case == "missing_fields":
        del payload["run_identity"]
    elif failure_case == "fingerprint":
        payload["fingerprint"] = "f" * 5_000
    else:
        payload = {"formula_tokens": [0], "legacy_" + "x" * 5_000: True}

    with pytest.raises(ArtifactCompatibilityError) as exc_info:
        load_artifact_json(_compact_json(payload), StrategyArtifact)

    message = str(exc_info.value)
    assert message.startswith("artifact JSON validation failed: cause=")
    assert len(message) <= 1_024


class DownstreamOrdinaryError(Exception):
    pass


RAW_LOADER_TARGETS = (
    (DatasetIdentity, dataset_identity),
    (ArtifactIdentity, artifact_identity),
    (
        TrainingRunIdentity,
        lambda: TrainingRunIdentity.create(artifact_identity()),
    ),
    (FoldEvidence, lambda: fold_evidence_with_index(0)),
    (StrategyArtifact, strategy_artifact),
)


def _patch_raw_loader_validation_failure(
    monkeypatch: pytest.MonkeyPatch,
    artifact_type: type[object],
    failure: BaseException,
) -> None:
    if artifact_type is DatasetIdentity:
        def fail_dataset(*args: object, **kwargs: object) -> object:
            raise failure

        monkeypatch.setattr(
            artifacts_module,
            "_strict_dataset_identity_from_dict",
            fail_dataset,
        )
        return

    def fail_from_dict(cls: type[object], payload: object) -> object:
        raise failure

    monkeypatch.setattr(artifact_type, "from_dict", classmethod(fail_from_dict))


@pytest.mark.parametrize(
    ("artifact_type", "value_factory"),
    RAW_LOADER_TARGETS,
    ids=["dataset", "artifact", "run", "fold", "strategy"],
)
@pytest.mark.parametrize(
    "failure_type",
    [KeyError, ValueError, RuntimeError, MemoryError, DownstreamOrdinaryError],
    ids=["key", "value", "runtime", "memory", "custom"],
)
def test_raw_json_load_wraps_every_downstream_ordinary_exception(
    monkeypatch: pytest.MonkeyPatch,
    artifact_type: type[object],
    value_factory,
    failure_type: type[Exception],
) -> None:
    failure = failure_type("safe downstream context")
    _patch_raw_loader_validation_failure(monkeypatch, artifact_type, failure)
    raw = _compact_json(value_factory().to_dict())

    messages: list[str] = []
    for _ in range(2):
        with pytest.raises(ArtifactCompatibilityError) as exc_info:
            load_artifact_json(raw, artifact_type)
        assert exc_info.value is not failure
        assert exc_info.value.__cause__ is failure
        messages.append(str(exc_info.value))

    assert messages[0] == messages[1]
    assert messages[0].startswith("artifact JSON validation failed:")
    assert len(messages[0]) <= 1_024


HOSTILE_DOWNSTREAM_CALLS = {
    "args": 0,
    "str": 0,
    "repr": 0,
    "format": 0,
    "class_metadata": 0,
    "bool": 0,
    "bytes": 0,
    "float": 0,
    "index": 0,
    "int": 0,
    "iter": 0,
    "len": 0,
}


class HostileDownstreamArgument:
    def __bool__(self) -> bool:
        HOSTILE_DOWNSTREAM_CALLS["bool"] += 1
        raise AssertionError("hostile bool")

    def __bytes__(self) -> bytes:
        HOSTILE_DOWNSTREAM_CALLS["bytes"] += 1
        raise AssertionError("hostile bytes")

    def __float__(self) -> float:
        HOSTILE_DOWNSTREAM_CALLS["float"] += 1
        raise AssertionError("hostile float")

    def __index__(self) -> int:
        HOSTILE_DOWNSTREAM_CALLS["index"] += 1
        raise AssertionError("hostile index")

    def __int__(self) -> int:
        HOSTILE_DOWNSTREAM_CALLS["int"] += 1
        raise AssertionError("hostile int")

    def __iter__(self):
        HOSTILE_DOWNSTREAM_CALLS["iter"] += 1
        raise AssertionError("hostile iter")

    def __len__(self) -> int:
        HOSTILE_DOWNSTREAM_CALLS["len"] += 1
        raise AssertionError("hostile len")

    def __str__(self) -> str:
        HOSTILE_DOWNSTREAM_CALLS["str"] += 1
        raise AssertionError("hostile argument str")

    def __repr__(self) -> str:
        HOSTILE_DOWNSTREAM_CALLS["repr"] += 1
        raise AssertionError("hostile argument repr")

    def __format__(self, format_spec: str) -> str:
        HOSTILE_DOWNSTREAM_CALLS["format"] += 1
        raise AssertionError("hostile argument format")


class HostileDownstreamMeta(type):
    def __getattribute__(cls, name: str) -> object:
        if name in {"__name__", "__qualname__", "__module__"}:
            HOSTILE_DOWNSTREAM_CALLS["class_metadata"] += 1
            raise AssertionError("hostile exception class metadata")
        return super().__getattribute__(name)


class HostileDownstreamError(Exception, metaclass=HostileDownstreamMeta):
    def __getattribute__(self, name: str) -> object:
        if name == "args":
            HOSTILE_DOWNSTREAM_CALLS["args"] += 1
            raise AssertionError("hostile exception args")
        return super().__getattribute__(name)

    def __str__(self) -> str:
        HOSTILE_DOWNSTREAM_CALLS["str"] += 1
        raise AssertionError("hostile exception str")

    def __repr__(self) -> str:
        HOSTILE_DOWNSTREAM_CALLS["repr"] += 1
        raise AssertionError("hostile exception repr")

    def __format__(self, format_spec: str) -> str:
        HOSTILE_DOWNSTREAM_CALLS["format"] += 1
        raise AssertionError("hostile exception format")


@pytest.mark.parametrize(
    ("artifact_type", "value_factory"),
    RAW_LOADER_TARGETS,
    ids=["dataset", "artifact", "run", "fold", "strategy"],
)
def test_raw_json_load_never_calls_hostile_downstream_exception_protocols(
    monkeypatch: pytest.MonkeyPatch,
    artifact_type: type[object],
    value_factory,
) -> None:
    for key in HOSTILE_DOWNSTREAM_CALLS:
        HOSTILE_DOWNSTREAM_CALLS[key] = 0
    failure = HostileDownstreamError(HostileDownstreamArgument())
    _patch_raw_loader_validation_failure(monkeypatch, artifact_type, failure)

    observed: BaseException | None = None
    try:
        load_artifact_json(_compact_json(value_factory().to_dict()), artifact_type)
    except BaseException as caught:
        observed = caught

    assert type(observed) is ArtifactCompatibilityError
    assert observed.__cause__ is failure
    assert len(str(observed)) <= 1_024
    assert HOSTILE_DOWNSTREAM_CALLS == dict.fromkeys(HOSTILE_DOWNSTREAM_CALLS, 0)


@pytest.mark.parametrize(
    "failure",
    [
        ValueError("x" * 100_000),
        ValueError(*(f"argument-{index}-" + "y" * 10_000 for index in range(100))),
        DownstreamOrdinaryError(HostileDownstreamArgument()),
    ],
    ids=["large-message", "many-huge-arguments", "hostile-huge-argument"],
)
def test_raw_json_load_bounds_large_downstream_exception_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
    failure: Exception,
) -> None:
    _patch_raw_loader_validation_failure(monkeypatch, StrategyArtifact, failure)
    raw = _compact_json(strategy_artifact().to_dict())

    messages: list[str] = []
    for _ in range(2):
        observed: BaseException | None = None
        try:
            load_artifact_json(raw, StrategyArtifact)
        except BaseException as caught:
            observed = caught
        assert type(observed) is ArtifactCompatibilityError
        assert observed.__cause__ is failure
        messages.append(str(observed))

    assert messages[0] == messages[1]
    assert len(messages[0]) <= 1_024


class DownstreamBaseFailure(BaseException):
    pass


@pytest.mark.parametrize(
    "failure",
    [KeyboardInterrupt(), SystemExit(), GeneratorExit(), DownstreamBaseFailure()],
    ids=["keyboard-interrupt", "system-exit", "generator-exit", "custom-base"],
)
def test_raw_json_load_propagates_non_exception_failures_by_identity(
    monkeypatch: pytest.MonkeyPatch,
    failure: BaseException,
) -> None:
    _patch_raw_loader_validation_failure(monkeypatch, StrategyArtifact, failure)

    with pytest.raises(BaseException) as exc_info:
        load_artifact_json(
            _compact_json(strategy_artifact().to_dict()),
            StrategyArtifact,
        )

    assert exc_info.value is failure


HOSTILE_CAUSE_PROTOCOL_CALLS = {
    "cause": 0,
    "args": 0,
    "traceback": 0,
    "context": 0,
    "suppress_context": 0,
    "class_metadata": 0,
    "str": 0,
    "repr": 0,
    "format": 0,
}


class HostileCauseProtocolMeta(type):
    def __getattribute__(cls, name: str) -> object:
        if name in {"__name__", "__qualname__", "__module__"}:
            HOSTILE_CAUSE_PROTOCOL_CALLS["class_metadata"] += 1
            raise RuntimeError("hostile exception class metadata escaped")
        return super().__getattribute__(name)


class HostileCauseProtocolError(
    ArtifactCompatibilityError,
    metaclass=HostileCauseProtocolMeta,
):
    def __getattribute__(self, name: str) -> object:
        protocol = {
            "__cause__": "cause",
            "args": "args",
            "__traceback__": "traceback",
            "__context__": "context",
            "__suppress_context__": "suppress_context",
            "__class__": "class_metadata",
        }.get(name)
        if protocol is not None:
            HOSTILE_CAUSE_PROTOCOL_CALLS[protocol] += 1
            raise RuntimeError(f"hostile {protocol} getter escaped")
        return super().__getattribute__(name)

    def __str__(self) -> str:
        HOSTILE_CAUSE_PROTOCOL_CALLS["str"] += 1
        raise RuntimeError("hostile exception str escaped")

    def __repr__(self) -> str:
        HOSTILE_CAUSE_PROTOCOL_CALLS["repr"] += 1
        raise RuntimeError("hostile exception repr escaped")

    def __format__(self, format_spec: str) -> str:
        HOSTILE_CAUSE_PROTOCOL_CALLS["format"] += 1
        raise RuntimeError("hostile exception format escaped")


def _reset_hostile_cause_protocol_calls() -> None:
    for key in HOSTILE_CAUSE_PROTOCOL_CALLS:
        HOSTILE_CAUSE_PROTOCOL_CALLS[key] = 0


def _set_builtin_exception_cause(
    failure: BaseException,
    cause: BaseException | None,
) -> None:
    BaseException.__cause__.__set__(failure, cause)


@pytest.mark.parametrize(
    ("artifact_type", "value_factory"),
    RAW_LOADER_TARGETS,
    ids=["dataset", "artifact", "run", "fold", "strategy"],
)
def test_raw_json_all_dispatch_paths_ignore_hostile_exception_metadata(
    monkeypatch: pytest.MonkeyPatch,
    artifact_type: type[object],
    value_factory,
) -> None:
    failure = HostileCauseProtocolError("untrusted")
    _patch_raw_loader_validation_failure(monkeypatch, artifact_type, failure)
    _reset_hostile_cause_protocol_calls()

    observed: BaseException | None = None
    try:
        load_artifact_json(_compact_json(value_factory().to_dict()), artifact_type)
    except BaseException as caught:
        observed = caught

    type_is_bounded_domain_error = type(observed) is ArtifactCompatibilityError
    cause_is_expected = (
        BaseException.__getattribute__(observed, "__cause__") is failure
    )
    message_is_bounded = len(str(observed)) <= 1_024
    BaseException.__context__.__set__(observed, None)
    del failure, observed
    assert type_is_bounded_domain_error
    assert cause_is_expected
    assert message_is_bounded
    assert HOSTILE_CAUSE_PROTOCOL_CALLS == dict.fromkeys(
        HOSTILE_CAUSE_PROTOCOL_CALLS,
        0,
    )


def _hostile_cause_case(case: str) -> tuple[BaseException, BaseException]:
    if case == "exact-no-cause":
        failure = ArtifactCompatibilityError("exact bounded failure")
        return failure, failure
    if case == "exact-safe-cause":
        failure = ArtifactCompatibilityError("exact bounded failure")
        safe_cause = RuntimeError("safe underlying cause")
        _set_builtin_exception_cause(failure, safe_cause)
        return failure, safe_cause

    failure = HostileCauseProtocolError("untrusted")
    if case == "subclass-no-cause":
        return failure, failure
    if case == "subclass-safe-cause":
        safe_cause = RuntimeError("safe underlying cause")
        _set_builtin_exception_cause(failure, safe_cause)
        return failure, safe_cause
    if case == "subclass-hostile-cause-object":
        hostile_cause = HostileCauseProtocolError("hostile underlying cause")
        _set_builtin_exception_cause(failure, hostile_cause)
        return failure, hostile_cause
    if case == "subclass-long-cause-chain":
        chain: BaseException = RuntimeError("tail")
        for index in range(300):
            parent = RuntimeError(f"cause-{index}")
            _set_builtin_exception_cause(parent, chain)
            chain = parent
        _set_builtin_exception_cause(failure, chain)
        return failure, chain
    if case == "subclass-cause-cycle":
        _set_builtin_exception_cause(failure, failure)
        return failure, failure
    raise AssertionError(case)


@pytest.mark.parametrize(
    "case",
    [
        "exact-no-cause",
        "exact-safe-cause",
        "subclass-no-cause",
        "subclass-safe-cause",
        "subclass-hostile-cause-object",
        "subclass-long-cause-chain",
        "subclass-cause-cycle",
    ],
)
def test_raw_json_cause_selection_is_safe_bounded_and_deterministic(
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    failure, expected_cause = _hostile_cause_case(case)
    _patch_raw_loader_validation_failure(monkeypatch, StrategyArtifact, failure)
    raw = _compact_json(strategy_artifact().to_dict())
    _reset_hostile_cause_protocol_calls()

    messages: list[str] = []
    observations_are_bounded_domain_errors = True
    causes_are_expected = True
    for _ in range(2):
        observed: BaseException | None = None
        try:
            load_artifact_json(raw, StrategyArtifact)
        except BaseException as caught:
            observed = caught
        observations_are_bounded_domain_errors &= (
            type(observed) is ArtifactCompatibilityError
        )
        causes_are_expected &= (
            BaseException.__getattribute__(observed, "__cause__") is expected_cause
        )
        messages.append(str(observed))
        BaseException.__context__.__set__(observed, None)

    del failure, expected_cause, observed
    assert observations_are_bounded_domain_errors
    assert causes_are_expected
    assert messages[0] == messages[1]
    assert len(messages[0]) <= 1_024
    assert HOSTILE_CAUSE_PROTOCOL_CALLS == dict.fromkeys(
        HOSTILE_CAUSE_PROTOCOL_CALLS,
        0,
    )


@pytest.mark.parametrize(
    "boundary",
    [
        "artifact-snapshot",
        "strategy-snapshot",
        "backtest-dataset-snapshot",
        "backtest-dataset-validation",
        "backtest-strategy-training-dataset",
        "backtest-strategy-generic",
    ],
)
def test_related_t06_wrappers_ignore_hostile_exception_metadata(
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
) -> None:
    failure = HostileCauseProtocolError("untrusted")
    expected_type: type[BaseException]

    if boundary == "artifact-snapshot":
        base = artifact_identity()

        class RaisingArtifactIdentity(ArtifactIdentity):
            def __getattribute__(self, name: str) -> object:
                armed = object.__getattribute__(self, "__dict__").get("_armed", False)
                if armed and name == "core_semantics_version":
                    raise failure
                return super().__getattribute__(name)

        value = RaisingArtifactIdentity(**base.__dict__)
        object.__setattr__(value, "_armed", True)
        action = lambda: artifacts_module._validated_artifact_identity_snapshot(
            value,
            context="hostile identity",
        )
        expected_type = ArtifactCompatibilityError
    elif boundary == "strategy-snapshot":
        base = strategy_artifact()

        class RaisingStrategyArtifact(StrategyArtifact):
            def __getattribute__(self, name: str) -> object:
                armed = object.__getattribute__(self, "__dict__").get("_armed", False)
                if armed and name == "schema_version":
                    raise failure
                return super().__getattribute__(name)

        value = RaisingStrategyArtifact(**base.__dict__)
        object.__setattr__(value, "_armed", True)
        action = lambda: artifacts_module._validated_strategy_artifact_snapshot(
            value,
            context="hostile strategy",
        )
        expected_type = ArtifactCompatibilityError
    else:
        strategy_value = strategy_artifact()
        dataset_value = dataset_identity()
        if boundary == "backtest-dataset-snapshot":
            monkeypatch.setattr(
                artifacts_module,
                "_snapshot_dataset_identity_payload",
                lambda *args, **kwargs: (_ for _ in ()).throw(failure),
            )
        elif boundary == "backtest-dataset-validation":
            monkeypatch.setattr(
                artifacts_module,
                "_dataset_identity_from_payload",
                lambda *args, **kwargs: (_ for _ in ()).throw(failure),
            )
        else:
            monkeypatch.setattr(
                artifacts_module,
                "_validated_strategy_artifact_snapshot",
                lambda *args, **kwargs: (_ for _ in ()).throw(failure),
            )
            if boundary == "backtest-strategy-training-dataset":
                original_safe_text = artifacts_module._safe_diagnostic_text

                def safe_text(value: object) -> str:
                    if value is failure:
                        return "training_dataset hostile failure"
                    return original_safe_text(value)

                monkeypatch.setattr(artifacts_module, "_safe_diagnostic_text", safe_text)
        action = lambda: artifacts_module.validate_backtest_dataset(
            strategy_value,
            dataset_value,
            artifacts_module.BacktestMode.IN_SAMPLE_REPLAY,
        )
        expected_type = artifacts_module.BacktestModeError

    _reset_hostile_cause_protocol_calls()
    observed: BaseException | None = None
    try:
        action()
    except BaseException as caught:
        observed = caught

    type_is_expected = type(observed) is expected_type
    cause_is_expected = (
        BaseException.__getattribute__(observed, "__cause__") is failure
    )
    message_is_bounded = len(str(observed)) <= 1_024
    BaseException.__context__.__set__(observed, None)
    del failure, observed
    assert type_is_expected
    assert cause_is_expected
    assert message_is_bounded
    assert HOSTILE_CAUSE_PROTOCOL_CALLS == dict.fromkeys(
        HOSTILE_CAUSE_PROTOCOL_CALLS,
        0,
    )
