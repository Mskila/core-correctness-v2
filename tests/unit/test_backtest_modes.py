from __future__ import annotations

from dataclasses import replace

import pytest

from data_pipeline.validation import DatasetIdentity
import model_core.artifacts as artifacts_module
from model_core.config import ModelConfig
from model_core.artifacts import (
    ArtifactIdentity,
    BacktestMode,
    FoldEvidence,
    StrategyArtifact,
    TrainingRunIdentity,
    sha256_json,
    validate_backtest_dataset,
)
from model_core.semantics import (
    CORE_SEMANTICS_VERSION,
    EXECUTION_SEMANTICS_VERSION,
    LABEL_LOOKAHEAD_BARS,
    LABEL_SEMANTICS_VERSION,
    ArtifactCompatibilityError,
    BacktestModeError,
    DataValidationError,
)
from model_core.vocab import FORMULA_VOCAB, VOCAB_VERSION


class BacktestIntegerSubclass(int):
    pass


class BacktestProtocolTrackingInteger(int):
    def __new__(cls, value: int) -> "BacktestProtocolTrackingInteger":
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


BACKTEST_HOSTILE_META_CALLS = {"name": 0}


class BacktestHostileIntegerMeta(type):
    def __getattribute__(cls, name: str) -> object:
        if name == "__name__":
            BACKTEST_HOSTILE_META_CALLS["name"] += 1
            raise AssertionError("HOSTILE_CLASS_NAME")
        return super().__getattribute__(name)


class BacktestHostileMetaInteger(int, metaclass=BacktestHostileIntegerMeta):
    pass


BACKTEST_HOSTILE_GETTER_EXCEPTION_CALLS = {"name": 0, "repr": 0, "str": 0}


class BacktestHostileGetterExceptionMeta(type):
    def __getattribute__(cls, name: str) -> object:
        if name == "__name__":
            BACKTEST_HOSTILE_GETTER_EXCEPTION_CALLS["name"] += 1
            raise AssertionError("HOSTILE_EXCEPTION_CLASS_NAME")
        return super().__getattribute__(name)


class BacktestHostileGetterError(
    Exception,
    metaclass=BacktestHostileGetterExceptionMeta,
):
    def __repr__(self) -> str:
        BACKTEST_HOSTILE_GETTER_EXCEPTION_CALLS["repr"] += 1
        raise AssertionError("HOSTILE_EXCEPTION_REPR")

    def __str__(self) -> str:
        BACKTEST_HOSTILE_GETTER_EXCEPTION_CALLS["str"] += 1
        raise AssertionError("HOSTILE_EXCEPTION_STR")


class BacktestCustomGetterBaseException(BaseException):
    pass


BACKTEST_HOSTILE_NUMERIC_CALLS = {
    "name": 0,
    "repr": 0,
    "str": 0,
    "int": 0,
    "index": 0,
    "eq": 0,
    "hash": 0,
    "float": 0,
}


class BacktestHostileNumericMeta(type):
    def __getattribute__(cls, name: str) -> object:
        if name == "__name__":
            BACKTEST_HOSTILE_NUMERIC_CALLS["name"] += 1
            raise AssertionError("HOSTILE_CLASS_NAME")
        return super().__getattribute__(name)


class BacktestHostileBoundaryInteger(int, metaclass=BacktestHostileNumericMeta):
    def __repr__(self) -> str:
        BACKTEST_HOSTILE_NUMERIC_CALLS["repr"] += 1
        raise AssertionError("HOSTILE_REPR")

    def __str__(self) -> str:
        BACKTEST_HOSTILE_NUMERIC_CALLS["str"] += 1
        raise AssertionError("HOSTILE_STR")

    def __int__(self) -> int:
        BACKTEST_HOSTILE_NUMERIC_CALLS["int"] += 1
        raise AssertionError("HOSTILE_INT")

    def __index__(self) -> int:
        BACKTEST_HOSTILE_NUMERIC_CALLS["index"] += 1
        raise AssertionError("HOSTILE_INDEX")

    def __eq__(self, other: object) -> bool:
        BACKTEST_HOSTILE_NUMERIC_CALLS["eq"] += 1
        raise AssertionError("HOSTILE_EQ")

    def __hash__(self) -> int:
        BACKTEST_HOSTILE_NUMERIC_CALLS["hash"] += 1
        raise AssertionError("HOSTILE_HASH")


class BacktestHostileBoundaryFloat(float, metaclass=BacktestHostileNumericMeta):
    def __repr__(self) -> str:
        BACKTEST_HOSTILE_NUMERIC_CALLS["repr"] += 1
        raise AssertionError("HOSTILE_REPR")

    def __str__(self) -> str:
        BACKTEST_HOSTILE_NUMERIC_CALLS["str"] += 1
        raise AssertionError("HOSTILE_STR")

    def __float__(self) -> float:
        BACKTEST_HOSTILE_NUMERIC_CALLS["float"] += 1
        raise AssertionError("HOSTILE_FLOAT_COERCION")

    def __eq__(self, other: object) -> bool:
        BACKTEST_HOSTILE_NUMERIC_CALLS["eq"] += 1
        raise AssertionError("HOSTILE_EQ")

    def __hash__(self) -> int:
        BACKTEST_HOSTILE_NUMERIC_CALLS["hash"] += 1
        raise AssertionError("HOSTILE_HASH")


class BacktestHostileBoundaryString(str, metaclass=BacktestHostileNumericMeta):
    def __repr__(self) -> str:
        BACKTEST_HOSTILE_NUMERIC_CALLS["repr"] += 1
        raise AssertionError("HOSTILE_REPR")

    def __str__(self) -> str:
        BACKTEST_HOSTILE_NUMERIC_CALLS["str"] += 1
        raise AssertionError("HOSTILE_STR")


def reset_backtest_hostile_numeric_calls() -> None:
    for key in BACKTEST_HOSTILE_NUMERIC_CALLS:
        BACKTEST_HOSTILE_NUMERIC_CALLS[key] = 0


class BacktestHostileList(list[object]):
    def __init__(self, values: list[object], failure: Exception) -> None:
        list.__init__(self, values)
        self.failure = failure
        self.protocol_calls = {"iter": 0, "len": 0, "getitem": 0, "repr": 0, "str": 0}

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


def dataset(
    *,
    symbol: str = "EURUSD",
    timeframe: str = "H1",
    start: int = 1_000,
    end: int = 10_000,
    data_fp: str = "a" * 64,
    time_fp: str = "b" * 64,
    bars: int = 10,
) -> DatasetIdentity:
    return DatasetIdentity(
        schema_version="ohlcv-v2",
        symbol=symbol,
        timeframe=timeframe,
        start_time_ns=start,
        end_time_ns=end,
        bars=bars,
        data_fingerprint=data_fp,
        time_fingerprint=time_fp,
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
    valid_source = dataset() if source is None else source
    unknown = set(changes).difference(DATASET_FIELDS)
    assert not unknown, f"unknown DatasetIdentity fields: {sorted(unknown)}"
    hostile = object.__new__(DatasetIdentity)
    for field in DATASET_FIELDS:
        value = changes[field] if field in changes else getattr(valid_source, field)
        object.__setattr__(hostile, field, value)
    return hostile


def hostile_dataset(
    field: str,
    *,
    failure: BaseException | None = None,
    **overrides: object,
) -> DatasetIdentity:
    values: dict[str, object] = {
        "schema_version": "ohlcv-v2",
        "symbol": "EURUSD",
        "timeframe": "H1",
        "start_time_ns": 1_000,
        "end_time_ns": 10_000,
        "bars": 10,
        "data_fingerprint": "a" * 64,
        "time_fingerprint": "b" * 64,
    }
    values.update(overrides)

    class HostileDatasetIdentity(DatasetIdentity):
        def __getattribute__(self, name: str) -> object:
            if name == field:
                actual_failure = (
                    failure
                    if failure is not None
                    else RuntimeError(f"hostile attribute access: {field}")
                )
                raise actual_failure
            return super().__getattribute__(name)

    return HostileDatasetIdentity(**values)  # type: ignore[arg-type]


def training_config() -> dict[str, object]:
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
        "entropy": {
            "coeff_max": ModelConfig.ENTROPY_COEFF_MAX,
            "coeff_power": ModelConfig.ENTROPY_COEFF_POWER,
            "collapse_thresh": ModelConfig.ENTROPY_COLLAPSE_THRESH,
            "collapse_steps": ModelConfig.ENTROPY_COLLAPSE_STEPS,
            "floor_enabled": ModelConfig.ENTROPY_FLOOR,
            "floor_thresh": ModelConfig.ENTROPY_FLOOR_THRESH,
            "floor_lambda": ModelConfig.ENTROPY_FLOOR_LAMBDA,
        },
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
            "use_lord_regularization": ModelConfig.USE_LORD_REGULARIZATION,
            "lord_decay_rate": ModelConfig.LORD_DECAY_RATE,
            "lord_num_iterations": ModelConfig.LORD_NUM_ITERATIONS,
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
        "random_seed": 42,
    }


def strategy() -> StrategyArtifact:
    config = training_config()
    identity = ArtifactIdentity(
        core_semantics_version=CORE_SEMANTICS_VERSION,
        vocab_version=VOCAB_VERSION,
        label_semantics_version=LABEL_SEMANTICS_VERSION,
        execution_semantics_version=EXECUTION_SEMANTICS_VERSION,
        symbol="EURUSD",
        timeframe="H1",
        training_dataset=dataset(),
        training_config=config,
        training_config_hash=sha256_json(config),
    )
    return StrategyArtifact.create(
        run_identity=TrainingRunIdentity.create(identity),
        formula_tokens=[0],
        decoded_formula=FORMULA_VOCAB.token_names[0],
        best_score=1.0,
        fold_evidence=[
            FoldEvidence(
                fold_index=index,
                train_start_time_ns=1_000,
                train_end_time_ns=2_000 + index * 1_500,
                val_start_time_ns=2_500 + index * 1_500,
                val_end_time_ns=3_000 + index * 1_500,
                effective_gap=LABEL_LOOKAHEAD_BARS,
                validation_metrics={"ic": 0.1 + index * 0.01, "net_return": 0.02},
            )
            for index in range(4)
        ],
        generated_at="2026-07-15T00:00:00Z",
    )


def test_backtest_rejects_five_block_strategy_without_four_fold_evidence() -> None:
    current = strategy()
    object.__setattr__(current, "fold_evidence", ())
    object.__setattr__(current, "fingerprint", sha256_json(current._payload_without_fingerprint()))

    with pytest.raises(
        BacktestModeError,
        match=r"fold_evidence count.*expected=4.*actual=0",
    ):
        validate_backtest_dataset(
            current,
            dataset(),
            BacktestMode.IN_SAMPLE_REPLAY,
        )


@pytest.mark.parametrize("fold_count", [0, 1, 2, 3, 5])
@pytest.mark.parametrize(
    "mode",
    [BacktestMode.IN_SAMPLE_REPLAY, BacktestMode.OUT_OF_SAMPLE_BACKTEST],
)
def test_backtest_revalidation_rejects_every_wrong_five_block_fold_count(
    fold_count: int,
    mode: BacktestMode,
) -> None:
    current = strategy()
    folds = list(current.fold_evidence)
    folds.append(
        FoldEvidence(
            fold_index=4,
            train_start_time_ns=1_000,
            train_end_time_ns=8_000,
            val_start_time_ns=8_500,
            val_end_time_ns=9_000,
            effective_gap=LABEL_LOOKAHEAD_BARS,
            validation_metrics={"ic": 0.14, "net_return": 0.02},
        )
    )
    object.__setattr__(current, "fold_evidence", tuple(folds[:fold_count]))
    object.__setattr__(current, "fingerprint", sha256_json(current._payload_without_fingerprint()))
    test_identity = (
        dataset()
        if mode is BacktestMode.IN_SAMPLE_REPLAY
        else dataset(
            start=11_000,
            end=20_000,
            data_fp="c" * 64,
            time_fp="d" * 64,
        )
    )

    with pytest.raises(
        BacktestModeError,
        match=rf"fold_evidence count.*expected=4.*actual={fold_count}",
    ):
        validate_backtest_dataset(current, test_identity, mode)


BACKTEST_STRATEGY_MUTATIONS = (
    ("schema_version", "schema"),
    ("run_identity", "run_identity"),
    ("formula_tokens", "formula token"),
    ("decoded_formula", "decoded_formula"),
    ("best_score", "best_score"),
    ("fold_evidence", "validation_metrics"),
    ("generated_at", "generated_at"),
    ("fingerprint", "fingerprint"),
)


def mutate_strategy_for_backtest(
    artifact: StrategyArtifact,
    mutation: str,
) -> None:
    original_payload = artifact._payload_without_fingerprint()
    recompute = True
    if mutation == "schema_version":
        object.__setattr__(artifact, "schema_version", "strategy-v1")
    elif mutation == "run_identity":
        object.__setattr__(artifact, "run_identity", "not-a-run-identity")
        recompute = False
    elif mutation == "formula_tokens":
        object.__setattr__(artifact, "formula_tokens", (True,))
    elif mutation == "decoded_formula":
        object.__setattr__(artifact, "decoded_formula", "DIFFERENT_FORMULA")
    elif mutation == "best_score":
        object.__setattr__(artifact, "best_score", True)
    elif mutation == "fold_evidence":
        fold = FoldEvidence(
            fold_index=0,
            train_start_time_ns=1_000,
            train_end_time_ns=4_000,
            val_start_time_ns=6_000,
            val_end_time_ns=9_000,
            effective_gap=LABEL_LOOKAHEAD_BARS,
            validation_metrics={"ic": 0.1},
        )
        object.__setattr__(fold, "validation_metrics", {"ic": "not-a-number"})
        object.__setattr__(artifact, "fold_evidence", (fold,))
    elif mutation == "generated_at":
        object.__setattr__(artifact, "generated_at", "2026-07-15T00:00:00")
    elif mutation == "fingerprint":
        object.__setattr__(artifact, "fingerprint", "0" * 64)
        recompute = False
    else:
        raise AssertionError(f"unknown strategy mutation: {mutation}")
    if recompute:
        if mutation == "fold_evidence":
            original_payload["fold_evidence"] = [
                {
                    "fold_index": 0,
                    "train_start_time_ns": 1_000,
                    "train_end_time_ns": 4_000,
                    "val_start_time_ns": 6_000,
                    "val_end_time_ns": 9_000,
                    "effective_gap": LABEL_LOOKAHEAD_BARS,
                    "validation_metrics": {"ic": "not-a-number"},
                }
            ]
            fingerprint_payload = original_payload
        else:
            fingerprint_payload = artifact._payload_without_fingerprint()
        object.__setattr__(
            artifact,
            "fingerprint",
            sha256_json(fingerprint_payload),
        )


def assert_mode_error(test_data: DatasetIdentity, mode: BacktestMode) -> str:
    with pytest.raises(BacktestModeError) as exc_info:
        validate_backtest_dataset(strategy(), test_data, mode)
    message = str(exc_info.value)
    assert f"mode={mode.value}" in message
    assert "train_range=[1000,10000]" in message
    assert f"test_range=[{test_data.start_time_ns},{test_data.end_time_ns}]" in message
    assert "train_data_fp=aaaaaaaaaaaa" in message
    assert f"test_data_fp={test_data.data_fingerprint[:12]}" in message
    return message


def test_exact_training_data_replay_succeeds() -> None:
    validate_backtest_dataset(strategy(), dataset(), BacktestMode.IN_SAMPLE_REPLAY)


@pytest.mark.parametrize("mode", ["standard", "ftmo", "forex"])
def test_current_reward_modes_survive_strategy_and_backtest_round_trip(
    mode: str,
) -> None:
    current = strategy()
    payload = current.to_dict()
    config = training_config()
    config["reward"]["mode"] = mode  # type: ignore[index]
    identity_payload = payload["run_identity"]["artifact_identity"]  # type: ignore[index]
    identity_payload["training_config"] = config
    identity_payload["training_config_hash"] = sha256_json(config)
    payload["fingerprint"] = sha256_json(
        {key: value for key, value in payload.items() if key != "fingerprint"}
    )
    round_tripped = StrategyArtifact.from_dict(payload)
    assert round_tripped.to_dict() == payload
    validate_backtest_dataset(
        round_tripped,
        dataset(),
        BacktestMode.IN_SAMPLE_REPLAY,
    )


class RewardModeLike(str):
    pass


@pytest.mark.parametrize(
    "mode",
    [
        "",
        " ",
        "totally-unknown-mode",
        "STANDARD",
        "FtMo",
        " standard",
        "standard ",
        1,
        True,
        None,
        [],
        RewardModeLike("standard"),
    ],
)
def test_backtest_rejects_bypass_mutated_reward_mode_before_fingerprint(
    mode: object,
) -> None:
    current = strategy()
    payload = current.to_dict()
    config = training_config()
    config["reward"]["mode"] = mode  # type: ignore[index]
    identity_payload = payload["run_identity"]["artifact_identity"]  # type: ignore[index]
    identity_payload["training_config"] = config
    identity_payload["training_config_hash"] = (
        sha256_json(config) if type(mode) is str else "0" * 64
    )
    payload["fingerprint"] = (
        sha256_json({key: value for key, value in payload.items() if key != "fingerprint"})
        if type(mode) is str
        else "0" * 64
    )
    identity = current.run_identity.artifact_identity
    object.__setattr__(identity, "training_config", config)
    object.__setattr__(
        identity,
        "training_config_hash",
        identity_payload["training_config_hash"],
    )
    object.__setattr__(current, "fingerprint", payload["fingerprint"])
    expected = (
        r"exact built-in string"
        if isinstance(mode, str) and type(mode) is not str
        else r"training_config\.reward\.mode.*standard.*ftmo.*forex"
    )
    with pytest.raises(BacktestModeError, match=expected):
        validate_backtest_dataset(
            current,
            dataset(),
            BacktestMode.IN_SAMPLE_REPLAY,
        )


@pytest.mark.parametrize("boundary", ["create", "direct", "from_dict", "to_dict"])
def test_strategy_revalidates_fingerprint_valid_unknown_reward_mode(
    boundary: str,
) -> None:
    current = strategy()
    payload = current.to_dict()
    config = training_config()
    config["reward"]["mode"] = "totally-unknown-mode"  # type: ignore[index]
    identity_payload = payload["run_identity"]["artifact_identity"]  # type: ignore[index]
    identity_payload["training_config"] = config
    identity_payload["training_config_hash"] = sha256_json(config)
    payload["fingerprint"] = sha256_json(
        {key: value for key, value in payload.items() if key != "fingerprint"}
    )
    if boundary == "from_dict":
        action = lambda: StrategyArtifact.from_dict(payload)
    else:
        identity = current.run_identity.artifact_identity
        object.__setattr__(identity, "training_config", config)
        object.__setattr__(identity, "training_config_hash", sha256_json(config))
        object.__setattr__(current, "fingerprint", payload["fingerprint"])
        if boundary == "create":
            action = lambda: StrategyArtifact.create(
                run_identity=current.run_identity,
                formula_tokens=[0],
                decoded_formula=FORMULA_VOCAB.token_names[0],
                best_score=1.0,
                fold_evidence=[],
                generated_at="2026-07-15T00:00:00Z",
            )
        elif boundary == "direct":
            action = lambda: StrategyArtifact(
                schema_version="strategy-v2",
                run_identity=current.run_identity,
                formula_tokens=(0,),
                decoded_formula=FORMULA_VOCAB.token_names[0],
                best_score=1.0,
                fold_evidence=(),
                generated_at="2026-07-15T00:00:00Z",
                fingerprint=payload["fingerprint"],  # type: ignore[arg-type]
            )
        else:
            action = current.to_dict
    with pytest.raises(
        ArtifactCompatibilityError,
        match=r"training_config\.reward\.mode.*standard.*ftmo.*forex",
    ):
        action()


@pytest.mark.parametrize(
    ("mutation", "message"),
    BACKTEST_STRATEGY_MUTATIONS,
)
def test_backtest_revalidates_every_strategy_field_after_bypass_mutation(
    mutation: str,
    message: str,
) -> None:
    artifact = strategy()
    mutate_strategy_for_backtest(artifact, mutation)
    with pytest.raises(BacktestModeError) as exc_info:
        validate_backtest_dataset(
            artifact,
            dataset(),
            BacktestMode.IN_SAMPLE_REPLAY,
        )
    error = str(exc_info.value)
    assert "invalid strategy artifact" in error
    assert message in error
    assert "mode=in_sample_replay" in error
    assert "test_range=[1000,10000]" in error


def test_backtest_rejects_bool_token_with_recomputed_strategy_fingerprint() -> None:
    artifact = strategy()
    object.__setattr__(artifact, "formula_tokens", (True,))
    object.__setattr__(
        artifact,
        "fingerprint",
        sha256_json(artifact._payload_without_fingerprint()),
    )
    with pytest.raises(BacktestModeError, match="formula token"):
        validate_backtest_dataset(
            artifact,
            dataset(),
            BacktestMode.IN_SAMPLE_REPLAY,
        )


def test_other_data_replay_fails() -> None:
    message = assert_mode_error(
        dataset(data_fp="c" * 64),
        BacktestMode.IN_SAMPLE_REPLAY,
    )
    assert "exact training data" in message


@pytest.mark.parametrize(
    "test_data",
    [
        dataset(),
        dataset(start=2_000, end=8_000, data_fp="c" * 64, time_fp="d" * 64),
        dataset(start=1_000, end=12_000, data_fp="c" * 64, time_fp="d" * 64),
        dataset(start=9_000, end=12_000, data_fp="c" * 64, time_fp="d" * 64),
        dataset(start=0, end=900, data_fp="c" * 64, time_fp="d" * 64),
    ],
)
def test_oos_rejects_training_equal_subset_covering_overlap_or_earlier(test_data: DatasetIdentity) -> None:
    assert_mode_error(test_data, BacktestMode.OUT_OF_SAMPLE_BACKTEST)


@pytest.mark.parametrize(
    "test_data",
    [
        dataset(start=11_000, end=20_000, data_fp="a" * 64, time_fp="d" * 64),
        dataset(start=11_000, end=20_000, data_fp="c" * 64, time_fp="b" * 64),
    ],
)
def test_oos_requires_both_data_and_time_fingerprints_to_differ(test_data: DatasetIdentity) -> None:
    assert_mode_error(test_data, BacktestMode.OUT_OF_SAMPLE_BACKTEST)


def test_strictly_later_independent_oos_succeeds() -> None:
    validate_backtest_dataset(
        strategy(),
        dataset(start=11_000, end=20_000, data_fp="c" * 64, time_fp="d" * 64),
        BacktestMode.OUT_OF_SAMPLE_BACKTEST,
    )


@pytest.mark.parametrize(
    "test_data",
    [dataset(symbol="GBPUSD"), dataset(timeframe="H4")],
)
def test_symbol_or_timeframe_mismatch_fails_in_every_mode(test_data: DatasetIdentity) -> None:
    for mode in BacktestMode:
        assert_mode_error(test_data, mode)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("core_semantics_version", "1"),
        ("vocab_version", "v-old"),
        ("label_semantics_version", "label-old"),
        ("execution_semantics_version", "execution-old"),
    ],
)
def test_backtest_revalidates_strategy_semantics(field: str, value: str) -> None:
    current = strategy()
    with pytest.raises(ArtifactCompatibilityError, match=field):
        replace(current.run_identity.artifact_identity, **{field: value})

    tampered = strategy()
    object.__setattr__(tampered.run_identity.artifact_identity, field, value)
    with pytest.raises(BacktestModeError, match=field):
        validate_backtest_dataset(tampered, dataset(), BacktestMode.IN_SAMPLE_REPLAY)


@pytest.mark.parametrize(
    ("container", "field", "actual", "expected"),
    [
        ("model", "input_dim", FORMULA_VOCAB.feature_count + 1, FORMULA_VOCAB.feature_count),
        ("walk_forward", "label_lookahead", LABEL_LOOKAHEAD_BARS + 1, LABEL_LOOKAHEAD_BARS),
    ],
)
def test_backtest_revalidates_authoritative_training_config_semantics(
    container: str,
    field: str,
    actual: int,
    expected: int,
) -> None:
    current = strategy()
    payload = current.to_dict()
    config = training_config()
    config[container][field] = actual  # type: ignore[index]
    identity = current.run_identity.artifact_identity
    object.__setattr__(identity, "training_config", config)
    object.__setattr__(identity, "training_config_hash", sha256_json(config))
    payload["run_identity"]["artifact_identity"]["training_config"] = config  # type: ignore[index]
    payload["run_identity"]["artifact_identity"]["training_config_hash"] = sha256_json(config)  # type: ignore[index]
    object.__setattr__(
        current,
        "fingerprint",
        sha256_json({key: value for key, value in payload.items() if key != "fingerprint"}),
    )
    with pytest.raises(
        BacktestModeError,
        match=rf"training_config\.{container}\.{field}.*expected={expected}.*actual={actual}",
    ):
        validate_backtest_dataset(current, dataset(), BacktestMode.IN_SAMPLE_REPLAY)


def test_backtest_revalidates_fold_gap_against_training_config() -> None:
    config = training_config()
    config["walk_forward"]["gap"] = 20  # type: ignore[index]
    identity = ArtifactIdentity(
        core_semantics_version=CORE_SEMANTICS_VERSION,
        vocab_version=VOCAB_VERSION,
        label_semantics_version=LABEL_SEMANTICS_VERSION,
        execution_semantics_version=EXECUTION_SEMANTICS_VERSION,
        symbol="EURUSD",
        timeframe="H1",
        training_dataset=dataset(),
        training_config=config,
        training_config_hash=sha256_json(config),
    )
    current = StrategyArtifact.create(
        run_identity=TrainingRunIdentity.create(identity),
        formula_tokens=[0],
        decoded_formula=FORMULA_VOCAB.token_names[0],
        best_score=1.0,
        fold_evidence=[
            replace(fold, effective_gap=20)
            for fold in strategy().fold_evidence
        ],
        generated_at="2026-07-15T00:00:00Z",
    )
    object.__setattr__(current.fold_evidence[0], "effective_gap", LABEL_LOOKAHEAD_BARS)
    object.__setattr__(current, "fingerprint", sha256_json(current._payload_without_fingerprint()))
    with pytest.raises(
        BacktestModeError,
        match=r"fold_evidence\[0\]\.effective_gap.*expected=20.*actual=2",
    ):
        validate_backtest_dataset(current, dataset(), BacktestMode.IN_SAMPLE_REPLAY)


def test_mode_must_be_explicit_enum() -> None:
    with pytest.raises(BacktestModeError, match="mode"):
        validate_backtest_dataset(strategy(), dataset(), "in_sample_replay")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "malformed",
    [
        bypass_dataset_identity(bars=True),
        bypass_dataset_identity(bars=-1),
        bypass_dataset_identity(start_time_ns=20_000, end_time_ns=10_000),
        bypass_dataset_identity(data_fingerprint="short"),
        bypass_dataset_identity(time_fingerprint="G" * 64),
    ],
)
def test_backtest_rejects_directly_constructed_malformed_test_identity(
    malformed: DatasetIdentity,
) -> None:
    message = assert_mode_error(malformed, BacktestMode.OUT_OF_SAMPLE_BACKTEST)
    assert "invalid test dataset identity" in message


def test_backtest_rejects_directly_constructed_malformed_training_identity() -> None:
    artifact = strategy()
    malformed = bypass_dataset_identity(
        artifact.run_identity.artifact_identity.training_dataset,
        data_fingerprint="short",
    )
    object.__setattr__(
        artifact.run_identity.artifact_identity,
        "training_dataset",
        malformed,
    )
    with pytest.raises(BacktestModeError) as exc_info:
        validate_backtest_dataset(
            artifact,
            dataset(start=11_000, end=20_000, data_fp="c" * 64, time_fp="d" * 64),
            BacktestMode.OUT_OF_SAMPLE_BACKTEST,
        )
    message = str(exc_info.value)
    assert "invalid training dataset identity" in message
    assert "mode=out_of_sample_backtest" in message
    assert "train_range=[1000,10000]" in message
    assert "train_data_fp=short" in message


def test_backtest_hostile_fingerprint_diagnostic_never_leaks_render_exception() -> None:
    class HostileText:
        def __str__(self) -> str:
            raise RuntimeError("hostile __str__")

        def __repr__(self) -> str:
            raise RuntimeError("hostile __repr__")

    malformed = dataset()
    object.__setattr__(malformed, "data_fingerprint", HostileText())

    with pytest.raises(BacktestModeError, match="data_fingerprint") as exc_info:
        validate_backtest_dataset(
            strategy(),
            malformed,
            BacktestMode.IN_SAMPLE_REPLAY,
        )
    assert "hostile __str__" not in str(exc_info.value)
    assert "hostile __repr__" not in str(exc_info.value)


@pytest.mark.parametrize(
    "field",
    [
        "schema_version",
        "symbol",
        "timeframe",
        "start_time_ns",
        "end_time_ns",
        "bars",
        "data_fingerprint",
        "time_fingerprint",
    ],
)
def test_hostile_test_dataset_attribute_access_is_a_mode_error(field: str) -> None:
    with pytest.raises(BacktestModeError) as exc_info:
        validate_backtest_dataset(
            strategy(),
            hostile_dataset(field),
            BacktestMode.IN_SAMPLE_REPLAY,
        )
    message = str(exc_info.value)
    assert "mode=in_sample_replay" in message
    assert f"field={field}" in message
    assert "test_range=" in message


@pytest.mark.parametrize("label", ["test", "training"])
@pytest.mark.parametrize(
    "exception_type",
    [ValueError, TypeError, KeyError, BacktestHostileGetterError],
    ids=["value", "type", "key", "custom-hostile"],
)
def test_dataset_getter_ordinary_exceptions_are_bounded_mode_errors(
    label: str,
    exception_type: type[Exception],
) -> None:
    for key in BACKTEST_HOSTILE_GETTER_EXCEPTION_CALLS:
        BACKTEST_HOSTILE_GETTER_EXCEPTION_CALLS[key] = 0
    failure = exception_type("getter failure")
    malformed = hostile_dataset("bars", failure=failure)
    artifact = strategy()
    test_identity = dataset()
    if label == "test":
        test_identity = malformed
    else:
        object.__setattr__(
            artifact.run_identity.artifact_identity,
            "training_dataset",
            malformed,
        )

    with pytest.raises(BacktestModeError) as exc_info:
        validate_backtest_dataset(
            artifact,
            test_identity,
            BacktestMode.IN_SAMPLE_REPLAY,
        )

    message = str(exc_info.value)
    assert f"invalid {label} dataset identity" in message
    assert f"{label}_dataset attribute access failed: field=bars" in message
    assert len(message) <= 1_024
    assert exc_info.value.__cause__ is failure
    assert BACKTEST_HOSTILE_GETTER_EXCEPTION_CALLS == {
        "name": 0,
        "repr": 0,
        "str": 0,
    }


@pytest.mark.parametrize("label", ["test", "training"])
@pytest.mark.parametrize(
    "exception_type",
    [
        KeyboardInterrupt,
        SystemExit,
        GeneratorExit,
        BacktestCustomGetterBaseException,
    ],
    ids=["keyboard-interrupt", "system-exit", "generator-exit", "custom-base"],
)
def test_dataset_getter_base_exceptions_propagate_by_exact_identity(
    label: str,
    exception_type: type[BaseException],
) -> None:
    failure = exception_type("getter base failure")
    malformed = hostile_dataset("bars", failure=failure)
    artifact = strategy()
    test_identity = dataset()
    if label == "test":
        test_identity = malformed
    else:
        object.__setattr__(
            artifact.run_identity.artifact_identity,
            "training_dataset",
            malformed,
        )

    with pytest.raises(exception_type) as exc_info:
        validate_backtest_dataset(
            artifact,
            test_identity,
            BacktestMode.IN_SAMPLE_REPLAY,
        )

    assert exc_info.value is failure


@pytest.mark.parametrize("render_mode", ["oversized", "throwing"])
def test_backtest_diagnostics_do_not_render_unknown_objects_and_are_bounded(
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

    malformed = dataset()
    object.__setattr__(malformed, "data_fingerprint", HostileDiagnostic())
    with pytest.raises(BacktestModeError) as exc_info:
        validate_backtest_dataset(
            strategy(),
            malformed,
            BacktestMode.IN_SAMPLE_REPLAY,
        )
    message = str(exc_info.value)
    assert "data_fingerprint" in message
    assert len(message) <= 1_024
    assert calls == {"repr": 0, "str": 0}


@pytest.mark.parametrize("error_type", [AssertionError, RuntimeError])
def test_backtest_propagates_unexpected_timeframe_helper_exceptions(
    monkeypatch: pytest.MonkeyPatch,
    error_type: type[Exception],
) -> None:
    failure = error_type("unexpected backtest helper failure")
    current = strategy()
    test_identity = dataset()

    def fail_normalization(value: object) -> str:
        raise failure

    monkeypatch.setattr(artifacts_module, "normalize_timeframe_name", fail_normalization)
    with pytest.raises(error_type) as exc_info:
        validate_backtest_dataset(
            current,
            test_identity,
            BacktestMode.IN_SAMPLE_REPLAY,
        )
    assert exc_info.value is failure


@pytest.mark.parametrize("error_type", [AssertionError, RuntimeError])
def test_backtest_propagates_unexpected_fold_parser_exceptions_unchanged(
    monkeypatch: pytest.MonkeyPatch,
    error_type: type[Exception],
) -> None:
    failure = error_type("internal-fold-bug")
    base = strategy()
    current = base

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
        validate_backtest_dataset(
            current,
            dataset(),
            BacktestMode.IN_SAMPLE_REPLAY,
        )

    assert type(exc_info.value) is error_type
    assert exc_info.value is failure


def test_backtest_fold_exception_boundary_still_rejects_malformed_data() -> None:
    current = strategy()
    fold = FoldEvidence(
        fold_index=0,
        train_start_time_ns=1_000,
        train_end_time_ns=4_000,
        val_start_time_ns=6_000,
        val_end_time_ns=9_000,
        effective_gap=LABEL_LOOKAHEAD_BARS,
        validation_metrics={"ic": 0.1},
    )
    object.__setattr__(fold, "validation_metrics", {"ic": "bad"})
    object.__setattr__(current, "fold_evidence", (fold,))

    with pytest.raises(BacktestModeError, match="validation_metrics"):
        validate_backtest_dataset(
            current,
            dataset(),
            BacktestMode.IN_SAMPLE_REPLAY,
        )


def test_backtest_translates_documented_timeframe_validation_with_exact_cause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failure = DataValidationError("documented backtest timeframe failure")
    current = strategy()
    test_identity = dataset()

    def fail_normalization(value: object) -> str:
        raise failure

    monkeypatch.setattr(artifacts_module, "normalize_timeframe_name", fail_normalization)
    with pytest.raises(BacktestModeError, match="timeframe") as exc_info:
        validate_backtest_dataset(
            current,
            test_identity,
            BacktestMode.IN_SAMPLE_REPLAY,
        )
    assert exc_info.value.__cause__ is failure


def test_hostile_training_dataset_attribute_access_is_a_mode_error() -> None:
    artifact = strategy()
    object.__setattr__(
        artifact.run_identity.artifact_identity,
        "training_dataset",
        hostile_dataset("bars"),
    )

    with pytest.raises(BacktestModeError) as exc_info:
        validate_backtest_dataset(
            artifact,
            dataset(),
            BacktestMode.IN_SAMPLE_REPLAY,
        )
    message = str(exc_info.value)
    assert "mode=in_sample_replay" in message
    assert "field=bars" in message
    assert "train_range=" in message


def test_backtest_revalidation_rejects_list_subclass_without_protocol_access() -> None:
    artifact = strategy()
    failure = RuntimeError("backtest-list-" + "B" * 5_000)
    hostile = BacktestHostileList([0], failure)
    object.__setattr__(artifact, "formula_tokens", hostile)

    with pytest.raises(BacktestModeError, match="invalid strategy artifact") as exc_info:
        validate_backtest_dataset(
            artifact,
            dataset(),
            BacktestMode.IN_SAMPLE_REPLAY,
        )

    assert len(str(exc_info.value)) <= 1_024
    assert hostile.protocol_calls == {
        "iter": 0,
        "len": 0,
        "getitem": 0,
        "repr": 0,
        "str": 0,
    }


def test_backtest_revalidation_rejects_fold_int_subclass_without_protocol_access() -> None:
    class HostileFoldInteger(int):
        def __repr__(self) -> str:
            raise AssertionError("fold integer repr must not run")

        def __lt__(self, other: object) -> bool:
            raise AssertionError("fold integer comparison must not run")

    artifact = strategy()
    fold = artifact.fold_evidence[0]
    object.__setattr__(fold, "effective_gap", HostileFoldInteger(LABEL_LOOKAHEAD_BARS))

    with pytest.raises(
        BacktestModeError,
        match="effective_gap.*exact built-in integer",
    ):
        validate_backtest_dataset(
            artifact,
            dataset(),
            BacktestMode.IN_SAMPLE_REPLAY,
        )


def test_backtest_revalidation_rejects_metric_str_subclass_key_without_protocol_access() -> None:
    class HostileMetricKey(str):
        def __repr__(self) -> str:
            raise AssertionError("metric key repr must not run")

        def __str__(self) -> str:
            raise AssertionError("metric key str must not run")

    artifact = strategy()
    fold = artifact.fold_evidence[0]
    object.__setattr__(fold, "validation_metrics", {HostileMetricKey("ic"): 0.1})

    with pytest.raises(
        BacktestModeError,
        match="key.*exact built-in string",
    ):
        validate_backtest_dataset(
            artifact,
            dataset(),
            BacktestMode.IN_SAMPLE_REPLAY,
        )


@pytest.mark.parametrize("blocks", [2, 4, 6])
def test_backtest_revalidation_rejects_walk_forward_fold_count_mismatch(blocks: int) -> None:
    artifact = strategy()
    identity = artifact.run_identity.artifact_identity
    config = training_config()
    config["walk_forward"]["blocks"] = blocks  # type: ignore[index]
    object.__setattr__(identity, "training_config", config)
    object.__setattr__(identity, "training_config_hash", sha256_json(config))
    with pytest.raises(
        BacktestModeError,
        match=rf"fold_evidence count.*expected={blocks - 1}.*actual=4",
    ):
        validate_backtest_dataset(artifact, dataset(), BacktestMode.IN_SAMPLE_REPLAY)


@pytest.mark.parametrize("nested_dataset", [False, True])
def test_backtest_revalidation_rejects_scalar_string_subclasses(
    nested_dataset: bool,
) -> None:
    class StringSubclass(str):
        pass

    artifact = strategy()
    identity = artifact.run_identity.artifact_identity
    if nested_dataset:
        object.__setattr__(
            identity,
            "training_dataset",
            bypass_dataset_identity(
                identity.training_dataset,
                symbol=StringSubclass("EURUSD"),
            ),
        )
    else:
        object.__setattr__(identity, "symbol", StringSubclass("EURUSD"))
    with pytest.raises(BacktestModeError, match="exact built-in string"):
        validate_backtest_dataset(artifact, dataset(), BacktestMode.IN_SAMPLE_REPLAY)


@pytest.mark.parametrize(
    "integer_type",
    [BacktestIntegerSubclass, BacktestProtocolTrackingInteger],
)
def test_backtest_revalidation_requires_exact_built_in_formula_token(
    integer_type: type[int],
) -> None:
    token = integer_type(0)
    artifact = strategy()
    object.__setattr__(artifact, "formula_tokens", (token,))

    with pytest.raises(BacktestModeError, match="exact built-in integer") as exc_info:
        validate_backtest_dataset(
            artifact,
            dataset(),
            BacktestMode.IN_SAMPLE_REPLAY,
        )

    assert len(str(exc_info.value)) <= 1_024
    if isinstance(token, BacktestProtocolTrackingInteger):
        assert token.protocol_calls == {"lt": 0, "ge": 0, "repr": 0}


def test_backtest_revalidation_accepts_exact_built_in_formula_token_zero() -> None:
    validate_backtest_dataset(
        strategy(),
        dataset(),
        BacktestMode.IN_SAMPLE_REPLAY,
    )


def test_backtest_formula_token_rejection_never_reads_hostile_metaclass_name() -> None:
    BACKTEST_HOSTILE_META_CALLS["name"] = 0
    artifact = strategy()
    object.__setattr__(artifact, "formula_tokens", (BacktestHostileMetaInteger(0),))

    with pytest.raises(
        BacktestModeError,
        match="actual_category=non-exact-integer",
    ) as exc_info:
        validate_backtest_dataset(
            artifact,
            dataset(),
            BacktestMode.IN_SAMPLE_REPLAY,
        )

    assert len(str(exc_info.value)) <= 1_024
    assert BACKTEST_HOSTILE_META_CALLS == {"name": 0}


@pytest.mark.parametrize(
    "location",
    ["fold.fold_index", "training_dataset.start_time_ns", "training_config.batch_size"],
)
def test_backtest_maps_non_token_hostile_integer_state_without_protocols(
    location: str,
) -> None:
    reset_backtest_hostile_numeric_calls()
    artifact = strategy()
    hostile = BacktestHostileBoundaryInteger(0 if location == "fold.fold_index" else 1_000)
    if location == "fold.fold_index":
        fold = artifact.fold_evidence[0]
        object.__setattr__(fold, "fold_index", hostile)
    else:
        identity = artifact.run_identity.artifact_identity
        if location.startswith("training_dataset"):
            object.__setattr__(
                identity,
                "training_dataset",
                bypass_dataset_identity(
                    identity.training_dataset,
                    start_time_ns=hostile,
                ),
            )
        else:
            config = dict(identity.training_config)
            config["batch_size"] = BacktestHostileBoundaryInteger(192)
            object.__setattr__(identity, "training_config", config)

    with pytest.raises(BacktestModeError) as exc_info:
        validate_backtest_dataset(artifact, dataset(), BacktestMode.IN_SAMPLE_REPLAY)

    message = str(exc_info.value)
    assert "invalid strategy artifact" in message
    assert len(message) <= 1_024
    assert BACKTEST_HOSTILE_NUMERIC_CALLS == {key: 0 for key in BACKTEST_HOSTILE_NUMERIC_CALLS}


def test_backtest_maps_hostile_best_score_without_float_coercion() -> None:
    reset_backtest_hostile_numeric_calls()
    artifact = strategy()
    object.__setattr__(artifact, "best_score", BacktestHostileBoundaryFloat(1.0))

    with pytest.raises(BacktestModeError, match="best_score") as exc_info:
        validate_backtest_dataset(artifact, dataset(), BacktestMode.IN_SAMPLE_REPLAY)

    assert len(str(exc_info.value)) <= 1_024
    assert BACKTEST_HOSTILE_NUMERIC_CALLS == {key: 0 for key in BACKTEST_HOSTILE_NUMERIC_CALLS}


def test_backtest_maps_hostile_string_class_metadata_without_protocols() -> None:
    reset_backtest_hostile_numeric_calls()
    artifact = strategy()
    object.__setattr__(
        artifact.run_identity.artifact_identity,
        "symbol",
        BacktestHostileBoundaryString("EURUSD"),
    )

    with pytest.raises(BacktestModeError, match="exact built-in string") as exc_info:
        validate_backtest_dataset(artifact, dataset(), BacktestMode.IN_SAMPLE_REPLAY)

    assert len(str(exc_info.value)) <= 1_024
    assert BACKTEST_HOSTILE_NUMERIC_CALLS == {key: 0 for key in BACKTEST_HOSTILE_NUMERIC_CALLS}


@pytest.mark.parametrize(
    "huge",
    [10**1000, -(10**1000), 10**5000, -(10**5000)],
    ids=["pos-1000", "neg-1000", "pos-5000", "neg-5000"],
)
@pytest.mark.parametrize("location", ["config", "metric", "best_score", "token", "dataset"])
def test_backtest_maps_huge_exact_integer_to_bounded_domain_error(
    huge: int,
    location: str,
) -> None:
    artifact = strategy()
    if location == "config":
        identity = artifact.run_identity.artifact_identity
        config = dict(identity.training_config)
        config["cost_rate"] = huge
        object.__setattr__(identity, "training_config", config)
    elif location == "metric":
        object.__setattr__(artifact.fold_evidence[0], "validation_metrics", {"ic": huge})
    elif location == "token":
        object.__setattr__(artifact, "formula_tokens", (huge,))
    elif location == "dataset":
        identity = artifact.run_identity.artifact_identity
        object.__setattr__(
            identity,
            "training_dataset",
            bypass_dataset_identity(identity.training_dataset, start_time_ns=huge),
        )
    else:
        object.__setattr__(artifact, "best_score", huge)

    with pytest.raises(BacktestModeError) as exc_info:
        validate_backtest_dataset(artifact, dataset(), BacktestMode.IN_SAMPLE_REPLAY)

    assert "invalid strategy artifact" in str(exc_info.value)
    assert len(str(exc_info.value)) <= 1_024
