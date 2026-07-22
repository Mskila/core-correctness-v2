from __future__ import annotations

import pathlib
import copy
import random

import numpy as np
import pytest
import torch

from data_pipeline.validation import DatasetIdentity
from model_core.semantics import DATA_CANONICALIZATION_VERSION, DATA_SCHEMA_VERSION
from model_core.artifacts import (
    ArtifactCompatibilityError,
    FoldEvidence,
    StrategyArtifact,
    TrainingRunIdentity,
)
from model_core.engine import AlphaEngine
from model_core.config import ModelConfig
import model_core.engine as engine_module
import training_service
from training_service import run_training_session


class FakeManager:
    symbols = ["EURUSD", "GBPUSD"]


def test_service_rejects_multi_symbol_before_training() -> None:
    with pytest.raises(ArtifactCompatibilityError, match="single.*symbol"):
        run_training_session(FakeManager(), source_path="fake", from_scratch=True,
                             random_seed=42)


def test_service_rejects_manager_dataset_symbol_mismatch_before_any_io(
    monkeypatch, tmp_path
) -> None:
    class MismatchedManager:
        symbols = ["GBPUSD"]
        data_identities = OneManager.data_identities

    old = tmp_path / "old-checkpoint.pt"
    old.write_bytes(b"old-checkpoint-bytes")
    monkeypatch.setattr(
        training_service,
        "_select_resume",
        lambda _identity: (_ for _ in ()).throw(
            AssertionError("resume lookup reached")
        ),
    )
    monkeypatch.setattr(
        training_service,
        "AlphaEngine",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("engine creation reached")
        ),
    )

    with pytest.raises(
        ArtifactCompatibilityError,
        match="manager_dataset_symbol.*expected=EURUSD.*actual=GBPUSD",
    ) as caught:
        run_training_session(
            MismatchedManager(), source_path="fake", from_scratch=False,
            random_seed=42,
        )
    assert len(str(caught.value)) <= 512
    assert old.read_bytes() == b"old-checkpoint-bytes"


@pytest.mark.parametrize("manager_symbol", ["EURUSD", " eurusd "])
def test_artifact_identity_accepts_only_normalized_equal_symbol_sources(
    manager_symbol
) -> None:
    class MatchingManager:
        symbols = [manager_symbol]
        data_identities = OneManager.data_identities

    identity = training_service._artifact_identity(MatchingManager(), 42)
    assert identity.symbol == "EURUSD"


@pytest.mark.parametrize("manager_symbol", [None, "", 7])
def test_artifact_identity_rejects_malformed_manager_symbol_boundedly(
    manager_symbol
) -> None:
    class MalformedManager:
        symbols = [manager_symbol]
        data_identities = OneManager.data_identities

    with pytest.raises(
        ArtifactCompatibilityError,
        match="manager_dataset_symbol.*expected=.*actual=",
    ) as caught:
        training_service._artifact_identity(MalformedManager(), 42)
    assert len(str(caught.value)) <= 512


class OneManager:
    symbols = ["EURUSD"]
    data_identities = (
        DatasetIdentity(
            schema_version=DATA_SCHEMA_VERSION,
            canonicalization_version=DATA_CANONICALIZATION_VERSION,
            time_unit="ns", gap_policy="segment", volume_type="tick",
            symbol="EURUSD", timeframe="H1",
            start_time_ns=1_000, end_time_ns=9_000_000, bars=9_000,
            data_fingerprint="a" * 64, time_fingerprint="b" * 64,
        ),
    )


class DummyEngine:
    def __init__(self, *, data_manager, target_symbol, run_identity, **kwargs):
        self.data_manager = data_manager
        self.target_symbol = target_symbol
        self.run_identity = run_identity
        self.best_formula = None
        self.best_metrics = None
        self.trained_from = None
        self.engine_options = kwargs

    def train(self, *, start_step):
        self.trained_from = start_step


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("USE_LORD_REGULARIZATION", False),
        ("LORD_DECAY_RATE", 0.5),
        ("LORD_NUM_ITERATIONS", 1),
    ],
)
def test_each_lord_control_changes_training_identity(
    monkeypatch, field: str, replacement: object
) -> None:
    baseline = training_service._artifact_identity(OneManager(), 42)
    monkeypatch.setattr(ModelConfig, field, replacement, raising=False)
    changed = training_service._artifact_identity(OneManager(), 42)

    assert set(baseline.training_config["lord"]) == {
        "use_lord_regularization",
        "lord_decay_rate",
        "lord_num_iterations",
    }
    assert baseline.training_config_hash != changed.training_config_hash
    assert baseline.fingerprint != changed.fingerprint


def test_service_passes_exact_identity_lord_controls_to_engine(
    monkeypatch,
) -> None:
    monkeypatch.setattr(training_service, "AlphaEngine", DummyEngine)
    engine = run_training_session(
        OneManager(), source_path="fake", from_scratch=True, random_seed=42
    )

    assert engine.engine_options == {
        "use_lord_regularization": ModelConfig.USE_LORD_REGULARIZATION,
        "lord_decay_rate": ModelConfig.LORD_DECAY_RATE,
        "lord_num_iterations": ModelConfig.LORD_NUM_ITERATIONS,
        "evaluation_workers": ModelConfig.EVALUATION_WORKERS,
    }
    assert dict(engine.run_identity.artifact_identity.training_config["lord"]) == {
        "use_lord_regularization": ModelConfig.USE_LORD_REGULARIZATION,
        "lord_decay_rate": ModelConfig.LORD_DECAY_RATE,
        "lord_num_iterations": ModelConfig.LORD_NUM_ITERATIONS,
    }


def test_service_forwards_explicit_evaluation_workers(monkeypatch) -> None:
    monkeypatch.setattr(training_service, "AlphaEngine", DummyEngine)
    engine = run_training_session(
        OneManager(), source_path="fake", from_scratch=True, random_seed=42,
        evaluation_workers=12,
    )
    assert engine.engine_options["evaluation_workers"] == 12


def test_from_scratch_never_scans_or_changes_old_artifacts(monkeypatch, tmp_path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    strategy_dir = tmp_path / "strategies"
    checkpoint_dir.mkdir(); strategy_dir.mkdir()
    old = [
        checkpoint_dir / "ckpt_v3_EURUSD_old.pt",
        strategy_dir / "best_EURUSD.json",
        tmp_path / "training_history_EURUSD.json",
    ]
    for index, path in enumerate(old):
        path.write_bytes(f"old-{index}".encode())
    before = [path.read_bytes() for path in old]
    monkeypatch.setattr(training_service, "CHECKPOINT_DIR", checkpoint_dir)
    monkeypatch.setattr(training_service, "STRATEGY_DIR", strategy_dir)
    monkeypatch.setattr(training_service, "AlphaEngine", DummyEngine)
    monkeypatch.setattr(
        training_service, "_select_resume",
        lambda identity: (_ for _ in ()).throw(AssertionError("from-scratch scanned")),
    )
    first = run_training_session(OneManager(), source_path="fake", from_scratch=True,
                                 random_seed=42)
    second = run_training_session(OneManager(), source_path="fake", from_scratch=True,
                                  random_seed=42)
    assert first.run_identity.run_id != second.run_identity.run_id
    assert [path.read_bytes() for path in old] == before


def test_resume_uses_internal_identity_not_filename(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(training_service, "CHECKPOINT_DIR", tmp_path)
    identity = training_service._artifact_identity(OneManager(), 42)
    run = TrainingRunIdentity(run_id="3" * 32, artifact_identity=identity)
    source = AlphaEngine(None, use_lord_regularization=ModelConfig.USE_LORD_REGULARIZATION,
                         target_symbol="EURUSD", run_identity=run)
    lying_name = tmp_path / "ckpt_v3_EURUSD_H1_wrong_filename_step_999.pt"
    source.save_checkpoint(7, str(lying_name))
    selected = training_service._select_resume(identity)
    assert selected is not None
    assert selected[0] == lying_name and selected[2] == 7


def test_incompatible_candidate_fails_explicitly(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(training_service, "CHECKPOINT_DIR", tmp_path)
    wanted = training_service._artifact_identity(OneManager(), 42)
    other = training_service._artifact_identity(OneManager(), 43)
    run = TrainingRunIdentity(run_id="4" * 32, artifact_identity=other)
    source = AlphaEngine(None, use_lord_regularization=ModelConfig.USE_LORD_REGULARIZATION,
                         target_symbol="EURUSD", run_identity=run)
    source.save_checkpoint(2, str(tmp_path / "ckpt_v3_EURUSD_candidate.pt"))
    with pytest.raises(ArtifactCompatibilityError, match="none has exact internal identity.*from-scratch"):
        training_service._select_resume(wanted)


def _folds():
    return [
        {
            "fold_index": index,
            "train_start_time_ns": 1_000,
            "train_end_time_ns": 2_000 + index * 1_000,
            "val_start_time_ns": 2_500 + index * 1_000,
            "val_end_time_ns": 3_000 + index * 1_000,
            "effective_gap": 20,
            "validation_metrics": {"score": 0.1 + index},
        }
        for index in range(4)
    ]


def test_strategy_publication_is_idempotent_and_immutable(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(training_service, "STRATEGY_DIR", tmp_path)
    identity = training_service._artifact_identity(OneManager(), 42)
    run = TrainingRunIdentity(run_id="5" * 32, artifact_identity=identity)
    engine = AlphaEngine(None, use_lord_regularization=ModelConfig.USE_LORD_REGULARIZATION,
                         target_symbol="EURUSD", run_identity=run)
    engine.best_formula = [0]
    engine.best_score = 1.0
    engine.best_metrics = {"fold_evidence": _folds()}
    path = training_service._save_strategy(engine)
    assert path is not None
    before = path.read_bytes()
    assert training_service._save_strategy(engine) == path
    assert path.read_bytes() == before
    engine.best_score = 2.0
    with pytest.raises(ArtifactCompatibilityError, match="immutable strategy content mismatch"):
        training_service._save_strategy(engine)
    assert path.read_bytes() == before


def test_corrupt_only_candidate_is_bounded_and_preserved(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(training_service, "CHECKPOINT_DIR", tmp_path)
    path = tmp_path / "ckpt_v3_EURUSD_corrupt.pt"
    path.write_bytes(b"not-a-checkpoint")
    before = path.read_bytes()
    identity = training_service._artifact_identity(OneManager(), 42)
    with pytest.raises(ArtifactCompatibilityError) as caught:
        training_service._select_resume(identity)
    assert "--from-scratch" in str(caught.value)
    assert len(str(caught.value)) <= 1024
    assert caught.value.__cause__ is not None
    assert path.read_bytes() == before


def test_corrupt_candidate_does_not_block_valid_internal_latest(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(training_service, "CHECKPOINT_DIR", tmp_path)
    corrupt = tmp_path / "ckpt_v3_EURUSD_aaa_step_999.pt"
    corrupt.write_bytes(b"corrupt")
    identity = training_service._artifact_identity(OneManager(), 42)
    run = TrainingRunIdentity(run_id="6" * 32, artifact_identity=identity)
    source = AlphaEngine(None, use_lord_regularization=ModelConfig.USE_LORD_REGULARIZATION,
                         target_symbol="EURUSD", run_identity=run)
    valid = tmp_path / "ckpt_v3_EURUSD_zzz_step_0.pt"
    source.save_checkpoint(17, str(valid))
    selected = training_service._select_resume(identity)
    assert selected is not None
    assert selected[0] == valid
    assert selected[2] == 17
    assert corrupt.read_bytes() == b"corrupt"


@pytest.mark.parametrize("symbol", ["EUR/USD", "EUR USD"])
def test_resume_discovers_canonicalized_filename_by_internal_identity(
    monkeypatch, tmp_path, symbol
) -> None:
    class CanonicalManager:
        symbols = [symbol]
        data_identities = (
            DatasetIdentity(
                schema_version=DATA_SCHEMA_VERSION,
                canonicalization_version=DATA_CANONICALIZATION_VERSION,
                time_unit="ns", gap_policy="segment", volume_type="tick",
                symbol=symbol, timeframe="H1",
                start_time_ns=1_000, end_time_ns=9_000_000, bars=9_000,
                data_fingerprint="c" * 64, time_fingerprint="d" * 64,
            ),
        )

    monkeypatch.setattr(training_service, "CHECKPOINT_DIR", tmp_path)
    identity = training_service._artifact_identity(CanonicalManager(), 42)
    run = TrainingRunIdentity(run_id="8" * 32, artifact_identity=identity)
    source = AlphaEngine(None, use_lord_regularization=ModelConfig.USE_LORD_REGULARIZATION,
                         target_symbol=symbol, run_identity=run)
    path = tmp_path / run.checkpoint_filename(23)
    source.save_checkpoint(23, str(path))
    selected = training_service._select_resume(identity)
    assert selected is not None
    assert selected[0] == path
    assert selected[1] == run
    assert selected[2] == 23


def _identity_for_symbol(symbol, fingerprint="e"):
    class Manager:
        symbols = [symbol]
        data_identities = (
            DatasetIdentity(
                schema_version=DATA_SCHEMA_VERSION,
                canonicalization_version=DATA_CANONICALIZATION_VERSION,
                time_unit="ns", gap_policy="segment", volume_type="tick",
                symbol=symbol, timeframe="H1",
                start_time_ns=1_000, end_time_ns=9_000_000, bars=9_000,
                data_fingerprint=fingerprint * 64,
                time_fingerprint="f" * 64,
            ),
        )
    return training_service._artifact_identity(Manager(), 42)


def _write_checkpoint(path, identity, step, run_id):
    run = TrainingRunIdentity(run_id=run_id, artifact_identity=identity)
    engine = AlphaEngine(
        None, use_lord_regularization=ModelConfig.USE_LORD_REGULARIZATION,
        target_symbol=identity.symbol, run_identity=run,
    )
    engine.save_checkpoint(step, str(path))
    return run


@pytest.mark.parametrize("kind", ["valid", "corrupt", "legacy"])
def test_unrelated_symbol_candidate_is_never_loaded_or_blocks_new_run(
    monkeypatch, tmp_path, kind
) -> None:
    monkeypatch.setattr(training_service, "CHECKPOINT_DIR", tmp_path)
    wanted = _identity_for_symbol("EURUSD", "a")
    unrelated_identity = _identity_for_symbol("GBPUSD", "b")
    unrelated = tmp_path / "ckpt_v3_GBPUSD_H1_foreign_step_999.pt"
    if kind == "valid":
        _write_checkpoint(unrelated, unrelated_identity, 999, "a" * 32)
    elif kind == "legacy":
        torch.save({"legacy": True}, unrelated)
    else:
        unrelated.write_bytes(b"corrupt unrelated checkpoint")
    before = (
        unrelated.read_bytes(), unrelated.stat().st_ino,
        unrelated.stat().st_mtime_ns, unrelated.stat().st_size,
    )
    real_load = training_service.torch.load
    loaded = []

    def tracking_load(path, *args, **kwargs):
        loaded.append(pathlib.Path(path))
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(training_service.torch, "load", tracking_load)
    assert training_service._select_resume(wanted) is None
    after = (
        unrelated.read_bytes(), unrelated.stat().st_ino,
        unrelated.stat().st_mtime_ns, unrelated.stat().st_size,
    )
    assert loaded == []
    assert after == before


def test_unrelated_newer_candidate_is_not_loaded_before_same_symbol_exact(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(training_service, "CHECKPOINT_DIR", tmp_path)
    wanted = _identity_for_symbol("EURUSD", "a")
    unrelated_identity = _identity_for_symbol("GBPUSD", "b")
    unrelated = tmp_path / "ckpt_v3_GBPUSD_newer_step_999.pt"
    exact = tmp_path / "ckpt_v3_EURUSD_older_step_1.pt"
    _write_checkpoint(unrelated, unrelated_identity, 999, "b" * 32)
    exact_run = _write_checkpoint(exact, wanted, 7, "c" * 32)
    unrelated_before = unrelated.read_bytes()
    real_load = training_service.torch.load
    loaded = []

    def tracking_load(path, *args, **kwargs):
        loaded.append(pathlib.Path(path))
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(training_service.torch, "load", tracking_load)
    selected = training_service._select_resume(wanted)
    assert selected == (exact, exact_run, 7)
    assert loaded == [exact]
    assert unrelated.read_bytes() == unrelated_before


@pytest.mark.parametrize("unrelated_symbol", ["EURUSDX", "eurusd", "EUR_USD"])
def test_resume_symbol_namespace_has_exact_component_boundary(
    monkeypatch, tmp_path, unrelated_symbol
) -> None:
    monkeypatch.setattr(training_service, "CHECKPOINT_DIR", tmp_path)
    wanted = _identity_for_symbol("EURUSD", "a")
    unrelated_identity = _identity_for_symbol(unrelated_symbol, "b")
    unrelated = tmp_path / f"ckpt_v3_{unrelated_symbol}_foreign_step_8.pt"
    _write_checkpoint(unrelated, unrelated_identity, 8, "d" * 32)
    real_load = training_service.torch.load
    loaded = []

    def tracking_load(path, *args, **kwargs):
        loaded.append(pathlib.Path(path))
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(training_service.torch, "load", tracking_load)
    assert training_service._select_resume(wanted) is None
    assert loaded == []


def test_strategy_publish_race_preserves_foreign_competitor(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(training_service, "STRATEGY_DIR", tmp_path)
    identity = training_service._artifact_identity(OneManager(), 42)
    run = TrainingRunIdentity(run_id="9" * 32, artifact_identity=identity)
    engine = AlphaEngine(None, use_lord_regularization=ModelConfig.USE_LORD_REGULARIZATION,
                         target_symbol="EURUSD", run_identity=run)
    engine.best_formula = [0]
    engine.best_score = 1.0
    engine.best_metrics = {"fold_evidence": _folds()}
    target = tmp_path / run.strategy_filename()
    foreign = b"foreign-strategy-race"
    real_link = training_service.os.link

    def competitor_then_link(temporary, destination):
        pathlib.Path(destination).write_bytes(foreign)
        return real_link(temporary, destination)

    monkeypatch.setattr(training_service.os, "link", competitor_then_link)
    with pytest.raises(ArtifactCompatibilityError, match="immutable.*conflict"):
        training_service._save_strategy(engine)
    assert target.read_bytes() == foreign
    assert not list(tmp_path.glob(f".{target.name}*.tmp"))


def test_strategy_publish_race_accepts_identical_competitor(
    monkeypatch, tmp_path
) -> None:
    seed_dir = tmp_path / "seed"
    seed_dir.mkdir()
    identity = training_service._artifact_identity(OneManager(), 42)
    run = TrainingRunIdentity(run_id="b" * 32, artifact_identity=identity)
    engine = AlphaEngine(None, use_lord_regularization=ModelConfig.USE_LORD_REGULARIZATION,
                         target_symbol="EURUSD", run_identity=run)
    engine.best_formula = [0]
    engine.best_score = 1.0
    engine.best_metrics = {"fold_evidence": _folds()}
    monkeypatch.setattr(training_service, "STRATEGY_DIR", seed_dir)
    identical = training_service._save_strategy(engine).read_bytes()
    monkeypatch.setattr(training_service, "STRATEGY_DIR", tmp_path)
    target = tmp_path / run.strategy_filename()
    real_link = training_service.os.link

    def competitor_then_link(temporary, destination):
        pathlib.Path(destination).write_bytes(identical)
        return real_link(temporary, destination)

    monkeypatch.setattr(training_service.os, "link", competitor_then_link)
    assert training_service._save_strategy(engine) == target
    assert target.read_bytes() == identical
    assert not list(tmp_path.glob(f".{target.name}*.tmp"))


def test_strategy_publication_translates_ordinary_write_failure(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(training_service, "STRATEGY_DIR", tmp_path)
    identity = training_service._artifact_identity(OneManager(), 42)
    run = TrainingRunIdentity(run_id="c" * 32, artifact_identity=identity)
    engine = AlphaEngine(None, use_lord_regularization=ModelConfig.USE_LORD_REGULARIZATION,
                         target_symbol="EURUSD", run_identity=run)
    engine.best_formula = [0]
    engine.best_score = 1.0
    engine.best_metrics = {"fold_evidence": _folds()}
    failure = OSError("write interrupted")
    monkeypatch.setattr(training_service.json, "dump", lambda *_a, **_k: (_ for _ in ()).throw(failure))
    with pytest.raises(ArtifactCompatibilityError) as caught:
        training_service._save_strategy(engine)
    assert caught.value.__cause__ is failure
    assert not (tmp_path / run.strategy_filename()).exists()
    assert not list(tmp_path.glob(f".{run.strategy_filename()}*.tmp"))


def test_strategy_publication_preserves_base_exception_identity(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(training_service, "STRATEGY_DIR", tmp_path)
    identity = training_service._artifact_identity(OneManager(), 42)
    run = TrainingRunIdentity(run_id="d" * 32, artifact_identity=identity)
    engine = AlphaEngine(None, use_lord_regularization=ModelConfig.USE_LORD_REGULARIZATION,
                         target_symbol="EURUSD", run_identity=run)
    engine.best_formula = [0]
    engine.best_score = 1.0
    engine.best_metrics = {"fold_evidence": _folds()}
    failure = KeyboardInterrupt("publication interrupted")
    monkeypatch.setattr(training_service.os, "link", lambda *_a: (_ for _ in ()).throw(failure))
    with pytest.raises(KeyboardInterrupt) as caught:
        training_service._save_strategy(engine)
    assert caught.value is failure
    assert not (tmp_path / run.strategy_filename()).exists()
    assert not list(tmp_path.glob(f".{run.strategy_filename()}*.tmp"))


class _HostileCleanupAbort(BaseException):
    def __str__(self):
        raise AssertionError("cleanup string protocol invoked")

    def __repr__(self):
        raise AssertionError("cleanup repr protocol invoked")


class _StrategyPublicationAbort(BaseException):
    pass


@pytest.mark.parametrize(
    "primary_factory",
    [
        lambda: KeyboardInterrupt("primary keyboard interrupt"),
        lambda: SystemExit(71),
        lambda: _StrategyPublicationAbort("primary custom abort"),
    ],
    ids=["keyboard-interrupt", "system-exit", "custom-base"],
)
def test_strategy_hostile_cleanup_never_replaces_primary_baseexception(
    monkeypatch, tmp_path, primary_factory
) -> None:
    monkeypatch.setattr(training_service, "STRATEGY_DIR", tmp_path)
    identity = training_service._artifact_identity(OneManager(), 42)
    run = TrainingRunIdentity(run_id="7" * 32, artifact_identity=identity)
    engine = AlphaEngine(None, use_lord_regularization=ModelConfig.USE_LORD_REGULARIZATION,
                         target_symbol="EURUSD", run_identity=run)
    engine.best_formula = [0]
    engine.best_score = 1.0
    engine.best_metrics = {"fold_evidence": _folds()}
    foreign = tmp_path / "legacy-strategy.json"
    foreign.write_bytes(b"FOREIGN-LEGACY-BYTES")
    primary = primary_factory()
    secondary = _HostileCleanupAbort("hostile cleanup")
    monkeypatch.setattr(
        training_service.os, "link", lambda *_args: (_ for _ in ()).throw(primary)
    )
    monkeypatch.setattr(
        pathlib.Path, "unlink", lambda *_args, **_kwargs: (_ for _ in ()).throw(secondary)
    )
    with pytest.raises(BaseException) as caught:
        training_service._save_strategy(engine)
    assert caught.value is primary
    assert caught.value.args == primary.args
    assert caught.value.__cause__ is primary.__cause__
    assert caught.value.__context__ is primary.__context__
    assert foreign.read_bytes() == b"FOREIGN-LEGACY-BYTES"


class _HostileIdentity:
    def __getattribute__(self, name):
        if name == "__class__":
            return object.__getattribute__(self, name)
        raise AssertionError("hostile identity protocol invoked")

    def __repr__(self):
        raise AssertionError("hostile identity repr invoked")


def _different_strategy_identity(run):
    return TrainingRunIdentity(
        run_id="e" * 32, artifact_identity=run.artifact_identity
    )


def _equal_distinct_strategy_identity(run):
    result = TrainingRunIdentity.from_dict(run.to_dict())
    assert result == run and result is not run
    return result


def _mutate_strategy_identity(engine, run, mutation) -> None:
    if mutation == "deleted":
        del engine.run_identity
    elif mutation == "none":
        engine.run_identity = None
    elif mutation == "different":
        engine.run_identity = _different_strategy_identity(run)
    elif mutation == "hostile":
        engine.run_identity = _HostileIdentity()
    else:
        engine.run_identity = _equal_distinct_strategy_identity(run)


@pytest.mark.parametrize("callback", ["fold-evidence", "artifact-create"])
@pytest.mark.parametrize(
    "mutation", ["deleted", "none", "different", "hostile", "equal-distinct"]
)
def test_strategy_validation_callbacks_reject_before_later_callbacks_or_io(
    monkeypatch, tmp_path, callback, mutation
) -> None:
    strategy_dir = tmp_path / "strategies"
    monkeypatch.setattr(training_service, "STRATEGY_DIR", strategy_dir)
    identity = training_service._artifact_identity(OneManager(), 42)
    run = TrainingRunIdentity(run_id="9" * 32, artifact_identity=identity)
    engine = AlphaEngine(
        None, use_lord_regularization=ModelConfig.USE_LORD_REGULARIZATION,
        target_symbol="EURUSD", run_identity=run,
    )
    engine.best_formula = [0]
    engine.best_score = 1.0
    engine.best_metrics = {"fold_evidence": _folds()}
    legacy = [
        tmp_path / "strategy.json",
        tmp_path / "best_EURUSD.json",
        tmp_path / "training_history_EURUSD.json",
    ]
    for index, path in enumerate(legacy):
        path.write_bytes(f"old-{index}".encode())
    before_files = {path: path.read_bytes() for path in legacy}
    before_model = copy.deepcopy(engine.model.state_dict())
    before_optimizer = copy.deepcopy(engine.opt.state_dict())
    before_rng = (
        random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    )
    calls = []
    mutated = False

    def mutate_once() -> None:
        nonlocal mutated
        if not mutated:
            mutated = True
            _mutate_strategy_identity(engine, run, mutation)

    real_fold_from_dict = FoldEvidence.from_dict
    real_create = StrategyArtifact.create
    if callback == "fold-evidence":
        def mutating_fold_from_dict(cls, payload):
            result = real_fold_from_dict(payload)
            mutate_once()
            return result

        monkeypatch.setattr(
            FoldEvidence, "from_dict", classmethod(mutating_fold_from_dict)
        )
        monkeypatch.setattr(
            engine,
            "_decode_formula",
            lambda *_args: (
                calls.append("decode"),
                "should-not-decode",
            )[1],
        )
    else:
        def mutating_create(cls, **kwargs):
            result = real_create(**kwargs)
            mutate_once()
            return result

        monkeypatch.setattr(
            StrategyArtifact, "create", classmethod(mutating_create)
        )

    real_filename = TrainingRunIdentity.strategy_filename
    real_exists = pathlib.Path.exists
    real_mkdir = pathlib.Path.mkdir
    real_mkstemp = training_service.tempfile.mkstemp
    real_dump = training_service.json.dump
    real_link = training_service.os.link

    def recording(name, operation):
        def wrapped(*args, **kwargs):
            calls.append(name)
            return operation(*args, **kwargs)
        return wrapped

    monkeypatch.setattr(
        TrainingRunIdentity,
        "strategy_filename",
        recording("filename", real_filename),
    )
    monkeypatch.setattr(pathlib.Path, "exists", recording("exists", real_exists))
    monkeypatch.setattr(pathlib.Path, "mkdir", recording("mkdir", real_mkdir))
    monkeypatch.setattr(
        training_service.tempfile, "mkstemp", recording("mkstemp", real_mkstemp)
    )
    monkeypatch.setattr(training_service.json, "dump", recording("dump", real_dump))
    monkeypatch.setattr(training_service.os, "link", recording("link", real_link))

    with pytest.raises(ArtifactCompatibilityError) as caught:
        training_service._save_strategy(engine)

    assert calls == []
    assert "run_identity" in str(caught.value) and len(str(caught.value)) <= 512
    assert engine.run_identity is run
    assert all(
        torch.equal(before_model[key], value)
        for key, value in engine.model.state_dict().items()
    )
    assert before_optimizer == engine.opt.state_dict()
    assert random.getstate() == before_rng[0]
    after_numpy = np.random.get_state()
    assert after_numpy[0] == before_rng[1][0]
    assert np.array_equal(after_numpy[1], before_rng[1][1])
    assert after_numpy[2:] == before_rng[1][2:]
    assert torch.equal(torch.get_rng_state(), before_rng[2])
    assert {path: path.read_bytes() for path in legacy} == before_files
    assert not strategy_dir.exists()


@pytest.mark.parametrize("boundary", ["mkdir", "mkstemp", "json-dump", "os-link"])
@pytest.mark.parametrize(
    "mutation", ["deleted", "none", "different", "hostile", "equal-distinct"]
)
def test_strategy_persistence_callback_rejects_immediately_and_cleans_owned_files(
    monkeypatch, tmp_path, boundary, mutation
) -> None:
    strategy_dir = tmp_path / "strategies"
    monkeypatch.setattr(training_service, "STRATEGY_DIR", strategy_dir)
    identity = training_service._artifact_identity(OneManager(), 42)
    run = TrainingRunIdentity(run_id="8" * 32, artifact_identity=identity)
    engine = AlphaEngine(
        None, use_lord_regularization=ModelConfig.USE_LORD_REGULARIZATION,
        target_symbol="EURUSD", run_identity=run,
    )
    engine.best_formula = [0]
    engine.best_score = 1.0
    engine.best_metrics = {"fold_evidence": _folds()}
    old = tmp_path / "best_EURUSD.json"
    old.write_bytes(b"OLD-V1-BYTES")
    before_model = copy.deepcopy(engine.model.state_dict())
    before_optimizer = copy.deepcopy(engine.opt.state_dict())
    before_rng = (
        random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    )
    calls = []

    def mutate() -> None:
        _mutate_strategy_identity(engine, run, mutation)

    if boundary == "mkdir":
        real = pathlib.Path.mkdir

        def callback(path, *args, **kwargs):
            result = real(path, *args, **kwargs)
            calls.append("mkdir")
            mutate()
            return result

        monkeypatch.setattr(pathlib.Path, "mkdir", callback)
    elif boundary == "mkstemp":
        real = training_service.tempfile.mkstemp

        def callback(*args, **kwargs):
            result = real(*args, **kwargs)
            calls.append("mkstemp")
            mutate()
            return result

        monkeypatch.setattr(training_service.tempfile, "mkstemp", callback)
    elif boundary == "json-dump":
        real = training_service.json.dump

        def callback(*args, **kwargs):
            result = real(*args, **kwargs)
            calls.append("json-dump")
            mutate()
            return result

        monkeypatch.setattr(training_service.json, "dump", callback)
    else:
        real = training_service.os.link

        def callback(*args, **kwargs):
            result = real(*args, **kwargs)
            calls.append("os-link")
            mutate()
            return result

        monkeypatch.setattr(training_service.os, "link", callback)

    with pytest.raises(ArtifactCompatibilityError) as caught:
        training_service._save_strategy(engine)

    assert calls == [boundary]
    assert "run_identity" in str(caught.value) and len(str(caught.value)) <= 512
    assert engine.run_identity is run
    assert all(
        torch.equal(before_model[key], value)
        for key, value in engine.model.state_dict().items()
    )
    assert before_optimizer == engine.opt.state_dict()
    assert random.getstate() == before_rng[0]
    after_numpy = np.random.get_state()
    assert after_numpy[0] == before_rng[1][0]
    assert np.array_equal(after_numpy[1], before_rng[1][1])
    assert after_numpy[2:] == before_rng[1][2:]
    assert torch.equal(torch.get_rng_state(), before_rng[2])
    assert old.read_bytes() == b"OLD-V1-BYTES"
    assert not list(strategy_dir.glob("*")) if strategy_dir.exists() else True


@pytest.mark.parametrize(
    "mutation",
    [
        "deleted",
        "none",
        "different",
        "hostile",
        "equal_distinct",
    ],
)
def test_strategy_revalidates_exact_identity_after_formula_decode(
    monkeypatch, tmp_path, mutation
) -> None:
    monkeypatch.setattr(training_service, "STRATEGY_DIR", tmp_path / "strategies")
    identity = training_service._artifact_identity(OneManager(), 42)
    run = TrainingRunIdentity(run_id="f" * 32, artifact_identity=identity)
    engine = AlphaEngine(
        None, use_lord_regularization=ModelConfig.USE_LORD_REGULARIZATION,
        target_symbol="EURUSD", run_identity=run,
    )
    engine.best_formula = [0]
    engine.best_score = 1.0
    engine.best_metrics = {"fold_evidence": _folds()}
    decoded_formula = engine._decode_formula(engine.best_formula)
    legacy = [
        tmp_path / "strategy.json",
        tmp_path / "best_EURUSD.json",
        tmp_path / "training_history_EURUSD.json",
    ]
    for index, path in enumerate(legacy):
        path.write_bytes(f"legacy-{index}".encode())
    before_files = {path: path.read_bytes() for path in legacy}
    before_model = copy.deepcopy(engine.model.state_dict())
    before_optimizer = copy.deepcopy(engine.opt.state_dict())
    before_rng = (
        random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    )
    calls = []
    real_filename = TrainingRunIdentity.strategy_filename
    real_exists = pathlib.Path.exists
    real_mkdir = pathlib.Path.mkdir
    real_mkstemp = training_service.tempfile.mkstemp
    real_dump = training_service.json.dump
    real_link = training_service.os.link
    real_create = StrategyArtifact.create

    def mutate_identity(_formula):
        if mutation == "deleted":
            del engine.run_identity
        elif mutation == "none":
            engine.run_identity = None
        elif mutation == "different":
            engine.run_identity = _different_strategy_identity(run)
        elif mutation == "hostile":
            engine.run_identity = _HostileIdentity()
        else:
            engine.run_identity = _equal_distinct_strategy_identity(run)
        return decoded_formula

    def record_filename(value):
        calls.append("filename")
        return real_filename(value)

    def record_exists(path):
        calls.append("exists")
        return real_exists(path)

    def record_mkdir(path, *args, **kwargs):
        calls.append("mkdir")
        return real_mkdir(path, *args, **kwargs)

    def record_mkstemp(*args, **kwargs):
        calls.append("mkstemp")
        return real_mkstemp(*args, **kwargs)

    def record_dump(*args, **kwargs):
        calls.append("dump")
        return real_dump(*args, **kwargs)

    def record_link(*args, **kwargs):
        calls.append("link")
        return real_link(*args, **kwargs)

    def record_create(cls, **kwargs):
        calls.append("artifact-create")
        return real_create(**kwargs)

    engine._decode_formula = mutate_identity
    monkeypatch.setattr(StrategyArtifact, "create", classmethod(record_create))
    monkeypatch.setattr(TrainingRunIdentity, "strategy_filename", record_filename)
    monkeypatch.setattr(pathlib.Path, "exists", record_exists)
    monkeypatch.setattr(pathlib.Path, "mkdir", record_mkdir)
    monkeypatch.setattr(training_service.tempfile, "mkstemp", record_mkstemp)
    monkeypatch.setattr(training_service.json, "dump", record_dump)
    monkeypatch.setattr(training_service.os, "link", record_link)
    caught = None
    published = None
    try:
        published = training_service._save_strategy(engine)
    except ArtifactCompatibilityError as exc:
        caught = exc

    assert (type(caught), published, calls) == (
        ArtifactCompatibilityError, None, []
    )
    assert len(str(caught)) <= 512
    assert "run_identity" in str(caught)
    assert engine.run_identity is run
    assert all(
        torch.equal(before_model[key], value)
        for key, value in engine.model.state_dict().items()
    )
    assert before_optimizer == engine.opt.state_dict()
    assert random.getstate() == before_rng[0]
    after_numpy = np.random.get_state()
    assert after_numpy[0] == before_rng[1][0]
    assert np.array_equal(after_numpy[1], before_rng[1][1])
    assert after_numpy[2:] == before_rng[1][2:]
    assert torch.equal(torch.get_rng_state(), before_rng[2])
    assert {path: path.read_bytes() for path in legacy} == before_files
    assert not (tmp_path / "strategies").exists()


@pytest.mark.parametrize(
    "replacement_factory",
    [
        lambda run: None,
        lambda run: _HostileIdentity(),
        lambda run: TrainingRunIdentity.from_dict(run.to_dict()),
    ],
    ids=["none", "hostile", "equal-distinct"],
)
def test_train_revalidates_identity_immediately_after_data_callback(
    monkeypatch, tmp_path, replacement_factory
) -> None:
    identity = training_service._artifact_identity(OneManager(), 42)
    run = TrainingRunIdentity(run_id="a" * 32, artifact_identity=identity)
    current = AlphaEngine(None, use_lord_regularization=ModelConfig.USE_LORD_REGULARIZATION,
                          target_symbol="EURUSD", run_identity=run)

    class TamperingManager:
        target_ret = torch.zeros(1, 30)
        target_valid = torch.ones(1, 30, dtype=torch.bool)
        bar_time = torch.arange(30, dtype=torch.int64).unsqueeze(0)

        @property
        def feat_tensor(self):
            current.run_identity = replacement_factory(run)
            return torch.zeros(1, 3, 30)

    current.data_manager = TamperingManager()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(engine_module, "formula_warmup_bars", lambda _length: 0)
    monkeypatch.setattr(engine_module, "required_training_bars", lambda **_kwargs: 1)
    monkeypatch.setattr(engine_module, "assert_minimum_bars", lambda *_a, **_k: None)
    monkeypatch.setattr(engine_module, "build_walk_forward_folds", lambda **_k: [])
    before_model = copy.deepcopy(current.model.state_dict())
    before_rng = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
    with pytest.raises(ArtifactCompatibilityError, match="run_identity.*expected=.*actual="):
        current.train(end_step=0, verbose_header=False)
    assert all(torch.equal(before_model[key], value)
               for key, value in current.model.state_dict().items())
    assert random.getstate() == before_rng[0]
    after_numpy = np.random.get_state()
    assert after_numpy[0] == before_rng[1][0]
    assert np.array_equal(after_numpy[1], before_rng[1][1])
    assert after_numpy[2:] == before_rng[1][2:]
    assert torch.equal(torch.get_rng_state(), before_rng[2])
    assert not list(tmp_path.rglob("*.pt"))
    assert not list(tmp_path.rglob("*.json"))
