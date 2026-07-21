import json
import os

import pytest
import torch

import web.progress as progress
from model_core.artifacts import StrategyArtifact, TrainingRunIdentity
from model_core.vocab import FORMULA_VOCAB
from tests.unit.test_artifacts import (
    artifact_identity_with_config,
    current_strategy_artifact as strategy_artifact,
    fold_evidence_with_index,
    training_config,
)


METRICS = (
    "avg_reward", "best_score", "val_score", "entropy", "ic_mean",
    "ic_stability", "sortino", "elite_pool_size", "init_entropy",
    "kl_uniform", "kl_prev", "top1_prob", "eff_vocab",
    "batch_uniq_tokens", "batch_uniq_fmls", "batch_fml_div",
)


class IntSubclass(int):
    pass


def _artifact(
    *, lord_enabled=False, generated_at="2026-07-15T00:00:00Z", token=0,
    run_id=None,
):
    config = training_config()
    config["lord"]["use_lord_regularization"] = lord_enabled
    identity = artifact_identity_with_config(config)
    run = (
        TrainingRunIdentity.create(identity)
        if run_id is None
        else TrainingRunIdentity(run_id=run_id, artifact_identity=identity)
    )
    return StrategyArtifact.create(
        run_identity=run,
        formula_tokens=[token],
        decoded_formula=FORMULA_VOCAB.token_names[token],
        best_score=1.25,
        fold_evidence=[fold_evidence_with_index(index) for index in range(4)],
        generated_at=generated_at,
        candidate_evaluation_count=1,
    )


def _layout(monkeypatch, tmp_path, *, train_steps=20):
    checkpoints = tmp_path / "checkpoints"
    strategies = tmp_path / "strategies"
    checkpoints.mkdir()
    strategies.mkdir()
    monkeypatch.setattr(progress, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(progress, "CHECKPOINT_DIR", checkpoints)
    monkeypatch.setattr(progress, "STRATEGIES_DIR", strategies)
    monkeypatch.setattr(progress.ModelConfig, "TRAIN_STEPS", train_steps)
    progress.invalidate_checkpoint_cache()
    return checkpoints, strategies


def _history_value(artifact, steps, *, stable_rank=None, include_identity=True):
    values = {
        "step": list(steps),
        **{name: [float(index + 1) for index in range(len(steps))] for name in METRICS},
        "stable_rank": ([] if stable_rank is None else list(stable_rank)),
    }
    if include_identity:
        values["run_identity"] = artifact.run_identity.to_dict()
    return values


def _checkpoint(path, artifact, *, step=10, history=None, rank_monitor=None):
    embedded = _history_value(artifact, range(step), include_identity=False) if history is None else history
    torch.save({
        "checkpoint_schema_version": "checkpoint-v3",
        "run_identity": artifact.run_identity.to_dict(),
        "step": step,
        "best_score": artifact.best_score,
        "best_formula": list(artifact.formula_tokens),
        "training_history": embedded,
        "rank_monitor_history": (
            embedded.get("stable_rank") if rank_monitor is None and isinstance(embedded, dict)
            else rank_monitor
        ),
    }, path)


def _history(path, artifact, *, steps, stable_rank=None, mutation=None):
    value = _history_value(artifact, steps, stable_rank=stable_rank)
    if mutation is not None:
        mutation(value)
    path.write_text(json.dumps(value), encoding="utf-8")
    return value


def _strategy(directory, artifact):
    path = directory / artifact.run_identity.strategy_filename()
    path.write_text(json.dumps(artifact.to_dict()), encoding="utf-8")
    return path


def test_progress_strategy_anchor_uses_chronological_generated_at(
    monkeypatch, tmp_path
) -> None:
    _, strategies = _layout(monkeypatch, tmp_path)
    earlier = _artifact(generated_at="2026-07-16T00:00:00Z", token=0)
    later = _artifact(generated_at="2026-07-16T00:00:00.500000Z", token=1)
    _strategy(strategies, earlier)
    _strategy(strategies, later)
    result = progress.get_symbol_progress("EURUSD")
    loaded = progress._load_strategy("EURUSD")
    assert result.best_formula == [1]
    assert result.run_id == later.run_identity.run_id
    assert loaded is not None
    assert loaded["formula"] == [1]


def test_progress_strategy_anchor_preserves_submicrosecond_order(
    monkeypatch, tmp_path
) -> None:
    _, strategies = _layout(monkeypatch, tmp_path)
    earlier = _artifact(
        generated_at="2026-07-16T00:00:00.0000001Z", token=0, run_id="f" * 32
    )
    later = _artifact(
        generated_at="2026-07-16T00:00:00.0000002Z", token=1, run_id="0" * 32
    )
    _strategy(strategies, earlier)
    _strategy(strategies, later)
    result = progress.get_symbol_progress("EURUSD")
    assert result.best_formula == [1]
    assert result.run_id == later.run_identity.run_id


def test_checkpoint_cache_reloads_on_subfloat_mtime_ns_change(
    monkeypatch, tmp_path
) -> None:
    checkpoints, _ = _layout(monkeypatch, tmp_path)
    earlier = _artifact(token=0, run_id="1" * 32)
    later = StrategyArtifact.create(
        run_identity=earlier.run_identity,
        formula_tokens=[1],
        decoded_formula=FORMULA_VOCAB.token_names[1],
        best_score=earlier.best_score,
        fold_evidence=earlier.fold_evidence,
        generated_at=earlier.generated_at,
        candidate_evaluation_count=1,
    )
    path = checkpoints / earlier.run_identity.checkpoint_filename(7)
    _checkpoint(path, earlier, step=7)
    first_ns = 1_700_020_000_000_000_000
    os.utime(path, ns=(first_ns, first_ns))
    first = progress._load_checkpoint_meta(path)
    _checkpoint(path, later, step=7)
    second_ns = first_ns + 100
    os.utime(path, ns=(second_ns, second_ns))
    assert float(first_ns / 1_000_000_000) == float(second_ns / 1_000_000_000)
    second = progress._load_checkpoint_meta(path)
    assert first["best_formula"] == [0]
    assert second["best_formula"] == [1]


def test_checkpoint_cache_reloads_changed_bytes_with_identical_mtime_ns(
    monkeypatch, tmp_path
) -> None:
    checkpoints, _ = _layout(monkeypatch, tmp_path)
    first_artifact = _artifact(token=0, run_id="1" * 32)
    second_artifact = StrategyArtifact.create(
        run_identity=first_artifact.run_identity,
        formula_tokens=[1],
        decoded_formula=FORMULA_VOCAB.token_names[1],
        best_score=first_artifact.best_score,
        fold_evidence=first_artifact.fold_evidence,
        generated_at=first_artifact.generated_at,
        candidate_evaluation_count=1,
    )
    path = checkpoints / first_artifact.run_identity.checkpoint_filename(7)
    fixed_ns = 1_700_020_050_000_000_000
    _checkpoint(path, first_artifact, step=7)
    os.utime(path, ns=(fixed_ns, fixed_ns))
    original_size = path.stat().st_size
    first = progress._load_checkpoint_meta(path)

    _checkpoint(path, second_artifact, step=7)
    assert path.stat().st_size == original_size
    os.utime(path, ns=(fixed_ns, fixed_ns))
    assert path.stat().st_mtime_ns == fixed_ns
    second = progress._load_checkpoint_meta(path)

    assert first["best_formula"] == [0]
    assert second["best_formula"] == [1]


def test_public_progress_mutation_cannot_poison_cached_checkpoint_metadata(
    monkeypatch, tmp_path
) -> None:
    checkpoints, _ = _layout(monkeypatch, tmp_path)
    artifact = _artifact(token=0, run_id="2" * 32)
    path = checkpoints / artifact.run_identity.checkpoint_filename(7)
    _checkpoint(path, artifact, step=7)
    disk_bytes = path.read_bytes()

    first = progress.get_symbol_progress("EURUSD")
    assert first.history is not None
    assert first.best_formula == [0]
    first.history["best_score"][0] = 999.0
    first.best_formula[0] = 1

    second = progress.get_symbol_progress("EURUSD")
    assert path.read_bytes() == disk_bytes
    assert second.history is not None
    assert second.history["best_score"][0] == 1.0
    assert second.best_formula == [0]
    assert second.history is not first.history
    assert second.history["best_score"] is not first.history["best_score"]
    assert second.best_formula is not first.best_formula


def test_history_only_anchor_uses_exact_mtime_ns(monkeypatch, tmp_path) -> None:
    _layout(monkeypatch, tmp_path)
    older = _artifact(run_id="f" * 32)
    newer = _artifact(run_id="0" * 32)
    older_path = tmp_path / older.run_identity.history_filename()
    newer_path = tmp_path / newer.run_identity.history_filename()
    _history(older_path, older, steps=[0])
    _history(newer_path, newer, steps=[0])
    older_ns = 1_700_020_100_000_000_000
    newer_ns = older_ns + 100
    os.utime(older_path, ns=(older_ns, older_ns))
    os.utime(newer_path, ns=(newer_ns, newer_ns))
    assert older_path.stat().st_mtime == newer_path.stat().st_mtime
    result = progress.get_symbol_progress("EURUSD")
    assert result.run_id == newer.run_identity.run_id


def test_one_coherent_v2_run_completes(monkeypatch, tmp_path) -> None:
    checkpoints, strategies = _layout(monkeypatch, tmp_path, train_steps=10)
    artifact = strategy_artifact()
    embedded = _history_value(artifact, range(10), include_identity=False)
    _checkpoint(checkpoints / artifact.run_identity.checkpoint_filename(10), artifact,
                history=embedded)
    _history(tmp_path / artifact.run_identity.history_filename(), artifact, steps=range(10))
    _strategy(strategies, artifact)
    result = progress.get_symbol_progress("EURUSD")
    assert result.status == "completed"
    assert result.current_step == 10
    assert result.artifact_fingerprint == artifact.run_identity.artifact_identity.fingerprint


def test_coherent_v2_run_is_not_contaminated_by_preserved_legacy_history(
    monkeypatch, tmp_path
) -> None:
    checkpoints, strategies = _layout(monkeypatch, tmp_path, train_steps=10)
    artifact = strategy_artifact()
    _checkpoint(checkpoints / artifact.run_identity.checkpoint_filename(10), artifact)
    _history(tmp_path / artifact.run_identity.history_filename(), artifact, steps=range(10))
    _strategy(strategies, artifact)
    legacy = tmp_path / "training_history_EURUSD.json"
    legacy.write_bytes(b'LEGACY-USER-BYTES-MUST-SURVIVE')
    before = legacy.read_bytes()

    result = progress.get_symbol_progress("EURUSD")

    assert result.status == "completed"
    assert result.current_step == 10
    assert result.incompatible_reasons == ()
    assert legacy.read_bytes() == before


def test_v1_history_never_advances_v2_progress(monkeypatch, tmp_path) -> None:
    _, strategies = _layout(monkeypatch, tmp_path)
    artifact = strategy_artifact()
    old = tmp_path / "training_history_EURUSD.json"
    old.write_text(json.dumps({"step": [8999], "best_score": [999]}), encoding="utf-8")
    before = old.read_bytes()
    _strategy(strategies, artifact)
    result = progress.get_symbol_progress("EURUSD")
    assert result.current_step == 0
    assert result.status == "incompatible"
    assert old.read_bytes() == before


def test_mixed_v2_fingerprints_are_incompatible(monkeypatch, tmp_path) -> None:
    checkpoints, strategies = _layout(monkeypatch, tmp_path)
    checkpoint_artifact = strategy_artifact()
    unrelated_strategy = strategy_artifact()
    _checkpoint(checkpoints / checkpoint_artifact.run_identity.checkpoint_filename(10), checkpoint_artifact)
    _history(tmp_path / checkpoint_artifact.run_identity.history_filename(), checkpoint_artifact,
             steps=range(10))
    _strategy(strategies, unrelated_strategy)
    result = progress.get_symbol_progress("EURUSD")
    assert result.status == "incompatible"
    assert result.current_step == 0


@pytest.mark.parametrize("payload_step", [7.0, True, IntSubclass(7), -1, 21])
def test_checkpoint_step_requires_exact_bounded_integer(
    monkeypatch, tmp_path, payload_step
) -> None:
    checkpoints, strategies = _layout(monkeypatch, tmp_path)
    artifact = strategy_artifact()
    path = checkpoints / artifact.run_identity.checkpoint_filename(7)
    _checkpoint(path, artifact, step=payload_step, history=_history_value(
        artifact, range(7), include_identity=False
    ))
    before = path.read_bytes()
    _strategy(strategies, artifact)
    result = progress.get_symbol_progress("EURUSD")
    assert result.status == "incompatible"
    assert result.current_step == 0
    assert path.read_bytes() == before


def test_checkpoint_filename_and_internal_step_must_agree(monkeypatch, tmp_path) -> None:
    checkpoints, _ = _layout(monkeypatch, tmp_path)
    artifact = strategy_artifact()
    path = checkpoints / artifact.run_identity.checkpoint_filename(6)
    _checkpoint(path, artifact, step=7, history=_history_value(
        artifact, range(7), include_identity=False
    ))
    result = progress.get_symbol_progress("EURUSD")
    assert result.status == "incompatible"
    assert result.current_step == 0


@pytest.mark.parametrize(
    ("label", "checkpoint_step", "history_steps"),
    [
        ("stale", 6, [0, 1, 2, 3, 4]),
        ("trailing-truncated", 6, [2, 3, 4]),
        ("ahead", 6, [0, 1, 2, 3, 4, 5, 6]),
        ("empty", 6, []),
        ("off-by-one", 6, [1, 2, 3]),
        ("zero-with-history", 0, [0]),
    ],
)
def test_checkpoint_history_requires_exact_next_step_relationship(
    monkeypatch, tmp_path, label, checkpoint_step, history_steps
) -> None:
    checkpoints, strategies = _layout(monkeypatch, tmp_path)
    artifact = strategy_artifact()
    embedded = _history_value(artifact, history_steps, include_identity=False)
    _checkpoint(checkpoints / artifact.run_identity.checkpoint_filename(checkpoint_step), artifact,
                step=checkpoint_step, history=embedded)
    _history(tmp_path / artifact.run_identity.history_filename(), artifact, steps=history_steps)
    _strategy(strategies, artifact)
    result = progress.get_symbol_progress("EURUSD")
    assert result.status == "incompatible", label
    assert result.current_step == 0


@pytest.mark.parametrize(
    ("lord_enabled", "steps", "stable_rank", "valid"),
    [
        (True, [0, 1], [1.0], True),
        (False, [0, 1], [], True),
        (True, [7, 8, 9, 10, 11, 12], [2.0], True),
        (True, [11, 12], [], True),
        (True, [], [], True),
        (True, [0, 1], None, False),
        (True, [0, 1], [True], False),
        (True, [0, 1], ["1"], False),
        (True, [0, 1], [{}], False),
        (True, [0, 1], [float("nan")], False),
        (True, [0, 1], [float("inf")], False),
        (True, [0, 1], [-1.0], False),
        (True, [0, 1], [], False),
        (True, [0, 1], [1.0, 2.0], False),
        (False, [0, 1], [1.0], False),
    ],
)
def test_stable_rank_obeys_lord_cadence_contract(
    monkeypatch, tmp_path, lord_enabled, steps, stable_rank, valid
) -> None:
    _, strategies = _layout(monkeypatch, tmp_path)
    artifact = _artifact(lord_enabled=lord_enabled)
    value = _history_value(artifact, steps, stable_rank=[])
    if stable_rank is None:
        value.pop("stable_rank")
    else:
        value["stable_rank"] = stable_rank
    path = tmp_path / artifact.run_identity.history_filename()
    path.write_text(json.dumps(value), encoding="utf-8")
    _strategy(strategies, artifact)
    result = progress.get_symbol_progress("EURUSD")
    assert (result.status != "incompatible") is valid
    if not valid:
        assert result.current_step == 0


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.pop("avg_reward"),
        lambda value: value.__setitem__("avg_reward", "bad"),
        lambda value: value.__setitem__("avg_reward", [1.0]),
        lambda value: value.__setitem__("avg_reward", [True, 2.0]),
        lambda value: value.__setitem__("avg_reward", [float("nan"), 2.0]),
        lambda value: value.__setitem__("step", [1, 1]),
        lambda value: value.__setitem__("step", [1, False]),
        lambda value: value.__setitem__("step", [1, 20]),
    ],
)
def test_malformed_standalone_history_is_incompatible(monkeypatch, tmp_path, mutation) -> None:
    _, strategies = _layout(monkeypatch, tmp_path)
    artifact = strategy_artifact()
    history = tmp_path / artifact.run_identity.history_filename()
    _history(history, artifact, steps=[1, 2], mutation=mutation)
    before = history.read_bytes()
    _strategy(strategies, artifact)
    result = progress.get_symbol_progress("EURUSD")
    assert result.status == "incompatible"
    assert result.current_step == 0
    assert history.read_bytes() == before


def test_malformed_embedded_history_never_advances_checkpoint(monkeypatch, tmp_path) -> None:
    checkpoints, strategies = _layout(monkeypatch, tmp_path)
    artifact = strategy_artifact()
    embedded = _history_value(artifact, [0, 1], include_identity=False)
    embedded.pop("avg_reward")
    path = checkpoints / artifact.run_identity.checkpoint_filename(2)
    _checkpoint(path, artifact, step=2, history=embedded)
    before = path.read_bytes()
    _strategy(strategies, artifact)
    result = progress.get_symbol_progress("EURUSD")
    assert result.status == "incompatible"
    assert result.current_step == 0
    assert path.read_bytes() == before


def test_rank_monitor_history_must_equal_embedded_stable_rank(monkeypatch, tmp_path) -> None:
    checkpoints, strategies = _layout(monkeypatch, tmp_path)
    artifact = _artifact(lord_enabled=True)
    embedded = _history_value(artifact, [0, 1], stable_rank=[1.0], include_identity=False)
    path = checkpoints / artifact.run_identity.checkpoint_filename(2)
    _checkpoint(path, artifact, step=2, history=embedded, rank_monitor=[2.0])
    _strategy(strategies, artifact)
    result = progress.get_symbol_progress("EURUSD")
    assert result.status == "incompatible"
    assert result.current_step == 0


def test_old_identity_without_lord_is_incompatible(monkeypatch, tmp_path) -> None:
    checkpoints, _ = _layout(monkeypatch, tmp_path)
    artifact = strategy_artifact()
    run = artifact.run_identity.to_dict()
    config = run["artifact_identity"]["training_config"]
    config.pop("lord")
    import hashlib
    run["artifact_identity"]["training_config_hash"] = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    path = checkpoints / artifact.run_identity.checkpoint_filename(2)
    embedded = _history_value(artifact, [0, 1], include_identity=False)
    torch.save({
        "checkpoint_schema_version": "checkpoint-v3", "run_identity": run,
        "step": 2, "training_history": embedded,
        "rank_monitor_history": embedded["stable_rank"],
    }, path)
    before = path.read_bytes()
    result = progress.get_symbol_progress("EURUSD")
    assert result.status == "incompatible"
    assert result.current_step == 0
    assert path.read_bytes() == before


def test_unrelated_symbol_corruption_does_not_contaminate_valid_progress(
    monkeypatch, tmp_path
) -> None:
    checkpoints, strategies = _layout(monkeypatch, tmp_path, train_steps=10)
    artifact = strategy_artifact()
    _checkpoint(checkpoints / artifact.run_identity.checkpoint_filename(10), artifact)
    _history(tmp_path / artifact.run_identity.history_filename(), artifact, steps=range(10))
    _strategy(strategies, artifact)
    unrelated = strategies / "best_v3_GBPUSD_H1_badbadbadbad_run_deadbeef.json"
    unrelated.write_bytes(b"{malformed")
    before = unrelated.read_bytes()
    result = progress.get_symbol_progress("EURUSD")
    assert result.status == "completed"
    assert result.current_step == 10
    assert result.incompatible_reasons == ()
    assert unrelated.read_bytes() == before


def test_same_symbol_corruption_fails_progress_closed(monkeypatch, tmp_path) -> None:
    checkpoints, strategies = _layout(monkeypatch, tmp_path, train_steps=10)
    artifact = strategy_artifact()
    _checkpoint(checkpoints / artifact.run_identity.checkpoint_filename(10), artifact)
    _history(tmp_path / artifact.run_identity.history_filename(), artifact, steps=range(10))
    _strategy(strategies, artifact)
    damaged = strategies / "best_v3_EURUSD_H1_badbadbadbad_run_deadbeef.json"
    damaged.write_bytes(b"{malformed")
    result = progress.get_symbol_progress("EURUSD")
    assert result.status == "incompatible"
    assert result.current_step == 0
