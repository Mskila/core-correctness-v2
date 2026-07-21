import hashlib
import io
import json
import os
import zipfile
from pathlib import Path

import pytest
import torch

import web.progress as progress
import web.training_package as packages
from model_core.artifacts import StrategyArtifact, TrainingRunIdentity
from model_core.semantics import ArtifactCompatibilityError
from model_core.vocab import FORMULA_VOCAB
from tests.unit.test_artifacts import (
    artifact_identity,
    current_strategy_artifact as strategy_artifact,
)
from tests.unit.test_web_progress_v2 import IntSubclass, _history_value


def _package(
    artifact, *, step=7, payload_step=None, manifest_step=None, names=None,
    embedded=None, standalone=None, rank_monitor=None,
):
    payload_step = step if payload_step is None else payload_step
    embedded = (_history_value(artifact, range(step), include_identity=False)
                if embedded is None else embedded)
    standalone = (_history_value(artifact, range(step), include_identity=True)
                  if standalone is None else standalone)
    checkpoint = io.BytesIO()
    torch.save({
        "checkpoint_schema_version": "checkpoint-v3",
        "run_identity": artifact.run_identity.to_dict(),
        "step": payload_step,
        "training_history": embedded,
        "rank_monitor_history": (
            embedded.get("stable_rank") if rank_monitor is None and isinstance(embedded, dict)
            else rank_monitor
        ),
    }, checkpoint)
    files = {
        f"checkpoints/{artifact.run_identity.checkpoint_filename(step)}": checkpoint.getvalue(),
        f"strategies/{artifact.run_identity.strategy_filename()}": json.dumps(artifact.to_dict()).encode(),
        artifact.run_identity.history_filename(): json.dumps(standalone).encode(),
    }
    if names:
        files = {new: data for new, data in zip(names, files.values())}
    manifest = {
        "package_schema": packages.PACKAGE_SCHEMA,
        "run_id": artifact.run_identity.run_id,
        "artifact_fingerprint": artifact.run_identity.artifact_identity.fingerprint,
        "step": step if manifest_step is None else manifest_step,
        "files": {name: hashlib.sha256(data).hexdigest() for name, data in files.items()},
    }
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in files.items():
            zf.writestr(name, data)
        zf.writestr("manifest.json", json.dumps(manifest))
    return buf.getvalue(), files


def _tree(root):
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def _layout(monkeypatch, tmp_path):
    checkpoints = tmp_path / "checkpoints"
    strategies = tmp_path / "strategies"
    checkpoints.mkdir(exist_ok=True)
    strategies.mkdir(exist_ok=True)
    monkeypatch.setattr(packages, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(packages, "CHECKPOINT_DIR", checkpoints)
    monkeypatch.setattr(packages, "STRATEGIES_DIR", strategies)
    monkeypatch.setattr(progress, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(progress, "CHECKPOINT_DIR", checkpoints)
    monkeypatch.setattr(progress, "STRATEGIES_DIR", strategies)
    progress.invalidate_checkpoint_cache()
    return checkpoints, strategies


def test_import_is_transactional_when_later_destination_conflicts(monkeypatch, tmp_path) -> None:
    artifact = strategy_artifact()
    content, _ = _package(artifact)
    _layout(monkeypatch, tmp_path)
    conflict = tmp_path / "strategies" / artifact.run_identity.strategy_filename()
    conflict.write_bytes(b"existing-different")
    before = _tree(tmp_path)
    with pytest.raises(ValueError, match="different bytes"):
        packages.import_training_package(content, "run.zip")
    assert _tree(tmp_path) == before


def test_import_is_idempotent_for_identical_bytes(monkeypatch, tmp_path) -> None:
    artifact = strategy_artifact()
    content, _ = _package(artifact)
    _layout(monkeypatch, tmp_path)
    first = packages.import_training_package(content, "run.zip")
    before = _tree(tmp_path)
    second = packages.import_training_package(content, "run.zip")
    assert second["installed"] == first["installed"]
    assert _tree(tmp_path) == before


def test_publication_failure_rolls_back_only_files_from_attempt(monkeypatch, tmp_path) -> None:
    artifact = strategy_artifact()
    content, _ = _package(artifact)
    _layout(monkeypatch, tmp_path)
    marker = tmp_path / "keep.bin"
    marker.write_bytes(b"unchanged")
    before = _tree(tmp_path)
    publish = packages._publish_immutable
    calls = 0

    def fail_after_first(relative, data):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected publication failure")
        return publish(relative, data)

    monkeypatch.setattr(packages, "_publish_immutable", fail_after_first)
    with pytest.raises(OSError, match="injected publication failure"):
        packages.import_training_package(content, "run.zip")
    assert _tree(tmp_path) == before


def test_noncanonical_v2_member_names_are_rejected_without_writes(monkeypatch, tmp_path) -> None:
    artifact = strategy_artifact()
    content, _ = _package(artifact, names=(
        "checkpoints/ckpt_EURUSD_step_7.pt",
        "strategies/best_EURUSD.json",
        "training_history_EURUSD.json",
    ))
    _layout(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="canonical"):
        packages.import_training_package(content, "legacy-names.zip")
    assert _tree(tmp_path) == {}


def test_hash_failure_preserves_existing_tree(monkeypatch, tmp_path) -> None:
    artifact = strategy_artifact()
    content, _ = _package(artifact)
    _layout(monkeypatch, tmp_path)
    marker = tmp_path / "keep.bin"
    marker.write_bytes(b"unchanged")
    tampered = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(content)) as source, zipfile.ZipFile(tampered, "w") as target:
        for name in source.namelist():
            data = source.read(name)
            if name == artifact.run_identity.history_filename():
                data += b"tampered"
            target.writestr(name, data)
    before = _tree(tmp_path)
    with pytest.raises(ValueError, match="hash mismatch"):
        packages.import_training_package(tampered.getvalue(), "tampered.zip")
    assert _tree(tmp_path) == before


def test_mixed_identity_failure_preserves_existing_tree(monkeypatch, tmp_path) -> None:
    artifact = strategy_artifact()
    other = strategy_artifact()
    content, _ = _package(artifact)
    mixed = io.BytesIO()
    strategy_name = f"strategies/{artifact.run_identity.strategy_filename()}"
    replacement = json.dumps(other.to_dict()).encode()
    with zipfile.ZipFile(io.BytesIO(content)) as source:
        manifest = json.loads(source.read("manifest.json"))
        manifest["files"][strategy_name] = hashlib.sha256(replacement).hexdigest()
        with zipfile.ZipFile(mixed, "w") as target:
            for name in source.namelist():
                if name == "manifest.json":
                    target.writestr(name, json.dumps(manifest))
                elif name == strategy_name:
                    target.writestr(name, replacement)
                else:
                    target.writestr(name, source.read(name))
    _layout(monkeypatch, tmp_path)
    marker = tmp_path / "keep.bin"
    marker.write_bytes(b"unchanged")
    before = _tree(tmp_path)
    with pytest.raises(ValueError, match="mixed"):
        packages.import_training_package(mixed.getvalue(), "mixed.zip")
    assert _tree(tmp_path) == before


def test_rollback_preserves_concurrently_replaced_foreign_bytes(monkeypatch, tmp_path) -> None:
    artifact = strategy_artifact()
    content, _ = _package(artifact)
    _layout(monkeypatch, tmp_path)
    publish = packages._publish_immutable
    foreign = b"CONCURRENT-FOREIGN-BYTES"
    first_destination = None
    calls = 0

    def replace_then_fail(relative, data):
        nonlocal calls, first_destination
        calls += 1
        if calls == 2:
            raise OSError("later publication failed")
        created = publish(relative, data)
        if created:
            first_destination = tmp_path / relative
            first_destination.write_bytes(foreign)
        return created

    monkeypatch.setattr(packages, "_publish_immutable", replace_then_fail)
    with pytest.raises(OSError, match="later publication failed"):
        packages.import_training_package(content, "run.zip")
    assert first_destination is not None
    assert first_destination.read_bytes() == foreign


def test_rollback_preserves_replacement_after_final_stat_before_delete(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(packages, "PROJECT_ROOT", tmp_path)
    relative = "checkpoints/owned.pt"
    owned = b"ATTEMPT-OWNED-BYTES"
    foreign = b"CONCURRENT-FOREIGN-BYTES"
    receipt = packages._publish_immutable(relative, owned)
    assert receipt is not None
    destination = tmp_path / relative
    original_stat = packages.Path.stat
    target_stat_calls = 0

    def replace_after_returning_final_owned_stat(path, *args, **kwargs):
        nonlocal target_stat_calls
        result = original_stat(path, *args, **kwargs)
        if path == destination:
            target_stat_calls += 1
            if target_stat_calls == 2:
                replacement = destination.with_name("foreign-replacement.pt")
                replacement.write_bytes(foreign)
                os.replace(replacement, destination)
        return result

    monkeypatch.setattr(packages.Path, "stat", replace_after_returning_final_owned_stat)
    packages._rollback_publication(receipt)
    assert target_stat_calls >= 2
    assert destination.exists()
    assert destination.read_bytes() == foreign


@pytest.mark.parametrize("payload_step", [7.0, True, IntSubclass(7), -1, 9001])
def test_import_rejects_coercive_or_bounded_checkpoint_steps_without_writes(
    monkeypatch, tmp_path, payload_step
) -> None:
    artifact = strategy_artifact()
    content, _ = _package(artifact, payload_step=payload_step)
    _layout(monkeypatch, tmp_path)
    marker = tmp_path / "legacy.bin"
    marker.write_bytes(b"OLD-BYTES")
    before = _tree(tmp_path)
    with pytest.raises((ValueError, ArtifactCompatibilityError)):
        packages.import_training_package(content, "run.zip")
    assert _tree(tmp_path) == before


@pytest.mark.parametrize("manifest_step", [7.0, True, 6, 9001])
def test_import_requires_exact_manifest_checkpoint_step_agreement(
    monkeypatch, tmp_path, manifest_step
) -> None:
    artifact = strategy_artifact()
    content, _ = _package(artifact, manifest_step=manifest_step)
    _layout(monkeypatch, tmp_path)
    with pytest.raises(ValueError):
        packages.import_training_package(content, "run.zip")
    assert _tree(tmp_path) == {}


@pytest.mark.parametrize(
    ("step", "history_steps"),
    [(7, []), (7, [0, 1, 2]), (7, list(range(8))), (0, [0])],
)
def test_import_rejects_inconsistent_checkpoint_histories_before_writes(
    monkeypatch, tmp_path, step, history_steps
) -> None:
    artifact = strategy_artifact()
    embedded = _history_value(artifact, history_steps, include_identity=False)
    standalone = _history_value(artifact, history_steps, include_identity=True)
    content, _ = _package(
        artifact, step=step, embedded=embedded, standalone=standalone
    )
    _layout(monkeypatch, tmp_path)
    with pytest.raises(ValueError):
        packages.import_training_package(content, "run.zip")
    assert _tree(tmp_path) == {}


def test_import_rejects_malformed_embedded_or_standalone_history_before_writes(
    monkeypatch, tmp_path
) -> None:
    artifact = strategy_artifact()
    embedded = _history_value(artifact, range(7), include_identity=False)
    embedded.pop("avg_reward")
    standalone = _history_value(artifact, range(7), include_identity=True)
    content, _ = _package(
        artifact, embedded=embedded, standalone=standalone,
        rank_monitor=embedded["stable_rank"],
    )
    _layout(monkeypatch, tmp_path)
    marker = tmp_path / "old.pkg"
    marker.write_bytes(b"PRESERVE")
    before = _tree(tmp_path)
    with pytest.raises(ValueError):
        packages.import_training_package(content, "run.zip")
    assert _tree(tmp_path) == before


def test_import_rejects_rank_monitor_mismatch_before_writes(monkeypatch, tmp_path) -> None:
    artifact = strategy_artifact()
    content, _ = _package(artifact, rank_monitor=[1.0])
    _layout(monkeypatch, tmp_path)
    with pytest.raises(ValueError):
        packages.import_training_package(content, "run.zip")
    assert _tree(tmp_path) == {}


def test_export_validates_complete_history_before_returning_package(monkeypatch, tmp_path) -> None:
    artifact = strategy_artifact()
    checkpoints, strategies = _layout(monkeypatch, tmp_path)
    embedded = _history_value(artifact, range(7), include_identity=False)
    checkpoint = checkpoints / artifact.run_identity.checkpoint_filename(7)
    stream = io.BytesIO()
    torch.save({
        "checkpoint_schema_version": "checkpoint-v3",
        "run_identity": artifact.run_identity.to_dict(), "step": 7,
        "training_history": embedded, "rank_monitor_history": [],
    }, stream)
    checkpoint.write_bytes(stream.getvalue())
    (strategies / artifact.run_identity.strategy_filename()).write_text(
        json.dumps(artifact.to_dict()), encoding="utf-8"
    )
    standalone = _history_value(artifact, range(7), include_identity=True)
    standalone["stable_rank"] = [1.0]
    history_path = tmp_path / artifact.run_identity.history_filename()
    history_path.write_text(json.dumps(standalone), encoding="utf-8")
    before = _tree(tmp_path)
    with pytest.raises(ValueError):
        packages.build_training_export_zip("EURUSD")
    assert _tree(tmp_path) == before


def _write_export_run(checkpoints, strategies, root, artifact, *, step=7, payload_step=None):
    embedded = _history_value(artifact, range(step), include_identity=False)
    stream = io.BytesIO()
    torch.save({
        "checkpoint_schema_version": "checkpoint-v3",
        "run_identity": artifact.run_identity.to_dict(),
        "step": step if payload_step is None else payload_step,
        "training_history": embedded,
        "rank_monitor_history": embedded["stable_rank"],
    }, stream)
    (checkpoints / artifact.run_identity.checkpoint_filename(step)).write_bytes(stream.getvalue())
    (strategies / artifact.run_identity.strategy_filename()).write_text(
        json.dumps(artifact.to_dict()), encoding="utf-8"
    )
    (root / artifact.run_identity.history_filename()).write_text(
        json.dumps(_history_value(artifact, range(step), include_identity=True)),
        encoding="utf-8",
    )


def _artifact_for_seed(seed: int, run_id: str | None = None) -> StrategyArtifact:
    template = strategy_artifact()
    identity = artifact_identity(seed=seed)
    run = (
        TrainingRunIdentity.create(identity)
        if run_id is None
        else TrainingRunIdentity(run_id=run_id, artifact_identity=identity)
    )
    return StrategyArtifact.create(
        run_identity=run,
        formula_tokens=template.formula_tokens,
        decoded_formula=template.decoded_formula,
        best_score=template.best_score,
        fold_evidence=template.fold_evidence,
        generated_at=template.generated_at,
        candidate_evaluation_count=1,
    )


def test_export_strategy_competition_uses_chronological_generated_at(
    monkeypatch, tmp_path
) -> None:
    checkpoints, strategies = _layout(monkeypatch, tmp_path)
    base = _artifact_for_seed(61)

    def same_run(generated_at, token):
        return StrategyArtifact.create(
            run_identity=base.run_identity,
            formula_tokens=[token],
            decoded_formula=FORMULA_VOCAB.token_names[token],
            best_score=base.best_score,
            fold_evidence=base.fold_evidence,
            generated_at=generated_at,
            candidate_evaluation_count=1,
        )

    earlier = same_run("2026-07-16T00:00:00Z", 0)
    later = same_run("2026-07-16T00:00:00.500000Z", 1)
    _write_export_run(checkpoints, strategies, tmp_path, earlier)
    source_paths = []
    for label, artifact in (("earlier", earlier), ("later", later)):
        directory = tmp_path / label
        directory.mkdir()
        path = directory / artifact.run_identity.strategy_filename()
        path.write_text(json.dumps(artifact.to_dict()), encoding="utf-8")
        source_paths.append(path)

    class _StrategySources:
        def glob(self, pattern):
            assert pattern == "best_v3_*.json"
            return list(source_paths)

    monkeypatch.setattr(packages, "STRATEGIES_DIR", _StrategySources())
    before = _tree(tmp_path)
    content, _ = packages.build_training_export_zip("EURUSD")
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        member = f"strategies/{base.run_identity.strategy_filename()}"
        selected = json.loads(archive.read(member))
    assert selected["generated_at"] == "2026-07-16T00:00:00.500000Z"
    assert selected["formula_tokens"] == [1]
    assert _tree(tmp_path) == before


def test_export_strategy_competition_preserves_submicrosecond_order(
    monkeypatch, tmp_path
) -> None:
    checkpoints, strategies = _layout(monkeypatch, tmp_path)
    base = _artifact_for_seed(62)

    def same_run(generated_at, token):
        return StrategyArtifact.create(
            run_identity=base.run_identity,
            formula_tokens=[token],
            decoded_formula=FORMULA_VOCAB.token_names[token],
            best_score=base.best_score,
            fold_evidence=base.fold_evidence,
            generated_at=generated_at,
            candidate_evaluation_count=1,
        )

    earlier = same_run("2026-07-16T00:00:00.0000001Z", 0)
    later = same_run("2026-07-16T00:00:00.0000002Z", 1)
    _write_export_run(checkpoints, strategies, tmp_path, earlier)
    source_paths = []
    for label, artifact in (("earlier-submicro", earlier), ("later-submicro", later)):
        directory = tmp_path / label
        directory.mkdir()
        path = directory / artifact.run_identity.strategy_filename()
        path.write_text(json.dumps(artifact.to_dict()), encoding="utf-8")
        source_paths.append(path)

    class _StrategySources:
        def glob(self, pattern):
            assert pattern == "best_v3_*.json"
            return list(source_paths)

    monkeypatch.setattr(packages, "STRATEGIES_DIR", _StrategySources())
    before = _tree(tmp_path)
    content, _ = packages.build_training_export_zip("EURUSD")
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        selected = json.loads(
            archive.read(f"strategies/{base.run_identity.strategy_filename()}")
        )
    assert selected["generated_at"] == "2026-07-16T00:00:00.0000002Z"
    assert selected["formula_tokens"] == [1]
    assert _tree(tmp_path) == before


def test_progress_and_export_choose_same_step_checkpoint_100ns_newer(
    monkeypatch, tmp_path
) -> None:
    checkpoints, strategies = _layout(monkeypatch, tmp_path)
    older = _artifact_for_seed(81, "f" * 32)
    newer = _artifact_for_seed(82, "0" * 32)
    _write_export_run(checkpoints, strategies, tmp_path, older)
    _write_export_run(checkpoints, strategies, tmp_path, newer)
    older_path = checkpoints / older.run_identity.checkpoint_filename(7)
    newer_path = checkpoints / newer.run_identity.checkpoint_filename(7)
    older_ns = 1_700_020_200_000_000_000
    newer_ns = older_ns + 100
    os.utime(older_path, ns=(older_ns, older_ns))
    os.utime(newer_path, ns=(newer_ns, newer_ns))
    assert older_path.name > newer_path.name
    assert older_path.stat().st_mtime == newer_path.stat().st_mtime
    assert older_path.stat().st_mtime_ns < newer_path.stat().st_mtime_ns
    progress.invalidate_checkpoint_cache()
    reported = progress.get_symbol_progress("EURUSD")
    content, _ = packages.build_training_export_zip("EURUSD")
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        manifest = json.loads(archive.read("manifest.json"))
    assert Path(reported.checkpoint_path).name == newer_path.name
    assert reported.run_id == newer.run_identity.run_id
    assert manifest["run_id"] == newer.run_identity.run_id


def _assert_export_matches_progress(symbol, content, filename, reported, artifacts):
    checkpoint_name = Path(reported.checkpoint_path).name
    expected = next(
        artifact
        for artifact in artifacts
        if artifact.run_identity.checkpoint_filename(7) == checkpoint_name
    )
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["run_id"] == expected.run_identity.run_id
        assert manifest["artifact_fingerprint"] == expected.run_identity.artifact_identity.fingerprint
        names = set(archive.namelist())
        assert f"checkpoints/{expected.run_identity.checkpoint_filename(7)}" in names
        assert f"strategies/{expected.run_identity.strategy_filename()}" in names
        assert expected.run_identity.history_filename() in names
    assert f"run_{expected.run_identity.run_id[:8]}_step_7.zip" in filename


def test_export_selects_progress_current_run_for_same_step_newer_mtime(
    monkeypatch, tmp_path
) -> None:
    checkpoints, strategies = _layout(monkeypatch, tmp_path)
    older = _artifact_for_seed(41)
    newer = _artifact_for_seed(42)
    _write_export_run(checkpoints, strategies, tmp_path, older)
    _write_export_run(checkpoints, strategies, tmp_path, newer)
    older_path = checkpoints / older.run_identity.checkpoint_filename(7)
    newer_path = checkpoints / newer.run_identity.checkpoint_filename(7)
    os.utime(older_path, (1_700_000_000, 1_700_000_000))
    os.utime(newer_path, (1_700_000_100, 1_700_000_100))
    progress.invalidate_checkpoint_cache()

    reported = progress.get_symbol_progress("EURUSD")
    before = _tree(tmp_path)
    content, filename = packages.build_training_export_zip("EURUSD")
    _assert_export_matches_progress("EURUSD", content, filename, reported, [older, newer])
    assert Path(reported.checkpoint_path).name == newer_path.name
    assert _tree(tmp_path) == before


def test_export_selects_progress_current_run_for_exact_mtime_filename_tie(
    monkeypatch, tmp_path
) -> None:
    checkpoints, strategies = _layout(monkeypatch, tmp_path)
    first = _artifact_for_seed(43)
    second = _artifact_for_seed(44)
    artifacts = [first, second]
    for artifact in artifacts:
        _write_export_run(checkpoints, strategies, tmp_path, artifact)
    tied_ns = 1_700_000_200_000_000_000
    paths = [checkpoints / artifact.run_identity.checkpoint_filename(7) for artifact in artifacts]
    for path in paths:
        os.utime(path, ns=(tied_ns, tied_ns))
    progress.invalidate_checkpoint_cache()

    reported = progress.get_symbol_progress("EURUSD")
    content, filename = packages.build_training_export_zip("EURUSD")
    _assert_export_matches_progress("EURUSD", content, filename, reported, artifacts)
    assert Path(reported.checkpoint_path).name == max(path.name for path in paths)


def test_valid_export_contains_one_identity_coherent_run(monkeypatch, tmp_path) -> None:
    artifact = strategy_artifact()
    checkpoints, strategies = _layout(monkeypatch, tmp_path)
    _write_export_run(checkpoints, strategies, tmp_path, artifact)
    before = _tree(tmp_path)
    content, filename = packages.build_training_export_zip("EURUSD")
    assert filename.endswith("_step_7.zip")
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["run_id"] == artifact.run_identity.run_id
        assert manifest["artifact_fingerprint"] == artifact.run_identity.artifact_identity.fingerprint
        assert manifest["step"] == 7
        assert set(manifest["files"]) == set(archive.namelist()) - {"manifest.json"}
    assert _tree(tmp_path) == before


def test_export_ignores_preserved_noncanonical_same_run_strategy_neighbor(
    monkeypatch, tmp_path
) -> None:
    artifact = strategy_artifact()
    checkpoints, strategies = _layout(monkeypatch, tmp_path)
    _write_export_run(checkpoints, strategies, tmp_path, artifact)
    canonical = strategies / artifact.run_identity.strategy_filename()
    noncanonical = strategies / "best_v2_EURUSD_H1_neighbor-copy.json"
    noncanonical.write_bytes(canonical.read_bytes())
    before = _tree(tmp_path)

    content, _ = packages.build_training_export_zip("EURUSD")

    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        member = f"strategies/{artifact.run_identity.strategy_filename()}"
        assert member in archive.namelist()
        assert archive.read(member) == canonical.read_bytes()
        assert not any("neighbor-copy" in name for name in archive.namelist())
    assert _tree(tmp_path) == before


def test_export_rejects_noncanonical_only_strategy_without_changing_bytes(
    monkeypatch, tmp_path
) -> None:
    artifact = strategy_artifact()
    checkpoints, strategies = _layout(monkeypatch, tmp_path)
    _write_export_run(checkpoints, strategies, tmp_path, artifact)
    canonical = strategies / artifact.run_identity.strategy_filename()
    noncanonical = strategies / "best_v2_EURUSD_H1_only-copy.json"
    noncanonical.write_bytes(canonical.read_bytes())
    canonical.unlink()
    before = _tree(tmp_path)

    with pytest.raises(ValueError, match="strategy|canonical"):
        packages.build_training_export_zip("EURUSD")
    assert _tree(tmp_path) == before


@pytest.mark.parametrize("payload_step", [7.0, IntSubclass(7)])
def test_export_rejects_coercive_checkpoint_step_without_changing_bytes(
    monkeypatch, tmp_path, payload_step
) -> None:
    artifact = strategy_artifact()
    checkpoints, strategies = _layout(monkeypatch, tmp_path)
    _write_export_run(
        checkpoints, strategies, tmp_path, artifact, payload_step=payload_step
    )
    before = _tree(tmp_path)
    with pytest.raises(ValueError, match="exact built-in integer"):
        packages.build_training_export_zip("EURUSD")
    assert _tree(tmp_path) == before
