from __future__ import annotations

import json

import pytest
import torch

import run_backtest
import training_service
from model_core.legacy_audit import inspect_legacy_artifact
from model_core.semantics import (
    CHECKPOINT_SCHEMA_VERSION,
    CORE_SEMANTICS_VERSION,
    EXECUTION_SEMANTICS_VERSION,
    REPORT_SCHEMA_VERSION,
    ArtifactCompatibilityError,
)
from model_core.versions import VERSION_CHANGE_TABLE
from tests.unit.test_artifacts import strategy_artifact


def test_core06_version_table_maps_every_semantic_boundary() -> None:
    assert CORE_SEMANTICS_VERSION == "5"
    assert EXECUTION_SEMANTICS_VERSION.endswith("-v4")
    assert CHECKPOINT_SCHEMA_VERSION == "checkpoint-v3"
    assert REPORT_SCHEMA_VERSION == "backtest-report-v3"
    assert set(VERSION_CHANGE_TABLE) == {
        "data_canonicalization",
        "dataset_schema",
        "reward_core",
        "vocab_operator",
        "execution",
        "validation",
        "checkpoint",
        "strategy",
        "history",
        "backtest_report",
    }


def test_formal_backtest_rejects_pre_core_fix_strategy_but_audit_is_read_only(
    tmp_path,
) -> None:
    legacy = strategy_artifact()
    path = tmp_path / "best_v2_EURUSD.json"
    path.write_text(json.dumps(legacy.to_dict()), encoding="utf-8")

    with pytest.raises(
        ArtifactCompatibilityError,
        match=r"pre-core-fix/incompatible.*strategy-v3",
    ):
        run_backtest.load_strategy(path)

    summary = inspect_legacy_artifact(path)
    assert summary.kind == "strategy"
    assert summary.schema_version == "strategy-v2"
    assert summary.pre_core_fix is True
    assert summary.rank_eligible is False
    assert summary.formal_output_allowed is False
    assert path.read_text(encoding="utf-8") == json.dumps(legacy.to_dict())


def test_pre_core_fix_checkpoint_rejection_names_schema_mismatch(tmp_path) -> None:
    path = tmp_path / "ckpt_v2_EURUSD.pt"
    torch.save({"checkpoint_schema_version": "checkpoint-v2"}, path)

    with pytest.raises(
        ArtifactCompatibilityError,
        match=r"pre-core-fix/incompatible.*checkpoint-v3.*checkpoint-v2",
    ):
        training_service._load_candidate(path)


def test_current_run_filenames_cannot_overwrite_v2_names() -> None:
    run = strategy_artifact().run_identity
    assert run.checkpoint_filename(1).startswith("ckpt_v3_")
    assert run.strategy_filename().startswith("best_v3_")
    assert run.history_filename().startswith("training_history_v3_")
