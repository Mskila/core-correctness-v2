import json

import pytest

from model_core.features import (
    FEATURE_REGISTRY,
    _load_active_feature_allowlist,
)
from model_core.ops import OPERATOR_REGISTRY
from model_core.semantics import ArtifactCompatibilityError
from model_core.vocab import FORMULA_VOCAB, compute_vocab_version


def _write(path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_missing_allowlist_uses_all_v2_features(tmp_path) -> None:
    missing = tmp_path / "missing.json"
    assert _load_active_feature_allowlist(missing) is None
    assert len(FEATURE_REGISTRY.feature_names) == 60


@pytest.mark.parametrize(
    "payload",
    [
        {"active_features": ["RET"]},
        {"core_semantics_version": "1", "active_features": ["RET"]},
        {"core_semantics_version": "2", "active_features": ["UNKNOWN"]},
        {"core_semantics_version": "2", "active_features": []},
    ],
)
def test_incompatible_allowlist_fails_closed(tmp_path, payload) -> None:
    path = tmp_path / "active_features.json"
    _write(path, payload)
    with pytest.raises(ArtifactCompatibilityError):
        _load_active_feature_allowlist(path)


def test_damaged_allowlist_json_fails_closed(tmp_path) -> None:
    path = tmp_path / "active_features.json"
    path.write_text("{not-json", encoding="utf-8")
    with pytest.raises(ArtifactCompatibilityError):
        _load_active_feature_allowlist(path)


def test_duplicate_active_feature_names_fail_closed(tmp_path) -> None:
    path = tmp_path / "active_features.json"
    _write(
        path,
        {"core_semantics_version": "2", "active_features": ["RET", "RET"]},
    )
    with pytest.raises(ArtifactCompatibilityError, match="duplicate"):
        _load_active_feature_allowlist(path)


def test_valid_allowlist_is_smaller_and_has_new_vocab_hash(tmp_path) -> None:
    path = tmp_path / "active_features.json"
    _write(
        path,
        {"core_semantics_version": "2", "active_features": ["RET", "RET5"]},
    )
    allowlist = _load_active_feature_allowlist(path)
    assert allowlist == {"RET", "RET5"}

    feature_entries = tuple(
        (spec.name, spec.lookback)
        for spec in FEATURE_REGISTRY.feature_specs
        if spec.name in allowlist
    )
    operator_entries = tuple(
        (spec.name, spec.arity, spec.lookback)
        for spec in OPERATOR_REGISTRY.operator_specs
    )
    smaller_version = compute_vocab_version(feature_entries, operator_entries)
    assert len(feature_entries) < FORMULA_VOCAB.feature_count
    assert smaller_version != FORMULA_VOCAB.version
