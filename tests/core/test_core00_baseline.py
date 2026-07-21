from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest
import torch

from benchmarks.core00 import BENCHMARK_SCHEMA_VERSION, run_benchmark
from tests.support.core00 import (
    CORE00_FIXTURE_SHA256,
    CORE00_FORMULA_DIGESTS,
    CORE00_REFERENCE_TRACE_SHA256,
    TRACE_SCHEMA_VERSION,
    load_formula_corpus,
    load_known_defects,
    load_ohlcv_fixture,
    resolve_formula_tokens,
    run_reference_trace,
    trace_digest,
)


pytestmark = pytest.mark.core

ROOT = Path(__file__).resolve().parents[2]


def test_fixed_ohlcv_fixture_is_local_complete_and_byte_stable() -> None:
    first = load_ohlcv_fixture()
    second = load_ohlcv_fixture()

    assert first.canonical_sha256 == CORE00_FIXTURE_SHA256
    assert second.canonical_sha256 == CORE00_FIXTURE_SHA256
    assert first.canonical_bytes == second.canonical_bytes
    for field in ("open", "high", "low", "close", "volume", "time_ns"):
        assert torch.equal(first.tensors[field], second.tensors[field])

    assert first.metadata["source"] == "local-core00-golden"
    assert first.metadata["symbol"] == "CORE00"
    assert first.metadata["timeframe"] == "1h"
    assert first.metadata["gap_policy"] == "explicitly-allowed-characterization"
    assert len(first.metadata["exit_timestamps_ns"]) == 2
    assert len(first.metadata["gap_after_indices"]) == 1
    assert bool((first.tensors["volume"] == 0).any())
    assert bool((first.tensors["high"] >= torch.maximum(first.tensors["open"], first.tensors["close"])).all())
    assert bool((first.tensors["low"] <= torch.minimum(first.tensors["open"], first.tensors["close"])).all())
    assert any(value != int(value) for value in first.payload["close"])
    close = first.tensors["close"]
    assert bool((close[0, 1:17] > close[0, :16]).all())
    assert bool((close[0, 17:] < close[0, 16:-1]).all())


def test_formula_corpus_covers_required_shapes_without_goldenizing_known_defects() -> None:
    corpus = load_formula_corpus()
    categories = {entry["category"] for entry in corpus}
    assert {
        "pure_feature", "unary", "binary", "ternary", "rolling",
        "ts_decay", "nonfinite_prone", "structural_boundary",
    } <= categories

    fixture = load_ohlcv_fixture()
    feature_tensor = fixture.feature_tensor()
    valid_digests = {}
    for entry in corpus:
        tokens = resolve_formula_tokens(entry["tokens"])
        assert tokens == resolve_formula_tokens(entry["tokens"])
        if entry["classification"] == "valid":
            output = fixture.vm.execute(tokens, feature_tensor)
            assert output is not None, entry["id"]
            assert bool(torch.isfinite(output).all()), entry["id"]
            valid_digests[entry["id"]] = fixture.tensor_digest(output)
        elif entry["classification"] == "structural_boundary":
            assert fixture.vm.execute(tokens, feature_tensor) is None
        elif entry["classification"] == "expected_error":
            assert fixture.vm.execute(tokens, feature_tensor) is None
            assert fixture.vm.last_error is not None
            assert fixture.vm.last_error.error_kind.value == entry["expected_error_kind"]
        else:
            assert entry["classification"] == "known_defect"
            assert entry["known_defect_ids"]
            assert "expected_tensor_digest" not in entry

    assert valid_digests
    expected_digests = {
        entry["id"]: entry["expected_tensor_digest"]
        for entry in corpus
        if entry["classification"] == "valid"
    }
    runtime_key = (sys.platform, torch.__version__.split("+")[0])
    expected_digests = CORE00_FORMULA_DIGESTS.get(runtime_key, expected_digests)
    assert valid_digests == expected_digests, valid_digests
    assert set(valid_digests) == {
        "feature-ret", "unary-abs", "binary-add", "ternary-if-gt",
        "rolling-mean-5", "decay-exp-5"
    }


def test_known_defect_register_has_exact_review_counts_and_no_correctness_oracle() -> None:
    defects = load_known_defects()
    assert len(defects) == 30
    assert len({item["id"] for item in defects}) == 30
    counts = {
        priority: sum(item["priority"] == priority for item in defects)
        for priority in ("P0", "P1", "P2", "P3")
    }
    assert counts == {"P0": 8, "P1": 13, "P2": 8, "P3": 1}
    resolved_core04 = {"F-010", "F-011", "F-012", "F-032", "F-039"}
    resolved_core05 = {"F-009", "F-026"}
    resolved_perf01 = {"F-020", "F-021", "F-033", "F-034"}
    assert {
        item["id"]
        for item in defects
        if item["status"] == "resolved-core04"
    } == resolved_core04
    assert {
        item["id"]
        for item in defects
        if item["status"] == "resolved-core05"
    } == resolved_core05
    assert {
        item["id"]
        for item in defects
        if item["status"] == "resolved-perf01"
    } == resolved_perf01
    assert all(
        item["characterization_policy"] == (
            "regression-covered"
            if item["id"] in resolved_core04 | resolved_core05 | resolved_perf01
            else "do-not-goldenize"
        )
        for item in defects
    )
    forbidden = {"expected", "expected_value", "golden", "accepted_behavior"}
    assert all(forbidden.isdisjoint(item) for item in defects)
    expected_packages = {
        "F-001": "CORE-01", "F-002": "CORE-02", "F-003": "CORE-02",
        "F-008": "CORE-03", "F-009": "CORE-05", "F-010": "CORE-04",
        "F-011": "CORE-04", "F-012": "CORE-04", "F-006": "CORE-01/CORE-07",
        "F-013": "CORE-02", "F-015": "CORE-07", "F-016": "CORE-07",
        "F-019": "CORE-08", "F-023": "CORE-01", "F-024": "CORE-03",
        "F-026": "CORE-05", "F-027": "CORE-01", "F-032": "CORE-04",
        "F-035": "CORE-02", "F-036": "CORE-02", "F-038": "CORE-07",
        "F-020": "PERF-01", "F-021": "PERF-01", "F-022": "CORE-03",
        "F-025": "CORE-03", "F-029": "CORE-08", "F-033": "PERF-01",
        "F-034": "PERF-01", "F-039": "CORE-04/CORE-08", "F-040": "CLEAN-01",
    }
    assert {item["id"]: item["work_package"] for item in defects} == expected_packages


@pytest.mark.deterministic
def test_short_reference_trace_repeats_exactly_for_same_seed(tmp_path) -> None:
    first = run_reference_trace(seed=20260721, steps=2, work_dir=tmp_path / "first")
    second = run_reference_trace(seed=20260721, steps=2, work_dir=tmp_path / "second")

    assert first["schema_version"] == TRACE_SCHEMA_VERSION
    assert trace_digest(first) == trace_digest(second)
    runtime_key = (sys.platform, torch.__version__.split("+")[0])
    assert runtime_key in CORE00_REFERENCE_TRACE_SHA256
    assert trace_digest(first) == CORE00_REFERENCE_TRACE_SHA256[runtime_key]
    assert first == second
    assert len(first["steps"]) == 2
    for step in first["steps"]:
        assert step["sampled_tokens"]
        assert len(step["factor_digests"]) == len(step["sampled_tokens"])
        assert len(step["fold_raw_scores"]) == len(step["sampled_tokens"])
        assert len(step["final_rewards"]) == len(step["sampled_tokens"])
    state = first["final_state"]
    assert state["best_formula"] is not None
    assert state["elite_pool"]
    assert state["factor_pool"]
    for name in ("model_digest", "optimizer_digest", "rng_digest"):
        assert len(state[name]) == 64


@pytest.mark.performance
def test_benchmark_smoke_emits_required_schema_and_metadata(tmp_path) -> None:
    output = tmp_path / "core00-benchmark.json"
    report = run_benchmark(iterations=1, output_path=output, repo_root=ROOT)
    loaded = json.loads(output.read_text(encoding="utf-8"))

    assert loaded == report
    assert report["schema_version"] == BENCHMARK_SCHEMA_VERSION
    assert set(report["metadata"]) >= {
        "machine", "python", "pytorch", "os", "cpu_threads",
        "dataset_fingerprint", "commit_sha",
    }
    assert report["metadata"]["dataset_fingerprint"] == CORE00_FIXTURE_SHA256
    assert len(report["metadata"]["commit_sha"]) == 40
    expected_stages = {
        "sampler", "vm", "fold_scoring", "decision", "backward",
        "transaction", "logging", "checkpoint",
    }
    assert set(report["timings_ns"]) == expected_stages
    for samples in report["timings_ns"].values():
        assert len(samples) == 1
        assert type(samples[0]) is int and samples[0] >= 0
    assert report["required_ci"] is False
