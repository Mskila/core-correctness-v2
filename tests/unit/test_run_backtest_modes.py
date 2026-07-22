from __future__ import annotations

import json

import pytest

import run_backtest
from model_core.artifacts import StrategyArtifact, sha256_json
from model_core.semantics import ArtifactCompatibilityError


def test_parser_requires_strategy_data_and_mode() -> None:
    parser = run_backtest.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
    with pytest.raises(SystemExit):
        parser.parse_args(["--strategy-file", "s.json", "--data-file", "d.parquet"])
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["--strategy-file", "s.json", "--data-file", "d.parquet", "--mode", "legacy"]
        )


def test_parser_accepts_only_explicit_v2_contract() -> None:
    args = run_backtest.build_parser().parse_args(
        [
            "--strategy-file", "s.json",
            "--data-file", "d.parquet",
            "--mode", "in_sample_replay",
            "--commission", "0.01",
            "--slippage", "0.02",
            "--output-dir", "reports",
            "--numeric-time-unit", "ms",
        ]
    )
    assert args.strategy_file == "s.json"
    assert args.data_file == "d.parquet"
    assert args.mode == "in_sample_replay"
    assert args.numeric_time_unit == "ms"

    with pytest.raises(SystemExit):
        run_backtest.build_parser().parse_args(
            [
                "--strategy-file", "s.json", "--data-file", "d.parquet",
                "--mode", "in_sample_replay", "--numeric-time-unit", "minutes",
            ]
        )


@pytest.mark.parametrize("payload", [[], {"formula": [0]}, {"symbol": "EURUSD", "formula": [0]}])
def test_load_strategy_rejects_legacy_and_identity_less_json(tmp_path, payload) -> None:
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    before = path.read_bytes()
    with pytest.raises(ArtifactCompatibilityError):
        run_backtest.load_strategy(path)
    assert path.read_bytes() == before


def test_missing_explicit_data_file_never_uses_strategy_source_path() -> None:
    parser = run_backtest.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["--strategy-file", "has-source-path.json", "--mode", "in_sample_replay"]
        )


@pytest.mark.parametrize("field", ["core_semantics_version", "vocab_version"])
def test_load_strategy_rejects_incompatible_core_or_vocab(tmp_path, field) -> None:
    from tests.unit.test_backtest_modes import strategy

    payload = strategy().to_dict()
    identity = payload["run_identity"]["artifact_identity"]
    identity[field] = "incompatible"
    identity["training_config_hash"] = sha256_json(identity["training_config"])
    path = tmp_path / "incompatible.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ArtifactCompatibilityError, match=field):
        run_backtest.load_strategy(path)


def test_load_strategy_returns_validated_artifact(tmp_path) -> None:
    from tests.unit.test_backtest_modes import strategy

    expected = strategy()
    path = tmp_path / "strategy.json"
    path.write_text(json.dumps(expected.to_dict()), encoding="utf-8")
    actual = run_backtest.load_strategy(path)
    assert isinstance(actual, StrategyArtifact)
    assert actual == expected


def test_missing_files_report_the_exact_explicit_path(tmp_path) -> None:
    missing = tmp_path / "missing.json"
    with pytest.raises(FileNotFoundError, match="missing.json"):
        run_backtest.load_strategy(missing)
