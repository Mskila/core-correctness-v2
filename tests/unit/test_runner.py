import json
from unittest.mock import ANY, MagicMock, patch

import pytest
import torch

import strategy_manager.runner as runner_module
from model_core.artifacts import StrategyArtifact, TrainingRunIdentity
from model_core.execution import factor_to_position
from model_core.vocab import FORMULA_VOCAB
from strategy_manager.runner import MT5StrategyRunner
from tests.unit.test_artifacts import artifact_identity, strategy_artifact


def _artifact(
    generated_at: str, token: int, run_id: str | None = None
) -> StrategyArtifact:
    template = strategy_artifact()
    identity = artifact_identity()
    run = (
        TrainingRunIdentity.create(identity)
        if run_id is None
        else TrainingRunIdentity(run_id=run_id, artifact_identity=identity)
    )
    return StrategyArtifact.create(
        run_identity=run,
        formula_tokens=[token],
        decoded_formula=FORMULA_VOCAB.token_names[token],
        best_score=1.0,
        fold_evidence=template.fold_evidence,
        generated_at=generated_at,
        candidate_evaluation_count=1,
    )


def _write(directory, artifact):
    path = directory / artifact.run_identity.strategy_filename()
    path.write_text(json.dumps(artifact.to_dict()), encoding="utf-8")
    return path


def test_all_missing_v2_strategies_fail_startup(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runner_module.Config, "SYMBOLS", ["EURUSD"])
    with pytest.raises(RuntimeError, match="identity-valid V2 strategy"):
        MT5StrategyRunner()


def test_latest_valid_artifact_is_selected_and_fingerprint_recorded(monkeypatch, tmp_path):
    strategies = tmp_path / "strategies"
    strategies.mkdir()
    older = _artifact("2026-07-15T00:00:00Z", 0)
    newer = _artifact("2026-07-16T00:00:00Z", 1)
    _write(strategies, older)
    _write(strategies, newer)
    (strategies / "best_EURUSD.json").write_text("[0]", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runner_module.Config, "SYMBOLS", ["EURUSD", "GBPUSD"])
    runner = MT5StrategyRunner()
    assert runner.symbol_formulas == {"EURUSD": [1]}
    assert runner.strategy_fingerprints == {"EURUSD": newer.fingerprint}
    assert "GBPUSD" not in runner.symbol_formulas


def test_runner_orders_generated_at_as_utc_instant(monkeypatch, tmp_path):
    strategies = tmp_path / "strategies"
    strategies.mkdir()
    earlier = _artifact("2026-07-16T00:00:00Z", 0)
    later = _artifact("2026-07-16T00:00:00.500000Z", 1)
    _write(strategies, earlier)
    _write(strategies, later)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runner_module.Config, "SYMBOLS", ["EURUSD"])
    runner = MT5StrategyRunner()
    assert runner.symbol_formulas == {"EURUSD": [1]}
    assert runner.strategy_fingerprints == {"EURUSD": later.fingerprint}


def test_runner_equal_utc_instants_use_filename_tie_break(monkeypatch, tmp_path):
    strategies = tmp_path / "strategies"
    strategies.mkdir()
    z_artifact = _artifact("2026-07-16T00:00:00.1Z", 0, "0" * 32)
    offset_artifact = _artifact(
        "2026-07-16T00:00:00.1000000+00:00", 1, "f" * 32
    )
    z_path = _write(strategies, z_artifact)
    offset_path = _write(strategies, offset_artifact)
    assert offset_path.name > z_path.name
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runner_module.Config, "SYMBOLS", ["EURUSD"])
    runner = MT5StrategyRunner()
    assert runner.symbol_formulas == {"EURUSD": [1]}
    assert runner.strategy_fingerprints == {"EURUSD": offset_artifact.fingerprint}


def test_runner_preserves_submicrosecond_generated_at_order(monkeypatch, tmp_path):
    strategies = tmp_path / "strategies"
    strategies.mkdir()
    earlier = _artifact("2026-07-16T00:00:00.0000001Z", 0, "f" * 32)
    later = _artifact("2026-07-16T00:00:00.0000002Z", 1, "0" * 32)
    earlier_path = _write(strategies, earlier)
    later_path = _write(strategies, later)
    assert earlier_path.name > later_path.name
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runner_module.Config, "SYMBOLS", ["EURUSD"])
    runner = MT5StrategyRunner()
    assert runner.symbol_formulas == {"EURUSD": [1]}
    assert runner.strategy_fingerprints == {"EURUSD": later.fingerprint}


def test_compute_targets_uses_one_formula_and_shared_position_mapping(monkeypatch):
    runner = MT5StrategyRunner.__new__(MT5StrategyRunner)
    runner._data_manager = MagicMock(symbols=["EURUSD", "GBPUSD"], raw_dict={})
    runner.symbol_formulas = {"EURUSD": [0]}
    runner.strategy_fingerprints = {"EURUSD": "f" * 64}
    runner.vm = MagicMock()
    factor = torch.tensor([[0.75, -0.25]], dtype=torch.float32)
    runner.vm.execute.return_value = factor
    with patch(
        "model_core.features.MT5FeatureEngineer.compute_features",
        return_value=torch.zeros(2, 1, 2),
    ), patch("model_core.walk_forward.formula_warmup_bars", return_value=2):
        actual = runner._compute_targets()
    expected = factor_to_position(
        factor[:, -1:], min_exposure=float(runner_module.Config.MIN_TRADE_EXPOSURE)
    ).item()
    torch.testing.assert_close(actual, torch.tensor([expected, 0.0]))
    runner.vm.execute.assert_called_once_with([0], ANY)


def _target_runner(symbols, formulas):
    runner = MT5StrategyRunner.__new__(MT5StrategyRunner)
    runner._data_manager = MagicMock(symbols=symbols, raw_dict={})
    runner.symbol_formulas = formulas
    runner.strategy_fingerprints = {symbol: "f" * 64 for symbol in formulas}
    runner.missing_strategy_reasons = {}
    runner.vm = MagicMock(return_value=None)
    runner.vm.execute.return_value = torch.tensor([[0.5]], dtype=torch.float32)
    return runner


def test_one_bar_short_warmup_stays_flat_without_invoking_vm(monkeypatch):
    runner = _target_runner(["EURUSD"], {"EURUSD": [0, 1]})
    monkeypatch.setattr("model_core.walk_forward.formula_warmup_bars", lambda length: 4)
    with patch(
        "model_core.features.MT5FeatureEngineer.compute_features",
        return_value=torch.zeros(1, 1, 3),
    ):
        actual = runner._compute_targets()
    torch.testing.assert_close(actual, torch.zeros(1))
    runner.vm.execute.assert_not_called()
    assert runner.target_reasons["EURUSD"] == "insufficient history: actual=3 required=4"


def test_exact_warmup_boundary_invokes_vm_and_shared_mapping(monkeypatch):
    runner = _target_runner(["EURUSD"], {"EURUSD": [0, 1]})
    factor = torch.tensor([[0.5]], dtype=torch.float32)
    runner.vm.execute.return_value = factor
    monkeypatch.setattr("model_core.walk_forward.formula_warmup_bars", lambda length: 4)
    with patch(
        "model_core.features.MT5FeatureEngineer.compute_features",
        return_value=torch.zeros(1, 1, 4),
    ):
        actual = runner._compute_targets()
    expected = factor_to_position(
        factor, min_exposure=float(runner_module.Config.MIN_TRADE_EXPOSURE)
    ).flatten()
    torch.testing.assert_close(actual, expected)
    runner.vm.execute.assert_called_once_with([0, 1], ANY)
    assert "EURUSD" not in runner.target_reasons


def test_insufficient_symbol_does_not_block_independent_valid_symbol(monkeypatch):
    runner = _target_runner(
        ["EURUSD", "GBPUSD"], {"EURUSD": [0, 1], "GBPUSD": [0]}
    )
    runner.vm.execute.return_value = torch.tensor([[1.0]], dtype=torch.float32)
    monkeypatch.setattr(
        "model_core.walk_forward.formula_warmup_bars",
        lambda length: 4 if length == 2 else 3,
    )
    with patch(
        "model_core.features.MT5FeatureEngineer.compute_features",
        return_value=torch.zeros(2, 1, 3),
    ):
        actual = runner._compute_targets()
    expected_valid = factor_to_position(
        torch.tensor([[1.0]]),
        min_exposure=float(runner_module.Config.MIN_TRADE_EXPOSURE),
    ).item()
    torch.testing.assert_close(actual, torch.tensor([0.0, expected_valid]))
    runner.vm.execute.assert_called_once_with([0], ANY)
    assert runner.target_reasons["EURUSD"] == "insufficient history: actual=3 required=4"


def test_shutdown_calls_mt5_shutdown():
    runner = MT5StrategyRunner.__new__(MT5StrategyRunner)
    runner.formula = [1, 2, 3]
    runner._fetcher = None
    mock_mt5 = MagicMock()
    with patch.object(runner_module, "mt5", mock_mt5):
        runner.shutdown()
    mock_mt5.shutdown.assert_called_once()
