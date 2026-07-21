from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import pytest
import torch

import model_core.engine as engine_module
import model_core.walk_forward as walk_forward_module
import training_service
from data_pipeline.data_manager import compute_forward_open_returns
from data_pipeline.parquet_manager import ParquetDataManager
from model_core.artifacts import StrategyArtifact, TrainingRunIdentity
from model_core.config import ModelConfig
from model_core.engine import AlphaEngine
from model_core.features import FEATURE_REGISTRY, MT5FeatureEngineer
from model_core.ops import OPERATOR_REGISTRY
from model_core.registry import Registry
from model_core.semantics import ArtifactCompatibilityError, BacktestModeError
from model_core.vm import StackVM
from run_backtest import load_strategy, run_backtest


_COMMISSION_PCT = 0.02
_SLIPPAGE_PCT = 0.01
_COST_RATE = (_COMMISSION_PCT + _SLIPPAGE_PCT) / 100.0


def _frame(start: pd.Timestamp, bars: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    log_open = np.log(100.0) + np.cumsum(rng.normal(0.0, 0.0015, bars))
    open_price = np.exp(log_open).astype(np.float32)
    close = (
        open_price * np.exp(rng.normal(0.0, 0.0008, bars))
    ).astype(np.float32)
    high = (
        np.maximum(open_price, close)
        * (1.0 + rng.uniform(0.0002, 0.0012, bars))
    ).astype(np.float32)
    low = (
        np.minimum(open_price, close)
        * (1.0 - rng.uniform(0.0002, 0.0012, bars))
    ).astype(np.float32)
    return pd.DataFrame(
        {
            "time": pd.date_range(start, periods=bars, freq="1h", tz="UTC"),
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": rng.integers(100, 10_000, bars).astype(np.float32),
        }
    )


def _write(path: Path, frame: pd.DataFrame) -> ParquetDataManager:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    manager = ParquetDataManager(path)
    manager.load()
    return manager


def _test_registry_warmup(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = Registry()
    for spec in FEATURE_REGISTRY.feature_specs:
        registry.register_feature(replace(spec, lookback=min(spec.lookback, 20)))
    for spec in OPERATOR_REGISTRY.operator_specs:
        registry.register_operator(replace(spec, lookback=min(spec.lookback, 5)))
    registry.freeze()
    monkeypatch.setattr(
        walk_forward_module,
        "MAX_FEATURE_LOOKBACK",
        max(spec.lookback for spec in registry.feature_specs),
    )
    monkeypatch.setattr(
        walk_forward_module,
        "MAX_OPERATOR_LOOKBACK",
        max(spec.lookback for spec in registry.operator_specs),
    )


def _bytes(paths: list[Path]) -> dict[Path, bytes]:
    return {path: path.read_bytes() for path in paths}


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _assert_final_liquidation(report: dict[str, object]) -> None:
    ledger = report["ledger"]
    assert isinstance(ledger, list) and len(ledger) >= 2
    previous = ledger[-2]
    final = ledger[-1]
    assert final["is_final_liquidation"] is True
    cost_rate = float(report["cost"]["cost_rate"])
    expected_cost = (
        abs(float(final["position"]) - float(previous["position"]))
        + abs(float(final["position"]))
    ) * cost_rate
    assert float(final["cost"]) == pytest.approx(
        expected_cost, rel=0.0, abs=1e-12
    )
    assert float(final["net_pnl"]) == pytest.approx(
        math.log1p(float(final["gross_pnl"]) - float(final["cost"])),
        rel=0.0,
        abs=1e-12,
    )


def _assert_ledger_matches_inputs(
    report: dict[str, object],
    manager: ParquetDataManager,
    strategy: StrategyArtifact,
) -> None:
    raw = manager.raw_dict
    factors = StackVM().execute(list(strategy.formula_tokens), manager.feat_tensor)
    assert factors is not None
    target_ret, target_valid = compute_forward_open_returns(raw["open"])
    min_exposure = float(
        strategy.run_identity.artifact_identity.training_config["neutral_band"]
    )
    positions = torch.tanh(factors)
    positions = torch.where(
        positions.abs() < min_exposure,
        torch.zeros_like(positions),
        positions,
    )
    previous_positions = torch.cat(
        [torch.zeros_like(positions[:, :1]), positions[:, :-1]], dim=1
    )
    turnovers = (positions - previous_positions).abs()

    assert report["formula_tokens"] == list(strategy.formula_tokens)
    assert report["min_exposure"] == min_exposure
    assert report["cost"] == {
        "commission_pct": _COMMISSION_PCT,
        "slippage_pct": _SLIPPAGE_PCT,
        "cost_rate": _COST_RATE,
        "total_cost_rate": _COST_RATE,
        "unit": "equity_fraction_simple_return",
    }
    assert target_valid.shape == factors.shape == positions.shape
    assert raw["time"].shape == factors.shape

    valid_count = int(target_valid[0].sum().item())
    ledger = report["ledger"]
    assert isinstance(ledger, list)
    assert len(ledger) == valid_count
    symbol = manager.symbols[0]
    times = raw["time"][0]
    for index, row in enumerate(ledger):
        position = float(positions[0, index].item())
        turnover = float(turnovers[0, index].item())
        is_final = index == valid_count - 1
        liquidation = abs(position) if is_final else 0.0
        asset_simple_return = math.expm1(float(target_ret[0, index].item()))
        gross_pnl = position * asset_simple_return
        cost = (turnover + liquidation) * _COST_RATE
        net_pnl = math.log1p(gross_pnl - cost)

        assert row["symbol"] == symbol
        assert row["signal_time_ns"] == int(times[index].item())
        assert row["entry_time_ns"] == int(times[index + 1].item())
        assert row["exit_time_ns"] == int(times[index + 2].item())
        assert row["position"] == pytest.approx(position, rel=0.0, abs=1e-12)
        assert row["gross_pnl"] == pytest.approx(gross_pnl, rel=0.0, abs=1e-12)
        assert row["cost"] == pytest.approx(cost, rel=0.0, abs=1e-12)
        assert row["net_pnl"] == pytest.approx(net_pnl, rel=0.0, abs=1e-12)
        assert row["is_final_liquidation"] is is_final


@pytest.fixture(scope="module")
def lifecycle(tmp_path_factory, request: pytest.FixtureRequest):
    root = tmp_path_factory.mktemp("core-correctness-v2")
    patch = pytest.MonkeyPatch()
    request.addfinalizer(patch.undo)
    patch.chdir(root)
    patch.setattr(ModelConfig, "BATCH_SIZE", 4)
    patch.setattr(ModelConfig, "TRAIN_STEPS", 4)
    patch.setattr(ModelConfig, "MAX_FORMULA_LEN", 3)
    patch.setattr(ModelConfig, "WF_N_BLOCKS", 3)
    patch.setattr(ModelConfig, "WF_MIN_FOLD_BARS", 200)
    patch.setattr(ModelConfig, "WF_GAP", 2)
    _test_registry_warmup(patch)

    checkpoint_dir = root / "checkpoints"
    strategy_dir = root / "strategies"
    patch.setattr(engine_module, "_CHECKPOINT_DIR", checkpoint_dir)
    patch.setattr(training_service, "CHECKPOINT_DIR", checkpoint_dir)
    patch.setattr(training_service, "STRATEGY_DIR", strategy_dir)
    checkpoint_dir.mkdir()
    strategy_dir.mkdir()

    training = _frame(pd.Timestamp("2025-01-01"), 1_700, seed=20260719)
    future_start = training["time"].iloc[-1] + pd.Timedelta(hours=1)
    future = _frame(future_start, 400, seed=20260720)
    overlap = pd.concat(
        [training.iloc[-100:].copy(), future.iloc[:300].copy()], ignore_index=True
    )

    train_path = root / "data" / "EURUSD_H1.parquet"
    copy_path = root / "copy" / "EURUSD_H1.parquet"
    oos_path = root / "oos" / "EURUSD_H1.parquet"
    overlap_path = root / "overlap" / "EURUSD_H1.parquet"
    manager = _write(train_path, training)
    _write(copy_path, training.copy(deep=True))
    oos_manager = _write(oos_path, future)
    _write(overlap_path, overlap)

    legacy_strategy = strategy_dir / "best_EURUSD.json"
    legacy_strategy.write_text(json.dumps([0, 1, 2]), encoding="utf-8")
    legacy_checkpoint = checkpoint_dir / "ckpt_EURUSD.pt"
    torch.save(
        {
            "step": 7,
            "model_state_dict": {},
            "optimizer_state_dict": {},
        },
        legacy_checkpoint,
    )
    legacy_history = root / "training_history_EURUSD.json"
    legacy_history.write_text(
        json.dumps({"step": [0, 1], "best_score": [0.1, 0.2]}),
        encoding="utf-8",
    )
    legacy_report = root / "reports" / "report_v1.json"
    legacy_report.parent.mkdir()
    legacy_report.write_text(
        json.dumps({"symbol": "EURUSD", "total_return": 0.1}),
        encoding="utf-8",
    )
    legacy_package = root / "packages" / "package_v1.zip"
    legacy_package.parent.mkdir()
    with zipfile.ZipFile(legacy_package, "w") as archive:
        archive.writestr("manifest.json", json.dumps({"schema_version": "v1"}))
    preserved = [
        legacy_strategy,
        legacy_checkpoint,
        legacy_history,
        legacy_report,
        legacy_package,
    ]
    old_bytes = _bytes(preserved)

    real_train = AlphaEngine.train

    def train_exactly_two_steps(engine: AlphaEngine, *args, **kwargs):
        assert kwargs.get("start_step", 0) == 0
        result = real_train(
            engine,
            start_step=0,
            end_step=2,
            verbose_header=False,
        )
        engine.best_formula = None
        return result

    patch.setattr(AlphaEngine, "train", train_exactly_two_steps)
    partial = training_service.run_training_session(
        manager,
        source_path=train_path,
        from_scratch=True,
        random_seed=123,
    )
    partial_run_identity = partial.run_identity
    first_checkpoints = sorted(checkpoint_dir.glob("ckpt_v3_*.pt"))
    assert len(first_checkpoints) == 1
    first_checkpoint = first_checkpoints[0]
    checkpoint_payload = torch.load(
        first_checkpoint, map_location="cpu", weights_only=False
    )
    checkpoint_run_identity = TrainingRunIdentity.from_dict(
        checkpoint_payload["run_identity"]
    )

    patch.setattr(AlphaEngine, "train", real_train)
    resumed = training_service.run_training_session(
        manager,
        source_path=train_path,
        from_scratch=False,
        random_seed=123,
    )
    strategy_paths = sorted(strategy_dir.glob("best_v3_EURUSD_H1_*.json"))
    assert len(strategy_paths) == 1
    strategy_path = strategy_paths[0]
    strategy = load_strategy(strategy_path)

    replay_path = run_backtest(
        strategy_file=strategy_path,
        data_file=train_path,
        mode="in_sample_replay",
        commission=_COMMISSION_PCT,
        slippage=_SLIPPAGE_PCT,
        output_dir=root / "replay",
    )
    oos_report_path = run_backtest(
        strategy_file=strategy_path,
        data_file=oos_path,
        mode="out_of_sample_backtest",
        commission=_COMMISSION_PCT,
        slippage=_SLIPPAGE_PCT,
        output_dir=root / "oos-report",
    )
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    oos_report = json.loads(oos_report_path.read_text(encoding="utf-8"))

    yield {
        "root": root,
        "training": training,
        "train_path": train_path,
        "copy_path": copy_path,
        "oos_path": oos_path,
        "overlap_path": overlap_path,
        "manager": manager,
        "oos_manager": oos_manager,
        "partial": partial,
        "resumed": resumed,
        "partial_run_identity": partial_run_identity,
        "checkpoint_payload": checkpoint_payload,
        "checkpoint_run_identity": checkpoint_run_identity,
        "checkpoint_paths": sorted(checkpoint_dir.glob("ckpt_v3_*.pt")),
        "strategy_path": strategy_path,
        "strategy": strategy,
        "replay_path": replay_path,
        "oos_report_path": oos_report_path,
        "replay": replay,
        "oos_report": oos_report,
        "preserved": preserved,
        "old_bytes": old_bytes,
        "legacy_strategy": legacy_strategy,
        "legacy_checkpoint": legacy_checkpoint,
        "legacy_history": legacy_history,
        "legacy_report": legacy_report,
        "legacy_package": legacy_package,
    }


def test_full_v2_lifecycle_resumes_identity_and_reconciles_reports(lifecycle) -> None:
    partial = lifecycle["partial"]
    resumed = lifecycle["resumed"]
    checkpoint = lifecycle["checkpoint_payload"]
    checkpoint_identity = lifecycle["checkpoint_run_identity"]
    strategy: StrategyArtifact = lifecycle["strategy"]
    artifact_identity = strategy.run_identity.artifact_identity

    assert partial is not resumed
    assert partial.training_history["step"] == [0, 1]
    assert resumed.training_history["step"] == [0, 1, 2, 3]
    assert checkpoint["checkpoint_schema_version"] == "checkpoint-v3"
    assert checkpoint["step"] == 1
    assert checkpoint_identity == lifecycle["partial_run_identity"]
    assert resumed.run_identity == checkpoint_identity == strategy.run_identity
    assert checkpoint_identity.artifact_identity == artifact_identity
    assert StrategyArtifact.from_dict(strategy.to_dict()) == strategy
    assert strategy.fingerprint == strategy.to_dict()["fingerprint"]

    assert strategy.fold_evidence
    for fold in strategy.fold_evidence:
        assert fold.train_start_time_ns < fold.train_end_time_ns
        assert fold.train_end_time_ns < fold.val_start_time_ns
        assert fold.val_start_time_ns < fold.val_end_time_ns
        assert fold.effective_gap == 2

    replay = lifecycle["replay"]
    oos_report = lifecycle["oos_report"]
    assert replay["report_schema"] == "backtest-report-v3"
    assert replay["mode"] == "in_sample_replay"
    assert replay["mode_label"] == "样本内复盘"
    assert oos_report["report_schema"] == "backtest-report-v3"
    assert oos_report["mode"] == "out_of_sample_backtest"
    assert oos_report["mode_label"] == "独立样本外回测"
    training_identity = artifact_identity.training_dataset
    expected_versions = {
        "artifact_schema": strategy.schema_version,
        "strategy_schema": strategy.schema_version,
        "core_semantics": artifact_identity.core_semantics_version,
        "vocab": artifact_identity.vocab_version,
        "label_semantics": artifact_identity.label_semantics_version,
        "execution_semantics": artifact_identity.execution_semantics_version,
    }
    for report, manager, test_identity in (
        (
            replay,
            lifecycle["manager"],
            lifecycle["manager"].data_identities[0],
        ),
        (
            oos_report,
            lifecycle["oos_manager"],
            lifecycle["oos_manager"].data_identities[0],
        ),
    ):
        assert report["symbol"] == artifact_identity.symbol
        assert report["timeframe"] == artifact_identity.timeframe
        assert report["strategy_fingerprint"] == strategy.fingerprint
        assert report["artifact_fingerprint"] == artifact_identity.fingerprint
        assert report["versions"] == expected_versions
        assert report["training_dataset"] == training_identity.to_dict()
        assert report["test_dataset"] == test_identity.to_dict()
        assert report["training_dataset"]["data_fingerprint"] == (
            checkpoint_identity.artifact_identity.training_dataset.data_fingerprint
        )
        assert report["test_dataset"]["data_fingerprint"] == (
            test_identity.data_fingerprint
        )
        assert report["training_start_time_ns"] == training_identity.start_time_ns
        assert report["training_end_time_ns"] == training_identity.end_time_ns
        assert report["test_start_time_ns"] == test_identity.start_time_ns
        assert report["test_end_time_ns"] == test_identity.end_time_ns

        _assert_ledger_matches_inputs(report, manager, strategy)

        reconciliation = report["ledger_reconciliation"]
        ledger_total = math.fsum(float(row["net_pnl"]) for row in report["ledger"])
        execution_total = float(reconciliation["execution_net_pnl"])
        difference = abs(ledger_total - execution_total)
        assert ledger_total == pytest.approx(
            reconciliation["ledger_net_pnl"], abs=1e-8
        )
        assert ledger_total == pytest.approx(execution_total, abs=1e-8)
        assert reconciliation["absolute_difference"] == pytest.approx(
            difference, rel=0.0, abs=1e-15
        )
        assert reconciliation["tolerance"] == 1e-8
        assert reconciliation["reconciled"] is (difference <= 1e-8)
        _assert_final_liquidation(report)


def test_illegal_oos_is_fail_closed_and_preserves_all_bytes(lifecycle) -> None:
    protected = [
        lifecycle["train_path"],
        lifecycle["copy_path"],
        lifecycle["overlap_path"],
        lifecycle["strategy_path"],
        *lifecycle["checkpoint_paths"],
        *lifecycle["preserved"],
    ]
    before = _bytes(protected)

    tree_before = _tree_bytes(lifecycle["root"])
    with pytest.raises(ArtifactCompatibilityError):
        load_strategy(lifecycle["legacy_strategy"])
    assert _tree_bytes(lifecycle["root"]) == tree_before

    tree_before = _tree_bytes(lifecycle["root"])
    with pytest.raises(ArtifactCompatibilityError):
        lifecycle["resumed"].load_checkpoint(str(lifecycle["legacy_checkpoint"]))
    assert _tree_bytes(lifecycle["root"]) == tree_before

    discovered_v2 = {
        *lifecycle["checkpoint_paths"],
        lifecycle["strategy_path"],
        lifecycle["replay_path"],
        lifecycle["oos_report_path"],
        *lifecycle["root"].glob("training_history_v3_*.json"),
    }
    assert set(lifecycle["preserved"]).isdisjoint(discovered_v2)
    assert all(path.name.startswith("ckpt_v3_") for path in lifecycle["checkpoint_paths"])
    assert lifecycle["strategy_path"].name.startswith("best_v3_")

    for label, data_path in (
        ("training", lifecycle["train_path"]),
        ("copy", lifecycle["copy_path"]),
        ("overlap", lifecycle["overlap_path"]),
    ):
        output = lifecycle["root"] / f"rejected-{label}"
        tree_before = _tree_bytes(lifecycle["root"])
        with pytest.raises(BacktestModeError):
            run_backtest(
                strategy_file=lifecycle["strategy_path"],
                data_file=data_path,
                mode="out_of_sample_backtest",
                output_dir=output,
            )
        assert not output.exists()
        assert _tree_bytes(lifecycle["root"]) == tree_before

    assert _bytes(protected) == before
    assert _bytes(lifecycle["preserved"]) == lifecycle["old_bytes"]


def test_future_tail_cannot_change_feature_or_selected_formula_prefix(
    lifecycle,
) -> None:
    changed = lifecycle["training"].copy(deep=True)
    cut = 1_200
    tail = changed.index > cut
    for column in ("open", "high", "low", "close"):
        changed[column] = changed[column].where(~tail, changed[column] * 1.07).astype(
            np.float32
        )
    changed["volume"] = changed["volume"].where(
        ~tail, changed["volume"] * 1.31
    ).astype(np.float32)
    changed_manager = _write(
        lifecycle["root"] / "changed" / "EURUSD_H1.parquet", changed
    )

    original_features = MT5FeatureEngineer.compute_features(
        lifecycle["manager"].raw_dict
    )
    changed_features = MT5FeatureEngineer.compute_features(changed_manager.raw_dict)
    assert torch.equal(
        original_features[..., : cut + 1], changed_features[..., : cut + 1]
    )
    feature_tail_difference = (
        original_features[..., cut + 1 :] - changed_features[..., cut + 1 :]
    ).abs()
    assert torch.count_nonzero(feature_tail_difference).item() > 0
    assert feature_tail_difference.max().item() > 1e-4

    formula = list(lifecycle["strategy"].formula_tokens)
    original_factor = StackVM().execute(formula, original_features)
    changed_factor = StackVM().execute(formula, changed_features)
    assert original_factor is not None
    assert changed_factor is not None
    torch.testing.assert_close(
        original_factor[..., : cut + 1],
        changed_factor[..., : cut + 1],
        rtol=0.0,
        atol=1e-6,
    )
    factor_tail_difference = (
        original_factor[..., cut + 1 :] - changed_factor[..., cut + 1 :]
    ).abs()
    assert torch.count_nonzero(factor_tail_difference).item() > 0
    assert factor_tail_difference.max().item() > 1e-6
