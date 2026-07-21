from __future__ import annotations

import json
import builtins
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import run_backtest as run_backtest_module

from data_pipeline.parquet_manager import ParquetDataManager
from model_core.artifacts import (
    ArtifactIdentity,
    FoldEvidence,
    StrategyArtifact,
    TrainingRunIdentity,
    sha256_json,
)
from model_core.semantics import (
    CORE_SEMANTICS_VERSION,
    EXECUTION_SEMANTICS_VERSION,
    LABEL_SEMANTICS_VERSION,
    BacktestModeError,
)
from model_core.vocab import FORMULA_VOCAB, VOCAB_VERSION
from run_backtest import run_backtest
from tests.unit.test_backtest_modes import training_config


def _frame(start: str, *, bars: int = 80) -> pd.DataFrame:
    x = np.arange(bars, dtype=np.float64)
    base = np.asarray(
        100.0 + 0.025 * x + 1.8 * np.sin(x / 2.7) + 0.7 * np.cos(x / 5.1),
        dtype=np.float32,
    )
    close = np.asarray(base + 0.2 * np.sin(x / 1.9), dtype=np.float32)
    return pd.DataFrame(
        {
            "time": pd.date_range(start, periods=bars, freq="4h", tz="UTC"),
            "open": base,
            "high": np.asarray(np.maximum(base, close) + 0.5, dtype=np.float32),
            "low": np.asarray(np.minimum(base, close) - 0.5, dtype=np.float32),
            "close": close,
            "volume": np.asarray(1000.0 + x, dtype=np.float32),
        }
    )


def _write_dataset(path: Path, frame: pd.DataFrame):
    frame.to_parquet(path, index=False)
    manager = ParquetDataManager(path)
    manager.load()
    return manager.data_identities[0]


def _strategy_for(training_identity, path: Path) -> StrategyArtifact:
    config = training_config()
    config["timeframe_reward"] = {
        "timeframe": training_identity.timeframe,
        "target_trades_per_day": 2.0,
        "target_bars_per_trade": 3.0,
    }
    identity = ArtifactIdentity(
        core_semantics_version=CORE_SEMANTICS_VERSION,
        vocab_version=VOCAB_VERSION,
        label_semantics_version=LABEL_SEMANTICS_VERSION,
        execution_semantics_version=EXECUTION_SEMANTICS_VERSION,
        symbol=training_identity.symbol,
        timeframe=training_identity.timeframe,
        training_dataset=training_identity,
        training_config=config,
        training_config_hash=sha256_json(config),
    )
    start = training_identity.start_time_ns
    span = training_identity.end_time_ns - start
    artifact = StrategyArtifact.create(
        run_identity=TrainingRunIdentity.create(identity),
        formula_tokens=[0],
        decoded_formula=FORMULA_VOCAB.token_names[0],
        best_score=1.0,
        fold_evidence=[
            FoldEvidence(
                fold_index=index,
                train_start_time_ns=start,
                train_end_time_ns=start + span * (index + 2) // 6,
                val_start_time_ns=start + span * (index + 2) // 6 + 1,
                val_end_time_ns=start + span * (index + 3) // 6,
                effective_gap=2,
                validation_metrics={"ic": 0.1, "net_return": 0.01},
            )
            for index in range(4)
        ],
        generated_at="2026-07-18T00:00:00Z",
    )
    path.write_text(json.dumps(artifact.to_dict()), encoding="utf-8")
    return artifact


@pytest.fixture
def lifecycle(tmp_path):
    train_path = tmp_path / "EURUSD_H4.parquet"
    oos_dir = tmp_path / "oos"
    oos_dir.mkdir()
    oos_path = oos_dir / "EURUSD_H4.parquet"
    overlap_dir = tmp_path / "overlap"
    overlap_dir.mkdir()
    overlap_path = overlap_dir / "EURUSD_H4.parquet"
    other_dir = tmp_path / "other"
    other_dir.mkdir()
    other_path = other_dir / "GBPUSD_H4.parquet"

    train_frame = _frame("2025-01-01")
    train_identity = _write_dataset(train_path, train_frame)
    future_start = train_frame["time"].iloc[-1] + pd.Timedelta(hours=4)
    oos_identity = _write_dataset(oos_path, _frame(str(future_start)))
    overlap = pd.concat(
        [train_frame.iloc[-8:], _frame(str(future_start), bars=40)], ignore_index=True
    )
    _write_dataset(overlap_path, overlap)
    _write_dataset(other_path, train_frame)
    strategy_path = tmp_path / "strategy.json"
    artifact = _strategy_for(train_identity, strategy_path)
    return {
        "strategy_path": strategy_path,
        "artifact": artifact,
        "train_path": train_path,
        "train_identity": train_identity,
        "oos_path": oos_path,
        "oos_identity": oos_identity,
        "overlap_path": overlap_path,
        "other_path": other_path,
        "tmp_path": tmp_path,
    }


def test_replay_report_has_complete_identity_cost_metrics_and_name(lifecycle) -> None:
    output = lifecycle["tmp_path"] / "replay-report"
    report_path = run_backtest(
        strategy_file=lifecycle["strategy_path"],
        data_file=lifecycle["train_path"],
        mode="in_sample_replay",
        commission=0.02,
        slippage=0.01,
        output_dir=output,
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert set(report) >= {
        "report_schema",
        "mode",
        "mode_label",
        "strategy_fingerprint",
        "artifact_fingerprint",
        "versions",
        "training_dataset",
        "test_dataset",
        "training_start_time_ns",
        "training_end_time_ns",
        "test_start_time_ns",
        "test_end_time_ns",
        "cost",
        "return_accounting",
        "trade_statistics",
        "min_exposure",
        "metrics",
        "ledger",
        "ledger_reconciliation",
        "generated_at",
    }
    assert report["report_schema"] == "backtest-report-v2"
    assert report["mode"] == "in_sample_replay"
    assert report["mode_label"] == "样本内复盘"
    assert report["strategy_fingerprint"] == lifecycle["artifact"].fingerprint
    assert report["training_dataset"] == lifecycle["train_identity"].to_dict()
    assert report["test_dataset"] == lifecycle["train_identity"].to_dict()
    assert report["cost"]["cost_rate"] == pytest.approx(0.0003)
    assert report["cost"]["unit"] == "equity_fraction_simple_return"
    assert report["return_accounting"]["net_fact"].startswith("net_log_return=log1p")
    assert report["trade_statistics"]["n_trades_definition"] == (
        "display_trades=entries+reversals"
    )
    assert set(report["metrics"]) >= {"periods_per_year", "sharpe", "sortino"}
    assert report["ledger_reconciliation"]["absolute_difference"] <= 1e-8
    assert "in_sample_replay_EURUSD" in report_path.name
    assert lifecycle["train_identity"].data_fingerprint[:12] in report_path.name
    assert lifecycle["artifact"].fingerprint[:12] in report_path.name
    assert not list(output.glob("*.tmp"))


def test_strict_future_oos_report_succeeds_with_explicit_identity(lifecycle) -> None:
    report_path = run_backtest(
        strategy_file=lifecycle["strategy_path"],
        data_file=lifecycle["oos_path"],
        mode="out_of_sample_backtest",
        output_dir=lifecycle["tmp_path"] / "oos-report",
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["mode"] == "out_of_sample_backtest"
    assert report["mode_label"] == "独立样本外回测"
    assert report["test_dataset"] == lifecycle["oos_identity"].to_dict()
    assert report["ledger_reconciliation"]["absolute_difference"] <= 1e-8


@pytest.mark.parametrize("dataset_key", ["train_path", "overlap_path"])
def test_invalid_oos_generates_no_report(lifecycle, dataset_key) -> None:
    output = lifecycle["tmp_path"] / f"rejected-{dataset_key}"
    with pytest.raises(BacktestModeError):
        run_backtest(
            strategy_file=lifecycle["strategy_path"],
            data_file=lifecycle[dataset_key],
            mode="out_of_sample_backtest",
            output_dir=output,
        )
    assert not output.exists()


def test_symbol_mismatch_is_rejected_without_remapping_or_report(lifecycle) -> None:
    output = lifecycle["tmp_path"] / "symbol-rejected"
    with pytest.raises(BacktestModeError, match="symbol mismatch"):
        run_backtest(
            strategy_file=lifecycle["strategy_path"],
            data_file=lifecycle["other_path"],
            mode="in_sample_replay",
            output_dir=output,
        )
    assert not output.exists()


def _output_snapshot(path: Path) -> dict[str, bytes]:
    if not path.exists():
        return {}
    return {
        item.relative_to(path).as_posix(): item.read_bytes()
        for item in sorted(path.rglob("*"))
        if item.is_file()
    }


def _seed_existing_output(lifecycle, output: Path) -> None:
    output.mkdir()
    stem = (
        "backtest_v2_in_sample_replay_EURUSD_"
        f"{lifecycle['train_identity'].data_fingerprint[:12]}_"
        f"{lifecycle['artifact'].fingerprint[:12]}"
    )
    (output / f"{stem}.json").write_bytes(b"pre-existing-report")
    (output / f"{stem}.png").write_bytes(b"pre-existing-chart")
    (output / "unrelated-legacy.bin").write_bytes(b"do-not-touch")


def test_chart_failure_preserves_exact_output_snapshot_and_leaves_no_temp(
    lifecycle, monkeypatch
) -> None:
    output = lifecycle["tmp_path"] / "chart-failure"
    _seed_existing_output(lifecycle, output)
    before = _output_snapshot(output)

    def fail_chart(*args, **kwargs):
        raise RuntimeError("forced chart failure")

    monkeypatch.setattr(run_backtest_module, "_write_equity_chart", fail_chart)
    with pytest.raises(RuntimeError, match="forced chart failure"):
        run_backtest_module.run_backtest(
            strategy_file=lifecycle["strategy_path"],
            data_file=lifecycle["train_path"],
            mode="in_sample_replay",
            output_dir=output,
        )

    assert _output_snapshot(output) == before
    assert not list(output.glob(".*.tmp"))


def test_report_replace_failure_removes_staged_chart_and_preserves_existing_bytes(
    lifecycle, monkeypatch
) -> None:
    output = lifecycle["tmp_path"] / "report-failure"
    output.mkdir()
    (output / "unrelated-legacy.bin").write_bytes(b"do-not-touch")
    before = _output_snapshot(output)
    real_publish = run_backtest_module._publish_no_replace
    real_chart = run_backtest_module._write_equity_chart
    chart_calls = 0

    def counted_chart(*args, **kwargs):
        nonlocal chart_calls
        chart_calls += 1
        return real_chart(*args, **kwargs)

    def fail_report_replace(source, destination):
        if Path(destination).suffix == ".json":
            raise OSError("forced report replace failure")
        return real_publish(source, destination)

    monkeypatch.setattr(run_backtest_module, "_write_equity_chart", counted_chart)
    monkeypatch.setattr(
        run_backtest_module, "_publish_no_replace", fail_report_replace
    )
    with pytest.raises(OSError, match="forced report replace failure"):
        run_backtest_module.run_backtest(
            strategy_file=lifecycle["strategy_path"],
            data_file=lifecycle["train_path"],
            mode="in_sample_replay",
            output_dir=output,
        )

    assert chart_calls == 1
    assert _output_snapshot(output) == before
    assert not [name for name in _output_snapshot(output) if ".tmp" in name]


def test_report_write_failure_removes_partial_stage_and_preserves_output(
    lifecycle, monkeypatch
) -> None:
    output = lifecycle["tmp_path"] / "report-write-failure"
    _seed_existing_output(lifecycle, output)
    before = _output_snapshot(output)
    real_chart = run_backtest_module._write_equity_chart
    chart_calls = 0

    def counted_chart(*args, **kwargs):
        nonlocal chart_calls
        chart_calls += 1
        return real_chart(*args, **kwargs)

    def fail_report_write(report, path):
        Path(path).write_bytes(b"partial-report-stage")
        raise OSError("forced report write failure")

    monkeypatch.setattr(run_backtest_module, "_write_equity_chart", counted_chart)
    monkeypatch.setattr(run_backtest_module, "_write_report_atomic", fail_report_write)
    with pytest.raises(OSError, match="forced report write failure"):
        run_backtest_module.run_backtest(
            strategy_file=lifecycle["strategy_path"],
            data_file=lifecycle["train_path"],
            mode="in_sample_replay",
            output_dir=output,
        )

    assert chart_calls == 1
    assert _output_snapshot(output) == before


def test_chart_publish_and_cleanup_failures_restore_finals_before_cleanup(
    lifecycle, monkeypatch
) -> None:
    output = lifecycle["tmp_path"] / "combined-failure"
    output.mkdir()
    (output / "unrelated-legacy.bin").write_bytes(b"do-not-touch")
    before = _output_snapshot(output)
    real_publish = run_backtest_module._publish_no_replace
    real_unlink = Path.unlink

    def fail_chart_publish(source, destination):
        if Path(destination).suffix == ".png":
            raise OSError("forced chart publication failure")
        return real_publish(source, destination)

    def fail_old_stage_unlink(path, *args, **kwargs):
        if ".stage" in path.name:
            raise OSError("forced stage cleanup failure")
        return real_unlink(path, *args, **kwargs)

    def fail_external_cleanup(*args, **kwargs):
        raise OSError("forced stage cleanup failure")

    monkeypatch.setattr(run_backtest_module, "_publish_no_replace", fail_chart_publish)
    monkeypatch.setattr(Path, "unlink", fail_old_stage_unlink)
    monkeypatch.setattr(
        run_backtest_module, "_delete_owned_transaction_tree", fail_external_cleanup
    )

    with pytest.raises(OSError, match="forced chart publication failure") as raised:
        run_backtest_module.run_backtest(
            strategy_file=lifecycle["strategy_path"],
            data_file=lifecycle["train_path"],
            mode="in_sample_replay",
            output_dir=output,
        )

    assert isinstance(raised.value.__cause__, OSError)
    assert "forced stage cleanup failure" in str(raised.value.__cause__)
    assert _output_snapshot(output) == before


def test_deterministic_stage_name_collision_never_mutates_unowned_files(
    lifecycle, monkeypatch
) -> None:
    output = lifecycle["tmp_path"] / "collision-failure"
    _seed_existing_output(lifecycle, output)
    collision_hex = "c" * 32
    stem = (
        "backtest_v2_in_sample_replay_EURUSD_"
        f"{lifecycle['train_identity'].data_fingerprint[:12]}_"
        f"{lifecycle['artifact'].fingerprint[:12]}"
    )
    (output / f".{stem}.{collision_hex}.report.stage").write_bytes(
        b"pre-existing-report-collision"
    )
    (output / f".{stem}.{collision_hex}.chart.stage").write_bytes(
        b"pre-existing-chart-collision"
    )
    before = _output_snapshot(output)

    class FixedUuid:
        hex = collision_hex

    monkeypatch.setattr(run_backtest_module, "uuid4", lambda: FixedUuid())
    monkeypatch.setattr(
        run_backtest_module,
        "_write_equity_chart",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            OSError("forced collision transaction failure")
        ),
    )

    with pytest.raises(OSError, match="forced collision transaction failure"):
        run_backtest_module.run_backtest(
            strategy_file=lifecycle["strategy_path"],
            data_file=lifecycle["train_path"],
            mode="in_sample_replay",
            output_dir=output,
        )

    assert _output_snapshot(output) == before


def test_cleanup_failure_after_commit_keeps_new_complete_pair(
    lifecycle, monkeypatch
) -> None:
    output = lifecycle["tmp_path"] / "success-cleanup-failure"
    output.mkdir()
    (output / "unrelated-legacy.bin").write_bytes(b"do-not-touch")
    before = _output_snapshot(output)

    def partially_remove_stages_then_fail(path, *args, **kwargs):
        transaction_root = Path(path)
        for stage_name in ("report.stage", "chart.stage"):
            stage = transaction_root / stage_name
            if stage.exists():
                stage.unlink()
        raise OSError("forced post-publication cleanup failure")

    monkeypatch.setattr(
        run_backtest_module,
        "_delete_owned_transaction_tree",
        partially_remove_stages_then_fail,
    )
    with pytest.raises(OSError, match="forced post-publication cleanup failure"):
        run_backtest_module.run_backtest(
            strategy_file=lifecycle["strategy_path"],
            data_file=lifecycle["train_path"],
            mode="in_sample_replay",
            output_dir=output,
        )

    after = _output_snapshot(output)
    assert set(after) == set(before) | {
        next(name for name in after if name.endswith(".json")),
        next(name for name in after if name.endswith(".png")),
    }
    assert after["unrelated-legacy.bin"] == before["unrelated-legacy.bin"]
    report_name = next(name for name in after if name.endswith(".json"))
    chart_name = next(name for name in after if name.endswith(".png"))
    assert after[report_name]
    assert after[chart_name].startswith(b"\x89PNG")


def test_publication_boundaries_contain_base_exceptions_for_safe_rollback(
    tmp_path, monkeypatch
) -> None:
    primary = KeyboardInterrupt("publication-boundary-primary")
    monkeypatch.setattr(
        run_backtest_module,
        "_write_owned_chart",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(primary),
    )
    raised = None
    try:
        _call_publication_transaction(tmp_path / "base-exception", monkeypatch)
    except BaseException as exc:
        raised = exc
    assert raised is primary
    assert not (tmp_path / "base-exception").exists()


def test_successful_rerun_is_identity_preserving_idempotent(lifecycle) -> None:
    output = lifecycle["tmp_path"] / "successful-rerun"
    output.mkdir()
    (output / "unrelated-legacy.bin").write_bytes(b"do-not-touch")
    unrelated_before = (output / "unrelated-legacy.bin").read_bytes()

    first_report = run_backtest_module.run_backtest(
        strategy_file=lifecycle["strategy_path"],
        data_file=lifecycle["train_path"],
        mode="in_sample_replay",
        output_dir=output,
    )
    first_identities = {
        path: (path.stat().st_dev, path.stat().st_ino, path.stat().st_mtime_ns)
        for path in (first_report, first_report.with_suffix(".png"))
    }
    second_report = run_backtest_module.run_backtest(
        strategy_file=lifecycle["strategy_path"],
        data_file=lifecycle["train_path"],
        mode="in_sample_replay",
        output_dir=output,
    )

    assert first_report == second_report
    for path, identity in first_identities.items():
        current = path.stat()
        assert (current.st_dev, current.st_ino, current.st_mtime_ns) == identity
    assert (output / "unrelated-legacy.bin").read_bytes() == unrelated_before
    assert set(_output_snapshot(output)) == {
        first_report.name,
        first_report.with_suffix(".png").name,
        "unrelated-legacy.bin",
    }


class ExactBaseException(BaseException):
    pass


def test_report_primary_error_survives_owned_temp_cleanup_failure_by_identity(
    lifecycle, monkeypatch
) -> None:
    output = lifecycle["tmp_path"] / "report-primary-cleanup-failure"
    _seed_existing_output(lifecycle, output)
    before = _output_snapshot(output)
    primary = OSError("PRIMARY report-stage replace failure")
    cleanup = OSError("CLEANUP owned-temp move failure")
    real_move = run_backtest_module._move_no_replace
    real_publish = run_backtest_module._publish_no_replace

    def fail_report(source, destination):
        if Path(destination).name == "report.stage":
            raise primary
        return real_publish(source, destination)

    def fail_cleanup(source, destination):
        if Path(source).name.endswith(".tmp") and Path(destination).name.endswith(
            ".cleanup"
        ):
            raise cleanup
        return real_move(source, destination)

    monkeypatch.setattr(run_backtest_module, "_publish_no_replace", fail_report)
    monkeypatch.setattr(run_backtest_module, "_move_no_replace", fail_cleanup)
    with pytest.raises(OSError) as raised:
        run_backtest_module.run_backtest(
            strategy_file=lifecycle["strategy_path"],
            data_file=lifecycle["train_path"],
            mode="in_sample_replay",
            output_dir=output,
        )

    assert raised.value is primary
    assert type(raised.value) is OSError
    assert raised.value.args == ("PRIMARY report-stage replace failure",)
    assert raised.value.__cause__ is cleanup
    assert _output_snapshot(output) == before


@pytest.mark.parametrize(
    "primary",
    [
        KeyboardInterrupt("exact-keyboard-interrupt"),
        SystemExit(59),
        GeneratorExit("exact-generator-exit"),
        ExactBaseException("exact-custom-base-exception"),
    ],
    ids=["keyboard-interrupt", "system-exit", "generator-exit", "custom-base"],
)
def test_report_base_exception_propagates_exactly_with_race_safe_cleanup(
    lifecycle, monkeypatch, primary
) -> None:
    output = lifecycle["tmp_path"] / f"base-{type(primary).__name__}"
    _seed_existing_output(lifecycle, output)
    before = _output_snapshot(output)
    cleanup_calls = 0
    real_move = run_backtest_module._move_no_replace
    real_publish = run_backtest_module._publish_no_replace

    def fail_report(source, destination):
        if Path(destination).name == "report.stage":
            raise primary
        return real_publish(source, destination)

    def fail_cleanup(source, destination):
        nonlocal cleanup_calls
        if Path(source).name.endswith(".tmp") and Path(destination).name.endswith(
            ".cleanup"
        ):
            cleanup_calls += 1
            raise OSError("race-safe cleanup failed")
        return real_move(source, destination)

    monkeypatch.setattr(run_backtest_module, "_publish_no_replace", fail_report)
    monkeypatch.setattr(run_backtest_module, "_move_no_replace", fail_cleanup)
    with pytest.raises(BaseException) as raised:
        run_backtest_module.run_backtest(
            strategy_file=lifecycle["strategy_path"],
            data_file=lifecycle["train_path"],
            mode="in_sample_replay",
            output_dir=output,
        )

    assert raised.value is primary
    assert cleanup_calls == 1
    assert primary.__cause__ is None
    assert BaseException.__getattribute__(primary, "__notes__") == [
        "secondary owned temporary cleanup failure"
    ]
    assert _output_snapshot(output) == before


def test_report_primary_error_with_successful_cleanup_keeps_exact_identity(
    lifecycle, monkeypatch
) -> None:
    output = lifecycle["tmp_path"] / "report-primary-clean-cleanup"
    _seed_existing_output(lifecycle, output)
    before = _output_snapshot(output)
    primary = OSError("PRIMARY with successful cleanup")
    real_publish = run_backtest_module._publish_no_replace

    def fail_report_stage_replace(source, destination):
        if Path(destination).name == "report.stage":
            raise primary
        return real_publish(source, destination)

    monkeypatch.setattr(
        run_backtest_module, "_publish_no_replace", fail_report_stage_replace
    )
    with pytest.raises(OSError) as raised:
        run_backtest_module.run_backtest(
            strategy_file=lifecycle["strategy_path"],
            data_file=lifecycle["train_path"],
            mode="in_sample_replay",
            output_dir=output,
        )

    assert raised.value is primary
    assert raised.value.__cause__ is None
    assert _output_snapshot(output) == before


def test_atomic_report_writer_has_no_finally_or_path_unlink(
    tmp_path, monkeypatch
) -> None:
    primary = OSError("report-publication-primary")
    destination = tmp_path / "report.stage"
    monkeypatch.setattr(
        run_backtest_module,
        "_publish_no_replace",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(primary),
    )
    with pytest.raises(OSError) as raised:
        run_backtest_module._write_report_atomic({"value": 1}, destination)
    assert raised.value is primary
    assert not destination.exists()
    assert not list(tmp_path.glob(".*.tmp"))


def _exception_graph_has_cycle(root: BaseException) -> bool:
    visiting: set[int] = set()
    complete: set[int] = set()

    def visit(node: BaseException | None) -> bool:
        if node is None:
            return False
        identity = id(node)
        if identity in visiting:
            return True
        if identity in complete:
            return False
        visiting.add(identity)
        if visit(node.__cause__) or visit(node.__context__):
            return True
        visiting.remove(identity)
        complete.add(identity)
        return False

    return visit(root)


def test_inner_same_cleanup_object_never_becomes_its_own_cause(
    lifecycle, monkeypatch
) -> None:
    output = lifecycle["tmp_path"] / "inner-self-cause"
    _seed_existing_output(lifecycle, output)
    before = _output_snapshot(output)
    primary = OSError("INNER SAME PRIMARY")
    real_move = run_backtest_module._move_no_replace
    real_publish = run_backtest_module._publish_no_replace

    def fail_report(source, destination):
        if Path(destination).name == "report.stage":
            raise primary
        return real_publish(source, destination)

    def fail_cleanup(source, destination):
        if Path(source).name.endswith(".tmp") and Path(destination).name.endswith(
            ".cleanup"
        ):
            raise primary
        return real_move(source, destination)

    monkeypatch.setattr(run_backtest_module, "_publish_no_replace", fail_report)
    monkeypatch.setattr(run_backtest_module, "_move_no_replace", fail_cleanup)
    with pytest.raises(OSError) as raised:
        run_backtest_module.run_backtest(
            strategy_file=lifecycle["strategy_path"],
            data_file=lifecycle["train_path"],
            mode="in_sample_replay",
            output_dir=output,
        )

    assert raised.value is primary
    assert primary.__cause__ is None
    assert not _exception_graph_has_cycle(primary)
    assert primary.__notes__ == ["secondary owned temporary cleanup failure"]
    assert _output_snapshot(output) == before


def test_inner_indirect_secondary_cycle_is_not_attached(lifecycle, monkeypatch) -> None:
    output = lifecycle["tmp_path"] / "inner-indirect-cycle"
    _seed_existing_output(lifecycle, output)
    before = _output_snapshot(output)
    primary = OSError("INNER INDIRECT PRIMARY")
    bridge = OSError("INNER BRIDGE")
    cleanup = OSError("INNER CYCLIC CLEANUP")
    bridge.__context__ = primary
    cleanup.__cause__ = bridge
    real_move = run_backtest_module._move_no_replace
    real_publish = run_backtest_module._publish_no_replace

    def fail_report(source, destination):
        if Path(destination).name == "report.stage":
            raise primary
        return real_publish(source, destination)

    def fail_cleanup(source, destination):
        if Path(source).name.endswith(".tmp") and Path(destination).name.endswith(
            ".cleanup"
        ):
            raise cleanup
        return real_move(source, destination)

    monkeypatch.setattr(run_backtest_module, "_publish_no_replace", fail_report)
    monkeypatch.setattr(run_backtest_module, "_move_no_replace", fail_cleanup)
    with pytest.raises(OSError) as raised:
        run_backtest_module.run_backtest(
            strategy_file=lifecycle["strategy_path"],
            data_file=lifecycle["train_path"],
            mode="in_sample_replay",
            output_dir=output,
        )

    assert raised.value is primary
    assert primary.__cause__ is None
    assert not _exception_graph_has_cycle(primary)
    assert primary.__notes__ == ["secondary owned temporary cleanup failure"]
    assert _output_snapshot(output) == before


@pytest.mark.parametrize("indirect", [False, True], ids=["same-object", "indirect"])
def test_outer_unsafe_secondary_never_creates_cause_cycle(
    lifecycle, monkeypatch, indirect
) -> None:
    output = lifecycle["tmp_path"] / f"outer-cycle-{indirect}"
    output.mkdir()
    (output / "unrelated-legacy.bin").write_bytes(b"do-not-touch")
    before = _output_snapshot(output)
    primary = OSError("OUTER PRIMARY")
    secondary = primary
    if indirect:
        bridge = OSError("OUTER BRIDGE")
        secondary = OSError("OUTER CYCLIC CLEANUP")
        bridge.__cause__ = primary
        secondary.__cause__ = bridge
    real_publish = run_backtest_module._publish_no_replace

    def fail_chart_publication(source, destination):
        if Path(destination).suffix == ".png":
            raise primary
        return real_publish(source, destination)

    monkeypatch.setattr(
        run_backtest_module, "_publish_no_replace", fail_chart_publication
    )
    monkeypatch.setattr(
        run_backtest_module,
        "_delete_owned_transaction_tree",
        lambda *args, **kwargs: (_ for _ in ()).throw(secondary),
    )
    with pytest.raises(OSError) as raised:
        run_backtest_module.run_backtest(
            strategy_file=lifecycle["strategy_path"],
            data_file=lifecycle["train_path"],
            mode="in_sample_replay",
            output_dir=output,
        )

    assert raised.value is primary
    assert primary.__cause__ is None
    assert not _exception_graph_has_cycle(primary)
    assert "secondary rollback/cleanup failure occurred" in primary.__notes__
    assert _output_snapshot(output) == before


def test_cycle_guard_rejects_existing_cycle_without_hostile_protocol_calls() -> None:
    primary = OSError("cycle-guard-primary")
    left = OSError("left")
    right = OSError("right")
    left.__cause__ = right
    right.__context__ = left
    assert run_backtest_module._cycle_safe_secondary(primary, left) is None

    class HostileProtocolError(Exception):
        def __str__(self):
            raise AssertionError("hostile __str__ invoked")

        def __repr__(self):
            raise AssertionError("hostile __repr__ invoked")

        def __getattribute__(self, name):
            if name in {"__cause__", "__context__"}:
                raise AssertionError("hostile __getattribute__ invoked")
            return super().__getattribute__(name)

    safe_candidate = HostileProtocolError()
    assert (
        run_backtest_module._cycle_safe_secondary(primary, safe_candidate)
        is safe_candidate
    )


PROBE_REPORT_BYTES = b'{\n  "report_schema": "backtest-report-v2"\n}\n'


def _call_publication_transaction(
    output: Path, monkeypatch, *, replace_report_writer: bool = True
) -> Path:
    def write_chart(_result, destination, _mode_label):
        destination.write(b"new-chart")

    monkeypatch.setattr(run_backtest_module, "_write_equity_chart", write_chart)
    return run_backtest_module._publish_output_set(
        report={"report_schema": "backtest-report-v2"},
        result=object(),
        output_dir=output,
        stem="transaction-probe",
        mode_label="probe",
    )


@pytest.mark.parametrize("failed_suffix", [".json", ".png"], ids=["report", "chart"])
def test_mutating_failed_publication_never_deletes_unowned_new_final(
    tmp_path, monkeypatch, failed_suffix
) -> None:
    output = tmp_path / f"unowned-{failed_suffix[1:]}"
    output.mkdir()
    (output / "unrelated.bin").write_bytes(b"unchanged")
    before = _output_snapshot(output)
    real_publish = run_backtest_module._publish_no_replace
    collision = b"UNOWNED-COLLISION"

    def mutate_then_fail(source, destination):
        destination = Path(destination)
        if destination.suffix == failed_suffix:
            destination.write_bytes(collision)
            raise OSError("publication mutated then failed")
        return real_publish(source, destination)

    monkeypatch.setattr(run_backtest_module, "_publish_no_replace", mutate_then_fail)
    with pytest.raises(OSError, match="publication mutated then failed"):
        _call_publication_transaction(output, monkeypatch)

    after = _output_snapshot(output)
    assert after == {**before, f"transaction-probe{failed_suffix}": collision}


def test_rollback_never_writes_or_restores_over_an_existing_final(
    tmp_path, monkeypatch
) -> None:
    output = tmp_path / "atomic-restore"
    output.mkdir()
    report_path = output / "transaction-probe.json"
    chart_path = output / "transaction-probe.png"
    (output / "unrelated.bin").write_bytes(b"unchanged")
    before = _output_snapshot(output)
    real_publish = run_backtest_module._publish_no_replace
    real_restore = run_backtest_module._restore_no_replace
    real_write_bytes = Path.write_bytes
    report_destination_calls = 0
    direct_final_writes = 0

    def fail_chart_publication(source, destination):
        if Path(destination) == chart_path:
            raise OSError("chart publication failed")
        return real_publish(source, destination)

    def count_forbidden_atomic_restore(source, destination):
        nonlocal report_destination_calls
        destination = Path(destination)
        if destination == report_path:
            report_destination_calls += 1
        return real_restore(source, destination)

    def reject_direct_final_write(path, data):
        nonlocal direct_final_writes
        if path in {report_path, chart_path}:
            direct_final_writes += 1
            real_write_bytes(path, b"PARTIAL")
            raise OSError("direct rollback write truncated final")
        return real_write_bytes(path, data)

    monkeypatch.setattr(
        run_backtest_module, "_publish_no_replace", fail_chart_publication
    )
    monkeypatch.setattr(
        run_backtest_module, "_restore_no_replace", count_forbidden_atomic_restore
    )
    monkeypatch.setattr(Path, "write_bytes", reject_direct_final_write)
    with pytest.raises(OSError, match="chart publication failed"):
        _call_publication_transaction(output, monkeypatch)

    assert direct_final_writes == 0
    assert report_destination_calls == 0
    assert _output_snapshot(output) == before


class HostileAddNoteError(OSError):
    def __getattribute__(self, name):
        if name == "add_note":
            raise AssertionError("virtual add_note lookup invoked")
        return super().__getattribute__(name)

    def __str__(self):
        raise AssertionError("hostile __str__ invoked")

    def __repr__(self):
        raise AssertionError("hostile __repr__ invoked")


@pytest.mark.parametrize("layer", ["inner", "outer"])
def test_hostile_add_note_cannot_replace_exact_primary(
    tmp_path, monkeypatch, layer
) -> None:
    output = tmp_path / f"hostile-note-{layer}"
    output.mkdir()
    if layer == "inner":
        (output / "transaction-probe.json").write_bytes(b"old-report")
        (output / "transaction-probe.png").write_bytes(b"old-chart")
    (output / "unrelated.bin").write_bytes(b"unchanged")
    before = _output_snapshot(output)
    primary = HostileAddNoteError("PRIMARY")
    cleanup = OSError("cleanup")
    real_publish = run_backtest_module._publish_no_replace
    real_move = run_backtest_module._move_no_replace

    if layer == "inner":
        def fail_report(source, destination):
            if Path(destination).name == "report.stage":
                raise primary
            return real_publish(source, destination)

        def fail_cleanup(source, destination):
            if Path(source).name.endswith(".tmp") and Path(destination).name.endswith(
                ".cleanup"
            ):
                raise cleanup
            return real_move(source, destination)

        monkeypatch.setattr(run_backtest_module, "_publish_no_replace", fail_report)
        monkeypatch.setattr(run_backtest_module, "_move_no_replace", fail_cleanup)
    else:
        def fail_replace(source, destination):
            if Path(destination).suffix == ".png":
                raise primary
            return real_publish(source, destination)

        monkeypatch.setattr(run_backtest_module, "_publish_no_replace", fail_replace)
        monkeypatch.setattr(
            run_backtest_module,
            "_delete_owned_transaction_tree",
            lambda *args, **kwargs: (_ for _ in ()).throw(cleanup),
        )

    with pytest.raises(HostileAddNoteError) as raised:
        _call_publication_transaction(
            output, monkeypatch, replace_report_writer=layer != "inner"
        )

    assert raised.value is primary
    notes = BaseException.__getattribute__(primary, "__notes__")
    assert notes == [
        "secondary owned temporary cleanup failure"
        if layer == "inner"
        else "secondary rollback/cleanup failure occurred"
    ]
    assert _output_snapshot(output) == before


@pytest.mark.parametrize(
    "primary_factory",
    [
        lambda: KeyboardInterrupt("exact-keyboard-interrupt"),
        lambda: SystemExit(66),
        lambda: GeneratorExit("exact-generator-exit"),
        lambda: ExactBaseException("exact-custom-base-exception"),
    ],
    ids=["keyboard-interrupt", "system-exit", "generator-exit", "custom-base"],
)
@pytest.mark.parametrize("failed_suffix", [".json", ".png"], ids=["report", "chart"])
@pytest.mark.parametrize(
    ("report_exists", "chart_exists"),
    [(False, False), (True, False), (False, True), (True, True)],
    ids=["none", "report-only", "chart-only", "both"],
)
def test_base_exception_at_each_publication_boundary_restores_exact_pair(
    tmp_path,
    monkeypatch,
    primary_factory,
    failed_suffix,
    report_exists,
    chart_exists,
) -> None:
    output = tmp_path / (
        f"base-boundary-{failed_suffix[1:]}-{report_exists}-{chart_exists}-"
        f"{primary_factory().__class__.__name__}"
    )
    output.mkdir()
    if report_exists:
        (output / "transaction-probe.json").write_bytes(b"old-report")
    if chart_exists:
        (output / "transaction-probe.png").write_bytes(b"old-chart")
    (output / "unrelated.bin").write_bytes(b"unchanged")
    before = _output_snapshot(output)
    primary = primary_factory()
    real_publish = run_backtest_module._publish_no_replace

    def fail_publication(source, destination):
        if Path(destination).suffix == failed_suffix:
            raise primary
        return real_publish(source, destination)

    monkeypatch.setattr(run_backtest_module, "_publish_no_replace", fail_publication)
    with pytest.raises(BaseException) as raised:
        _call_publication_transaction(output, monkeypatch)

    if report_exists or chart_exists:
        assert isinstance(raised.value, FileExistsError)
        assert primary.__traceback__ is None
    else:
        assert raised.value is primary
        assert type(raised.value) is type(primary)
        assert raised.value.args == primary.args
    assert _output_snapshot(output) == before


@pytest.mark.parametrize("collision_suffix", [".json", ".png"], ids=["report", "chart"])
@pytest.mark.parametrize(
    ("report_exists", "chart_exists"),
    [(False, False), (True, False), (False, True), (True, True)],
    ids=["none", "report-only", "chart-only", "both"],
)
def test_concurrent_final_creation_is_an_exclusive_collision_across_presence_states(
    tmp_path,
    monkeypatch,
    collision_suffix,
    report_exists,
    chart_exists,
) -> None:
    output = tmp_path / (
        f"exclusive-{collision_suffix[1:]}-{report_exists}-{chart_exists}"
    )
    output.mkdir()
    report_path = output / "transaction-probe.json"
    chart_path = output / "transaction-probe.png"
    if report_exists:
        report_path.write_bytes(b"old-report")
    if chart_exists:
        chart_path.write_bytes(b"old-chart")
    (output / "unrelated.bin").write_bytes(b"unchanged")
    competitor = (
        b"CONCURRENT-REPORT" if collision_suffix == ".json" else b"CONCURRENT-CHART"
    )
    real_publish = run_backtest_module._publish_no_replace
    inserted = False

    def insert_competitor_before_publication(source, destination):
        nonlocal inserted
        destination = Path(destination)
        if not inserted and destination.suffix == collision_suffix:
            inserted = True
            destination.write_bytes(competitor)
        return real_publish(source, destination)

    monkeypatch.setattr(
        run_backtest_module, "_publish_no_replace", insert_competitor_before_publication
    )
    with pytest.raises(FileExistsError):
        _call_publication_transaction(output, monkeypatch)

    expected = {"unrelated.bin": b"unchanged"}
    if report_exists or chart_exists:
        if report_exists:
            expected[report_path.name] = b"old-report"
        if chart_exists:
            expected[chart_path.name] = b"old-chart"
        assert not inserted
    else:
        expected[
            report_path.name if collision_suffix == ".json" else chart_path.name
        ] = competitor
        assert inserted
    assert _output_snapshot(output) == expected
    recoveries = list(
        output.parent.glob(
            f".transaction-probe{collision_suffix}.recovery-*"
        )
    )
    assert recoveries == []


@pytest.mark.parametrize(
    "later_error",
    [OSError("later chart ordinary failure"), KeyboardInterrupt("later chart interrupt")],
    ids=["ordinary", "base-exception"],
)
def test_report_competitor_aborts_before_later_chart_failure_without_deletion(
    tmp_path, monkeypatch, later_error
) -> None:
    output = tmp_path / f"report-race-{type(later_error).__name__}"
    output.mkdir()
    (output / "unrelated.bin").write_bytes(b"unchanged")
    report_path = output / "transaction-probe.json"
    chart_path = output / "transaction-probe.png"
    competitor = b"UNOWNED-REPORT-BEFORE-PUBLISH"
    real_publish = run_backtest_module._publish_no_replace
    chart_boundary_calls = 0

    def race_then_later_failure(source, destination):
        nonlocal chart_boundary_calls
        destination = Path(destination)
        if destination == report_path:
            destination.write_bytes(competitor)
        if destination == chart_path:
            chart_boundary_calls += 1
            raise later_error
        return real_publish(source, destination)

    monkeypatch.setattr(
        run_backtest_module, "_publish_no_replace", race_then_later_failure
    )
    with pytest.raises(FileExistsError):
        _call_publication_transaction(output, monkeypatch)

    assert chart_boundary_calls == 0
    assert _output_snapshot(output) == {
        report_path.name: competitor,
        "unrelated.bin": b"unchanged",
    }


def test_atomic_no_replace_primitive_succeeds_once_and_preserves_collision(
    tmp_path,
) -> None:
    staged = tmp_path / "staged"
    final = tmp_path / "final"
    staged.write_bytes(b"candidate")

    run_backtest_module._publish_no_replace(staged, final)
    assert final.read_bytes() == b"candidate"
    final.unlink()
    final.write_bytes(b"competitor")

    with pytest.raises(FileExistsError):
        run_backtest_module._publish_no_replace(staged, final)
    assert final.read_bytes() == b"competitor"


def test_publication_uses_link_no_replace_and_never_displaces_existing_finals(
    tmp_path, monkeypatch
) -> None:
    output = tmp_path / "occupied-pair"
    output.mkdir()
    (output / "transaction-probe.json").write_bytes(b"FOREIGN-REPORT")
    (output / "transaction-probe.png").write_bytes(b"FOREIGN-CHART")
    before = _output_snapshot(output)
    with pytest.raises(FileExistsError):
        _call_publication_transaction(output, monkeypatch)
    assert _output_snapshot(output) == before


def test_no_replace_mutate_then_raise_does_not_grant_rollback_ownership(
    tmp_path, monkeypatch
) -> None:
    output = tmp_path / "link-mutates-then-raises"
    output.mkdir()
    (output / "unrelated.bin").write_bytes(b"unchanged")
    report_path = output / "transaction-probe.json"
    real_link = run_backtest_module.os.link

    def link_then_raise(source, destination):
        real_link(source, destination)
        if Path(destination) == report_path:
            raise OSError("link mutated then raised")

    monkeypatch.setattr(run_backtest_module.os, "link", link_then_raise)
    with pytest.raises(OSError, match="link mutated then raised"):
        _call_publication_transaction(output, monkeypatch)

    assert _output_snapshot(output) == {
        report_path.name: PROBE_REPORT_BYTES,
        "unrelated.bin": b"unchanged",
    }


@pytest.mark.parametrize("replaced_suffix", [".json", ".png"], ids=["report", "chart"])
def test_competitor_after_link_before_ownership_record_is_never_claimed(
    tmp_path, monkeypatch, replaced_suffix
) -> None:
    output = tmp_path / f"post-link-race-{replaced_suffix[1:]}"
    output.mkdir()
    report_path = output / "transaction-probe.json"
    chart_path = output / "transaction-probe.png"
    (output / "unrelated.bin").write_bytes(b"unchanged")
    competitor = b"POST-LINK-REPORT" if replaced_suffix == ".json" else b"POST-LINK-CHART"
    competitor_stage = tmp_path / f"competitor{replaced_suffix}"
    competitor_stage.write_bytes(competitor)
    real_publish = run_backtest_module._publish_no_replace
    real_replace = run_backtest_module.os.replace

    def replace_after_link(source, destination):
        destination = Path(destination)
        real_publish(source, destination)
        if destination.suffix == replaced_suffix:
            real_replace(competitor_stage, destination)

    monkeypatch.setattr(run_backtest_module, "_publish_no_replace", replace_after_link)
    if replaced_suffix == ".json":
        def fail_chart(source, destination):
            destination = Path(destination)
            if destination == chart_path:
                raise OSError("later chart failure")
            return replace_after_link(source, destination)

        monkeypatch.setattr(run_backtest_module, "_publish_no_replace", fail_chart)
        expected_error = "later chart failure"
    else:
        monkeypatch.setattr(
            run_backtest_module,
            "_delete_owned_transaction_tree",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                OSError("post-commit cleanup failure")
            ),
        )
        expected_error = "post-commit cleanup failure"

    with pytest.raises(OSError, match=expected_error):
        _call_publication_transaction(output, monkeypatch)

    snapshot = _output_snapshot(output)
    assert snapshot[f"transaction-probe{replaced_suffix}"] == competitor
    assert snapshot["unrelated.bin"] == b"unchanged"


@pytest.mark.parametrize("replaced_suffix", [".json", ".png"], ids=["report", "chart"])
def test_occupied_pair_is_rejected_without_attempting_displacement(
    tmp_path, monkeypatch, replaced_suffix
) -> None:
    output = tmp_path / f"pre-displacement-race-{replaced_suffix[1:]}"
    output.mkdir()
    report_path = output / "transaction-probe.json"
    chart_path = output / "transaction-probe.png"
    report_path.write_bytes(b"old-report")
    chart_path.write_bytes(b"old-chart")
    (output / "unrelated.bin").write_bytes(b"unchanged")
    competitor = (
        b"PRE-DISPLACE-REPORT" if replaced_suffix == ".json" else b"PRE-DISPLACE-CHART"
    )
    competitor_stage = tmp_path / f"pre-displace{replaced_suffix}"
    competitor_stage.write_bytes(competitor)
    real_replace = run_backtest_module.os.replace
    real_move = run_backtest_module._move_no_replace
    injected = False

    def replace_before_backup_move(source, destination):
        nonlocal injected
        source = Path(source)
        destination = Path(destination)
        if not injected and source.suffix == replaced_suffix and destination.suffix == ".backup":
            injected = True
            real_replace(competitor_stage, source)
        return real_move(source, destination)

    monkeypatch.setattr(run_backtest_module, "_move_no_replace", replace_before_backup_move)
    with pytest.raises(FileExistsError):
        _call_publication_transaction(output, monkeypatch)

    assert not injected
    expected = {
        report_path.name: b"old-report",
        chart_path.name: b"old-chart",
        "unrelated.bin": b"unchanged",
    }
    assert _output_snapshot(output) == expected
    assert competitor_stage.read_bytes() == competitor


def test_competitor_between_ownership_check_and_removal_is_quarantined_not_deleted(
    tmp_path, monkeypatch
) -> None:
    output = tmp_path / "pre-removal-race"
    output.mkdir()
    report_path = output / "transaction-probe.json"
    chart_path = output / "transaction-probe.png"
    (output / "unrelated.bin").write_bytes(b"unchanged")
    competitor_stage = tmp_path / "pre-removal-competitor.json"
    competitor_stage.write_bytes(b"PRE-REMOVAL-COMPETITOR")
    real_publish = run_backtest_module._publish_no_replace
    real_replace = run_backtest_module.os.replace
    real_move = run_backtest_module._move_no_replace
    injected = False

    def fail_chart(source, destination):
        if Path(destination) == chart_path:
            raise OSError("chart failure starts rollback")
        return real_publish(source, destination)

    def replace_before_quarantine(source, destination):
        nonlocal injected
        source = Path(source)
        destination = Path(destination)
        if (
            source == report_path
            and destination.name.endswith("json.rollback.quarantine")
            and not injected
        ):
            injected = True
            real_replace(competitor_stage, report_path)
        return real_move(source, destination)

    monkeypatch.setattr(run_backtest_module, "_publish_no_replace", fail_chart)
    monkeypatch.setattr(run_backtest_module, "_move_no_replace", replace_before_quarantine)
    with pytest.raises(OSError, match="chart failure starts rollback"):
        _call_publication_transaction(output, monkeypatch)

    assert injected
    assert _output_snapshot(output) == {
        report_path.name: b"PRE-REMOVAL-COMPETITOR",
        "unrelated.bin": b"unchanged",
    }


def test_cleanup_failure_cannot_turn_occupied_pair_into_new_outputs(
    tmp_path, monkeypatch
) -> None:
    output = tmp_path / "backup-cleanup-race"
    output.mkdir()
    report_path = output / "transaction-probe.json"
    chart_path = output / "transaction-probe.png"
    report_path.write_bytes(b"old-report")
    chart_path.write_bytes(b"old-chart")
    (output / "unrelated.bin").write_bytes(b"unchanged")
    def remove_backups_then_fail(path, *args, **kwargs):
        transaction_root = Path(path)
        for backup_name in ("report.backup", "chart.backup"):
            backup = transaction_root / backup_name
            if backup.exists():
                backup.unlink()
        raise OSError("cleanup failed after backups removed")

    monkeypatch.setattr(
        run_backtest_module,
        "_delete_owned_transaction_tree",
        remove_backups_then_fail,
    )
    with pytest.raises(FileExistsError, match="occupied output pair differs") as raised:
        _call_publication_transaction(output, monkeypatch)

    assert isinstance(raised.value.__cause__, OSError)
    assert _output_snapshot(output) == {
        report_path.name: b"old-report",
        chart_path.name: b"old-chart",
        "unrelated.bin": b"unchanged",
    }


@pytest.mark.parametrize("phase", ["fsync", "close", "stat"])
def test_staged_durability_failures_preserve_exact_existing_pair(
    tmp_path, monkeypatch, phase
) -> None:
    output = tmp_path / f"staged-{phase}-failure"
    output.mkdir()
    (output / "transaction-probe.json").write_bytes(b"old-report")
    (output / "transaction-probe.png").write_bytes(b"old-chart")
    (output / "unrelated.bin").write_bytes(b"unchanged")
    before = _output_snapshot(output)
    primary = OSError(f"staged {phase} failure")

    if phase == "fsync":
        monkeypatch.setattr(
            run_backtest_module.os,
            "fsync",
            lambda *args, **kwargs: (_ for _ in ()).throw(primary),
        )
    elif phase == "close":
        real_reserve = run_backtest_module._reserve_owned_file

        class CloseFailureStream:
            def __init__(self, stream):
                self.stream = stream
                self.close_calls = 0

            def __getattr__(self, name):
                return getattr(self.stream, name)

            def close(self):
                self.close_calls += 1
                if self.close_calls == 1:
                    self.stream.close()
                    raise primary

        def fail_stage_close(path, ownership):
            stream = real_reserve(path, ownership)
            if path.name == "chart.stage":
                return CloseFailureStream(stream)
            return stream

        monkeypatch.setattr(
            run_backtest_module, "_reserve_owned_file", fail_stage_close
        )
    else:
        real_stat = Path.stat

        def fail_stage_stat(path, *args, **kwargs):
            if path.name == "report.stage":
                raise primary
            return real_stat(path, *args, **kwargs)

        monkeypatch.setattr(Path, "stat", fail_stage_stat)

    with pytest.raises(OSError) as raised:
        _call_publication_transaction(output, monkeypatch)

    assert raised.value is primary
    assert _output_snapshot(output) == before


def test_occupied_pair_never_reaches_a_replacing_backup_move(
    tmp_path, monkeypatch
) -> None:
    output = tmp_path / "backup-move-mutates"
    output.mkdir()
    report_path = output / "transaction-probe.json"
    (output / "transaction-probe.json").write_bytes(b"old-report")
    (output / "transaction-probe.png").write_bytes(b"old-chart")
    (output / "unrelated.bin").write_bytes(b"unchanged")
    before = _output_snapshot(output)
    primary = OSError("backup move completed then raised")
    real_move = run_backtest_module._move_no_replace
    attempted = False

    def move_then_raise(source, destination):
        nonlocal attempted
        result = real_move(source, destination)
        if Path(source) == report_path and Path(destination).name == "report.backup":
            attempted = True
            raise primary
        return result

    monkeypatch.setattr(run_backtest_module, "_move_no_replace", move_then_raise)
    with pytest.raises(OSError) as raised:
        _call_publication_transaction(output, monkeypatch)

    assert isinstance(raised.value, FileExistsError)
    assert raised.value is not primary
    assert not attempted
    assert _output_snapshot(output) == before


def test_quarantine_move_that_completes_then_raises_keeps_primary_and_snapshot(
    tmp_path, monkeypatch
) -> None:
    output = tmp_path / "quarantine-move-mutates"
    output.mkdir()
    (output / "unrelated.bin").write_bytes(b"unchanged")
    before = _output_snapshot(output)
    report_path = output / "transaction-probe.json"
    chart_path = output / "transaction-probe.png"
    primary = OSError("chart publication primary")
    move_error = OSError("quarantine move completed then raised")
    real_publish = run_backtest_module._publish_no_replace
    real_move = run_backtest_module._move_no_replace

    def fail_chart(source, destination):
        if Path(destination) == chart_path:
            raise primary
        return real_publish(source, destination)

    def move_then_raise(source, destination):
        result = real_move(source, destination)
        if Path(source) == report_path and Path(destination).name.endswith(
            ".rollback.quarantine"
        ):
            raise move_error
        return result

    monkeypatch.setattr(run_backtest_module, "_publish_no_replace", fail_chart)
    monkeypatch.setattr(run_backtest_module, "_move_no_replace", move_then_raise)
    with pytest.raises(OSError) as raised:
        _call_publication_transaction(output, monkeypatch)

    assert raised.value is primary
    assert raised.value.__cause__ is move_error
    assert _output_snapshot(output) == before


@pytest.mark.parametrize("suffix", [".json", ".png"], ids=["report", "chart"])
def test_occupied_pair_never_acquires_a_backup_destination(
    tmp_path, monkeypatch, suffix
) -> None:
    output = tmp_path / f"foreign-backup-{suffix[1:]}"
    output.mkdir()
    (output / "transaction-probe.json").write_bytes(b"old-report")
    (output / "transaction-probe.png").write_bytes(b"old-chart")
    (output / "unrelated.bin").write_bytes(b"unchanged")
    before = _output_snapshot(output)
    foreign = b"FOREIGN-BACKUP-DESTINATION"
    real_move = run_backtest_module._move_no_replace
    captured: list[Path] = []

    def occupy_backup_then_move(source, destination):
        destination = Path(destination)
        if destination.suffix == ".backup" and Path(source).suffix == suffix:
            destination.write_bytes(foreign)
            captured.append(destination)
        return real_move(source, destination)

    monkeypatch.setattr(run_backtest_module, "_move_no_replace", occupy_backup_then_move)
    monkeypatch.setattr(
        run_backtest_module,
        "_delete_owned_transaction_tree",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            OSError("retain transaction for inspection")
        ),
    )
    with pytest.raises(OSError):
        _call_publication_transaction(output, monkeypatch)

    assert captured == []
    assert _output_snapshot(output) == before


def test_preoccupied_quarantine_destination_is_never_overwritten(
    tmp_path, monkeypatch
) -> None:
    output = tmp_path / "foreign-quarantine"
    output.mkdir()
    report_path = output / "transaction-probe.json"
    chart_path = output / "transaction-probe.png"
    (output / "unrelated.bin").write_bytes(b"unchanged")
    foreign = b"FOREIGN-QUARANTINE-DESTINATION"
    real_publish = run_backtest_module._publish_no_replace
    real_move = run_backtest_module._move_no_replace
    captured: list[Path] = []

    def fail_chart(source, destination):
        if Path(destination) == chart_path:
            raise OSError("chart failure")
        return real_publish(source, destination)

    def occupy_quarantine_then_move(source, destination):
        destination = Path(destination)
        if Path(source) == report_path and destination.name.endswith(
            ".rollback.quarantine"
        ):
            destination.write_bytes(foreign)
            captured.append(destination)
        return real_move(source, destination)

    monkeypatch.setattr(run_backtest_module, "_publish_no_replace", fail_chart)
    monkeypatch.setattr(
        run_backtest_module, "_move_no_replace", occupy_quarantine_then_move
    )
    monkeypatch.setattr(
        run_backtest_module,
        "_delete_owned_transaction_tree",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            OSError("retain transaction for inspection")
        ),
    )
    with pytest.raises(OSError, match="chart failure"):
        _call_publication_transaction(output, monkeypatch)

    assert len(captured) == 1
    retained = _paths_with_bytes(tmp_path, foreign)
    assert 1 <= len(retained) <= 2
    assert any(path.name == "json.rollback.quarantine" for path in retained)


@pytest.mark.parametrize("suffix", [".json", ".png"], ids=["report", "chart"])
def test_no_replace_move_preserves_each_preoccupied_quarantine_destination(
    tmp_path, suffix
) -> None:
    source = tmp_path / f"candidate{suffix}"
    quarantine = tmp_path / f"{suffix[1:]}.rollback.quarantine"
    source.write_bytes(b"OWNED-CANDIDATE")
    quarantine.write_bytes(b"FOREIGN-QUARANTINE-DESTINATION")

    with pytest.raises(FileExistsError):
        run_backtest_module._move_no_replace(source, quarantine)

    assert source.read_bytes() == b"OWNED-CANDIDATE"
    assert quarantine.read_bytes() == b"FOREIGN-QUARANTINE-DESTINATION"


def test_occupied_recovery_destination_preserves_both_exact_objects(tmp_path) -> None:
    source = tmp_path / "foreign-source"
    source.write_bytes(b"FOREIGN-SOURCE")
    signature = run_backtest_module._file_signature(source)
    recovery = tmp_path / (
        f".probe.recovery-{signature[3].hex()[:16]}-"
        f"{signature[0]:x}-{signature[1]:x}"
    )
    recovery.write_bytes(b"FOREIGN-RECOVERY-DESTINATION")

    errors = run_backtest_module._move_to_recovery(
        source, recovery_root=tmp_path, label="probe"
    )

    assert len(errors) == 1
    assert isinstance(errors[0], FileExistsError)
    assert source.read_bytes() == b"FOREIGN-SOURCE"
    assert recovery.read_bytes() == b"FOREIGN-RECOVERY-DESTINATION"


def test_report_stage_collision_is_no_replace_and_retains_owned_temp(tmp_path) -> None:
    destination = tmp_path / "report.stage"
    destination.write_bytes(b"FOREIGN-REPORT-STAGE")

    with pytest.raises(FileExistsError):
        run_backtest_module._write_report_atomic({"value": 1}, destination)

    assert destination.read_bytes() == b"FOREIGN-REPORT-STAGE"
    recoveries = list(tmp_path.glob(".report-temp.recovery-*"))
    assert len(recoveries) == 1
    assert json.loads(recoveries[0].read_text(encoding="utf-8")) == {"value": 1}


def _paths_with_bytes(root: Path, expected: bytes) -> list[Path]:
    matches: list[Path] = []
    for path in root.rglob("*"):
        if path.is_file() and path.read_bytes() == expected:
            matches.append(path)
    return matches


def test_report_temp_swap_before_ordinary_cleanup_preserves_competitor(
    tmp_path, monkeypatch
) -> None:
    destination = tmp_path / "report.stage"
    foreign_stage = tmp_path / "foreign-source"
    foreign = b"FOREIGN-REPORT-TEMP"
    foreign_stage.write_bytes(foreign)
    primary = OSError("report replacement failure")
    real_reserve = run_backtest_module._reserve_owned_file

    def fail_publication_then_cleanup_safely(path, ownership):
        result = real_reserve(path, ownership)
        if Path(path).name.startswith(".report.stage."):
            with pytest.raises(PermissionError):
                run_backtest_module.os.replace(foreign_stage, path)
            raise primary
        return result

    monkeypatch.setattr(
        run_backtest_module,
        "_reserve_owned_file",
        fail_publication_then_cleanup_safely,
    )
    with pytest.raises(OSError) as raised:
        run_backtest_module._write_report_atomic({"value": 1}, destination)

    assert raised.value is primary
    matches = _paths_with_bytes(tmp_path, foreign)
    assert len(matches) == 1
    assert matches[0] == foreign_stage


@pytest.mark.parametrize(
    "primary",
    [
        KeyboardInterrupt("temp keyboard interrupt"),
        SystemExit(79),
        GeneratorExit("temp generator exit"),
        ExactBaseException("temp custom base"),
    ],
    ids=["keyboard", "system-exit", "generator", "custom-base"],
)
def test_report_temp_swap_before_outer_base_cleanup_preserves_competitor(
    tmp_path, monkeypatch, primary
) -> None:
    output = tmp_path / f"base-temp-{type(primary).__name__}"
    output.mkdir()
    foreign_source = tmp_path / f"foreign-{type(primary).__name__}"
    foreign = b"FOREIGN-REPORT-TEMP-BASE"
    foreign_source.write_bytes(foreign)
    real_reserve = run_backtest_module._reserve_owned_file

    def swap_temp_then_raise(path, ownership):
        result = real_reserve(path, ownership)
        if Path(path).name.startswith(".report.stage."):
            with pytest.raises(PermissionError):
                run_backtest_module.os.replace(foreign_source, path)
            raise primary
        return result

    monkeypatch.setattr(
        run_backtest_module, "_reserve_owned_file", swap_temp_then_raise
    )
    with pytest.raises(BaseException) as raised:
        _call_publication_transaction(
            output, monkeypatch, replace_report_writer=False
        )

    assert raised.value is primary
    matches = _paths_with_bytes(tmp_path, foreign)
    assert len(matches) == 1
    assert matches[0] == foreign_source


@pytest.mark.parametrize(
    ("case", "report_bytes", "chart_bytes", "idempotent"),
    [
        ("report-only", b"FOREIGN-REPORT", None, False),
        ("chart-only", None, b"FOREIGN-CHART", False),
        ("both-foreign", b"FOREIGN-REPORT", b"FOREIGN-CHART", False),
        ("partial-candidate", PROBE_REPORT_BYTES, None, False),
        ("pair-inconsistent", PROBE_REPORT_BYTES, b"FOREIGN-CHART", False),
        ("malformed", b"{not-json", b"not-a-png", False),
        ("identical-valid-pair", PROBE_REPORT_BYTES, b"new-chart", True),
    ],
)
def test_occupied_finals_are_fail_closed_or_truly_idempotent(
    tmp_path, monkeypatch, case, report_bytes, chart_bytes, idempotent
) -> None:
    output = tmp_path / f"occupied-{case}"
    output.mkdir()
    report_path = output / "transaction-probe.json"
    chart_path = output / "transaction-probe.png"
    if report_bytes is not None:
        report_path.write_bytes(report_bytes)
    if chart_bytes is not None:
        chart_path.write_bytes(chart_bytes)
    (output / "unrelated.bin").write_bytes(b"UNCHANGED")
    before = _output_snapshot(output)
    identities = {
        path: (path.stat().st_dev, path.stat().st_ino, path.stat().st_mtime_ns)
        for path in (report_path, chart_path)
        if path.exists()
    }

    if idempotent:
        returned = _call_publication_transaction(output, monkeypatch)
        assert returned == report_path
    else:
        with pytest.raises(FileExistsError):
            _call_publication_transaction(output, monkeypatch)

    assert _output_snapshot(output) == before
    for path, identity in identities.items():
        current = path.stat()
        assert (current.st_dev, current.st_ino, current.st_mtime_ns) == identity


def test_foreign_complete_pair_occurrences_are_never_consumed(tmp_path, monkeypatch) -> None:
    output = tmp_path / "formal-foreign-pair"
    output.mkdir()
    report_path = output / "transaction-probe.json"
    chart_path = output / "transaction-probe.png"
    report_path.write_bytes(b"FOREIGN-CORRUPT-REPORT")
    chart_path.write_bytes(b"FOREIGN-CORRUPT-CHART")
    before = _output_snapshot(output)

    with pytest.raises(FileExistsError):
        _call_publication_transaction(output, monkeypatch)

    assert _output_snapshot(output) == before
    assert _paths_with_bytes(tmp_path, b"FOREIGN-CORRUPT-REPORT") == [report_path]
    assert _paths_with_bytes(tmp_path, b"FOREIGN-CORRUPT-CHART") == [chart_path]


@pytest.mark.parametrize("boundary", ["report", "chart"])
def test_transaction_root_directory_swap_preserves_newcomer(
    tmp_path, monkeypatch, boundary
) -> None:
    output = tmp_path / f"root-swap-{boundary}"
    output.mkdir()
    real_publish = run_backtest_module._publish_no_replace
    parked: list[Path] = []
    newcomer: list[Path] = []

    def swap_root_after_publication(source, destination):
        source = Path(source)
        destination = Path(destination)
        real_publish(source, destination)
        if destination.suffix == (".json" if boundary == "report" else ".png"):
            root = source.parent
            parked_root = root.with_name(f"{root.name}.parked")
            run_backtest_module.os.rename(root, parked_root)
            root.mkdir()
            foreign = root / "FOREIGN-NEWCOMER.bin"
            foreign.write_bytes(b"FOREIGN-NEWCOMER")
            parked.append(parked_root)
            newcomer.append(foreign)

    monkeypatch.setattr(
        run_backtest_module, "_publish_no_replace", swap_root_after_publication
    )
    with pytest.raises(OSError):
        _call_publication_transaction(output, monkeypatch)

    assert len(parked) == 1
    assert len(newcomer) == 1
    assert newcomer[0].read_bytes() == b"FOREIGN-NEWCOMER"
    assert parked[0].is_dir()


@pytest.mark.parametrize(
    "primary",
    [
        OSError("root-swap-ordinary"),
        KeyboardInterrupt("root-swap-keyboard"),
        SystemExit(84),
        GeneratorExit("root-swap-generator"),
        ExactBaseException("root-swap-custom-base"),
    ],
    ids=["ordinary", "keyboard", "system-exit", "generator", "custom-base"],
)
def test_transaction_root_swap_cleanup_never_masks_publication_primary(
    tmp_path, monkeypatch, primary
) -> None:
    output = tmp_path / f"root-primary-{type(primary).__name__}"
    output.mkdir()
    chart_path = output / "transaction-probe.png"
    real_publish = run_backtest_module._publish_no_replace
    newcomer_paths: list[Path] = []

    def fail_chart(source, destination):
        if Path(destination) == chart_path:
            root = Path(source).parent
            parked_root = root.with_name(f"{root.name}.parked-primary")
            run_backtest_module.os.rename(root, parked_root)
            root.mkdir()
            newcomer = root / "FOREIGN-NEWCOMER.bin"
            newcomer.write_bytes(b"FOREIGN-NEWCOMER")
            newcomer_paths.append(newcomer)
            raise primary
        return real_publish(source, destination)

    monkeypatch.setattr(run_backtest_module, "_publish_no_replace", fail_chart)
    with pytest.raises(BaseException) as raised:
        _call_publication_transaction(output, monkeypatch)

    assert raised.value is primary
    assert raised.value.__traceback__ is not None
    assert len(newcomer_paths) == 1
    assert newcomer_paths[0].read_bytes() == b"FOREIGN-NEWCOMER"
    if isinstance(primary, Exception):
        assert isinstance(primary.__cause__, OSError)
    else:
        assert primary.__cause__ is None


@pytest.mark.parametrize("newcomer_kind", ["file", "directory-symlink"])
def test_transaction_root_non_directory_swap_is_never_followed_or_deleted(
    tmp_path, monkeypatch, newcomer_kind
) -> None:
    output = tmp_path / f"root-{newcomer_kind}"
    output.mkdir()
    chart_path = output / "transaction-probe.png"
    real_publish = run_backtest_module._publish_no_replace
    newcomer_paths: list[Path] = []
    symlink_target = tmp_path / "FOREIGN-SYMLINK-TARGET"

    def swap_after_chart(source, destination):
        source = Path(source)
        destination = Path(destination)
        real_publish(source, destination)
        if destination == chart_path:
            root = source.parent
            run_backtest_module.os.rename(root, root.with_name(f"{root.name}.parked"))
            if newcomer_kind == "file":
                root.write_bytes(b"FOREIGN-ROOT-FILE")
            else:
                symlink_target.mkdir()
                (symlink_target / "FOREIGN.bin").write_bytes(b"FOREIGN-SYMLINK-BYTES")
                run_backtest_module.os.symlink(
                    symlink_target, root, target_is_directory=True
                )
            newcomer_paths.append(root)

    monkeypatch.setattr(run_backtest_module, "_publish_no_replace", swap_after_chart)
    with pytest.raises(OSError):
        _call_publication_transaction(output, monkeypatch)

    newcomer = newcomer_paths[0]
    if newcomer_kind == "file":
        assert newcomer.read_bytes() == b"FOREIGN-ROOT-FILE"
    else:
        assert newcomer.is_symlink()
        assert (symlink_target / "FOREIGN.bin").read_bytes() == b"FOREIGN-SYMLINK-BYTES"


@pytest.mark.parametrize(
    "primary",
    [
        OSError("transaction move completed then raised"),
        KeyboardInterrupt("transaction move keyboard"),
        SystemExit(84),
        GeneratorExit("transaction move generator"),
        ExactBaseException("transaction move custom base"),
    ],
    ids=["ordinary", "keyboard", "system-exit", "generator", "custom-base"],
)
def test_transaction_root_move_completes_then_raises_keeps_pair_and_primary(
    tmp_path, monkeypatch, primary
) -> None:
    output = tmp_path / "root-move-then-raise"
    output.mkdir()
    real_move = run_backtest_module._move_no_replace
    injected = False

    def move_then_raise(source, destination):
        nonlocal injected
        source = Path(source)
        destination = Path(destination)
        result = real_move(source, destination)
        if (
            not injected
            and source.name.startswith(".alphamaster-backtest-")
            and destination.name.endswith(".cleanup")
        ):
            injected = True
            raise primary
        return result

    monkeypatch.setattr(run_backtest_module, "_move_no_replace", move_then_raise)
    with pytest.raises(BaseException) as raised:
        _call_publication_transaction(output, monkeypatch)

    assert injected
    assert raised.value is primary
    assert _output_snapshot(output) == {
        "transaction-probe.json": PROBE_REPORT_BYTES,
        "transaction-probe.png": b"new-chart",
    }
    assert not list(tmp_path.glob(".alphamaster-backtest-*"))


class HostileCleanupError(OSError):
    def __str__(self):
        raise AssertionError("hostile cleanup __str__ invoked")

    def __repr__(self):
        raise AssertionError("hostile cleanup __repr__ invoked")

    def __getattribute__(self, name):
        if name in {"__cause__", "__context__", "add_note"}:
            raise AssertionError("hostile cleanup protocol invoked")
        return super().__getattribute__(name)


def test_hostile_root_cleanup_secondary_preserves_primary_and_newcomer(
    tmp_path, monkeypatch
) -> None:
    output = tmp_path / "hostile-root-cleanup"
    output.mkdir()
    chart_path = output / "transaction-probe.png"
    primary = OSError("publication primary")
    secondary = HostileCleanupError("cleanup secondary")
    real_publish = run_backtest_module._publish_no_replace
    real_move = run_backtest_module._move_no_replace
    newcomer_paths: list[Path] = []

    def swap_root_then_fail(source, destination):
        if Path(destination) == chart_path:
            root = Path(source).parent
            run_backtest_module.os.rename(root, root.with_name(f"{root.name}.parked"))
            root.mkdir()
            newcomer = root / "FOREIGN.bin"
            newcomer.write_bytes(b"FOREIGN")
            newcomer_paths.append(newcomer)
            raise primary
        return real_publish(source, destination)

    def hostile_move(source, destination):
        result = real_move(source, destination)
        if Path(source).name.startswith(".alphamaster-backtest-"):
            raise secondary
        return result

    monkeypatch.setattr(run_backtest_module, "_publish_no_replace", swap_root_then_fail)
    monkeypatch.setattr(run_backtest_module, "_move_no_replace", hostile_move)
    with pytest.raises(OSError) as raised:
        _call_publication_transaction(output, monkeypatch)

    assert raised.value is primary
    assert BaseException.__getattribute__(primary, "__cause__") is secondary
    assert newcomer_paths[0].read_bytes() == b"FOREIGN"


def test_cleanup_path_swap_immediately_before_delete_lock_preserves_newcomer(
    tmp_path, monkeypatch
) -> None:
    output = tmp_path / "pre-delete-lock-swap"
    output.mkdir()
    real_open = run_backtest_module._open_owned_delete_handle
    newcomer_paths: list[Path] = []
    injected = False

    def swap_before_lock(path, identity, *, directory):
        nonlocal injected
        path = Path(path)
        if directory and not injected and path.name.endswith(".cleanup"):
            injected = True
            run_backtest_module.os.rename(path, path.with_name(f"{path.name}.parked"))
            path.mkdir()
            newcomer = path / "FOREIGN.bin"
            newcomer.write_bytes(b"FOREIGN-BEFORE-DELETE-LOCK")
            newcomer_paths.append(newcomer)
        return real_open(path, identity, directory=directory)

    monkeypatch.setattr(
        run_backtest_module, "_open_owned_delete_handle", swap_before_lock
    )
    with pytest.raises(FileExistsError, match="no longer names the owned object"):
        _call_publication_transaction(output, monkeypatch)

    assert injected
    assert newcomer_paths[0].read_bytes() == b"FOREIGN-BEFORE-DELETE-LOCK"


def test_open_owned_directory_handle_blocks_path_rename_until_closed(tmp_path) -> None:
    owned = tmp_path / "owned-directory"
    owned.mkdir()
    identity = run_backtest_module._directory_identity(owned)
    handle = run_backtest_module._open_owned_delete_handle(
        owned, identity, directory=True
    )
    try:
        with pytest.raises(PermissionError):
            run_backtest_module.os.rename(owned, tmp_path / "replacement")
        assert owned.is_dir()
    finally:
        run_backtest_module._close_windows_handle(handle)


def test_transaction_cleanup_has_no_unconditional_recursive_path_delete(
    tmp_path,
) -> None:
    root, identity = run_backtest_module._allocate_owned_directory(
        tmp_path, ".behavior-cleanup-"
    )
    foreign = root / "FOREIGN.bin"
    foreign.write_bytes(b"FOREIGN-MUST-SURVIVE")
    before = foreign.stat()
    errors = run_backtest_module._cleanup_owned_transaction(root, identity, {})
    matches = _paths_with_bytes(tmp_path, b"FOREIGN-MUST-SURVIVE")
    assert errors and len(matches) == 1
    after = matches[0].stat()
    assert (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) == (
        before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns
    )


def test_idempotent_pair_change_during_cleanup_fails_before_success(
    tmp_path, monkeypatch
) -> None:
    output = tmp_path / "idempotent-cleanup-race"
    output.mkdir()
    report_path = output / "transaction-probe.json"
    chart_path = output / "transaction-probe.png"
    report_path.write_bytes(PROBE_REPORT_BYTES)
    chart_path.write_bytes(b"new-chart")
    competitor = tmp_path / "competitor.json"
    competitor.write_bytes(b"FOREIGN-DURING-CLEANUP")
    real_delete = run_backtest_module._delete_owned_transaction_tree
    injected = False

    def replace_final_then_cleanup(root, identity, owned_children):
        nonlocal injected
        if not injected:
            injected = True
            run_backtest_module.os.replace(competitor, report_path)
        return real_delete(root, identity, owned_children)

    monkeypatch.setattr(
        run_backtest_module, "_delete_owned_transaction_tree", replace_final_then_cleanup
    )
    with pytest.raises(FileExistsError, match="changed before idempotent return"):
        _call_publication_transaction(output, monkeypatch)

    assert injected
    assert report_path.read_bytes() == b"FOREIGN-DURING-CLEANUP"
    assert chart_path.read_bytes() == b"new-chart"


@pytest.mark.parametrize("with_sentinel", [False, True], ids=["empty", "sentinel"])
def test_output_root_creation_race_never_claims_or_removes_competitor(
    tmp_path, monkeypatch, with_sentinel
) -> None:
    output = tmp_path / f"root-create-race-{with_sentinel}"
    primary = OSError("transaction allocation failed")
    real_mkdir = Path.mkdir
    real_move = run_backtest_module._move_no_replace
    competitor_identity: list[tuple[int, int]] = []

    def competitor_wins_mkdir(path, mode=0o777, parents=False, exist_ok=False):
        if Path(path) == output and not output.exists():
            real_mkdir(output)
            if with_sentinel:
                (output / "FOREIGN.bin").write_bytes(b"FOREIGN-ROOT")
            current = output.stat()
            competitor_identity.append((current.st_dev, current.st_ino))
        return real_mkdir(path, mode=mode, parents=parents, exist_ok=exist_ok)

    if hasattr(run_backtest_module, "_reserve_owned_directory"):
        real_reserve = run_backtest_module._reserve_owned_directory

        def competitor_wins_move(source, destination):
            if Path(destination) == output and not output.exists():
                real_mkdir(output)
                if with_sentinel:
                    (output / "FOREIGN.bin").write_bytes(b"FOREIGN-ROOT")
                current = output.stat()
                competitor_identity.append((current.st_dev, current.st_ino))
            return real_move(source, destination)

        monkeypatch.setattr(
            run_backtest_module, "_move_no_replace", competitor_wins_move
        )

        def fail_transaction_reservation(path, ownership):
            if Path(path).name.startswith(".alphamaster-backtest-"):
                raise primary
            return real_reserve(path, ownership)

        monkeypatch.setattr(
            run_backtest_module,
            "_reserve_owned_directory",
            fail_transaction_reservation,
        )
    else:
        monkeypatch.setattr(Path, "mkdir", competitor_wins_mkdir)
        monkeypatch.setattr(
            run_backtest_module.tempfile,
            "mkdtemp",
            lambda **_kwargs: (_ for _ in ()).throw(primary),
        )

    with pytest.raises(OSError) as raised:
        _call_publication_transaction(output, monkeypatch)

    assert raised.value is primary
    assert output.is_dir()
    current = output.stat()
    assert (current.st_dev, current.st_ino) == competitor_identity[0]
    expected = {"FOREIGN.bin": b"FOREIGN-ROOT"} if with_sentinel else {}
    assert _output_snapshot(output) == expected


@pytest.mark.parametrize(
    "primary",
    [
        OSError("output-root rollback ordinary"),
        KeyboardInterrupt("output-root rollback keyboard"),
        SystemExit(90),
        GeneratorExit("output-root rollback generator"),
        ExactBaseException("output-root rollback custom"),
    ],
    ids=["ordinary", "keyboard", "system-exit", "generator", "custom-base"],
)
@pytest.mark.parametrize("boundary", ["report", "chart"])
def test_output_root_swap_after_publication_preserves_exact_newcomer_and_primary(
    tmp_path, monkeypatch, primary, boundary
) -> None:
    output = tmp_path / f"output-root-swap-{boundary}-{type(primary).__name__}"
    boundary_suffix = "json" if boundary == "report" else "png"
    boundary_path = output / f"transaction-probe.{boundary_suffix}"
    real_publish = run_backtest_module._publish_no_replace
    newcomer_identity: list[tuple[int, int]] = []

    def swap_root_then_fail(source, destination):
        result = real_publish(source, destination)
        if Path(destination) == boundary_path:
            run_backtest_module.os.rename(
                output, tmp_path / f"parked-{boundary}-{type(primary).__name__}"
            )
            output.mkdir()
            current = output.stat()
            newcomer_identity.append((current.st_dev, current.st_ino))
            raise primary
        return result

    monkeypatch.setattr(run_backtest_module, "_publish_no_replace", swap_root_then_fail)
    with pytest.raises(BaseException) as raised:
        _call_publication_transaction(output, monkeypatch)

    assert raised.value is primary
    assert output.is_dir()
    current = output.stat()
    assert (current.st_dev, current.st_ino) == newcomer_identity[0]
    assert _output_snapshot(output) == {}


@pytest.mark.parametrize(
    "newcomer_kind", ["empty-directory", "sentinel-directory", "file", "directory-symlink"]
)
def test_output_root_swap_immediately_before_cleanup_lock_is_preserved(
    tmp_path, monkeypatch, newcomer_kind
) -> None:
    output = tmp_path / f"output-root-pre-lock-swap-{newcomer_kind}"
    primary = OSError("transaction allocation failed")
    real_open = run_backtest_module._open_owned_delete_handle
    injected = False
    newcomer_identity: list[tuple[int, int]] = []

    def swap_before_output_root_lock(path, identity, *, directory):
        nonlocal injected
        path = Path(path)
        if directory and path == output and not injected:
            injected = True
            run_backtest_module.os.rename(
                output, tmp_path / f"parked-output-root-{newcomer_kind}"
            )
            if newcomer_kind == "file":
                output.write_bytes(b"FOREIGN-ROOT-FILE")
            elif newcomer_kind == "directory-symlink":
                target = tmp_path / "FOREIGN-ROOT-TARGET"
                target.mkdir()
                (target / "FOREIGN.bin").write_bytes(b"FOREIGN-SYMLINK")
                run_backtest_module.os.symlink(target, output, target_is_directory=True)
            else:
                output.mkdir()
                if newcomer_kind == "sentinel-directory":
                    (output / "FOREIGN.bin").write_bytes(b"FOREIGN-DIRECTORY")
            current = output.lstat()
            newcomer_identity.append((current.st_dev, current.st_ino))
        return real_open(path, identity, directory=directory)

    monkeypatch.setattr(
        run_backtest_module, "_open_owned_delete_handle", swap_before_output_root_lock
    )
    real_reserve = run_backtest_module._reserve_owned_directory

    def fail_transaction_reservation(path, ownership):
        if Path(path).name.startswith(".alphamaster-backtest-"):
            raise primary
        return real_reserve(path, ownership)

    monkeypatch.setattr(
        run_backtest_module, "_reserve_owned_directory", fail_transaction_reservation
    )

    with pytest.raises(OSError) as raised:
        _call_publication_transaction(output, monkeypatch)

    assert raised.value is primary
    assert injected
    current = output.lstat()
    assert (current.st_dev, current.st_ino) == newcomer_identity[0]
    if newcomer_kind == "file":
        assert output.read_bytes() == b"FOREIGN-ROOT-FILE"
    elif newcomer_kind == "directory-symlink":
        assert output.is_symlink()
        assert (
            tmp_path / "FOREIGN-ROOT-TARGET" / "FOREIGN.bin"
        ).read_bytes() == b"FOREIGN-SYMLINK"
    elif newcomer_kind == "sentinel-directory":
        assert (output / "FOREIGN.bin").read_bytes() == b"FOREIGN-DIRECTORY"
    else:
        assert output.is_dir()


def test_only_exact_owned_empty_output_root_is_cleanup_eligible(
    tmp_path, monkeypatch
) -> None:
    primary = OSError("transaction allocation failed")
    real_reserve = run_backtest_module._reserve_owned_directory
    calls = 0

    def fail_transaction_reservation(path, ownership):
        nonlocal calls
        if Path(path).name.startswith(".alphamaster-backtest-"):
            calls += 1
            raise primary
        return real_reserve(path, ownership)

    monkeypatch.setattr(
        run_backtest_module, "_reserve_owned_directory", fail_transaction_reservation
    )
    owned = tmp_path / "owned-empty-output-root"
    with pytest.raises(OSError) as raised:
        _call_publication_transaction(owned, monkeypatch)
    assert raised.value is primary
    assert not owned.exists()

    preexisting = tmp_path / "preexisting-empty-output-root"
    preexisting.mkdir()
    before = preexisting.stat()
    calls = 0
    with pytest.raises(OSError) as raised:
        _call_publication_transaction(preexisting, monkeypatch)
    assert raised.value is primary
    after = preexisting.stat()
    assert (after.st_dev, after.st_ino) == (before.st_dev, before.st_ino)


def test_output_root_cannot_be_swapped_after_exact_cleanup_lock(tmp_path, monkeypatch) -> None:
    output = tmp_path / "output-root-after-lock"
    primary = OSError("transaction allocation failed")
    real_dispose = run_backtest_module._dispose_owned_handle
    attempted = False

    def attempt_swap_while_locked(handle):
        nonlocal attempted
        attempted = True
        with pytest.raises(PermissionError):
            run_backtest_module.os.rename(output, tmp_path / "foreign-swap-target")
        return real_dispose(handle)

    monkeypatch.setattr(run_backtest_module, "_dispose_owned_handle", attempt_swap_while_locked)
    real_reserve = run_backtest_module._reserve_owned_directory

    def fail_transaction_reservation(path, ownership):
        if Path(path).name.startswith(".alphamaster-backtest-"):
            raise primary
        return real_reserve(path, ownership)

    monkeypatch.setattr(
        run_backtest_module, "_reserve_owned_directory", fail_transaction_reservation
    )
    with pytest.raises(OSError) as raised:
        _call_publication_transaction(output, monkeypatch)

    assert raised.value is primary
    assert attempted
    assert not output.exists()


def test_publisher_has_no_boolean_or_raw_output_root_cleanup_mutant(
    tmp_path, monkeypatch
) -> None:
    output = tmp_path / "preexisting-empty-root"
    output.mkdir()
    before = output.stat()
    real_allocate = run_backtest_module._allocate_owned_directory
    primary = OSError("transaction-allocation-primary")

    def fail_transaction(parent, prefix):
        if prefix == ".alphamaster-backtest-":
            raise primary
        return real_allocate(parent, prefix)

    monkeypatch.setattr(run_backtest_module, "_allocate_owned_directory", fail_transaction)
    with pytest.raises(OSError) as raised:
        _call_publication_transaction(output, monkeypatch)
    after = output.stat()
    assert raised.value is primary
    assert (after.st_dev, after.st_ino, after.st_mtime_ns) == (
        before.st_dev, before.st_ino, before.st_mtime_ns
    )


@pytest.mark.parametrize("suffix", [".json", ".png"], ids=["report", "chart"])
def test_new_pair_is_revalidated_after_transaction_cleanup(
    tmp_path, monkeypatch, suffix
) -> None:
    output = tmp_path / f"post-cleanup-new-{suffix[1:]}"
    final_path = output / f"transaction-probe{suffix}"
    competitor = tmp_path / f"FOREIGN-SWAP{suffix}"
    competitor.write_bytes(b"FOREIGN-SWAP")
    real_cleanup = run_backtest_module._cleanup_owned_transaction
    injected = False

    def replace_final_after_cleanup(root, identity, owned_children=None):
        nonlocal injected
        errors = real_cleanup(root, identity, owned_children)
        if not injected and final_path.exists():
            injected = True
            run_backtest_module.os.replace(competitor, final_path)
        return errors

    monkeypatch.setattr(
        run_backtest_module, "_cleanup_owned_transaction", replace_final_after_cleanup
    )
    with pytest.raises(FileExistsError, match="changed after transaction cleanup"):
        _call_publication_transaction(output, monkeypatch)

    assert injected
    assert final_path.read_bytes() == b"FOREIGN-SWAP"
    assert _output_snapshot(output) == {final_path.name: b"FOREIGN-SWAP"}
    assert not list(tmp_path.glob(".alphamaster-backtest-*"))
    assert not list(tmp_path.glob(".alphamaster-rollback-*"))


@pytest.mark.parametrize("suffix", [".json", ".png"], ids=["report", "chart"])
def test_idempotent_pair_mtime_is_revalidated_after_cleanup(
    tmp_path, monkeypatch, suffix
) -> None:
    output = tmp_path / f"post-cleanup-idempotent-{suffix[1:]}"
    output.mkdir()
    report_path = output / "transaction-probe.json"
    chart_path = output / "transaction-probe.png"
    report_path.write_bytes(PROBE_REPORT_BYTES)
    chart_path.write_bytes(b"new-chart")
    final_path = output / f"transaction-probe{suffix}"
    before = final_path.stat()
    real_cleanup = run_backtest_module._cleanup_owned_transaction
    injected = False

    def mutate_mtime_after_cleanup(root, identity, owned_children=None):
        nonlocal injected
        errors = real_cleanup(root, identity, owned_children)
        if not injected:
            injected = True
            run_backtest_module.os.utime(
                final_path,
                ns=(before.st_atime_ns, before.st_mtime_ns + 2_000_000_000),
            )
        return errors

    monkeypatch.setattr(
        run_backtest_module, "_cleanup_owned_transaction", mutate_mtime_after_cleanup
    )
    with pytest.raises(FileExistsError, match="changed before idempotent return"):
        _call_publication_transaction(output, monkeypatch)

    after = final_path.stat()
    assert injected
    assert (after.st_dev, after.st_ino, after.st_size) == (
        before.st_dev,
        before.st_ino,
        before.st_size,
    )
    assert after.st_mtime_ns == before.st_mtime_ns + 2_000_000_000


@pytest.mark.parametrize(
    "secondary",
    [
        KeyboardInterrupt("rollback stat keyboard"),
        SystemExit(94),
        GeneratorExit("rollback stat generator"),
        ExactBaseException("rollback stat custom"),
    ],
    ids=["keyboard", "system-exit", "generator", "custom-base"],
)
def test_rollback_base_exception_cannot_mask_publication_primary(
    tmp_path, monkeypatch, secondary
) -> None:
    output = tmp_path / f"rollback-secondary-{type(secondary).__name__}"
    report_path = output / "transaction-probe.json"
    chart_path = output / "transaction-probe.png"
    primary = OSError("chart publication primary")
    real_publish = run_backtest_module._publish_no_replace
    real_stat = Path.stat
    publication_failed = False

    def fail_chart(source, destination):
        nonlocal publication_failed
        if Path(destination) == chart_path:
            publication_failed = True
            raise primary
        return real_publish(source, destination)

    def hostile_stat(path, *args, **kwargs):
        if publication_failed and Path(path) == report_path:
            raise secondary
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(run_backtest_module, "_publish_no_replace", fail_chart)
    monkeypatch.setattr(Path, "stat", hostile_stat)
    with pytest.raises(OSError) as raised:
        _call_publication_transaction(output, monkeypatch)

    assert raised.value is primary
    assert BaseException.__getattribute__(primary, "__cause__") is secondary
    assert primary.__traceback__ is not None


@pytest.mark.parametrize(
    "primary",
    [
        OSError("output allocation ordinary"),
        KeyboardInterrupt("output allocation keyboard"),
        SystemExit(94),
        GeneratorExit("output allocation generator"),
        ExactBaseException("output allocation custom"),
    ],
    ids=["ordinary", "keyboard", "system-exit", "generator", "custom-base"],
)
def test_output_root_create_then_raise_cleans_exact_created_object(
    tmp_path, monkeypatch, primary
) -> None:
    output = tmp_path / f"output-create-raise-{type(primary).__name__}"
    real_reserve = getattr(run_backtest_module, "_reserve_owned_directory", None)
    if real_reserve is None:
        real_mkdir = Path.mkdir

        def mkdir_then_raise(path, mode=0o777, parents=False, exist_ok=False):
            result = real_mkdir(path, mode=mode, parents=parents, exist_ok=exist_ok)
            if Path(path) == output:
                raise primary
            return result

        monkeypatch.setattr(Path, "mkdir", mkdir_then_raise)
    else:
        def reserve_then_raise(path, ownership):
            result = real_reserve(path, ownership)
            if Path(path).name.startswith(".alphamaster-output-"):
                raise primary
            return result

        monkeypatch.setattr(
            run_backtest_module, "_reserve_owned_directory", reserve_then_raise
        )

    with pytest.raises(BaseException) as raised:
        _call_publication_transaction(output, monkeypatch)

    assert raised.value is primary
    assert not output.exists()
    assert not list(tmp_path.glob(".alphamaster-output-*"))


@pytest.mark.parametrize("kind", ["transaction", "rollback"])
@pytest.mark.parametrize(
    "allocation_error",
    [
        OSError("allocation completed then raised"),
        KeyboardInterrupt("allocation keyboard"),
        SystemExit(94),
        GeneratorExit("allocation generator"),
        ExactBaseException("allocation custom"),
    ],
    ids=["ordinary", "keyboard", "system-exit", "generator", "custom-base"],
)
def test_owned_directory_allocator_create_then_raise_leaves_no_root(
    tmp_path, monkeypatch, kind, allocation_error
) -> None:
    output = tmp_path / f"allocator-create-raise-{kind}"
    publication_primary = OSError("chart publication primary")
    allocation_label = "backtest" if kind == "transaction" else "rollback"
    real_reserve = getattr(run_backtest_module, "_reserve_owned_directory", None)
    if real_reserve is None:
        real_mkdtemp = run_backtest_module.tempfile.mkdtemp

        def mkdtemp_then_raise(*args, **kwargs):
            path = Path(real_mkdtemp(*args, **kwargs))
            prefix = kwargs.get("prefix", "")
            if prefix.startswith(f".alphamaster-{allocation_label}-"):
                raise allocation_error
            return str(path)

        monkeypatch.setattr(run_backtest_module.tempfile, "mkdtemp", mkdtemp_then_raise)
    else:
        def reserve_then_raise(path, ownership):
            result = real_reserve(path, ownership)
            if Path(path).name.startswith(f".alphamaster-{allocation_label}-"):
                raise allocation_error
            return result

        monkeypatch.setattr(
            run_backtest_module, "_reserve_owned_directory", reserve_then_raise
        )

    if kind == "rollback":
        chart_path = output / "transaction-probe.png"
        real_publish = run_backtest_module._publish_no_replace

        def fail_chart(source, destination):
            if Path(destination) == chart_path:
                raise publication_primary
            return real_publish(source, destination)

        monkeypatch.setattr(run_backtest_module, "_publish_no_replace", fail_chart)
        expected = publication_primary
    else:
        expected = allocation_error

    with pytest.raises(BaseException) as raised:
        _call_publication_transaction(output, monkeypatch)

    assert raised.value is expected
    if kind == "rollback":
        assert BaseException.__getattribute__(publication_primary, "__cause__") is allocation_error
    assert not list(tmp_path.glob(f".alphamaster-{allocation_label}-*"))
    assert not output.exists()


@pytest.mark.parametrize(
    "primary",
    [
        OSError("partial temp ordinary"),
        KeyboardInterrupt("partial temp keyboard"),
        SystemExit(94),
        GeneratorExit("partial temp generator"),
        ExactBaseException("partial temp custom"),
    ],
    ids=["ordinary", "keyboard", "system-exit", "generator", "custom-base"],
)
def test_partial_report_temp_write_is_exact_owned_before_writer(
    tmp_path, monkeypatch, primary
) -> None:
    report_path = tmp_path / f"report-{type(primary).__name__}.stage"
    real_reserve = run_backtest_module._reserve_owned_file

    class PartialStream:
        def __init__(self, stream):
            self.stream = stream

        def write(self, _data):
            self.stream.write(b"PARTIAL")
            raise primary

        def __getattr__(self, name):
            return getattr(self.stream, name)

    def partial_then_raise(path, ownership):
        stream = real_reserve(path, ownership)
        if Path(path).name.startswith(f".{report_path.name}."):
            return PartialStream(stream)
        return stream

    monkeypatch.setattr(
        run_backtest_module, "_reserve_owned_file", partial_then_raise
    )
    with pytest.raises(BaseException) as raised:
        run_backtest_module._write_report_atomic(
            {"report_schema": "backtest-report-v2"}, report_path
        )

    assert raised.value is primary
    assert not report_path.exists()
    assert not list(tmp_path.glob(f".{report_path.name}.*.tmp"))


@pytest.mark.parametrize(
    "primary",
    [
        OSError("partial chart ordinary"),
        KeyboardInterrupt("partial chart keyboard"),
        SystemExit(97),
        GeneratorExit("partial chart generator"),
        ExactBaseException("partial chart custom"),
    ],
    ids=["ordinary", "keyboard", "system-exit", "generator", "custom-base"],
)
def test_partial_chart_stage_is_exact_owned_before_renderer(
    tmp_path, monkeypatch, primary
) -> None:
    output = tmp_path / f"partial-chart-stage-{type(primary).__name__}"

    def partial_chart(_result, path, _mode_label):
        path.write(b"PARTIAL-CHART")
        raise primary

    monkeypatch.setattr(run_backtest_module, "_write_equity_chart", partial_chart)
    monkeypatch.setattr(
        run_backtest_module,
        "_write_report_atomic",
        lambda report, path: Path(path).write_bytes(PROBE_REPORT_BYTES),
    )
    with pytest.raises(BaseException) as raised:
        run_backtest_module._publish_output_set(
            report={"report_schema": "backtest-report-v2"},
            result=object(),
            output_dir=output,
            stem="transaction-probe",
            mode_label="probe",
        )

    assert raised.value is primary
    assert not output.exists()
    assert not list(tmp_path.glob(".alphamaster-backtest-*"))
    assert not _paths_with_bytes(tmp_path, b"PARTIAL-CHART")


def test_chart_stage_swap_preserves_foreign_renderer_newcomer(
    tmp_path, monkeypatch
) -> None:
    output = tmp_path / "foreign-chart-stage"
    foreign_source = tmp_path / "FOREIGN-CHART.bin"
    foreign_source.write_bytes(b"FOREIGN-CHART-STAGE")
    def swap_chart(_result, path, _mode_label):
        staged = next(tmp_path.glob(".alphamaster-backtest-*/chart.stage"))
        run_backtest_module.os.replace(foreign_source, staged)

    monkeypatch.setattr(run_backtest_module, "_write_equity_chart", swap_chart)
    monkeypatch.setattr(
        run_backtest_module,
        "_write_report_atomic",
        lambda report, path: Path(path).write_bytes(PROBE_REPORT_BYTES),
    )
    with pytest.raises(PermissionError):
        run_backtest_module._publish_output_set(
            report={"report_schema": "backtest-report-v2"},
            result=object(),
            output_dir=output,
            stem="transaction-probe",
            mode_label="probe",
        )

    matches = _paths_with_bytes(tmp_path, b"FOREIGN-CHART-STAGE")
    assert len(matches) == 1
    assert matches[0] == foreign_source


def test_repair94_mutation_sentinels_cover_all_publication_boundaries(
    tmp_path, monkeypatch
) -> None:
    report_path = tmp_path / "report.stage"
    chart_path = tmp_path / "chart.stage"
    report_signature = run_backtest_module._write_report_atomic(
        {"report_schema": "backtest-report-v2"}, report_path
    )
    monkeypatch.setattr(
        run_backtest_module,
        "_write_equity_chart",
        lambda _result, stream, _label: stream.write(b"chart-bytes"),
    )
    chart_signature = run_backtest_module._write_owned_chart(
        object(), chart_path, "probe"
    )
    assert report_signature[:2] == run_backtest_module._directory_identity(report_path)
    assert chart_signature[:2] == run_backtest_module._directory_identity(chart_path)
    assert report_signature[4] == report_path.stat().st_mtime_ns
    assert chart_signature[4] == chart_path.stat().st_mtime_ns
    root, identity = run_backtest_module._allocate_owned_directory(
        tmp_path, ".behavior-allocation-"
    )
    assert run_backtest_module._cleanup_owned_transaction(root, identity, {}) == []


@pytest.mark.parametrize(
    "kind", ["file", "directory", "file-symlink", "directory-symlink"]
)
def test_transaction_cleanup_never_adopts_unmanifested_child(
    tmp_path, kind
) -> None:
    root, identity = run_backtest_module._allocate_owned_directory(
        tmp_path, ".repair97-manifest-"
    )
    target = root / "FOREIGN-NEWCOMER.bin"
    if kind == "file":
        target.write_bytes(b"FOREIGN-MUST-SURVIVE")
        observed = target
    elif kind == "directory":
        target.mkdir()
        observed = target / "FOREIGN.bin"
        observed.write_bytes(b"FOREIGN-MUST-SURVIVE")
    elif kind == "file-symlink":
        source = tmp_path / "FOREIGN-SYMLINK-SOURCE.bin"
        source.write_bytes(b"FOREIGN-MUST-SURVIVE")
        run_backtest_module.os.symlink(source, target)
        observed = source
    else:
        source = tmp_path / "FOREIGN-SYMLINK-DIRECTORY"
        source.mkdir()
        observed = source / "FOREIGN.bin"
        observed.write_bytes(b"FOREIGN-MUST-SURVIVE")
        run_backtest_module.os.symlink(source, target, target_is_directory=True)
    before = observed.stat()
    before_bytes = observed.read_bytes()

    errors = run_backtest_module._cleanup_owned_transaction(root, identity)

    assert errors
    if "symlink" in kind:
        matches = [observed]
        assert any(
            path.is_symlink()
            for path in tmp_path.glob(".repair97-manifest-*/*")
        )
    else:
        matches = _paths_with_bytes(tmp_path, before_bytes)
    assert len(matches) == 1
    after = matches[0].stat()
    assert matches[0].read_bytes() == before_bytes
    assert (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) == (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )


def test_manifest_child_swap_before_delete_preserves_foreign_and_cleans_sibling(
    tmp_path, monkeypatch
) -> None:
    root, root_identity = run_backtest_module._allocate_owned_directory(
        tmp_path, ".repair97-child-swap-"
    )
    manifest: dict[str, tuple[int, int]] = {}
    for name in ("owned.bin", "sibling.bin"):
        ownership: list[tuple[int, int]] = []
        stream = run_backtest_module._reserve_owned_file(root / name, ownership)
        stream.write(name.encode("ascii"))
        stream.close()
        manifest[name] = ownership[0]
    foreign = tmp_path / "FOREIGN-SWAP.bin"
    foreign.write_bytes(b"FOREIGN-CHILD-SWAP")
    before = foreign.stat()
    real_open = run_backtest_module._open_owned_delete_handle
    injected = False

    def swap_before_child_lock(path, identity, *, directory):
        nonlocal injected
        if not directory and Path(path).name == "owned.bin" and not injected:
            injected = True
            run_backtest_module.os.replace(foreign, path)
        return real_open(path, identity, directory=directory)

    monkeypatch.setattr(
        run_backtest_module, "_open_owned_delete_handle", swap_before_child_lock
    )
    errors = run_backtest_module._cleanup_owned_transaction(
        root, root_identity, manifest
    )

    assert injected
    assert errors
    matches = _paths_with_bytes(tmp_path, b"FOREIGN-CHILD-SWAP")
    assert len(matches) == 1
    after = matches[0].stat()
    assert (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) == (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    assert not _paths_with_bytes(tmp_path, b"sibling.bin")


@pytest.mark.parametrize("stage", ["report", "chart"])
def test_reserved_stage_swap_before_first_write_never_mutates_foreign(
    tmp_path, monkeypatch, stage
) -> None:
    output = tmp_path / f"reserved-stage-swap-{stage}"
    foreign = tmp_path / f"FOREIGN-{stage}.bin"
    foreign.write_bytes(b"FOREIGN-STAGE-MUST-STAY-EXACT")
    before = foreign.stat()
    real_reserve = run_backtest_module._reserve_owned_file
    injected = False
    allocating = True

    def reserve_then_swap(path, ownership):
        nonlocal injected
        result = real_reserve(path, ownership)
        path = Path(path)
        is_target = (
            path.name == "chart.stage"
            if stage == "chart"
            else path.name.startswith(".report.stage.")
        )
        if is_target and not injected:
            injected = True
            run_backtest_module.os.replace(foreign, path)
        return result

    monkeypatch.setattr(run_backtest_module, "_reserve_owned_file", reserve_then_swap)
    with pytest.raises(BaseException):
        _call_publication_transaction(output, monkeypatch, replace_report_writer=False)

    assert injected
    matches = _paths_with_bytes(tmp_path, b"FOREIGN-STAGE-MUST-STAY-EXACT")
    assert len(matches) == 1
    after = matches[0].stat()
    assert (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) == (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )


def test_directory_creation_identity_is_not_captured_from_replaceable_path(
    tmp_path, monkeypatch
) -> None:
    real_identity = run_backtest_module._directory_identity
    injected = False
    newcomer: list[Path] = []
    parked: list[Path] = []

    def swap_before_path_identity(path):
        nonlocal injected
        path = Path(path)
        if allocating and path.name.startswith(".repair97-directory-") and not injected:
            injected = True
            parked_path = path.with_name(f"{path.name}.genuine-parked")
            run_backtest_module.os.rename(path, parked_path)
            path.mkdir()
            sentinel = path / "FOREIGN.bin"
            sentinel.write_bytes(b"FOREIGN-DIRECTORY-SENTINEL")
            parked.append(parked_path)
            newcomer.append(sentinel)
        return real_identity(path)

    monkeypatch.setattr(run_backtest_module, "_directory_identity", swap_before_path_identity)
    root, identity = run_backtest_module._allocate_owned_directory(
        tmp_path, ".repair97-directory-"
    )
    allocating = False
    errors = run_backtest_module._cleanup_owned_transaction(root, identity)

    if injected:
        assert errors
        assert newcomer[0].read_bytes() == b"FOREIGN-DIRECTORY-SENTINEL"
        assert parked[0].is_dir()
    else:
        assert not errors
        assert not root.exists()


def test_repair97_ownership_mutation_sentinels(tmp_path) -> None:
    root, identity = run_backtest_module._allocate_owned_directory(
        tmp_path, ".behavior-manifest-"
    )
    ownership = []
    stream = run_backtest_module._reserve_owned_file(root / "owned.bin", ownership)
    stream.write(b"OWNED")
    stream.close()
    foreign = root / "FOREIGN.bin"
    foreign.write_bytes(b"FOREIGN")
    errors = run_backtest_module._cleanup_owned_transaction(
        root, identity, {"owned.bin": ownership[0]}
    )
    assert errors
    assert not _paths_with_bytes(tmp_path, b"OWNED")
    assert len(_paths_with_bytes(tmp_path, b"FOREIGN")) == 1


class _Repair101BaseError(BaseException):
    pass


@pytest.mark.parametrize("stage", ["report", "chart"])
@pytest.mark.parametrize(
    "error_type",
    [OSError, KeyboardInterrupt, SystemExit, GeneratorExit, _Repair101BaseError],
)
def test_repair101_never_reopens_reserved_stage_path_for_sync(
    tmp_path, monkeypatch, stage, error_type
) -> None:
    output = tmp_path / f"repair101-no-reopen-{stage}-{error_type.__name__}"
    primary = error_type("forbidden staged pathname reopen")
    real_open = Path.open
    observed: list[Path] = []

    def reject_post_reservation_reopen(path, *args, **kwargs):
        mode = args[0] if args else kwargs.get("mode", "r")
        if mode == "rb+" and path.name in {"report.stage", "chart.stage"}:
            observed.append(path)
            if path.name == f"{stage}.stage":
                raise primary
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", reject_post_reservation_reopen)
    raised: BaseException | None = None
    published: Path | None = None
    try:
        published = _call_publication_transaction(
            output, monkeypatch, replace_report_writer=False
        )
    except BaseException as exc:
        raised = exc

    assert raised is None
    assert published is not None
    assert published.read_bytes() == PROBE_REPORT_BYTES
    assert (output / "transaction-probe.png").read_bytes() == b"new-chart"
    assert observed == []


def test_repair101_reserved_stage_streams_are_fsynced_then_closed_once(
    tmp_path, monkeypatch
) -> None:
    output = tmp_path / "repair101-stream-lifetime"
    real_reserve = run_backtest_module._reserve_owned_file
    real_fsync = run_backtest_module.os.fsync
    close_counts: dict[str, int] = {}
    labels_by_descriptor: dict[int, str] = {}
    synced: list[str] = []

    class TrackedStream:
        def __init__(self, stream, label):
            self.stream = stream
            self.label = label

        def __getattr__(self, name):
            return getattr(self.stream, name)

        def close(self):
            assert self.stream.fileno() in labels_by_descriptor
            assert self.label in synced
            close_counts[self.label] = close_counts.get(self.label, 0) + 1
            self.stream.close()

    def track_reserved_stream(path, ownership):
        stream = real_reserve(path, ownership)
        label = "report" if path.name.startswith(".report.stage.") else path.name
        labels_by_descriptor[stream.fileno()] = label
        return TrackedStream(stream, label)

    def track_fsync(descriptor):
        synced.append(labels_by_descriptor[descriptor])
        return real_fsync(descriptor)

    monkeypatch.setattr(
        run_backtest_module, "_reserve_owned_file", track_reserved_stream
    )
    monkeypatch.setattr(run_backtest_module.os, "fsync", track_fsync)
    published = _call_publication_transaction(
        output, monkeypatch, replace_report_writer=False
    )

    assert published.read_bytes() == PROBE_REPORT_BYTES
    assert close_counts == {"report": 1, "chart.stage": 1}
    assert synced == ["chart.stage", "report"]


@pytest.mark.parametrize(
    "forgery", ["device", "inode", "both", "raise", "swapped"]
)
def test_repair101_directory_identity_never_comes_from_post_create_path_lookup(
    tmp_path, monkeypatch, forgery
) -> None:
    prefix = f".repair101-handle-identity-{forgery}-"
    real_lstat = Path.lstat
    observed: list[Path] = []
    parked: list[Path] = []

    class ForgedStat:
        def __init__(self, original, *, device_delta: int, inode_delta: int):
            self._original = original
            self.st_dev = original.st_dev + device_delta
            self.st_ino = original.st_ino + inode_delta

        def __getattr__(self, name):
            return getattr(self._original, name)

    def forge_post_create_identity(path, *args, **kwargs):
        if path.name.startswith(prefix):
            observed.append(path)
            if forgery == "raise":
                raise OSError("post-create pathname identity is forbidden")
            if forgery == "swapped":
                parked_path = path.with_name(f"{path.name}.genuine-parked")
                run_backtest_module.os.rename(path, parked_path)
                path.mkdir()
                (path / "FOREIGN.bin").write_bytes(b"FOREIGN-DIRECTORY")
                parked.append(parked_path)
            original = real_lstat(path, *args, **kwargs)
            return ForgedStat(
                original,
                device_delta=1 if forgery in {"device", "both"} else 0,
                inode_delta=1 if forgery in {"inode", "both"} else 0,
            )
        return real_lstat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", forge_post_create_identity)
    root, identity = run_backtest_module._allocate_owned_directory(
        tmp_path, prefix
    )
    errors = run_backtest_module._cleanup_owned_empty_directory(root, identity)

    assert observed == []
    assert parked == []
    assert errors == []
    assert not root.exists()


@pytest.mark.parametrize(
    ("volume_delta", "file_id_delta"),
    [(1, 0), (0, 1), (1, 1)],
    ids=["volume-only", "file-id-only", "both"],
)
def test_repair101_cleanup_rejects_every_complete_handle_identity_mismatch(
    tmp_path, monkeypatch, volume_delta, file_id_delta
) -> None:
    root, identity = run_backtest_module._allocate_owned_directory(
        tmp_path, ".repair101-cleanup-identity-"
    )
    real_identity = run_backtest_module._windows_handle_identity

    def forged_handle_identity(handle):
        volume, file_id = real_identity(handle)
        return volume + volume_delta, file_id + file_id_delta

    monkeypatch.setattr(
        run_backtest_module, "_windows_handle_identity", forged_handle_identity
    )
    errors = run_backtest_module._cleanup_owned_empty_directory(root, identity)

    assert errors
    assert root.is_dir()


def test_repair101_ownership_mutation_sentinels(tmp_path, monkeypatch) -> None:
    real_open = Path.open

    def reject_path_reopen(path, *args, **kwargs):
        if ".stage" in Path(path).name:
            raise AssertionError("stage path reopened")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", reject_path_reopen)
    report = run_backtest_module._write_report_atomic(
        {"report_schema": "backtest-report-v2"}, tmp_path / "report.stage"
    )
    monkeypatch.setattr(
        run_backtest_module,
        "_write_equity_chart",
        lambda _result, stream, _label: stream.write(b"chart"),
    )
    chart = run_backtest_module._write_owned_chart(
        object(), tmp_path / "chart.stage", "probe"
    )
    assert report[2] > 0 and chart[2] == 5


class _Repair104BaseError(BaseException):
    pass


@pytest.mark.parametrize("scenario", ["first", "idempotent", "different"])
@pytest.mark.parametrize(
    "error_type",
    [OSError, KeyboardInterrupt, SystemExit, GeneratorExit, _Repair104BaseError],
)
def test_repair104_never_reopens_any_staged_path_for_data(
    tmp_path, monkeypatch, scenario, error_type
) -> None:
    output = tmp_path / f"repair104-stage-{scenario}-{error_type.__name__}"
    if scenario == "idempotent":
        _call_publication_transaction(output, monkeypatch, replace_report_writer=False)
        monkeypatch.undo()
    elif scenario == "different":
        output.mkdir()
        (output / "transaction-probe.json").write_bytes(b"FOREIGN-REPORT")
        (output / "transaction-probe.png").write_bytes(b"FOREIGN-CHART")
    before = _output_snapshot(output)
    primary = error_type("forbidden staged pathname reopen")
    observed: list[tuple[str, str]] = []
    real_path_open = Path.open
    real_read_bytes = Path.read_bytes
    real_read_text = Path.read_text
    real_builtin_open = builtins.open
    real_os_open = run_backtest_module.os.open

    def is_stage_path(value) -> bool:
        try:
            name = Path(value).name
        except TypeError:
            return False
        return "report.stage" in name or "chart.stage" in name

    def reject_path_open(path, *args, **kwargs):
        if is_stage_path(path):
            mode = args[0] if args else kwargs.get("mode", "r")
            observed.append(("Path.open", mode))
            raise primary
        return real_path_open(path, *args, **kwargs)

    def reject_read_bytes(path, *args, **kwargs):
        if is_stage_path(path):
            observed.append(("Path.read_bytes", "rb"))
            raise primary
        return real_read_bytes(path, *args, **kwargs)

    def reject_read_text(path, *args, **kwargs):
        if is_stage_path(path):
            observed.append(("Path.read_text", "r"))
            raise primary
        return real_read_text(path, *args, **kwargs)

    def reject_builtin_open(file, *args, **kwargs):
        if is_stage_path(file):
            mode = args[0] if args else kwargs.get("mode", "r")
            observed.append(("builtins.open", mode))
            raise primary
        return real_builtin_open(file, *args, **kwargs)

    def reject_os_open(path, flags, *args, **kwargs):
        if is_stage_path(path):
            observed.append(("os.open", str(flags)))
            raise primary
        return real_os_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(Path, "open", reject_path_open)
    monkeypatch.setattr(Path, "read_bytes", reject_read_bytes)
    monkeypatch.setattr(Path, "read_text", reject_read_text)
    monkeypatch.setattr(builtins, "open", reject_builtin_open)
    monkeypatch.setattr(run_backtest_module.os, "open", reject_os_open)
    raised: BaseException | None = None
    try:
        _call_publication_transaction(output, monkeypatch, replace_report_writer=False)
    except BaseException as exc:
        raised = exc

    assert observed == []
    if scenario == "different":
        assert isinstance(raised, FileExistsError)
        assert _output_snapshot(output) == before
    else:
        assert raised is None
        assert _output_snapshot(output) == {
            "transaction-probe.json": PROBE_REPORT_BYTES,
            "transaction-probe.png": b"new-chart",
        }


@pytest.mark.parametrize("stage", ["report", "chart"])
@pytest.mark.parametrize(
    "error_type",
    [OSError, KeyboardInterrupt, SystemExit, GeneratorExit, _Repair104BaseError],
)
def test_repair104_hostile_close_is_attempted_exactly_once(
    tmp_path, monkeypatch, stage, error_type
) -> None:
    primary = error_type("HOSTILE-CLOSE")
    destination = tmp_path / f"{stage}.stage"
    unrelated = tmp_path / "FOREIGN.bin"
    unrelated.write_bytes(b"FOREIGN-MUST-STAY-EXACT")
    before = unrelated.stat()
    real_reserve = run_backtest_module._reserve_owned_file
    real_fsync = run_backtest_module.os.fsync
    close_count = 0
    flush_count = 0
    fsync_count = 0
    target_descriptor: int | None = None

    class HostileCloseStream:
        def __init__(self, stream):
            self.stream = stream

        def __getattr__(self, name):
            return getattr(self.stream, name)

        def flush(self):
            nonlocal flush_count
            flush_count += 1
            return self.stream.flush()

        def close(self):
            nonlocal close_count
            close_count += 1
            self.stream.close()
            raise primary

    def reserve_hostile_stream(path, ownership):
        nonlocal target_descriptor
        stream = real_reserve(path, ownership)
        is_target = (
            path.name == "chart.stage"
            if stage == "chart"
            else path.name.startswith(".report.stage.")
        )
        if is_target:
            target_descriptor = stream.fileno()
            return HostileCloseStream(stream)
        return stream

    def count_target_fsync(descriptor):
        nonlocal fsync_count
        if descriptor == target_descriptor:
            fsync_count += 1
        return real_fsync(descriptor)

    monkeypatch.setattr(
        run_backtest_module, "_reserve_owned_file", reserve_hostile_stream
    )
    monkeypatch.setattr(run_backtest_module.os, "fsync", count_target_fsync)
    monkeypatch.setattr(
        run_backtest_module,
        "_write_equity_chart",
        lambda _result, stream, _label: stream.write(b"new-chart"),
    )
    raised: BaseException | None = None
    try:
        if stage == "report":
            run_backtest_module._write_report_atomic(
                {"report_schema": "backtest-report-v2"}, destination
            )
        else:
            run_backtest_module._write_owned_chart(object(), destination, "probe")
    except BaseException as exc:
        raised = exc

    assert raised is primary
    assert primary.__traceback__ is not None
    assert close_count == 1
    assert flush_count == 1
    assert fsync_count == 1
    assert not destination.exists()
    after = unrelated.stat()
    assert unrelated.read_bytes() == b"FOREIGN-MUST-STAY-EXACT"
    assert (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) == (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )


def test_repair104_ownership_mutation_sentinels(tmp_path, monkeypatch) -> None:
    real_reserve = run_backtest_module._reserve_owned_file
    close_counts = []

    class CountClose:
        def __init__(self, stream):
            self.stream = stream
            self.count = 0
            close_counts.append(self)

        def __getattr__(self, name):
            return getattr(self.stream, name)

        def close(self):
            self.count += 1
            return self.stream.close()

    monkeypatch.setattr(
        run_backtest_module,
        "_reserve_owned_file",
        lambda path, ownership: CountClose(real_reserve(path, ownership)),
    )
    run_backtest_module._write_report_atomic({"value": 1}, tmp_path / "report.stage")
    monkeypatch.setattr(
        run_backtest_module,
        "_write_equity_chart",
        lambda _result, stream, _label: stream.write(b"chart"),
    )
    run_backtest_module._write_owned_chart(object(), tmp_path / "chart.stage", "probe")
    assert [item.count for item in close_counts] == [1, 1]


class _Repair108BaseError(BaseException):
    pass


@pytest.mark.parametrize("stage", ["report", "chart"])
@pytest.mark.parametrize("pattern", ["half", "one", "varying"])
def test_repair108_positive_short_writes_publish_complete_exact_bytes(
    tmp_path, monkeypatch, stage, pattern
) -> None:
    destination = tmp_path / f"{stage}.stage"
    report = {"report_schema": "backtest-report-v2", "value": "complete"}
    report_bytes = (
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    chart_bytes = b"complete-chart-payload"
    expected = report_bytes if stage == "report" else chart_bytes
    real_reserve = run_backtest_module._reserve_owned_file
    close_count = 0
    write_count = 0

    class ShortWriteStream:
        def __init__(self, stream):
            self.stream = stream

        def __getattr__(self, name):
            return getattr(self.stream, name)

        def write(self, data):
            nonlocal write_count
            write_count += 1
            remaining = len(data)
            if pattern == "one":
                accepted = 1
            elif pattern == "half":
                accepted = max(1, remaining // 2)
            else:
                accepted = min(remaining, (3, 1, 7)[(write_count - 1) % 3])
            actual = self.stream.write(data[:accepted])
            assert actual == accepted
            return accepted

        def close(self):
            nonlocal close_count
            close_count += 1
            return self.stream.close()

    def reserve_short_stream(path, ownership):
        return ShortWriteStream(real_reserve(path, ownership))

    monkeypatch.setattr(
        run_backtest_module, "_reserve_owned_file", reserve_short_stream
    )
    monkeypatch.setattr(
        run_backtest_module,
        "_write_equity_chart",
        lambda _result, stream, _label: stream.write(chart_bytes),
    )
    signature = (
        run_backtest_module._write_report_atomic(report, destination)
        if stage == "report"
        else run_backtest_module._write_owned_chart(object(), destination, "probe")
    )

    published = destination.read_bytes()
    assert write_count > 1
    assert published == expected
    assert signature[2] == len(published)
    assert signature[3] == run_backtest_module.hashlib.sha256(published).digest()
    assert close_count == 1


@pytest.mark.parametrize("stage", ["report", "chart"])
@pytest.mark.parametrize(
    ("invalid_kind", "invalid_value"),
    [
        ("zero", 0),
        ("none", None),
        ("negative", -1),
        ("oversized", "oversized"),
        ("non-int", "invalid"),
        ("bool", True),
    ],
)
def test_repair108_invalid_write_progress_fails_before_publication(
    tmp_path, monkeypatch, stage, invalid_kind, invalid_value
) -> None:
    destination = tmp_path / f"{stage}.stage"
    unrelated = tmp_path / "FOREIGN.bin"
    unrelated.write_bytes(b"FOREIGN-EXACT")
    before = unrelated.stat()
    real_reserve = run_backtest_module._reserve_owned_file
    close_count = 0

    class InvalidProgressStream:
        def __init__(self, stream):
            self.stream = stream

        def __getattr__(self, name):
            return getattr(self.stream, name)

        def write(self, data):
            if invalid_kind == "oversized":
                return len(data) + 1
            return invalid_value

        def close(self):
            nonlocal close_count
            close_count += 1
            return self.stream.close()

    monkeypatch.setattr(
        run_backtest_module,
        "_reserve_owned_file",
        lambda path, ownership: InvalidProgressStream(
            real_reserve(path, ownership)
        ),
    )
    monkeypatch.setattr(
        run_backtest_module,
        "_write_equity_chart",
        lambda _result, stream, _label: stream.write(b"chart-payload"),
    )
    with pytest.raises(OSError, match="write progress"):
        if stage == "report":
            run_backtest_module._write_report_atomic(
                {"report_schema": "backtest-report-v2"}, destination
            )
        else:
            run_backtest_module._write_owned_chart(object(), destination, "probe")

    assert close_count == 1
    assert not destination.exists()
    after = unrelated.stat()
    assert unrelated.read_bytes() == b"FOREIGN-EXACT"
    assert (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) == (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )


@pytest.mark.parametrize("stage", ["report", "chart"])
@pytest.mark.parametrize(
    "error_type",
    [OSError, KeyboardInterrupt, SystemExit, GeneratorExit, _Repair108BaseError],
)
def test_repair108_partial_progress_then_exception_preserves_exact_primary(
    tmp_path, monkeypatch, stage, error_type
) -> None:
    destination = tmp_path / f"{stage}.stage"
    primary = error_type("partial-write-primary")
    real_reserve = run_backtest_module._reserve_owned_file
    close_count = 0
    write_count = 0

    class PartialThenRaiseStream:
        def __init__(self, stream):
            self.stream = stream

        def __getattr__(self, name):
            return getattr(self.stream, name)

        def write(self, data):
            nonlocal write_count
            write_count += 1
            if write_count == 2:
                raise primary
            accepted = max(1, len(data) // 2)
            self.stream.write(data[:accepted])
            return accepted

        def close(self):
            nonlocal close_count
            close_count += 1
            return self.stream.close()

    monkeypatch.setattr(
        run_backtest_module,
        "_reserve_owned_file",
        lambda path, ownership: PartialThenRaiseStream(
            real_reserve(path, ownership)
        ),
    )
    monkeypatch.setattr(
        run_backtest_module,
        "_write_equity_chart",
        lambda _result, stream, _label: stream.write(b"chart-payload"),
    )
    raised: BaseException | None = None
    try:
        if stage == "report":
            run_backtest_module._write_report_atomic(
                {"report_schema": "backtest-report-v2"}, destination
            )
        else:
            run_backtest_module._write_owned_chart(object(), destination, "probe")
    except BaseException as exc:
        raised = exc

    assert raised is primary
    assert primary.__traceback__ is not None
    assert write_count == 2
    assert close_count == 1
    assert not destination.exists()


def test_repair108_short_write_idempotent_pair_remains_exact(
    tmp_path, monkeypatch
) -> None:
    output = tmp_path / "repair108-idempotent"
    real_reserve = run_backtest_module._reserve_owned_file

    class HalfWriteStream:
        def __init__(self, stream):
            self.stream = stream

        def __getattr__(self, name):
            return getattr(self.stream, name)

        def write(self, data):
            accepted = max(1, len(data) // 2)
            self.stream.write(data[:accepted])
            return accepted

    monkeypatch.setattr(
        run_backtest_module,
        "_reserve_owned_file",
        lambda path, ownership: HalfWriteStream(real_reserve(path, ownership)),
    )
    first = _call_publication_transaction(
        output, monkeypatch, replace_report_writer=False
    )
    before = _output_snapshot(output)
    second = _call_publication_transaction(
        output, monkeypatch, replace_report_writer=False
    )

    assert first == second
    assert _output_snapshot(output) == before == {
        "transaction-probe.json": PROBE_REPORT_BYTES,
        "transaction-probe.png": b"new-chart",
    }


def test_repair108_short_write_mutation_sentinels(tmp_path, monkeypatch) -> None:
    real_reserve = run_backtest_module._reserve_owned_file
    write_counts = []

    class HalfWrite:
        def __init__(self, stream):
            self.stream = stream
            self.count = 0
            write_counts.append(self)

        def __getattr__(self, name):
            return getattr(self.stream, name)

        def write(self, payload):
            self.count += 1
            accepted = max(1, len(payload) // 2)
            self.stream.write(payload[:accepted])
            return accepted

    monkeypatch.setattr(
        run_backtest_module,
        "_reserve_owned_file",
        lambda path, ownership: HalfWrite(real_reserve(path, ownership)),
    )
    report_path = tmp_path / "report.stage"
    report = {"report_schema": "backtest-report-v2", "value": "complete"}
    signature = run_backtest_module._write_report_atomic(report, report_path)
    expected = (
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    assert report_path.read_bytes() == expected
    assert signature[2] == len(expected)
    assert write_counts[0].count > 1


class _Repair112BaseError(BaseException):
    pass


def test_repair112_preexisting_directory_symlink_is_never_traversed(
    tmp_path, monkeypatch
) -> None:
    foreign = tmp_path / "FOREIGN-TARGET"
    foreign.mkdir()
    sentinel = foreign / "FOREIGN.bin"
    sentinel.write_bytes(b"FOREIGN-TARGET-MUST-STAY-EXACT")
    before = sentinel.stat()
    output = tmp_path / "requested-output"
    run_backtest_module.os.symlink(foreign, output, target_is_directory=True)
    link_before = output.lstat()

    with pytest.raises(OSError, match="ordinary directory"):
        _call_publication_transaction(output, monkeypatch)

    link_after = output.lstat()
    after = sentinel.stat()
    assert output.is_symlink()
    assert sentinel.read_bytes() == b"FOREIGN-TARGET-MUST-STAY-EXACT"
    assert sorted(path.name for path in foreign.iterdir()) == ["FOREIGN.bin"]
    assert (link_after.st_dev, link_after.st_ino, link_after.st_mtime_ns) == (
        link_before.st_dev,
        link_before.st_ino,
        link_before.st_mtime_ns,
    )
    assert (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) == (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )


def test_repair112_preexisting_ordinary_output_directory_still_succeeds(
    tmp_path, monkeypatch
) -> None:
    output = tmp_path / "ordinary-output"
    output.mkdir()
    identity = run_backtest_module._directory_identity(output)

    published = _call_publication_transaction(output, monkeypatch)

    assert published == output / "transaction-probe.json"
    assert run_backtest_module._directory_identity(output) == identity
    assert _output_snapshot(output) == {
        "transaction-probe.json": PROBE_REPORT_BYTES,
        "transaction-probe.png": b"new-chart",
    }


@pytest.mark.parametrize(
    "error_type",
    [OSError, KeyboardInterrupt, SystemExit, GeneratorExit, _Repair112BaseError],
)
@pytest.mark.parametrize("with_cleanup_error", [False, True])
def test_repair112_cleanup_evidence_preserves_exact_primary_traceback(
    error_type, with_cleanup_error
) -> None:
    primary = error_type("repair112-primary")
    cleanup = OSError("repair112-cleanup")
    original_traceback = None
    raised = None

    try:
        try:
            raise primary
        except BaseException as caught:
            original_traceback = caught.__traceback__
            run_backtest_module._raise_primary_after_cleanup(
                caught,
                [cleanup] if with_cleanup_error else [],
                "secondary repair112 cleanup failure",
            )
            raise
    except BaseException as exc:
        raised = exc

    assert raised is primary
    assert raised.__traceback__ is original_traceback
    if with_cleanup_error and isinstance(primary, Exception):
        assert BaseException.__getattribute__(primary, "__cause__") is cleanup
    else:
        assert BaseException.__getattribute__(primary, "__cause__") is None


def test_repair112_publication_ownership_mutation_sentinels(
    tmp_path, monkeypatch
) -> None:
    foreign = tmp_path / "FOREIGN-ROOT"
    foreign.mkdir()
    output = tmp_path / "linked-output"
    run_backtest_module.os.symlink(foreign, output, target_is_directory=True)
    before = output.lstat()
    with pytest.raises(OSError, match="ordinary directory"):
        _call_publication_transaction(output, monkeypatch)
    after = output.lstat()
    assert not list(foreign.iterdir())
    assert (after.st_dev, after.st_ino, after.st_mtime_ns) == (
        before.st_dev, before.st_ino, before.st_mtime_ns
    )


def _repair115_entry_snapshot(path: Path):
    current = path.lstat()
    return (
        current.st_dev,
        current.st_ino,
        current.st_size,
        current.st_mtime_ns,
        getattr(current, "st_file_attributes", 0),
        getattr(current, "st_reparse_tag", 0),
        run_backtest_module.os.readlink(path) if path.is_symlink() else None,
    )


@pytest.mark.parametrize("linked", [("report",), ("chart",), ("report", "chart")])
@pytest.mark.parametrize("matching", [True, False])
def test_repair115_final_file_symlinks_fail_before_staging_and_preserve_targets(
    tmp_path, monkeypatch, linked, matching
) -> None:
    output = tmp_path / "repair115-symlink-output"
    foreign = tmp_path / "FOREIGN-FINALS"
    foreign.mkdir()
    _call_publication_transaction(output, monkeypatch)
    paths = {
        "report": output / "transaction-probe.json",
        "chart": output / "transaction-probe.png",
    }
    target_snapshots = {}
    link_snapshots = {}
    ordinary_snapshots = {}
    for label, final_path in paths.items():
        if label in linked:
            target = foreign / final_path.name
            run_backtest_module.os.replace(final_path, target)
            if not matching:
                target.write_bytes(f"FOREIGN-{label}-DIFFERENT".encode("ascii"))
            run_backtest_module.os.symlink(target, final_path)
            target_stat = target.stat()
            target_snapshots[label] = (
                target.read_bytes(),
                target_stat.st_dev,
                target_stat.st_ino,
                target_stat.st_size,
                target_stat.st_mtime_ns,
            )
            link_snapshots[label] = _repair115_entry_snapshot(final_path)
        else:
            current = final_path.stat()
            ordinary_snapshots[label] = (
                final_path.read_bytes(),
                current.st_dev,
                current.st_ino,
                current.st_size,
                current.st_mtime_ns,
            )

    stage_calls = 0
    real_chart_writer = run_backtest_module._write_owned_chart

    def count_stage(*args, **kwargs):
        nonlocal stage_calls
        stage_calls += 1
        return real_chart_writer(*args, **kwargs)

    monkeypatch.setattr(run_backtest_module, "_write_owned_chart", count_stage)
    with pytest.raises(OSError, match="ordinary regular file"):
        _call_publication_transaction(output, monkeypatch)

    assert stage_calls == 0
    for label, final_path in paths.items():
        if label in linked:
            target = foreign / final_path.name
            current = target.stat()
            assert _repair115_entry_snapshot(final_path) == link_snapshots[label]
            assert (
                target.read_bytes(),
                current.st_dev,
                current.st_ino,
                current.st_size,
                current.st_mtime_ns,
            ) == target_snapshots[label]
        else:
            current = final_path.stat()
            assert (
                final_path.read_bytes(),
                current.st_dev,
                current.st_ino,
                current.st_size,
                current.st_mtime_ns,
            ) == ordinary_snapshots[label]


@pytest.mark.parametrize("linked", [("report",), ("chart",), ("report", "chart")])
def test_repair115_broken_final_symlinks_fail_before_staging(
    tmp_path, monkeypatch, linked
) -> None:
    output = tmp_path / "repair115-broken-output"
    foreign = tmp_path / "FOREIGN-PARKED"
    foreign.mkdir()
    _call_publication_transaction(output, monkeypatch)
    paths = {
        "report": output / "transaction-probe.json",
        "chart": output / "transaction-probe.png",
    }
    parked_snapshots = {}
    link_snapshots = {}
    for label, final_path in paths.items():
        if label in linked:
            parked = foreign / final_path.name
            run_backtest_module.os.replace(final_path, parked)
            parked_stat = parked.stat()
            parked_snapshots[label] = (
                parked.read_bytes(),
                parked_stat.st_dev,
                parked_stat.st_ino,
                parked_stat.st_size,
                parked_stat.st_mtime_ns,
            )
            run_backtest_module.os.symlink(foreign / f"MISSING-{label}", final_path)
            link_snapshots[label] = _repair115_entry_snapshot(final_path)

    stage_calls = 0
    real_chart_writer = run_backtest_module._write_owned_chart

    def count_stage(*args, **kwargs):
        nonlocal stage_calls
        stage_calls += 1
        return real_chart_writer(*args, **kwargs)

    monkeypatch.setattr(run_backtest_module, "_write_owned_chart", count_stage)
    with pytest.raises(OSError, match="ordinary regular file"):
        _call_publication_transaction(output, monkeypatch)

    assert stage_calls == 0
    for label in linked:
        final_path = paths[label]
        parked = foreign / final_path.name
        current = parked.stat()
        assert _repair115_entry_snapshot(final_path) == link_snapshots[label]
        assert (
            parked.read_bytes(),
            current.st_dev,
            current.st_ino,
            current.st_size,
            current.st_mtime_ns,
        ) == parked_snapshots[label]


def test_repair115_ordinary_regular_pair_remains_idempotent(
    tmp_path, monkeypatch
) -> None:
    output = tmp_path / "repair115-ordinary-output"
    first = _call_publication_transaction(output, monkeypatch)
    before = _output_snapshot(output)

    second = _call_publication_transaction(output, monkeypatch)

    assert first == second
    assert _output_snapshot(output) == before


def test_repair115_final_entry_mutation_sentinels(tmp_path) -> None:
    target = tmp_path / "FOREIGN-FINAL.json"
    target.write_bytes(b"FOREIGN-EXACT")
    final = tmp_path / "report.json"
    run_backtest_module.os.symlink(target, final)
    link_before = _repair115_entry_snapshot(final)
    target_before = target.stat()
    with pytest.raises(OSError, match="ordinary regular file"):
        run_backtest_module._ordinary_final_identity_if_present(final)
    target_after = target.stat()
    assert _repair115_entry_snapshot(final) == link_before
    assert target.read_bytes() == b"FOREIGN-EXACT"
    assert (target_after.st_dev, target_after.st_ino, target_after.st_mtime_ns) == (
        target_before.st_dev, target_before.st_ino, target_before.st_mtime_ns
    )


def test_repair120_signature_open_rejects_path_swap_without_touching_objects(
    tmp_path, monkeypatch
) -> None:
    artifact = tmp_path / "approved.bin"
    artifact.write_bytes(b"APPROVED-EXACT")
    approved_identity = run_backtest_module._ordinary_final_identity_if_present(
        artifact
    )
    parked = tmp_path / "approved-parked.bin"
    foreign = tmp_path / "FOREIGN.bin"
    foreign.write_bytes(b"FOREIGN-MUST-STAY-EXACT")
    foreign_before = foreign.stat()
    real_open = run_backtest_module._open_owned_read_stream
    injected = False

    def swap_before_owned_open(path, expected_identity):
        nonlocal injected
        if not injected:
            injected = True
            run_backtest_module.os.replace(path, parked)
            run_backtest_module.os.symlink(foreign, path)
        return real_open(path, expected_identity)

    monkeypatch.setattr(
        run_backtest_module, "_open_owned_read_stream", swap_before_owned_open
    )
    with pytest.raises((FileExistsError, OSError)):
        run_backtest_module._file_signature(artifact, approved_identity)

    foreign_after = foreign.stat()
    assert injected
    assert artifact.is_symlink()
    assert parked.read_bytes() == b"APPROVED-EXACT"
    assert foreign.read_bytes() == b"FOREIGN-MUST-STAY-EXACT"
    assert (
        foreign_after.st_dev,
        foreign_after.st_ino,
        foreign_after.st_size,
        foreign_after.st_mtime_ns,
    ) == (
        foreign_before.st_dev,
        foreign_before.st_ino,
        foreign_before.st_size,
        foreign_before.st_mtime_ns,
    )


def test_repair120_large_idempotent_pair_scans_each_final_once_without_read_bytes(
    tmp_path, monkeypatch
) -> None:
    output = tmp_path / "repair120-large-output"
    report = {
        "report_schema": "backtest-report-v2",
        "generated_at": "2026-07-19T00:00:00Z",
        "payload": "R" * (3 * 1024 * 1024 + 31),
    }
    chart = b"C" * (3 * 1024 * 1024 + 19)
    monkeypatch.setattr(
        run_backtest_module,
        "_write_equity_chart",
        lambda _result, stream, _label: stream.write(chart),
    )
    first = run_backtest_module._publish_output_set(
        report=report,
        result=object(),
        output_dir=output,
        stem="large",
        mode_label="probe",
    )
    finals = {output / "large.json", output / "large.png"}
    before = {}
    for path in finals:
        current = path.stat()
        with path.open("rb") as stream:
            payload = stream.read()
        before[path] = (
            current.st_dev,
            current.st_ino,
            current.st_size,
            current.st_mtime_ns,
            run_backtest_module.hashlib.sha256(payload).digest(),
        )
    signature_counts = {path: 0 for path in finals}
    full_reads = []
    opened_paths = []
    read_requests = {path: [] for path in finals}
    total_bytes = {path: 0 for path in finals}
    real_signature = run_backtest_module._file_signature
    real_open = run_backtest_module._open_owned_read_stream
    real_read_bytes = Path.read_bytes

    def count_signature(path, *args, **kwargs):
        path = Path(path)
        if path in signature_counts:
            signature_counts[path] += 1
        return real_signature(path, *args, **kwargs)

    def count_full_read(path, *args, **kwargs):
        path = Path(path)
        if path in finals:
            full_reads.append((path.name, path.stat().st_size))
        return real_read_bytes(path, *args, **kwargs)

    class ReadRecorder:
        def __init__(self, path, stream):
            self.path = path
            self.stream = stream

        def __getattr__(self, name):
            return getattr(self.stream, name)

        def read(self, size=-1):
            data = self.stream.read(size)
            if self.path in read_requests:
                read_requests[self.path].append(size)
                total_bytes[self.path] += len(data)
            return data

        def __enter__(self):
            self.stream.__enter__()
            return self

        def __exit__(self, *args):
            return self.stream.__exit__(*args)

    def record_open(path, expected_identity):
        path = Path(path)
        opened_paths.append(path)
        return ReadRecorder(path, real_open(path, expected_identity))

    monkeypatch.setattr(run_backtest_module, "_file_signature", count_signature)
    monkeypatch.setattr(run_backtest_module, "_open_owned_read_stream", record_open)
    monkeypatch.setattr(Path, "read_bytes", count_full_read)
    rerun_report = dict(report)
    rerun_report["generated_at"] = "2099-01-01T00:00:00Z"
    second = run_backtest_module._publish_output_set(
        report=rerun_report,
        result=object(),
        output_dir=output,
        stem="large",
        mode_label="probe",
    )

    assert first == second
    final_sizes = {path: path.stat().st_size for path in finals}
    assert all(size >= 3 * 1024 * 1024 for size in final_sizes.values()), final_sizes
    assert full_reads == []
    assert signature_counts == {path: 1 for path in finals}
    assert set(opened_paths) == finals
    for path in finals:
        assert read_requests[path]
        assert all(0 < request <= 1024 * 1024 for request in read_requests[path])
        assert total_bytes[path] == final_sizes[path]
        current = path.stat()
        with path.open("rb") as stream:
            payload = stream.read()
        assert (
            current.st_dev,
            current.st_ino,
            current.st_size,
            current.st_mtime_ns,
            run_backtest_module.hashlib.sha256(payload).digest(),
        ) == before[path]
    with (output / "large.json").open("rb") as stream:
        published_report = json.load(stream)
    assert published_report["generated_at"] == "2026-07-19T00:00:00Z"


def test_repair120_signature_reader_uses_bounded_chunks(
    tmp_path, monkeypatch
) -> None:
    artifact = tmp_path / "large-artifact.bin"
    artifact.write_bytes(b"A" * (3 * 1024 * 1024 + 17))
    identity = run_backtest_module._ordinary_final_identity_if_present(artifact)
    read_requests = []
    real_open = run_backtest_module._open_owned_read_stream

    class ReadRecorder:
        def __init__(self, stream):
            self.stream = stream

        def __getattr__(self, name):
            return getattr(self.stream, name)

        def read(self, size=-1):
            read_requests.append(size)
            return self.stream.read(size)

        def __enter__(self):
            self.stream.__enter__()
            return self

        def __exit__(self, *args):
            return self.stream.__exit__(*args)

    monkeypatch.setattr(
        run_backtest_module,
        "_open_owned_read_stream",
        lambda path, expected: ReadRecorder(real_open(path, expected)),
    )
    signature = run_backtest_module._file_signature(artifact)

    assert signature[2] == artifact.stat().st_size
    assert read_requests
    assert all(0 < request <= 1024 * 1024 for request in read_requests)
