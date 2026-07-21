from __future__ import annotations

import importlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_dead_helpers_and_compat_config_are_removed() -> None:
    ops_source = (ROOT / "model_core" / "ops.py").read_text(encoding="utf-8")

    assert "def _ema(" not in ops_source
    assert not (ROOT / "strategy_manager" / "config.py").exists()


def test_core_modules_import_without_historical_entrypoints() -> None:
    modules = (
        "model_core.alphagpt",
        "model_core.artifacts",
        "model_core.backtest",
        "model_core.engine",
        "model_core.execution",
        "model_core.features",
        "model_core.formula_evaluation",
        "model_core.ops",
        "model_core.parallel_evaluator",
        "model_core.vm",
        "strategy_manager.runner",
        "strategy_manager.signal",
    )

    for module_name in modules:
        assert importlib.import_module(module_name).__name__ == module_name


def test_obsolete_root_entrypoints_are_archived_and_marked_non_core() -> None:
    archived = (
        "analyze_ckpt.py",
        "backtest_all_groups.py",
        "benchmark_speed.py",
        "run_v1_backtest.py",
        "train_ftmo.py",
        "train_index.py",
        "verify_all_strategies.py",
    )
    notice = (ROOT / "extras" / "experiments" / "README.md").read_text(
        encoding="utf-8"
    )

    for filename in archived:
        assert not (ROOT / filename).exists()
        assert (ROOT / "extras" / "experiments" / filename).is_file()
    assert "不是 Core Correctness V2 正式入口" in notice


def test_current_docs_reject_legacy_artifacts_without_deleting_them() -> None:
    strategy_docs = (ROOT / "strategies" / "README.md").read_text(encoding="utf-8")
    history_docs = (ROOT / "extras" / "history" / "README.md").read_text(
        encoding="utf-8"
    )

    assert "明确拒绝加载" in strategy_docs
    assert "不会删除、重命名、重标或静默迁移" in strategy_docs
    assert "不是当前功能合同" in history_docs


def test_core_sources_have_no_unqualified_dead_code_ignore() -> None:
    for package in ("model_core", "strategy_manager"):
        for path in (ROOT / package).glob("*.py"):
            for line_number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                if "# noqa" in line:
                    assert "# noqa:" in line, f"{path}:{line_number} has a bare noqa"
