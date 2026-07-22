from __future__ import annotations

import os
import multiprocessing
import copy
import random
from concurrent.futures.process import BrokenProcessPool
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from model_core.backtest import MT5Backtest
from model_core.formula_evaluation import (
    RawEvaluationContext,
    ReferenceCpuEvaluator,
)
from model_core.parallel_evaluator import ParallelCpuEvaluator
import model_core.parallel_evaluator as parallel_module
from model_core.vm import StackVM
from model_core.walk_forward import WalkForwardFold
from tests.support.core00 import (
    load_formula_corpus,
    load_ohlcv_fixture,
    resolve_formula_tokens,
)
import model_core.engine as engine_module
from model_core.config import ModelConfig
from model_core.engine import AlphaEngine
from model_core.artifacts import TrainingRunIdentity
from tests.unit.test_training_alignment import (
    _TEST_ARTIFACT_IDENTITY,
    _TEST_TRAINING_RUN_ID,
    _TinyPolicy,
    _TinySampler,
)
from benchmarks.perf02 import PERF02_BENCHMARK_SCHEMA_VERSION, run_benchmark


def _context() -> tuple[RawEvaluationContext, list[list[int]]]:
    fixture = load_ohlcv_fixture()
    features = fixture.feature_tensor()
    target = torch.tensor(
        [[0.01, -0.02, 0.015, -0.01, 0.005] * 8],
        dtype=features.dtype,
    )
    valid = torch.ones_like(target, dtype=torch.bool)
    valid[:, -1] = False
    fold = WalkForwardFold(
        fold_index=0,
        train_start=0,
        train_end=16,
        val_start=18,
        val_end=34,
        effective_gap=2,
    )
    formulas = [
        resolve_formula_tokens(row["tokens"])
        for row in load_formula_corpus()
        if row["classification"] == "valid"
    ]
    return (
        RawEvaluationContext(
            features=features,
            target_ret=target,
            target_valid=valid,
            bar_time_ns=fixture.tensors["time_ns"],
            folds=(fold,),
            selection_index=torch.arange(0, 34, dtype=torch.long),
            cost_rate=0.001,
            timeframe="H1",
            target_trades_per_day=2.0,
            oos_gate_scale=0.5,
        ),
        formulas,
    )


def _force_worker_input_mutation() -> None:
    evaluator = parallel_module._WORKER_EVALUATOR
    if evaluator is None:
        raise RuntimeError("worker evaluator is unavailable")
    evaluator.context.features.zero_()


def _assert_raw_equal(left, right) -> None:
    assert left.formula_index == right.formula_index
    assert left.formula == right.formula
    assert left.constant == right.constant
    assert left.error == right.error
    assert left.selection_ic == right.selection_ic
    assert left.selection_ic_stability == right.selection_ic_stability
    assert left.exposure == right.exposure
    assert left.factor is not None and right.factor is not None
    assert torch.equal(left.factor, right.factor)
    assert len(left.fold_train_scores) == len(right.fold_train_scores)
    assert all(
        torch.equal(a, b)
        for a, b in zip(left.fold_train_scores, right.fold_train_scores)
    )
    assert all(
        torch.equal(a, b)
        for a, b in zip(left.fold_val_scores, right.fold_val_scores)
    )
    assert left.fold_ics == right.fold_ics


def test_reference_and_spawn_parallel_raw_results_are_exact_and_ordered() -> None:
    context, formulas = _context()
    reference = ReferenceCpuEvaluator(context, vm=StackVM(), backtest=MT5Backtest(
        cost_rate=context.cost_rate,
        timeframe=context.timeframe,
        target_trades_per_day=context.target_trades_per_day,
        oos_gate_scale=context.oos_gate_scale,
    ))
    expected = reference.evaluate_batch(formulas, step=7)

    with ParallelCpuEvaluator(context, workers=2, timeout_seconds=30.0) as parallel:
        actual = parallel.evaluate_batch(formulas, step=7)
        worker_pids = parallel.worker_pids

    assert [item.formula_index for item in actual] == list(range(len(formulas)))
    assert len(worker_pids) == 2
    assert all(pid != os.getpid() for pid in worker_pids)
    for left, right in zip(expected, actual):
        _assert_raw_equal(left, right)


@pytest.mark.parametrize("workers", [1, 2])
def test_parallel_evaluator_reuses_workers_and_closes_them(workers: int) -> None:
    context, formulas = _context()
    evaluator = ParallelCpuEvaluator(context, workers=workers, timeout_seconds=30.0)
    first = evaluator.evaluate_batch(formulas[:2], step=1)
    first_pids = evaluator.worker_pids
    second = evaluator.evaluate_batch(formulas[2:4], step=2)

    assert first and second
    assert set(first_pids).issubset(evaluator.worker_pids)
    assert len(evaluator.worker_pids) <= workers
    evaluator.close()
    assert evaluator.closed is True
    assert evaluator.worker_pids == ()


def test_chunked_parallel_evaluation_is_exact_and_ordered() -> None:
    context, formulas = _context()
    formulas = formulas * 3
    expected = ReferenceCpuEvaluator(context).evaluate_batch(formulas, step=9)

    with ParallelCpuEvaluator(
        context,
        workers=2,
        timeout_seconds=30.0,
        chunk_size=4,
    ) as evaluator:
        actual = evaluator.evaluate_batch(formulas, step=9)

    assert [item.formula_index for item in actual] == list(range(len(formulas)))
    for left, right in zip(expected, actual):
        _assert_raw_equal(left, right)


@pytest.mark.parametrize("chunk_size", [0, -1, 1.5, True])
def test_parallel_evaluator_rejects_invalid_chunk_size(chunk_size: object) -> None:
    context, _ = _context()
    with pytest.raises(ValueError, match="chunk_size"):
        ParallelCpuEvaluator(context, workers=2, chunk_size=chunk_size)  # type: ignore[arg-type]


def test_parallel_evaluator_rejects_non_context_object() -> None:
    with pytest.raises(TypeError, match="exact RawEvaluationContext"):
        ParallelCpuEvaluator(object(), workers=2)


def test_parallel_context_is_detached_from_later_caller_mutation() -> None:
    context, formulas = _context()
    expected = ReferenceCpuEvaluator(context).evaluate_batch(formulas[:1], step=1)
    evaluator = ParallelCpuEvaluator(context, workers=1, timeout_seconds=30.0)
    context.features.zero_()

    actual = evaluator.evaluate_batch(formulas[:1], step=1)
    evaluator.close()

    _assert_raw_equal(expected[0], actual[0])


def test_main_process_keyboard_interrupt_aborts_pool_without_orphans(
    monkeypatch,
) -> None:
    context, formulas = _context()
    evaluator = ParallelCpuEvaluator(context, workers=2, timeout_seconds=30.0)
    evaluator.evaluate_batch(formulas, step=1)
    worker_pids = set(evaluator.worker_pids)

    def interrupt(*_args, **_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(parallel_module.concurrent.futures, "wait", interrupt)
    with pytest.raises(KeyboardInterrupt):
        evaluator.evaluate_batch(formulas, step=2)

    assert evaluator.closed is True
    assert evaluator.worker_pids == ()
    assert worker_pids.isdisjoint(
        process.pid for process in multiprocessing.active_children()
    )


def test_batch_timeout_aborts_pool_without_partial_results_or_orphans() -> None:
    context, formulas = _context()
    evaluator = ParallelCpuEvaluator(
        context,
        workers=2,
        timeout_seconds=1e-9,
    )

    with pytest.raises(TimeoutError, match="batch timed out"):
        evaluator.evaluate_batch(formulas * 10, step=1)

    assert evaluator.closed is True
    assert evaluator.worker_pids == ()


def test_unexpected_worker_exit_fails_whole_batch_and_closes_pool() -> None:
    context, formulas = _context()
    evaluator = ParallelCpuEvaluator(context, workers=2, timeout_seconds=30.0)
    evaluator.evaluate_batch(formulas, step=1)
    processes = tuple(evaluator._executor._processes.values())
    processes[0].terminate()
    processes[0].join(timeout=5.0)

    with pytest.raises(BrokenProcessPool):
        evaluator.evaluate_batch(formulas, step=2)

    assert evaluator.closed is True
    assert evaluator.worker_pids == ()
    assert all(not process.is_alive() for process in processes)


def test_worker_input_mutation_cannot_change_main_process_source_tensor() -> None:
    context, formulas = _context()
    original = context.features.clone()
    evaluator = ParallelCpuEvaluator(context, workers=1, timeout_seconds=30.0)
    evaluator.evaluate_batch(formulas[:1], step=1)

    evaluator._executor.submit(_force_worker_input_mutation).result(timeout=30.0)
    evaluator.close()

    assert torch.equal(context.features, original)


@pytest.mark.skipif(
    "forkserver" not in multiprocessing.get_all_start_methods(),
    reason="forkserver is unavailable on this platform",
)
def test_forkserver_matches_spawn_when_platform_supports_it() -> None:
    context, formulas = _context()
    expected = ReferenceCpuEvaluator(context).evaluate_batch(formulas, step=1)
    with ParallelCpuEvaluator(
        context,
        workers=2,
        timeout_seconds=30.0,
        start_method="forkserver",
    ) as evaluator:
        actual = evaluator.evaluate_batch(formulas, step=1)

    for left, right in zip(expected, actual):
        _assert_raw_equal(left, right)


def _run_engine_trace(
    monkeypatch,
    tmp_path,
    workers: int | None,
    *,
    torch_threads: int = 1,
    chunk_size: int = 1,
) -> dict:
    context, _ = _context()
    folds = (
        WalkForwardFold(0, 0, 10, 12, 20, 2),
        WalkForwardFold(1, 0, 20, 22, 34, 2),
    )
    monkeypatch.setattr(engine_module, "formula_warmup_bars", lambda _length: 1)
    monkeypatch.setattr(engine_module, "required_training_bars", lambda **_kwargs: 1)
    monkeypatch.setattr(engine_module, "assert_minimum_bars", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        engine_module,
        "build_walk_forward_folds",
        lambda **_kwargs: list(folds),
    )
    monkeypatch.setattr(ModelConfig, "BATCH_SIZE", 4)
    monkeypatch.setattr(ModelConfig, "ELITE_REPLAY_FRAC", 0.25)
    monkeypatch.setattr(ModelConfig, "MAX_FORMULA_LEN", 1)
    monkeypatch.setattr(ModelConfig, "WF_N_BLOCKS", len(folds))
    monkeypatch.setattr(ModelConfig, "ENTROPY_COLLAPSE_THRESH", -1.0)
    monkeypatch.setattr(ModelConfig, "EVALUATION_TORCH_THREADS", torch_threads)
    monkeypatch.setattr(ModelConfig, "EVALUATION_CHUNK_SIZE", chunk_size)
    history_path = tmp_path / f"history-{workers}.json"
    monkeypatch.setattr(
        TrainingRunIdentity,
        "history_filename",
        lambda _identity: str(history_path),
    )
    monkeypatch.setattr(engine_module, "_CHECKPOINT_DIR", tmp_path)
    monkeypatch.setattr(
        engine_module,
        "_publish_training_history",
        lambda _engine: history_path,
    )
    monkeypatch.setattr(
        engine_module,
        "_save_checkpoint",
        lambda _engine, step: tmp_path / f"checkpoint-{workers}-{step}.pt",
    )

    engine = AlphaEngine.__new__(AlphaEngine)
    engine.run_identity = TrainingRunIdentity(
        run_id=_TEST_TRAINING_RUN_ID,
        artifact_identity=_TEST_ARTIFACT_IDENTITY,
    )
    engine.data_manager = SimpleNamespace(
        feat_tensor=context.features.clone(),
        target_ret=context.target_ret.clone(),
        target_valid=context.target_valid.clone(),
        bar_time=context.bar_time_ns.clone(),
    )
    engine.n_folds = len(folds)
    engine.target_symbol = None
    engine.evaluation_workers = workers
    engine._active_formula_evaluator = None
    engine.model = _TinyPolicy()
    engine.opt = torch.optim.SGD(engine.model.parameters(), lr=0.01)
    engine.sampler = _TinySampler()
    engine.vm = StackVM()
    config = _TEST_ARTIFACT_IDENTITY.training_config
    reward = config["reward"]
    timeframe = config["timeframe_reward"]
    engine._reward_config = dict(reward)
    lord = config["lord"]
    engine.use_lord_regularization = lord["use_lord_regularization"]
    engine.lord_decay_rate = lord["lord_decay_rate"]
    engine.lord_num_iterations = lord["lord_num_iterations"]
    engine.bt = MT5Backtest(
        cost_rate=float(config["cost_rate"]),
        timeframe=str(timeframe["timeframe"]),
        target_trades_per_day=float(timeframe["target_trades_per_day"]),
        oos_gate_scale=float(reward["oos_gate_scale"]),
    )
    engine.use_lord = False
    engine.lord_opt = None
    engine.rank_monitor = None
    engine.best_score = -float("inf")
    engine.best_formula = None
    engine.best_metrics = None
    engine._best_snapshot = None
    engine.training_history = {
        "step": [], "avg_reward": [], "best_score": [],
        "val_score": [], "stable_rank": [],
    }
    engine._restart_count = 0
    engine.factor_pool = []
    engine.factor_pool_scores = []
    engine._factor_pool_counter = 0
    engine._elite_pool = []
    engine.elite_pool_ages = []
    engine._elite_counter = 0
    engine._best_update_step = 0
    engine._stagnation_steps = 0
    engine._reward_ema = None
    engine._reward_ema_step = 0
    engine._low_entropy_streak = 0
    engine._previous_initial_distribution = None
    engine._owned_public_artifacts = {}
    engine._distribution_stats = lambda _previous: {
        "dist": torch.tensor([0.5, 0.5]),
        "entropy": 0.5,
        "kl_uniform": 0.0,
        "top1_prob": 0.5,
        "top5_prob": 1.0,
        "eff_vocab": 2.0,
        "prob_std": 0.0,
        "kl_prev": 0.0,
    }
    engine._save_strategy_live = lambda *_args: None
    engine._decode_formula = lambda formula: str(formula)

    random.seed(20260722)
    np.random.seed(20260722)
    torch.manual_seed(20260722)
    engine.train(end_step=2, verbose_header=False)
    trace = {
        "history": copy.deepcopy(engine.training_history),
        "best": (engine.best_score, copy.deepcopy(engine.best_formula)),
        "elite": copy.deepcopy(engine._elite_pool),
        "factor_pool": [
            (score, counter, factor.clone())
            for score, counter, factor in engine.factor_pool
        ],
        "model": {
            name: value.detach().clone()
            for name, value in engine.model.state_dict().items()
        },
        "optimizer": copy.deepcopy(engine.opt.state_dict()),
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state().clone(),
    }
    engine._close_active_formula_evaluator()
    return trace


def _assert_engine_trace_exact(left: dict, right: dict) -> None:
    assert left["history"] == right["history"]
    assert left["best"] == right["best"]
    assert left["elite"] == right["elite"]
    assert left["python_rng"] == right["python_rng"]
    assert left["numpy_rng"][0] == right["numpy_rng"][0]
    assert np.array_equal(left["numpy_rng"][1], right["numpy_rng"][1])
    assert left["numpy_rng"][2:] == right["numpy_rng"][2:]
    assert torch.equal(left["torch_rng"], right["torch_rng"])
    assert left["optimizer"] == right["optimizer"]
    assert left["model"].keys() == right["model"].keys()
    assert all(
        torch.equal(left["model"][name], right["model"][name])
        for name in left["model"]
    )
    assert len(left["factor_pool"]) == len(right["factor_pool"])
    for (ls, lc, lf), (rs, rc, rf) in zip(
        left["factor_pool"], right["factor_pool"]
    ):
        assert (ls, lc) == (rs, rc)
        assert torch.equal(lf, rf)


@pytest.mark.parametrize("workers", [1, 2, 4, 8])
def test_engine_trace_is_exact_across_reference_and_worker_counts(
    monkeypatch,
    tmp_path,
    workers: int,
) -> None:
    reference = _run_engine_trace(monkeypatch, tmp_path / "reference", None)
    actual = _run_engine_trace(monkeypatch, tmp_path / f"workers-{workers}", workers)

    _assert_engine_trace_exact(reference, actual)


def test_engine_trace_is_exact_with_worker_threads_and_chunking(
    monkeypatch,
    tmp_path,
) -> None:
    reference = _run_engine_trace(monkeypatch, tmp_path / "reference-mixed", None)
    actual = _run_engine_trace(
        monkeypatch,
        tmp_path / "workers-mixed",
        4,
        torch_threads=2,
        chunk_size=2,
    )

    _assert_engine_trace_exact(reference, actual)


def test_perf02_benchmark_recommends_only_measured_exact_variant(tmp_path) -> None:
    output = tmp_path / "perf02.json"
    report = run_benchmark(
        worker_counts=(1, 2),
        iterations=1,
        output_path=output,
    )

    assert output.exists()
    assert report["schema_version"] == PERF02_BENCHMARK_SCHEMA_VERSION
    assert report["required_ci"] is False
    assert report["parity"]["exact"] is True
    assert report["recommended_workers"] in (1, 2)
    assert report["recommendation_candidates"] == [1, 2]
    assert set(report["variants"]) == {"1", "2"}
    assert all(item["exact_parity"] for item in report["variants"].values())
