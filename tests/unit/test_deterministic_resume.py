from __future__ import annotations

import random

import numpy as np
import torch

from model_core.backtest import MT5Backtest
from model_core.engine import AlphaEngine
from tests.unit.test_training_alignment import _failure_training_engine


def test_deterministic_resume_drives_real_alphaengine_train(
    monkeypatch, tmp_path
) -> None:
    calls = []
    real_train = AlphaEngine.train

    def traced_train(engine, *args, **kwargs):
        calls.append((kwargs.get("start_step", 0), kwargs.get("end_step")))
        return real_train(engine, *args, **kwargs)

    monkeypatch.setattr(AlphaEngine, "train", traced_train)
    engine = _actual_training_engine(monkeypatch, tmp_path)
    engine.train(start_step=0, end_step=1, verbose_header=False)

    assert calls == [(0, 1)]
    assert engine.training_history["step"] == [0]


def _assert_nested_equal(left, right) -> None:
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        assert np.array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _assert_nested_equal(a, b)
    else:
        assert left == right


def _actual_training_engine(monkeypatch, tmp_path) -> AlphaEngine:
    engine, factor, _calls, _before = _failure_training_engine(
        monkeypatch, tmp_path, lambda _formula, _features: factor.clone()
    )
    return engine


def _training_state(engine: AlphaEngine) -> dict:
    return {
        "model": engine.model.state_dict(),
        "optimizer": engine.opt.state_dict(),
        "best": (
            engine.best_score, engine.best_formula, engine.best_metrics,
            engine._best_snapshot, engine._best_update_step,
            engine._stagnation_steps,
        ),
        "factor_pool": (
            engine.factor_pool, engine.factor_pool_scores,
            engine._factor_pool_counter,
        ),
        "elite_pool": (
            engine._elite_pool, engine.elite_pool_ages, engine._elite_counter,
        ),
        "ema": (engine._reward_ema, engine._reward_ema_step),
        "entropy": (
            engine._low_entropy_streak,
            engine._previous_initial_distribution,
        ),
        "histories": (
            engine.training_history,
            [] if engine.rank_monitor is None else engine.rank_monitor.history,
        ),
        "counters": (
            engine._restart_count, engine._best_update_step,
            engine._stagnation_steps,
        ),
    }


def test_continuous_and_resumed_state_are_bit_exact(monkeypatch, tmp_path) -> None:
    torch.use_deterministic_algorithms(True)
    random.seed(123); np.random.seed(123); torch.manual_seed(123)
    continuous = _actual_training_engine(monkeypatch, tmp_path / "continuous")
    continuous.train(start_step=0, end_step=4, verbose_header=False)
    continuous_next = (random.random(), np.random.random(), torch.rand(2))

    random.seed(123); np.random.seed(123); torch.manual_seed(123)
    split = _actual_training_engine(monkeypatch, tmp_path / "split")
    split.train(start_step=0, end_step=2, verbose_header=False)
    path = tmp_path / "split-v2.pt"
    AlphaEngine.save_checkpoint(split, 1, str(path))
    resumed = _actual_training_engine(monkeypatch, tmp_path / "resumed")
    assert AlphaEngine.load_checkpoint(resumed, str(path)) == 2
    resumed.train(start_step=2, end_step=4, verbose_header=False)
    resumed_next = (random.random(), np.random.random(), torch.rand(2))
    _assert_nested_equal(_training_state(continuous), _training_state(resumed))
    assert continuous_next[0] == resumed_next[0]
    assert continuous_next[1] == resumed_next[1]
    assert torch.equal(continuous_next[2], resumed_next[2])


def _promoted_execution_training_engine(monkeypatch, tmp_path) -> AlphaEngine:
    engine = _actual_training_engine(monkeypatch, tmp_path)
    factors = torch.linspace(-0.8, 0.8, 20, dtype=torch.float32).unsqueeze(0)
    factors[0, 14] = torch.tensor(
        np.arctanh(0.37995398044586182), dtype=torch.float32
    )
    factors[0, 15] = 0.40000590682029724
    target_ret = torch.tensor(
        [[
            -0.002,
            0.001,
            -0.0015,
            0.0012,
            -0.001,
            0.001,
            -0.002,
            0.001,
            -0.001,
            0.002,
            -0.001,
            0.001,
            -0.002,
            0.001,
            0.0,
            0.0010838214075192809,
            -0.002,
            0.001,
            0.0,
            0.0,
        ]],
        dtype=torch.float32,
    )
    engine.data_manager.target_ret = target_ret
    engine.vm.execute = lambda _formula, _features: factors.clone()
    engine.bt = MT5Backtest(cost_rate=1.0e-4)
    return engine


def test_deterministic_resume_reaches_formerly_failing_fourth_execution_step(
    monkeypatch, tmp_path
) -> None:
    torch.use_deterministic_algorithms(True)
    random.seed(321)
    np.random.seed(321)
    torch.manual_seed(321)
    continuous = _promoted_execution_training_engine(
        monkeypatch, tmp_path / "promoted-continuous"
    )
    continuous.train(start_step=0, end_step=4, verbose_header=False)

    random.seed(321)
    np.random.seed(321)
    torch.manual_seed(321)
    split = _promoted_execution_training_engine(
        monkeypatch, tmp_path / "promoted-split"
    )
    split.train(start_step=0, end_step=2, verbose_header=False)
    checkpoint = tmp_path / "promoted-split-v2.pt"
    AlphaEngine.save_checkpoint(split, 1, str(checkpoint))
    resumed = _promoted_execution_training_engine(
        monkeypatch, tmp_path / "promoted-resumed"
    )
    assert AlphaEngine.load_checkpoint(resumed, str(checkpoint)) == 2
    resumed.train(start_step=2, end_step=4, verbose_header=False)

    assert continuous.training_history["step"] == [0, 1, 2, 3]
    assert resumed.training_history["step"] == [0, 1, 2, 3]
    _assert_nested_equal(_training_state(continuous), _training_state(resumed))
