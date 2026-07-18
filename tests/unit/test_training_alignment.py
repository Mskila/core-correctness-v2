"""Training alignment and fail-closed entry contracts."""

import ctypes
import inspect
import copy
import gc
import json
import math
import os
import pathlib
import random
import stat
import subprocess
import sys
import textwrap
import time
import threading
import weakref
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

import model_core.engine as engine_module
import model_core.backtest as backtest_module
from model_core.backtest import MT5Backtest
from model_core.artifacts import TrainingRunIdentity
from model_core.config import ModelConfig
from model_core.engine import AlphaEngine
from model_core.island_engine import IslandAlphaEngine
from model_core.semantics import (
    ArtifactCompatibilityError,
    DataValidationError,
    InsufficientWalkForwardDataError,
)
from model_core.vm import StackVM
from model_core.walk_forward import WalkForwardFold
from tests.unit.test_artifacts import artifact_identity


engine_module._test_strategy_file_for_symbol = lambda symbol: str(
    engine_module.pathlib.Path("strategies") / f"best_{symbol}.json"
)


_TEST_TRAINING_RUN_ID = "7" * 32
_TEST_ARTIFACT_IDENTITY = artifact_identity()


def _install_formal_training_identity(engine: AlphaEngine) -> AlphaEngine:
    engine.run_identity = TrainingRunIdentity(
        run_id=_TEST_TRAINING_RUN_ID,
        artifact_identity=_TEST_ARTIFACT_IDENTITY,
    )
    for field, value in (
        ("best_metrics", None),
        ("factor_pool_scores", []),
        ("elite_pool_ages", []),
        ("_previous_initial_distribution", None),
        ("rank_monitor", None),
    ):
        if not hasattr(engine, field):
            setattr(engine, field, value)
    return engine


def _run_isolated_engine_import(script_body: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script_body)],
        cwd=str(engine_module.pathlib.Path(__file__).resolve().parents[2]),
        text=True,
        capture_output=True,
        check=False,
    )


def test_fresh_engine_import_uses_exact_shared_execution_semantics() -> None:
    result = _run_isolated_engine_import(
        """
        import torch
        import model_core.engine as engine
        from strategy_manager.signal import compute_target_positions_stateless as shared

        factors = torch.tensor([0.01, -0.01])
        actual = engine.compute_target_positions_stateless(factors)
        expected = shared(factors)
        assert torch.equal(actual, expected)
        assert actual.tolist() == [0.0, 0.0]
        print("SHARED_EXECUTION_EXACT")
        """
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "SHARED_EXECUTION_EXACT"


def test_fresh_engine_import_fails_closed_when_shared_execution_is_missing() -> None:
    result = _run_isolated_engine_import(
        """
        import builtins

        original_import = builtins.__import__
        def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "strategy_manager.signal":
                raise ImportError("forced missing strategy_manager.signal")
            return original_import(name, globals, locals, fromlist, level)
        builtins.__import__ = blocked_import

        try:
            import model_core.engine
        except ImportError as exc:
            assert "forced missing strategy_manager.signal" in str(exc)
            print("FAILED_CLOSED")
        else:
            raise SystemExit("engine installed a substitute execution algorithm")
        """
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "FAILED_CLOSED"


def _valid_training_manager(*, symbols: int = 1, bars: int = 30):
    return SimpleNamespace(
        feat_tensor=torch.zeros(symbols, 3, bars, dtype=torch.float32),
        target_ret=torch.zeros(symbols, bars, dtype=torch.float32),
        target_valid=torch.ones(symbols, bars, dtype=torch.bool),
        bar_time=torch.arange(bars, dtype=torch.int64)
        .unsqueeze(0)
        .expand(symbols, -1),
    )


def _boundary_engine(
    data_manager, *, n_folds: int = ModelConfig.WF_N_BLOCKS
) -> AlphaEngine:
    engine = AlphaEngine.__new__(AlphaEngine)
    _install_formal_training_identity(engine)
    engine.data_manager = data_manager
    engine.n_folds = n_folds
    return engine


@pytest.mark.parametrize(
    ("field", "invalid_value", "message"),
    [
        ("target_ret", torch.zeros(30), "target_ret.*rank 2"),
        ("target_ret", torch.zeros(1, 1, 30), "target_ret.*rank 2"),
        ("target_valid", torch.ones(30, dtype=torch.bool), "target_valid.*rank 2"),
        (
            "target_valid",
            torch.ones(1, 1, 30, dtype=torch.bool),
            "target_valid.*rank 2",
        ),
        ("bar_time", torch.arange(30, dtype=torch.int64), "bar_time.*rank 2"),
        (
            "bar_time",
            torch.arange(30, dtype=torch.int64).reshape(1, 1, 30),
            "bar_time.*rank 2",
        ),
        ("target_valid", torch.ones(1, 29, dtype=torch.bool), "target_valid.*shape"),
        (
            "bar_time",
            torch.arange(29, dtype=torch.int64).unsqueeze(0),
            "bar_time.*shape",
        ),
        ("feat_tensor", torch.zeros(2, 3, 30), "feat_tensor.*N/T"),
        ("feat_tensor", torch.zeros(1, 3, 29), "feat_tensor.*N/T"),
        ("feat_tensor", torch.zeros(1, 3, 30, dtype=torch.int64), "feat_tensor.*floating"),
        ("target_ret", torch.zeros(1, 30, dtype=torch.int64), "target_ret.*floating"),
        (
            "target_valid",
            torch.ones(1, 30, dtype=torch.int64),
            "target_valid.*torch.bool",
        ),
        ("bar_time", torch.arange(30, dtype=torch.float32).unsqueeze(0), "bar_time.*torch.int64"),
        ("target_ret", torch.zeros(1, 30, dtype=torch.float64), "target_ret.*dtype"),
        ("feat_tensor", torch.empty(1, 3, 30, device="meta"), "tensor devices"),
    ],
)
def test_training_validates_complete_tensor_contract_before_work_or_zero_step_return(
    monkeypatch, field: str, invalid_value: torch.Tensor, message: str
) -> None:
    manager = _valid_training_manager()
    setattr(manager, field, invalid_value)
    engine = _boundary_engine(manager)

    def unexpected_work(*_args, **_kwargs):
        raise AssertionError("training work ran before tensor-contract validation")

    monkeypatch.setattr(engine_module, "formula_warmup_bars", unexpected_work)
    monkeypatch.setattr(engine_module, "required_training_bars", unexpected_work)
    monkeypatch.setattr(engine_module, "assert_minimum_bars", unexpected_work)
    monkeypatch.setattr(engine_module, "build_walk_forward_folds", unexpected_work)

    with pytest.raises(DataValidationError, match=message):
        engine.train(end_step=0, verbose_header=False)


def test_training_boundary_does_not_translate_internal_attribute_errors() -> None:
    class BrokenManager:
        @property
        def feat_tensor(self):
            raise AttributeError("internal feature property bug")

        target_ret = torch.zeros(1, 30)
        target_valid = torch.ones(1, 30, dtype=torch.bool)
        bar_time = torch.arange(30, dtype=torch.int64).unsqueeze(0)

    engine = _boundary_engine(BrokenManager())
    with pytest.raises(AttributeError, match="internal feature property bug"):
        engine.train(end_step=0, verbose_header=False)


class _IntSubclass(int):
    pass


def test_default_n_folds_is_derived_from_model_config(monkeypatch) -> None:
    assert ModelConfig.WF_N_BLOCKS == 5
    captured = []

    class ValidationReached(RuntimeError):
        pass

    def capture(value):
        captured.append(value)
        raise ValidationReached

    monkeypatch.setattr(ModelConfig, "WF_N_BLOCKS", 3)
    monkeypatch.setattr(AlphaEngine, "_validate_n_folds", staticmethod(capture))

    with pytest.raises(ValidationReached):
        AlphaEngine()

    assert captured == [3]


@pytest.mark.parametrize(
    "n_folds",
    [None, True, False, -1, 0, 1, 2, 3, 4, 6, 5.0, "5", _IntSubclass(5)],
)
def test_alpha_engine_rejects_divergent_or_invalid_n_folds_before_model_construction(
    monkeypatch, n_folds
) -> None:
    def unexpected_model_construction():
        raise AssertionError("model construction ran before n_folds validation")

    monkeypatch.setattr(engine_module, "AlphaGPT", unexpected_model_construction)

    with pytest.raises(
        InsufficientWalkForwardDataError,
        match=r"n_folds.*expected=5.*actual=",
    ):
        AlphaEngine(n_folds=n_folds)


def test_n_folds_domain_error_message_is_bounded(monkeypatch) -> None:
    def unexpected_model_construction():
        raise AssertionError("model construction ran before n_folds validation")

    monkeypatch.setattr(engine_module, "AlphaGPT", unexpected_model_construction)

    with pytest.raises(InsufficientWalkForwardDataError) as captured:
        AlphaEngine(n_folds="x" * 10_000)

    message = str(captured.value)
    assert "expected=5" in message
    assert "actual=" in message
    assert len(message) <= 200


class _HostileNFolds:
    calls = 0

    def _fail(self):
        type(self).calls += 1
        raise RuntimeError("hostile repr executed")

    __repr__ = lambda self: self._fail()
    __str__ = lambda self: self._fail()
    __int__ = lambda self: self._fail()
    __index__ = lambda self: self._fail()
    __hash__ = lambda self: self._fail()
    __eq__ = lambda self, other: self._fail()


@pytest.mark.parametrize(
    "value",
    [_HostileNFolds(), 10**5000],
    ids=["hostile-protocols", "huge-exact-int"],
)
def test_n_folds_diagnostics_never_execute_protocols_or_render_huge_ints(
    monkeypatch, value
) -> None:
    _HostileNFolds.calls = 0
    monkeypatch.setattr(
        engine_module,
        "AlphaGPT",
        lambda: (_ for _ in ()).throw(
            AssertionError("model construction ran before n_folds validation")
        ),
    )

    with pytest.raises(InsufficientWalkForwardDataError) as captured:
        AlphaEngine(n_folds=value)

    message = str(captured.value)
    assert len(message) <= 200
    assert _HostileNFolds.calls == 0
    if type(value) is int:
        assert "bit_length=" in message


def test_invalid_configured_n_folds_fail_before_model_or_topology_work(
    monkeypatch,
) -> None:
    invalid_values = (
        None,
        True,
        False,
        -1,
        0,
        1,
        3.0,
        "3",
        _IntSubclass(3),
        _HostileNFolds(),
    )

    def unexpected_model_construction():
        raise AssertionError("model construction ran before config validation")

    monkeypatch.setattr(engine_module, "AlphaGPT", unexpected_model_construction)
    for configured in invalid_values:
        _HostileNFolds.calls = 0
        monkeypatch.setattr(ModelConfig, "WF_N_BLOCKS", configured)
        with pytest.raises(InsufficientWalkForwardDataError) as captured:
            AlphaEngine()
        message = str(captured.value)
        assert "ModelConfig.WF_N_BLOCKS" in message
        assert "exact built-in int >=2" in message
        assert len(message) <= 200
        assert _HostileNFolds.calls == 0


def test_public_n_folds_must_equal_current_config_before_model_construction(
    monkeypatch,
) -> None:
    monkeypatch.setattr(ModelConfig, "WF_N_BLOCKS", 3)
    monkeypatch.setattr(
        engine_module,
        "AlphaGPT",
        lambda: (_ for _ in ()).throw(
            AssertionError("model construction ran before divergence validation")
        ),
    )

    with pytest.raises(
        InsufficientWalkForwardDataError,
        match=r"n_folds.*expected=3.*actual=4",
    ):
        AlphaEngine(n_folds=4)


def test_configured_three_blocks_drive_sizing_and_two_fold_topology(
    monkeypatch,
) -> None:
    real_required = engine_module.required_training_bars
    real_build = engine_module.build_walk_forward_folds
    captured = {}

    def capture_required(**kwargs):
        captured["required_kwargs"] = kwargs.copy()
        captured["required_bars"] = real_required(**kwargs)
        return captured["required_bars"]

    def capture_build(**kwargs):
        captured["build_kwargs"] = kwargs.copy()
        captured["folds"] = real_build(**kwargs)
        return captured["folds"]

    monkeypatch.setattr(engine_module, "formula_warmup_bars", lambda _length: 2)
    monkeypatch.setattr(engine_module, "required_training_bars", capture_required)
    monkeypatch.setattr(engine_module, "build_walk_forward_folds", capture_build)
    monkeypatch.setattr(ModelConfig, "WF_N_BLOCKS", 3)
    monkeypatch.setattr(ModelConfig, "WF_MIN_FOLD_BARS", 2)
    monkeypatch.setattr(ModelConfig, "WF_GAP", 2)

    engine = _boundary_engine(_valid_training_manager(bars=30), n_folds=3)
    engine.train(end_step=0, verbose_header=False)

    assert captured["required_kwargs"]["n_blocks"] == ModelConfig.WF_N_BLOCKS == 3
    assert captured["build_kwargs"]["n_blocks"] == ModelConfig.WF_N_BLOCKS == 3
    assert captured["required_bars"] == 14
    assert len(captured["folds"]) == 2
    assert [fold.fold_index for fold in captured["folds"]] == [0, 1]


@pytest.mark.parametrize("mutated", [2, 3, 4, 6, True, 5.0, "5", _IntSubclass(5)])
def test_training_rejects_mutated_topology_before_minimum_bars_or_fold_work(
    monkeypatch, mutated
) -> None:
    engine = _boundary_engine(_valid_training_manager())
    engine.n_folds = mutated

    def unexpected_work(*_args, **_kwargs):
        raise AssertionError("training topology work ran before n_folds revalidation")

    monkeypatch.setattr(engine_module, "formula_warmup_bars", unexpected_work)
    monkeypatch.setattr(engine_module, "required_training_bars", unexpected_work)
    monkeypatch.setattr(engine_module, "assert_minimum_bars", unexpected_work)
    monkeypatch.setattr(engine_module, "build_walk_forward_folds", unexpected_work)

    with pytest.raises(
        InsufficientWalkForwardDataError,
        match=r"n_folds.*expected=5.*actual=",
    ):
        engine.train(end_step=0, verbose_header=False)


def test_compute_ic_requires_and_uses_explicit_shared_valid_mask() -> None:
    factor = torch.tensor([[1.0, 2.0, 4.0, 1.0e6, -1.0e6]])
    target = torch.tensor([[1.0, 2.0, 4.0, -1.0e6, 1.0e6]])
    valid = torch.tensor([[True, True, True, False, False]])

    with pytest.raises(TypeError):
        AlphaEngine._compute_ic(factor, target)

    positional = AlphaEngine._compute_ic(factor, target, valid)
    keyword = AlphaEngine._compute_ic(
        factor, target, target_valid=valid
    )

    assert positional == pytest.approx(1.0)
    assert keyword == pytest.approx(positional)
    with pytest.raises(TypeError):
        AlphaEngine._compute_ic(factor, target, valid_mask=valid)


def test_ic_compares_factor_t_with_target_t() -> None:
    factor = torch.tensor([[0.0, 1.0, 4.0, 2.0, 99.0, -99.0]])
    target = factor.clone()
    valid = torch.tensor([[True, True, True, True, False, False]])

    same_bar_ic = AlphaEngine._compute_ic(factor, target, valid)
    one_bar_shift_ic = AlphaEngine._compute_ic(
        factor[:, :-1],
        target[:, 1:],
        valid[:, :-1] & valid[:, 1:],
    )

    assert same_bar_ic == pytest.approx(1.0)
    assert one_bar_shift_ic != pytest.approx(same_bar_ic)


def test_invalid_tail_extremes_do_not_change_ic() -> None:
    factor = torch.tensor([[1.0, 2.0, 4.0, 8.0, 16.0]])
    target = factor.clone()
    valid = torch.tensor([[True, True, True, False, False]])
    baseline = AlphaEngine._compute_ic(factor, target, valid)
    baseline_stability = AlphaEngine._compute_ic_stability(factor, target, valid)

    changed_factor = factor.clone()
    changed_target = target.clone()
    changed_factor[:, -2:] = torch.tensor([[-1.0e20, 1.0e20]])
    changed_target[:, -2:] = torch.tensor([[1.0e20, -1.0e20]])
    changed = AlphaEngine._compute_ic(changed_factor, changed_target, valid)
    changed_stability = AlphaEngine._compute_ic_stability(
        changed_factor, changed_target, valid
    )

    assert changed == pytest.approx(baseline)
    assert changed_stability == pytest.approx(baseline_stability)


@pytest.mark.parametrize("scale", [1.0, 1.0e-4, 1.0e-6, 1.0e-7])
@pytest.mark.parametrize("direction", [1.0, -1.0])
def test_engine_ic_and_stability_are_scale_invariant_and_match_backtest(
    scale, direction
) -> None:
    factor = torch.tensor([[1.0, 2.0, 4.0, 8.0]]) * scale
    target = factor * direction
    valid = torch.ones_like(factor, dtype=torch.bool)

    assert AlphaEngine._compute_ic(factor, target, valid) == pytest.approx(
        direction, abs=1.0e-6
    )
    assert AlphaEngine._compute_ic_stability(
        factor, target, valid
    ) == pytest.approx(direction, abs=1.0e-6)
    assert MT5Backtest()._ts_ic_stability(
        factor, target, valid
    ) == pytest.approx(direction, abs=1.0e-6)


def test_fold_selection_merges_intervals_before_materializing(monkeypatch) -> None:
    folds = [
        WalkForwardFold(i, 0, 125_000, 125_000, 249_996, 0)
        for i in range(100)
    ]
    generated = []
    real_arange = torch.arange

    def capture_arange(start, end, **kwargs):
        generated.append(end - start)
        return real_arange(start, end, **kwargs)

    monkeypatch.setattr(engine_module.torch, "arange", capture_arange)
    selection = AlphaEngine._fold_selection_index(250_000, folds)

    assert torch.equal(selection, real_arange(249_996, dtype=torch.long))
    assert generated == [249_996]


def test_engine_and_backtest_delegate_to_one_shared_ic_implementation(
    monkeypatch,
) -> None:
    calls = []
    real_compute = backtest_module.compute_ic_metrics

    def traced_compute(*args, **kwargs):
        calls.append(args)
        return real_compute(*args, **kwargs)

    monkeypatch.setattr(engine_module, "compute_ic_metrics", traced_compute)
    monkeypatch.setattr(backtest_module, "compute_ic_metrics", traced_compute)
    factor = torch.tensor([[1.0, 2.0, 4.0, 8.0]])
    valid = torch.ones_like(factor, dtype=torch.bool)

    engine_result = AlphaEngine._compute_ic_components(factor, factor, valid)
    backtest_result = MT5Backtest()._ts_ic_stability(factor, factor, valid)

    assert len(calls) == 2
    assert torch.equal(engine_result[1], torch.tensor(1.0))
    assert backtest_result == pytest.approx(1.0)


def test_fold_selection_exact_sorted_union_excludes_gaps_and_exterior() -> None:
    folds = [
        WalkForwardFold(0, 2, 5, 8, 10, 3),
        WalkForwardFold(1, 2, 10, 13, 15, 3),
    ]

    selection = AlphaEngine._fold_selection_index(18, folds, device="cpu")

    assert selection.dtype is torch.long
    assert selection.device == torch.device("cpu")
    assert selection.tolist() == [2, 3, 4, 5, 6, 7, 8, 9, 13, 14]


def test_fold_selection_runtime_is_not_proportional_to_fold_count() -> None:
    def measure(count):
        folds = [
            WalkForwardFold(i, 0, 125_000, 125_000, 249_996, 0)
            for i in range(count)
        ]
        started = time.perf_counter()
        output = AlphaEngine._fold_selection_index(250_000, folds)
        return time.perf_counter() - started, output

    fast_time, fast_output = min((measure(5) for _ in range(3)), key=lambda v: v[0])
    slow_time, slow_output = min(
        (measure(100) for _ in range(3)), key=lambda v: v[0]
    )

    assert torch.equal(fast_output, slow_output)
    assert slow_time < max(0.5, fast_time * 5.0)


def test_training_rejects_insufficient_bars_before_sampling() -> None:
    class TinyDataManager:
        feat_tensor = torch.zeros(1, 3, 10)
        target_ret = torch.zeros(1, 10)
        target_valid = torch.tensor([[True] * 8 + [False, False]])
        bar_time = torch.arange(10, dtype=torch.int64).unsqueeze(0)

    engine = _boundary_engine(TinyDataManager())
    with pytest.raises(
        DataValidationError,
        match="insufficient bars.*expected at least.*actual",
    ):
        engine.train(end_step=0, verbose_header=False)


def test_fold_selection_penalty_and_pool_ignore_excluded_bars() -> None:
    folds = [
        WalkForwardFold(0, 2, 5, 7, 9, 2),
        WalkForwardFold(1, 2, 9, 11, 13, 2),
    ]
    factor = torch.arange(15, dtype=torch.float32).unsqueeze(0)
    changed = factor.clone()
    changed[:, [0, 1, 9, 10, 13, 14]] = torch.tensor(
        [[1.0e6, -1.0e6, -2.0e6, 2.0e6, 3.0e6, -3.0e6]]
    )

    engine = AlphaEngine.__new__(AlphaEngine)
    engine.factor_pool = []
    engine._factor_pool_counter = 0
    selection_index = engine._fold_selection_index(factor.shape[-1], folds)
    selection_factor = engine._select_fold_bars(factor, selection_index)
    changed_selection_factor = engine._select_fold_bars(changed, selection_index)
    engine._update_factor_pool(1.0, selection_factor)

    stored = engine.factor_pool[0][2]
    expected = factor[:, [2, 3, 4, 5, 6, 7, 8, 11, 12]]
    assert torch.equal(stored, expected)
    assert stored.shape[-1] < factor.shape[-1]

    reward = torch.tensor(1.0)
    expected_penalty = torch.tensor(ModelConfig.CORR_PENALTY)
    assert torch.equal(
        engine._apply_corr_penalty(reward, selection_factor), expected_penalty
    )
    assert torch.equal(
        engine._apply_corr_penalty(reward, changed_selection_factor), expected_penalty
    )

    mutant = AlphaEngine.__new__(AlphaEngine)
    mutant.factor_pool = [(1.0, 0, factor)]
    assert not torch.equal(mutant._apply_corr_penalty(reward, changed), expected_penalty)


class _TinyPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(2))

    def forward(self, tokens: torch.Tensor):
        return self.logits.expand(tokens.shape[0], -1), None, None


class _TinySampler:
    delta = {0: 0, 1: 0}

    @staticmethod
    def apply_mask_to_logits(logits, *_args, **_kwargs):
        return logits

    @staticmethod
    def update_infection(_token: int, infected_chain_len: int) -> int:
        return infected_chain_len


class _RecordingBacktest:
    def __init__(self) -> None:
        self.fold_inputs = []

    def evaluate_fold(self, **kwargs):
        for start_name, end_name in (
            ("train_start", "train_end"),
            ("val_start", "val_end"),
        ):
            start = kwargs[start_name]
            end = kwargs[end_name]
            self.fold_inputs.append(
                tuple(
                    kwargs[name][..., start:end].detach().clone()
                    for name in ("factors", "target_ret", "target_valid")
                )
            )
        score = kwargs["factors"].new_tensor(1.0)
        return score, score


class _TrainingAbort(BaseException):
    pass


def _selection_indices(folds: list[WalkForwardFold]) -> list[int]:
    return sorted(
        {
            index
            for fold in folds
            for start, end in (
                (fold.train_start, fold.train_end),
                (fold.val_start, fold.val_end),
            )
            for index in range(start, end)
        }
    )


def _run_one_training_step(
    monkeypatch,
    tmp_path,
    factor: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    folds: list[WalkForwardFold],
    *,
    fixed_exposure: float | None = None,
):
    monkeypatch.setattr(engine_module, "formula_warmup_bars", lambda _length: 2)
    monkeypatch.setattr(engine_module, "required_training_bars", lambda **_kwargs: 1)
    monkeypatch.setattr(engine_module, "assert_minimum_bars", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        engine_module, "build_walk_forward_folds", lambda **_kwargs: folds
    )
    monkeypatch.setattr(
        engine_module,
        "_test_strategy_file_for_symbol",
        lambda _symbol: str(tmp_path / "strategy.json"),
    )
    monkeypatch.setattr(ModelConfig, "BATCH_SIZE", 2)
    monkeypatch.setattr(ModelConfig, "ELITE_REPLAY_FRAC", 0.5)
    monkeypatch.setattr(ModelConfig, "MAX_FORMULA_LEN", 1)
    monkeypatch.setattr(ModelConfig, "ENTROPY_COLLAPSE_THRESH", -1.0)

    exposure_inputs = []

    def record_positions(values: torch.Tensor) -> torch.Tensor:
        exposure_inputs.append(values.detach().clone())
        if fixed_exposure is not None:
            return torch.full_like(values, fixed_exposure)
        return torch.tanh(values)

    monkeypatch.setattr(
        engine_module, "compute_target_positions_stateless", record_positions
    )

    engine = AlphaEngine.__new__(AlphaEngine)
    _install_formal_training_identity(engine)
    engine.data_manager = SimpleNamespace(
        feat_tensor=torch.zeros(
            factor.shape[0], 1, factor.shape[1], dtype=factor.dtype
        ),
        target_ret=target,
        target_valid=valid,
        bar_time=torch.arange(factor.shape[1], dtype=torch.int64)
        .unsqueeze(0)
        .expand(factor.shape[0], -1),
    )
    engine.n_folds = ModelConfig.WF_N_BLOCKS
    engine.model = _TinyPolicy()
    engine.opt = torch.optim.SGD(engine.model.parameters(), lr=0.01)
    engine.sampler = _TinySampler()
    engine.vm = SimpleNamespace(execute=lambda _formula, _features: factor.clone())
    engine.bt = _RecordingBacktest()
    engine.use_lord = False
    engine.lord_opt = None
    engine.target_symbol = None
    engine.best_score = -float("inf")
    engine.best_formula = None
    engine._best_snapshot = None
    engine.training_history = {
        "step": [],
        "avg_reward": [],
        "best_score": [],
        "val_score": [],
        "stable_rank": [],
    }
    engine._restart_count = 0
    selected = factor[..., _selection_indices(folds)].clone()
    engine.factor_pool = [(1.0, 0, selected)]
    engine._factor_pool_counter = 1
    engine._elite_pool = []
    engine._elite_counter = 0
    engine._best_update_step = 0
    engine._stagnation_steps = 0
    engine._reward_ema = None
    engine._reward_ema_step = 0
    engine._low_entropy_streak = 0
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
    engine._save_training_history_live = lambda: None
    engine.save_checkpoint = lambda _step: tmp_path / "checkpoint.pt"
    engine._decode_formula = lambda _formula: "tiny"

    torch.manual_seed(1234)
    engine.train(end_step=1, verbose_header=False)
    history_keys = (
        "avg_reward",
        "val_score",
        "best_score",
        "ic_mean",
        "ic_stability",
        "sortino",
        "elite_pool_size",
    )
    return {
        "fold_inputs": engine.bt.fold_inputs,
        "history": {key: engine.training_history[key] for key in history_keys},
        "best": (engine.best_score, engine.best_formula),
        "elite": engine._elite_pool,
        "pool": [entry[2].clone() for entry in engine.factor_pool],
        "exposure_inputs": exposure_inputs,
    }


def _assert_training_outcomes_equal(left, right) -> None:
    assert left["history"] == right["history"]
    assert left["best"] == right["best"]
    assert left["elite"] == right["elite"]
    for key in ("fold_inputs", "pool", "exposure_inputs"):
        assert len(left[key]) == len(right[key])
        for left_item, right_item in zip(left[key], right[key]):
            if isinstance(left_item, tuple):
                assert all(
                    torch.equal(a, b) for a, b in zip(left_item, right_item)
                )
            else:
                assert torch.equal(left_item, right_item)


def _failure_training_engine(
    monkeypatch, tmp_path, execute, *, bars: int = 20, batch_size: int = 3
):
    folds = [
        WalkForwardFold(0, 2, 6, 8, 12, 2),
        WalkForwardFold(1, 2, 12, 14, 18, 2),
    ]
    factor = torch.linspace(-2.0, 2.0, bars).unsqueeze(0)
    monkeypatch.setattr(engine_module, "formula_warmup_bars", lambda _length: 2)
    monkeypatch.setattr(engine_module, "required_training_bars", lambda **_kwargs: 1)
    monkeypatch.setattr(engine_module, "assert_minimum_bars", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        engine_module, "build_walk_forward_folds", lambda **_kwargs: folds
    )
    monkeypatch.setattr(
        engine_module,
        "_test_strategy_file_for_symbol",
        lambda _symbol: str(tmp_path / "strategy.json"),
    )
    monkeypatch.setattr(ModelConfig, "BATCH_SIZE", batch_size)
    monkeypatch.setattr(ModelConfig, "ELITE_REPLAY_FRAC", 0.5)
    monkeypatch.setattr(ModelConfig, "MAX_FORMULA_LEN", 1)
    monkeypatch.setattr(ModelConfig, "ENTROPY_COLLAPSE_THRESH", -1.0)

    engine = AlphaEngine.__new__(AlphaEngine)
    _install_formal_training_identity(engine)
    monkeypatch.setattr(
        TrainingRunIdentity,
        "history_filename",
        lambda _identity: (
            f"training_history_{engine.target_symbol}.json"
            if engine.target_symbol else "training_history.json"
        ),
    )
    monkeypatch.setattr(
        TrainingRunIdentity,
        "checkpoint_filename",
        lambda _identity, step: f"ckpt_EURUSD_step_{step + 1:04d}.pt",
    )
    training_payload = SimpleNamespace(
        feat_tensor=torch.zeros(1, 1, factor.shape[1]),
        target_ret=factor.clone(),
        target_valid=torch.ones_like(factor, dtype=torch.bool),
        bar_time=(
            torch.arange(factor.shape[1], dtype=torch.int64) * 3_600_000_000_000
        ).unsqueeze(0),
    )
    engine.data_manager = training_payload
    engine.n_folds = ModelConfig.WF_N_BLOCKS
    engine.model = _TinyPolicy()
    engine.opt = torch.optim.SGD(engine.model.parameters(), lr=0.01)
    engine.sampler = _TinySampler()
    engine.vm = SimpleNamespace(execute=execute)
    engine.bt = _RecordingBacktest()
    engine.use_lord = False
    engine.lord_opt = None
    engine.target_symbol = None
    engine.best_score = -float("inf")
    engine.best_formula = None
    engine._best_snapshot = None
    engine.training_history = {
        "step": [],
        "avg_reward": [],
        "best_score": [],
        "val_score": [],
        "stable_rank": [],
    }
    engine._restart_count = 0
    engine.factor_pool = []
    engine._factor_pool_counter = 0
    engine._elite_pool = []
    engine._elite_counter = 0
    engine._best_update_step = 0
    engine._stagnation_steps = 0
    engine._reward_ema = None
    engine._reward_ema_step = 0
    engine._low_entropy_streak = 0
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
    calls = {"optimizer": 0, "strategy": 0, "history": 0, "checkpoint": 0}
    original_step = engine.opt.step

    def record_optimizer_step(*args, **kwargs):
        calls["optimizer"] += 1
        return original_step(*args, **kwargs)

    engine.opt.step = record_optimizer_step
    engine._save_strategy_live = lambda *_args: calls.__setitem__(
        "strategy", calls["strategy"] + 1
    )
    engine._save_training_history_live = lambda: calls.__setitem__(
        "history", calls["history"] + 1
    )

    def record_checkpoint(_step):
        calls["checkpoint"] += 1
        return tmp_path / "checkpoint.pt"

    engine.save_checkpoint = record_checkpoint
    engine._decode_formula = lambda _formula: "tiny"
    engine.train = AlphaEngine.train.__get__(engine, AlphaEngine)
    before = {
        "model": {name: value.detach().clone() for name, value in engine.model.state_dict().items()},
        "optimizer": engine.opt.state_dict(),
        "history": {name: list(values) for name, values in engine.training_history.items()},
        "factor_pool": list(engine.factor_pool),
        "elite_pool": list(engine._elite_pool),
        "best": (engine.best_score, engine.best_formula, engine._best_snapshot),
        "reward_ema": (engine._reward_ema, engine._reward_ema_step),
        "counters": (
            engine._factor_pool_counter,
            engine._elite_counter,
            engine._best_update_step,
            engine._stagnation_steps,
            engine._restart_count,
            engine._low_entropy_streak,
        ),
    }
    return engine, factor, calls, before


def _install_legacy_final_strategy_oracle(
    engine: AlphaEngine, *, enforce_ownership: bool = False
) -> None:
    """Retain the historical publication oracle without altering V2 identity."""
    def publish(*_args) -> None:
        from model_core.vocab import VOCAB_VERSION

        target = engine_module.pathlib.Path(
            engine_module._test_strategy_file_for_symbol(engine.target_symbol)
        )
        if enforce_ownership:
            engine._assert_artifact_owned_or_absent(target)
        engine_module._atomic_json_replace(
            target,
            {
                "vocab_version": VOCAB_VERSION,
                "symbol": engine.target_symbol,
                "formula": engine.best_formula,
                "best_score": engine.best_score,
            },
        )

    engine._save_strategy_live = publish


def _assert_no_failed_batch_side_effects(
    engine, calls, before, *, scoring_may_have_run: bool = False
) -> None:
    assert calls == {"optimizer": 0, "strategy": 0, "history": 0, "checkpoint": 0}
    assert engine.opt.state_dict() == before["optimizer"]
    assert engine.training_history == before["history"]
    assert engine.factor_pool == before["factor_pool"]
    assert engine._elite_pool == before["elite_pool"]
    assert (engine.best_score, engine.best_formula, engine._best_snapshot) == before["best"]
    assert (engine._reward_ema, engine._reward_ema_step) == before["reward_ema"]
    assert (
        engine._factor_pool_counter,
        engine._elite_counter,
        engine._best_update_step,
        engine._stagnation_steps,
        engine._restart_count,
        engine._low_entropy_streak,
    ) == before["counters"]
    for name, expected in before["model"].items():
        assert torch.equal(engine.model.state_dict()[name], expected)
    if not scoring_may_have_run:
        assert engine.bt.fold_inputs == []


def test_training_fails_closed_on_first_vm_none_without_batch_side_effects(
    monkeypatch, tmp_path
) -> None:
    vm_calls = []

    def execute(formula, _features):
        vm_calls.append(list(formula))
        return None

    engine, _factor, calls, before = _failure_training_engine(
        monkeypatch, tmp_path, execute
    )

    with pytest.raises(
        RuntimeError,
        match=r"program evaluation failed.*step=0.*formula_index=0.*formula_length=1",
    ):
        engine.train(end_step=1, verbose_header=False)

    assert len(vm_calls) == 1
    _assert_no_failed_batch_side_effects(engine, calls, before)


def test_training_preflights_batch_before_scoring_valid_then_none(
    monkeypatch, tmp_path
) -> None:
    vm_calls = []

    def execute(formula, _features):
        vm_calls.append(list(formula))
        return factor.clone() if len(vm_calls) == 1 else None

    engine, factor, calls, before = _failure_training_engine(
        monkeypatch, tmp_path, execute
    )

    with pytest.raises(RuntimeError, match=r"formula_index=1"):
        engine.train(end_step=1, verbose_header=False)

    assert len(vm_calls) == 2
    _assert_no_failed_batch_side_effects(engine, calls, before)


@pytest.mark.parametrize(
    "error_type",
    [RuntimeError, AssertionError, KeyboardInterrupt, SystemExit, _TrainingAbort],
)
def test_training_preflight_exception_traceback_releases_execute_tensor(
    monkeypatch, tmp_path, error_type
) -> None:
    failure = error_type("preflight execute failed")
    result_refs = []
    vm_calls = 0

    def execute(_formula, _features):
        nonlocal vm_calls
        vm_calls += 1
        result = torch.ones(192, 192)
        result_refs.append(weakref.ref(result))
        raise failure

    engine, _factor, calls, before = _failure_training_engine(
        monkeypatch, tmp_path, execute, batch_size=ModelConfig.BATCH_SIZE
    )

    with pytest.raises(error_type) as caught:
        engine.train(end_step=1, verbose_header=False)

    assert caught.value is failure
    assert caught.value.__traceback__ is not None
    assert vm_calls == 1
    assert result_refs and result_refs[0]() is None
    _assert_no_failed_batch_side_effects(engine, calls, before)


def test_training_preflight_releases_each_result_before_second_pass(
    monkeypatch, tmp_path
) -> None:
    result_refs = []
    vm_calls = []
    live_before_execute = []
    peak_live_results = 0

    def execute(formula, _features):
        nonlocal peak_live_results
        live_before_execute.append(
            sum(result_ref() is not None for result_ref in result_refs)
        )
        vm_calls.append(list(formula))
        result = factor.clone()
        result_refs.append(weakref.ref(result))
        peak_live_results = max(
            peak_live_results,
            sum(result_ref() is not None for result_ref in result_refs),
        )
        return result

    engine, factor, _calls, _before = _failure_training_engine(
        monkeypatch, tmp_path, execute
    )

    engine.train(end_step=1, verbose_header=False)

    assert len(vm_calls) == 6
    assert live_before_execute == [0] * len(vm_calls)
    assert peak_live_results == 1


def test_training_second_pass_first_none_fails_before_scoring(
    monkeypatch, tmp_path
) -> None:
    vm_calls = []

    def execute(formula, _features):
        vm_calls.append(list(formula))
        return None if len(vm_calls) == 4 else factor.clone()

    engine, factor, calls, before = _failure_training_engine(
        monkeypatch, tmp_path, execute
    )

    with pytest.raises(RuntimeError, match=r"formula_index=0"):
        engine.train(end_step=1, verbose_header=False)

    assert len(vm_calls) == 4
    _assert_no_failed_batch_side_effects(
        engine, calls, before, scoring_may_have_run=True
    )


def test_training_late_second_pass_none_is_batch_transactional_at_real_batch_size(
    monkeypatch, tmp_path
) -> None:
    batch_size = ModelConfig.BATCH_SIZE
    vm_calls = 0

    def execute(_formula, _features):
        nonlocal vm_calls
        vm_calls += 1
        if vm_calls == batch_size + 2:
            return None
        return factor.clone()

    engine, factor, calls, before = _failure_training_engine(
        monkeypatch, tmp_path, execute, batch_size=batch_size
    )

    with pytest.raises(RuntimeError, match=r"formula_index=1"):
        engine.train(end_step=1, verbose_header=False)

    assert vm_calls == batch_size + 2
    _assert_no_failed_batch_side_effects(
        engine, calls, before, scoring_may_have_run=True
    )


@pytest.mark.parametrize(
    "error_type",
    [RuntimeError, AssertionError, KeyboardInterrupt, SystemExit, _TrainingAbort],
)
def test_training_late_scoring_exception_is_batch_transactional(
    monkeypatch, tmp_path, error_type
) -> None:
    failure = error_type("later fold scoring failed")
    vm_calls = 0

    def execute(_formula, _features):
        nonlocal vm_calls
        vm_calls += 1
        return factor.clone()

    engine, factor, calls, before = _failure_training_engine(
        monkeypatch, tmp_path, execute
    )
    real_evaluate = engine.bt.evaluate_fold
    scoring_calls = 0

    def evaluate_fold(**kwargs):
        nonlocal scoring_calls
        scoring_calls += 1
        if scoring_calls == 3:
            raise failure
        return real_evaluate(**kwargs)

    engine.bt.evaluate_fold = evaluate_fold

    with pytest.raises(error_type) as caught:
        engine.train(end_step=1, verbose_header=False)

    assert caught.value is failure
    assert vm_calls == 5
    _assert_no_failed_batch_side_effects(
        engine, calls, before, scoring_may_have_run=True
    )


@pytest.mark.parametrize(
    "error_type",
    [RuntimeError, AssertionError, KeyboardInterrupt, SystemExit, _TrainingAbort],
)
def test_real_fold_scoring_traceback_does_not_retain_vm_result(
    monkeypatch, tmp_path, error_type
) -> None:
    batch_size = ModelConfig.BATCH_SIZE
    failure = error_type("real fold scoring failed")
    result_refs = []
    selection_refs = []
    vm_calls = 0

    def execute(_formula, _features):
        nonlocal vm_calls
        vm_calls += 1
        result = factor.clone()
        result_refs.append(weakref.ref(result))
        return result

    engine, factor, calls, before = _failure_training_engine(
        monkeypatch, tmp_path, execute, batch_size=batch_size
    )
    engine.data_manager.target_ret = factor.clone() * 1.0e-4
    engine.bt = MT5Backtest()
    real_select = engine._select_fold_bars

    def select_fold_bars(values, index):
        selected = real_select(values, index)
        selection_refs.append(weakref.ref(selected))
        return selected

    engine._select_fold_bars = select_fold_bars
    real_multi_objective = engine.bt._multi_objective
    multi_objective_calls = 0

    def fail_multi_objective(*args, **kwargs):
        nonlocal multi_objective_calls
        multi_objective_calls += 1
        if multi_objective_calls == 5:
            raise failure
        return real_multi_objective(*args, **kwargs)

    engine.bt._multi_objective = fail_multi_objective

    with pytest.raises(error_type) as caught:
        engine.train(end_step=1, verbose_header=False)

    assert caught.value is failure
    assert caught.value.__traceback__ is not None
    assert vm_calls == batch_size + 2
    assert multi_objective_calls == 5
    assert all(result_ref() is None for result_ref in result_refs)
    assert selection_refs and selection_refs[-1]() is None
    assert calls == {"optimizer": 0, "strategy": 0, "history": 0, "checkpoint": 0}
    assert engine.factor_pool == before["factor_pool"]
    assert engine._elite_pool == before["elite_pool"]


def test_successful_batch_commits_buffered_actions_in_formula_order(
    monkeypatch, tmp_path
) -> None:
    vm_calls = 0

    def execute(_formula, _features):
        nonlocal vm_calls
        vm_calls += 1
        return factor.clone()

    engine, factor, calls, _before = _failure_training_engine(
        monkeypatch, tmp_path, execute
    )
    scoring_calls = 0

    def evaluate_fold(**kwargs):
        nonlocal scoring_calls
        scoring_calls += 1
        assert engine.best_score == -float("inf")
        assert engine.factor_pool == []
        assert engine._elite_pool == []
        assert calls["strategy"] == 0
        score = kwargs["factors"].new_tensor((scoring_calls + 1) // 2)
        return score, score

    engine.bt.evaluate_fold = evaluate_fold
    monkeypatch.setattr(ModelConfig, "CORR_THRESHOLD", 2.0)
    committed = []

    def record_strategy(*_args):
        calls["strategy"] += 1
        committed.append(
            (engine.best_score, list(engine.best_formula), len(engine.factor_pool))
        )

    engine._save_strategy_live = record_strategy
    engine.train(end_step=1, verbose_header=False)

    assert vm_calls == 6
    assert scoring_calls == 6
    assert [score for score, _formula, _pool_size in committed] == [3.0]
    assert [pool_size for _score, _formula, pool_size in committed] == [3]
    assert engine.best_score == 3.0
    assert len(engine.factor_pool) == 3
    assert calls == {"optimizer": 1, "strategy": 1, "history": 1, "checkpoint": 1}


def _commit_test_engine() -> AlphaEngine:
    engine = AlphaEngine.__new__(AlphaEngine)
    engine.best_score = -1.0
    engine.best_formula = [99]
    engine._best_snapshot = {"old": torch.tensor(1.0)}
    engine._best_update_step = 7
    engine._stagnation_steps = 8
    engine.factor_pool = []
    engine._factor_pool_counter = 0
    engine._elite_pool = []
    engine._elite_counter = 0
    engine._decode_formula = lambda formula: str(formula)
    return engine


def _pending_commit_actions():
    snapshot = {"weight": torch.ones(64, 64)}
    buffered = torch.ones(64, 64)
    refs = [weakref.ref(snapshot["weight"]), weakref.ref(buffered)]
    actions = [
        ("elite", 1.0, [1], 10),
        ("best", 2.0, [2], snapshot, 10, buffered, -1.0, 0.5, 0.5),
        ("elite", 2.0, [2], 10),
    ]
    return actions, refs


@pytest.mark.parametrize("failure_index", [0, 1, 2])
@pytest.mark.parametrize(
    "error_type",
    [RuntimeError, AssertionError, KeyboardInterrupt, SystemExit, _TrainingAbort],
)
def test_commit_replay_failure_is_transactional_and_releases_buffered_tensors(
    monkeypatch, failure_index, error_type
) -> None:
    engine = _commit_test_engine()
    before = (
        engine.best_score,
        list(engine.best_formula),
        engine._best_snapshot,
        engine._best_update_step,
        engine._stagnation_steps,
        list(engine.factor_pool),
        engine._factor_pool_counter,
        list(engine._elite_pool),
        engine._elite_counter,
    )
    actions, refs = _pending_commit_actions()
    failure = error_type("commit replay failed")
    replay_calls = 0
    strategy_writes = []

    real_elite = AlphaEngine._update_elite_pool
    real_factor = AlphaEngine._update_factor_pool

    def maybe_fail(callback, *args):
        nonlocal replay_calls
        current = replay_calls
        replay_calls += 1
        if current == failure_index:
            raise failure
        return callback(engine, *args)

    monkeypatch.setattr(
        engine,
        "_update_elite_pool",
        lambda *args: maybe_fail(real_elite, *args),
    )
    monkeypatch.setattr(
        engine,
        "_update_factor_pool",
        lambda *args: maybe_fail(real_factor, *args),
    )
    monkeypatch.setattr(engine, "_save_strategy_live", lambda: strategy_writes.append(1))

    with pytest.raises(error_type) as caught:
        engine._commit_pending_actions(actions)

    assert caught.value is failure
    assert caught.value.__traceback__ is not None
    assert strategy_writes == []
    assert actions == []
    assert (
        engine.best_score,
        engine.best_formula,
        engine._best_snapshot,
        engine._best_update_step,
        engine._stagnation_steps,
        engine.factor_pool,
        engine._factor_pool_counter,
        engine._elite_pool,
        engine._elite_counter,
    ) == before
    gc.collect()
    assert all(ref() is None for ref in refs)


@pytest.mark.parametrize(
    "error_type",
    [RuntimeError, AssertionError, KeyboardInterrupt, SystemExit, _TrainingAbort],
)
def test_final_strategy_write_failure_rolls_back_commit_and_releases_tensors(
    monkeypatch, error_type
) -> None:
    engine = _commit_test_engine()
    before = (
        engine.best_score,
        list(engine.best_formula),
        engine._best_snapshot,
        engine._best_update_step,
        engine._stagnation_steps,
        list(engine.factor_pool),
        engine._factor_pool_counter,
        list(engine._elite_pool),
        engine._elite_counter,
    )
    actions, refs = _pending_commit_actions()
    failure = error_type("strategy write failed")
    completed_writes = []

    def fail_write():
        raise failure

    monkeypatch.setattr(engine, "_save_strategy_live", fail_write)
    with pytest.raises(error_type) as caught:
        engine._commit_pending_actions(actions)

    assert caught.value is failure
    assert completed_writes == []
    assert actions == []
    assert (
        engine.best_score,
        engine.best_formula,
        engine._best_snapshot,
        engine._best_update_step,
        engine._stagnation_steps,
        engine.factor_pool,
        engine._factor_pool_counter,
        engine._elite_pool,
        engine._elite_counter,
    ) == before
    gc.collect()
    assert all(ref() is None for ref in refs)


def test_live_strategy_replace_failure_preserves_old_bytes_and_exception_identity(
    monkeypatch, tmp_path
) -> None:
    strategy_path = tmp_path / "strategy.json"
    old_bytes = b'{"old": true}\n'
    strategy_path.write_bytes(old_bytes)
    engine = _commit_test_engine()
    engine.target_symbol = "EURUSD"
    engine._record_owned_artifact(strategy_path)
    failure = _TrainingAbort("atomic replace failed")

    monkeypatch.setattr(
        engine_module,
        "_test_strategy_file_for_symbol",
        lambda _symbol: str(strategy_path),
    )

    def fail_replace(_self, _target):
        raise failure

    monkeypatch.setattr(engine_module.pathlib.Path, "replace", fail_replace)

    with pytest.raises(ArtifactCompatibilityError) as caught:
        engine._save_strategy_live()

    assert caught.value is not failure
    assert "run_identity mismatch" in str(caught.value)
    assert strategy_path.read_bytes() == old_bytes
    assert not strategy_path.with_name(f".{strategy_path.name}.tmp").exists()


def test_identity_free_live_strategy_rejects_before_filename_or_filesystem(
    monkeypatch, tmp_path
) -> None:
    strategy_path = tmp_path / "best_EURUSD.json"
    strategy_path.write_bytes(b"IDENTITY-FREE-LEGACY-BYTES")
    engine = _commit_test_engine()
    engine.target_symbol = "EURUSD"
    engine.run_identity = None
    monkeypatch.setattr(
        engine_module,
        "_test_strategy_file_for_symbol",
        lambda _symbol: (_ for _ in ()).throw(
            AssertionError("legacy filename reached before identity rejection")
        ),
    )
    with pytest.raises(ArtifactCompatibilityError) as caught:
        engine._save_strategy_live()
    assert "run_identity" in str(caught.value)
    assert strategy_path.read_bytes() == b"IDENTITY-FREE-LEGACY-BYTES"
    assert not list(tmp_path.glob(".*.tmp"))


def test_new_and_elite_entropy_paths_share_the_stable_guardrail(
    monkeypatch, tmp_path
) -> None:
    def execute(_formula, _features):
        return factor.clone()

    engine, factor, _calls, _before = _failure_training_engine(
        monkeypatch, tmp_path, execute
    )
    real_entropy = engine._stable_categorical_entropy
    observed_shapes = []

    def record_entropy(logits):
        observed_shapes.append(tuple(logits.shape))
        return real_entropy(logits)

    engine._stable_categorical_entropy = record_entropy
    engine.train(end_step=1, verbose_header=False)

    assert observed_shapes == [(2, 2), (1, 2)]


def test_training_preflight_100k_real_batch_has_bounded_factor_retention(
    monkeypatch, tmp_path
) -> None:
    real_batch_size = ModelConfig.BATCH_SIZE
    result_refs = []
    vm_calls = 0
    live_before_execute = []
    peak_live_results = 0

    def execute(_formula, _features):
        nonlocal vm_calls, peak_live_results
        live_before_execute.append(
            sum(result_ref() is not None for result_ref in result_refs)
        )
        vm_calls += 1
        if vm_calls == real_batch_size:
            return None
        result = factor.clone()
        result_refs.append(weakref.ref(result))
        peak_live_results = max(
            peak_live_results,
            sum(result_ref() is not None for result_ref in result_refs),
        )
        return result

    engine, factor, calls, before = _failure_training_engine(
        monkeypatch,
        tmp_path,
        execute,
        bars=100_000,
        batch_size=real_batch_size,
    )

    started = time.perf_counter()
    with pytest.raises(RuntimeError, match=rf"formula_index={real_batch_size - 1}"):
        engine.train(end_step=1, verbose_header=False)
    elapsed = time.perf_counter() - started

    assert vm_calls == real_batch_size
    assert live_before_execute == [0] * vm_calls
    assert peak_live_results == 1
    assert elapsed < 30.0
    _assert_no_failed_batch_side_effects(engine, calls, before)


def test_training_fails_closed_when_real_stack_vm_swallows_operator_failure(
    monkeypatch, tmp_path
) -> None:
    vm = StackVM()
    unary_token = next(token for token, arity in vm.arity_map.items() if arity == 1)

    def broken_operator(_value):
        raise RuntimeError("unexpected unary operator failure")

    vm.op_map[unary_token] = broken_operator

    def execute(_formula, features):
        return vm.execute([0, unary_token], features)

    engine, _factor, calls, before = _failure_training_engine(
        monkeypatch, tmp_path, execute
    )

    with pytest.raises(RuntimeError, match=r"program evaluation failed.*formula_index=0"):
        engine.train(end_step=1, verbose_header=False)

    _assert_no_failed_batch_side_effects(engine, calls, before)


@pytest.mark.parametrize("error_type", [RuntimeError, AssertionError])
def test_training_propagates_direct_vm_exceptions_unchanged(
    monkeypatch, tmp_path, error_type
) -> None:
    failure = error_type("direct vm failure")

    def execute(_formula, _features):
        raise failure

    engine, _factor, calls, before = _failure_training_engine(
        monkeypatch, tmp_path, execute
    )

    with pytest.raises(error_type) as caught:
        engine.train(end_step=1, verbose_header=False)

    assert caught.value is failure
    _assert_no_failed_batch_side_effects(engine, calls, before)


def test_exact_constant_factor_keeps_distinct_minus_two_training_path(
    monkeypatch, tmp_path
) -> None:
    result_refs = []
    live_before_execute = []
    peak_live_results = 0

    def execute(_formula, features):
        nonlocal peak_live_results
        live_before_execute.append(
            sum(result_ref() is not None for result_ref in result_refs)
        )
        result = torch.full(
            (features.shape[0], features.shape[-1]),
            3.0,
            dtype=features.dtype,
            device=features.device,
        )
        result_refs.append(weakref.ref(result))
        peak_live_results = max(
            peak_live_results,
            sum(result_ref() is not None for result_ref in result_refs),
        )
        return result

    engine, _factor, calls, _before = _failure_training_engine(
        monkeypatch, tmp_path, execute
    )

    engine.train(end_step=1, verbose_header=False)

    assert engine.training_history["step"] == [0]
    assert engine.training_history["avg_reward"] == [-2.0]
    assert engine.training_history["val_score"] == [-2.0]
    assert engine.bt.fold_inputs == []
    assert engine.factor_pool == []
    assert calls == {"optimizer": 1, "strategy": 0, "history": 1, "checkpoint": 1}
    assert live_before_execute == [0] * len(live_before_execute)
    assert peak_live_results == 1


def test_training_releases_scoring_result_on_downstream_exception(
    monkeypatch, tmp_path
) -> None:
    result_refs = []
    live_before_execute = []
    failure = RuntimeError("fold scoring failed")

    def execute(_formula, _features):
        live_before_execute.append(
            sum(result_ref() is not None for result_ref in result_refs)
        )
        result = factor.clone()
        result_refs.append(weakref.ref(result))
        return result

    engine, factor, calls, before = _failure_training_engine(
        monkeypatch, tmp_path, execute
    )

    def fail_scoring(_score, _selection_factor, **_kwargs):
        raise failure

    engine._apply_corr_penalty = fail_scoring

    with pytest.raises(RuntimeError) as caught:
        engine.train(end_step=1, verbose_header=False)

    assert caught.value is failure
    assert live_before_execute == [0] * len(live_before_execute)
    assert all(result_ref() is None for result_ref in result_refs)
    engine.bt.fold_inputs.clear()
    _assert_no_failed_batch_side_effects(engine, calls, before)


def test_training_loop_constant_eligibility_ignores_fold_exterior(
    monkeypatch, tmp_path
) -> None:
    folds = [
        WalkForwardFold(0, 2, 6, 8, 12, 2),
        WalkForwardFold(1, 2, 12, 14, 18, 2),
    ]
    factor = torch.ones(1, 20)
    changed = factor.clone()
    outside = sorted(set(range(20)) - set(_selection_indices(folds)))
    changed[:, outside] = torch.linspace(-1.0e7, 1.0e7, len(outside))
    target = torch.arange(20, dtype=torch.float32).unsqueeze(0)
    valid = torch.ones_like(target, dtype=torch.bool)

    assert factor.std() < 1e-4
    assert changed.std() > 1e4
    baseline = _run_one_training_step(
        monkeypatch, tmp_path / "baseline", factor, target, valid, folds
    )
    mutated = _run_one_training_step(
        monkeypatch, tmp_path / "mutated", changed, target, valid, folds
    )

    assert baseline["fold_inputs"] == []
    assert mutated["fold_inputs"] == []
    _assert_training_outcomes_equal(baseline, mutated)


@pytest.mark.parametrize(
    ("dtype", "scale_name"),
    [
        (torch.float32, "1e-7"),
        (torch.float32, "min-subnormal"),
        (torch.float64, "1e-12"),
        (torch.float64, "min-subnormal"),
    ],
)
@pytest.mark.parametrize("direction", [1.0, -1.0])
def test_training_loop_scores_every_exactly_nonconstant_scale_like_control(
    monkeypatch, tmp_path, dtype: torch.dtype, scale_name: str, direction: float
) -> None:
    folds = [
        WalkForwardFold(0, 2, 6, 8, 12, 2),
        WalkForwardFold(1, 2, 12, 14, 18, 2),
    ]
    pattern = torch.tensor([-2.0, -1.0, 1.0, 2.0], dtype=dtype).repeat(5).unsqueeze(0)
    if scale_name == "min-subnormal":
        scale = torch.nextafter(
            torch.tensor(0.0, dtype=dtype),
            torch.tensor(float("inf"), dtype=dtype),
        )
    else:
        scale = torch.tensor(float(scale_name), dtype=dtype)
    tiny = pattern * scale * direction
    control = pattern * direction
    valid = torch.ones_like(tiny, dtype=torch.bool)

    assert not torch.equal(tiny, tiny[..., :1].expand_as(tiny))
    tiny_outcome = _run_one_training_step(
        monkeypatch,
        tmp_path / f"tiny-{dtype}-{scale_name}-{direction}",
        tiny,
        tiny.clone(),
        valid,
        folds,
        fixed_exposure=0.1,
    )
    control_outcome = _run_one_training_step(
        monkeypatch,
        tmp_path / f"control-{dtype}-{scale_name}-{direction}",
        control,
        control.clone(),
        valid,
        folds,
        fixed_exposure=0.1,
    )

    assert len(tiny_outcome["fold_inputs"]) == 8
    for key in ("ic_mean", "ic_stability"):
        assert tiny_outcome["history"][key] == pytest.approx(
            control_outcome["history"][key], rel=0, abs=1.0e-15
        )
    for key in set(tiny_outcome["history"]) - {"ic_mean", "ic_stability"}:
        assert tiny_outcome["history"][key] == control_outcome["history"][key]
    assert tiny_outcome["best"] == control_outcome["best"]
    assert tiny_outcome["elite"] == control_outcome["elite"]


def test_training_loop_tiny_variation_uses_only_selected_fold_support(
    monkeypatch, tmp_path
) -> None:
    folds = [
        WalkForwardFold(0, 2, 6, 8, 12, 2),
        WalkForwardFold(1, 2, 12, 14, 18, 2),
    ]
    selected = _selection_indices(folds)
    outside = sorted(set(range(20)) - set(selected))
    factor = torch.ones(1, 20, dtype=torch.float32)
    factor[:, selected] += torch.linspace(0.0, 1.0e-7, len(selected))
    changed = factor.clone()
    changed[:, outside] = torch.linspace(-1.0e20, 1.0e20, len(outside))
    target = factor.clone()
    valid = torch.ones_like(factor, dtype=torch.bool)

    baseline = _run_one_training_step(
        monkeypatch,
        tmp_path / "tiny-exterior-baseline",
        factor,
        target,
        valid,
        folds,
        fixed_exposure=0.1,
    )
    mutated = _run_one_training_step(
        monkeypatch,
        tmp_path / "tiny-exterior-mutated",
        changed,
        target,
        valid,
        folds,
        fixed_exposure=0.1,
    )

    assert baseline["fold_inputs"]
    assert baseline["history"] == mutated["history"]
    assert baseline["best"] == mutated["best"]
    assert baseline["elite"] == mutated["elite"]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("direction", [1.0, -1.0])
def test_corr_penalty_is_scale_invariant_down_to_min_subnormal(
    dtype: torch.dtype, direction: float
) -> None:
    pattern = torch.tensor([[-2.0, -1.0, 1.0, 2.0]], dtype=dtype)
    min_subnormal = torch.nextafter(
        torch.tensor(0.0, dtype=dtype),
        torch.tensor(float("inf"), dtype=dtype),
    )
    engine = AlphaEngine.__new__(AlphaEngine)
    engine.factor_pool = [(1.0, 7, pattern.clone())]
    reward = torch.tensor(1.0, dtype=dtype)
    expected = reward * ModelConfig.CORR_PENALTY

    for scale in (1.0, 1.0e-3, 1.0e-7, min_subnormal):
        candidate = pattern * scale * direction
        assert not torch.equal(candidate, candidate[..., :1].expand_as(candidate))
        actual = engine._apply_corr_penalty(reward, candidate)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert actual.dtype is reward.dtype
        assert actual.device == reward.device


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_corr_normalization_stays_finite_centered_and_unit_at_min_subnormal(
    dtype: torch.dtype,
) -> None:
    min_subnormal = torch.nextafter(
        torch.tensor(0.0, dtype=dtype),
        torch.tensor(float("inf"), dtype=dtype),
    )
    values = torch.tensor([-2.0, -1.0, 1.0, 2.0], dtype=dtype) * min_subnormal

    normalized = engine_module._scale_invariant_centered(values)

    assert torch.isfinite(normalized).all()
    tolerance = torch.finfo(dtype).eps
    assert normalized.mean().item() == pytest.approx(0.0, abs=tolerance)
    assert normalized.norm().item() == pytest.approx(1.0, abs=tolerance)


def test_corr_penalty_ignores_constant_candidates_and_pool_entries_without_mutation(
) -> None:
    engine = AlphaEngine.__new__(AlphaEngine)
    constant_pool = torch.full((1, 4), 9.0)
    correlated_pool = torch.tensor([[-2.0, -1.0, 1.0, 2.0]])
    engine.factor_pool = [
        (2.0, 4, constant_pool),
        (1.0, 5, correlated_pool),
    ]
    before = [(score, count, factor.clone()) for score, count, factor in engine.factor_pool]

    positive = engine._apply_corr_penalty(torch.tensor(1.0), correlated_pool * 1.0e-7)
    negative = engine._apply_corr_penalty(torch.tensor(-1.0), correlated_pool * -1.0e-7)
    constant = engine._apply_corr_penalty(torch.tensor(1.0), torch.full((1, 4), 3.0))

    assert positive.item() == pytest.approx(ModelConfig.CORR_PENALTY)
    assert negative.item() == pytest.approx(-ModelConfig.CORR_PENALTY)
    assert constant.item() == pytest.approx(1.0)
    assert [(score, count) for score, count, _factor in engine.factor_pool] == [
        (score, count) for score, count, _factor in before
    ]
    for (_, _, actual), (_, _, expected) in zip(engine.factor_pool, before):
        assert actual.data_ptr() != expected.data_ptr()
        assert torch.equal(actual, expected)

    engine.factor_pool = [(2.0, 4, constant_pool)]
    unchanged = engine._apply_corr_penalty(torch.tensor(1.0), correlated_pool)
    assert unchanged.item() == pytest.approx(1.0)


@pytest.mark.parametrize("scale", [1.0, 1.0e-7, -1.0, -1.0e-7])
def test_corr_penalty_centers_affinely_shifted_factors(scale: float) -> None:
    pool_factor = torch.tensor([[-2.0, -1.0, 1.0, 2.0]], dtype=torch.float64)
    candidate = pool_factor * scale + 1000.0
    assert not torch.equal(candidate, candidate[..., :1].expand_as(candidate))
    engine = AlphaEngine.__new__(AlphaEngine)
    engine.factor_pool = [(1.0, 0, pool_factor)]

    actual = engine._apply_corr_penalty(torch.tensor(1.0), candidate)

    assert actual.item() == pytest.approx(ModelConfig.CORR_PENALTY)


@pytest.mark.parametrize("mutation", ["factor", "target", "valid"])
def test_training_loop_history_and_selection_ignore_fold_exterior(
    monkeypatch, tmp_path, mutation: str
) -> None:
    folds = [
        WalkForwardFold(0, 2, 6, 8, 12, 2),
        WalkForwardFold(1, 2, 12, 14, 18, 2),
    ]
    selected = _selection_indices(folds)
    outside = sorted(set(range(20)) - set(selected))
    factor = torch.zeros(2, 20)
    selected_values = torch.linspace(-2.0, 2.0, len(selected))
    factor[0, selected] = selected_values
    factor[1, selected] = selected_values.square()
    target = factor.clone()
    valid = torch.ones_like(target, dtype=torch.bool)

    changed_factor = factor.clone()
    changed_target = target.clone()
    changed_valid = valid.clone()
    extremes = torch.linspace(-1.0e7, 1.0e7, len(outside))
    if mutation == "factor":
        changed_factor[0, outside] = extremes
        changed_factor[1, outside] = -extremes
    elif mutation == "target":
        changed_target[0, outside] = -extremes
        changed_target[1, outside] = extremes
    else:
        factor[0, outside] = extremes
        factor[1, outside] = -extremes
        target[0, outside] = -extremes
        target[1, outside] = extremes
        changed_factor = factor.clone()
        changed_target = target.clone()
        valid[:, outside] = False
        changed_valid[:, outside] = True

    assert AlphaEngine._compute_ic(
        factor, target, valid
    ) != pytest.approx(
        AlphaEngine._compute_ic(changed_factor, changed_target, changed_valid)
    )
    baseline = _run_one_training_step(
        monkeypatch,
        tmp_path / f"baseline-{mutation}",
        factor,
        target,
        valid,
        folds,
    )
    mutated = _run_one_training_step(
        monkeypatch,
        tmp_path / f"mutated-{mutation}",
        changed_factor,
        changed_target,
        changed_valid,
        folds,
    )

    assert baseline["fold_inputs"]
    _assert_training_outcomes_equal(baseline, mutated)


def _clone_transaction_value(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _clone_transaction_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_transaction_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_transaction_value(item) for item in value)
    return value


def _assert_transaction_value_equal(actual, expected) -> None:
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor)
        assert torch.equal(actual, expected)
        return
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_transaction_value_equal(actual[key], expected[key])
        return
    if isinstance(expected, (list, tuple)):
        assert type(actual) is type(expected)
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            _assert_transaction_value_equal(actual_item, expected_item)
        return
    assert actual == expected


def _late_failure_transaction_engine(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    def execute(_formula, _features):
        return factor.clone()

    engine, factor, calls, _before = _failure_training_engine(
        monkeypatch, tmp_path, execute
    )
    engine.target_symbol = "EURUSD"
    monkeypatch.setattr(engine_module, "_CHECKPOINT_DIR", tmp_path / "checkpoints")

    engine.opt = torch.optim.Adam(engine.model.parameters(), lr=0.01)
    prime_loss = sum(parameter.square().sum() for parameter in engine.model.parameters())
    prime_loss.backward()
    engine.opt.step()
    engine.opt.zero_grad()

    engine.best_score = 0.25
    engine.best_formula = [0]
    engine._best_snapshot = _clone_transaction_value(engine.model.state_dict())
    engine._best_update_step = 7
    engine._stagnation_steps = 3
    selection_indices = [*range(2, 12), *range(14, 18)]
    engine.factor_pool = [(0.2, 4, factor[:, selection_indices].clone())]
    engine._factor_pool_counter = 5
    engine._elite_pool = [(0.2, 6, [0], 1)]
    engine._elite_counter = 7
    engine._reward_ema = 0.125
    engine._reward_ema_step = 8
    engine._restart_count = 2
    engine._low_entropy_streak = 1
    for key in engine.training_history:
        engine.training_history[key] = ["prior"]

    strategy_path = tmp_path / "strategy.json"
    history_path = tmp_path / "training_history_EURUSD.json"
    checkpoint_path = tmp_path / "checkpoints" / "ckpt_EURUSD_step_0001.pt"
    checkpoint_path.parent.mkdir(parents=True)
    artifacts = {
        strategy_path: b"prior-strategy-bytes",
        history_path: b"prior-history-bytes",
        checkpoint_path: b"prior-checkpoint-bytes",
    }
    for path, payload in artifacts.items():
        path.write_bytes(payload)

    engine._record_owned_artifact(strategy_path)
    engine._record_owned_artifact(checkpoint_path)

    real_step = engine.opt.step

    def counted_step(*args, **kwargs):
        calls["optimizer"] += 1
        return real_step(*args, **kwargs)

    engine.opt.step = counted_step

    real_strategy = AlphaEngine._save_strategy_live.__get__(engine, AlphaEngine)

    def counted_strategy(*_args):
        calls["strategy"] += 1
        return real_strategy()

    real_history = AlphaEngine._save_training_history_live.__get__(engine, AlphaEngine)

    def counted_history():
        calls["history"] += 1
        return real_history()

    formal_checkpoint = AlphaEngine.save_checkpoint.__get__(engine, AlphaEngine)

    def counted_checkpoint(step):
        calls["checkpoint"] += 1
        return formal_checkpoint(step)

    engine._save_strategy_live = counted_strategy
    engine._save_training_history_live = counted_history
    engine.save_checkpoint = counted_checkpoint

    before = {
        "model": _clone_transaction_value(engine.model.state_dict()),
        "optimizer": _clone_transaction_value(engine.opt.state_dict()),
        "best": _clone_transaction_value(
            (
                engine.best_score,
                engine.best_formula,
                engine._best_snapshot,
                engine._best_update_step,
                engine._stagnation_steps,
            )
        ),
        "pool": _clone_transaction_value(
            (
                engine.factor_pool,
                engine._factor_pool_counter,
                engine._elite_pool,
                engine._elite_counter,
            )
        ),
        "reward": (engine._reward_ema, engine._reward_ema_step),
        "history": _clone_transaction_value(engine.training_history),
        "counters": (engine._restart_count, engine._low_entropy_streak),
    }
    return engine, calls, before, artifacts


def _inject_late_training_failure(engine, calls, artifacts, stage, failure):
    strategy_path, history_path, checkpoint_path = artifacts

    def raise_failure(*_args, **_kwargs):
        raise failure

    if stage == "backward":
        def fail_backward(_gradient):
            raise failure

        next(engine.model.parameters()).register_hook(fail_backward)
    elif stage == "optimizer_before":
        def fail_optimizer_before(*_args, **_kwargs):
            calls["optimizer"] += 1
            raise failure

        engine.opt.step = fail_optimizer_before
    elif stage == "optimizer_after":
        real_step = engine.opt.step

        def fail_optimizer_after(*args, **kwargs):
            result = real_step(*args, **kwargs)
            raise failure

        engine.opt.step = fail_optimizer_after
    elif stage == "lord":
        engine.use_lord = True
        engine.lord_opt = SimpleNamespace(step=raise_failure)
    elif stage == "distribution":
        engine._distribution_stats = raise_failure
    elif stage == "strategy":
        def fail_strategy(*_args):
            calls["strategy"] += 1
            raise failure

        engine._save_strategy_live = fail_strategy
    elif stage == "history":
        def fail_history():
            calls["history"] += 1
            failure._artifact_publication_claims = {
                engine_module.pathlib.Path(history_path.name): (
                    True,
                    b"partial-new-history",
                )
            }
            history_path.write_bytes(b"partial-new-history")
            _bind_test_publication_identity(
                engine_module.pathlib.Path(history_path.name)
            )
            raise failure

        engine._save_training_history_live = fail_history
    elif stage == "checkpoint":
        def fail_checkpoint(_step):
            calls["checkpoint"] += 1
            failure._artifact_publication_claims = {
                checkpoint_path: (True, b"partial-new-checkpoint")
            }
            checkpoint_path.write_bytes(b"partial-new-checkpoint")
            _bind_test_publication_identity(checkpoint_path)
            raise failure

        engine.save_checkpoint = fail_checkpoint
    else:  # pragma: no cover - test helper guard
        raise AssertionError(stage)


def _assert_late_training_failure_rolled_back(engine, before, artifacts) -> None:
    _assert_transaction_value_equal(engine.model.state_dict(), before["model"])
    _assert_transaction_value_equal(engine.opt.state_dict(), before["optimizer"])
    _assert_transaction_value_equal(
        (
            engine.best_score,
            engine.best_formula,
            engine._best_snapshot,
            engine._best_update_step,
            engine._stagnation_steps,
        ),
        before["best"],
    )
    _assert_transaction_value_equal(
        (
            engine.factor_pool,
            engine._factor_pool_counter,
            engine._elite_pool,
            engine._elite_counter,
        ),
        before["pool"],
    )
    assert (engine._reward_ema, engine._reward_ema_step) == before["reward"]
    _assert_transaction_value_equal(engine.training_history, before["history"])
    assert (engine._restart_count, engine._low_entropy_streak) == before["counters"]
    for path, payload in artifacts.items():
        assert path.read_bytes() == payload
    root = next(iter(artifacts)).parent
    assert not list(root.rglob(".*.tmp"))


@pytest.mark.parametrize(
    "error_type",
    [RuntimeError, AssertionError, KeyboardInterrupt, SystemExit, _TrainingAbort],
)
def test_late_optimizer_mutation_preserves_exact_baseexception_and_rolls_back_batch(
    monkeypatch, tmp_path, error_type
) -> None:
    engine, calls, before, artifacts = _late_failure_transaction_engine(
        monkeypatch, tmp_path
    )
    failure = error_type("late optimizer mutation")
    _inject_late_training_failure(
        engine, calls, artifacts, "optimizer_after", failure
    )

    with pytest.raises(error_type) as caught:
        engine.train(end_step=1, verbose_header=False)

    assert caught.value is failure
    assert caught.value.__traceback__ is not None
    assert any(
        frame.function == "fail_optimizer_after"
        for frame in inspect.getinnerframes(caught.value.__traceback__)
    )
    _assert_late_training_failure_rolled_back(engine, before, artifacts)


@pytest.mark.parametrize(
    "stage",
    [
        "backward",
        "optimizer_before",
        "lord",
        "distribution",
        "strategy",
        "history",
        "checkpoint",
    ],
)
def test_every_late_batch_boundary_is_fail_closed(monkeypatch, tmp_path, stage) -> None:
    engine, calls, before, artifacts = _late_failure_transaction_engine(
        monkeypatch, tmp_path
    )
    failure = RuntimeError(f"late failure at {stage}")
    _inject_late_training_failure(engine, calls, artifacts, stage, failure)

    with pytest.raises(RuntimeError) as caught:
        engine.train(end_step=1, verbose_header=False)

    assert caught.value is failure
    assert caught.value.__traceback__ is not None
    assert not getattr(failure, "__notes__", ()), getattr(failure, "__notes__", ())
    _assert_late_training_failure_rolled_back(engine, before, artifacts)
    if stage in {"backward", "optimizer_before", "lord", "distribution"}:
        assert calls["strategy"] == 0
        assert calls["history"] == 0
        assert calls["checkpoint"] == 0


class _MutableProtocolSampler(_TinySampler):
    def __init__(self) -> None:
        self.calls = 0

    def state_dict(self):
        return {"calls": self.calls}

    def load_state_dict(self, state) -> None:
        self.calls = state["calls"]

    def apply_mask_to_logits(self, logits, *_args, **_kwargs):
        self.calls += 1
        return logits + logits.new_tensor([0.0, self.calls * 0.01])


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _transactional_outcome(engine: AlphaEngine) -> dict:
    return {
        "model": _clone_transaction_value(engine.model.state_dict()),
        "optimizer": _clone_transaction_value(engine.opt.state_dict()),
        "gradients": [
            None if parameter.grad is None else parameter.grad.detach().clone()
            for parameter in engine.model.parameters()
        ],
        "sampler": copy.deepcopy(engine.sampler.state_dict()),
        "best": _clone_transaction_value(
            (engine.best_score, engine.best_formula, engine._best_snapshot)
        ),
        "pool": _clone_transaction_value(
            (engine.factor_pool, engine._elite_pool)
        ),
        "history": _clone_transaction_value(engine.training_history),
        "state": (
            engine._factor_pool_counter,
            engine._elite_counter,
            engine._best_update_step,
            engine._stagnation_steps,
            engine._reward_ema,
            engine._reward_ema_step,
            engine._restart_count,
            engine._low_entropy_streak,
        ),
    }


def test_failed_batch_retry_restores_all_rng_and_sampler_state_before_sampling(
    monkeypatch, tmp_path
) -> None:
    def make_engine():
        engine, factor, _calls, _before = _failure_training_engine(
            monkeypatch, tmp_path, lambda _formula, _features: factor.clone()
        )
        engine.sampler = _MutableProtocolSampler()
        engine._elite_pool = [(0.25, 0, [0], 0), (0.5, 1, [1], 0)]
        engine._elite_counter = 2
        engine.best_score = 100.0
        engine.use_lord = True

        def lord_step():
            with torch.no_grad():
                for parameter in engine.model.parameters():
                    parameter.add_(torch.randn_like(parameter) * 0.001)

        engine.lord_opt = SimpleNamespace(step=lord_step)
        engine.rank_monitor = SimpleNamespace(compute=lambda: 1.0)

        def distribution_stats(_previous):
            random.random()
            np.random.random()
            torch.rand(1)
            return {
                "dist": torch.tensor([0.5, 0.5]),
                "entropy": 0.5,
                "kl_uniform": 0.0,
                "top1_prob": 0.5,
                "top5_prob": 1.0,
                "eff_vocab": 2.0,
                "prob_std": 0.0,
                "kl_prev": 0.0,
            }

        engine._distribution_stats = distribution_stats
        monkeypatch.setattr(ModelConfig, "ENTROPY_COLLAPSE_THRESH", 1.0)
        monkeypatch.setattr(ModelConfig, "ENTROPY_COLLAPSE_STEPS", 1)
        monkeypatch.setattr(ModelConfig, "MAX_RESTARTS", 1)
        monkeypatch.setattr(ModelConfig, "ADAPTIVE_NOISE", True)
        return engine

    control = make_engine()
    control_draws = []
    control_step = control.opt.step

    def control_optimizer_step(*args, **kwargs):
        result = control_step(*args, **kwargs)
        control_draws.append(
            (random.random(), float(np.random.random()), torch.rand(4).clone())
        )
        return result

    control.opt.step = control_optimizer_step
    _seed_all(77)
    control.train(end_step=1, verbose_header=False)
    control_outcome = _transactional_outcome(control)
    control_next = (random.random(), float(np.random.random()), torch.rand(4))

    retry = make_engine()
    retry_draws = []
    retry_step = retry.opt.step
    failure = KeyboardInterrupt("post-real-optimizer interruption")
    attempts = 0

    def interrupted_optimizer_step(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        result = retry_step(*args, **kwargs)
        retry_draws.append(
            (random.random(), float(np.random.random()), torch.rand(4).clone())
        )
        if attempts == 1:
            raise failure
        return result

    retry.opt.step = interrupted_optimizer_step
    _seed_all(77)
    with pytest.raises(KeyboardInterrupt) as caught:
        retry.train(end_step=1, verbose_header=False)
    assert caught.value is failure
    assert caught.value.__traceback__ is not None
    retry.train(end_step=1, verbose_header=False)
    retry_outcome = _transactional_outcome(retry)
    retry_next = (random.random(), float(np.random.random()), torch.rand(4))

    assert attempts == 2
    assert retry_draws[1][0:2] == control_draws[0][0:2]
    assert torch.equal(retry_draws[1][2], control_draws[0][2])
    _assert_transaction_value_equal(retry_outcome, control_outcome)
    assert retry_next[0:2] == control_next[0:2]
    assert torch.equal(retry_next[2], control_next[2])


def test_real_engine_preserves_extreme_finite_float64_scores_through_optimizer(
    monkeypatch, tmp_path
) -> None:
    engine, factor, _calls, _before = _failure_training_engine(
        monkeypatch,
        tmp_path,
        lambda _formula, _features: factor.clone(),
    )
    factor = factor.to(torch.float64)
    engine.data_manager.feat_tensor = engine.data_manager.feat_tensor.to(torch.float64)
    engine.data_manager.target_ret = factor.clone()
    engine.vm.execute = lambda _formula, _features: factor.clone()
    monkeypatch.setattr(ModelConfig, "CORR_THRESHOLD", 2.0)
    witness = 5.328095413325541e51
    score_calls = 0
    observed_scores = []

    def extreme_scores(**kwargs):
        nonlocal score_calls
        formula_index = score_calls // 2
        scale = (1.0, 0.5, -0.25)[formula_index]
        score_calls += 1
        value = kwargs["factors"].new_tensor(witness * scale)
        observed_scores.append(value)
        return value, value

    engine.bt.evaluate_fold = extreme_scores
    before = [parameter.detach().clone() for parameter in engine.model.parameters()]
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        engine.train(end_step=1, verbose_header=False)

    assert score_calls == 6
    assert all(score.dtype == torch.float64 for score in observed_scores)
    assert all(torch.isfinite(score) for score in observed_scores)
    gradients = [parameter.grad for parameter in engine.model.parameters()]
    assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
    assert any(torch.count_nonzero(gradient).item() for gradient in gradients)
    assert all(torch.isfinite(parameter).all() for parameter in engine.model.parameters())
    assert any(not torch.equal(old, new) for old, new in zip(before, engine.model.parameters()))


def test_float32_engine_inputs_preserve_promoted_float64_scorer_outputs(
    monkeypatch, tmp_path
) -> None:
    engine, factor, _calls, _before = _failure_training_engine(
        monkeypatch,
        tmp_path,
        lambda _formula, _features: factor.clone(),
    )
    assert factor.dtype == torch.float32
    assert engine.data_manager.feat_tensor.dtype == torch.float32
    assert engine.data_manager.target_ret.dtype == torch.float32
    monkeypatch.setattr(ModelConfig, "CORR_THRESHOLD", 2.0)
    witness = 5.328095413325541e51
    score_calls = 0
    observed_scores = []

    def promoted_scores(**_kwargs):
        nonlocal score_calls
        formula_index = score_calls // 2
        scale = (1.0, 0.5, -0.25)[formula_index]
        score_calls += 1
        value = torch.tensor(witness * scale, dtype=torch.float64)
        observed_scores.append(value)
        return value, value

    engine.bt.evaluate_fold = promoted_scores
    before = [parameter.detach().clone() for parameter in engine.model.parameters()]
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        engine.train(end_step=1, verbose_header=False)

    assert score_calls == 6
    assert all(score.dtype == torch.float64 for score in observed_scores)
    assert all(torch.isfinite(score) for score in observed_scores)
    assert math.isfinite(engine.best_score)
    assert all(math.isfinite(value) for value in engine.training_history["avg_reward"])
    gradients = [parameter.grad for parameter in engine.model.parameters()]
    assert all(
        gradient is not None and torch.isfinite(gradient).all()
        for gradient in gradients
    )
    assert any(torch.count_nonzero(gradient).item() for gradient in gradients)
    assert all(torch.isfinite(parameter).all() for parameter in engine.model.parameters())
    assert any(not torch.equal(old, new) for old, new in zip(before, engine.model.parameters()))


@pytest.mark.parametrize(
    ("input_dtype", "score_dtype"),
    [
        (torch.float32, torch.float64),
        (torch.float64, torch.float64),
        (torch.float32, torch.float32),
    ],
)
def test_real_engine_mixed_input_and_score_dtypes_keep_ordinary_gradients(
    monkeypatch, tmp_path, input_dtype, score_dtype
) -> None:
    engine, factor, _calls, _before = _failure_training_engine(
        monkeypatch,
        tmp_path,
        lambda _formula, _features: factor.clone(),
    )
    factor = factor.to(input_dtype)
    engine.data_manager.feat_tensor = engine.data_manager.feat_tensor.to(input_dtype)
    engine.data_manager.target_ret = factor.clone()
    engine.vm.execute = lambda _formula, _features: factor.clone()
    monkeypatch.setattr(ModelConfig, "CORR_THRESHOLD", 2.0)
    score_calls = 0

    def ordinary_scores(**_kwargs):
        nonlocal score_calls
        formula_index = score_calls // 2
        score_calls += 1
        value = torch.tensor(
            (4.0, 2.0, -1.0)[formula_index], dtype=score_dtype
        )
        return value, value

    engine.bt.evaluate_fold = ordinary_scores
    before = [parameter.detach().clone() for parameter in engine.model.parameters()]
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        engine.train(end_step=1, verbose_header=False)

    assert score_calls == 6
    assert math.isfinite(engine.best_score)
    gradients = [parameter.grad for parameter in engine.model.parameters()]
    assert all(
        gradient is not None and torch.isfinite(gradient).all()
        for gradient in gradients
    )
    assert any(torch.count_nonzero(gradient).item() for gradient in gradients)
    assert all(torch.isfinite(parameter).all() for parameter in engine.model.parameters())
    assert any(not torch.equal(old, new) for old, new in zip(before, engine.model.parameters()))


@pytest.mark.parametrize("target_exists", [False, True])
def test_real_final_train_publication_is_atomic_on_partial_json_failure(
    monkeypatch, tmp_path, target_exists
) -> None:
    strategy_path = tmp_path / "final-strategy.json"
    old_bytes = b'{"committed": true}\n'
    if target_exists:
        strategy_path.write_bytes(old_bytes)
    engine, factor, _calls, _before = _failure_training_engine(
        monkeypatch, tmp_path, lambda _formula, _features: factor.clone()
    )
    engine.target_symbol = "EURUSD"
    engine.best_formula = [1]
    engine.best_score = 12.5
    monkeypatch.setattr(ModelConfig, "TRAIN_STEPS", 1)
    monkeypatch.setattr(
        engine_module, "_test_strategy_file_for_symbol", lambda _symbol: str(strategy_path),
        raising=False,
    )
    _install_legacy_final_strategy_oracle(engine)
    failure = OSError("final json dump interrupted")
    real_dump = json.dump

    def partial_dump(payload, fp, *args, **kwargs):
        if set(payload) == {"vocab_version", "symbol", "formula", "best_score"}:
            fp.write("PARTIAL-FINAL")
            fp.flush()
            raise failure
        return real_dump(payload, fp, *args, **kwargs)

    monkeypatch.setattr(engine_module.json, "dump", partial_dump)
    with pytest.raises(OSError) as caught:
        engine.train(end_step=1, verbose_header=False)

    assert caught.value is failure
    if target_exists:
        assert strategy_path.read_bytes() == old_bytes
    else:
        assert not strategy_path.exists()
    assert not list(tmp_path.glob(f".{strategy_path.name}*.tmp"))


@pytest.mark.parametrize(
    ("initial", "external"),
    [
        (b"ORIGINAL", b"EXTERNAL-REPLACE"),
        (None, b"EXTERNAL-CREATE"),
        (b"ORIGINAL", None),
    ],
)
def test_batch_rollback_preserves_concurrent_external_exact_version_and_failure(
    tmp_path, initial, external
) -> None:
    path = tmp_path / "strategy.json"
    if initial is not None:
        path.write_bytes(initial)
    engine = _commit_test_engine()
    engine.model = _TinyPolicy()
    engine.opt = torch.optim.AdamW(engine.model.parameters(), lr=0.01)
    engine._reward_ema = None
    engine._reward_ema_step = 0
    engine.training_history = {"step": []}
    engine._restart_count = 0
    engine._low_entropy_streak = 0
    transaction = engine_module._BatchTransaction(engine, [path])
    if external is None:
        path.unlink()
    else:
        path.write_bytes(external)
    failure = _TrainingAbort("original batch failure")

    with pytest.raises(_TrainingAbort) as caught:
        transaction.run(lambda: (_ for _ in ()).throw(failure))

    assert caught.value is failure
    assert caught.value.__traceback__ is not None
    assert (path.read_bytes() if path.exists() else None) == external
    assert any("rollback conflict" in note for note in getattr(failure, "__notes__", ()))


def test_artifact_publication_rollback_restores_external_prepublication_version(
    tmp_path,
) -> None:
    path = tmp_path / "strategy.json"
    path.write_bytes(b"ORIGINAL")
    transaction = engine_module._BatchTransaction(
        _direct_transaction_engine(), [path]
    )
    path.write_bytes(b"EXTERNAL-BEFORE-PUBLICATION")
    transaction.run_artifact(lambda: _publish_exact(path, b"OWNED"), [path])
    failure = _TrainingAbort("fail after owned publication")

    with pytest.raises(_TrainingAbort) as caught:
        transaction.run(lambda: (_ for _ in ()).throw(failure))

    assert caught.value is failure
    assert path.read_bytes() == b"EXTERNAL-BEFORE-PUBLICATION"


@pytest.mark.parametrize("target_exists", [False, True])
@pytest.mark.parametrize("stage", ["dump", "flush", "close", "replace"])
def test_atomic_final_json_failure_matrix_preserves_target_and_cleans_owned_temp(
    monkeypatch, tmp_path, target_exists, stage
) -> None:
    target = tmp_path / "strategy.json"
    old_bytes = b'{"old": true}\n'
    if target_exists:
        target.write_bytes(old_bytes)
    failure = MemoryError(f"atomic final publication {stage} failure")
    real_open = open
    real_dump = engine_module.json.dump
    real_replace = engine_module.pathlib.Path.replace

    class FileProxy:
        def __init__(self, wrapped):
            self.wrapped = wrapped
            self.close_calls = 0

        def write(self, value):
            return self.wrapped.write(value)

        def flush(self):
            if stage == "flush":
                raise failure
            return self.wrapped.flush()

        def close(self):
            self.close_calls += 1
            if stage == "close" and self.close_calls == 1:
                raise failure
            return self.wrapped.close()

    def controlled_open(path, mode="r", *args, **kwargs):
        opened = real_open(path, mode, *args, **kwargs)
        if engine_module.pathlib.Path(path).name.startswith(f".{target.name}."):
            return FileProxy(opened)
        return opened

    def controlled_dump(payload, fp, *args, **kwargs):
        if stage == "dump":
            raise failure
        return real_dump(payload, fp, *args, **kwargs)

    def controlled_replace(source, destination):
        if stage == "replace" and engine_module.pathlib.Path(destination) == target:
            raise failure
        return real_replace(source, destination)

    monkeypatch.setattr(engine_module, "open", controlled_open, raising=False)
    monkeypatch.setattr(engine_module.json, "dump", controlled_dump)
    monkeypatch.setattr(engine_module.pathlib.Path, "replace", controlled_replace)

    with pytest.raises(MemoryError) as caught:
        engine_module._atomic_json_replace(
            target,
            {"vocab_version": "v", "symbol": "EURUSD", "formula": [1], "best_score": 1.0},
            indent=2,
        )

    assert caught.value is failure
    assert (target.read_bytes() if target.exists() else None) == (
        old_bytes if target_exists else None
    )
    assert not list(tmp_path.glob(f".{target.name}*.tmp"))


def test_atomic_final_json_handles_actual_windows_read_only_target(tmp_path) -> None:
    target = tmp_path / "strategy.json"
    old_bytes = b'{"old": true}\n'
    target.write_bytes(old_bytes)
    target.chmod(stat.S_IREAD)
    payload = {
        "vocab_version": "v",
        "symbol": "EURUSD",
        "formula": [1],
        "best_score": 1.0,
    }
    try:
        try:
            engine_module._atomic_json_replace(target, payload, indent=2)
        except OSError:
            assert target.read_bytes() == old_bytes
        else:
            assert json.loads(target.read_text()) == payload
    finally:
        target.chmod(stat.S_IREAD | stat.S_IWRITE)
    assert not list(tmp_path.glob(f".{target.name}*.tmp"))


def _direct_transaction_engine() -> AlphaEngine:
    engine = _commit_test_engine()
    engine.model = _TinyPolicy()
    engine.opt = torch.optim.AdamW(engine.model.parameters(), lr=0.01)
    engine.sampler = _MutableProtocolSampler()
    engine._reward_ema = None
    engine._reward_ema_step = 0
    engine.training_history = {"step": []}
    engine._restart_count = 0
    engine._low_entropy_streak = 0
    return engine


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt, _TrainingAbort])
def test_batch_transaction_rollback_restores_exact_identity_object(
    error_type,
) -> None:
    engine = _direct_transaction_engine()
    _install_formal_training_identity(engine)
    entry = engine.run_identity
    transaction = engine_module._BatchTransaction(engine, [], entry)
    replacement = TrainingRunIdentity.from_dict(entry.to_dict())
    assert replacement == entry and replacement is not entry
    failure = error_type("batch identity failure")

    def mutate_then_fail():
        engine.run_identity = replacement
        raise failure

    with pytest.raises(BaseException) as caught:
        transaction.run(mutate_then_fail)

    assert caught.value is failure
    assert engine.run_identity is entry


def test_v1_strategy_filename_helper_is_absent_from_production() -> None:
    assert not hasattr(engine_module, "_strategy_file_for_symbol")
    with pytest.raises(AttributeError):
        getattr(engine_module, "_strategy_file_for_symbol")


def _direct_checkpoint_engine() -> AlphaEngine:
    engine = _direct_transaction_engine()
    _install_formal_training_identity(engine)
    return engine


def _publish_exact(path, payload):
    path.write_bytes(payload)
    _bind_test_publication_identity(path)
    return engine_module._ArtifactPublicationReceipt(
        None, {path: (True, payload)}
    )


def _bind_test_publication_identity(path) -> None:
    active = engine_module._ACTIVE_ARTIFACT_PUBLICATION.get()
    assert active is not None
    transaction, _before = active
    _version, identity = engine_module._artifact_path_observation(path)
    assert identity is not None
    transaction._publication_identities[engine_module.pathlib.Path(path)] = identity


def _publish_exact_versions(versions):
    claims = {}
    for path, payload in versions.items():
        path.write_bytes(payload)
        _bind_test_publication_identity(path)
        claims[path] = (True, payload)
    return engine_module._ArtifactPublicationReceipt(None, claims)


def test_batch_rollback_restores_large_multiple_transaction_owned_artifacts(
    tmp_path,
) -> None:
    existing = tmp_path / "existing.bin"
    absent = tmp_path / "absent.bin"
    original = bytes(range(256)) * 8192
    existing.write_bytes(original)
    transaction = engine_module._BatchTransaction(
        _direct_transaction_engine(), [existing, absent]
    )

    def publish_owned_versions():
        return _publish_exact_versions(
            {
                existing: b"OWNED-REPLACEMENT",
                absent: b"OWNED-CREATE",
            }
        )

    transaction.run_artifact(publish_owned_versions, [existing, absent])
    failure = _TrainingAbort("fail after multi-artifact publication")
    with pytest.raises(_TrainingAbort) as caught:
        transaction.run(lambda: (_ for _ in ()).throw(failure))

    assert caught.value is failure
    assert existing.read_bytes() == original
    assert not absent.exists()
    assert not list(tmp_path.glob(".*.tmp"))


def test_batch_rollback_replace_failure_does_not_mask_original_or_leave_temp(
    monkeypatch, tmp_path
) -> None:
    path = tmp_path / "strategy.json"
    path.write_bytes(b"ORIGINAL")
    transaction = engine_module._BatchTransaction(
        _direct_transaction_engine(), [path]
    )
    transaction.run_artifact(lambda: _publish_exact(path, b"OWNED"), [path])
    rollback_failure = OSError("rollback replace denied")
    if engine_module.os.name == "nt":
        def fail_rollback_copy(source, kernel32, handle):
            raise rollback_failure

        monkeypatch.setattr(
            engine_module, "_windows_copy_artifact_to_handle", fail_rollback_copy
        )
    else:
        original_replace = engine_module.pathlib.Path.replace

        def fail_rollback_replace(source, target):
            if ".rollback." in source.name:
                raise rollback_failure
            return original_replace(source, target)

        monkeypatch.setattr(
            engine_module.pathlib.Path, "replace", fail_rollback_replace
        )
    failure = _TrainingAbort("original failure wins")
    with pytest.raises(_TrainingAbort) as caught:
        transaction.run(lambda: (_ for _ in ()).throw(failure))

    assert caught.value is failure
    assert path.read_bytes() == b"OWNED"
    assert any("rollback replace denied" in note for note in failure.__notes__)
    assert not list(tmp_path.glob(".*.tmp"))


def test_batch_rollback_cas_preserves_interleave_after_locked_version_check(
    monkeypatch, tmp_path,
) -> None:
    path = tmp_path / "strategy.json"
    path.write_bytes(b"ORIGINAL")
    transaction = engine_module._BatchTransaction(
        _direct_transaction_engine(), [path]
    )
    transaction.run_artifact(lambda: _publish_exact(path, b"OWNED"), [path])

    if engine_module.os.name == "nt":
        def interleave(_path, kernel32, handle):
            engine_module._windows_write_handle(
                kernel32, handle, b"EXTERNAL-AFTER-CHECK"
            )

        transaction._cas_interleave_hook = interleave
    else:
        real_version = engine_module._BatchTransaction._artifact_version
        rollback_reads = 0

        def interleaved_version(candidate):
            nonlocal rollback_reads
            if candidate == path:
                rollback_reads += 1
                if rollback_reads == 2:
                    path.write_bytes(b"EXTERNAL-AFTER-CHECK")
            return real_version(candidate)

        monkeypatch.setattr(
            engine_module._BatchTransaction,
            "_artifact_version",
            staticmethod(interleaved_version),
        )

    failure = _TrainingAbort("original failure survives CAS conflict")
    with pytest.raises(_TrainingAbort) as caught:
        transaction.run(lambda: (_ for _ in ()).throw(failure))

    assert caught.value is failure
    assert path.read_bytes() == b"EXTERNAL-AFTER-CHECK"
    assert any("rollback conflict" in note for note in failure.__notes__)


def test_transaction_and_publication_mutation_guards_use_exact_content_not_time(
    tmp_path,
) -> None:
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"ORIGINAL")
    transaction = engine_module._BatchTransaction(
        _direct_transaction_engine(), [path]
    )
    transaction.run_artifact(
        lambda: _publish_exact(path, b"OWNED---"), [path]
    )
    timestamp = path.stat().st_mtime_ns
    path.write_bytes(b"FOREIGN-")
    os.utime(path, ns=(timestamp, timestamp))
    failure = _TrainingAbort("content changed without timestamp evidence")
    with pytest.raises(_TrainingAbort) as caught:
        transaction.run(lambda: (_ for _ in ()).throw(failure))

    assert caught.value is failure
    assert path.read_bytes() == b"FOREIGN-"
    assert any("rollback conflict" in note for note in failure.__notes__)


def test_pure_transaction_operations_do_not_rescan_or_materialize_artifacts(
    monkeypatch, tmp_path
) -> None:
    paths = [tmp_path / f"artifact-{index}.bin" for index in range(6)]
    for index, path in enumerate(paths[:3]):
        path.write_bytes(bytes([index]) * 4096)
    attempts = []
    real_snapshot = engine_module._snapshot_artifact_to_owned_backup

    def counted_snapshot(path):
        attempts.append(path)
        return real_snapshot(path)

    monkeypatch.setattr(
        engine_module,
        "_snapshot_artifact_to_owned_backup",
        counted_snapshot,
    )
    transaction = engine_module._BatchTransaction(
        _direct_transaction_engine(), paths
    )
    construction_attempts = len(attempts)

    for value in range(8):
        assert transaction.run(lambda item=value: item) == value

    assert construction_attempts == len(paths)
    assert len(attempts) == construction_attempts


def test_artifact_publication_observes_only_its_declared_path(
    monkeypatch, tmp_path
) -> None:
    paths = [tmp_path / f"artifact-{index}.bin" for index in range(6)]
    for path in paths:
        path.write_bytes(b"ORIGINAL")
    attempts = []
    real_observation = engine_module._BatchTransaction._artifact_observation

    def counted_observation(path):
        attempts.append(path)
        return real_observation(path)

    monkeypatch.setattr(
        engine_module._BatchTransaction,
        "_artifact_observation",
        staticmethod(counted_observation),
    )
    transaction = engine_module._BatchTransaction(
        _direct_transaction_engine(), paths
    )
    attempts.clear()

    transaction.run_artifact(
        lambda: _publish_exact(paths[2], b"OWNED"), [paths[2]]
    )

    assert attempts == [paths[2], paths[2]]


def test_real_one_step_training_artifact_probes_are_bounded_by_publications(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(engine_module, "_CHECKPOINT_DIR", tmp_path / "checkpoints")
    engine, factor, _calls, _before = _failure_training_engine(
        monkeypatch, tmp_path, lambda _formula, _features: factor.clone()
    )
    engine.target_symbol = "EURUSD"
    engine.best_score = float("inf")
    attempts = []
    construction_attempts = []
    real_observation = engine_module._BatchTransaction._artifact_observation
    real_snapshot = engine_module._snapshot_artifact_to_owned_backup

    def counted_observation(path):
        attempts.append(path)
        return real_observation(path)

    def counted_snapshot(path):
        construction_attempts.append(path)
        return real_snapshot(path)

    monkeypatch.setattr(
        engine_module._BatchTransaction,
        "_artifact_observation",
        staticmethod(counted_observation),
    )
    monkeypatch.setattr(
        engine_module,
        "_snapshot_artifact_to_owned_backup",
        counted_snapshot,
    )
    engine.train(end_step=1, verbose_header=False)

    # Six construction probes plus two paths before/after history and checkpoint.
    assert len(construction_attempts) == 6
    assert len(attempts) == 8
    assert len(construction_attempts) + len(attempts) == 14
    counts = {path: attempts.count(path) for path in set(attempts)}
    assert set(counts.values()) == {2}


def test_real_train_restart_failure_rolls_back_prebatch_state_and_optimizer_identity(
    monkeypatch, tmp_path
) -> None:
    engine, factor, _calls, _before = _failure_training_engine(
        monkeypatch, tmp_path, lambda _formula, _features: factor.clone()
    )
    monkeypatch.setattr(ModelConfig, "ENTROPY_COLLAPSE_THRESH", 1.0)
    monkeypatch.setattr(ModelConfig, "ENTROPY_COLLAPSE_STEPS", 1)
    monkeypatch.setattr(ModelConfig, "MAX_RESTARTS", 1)
    monkeypatch.setattr(ModelConfig, "FULL_RESET_EVERY", 1)
    failure = RuntimeError("reset mutated logits before failing")
    original_optimizer = engine.opt
    model_before = _clone_transaction_value(engine.model.state_dict())
    optimizer_before = _clone_transaction_value(engine.opt.state_dict())
    history_before = _clone_transaction_value(engine.training_history)
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()

    def mutate_then_fail():
        with torch.no_grad():
            engine.model.logits.add_(123.0)
        raise failure

    engine.model.reset_parameters = mutate_then_fail
    with pytest.raises(RuntimeError) as caught:
        engine.train(end_step=1, verbose_header=False)

    assert caught.value is failure
    assert caught.value.__traceback__ is not None
    assert engine.opt is original_optimizer
    _assert_transaction_value_equal(engine.model.state_dict(), model_before)
    _assert_transaction_value_equal(engine.opt.state_dict(), optimizer_before)
    _assert_transaction_value_equal(engine.training_history, history_before)
    assert (engine._restart_count, engine._low_entropy_streak) == (0, 0)
    assert random.getstate() == python_before
    numpy_after = np.random.get_state()
    assert numpy_after[0] == numpy_before[0]
    assert np.array_equal(numpy_after[1], numpy_before[1])
    assert numpy_after[2:] == numpy_before[2:]
    assert torch.equal(torch.get_rng_state(), torch_before)


@pytest.mark.parametrize(
    "stage",
    [
        "reset_before",
        "reset_after",
        "snapshot_before",
        "snapshot_after",
        "noise_before",
        "noise_after",
        "optimizer_before",
        "optimizer_after",
    ],
)
def test_every_adaptive_restart_mutation_failure_rolls_back_complete_batch(
    monkeypatch, tmp_path, stage
) -> None:
    engine, calls, before, artifacts = _late_failure_transaction_engine(
        monkeypatch, tmp_path
    )
    engine.sampler = _MutableProtocolSampler()
    sampler_before = copy.deepcopy(engine.sampler.state_dict())
    original_optimizer = engine.opt
    monkeypatch.setattr(ModelConfig, "ENTROPY_COLLAPSE_THRESH", 1.0)
    monkeypatch.setattr(ModelConfig, "ENTROPY_COLLAPSE_STEPS", 1)
    monkeypatch.setattr(ModelConfig, "MAX_RESTARTS", 4)
    monkeypatch.setattr(ModelConfig, "PARTIAL_RESET", True)
    monkeypatch.setattr(ModelConfig, "PARTIAL_RESET_LAYERS", ("logits",))
    monkeypatch.setattr(
        ModelConfig,
        "FULL_RESET_EVERY",
        1 if stage.startswith("reset") else 99,
    )
    failure = _TrainingAbort(f"restart failure at {stage}")
    injected = False

    if stage.startswith("reset"):
        def fail_reset():
            nonlocal injected
            if not injected:
                injected = True
                if stage.endswith("after"):
                    with torch.no_grad():
                        engine.model.logits.add_(77.0)
                raise failure

        engine.model.reset_parameters = fail_reset
    elif stage.startswith("snapshot"):
        real_load = engine.model.load_state_dict

        def fail_snapshot(state, *args, **kwargs):
            nonlocal injected
            if not injected:
                injected = True
                if stage.endswith("after"):
                    real_load(state, *args, **kwargs)
                raise failure
            return real_load(state, *args, **kwargs)

        engine.model.load_state_dict = fail_snapshot
    elif stage.startswith("noise"):
        real_randn_like = engine_module.torch.randn_like

        def fail_noise(value, *args, **kwargs):
            nonlocal injected
            if not injected:
                injected = True
                if stage.endswith("after"):
                    real_randn_like(value, *args, **kwargs)
                raise failure
            return real_randn_like(value, *args, **kwargs)

        monkeypatch.setattr(engine_module.torch, "randn_like", fail_noise)
    else:
        real_replace = engine._replace_optimizer_for_restart

        def fail_optimizer_replace():
            nonlocal injected
            if not injected:
                injected = True
                if stage.endswith("after"):
                    real_replace()
                raise failure
            return real_replace()

        engine._replace_optimizer_for_restart = fail_optimizer_replace

    _seed_all(923)
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()
    with pytest.raises(_TrainingAbort) as caught:
        engine.train(end_step=1, verbose_header=False)

    assert caught.value is failure
    assert caught.value.__traceback__ is not None
    assert injected
    assert engine.opt is original_optimizer
    _assert_late_training_failure_rolled_back(engine, before, artifacts)
    assert engine.sampler.state_dict() == sampler_before
    assert random.getstate() == python_before
    numpy_after = np.random.get_state()
    assert numpy_after[0] == numpy_before[0]
    assert np.array_equal(numpy_after[1], numpy_before[1])
    assert numpy_after[2:] == numpy_before[2:]
    assert torch.equal(torch.get_rng_state(), torch_before)
    assert calls["optimizer"] == 1


@pytest.mark.parametrize(
    "branch", ["full_reset", "best_snapshot", "no_snapshot", "max_restarts"]
)
def test_every_adaptive_restart_branch_commits_one_new_optimizer(
    monkeypatch, tmp_path, branch
) -> None:
    engine, factor, _calls, _before = _failure_training_engine(
        monkeypatch, tmp_path, lambda _formula, _features: factor.clone()
    )
    engine.best_score = float("inf")
    monkeypatch.setattr(ModelConfig, "ENTROPY_COLLAPSE_THRESH", 1.0)
    monkeypatch.setattr(ModelConfig, "ENTROPY_COLLAPSE_STEPS", 1)
    monkeypatch.setattr(ModelConfig, "MAX_RESTARTS", 2)
    monkeypatch.setattr(ModelConfig, "PARTIAL_RESET", True)
    monkeypatch.setattr(ModelConfig, "PARTIAL_RESET_LAYERS", ("logits",))
    monkeypatch.setattr(
        ModelConfig, "FULL_RESET_EVERY", 1 if branch == "full_reset" else 99
    )
    if branch in {"best_snapshot", "max_restarts"}:
        engine._best_snapshot = _clone_transaction_value(engine.model.state_dict())
    else:
        engine._best_snapshot = None
    if branch == "max_restarts":
        engine._restart_count = ModelConfig.MAX_RESTARTS
    original_optimizer = engine.opt
    real_replace = engine._replace_optimizer_for_restart
    replacements = 0

    def counted_replace():
        nonlocal replacements
        replacements += 1
        return real_replace()

    engine._replace_optimizer_for_restart = counted_replace
    engine.train(end_step=1, verbose_header=False)

    assert replacements == 1
    assert engine.opt is not original_optimizer
    assert isinstance(engine.opt, torch.optim.AdamW)
    if branch == "max_restarts":
        assert engine._restart_count == ModelConfig.MAX_RESTARTS
    else:
        assert engine._restart_count == 1
    assert engine._low_entropy_streak == 0


def test_failed_restart_after_optimizer_replacement_retries_like_control(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(engine_module, "_CHECKPOINT_DIR", tmp_path / "checkpoints")
    monkeypatch.setattr(ModelConfig, "ENTROPY_COLLAPSE_THRESH", 1.0)
    monkeypatch.setattr(ModelConfig, "ENTROPY_COLLAPSE_STEPS", 1)
    monkeypatch.setattr(ModelConfig, "MAX_RESTARTS", 2)
    monkeypatch.setattr(ModelConfig, "FULL_RESET_EVERY", 99)

    def make_engine():
        engine, factor, _calls, _before = _failure_training_engine(
            monkeypatch, tmp_path, lambda _formula, _features: factor.clone()
        )
        monkeypatch.setattr(ModelConfig, "ENTROPY_COLLAPSE_THRESH", 1.0)
        monkeypatch.setattr(ModelConfig, "ENTROPY_COLLAPSE_STEPS", 1)
        monkeypatch.setattr(ModelConfig, "MAX_RESTARTS", 2)
        monkeypatch.setattr(ModelConfig, "FULL_RESET_EVERY", 99)
        engine.best_score = float("inf")
        engine._best_snapshot = None
        engine.sampler = _MutableProtocolSampler()
        return engine

    control = make_engine()
    _seed_all(451)
    control.train(end_step=1, verbose_header=False)
    control_outcome = _transactional_outcome(control)
    control_next = (random.random(), float(np.random.random()), torch.rand(4))

    retry = make_engine()
    original_optimizer = retry.opt
    real_replace = retry._replace_optimizer_for_restart
    failure = KeyboardInterrupt("fail after optimizer replacement")
    attempts = 0

    def fail_once_after_replace():
        nonlocal attempts
        attempts += 1
        real_replace()
        if attempts == 1:
            raise failure

    retry._replace_optimizer_for_restart = fail_once_after_replace
    _seed_all(451)
    with pytest.raises(KeyboardInterrupt) as caught:
        retry.train(end_step=1, verbose_header=False)
    assert caught.value is failure
    assert retry.opt is original_optimizer
    retry.train(end_step=1, verbose_header=False)
    retry_outcome = _transactional_outcome(retry)
    retry_next = (random.random(), float(np.random.random()), torch.rand(4))

    assert attempts == 2
    _assert_transaction_value_equal(retry_outcome, control_outcome)
    assert retry_next[:2] == control_next[:2]
    assert torch.equal(retry_next[2], control_next[2])


@pytest.mark.parametrize("target_exists", [False, True])
@pytest.mark.parametrize(
    ("stage", "error_type"),
    [
        ("dump", OSError),
        ("flush", MemoryError),
        ("close", KeyboardInterrupt),
        ("replace", _TrainingAbort),
    ],
)
def test_real_final_training_history_publication_failure_matrix_is_atomic(
    monkeypatch, tmp_path, target_exists, stage, error_type
) -> None:
    monkeypatch.chdir(tmp_path)
    history_path = tmp_path / "training_history_EURUSD.json"
    prior = b'{"complete":true}\n'
    if target_exists:
        history_path.write_bytes(prior)
    engine, factor, _calls, _before = _failure_training_engine(
        monkeypatch, tmp_path, lambda _formula, _features: factor.clone()
    )
    engine.target_symbol = "EURUSD"
    engine.best_score = float("inf")
    engine.best_formula = [0]
    monkeypatch.setattr(ModelConfig, "TRAIN_STEPS", 1)
    failure = error_type(f"final history {stage} interrupted")
    real_open = open
    real_dump = engine_module.json.dump
    real_replace = engine_module.pathlib.Path.replace
    real_windows_replace = engine_module._windows_replace_file

    class FileProxy:
        def __init__(self, wrapped):
            self.wrapped = wrapped
            self.close_calls = 0

        def write(self, value):
            return self.wrapped.write(value)

        def flush(self):
            if stage == "flush":
                raise failure
            return self.wrapped.flush()

        def close(self):
            self.close_calls += 1
            if stage == "close" and self.close_calls == 1:
                raise failure
            return self.wrapped.close()

    def controlled_open(path, mode="r", *args, **kwargs):
        opened = real_open(path, mode, *args, **kwargs)
        name = engine_module.pathlib.Path(path).name
        if name.startswith(f".{history_path.name}."):
            return FileProxy(opened)
        return opened

    def partial_history_dump(payload, fp, *args, **kwargs):
        if stage == "dump" and payload is engine.training_history:
            fp.write("PARTIAL-HISTORY")
            fp.flush()
            raise failure
        return real_dump(payload, fp, *args, **kwargs)

    def controlled_replace(source, destination):
        if (
            stage == "replace"
            and engine_module.pathlib.Path(destination).name == history_path.name
        ):
            raise failure
        return real_replace(source, destination)

    def controlled_windows_replace(target, replacement, *args, **kwargs):
        if (
            stage == "replace"
            and engine_module.pathlib.Path(target).name == history_path.name
        ):
            raise failure
        return real_windows_replace(target, replacement, *args, **kwargs)

    monkeypatch.setattr(engine_module, "open", controlled_open, raising=False)
    monkeypatch.setattr(engine_module.json, "dump", partial_history_dump)
    monkeypatch.setattr(engine_module.pathlib.Path, "replace", controlled_replace)
    monkeypatch.setattr(
        engine_module, "_windows_replace_file", controlled_windows_replace
    )
    with pytest.raises(error_type) as caught:
        engine.train(end_step=1, verbose_header=False)

    assert caught.value is failure
    assert caught.value.__traceback__ is not None
    assert (history_path.read_bytes() if history_path.exists() else None) == (
        prior if target_exists else None
    )
    assert not list(tmp_path.glob(f".{history_path.name}.*.tmp"))


def test_final_training_history_success_preserves_payload_on_unicode_path(
    monkeypatch, tmp_path
) -> None:
    unicode_root = tmp_path / "历史曲线"
    unicode_root.mkdir()
    monkeypatch.chdir(unicode_root)
    engine, factor, _calls, _before = _failure_training_engine(
        monkeypatch, unicode_root, lambda _formula, _features: factor.clone()
    )
    engine.target_symbol = "欧元美元"
    engine.best_score = float("inf")
    engine.best_formula = [0]
    monkeypatch.setattr(ModelConfig, "TRAIN_STEPS", 1)

    engine.train(end_step=1, verbose_header=False)

    history_path = unicode_root / "training_history_欧元美元.json"
    assert json.loads(history_path.read_text()) == engine.training_history
    assert not list(unicode_root.glob(f".{history_path.name}.*.tmp"))


def test_atomic_final_history_handles_actual_windows_read_only_target(
    tmp_path,
) -> None:
    target = tmp_path / "training_history_只读.json"
    old_bytes = b'{"complete":true}\n'
    target.write_bytes(old_bytes)
    target.chmod(stat.S_IREAD)
    payload = {"step": [0], "avg_reward": [1.25]}
    try:
        try:
            engine_module._atomic_json_replace(target, payload)
        except OSError:
            assert target.read_bytes() == old_bytes
        else:
            assert json.loads(target.read_text()) == payload
    finally:
        target.chmod(stat.S_IREAD | stat.S_IWRITE)
    assert not list(tmp_path.glob(f".{target.name}.*.tmp"))


def test_final_history_mutation_guard_rejects_direct_target_write(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.chdir(tmp_path)
    engine, factor, _calls, _before = _failure_training_engine(
        monkeypatch, tmp_path, lambda _formula, _features: factor.clone()
    )
    monkeypatch.setattr(ModelConfig, "TRAIN_STEPS", 1)
    publications = []
    real_atomic = engine_module._atomic_json_replace

    def traced_atomic(path, payload):
        publications.append(pathlib.Path(path))
        return real_atomic(path, payload)

    monkeypatch.setattr(engine_module, "_atomic_json_replace", traced_atomic)
    engine.train(end_step=1, verbose_header=False)

    assert len(publications) == 1
    assert json.loads(publications[0].read_text()) == engine.training_history
    assert not list(tmp_path.glob(f".{publications[0].name}.*.tmp"))


def test_repair23_transaction_and_restart_mutation_guards(
    monkeypatch, tmp_path
) -> None:
    engine, factor, _calls, _before = _failure_training_engine(
        monkeypatch, tmp_path, lambda _formula, _features: factor.clone()
    )
    monkeypatch.setattr(ModelConfig, "MIGRATION_INTERVAL", 1)
    events = []
    real_restart = engine._apply_adaptive_restart
    real_checkpoint = engine.save_checkpoint
    real_commit = engine_module._BatchTransaction.commit

    def restart(*args, **kwargs):
        events.append("restart")
        return real_restart(*args, **kwargs)

    def checkpoint(*args, **kwargs):
        events.append("checkpoint")
        return real_checkpoint(*args, **kwargs)

    def migration(*_args):
        events.append("migration")

    def commit(transaction):
        events.append("commit")
        return real_commit(transaction)

    engine._apply_adaptive_restart = restart
    engine.save_checkpoint = checkpoint
    monkeypatch.setattr(engine_module._BatchTransaction, "commit", commit)
    engine.train(
        start_step=0, end_step=1, migration_hook=migration,
        verbose_header=False,
    )

    assert events == ["restart", "checkpoint", "migration", "commit"]


def test_callback_without_publication_claim_never_owns_external_write(tmp_path) -> None:
    path = tmp_path / "strategy.json"
    path.write_bytes(b"ORIGINAL")
    transaction = engine_module._BatchTransaction(
        _direct_transaction_engine(), [path]
    )

    def callback_without_write() -> None:
        writer = threading.Thread(
            target=path.write_bytes,
            args=(b"EXTERNAL-DURING-PUBLICATION",),
        )
        writer.start()
        writer.join()

    transaction.run_artifact(callback_without_write, [path])
    failure = _TrainingAbort("fail after external publication-window write")
    cause = ValueError("publication cause")
    context = LookupError("publication context")
    failure.__cause__ = cause
    failure.__context__ = context
    with pytest.raises(_TrainingAbort) as caught:
        transaction.run(lambda: (_ for _ in ()).throw(failure))

    assert caught.value is failure
    assert caught.value.__traceback__ is not None
    assert caught.value.__cause__ is cause
    assert caught.value.__context__ is context
    assert path.read_bytes() == b"EXTERNAL-DURING-PUBLICATION"


def test_transaction_claim_loses_ownership_to_external_publication_race(
    tmp_path,
) -> None:
    path = tmp_path / "strategy.json"
    path.write_bytes(b"ORIGINAL")
    transaction = engine_module._BatchTransaction(
        _direct_transaction_engine(), [path]
    )

    def publish_then_lose_race():
        path.write_bytes(b"TRANSACTION-PUBLICATION")
        writer = threading.Thread(
            target=path.write_bytes,
            args=(b"EXTERNAL-AFTER-TRANSACTION-WRITE",),
        )
        writer.start()
        writer.join()
        return engine_module._ArtifactPublicationReceipt(
            None, {path: (True, b"TRANSACTION-PUBLICATION")}
        )

    transaction.run_artifact(publish_then_lose_race, [path])
    failure = _TrainingAbort("fail after competing publication")
    with pytest.raises(_TrainingAbort) as caught:
        transaction.run(lambda: (_ for _ in ()).throw(failure))

    assert caught.value is failure
    assert path.read_bytes() == b"EXTERNAL-AFTER-TRANSACTION-WRITE"
    assert any("rollback conflict" in note for note in failure.__notes__)
    assert len(failure.__notes__) == 1
    assert len(failure.__notes__[0]) <= 400


def test_real_history_publication_fails_closed_on_external_before_replace(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "training_history_EURUSD.json"
    path.write_bytes(b"ORIGINAL")
    engine = _direct_transaction_engine()
    engine.target_symbol = "EURUSD"
    engine.training_history = {"step": [1], "avg_reward": [0.25]}
    pathlib_path = engine_module.pathlib.Path(path.name)
    transaction = engine_module._BatchTransaction(
        engine,
        [
            pathlib_path,
            pathlib_path.with_name(f".{pathlib_path.name}.tmp"),
        ],
    )
    interleaved = False

    def external_before_replace(target):
        nonlocal interleaved
        assert target == pathlib_path
        interleaved = True
        pathlib_path.write_bytes(b"EXTERNAL-BEFORE-REPLACE")

    transaction._publication_cas_interleave_hook = external_before_replace
    with pytest.raises(RuntimeError, match="publication conflict"):
        transaction.run_artifact(
            engine._save_training_history_live,
            [pathlib_path, pathlib_path.with_name(f".{pathlib_path.name}.tmp")],
        )

    assert interleaved
    assert path.read_bytes() == b"EXTERNAL-BEFORE-REPLACE"
    assert not list(tmp_path.glob(".*.tmp"))


@pytest.mark.parametrize("target_exists", [False, True])
@pytest.mark.parametrize("stage", ["partial", "flush", "close", "replace"])
def test_live_history_transaction_publication_failure_is_atomic(
    monkeypatch, tmp_path, target_exists, stage
) -> None:
    monkeypatch.chdir(tmp_path)
    path = engine_module.pathlib.Path("training_history_EURUSD.json")
    temporary = path.with_name(f".{path.name}.tmp")
    prior = b'EXTERNAL-OR-PRIOR-EXACT-BYTES\x00\xff'
    if target_exists:
        path.write_bytes(prior)
    engine = _direct_transaction_engine()
    engine.target_symbol = "EURUSD"
    engine.training_history = {"step": [1], "avg_reward": [0.25]}
    transaction = engine_module._BatchTransaction(engine, [path, temporary])
    failure = _TrainingAbort(f"live publication {stage} failure")
    real_open = open
    real_dump = engine_module.json.dump
    real_replace = engine_module.pathlib.Path.replace
    real_windows_replace = engine_module._windows_replace_file

    class TemporaryProxy:
        def __init__(self, wrapped):
            self.wrapped = wrapped

        def __getattr__(self, name):
            return getattr(self.wrapped, name)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, exc_traceback):
            self.close()
            return False

        def write(self, value):
            return self.wrapped.write(value)

        def flush(self):
            if stage == "flush":
                raise failure
            return self.wrapped.flush()

        def close(self):
            self.wrapped.close()
            if stage == "close":
                raise failure

    def controlled_open(candidate, mode="r", *args, **kwargs):
        opened = real_open(candidate, mode, *args, **kwargs)
        if engine_module.pathlib.Path(candidate) == temporary:
            return TemporaryProxy(opened)
        return opened

    def controlled_dump(payload, fp, *args, **kwargs):
        if stage == "partial":
            fp.write("PARTIAL-PUBLICATION")
            raise failure
        return real_dump(payload, fp, *args, **kwargs)

    def controlled_replace(source, destination):
        if stage == "replace" and source == temporary and destination == path:
            raise failure
        return real_replace(source, destination)

    def controlled_windows_replace(target, replacement, *args, **kwargs):
        if stage == "replace" and target == path and replacement == temporary:
            raise failure
        return real_windows_replace(target, replacement, *args, **kwargs)

    monkeypatch.setattr(engine_module, "open", controlled_open, raising=False)
    monkeypatch.setattr(engine_module.json, "dump", controlled_dump)
    monkeypatch.setattr(engine_module.pathlib.Path, "replace", controlled_replace)
    monkeypatch.setattr(
        engine_module, "_windows_replace_file", controlled_windows_replace
    )
    with pytest.raises(_TrainingAbort) as caught:
        transaction.run_artifact(
            engine._save_training_history_live, [path, temporary]
        )

    assert caught.value is failure
    assert caught.value.__traceback__ is not None
    assert (path.read_bytes() if path.exists() else None) == (
        prior if target_exists else None
    )
    assert not temporary.exists()


def test_atomic_replace_entry_competing_write_is_excluded_or_preserved(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.chdir(tmp_path)
    path = engine_module.pathlib.Path("training_history_EURUSD.json")
    temporary = path.with_name(f".{path.name}.tmp")
    path.write_bytes(b"ORIGINAL")
    engine = _direct_transaction_engine()
    engine.target_symbol = "EURUSD"
    engine.training_history = {"step": [1]}
    transaction = engine_module._BatchTransaction(engine, [path, temporary])
    failure = _TrainingAbort("competing write at atomic replace entry")
    real_replace = engine_module.pathlib.Path.replace
    real_windows_replace = engine_module._windows_replace_file
    competing_write_error = None

    def competing_replace(source, destination):
        nonlocal competing_write_error
        if source == temporary and destination == path:
            try:
                path.write_bytes(b"EXTERNAL-AT-REPLACE")
            except OSError as error:
                competing_write_error = error
            raise failure
        return real_replace(source, destination)

    def competing_windows_replace(target, replacement, *args, **kwargs):
        nonlocal competing_write_error
        if target == path and replacement == temporary:
            try:
                path.write_bytes(b"EXTERNAL-AT-REPLACE")
            except OSError as error:
                competing_write_error = error
            raise failure
        return real_windows_replace(target, replacement, *args, **kwargs)

    monkeypatch.setattr(engine_module.pathlib.Path, "replace", competing_replace)
    monkeypatch.setattr(
        engine_module, "_windows_replace_file", competing_windows_replace
    )
    with pytest.raises(_TrainingAbort) as caught:
        transaction.run_artifact(
            engine._save_training_history_live, [path, temporary]
        )

    assert caught.value is failure
    if engine_module.os.name == "nt":
        assert competing_write_error is not None
        assert path.read_bytes() == b"ORIGINAL"
    else:
        assert competing_write_error is None
        assert path.read_bytes() == b"EXTERNAL-AT-REPLACE"
    assert not temporary.exists()


@pytest.mark.parametrize("target_exists", [False, True])
@pytest.mark.parametrize("external_api", ["os_replace", "replace_file"])
def test_windows_guard_detects_external_atomic_replace_after_baseline_check(
    monkeypatch, tmp_path, target_exists, external_api
) -> None:
    monkeypatch.chdir(tmp_path)
    path = engine_module.pathlib.Path("training_history_EURUSD.json")
    temporary = path.with_name(f".{path.name}.tmp")
    external_temp = path.with_name(f".{path.name}.external.tmp")
    if target_exists:
        path.write_bytes(b"ORIGINAL")
    engine = _direct_transaction_engine()
    engine.target_symbol = "EURUSD"
    engine.training_history = {"step": [1]}
    transaction = engine_module._BatchTransaction(engine, [path, temporary])
    external = (
        b"EXTERNAL-ATOMIC-REPLACE-PRESENT"
        if target_exists
        else b"EXTERNAL-ATOMIC-REPLACE-PLACEHOLDER"
    )
    replace_succeeded = False
    replace_failure = None

    def external_replace_after_locked_check(target):
        nonlocal replace_succeeded, replace_failure
        assert target == path
        external_temp.write_bytes(external)
        try:
            if external_api == "replace_file" and engine_module.os.name == "nt":
                engine_module._windows_replace_file(target, external_temp)
            else:
                engine_module.os.replace(external_temp, target)
        except OSError as failure:
            replace_failure = failure
            raise
        replace_succeeded = True

    transaction._publication_locked_interleave_hook = (
        external_replace_after_locked_check
    )
    if engine_module.os.name == "nt":
        if external_api == "os_replace":
            with pytest.raises((PermissionError, RuntimeError)) as caught:
                transaction.run_artifact(
                    engine._save_training_history_live, [path, temporary]
                )
            if replace_succeeded:
                assert replace_failure is None
                assert isinstance(caught.value, RuntimeError)
                assert "publication conflict" in str(caught.value)
                assert path.read_bytes() == external
                assert not external_temp.exists()
            else:
                assert caught.value is replace_failure
                assert (path.read_bytes() if path.exists() else None) == (
                    b"ORIGINAL" if target_exists else None
                )
                assert external_temp.read_bytes() == external
        else:
            with pytest.raises(RuntimeError, match="publication conflict"):
                transaction.run_artifact(
                    engine._save_training_history_live, [path, temporary]
                )
            assert replace_succeeded
            assert replace_failure is None
            assert path.read_bytes() == external
            assert not external_temp.exists()
    else:
        transaction._publication_cas_interleave_hook = (
            external_replace_after_locked_check
        )
        transaction._publication_locked_interleave_hook = None
        with pytest.raises(RuntimeError, match="publication conflict"):
            transaction.run_artifact(
                engine._save_training_history_live, [path, temporary]
            )
        assert replace_succeeded
        assert path.read_bytes() == external
        assert not external_temp.exists()
    assert not temporary.exists()


@pytest.mark.parametrize("target_exists", [False, True])
def test_windows_publication_preserves_external_replace_after_final_link_check(
    monkeypatch, tmp_path, target_exists
) -> None:
    assert engine_module.os.name == "nt"
    monkeypatch.chdir(tmp_path)
    path = engine_module.pathlib.Path("training_history_欧元美元.json")
    temporary = path.with_name(f".{path.name}.tmp")
    external_temp = path.with_name(f".{path.name}.external.tmp")
    foreign_sibling = path.with_name(f".{path.name}.foreign.keep")
    if target_exists:
        path.write_bytes(b"ORIGINAL")
    foreign_sibling.write_bytes(b"UNRELATED-FOREIGN-SIBLING")
    engine = _direct_transaction_engine()
    engine.target_symbol = "欧元美元"
    engine.training_history = {"step": [1]}
    transaction = engine_module._BatchTransaction(engine, [path, temporary])
    external = (
        b"EXTERNAL-AFTER-FINAL-LINK-CHECK-PRESENT"
        if target_exists
        else b"EXTERNAL-AFTER-FINAL-LINK-CHECK-PLACEHOLDER"
    )
    event_order = []
    real_windows_replace = engine_module._windows_replace_file

    def external_then_candidate(target, replacement, *args, **kwargs):
        if target == path and replacement == temporary:
            assert path.exists()
            external_temp.write_bytes(external)
            real_windows_replace(target, external_temp)
            assert path.read_bytes() == external
            assert not external_temp.exists()
            event_order.append("external-observable")
        result = real_windows_replace(target, replacement, *args, **kwargs)
        if target == path and replacement == temporary:
            event_order.append("candidate-returned")
        return result

    monkeypatch.setattr(
        engine_module, "_windows_replace_file", external_then_candidate
    )
    with pytest.raises(RuntimeError, match="publication conflict"):
        transaction.run_artifact(
            engine._save_training_history_live, [path, temporary]
        )

    assert event_order[0] == "external-observable"
    assert path.read_bytes() == external
    assert foreign_sibling.read_bytes() == b"UNRELATED-FOREIGN-SIBLING"
    assert not temporary.exists()
    assert not external_temp.exists()


def _repair100_engine(monkeypatch, tmp_path):
    engine = _direct_checkpoint_engine()
    engine.sampler = _MutableProtocolSampler()
    parameter = next(engine.model.parameters())
    engine.opt.state[parameter] = {
        "step": torch.tensor(2.0),
        "nested": {"moment": torch.full_like(parameter, 0.25)},
    }
    engine.factor_pool = [(0.5, 3, torch.tensor([1.0, 2.0]))]
    engine.factor_pool_scores = [0.5]
    engine._elite_pool = [(0.7, 4, [1, 2], 5)]
    engine.elite_pool_ages = [5]
    engine.training_history = {"step": [1], "nested": {"reward": [0.5]}}
    engine.scheduler = SimpleNamespace(state={"epoch": 3, "nested": [1, 2]})
    engine.scaler = SimpleNamespace(state={"scale": torch.tensor(8.0)})
    monkeypatch.chdir(tmp_path)
    return engine


def _repair100_state_snapshot(engine):
    return {
        "transactional": _transactional_outcome(engine),
        "factor_pool_scores": _clone_transaction_value(engine.factor_pool_scores),
        "elite_pool_ages": _clone_transaction_value(engine.elite_pool_ages),
        "scheduler": _clone_transaction_value(engine.scheduler.__dict__),
        "scaler": _clone_transaction_value(engine.scaler.__dict__),
        "optimizer_object": engine.opt,
        "scheduler_object": engine.scheduler,
        "scaler_object": engine.scaler,
    }


def _repair100_rng_snapshot():
    return (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state().clone(),
        [state.clone() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_initialized()
        else [],
    )


def _assert_repair100_rng_equal(actual, expected) -> None:
    assert actual[0] == expected[0]
    assert actual[1][0] == expected[1][0]
    assert np.array_equal(actual[1][1], expected[1][1])
    assert actual[1][2:] == expected[1][2:]
    assert torch.equal(actual[2], expected[2])
    assert len(actual[3]) == len(expected[3])
    assert all(torch.equal(left, right) for left, right in zip(actual[3], expected[3]))


def _mutate_repair100_entry(engine, entry, identity_variant) -> None:
    _replace_training_identity(engine, entry, identity_variant)
    engine.best_score = 98765.0
    with torch.no_grad():
        next(engine.model.parameters()).add_(11.0)
    engine.opt.param_groups[0]["lr"] = 9.0
    next(iter(engine.opt.state.values()))["nested"]["moment"].add_(7.0)
    engine.factor_pool[0][2].add_(5.0)
    engine.factor_pool_scores.append(99.0)
    engine._elite_pool.append((9.0, 99, [9], 99))
    engine.elite_pool_ages.append(99)
    engine.training_history["nested"]["reward"].append(99.0)
    engine.sampler.calls = 99
    engine.scheduler.state["nested"].append(99)
    engine.scaler.state["scale"].add_(4.0)
    random.random()
    np.random.random()
    torch.rand(3)
    if torch.cuda.is_initialized():
        torch.cuda.get_rng_state_all()


@pytest.mark.parametrize(
    "identity_variant", ["equal-distinct", "different", "deleted", "none", "hostile"]
)
def test_true_entry_snapshot_precedes_first_deepcopy_and_restores_everything(
    monkeypatch, tmp_path, identity_variant
) -> None:
    engine = _repair100_engine(monkeypatch, tmp_path)
    entry = engine.run_identity
    _seed_all(10001)
    before_state = _repair100_state_snapshot(engine)
    before_rng = _repair100_rng_snapshot()
    real_deepcopy = copy.deepcopy
    real_state_dict = engine.model.state_dict
    deepcopy_calls = 0
    later_callbacks = []

    def mutating_deepcopy(value, *args, **kwargs):
        nonlocal deepcopy_calls
        deepcopy_calls += 1
        if deepcopy_calls == 1:
            _mutate_repair100_entry(engine, entry, identity_variant)
        return real_deepcopy(value, *args, **kwargs)

    def forbidden_state_dict(*_args, **_kwargs):
        later_callbacks.append("model.state_dict")
        raise AssertionError("later snapshot callback reached")

    monkeypatch.setattr(copy, "deepcopy", mutating_deepcopy)
    monkeypatch.setattr(engine.model, "state_dict", forbidden_state_dict)
    transaction = engine._begin_batch_transaction(1, entry)
    transaction.rollback()

    monkeypatch.setattr(copy, "deepcopy", real_deepcopy)
    monkeypatch.setattr(engine.model, "state_dict", real_state_dict)
    assert deepcopy_calls == 0
    assert later_callbacks == []
    assert engine.run_identity is entry
    _assert_transaction_value_equal(_repair100_state_snapshot(engine), before_state)
    _assert_repair100_rng_equal(_repair100_rng_snapshot(), before_rng)
    assert list(tmp_path.rglob("*")) == []


@pytest.mark.parametrize(
    "error_type", [RuntimeError, KeyboardInterrupt, SystemExit, GeneratorExit, _TrainingAbort]
)
def test_first_deepcopy_baseexception_restores_true_entry_and_primary_graph(
    monkeypatch, tmp_path, error_type
) -> None:
    engine = _repair100_engine(monkeypatch, tmp_path)
    entry = engine.run_identity
    _seed_all(10002)
    before_state = _repair100_state_snapshot(engine)
    before_rng = _repair100_rng_snapshot()
    real_deepcopy = copy.deepcopy
    failure = error_type("first deepcopy failure")
    cause = ValueError("deepcopy cause")
    context = LookupError("deepcopy context")
    failure.__cause__ = cause
    failure.__context__ = context

    def failing_deepcopy(value, *args, **kwargs):
        _mutate_repair100_entry(engine, entry, "equal-distinct")
        raise failure

    monkeypatch.setattr(copy, "deepcopy", failing_deepcopy)
    transaction = engine._begin_batch_transaction(1, entry)
    transaction.rollback()

    monkeypatch.setattr(copy, "deepcopy", real_deepcopy)
    assert failure.__traceback__ is None
    assert engine.run_identity is entry
    _assert_transaction_value_equal(_repair100_state_snapshot(engine), before_state)
    _assert_repair100_rng_equal(_repair100_rng_snapshot(), before_rng)
    assert list(tmp_path.rglob("*")) == []


@pytest.mark.parametrize("trigger_call", [2, 5, 11])
def test_deeper_deepcopy_mutation_reuses_true_entry_not_intermediate_snapshot(
    monkeypatch, tmp_path, trigger_call
) -> None:
    engine = _repair100_engine(monkeypatch, tmp_path)
    entry = engine.run_identity
    _seed_all(10003)
    before_state = _repair100_state_snapshot(engine)
    before_rng = _repair100_rng_snapshot()
    real_deepcopy = copy.deepcopy
    deepcopy_calls = 0

    def mutating_deepcopy(value, *args, **kwargs):
        nonlocal deepcopy_calls
        deepcopy_calls += 1
        if deepcopy_calls == trigger_call:
            _mutate_repair100_entry(engine, entry, "different")
        return real_deepcopy(value, *args, **kwargs)

    monkeypatch.setattr(copy, "deepcopy", mutating_deepcopy)
    transaction = engine._begin_batch_transaction(1, entry)
    transaction.rollback()

    monkeypatch.setattr(copy, "deepcopy", real_deepcopy)
    assert deepcopy_calls == 0
    assert engine.run_identity is entry
    _assert_transaction_value_equal(_repair100_state_snapshot(engine), before_state)
    _assert_repair100_rng_equal(_repair100_rng_snapshot(), before_rng)
    assert list(tmp_path.rglob("*")) == []


def test_normal_true_entry_snapshot_and_direct_rollback_preserve_identity(
    monkeypatch, tmp_path
) -> None:
    engine = _repair100_engine(monkeypatch, tmp_path)
    entry = engine.run_identity
    before_state = _repair100_state_snapshot(engine)
    before_rng = _repair100_rng_snapshot()
    transaction = engine._begin_batch_transaction(1, entry)
    assert transaction._entry_run_identity is entry
    transaction.rollback()
    assert engine.run_identity is entry
    _assert_transaction_value_equal(_repair100_state_snapshot(engine), before_state)
    _assert_repair100_rng_equal(_repair100_rng_snapshot(), before_rng)


class _Repair105UnsafeTensor(torch.Tensor):
    callback = None
    failure = None

    @classmethod
    def create(cls, value):
        return torch.Tensor._make_subclass(cls, value, value.requires_grad)

    @classmethod
    def trip(cls):
        if cls.callback is not None:
            cls.callback()
        if cls.failure is not None:
            raise cls.failure

    def detach(self):
        type(self).trip()
        return super().detach()

    def clone(self, *args, **kwargs):
        type(self).trip()
        return super().clone(*args, **kwargs)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        cls.trip()
        return super().__torch_function__(func, types, args, kwargs or {})


class _Repair105UnsafeParameter(torch.nn.Parameter):
    callback = None
    failure = None

    @classmethod
    def trip(cls):
        if cls.callback is not None:
            cls.callback()
        if cls.failure is not None:
            raise cls.failure

    def detach(self):
        type(self).trip()
        return super().detach()

    def clone(self, *args, **kwargs):
        type(self).trip()
        return super().clone(*args, **kwargs)


class _Repair105HostileDict(dict):
    callback = None

    def _trip(self):
        if type(self).callback is not None:
            type(self).callback()
        raise AssertionError("hostile mapping protocol invoked")

    items = _trip
    values = _trip
    keys = _trip
    __iter__ = _trip
    __len__ = _trip
    __getitem__ = _trip


class _Repair105HostileList(list):
    callback = None

    def _trip(self, *_args):
        if type(self).callback is not None:
            type(self).callback()
        raise AssertionError("hostile list protocol invoked")

    __iter__ = _trip
    __len__ = _trip
    __getitem__ = _trip


class _Repair105HostileTuple(tuple):
    callback = None

    def _trip(self, *_args):
        if type(self).callback is not None:
            type(self).callback()
        raise AssertionError("hostile tuple protocol invoked")

    __iter__ = _trip
    __len__ = _trip
    __getitem__ = _trip


def _repair105_first_parameter_slot(engine):
    for module in engine.model.modules():
        attributes = object.__getattribute__(module, "__dict__")
        parameters = attributes.get("_parameters", {})
        for name, parameter in dict.items(parameters):
            if parameter is not None:
                return module, name, parameter
    raise AssertionError("test model has no parameter")


def _install_repair105_unsafe_tensor(engine, location):
    module, name, parameter = _repair105_first_parameter_slot(engine)
    base = torch.ones_like(parameter.detach())
    unsafe_tensor = _Repair105UnsafeTensor.create(base)
    if location == "parameter":
        unsafe_parameter = _Repair105UnsafeParameter(
            parameter.detach().clone(), requires_grad=parameter.requires_grad
        )
        object.__getattribute__(module, "__dict__")["_parameters"][name] = unsafe_parameter
        return unsafe_parameter
    if location == "buffer":
        object.__getattribute__(engine.model, "__dict__")["_buffers"]["repair105"] = unsafe_tensor
    elif location == "gradient":
        parameter.grad = unsafe_tensor
    elif location == "optimizer":
        engine.opt.state[parameter]["repair105"] = {"nested": unsafe_tensor}
    elif location == "best":
        engine.best_metrics = {"repair105": unsafe_tensor}
    elif location == "pool":
        engine.factor_pool.append((0.9, 9, unsafe_tensor))
    elif location == "history":
        engine.training_history["repair105"] = [unsafe_tensor]
    elif location == "sampler":
        engine.sampler.repair105 = unsafe_tensor
    elif location == "scheduler":
        engine.scheduler.state["repair105"] = unsafe_tensor
    elif location == "scaler":
        engine.scaler.state["repair105"] = unsafe_tensor
    else:
        raise AssertionError(location)
    return unsafe_tensor


@pytest.mark.parametrize(
    "location",
    [
        "parameter", "buffer", "gradient", "optimizer", "best",
        "pool", "history", "sampler", "scheduler", "scaler",
    ],
)
def test_unsafe_tensor_subclass_is_rejected_before_any_callback(
    monkeypatch, tmp_path, location
) -> None:
    engine = _repair100_engine(monkeypatch, tmp_path)
    entry = engine.run_identity
    installed = _install_repair105_unsafe_tensor(engine, location)
    _seed_all(10501)
    before_rng = _repair100_rng_snapshot()
    before_score = engine.best_score
    before_optimizer = engine.opt
    callbacks = []
    later_callbacks = []

    def malicious_callback():
        callbacks.append(location)
        engine.run_identity = TrainingRunIdentity.from_dict(entry.to_dict())
        engine.best_score = 105.0
        random.random()
        np.random.random()
        torch.rand(1)

    def forbidden_deepcopy(*_args, **_kwargs):
        later_callbacks.append("deepcopy")
        raise AssertionError("later snapshot callback reached")

    _Repair105UnsafeTensor.callback = malicious_callback
    _Repair105UnsafeTensor.failure = RuntimeError("unsafe tensor callback")
    _Repair105UnsafeParameter.callback = malicious_callback
    _Repair105UnsafeParameter.failure = RuntimeError("unsafe parameter callback")
    monkeypatch.setattr(copy, "deepcopy", forbidden_deepcopy)
    try:
        with pytest.raises(ArtifactCompatibilityError, match="unsafe|unsupported"):
            engine._begin_batch_transaction(1, entry)
    finally:
        _Repair105UnsafeTensor.callback = None
        _Repair105UnsafeTensor.failure = None
        _Repair105UnsafeParameter.callback = None
        _Repair105UnsafeParameter.failure = None

    assert callbacks == []
    assert later_callbacks == []
    assert engine.run_identity is entry
    assert engine.best_score == before_score
    assert engine.opt is before_optimizer
    assert installed is installed
    _assert_repair100_rng_equal(_repair100_rng_snapshot(), before_rng)
    assert list(tmp_path.rglob("*")) == []


@pytest.mark.parametrize(
    "error_type", [RuntimeError, KeyboardInterrupt, SystemExit, GeneratorExit, _TrainingAbort]
)
def test_unsafe_parameter_primary_never_runs_before_true_snapshot(
    monkeypatch, tmp_path, error_type
) -> None:
    engine = _repair100_engine(monkeypatch, tmp_path)
    entry = engine.run_identity
    _install_repair105_unsafe_tensor(engine, "parameter")
    failure = error_type("pre-guard tensor callback")
    callbacks = []
    _Repair105UnsafeParameter.callback = lambda: callbacks.append(failure)
    _Repair105UnsafeParameter.failure = failure
    try:
        with pytest.raises(BaseException) as caught:
            engine._begin_batch_transaction(1, entry)
    finally:
        _Repair105UnsafeParameter.callback = None
        _Repair105UnsafeParameter.failure = None
    assert isinstance(caught.value, ArtifactCompatibilityError)
    assert callbacks == []
    assert engine.run_identity is entry


@pytest.mark.parametrize(
    "container_type", [_Repair105HostileDict, _Repair105HostileList, _Repair105HostileTuple]
)
def test_snapshot_preflight_rejects_container_subclasses_without_protocols(
    monkeypatch, tmp_path, container_type
) -> None:
    engine = _repair100_engine(monkeypatch, tmp_path)
    entry = engine.run_identity
    callbacks = []
    container_type.callback = lambda: callbacks.append(container_type.__name__)
    if issubclass(container_type, dict):
        engine.training_history = container_type({"unsafe": 1})
    else:
        engine.factor_pool = container_type([(0.1, 1, torch.tensor([1.0]))])
    try:
        with pytest.raises(ArtifactCompatibilityError, match="unsafe|unsupported"):
            engine._begin_batch_transaction(1, entry)
    finally:
        container_type.callback = None
    assert callbacks == []
    assert engine.run_identity is entry
    assert list(tmp_path.rglob("*")) == []


def test_exact_base_snapshot_preserves_internal_tensor_aliases(monkeypatch, tmp_path):
    engine = _repair100_engine(monkeypatch, tmp_path)
    entry = engine.run_identity
    shared = torch.arange(6.0).reshape(2, 3).t()
    assert not shared.is_contiguous()
    parameter = engine.opt.param_groups[0]["params"][0]
    engine.training_history = {"left": shared, "right": shared}
    transaction = engine._begin_batch_transaction(1, entry)
    engine.training_history = {}
    transaction.rollback()
    assert engine.run_identity is entry
    assert engine.training_history["left"] is engine.training_history["right"]
    assert type(engine.training_history["left"]) is torch.Tensor
    assert engine.training_history["left"].stride() == shared.stride()
    assert engine.opt.param_groups[0]["params"][0] is parameter
    assert parameter in engine.opt.state


class _Repair109DispatchMode(torch.utils._python_dispatch.TorchDispatchMode):
    def __init__(self, callback, failure):
        super().__init__()
        self.callback = callback
        self.failure = failure

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.callback()
        raise self.failure


class _Repair109FunctionMode(torch.overrides.TorchFunctionMode):
    def __init__(self, callback, failure):
        super().__init__()
        self.callback = callback
        self.failure = failure

    def __torch_function__(self, func, types, args=(), kwargs=None):
        self.callback()
        raise self.failure


@pytest.mark.parametrize(
    "mode_type", [_Repair109DispatchMode, _Repair109FunctionMode]
)
@pytest.mark.parametrize(
    "error_type", [RuntimeError, KeyboardInterrupt, SystemExit, GeneratorExit, _TrainingAbort]
)
def test_active_global_torch_mode_rejects_before_dispatch_and_snapshot(
    monkeypatch, tmp_path, mode_type, error_type
) -> None:
    engine = _repair100_engine(monkeypatch, tmp_path)
    entry = engine.run_identity
    _seed_all(10901)
    before_rng = _repair100_rng_snapshot()
    before_score = engine.best_score
    before_lr = engine.opt.param_groups[0]["lr"]
    failure = error_type("global torch mode callback")
    callbacks = []

    def hostile_callback():
        callbacks.append(type(failure).__name__)
        engine.run_identity = TrainingRunIdentity.from_dict(entry.to_dict())
        engine.best_score = 109.0
        engine.opt.param_groups[0]["lr"] = 109.0
        random.random()
        np.random.random()

    mode = mode_type(hostile_callback, failure)
    with mode:
        with pytest.raises(BaseException) as caught:
            engine._begin_batch_transaction(1, entry)

    assert isinstance(caught.value, ArtifactCompatibilityError)
    assert callbacks == []
    assert engine.run_identity is entry
    assert engine.best_score == before_score
    assert engine.opt.param_groups[0]["lr"] == before_lr
    _assert_repair100_rng_equal(_repair100_rng_snapshot(), before_rng)
    assert list(tmp_path.rglob("*")) == []


@pytest.mark.parametrize(
    "mode_type", [_Repair109DispatchMode, _Repair109FunctionMode]
)
def test_nested_global_torch_modes_reject_without_checking_only_top(
    monkeypatch, tmp_path, mode_type
) -> None:
    engine = _repair100_engine(monkeypatch, tmp_path)
    entry = engine.run_identity
    callbacks = []
    lower = mode_type(lambda: callbacks.append("lower"), RuntimeError("lower"))
    upper = mode_type(lambda: callbacks.append("upper"), RuntimeError("upper"))
    with lower, upper:
        with pytest.raises(ArtifactCompatibilityError, match="active.*mode"):
            engine._begin_batch_transaction(1, entry)
    assert callbacks == []
    assert engine.run_identity is entry


def test_snapshot_preflight_has_no_isinstance_or_mode_stack_mutation(
    monkeypatch, tmp_path
) -> None:
    engine = _repair100_engine(monkeypatch, tmp_path)
    entry = engine.run_identity
    callbacks = []
    before_dispatch = torch._C._len_torch_dispatch_stack()
    before_function = torch.overrides._len_torch_function_stack()
    dispatch = _Repair109DispatchMode(
        lambda: callbacks.append("dispatch"), RuntimeError("dispatch")
    )
    function = _Repair109FunctionMode(
        lambda: callbacks.append("function"), RuntimeError("function")
    )

    with dispatch, function:
        active_depths = (
            torch._C._len_torch_dispatch_stack(),
            torch.overrides._len_torch_function_stack(),
        )
        with pytest.raises(ArtifactCompatibilityError, match="active.*mode"):
            engine._begin_batch_transaction(1, entry)
        assert (
            torch._C._len_torch_dispatch_stack(),
            torch.overrides._len_torch_function_stack(),
        ) == active_depths

    assert callbacks == []
    assert engine.run_identity is entry
    assert torch._C._len_torch_dispatch_stack() == before_dispatch
    assert torch.overrides._len_torch_function_stack() == before_function


def test_no_global_torch_mode_keeps_exact_base_snapshot_supported(
    monkeypatch, tmp_path
) -> None:
    engine = _repair100_engine(monkeypatch, tmp_path)
    entry = engine.run_identity
    transaction = engine._begin_batch_transaction(1, entry)
    transaction.rollback()
    assert engine.run_identity is entry


@pytest.mark.parametrize(
    "error_type", [RuntimeError, KeyboardInterrupt, SystemExit, GeneratorExit, _TrainingAbort]
)
def test_migration_failure_rolls_back_entire_training_batch(
    monkeypatch, tmp_path, error_type
) -> None:
    monkeypatch.chdir(tmp_path)
    engine, factor, _calls, before = _failure_training_engine(
        monkeypatch, tmp_path, lambda _formula, _features: factor.clone()
    )
    monkeypatch.setattr(ModelConfig, "MIGRATION_INTERVAL", 1)
    failure = error_type("migration callback failure")
    _seed_all(11401)
    before_rng = _repair100_rng_snapshot()

    def migration_callback(current, _step):
        current.best_score = 114.0
        random.random()
        np.random.random()
        torch.rand(1)
        raise failure

    with pytest.raises(BaseException) as caught:
        engine.train(
            start_step=0, end_step=1,
            migration_hook=migration_callback, verbose_header=False,
        )

    assert caught.value is failure
    for name, expected in before["model"].items():
        assert torch.equal(engine.model.state_dict()[name], expected)
    _assert_transaction_value_equal(engine.opt.state_dict(), before["optimizer"])
    assert engine.training_history == before["history"]
    assert engine.factor_pool == before["factor_pool"]
    assert engine._elite_pool == before["elite_pool"]
    assert (engine.best_score, engine.best_formula, engine._best_snapshot) == before["best"]
    _assert_repair100_rng_equal(_repair100_rng_snapshot(), before_rng)


@pytest.mark.parametrize(
    "error_type", [RuntimeError, KeyboardInterrupt, SystemExit, GeneratorExit, _TrainingAbort]
)
def test_transaction_rollback_preserves_cross_component_tensor_object(
    monkeypatch, tmp_path, error_type
) -> None:
    engine = _repair100_engine(monkeypatch, tmp_path)
    entry = engine.run_identity
    parameter = engine.opt.param_groups[0]["params"][0]
    shared = torch.tensor([1.0, 2.0])
    engine.opt.state[parameter]["cross_shared"] = {"value": shared}
    engine.training_history["cross_shared"] = {"value": shared}
    before_value = shared.clone()
    failure = error_type("cross-component rollback")
    transaction = engine._begin_batch_transaction(1, entry)

    def mutate_then_fail():
        shared.add_(9.0)
        raise failure

    with pytest.raises(BaseException) as caught:
        transaction.run(mutate_then_fail)

    assert caught.value is failure
    optimizer_shared = engine.opt.state[parameter]["cross_shared"]["value"]
    history_shared = engine.training_history["cross_shared"]["value"]
    assert optimizer_shared is shared
    assert history_shared is shared
    assert optimizer_shared is history_shared
    assert torch.equal(shared, before_value)
    assert engine.opt.param_groups[0]["params"][0] is parameter


@pytest.mark.parametrize(
    "error_type", [RuntimeError, KeyboardInterrupt, SystemExit, GeneratorExit, _TrainingAbort]
)
def test_rank_monitor_history_is_in_exact_transaction_graph(
    monkeypatch, tmp_path, error_type
) -> None:
    engine = _repair100_engine(monkeypatch, tmp_path)
    shared = torch.tensor([7.25])
    history = [shared]
    engine.rank_monitor = SimpleNamespace(history=history)
    engine.training_history["rank_alias"] = shared
    failure = error_type("late rank failure")
    transaction = engine._begin_batch_transaction(1, engine.run_identity)

    def mutate_then_fail():
        shared.add_(3.0)
        engine.rank_monitor.history.append(torch.tensor([8.5]))
        raise failure

    with pytest.raises(BaseException) as caught:
        transaction.run(mutate_then_fail)

    assert caught.value is failure
    assert len(engine.rank_monitor.history) == 1
    assert engine.rank_monitor.history[0] is shared
    assert engine.training_history["rank_alias"] is shared
    assert torch.equal(shared, torch.tensor([7.25]))


@pytest.mark.parametrize("history_size", [1, 10_000, 100_000])
def test_batch_snapshot_has_no_redundant_deepcopy_or_state_dict_pass(
    monkeypatch, tmp_path, history_size
) -> None:
    engine = _repair100_engine(monkeypatch, tmp_path)
    engine.training_history["scaling"] = list(range(history_size))
    scaling = engine.training_history["scaling"]
    counts = {
        "deepcopy": 0, "model_state": 0, "optimizer_state": 0,
        "history_capture_visits": 0, "history_preflight_visits": 0,
        "history_sync_visits": 0,
    }
    real_deepcopy = engine_module.copy.deepcopy
    real_model_state = engine.model.state_dict
    real_optimizer_state = engine.opt.state_dict
    real_capture = engine_module._capture_true_transaction_graph
    real_exact_capture = engine_module._capture_exact_object_graph
    real_preflight = engine_module._preflight_true_transaction_value
    capture_active = False
    sync_active = False

    def counted_deepcopy(*args, **kwargs):
        counts["deepcopy"] += 1
        return real_deepcopy(*args, **kwargs)

    def counted_model_state(*args, **kwargs):
        counts["model_state"] += 1
        return real_model_state(*args, **kwargs)

    def counted_optimizer_state(*args, **kwargs):
        counts["optimizer_state"] += 1
        return real_optimizer_state(*args, **kwargs)

    def counted_capture(value, *args, **kwargs):
        nonlocal capture_active
        root = value is scaling
        if root:
            capture_active = True
        if capture_active:
            counts["history_capture_visits"] += 1
        try:
            return real_capture(value, *args, **kwargs)
        finally:
            if root:
                capture_active = False

    def counted_preflight(value, path, seen):
        if path.startswith("training_history"):
            counts["history_preflight_visits"] += 1
        return real_preflight(value, path, seen)

    def counted_exact_capture(value, *args, **kwargs):
        nonlocal sync_active
        root = value is engine.training_history
        if root:
            sync_active = True
        if sync_active:
            counts["history_sync_visits"] += 1
        try:
            return real_exact_capture(value, *args, **kwargs)
        finally:
            if root:
                sync_active = False

    monkeypatch.setattr(engine_module.copy, "deepcopy", counted_deepcopy)
    monkeypatch.setattr(engine.model, "state_dict", counted_model_state)
    monkeypatch.setattr(engine.opt, "state_dict", counted_optimizer_state)
    monkeypatch.setattr(
        engine_module, "_capture_true_transaction_graph", counted_capture
    )
    monkeypatch.setattr(
        engine_module, "_preflight_true_transaction_value", counted_preflight
    )
    monkeypatch.setattr(
        engine_module, "_capture_exact_object_graph", counted_exact_capture
    )
    transaction = engine._begin_batch_transaction(1, engine.run_identity)

    assert counts["deepcopy"] == 0
    assert counts["model_state"] == 0
    assert counts["optimizer_state"] == 0
    assert counts["history_capture_visits"] <= 1
    assert counts["history_preflight_visits"] <= 16
    assert counts["history_sync_visits"] >= history_size
    for dead_field in (
        "model_state", "optimizer_state", "state", "gradients",
        "python_rng_state", "numpy_rng_state", "torch_cpu_rng_state",
        "torch_cuda_rng_state", "sampler_state", "optimizer",
    ):
        assert not hasattr(transaction, dead_field)
    transaction.rollback()
    counts["history_sync_visits"] = 0
    for step in range(2, 5):
        repeated = engine._begin_batch_transaction(step, engine.run_identity)
        repeated.commit()
    assert counts["history_sync_visits"] == 0


def test_batch_history_rollback_is_append_bounded_and_preserves_aliases(
    monkeypatch, tmp_path
) -> None:
    engine = _repair100_engine(monkeypatch, tmp_path)
    shared = list(range(10_000))
    history = {"step": shared, "alias": shared}
    engine.training_history = history
    entry_values = list(shared)
    failure = _TrainingAbort("history append failed")
    transaction = engine._begin_batch_transaction(1, engine.run_identity)

    def append_then_fail():
        shared[0] = 999_999
        shared.append(10_000)
        engine.training_history["new_metric"] = [1.0]
        raise failure

    with pytest.raises(_TrainingAbort) as caught:
        transaction.run(append_then_fail)

    assert caught.value is failure
    assert engine.training_history is history
    assert engine.training_history["step"] is shared
    assert engine.training_history["alias"] is shared
    assert shared == entry_values
    assert "new_metric" not in history

    success = engine._begin_batch_transaction(2, engine.run_identity)
    shared.append(10_000)
    success.commit()
    assert engine.training_history is history
    assert engine.training_history["step"] is shared
    assert engine.training_history["alias"] is shared
    assert shared[-1] == 10_000


def test_batch_history_rollback_uses_latest_successful_commit(
    monkeypatch, tmp_path
) -> None:
    engine = _repair100_engine(monkeypatch, tmp_path)
    shared = list(range(10_000))
    history = {"step": shared, "alias": shared}
    engine.training_history = history

    successful = engine._begin_batch_transaction(1, engine.run_identity)
    shared.append(10_000)
    successful.commit()
    committed = list(shared)
    failure = _TrainingAbort("rollback must use latest committed history")
    failing = engine._begin_batch_transaction(2, engine.run_identity)

    def mutate_then_fail():
        shared[0] = 999_999
        shared.append(10_001)
        engine.training_history["new_metric"] = [1.0]
        raise failure

    with pytest.raises(_TrainingAbort) as caught:
        failing.run(mutate_then_fail)

    assert caught.value is failure
    assert engine.training_history is history
    assert engine.training_history["step"] is shared
    assert engine.training_history["alias"] is shared
    assert shared == committed
    assert shared[-1] == 10_000
    assert "new_metric" not in history


def test_island_engine_rejects_before_manager_or_artifact_access() -> None:
    accesses = []

    class HostileManager:
        def __getattribute__(self, name):
            accesses.append(name)
            raise AssertionError("island rejection touched the manager")

    with pytest.raises(
        ArtifactCompatibilityError, match="only single-symbol training"
    ):
        IslandAlphaEngine(HostileManager())

    assert accesses == []


class _HostileIdentityReplacement:
    def __repr__(self):
        raise AssertionError("hostile identity repr must not run")

    def __str__(self):
        raise AssertionError("hostile identity str must not run")


class _HostileRollbackCleanup(BaseException):
    def __repr__(self):
        raise AssertionError("hostile rollback repr must not run")

    def __str__(self):
        raise AssertionError("hostile rollback str must not run")


def _replace_training_identity(engine, entry, variant: str) -> None:
    if variant == "deleted":
        object.__delattr__(engine, "run_identity")
    elif variant == "none":
        object.__setattr__(engine, "run_identity", None)
    elif variant == "different":
        replacement = TrainingRunIdentity.from_dict(entry.to_dict())
        object.__setattr__(replacement, "run_id", "8" * 32)
        object.__setattr__(engine, "run_identity", replacement)
    elif variant == "equal-distinct":
        replacement = TrainingRunIdentity.from_dict(entry.to_dict())
        assert replacement == entry and replacement is not entry
        object.__setattr__(engine, "run_identity", replacement)
    elif variant == "hostile":
        object.__setattr__(engine, "run_identity", _HostileIdentityReplacement())
    else:
        raise AssertionError(f"unknown variant: {variant}")


@pytest.mark.parametrize(
    "variant", ["deleted", "none", "different", "equal-distinct", "hostile"]
)
def test_batch_setup_filename_callback_restores_exact_entry_before_next_boundary(
    monkeypatch, tmp_path, variant
) -> None:
    monkeypatch.chdir(tmp_path)
    engine = _direct_checkpoint_engine()
    entry = engine.run_identity
    next_boundary_calls = []
    real_strategy_filename = TrainingRunIdentity.strategy_filename

    def mutating_strategy_filename(identity):
        result = real_strategy_filename(identity)
        _replace_training_identity(engine, entry, variant)
        return result

    def forbidden_history_filename(_identity):
        next_boundary_calls.append("history")
        raise AssertionError("next filename callback reached")

    monkeypatch.setattr(
        TrainingRunIdentity, "strategy_filename", mutating_strategy_filename
    )
    monkeypatch.setattr(
        TrainingRunIdentity, "history_filename", forbidden_history_filename
    )

    with pytest.raises(ArtifactCompatibilityError):
        engine._begin_batch_transaction(1, entry)

    assert engine.run_identity is entry
    assert next_boundary_calls == []
    assert list(tmp_path.rglob("*")) == []


@pytest.mark.parametrize(
    "error_type", [RuntimeError, KeyboardInterrupt, SystemExit, GeneratorExit, _TrainingAbort]
)
def test_batch_setup_state_snapshot_failure_restores_entry_state_and_rng(
    monkeypatch, tmp_path, error_type
) -> None:
    monkeypatch.chdir(tmp_path)
    engine = _direct_checkpoint_engine()
    entry = engine.run_identity
    before_state = _transactional_outcome(engine)
    _seed_all(9501)
    before_rng = (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state().clone(),
    )
    real_state_dict = engine.model.state_dict
    failure = error_type("batch setup snapshot failure")
    cause = ValueError("setup cause")
    context = LookupError("setup context")
    failure.__cause__ = cause
    failure.__context__ = context

    def mutate_then_fail(*args, **kwargs):
        real_state_dict(*args, **kwargs)
        replacement = TrainingRunIdentity.from_dict(entry.to_dict())
        object.__setattr__(engine, "run_identity", replacement)
        engine.best_score = 12345.0
        random.random()
        raise failure

    monkeypatch.setattr(engine.model, "state_dict", mutate_then_fail)

    transaction = engine._begin_batch_transaction(1, entry)
    transaction.rollback()

    monkeypatch.setattr(engine.model, "state_dict", real_state_dict)
    assert failure.__traceback__ is None
    assert engine.run_identity is entry
    _assert_transaction_value_equal(_transactional_outcome(engine), before_state)
    assert random.getstate() == before_rng[0]
    assert all(
        np.array_equal(left, right)
        for left, right in zip(np.random.get_state(), before_rng[1])
    )
    assert torch.equal(torch.get_rng_state(), before_rng[2])
    assert list(tmp_path.rglob("*")) == []


@pytest.mark.parametrize(
    "variant", ["deleted", "none", "different", "equal-distinct", "hostile"]
)
@pytest.mark.parametrize("target_exists", [False, True])
def test_final_history_publication_identity_mutation_rolls_back_before_decode(
    monkeypatch, tmp_path, variant, target_exists
) -> None:
    monkeypatch.chdir(tmp_path)
    history_path = tmp_path / "training_history_EURUSD.json"
    prior = b'FOREIGN-OR-PRIOR-HISTORY\x00\xff'
    if target_exists:
        history_path.write_bytes(prior)
    prior_identity = (
        engine_module._artifact_path_observation(history_path)[1]
        if target_exists
        else None
    )
    engine, factor, _calls, _before = _failure_training_engine(
        monkeypatch, tmp_path, lambda _formula, _features: factor.clone()
    )
    engine.target_symbol = "EURUSD"
    engine.sampler = _MutableProtocolSampler()
    engine.best_score = float("inf")
    engine.best_formula = [0]
    monkeypatch.setattr(ModelConfig, "TRAIN_STEPS", 1)
    entry = engine.run_identity
    real_atomic = engine_module._atomic_json_replace
    publication_snapshot = {}
    decode_after_publication = []

    def mutating_atomic(path, payload, *args, **kwargs):
        result = real_atomic(path, payload, *args, **kwargs)
        publication_snapshot["state"] = _transactional_outcome(engine)
        publication_snapshot["rng"] = (
            random.getstate(),
            np.random.get_state(),
            torch.get_rng_state().clone(),
        )
        _replace_training_identity(engine, entry, variant)
        return result

    def decode_after_atomic(_formula):
        if publication_snapshot:
            decode_after_publication.append(True)
            raise AssertionError("decode reached after invalid history publication")
        return "tiny"

    monkeypatch.setattr(engine_module, "_atomic_json_replace", mutating_atomic)
    engine._decode_formula = decode_after_atomic

    with pytest.raises(ArtifactCompatibilityError):
        engine.train(end_step=1, verbose_header=False)

    assert engine.run_identity is entry
    assert decode_after_publication == []
    _assert_transaction_value_equal(
        _transactional_outcome(engine), publication_snapshot["state"]
    )
    after_rng = publication_snapshot["rng"]
    assert random.getstate() == after_rng[0]
    assert all(
        np.array_equal(left, right)
        for left, right in zip(np.random.get_state(), after_rng[1])
    )
    assert torch.equal(torch.get_rng_state(), after_rng[2])
    assert (history_path.read_bytes() if history_path.exists() else None) == (
        prior if target_exists else None
    )
    if target_exists:
        assert engine_module._artifact_path_observation(history_path)[1] == prior_identity
    assert not list(tmp_path.glob(".training_history_EURUSD.json.*.tmp"))


def test_final_history_identity_error_survives_hostile_rollback_cleanup(
    monkeypatch, tmp_path
) -> None:
    assert engine_module.os.name == "nt"
    monkeypatch.chdir(tmp_path)
    history_path = tmp_path / "training_history_EURUSD.json"
    prior = b"PRIOR-HISTORY-OBJECT"
    history_path.write_bytes(prior)
    prior_identity = engine_module._artifact_path_observation(history_path)[1]
    engine, factor, _calls, _before = _failure_training_engine(
        monkeypatch, tmp_path, lambda _formula, _features: factor.clone()
    )
    engine.target_symbol = "EURUSD"
    engine.sampler = _MutableProtocolSampler()
    engine.best_score = float("inf")
    engine.best_formula = [0]
    monkeypatch.setattr(ModelConfig, "TRAIN_STEPS", 1)
    entry = engine.run_identity
    real_atomic = engine_module._atomic_json_replace
    real_windows_replace = engine_module._windows_replace_file
    rollback_cleanup = _HostileRollbackCleanup("hostile rollback cleanup")
    replace_calls = 0

    def mutating_atomic(path, payload, *args, **kwargs):
        result = real_atomic(path, payload, *args, **kwargs)
        engine.run_identity = TrainingRunIdentity.from_dict(entry.to_dict())
        return result

    def fail_after_exact_rollback(*args, **kwargs):
        nonlocal replace_calls
        replace_calls += 1
        result = real_windows_replace(*args, **kwargs)
        if replace_calls == 2:
            raise rollback_cleanup
        return result

    monkeypatch.setattr(engine_module, "_atomic_json_replace", mutating_atomic)
    monkeypatch.setattr(engine_module, "_windows_replace_file", fail_after_exact_rollback)

    with pytest.raises(ArtifactCompatibilityError) as caught:
        engine.train(end_step=1, verbose_header=False)

    assert engine.run_identity is entry
    assert replace_calls == 2
    assert any(
        "_HostileRollbackCleanup" in note
        for note in getattr(caught.value, "__notes__", ())
    )
    assert history_path.read_bytes() == prior
    assert engine_module._artifact_path_observation(history_path)[1] == prior_identity
    assert not list(tmp_path.glob(".training_history_EURUSD.json.*"))


@pytest.mark.parametrize("target_exists", [False, True])
def test_windows_publication_preserves_external_replace_after_candidate_kernel_call(
    monkeypatch, tmp_path, target_exists
) -> None:
    assert engine_module.os.name == "nt"
    monkeypatch.chdir(tmp_path)
    path = engine_module.pathlib.Path("training_history_EURUSD.json")
    temporary = path.with_name(f".{path.name}.tmp")
    external_temp = path.with_name(f".{path.name}.external-after-candidate.tmp")
    if target_exists:
        path.write_bytes(b"ORIGINAL")
    engine = _direct_transaction_engine()
    engine.target_symbol = "EURUSD"
    engine.training_history = {"step": [1]}
    transaction = engine_module._BatchTransaction(engine, [path, temporary])
    external = (
        b"EXTERNAL-AFTER-CANDIDATE-PRESENT"
        if target_exists
        else b"EXTERNAL-AFTER-CANDIDATE-PLACEHOLDER"
    )
    external_observable = False
    real_windows_replace = engine_module._windows_replace_file

    def candidate_then_external(target, replacement, *args, **kwargs):
        nonlocal external_observable
        result = real_windows_replace(target, replacement, *args, **kwargs)
        if target == path and replacement == temporary:
            external_temp.write_bytes(external)
            real_windows_replace(target, external_temp)
            assert path.read_bytes() == external
            external_observable = True
        return result

    monkeypatch.setattr(
        engine_module, "_windows_replace_file", candidate_then_external
    )
    transaction.run_artifact(
        engine._save_training_history_live, [path, temporary]
    )

    assert external_observable
    assert path.read_bytes() == external
    assert path not in transaction.owned_artifacts
    failure = _TrainingAbort("rollback after later external publication")
    with pytest.raises(_TrainingAbort) as caught:
        transaction.run(lambda: (_ for _ in ()).throw(failure))
    assert caught.value is failure
    assert path.read_bytes() == external
    assert any("rollback conflict" in note for note in failure.__notes__)
    assert not temporary.exists()
    assert not external_temp.exists()


@pytest.mark.parametrize("target_exists", [False, True])
def test_windows_publication_receipt_collision_preserves_foreign_sibling(
    monkeypatch, tmp_path, target_exists
) -> None:
    assert engine_module.os.name == "nt"
    monkeypatch.chdir(tmp_path)
    path = engine_module.pathlib.Path("training_history_EURUSD.json")
    temporary = path.with_name(f".{path.name}.tmp")
    receipt = temporary.with_name(f"{temporary.name}.receipt")
    prior = b"ORIGINAL"
    if target_exists:
        path.write_bytes(prior)
    receipt.write_bytes(b"FOREIGN-RECEIPT-SIBLING")
    engine = _direct_transaction_engine()
    engine.target_symbol = "EURUSD"
    engine.training_history = {"step": [1]}
    transaction = engine_module._BatchTransaction(engine, [path, temporary])

    with pytest.raises(RuntimeError, match="publication receipt conflict"):
        transaction.run_artifact(
            engine._save_training_history_live, [path, temporary]
        )

    assert (path.read_bytes() if path.exists() else None) == (
        prior if target_exists else None
    )
    assert receipt.read_bytes() == b"FOREIGN-RECEIPT-SIBLING"
    assert not temporary.exists()


@pytest.mark.parametrize("target_exists", [False, True])
def test_windows_same_bytes_new_identity_is_never_owned_or_rolled_back(
    monkeypatch, tmp_path, target_exists
) -> None:
    assert engine_module.os.name == "nt"
    monkeypatch.chdir(tmp_path)
    path = engine_module.pathlib.Path("training_history_EURUSD.json")
    temporary = path.with_name(f".{path.name}.tmp")
    external_temp = path.with_name(f".{path.name}.same-bytes-external.tmp")
    prior = b"ORIGINAL" if target_exists else None
    if prior is not None:
        path.write_bytes(prior)
    engine = _direct_transaction_engine()
    engine.target_symbol = "EURUSD"
    engine.training_history = {"step": [1], "avg_reward": [0.25]}
    transaction = engine_module._BatchTransaction(engine, [path, temporary])
    real_windows_replace = engine_module._windows_replace_file
    identities = {}

    def candidate_then_same_bytes_external(target, replacement, *args, **kwargs):
        result = real_windows_replace(target, replacement, *args, **kwargs)
        if target == path and replacement == temporary:
            candidate_kernel32, candidate_handle = result[2:]
            identities["candidate"] = engine_module._windows_file_identity(
                candidate_kernel32, candidate_handle
            )
            candidate_version = engine_module._windows_handle_version(
                candidate_kernel32, candidate_handle
            )
            candidate_payload = json.dumps(engine.training_history).encode("utf-8")
            assert candidate_version == (
                True,
                len(candidate_payload),
                engine_module.hashlib.sha256(candidate_payload).hexdigest(),
            )
            external_temp.write_bytes(candidate_payload)
            real_windows_replace(target, external_temp)
            current_kernel32, current_handle = engine_module._windows_open_exclusive(
                path,
                desired_access=0x80000000,
                share_mode=0x00000001 | 0x00000002 | 0x00000004,
            )
            try:
                identities["external"] = engine_module._windows_file_identity(
                    current_kernel32, current_handle
                )
            finally:
                assert current_kernel32.CloseHandle(current_handle)
            assert engine_module._artifact_path_version(path) == candidate_version
        return result

    monkeypatch.setattr(
        engine_module, "_windows_replace_file", candidate_then_same_bytes_external
    )
    transaction.run_artifact(
        engine._save_training_history_live, [path, temporary]
    )
    external = path.read_bytes()

    assert identities["candidate"] != identities["external"]
    assert path not in transaction.owned_artifacts
    failure = _TrainingAbort("rollback must preserve same-byte external identity")
    with pytest.raises(_TrainingAbort) as caught:
        transaction.run(lambda: (_ for _ in ()).throw(failure))
    assert caught.value is failure
    assert path.read_bytes() == external
    assert any("rollback conflict" in note for note in failure.__notes__)
    assert not temporary.exists()
    assert not external_temp.exists()


@pytest.mark.parametrize("target_exists", [False, True])
def test_windows_receipt_created_at_helper_boundary_is_never_overwritten(
    monkeypatch, tmp_path, target_exists
) -> None:
    assert engine_module.os.name == "nt"
    monkeypatch.chdir(tmp_path)
    path = engine_module.pathlib.Path("training_history_EURUSD.json")
    temporary = path.with_name(f".{path.name}.tmp")
    prior = b"ORIGINAL" if target_exists else None
    if prior is not None:
        path.write_bytes(prior)
    engine = _direct_transaction_engine()
    engine.target_symbol = "EURUSD"
    engine.training_history = {"step": [1]}
    transaction = engine_module._BatchTransaction(engine, [path, temporary])
    real_windows_replace = engine_module._windows_replace_file
    foreign = b"FOREIGN-RECEIPT-CREATED-AT-HELPER-BOUNDARY"
    raced_receipts = []

    def create_receipt_then_publish(target, replacement, backup=None):
        if target == path and replacement == temporary and backup is not None:
            engine_module.pathlib.Path(backup).write_bytes(foreign)
            raced_receipts.append(engine_module.pathlib.Path(backup))
        return real_windows_replace(target, replacement, backup)

    monkeypatch.setattr(
        engine_module, "_windows_replace_file", create_receipt_then_publish
    )
    with pytest.raises(RuntimeError, match="publication receipt conflict"):
        transaction.run_artifact(
            engine._save_training_history_live, [path, temporary]
        )

    assert len(raced_receipts) == 1
    assert raced_receipts[0].read_bytes() == foreign
    assert (path.read_bytes() if path.exists() else None) == prior
    assert not temporary.exists()


@pytest.mark.parametrize("target_exists", [False, True])
def test_windows_post_publication_first_close_failure_closes_all_and_rolls_back(
    monkeypatch, tmp_path, target_exists
) -> None:
    assert engine_module.os.name == "nt"
    monkeypatch.chdir(tmp_path)
    path = engine_module.pathlib.Path("training_history_EURUSD.json")
    temporary = path.with_name(f".{path.name}.tmp")
    receipt = temporary.with_name(f"{temporary.name}.receipt")
    prior = b"ORIGINAL" if target_exists else None
    if prior is not None:
        path.write_bytes(prior)
    engine = _direct_transaction_engine()
    engine.target_symbol = "EURUSD"
    engine.training_history = {"step": [1], "avg_reward": [0.25]}
    transaction = engine_module._BatchTransaction(engine, [path, temporary])
    real_windows_replace = engine_module._windows_replace_file
    close_calls = []

    class CloseProxy:
        def __init__(self, wrapped):
            self.wrapped = wrapped

        def __getattr__(self, name):
            return getattr(self.wrapped, name)

        def CloseHandle(self, handle):
            close_calls.append(handle)
            closed = self.wrapped.CloseHandle(handle)
            if len(close_calls) == 1:
                assert closed
                ctypes.set_last_error(5)
                return 0
            return closed

    def replace_then_fail_first_close(target, replacement, *args, **kwargs):
        result = real_windows_replace(target, replacement, *args, **kwargs)
        if target == path and replacement == temporary:
            return (
                CloseProxy(result[0]),
                result[1],
                CloseProxy(result[2]),
                result[3],
            )
        return result

    monkeypatch.setattr(
        engine_module, "_windows_replace_file", replace_then_fail_first_close
    )
    with pytest.raises(OSError) as caught:
        transaction.run_artifact(
            engine._save_training_history_live, [path, temporary]
        )

    assert "first post-publication" not in str(caught.value)
    assert len(close_calls) == 2
    assert not transaction.active
    assert (path.read_bytes() if path.exists() else None) == prior
    assert not temporary.exists()
    assert not receipt.exists()


@pytest.mark.parametrize("target_exists", [False, True])
def test_windows_post_publication_receipt_open_failure_restores_prior_or_absence(
    monkeypatch, tmp_path, target_exists
) -> None:
    assert engine_module.os.name == "nt"
    monkeypatch.chdir(tmp_path)
    path = engine_module.pathlib.Path("training_history_EURUSD.json")
    temporary = path.with_name(f".{path.name}.tmp")
    receipt = temporary.with_name(f"{temporary.name}.receipt")
    displaced = receipt / "displaced"
    foreign_sibling = path.with_name(f".{path.name}.foreign.keep")
    foreign_temporary = path.with_name(f".{path.name}.foreign.tmp")
    prior = b"ORIGINAL" if target_exists else None
    if prior is not None:
        path.write_bytes(prior)
    foreign_sibling.write_bytes(b"UNRELATED-SIBLING")
    foreign_temporary.write_bytes(b"UNRELATED-FOREIGN-TEMPORARY")
    engine = _direct_transaction_engine()
    engine.target_symbol = "EURUSD"
    engine.training_history = {"step": [1], "avg_reward": [0.25]}
    transaction = engine_module._BatchTransaction(engine, [path, temporary])
    failure = _TrainingAbort("post-publication receipt entry open failed")
    real_open_exclusive = engine_module._windows_open_exclusive
    observed_candidate = []

    def fail_first_displaced_open(open_path, *args, **kwargs):
        if engine_module.pathlib.Path(open_path) == displaced:
            assert displaced.exists()
            candidate = path.read_bytes()
            assert candidate != prior
            observed_candidate.append(candidate)
            raise failure
        return real_open_exclusive(open_path, *args, **kwargs)

    monkeypatch.setattr(
        engine_module, "_windows_open_exclusive", fail_first_displaced_open
    )
    with pytest.raises(_TrainingAbort) as caught:
        transaction.run_artifact(
            engine._save_training_history_live, [path, temporary]
        )

    assert caught.value is failure
    assert caught.value.__cause__ is None
    assert caught.value.__traceback__ is not None
    assert len(observed_candidate) == 1
    assert not transaction.active
    assert (path.read_bytes() if path.exists() else None) == prior
    assert not temporary.exists()
    assert not displaced.exists()
    assert not receipt.exists()
    assert foreign_sibling.read_bytes() == b"UNRELATED-SIBLING"
    assert foreign_temporary.read_bytes() == b"UNRELATED-FOREIGN-TEMPORARY"


@pytest.mark.parametrize("target_exists", [False, True])
def test_publication_guard_close_failure_restores_prior_or_absence(
    monkeypatch, tmp_path, target_exists
) -> None:
    monkeypatch.chdir(tmp_path)
    path = engine_module.pathlib.Path("training_history_EURUSD.json")
    temporary = path.with_name(f".{path.name}.tmp")
    prior = b"PRIOR-BEFORE-GUARDED-ATOMIC-REPLACE"
    if target_exists:
        path.write_bytes(prior)
    engine = _direct_transaction_engine()
    engine.target_symbol = "EURUSD"
    engine.training_history = {"step": [1], "avg_reward": [0.25]}
    transaction = engine_module._BatchTransaction(engine, [path, temporary])

    if engine_module.os.name == "nt":
        real_open_exclusive = engine_module._windows_open_exclusive
        close_calls = 0

        class Kernel32Proxy:
            def __init__(self, wrapped):
                self.wrapped = wrapped

            def __getattr__(self, name):
                return getattr(self.wrapped, name)

            def CloseHandle(self, handle):
                nonlocal close_calls
                close_calls += 1
                closed = self.wrapped.CloseHandle(handle)
                if close_calls == 1:
                    assert closed
                    return 0
                return closed

        def close_failing_open(*args, **kwargs):
            kernel32, handle = real_open_exclusive(*args, **kwargs)
            return Kernel32Proxy(kernel32), handle

        monkeypatch.setattr(
            engine_module, "_windows_open_exclusive", close_failing_open
        )
        expected_type = OSError
        expected_failure = None
    else:
        real_replace = engine_module.pathlib.Path.replace
        expected_failure = _TrainingAbort("post-replace publication close failed")
        expected_type = _TrainingAbort

        def replace_then_fail(source, destination):
            result = real_replace(source, destination)
            expected_failure._artifact_publication_claims = {
                path: (True, path.read_bytes()),
                temporary: (False, None),
            }
            raise expected_failure

        monkeypatch.setattr(
            engine_module.pathlib.Path, "replace", replace_then_fail
        )

    with pytest.raises(expected_type) as caught:
        transaction.run_artifact(
            engine._save_training_history_live, [path, temporary]
        )

    if expected_failure is not None:
        assert caught.value is expected_failure
    assert caught.value.__traceback__ is not None
    assert (path.read_bytes() if path.exists() else None) == (
        prior if target_exists else None
    )
    assert not temporary.exists()


def test_save_checkpoint_direct_result_preserves_plain_string_contract(
    monkeypatch, tmp_path
) -> None:
    engine = _direct_checkpoint_engine()
    path = str(tmp_path / "public-checkpoint.pt")

    result = engine.save_checkpoint(7, path)

    assert type(result) is str
    assert result == path
    assert engine_module.os.fspath(result) == path
    assert engine_module.pathlib.Path(result).is_file()
    with open(result, "rb") as fp:
        assert fp.read(2) == b"PK"
    payload = torch.load(result, map_location="cpu", weights_only=False)
    assert payload["step"] == 7


@pytest.mark.parametrize("error_type", [KeyboardInterrupt, _TrainingAbort])
def test_post_restart_checkpoint_baseexception_rolls_back_complete_batch(
    monkeypatch, tmp_path, error_type
) -> None:
    engine, calls, before, artifacts = _late_failure_transaction_engine(
        monkeypatch, tmp_path
    )
    original_optimizer = engine.opt
    monkeypatch.setattr(ModelConfig, "ENTROPY_COLLAPSE_THRESH", 1.0)
    monkeypatch.setattr(ModelConfig, "ENTROPY_COLLAPSE_STEPS", 1)
    monkeypatch.setattr(ModelConfig, "MAX_RESTARTS", 4)
    monkeypatch.setattr(ModelConfig, "FULL_RESET_EVERY", 1)
    failure = error_type("post-restart checkpoint publication failed")
    cause = ValueError("checkpoint cause")
    context = LookupError("checkpoint context")
    failure.__cause__ = cause
    failure.__context__ = context
    _inject_late_training_failure(engine, calls, artifacts, "checkpoint", failure)

    with pytest.raises(error_type) as caught:
        engine.train(end_step=1, verbose_header=False)

    assert caught.value is failure
    assert caught.value.__traceback__ is not None
    assert caught.value.__cause__ is cause
    assert caught.value.__context__ is context
    assert engine.opt is original_optimizer
    _assert_late_training_failure_rolled_back(engine, before, artifacts)


def test_successful_restart_checkpoint_matches_post_restart_supported_state(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.chdir(tmp_path)
    checkpoint_dir = tmp_path / "checkpoints"
    monkeypatch.setattr(engine_module, "_CHECKPOINT_DIR", checkpoint_dir)
    engine, factor, _calls, _before = _failure_training_engine(
        monkeypatch, tmp_path, lambda _formula, _features: factor.clone()
    )
    engine.target_symbol = "EURUSD"
    engine.best_score = float("inf")
    engine.save_checkpoint = AlphaEngine.save_checkpoint.__get__(engine, AlphaEngine)
    monkeypatch.setattr(ModelConfig, "ENTROPY_COLLAPSE_THRESH", 1.0)
    monkeypatch.setattr(ModelConfig, "ENTROPY_COLLAPSE_STEPS", 1)
    monkeypatch.setattr(ModelConfig, "MAX_RESTARTS", 2)
    monkeypatch.setattr(ModelConfig, "FULL_RESET_EVERY", 1)

    engine.train(end_step=1, verbose_header=False)

    checkpoint_path = checkpoint_dir / "ckpt_EURUSD_step_0001.pt"
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert payload["restart_count"] == engine._restart_count == 1
    _assert_transaction_value_equal(
        payload["model_state_dict"], engine.model.state_dict()
    )
    _assert_transaction_value_equal(
        payload["optimizer_state_dict"], engine.opt.state_dict()
    )

    restored, factor, _calls, _before = _failure_training_engine(
        monkeypatch, tmp_path, lambda _formula, _features: factor.clone()
    )
    restored.target_symbol = "EURUSD"
    restored.opt = torch.optim.AdamW(restored.model.parameters(), lr=1e-3)
    assert restored.load_checkpoint(str(checkpoint_path)) == 1
    _assert_transaction_value_equal(restored.model.state_dict(), engine.model.state_dict())
    _assert_transaction_value_equal(restored.opt.state_dict(), engine.opt.state_dict())
    assert restored._restart_count == engine._restart_count
    assert restored.best_score == engine.best_score
    assert restored.best_formula == engine.best_formula

    _seed_all(1409)
    engine.train(start_step=1, end_step=2, verbose_header=False)
    _seed_all(1409)
    restored.train(start_step=1, end_step=2, verbose_header=False)
    _assert_transaction_value_equal(
        restored.model.state_dict(), engine.model.state_dict()
    )
    _assert_transaction_value_equal(
        restored.opt.state_dict(), engine.opt.state_dict()
    )
    _assert_transaction_value_equal(restored.factor_pool, engine.factor_pool)
    _assert_transaction_value_equal(restored._elite_pool, engine._elite_pool)
    _assert_transaction_value_equal(
        restored.training_history, engine.training_history
    )
    assert restored._restart_count == engine._restart_count


_EXPECTED_ARTIFACT_IO_CHUNK_SIZE = 1024 * 1024


def _write_repeated_file(path, size: int, *, byte: int) -> None:
    chunk = bytes([byte]) * min(64 * 1024, size)
    remaining = size
    with open(path, "wb") as fp:
        while remaining:
            payload = chunk[: min(len(chunk), remaining)]
            fp.write(payload)
            remaining -= len(payload)


def _stream_file_identity(path, *, opener=open) -> tuple[int, str]:
    digest = engine_module.hashlib.sha256()
    size = 0
    with opener(path, "rb") as fp:
        while True:
            chunk = fp.read(64 * 1024)
            if not chunk:
                break
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def _install_bounded_artifact_read_probe(monkeypatch):
    real_open = open
    real_read_bytes = engine_module.pathlib.Path.read_bytes
    requested_sizes = []
    returned_sizes = []

    class BoundedReadProxy:
        def __init__(self, wrapped):
            self.wrapped = wrapped

        def __enter__(self):
            self.wrapped.__enter__()
            return self

        def __exit__(self, exc_type, exc_value, exc_traceback):
            return self.wrapped.__exit__(exc_type, exc_value, exc_traceback)

        def __getattr__(self, name):
            return getattr(self.wrapped, name)

        def read(self, size=-1):
            assert size is not None and size >= 0, "unbounded artifact read detected"
            requested_sizes.append(size)
            payload = self.wrapped.read(size)
            if isinstance(payload, (bytes, bytearray)):
                returned_sizes.append(len(payload))
            return payload

    def controlled_open(path, mode="r", *args, **kwargs):
        opened = real_open(path, mode, *args, **kwargs)
        if "r" in mode and "b" in mode:
            return BoundedReadProxy(opened)
        return opened

    def forbidden_read_bytes(path):
        raise AssertionError(f"Path.read_bytes whole-file materialization: {path}")

    monkeypatch.setattr(engine_module, "open", controlled_open, raising=False)
    monkeypatch.setattr(
        engine_module.pathlib.Path, "read_bytes", forbidden_read_bytes
    )
    return SimpleNamespace(
        real_open=real_open,
        real_read_bytes=real_read_bytes,
        requested_sizes=requested_sizes,
        returned_sizes=returned_sizes,
    )


def _largest_retained_bytes(value, seen=None) -> int:
    if seen is None:
        seen = set()
    identity = id(value)
    if identity in seen:
        return 0
    seen.add(identity)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return len(value)
    if isinstance(value, dict):
        return max(
            [_largest_retained_bytes(item, seen) for pair in value.items() for item in pair]
            or [0]
        )
    if isinstance(value, (list, tuple, set)):
        return max([_largest_retained_bytes(item, seen) for item in value] or [0])
    if hasattr(value, "__dict__") and value.__class__.__module__ == engine_module.__name__:
        return _largest_retained_bytes(vars(value), seen)
    return 0


@pytest.mark.parametrize("target_exists", [False, True])
def test_windows_post_publication_target_open_failure_restores_and_closes_all(
    monkeypatch, tmp_path, target_exists
) -> None:
    assert engine_module.os.name == "nt"
    monkeypatch.chdir(tmp_path)
    path = engine_module.pathlib.Path("training_history_EURUSD.json")
    temporary = path.with_name(f".{path.name}.tmp")
    receipt = temporary.with_name(f"{temporary.name}.receipt")
    displaced = receipt / "displaced"
    prior = b"EXACT-PRIOR-BYTES\x00\xff" if target_exists else None
    if prior is not None:
        path.write_bytes(prior)
    foreign_sibling = path.with_name(f".{path.name}.foreign.keep")
    foreign_temporary = path.with_name(f".{path.name}.foreign.tmp")
    foreign_sibling.write_bytes(b"UNRELATED-SIBLING")
    foreign_temporary.write_bytes(b"UNRELATED-FOREIGN-TEMPORARY")
    engine = _direct_transaction_engine()
    engine.target_symbol = "EURUSD"
    engine.training_history = {"step": [1], "avg_reward": [0.25]}
    transaction = engine_module._BatchTransaction(engine, [path, temporary])
    failure = _TrainingAbort("first post-publication destination open failed")
    cause = ValueError("target-open cause")
    context = LookupError("target-open context")
    failure.__cause__ = cause
    failure.__context__ = context
    real_open_exclusive = engine_module._windows_open_exclusive
    opened_tokens = []
    close_attempts = []
    displaced_open_succeeded = False
    injected = False
    observed_candidate = []

    class CloseTrackingKernel32:
        def __init__(self, wrapped, token):
            self.wrapped = wrapped
            self.token = token

        def __getattr__(self, name):
            return getattr(self.wrapped, name)

        def CloseHandle(self, handle):
            close_attempts.append(self.token)
            return self.wrapped.CloseHandle(handle)

    def fail_first_post_publication_target_open(open_path, *args, **kwargs):
        nonlocal displaced_open_succeeded, injected
        candidate_path = engine_module.pathlib.Path(open_path)
        if candidate_path == path and displaced_open_succeeded and not injected:
            injected = True
            observed_candidate.append(path.read_bytes())
            raise failure
        kernel32, handle = real_open_exclusive(open_path, *args, **kwargs)
        token = len(opened_tokens)
        opened_tokens.append(token)
        if candidate_path == displaced:
            displaced_open_succeeded = True
        return CloseTrackingKernel32(kernel32, token), handle

    monkeypatch.setattr(
        engine_module, "_windows_open_exclusive", fail_first_post_publication_target_open
    )
    with pytest.raises(_TrainingAbort) as caught:
        transaction.run_artifact(
            engine._save_training_history_live, [path, temporary]
        )

    assert caught.value is failure
    assert caught.value.__traceback__ is not None
    assert caught.value.__cause__ is cause
    assert caught.value.__context__ is context
    assert displaced_open_succeeded and injected
    assert len(observed_candidate) == 1
    assert observed_candidate[0] != prior
    assert not transaction.active
    assert (path.read_bytes() if path.exists() else None) == prior
    assert not temporary.exists()
    assert not displaced.exists()
    assert not receipt.exists()
    assert foreign_sibling.read_bytes() == b"UNRELATED-SIBLING"
    assert foreign_temporary.read_bytes() == b"UNRELATED-FOREIGN-TEMPORARY"
    assert opened_tokens
    assert all(close_attempts.count(token) == 1 for token in opened_tokens)


@pytest.mark.parametrize("size_mib", [1, 8, 32])
def test_transaction_checkpoint_io_is_chunk_bounded_and_rolls_back_exactly(
    monkeypatch, tmp_path, size_mib
) -> None:
    target = tmp_path / f"checkpoint-{size_mib}mib.pt"
    temporary = target.with_name(f".{target.name}.tmp")
    size = size_mib * 1024 * 1024
    _write_repeated_file(target, size, byte=0x31 + size_mib)
    expected = _stream_file_identity(target)
    probe = _install_bounded_artifact_read_probe(monkeypatch)
    engine = _direct_checkpoint_engine()
    engine._record_owned_artifact(target)
    transaction = engine_module._BatchTransaction(engine, [target, temporary])

    transaction.run_artifact(engine.save_checkpoint, [target, temporary], 7, str(target))
    retained_payload = _largest_retained_bytes(transaction)
    failure = _TrainingAbort(f"rollback {size_mib} MiB checkpoint")
    with pytest.raises(_TrainingAbort) as caught:
        transaction.run(lambda: (_ for _ in ()).throw(failure))

    assert caught.value is failure
    assert _stream_file_identity(target, opener=probe.real_open) == expected
    assert not temporary.exists()
    assert not list(tmp_path.glob(f".{target.name}.transaction-backup.*"))
    assert probe.requested_sizes
    assert max(probe.requested_sizes) == _EXPECTED_ARTIFACT_IO_CHUNK_SIZE
    assert max(probe.returned_sizes) <= _EXPECTED_ARTIFACT_IO_CHUNK_SIZE
    assert retained_payload <= 64
    request_histogram = {
        requested: probe.requested_sizes.count(requested)
        for requested in sorted(set(probe.requested_sizes))
    }
    print(
        "BOUNDED_ARTIFACT_IO "
        f"size_mib={size_mib} attempts={len(probe.requested_sizes)} "
        f"request_histogram={request_histogram} "
        f"max_requested={max(probe.requested_sizes)} "
        f"max_returned={max(probe.returned_sizes)} "
        f"max_retained_bytes={retained_payload}"
    )


@pytest.mark.parametrize("target_exists", [False, True])
@pytest.mark.parametrize("outcome", ["commit", "baseexception"])
def test_transaction_owned_backups_cleanup_on_success_and_baseexception(
    monkeypatch, tmp_path, target_exists, outcome
) -> None:
    target = tmp_path / f"owned-{target_exists}-{outcome}.pt"
    temporary = target.with_name(f".{target.name}.tmp")
    original = None
    if target_exists:
        _write_repeated_file(
            target, _EXPECTED_ARTIFACT_IO_CHUNK_SIZE * 2 + 17, byte=0x5A
        )
        original = _stream_file_identity(target)
    probe = _install_bounded_artifact_read_probe(monkeypatch)
    engine = _direct_checkpoint_engine()
    if target_exists:
        engine._record_owned_artifact(target)
    transaction = engine_module._BatchTransaction(engine, [target, temporary])
    transaction.run_artifact(engine.save_checkpoint, [target, temporary], 9, str(target))

    if outcome == "commit":
        transaction.commit()
        assert target.exists()
        assert _stream_file_identity(target, opener=probe.real_open) != original
    else:
        failure = KeyboardInterrupt("bounded rollback")
        with pytest.raises(KeyboardInterrupt) as caught:
            transaction.run(lambda: (_ for _ in ()).throw(failure))
        assert caught.value is failure
        if original is None:
            assert not target.exists()
        else:
            assert _stream_file_identity(target, opener=probe.real_open) == original

    assert not transaction.active
    assert not temporary.exists()
    assert not list(tmp_path.glob(f".{target.name}.transaction-backup.*"))
    assert not list(tmp_path.glob(f"{temporary.name}.receipt"))
    assert probe.requested_sizes
    assert max(probe.requested_sizes) <= _EXPECTED_ARTIFACT_IO_CHUNK_SIZE
    assert max(probe.returned_sizes) <= _EXPECTED_ARTIFACT_IO_CHUNK_SIZE


@pytest.mark.parametrize("mutant", ["path_read_bytes", "unbounded_read"])
def test_bounded_artifact_io_probe_rejects_whole_file_mutants(
    monkeypatch, tmp_path, mutant
) -> None:
    path = tmp_path / "mutant.bin"
    path.write_bytes(b"MUTANT-PAYLOAD")
    probe = _install_bounded_artifact_read_probe(monkeypatch)

    with pytest.raises(AssertionError, match="whole-file|unbounded"):
        if mutant == "path_read_bytes":
            path.read_bytes()
        else:
            with engine_module.open(path, "rb") as fp:
                fp.read()

    assert probe.requested_sizes == []


def test_live_strategy_refuses_preexisting_unowned_legacy_artifact(
    monkeypatch, tmp_path
) -> None:
    assert engine_module.os.name == "nt"
    strategy_path = tmp_path / "best_EURUSD.json"
    legacy = b"LEGACY-STRATEGY-WITHOUT-V2-IDENTITY\x00\xff"
    strategy_path.write_bytes(legacy)
    legacy_identity = engine_module._windows_path_identity(strategy_path)
    engine = _commit_test_engine()
    engine.target_symbol = "EURUSD"
    before = (engine.best_score, list(engine.best_formula))
    monkeypatch.setattr(
        engine_module,
        "_test_strategy_file_for_symbol",
        lambda _symbol: str(strategy_path),
    )

    with pytest.raises(ArtifactCompatibilityError, match="run_identity mismatch"):
        engine._save_strategy_live()

    assert (engine.best_score, engine.best_formula) == before
    assert strategy_path.read_bytes() == legacy
    assert engine_module._windows_path_identity(strategy_path) == legacy_identity
    assert not list(tmp_path.glob(f".{strategy_path.name}*.tmp"))
    assert not list(tmp_path.glob(f".{strategy_path.name}.transaction-backup.*"))


def test_final_strategy_refuses_preexisting_unowned_legacy_artifact(
    monkeypatch, tmp_path
) -> None:
    assert engine_module.os.name == "nt"
    monkeypatch.chdir(tmp_path)
    strategy_path = tmp_path / "best_EURUSD.json"
    legacy = b"LEGACY-FINAL-STRATEGY-WITHOUT-V2-IDENTITY"
    strategy_path.write_bytes(legacy)
    legacy_identity = engine_module._windows_path_identity(strategy_path)
    engine, factor, _calls, _before = _failure_training_engine(
        monkeypatch, tmp_path, lambda _formula, _features: factor.clone()
    )
    engine.target_symbol = "EURUSD"
    engine.best_formula = [0]
    engine.best_score = float("inf")
    monkeypatch.setattr(ModelConfig, "TRAIN_STEPS", 1)
    monkeypatch.setattr(
        engine_module,
        "_test_strategy_file_for_symbol",
        lambda _symbol: str(strategy_path),
    )
    _install_legacy_final_strategy_oracle(engine, enforce_ownership=True)

    with pytest.raises(RuntimeError, match="artifact publication conflict"):
        engine.train(end_step=1, verbose_header=False)

    assert engine.best_formula == [0]
    assert engine.best_score == float("inf")
    assert strategy_path.read_bytes() == legacy
    assert engine_module._windows_path_identity(strategy_path) == legacy_identity
    assert not list(tmp_path.glob(f".{strategy_path.name}*.tmp"))
    assert not list(tmp_path.glob(f".{strategy_path.name}.transaction-backup.*"))


def test_save_checkpoint_refuses_preexisting_unowned_legacy_artifact(
    tmp_path,
) -> None:
    assert engine_module.os.name == "nt"
    checkpoint_path = tmp_path / "legacy-checkpoint.pt"
    legacy = b"LEGACY-CHECKPOINT-WITHOUT-V2-IDENTITY\x00\xff"
    checkpoint_path.write_bytes(legacy)
    legacy_identity = engine_module._windows_path_identity(checkpoint_path)
    engine = _direct_checkpoint_engine()
    before = (engine.best_score, list(engine.best_formula), engine._restart_count)

    with pytest.raises(RuntimeError, match="artifact publication conflict"):
        engine.save_checkpoint(7, str(checkpoint_path))

    assert (engine.best_score, engine.best_formula, engine._restart_count) == before
    assert checkpoint_path.read_bytes() == legacy
    assert engine_module._windows_path_identity(checkpoint_path) == legacy_identity
    assert not checkpoint_path.with_name(f".{checkpoint_path.name}.tmp").exists()
    assert not list(
        tmp_path.glob(f".{checkpoint_path.name}.transaction-backup.*")
    )


def test_live_strategy_allows_repeated_updates_owned_by_same_engine(
    monkeypatch, tmp_path
) -> None:
    assert engine_module.os.name == "nt"
    strategy_path = tmp_path / "best_EURUSD.json"
    engine = _commit_test_engine()
    engine.target_symbol = "EURUSD"
    monkeypatch.setattr(
        engine_module,
        "_test_strategy_file_for_symbol",
        lambda _symbol: str(strategy_path),
    )

    with pytest.raises(ArtifactCompatibilityError, match="run_identity mismatch"):
        engine._save_strategy_live()
    assert not strategy_path.exists()
    engine.best_formula = [100]
    engine.best_score = 3.5
    with pytest.raises(ArtifactCompatibilityError, match="run_identity mismatch"):
        engine._save_strategy_live()

    assert not strategy_path.exists()
    assert engine.best_formula == [100]
    assert engine.best_score == 3.5
    assert not list(tmp_path.glob(f".{strategy_path.name}*.tmp"))


def test_live_strategy_repeated_update_uses_engine_metadata_without_whole_file_read(
    monkeypatch, tmp_path
) -> None:
    strategy_path = tmp_path / "best_EURUSD.json"
    engine = _commit_test_engine()
    engine.target_symbol = "EURUSD"
    engine.timeframe = "M15"
    engine.data_file = "owned-training-data.parquet"
    engine.mode = "train"
    engine.train_steps = 37
    monkeypatch.setattr(
        engine_module,
        "_test_strategy_file_for_symbol",
        lambda _symbol: str(strategy_path),
    )
    with pytest.raises(ArtifactCompatibilityError, match="run_identity mismatch"):
        engine._save_strategy_live()
    assert not strategy_path.exists()

    real_read_text = engine_module.pathlib.Path.read_text
    real_open = open
    requested_sizes = []

    class BoundedBinaryRead:
        def __init__(self, wrapped):
            self.wrapped = wrapped

        def __enter__(self):
            self.wrapped.__enter__()
            return self

        def __exit__(self, exc_type, exc_value, exc_traceback):
            return self.wrapped.__exit__(exc_type, exc_value, exc_traceback)

        def __getattr__(self, name):
            return getattr(self.wrapped, name)

        def read(self, size=-1):
            assert size is not None and size >= 0, "unbounded strategy read"
            requested_sizes.append(size)
            return self.wrapped.read(size)

    def bounded_open(path, mode="r", *args, **kwargs):
        opened = real_open(path, mode, *args, **kwargs)
        if "r" in mode and "b" in mode:
            return BoundedBinaryRead(opened)
        return opened

    def forbidden_read_text(path, *args, **kwargs):
        raise AssertionError(f"Path.read_text whole-file materialization: {path}")

    monkeypatch.setattr(engine_module, "open", bounded_open, raising=False)
    monkeypatch.setattr(
        engine_module.pathlib.Path, "read_text", forbidden_read_text
    )
    engine.best_formula = [100]
    engine.best_score = 3.5
    with pytest.raises(ArtifactCompatibilityError, match="run_identity mismatch"):
        engine._save_strategy_live()

    assert not strategy_path.exists()
    assert engine.best_formula == [100]
    assert engine.best_score == 3.5
    assert engine.timeframe == "M15"
    assert engine.data_file == "owned-training-data.parquet"
    assert engine.mode == "train"
    assert engine.train_steps == 37
    assert requested_sizes == []
    assert not list(tmp_path.glob(f".{strategy_path.name}*.tmp"))


def test_save_checkpoint_allows_repeated_updates_owned_by_same_engine(
    tmp_path,
) -> None:
    assert engine_module.os.name == "nt"
    checkpoint_path = tmp_path / "owned-checkpoint.pt"
    engine = _direct_checkpoint_engine()

    assert engine.save_checkpoint(1, str(checkpoint_path)) == str(checkpoint_path)
    first_identity = engine_module._windows_path_identity(checkpoint_path)
    assert engine.save_checkpoint(2, str(checkpoint_path)) == str(checkpoint_path)

    assert engine_module._windows_path_identity(checkpoint_path) != first_identity
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert payload["step"] == 2
    assert not checkpoint_path.with_name(f".{checkpoint_path.name}.tmp").exists()


@pytest.mark.parametrize("target_exists", [False, True])
def test_windows_early_publication_cas_preserves_new_foreign_identity(
    monkeypatch, tmp_path, target_exists
) -> None:
    assert engine_module.os.name == "nt"
    monkeypatch.chdir(tmp_path)
    path = engine_module.pathlib.Path("training_history_EURUSD.json")
    temporary = path.with_name(f".{path.name}.tmp")
    receipt = temporary.with_name(f"{temporary.name}.receipt")
    external_temp = path.with_name(f".{path.name}.early-foreign.tmp")
    foreign_sibling = path.with_name(f".{path.name}.foreign.keep")
    original = b"SAME-BYTES-BASELINE"
    if target_exists:
        path.write_bytes(original)
    foreign_sibling.write_bytes(b"UNRELATED-FOREIGN-SIBLING")
    engine = _direct_transaction_engine()
    engine.target_symbol = "EURUSD"
    engine.training_history = {"step": [1], "avg_reward": [0.25]}
    transaction = engine_module._BatchTransaction(engine, [path, temporary])
    foreign = original if target_exists else b"FOREIGN-APPEARED-BEFORE-OPEN"
    observed = {}

    def replace_before_exclusive_open(target):
        assert target == path
        external_temp.write_bytes(foreign)
        if target_exists:
            engine_module._windows_replace_file(path, external_temp)
        else:
            engine_module.os.replace(external_temp, path)
        observed["identity"] = engine_module._windows_path_identity(path)

    transaction._publication_cas_interleave_hook = replace_before_exclusive_open
    with pytest.raises(RuntimeError, match="artifact publication conflict"):
        transaction.run_artifact(
            engine._save_training_history_live, [path, temporary]
        )

    assert not transaction.active
    assert path.read_bytes() == foreign
    assert engine_module._windows_path_identity(path) == observed["identity"]
    assert path not in transaction.owned_artifacts
    assert foreign_sibling.read_bytes() == b"UNRELATED-FOREIGN-SIBLING"
    assert not temporary.exists()
    assert not external_temp.exists()
    assert not receipt.exists()
    assert not list(tmp_path.glob(f".{path.name}.transaction-backup.*"))


def test_checkpoint_rollback_reconciles_same_engine_ownership_and_absence(
    tmp_path,
) -> None:
    assert engine_module.os.name == "nt"
    checkpoint_path = tmp_path / "owned-checkpoint.pt"
    checkpoint_temp = checkpoint_path.with_name(f".{checkpoint_path.name}.tmp")
    absent_path = tmp_path / "created-inside-transaction.pt"
    absent_temp = absent_path.with_name(f".{absent_path.name}.tmp")
    foreign_sibling = tmp_path / "unrelated.foreign.keep"
    foreign_sibling.write_bytes(b"UNRELATED-FOREIGN-SIBLING")
    engine = _direct_checkpoint_engine()

    assert engine.save_checkpoint(1, str(checkpoint_path)) == str(checkpoint_path)
    version_one_bytes = checkpoint_path.read_bytes()
    version_one = engine_module._artifact_path_observation(checkpoint_path)
    transaction = engine_module._BatchTransaction(
        engine,
        [checkpoint_path, checkpoint_temp, absent_path, absent_temp],
    )

    transaction.run_artifact(
        engine.save_checkpoint,
        [checkpoint_path, checkpoint_temp],
        2,
        str(checkpoint_path),
    )
    transaction.run_artifact(
        engine.save_checkpoint,
        [absent_path, absent_temp],
        2,
        str(absent_path),
    )
    version_two = engine_module._artifact_path_observation(checkpoint_path)
    failure = _TrainingAbort("rollback ownership reconciliation")
    cause = ValueError("ownership cause")
    context = LookupError("ownership context")
    failure.__cause__ = cause
    failure.__context__ = context

    with pytest.raises(_TrainingAbort) as caught:
        transaction.run(lambda: (_ for _ in ()).throw(failure))

    assert caught.value is failure
    assert caught.value.__traceback__ is not None
    assert caught.value.__cause__ is cause
    assert caught.value.__context__ is context
    assert not transaction.active
    assert checkpoint_path.read_bytes() == version_one_bytes
    restored = engine_module._artifact_path_observation(checkpoint_path)
    assert restored[0] == version_one[0]
    assert restored != version_two
    records = engine._artifact_ownership_records()
    checkpoint_key = engine._artifact_ownership_key(checkpoint_path)
    absent_key = engine._artifact_ownership_key(absent_path)
    assert records[checkpoint_key] == restored
    assert not absent_path.exists()
    assert absent_key not in records
    assert not checkpoint_temp.exists()
    assert not absent_temp.exists()
    assert not list(tmp_path.glob(".*.transaction-backup.*"))
    assert foreign_sibling.read_bytes() == b"UNRELATED-FOREIGN-SIBLING"

    assert engine.save_checkpoint(3, str(checkpoint_path)) == str(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert payload["step"] == 3


@pytest.mark.parametrize(
    "restore_boundary", ["windows_post_close", "nonwindows_post_replace"]
)
def test_checkpoint_rollback_never_claims_post_restore_foreign_identity(
    monkeypatch, tmp_path, restore_boundary
) -> None:
    assert engine_module.os.name == "nt"
    checkpoint_path = tmp_path / "owned-checkpoint.pt"
    checkpoint_temp = checkpoint_path.with_name(f".{checkpoint_path.name}.tmp")
    absent_path = tmp_path / "created-inside-transaction.pt"
    absent_temp = absent_path.with_name(f".{absent_path.name}.tmp")
    foreign_temp = tmp_path / "same-content-foreign.tmp"
    foreign_sibling = tmp_path / "unrelated.foreign.keep"
    foreign_sibling.write_bytes(b"UNRELATED-FOREIGN-SIBLING")
    boundary_calls = []
    if restore_boundary == "nonwindows_post_replace":
        real_replace = engine_module.pathlib.Path.replace
        real_os = engine_module.os

        class NonWindowsOS:
            name = "posix"

            def __getattr__(self, name):
                return getattr(real_os, name)

        def interleaved_replace(source, destination):
            result = real_replace(source, destination)
            if ".rollback." in source.name and destination == checkpoint_path:
                boundary_calls.append("real-replace-complete")
                real_replace(foreign_temp, checkpoint_path)
            return result

        monkeypatch.setattr(engine_module, "os", NonWindowsOS())
        monkeypatch.setattr(
            engine_module.pathlib.Path, "replace", interleaved_replace
        )
    engine = _direct_checkpoint_engine()

    assert engine.save_checkpoint(1, str(checkpoint_path)) == str(checkpoint_path)
    version_one_bytes = checkpoint_path.read_bytes()
    transaction = engine_module._BatchTransaction(
        engine,
        [checkpoint_path, checkpoint_temp, absent_path, absent_temp],
    )
    transaction.run_artifact(
        engine.save_checkpoint,
        [checkpoint_path, checkpoint_temp],
        2,
        str(checkpoint_path),
    )
    transaction.run_artifact(
        engine.save_checkpoint,
        [absent_path, absent_temp],
        2,
        str(absent_path),
    )
    foreign_temp.write_bytes(version_one_bytes)

    if restore_boundary == "windows_post_close":
        real_open_exclusive = engine_module._windows_open_exclusive

        class PostCloseForeignReplacement:
            def __init__(self, wrapped):
                self.wrapped = wrapped

            def __getattr__(self, name):
                return getattr(self.wrapped, name)

            def CloseHandle(self, handle):
                closed = self.wrapped.CloseHandle(handle)
                assert closed
                boundary_calls.append("real-close-complete")
                engine_module._windows_replace_file(checkpoint_path, foreign_temp)
                return closed

        def interleaved_open(path, *args, **kwargs):
            kernel32, handle = real_open_exclusive(path, *args, **kwargs)
            if engine_module.pathlib.Path(path) == checkpoint_path:
                return PostCloseForeignReplacement(kernel32), handle
            return kernel32, handle

        monkeypatch.setattr(
            engine_module, "_windows_open_exclusive", interleaved_open
        )
    failure = _TrainingAbort(f"post-restore identity race: {restore_boundary}")
    cause = ValueError("post-restore cause")
    context = LookupError("post-restore context")
    failure.__cause__ = cause
    failure.__context__ = context
    with pytest.raises(_TrainingAbort) as caught:
        transaction.run(lambda: (_ for _ in ()).throw(failure))

    assert caught.value is failure
    assert caught.value.__traceback__ is not None
    assert caught.value.__cause__ is cause
    assert caught.value.__context__ is context
    assert not transaction.active
    assert boundary_calls == [
        "real-close-complete"
        if restore_boundary == "windows_post_close"
        else "real-replace-complete"
    ]
    foreign_observation = engine_module._artifact_path_observation(checkpoint_path)
    foreign_identity = foreign_observation[1]
    assert checkpoint_path.read_bytes() == version_one_bytes
    assert foreign_identity is not None
    records = engine._artifact_ownership_records()
    checkpoint_key = engine._artifact_ownership_key(checkpoint_path)
    absent_key = engine._artifact_ownership_key(absent_path)
    recorded = records.get(checkpoint_key)

    save_failure = None
    try:
        engine.save_checkpoint(3, str(checkpoint_path))
    except RuntimeError as caught_save_failure:
        save_failure = caught_save_failure

    assert recorded != foreign_observation and save_failure is not None, (
        f"record_eq_foreign={recorded == foreign_observation} "
        f"subsequent_save_succeeded={save_failure is None}"
    )
    assert "artifact publication conflict" in str(save_failure)
    assert checkpoint_path.read_bytes() == version_one_bytes
    assert engine_module._artifact_path_observation(checkpoint_path)[1] == foreign_identity
    assert not absent_path.exists()
    assert absent_key not in records
    assert not checkpoint_temp.exists()
    assert not absent_temp.exists()
    assert not foreign_temp.exists()
    assert not list(tmp_path.glob(".*.transaction-backup.*"))
    assert not list(tmp_path.glob(".*.receipt"))
    assert foreign_sibling.read_bytes() == b"UNRELATED-FOREIGN-SIBLING"


@pytest.mark.parametrize("target_exists", [False, True])
def test_nonwindows_rollback_preserves_same_bytes_new_foreign_identity(
    monkeypatch, tmp_path, target_exists
) -> None:
    assert engine_module.os.name == "nt"
    monkeypatch.chdir(tmp_path)
    real_os = engine_module.os

    class NonWindowsOS:
        name = "posix"

        def __getattr__(self, name):
            return getattr(real_os, name)

    monkeypatch.setattr(engine_module, "os", NonWindowsOS())
    path = engine_module.pathlib.Path("training_history_EURUSD.json")
    temporary = path.with_name(f".{path.name}.tmp")
    foreign_temp = path.with_name(f".{path.name}.same-bytes-foreign.tmp")
    foreign_sibling = path.with_name(f".{path.name}.foreign.keep")
    original = b"ORIGINAL-HISTORY" if target_exists else None
    if original is not None:
        path.write_bytes(original)
    foreign_sibling.write_bytes(b"UNRELATED-FOREIGN-SIBLING")
    engine = _direct_transaction_engine()
    engine.target_symbol = "EURUSD"
    engine.training_history = {"step": [1], "avg_reward": [0.25]}
    transaction = engine_module._BatchTransaction(engine, [path, temporary])

    transaction.run_artifact(
        engine._save_training_history_live, [path, temporary]
    )
    claimed = transaction.owned_artifacts[path]
    assert transaction.publication_conflicts == []
    assert transaction.owned_artifacts[temporary] == (
        engine_module._ABSENT_ARTIFACT_VERSION,
        None,
    )
    published_bytes = path.read_bytes()
    published_observation = engine_module._artifact_path_observation(path)
    foreign_temp.write_bytes(published_bytes)
    foreign_before = engine_module._artifact_path_observation(foreign_temp)
    assert foreign_before[0] == published_observation[0]
    assert foreign_before[1] != published_observation[1]
    foreign_temp.replace(path)
    foreign_observation = engine_module._artifact_path_observation(path)
    assert foreign_observation == foreign_before

    failure = _TrainingAbort(f"non-Windows rollback conflict: {target_exists}")
    cause = ValueError("non-Windows rollback cause")
    context = LookupError("non-Windows rollback context")
    failure.__cause__ = cause
    failure.__context__ = context
    with pytest.raises(_TrainingAbort) as caught:
        transaction.run(lambda: (_ for _ in ()).throw(failure))

    assert caught.value is failure
    assert caught.value.__traceback__ is not None
    assert caught.value.__cause__ is cause
    assert caught.value.__context__ is context
    assert not transaction.active
    preserved = (
        path.exists()
        and path.read_bytes() == published_bytes
        and engine_module._artifact_path_observation(path) == foreign_observation
    )
    assert claimed[1] is not None and preserved, (
        f"claimed_identity={claimed[1]} foreign_preserved={preserved} "
        f"target_exists={target_exists}"
    )
    assert any(
        "batch rollback conflict" in note
        and "file-identity-changed" in note
        for note in getattr(caught.value, "__notes__", ())
    )
    assert not temporary.exists()
    assert not foreign_temp.exists()
    assert not list(tmp_path.glob(f".{path.name}.transaction-backup.*"))
    assert not list(tmp_path.glob(f".{temporary.name}.receipt"))
    assert foreign_sibling.read_bytes() == b"UNRELATED-FOREIGN-SIBLING"
