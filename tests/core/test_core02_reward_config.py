from __future__ import annotations

import math
import inspect

import pytest
import torch

from data_pipeline.validation import DataValidationError
from config import Config
from model_core.artifacts import sha256_json
from model_core.config import ModelConfig
from model_core.execution import factor_to_position
from model_core.reward import apply_ic_gate, apply_oos_gate, target_bars_per_trade
from model_core.semantics import CORE_SEMANTICS_VERSION
from training_service import _training_config


@pytest.mark.core
@pytest.mark.parametrize("base", [-3.0, 0.0, 4.0])
def test_oos_gate_is_sign_safe_and_monotone_for_every_base_sign(base: float) -> None:
    scores = [
        apply_oos_gate(base, sortino).final
        for sortino in (-3.0, -1.0, 0.0, 0.5, 3.0)
    ]

    assert scores == sorted(scores)
    assert scores[2] == pytest.approx(base)
    assert all(score <= base for score in scores[:3])
    assert all(score >= base for score in scores[2:])


@pytest.mark.core
@pytest.mark.parametrize("base", [-3.0, 0.0, 4.0])
def test_ic_gate_adjustment_direction_is_independent_of_reward_sign(base: float) -> None:
    positive = apply_ic_gate(torch.tensor(base), 0.5)
    neutral = apply_ic_gate(torch.tensor(base), 0.0)
    negative = apply_ic_gate(torch.tensor(base), -0.5)

    assert positive.adjustment.item() >= 0.0
    assert neutral.adjustment.item() == 0.0
    assert negative.adjustment.item() <= 0.0
    assert positive.final.item() >= neutral.final.item() >= negative.final.item()


@pytest.mark.core
@pytest.mark.parametrize("base", [-2.0, 2.0])
def test_ic_gate_preserves_ordering_gradient_for_each_reward_sign(base: float) -> None:
    positive_reward = torch.tensor(base, requires_grad=True)
    positive = apply_ic_gate(positive_reward, 0.5)
    positive.final.backward()
    positive_gradient = positive_reward.grad.item()

    negative_reward = torch.tensor(base, requires_grad=True)
    negative = apply_ic_gate(negative_reward, -0.5)
    negative.final.backward()

    assert positive_gradient > 0.0
    assert negative_reward.grad.item() > 0.0


@pytest.mark.core
@pytest.mark.parametrize("invalid", [math.nan, math.inf, -math.inf])
def test_reward_gates_reject_non_finite_inputs(invalid: float) -> None:
    with pytest.raises(DataValidationError, match="finite"):
        apply_oos_gate(1.0, invalid)
    with pytest.raises(DataValidationError, match="finite"):
        apply_ic_gate(torch.tensor(1.0), invalid)


@pytest.mark.core
def test_gate_audit_logs_show_base_adjustment_and_final(caplog) -> None:
    caplog.set_level("DEBUG", logger="model_core.reward")
    apply_oos_gate(-1.0, 0.5)
    apply_ic_gate(torch.tensor(-1.0), 0.5)

    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "OOS gate base=" in message
        and "adjustment=" in message
        and "final=" in message
        for message in messages
    )
    assert any(
        "IC gate base=" in message
        and "adjustment=" in message
        and "final=" in message
        for message in messages
    )


@pytest.mark.core
def test_daily_trade_target_is_timeframe_aware() -> None:
    assert target_bars_per_trade("M1", 2.0) == pytest.approx(720.0)
    assert target_bars_per_trade("H1", 2.0) == pytest.approx(12.0)
    assert target_bars_per_trade("H4", 2.0) == pytest.approx(3.0)
    assert target_bars_per_trade("D1", 2.0) == pytest.approx(0.5)


@pytest.mark.core
def test_training_snapshot_contains_every_gate_and_timeframe_input() -> None:
    h1 = _training_config(42, timeframe="H1")
    m1 = _training_config(42, timeframe="M1")

    assert h1 == ModelConfig.training_config_snapshot(42, timeframe="H1")
    assert h1["reward"]["oos_gate_scale"] == ModelConfig.OOS_GATE_SCALE
    assert h1["reward"]["ic_gate_scale_floor"] == ModelConfig.IC_GATE_SCALE_FLOOR
    assert h1["timeframe_reward"]["target_bars_per_trade"] == pytest.approx(12.0)
    assert m1["timeframe_reward"]["target_bars_per_trade"] == pytest.approx(720.0)
    assert h1 != m1


@pytest.mark.core
def test_gate_timeframe_and_exposure_changes_alter_training_hash() -> None:
    baseline = _training_config(42, timeframe="H1")
    changed_oos = _training_config(42, timeframe="H1")
    changed_oos["reward"]["oos_gate_scale"] = 0.75
    changed_floor = _training_config(42, timeframe="H1")
    changed_floor["reward"]["ic_gate_scale_floor"] = 0.1
    changed_exposure = _training_config(42, timeframe="H1")
    changed_exposure["neutral_band"] = 0.2
    changed_timeframe = _training_config(42, timeframe="H4")

    hashes = {
        sha256_json(value)
        for value in (
            baseline,
            changed_oos,
            changed_floor,
            changed_exposure,
            changed_timeframe,
        )
    }
    assert len(hashes) == 5


@pytest.mark.core
def test_training_parameters_have_one_authoritative_exporter() -> None:
    for stale_name in ("BATCH_SIZE", "TRAIN_STEPS", "MAX_FORMULA_LEN", "DEVICE"):
        assert not hasattr(Config, stale_name)
    assert "training_config_snapshot" in inspect.getsource(_training_config)


@pytest.mark.core
@pytest.mark.parametrize("invalid", [1.0, 1.1, math.nan, math.inf, -math.inf])
def test_min_exposure_rejects_one_non_finite_and_above(invalid: float) -> None:
    with pytest.raises(DataValidationError, match=r"0 <= min_exposure < 1"):
        factor_to_position(torch.tensor([[0.0]]), min_exposure=invalid)


@pytest.mark.core
@pytest.mark.parametrize("valid", [0.0, 0.05, 0.999999])
def test_min_exposure_accepts_documented_range(valid: float) -> None:
    factor_to_position(torch.tensor([[0.0]]), min_exposure=valid)


@pytest.mark.core
def test_core02_bumps_training_semantics_version() -> None:
    assert CORE_SEMANTICS_VERSION == "3"
