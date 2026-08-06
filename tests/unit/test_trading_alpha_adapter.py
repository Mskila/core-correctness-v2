from __future__ import annotations

import json
from dataclasses import replace

import pytest

import trading_core.alpha_adapter as alpha_adapter
from model_core.semantics import ArtifactCompatibilityError
from tests.unit.test_artifacts import current_strategy_artifact
from trading_core import ClosedBarV1, TimeframeSeriesV1


def _strategy_path(tmp_path):
    artifact = current_strategy_artifact()
    path = tmp_path / artifact.run_identity.strategy_filename()
    path.write_text(json.dumps(artifact.to_dict()), encoding="utf-8")
    return artifact, path


def _series(count: int) -> TimeframeSeriesV1:
    return TimeframeSeriesV1(
        symbol="EURUSD",
        timeframe="H1",
        tick=0.00001,
        bars=tuple(
            ClosedBarV1(
                timestamp=index * 3_600,
                open=1.1,
                high=max(1.2, 1.11 + index * 0.001),
                low=min(1.0, 1.09 + index * 0.001),
                close=1.1 + index * 0.001,
                volume=100.0 + index,
            )
            for index in range(count)
        ),
    )


def test_load_alpha_strategy_revalidates_identity_and_canonical_name(tmp_path) -> None:
    artifact, path = _strategy_path(tmp_path)
    loaded = alpha_adapter.load_alpha_strategy(
        path,
        expected_symbol="EURUSD",
        expected_timeframe="H1",
    )

    assert loaded.formula_tokens == artifact.formula_tokens
    assert loaded.strategy_fingerprint == artifact.fingerprint
    assert loaded.dataset_fingerprint == (
        artifact.run_identity.artifact_identity.training_dataset.data_fingerprint
    )

    with pytest.raises(ArtifactCompatibilityError, match="identity"):
        alpha_adapter.load_alpha_strategy(
            path,
            expected_symbol="XAUUSD",
            expected_timeframe="H1",
        )

    renamed = tmp_path / "renamed.json"
    renamed.write_bytes(path.read_bytes())
    with pytest.raises(ArtifactCompatibilityError, match="filename"):
        alpha_adapter.load_alpha_strategy(
            renamed,
            expected_symbol="EURUSD",
            expected_timeframe="H1",
        )


def test_alpha_observation_preserves_live_position_and_exact_trailing_window(
    monkeypatch, tmp_path
) -> None:
    _, path = _strategy_path(tmp_path)
    strategy = alpha_adapter.load_alpha_strategy(
        path,
        expected_symbol="EURUSD",
        expected_timeframe="H1",
    )
    seen: dict[str, object] = {}

    monkeypatch.setattr(alpha_adapter, "formula_warmup_bars", lambda _length: 3)

    def fake_evaluate(formula, raw):
        seen["formula"] = formula
        seen["close"] = raw["close"].tolist()
        return {
            "state": "ok",
            "direction": "SHORT",
            "strength": 0.375,
            "position": -0.375,
            "factor_value": -0.42,
            "bars_used": 3,
            "message": "",
        }

    monkeypatch.setattr(alpha_adapter, "evaluate_signal", fake_evaluate)
    observation = alpha_adapter.evaluate_alpha_observation(
        strategy,
        _series(5),
        decision_close_timestamp=5 * 3_600,
    )

    assert seen["formula"] == list(strategy.formula_tokens)
    assert seen["close"][0] == pytest.approx([1.102, 1.103, 1.104])
    assert observation.position == -0.375
    assert observation.strength == 0.375
    assert observation.factor_value == -0.42
    assert observation.bar_close_timestamp == 5 * 3_600
    assert observation.bars_used == 3


def test_alpha_observation_fails_closed_when_enabled_history_is_short(
    monkeypatch, tmp_path
) -> None:
    _, path = _strategy_path(tmp_path)
    strategy = alpha_adapter.load_alpha_strategy(
        path,
        expected_symbol="EURUSD",
        expected_timeframe="H1",
    )
    monkeypatch.setattr(alpha_adapter, "formula_warmup_bars", lambda _length: 4)

    with pytest.raises(ValueError, match="insufficient.*H1"):
        alpha_adapter.evaluate_alpha_observation(
            strategy,
            _series(3),
            decision_close_timestamp=3 * 3_600,
        )


def test_batched_rolling_alpha_path_matches_individual_live_path(tmp_path) -> None:
    _, path = _strategy_path(tmp_path)
    strategy = alpha_adapter.load_alpha_strategy(
        path,
        expected_symbol="EURUSD",
        expected_timeframe="H1",
    )
    required = alpha_adapter.formula_warmup_bars(len(strategy.formula_tokens))
    series = _series(required + 3)
    indices = (required - 1, required, required + 2)

    batched = alpha_adapter.evaluate_alpha_observations_at_indices(
        strategy,
        series,
        end_indices=indices,
        batch_size=2,
    )
    individual = tuple(
        alpha_adapter.evaluate_alpha_observation(
            strategy,
            series,
            decision_close_timestamp=(index + 1) * 3_600,
        )
        for index in indices
    )

    assert batched == individual

    causal = alpha_adapter.evaluate_alpha_observations_causal_prefix(
        strategy,
        series,
        end_indices=indices,
    )
    assert causal == individual

    future = replace(
        series,
        bars=(
            *series.bars,
            ClosedBarV1(
                timestamp=len(series.bars) * 3_600,
                open=9.0,
                high=10.0,
                low=8.0,
                close=9.5,
                volume=999.0,
            ),
        ),
    )
    assert alpha_adapter.evaluate_alpha_observations_causal_prefix(
        strategy,
        future,
        end_indices=indices,
    ) == causal
