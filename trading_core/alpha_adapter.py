"""Strict V2/V3 strategy loading and shared Alpha signal observation."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from data_pipeline.validation import normalize_timeframe_name
from model_core.artifacts import STRATEGY_SCHEMA_VERSION, StrategyArtifact
from model_core.execution import factor_to_position
from model_core.features import MT5FeatureEngineer
from model_core.semantics import ArtifactCompatibilityError
from model_core.vm import StackVM
from model_core.walk_forward import formula_warmup_bars
from strategy_manager.live_signal import evaluate_signal, min_exposure

from .models import AlphaObservationV1, LoadedAlphaStrategyV1, TimeframeSeriesV1
from .time_alignment import bar_close_timestamp


_BATCH_VM = StackVM()


def load_alpha_strategy(
    path: str | Path,
    *,
    expected_symbol: str,
    expected_timeframe: str,
) -> LoadedAlphaStrategyV1:
    strategy_path = Path(path)
    try:
        payload = json.loads(strategy_path.read_text(encoding="utf-8"))
        artifact = StrategyArtifact.from_dict(payload)
    except ArtifactCompatibilityError:
        raise
    except Exception as exc:  # noqa: BLE001 - normalize the public artifact boundary
        raise ArtifactCompatibilityError(f"strategy artifact cannot be loaded: {exc}") from exc
    if artifact.schema_version != STRATEGY_SCHEMA_VERSION:
        raise ArtifactCompatibilityError(
            "strategy schema is not current: "
            f"expected={STRATEGY_SCHEMA_VERSION!r} actual={artifact.schema_version!r}"
        )
    canonical_name = artifact.run_identity.strategy_filename()
    if strategy_path.name != canonical_name:
        raise ArtifactCompatibilityError(
            f"strategy filename is not canonical: expected={canonical_name} actual={strategy_path.name}"
        )
    identity = artifact.run_identity.artifact_identity
    timeframe = normalize_timeframe_name(expected_timeframe)
    if identity.symbol != expected_symbol or identity.timeframe != timeframe:
        raise ArtifactCompatibilityError(
            "strategy identity mismatch: "
            f"expected={expected_symbol}/{timeframe} "
            f"actual={identity.symbol}/{identity.timeframe}"
        )
    return LoadedAlphaStrategyV1(
        symbol=identity.symbol,
        timeframe=identity.timeframe,
        formula_tokens=artifact.formula_tokens,
        strategy_fingerprint=artifact.fingerprint,
        dataset_fingerprint=identity.training_dataset.data_fingerprint,
        best_score=float(artifact.best_score),
    )


def _raw_dict(bars):
    def tensor(values):
        return torch.tensor([values], dtype=torch.float32)

    return {
        "open": tensor([bar.open for bar in bars]),
        "high": tensor([bar.high for bar in bars]),
        "low": tensor([bar.low for bar in bars]),
        "close": tensor([bar.close for bar in bars]),
        "volume": tensor([bar.volume for bar in bars]),
        "time": tensor([float(bar.timestamp) for bar in bars]),
    }


def evaluate_alpha_observation(
    strategy: LoadedAlphaStrategyV1,
    series: TimeframeSeriesV1,
    *,
    decision_close_timestamp: int,
) -> AlphaObservationV1:
    if series.symbol != strategy.symbol or series.timeframe != strategy.timeframe:
        raise ValueError("Alpha strategy and series identity mismatch")
    eligible = tuple(
        bar
        for bar in series.bars
        if bar_close_timestamp(bar, series.timeframe) <= decision_close_timestamp
    )
    required = formula_warmup_bars(len(strategy.formula_tokens))
    if len(eligible) < required:
        raise ValueError(
            f"insufficient {series.timeframe} Alpha history: {len(eligible)}/{required}"
        )
    trailing = eligible[-required:]
    result = evaluate_signal(list(strategy.formula_tokens), _raw_dict(trailing))
    if result.get("state") != "ok":
        raise ValueError(
            f"Alpha evaluation failed for {series.timeframe}: "
            f"{result.get('message', result.get('state', 'unknown'))}"
        )
    position = float(result["position"])
    return AlphaObservationV1(
        symbol=series.symbol,
        timeframe=series.timeframe,
        bar_close_timestamp=bar_close_timestamp(trailing[-1], series.timeframe),
        strategy_fingerprint=strategy.strategy_fingerprint,
        position=position,
        strength=abs(position),
        factor_value=float(result["factor_value"]),
        bars_used=int(result["bars_used"]),
    )


def evaluate_alpha_observations_at_indices(
    strategy: LoadedAlphaStrategyV1,
    series: TimeframeSeriesV1,
    *,
    end_indices: tuple[int, ...],
    batch_size: int = 16,
) -> tuple[AlphaObservationV1, ...]:
    """Evaluate exact live-sized trailing windows in bounded tensor batches."""

    if series.symbol != strategy.symbol or series.timeframe != strategy.timeframe:
        raise ValueError("Alpha strategy and series identity mismatch")
    if type(end_indices) is not tuple or any(type(index) is not int for index in end_indices):
        raise TypeError("end_indices must be an exact tuple of integers")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    required = formula_warmup_bars(len(strategy.formula_tokens))
    if any(index < required - 1 or index >= len(series.bars) for index in end_indices):
        raise ValueError("Alpha batch index lacks its exact trailing warmup window")

    output: list[AlphaObservationV1] = []
    fields = ("open", "high", "low", "close", "volume", "timestamp")
    with torch.no_grad():
        for offset in range(0, len(end_indices), batch_size):
            chunk = end_indices[offset : offset + batch_size]
            rows = [series.bars[index - required + 1 : index + 1] for index in chunk]
            raw: dict[str, torch.Tensor] = {}
            for field in fields:
                key = "time" if field == "timestamp" else field
                raw[key] = torch.tensor(
                    [[float(getattr(bar, field)) for bar in row] for row in rows],
                    dtype=torch.float32,
                )
            features = MT5FeatureEngineer.compute_features(raw)
            factors = _BATCH_VM.execute(list(strategy.formula_tokens), features)
            if factors is None or factors.ndim != 2 or factors.shape[1] == 0:
                raise ValueError(f"Alpha batch evaluation failed for {series.timeframe}")
            positions = factor_to_position(
                factors[:, -1:],
                min_exposure=min_exposure(),
            )
            for row_index, end_index in enumerate(chunk):
                position = float(positions[row_index, 0].item())
                factor_value = round(float(factors[row_index, -1].item()), 6)
                output.append(
                    AlphaObservationV1(
                        symbol=series.symbol,
                        timeframe=series.timeframe,
                        bar_close_timestamp=bar_close_timestamp(
                            series.bars[end_index], series.timeframe
                        ),
                        strategy_fingerprint=strategy.strategy_fingerprint,
                        position=position,
                        strength=abs(position),
                        factor_value=factor_value,
                        bars_used=required,
                    )
                )
    return tuple(output)


def evaluate_alpha_observations_causal_prefix(
    strategy: LoadedAlphaStrategyV1,
    series: TimeframeSeriesV1,
    *,
    end_indices: tuple[int, ...],
) -> tuple[AlphaObservationV1, ...]:
    """Evaluate a causal prefix once and select requested warmed-up endpoints.

    The supplied series is sliced at the greatest requested index before any
    feature is computed, so a caller cannot accidentally include the reserved
    holdout tail.  Release tooling additionally compares deterministic anchor
    points against :func:`evaluate_alpha_observations_at_indices`.
    """

    if series.symbol != strategy.symbol or series.timeframe != strategy.timeframe:
        raise ValueError("Alpha strategy and series identity mismatch")
    if type(end_indices) is not tuple or any(type(index) is not int for index in end_indices):
        raise TypeError("end_indices must be an exact tuple of integers")
    if not end_indices:
        return ()
    if any(end_indices[index] <= end_indices[index - 1] for index in range(1, len(end_indices))):
        raise ValueError("Alpha end indices must be strictly increasing")
    required = formula_warmup_bars(len(strategy.formula_tokens))
    if end_indices[0] < required - 1 or end_indices[-1] >= len(series.bars):
        raise ValueError("Alpha causal index lacks its exact warmup prefix")
    prefix = series.bars[: end_indices[-1] + 1]
    raw = _raw_dict(prefix)
    with torch.no_grad():
        features = MT5FeatureEngineer.compute_features(raw)
        factors = _BATCH_VM.execute(list(strategy.formula_tokens), features)
        if factors is None or factors.ndim != 2 or factors.shape[1] == 0:
            raise ValueError(f"Alpha causal evaluation failed for {series.timeframe}")
        positions = factor_to_position(factors, min_exposure=min_exposure())
    return tuple(
        AlphaObservationV1(
            symbol=series.symbol,
            timeframe=series.timeframe,
            bar_close_timestamp=bar_close_timestamp(series.bars[index], series.timeframe),
            strategy_fingerprint=strategy.strategy_fingerprint,
            position=float(positions[0, index].item()),
            strength=abs(float(positions[0, index].item())),
            factor_value=round(float(factors[0, index].item()), 6),
            bars_used=required,
        )
        for index in end_indices
    )
