from __future__ import annotations

import pytest

from model_core.validation_protocol import (
    DatasetLayer,
    ExperimentProtocol,
    FinalHoldoutAccessError,
    FinalHoldoutStateError,
    ProtocolConsumer,
)
from model_core.artifacts import StrategyArtifact
from model_core.walk_forward import build_walk_forward_folds
from model_core.semantics import InsufficientWalkForwardDataError
from tests.unit.test_artifacts import dataset_identity, strategy_artifact
import training_service


def test_training_and_web_cannot_read_final_holdout() -> None:
    protocol = ExperimentProtocol.create(
        experiment_id="experiment-a",
        search_dataset=dataset_identity(),
        final_holdout=dataset_identity(
            start=10_000,
            end=18_000,
            data_fingerprint="b" * 64,
        ),
    )

    for consumer in (ProtocolConsumer.TRAINING, ProtocolConsumer.WEB_PROGRESS):
        with pytest.raises(FinalHoldoutAccessError, match="final holdout"):
            protocol.dataset_for(consumer, DatasetLayer.FINAL_HOLDOUT)

    class FinalManager:
        dataset_layer = DatasetLayer.FINAL_HOLDOUT

    with pytest.raises(FinalHoldoutAccessError, match="training.*final holdout"):
        training_service.run_training_session(
            FinalManager(),
            source_path=None,
            from_scratch=True,
            random_seed=42,
        )


def test_two_observation_fold_is_rejected() -> None:
    with pytest.raises(
        InsufficientWalkForwardDataError,
        match=r"min_fold_bars.*>= 200",
    ):
        build_walk_forward_folds(
            total_bars=1_000,
            n_blocks=3,
            configured_gap=2,
            min_fold_bars=2,
            warmup_bars=0,
            label_lookahead=2,
        )


def test_final_holdout_requires_freeze_and_is_consumed_once() -> None:
    protocol = ExperimentProtocol.create(
        experiment_id="experiment-a",
        search_dataset=dataset_identity(),
        final_holdout=dataset_identity(
            start=10_000,
            end=18_000,
            data_fingerprint="b" * 64,
        ),
    )
    protocol.record_candidate_evaluations(7)

    with pytest.raises(FinalHoldoutStateError, match="frozen"):
        protocol.dataset_for(
            ProtocolConsumer.FINAL_EVALUATOR,
            DatasetLayer.FINAL_HOLDOUT,
        )

    frozen = protocol.freeze_strategy("strategy-fingerprint")
    final_dataset = frozen.dataset_for(
        ProtocolConsumer.FINAL_EVALUATOR,
        DatasetLayer.FINAL_HOLDOUT,
    )
    evaluated = frozen.record_final_evaluation(
        final_dataset,
        metrics={"net_return": 0.12, "sortino": 1.4},
    )

    with pytest.raises(FinalHoldoutStateError, match="new experiment identity"):
        evaluated.record_candidate_evaluations(1)
    with pytest.raises(FinalHoldoutStateError, match="already consumed"):
        evaluated.dataset_for(
            ProtocolConsumer.FINAL_EVALUATOR,
            DatasetLayer.FINAL_HOLDOUT,
        )

    assert evaluated.final_oos_evidence is not None
    assert evaluated.final_oos_evidence.dataset == final_dataset
    assert evaluated.candidate_evaluation_count == 7


def test_strategy_evidence_separates_selection_from_final_oos() -> None:
    legacy = strategy_artifact()
    selected = StrategyArtifact.create(
        run_identity=legacy.run_identity,
        formula_tokens=legacy.formula_tokens,
        decoded_formula=legacy.decoded_formula,
        best_score=legacy.best_score,
        fold_evidence=legacy.fold_evidence,
        generated_at=legacy.generated_at,
        candidate_evaluation_count=23,
    )

    selection_payload = selected.to_dict()
    assert selection_payload["schema_version"] == "strategy-v3"
    assert "fold_evidence" not in selection_payload
    assert selection_payload["final_oos_evidence"] is None
    assert selected.selection_evidence["candidate_evaluation_count"] == 23
    assert selected.selection_evidence["random_seed"] == 42
    assert selected.selection_evidence["fold_count"] == 4
    assert selected.selection_evidence["timeframe"] == "H1"
    assert legacy.selection_evidence["protocol"] == (
        "legacy-fold-evidence-as-selection-only"
    )
    assert legacy.final_oos_evidence is None

    protocol = ExperimentProtocol.create(
        experiment_id=selected.run_identity.run_id,
        search_dataset=selected.run_identity.artifact_identity.training_dataset,
        final_holdout=dataset_identity(
            start=10_000,
            end=18_000,
            data_fingerprint="b" * 64,
        ),
    )
    protocol.record_candidate_evaluations(23).freeze_strategy(selected.fingerprint)
    final_dataset = protocol.dataset_for(
        ProtocolConsumer.FINAL_EVALUATOR,
        DatasetLayer.FINAL_HOLDOUT,
    )
    protocol.record_final_evaluation(final_dataset, metrics={"net_return": 0.12})
    assert protocol.final_oos_evidence is not None
    finalized = selected.with_final_oos_evidence(protocol.final_oos_evidence)

    round_trip = StrategyArtifact.from_dict(finalized.to_dict())
    assert round_trip.final_oos_evidence is not None
    assert round_trip.final_oos_evidence.dataset == final_dataset
    assert round_trip.to_dict()["final_oos_evidence"]["holdout_start_time_ns"] == (
        final_dataset.start_time_ns
    )
