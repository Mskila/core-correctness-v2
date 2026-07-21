"""Three-layer experiment protocol with an untouched, single-use final holdout."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import math
from types import MappingProxyType
from typing import Mapping

from data_pipeline.validation import DatasetIdentity


class FinalHoldoutAccessError(RuntimeError):
    """Raised when a consumer crosses the final-holdout access boundary."""


class FinalHoldoutStateError(RuntimeError):
    """Raised when an experiment violates the freeze/evaluate lifecycle."""


class DatasetLayer(str, Enum):
    SEARCH_TRAIN = "search-train"
    SEARCH_VALIDATION = "search-validation"
    FINAL_HOLDOUT = "final-untouched-holdout"


class ProtocolConsumer(str, Enum):
    TRAINING = "training"
    WEB_PROGRESS = "web-progress"
    FINAL_EVALUATOR = "final-evaluator"


def require_search_layer(source: object, consumer: ProtocolConsumer) -> None:
    """Reject a data source explicitly labelled as the untouched final layer."""
    layer = getattr(source, "dataset_layer", None)
    if layer in {DatasetLayer.FINAL_HOLDOUT, DatasetLayer.FINAL_HOLDOUT.value}:
        raise FinalHoldoutAccessError(
            f"{consumer.value} cannot access the final holdout"
        )


@dataclass(frozen=True)
class FinalOOSEvidence:
    experiment_id: str
    strategy_fingerprint: str
    dataset: DatasetIdentity
    evaluated_at: str
    metrics: Mapping[str, float]


@dataclass
class ExperimentProtocol:
    """Own the search/freeze/final-evaluation state for one experiment identity."""

    experiment_id: str
    search_dataset: DatasetIdentity
    final_holdout: DatasetIdentity
    candidate_evaluation_count: int = 0
    frozen_strategy_fingerprint: str | None = None
    final_oos_evidence: FinalOOSEvidence | None = None
    _state: str = "search"

    @classmethod
    def create(
        cls,
        *,
        experiment_id: str,
        search_dataset: DatasetIdentity,
        final_holdout: DatasetIdentity,
    ) -> "ExperimentProtocol":
        if type(experiment_id) is not str or not experiment_id.strip():
            raise ValueError("experiment_id must be a non-empty string")
        if search_dataset == final_holdout:
            raise ValueError(
                "search dataset and final holdout must have distinct identities"
            )
        if search_dataset.symbol != final_holdout.symbol:
            raise ValueError("search dataset and final holdout symbol mismatch")
        if search_dataset.timeframe != final_holdout.timeframe:
            raise ValueError("search dataset and final holdout timeframe mismatch")
        if final_holdout.start_time_ns <= search_dataset.end_time_ns:
            raise ValueError(
                "final holdout must start strictly after the search dataset ends"
            )
        return cls(
            experiment_id=experiment_id,
            search_dataset=search_dataset,
            final_holdout=final_holdout,
        )

    def record_candidate_evaluations(self, count: int) -> "ExperimentProtocol":
        if self._state != "search":
            raise FinalHoldoutStateError(
                "tuning after strategy freeze/final evaluation requires a new "
                "experiment identity"
            )
        if type(count) is not int or count < 1:
            raise ValueError("candidate evaluation increment must be a positive integer")
        self.candidate_evaluation_count += count
        return self

    def freeze_strategy(self, strategy_fingerprint: str) -> "ExperimentProtocol":
        if self._state != "search":
            raise FinalHoldoutStateError("strategy is already frozen")
        if (
            type(strategy_fingerprint) is not str
            or not strategy_fingerprint.strip()
        ):
            raise ValueError("strategy_fingerprint must be a non-empty string")
        self.frozen_strategy_fingerprint = strategy_fingerprint
        self._state = "frozen"
        return self

    def dataset_for(
        self,
        consumer: ProtocolConsumer,
        layer: DatasetLayer,
    ) -> DatasetIdentity:
        if layer is not DatasetLayer.FINAL_HOLDOUT:
            if consumer is ProtocolConsumer.FINAL_EVALUATOR:
                raise FinalHoldoutAccessError(
                    "final evaluator may only access the final holdout"
                )
            return self.search_dataset
        if consumer in {
            ProtocolConsumer.TRAINING,
            ProtocolConsumer.WEB_PROGRESS,
        }:
            raise FinalHoldoutAccessError(
                f"{consumer.value} cannot access the final holdout"
            )
        if consumer is not ProtocolConsumer.FINAL_EVALUATOR:
            raise FinalHoldoutAccessError("unknown final holdout consumer")
        if self._state == "search":
            raise FinalHoldoutStateError(
                "final holdout cannot be loaded until the strategy is frozen"
            )
        if self._state != "frozen":
            raise FinalHoldoutStateError("final holdout is already consumed")
        self._state = "final-in-progress"
        return self.final_holdout

    def record_final_evaluation(
        self,
        dataset: DatasetIdentity,
        *,
        metrics: Mapping[str, float],
    ) -> "ExperimentProtocol":
        if self._state != "final-in-progress":
            raise FinalHoldoutStateError(
                "final evaluation must claim the frozen holdout exactly once"
            )
        if dataset != self.final_holdout:
            raise FinalHoldoutStateError("final evaluation dataset identity mismatch")
        if type(metrics) is not dict or not metrics:
            raise ValueError("final OOS metrics must be a non-empty exact dict")
        clean_metrics: dict[str, float] = {}
        for name, value in metrics.items():
            if type(name) is not str or not name:
                raise ValueError("final OOS metric names must be non-empty strings")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"final OOS metric {name!r} must be numeric")
            numeric = float(value)
            if not math.isfinite(numeric):
                raise ValueError(f"final OOS metric {name!r} must be finite")
            clean_metrics[name] = numeric
        assert self.frozen_strategy_fingerprint is not None
        self.final_oos_evidence = FinalOOSEvidence(
            experiment_id=self.experiment_id,
            strategy_fingerprint=self.frozen_strategy_fingerprint,
            dataset=dataset,
            evaluated_at=datetime.now(timezone.utc).isoformat(),
            metrics=MappingProxyType(clean_metrics),
        )
        self._state = "final-evaluated"
        return self
