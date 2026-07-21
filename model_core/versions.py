"""Authoritative CORE-06 compatibility boundary map."""

from __future__ import annotations

from types import MappingProxyType

from .semantics import (
    CHECKPOINT_SCHEMA_VERSION,
    CORE_SEMANTICS_VERSION,
    DATA_CANONICALIZATION_VERSION,
    DATA_SCHEMA_VERSION,
    EXECUTION_SEMANTICS_VERSION,
    HISTORY_SCHEMA_VERSION,
    REPORT_SCHEMA_VERSION,
    STRATEGY_SCHEMA_VERSION,
    VALIDATION_SCHEMA_VERSION,
)
from .vocab import VOCAB_VERSION


VERSION_CHANGE_TABLE = MappingProxyType(
    {
        "data_canonicalization": DATA_CANONICALIZATION_VERSION,
        "dataset_schema": DATA_SCHEMA_VERSION,
        "reward_core": CORE_SEMANTICS_VERSION,
        "vocab_operator": VOCAB_VERSION,
        "execution": EXECUTION_SEMANTICS_VERSION,
        "validation": VALIDATION_SCHEMA_VERSION,
        "checkpoint": CHECKPOINT_SCHEMA_VERSION,
        "strategy": STRATEGY_SCHEMA_VERSION,
        "history": HISTORY_SCHEMA_VERSION,
        "backtest_report": REPORT_SCHEMA_VERSION,
    }
)
