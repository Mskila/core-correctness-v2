from __future__ import annotations

CORE_SEMANTICS_VERSION = "5"
DATA_SCHEMA_VERSION = "ohlcv-v4"
DATA_CANONICALIZATION_VERSION = "float32-le-ns-gap-volume-v2"
LABEL_SEMANTICS_VERSION = "close-t__open-t1-to-open-t2-v2"
EXECUTION_SEMANTICS_VERSION = "tanh-threshold-cost-liquidate-v4"
VALIDATION_SCHEMA_VERSION = "temporal-validation-v2"
CHECKPOINT_SCHEMA_VERSION = "checkpoint-v3"
STRATEGY_SCHEMA_VERSION = "strategy-v3"
HISTORY_SCHEMA_VERSION = "training-history-v3"
REPORT_SCHEMA_VERSION = "backtest-report-v3"
LABEL_LOOKAHEAD_BARS = 2


class CoreCorrectnessError(RuntimeError):
    """V2 核心正确性错误基类。"""


class DataValidationError(CoreCorrectnessError):
    pass


class DatasetAlignmentError(CoreCorrectnessError):
    pass


class ArtifactCompatibilityError(CoreCorrectnessError):
    pass


class InsufficientWalkForwardDataError(CoreCorrectnessError):
    pass


class BacktestModeError(CoreCorrectnessError):
    pass
