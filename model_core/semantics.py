from __future__ import annotations

CORE_SEMANTICS_VERSION = "2"
DATA_SCHEMA_VERSION = "ohlcv-v2"
LABEL_SEMANTICS_VERSION = "close-t__open-t1-to-open-t2-v2"
EXECUTION_SEMANTICS_VERSION = "tanh-threshold-cost-liquidate-v2"
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
