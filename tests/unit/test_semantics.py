import pytest

from model_core.semantics import (
    CORE_SEMANTICS_VERSION,
    DATA_SCHEMA_VERSION,
    EXECUTION_SEMANTICS_VERSION,
    LABEL_LOOKAHEAD_BARS,
    LABEL_SEMANTICS_VERSION,
    ArtifactCompatibilityError,
    BacktestModeError,
    DataValidationError,
    DatasetAlignmentError,
    InsufficientWalkForwardDataError,
)


def test_v2_semantics_constants_are_explicit() -> None:
    assert CORE_SEMANTICS_VERSION == "2"
    assert DATA_SCHEMA_VERSION == "ohlcv-v2"
    assert LABEL_SEMANTICS_VERSION == "close-t__open-t1-to-open-t2-v2"
    assert EXECUTION_SEMANTICS_VERSION == "tanh-threshold-cost-liquidate-v2"
    assert LABEL_LOOKAHEAD_BARS == 2


@pytest.mark.parametrize(
    "error_type",
    [
        DataValidationError,
        DatasetAlignmentError,
        ArtifactCompatibilityError,
        InsufficientWalkForwardDataError,
        BacktestModeError,
    ],
)
def test_core_errors_are_runtime_errors(error_type: type[Exception]) -> None:
    assert issubclass(error_type, RuntimeError)
