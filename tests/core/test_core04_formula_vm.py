from __future__ import annotations

import pandas as pd
import pytest
import torch

from data_pipeline.validation import canonicalize_ohlcv
from data_pipeline.parquet_manager import ParquetDataManager, inspect_parquet_file
from model_core.features import FEATURE_METADATA_BY_NAME, FEATURE_REGISTRY
from model_core.ops import DomainError, OPERATOR_REGISTRY, ShapeError
from model_core.semantics import CORE_SEMANTICS_VERSION
from model_core.vm import (
    FormulaDomain,
    FormulaErrorKind,
    FormulaEvaluationError,
    StackVM,
    analyze_formula_structure,
    validate_formula_structure,
)
from model_core.vocab import FORMULA_VOCAB, VOCAB_VERSION


def _token(name: str) -> int:
    return FORMULA_VOCAB.token_names.index(name)


@pytest.mark.core
def test_exp_decay_weights_the_most_recent_sample_most() -> None:
    spec = next(
        item
        for item in OPERATOR_REGISTRY.operator_specs
        if item.name == "TS_DECAY_EXP_5"
    )
    values = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    result = spec.transform(values)
    assert result[0, -1].item() == pytest.approx(129.0 / 31.0)
    assert result[0, -1].item() > values[0].mean().item()


@pytest.mark.core
def test_postfix_abstract_interpreter_tracks_stack_and_operand_domains() -> None:
    ret = _token("RET")
    abs_op = _token("ABS")
    neg = _token("NEG")
    add = _token("ADD")
    if_gt = _token("IF_GT")

    assert analyze_formula_structure([ret]).final_domain is FormulaDomain.SIGNED
    assert (
        analyze_formula_structure([ret, abs_op]).final_domain
        is FormulaDomain.NONNEGATIVE
    )
    assert (
        analyze_formula_structure([ret, abs_op, neg]).final_domain
        is FormulaDomain.NONPOSITIVE
    )
    assert analyze_formula_structure([ret, ret, add]).final_stack_depth == 1
    assert analyze_formula_structure([ret, ret, ret, if_gt]).final_stack_depth == 1

    assert validate_formula_structure([add], FORMULA_VOCAB.token_names)
    assert validate_formula_structure([ret, ret], FORMULA_VOCAB.token_names)


@pytest.mark.core
def test_std_and_quantile_domains_are_not_static_sign_restorers() -> None:
    ret = _token("RET")
    rank = _token("TS_RANK_5")
    std = _token("TS_STD_5")
    quantile = _token("TS_QUANTILE_10")

    std_analysis = analyze_formula_structure([ret, rank, std])
    quantile_analysis = analyze_formula_structure([ret, rank, quantile])
    assert std_analysis.final_domain is FormulaDomain.NONNEGATIVE
    assert quantile_analysis.final_domain is FormulaDomain.NONNEGATIVE
    assert std_analysis.violations
    assert quantile_analysis.violations


@pytest.mark.core
@pytest.mark.parametrize(
    ("kind", "operator", "replacement"),
    [
        (FormulaErrorKind.DOMAIN_ERROR, "DIV", DomainError("division by zero")),
        (FormulaErrorKind.SHAPE_ERROR, "ADD", ShapeError("bad shape")),
        (FormulaErrorKind.OPERATOR_ERROR, "NEG", RuntimeError("broken operator")),
    ],
)
def test_vm_preserves_structured_operator_failures(
    kind: FormulaErrorKind,
    operator: str,
    replacement: Exception,
) -> None:
    vm = StackVM()
    token = _token(operator)
    arity = vm.arity_map[token]

    def fail(*_args):
        raise replacement

    vm.op_map[token] = fail
    formula = [0] * arity + [token]
    features = torch.ones(1, FORMULA_VOCAB.feature_count, 4)
    with pytest.raises(FormulaEvaluationError) as caught:
        vm.evaluate(formula, features)
    error = caught.value
    assert error.formula == tuple(formula)
    assert error.token_index == arity
    assert error.operator == operator
    assert error.error_kind is kind


@pytest.mark.core
def test_vm_rejects_nonfinite_intermediates_and_legacy_adapter_keeps_reason() -> None:
    vm = StackVM()
    features = torch.ones(1, FORMULA_VOCAB.feature_count, 4)
    features[:, 0, 2] = float("inf")
    with pytest.raises(FormulaEvaluationError) as caught:
        vm.evaluate([0], features)
    assert caught.value.error_kind is FormulaErrorKind.NONFINITE

    assert vm.execute([0], features) is None
    assert vm.last_error is not None
    assert vm.last_error.error_kind is FormulaErrorKind.NONFINITE


@pytest.mark.core
def test_vm_wraps_normalization_failures_with_structured_context(monkeypatch) -> None:
    vm = StackVM()

    def fail(_value):
        raise RuntimeError("normalizer failed")

    monkeypatch.setattr(vm, "_normalize_output", fail)
    features = torch.ones(1, FORMULA_VOCAB.feature_count, 4)
    with pytest.raises(FormulaEvaluationError) as caught:
        vm.evaluate([0], features)
    assert caught.value.error_kind is FormulaErrorKind.OPERATOR_ERROR
    assert caught.value.operator == "NORMALIZE"


@pytest.mark.core
def test_vm_rejects_division_by_zero_and_overflow_without_coercion() -> None:
    vm = StackVM()
    feature_count = FORMULA_VOCAB.feature_count
    features = torch.ones(1, feature_count, 4)
    features[:, 1] = 0.0
    with pytest.raises(FormulaEvaluationError) as division:
        vm.evaluate([0, 1, _token("DIV")], features)
    assert division.value.error_kind is FormulaErrorKind.DOMAIN_ERROR

    features[:, 0] = torch.finfo(features.dtype).max
    with pytest.raises(FormulaEvaluationError) as overflow:
        vm.evaluate([0, 0, _token("MUL")], features)
    assert overflow.value.error_kind is FormulaErrorKind.NONFINITE


@pytest.mark.core
def test_feature_and_dataset_metadata_publish_volume_semantics(tmp_path) -> None:
    assert set(FEATURE_METADATA_BY_NAME) == set(FEATURE_REGISTRY.feature_names)
    for spec in FEATURE_REGISTRY.feature_specs:
        metadata = FEATURE_METADATA_BY_NAME[spec.name]
        assert metadata.implementation
        assert metadata.definition
        if spec.category == "volume":
            assert metadata.volume_input == "dataset.volume"

    frame = pd.DataFrame(
        {
            "time": pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC"),
            "open": [1.0, 1.1, 1.2, 1.3],
            "high": [1.1, 1.2, 1.3, 1.4],
            "low": [0.9, 1.0, 1.1, 1.2],
            "close": [1.05, 1.15, 1.25, 1.35],
            "base_volume": [10.0, 11.0, 12.0, 13.0],
        }
    )
    dataset = canonicalize_ohlcv(
        frame,
        symbol="CORE04",
        timeframe="H1",
        volume_type="base",
    )
    assert dataset.volume_type == "base"
    assert dataset.identity.volume_type == "base"

    parquet_path = tmp_path / "CORE04_H1.parquet"
    frame.to_parquet(parquet_path, index=False)
    manager = ParquetDataManager(parquet_path, volume_type="base")
    manager.load()
    assert manager.loaded_volume_type == "base"
    assert inspect_parquet_file(parquet_path, volume_type="base")["volume_type"] == "base"


@pytest.mark.core
def test_core04_bumps_formula_semantics_versions() -> None:
    assert CORE_SEMANTICS_VERSION == "4"
    assert VOCAB_VERSION == FORMULA_VOCAB.version
