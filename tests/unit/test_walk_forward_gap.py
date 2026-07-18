"""Strict walk-forward gap and minimum-data contracts."""

from dataclasses import fields

import pytest

import model_core.walk_forward as walk_forward_module
from model_core.features import MAX_FEATURE_LOOKBACK
from model_core.ops import MAX_OPERATOR_LOOKBACK
from model_core.semantics import InsufficientWalkForwardDataError
from model_core.walk_forward import (
    WalkForwardFold,
    build_walk_forward_folds,
    formula_warmup_bars,
    required_training_bars,
)


class _IntSubclass(int):
    pass


class _HostileProtocols:
    calls = 0

    def _fail(self):
        type(self).calls += 1
        raise RuntimeError("hostile protocol executed")

    __repr__ = lambda self: self._fail()
    __str__ = lambda self: self._fail()
    __int__ = lambda self: self._fail()
    __index__ = lambda self: self._fail()
    __hash__ = lambda self: self._fail()
    __eq__ = lambda self, other: self._fail()


_HUGE_INTEGERS = [
    pytest.param(10**1000, id="positive-1000-digits"),
    pytest.param(-(10**1000), id="negative-1000-digits"),
    pytest.param(10**5000, id="positive-5000-digits"),
    pytest.param(-(10**5000), id="negative-5000-digits"),
]


def test_minimum_scorable_fold_observations_is_explicitly_two() -> None:
    assert walk_forward_module.MIN_SCORABLE_FOLD_OBSERVATIONS == 2


def test_walk_forward_fold_has_exact_fields() -> None:
    assert [field.name for field in fields(WalkForwardFold)] == [
        "fold_index",
        "train_start",
        "train_end",
        "val_start",
        "val_end",
        "effective_gap",
    ]


def test_walk_forward_keeps_full_effective_gap_and_expands_training() -> None:
    folds = build_walk_forward_folds(
        total_bars=1600,
        n_blocks=5,
        configured_gap=20,
        min_fold_bars=200,
        warmup_bars=400,
        label_lookahead=2,
    )

    assert len(folds) == 4
    assert [fold.fold_index for fold in folds] == [0, 1, 2, 3]
    assert all(fold.train_start == 400 for fold in folds)
    assert all(fold.effective_gap == 20 for fold in folds)
    assert all(fold.val_start - fold.train_end == 20 for fold in folds)
    assert all(fold.val_end - fold.val_start >= 200 for fold in folds)
    assert all(
        current.train_end > previous.train_end
        for previous, current in zip(folds, folds[1:])
    )
    assert folds[-1].val_end <= 1600 - 2


def test_effective_gap_is_never_less_than_label_lookahead() -> None:
    folds = build_walk_forward_folds(
        total_bars=1200,
        n_blocks=5,
        configured_gap=0,
        min_fold_bars=100,
        warmup_bars=100,
        label_lookahead=2,
    )
    assert all(fold.effective_gap == 2 for fold in folds)
    assert all(fold.val_start - fold.train_end == 2 for fold in folds)


def test_required_training_bars_uses_all_gaps_and_label_tail() -> None:
    assert required_training_bars(
        warmup_bars=400,
        label_lookahead=2,
        n_blocks=5,
        min_fold_bars=200,
        configured_gap=20,
    ) == 1482


@pytest.mark.parametrize("formula_length", _HUGE_INTEGERS)
def test_formula_warmup_rejects_operationally_excessive_exact_integers(
    formula_length: int,
) -> None:
    with pytest.raises(ValueError) as exc_info:
        formula_warmup_bars(formula_length)

    message = str(exc_info.value)
    assert "formula_length" in message
    assert len(message) <= 160


@pytest.mark.parametrize(
    "parameter",
    [
        "warmup_bars",
        "label_lookahead",
        "n_blocks",
        "min_fold_bars",
        "configured_gap",
    ],
)
@pytest.mark.parametrize("huge", _HUGE_INTEGERS)
def test_required_training_bars_rejects_operationally_excessive_exact_integers(
    parameter: str,
    huge: int,
) -> None:
    arguments = {
        "warmup_bars": 2137,
        "label_lookahead": 2,
        "n_blocks": 5,
        "min_fold_bars": 200,
        "configured_gap": 20,
    }
    arguments[parameter] = huge

    with pytest.raises(ValueError) as exc_info:
        required_training_bars(**arguments)

    message = str(exc_info.value)
    assert parameter in message
    assert len(message) <= 160


def test_public_sizing_apis_preserve_approved_operational_topology() -> None:
    assert formula_warmup_bars(8) == 2137
    assert required_training_bars(
        warmup_bars=2137,
        label_lookahead=2,
        n_blocks=5,
        min_fold_bars=200,
        configured_gap=20,
    ) == 3219


def test_required_training_bars_rejects_one_bar_folds() -> None:
    with pytest.raises(
        ValueError,
        match=r"min_fold_bars must be an integer >= 2",
    ):
        required_training_bars(
            warmup_bars=0,
            label_lookahead=2,
            n_blocks=2,
            min_fold_bars=1,
            configured_gap=2,
        )


def test_builder_rejects_one_bar_folds_with_domain_details() -> None:
    with pytest.raises(InsufficientWalkForwardDataError) as exc_info:
        build_walk_forward_folds(
            total_bars=6,
            n_blocks=2,
            configured_gap=2,
            min_fold_bars=1,
            warmup_bars=0,
            label_lookahead=2,
        )

    message = str(exc_info.value)
    for fragment in (
        "parameter=min_fold_bars",
        "value=1",
        "reason=min_fold_bars must be an integer >= 2",
        "required=",
        "actual=6",
        "warmup=0",
        "gap=2",
        "blocks=2",
        "min_fold_bars=1",
    ):
        assert fragment in message


def test_insufficient_data_never_reduces_gap_or_returns_a_fake_fold() -> None:
    with pytest.raises(InsufficientWalkForwardDataError) as exc_info:
        build_walk_forward_folds(
            total_bars=500,
            n_blocks=5,
            configured_gap=20,
            min_fold_bars=200,
            warmup_bars=200,
            label_lookahead=2,
        )

    message = str(exc_info.value)
    for fragment in (
        "required=",
        "actual=500",
        "warmup=200",
        "gap=20",
        "blocks=5",
        "min_fold_bars=200",
    ):
        assert fragment in message


@pytest.mark.parametrize("n_blocks", [-3, -1, 0, 1])
def test_fewer_than_two_blocks_fails_closed_with_domain_details(
    n_blocks: int,
) -> None:
    with pytest.raises(InsufficientWalkForwardDataError) as exc_info:
        build_walk_forward_folds(
            total_bars=500,
            n_blocks=n_blocks,
            configured_gap=20,
            min_fold_bars=200,
            warmup_bars=100,
            label_lookahead=2,
        )

    message = str(exc_info.value)
    for fragment in (
        "required=522",
        "actual=500",
        "warmup=100",
        "gap=20",
        f"blocks={n_blocks}",
        "min_fold_bars=200",
        "parameter=n_blocks",
        f"value={n_blocks!r}",
        "reason=at least two blocks are required",
    ):
        assert fragment in message


@pytest.mark.parametrize(
    ("parameter", "invalid_value"),
    [
        ("total_bars", -1),
        ("configured_gap", -1),
        ("min_fold_bars", 0),
        ("warmup_bars", -1),
        ("label_lookahead", -1),
        ("total_bars", True),
        ("n_blocks", False),
        ("configured_gap", 1.5),
        ("min_fold_bars", "200"),
        ("warmup_bars", None),
        ("label_lookahead", 2.0),
        ("n_blocks", "5"),
    ],
)
def test_invalid_public_build_arguments_use_walk_forward_domain_error(
    parameter: str, invalid_value,
) -> None:
    arguments = {
        "total_bars": 1600,
        "n_blocks": 5,
        "configured_gap": 20,
        "min_fold_bars": 200,
        "warmup_bars": 400,
        "label_lookahead": 2,
    }
    arguments[parameter] = invalid_value

    with pytest.raises(InsufficientWalkForwardDataError) as exc_info:
        build_walk_forward_folds(**arguments)

    message = str(exc_info.value)
    for fragment in (
        "invalid walk-forward configuration",
        f"parameter={parameter}",
        f"value={invalid_value!r}",
        "reason=",
        "required=",
        f"actual={arguments['total_bars']!r}",
        f"warmup={arguments['warmup_bars']!r}",
        f"gap={arguments['configured_gap']!r}",
        f"blocks={arguments['n_blocks']!r}",
        f"min_fold_bars={arguments['min_fold_bars']!r}",
    ):
        assert fragment in message


def test_formula_warmup_bars_uses_declared_feature_and_operator_history() -> None:
    assert formula_warmup_bars(1) == MAX_FEATURE_LOOKBACK + MAX_OPERATOR_LOOKBACK - 1
    assert formula_warmup_bars(3) == MAX_FEATURE_LOOKBACK + 3 * (
        MAX_OPERATOR_LOOKBACK - 1
    )


@pytest.mark.parametrize("formula_length", [0, -1])
def test_formula_warmup_rejects_non_positive_length(formula_length: int) -> None:
    with pytest.raises(ValueError, match="formula_length must be >= 1"):
        formula_warmup_bars(formula_length)


@pytest.mark.parametrize("value", [True, _IntSubclass(1)])
def test_formula_warmup_requires_exact_builtin_int(value) -> None:
    with pytest.raises(ValueError, match="formula_length must be >= 1"):
        formula_warmup_bars(value)


@pytest.mark.parametrize(
    "parameter",
    [
        "warmup_bars",
        "label_lookahead",
        "n_blocks",
        "min_fold_bars",
        "configured_gap",
    ],
)
@pytest.mark.parametrize("value", [True, _IntSubclass(5)])
def test_required_training_bars_requires_exact_builtin_int(parameter, value) -> None:
    arguments = {
        "warmup_bars": 400,
        "label_lookahead": 2,
        "n_blocks": 5,
        "min_fold_bars": 200,
        "configured_gap": 20,
    }
    arguments[parameter] = value
    with pytest.raises(ValueError):
        required_training_bars(**arguments)


@pytest.mark.parametrize(
    "parameter",
    [
        "total_bars",
        "warmup_bars",
        "label_lookahead",
        "n_blocks",
        "min_fold_bars",
        "configured_gap",
    ],
)
@pytest.mark.parametrize(
    "value",
    [True, _IntSubclass(5), _HostileProtocols()],
    ids=["bool", "int-subclass", "hostile-protocols"],
)
def test_builder_rejects_non_exact_ints_without_caller_protocols(
    parameter, value
) -> None:
    _HostileProtocols.calls = 0
    arguments = {
        "total_bars": 1600,
        "n_blocks": 5,
        "configured_gap": 20,
        "min_fold_bars": 200,
        "warmup_bars": 400,
        "label_lookahead": 2,
    }
    arguments[parameter] = value

    with pytest.raises(InsufficientWalkForwardDataError) as captured:
        build_walk_forward_folds(**arguments)

    assert len(str(captured.value)) <= 500
    assert _HostileProtocols.calls == 0


def test_builder_huge_exact_int_diagnostic_is_bounded_without_stringifying_value() -> None:
    huge = 10**5000
    with pytest.raises(InsufficientWalkForwardDataError) as captured:
        build_walk_forward_folds(
            total_bars=1600,
            n_blocks=huge,
            configured_gap=20,
            min_fold_bars=200,
            warmup_bars=400,
            label_lookahead=2,
        )

    message = str(captured.value)
    assert "bit_length=" in message
    assert len(message) <= 500
