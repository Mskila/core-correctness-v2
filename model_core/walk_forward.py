"""Strict expanding-window walk-forward splits for V2 training."""

from __future__ import annotations

from dataclasses import dataclass

from model_core.features import MAX_FEATURE_LOOKBACK
from model_core.ops import MAX_OPERATOR_LOOKBACK
from model_core.semantics import InsufficientWalkForwardDataError


MIN_SCORABLE_FOLD_OBSERVATIONS = 200
MIN_EXECUTION_SEGMENT_OBSERVATIONS = 2
MAX_OPERATIONAL_INTEGER = (1 << 63) - 1


@dataclass(frozen=True)
class WalkForwardFold:
    fold_index: int
    train_start: int
    train_end: int
    val_start: int
    val_end: int
    effective_gap: int


def _require_non_negative_int(name: str, value: int) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    if value > MAX_OPERATIONAL_INTEGER:
        raise ValueError(f"{name} exceeds the operational integer limit")


def _require_positive_int(name: str, value: int) -> None:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    if value > MAX_OPERATIONAL_INTEGER:
        raise ValueError(f"{name} exceeds the operational integer limit")


def _require_scorable_fold_size(name: str, value: int) -> None:
    if (
        type(value) is not int
        or value < MIN_SCORABLE_FOLD_OBSERVATIONS
    ):
        raise ValueError(
            f"{name} must be an integer >= {MIN_SCORABLE_FOLD_OBSERVATIONS}"
        )
    if value > MAX_OPERATIONAL_INTEGER:
        raise ValueError(f"{name} exceeds the operational integer limit")


def required_training_bars(
    *,
    warmup_bars: int,
    label_lookahead: int,
    n_blocks: int,
    min_fold_bars: int,
    configured_gap: int,
) -> int:
    """Return the minimum series length without shrinking any isolation gap."""
    _require_non_negative_int("warmup_bars", warmup_bars)
    _require_non_negative_int("label_lookahead", label_lookahead)
    _require_positive_int("n_blocks", n_blocks)
    _require_scorable_fold_size("min_fold_bars", min_fold_bars)
    _require_non_negative_int("configured_gap", configured_gap)
    effective_gap = max(configured_gap, label_lookahead)
    return (
        warmup_bars
        + label_lookahead
        + n_blocks * min_fold_bars
        + (n_blocks - 1) * effective_gap
    )


def formula_warmup_bars(formula_length: int) -> int:
    """Conservative raw-history requirement for a formula of the given length."""
    if type(formula_length) is not int or formula_length < 1:
        raise ValueError("formula_length must be >= 1")
    if formula_length > MAX_OPERATIONAL_INTEGER:
        raise ValueError("formula_length exceeds the operational integer limit")
    return MAX_FEATURE_LOOKBACK + formula_length * max(0, MAX_OPERATOR_LOOKBACK - 1)


def _safe_diagnostic_value(value: object, *, limit: int = 80) -> str:
    """Render bounded diagnostics without invoking caller-defined protocols."""
    value_type = type(value)
    if value_type is int:
        bits = int.bit_length(value)
        if bits > 256:
            return f"<int bit_length={bits}>"
        return repr(value)
    if value_type in {bool, float, str, type(None)}:
        rendered = repr(value)
        if len(rendered) <= limit:
            return rendered
        return rendered[: limit - 3] + "..."
    return f"<object type={value_type.__name__}>"


def _invalid_build_configuration_error(
    *,
    parameter: str,
    value: object,
    reason: str,
    total_bars: object,
    n_blocks: object,
    configured_gap: object,
    min_fold_bars: object,
    warmup_bars: object,
    label_lookahead: object,
) -> InsufficientWalkForwardDataError:
    try:
        required: int | str = required_training_bars(
            warmup_bars=warmup_bars,
            label_lookahead=label_lookahead,
            n_blocks=n_blocks,
            min_fold_bars=min_fold_bars,
            configured_gap=configured_gap,
        )
    except (TypeError, ValueError):
        required = "unavailable"
    return InsufficientWalkForwardDataError(
        "invalid walk-forward configuration: "
        f"parameter={parameter} value={_safe_diagnostic_value(value)} reason={reason}; "
        f"required={_safe_diagnostic_value(required)} "
        f"actual={_safe_diagnostic_value(total_bars)} "
        f"warmup={_safe_diagnostic_value(warmup_bars)} "
        f"gap={_safe_diagnostic_value(configured_gap)} "
        f"blocks={_safe_diagnostic_value(n_blocks)} "
        f"min_fold_bars={_safe_diagnostic_value(min_fold_bars)}"
    )


def build_walk_forward_folds(
    *,
    total_bars: int,
    n_blocks: int,
    configured_gap: int,
    min_fold_bars: int,
    warmup_bars: int,
    label_lookahead: int,
) -> list[WalkForwardFold]:
    """Build strict expanding-window folds after excluding warmup and label tails."""
    validations = (
        ("total_bars", total_bars, _require_non_negative_int),
        ("warmup_bars", warmup_bars, _require_non_negative_int),
        ("label_lookahead", label_lookahead, _require_non_negative_int),
        ("min_fold_bars", min_fold_bars, _require_scorable_fold_size),
        ("configured_gap", configured_gap, _require_non_negative_int),
    )
    for parameter, value, validator in validations:
        try:
            validator(parameter, value)
        except ValueError as exc:
            raise _invalid_build_configuration_error(
                parameter=parameter,
                value=value,
                reason=str(exc),
                total_bars=total_bars,
                n_blocks=n_blocks,
                configured_gap=configured_gap,
                min_fold_bars=min_fold_bars,
                warmup_bars=warmup_bars,
                label_lookahead=label_lookahead,
            ) from exc
    if type(n_blocks) is not int:
        reason = "n_blocks must be an integer"
        raise _invalid_build_configuration_error(
            parameter="n_blocks",
            value=n_blocks,
            reason=reason,
            total_bars=total_bars,
            n_blocks=n_blocks,
            configured_gap=configured_gap,
            min_fold_bars=min_fold_bars,
            warmup_bars=warmup_bars,
            label_lookahead=label_lookahead,
        )
    if n_blocks > MAX_OPERATIONAL_INTEGER:
        raise _invalid_build_configuration_error(
            parameter="n_blocks",
            value=n_blocks,
            reason="n_blocks exceeds the operational integer limit",
            total_bars=total_bars,
            n_blocks=n_blocks,
            configured_gap=configured_gap,
            min_fold_bars=min_fold_bars,
            warmup_bars=warmup_bars,
            label_lookahead=label_lookahead,
        )
    effective_gap = max(configured_gap, label_lookahead)
    if n_blocks < 2:
        minimum_valid_blocks = 2
        required = (
            warmup_bars
            + label_lookahead
            + minimum_valid_blocks * min_fold_bars
            + (minimum_valid_blocks - 1) * effective_gap
        )
        raise InsufficientWalkForwardDataError(
            "insufficient walk-forward data: "
            f"required={_safe_diagnostic_value(required)} "
            f"actual={_safe_diagnostic_value(total_bars)} "
            f"warmup={_safe_diagnostic_value(warmup_bars)} "
            f"gap={_safe_diagnostic_value(effective_gap)} "
            f"blocks={_safe_diagnostic_value(n_blocks)} "
            f"min_fold_bars={_safe_diagnostic_value(min_fold_bars)} "
            f"parameter=n_blocks value={_safe_diagnostic_value(n_blocks)} "
            "reason=at least two blocks are required"
        )
    required = required_training_bars(
        warmup_bars=warmup_bars,
        label_lookahead=label_lookahead,
        n_blocks=n_blocks,
        min_fold_bars=min_fold_bars,
        configured_gap=configured_gap,
    )
    eligible_bars = total_bars - warmup_bars - label_lookahead
    usable_bars = eligible_bars - (n_blocks - 1) * effective_gap
    block_bars = usable_bars // n_blocks

    if block_bars < min_fold_bars:
        raise InsufficientWalkForwardDataError(
            "insufficient walk-forward data: "
            f"required={_safe_diagnostic_value(required)} "
            f"actual={_safe_diagnostic_value(total_bars)} "
            f"warmup={_safe_diagnostic_value(warmup_bars)} "
            f"gap={_safe_diagnostic_value(effective_gap)} "
            f"blocks={_safe_diagnostic_value(n_blocks)} "
            f"min_fold_bars={_safe_diagnostic_value(min_fold_bars)}"
        )

    folds: list[WalkForwardFold] = []
    train_start = warmup_bars
    first_block_end = train_start + block_bars
    for validation_block in range(1, n_blocks):
        train_end = first_block_end + (validation_block - 1) * (
            effective_gap + block_bars
        )
        val_start = train_end + effective_gap
        val_end = val_start + block_bars
        folds.append(
            WalkForwardFold(
                fold_index=validation_block - 1,
                train_start=train_start,
                train_end=train_end,
                val_start=val_start,
                val_end=val_end,
                effective_gap=effective_gap,
            )
        )
    return folds
