"""Fixed Stage-2 ablation definitions and holdout isolation."""

from __future__ import annotations

from .fusion import (
    ALPHA_H1,
    ALPHA_M15,
    PA_H1,
    PA_M15_CONTEXT,
    PA_M15_PATTERN,
    PA_M5_TIMING,
    default_module_selection,
)
from .models import AblationVariantV1, AblationWindowV1, ModuleSelectionV1


def freeze_ablation_window(
    decision_close_timestamps: tuple[int, ...],
    *,
    holdout_days: int = 7,
) -> AblationWindowV1:
    if type(decision_close_timestamps) is not tuple or any(
        type(value) is not int for value in decision_close_timestamps
    ):
        raise TypeError("decision_close_timestamps must be an exact tuple of integers")
    if not decision_close_timestamps:
        raise ValueError("decision_close_timestamps must not be empty")
    if any(
        decision_close_timestamps[index] <= decision_close_timestamps[index - 1]
        for index in range(1, len(decision_close_timestamps))
    ):
        raise ValueError("decision close timestamps must be strictly increasing")
    if type(holdout_days) is not int or holdout_days < 1:
        raise ValueError("holdout_days must be a positive integer")
    latest = decision_close_timestamps[-1]
    cutoff = latest - holdout_days * 86_400
    preholdout = tuple(value for value in decision_close_timestamps if value <= cutoff)
    holdout = tuple(value for value in decision_close_timestamps if value > cutoff)
    if not preholdout or not holdout:
        raise ValueError("ablation window must contain both preholdout and holdout decisions")
    return AblationWindowV1(
        latest_decision_close=latest,
        cutoff_close=cutoff,
        preholdout_closes=preholdout,
        holdout_closes=holdout,
    )


def _without(
    selection: ModuleSelectionV1,
    *,
    direction: frozenset[str] = frozenset(),
    entry: frozenset[str] = frozenset(),
) -> ModuleSelectionV1:
    return ModuleSelectionV1(
        direction_modules=selection.direction_modules.difference(direction),
        entry_modules=selection.entry_modules.difference(entry),
    )


def default_ablation_variants() -> tuple[AblationVariantV1, ...]:
    default = default_module_selection()
    return (
        AblationVariantV1("default", default),
        AblationVariantV1(
            "without_h1_alpha",
            _without(default, direction=frozenset({ALPHA_H1})),
        ),
        AblationVariantV1(
            "without_h1_pa",
            _without(default, direction=frozenset({PA_H1})),
        ),
        AblationVariantV1(
            "without_m15_alpha",
            _without(
                default,
                direction=frozenset({ALPHA_M15}),
                entry=frozenset({ALPHA_M15}),
            ),
        ),
        AblationVariantV1(
            "without_m15_pa_context",
            _without(default, direction=frozenset({PA_M15_CONTEXT})),
        ),
        AblationVariantV1(
            "without_m15_pa_pattern",
            _without(default, entry=frozenset({PA_M15_PATTERN})),
        ),
        AblationVariantV1(
            "without_m5_pa",
            _without(default, entry=frozenset({PA_M5_TIMING})),
        ),
        AblationVariantV1("with_m5_alpha", default_module_selection(enable_m5_alpha=True)),
    )
