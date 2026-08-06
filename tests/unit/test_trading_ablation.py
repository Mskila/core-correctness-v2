from __future__ import annotations

from trading_core import (
    M5_ALPHA,
    default_ablation_variants,
    freeze_ablation_window,
    resolve_fusion_weights,
)


def test_ablation_window_reserves_latest_seven_days_without_overlap() -> None:
    day = 86_400
    timestamps = tuple(index * day for index in range(12))
    window = freeze_ablation_window(timestamps, holdout_days=7)

    assert window.latest_decision_close == 11 * day
    assert window.cutoff_close == 4 * day
    assert window.preholdout_closes == tuple(index * day for index in range(5))
    assert window.holdout_closes == tuple(index * day for index in range(5, 12))
    assert set(window.preholdout_closes).isdisjoint(window.holdout_closes)


def test_fixed_ablation_variants_do_not_grid_search_thresholds() -> None:
    variants = default_ablation_variants()
    names = [variant.name for variant in variants]

    assert names == [
        "default",
        "without_h1_alpha",
        "without_h1_pa",
        "without_m15_alpha",
        "without_m15_pa_context",
        "without_m15_pa_pattern",
        "without_m5_pa",
        "with_m5_alpha",
    ]
    assert M5_ALPHA not in variants[0].selection.entry_modules
    assert M5_ALPHA in variants[-1].selection.entry_modules
    assert dict(resolve_fusion_weights(variants[-1].selection).entry)[M5_ALPHA] == 0.1
