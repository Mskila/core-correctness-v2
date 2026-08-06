from __future__ import annotations

from dataclasses import replace

import pytest

from pa_core import StrategyCandidate
from trading_core import (
    ALPHA_H1,
    ALPHA_M15,
    M5_ALPHA,
    PA_H1,
    PA_M15_CONTEXT,
    PA_M15_PATTERN,
    PA_M5_TIMING,
    AlphaObservationV1,
    CandidateFreshnessStateV1,
    FusionInputsV1,
    ModuleSelectionV1,
    PACandidateOccurrenceV1,
    PAObservationV1,
    classify_candidate_freshness,
    default_module_selection,
    fuse_signals,
    resolve_fusion_weights,
)


DECISION_CLOSE = 10_800


def _alpha(timeframe: str, position: float) -> AlphaObservationV1:
    return AlphaObservationV1(
        symbol="XAUUSD",
        timeframe=timeframe,
        bar_close_timestamp=DECISION_CLOSE if timeframe != "H1" else 10_000,
        strategy_fingerprint=f"{timeframe.lower()}-fingerprint",
        position=position,
        strength=abs(position),
        factor_value=position,
        bars_used=2_137,
    )


def _candidate(direction: str = "long", *, trigger_at: int = 9_900) -> StrategyCandidate:
    return StrategyCandidate(
        family="trend_continuation",
        direction=direction,
        setup_type="H2" if direction == "long" else "L2",
        trigger_at=trigger_at,
        evidence=("fixture",),
        invalidation_anchor=99.0 if direction == "long" else 101.0,
        target_anchors=(103.0,) if direction == "long" else (97.0,),
        source_policy="formula_candidate_only",
        chase_forbidden=False,
    )


def _pa(
    timeframe: str,
    direction_score: float,
    *,
    candidate_direction: str | None = None,
    barbwire: bool = False,
    new: bool = True,
) -> PAObservationV1:
    occurrences = ()
    if candidate_direction is not None:
        occurrences = (
            PACandidateOccurrenceV1(
                candidate=_candidate(candidate_direction),
                observed_at=DECISION_CLOSE,
                is_new=new,
            ),
        )
    return PAObservationV1(
        symbol="XAUUSD",
        timeframe=timeframe,
        bar_close_timestamp=DECISION_CLOSE if timeframe != "H1" else 10_000,
        direction_score=direction_score,
        barbwire=barbwire,
        extreme_range=False,
        atr=2.0,
        current_open=100.0,
        current_high=100.8,
        current_low=99.4,
        current_close=100.4,
        tick=0.01,
        supports=(99.5, 99.0),
        resistances=(101.5, 103.0),
        candidates=occurrences,
    )


def _inputs(**overrides) -> FusionInputsV1:
    values = {
        "symbol": "XAUUSD",
        "decision_close_timestamp": DECISION_CLOSE,
        "h1_alpha": _alpha("H1", 0.9),
        "h1_pa": _pa("H1", 0.8),
        "m15_alpha": _alpha("M15", 0.8),
        "m15_pa": _pa("M15", 0.8, candidate_direction="long"),
        "m5_pa": _pa("M5", 0.7, candidate_direction="long"),
        "m5_alpha": None,
    }
    values.update(overrides)
    return FusionInputsV1(**values)


def test_default_and_m5_alpha_weights_are_exact() -> None:
    default = resolve_fusion_weights(default_module_selection())
    assert dict(default.direction) == {
        ALPHA_H1: 0.40,
        PA_H1: 0.25,
        ALPHA_M15: 0.20,
        PA_M15_CONTEXT: 0.15,
    }
    assert dict(default.entry) == {
        PA_M15_PATTERN: 0.45,
        PA_M5_TIMING: 0.30,
        ALPHA_M15: 0.25,
    }

    enabled = resolve_fusion_weights(default_module_selection(enable_m5_alpha=True))
    assert dict(enabled.entry) == pytest.approx(
        {
            PA_M15_PATTERN: 0.405,
            PA_M5_TIMING: 0.270,
            ALPHA_M15: 0.225,
            M5_ALPHA: 0.100,
        }
    )


def test_disabled_modules_are_renormalized_only_within_their_group() -> None:
    selection = ModuleSelectionV1(
        direction_modules=frozenset({ALPHA_H1, PA_H1}),
        entry_modules=frozenset({PA_M15_PATTERN, ALPHA_M15}),
    )
    weights = resolve_fusion_weights(selection)

    assert dict(weights.direction) == pytest.approx(
        {ALPHA_H1: 0.40 / 0.65, PA_H1: 0.25 / 0.65}
    )
    assert dict(weights.entry) == pytest.approx(
        {PA_M15_PATTERN: 0.45 / 0.70, ALPHA_M15: 0.25 / 0.70}
    )


def test_matching_direction_and_fresh_pa_candidates_pass_both_thresholds() -> None:
    result = fuse_signals(_inputs())

    assert result.accepted is True
    assert result.side == "long"
    assert result.reject_reasons == ()
    assert result.direction_score == pytest.approx(0.84)
    assert result.entry_score == pytest.approx(0.95)


def test_old_pa_candidate_does_not_retrigger_entry() -> None:
    stale_m15 = _pa("M15", 0.8, candidate_direction="long", new=False)
    stale_m5 = _pa("M5", 0.7, candidate_direction="long", new=False)
    result = fuse_signals(_inputs(m15_pa=stale_m15, m5_pa=stale_m5))

    assert result.accepted is False
    assert result.entry_score == pytest.approx(0.20)
    assert "entry_below_threshold" in result.reject_reasons


def test_strong_h1_m15_conflict_rejects_before_entry() -> None:
    result = fuse_signals(
        _inputs(
            m15_alpha=_alpha("M15", -1.0),
            m15_pa=_pa("M15", -1.0, candidate_direction="short"),
        )
    )

    assert result.accepted is False
    assert "strong_timeframe_conflict" in result.reject_reasons


def test_enabled_barbwire_and_missing_module_fail_closed() -> None:
    noisy = fuse_signals(_inputs(m15_pa=_pa("M15", 0.8, barbwire=True)))
    assert noisy.accepted is False
    assert "barbwire:M15" in noisy.reject_reasons

    missing = fuse_signals(_inputs(h1_alpha=None))
    assert missing.accepted is False
    assert "insufficient_data:alpha_h1" in missing.reject_reasons


def test_direction_and_entry_groups_cannot_be_empty() -> None:
    empty_direction = ModuleSelectionV1(
        direction_modules=frozenset(),
        entry_modules=frozenset({PA_M15_PATTERN}),
    )
    with pytest.raises(ValueError, match="direction modules"):
        resolve_fusion_weights(empty_direction)

    empty_entry = replace(
        default_module_selection(),
        entry_modules=frozenset(),
    )
    with pytest.raises(ValueError, match="entry modules"):
        resolve_fusion_weights(empty_entry)


def test_candidate_freshness_is_stable_and_can_be_seeded_without_emission() -> None:
    first = _candidate(trigger_at=9_000)
    occurrences, state = classify_candidate_freshness(
        (first,),
        observed_at=DECISION_CLOSE,
        previous=CandidateFreshnessStateV1(),
    )
    assert occurrences[0].is_new is True

    repeated, state = classify_candidate_freshness(
        (first,),
        observed_at=DECISION_CLOSE + 900,
        previous=state,
    )
    assert repeated[0].is_new is False

    later, state = classify_candidate_freshness(
        (_candidate(trigger_at=9_900),),
        observed_at=DECISION_CLOSE + 1_800,
        previous=state,
    )
    assert later[0].is_new is True

    seeded, seeded_state = classify_candidate_freshness(
        (_candidate(trigger_at=10_800),),
        observed_at=DECISION_CLOSE + 2_700,
        previous=CandidateFreshnessStateV1(),
        emit_new=False,
    )
    assert seeded[0].is_new is False
    assert seeded_state.seen_keys
