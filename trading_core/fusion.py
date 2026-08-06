"""Deterministic multi-timeframe Alpha and PA score fusion."""

from __future__ import annotations

from pa_core import StrategyCandidate

from .models import (
    CandidateFreshnessStateV1,
    FusionInputsV1,
    FusionSignalV1,
    FusionWeightsV1,
    ModuleSelectionV1,
    PACandidateOccurrenceV1,
)


ALPHA_H1 = "alpha_h1"
PA_H1 = "pa_h1"
ALPHA_M15 = "alpha_m15"
PA_M15_CONTEXT = "pa_m15_context"
PA_M15_PATTERN = "pa_m15_pattern"
PA_M5_TIMING = "pa_m5_timing"
M5_ALPHA = "alpha_m5"

DIRECTION_THRESHOLD = 0.60
ENTRY_THRESHOLD = 0.65
STRONG_CONTEXT_THRESHOLD = 0.60

_DIRECTION_BASE = (
    (ALPHA_H1, 0.40),
    (PA_H1, 0.25),
    (ALPHA_M15, 0.20),
    (PA_M15_CONTEXT, 0.15),
)
_ENTRY_DEFAULT = (
    (PA_M15_PATTERN, 0.45),
    (PA_M5_TIMING, 0.30),
    (ALPHA_M15, 0.25),
)
_ENTRY_WITH_M5_ALPHA = (
    (PA_M15_PATTERN, 0.405),
    (PA_M5_TIMING, 0.270),
    (ALPHA_M15, 0.225),
    (M5_ALPHA, 0.100),
)


def default_module_selection(*, enable_m5_alpha: bool = False) -> ModuleSelectionV1:
    entry = {PA_M15_PATTERN, PA_M5_TIMING, ALPHA_M15}
    if enable_m5_alpha:
        entry.add(M5_ALPHA)
    return ModuleSelectionV1(
        direction_modules=frozenset(name for name, _ in _DIRECTION_BASE),
        entry_modules=frozenset(entry),
    )


def _normalize(
    base: tuple[tuple[str, float], ...], enabled: frozenset[str], *, group: str
) -> tuple[tuple[str, float], ...]:
    selected = tuple((name, weight) for name, weight in base if name in enabled)
    unknown = enabled.difference(name for name, _ in base)
    if unknown:
        raise ValueError(f"unknown {group} modules: {sorted(unknown)}")
    if not selected:
        raise ValueError(f"at least one {group} modules entry is required")
    total = sum(weight for _, weight in selected)
    return tuple((name, weight / total) for name, weight in selected)


def resolve_fusion_weights(selection: ModuleSelectionV1) -> FusionWeightsV1:
    entry_base = _ENTRY_WITH_M5_ALPHA if M5_ALPHA in selection.entry_modules else _ENTRY_DEFAULT
    return FusionWeightsV1(
        direction=_normalize(
            _DIRECTION_BASE,
            selection.direction_modules,
            group="direction",
        ),
        entry=_normalize(entry_base, selection.entry_modules, group="entry"),
    )


def _candidate_key(candidate: StrategyCandidate) -> str:
    return "|".join(
        (
            candidate.family,
            candidate.direction,
            candidate.setup_type,
            str(candidate.trigger_at),
            candidate.source_policy,
        )
    )


def classify_candidate_freshness(
    candidates: tuple[StrategyCandidate, ...],
    *,
    observed_at: int,
    previous: CandidateFreshnessStateV1,
    emit_new: bool = True,
) -> tuple[tuple[PACandidateOccurrenceV1, ...], CandidateFreshnessStateV1]:
    seen = set(previous.seen_keys)
    occurrences: list[PACandidateOccurrenceV1] = []
    for candidate in candidates:
        key = _candidate_key(candidate)
        is_new = emit_new and key not in seen
        occurrences.append(
            PACandidateOccurrenceV1(
                candidate=candidate,
                observed_at=observed_at,
                is_new=is_new,
            )
        )
        seen.add(key)
    return tuple(occurrences), CandidateFreshnessStateV1(frozenset(seen))


def _candidate_score(occurrences: tuple[PACandidateOccurrenceV1, ...]) -> tuple[float, bool]:
    directions = {
        occurrence.candidate.direction
        for occurrence in occurrences
        if occurrence.is_new
    }
    if not directions:
        return 0.0, False
    if directions == {"long"}:
        return 1.0, False
    if directions == {"short"}:
        return -1.0, False
    return 0.0, True


def _append_once(reasons: list[str], reason: str) -> None:
    if reason not in reasons:
        reasons.append(reason)


def _validate_observation_times(inputs: FusionInputsV1, reasons: list[str]) -> None:
    observations = (
        ("alpha_h1", inputs.h1_alpha, False),
        ("pa_h1", inputs.h1_pa, False),
        ("alpha_m15", inputs.m15_alpha, True),
        ("pa_m15", inputs.m15_pa, True),
        ("pa_m5", inputs.m5_pa, True),
        ("alpha_m5", inputs.m5_alpha, True),
    )
    for name, observation, exact in observations:
        if observation is None:
            continue
        if observation.symbol != inputs.symbol:
            _append_once(reasons, f"identity_mismatch:{name}")
        timestamp = observation.bar_close_timestamp
        if timestamp > inputs.decision_close_timestamp or (
            exact and timestamp != inputs.decision_close_timestamp
        ):
            _append_once(reasons, f"timeframe_misaligned:{name}")


def _timeframe_context(
    values: dict[str, float],
    enabled: frozenset[str],
    names: frozenset[str],
) -> float | None:
    selected = [(name, weight) for name, weight in _DIRECTION_BASE if name in enabled & names]
    if not selected:
        return None
    total = sum(weight for _, weight in selected)
    return sum(values[name] * weight for name, weight in selected) / total


def fuse_signals(
    inputs: FusionInputsV1,
    selection: ModuleSelectionV1 | None = None,
) -> FusionSignalV1:
    active = selection or default_module_selection()
    weights = resolve_fusion_weights(active)
    reasons: list[str] = []
    _validate_observation_times(inputs, reasons)

    observations = {
        ALPHA_H1: inputs.h1_alpha,
        PA_H1: inputs.h1_pa,
        ALPHA_M15: inputs.m15_alpha,
        PA_M15_CONTEXT: inputs.m15_pa,
        PA_M15_PATTERN: inputs.m15_pa,
        PA_M5_TIMING: inputs.m5_pa,
        M5_ALPHA: inputs.m5_alpha,
    }
    for module in (*active.direction_modules, *active.entry_modules):
        if observations[module] is None:
            _append_once(reasons, f"insufficient_data:{module}")

    pa_checks = (
        ("H1", inputs.h1_pa, PA_H1 in active.direction_modules),
        (
            "M15",
            inputs.m15_pa,
            PA_M15_CONTEXT in active.direction_modules
            or PA_M15_PATTERN in active.entry_modules,
        ),
        ("M5", inputs.m5_pa, PA_M5_TIMING in active.entry_modules),
    )
    for timeframe, observation, enabled in pa_checks:
        if not enabled or observation is None:
            continue
        if observation.barbwire:
            _append_once(reasons, f"barbwire:{timeframe}")
        if observation.extreme_range:
            _append_once(reasons, f"extreme_range:{timeframe}")

    direction_values: dict[str, float] = {}
    if inputs.h1_alpha is not None:
        direction_values[ALPHA_H1] = inputs.h1_alpha.position
    if inputs.h1_pa is not None:
        direction_values[PA_H1] = inputs.h1_pa.direction_score
    if inputs.m15_alpha is not None:
        direction_values[ALPHA_M15] = inputs.m15_alpha.position
    if inputs.m15_pa is not None:
        direction_values[PA_M15_CONTEXT] = inputs.m15_pa.direction_score

    direction_score: float | None = None
    side: str | None = None
    if all(name in direction_values for name, _ in weights.direction):
        direction_score = sum(
            direction_values[name] * weight for name, weight in weights.direction
        )
        h1_context = _timeframe_context(
            direction_values,
            active.direction_modules,
            frozenset({ALPHA_H1, PA_H1}),
        )
        m15_context = _timeframe_context(
            direction_values,
            active.direction_modules,
            frozenset({ALPHA_M15, PA_M15_CONTEXT}),
        )
        if (
            h1_context is not None
            and m15_context is not None
            and abs(h1_context) >= STRONG_CONTEXT_THRESHOLD
            and abs(m15_context) >= STRONG_CONTEXT_THRESHOLD
            and h1_context * m15_context < 0.0
        ):
            _append_once(reasons, "strong_timeframe_conflict")
        if abs(direction_score) < DIRECTION_THRESHOLD:
            _append_once(reasons, "direction_below_threshold")
        else:
            side = "long" if direction_score > 0.0 else "short"

    m15_pa_score, m15_ambiguous = (
        _candidate_score(inputs.m15_pa.candidates)
        if inputs.m15_pa is not None
        else (0.0, False)
    )
    m5_pa_score, m5_ambiguous = (
        _candidate_score(inputs.m5_pa.candidates)
        if inputs.m5_pa is not None
        else (0.0, False)
    )
    if PA_M15_PATTERN in active.entry_modules and m15_ambiguous:
        _append_once(reasons, "ambiguous_pa:M15")
    if PA_M5_TIMING in active.entry_modules and m5_ambiguous:
        _append_once(reasons, "ambiguous_pa:M5")

    entry_values: dict[str, float] = {
        PA_M15_PATTERN: m15_pa_score,
        PA_M5_TIMING: m5_pa_score,
    }
    if inputs.m15_alpha is not None:
        entry_values[ALPHA_M15] = inputs.m15_alpha.position
    if inputs.m5_alpha is not None:
        entry_values[M5_ALPHA] = inputs.m5_alpha.position

    entry_score: float | None = None
    if side is not None and all(name in entry_values for name, _ in weights.entry):
        side_sign = 1.0 if side == "long" else -1.0
        entry_score = sum(
            side_sign * entry_values[name] * weight for name, weight in weights.entry
        )
        if entry_score < ENTRY_THRESHOLD:
            _append_once(reasons, "entry_below_threshold")

    selected: tuple[StrategyCandidate, ...] = ()
    if side is not None and inputs.m15_pa is not None:
        selected = tuple(
            occurrence.candidate
            for occurrence in inputs.m15_pa.candidates
            if occurrence.is_new and occurrence.candidate.direction == side
        )
    return FusionSignalV1(
        accepted=not reasons and side is not None and entry_score is not None,
        side=side,
        direction_score=direction_score,
        entry_score=entry_score,
        reject_reasons=tuple(reasons),
        direction_weights=weights.direction,
        entry_weights=weights.entry,
        selected_m15_candidates=selected,
    )
