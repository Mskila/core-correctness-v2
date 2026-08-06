# SPDX-License-Identifier: AGPL-3.0-or-later
"""Read-only PA strategy-family candidate detector.

Strategy-family semantics are derived from the PA_Agent playbooks fixed at
d92ecd827fe671a589b7fdfdbba41e5e98081d87 (AGPL-3.0-or-later), Copyright
(C) 2026 PA Agent Contributors.  This module deliberately emits no order,
entry, stop, volume, or execution permission; those belong to later stages.
"""

from __future__ import annotations

from .models import FormulaResult, StrategyCandidate, StructureSnapshot


def _dedupe(values: list[float]) -> tuple[float, ...]:
    output: list[float] = []
    seen: set[float] = set()
    for value in values:
        key = round(float(value), 8)
        if key not in seen:
            seen.add(key)
            output.append(float(value))
    return tuple(output)


class PAStrategyDetector:
    """Apply the four Stage-1A strategy gates to a structure snapshot."""

    def __init__(self, snapshot: StructureSnapshot):
        if type(snapshot) is not StructureSnapshot:
            raise TypeError("snapshot must be an exact StructureSnapshot")
        self.snapshot = snapshot

    def _anchors(self, direction: str) -> tuple[float, tuple[float, ...]] | None:
        levels = self.snapshot.support_resistance
        if direction == "long":
            if not levels.supports:
                return None
            targets = [*levels.resistances]
            targets.extend(
                move.target_price
                for move in self.snapshot.measured_moves
                if move.completed and move.kind.endswith("up")
            )
            invalidation = levels.supports[0]
        else:
            if not levels.resistances:
                return None
            targets = [*levels.supports]
            targets.extend(
                move.target_price
                for move in self.snapshot.measured_moves
                if move.completed and move.kind.endswith("down")
            )
            invalidation = levels.resistances[0]
        deduped = _dedupe(targets)
        return (invalidation, deduped) if deduped else None

    def _trend_candidate(self) -> StrategyCandidate | None:
        direction = self.snapshot.direction.direction
        always = self.snapshot.always_in.state
        if direction == "bullish" and always != "AIS":
            side = "long"
            hl_prefix = "H"
        elif direction == "bearish" and always != "AIL":
            side = "short"
            hl_prefix = "L"
        else:
            return None
        allowed_environments = {
            "spike_candidate",
            "standard_spike",
            "micro_channel",
            "tight_channel",
            "normal_channel",
            "broad_channel",
        }
        if not (
            self.snapshot.environment.spike
            or allowed_environments.intersection(self.snapshot.environment.labels)
        ):
            return None
        setup = None
        if self.snapshot.hl_state.candidate.startswith(hl_prefix):
            setup = self.snapshot.hl_state.candidate
        else:
            accepted = {
                "wedge_pullback",
                "ema_pullback",
                "failed_countertrend_breakout",
            }
            setup = next(
                (
                    pattern.pattern
                    for pattern in self.snapshot.patterns
                    if pattern.pattern in accepted
                ),
                None,
            )
        if setup is None:
            return None
        if (
            self.snapshot.range_state.trading_range
            and self.snapshot.range_state.zone == "middle_third"
        ):
            return None
        anchors = self._anchors(side)
        if anchors is None:
            return None
        invalidation, targets = anchors
        return StrategyCandidate(
            family="trend_continuation",
            direction=side,
            setup_type=setup,
            trigger_at=self.snapshot.as_of,
            evidence=(
                f"direction:{direction}",
                f"always_in:{always}",
                f"environment:{self.snapshot.environment.labels}",
                f"setup:{setup}",
            ),
            invalidation_anchor=invalidation,
            target_anchors=targets,
            source_policy="formula_candidate_only",
            chase_forbidden=self.snapshot.environment.climax_triggered,
        )

    def _breakout_candidates(self) -> tuple[StrategyCandidate, ...]:
        direction = self.snapshot.direction.direction
        output: list[StrategyCandidate] = []
        for event in self.snapshot.breakout_events:
            side = "long" if event.direction == "up" else "short"
            if (side == "long" and direction == "bearish") or (
                side == "short" and direction == "bullish"
            ):
                continue
            if not event.follow_through:
                continue
            if event.retest_at is None and event.failed_failure_at is None:
                continue
            if self.snapshot.range_state.trading_range and event.failed_failure_at is None:
                continue
            anchors = self._anchors(side)
            if anchors is None:
                continue
            invalidation, targets = anchors
            setup = "failed_failure" if event.failed_failure_at is not None else "breakout_retest"
            trigger_at = event.failed_failure_at or event.retest_at or event.breakout_at
            output.append(
                StrategyCandidate(
                    family="breakout",
                    direction=side,
                    setup_type=setup,
                    trigger_at=trigger_at,
                    evidence=(
                        f"level:{event.level}",
                        f"breakout_at:{event.breakout_at}",
                        f"follow:{event.follow_through}",
                    ),
                    invalidation_anchor=invalidation,
                    target_anchors=targets,
                    source_policy="formula_candidate_only",
                    chase_forbidden=self.snapshot.environment.climax_triggered,
                )
            )
        return tuple(output)

    def _reversal_candidate(self) -> StrategyCandidate | None:
        mtr = next(
            (pattern for pattern in self.snapshot.patterns if pattern.pattern == "mtr"), None
        )
        if mtr is None:
            return None
        confirmations = {
            "wedge_reversal",
            "double_top",
            "double_bottom",
            "failed_final_flag",
        }
        confirmation = next(
            (
                pattern
                for pattern in self.snapshot.patterns
                if pattern.pattern in confirmations
                and pattern.direction == mtr.direction
                and pattern.follow_through
            ),
            None,
        )
        if confirmation is None:
            return None
        side = "long" if mtr.direction == "bullish" else "short"
        anchors = self._anchors(side)
        if anchors is None:
            return None
        invalidation, targets = anchors
        return StrategyCandidate(
            family="reversal",
            direction=side,
            setup_type=confirmation.pattern,
            trigger_at=max(mtr.confirmed_at, confirmation.confirmed_at),
            evidence=(*mtr.evidence, *confirmation.evidence),
            invalidation_anchor=invalidation,
            target_anchors=targets,
            source_policy="diagnostic_only",
            chase_forbidden=False,
        )

    def _range_candidate(self) -> StrategyCandidate | None:
        state = self.snapshot.range_state
        if not state.trading_range or state.extreme:
            return None
        if state.zone == "lower_third":
            side = "long"
            required = "H2"
            failure_direction = "down"
        elif state.zone == "upper_third":
            side = "short"
            required = "L2"
            failure_direction = "up"
        else:
            return None
        second_entry = any(trigger.label == required for trigger in self.snapshot.hl_state.triggers)
        failed_break = any(
            event.direction == failure_direction and event.failure_at is not None
            for event in self.snapshot.breakout_events
        )
        if not second_entry and not failed_break:
            return None
        anchors = self._anchors(side)
        if anchors is None:
            return None
        invalidation, targets = anchors
        return StrategyCandidate(
            family="range",
            direction=side,
            setup_type=required if second_entry else "failed_breakout",
            trigger_at=self.snapshot.as_of,
            evidence=(f"zone:{state.zone}", f"second_entry:{second_entry}"),
            invalidation_anchor=invalidation,
            target_anchors=targets,
            source_policy="formula_candidate_only",
            chase_forbidden=False,
        )

    def detect(self) -> tuple[StrategyCandidate, ...]:
        if self.snapshot.barbwire.candidate or self.snapshot.range_state.extreme:
            return ()
        output: list[StrategyCandidate] = []
        trend = self._trend_candidate()
        if trend is not None:
            output.append(trend)
        output.extend(self._breakout_candidates())
        reversal = self._reversal_candidate()
        if reversal is not None:
            output.append(reversal)
        range_candidate = self._range_candidate()
        if range_candidate is not None:
            output.append(range_candidate)
        return tuple(output)

    def detect_result(self) -> FormulaResult:
        source_start = min(
            (result.source_start for result in self.snapshot.results),
            default=self.snapshot.as_of,
        )
        return FormulaResult(
            formula_id="strategy.pa_candidates",
            value=self.detect(),
            valid=True,
            as_of=self.snapshot.as_of,
            source_start=source_start,
            source_end=self.snapshot.as_of,
            confirmed_at=self.snapshot.as_of,
            invalid_reason=None,
            provenance_class="D1",
        )
