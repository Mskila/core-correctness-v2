from __future__ import annotations

from dataclasses import replace

from pa_core import (
    AlwaysInState,
    BarbwireState,
    BreakoutEvent,
    ChannelState,
    DirectionState,
    EnvironmentState,
    HLState,
    HLTrigger,
    MeasuredMove,
    PAStrategyDetector,
    PatternCandidate,
    RangeState,
    StructureSnapshot,
    SupportResistance,
)
from tests.support.pa_fixtures import BASE_TS


def base_snapshot() -> StructureSnapshot:
    return StructureSnapshot(
        as_of=BASE_TS,
        range_state=RangeState(
            high=110.0,
            low=98.0,
            width_atr=6.0,
            price_position=0.25,
            zone="lower_third",
            lookback_bars=40,
            trading_range=False,
            extreme=False,
        ),
        swing_structure="HH_HL",
        support_resistance=SupportResistance(
            supports=(99.0,), resistances=(110.0,), fallback=False
        ),
        direction=DirectionState(
            direction="bullish",
            score=4,
            components=(1, 1, 1, 1, 0),
            strength=0.8,
        ),
        always_in=AlwaysInState(
            state="AIL",
            strength="strong",
            source_window="near",
            above_ratio=0.9,
            below_ratio=0.1,
        ),
        channel=ChannelState(
            label="normal_channel",
            confirmed=True,
            group_count=3,
            pullback_ratio=0.4,
            parallel_error=0.1,
            max_residual_atr=0.2,
            direction="bullish",
        ),
        barbwire=BarbwireState(score=0.2, candidate=False, components=()),
        breakout_events=(),
        hl_state=HLState(
            triggers=(HLTrigger("H2", BASE_TS, BASE_TS - 900),),
            candidate="H2",
            wedge_check_required=False,
        ),
        environment=EnvironmentState(
            labels=("normal_channel",),
            spike=False,
            micro_channel=False,
            climax_triggered=False,
        ),
        patterns=(),
        measured_moves=(
            MeasuredMove("range_up", 112.0, 6.0, BASE_TS - 36000, BASE_TS, True),
            MeasuredMove("range_down", 96.0, 6.0, BASE_TS - 36000, BASE_TS, True),
        ),
    )


def families(snapshot: StructureSnapshot) -> set[str]:
    return {candidate.family for candidate in PAStrategyDetector(snapshot).detect()}


def test_trend_continuation_candidate_contains_only_structure_anchors() -> None:
    candidates = PAStrategyDetector(base_snapshot()).detect()
    candidate = next(item for item in candidates if item.family == "trend_continuation")

    assert candidate.direction == "long"
    assert candidate.setup_type == "H2"
    assert candidate.invalidation_anchor == 99.0
    assert candidate.target_anchors == (110.0, 112.0)
    assert candidate.executable is False


def test_breakout_candidate_requires_follow_and_retest_or_failed_failure() -> None:
    event = BreakoutEvent(
        direction="up",
        level=110.0,
        breakout_at=BASE_TS,
        follow_through=True,
        retest_at=BASE_TS + 900,
    )
    snapshot = replace(base_snapshot(), breakout_events=(event,))
    assert "breakout" in families(snapshot)

    no_follow = replace(event, follow_through=False)
    assert "breakout" not in families(replace(snapshot, breakout_events=(no_follow,)))


def test_reversal_candidate_is_diagnostic_only_and_requires_all_gates() -> None:
    mtr = PatternCandidate(
        pattern="mtr",
        direction="bearish",
        confirmed_at=BASE_TS,
        evidence=("trend", "trendline_break", "recovery_failure", "extreme_retest"),
        diagnostic_only=True,
    )
    double_top = PatternCandidate(
        pattern="double_top",
        direction="bearish",
        confirmed_at=BASE_TS,
        evidence=("second_test",),
        follow_through=True,
        diagnostic_only=True,
    )
    snapshot = replace(base_snapshot(), patterns=(mtr, double_top))
    candidate = next(
        item for item in PAStrategyDetector(snapshot).detect() if item.family == "reversal"
    )

    assert candidate.direction == "short"
    assert candidate.source_policy == "diagnostic_only"
    assert candidate.executable is False

    assert "reversal" not in families(replace(snapshot, patterns=(double_top,)))
    assert "reversal" not in families(
        replace(snapshot, patterns=(mtr, replace(double_top, follow_through=False)))
    )


def test_range_candidate_needs_edge_second_entry_and_non_barbwire() -> None:
    snapshot = replace(
        base_snapshot(),
        range_state=replace(
            base_snapshot().range_state,
            price_position=0.8,
            zone="upper_third",
            trading_range=True,
        ),
        direction=DirectionState("neutral", 0, (0, 0, 0, 0, 0), 0.0),
        always_in=AlwaysInState("neutral", "none", "none", 0.5, 0.5),
        channel=ChannelState("trading_range", False, 0, None, None, None, None),
        hl_state=HLState(
            triggers=(HLTrigger("L2", BASE_TS, BASE_TS - 900),),
            candidate="L2",
            wedge_check_required=False,
        ),
        environment=EnvironmentState(("trading_range",), False, False, False),
    )
    candidate = next(
        item for item in PAStrategyDetector(snapshot).detect() if item.family == "range"
    )
    assert candidate.direction == "short"

    middle = replace(snapshot, range_state=replace(snapshot.range_state, zone="middle_third"))
    assert "range" not in families(middle)

    barbwire = replace(snapshot, barbwire=BarbwireState(0.6, True, ("overlap", "doji")))
    assert PAStrategyDetector(barbwire).detect() == ()


def test_detector_never_creates_order_prices_or_trade_permission() -> None:
    for candidate in PAStrategyDetector(base_snapshot()).detect():
        assert candidate.executable is False
        assert not hasattr(candidate, "entry_price")
        assert not hasattr(candidate, "stop_loss")
        assert not hasattr(candidate, "volume")
