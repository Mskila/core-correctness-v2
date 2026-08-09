from __future__ import annotations

from dataclasses import replace

from pa_core import StrategyCandidate
from trading_core import (
    PA_H1,
    PA_M15_CONTEXT,
    PA_M15_PATTERN,
    PA_M5_TIMING,
    CandidateFreshnessStateV1,
    ClosedBarV1,
    InstrumentConstraintsV1,
    ModuleSelectionV1,
    PACandidateOccurrenceV1,
    PAObservationV1,
    TimeframeSeriesV1,
    TradingConfigV1,
    bar_close_timestamp,
)
import web.trading_preview as preview


DECISION_CLOSE = 3_600 * 500


def _series(timeframe: str, duration: int) -> TimeframeSeriesV1:
    bars = tuple(
        ClosedBarV1(
            timestamp=DECISION_CLOSE - duration * (384 - index),
            open=2399.8,
            high=2400.5,
            low=2399.0,
            close=2400.0,
            volume=100.0,
        )
        for index in range(384)
    )
    return TimeframeSeriesV1(
        symbol="XAUUSD",
        timeframe=timeframe,
        bars=bars,
        tick=0.01,
    )


class _Market:
    def __init__(self) -> None:
        self.series = {
            "H1": _series("H1", 3600),
            "M15": _series("M15", 900),
            "M5": _series("M5", 300),
        }
        self.fetches: list[tuple[str, str, int]] = []

    def fetch_series(self, symbol: str, timeframe: str, count: int):
        self.fetches.append((symbol, timeframe, count))
        return replace(self.series[timeframe], bars=self.series[timeframe].bars[-count:])

    def instrument_constraints(self, symbol: str):
        assert symbol == "XAUUSD"
        return InstrumentConstraintsV1(tick=0.01)


def _candidate(family: str) -> StrategyCandidate:
    return StrategyCandidate(
        family=family,
        direction="long",
        setup_type="breakout_retest" if family == "breakout" else "H2",
        trigger_at=DECISION_CLOSE - 900,
        evidence=(f"family:{family}",),
        invalidation_anchor=2398.0,
        target_anchors=(2404.0, 2406.0),
        source_policy="formula_candidate_only",
        chase_forbidden=False,
    )


def _pa_observation(series: TimeframeSeriesV1) -> PAObservationV1:
    candidates = ()
    if series.timeframe in {"M15", "M5"}:
        candidates = tuple(
            PACandidateOccurrenceV1(
                candidate=_candidate(family),
                observed_at=DECISION_CLOSE,
                is_new=True,
            )
            for family in ("trend_continuation", "breakout")
        )
    return PAObservationV1(
        symbol="XAUUSD",
        timeframe=series.timeframe,
        bar_close_timestamp=bar_close_timestamp(series.bars[-1], series.timeframe),
        direction_score=1.0,
        barbwire=False,
        extreme_range=False,
        atr=2.0,
        current_open=2399.8,
        current_high=2400.5,
        current_low=2399.0,
        current_close=2400.0,
        tick=0.01,
        supports=(2399.0,),
        resistances=(2404.0, 2406.0),
        candidates=candidates,
    )


def test_engine_aligns_closed_timeframes_filters_families_and_never_executes(
    monkeypatch,
) -> None:
    market = _Market()

    def fake_pa(series, *, decision_close_timestamp, previous, emit_new=True):
        assert decision_close_timestamp == DECISION_CLOSE
        assert isinstance(previous, CandidateFreshnessStateV1)
        return _pa_observation(series), previous

    monkeypatch.setattr(preview, "build_pa_observation", fake_pa)
    config = TradingConfigV1(
        mode="rules",
        direction_module_ids=(PA_H1, PA_M15_CONTEXT),
        entry_module_ids=(PA_M15_PATTERN, PA_M5_TIMING),
        pa_family_ids=("breakout",),
        tp1_lots=0.02,
        tp2_lots=0.01,
    )
    assert config.module_selection() == ModuleSelectionV1(
        direction_modules=frozenset({PA_H1, PA_M15_CONTEXT}),
        entry_modules=frozenset({PA_M15_PATTERN, PA_M5_TIMING}),
    )
    engine = preview.TradingPreviewEngine(market)

    decision = engine.evaluate_at(config, DECISION_CLOSE)
    payload = decision.to_payload()

    assert decision.fusion_accepted is True
    assert decision.review.verdict == "approve"
    assert decision.review.selected_plan_id in {plan.plan_id for plan in decision.plans}
    assert decision.plans
    assert {plan.family for plan in decision.plans} == {"breakout"}
    assert payload["close_times"] == {
        "H1": DECISION_CLOSE,
        "M15": DECISION_CLOSE,
        "M5": DECISION_CLOSE,
    }
    assert payload["execution"]["enabled"] is False
    assert all(plan["executable"] is False for plan in payload["plans"])
    assert {timeframe for _, timeframe, _ in market.fetches} == {"H1", "M15", "M5"}
    assert all(count >= 384 for _, _, count in market.fetches)
