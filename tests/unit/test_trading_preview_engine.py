from __future__ import annotations

from dataclasses import replace

from pa_core import StrategyCandidate
from trading_core import (
    PA_H1,
    PA_M15_CONTEXT,
    PA_M15_PATTERN,
    PA_M5_TIMING,
    AlphaObservationV1,
    CandidateFreshnessStateV1,
    ClosedBarV1,
    InstrumentConstraintsV1,
    LoadedAlphaStrategyV1,
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


class _ReplayMarket(_Market):
    def __init__(self) -> None:
        super().__init__()
        self.series = {
            timeframe: replace(
                source,
                bars=(
                    *source.bars,
                    replace(
                        source.bars[-1],
                        timestamp=source.bars[-1].timestamp + duration,
                    ),
                ),
            )
            for timeframe, source, duration in (
                ("H1", self.series["H1"], 3_600),
                ("M15", self.series["M15"], 900),
                ("M5", self.series["M5"], 300),
            )
        }
        self.decision_close = DECISION_CLOSE

    def fetch_series(self, symbol: str, timeframe: str, count: int):
        self.fetches.append((symbol, timeframe, count))
        source = self.series[timeframe]
        eligible = tuple(
            bar
            for bar in source.bars
            if bar_close_timestamp(bar, timeframe) <= self.decision_close
        )
        return replace(source, bars=eligible[-count:])


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


def _strategy(timeframe: str) -> LoadedAlphaStrategyV1:
    return LoadedAlphaStrategyV1(
        symbol="XAUUSD",
        timeframe=timeframe,
        formula_tokens=(1,),
        strategy_fingerprint=f"strategy-{timeframe}",
        dataset_fingerprint=f"dataset-{timeframe}",
        best_score=1.0,
    )


def _alpha_observation(
    strategy: LoadedAlphaStrategyV1,
    series: TimeframeSeriesV1,
) -> AlphaObservationV1:
    position = 0.2 if series.timeframe == "H1" else 0.3
    return AlphaObservationV1(
        symbol=series.symbol,
        timeframe=series.timeframe,
        bar_close_timestamp=bar_close_timestamp(series.bars[-1], series.timeframe),
        strategy_fingerprint=strategy.strategy_fingerprint,
        position=position,
        strength=position,
        factor_value=position,
        bars_used=len(series.bars),
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


def test_engine_accepts_an_instance_alpha_observation_provider() -> None:
    provider = lambda *args, **kwargs: None

    engine = preview.TradingPreviewEngine(
        _Market(),
        alpha_observation_provider=provider,
    )

    assert engine.alpha_observation_provider is provider


def test_engine_without_injection_uses_the_live_single_alpha_path(monkeypatch) -> None:
    calls: list[tuple[str, int]] = []

    def fake_alpha(strategy, series, *, decision_close_timestamp):
        calls.append((series.timeframe, decision_close_timestamp))
        return _alpha_observation(strategy, series)

    def fake_pa(series, *, decision_close_timestamp, previous, emit_new=True):
        return replace(_pa_observation(series), candidates=()), previous

    monkeypatch.setattr(preview, "evaluate_alpha_observation", fake_alpha)
    monkeypatch.setattr(preview, "build_pa_observation", fake_pa)
    engine = preview.TradingPreviewEngine(
        _Market(),
        strategy_loader=lambda symbol, timeframe: _strategy(timeframe),
    )

    engine.evaluate_at(TradingConfigV1.default(), DECISION_CLOSE)

    assert calls == [("H1", DECISION_CLOSE), ("M15", DECISION_CLOSE)]


def test_injected_alpha_matches_legacy_across_decisions_and_future_tail(
    monkeypatch,
) -> None:
    decisions = (DECISION_CLOSE - 900, DECISION_CLOSE)

    def fake_alpha(strategy, series, *, decision_close_timestamp):
        assert all(
            bar_close_timestamp(bar, series.timeframe) <= decision_close_timestamp
            for bar in series.bars
        )
        return _alpha_observation(strategy, series)

    def fake_pa(series, *, decision_close_timestamp, previous, emit_new=True):
        observation = replace(_pa_observation(series), candidates=())
        if series.timeframe == "H1":
            return observation, previous
        key = f"{series.timeframe}:candidate"
        occurrence = PACandidateOccurrenceV1(
            candidate=replace(
                _candidate("trend_continuation"),
                trigger_at=decision_close_timestamp - 900,
            ),
            observed_at=decision_close_timestamp,
            is_new=key not in previous.seen_keys,
        )
        return (
            replace(observation, candidates=(occurrence,)),
            CandidateFreshnessStateV1(previous.seen_keys | {key}),
        )

    monkeypatch.setattr(preview, "evaluate_alpha_observation", fake_alpha)
    monkeypatch.setattr(preview, "build_pa_observation", fake_pa)
    monkeypatch.setattr(preview.time, "time", lambda: 1.0)
    legacy_market = _ReplayMarket()
    injected_market = _ReplayMarket()
    cached: dict[tuple[str, int], AlphaObservationV1] = {}
    for timeframe in ("H1", "M15"):
        strategy = _strategy(timeframe)
        for decision_close in decisions:
            injected_market.decision_close = decision_close
            visible = injected_market.fetch_series("XAUUSD", timeframe, 384)
            cached[(timeframe, decision_close)] = fake_alpha(
                strategy,
                visible,
                decision_close_timestamp=decision_close,
            )

    def provider(strategy, series, *, decision_close_timestamp):
        return cached[(series.timeframe, decision_close_timestamp)]

    legacy = preview.TradingPreviewEngine(
        legacy_market,
        strategy_loader=lambda symbol, timeframe: _strategy(timeframe),
    )
    injected = preview.TradingPreviewEngine(
        injected_market,
        strategy_loader=lambda symbol, timeframe: _strategy(timeframe),
        alpha_observation_provider=provider,
    )
    legacy_decisions = []
    injected_decisions = []
    for decision_close in decisions:
        legacy_market.decision_close = decision_close
        injected_market.decision_close = decision_close
        legacy_decisions.append(
            legacy.evaluate_at(TradingConfigV1.default(), decision_close)
        )
        injected_decisions.append(
            injected.evaluate_at(TradingConfigV1.default(), decision_close)
        )

    assert injected_decisions == legacy_decisions
    assert any("existing" in item for item in injected_decisions[-1].pa_evidence)
    assert all(
        observation.bar_close_timestamp <= decision_close
        for (_timeframe, decision_close), observation in cached.items()
    )


def test_injected_alpha_preserves_missing_and_failed_warning_behavior(
    monkeypatch,
) -> None:
    def strategy_loader(symbol, timeframe):
        if timeframe == "M15":
            raise ValueError("missing strategy")
        return _strategy(timeframe)

    def failed_alpha(strategy, series, *, decision_close_timestamp):
        raise ValueError("alpha failed")

    def fake_pa(series, *, decision_close_timestamp, previous, emit_new=True):
        return replace(_pa_observation(series), candidates=()), previous

    monkeypatch.setattr(preview, "evaluate_alpha_observation", failed_alpha)
    monkeypatch.setattr(preview, "build_pa_observation", fake_pa)
    monkeypatch.setattr(preview.time, "time", lambda: 1.0)
    legacy = preview.TradingPreviewEngine(_Market(), strategy_loader=strategy_loader)
    injected = preview.TradingPreviewEngine(
        _Market(),
        strategy_loader=strategy_loader,
        alpha_observation_provider=failed_alpha,
    )

    legacy_decision = legacy.evaluate_at(TradingConfigV1.default(), DECISION_CLOSE)
    injected_decision = injected.evaluate_at(TradingConfigV1.default(), DECISION_CLOSE)

    assert injected_decision == legacy_decision
    assert injected_decision.input_warnings == (
        "alpha_m15:missing strategy",
        "alpha_h1:alpha failed",
    )
