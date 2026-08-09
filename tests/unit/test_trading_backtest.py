from __future__ import annotations

import asyncio
import copy
import json
from dataclasses import replace
from datetime import datetime, timezone

import pytest

from trading_core import (
    APPROVED_SELECTED_PLAN,
    ClosedBarV1,
    DecisionReviewV1,
    InstrumentConstraintsV1,
    OrderPlanCandidateV1,
    TimeframeSeriesV1,
    TradingConfigV1,
    bar_close_timestamp,
)
from trading_core.trading_backtest import (
    CachedDecisionReviewer,
    HistoricalExecutionSimulator,
    ReplaySeriesMarket,
    TradingBacktestRunner,
    canonical_dataset_identity,
    seven_natural_day_window,
    write_backtest_report,
)


def _ts(value: str) -> int:
    return int(datetime.fromisoformat(value).replace(tzinfo=timezone.utc).timestamp())


def _bar(open_ts: int, low: float, high: float, close: float = 100.0) -> ClosedBarV1:
    return ClosedBarV1(
        timestamp=open_ts,
        open=close,
        high=high,
        low=low,
        close=close,
        volume=1.0,
    )


def _plan(
    order_type: str = "market",
    *,
    side: str = "long",
    entry: float = 100.0,
    trigger: float | None = None,
    limit: float | None = None,
) -> OrderPlanCandidateV1:
    return OrderPlanCandidateV1(
        plan_id=f"{side}-{order_type}",
        style="pa_primary",
        side=side,
        family="breakout",
        setup_type="test",
        order_type=order_type,
        entry_price=entry,
        trigger_price=trigger,
        limit_price=limit,
        stop_loss=98.0 if side == "long" else 102.0,
        take_profit_1=102.0 if side == "long" else 98.0,
        take_profit_2=104.0 if side == "long" else 96.0,
        risk_distance=2.0,
        take_profit_1_r=1.0,
        take_profit_2_r=2.0,
        source_trigger_at=1,
        source_policy="test",
        policy_origin="test",
    )


def test_seven_natural_day_window_keeps_weekend_in_calendar() -> None:
    latest = _ts("2026-08-10T12:45:00")
    window = seven_natural_day_window(latest)
    assert window.start_close == _ts("2026-08-04T00:00:00")
    assert window.end_close == latest
    # Saturday and Sunday are part of the requested interval even if MT5 returns no bars.
    assert window.calendar_days == 7


def test_canonical_dataset_identity_is_utc_ordered_and_no_future() -> None:
    end = _ts("2026-08-10T12:45:00")
    series = TimeframeSeriesV1(
        symbol="XAUUSD",
        timeframe="M15",
        tick=0.01,
        bars=(
            _bar(end - 1800, 99, 101),
            _bar(end - 900, 99, 101),
            _bar(end, 99, 101),  # closes after end and must be excluded
        ),
    )
    identity = canonical_dataset_identity(series, end_close=end, requested_warmup=10)
    assert identity["timezone"] == "UTC"
    assert identity["bar_count"] == 2
    assert identity["last_close"] == end
    assert identity["available_warmup"] == 1
    assert len(identity["sha256"]) == 64


class _Reviewer:
    def __init__(self) -> None:
        self.calls = 0

    async def review(self, request):
        self.calls += 1
        return DecisionReviewV1(
            decision_id=request.decision_id,
            verdict="approve",
            selected_plan_id=request.plans[0].plan_id,
            reason_code=APPROVED_SELECTED_PLAN,
            summary_zh="测试批准。",
        )


def test_cached_reviewer_hits_and_rejects_mismatched_entry(tmp_path, decision_request) -> None:
    reviewer = _Reviewer()
    cache_path = tmp_path / "reviews.json"
    identity = {"model": "fake-v1", "review_schema_version": "test-v1"}
    cached = CachedDecisionReviewer(reviewer, cache_path, reviewer_identity=identity)
    first = asyncio.run(cached.review(decision_request))
    second = asyncio.run(cached.review(decision_request))
    assert first == second
    assert reviewer.calls == 1
    assert (cached.hits, cached.misses) == (1, 1)

    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    payload["entries"][decision_request.decision_id]["decision_id"] = "decision-v1-wrong"
    cache_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="decision_id"):
        asyncio.run(
            CachedDecisionReviewer(
                reviewer, cache_path, reviewer_identity=identity
            ).review(decision_request)
        )


def test_cached_reviewer_rejects_identity_mismatch(tmp_path, decision_request) -> None:
    cache_path = tmp_path / "reviews.json"
    first = CachedDecisionReviewer(
        _Reviewer(), cache_path, reviewer_identity={"model": "fake-v1"}
    )
    asyncio.run(first.review(decision_request))
    changed = CachedDecisionReviewer(
        _Reviewer(), cache_path, reviewer_identity={"model": "fake-v2"}
    )
    with pytest.raises(ValueError, match="reviewer identity"):
        asyncio.run(changed.review(decision_request))


@pytest.mark.parametrize(
    "changed_key", ["reviewer_implementation", "runtime_implementation"]
)
def test_cached_reviewer_rejects_concrete_implementation_change(
    tmp_path, decision_request, changed_key
) -> None:
    cache_path = tmp_path / "reviews.json"
    identity = {
        "reviewer_implementation": "package.CodexDecisionReviewerV1",
        "runtime_implementation": "package.OpenAICodexRuntimeV1",
    }
    asyncio.run(
        CachedDecisionReviewer(
            _Reviewer(), cache_path, reviewer_identity=identity
        ).review(decision_request)
    )
    changed = {**identity, changed_key: f"changed.{changed_key}"}
    with pytest.raises(ValueError, match="reviewer identity"):
        asyncio.run(
            CachedDecisionReviewer(
                _Reviewer(), cache_path, reviewer_identity=changed
            ).review(decision_request)
        )


def test_cached_hybrid_non_candidate_never_calls_underlying(tmp_path, decision_request) -> None:
    reviewer = _Reviewer()
    # Rebuild because decision_id binds the complete canonical request.
    from trading_core import OrderPlanGenerationV1, build_decision_review_request

    request = build_decision_review_request(
        decision_request.inputs,
        replace(decision_request.signal, accepted=False, side=None),
        OrderPlanGenerationV1(plans=(), reject_reasons=("entry_threshold",)),
    )
    cached = CachedDecisionReviewer(reviewer, tmp_path / "cache.json")
    asyncio.run(cached.review(request))
    assert reviewer.calls == 0
    assert (cached.hits, cached.misses) == (0, 0)


@pytest.fixture
def decision_request():
    from tests.unit.test_trading_decision_reviewer import _request

    return _request()


def test_market_fill_and_adverse_first_same_m5_bar() -> None:
    sim = HistoricalExecutionSimulator(TradingConfigV1.default(), spread_points=0.0)
    sim.submit(_plan(), decision_close=900)
    assert sim.snapshot()["filled"] == 1  # market fills at decision time
    sim.process_bar(_bar(900, 97.0, 103.0), close_timestamp=1200)
    result = sim.finish()
    assert result["placed"] == 1
    assert result["filled"] == 1
    assert result["sl"] == 1
    assert result["tp1"] == 0
    assert result["basic_r"]["total"] == -1.0


@pytest.mark.parametrize(
    ("side", "bar"),
    [
        ("long", _bar(900, 89.0, 92.0, 90.0)),
        ("short", _bar(900, 108.0, 111.0, 110.0)),
    ],
)
def test_gap_through_stop_exits_at_stated_stop_for_both_sides(side, bar) -> None:
    sim = HistoricalExecutionSimulator(TradingConfigV1.default(), spread_points=0.0)
    sim.submit(_plan(side=side), decision_close=900)
    sim.process_bar(bar, close_timestamp=1200)
    result = sim.finish()
    assert result["sl"] == 1
    assert result["open_at_end"] == 0
    assert result["basic_r"]["total"] == -1.0


@pytest.mark.parametrize(
    ("side", "bar"),
    [
        ("long", _bar(900, 103.0, 106.0, 104.0)),
        ("short", _bar(900, 94.0, 97.0, 96.0)),
    ],
)
def test_gap_through_target_exits_at_stated_target_for_both_sides(side, bar) -> None:
    sim = HistoricalExecutionSimulator(TradingConfigV1.default(), spread_points=0.0)
    sim.submit(_plan(side=side), decision_close=900)
    sim.process_bar(bar, close_timestamp=1200)
    result = sim.finish()
    assert result["tp1"] == 1
    assert result["sl"] == 0
    assert result["open_at_end"] == 0
    assert result["basic_r"]["total"] == 1.0


def test_gap_bar_touching_stop_and_target_remains_adverse_first() -> None:
    sim = HistoricalExecutionSimulator(TradingConfigV1.default(), spread_points=0.0)
    sim.submit(_plan(), decision_close=900)
    sim.process_bar(_bar(900, 89.0, 106.0, 100.0), close_timestamp=1200)
    result = sim.finish()
    assert result["sl"] == 1
    assert result["tp1"] == 0
    assert result["basic_r"]["total"] == -1.0


def test_post_tp1_break_even_gap_is_detected_adverse_first() -> None:
    config = replace(TradingConfigV1.default(), tp1_lots=0.01, tp2_lots=0.01)
    sim = HistoricalExecutionSimulator(config, spread_points=0.2)
    sim.submit(_plan(), decision_close=900)
    sim.process_bar(_bar(900, 100.3, 102.5, 101.0), close_timestamp=1200)
    sim.process_bar(_bar(1200, 95.0, 99.0, 97.0), close_timestamp=1500)
    result = sim.finish()
    assert result["tp1"] == 1
    assert result["sl"] == 1
    assert [row["exit"] for row in result["realized_legs"]] == ["tp1", "sl"]


@pytest.mark.parametrize(
    ("side", "tp1_bar", "tp2_gap_bar"),
    [
        (
            "long",
            _bar(900, 100.1, 103.0, 101.0),
            _bar(1200, 105.0, 107.0, 106.0),
        ),
        (
            "short",
            _bar(900, 97.5, 99.5, 99.0),
            _bar(1200, 92.0, 95.0, 94.0),
        ),
    ],
)
def test_tp2_gap_through_after_tp1_exits_at_stated_target(
    side, tp1_bar, tp2_gap_bar
) -> None:
    config = replace(TradingConfigV1.default(), tp1_lots=0.01, tp2_lots=0.01)
    sim = HistoricalExecutionSimulator(config, spread_points=0.0)
    sim.submit(_plan(side=side), decision_close=900)
    sim.process_bar(tp1_bar, close_timestamp=1200)
    sim.process_bar(tp2_gap_bar, close_timestamp=1500)
    result = sim.finish()
    assert result["tp1"] == 1
    assert result["tp2"] == 1
    assert result["sl"] == 0
    assert result["basic_r"]["total"] == 1.5


@pytest.mark.parametrize(
    ("plan", "bar", "filled"),
    [
        (_plan("limit", entry=99.0, limit=99.0), _bar(900, 98.5, 100.0), True),
        (_plan("stop", entry=101.0, trigger=101.0), _bar(900, 100.0, 101.5), True),
        (
            _plan("stop_limit", entry=100.5, trigger=101.0, limit=100.5),
            _bar(900, 100.0, 101.5),
            True,
        ),
    ],
)
def test_pending_fill_semantics(plan, bar, filled) -> None:
    sim = HistoricalExecutionSimulator(TradingConfigV1.default(), spread_points=0.0)
    sim.submit(plan, decision_close=900)
    sim.process_bar(bar, close_timestamp=1200)
    assert bool(sim.snapshot()["filled"]) is filled


@pytest.mark.parametrize(
    ("plan", "bar"),
    [
        (_plan("limit", side="long", entry=99.0, limit=99.0), _bar(900, 95.0, 98.0, 97.0)),
        (_plan("limit", side="short", entry=101.0, limit=101.0), _bar(900, 102.0, 105.0, 103.0)),
        (_plan("stop", side="long", entry=101.0, trigger=101.0), _bar(900, 102.0, 105.0, 103.0)),
        (_plan("stop", side="short", entry=99.0, trigger=99.0), _bar(900, 95.0, 98.0, 97.0)),
    ],
)
def test_pending_gap_through_fills_by_side(plan, bar) -> None:
    sim = HistoricalExecutionSimulator(TradingConfigV1.default(), spread_points=0.0)
    sim.submit(plan, decision_close=900)
    sim.process_bar(bar, close_timestamp=1200)
    assert sim.snapshot()["filled"] == 1


def test_stop_limit_same_bar_trigger_then_limit_is_deterministic_and_adverse_first() -> None:
    sim = HistoricalExecutionSimulator(TradingConfigV1.default(), spread_points=0.0)
    plan = _plan("stop_limit", side="long", entry=100.5, trigger=101.0, limit=100.5)
    sim.submit(plan, decision_close=900)
    sim.process_bar(_bar(900, 97.0, 102.0), close_timestamp=1200)
    result = sim.finish()
    assert result["filled"] == 1
    assert result["stop_limit_triggered"] == 1
    assert result["sl"] == 1


def test_pending_expires_before_next_m15_and_no_pyramiding_reverse_cooldown() -> None:
    sim = HistoricalExecutionSimulator(TradingConfigV1.default(), spread_points=0.0)
    sim.submit(_plan("limit", entry=90.0, limit=90.0), decision_close=900)
    sim.before_decision(1800)
    assert sim.snapshot()["expired_cancelled"] == 1

    sim.submit(_plan(), decision_close=1800)
    assert sim.submit(_plan(), decision_close=2700) == "no_pyramiding"
    assert sim.submit(_plan(side="short"), decision_close=2700) == "reverse_exit"
    assert sim.submit(_plan(side="short"), decision_close=2700) == "reverse_cooldown"
    assert sim.submit(_plan(side="short"), decision_close=3600) == "placed"


def test_reverse_exit_records_r_and_waits_until_next_m15() -> None:
    sim = HistoricalExecutionSimulator(TradingConfigV1.default(), spread_points=0.0)
    sim.submit(_plan(entry=100.0), decision_close=900)
    assert (
        sim.submit(
            _plan(side="short", entry=99.0),
            decision_close=1800,
            market_price=101.0,
        )
        == "reverse_exit"
    )
    assert sim.snapshot()["realized_legs"][0]["r"] == 0.5
    assert sim.submit(_plan(side="short", entry=101.0), decision_close=1800) == "reverse_cooldown"
    assert sim.submit(_plan(side="short", entry=101.0), decision_close=2700) == "placed"


def test_tp1_moves_tp2_to_cost_buffer_and_accounts_legs() -> None:
    config = replace(TradingConfigV1.default(), tp1_lots=0.01, tp2_lots=0.01)
    sim = HistoricalExecutionSimulator(config, spread_points=0.2)
    sim.submit(_plan(), decision_close=900)
    sim.process_bar(_bar(900, 100.3, 102.5, 101.0), close_timestamp=1200)
    sim.process_bar(_bar(1200, 100.3, 104.5, 101.0), close_timestamp=1500)
    result = sim.finish()
    assert result["tp1"] == 1
    assert result["tp2"] == 1
    assert result["realized_legs"][0]["r"] == 1.0
    assert result["realized_legs"][1]["r"] == 2.0
    assert result["basic_r"]["wins"] == 1
    assert result["basic_r"]["total"] == 1.5


def test_r_contribution_uses_configured_lot_proportions() -> None:
    config = replace(TradingConfigV1.default(), tp1_lots=0.01, tp2_lots=0.03)
    sim = HistoricalExecutionSimulator(config, spread_points=0.0)
    sim.submit(_plan(), decision_close=900)
    sim.process_bar(_bar(900, 100.1, 102.5, 101.0), close_timestamp=1200)
    sim.process_bar(_bar(1200, 100.1, 104.5, 101.0), close_timestamp=1500)
    result = sim.finish()
    assert [row["r_contribution"] for row in result["realized_legs"]] == [0.25, 1.5]
    assert result["basic_r"] == {
        "total": 1.75,
        "average": 1.75,
        "wins": 1,
        "losses": 0,
        "flat": 0,
    }


def test_two_leg_stop_is_one_exit_event_with_two_leg_results() -> None:
    config = replace(TradingConfigV1.default(), tp1_lots=0.01, tp2_lots=0.03)
    sim = HistoricalExecutionSimulator(config, spread_points=0.0)
    sim.submit(_plan(), decision_close=900)
    sim.process_bar(_bar(900, 97.0, 100.5), close_timestamp=1200)
    result = sim.finish()
    assert result["sl"] == 1
    assert len(result["realized_legs"]) == 2
    assert result["basic_r"]["total"] == -1.0


@pytest.mark.parametrize(
    "changes",
    [
        {"take_profit_2": None},
        {"take_profit_2_r": None},
    ],
)
def test_configured_tp2_requires_complete_plan_before_counters_mutate(changes) -> None:
    config = replace(TradingConfigV1.default(), tp1_lots=0.01, tp2_lots=0.02)
    incomplete = replace(_plan(), **changes)
    sim = HistoricalExecutionSimulator(config, spread_points=0.0)
    with pytest.raises(ValueError, match="TP2"):
        sim.submit(incomplete, decision_close=900)
    result = sim.snapshot()
    assert result["placed"] == 0
    assert result["filled"] == 0
    assert sim.position is None and sim.pending is None


def test_missing_tp2_is_valid_when_tp2_volume_is_zero() -> None:
    incomplete = replace(_plan(), take_profit_2=None, take_profit_2_r=None)
    sim = HistoricalExecutionSimulator(TradingConfigV1.default(), spread_points=0.0)
    assert sim.submit(incomplete, decision_close=900) == "placed"
    assert sim.snapshot()["filled"] == 1


def test_invalid_tp2_cannot_mutate_existing_opposite_position() -> None:
    config = replace(TradingConfigV1.default(), tp1_lots=0.01, tp2_lots=0.02)
    sim = HistoricalExecutionSimulator(config, spread_points=0.0)
    sim.submit(_plan(side="long"), decision_close=900)
    snapshot_before = copy.deepcopy(sim.snapshot())
    position_before = sim.position
    pending_before = sim.pending
    cooldown_before = sim.cooldown_until
    invalid = replace(
        _plan(side="short"), take_profit_2=None, take_profit_2_r=None
    )

    with pytest.raises(ValueError, match="TP2"):
        sim.submit(invalid, decision_close=1800, market_price=101.0)

    assert sim.snapshot() == snapshot_before
    assert sim.position is position_before
    assert sim.pending is pending_before
    assert sim.cooldown_until == cooldown_before


def test_invalid_tp2_cannot_expire_or_replace_existing_pending_order() -> None:
    config = replace(TradingConfigV1.default(), tp1_lots=0.01, tp2_lots=0.02)
    sim = HistoricalExecutionSimulator(config, spread_points=0.0)
    sim.submit(_plan("limit", entry=90.0, limit=90.0), decision_close=900)
    snapshot_before = copy.deepcopy(sim.snapshot())
    position_before = sim.position
    pending_before = sim.pending
    cooldown_before = sim.cooldown_until
    invalid = replace(_plan(), take_profit_2=None, take_profit_2_r=None)

    with pytest.raises(ValueError, match="TP2"):
        sim.submit(invalid, decision_close=1800)

    assert sim.snapshot() == snapshot_before
    assert sim.position is position_before
    assert sim.pending is pending_before
    assert sim.cooldown_until == cooldown_before


def test_invalid_tp2_cannot_bypass_validation_via_no_pyramiding() -> None:
    config = replace(TradingConfigV1.default(), tp1_lots=0.01, tp2_lots=0.02)
    sim = HistoricalExecutionSimulator(config, spread_points=0.0)
    sim.submit(_plan(side="long"), decision_close=900)
    snapshot_before = copy.deepcopy(sim.snapshot())
    position_before = sim.position
    invalid = replace(
        _plan(side="long"), take_profit_2=None, take_profit_2_r=None
    )

    with pytest.raises(ValueError, match="TP2"):
        sim.submit(invalid, decision_close=1800)

    assert sim.snapshot() == snapshot_before
    assert sim.position is position_before
    assert sim.pending is None
    assert sim.cooldown_until is None


def test_invalid_tp2_cannot_bypass_validation_via_reverse_cooldown() -> None:
    config = replace(TradingConfigV1.default(), tp1_lots=0.01, tp2_lots=0.02)
    sim = HistoricalExecutionSimulator(config, spread_points=0.0)
    sim.submit(_plan(side="long"), decision_close=900)
    assert (
        sim.submit(_plan(side="short"), decision_close=1800, market_price=101.0)
        == "reverse_exit"
    )
    snapshot_before = copy.deepcopy(sim.snapshot())
    position_before = sim.position
    pending_before = sim.pending
    cooldown_before = sim.cooldown_until
    invalid = replace(
        _plan(side="short"), take_profit_2=None, take_profit_2_r=None
    )

    with pytest.raises(ValueError, match="TP2"):
        sim.submit(invalid, decision_close=1800)

    assert sim.snapshot() == snapshot_before
    assert sim.position is position_before
    assert sim.pending is pending_before
    assert sim.cooldown_until == cooldown_before


def test_realized_trades_keep_same_plan_id_at_distinct_decision_closes() -> None:
    sim = HistoricalExecutionSimulator(TradingConfigV1.default(), spread_points=0.0)
    plan = _plan()
    sim.submit(plan, decision_close=900)
    sim.process_bar(_bar(900, 100.0, 103.0, 101.0), close_timestamp=1200)
    sim.submit(plan, decision_close=1800)
    sim.process_bar(_bar(1800, 100.0, 103.0, 101.0), close_timestamp=2100)
    result = sim.finish()
    assert result["realized_trades"] == [
        {
            "decision_close": 900,
            "plan_id": plan.plan_id,
            "side": "long",
            "r": 1.0,
        },
        {
            "decision_close": 1800,
            "plan_id": plan.plan_id,
            "side": "long",
            "r": 1.0,
        },
    ]
    assert result["basic_r"] == {
        "total": 2.0,
        "average": 1.0,
        "wins": 2,
        "losses": 0,
        "flat": 0,
    }


def test_report_has_required_identity_and_atomic_latest(tmp_path) -> None:
    report = {
        "schema_version": "trading-backtest-report-v1",
        "engine_version": "trading-backtest-v1",
        "data": {"timezone": "UTC"},
        "config": TradingConfigV1.default().to_payload(),
        "strategy_fingerprints": {},
        "resolved_weights": {},
        "modes": {"rules": {}, "rules_codex": {}},
        "assumptions": {"spread": "historical OHLC has no bid/ask"},
    }
    immutable, latest = write_backtest_report(report, tmp_path, end_close=123456)
    assert immutable.exists() and latest.exists()
    assert json.loads(latest.read_text(encoding="utf-8")) == report
    assert not list(tmp_path.glob("*.tmp"))


def test_replay_market_serves_only_prefix_and_each_m15_is_run_once() -> None:
    closes = (900, 1800, 2700)
    m15 = TimeframeSeriesV1(
        symbol="XAUUSD",
        timeframe="M15",
        tick=0.01,
        bars=tuple(_bar(close - 900, 99, 101) for close in closes),
    )
    market = ReplaySeriesMarket(
        {"H1": TimeframeSeriesV1("XAUUSD", "H1", (), 0.01), "M15": m15, "M5": TimeframeSeriesV1("XAUUSD", "M5", (), 0.01)},
        constraints=None,
    )
    market.set_decision_close(1800)
    served = market.fetch_series("XAUUSD", "M15", 99)
    assert [bar_close_timestamp(bar, "M15") for bar in served.bars] == [900, 1800]
    with pytest.raises(RuntimeError, match="already evaluated"):
        market.set_decision_close(1800)


def test_runner_builds_complete_mode_report_with_identity(monkeypatch, tmp_path) -> None:
    required = {
        "checked_m15",
        "raw_trigger_points",
        "raw_pa_trigger_points",
        "raw_factor_trigger_points",
        "qualified_candidate_points",
        "codex",
        "placed",
        "filled",
        "unfilled",
        "expired_cancelled",
        "side_distribution",
        "order_type_distribution",
        "pa_family_distribution",
        "tp1",
        "tp2",
        "sl",
        "reverse_exit",
        "realized_legs",
        "realized_trades",
        "basic_r",
        "open_at_end",
    }
    report = TradingBacktestRunner.empty_report_for_test(
        config=TradingConfigV1.default(), end_close=2700
    )
    assert report["schema_version"] == "trading-backtest-report-v1"
    assert set(report["modes"]) == {"rules", "rules_codex"}
    assert required <= set(report["modes"]["rules"])
    assert {"H1", "M15", "M5"} <= set(report["data"]["series"])
    assert "resolved_weights" in report and "strategy_fingerprints" in report
    assert {
        "sdk_version",
        "model",
        "reasoning_effort",
        "service_tier",
        "review_schema_version",
        "cache_schema_version",
        "prompt_contract_version",
        "reviewer_implementation",
        "runtime_implementation",
    } <= set(report["reviewer_identity"])
    assert report["reviewer_identity"]["reviewer_implementation"].endswith(
        ".CodexDecisionReviewer"
    )
    assert report["reviewer_identity"]["runtime_implementation"].endswith(
        ".OpenAICodexRuntime"
    )
    assert "config_hash" in report["config"]
    assert report["modes"]["rules"]["config"]["mode"] == "rules"
    assert report["modes"]["rules_codex"]["config"]["mode"] == "rules_codex"
    assert (
        report["modes"]["rules"]["config"]["config_hash"]
        != report["modes"]["rules_codex"]["config"]["config_hash"]
    )


def test_runner_resolves_formula_warmup_before_fetch(monkeypatch, tmp_path) -> None:
    from types import SimpleNamespace
    import trading_core.trading_backtest as module

    loaded = []

    def fake_load(path, *, expected_symbol, expected_timeframe):
        loaded.append(expected_timeframe)
        return SimpleNamespace(
            formula_tokens=tuple(range(8)),
            strategy_fingerprint=f"strategy-{expected_timeframe}",
            dataset_fingerprint=f"dataset-{expected_timeframe}",
        )

    monkeypatch.setattr(module, "load_alpha_strategy", fake_load)
    monkeypatch.setattr(module, "formula_warmup_bars", lambda token_count: 2137)
    runner = TradingBacktestRunner(
        market=object(),
        output_dir=tmp_path,
        strategy_resolver=lambda symbol, timeframe: tmp_path / f"{timeframe}.json",
    )
    identities, warmups, frozen = runner._resolve_strategy_inputs(TradingConfigV1.default())
    assert loaded == ["H1", "M15"]
    assert warmups == {"H1": 2137, "M15": 2137, "M5": 384}
    assert identities["H1"]["strategy_fingerprint"] == "strategy-H1"
    assert frozen["H1"].strategy_fingerprint == "strategy-H1"


def test_runner_end_to_end_reuses_one_engine_and_never_exposes_future(monkeypatch, tmp_path) -> None:
    import threading
    from types import SimpleNamespace

    closes = (900, 1800, 2700)
    series = {
        "H1": TimeframeSeriesV1(
            "XAUUSD", "H1", (_bar(-3600, 99, 101),), 0.01
        ),
        "M15": TimeframeSeriesV1(
            "XAUUSD",
            "M15",
            tuple(_bar(close - 900, 99, 101) for close in closes),
            0.01,
        ),
        "M5": TimeframeSeriesV1(
            "XAUUSD",
            "M5",
            tuple(_bar(close - 300, 99, 101) for close in range(300, 2701, 300)),
            0.01,
        ),
    }
    instances = []
    frozen_strategy = SimpleNamespace(strategy_fingerprint="frozen")
    loaded_strategy_ids = []

    class FakeEngine:
        def __init__(self, *, market, reviewer_factory, strategy_loader):
            self.market = market
            self.calls = []
            loaded_strategy_ids.append(id(strategy_loader("XAUUSD", "H1")))
            instances.append(self)

        def evaluate_at(self, config, close):
            visible = self.market.fetch_series("XAUUSD", "M15", 99)
            assert max(bar_close_timestamp(bar, "M15") for bar in visible.bars) == close
            self.calls.append(close)
            return SimpleNamespace(
                review=SimpleNamespace(selected_plan_id=None, verdict="reject", reason_code="reject_entry_quality"),
                plans=(),
                pa_evidence=(),
                alpha_scores=(("H1", 0.0), ("M15", 0.0), ("M5", None)),
                fusion_accepted=False,
            )

    market = SimpleNamespace(
        instrument_constraints=lambda symbol: InstrumentConstraintsV1(tick=0.01)
    )
    runner = TradingBacktestRunner(
        market=market, output_dir=tmp_path, preview_engine_factory=FakeEngine
    )
    monkeypatch.setattr(
        runner,
        "_resolve_strategy_inputs",
        lambda config: (
            {"H1": {"strategy_fingerprint": "frozen"}},
            {"H1": 0, "M15": 0, "M5": 0},
            {"H1": frozen_strategy},
        ),
    )
    monkeypatch.setattr(runner, "_acquire", lambda config, warmups: (seven_natural_day_window(2700), series))
    result = runner.run(
        TradingConfigV1.default(),
        modes=("rules", "rules_codex"),
        stop_event=threading.Event(),
        progress=lambda *args: None,
    )
    assert len(instances) == 2
    assert all(instance.calls == [900, 1800, 2700] for instance in instances)
    assert loaded_strategy_ids == [id(frozen_strategy), id(frozen_strategy)]
    assert result["modes"]["rules"]["checked_m15"] == 3


def test_runner_fails_when_available_warmup_is_short(monkeypatch, tmp_path) -> None:
    import threading
    from types import SimpleNamespace

    series = {
        timeframe: TimeframeSeriesV1(
            "XAUUSD",
            timeframe,
            (_bar(0, 99, 101),),
            0.01,
        )
        for timeframe in ("H1", "M15", "M5")
    }
    runner = TradingBacktestRunner(
        market=SimpleNamespace(
            instrument_constraints=lambda symbol: InstrumentConstraintsV1(tick=0.01)
        ),
        output_dir=tmp_path,
    )
    monkeypatch.setattr(
        runner,
        "_resolve_strategy_inputs",
        lambda config: ({}, {"H1": 384, "M15": 384, "M5": 384}, {}),
    )
    monkeypatch.setattr(
        runner,
        "_acquire",
        lambda config, warmups: (seven_natural_day_window(3600), series),
    )
    with pytest.raises(RuntimeError, match="insufficient historical warmup"):
        runner.run(
            TradingConfigV1.default(),
            modes=("rules",),
            stop_event=threading.Event(),
            progress=lambda *args: None,
        )
