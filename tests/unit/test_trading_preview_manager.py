from __future__ import annotations

from dataclasses import replace

from trading_core import DecisionReviewV1, TradeDecisionV1, TradingConfigV1
from web.trading_preview import TradingPreviewManager


def _decision(config: TradingConfigV1, close: int) -> TradeDecisionV1:
    decision_id = f"decision-{close}-{config.config_hash[:8]}"
    return TradeDecisionV1(
        decision_id=decision_id,
        symbol="XAUUSD",
        decision_close_timestamp=close,
        h1_close_timestamp=close - close % 3600,
        m15_close_timestamp=close,
        m5_close_timestamp=close,
        config=config,
        alpha_scores=(("H1", None), ("M15", None), ("M5", None)),
        pa_scores=(("H1", 0.7), ("M15", 0.8), ("M5", 0.9)),
        pa_evidence=(),
        input_warnings=(),
        direction_score=0.75,
        entry_score=0.8,
        side=None,
        fusion_accepted=False,
        fusion_reject_reasons=("no_fresh_m15_pa_candidate",),
        plan_reject_reasons=("fusion_rejected",),
        plans=(),
        review=DecisionReviewV1(
            decision_id=decision_id,
            verdict="reject",
            selected_plan_id=None,
            reason_code="reject_entry_quality",
            summary_zh="当前没有合格候选，本周期仅展示观察结果。",
        ),
        final_action="preview_rejected",
        created_at=float(close),
    )


class _FakeEngine:
    def __init__(self) -> None:
        self.close = 900
        self.evaluated: list[tuple[TradingConfigV1, int]] = []
        self.failure: Exception | None = None

    def latest_m15_close(self, symbol: str) -> int:
        assert symbol == "XAUUSD"
        return self.close

    def evaluate_at(self, config: TradingConfigV1, close: int) -> TradeDecisionV1:
        self.evaluated.append((config, close))
        if self.failure is not None:
            raise self.failure
        return _decision(config, close)


def _manager(engine: _FakeEngine, saved: list[dict]) -> TradingPreviewManager:
    return TradingPreviewManager(
        engine,
        settings_loader=lambda: {"trading_config": {}},
        settings_saver=lambda payload: saved.append(payload) or payload,
        poll_seconds=0.01,
    )


def test_manager_processes_each_closed_m15_once_and_applies_config_next_bar() -> None:
    engine = _FakeEngine()
    saved: list[dict] = []
    manager = _manager(engine, saved)

    assert manager.run_once() is True
    assert manager.run_once() is False
    assert len(engine.evaluated) == 1
    first_config = engine.evaluated[0][0]

    hybrid = replace(first_config, mode="rules_codex", tp1_lots=0.02)
    config_state = manager.update_config(hybrid)
    assert config_state["pending_next_m15"] is True
    assert manager.run_once() is False
    assert len(engine.evaluated) == 1

    engine.close = 1800
    assert manager.run_once() is True
    assert engine.evaluated[-1] == (hybrid, 1800)
    assert manager.config_state()["pending_next_m15"] is False
    assert manager.status()["decision_count"] == 2
    assert manager.decisions(limit=1)[0]["config_hash"] == hybrid.config_hash
    assert saved == [{"trading_config": hybrid.identity_payload()}]


def test_manager_does_not_repeat_a_failed_attempt_on_the_same_m15() -> None:
    engine = _FakeEngine()
    engine.failure = RuntimeError("preview failed")
    manager = _manager(engine, [])

    assert manager.run_once() is False
    assert manager.run_once() is False
    assert len(engine.evaluated) == 1
    assert manager.status()["last_error"] == "preview failed"

    engine.close = 1800
    assert manager.run_once() is False
    assert len(engine.evaluated) == 2


def test_manager_falls_back_to_default_when_persisted_config_is_invalid() -> None:
    manager = TradingPreviewManager(
        _FakeEngine(),
        settings_loader=lambda: {"trading_config": {"mode": "broken"}},
        settings_saver=lambda payload: payload,
    )

    state = manager.config_state()

    assert state["config"]["mode"] == "rules"
    assert state["config"]["symbol"] == "XAUUSD"
