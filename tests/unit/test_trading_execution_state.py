from __future__ import annotations

import json

from trading_core import ManagedTradeV1, TradingExecutionStateV1
from web.trading_state import TradingStateStore


def _managed_trade() -> ManagedTradeV1:
    return ManagedTradeV1(
        decision_id="decision-1",
        plan_id="plan-1",
        decision_m15_close=1_800,
        config_hash="a" * 64,
        account_mode="netting",
        side="long",
        entry_price=2400.0,
        stop_loss=2398.0,
        take_profit_1=2403.0,
        take_profit_2=2404.0,
        tp1_volume=0.01,
        tp2_volume=0.02,
        position_tickets=(501,),
        pending_tickets=(),
        tp1_ticket=501,
        tp2_ticket=501,
    )


def test_execution_state_round_trips_through_atomic_store(tmp_path) -> None:
    store = TradingStateStore(
        state_path=tmp_path / "state.json",
        journal_path=tmp_path / "events.jsonl",
    )
    state = TradingExecutionStateV1(
        last_processed_m15=1_800,
        config_hash="a" * 64,
        cooldown_until_m15=2_700,
        managed_trade=_managed_trade(),
        receipt_count=3,
    )

    store.save(state)

    assert store.load() == state
    assert not list(tmp_path.glob("*.tmp"))


def test_execution_journal_records_decisions_and_receipts_as_jsonl(tmp_path) -> None:
    store = TradingStateStore(
        state_path=tmp_path / "state.json",
        journal_path=tmp_path / "events.jsonl",
    )

    store.append_event("decision", {"decision_id": "d-1"})
    store.append_event("receipt", {"retcode": 10009, "confirmed": True})

    rows = [json.loads(line) for line in store.journal_path.read_text("utf-8").splitlines()]
    assert [row["event"] for row in rows] == ["decision", "receipt"]
    assert rows[0]["payload"]["decision_id"] == "d-1"
    assert rows[1]["payload"]["confirmed"] is True
