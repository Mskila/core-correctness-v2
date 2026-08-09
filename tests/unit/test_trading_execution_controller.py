from __future__ import annotations

from dataclasses import replace

from trading_core import (
    BrokerAccountV1,
    BrokerOrderV1,
    BrokerPositionV1,
    BrokerTickV1,
    DecisionReviewV1,
    ExecutionEnvironmentV1,
    ExecutionReceiptV1,
    OrderPlanCandidateV1,
    TradeDecisionV1,
    TradingConfigV1,
)
from web.trading_execution import TradingExecutionController
from web.trading_state import TradingStateStore


def _plan(*, side: str = "long", order_type: str = "market") -> OrderPlanCandidateV1:
    return OrderPlanCandidateV1(
        plan_id=f"plan-{side}-{order_type}",
        style="pa_primary",
        side=side,
        family="breakout",
        setup_type="breakout_retest",
        order_type=order_type,
        entry_price=2400.0,
        trigger_price=2401.0 if order_type == "stop" else None,
        limit_price=2399.0 if order_type == "limit" else None,
        stop_loss=2398.0 if side == "long" else 2402.0,
        take_profit_1=2403.0 if side == "long" else 2397.0,
        take_profit_2=2404.0 if side == "long" else 2396.0,
        risk_distance=2.0,
        take_profit_1_r=1.5,
        take_profit_2_r=2.0,
        source_trigger_at=900,
        source_policy="formula_candidate_only",
        policy_origin="pa_formula_candidate",
    )


def _decision(
    close: int,
    *,
    side: str = "long",
    order_type: str = "market",
    tp2_lots: float = 0.01,
) -> TradeDecisionV1:
    plan = _plan(side=side, order_type=order_type)
    config = replace(TradingConfigV1.default(), tp1_lots=0.01, tp2_lots=tp2_lots)
    review = DecisionReviewV1(
        decision_id=f"decision-{close}",
        verdict="approve",
        selected_plan_id=plan.plan_id,
        reason_code="approved_selected_plan",
        summary_zh="规则通过",
    )
    return TradeDecisionV1(
        decision_id=review.decision_id,
        symbol="XAUUSD",
        decision_close_timestamp=close,
        h1_close_timestamp=close,
        m15_close_timestamp=close,
        m5_close_timestamp=close,
        config=config,
        alpha_scores=(),
        pa_scores=(),
        pa_evidence=(),
        input_warnings=(),
        direction_score=1.0 if side == "long" else -1.0,
        entry_score=1.0 if side == "long" else -1.0,
        side=side,
        fusion_accepted=True,
        fusion_reject_reasons=(),
        plan_reject_reasons=(),
        plans=(plan,),
        review=review,
        final_action="preview_approved",
        created_at=1.0,
    )


class _FakeAdapter:
    def __init__(self, mode: str = "hedging") -> None:
        self.mode = mode
        self.positions_rows: list[BrokerPositionV1] = []
        self.orders_rows: list[BrokerOrderV1] = []
        self.placed: list[tuple[str, float, float, str]] = []
        self.cancelled: list[int] = []
        self.closed: list[tuple[int, float]] = []
        self.modified: list[tuple[int, float, float]] = []
        self.next_ticket = 100
        self.tick_row = BrokerTickV1(symbol="XAUUSD", bid=2403.1, ask=2403.2, point=0.01)
        self.tp_closed_tickets: set[int] = set()

    def preflight(self, symbol, volumes):
        assert symbol == "XAUUSD"
        return ExecutionEnvironmentV1(
            account=BrokerAccountV1(
                login=123,
                server="Demo",
                position_mode=self.mode,
                trade_allowed=True,
                trade_expert=True,
            ),
            symbol=symbol,
            point=0.01,
            digits=2,
            volume_min=0.01,
            volume_max=10.0,
            volume_step=0.01,
            min_stop_distance=0.1,
            order_mode=15,
        )

    def positions(self, symbol):
        return tuple(self.positions_rows)

    def orders(self, symbol):
        return tuple(self.orders_rows)

    def tick(self, symbol):
        return self.tick_row

    def position_closed_by_take_profit(self, ticket):
        return ticket in self.tp_closed_tickets

    def place_plan(self, plan, *, volume, take_profit, leg):
        self.placed.append((plan.plan_id, volume, take_profit, leg))
        self.next_ticket += 1
        if plan.order_type == "market":
            row = BrokerPositionV1(
                ticket=self.next_ticket,
                symbol="XAUUSD",
                side=plan.side,
                magic=20250101,
                volume=volume,
                price_open=plan.entry_price,
                stop_loss=plan.stop_loss,
                take_profit=take_profit,
                comment=f"AM5:{leg}",
            )
            if self.mode == "netting" and self.positions_rows:
                old = self.positions_rows[0]
                row = replace(old, volume=old.volume + volume, take_profit=take_profit)
                self.positions_rows[0] = row
            else:
                self.positions_rows.append(row)
            return ExecutionReceiptV1.ok(
                action="place", retcode=10009, message="done",
                position_ticket=row.ticket, confirmation_kind="position"
            )
        row = BrokerOrderV1(
            ticket=self.next_ticket,
            symbol="XAUUSD",
            side=plan.side,
            magic=20250101,
            volume=volume,
            price_open=plan.entry_price,
            stop_loss=plan.stop_loss,
            take_profit=take_profit,
            order_type=plan.order_type,
            comment=f"AM5:{leg}",
        )
        self.orders_rows.append(row)
        return ExecutionReceiptV1.ok(
            action="place", retcode=10008, message="placed",
            order_ticket=row.ticket, confirmation_kind="order"
        )

    def cancel_order(self, order):
        self.cancelled.append(order.ticket)
        self.orders_rows = [row for row in self.orders_rows if row.ticket != order.ticket]
        return ExecutionReceiptV1.ok(
            action="cancel", retcode=10009, message="cancelled",
            order_ticket=order.ticket, confirmation_kind="cancelled"
        )

    def close_position(self, position, *, volume):
        self.closed.append((position.ticket, volume))
        remaining = position.volume - volume
        if remaining > 1e-9:
            self.positions_rows = [
                replace(row, volume=remaining) if row.ticket == position.ticket else row
                for row in self.positions_rows
            ]
        else:
            self.positions_rows = [row for row in self.positions_rows if row.ticket != position.ticket]
        return ExecutionReceiptV1.ok(
            action="close", retcode=10009, message="closed",
            position_ticket=position.ticket, confirmation_kind="closed"
        )

    def modify_position(self, position, *, stop_loss, take_profit):
        self.modified.append((position.ticket, stop_loss, take_profit))
        self.positions_rows = [
            replace(row, stop_loss=stop_loss, take_profit=take_profit)
            if row.ticket == position.ticket else row
            for row in self.positions_rows
        ]
        return ExecutionReceiptV1.ok(
            action="modify", retcode=10009, message="modified",
            position_ticket=position.ticket, confirmation_kind="modified"
        )


def _controller(tmp_path, adapter):
    return TradingExecutionController(
        adapter=adapter,
        store=TradingStateStore(
            state_path=tmp_path / "state.json",
            journal_path=tmp_path / "events.jsonl",
        ),
        management_poll_seconds=0.01,
    )


def test_manual_enable_is_runtime_only_and_restart_returns_disabled(tmp_path) -> None:
    adapter = _FakeAdapter()
    controller = _controller(tmp_path, adapter)

    enabled = controller.enable(TradingConfigV1.default())
    restarted = _controller(tmp_path, adapter)

    assert enabled["execution_enabled"] is True
    assert restarted.status()["execution_enabled"] is False
    assert restarted.status()["management_enabled"] is True


def test_hedging_places_two_independent_legs_and_never_pyramids(tmp_path) -> None:
    adapter = _FakeAdapter("hedging")
    controller = _controller(tmp_path, adapter)
    controller.enable(_decision(900).config)

    first = controller.process_decision(_decision(900))
    second = controller.process_decision(_decision(1_800))

    assert first["action"] == "opened_hedging"
    assert [row[3] for row in adapter.placed] == ["tp1", "tp2"]
    assert second["action"] == "no_pyramiding"
    assert len(adapter.placed) == 2


def test_hedging_second_leg_failure_persists_first_leg_for_management(tmp_path) -> None:
    class _SecondLegFails(_FakeAdapter):
        def place_plan(self, plan, *, volume, take_profit, leg):
            if leg == "tp2":
                raise RuntimeError("second leg rejected")
            return super().place_plan(
                plan, volume=volume, take_profit=take_profit, leg=leg
            )

    adapter = _SecondLegFails("hedging")
    controller = _controller(tmp_path, adapter)
    decision = _decision(900)
    controller.enable(decision.config)

    result = controller.process_decision(decision)

    assert result["action"] == "order_failed"
    assert result["execution_enabled"] is False
    persisted = controller.store.load().managed_trade
    assert persisted is not None
    assert persisted.tp1_ticket == 101
    assert persisted.tp2_ticket is None


def test_netting_merges_volume_then_partially_closes_tp1_and_moves_to_break_even(tmp_path) -> None:
    adapter = _FakeAdapter("netting")
    controller = _controller(tmp_path, adapter)
    decision = _decision(900)
    controller.enable(decision.config)

    opened = controller.process_decision(decision)
    managed = controller.manage_once()

    assert opened["action"] == "opened_netting"
    assert adapter.placed == [(decision.plans[0].plan_id, 0.02, 2404.0, "net")]
    assert adapter.closed == [(adapter.positions_rows[0].ticket, 0.01)]
    assert managed["action"] == "tp1_break_even"
    assert adapter.modified[-1][1] == 2400.1


def test_netting_persists_tp1_before_break_even_retry_after_restart(tmp_path) -> None:
    class _FirstModifyFails(_FakeAdapter):
        def __init__(self) -> None:
            super().__init__("netting")
            self.fail_modify = True

        def modify_position(self, position, *, stop_loss, take_profit):
            if self.fail_modify:
                self.fail_modify = False
                raise RuntimeError("temporary modify failure")
            return super().modify_position(
                position,
                stop_loss=stop_loss,
                take_profit=take_profit,
            )

    adapter = _FirstModifyFails()
    controller = _controller(tmp_path, adapter)
    decision = _decision(900)
    controller.enable(decision.config)
    controller.process_decision(decision)

    failed = controller.manage_once()
    persisted = controller.store.load().managed_trade

    assert failed["action"] == "position_management_failed"
    assert len(adapter.closed) == 1
    assert persisted is not None
    assert persisted.tp1_done is True
    assert persisted.break_even_applied is False

    restarted = _controller(tmp_path, adapter)
    retried = restarted.manage_once()

    assert retried["action"] == "protection_restored"
    assert len(adapter.closed) == 1
    assert restarted.store.load().managed_trade.break_even_applied is True


def test_hedging_tp1_disappearance_moves_only_tp2_to_break_even(tmp_path) -> None:
    adapter = _FakeAdapter("hedging")
    controller = _controller(tmp_path, adapter)
    decision = _decision(900)
    controller.enable(decision.config)
    controller.process_decision(decision)
    tp1, tp2 = adapter.positions_rows
    adapter.positions_rows = [tp2]
    adapter.tp_closed_tickets.add(tp1.ticket)

    managed = controller.manage_once()

    assert managed["action"] == "tp1_break_even"
    assert adapter.modified == [(tp2.ticket, 2400.1, 2404.0)]
    assert all(ticket != tp1.ticket for ticket, _, _ in adapter.modified)


def test_pending_orders_expire_before_next_m15_and_are_rebuilt(tmp_path) -> None:
    adapter = _FakeAdapter("hedging")
    controller = _controller(tmp_path, adapter)
    controller.enable(_decision(900, order_type="stop").config)

    controller.process_decision(_decision(900, order_type="stop"))
    assert len(adapter.orders_rows) == 2
    result = controller.process_decision(_decision(1_800, order_type="stop"))

    assert adapter.cancelled == [101, 102]
    assert result["action"] == "opened_hedging"
    assert len(adapter.orders_rows) == 2


def test_manual_disable_cancels_system_pending_orders_but_keeps_management_on(tmp_path) -> None:
    adapter = _FakeAdapter("hedging")
    controller = _controller(tmp_path, adapter)
    decision = _decision(900, order_type="stop")
    controller.enable(decision.config)
    controller.process_decision(decision)

    disabled = controller.disable()

    assert disabled["execution_enabled"] is False
    assert disabled["management_enabled"] is True
    assert adapter.cancelled == [101, 102]
    assert adapter.orders_rows == []


def test_manual_disable_cancels_orphan_system_pending_while_runtime_is_already_off(
    tmp_path,
) -> None:
    adapter = _FakeAdapter("hedging")
    adapter.orders_rows.append(
        BrokerOrderV1(
            ticket=77,
            symbol="XAUUSD",
            side="long",
            magic=20250101,
            volume=0.01,
            price_open=2401.0,
            stop_loss=2398.0,
            take_profit=2404.0,
            order_type="stop",
            comment="AM5:orphan",
        )
    )
    controller = _controller(tmp_path, adapter)

    disabled = controller.disable()

    assert disabled["execution_enabled"] is False
    assert disabled["management_enabled"] is True
    assert adapter.cancelled == [77]
    assert adapter.orders_rows == []


def test_reverse_signal_closes_only_system_positions_and_waits_for_next_bar(tmp_path) -> None:
    adapter = _FakeAdapter("hedging")
    controller = _controller(tmp_path, adapter)
    controller.enable(_decision(900).config)
    controller.process_decision(_decision(900))

    reversed_now = controller.process_decision(_decision(1_800, side="short"))
    reversed_next = controller.process_decision(_decision(2_700, side="short"))

    assert reversed_now["action"] == "reverse_closed_wait_next_m15"
    assert len(adapter.closed) == 2
    assert reversed_next["action"] == "opened_hedging"


def test_external_position_is_never_touched_and_blocks_new_entry(tmp_path) -> None:
    adapter = _FakeAdapter("hedging")
    adapter.positions_rows.append(
        BrokerPositionV1(
            ticket=88,
            symbol="XAUUSD",
            side="long",
            magic=0,
            volume=0.1,
            price_open=2400.0,
            stop_loss=0.0,
            take_profit=0.0,
            comment="manual",
        )
    )
    controller = _controller(tmp_path, adapter)

    status = controller.enable(_decision(900).config)

    assert status["execution_enabled"] is False
    assert status["blocker"] == "external_xauusd_exposure"
    assert adapter.closed == []
    assert adapter.placed == []
