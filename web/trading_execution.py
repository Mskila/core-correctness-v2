"""Stage-5 manual-enable execution controller and position lifecycle."""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import replace
from typing import Any, Protocol

from trading_core import (
    ALPHAMASTER_MAGIC,
    BrokerOrderV1,
    BrokerPositionV1,
    BrokerTickV1,
    ExecutionEnvironmentV1,
    ExecutionReceiptV1,
    ManagedTradeV1,
    OrderPlanCandidateV1,
    TradeDecisionV1,
    TradingConfigV1,
    TradingExecutionStateV1,
)
from web.trading_execution_mt5 import mt5_execution_adapter
from web.trading_state import TradingStateStore, trading_state_store


_M15_SECONDS = 15 * 60


class ExecutionAdapter(Protocol):
    def preflight(
        self, symbol: str, volumes: tuple[float, ...]
    ) -> ExecutionEnvironmentV1:
        ...

    def positions(self, symbol: str) -> tuple[BrokerPositionV1, ...]:
        ...

    def orders(self, symbol: str) -> tuple[BrokerOrderV1, ...]:
        ...

    def tick(self, symbol: str) -> BrokerTickV1:
        ...

    def position_closed_by_take_profit(self, ticket: int) -> bool | None:
        ...

    def place_plan(
        self,
        plan: OrderPlanCandidateV1,
        *,
        volume: float,
        take_profit: float,
        leg: str,
    ) -> ExecutionReceiptV1:
        ...

    def cancel_order(self, order: BrokerOrderV1) -> ExecutionReceiptV1:
        ...

    def close_position(
        self, position: BrokerPositionV1, *, volume: float
    ) -> ExecutionReceiptV1:
        ...

    def modify_position(
        self,
        position: BrokerPositionV1,
        *,
        stop_loss: float,
        take_profit: float,
    ) -> ExecutionReceiptV1:
        ...


class TradingExecutionController:
    """Keep execution off after every process start and manage only our magic number."""

    def __init__(
        self,
        adapter: ExecutionAdapter = mt5_execution_adapter,
        *,
        store: TradingStateStore = trading_state_store,
        management_poll_seconds: float = 2.0,
    ) -> None:
        self.adapter = adapter
        self.store = store
        self.management_poll_seconds = management_poll_seconds
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._execution_enabled = False
        self._management_enabled = True
        self._environment: ExecutionEnvironmentV1 | None = None
        self._blocker = "manual_enable_required"
        self._last_error = ""
        self._receipt_decision_id: str | None = None
        self._receipt_plan_id: str | None = None
        self._recent_receipts: deque[ExecutionReceiptV1] = deque(maxlen=30)
        self._operation_receipts: list[ExecutionReceiptV1] = []
        try:
            self._state = self.store.load()
        except RuntimeError as exc:
            self._state = TradingExecutionStateV1()
            self._management_enabled = False
            self._blocker = "invalid_persisted_state"
            self._last_error = str(exc)

    @staticmethod
    def _system_positions(
        rows: tuple[BrokerPositionV1, ...],
    ) -> tuple[BrokerPositionV1, ...]:
        return tuple(row for row in rows if row.magic == ALPHAMASTER_MAGIC)

    @staticmethod
    def _system_orders(rows: tuple[BrokerOrderV1, ...]) -> tuple[BrokerOrderV1, ...]:
        return tuple(row for row in rows if row.magic == ALPHAMASTER_MAGIC)

    @staticmethod
    def _has_external_exposure(
        positions: tuple[BrokerPositionV1, ...],
        orders: tuple[BrokerOrderV1, ...],
    ) -> bool:
        return any(row.magic != ALPHAMASTER_MAGIC for row in (*positions, *orders))

    def _persist(self) -> None:
        self.store.save(self._state)

    def _record_receipt(self, receipt: ExecutionReceiptV1) -> None:
        self._recent_receipts.append(receipt)
        self._operation_receipts.append(receipt)
        payload = receipt.to_payload()
        payload["decision_id"] = self._receipt_decision_id
        payload["plan_id"] = self._receipt_plan_id
        self.store.append_event("receipt", payload)
        self._state = replace(
            self._state,
            receipt_count=self._state.receipt_count + 1,
        )

    def _result(self, action: str, **details: Any) -> dict[str, Any]:
        return {
            "action": action,
            "receipts": [receipt.to_payload() for receipt in self._operation_receipts],
            **details,
            **self.status(),
        }

    def enable(self, config: TradingConfigV1) -> dict[str, Any]:
        volumes = (config.tp1_lots,) + (
            (config.tp2_lots,) if config.tp2_lots > 0.0 else ()
        )
        with self._lock:
            if not self._management_enabled:
                self._execution_enabled = False
                return self.status()
            try:
                environment = self.adapter.preflight(config.symbol, volumes)
                positions = self.adapter.positions(config.symbol)
                orders = self.adapter.orders(config.symbol)
                self._persist()
            except Exception as exc:  # noqa: BLE001 - user-visible operational blocker
                self._execution_enabled = False
                self._blocker = "mt5_preflight_failed"
                self._last_error = str(exc)
                return self.status()
            self._environment = environment
            if self._has_external_exposure(positions, orders):
                self._execution_enabled = False
                self._blocker = "external_xauusd_exposure"
                self._last_error = "检测到非 AlphaMaster 的 XAUUSD 持仓或挂单；未触碰并阻止新开仓"
                return self.status()
            self._execution_enabled = True
            self._blocker = ""
            self._last_error = ""
            return self.status()

    def disable(self) -> dict[str, Any]:
        with self._lock:
            self._execution_enabled = False
            self._blocker = "manual_enable_required"
            self._last_error = ""
            self._operation_receipts = []
            # Query every time, including after restart or an earlier failed
            # send: broker-side AlphaMaster pending orders can outlive the
            # local enabled flag or be absent from the last persisted state.
            try:
                orders = self.adapter.orders("XAUUSD")
                cancelled = self._cancel_system_orders(orders)
                remaining_orders = self._system_orders(self.adapter.orders("XAUUSD"))
                positions = self._system_positions(self.adapter.positions("XAUUSD"))
                if cancelled and not remaining_orders and not positions:
                    self._state = replace(self._state, managed_trade=None)
                if not cancelled or remaining_orders:
                    self._blocker = "pending_cancel_unconfirmed"
                    self._last_error = "停用时系统挂单撤销未被 MT5 查询确认"
            except Exception as exc:  # noqa: BLE001 - entries remain disabled
                self._blocker = "pending_cancel_failed"
                self._last_error = str(exc)
            self._persist()
            return self.status()

    def start_management(self) -> dict[str, Any]:
        with self._lock:
            if not self._management_enabled:
                return self.status()
            if self._thread and self._thread.is_alive():
                return self.status()
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._management_loop,
                name="trading-position-manager",
                daemon=True,
            )
            self._thread.start()
            return self.status()

    def stop_management(self) -> dict[str, Any]:
        with self._lock:
            self._execution_enabled = False
            self._stop_event.set()
            self._blocker = "manual_enable_required"
            return self.status()

    def _management_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.manage_once()
            except Exception as exc:  # noqa: BLE001 - retried without enabling entries
                with self._lock:
                    self._last_error = str(exc)
            self._stop_event.wait(self.management_poll_seconds)

    def _cancel_system_orders(
        self,
        orders: tuple[BrokerOrderV1, ...],
    ) -> bool:
        all_confirmed = True
        for order in self._system_orders(orders):
            receipt = self.adapter.cancel_order(order)
            self._record_receipt(receipt)
            all_confirmed = all_confirmed and receipt.confirmed
        return all_confirmed

    def _close_system_positions(
        self,
        positions: tuple[BrokerPositionV1, ...],
    ) -> bool:
        all_confirmed = True
        for position in self._system_positions(positions):
            receipt = self.adapter.close_position(position, volume=position.volume)
            self._record_receipt(receipt)
            all_confirmed = all_confirmed and receipt.confirmed
        return all_confirmed

    @staticmethod
    def _selected_plan(decision: TradeDecisionV1) -> OrderPlanCandidateV1 | None:
        selected_id = decision.review.selected_plan_id
        if decision.review.verdict != "approve" or selected_id is None:
            return None
        return next((plan for plan in decision.plans if plan.plan_id == selected_id), None)

    def process_decision(self, decision: TradeDecisionV1) -> dict[str, Any]:
        """Consume one closed-M15 decision; never retry the same persisted boundary."""

        with self._lock:
            close = decision.decision_close_timestamp
            self._operation_receipts = []
            if self._state.last_processed_m15 is not None and close <= self._state.last_processed_m15:
                return self._result("duplicate_m15_ignored")
            self._receipt_decision_id = decision.decision_id
            self._receipt_plan_id = None
            self.store.append_event("decision", decision.to_payload())
            try:
                positions = self.adapter.positions(decision.symbol)
                orders = self.adapter.orders(decision.symbol)
                expired = self._cancel_system_orders(orders)
            except Exception as exc:  # noqa: BLE001 - fail closed for this M15
                self._state = replace(
                    self._state,
                    last_processed_m15=close,
                    config_hash=decision.config.config_hash,
                )
                self._execution_enabled = False
                self._blocker = "mt5_state_query_failed"
                self._last_error = str(exc)
                self._persist()
                return self._result("mt5_state_query_failed")

            self._state = replace(
                self._state,
                last_processed_m15=close,
                config_hash=decision.config.config_hash,
            )
            if not expired:
                self._execution_enabled = False
                self._blocker = "pending_cancel_unconfirmed"
                self._last_error = "系统挂单撤销未被 MT5 查询确认；已停止新开仓"
                self._persist()
                return self._result("pending_cancel_unconfirmed")

            positions = self.adapter.positions(decision.symbol)
            orders = self.adapter.orders(decision.symbol)
            if self._has_external_exposure(positions, orders):
                self._execution_enabled = False
                self._blocker = "external_xauusd_exposure"
                self._last_error = "检测到非 AlphaMaster 的 XAUUSD 持仓或挂单；未触碰并阻止新开仓"
                self._persist()
                return self._result("external_exposure_blocked")

            selected = self._selected_plan(decision)
            if selected is None:
                self._persist()
                return self._result("decision_rejected")
            self._receipt_plan_id = selected.plan_id
            if not self._execution_enabled:
                self._persist()
                return self._result("execution_disabled")
            if (
                self._state.cooldown_until_m15 is not None
                and close < self._state.cooldown_until_m15
            ):
                self._persist()
                return self._result("reverse_cooldown")

            system_positions = self._system_positions(positions)
            if system_positions:
                if any(position.side == selected.side for position in system_positions):
                    self._persist()
                    return self._result("no_pyramiding")
                closed = self._close_system_positions(system_positions)
                self._state = replace(
                    self._state,
                    cooldown_until_m15=close + _M15_SECONDS,
                    managed_trade=None if closed else self._state.managed_trade,
                )
                if not closed:
                    self._execution_enabled = False
                    self._blocker = "reverse_close_unconfirmed"
                    self._last_error = "反向前平仓未被 MT5 查询确认；已停止新开仓"
                self._persist()
                return self._result(
                    "reverse_closed_wait_next_m15" if closed else "reverse_close_unconfirmed"
                )

            try:
                volumes = (decision.config.tp1_lots,) + (
                    (decision.config.tp2_lots,) if decision.config.tp2_lots > 0 else ()
                )
                environment = self.adapter.preflight(decision.symbol, volumes)
                self._environment = environment
                if decision.config.tp2_lots > 0 and selected.take_profit_2 is None:
                    raise ValueError("selected plan has no TP2 target")
                if environment.account.position_mode == "hedging":
                    result = self._open_hedging(decision, selected, environment)
                else:
                    result = self._open_netting(decision, selected, environment)
                self._last_error = ""
            except Exception as exc:  # noqa: BLE001 - no fallback order is permitted
                self._execution_enabled = False
                self._blocker = "order_validation_or_send_failed"
                self._last_error = str(exc)
                self._persist()
                return self._result("order_failed")
            self._persist()
            return self._result(result)

    def _open_hedging(
        self,
        decision: TradeDecisionV1,
        plan: OrderPlanCandidateV1,
        environment: ExecutionEnvironmentV1,
    ) -> str:
        specs = [("tp1", decision.config.tp1_lots, plan.take_profit_1)]
        if decision.config.tp2_lots > 0 and plan.take_profit_2 is not None:
            specs.append(("tp2", decision.config.tp2_lots, plan.take_profit_2))
        position_tickets: list[int] = []
        pending_tickets: list[int] = []
        leg_tickets: dict[str, int] = {}
        confirmed_count = 0
        executed_entry = plan.entry_price
        for leg, volume, target in specs:
            receipt = self.adapter.place_plan(
                plan,
                volume=volume,
                take_profit=target,
                leg=leg,
            )
            ticket = receipt.position_ticket or receipt.order_ticket
            if receipt.accepted and ticket is not None:
                if not leg_tickets and receipt.executed_price is not None:
                    executed_entry = receipt.executed_price
                leg_tickets[leg] = ticket
                if receipt.confirmation_kind == "position" or plan.order_type == "market":
                    position_tickets.append(ticket)
                else:
                    pending_tickets.append(ticket)
                self._state = replace(
                    self._state,
                    managed_trade=ManagedTradeV1(
                        decision_id=decision.decision_id,
                        plan_id=plan.plan_id,
                        decision_m15_close=decision.decision_close_timestamp,
                        config_hash=decision.config.config_hash,
                        account_mode=environment.account.position_mode,
                        side=plan.side,
                        entry_price=executed_entry,
                        stop_loss=plan.stop_loss,
                        take_profit_1=plan.take_profit_1,
                        take_profit_2=plan.take_profit_2,
                        tp1_volume=decision.config.tp1_lots,
                        tp2_volume=decision.config.tp2_lots,
                        position_tickets=tuple(position_tickets),
                        pending_tickets=tuple(pending_tickets),
                        tp1_ticket=leg_tickets.get("tp1"),
                        tp2_ticket=leg_tickets.get("tp2"),
                    ),
                )
            self._record_receipt(receipt)
            if receipt.confirmed and ticket is not None:
                confirmed_count += 1
            else:
                self._execution_enabled = False
                self._blocker = "order_confirmation_failed"
                self._last_error = "MT5 接受请求但未能从持仓/挂单查询确认；已停止新开仓"
                break
        return "opened_hedging" if confirmed_count == len(specs) else "opening_partial"

    def _open_netting(
        self,
        decision: TradeDecisionV1,
        plan: OrderPlanCandidateV1,
        environment: ExecutionEnvironmentV1,
    ) -> str:
        total = decision.config.tp1_lots + decision.config.tp2_lots
        target = (
            plan.take_profit_2
            if decision.config.tp2_lots > 0 and plan.take_profit_2 is not None
            else plan.take_profit_1
        )
        receipt = self.adapter.place_plan(
            plan,
            volume=total,
            take_profit=target,
            leg="net",
        )
        ticket = receipt.position_ticket or receipt.order_ticket
        if receipt.accepted and ticket is not None:
            self._state = replace(
                self._state,
                managed_trade=ManagedTradeV1(
                    decision_id=decision.decision_id,
                    plan_id=plan.plan_id,
                    decision_m15_close=decision.decision_close_timestamp,
                    config_hash=decision.config.config_hash,
                    account_mode=environment.account.position_mode,
                    side=plan.side,
                    entry_price=receipt.executed_price or plan.entry_price,
                    stop_loss=plan.stop_loss,
                    take_profit_1=plan.take_profit_1,
                    take_profit_2=plan.take_profit_2,
                    tp1_volume=decision.config.tp1_lots,
                    tp2_volume=decision.config.tp2_lots,
                    position_tickets=(
                        (ticket,)
                        if receipt.position_ticket is not None or plan.order_type == "market"
                        else ()
                    ),
                    pending_tickets=(
                        (ticket,) if plan.order_type != "market" else ()
                    ),
                    tp1_ticket=ticket,
                    tp2_ticket=ticket if decision.config.tp2_lots > 0 else None,
                ),
            )
        self._record_receipt(receipt)
        if not receipt.confirmed or ticket is None:
            self._execution_enabled = False
            self._blocker = "order_confirmation_failed"
            self._last_error = "MT5 接受请求但未能从持仓/挂单查询确认；已停止新开仓"
            return "order_confirmation_failed"
        return "opened_netting"

    @staticmethod
    def _crossed_tp1(managed: ManagedTradeV1, tick: BrokerTickV1) -> bool:
        if managed.side == "long":
            return tick.bid >= managed.take_profit_1
        return tick.ask <= managed.take_profit_1

    @staticmethod
    def _break_even(managed: ManagedTradeV1, tick: BrokerTickV1) -> float:
        buffer = max(tick.spread, tick.point)
        if managed.side == "long":
            return managed.entry_price + buffer
        return managed.entry_price - buffer

    @staticmethod
    def _position_by_ticket(
        positions: tuple[BrokerPositionV1, ...], ticket: int | None
    ) -> BrokerPositionV1 | None:
        if ticket is None:
            return None
        return next((position for position in positions if position.ticket == ticket), None)

    def manage_once(self) -> dict[str, Any]:
        """Manage only persisted AlphaMaster exposure, even while new entries are disabled."""

        with self._lock:
            managed = self._state.managed_trade
            self._operation_receipts = []
            if not self._management_enabled or managed is None:
                return self._result("management_idle")
            self._receipt_decision_id = managed.decision_id
            self._receipt_plan_id = managed.plan_id
            try:
                positions = self.adapter.positions("XAUUSD")
                orders = self.adapter.orders("XAUUSD")
                system_positions = self._system_positions(positions)
                system_orders = self._system_orders(orders)
                if system_orders and not system_positions:
                    return self._result("pending_wait")
                if not system_positions and not system_orders:
                    self._state = replace(self._state, managed_trade=None)
                    self._persist()
                    return self._result("managed_trade_closed")
                if managed.account_mode == "netting":
                    action = self._manage_netting(managed, system_positions)
                else:
                    action = self._manage_hedging(managed, system_positions)
                self._persist()
                self._last_error = ""
                return self._result(action)
            except Exception as exc:  # noqa: BLE001 - management retries, entries stop
                self._execution_enabled = False
                self._blocker = "position_management_failed"
                self._last_error = str(exc)
                self._persist()
                return self._result("position_management_failed")

    def _manage_netting(
        self,
        managed: ManagedTradeV1,
        positions: tuple[BrokerPositionV1, ...],
    ) -> str:
        if not positions:
            return "management_idle"
        position = positions[0]
        managed = replace(managed, entry_price=position.price_open)
        updated = replace(
            managed,
            position_tickets=(position.ticket,),
            pending_tickets=(),
            tp1_ticket=position.ticket,
            tp2_ticket=position.ticket if managed.tp2_volume > 0 else None,
        )
        final_target = (
            managed.take_profit_2
            if managed.tp2_volume > 0 and managed.take_profit_2 is not None
            else managed.take_profit_1
        )
        if managed.tp2_volume > 0 and not managed.tp1_done:
            tick = self.adapter.tick(position.symbol)
            if self._crossed_tp1(managed, tick):
                closed = self.adapter.close_position(position, volume=managed.tp1_volume)
                self._record_receipt(closed)
                if not closed.confirmed:
                    raise RuntimeError("netting TP1 partial close was not confirmed")
                remaining = self._system_positions(self.adapter.positions(position.symbol))
                if not remaining:
                    self._state = replace(self._state, managed_trade=None)
                    return "managed_trade_closed"
                position = remaining[0]
                updated = replace(
                    updated,
                    position_tickets=(position.ticket,),
                    pending_tickets=(),
                    tp1_done=True,
                )
                # The partial close is irreversible.  Persist it before the
                # separate broker request that moves the remaining protection,
                # so a restart cannot close TP1 volume a second time.
                self._state = replace(self._state, managed_trade=updated)
                self._persist()
                break_even = self._break_even(updated, tick)
                modified = self.adapter.modify_position(
                    position,
                    stop_loss=break_even,
                    take_profit=float(final_target),
                )
                self._record_receipt(modified)
                if not modified.confirmed:
                    raise RuntimeError("netting break-even modification was not confirmed")
                updated = replace(
                    updated,
                    position_tickets=(position.ticket,),
                    tp1_done=True,
                    break_even_applied=True,
                )
                self._state = replace(self._state, managed_trade=updated)
                return "tp1_break_even"
        expected_stop = managed.stop_loss
        if managed.tp1_done or managed.break_even_applied:
            expected_stop = self._break_even(managed, self.adapter.tick(position.symbol))
        tolerance = self._environment.point if self._environment is not None else 1e-9
        if (
            abs(position.stop_loss - expected_stop) > tolerance * 0.5
            or abs(position.take_profit - float(final_target)) > tolerance * 0.5
        ):
            receipt = self.adapter.modify_position(
                position,
                stop_loss=expected_stop,
                take_profit=float(final_target),
            )
            self._record_receipt(receipt)
            if not receipt.confirmed:
                raise RuntimeError("netting protection restore was not confirmed")
            if managed.tp1_done:
                updated = replace(updated, break_even_applied=True)
            action = "protection_restored"
        else:
            action = "position_managed"
        self._state = replace(self._state, managed_trade=updated)
        return action

    def _manage_hedging(
        self,
        managed: ManagedTradeV1,
        positions: tuple[BrokerPositionV1, ...],
    ) -> str:
        if not positions:
            return "management_idle"
        tolerance = self._environment.point if self._environment is not None else 1e-6
        tp1 = self._position_by_ticket(positions, managed.tp1_ticket)
        tp2 = self._position_by_ticket(positions, managed.tp2_ticket)
        if tp1 is None and not managed.tp1_done:
            tp1 = next(
                (
                    row
                    for row in positions
                    if abs(row.take_profit - managed.take_profit_1) <= tolerance
                ),
                None,
            )
        if tp2 is None and managed.tp2_volume > 0 and managed.take_profit_2 is not None:
            tp2 = next(
                (
                    row
                    for row in positions
                    if abs(row.take_profit - managed.take_profit_2) <= tolerance
                ),
                None,
            )
        reference_position = tp2 or tp1 or positions[0]
        managed = replace(managed, entry_price=reference_position.price_open)
        if tp1 is None and managed.tp1_ticket is not None and managed.tp2_volume > 0:
            if tp2 is not None and not managed.break_even_applied:
                tp1_hit = self.adapter.position_closed_by_take_profit(managed.tp1_ticket)
                if tp1_hit is not True:
                    self._state = replace(self._state, managed_trade=managed)
                    return "tp1_close_unverified"
                tick = self.adapter.tick(tp2.symbol)
                break_even = self._break_even(managed, tick)
                receipt = self.adapter.modify_position(
                    tp2,
                    stop_loss=break_even,
                    take_profit=float(managed.take_profit_2),
                )
                self._record_receipt(receipt)
                if not receipt.confirmed:
                    raise RuntimeError("hedging break-even modification was not confirmed")
                managed = replace(
                    managed,
                    tp1_done=True,
                    break_even_applied=True,
                    position_tickets=tuple(row.ticket for row in positions),
                    pending_tickets=(),
                )
                self._state = replace(self._state, managed_trade=managed)
                return "tp1_break_even"
        protection_restored = False
        for position, target in (
            (tp1, managed.take_profit_1),
            (tp2, managed.take_profit_2),
        ):
            if position is None or target is None:
                continue
            expected_stop = managed.stop_loss
            if managed.break_even_applied and position is tp2:
                expected_stop = self._break_even(managed, self.adapter.tick(position.symbol))
            if (
                abs(position.stop_loss - expected_stop) <= tolerance * 0.5
                and abs(position.take_profit - target) <= tolerance * 0.5
            ):
                continue
            receipt = self.adapter.modify_position(
                position,
                stop_loss=expected_stop,
                take_profit=target,
            )
            self._record_receipt(receipt)
            if not receipt.confirmed:
                raise RuntimeError("hedging protection restore was not confirmed")
            protection_restored = True
        managed = replace(
            managed,
            entry_price=reference_position.price_open,
            position_tickets=tuple(row.ticket for row in positions),
            pending_tickets=(),
            tp1_ticket=None if tp1 is None else tp1.ticket,
            tp2_ticket=None if tp2 is None else tp2.ticket,
        )
        self._state = replace(self._state, managed_trade=managed)
        return "protection_restored" if protection_restored else "position_managed"

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "execution_enabled": self._execution_enabled,
                "management_enabled": self._management_enabled,
                "management_running": bool(self._thread and self._thread.is_alive()),
                "blocker": self._blocker,
                "last_error": self._last_error,
                "account": (
                    None
                    if self._environment is None
                    else self._environment.account.to_payload()
                ),
                "state": self._state.to_payload(),
                "recent_receipts": [
                    receipt.to_payload() for receipt in reversed(self._recent_receipts)
                ],
            }


trading_execution_controller = TradingExecutionController()
