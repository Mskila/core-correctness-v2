"""Mutating MT5 adapter used only by the explicitly enabled Stage-5 controller."""

from __future__ import annotations

import math
from decimal import Decimal, InvalidOperation
from typing import Any

from trading_core import (
    ALPHAMASTER_MAGIC,
    BrokerAccountV1,
    BrokerOrderV1,
    BrokerPositionV1,
    BrokerTickV1,
    ExecutionEnvironmentV1,
    ExecutionReceiptV1,
    OrderPlanCandidateV1,
)
from web.data_sources.mt5_source import MT5_API_LOCK


_MINIMUM_R = {
    "trend_continuation": 1.5,
    "breakout": 1.5,
    "reversal": 1.2,
    "range": 1.2,
}


def _value(row: object, name: str, default: Any = None) -> Any:
    if isinstance(row, dict):
        return row.get(name, default)
    return getattr(row, name, default)


def _positive_ticket(value: object) -> int | None:
    return int(value) if type(value) is int and value > 0 else None


def _request_pairs(request: dict[str, Any]) -> tuple[tuple[str, object], ...]:
    return tuple(
        (key, value)
        for key, value in sorted(request.items())
        if value is None or type(value) in {str, int, float, bool}
    )


class MT5ExecutionAdapter:
    """Validate, send and independently confirm AlphaMaster MT5 requests."""

    def __init__(self, *, mt5: object | None = None) -> None:
        self._mt5 = mt5

    def _module(self):
        if self._mt5 is not None:
            return self._mt5
        try:
            import MetaTrader5 as mt5
        except ImportError as exc:
            raise RuntimeError("未安装 MetaTrader5：pip install -r requirements-mt5.txt") from exc
        return mt5

    def _initialize(self):
        mt5 = self._module()
        if not mt5.initialize():
            raise RuntimeError(f"MT5 初始化失败 {mt5.last_error()}；请确认默认终端已打开并登录")
        return mt5

    @staticmethod
    def _validate_volume(
        volume: float,
        *,
        minimum: float,
        maximum: float,
        step: float,
    ) -> None:
        try:
            value = Decimal(str(volume))
            low = Decimal(str(minimum))
            high = Decimal(str(maximum))
            increment = Decimal(str(step))
        except (InvalidOperation, ValueError) as exc:
            raise ValueError("volume must be a finite broker lot value") from exc
        if not value.is_finite() or value < low or value > high:
            raise ValueError(f"volume must be in [{minimum}, {maximum}]")
        units = (value - low) / increment
        if units != units.to_integral_value():
            raise ValueError(f"volume does not match broker volume_step={step}; it was not rounded")

    def _environment(self, mt5, symbol: str) -> ExecutionEnvironmentV1:
        account = mt5.account_info()
        terminal = mt5.terminal_info()
        if account is None or terminal is None:
            raise RuntimeError(f"MT5 未返回账户/终端信息 {mt5.last_error()}")
        if not mt5.symbol_select(symbol, True):
            raise RuntimeError(f"MT5 无法选择 {symbol} {mt5.last_error()}")
        info = mt5.symbol_info(symbol)
        if info is None:
            raise RuntimeError(f"MT5 无法读取 {symbol} 合约信息 {mt5.last_error()}")
        trade_allowed = bool(
            _value(account, "trade_allowed", False)
            and _value(terminal, "trade_allowed", False)
        )
        trade_expert = bool(_value(terminal, "trade_expert", False))
        if not trade_allowed:
            raise RuntimeError("MT5 当前账户或终端未允许交易")
        if not trade_expert:
            raise RuntimeError("MT5 当前终端未允许算法交易")
        hedging = getattr(mt5, "ACCOUNT_MARGIN_MODE_RETAIL_HEDGING", 2)
        position_mode = (
            "hedging" if _value(account, "margin_mode") == hedging else "netting"
        )
        point = float(_value(info, "point", 0.0) or 0.0)
        tick_size = float(_value(info, "trade_tick_size", point) or point)
        if point <= 0 or tick_size <= 0:
            raise RuntimeError(f"MT5 返回的 {symbol} point/tick size 无效")
        return ExecutionEnvironmentV1(
            account=BrokerAccountV1(
                login=int(_value(account, "login", 0) or 0),
                server=str(_value(account, "server", "") or ""),
                position_mode=position_mode,
                trade_allowed=trade_allowed,
                trade_expert=trade_expert,
            ),
            symbol=symbol,
            point=tick_size,
            digits=int(_value(info, "digits", 0) or 0),
            volume_min=float(_value(info, "volume_min", 0.0) or 0.0),
            volume_max=float(_value(info, "volume_max", 0.0) or 0.0),
            volume_step=float(_value(info, "volume_step", 0.0) or 0.0),
            min_stop_distance=max(0, int(_value(info, "trade_stops_level", 0) or 0)) * point,
            order_mode=int(_value(info, "order_mode", 0) or 0),
        )

    def preflight(
        self,
        symbol: str,
        volumes: tuple[float, ...],
    ) -> ExecutionEnvironmentV1:
        with MT5_API_LOCK:
            mt5 = self._initialize()
            environment = self._environment(mt5, symbol)
            for volume in volumes:
                self._validate_volume(
                    volume,
                    minimum=environment.volume_min,
                    maximum=environment.volume_max,
                    step=environment.volume_step,
                )
            return environment

    @staticmethod
    def _side(mt5, row_type: int) -> str:
        long_types = {
            getattr(mt5, "ORDER_TYPE_BUY", 0),
            getattr(mt5, "ORDER_TYPE_BUY_LIMIT", 2),
            getattr(mt5, "ORDER_TYPE_BUY_STOP", 4),
            getattr(mt5, "ORDER_TYPE_BUY_STOP_LIMIT", 6),
        }
        return "long" if row_type in long_types else "short"

    @staticmethod
    def _pending_type(mt5, row_type: int) -> str:
        if row_type in {
            getattr(mt5, "ORDER_TYPE_BUY_LIMIT", 2),
            getattr(mt5, "ORDER_TYPE_SELL_LIMIT", 3),
        }:
            return "limit"
        if row_type in {
            getattr(mt5, "ORDER_TYPE_BUY_STOP_LIMIT", 6),
            getattr(mt5, "ORDER_TYPE_SELL_STOP_LIMIT", 7),
        }:
            return "stop_limit"
        return "stop"

    def positions(self, symbol: str) -> tuple[BrokerPositionV1, ...]:
        with MT5_API_LOCK:
            mt5 = self._initialize()
            rows = mt5.positions_get(symbol=symbol)
            if rows is None:
                raise RuntimeError(f"MT5 positions_get 失败 {mt5.last_error()}")
            return tuple(
                BrokerPositionV1(
                    ticket=int(_value(row, "ticket")),
                    symbol=str(_value(row, "symbol")),
                    side=self._side(mt5, int(_value(row, "type"))),
                    magic=int(_value(row, "magic", 0) or 0),
                    volume=float(_value(row, "volume")),
                    price_open=float(_value(row, "price_open")),
                    stop_loss=float(_value(row, "sl", 0.0) or 0.0),
                    take_profit=float(_value(row, "tp", 0.0) or 0.0),
                    comment=str(_value(row, "comment", "") or ""),
                )
                for row in rows
            )

    def orders(self, symbol: str) -> tuple[BrokerOrderV1, ...]:
        with MT5_API_LOCK:
            mt5 = self._initialize()
            rows = mt5.orders_get(symbol=symbol)
            if rows is None:
                raise RuntimeError(f"MT5 orders_get 失败 {mt5.last_error()}")
            return tuple(
                BrokerOrderV1(
                    ticket=int(_value(row, "ticket")),
                    symbol=str(_value(row, "symbol")),
                    side=self._side(mt5, int(_value(row, "type"))),
                    magic=int(_value(row, "magic", 0) or 0),
                    volume=float(
                        _value(row, "volume_current", _value(row, "volume_initial", 0.0))
                    ),
                    price_open=float(_value(row, "price_open")),
                    stop_loss=float(_value(row, "sl", 0.0) or 0.0),
                    take_profit=float(_value(row, "tp", 0.0) or 0.0),
                    order_type=self._pending_type(mt5, int(_value(row, "type"))),
                    comment=str(_value(row, "comment", "") or ""),
                )
                for row in rows
            )

    def tick(self, symbol: str) -> BrokerTickV1:
        with MT5_API_LOCK:
            mt5 = self._initialize()
            environment = self._environment(mt5, symbol)
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                raise RuntimeError(f"MT5 未返回 {symbol} 最新报价 {mt5.last_error()}")
            return BrokerTickV1(
                symbol=symbol,
                bid=float(_value(tick, "bid", 0.0) or 0.0),
                ask=float(_value(tick, "ask", 0.0) or 0.0),
                point=environment.point,
            )

    def position_closed_by_take_profit(self, ticket: int) -> bool | None:
        """Return True only when MT5 history identifies a TP close for the position."""

        with MT5_API_LOCK:
            mt5 = self._initialize()
            rows = mt5.history_deals_get(position=ticket)
            if rows is None:
                raise RuntimeError(f"MT5 history_deals_get 失败 {mt5.last_error()}")
            if not rows:
                return None
            tp_reason = getattr(mt5, "DEAL_REASON_TP", 5)
            return any(int(_value(row, "reason", -1)) == tp_reason for row in rows)

    @staticmethod
    def _normalized_price(value: float, *, digits: int) -> float:
        if not math.isfinite(value) or value <= 0:
            raise ValueError("order price must be a positive finite number")
        return round(value, digits)

    @staticmethod
    def _filling_mode(mt5, info: object) -> int:
        execution = int(_value(info, "trade_exemode", 0) or 0)
        market_execution = getattr(mt5, "SYMBOL_TRADE_EXECUTION_MARKET", 2)
        filling = int(_value(info, "filling_mode", 0) or 0)
        if execution != market_execution:
            return getattr(mt5, "ORDER_FILLING_RETURN", 2)
        if filling & getattr(mt5, "SYMBOL_FILLING_IOC", 2):
            return getattr(mt5, "ORDER_FILLING_IOC", 1)
        if filling & getattr(mt5, "SYMBOL_FILLING_FOK", 1):
            return getattr(mt5, "ORDER_FILLING_FOK", 0)
        raise ValueError("broker exposes no supported market filling mode")

    @staticmethod
    def _required_order_flag(mt5, order_type: str) -> int:
        return {
            "market": getattr(mt5, "SYMBOL_ORDER_MARKET", 1),
            "limit": getattr(mt5, "SYMBOL_ORDER_LIMIT", 2),
            "stop": getattr(mt5, "SYMBOL_ORDER_STOP", 4),
            "stop_limit": getattr(mt5, "SYMBOL_ORDER_STOP_LIMIT", 8),
        }[order_type]

    def _build_request(
        self,
        mt5,
        plan: OrderPlanCandidateV1,
        *,
        volume: float,
        take_profit: float,
        leg: str,
        environment: ExecutionEnvironmentV1,
    ) -> dict[str, Any]:
        info = mt5.symbol_info(plan_symbol := "XAUUSD")
        if info is None:
            raise RuntimeError(f"MT5 无法读取 {plan_symbol} 合约信息 {mt5.last_error()}")
        required = self._required_order_flag(mt5, plan.order_type)
        if environment.order_mode & required != required:
            raise ValueError(f"broker does not support order_type={plan.order_type}")
        tick = mt5.symbol_info_tick(plan_symbol)
        if tick is None:
            raise RuntimeError(f"MT5 未返回 {plan_symbol} 最新报价 {mt5.last_error()}")
        bid = float(_value(tick, "bid", 0.0) or 0.0)
        ask = float(_value(tick, "ask", 0.0) or 0.0)
        if bid <= 0 or ask < bid:
            raise RuntimeError(f"MT5 返回的 {plan_symbol} 报价无效")
        digits = environment.digits
        if plan.order_type == "market":
            entry = ask if plan.side == "long" else bid
            action = mt5.TRADE_ACTION_DEAL
            order_type = mt5.ORDER_TYPE_BUY if plan.side == "long" else mt5.ORDER_TYPE_SELL
        elif plan.order_type == "limit":
            entry = float(plan.limit_price if plan.limit_price is not None else plan.entry_price)
            action = mt5.TRADE_ACTION_PENDING
            order_type = (
                mt5.ORDER_TYPE_BUY_LIMIT if plan.side == "long" else mt5.ORDER_TYPE_SELL_LIMIT
            )
        elif plan.order_type == "stop":
            entry = float(plan.trigger_price if plan.trigger_price is not None else plan.entry_price)
            action = mt5.TRADE_ACTION_PENDING
            order_type = (
                mt5.ORDER_TYPE_BUY_STOP if plan.side == "long" else mt5.ORDER_TYPE_SELL_STOP
            )
        else:
            trigger_price = plan.trigger_price
            limit_price = plan.limit_price
            if trigger_price is None or limit_price is None:
                raise ValueError("stop_limit plan requires trigger_price and limit_price")
            entry = float(trigger_price)
            action = mt5.TRADE_ACTION_PENDING
            order_type = (
                mt5.ORDER_TYPE_BUY_STOP_LIMIT
                if plan.side == "long"
                else mt5.ORDER_TYPE_SELL_STOP_LIMIT
            )
        entry = self._normalized_price(entry, digits=digits)
        stop_loss = self._normalized_price(plan.stop_loss, digits=digits)
        target = self._normalized_price(take_profit, digits=digits)
        distance = environment.min_stop_distance
        tolerance = environment.point * 1e-6
        if plan.side == "long":
            if stop_loss > entry - distance + tolerance or target < entry + distance - tolerance:
                raise ValueError("long SL/TP violates broker stop distance")
            if plan.order_type == "limit" and entry > ask - distance + tolerance:
                raise ValueError("buy limit violates broker pending distance")
            if plan.order_type in {"stop", "stop_limit"} and entry < ask + distance - tolerance:
                raise ValueError("buy stop violates broker pending distance")
        else:
            if stop_loss < entry + distance - tolerance or target > entry - distance + tolerance:
                raise ValueError("short SL/TP violates broker stop distance")
            if plan.order_type == "limit" and entry < bid + distance - tolerance:
                raise ValueError("sell limit violates broker pending distance")
            if plan.order_type in {"stop", "stop_limit"} and entry > bid - distance + tolerance:
                raise ValueError("sell stop violates broker pending distance")
        risk = abs(entry - stop_loss)
        reward = abs(target - entry)
        minimum_r = _MINIMUM_R.get(plan.family)
        if minimum_r is None or risk <= 0 or reward / risk + 1e-12 < minimum_r:
            raise ValueError("broker-time price moved the selected plan below its minimum R")
        request: dict[str, Any] = {
            "action": action,
            "symbol": plan_symbol,
            "volume": volume,
            "type": order_type,
            "price": entry,
            "sl": stop_loss,
            "tp": target,
            "deviation": 20,
            "magic": ALPHAMASTER_MAGIC,
            "comment": f"AM5:{leg}:{plan.plan_id[-12:]}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": (
                self._filling_mode(mt5, info)
                if plan.order_type == "market"
                else getattr(mt5, "ORDER_FILLING_RETURN", 2)
            ),
        }
        if plan.order_type == "stop_limit":
            limit_price = plan.limit_price
            if limit_price is None:
                raise ValueError("stop_limit plan requires trigger_price and limit_price")
            request["stoplimit"] = self._normalized_price(float(limit_price), digits=digits)
        return request

    @staticmethod
    def _result_receipt(
        *,
        action: str,
        result: object | None,
        request: dict[str, Any],
        accepted: bool,
        confirmed: bool,
        confirmation_kind: str | None,
        position_ticket: int | None = None,
        executed_price: float | None = None,
    ) -> ExecutionReceiptV1:
        result_price = None if result is None else float(_value(result, "price", 0.0) or 0.0)
        return ExecutionReceiptV1(
            action=action,
            accepted=accepted,
            confirmed=confirmed,
            retcode=None if result is None else int(_value(result, "retcode", -1)),
            message=("MT5 returned no result" if result is None else str(_value(result, "comment", ""))),
            order_ticket=None if result is None else _positive_ticket(_value(result, "order")),
            deal_ticket=None if result is None else _positive_ticket(_value(result, "deal")),
            position_ticket=position_ticket,
            confirmation_kind=confirmation_kind,
            executed_price=(
                executed_price
                if executed_price is not None
                else (result_price if result_price and result_price > 0 else None)
            ),
            request=_request_pairs(request),
        )

    def place_plan(
        self,
        plan: OrderPlanCandidateV1,
        *,
        volume: float,
        take_profit: float,
        leg: str,
    ) -> ExecutionReceiptV1:
        with MT5_API_LOCK:
            mt5 = self._initialize()
            environment = self._environment(mt5, "XAUUSD")
            self._validate_volume(
                volume,
                minimum=environment.volume_min,
                maximum=environment.volume_max,
                step=environment.volume_step,
            )
            request = self._build_request(
                mt5,
                plan,
                volume=volume,
                take_profit=take_profit,
                leg=leg,
                environment=environment,
            )
            checked = mt5.order_check(request)
            if checked is None or int(_value(checked, "retcode", -1)) != 0:
                return self._result_receipt(
                    action="place",
                    result=checked,
                    request=request,
                    accepted=False,
                    confirmed=False,
                    confirmation_kind=None,
                )
            result = mt5.order_send(request)
            accepted_codes = {
                getattr(mt5, "TRADE_RETCODE_PLACED", 10008),
                getattr(mt5, "TRADE_RETCODE_DONE", 10009),
                getattr(mt5, "TRADE_RETCODE_DONE_PARTIAL", 10010),
            }
            accepted = result is not None and int(_value(result, "retcode", -1)) in accepted_codes
            order_ticket = None if result is None else _positive_ticket(_value(result, "order"))
            if not accepted:
                return self._result_receipt(
                    action="place",
                    result=result,
                    request=request,
                    accepted=False,
                    confirmed=False,
                    confirmation_kind=None,
                )
            if plan.order_type == "market":
                rows = mt5.positions_get(symbol="XAUUSD")
                candidates = tuple(rows or ())
                match = next(
                    (
                        row
                        for row in candidates
                        if int(_value(row, "magic", 0) or 0) == ALPHAMASTER_MAGIC
                        and self._side(mt5, int(_value(row, "type"))) == plan.side
                        and str(_value(row, "comment", "")) == request["comment"]
                    ),
                    None,
                )
                if match is None and order_ticket is not None:
                    match = next(
                        (row for row in candidates if int(_value(row, "ticket", 0)) == order_ticket),
                        None,
                    )
                if match is None:
                    matching_system = tuple(
                        row
                        for row in candidates
                        if int(_value(row, "magic", 0) or 0) == ALPHAMASTER_MAGIC
                        and self._side(mt5, int(_value(row, "type"))) == plan.side
                    )
                    if len(matching_system) == 1:
                        match = matching_system[0]
                return self._result_receipt(
                    action="place",
                    result=result,
                    request=request,
                    accepted=True,
                    confirmed=match is not None,
                    confirmation_kind="position" if match is not None else None,
                    position_ticket=(
                        None if match is None else int(_value(match, "ticket"))
                    ),
                    executed_price=(
                        None if match is None else float(_value(match, "price_open"))
                    ),
                )
            rows = () if order_ticket is None else (mt5.orders_get(ticket=order_ticket) or ())
            confirmed = bool(rows)
            return self._result_receipt(
                action="place",
                result=result,
                request=request,
                accepted=True,
                confirmed=confirmed,
                confirmation_kind="order" if confirmed else None,
            )

    def _simple_send(
        self,
        *,
        mt5,
        action: str,
        request: dict[str, Any],
    ) -> tuple[object | None, bool]:
        checked = mt5.order_check(request)
        if checked is None or int(_value(checked, "retcode", -1)) != 0:
            return checked, False
        result = mt5.order_send(request)
        accepted = result is not None and int(_value(result, "retcode", -1)) in {
            getattr(mt5, "TRADE_RETCODE_DONE", 10009),
            getattr(mt5, "TRADE_RETCODE_DONE_PARTIAL", 10010),
        }
        return result, accepted

    def cancel_order(self, order: BrokerOrderV1) -> ExecutionReceiptV1:
        with MT5_API_LOCK:
            mt5 = self._initialize()
            request = {
                "action": mt5.TRADE_ACTION_REMOVE,
                "order": order.ticket,
                "symbol": order.symbol,
                "magic": ALPHAMASTER_MAGIC,
                "comment": "AM5:expire",
            }
            result, accepted = self._simple_send(mt5=mt5, action="cancel", request=request)
            remaining = mt5.orders_get(ticket=order.ticket) if accepted else (order,)
            confirmed = accepted and remaining is not None and len(remaining) == 0
            return self._result_receipt(
                action="cancel",
                result=result,
                request=request,
                accepted=accepted,
                confirmed=confirmed,
                confirmation_kind="cancelled" if confirmed else None,
            )

    def close_position(
        self,
        position: BrokerPositionV1,
        *,
        volume: float,
    ) -> ExecutionReceiptV1:
        with MT5_API_LOCK:
            mt5 = self._initialize()
            environment = self._environment(mt5, position.symbol)
            self._validate_volume(
                volume,
                minimum=environment.volume_min,
                maximum=environment.volume_max,
                step=environment.volume_step,
            )
            tick = mt5.symbol_info_tick(position.symbol)
            if tick is None:
                raise RuntimeError(f"MT5 未返回 {position.symbol} 最新报价 {mt5.last_error()}")
            closing_long = position.side == "long"
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "position": position.ticket,
                "symbol": position.symbol,
                "volume": volume,
                "type": mt5.ORDER_TYPE_SELL if closing_long else mt5.ORDER_TYPE_BUY,
                "price": float(_value(tick, "bid" if closing_long else "ask")),
                "deviation": 20,
                "magic": ALPHAMASTER_MAGIC,
                "comment": "AM5:close",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": self._filling_mode(mt5, mt5.symbol_info(position.symbol)),
            }
            result, accepted = self._simple_send(mt5=mt5, action="close", request=request)
            rows = mt5.positions_get(ticket=position.ticket) if accepted else (position,)
            expected_max = max(0.0, position.volume - volume)
            first = None if rows is None else next(iter(rows), None)
            confirmed = accepted and rows is not None and (
                first is None
                or float(_value(first, "volume", position.volume))
                <= expected_max + environment.volume_step * 1e-6
            )
            return self._result_receipt(
                action="close",
                result=result,
                request=request,
                accepted=accepted,
                confirmed=confirmed,
                confirmation_kind="closed" if confirmed else None,
                position_ticket=position.ticket,
            )

    def modify_position(
        self,
        position: BrokerPositionV1,
        *,
        stop_loss: float,
        take_profit: float,
    ) -> ExecutionReceiptV1:
        with MT5_API_LOCK:
            mt5 = self._initialize()
            environment = self._environment(mt5, position.symbol)
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "position": position.ticket,
                "symbol": position.symbol,
                "sl": self._normalized_price(stop_loss, digits=environment.digits),
                "tp": self._normalized_price(take_profit, digits=environment.digits),
                "magic": ALPHAMASTER_MAGIC,
                "comment": "AM5:breakeven",
            }
            result, accepted = self._simple_send(mt5=mt5, action="modify", request=request)
            rows = mt5.positions_get(ticket=position.ticket) if accepted else ()
            first = next(iter(rows or ()), None)
            confirmed = (
                first is not None
                and math.isclose(
                    float(_value(first, "sl", 0.0)),
                    request["sl"],
                    rel_tol=0.0,
                    abs_tol=environment.point * 0.5,
                )
                and math.isclose(
                    float(_value(first, "tp", 0.0)),
                    request["tp"],
                    rel_tol=0.0,
                    abs_tol=environment.point * 0.5,
                )
            )
            return self._result_receipt(
                action="modify",
                result=result,
                request=request,
                accepted=accepted,
                confirmed=confirmed,
                confirmation_kind="modified" if confirmed else None,
                position_ticket=position.ticket,
            )


mt5_execution_adapter = MT5ExecutionAdapter()
