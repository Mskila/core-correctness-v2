from __future__ import annotations

from collections import namedtuple

import pytest

from trading_core import OrderPlanCandidateV1
from web.trading_execution_mt5 import MT5ExecutionAdapter


Row = namedtuple("Row", "retcode comment order deal volume price")
Position = namedtuple(
    "Position", "ticket symbol type magic volume price_open sl tp comment"
)
Order = namedtuple(
    "Order", "ticket symbol type magic volume_current price_open sl tp comment"
)


def _plan(order_type: str = "market", side: str = "long") -> OrderPlanCandidateV1:
    if side == "long":
        trigger = 2401.0
        limit = 2399.0
        stop_loss = 2398.0
        tp1 = 2406.0 if order_type in {"stop", "stop_limit"} else 2403.0
        tp2 = 2407.0 if order_type in {"stop", "stop_limit"} else 2404.0
    else:
        trigger = 2399.0
        limit = 2401.0
        stop_loss = 2402.0
        tp1 = 2394.0 if order_type in {"stop", "stop_limit"} else 2397.0
        tp2 = 2393.0 if order_type in {"stop", "stop_limit"} else 2396.0
    return OrderPlanCandidateV1(
        plan_id="stage2-plan",
        style="pa_primary",
        side=side,
        family="breakout",
        setup_type="breakout_retest",
        order_type=order_type,
        entry_price=2400.0,
        trigger_price=trigger if order_type in {"stop", "stop_limit"} else None,
        limit_price=limit if order_type in {"limit", "stop_limit"} else None,
        stop_loss=stop_loss,
        take_profit_1=tp1,
        take_profit_2=tp2,
        risk_distance=2.0,
        take_profit_1_r=1.5,
        take_profit_2_r=2.0,
        source_trigger_at=10,
        source_policy="formula_candidate_only",
        policy_origin="pa_formula_candidate",
    )


class _FakeMT5:
    ACCOUNT_MARGIN_MODE_RETAIL_HEDGING = 2
    TRADE_ACTION_DEAL = 1
    TRADE_ACTION_PENDING = 5
    TRADE_ACTION_SLTP = 6
    TRADE_ACTION_REMOVE = 8
    ORDER_TYPE_BUY = 0
    ORDER_TYPE_SELL = 1
    ORDER_TYPE_BUY_LIMIT = 2
    ORDER_TYPE_SELL_LIMIT = 3
    ORDER_TYPE_BUY_STOP = 4
    ORDER_TYPE_SELL_STOP = 5
    ORDER_TYPE_BUY_STOP_LIMIT = 6
    ORDER_TYPE_SELL_STOP_LIMIT = 7
    ORDER_TIME_GTC = 0
    ORDER_FILLING_FOK = 0
    ORDER_FILLING_IOC = 1
    ORDER_FILLING_RETURN = 2
    SYMBOL_FILLING_FOK = 1
    SYMBOL_FILLING_IOC = 2
    SYMBOL_TRADE_EXECUTION_MARKET = 2
    SYMBOL_ORDER_MARKET = 1
    SYMBOL_ORDER_LIMIT = 2
    SYMBOL_ORDER_STOP = 4
    SYMBOL_ORDER_STOP_LIMIT = 8
    TRADE_RETCODE_PLACED = 10008
    TRADE_RETCODE_DONE = 10009
    TRADE_RETCODE_DONE_PARTIAL = 10010
    DEAL_REASON_TP = 5

    def __init__(self, *, mode: int = 2) -> None:
        self.mode = mode
        self.calls: list[tuple[str, dict]] = []
        self.positions: list[Position] = []
        self.orders: list[Order] = []
        self.bid = 2399.9
        self.ask = 2400.0
        self.history_rows = []

    def initialize(self):
        return True

    def last_error(self):
        return (0, "ok")

    def account_info(self):
        return type("Account", (), {
            "login": 123,
            "server": "Demo",
            "margin_mode": self.mode,
            "trade_allowed": True,
        })()

    def terminal_info(self):
        return type("Terminal", (), {"trade_allowed": True, "trade_expert": True})()

    def symbol_select(self, symbol, selected):
        return True

    def symbol_info(self, symbol):
        return type("Symbol", (), {
            "point": 0.01,
            "trade_tick_size": 0.01,
            "digits": 2,
            "trade_stops_level": 10,
            "volume_min": 0.01,
            "volume_max": 10.0,
            "volume_step": 0.01,
            "filling_mode": self.SYMBOL_FILLING_FOK | self.SYMBOL_FILLING_IOC,
            "trade_exemode": self.SYMBOL_TRADE_EXECUTION_MARKET,
            "order_mode": 15,
            "visible": True,
        })()

    def symbol_info_tick(self, symbol):
        return type("Tick", (), {"bid": self.bid, "ask": self.ask})()

    def order_check(self, request):
        self.calls.append(("check", dict(request)))
        return Row(
            0,
            "check ok",
            0,
            0,
            request.get("volume", 0.0),
            request.get("price", 0.0),
        )

    def order_send(self, request):
        self.calls.append(("send", dict(request)))
        if request["action"] == self.TRADE_ACTION_SLTP:
            self.positions = [
                row._replace(sl=request["sl"], tp=request["tp"])
                if row.ticket == request["position"]
                else row
                for row in self.positions
            ]
            position = next(row for row in self.positions if row.ticket == request["position"])
            return Row(
                self.TRADE_RETCODE_DONE,
                "modified",
                position.ticket,
                0,
                position.volume,
                position.price_open,
            )
        if request["action"] == self.TRADE_ACTION_PENDING:
            self.orders = [
                Order(701, request["symbol"], request["type"], request["magic"],
                      request["volume"], request["price"], request["sl"], request["tp"], request["comment"])
            ]
            return Row(self.TRADE_RETCODE_PLACED, "placed", 701, 0, request["volume"], request["price"])
        self.positions = [
            Position(501, request["symbol"], request["type"], request["magic"],
                     request["volume"], request["price"], request["sl"], request["tp"], request["comment"])
        ]
        return Row(self.TRADE_RETCODE_DONE, "done", 501, 601, request["volume"], request["price"])

    def positions_get(self, *, symbol=None, ticket=None):
        if ticket is not None:
            return tuple(row for row in self.positions if row.ticket == ticket)
        return tuple(self.positions)

    def orders_get(self, *, symbol=None, ticket=None):
        if ticket is not None:
            return tuple(row for row in self.orders if row.ticket == ticket)
        return tuple(self.orders)

    def history_deals_get(self, *, position):
        return tuple(self.history_rows)


def test_preflight_rejects_invalid_volume_without_rounding() -> None:
    adapter = MT5ExecutionAdapter(mt5=_FakeMT5())

    with pytest.raises(ValueError, match="volume_step"):
        adapter.preflight("XAUUSD", (0.015,))


def test_market_order_is_checked_sent_and_confirmed_from_position_query() -> None:
    mt5 = _FakeMT5()
    adapter = MT5ExecutionAdapter(mt5=mt5)

    receipt = adapter.place_plan(_plan(), volume=0.02, take_profit=2403.0, leg="tp1")

    assert [name for name, _ in mt5.calls] == ["check", "send"]
    assert receipt.accepted is True
    assert receipt.confirmed is True
    assert receipt.confirmation_kind == "position"
    assert receipt.position_ticket == 501
    request = mt5.calls[-1][1]
    assert request["magic"] == 20250101
    assert request["type"] == mt5.ORDER_TYPE_BUY
    assert request["volume"] == 0.02
    assert request["tp"] == 2403.0


def test_market_order_is_rejected_when_live_quote_degrades_minimum_r() -> None:
    mt5 = _FakeMT5()
    mt5.bid = 2400.9
    mt5.ask = 2401.0
    adapter = MT5ExecutionAdapter(mt5=mt5)

    with pytest.raises(ValueError, match="minimum R"):
        adapter.place_plan(_plan(), volume=0.02, take_profit=2403.0, leg="tp1")

    assert mt5.calls == []


def test_pending_stop_is_confirmed_only_after_active_order_query() -> None:
    mt5 = _FakeMT5()
    adapter = MT5ExecutionAdapter(mt5=mt5)
    plan = _plan("stop")

    receipt = adapter.place_plan(
        plan, volume=0.01, take_profit=plan.take_profit_1, leg="tp1"
    )

    assert receipt.accepted is True
    assert receipt.confirmed is True
    assert receipt.confirmation_kind == "order"
    assert receipt.order_ticket == 701
    assert mt5.calls[-1][1]["action"] == mt5.TRADE_ACTION_PENDING
    assert mt5.calls[-1][1]["type"] == mt5.ORDER_TYPE_BUY_STOP
    assert mt5.calls[-1][1]["type_filling"] == mt5.ORDER_FILLING_RETURN


def test_tp1_close_requires_take_profit_reason_from_mt5_history() -> None:
    mt5 = _FakeMT5()
    adapter = MT5ExecutionAdapter(mt5=mt5)

    assert adapter.position_closed_by_take_profit(501) is None
    mt5.history_rows = [type("Deal", (), {"reason": 3})()]
    assert adapter.position_closed_by_take_profit(501) is False
    mt5.history_rows.append(type("Deal", (), {"reason": mt5.DEAL_REASON_TP})())
    assert adapter.position_closed_by_take_profit(501) is True


def test_modify_position_requires_both_stop_loss_and_take_profit_query_confirmation() -> None:
    class _SlOnlyMT5(_FakeMT5):
        def order_send(self, request):
            if request["action"] != self.TRADE_ACTION_SLTP:
                return super().order_send(request)
            self.calls.append(("send", dict(request)))
            self.positions = [
                row._replace(sl=request["sl"])
                if row.ticket == request["position"]
                else row
                for row in self.positions
            ]
            position = next(row for row in self.positions if row.ticket == request["position"])
            return Row(
                self.TRADE_RETCODE_DONE,
                "modified",
                position.ticket,
                0,
                position.volume,
                position.price_open,
            )

    mt5 = _SlOnlyMT5()
    mt5.positions = [
        Position(501, "XAUUSD", mt5.ORDER_TYPE_BUY, 20250101, 0.01, 2400.0, 2398.0, 2403.0, "AM5:tp2")
    ]
    adapter = MT5ExecutionAdapter(mt5=mt5)

    receipt = adapter.modify_position(
        adapter.positions("XAUUSD")[0],
        stop_loss=2400.1,
        take_profit=2404.0,
    )

    assert receipt.accepted is True
    assert receipt.confirmed is False
