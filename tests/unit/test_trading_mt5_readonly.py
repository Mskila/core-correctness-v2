from __future__ import annotations

from collections import namedtuple

from web.data_sources.base import Bar
import web.trading_mt5 as trading_mt5


class _FakeMT5:
    ACCOUNT_TRADE_MODE_DEMO = 0
    ACCOUNT_TRADE_MODE_CONTEST = 1
    ACCOUNT_TRADE_MODE_REAL = 2
    ACCOUNT_MARGIN_MODE_RETAIL_HEDGING = 2

    def initialize(self):
        return True

    def last_error(self):
        return (0, "ok")

    def account_info(self):
        Account = namedtuple(
            "Account",
            "login server company currency balance equity trade_mode margin_mode trade_allowed",
        )
        return Account(123456, "Demo-Server", "Broker", "USD", 1000.0, 1005.0, 0, 2, True)

    def terminal_info(self):
        Terminal = namedtuple("Terminal", "name trade_allowed trade_expert")
        return Terminal("Terminal", True, True)

    def symbol_select(self, symbol, selected):
        return symbol == "XAUUSD" and selected

    def symbol_info(self, symbol):
        Symbol = namedtuple("Symbol", "point trade_stops_level")
        return Symbol(0.01, 20) if symbol == "XAUUSD" else None


class _FakeSource:
    def fetch_bars(self, symbol, timeframe, count, drop_forming=True):
        assert (symbol, timeframe, drop_forming) == ("XAUUSD", "15m", True)
        return [
            Bar(ts=0, open=2400.0, high=2401.0, low=2399.0, close=2400.5, volume=10.0),
            Bar(ts=900, open=2400.5, high=2402.0, low=2400.0, close=2401.0, volume=11.0),
        ][-count:]


def test_account_snapshot_reports_account_and_position_modes_without_execution(
    monkeypatch,
) -> None:
    market = trading_mt5.MT5ReadOnlyMarket()
    monkeypatch.setattr(market, "_module", lambda: _FakeMT5())

    account = market.account_snapshot()

    assert account["connected"] is True
    assert account["account_kind"] == "demo"
    assert account["position_mode"] == "hedging"
    assert account["trade_allowed"] is True
    assert account["read_only"] is True
    assert account["execution_enabled"] is False
    assert not hasattr(market, "order_send")


def test_closed_bars_and_numeric_constraints_are_read_only_inputs(monkeypatch) -> None:
    market = trading_mt5.MT5ReadOnlyMarket()
    monkeypatch.setattr(market, "_module", lambda: _FakeMT5())
    monkeypatch.setattr(trading_mt5, "get_source", lambda kind: _FakeSource())

    constraints = market.instrument_constraints("XAUUSD")
    series = market.fetch_series("XAUUSD", "M15", 2)

    assert constraints.tick == 0.01
    assert constraints.min_stop_distance == 0.2
    assert constraints.min_pending_distance == 0.2
    assert series.timeframe == "M15"
    assert tuple(bar.timestamp for bar in series.bars) == (0, 900)
    assert all(bar.closed is True for bar in series.bars)
