"""Read-only MT5 boundary used by the Stage-4 trading preview."""

from __future__ import annotations

from typing import Any

from trading_core import ClosedBarV1, InstrumentConstraintsV1, TimeframeSeriesV1
from web.data_sources.factory import get_source
from web.data_sources.mt5_source import MT5_API_LOCK


_SOURCE_TIMEFRAMES = {"H1": "1h", "M15": "15m", "M5": "5m"}


def _namedtuple_dict(value: object) -> dict[str, Any]:
    as_dict = getattr(value, "_asdict", None)
    return dict(as_dict()) if callable(as_dict) else {}


class MT5ReadOnlyMarket:
    """Fetch account metadata and closed bars without exposing order methods."""

    def _module(self):
        try:
            import MetaTrader5 as mt5
        except ImportError as exc:
            raise RuntimeError("未安装 MetaTrader5：pip install -r requirements-mt5.txt") from exc
        return mt5

    def _initialize(self):
        mt5 = self._module()
        if not mt5.initialize():
            raise RuntimeError(f"MT5 初始化失败 {mt5.last_error()}；请确认终端已打开并登录")
        return mt5

    def account_snapshot(self) -> dict[str, Any]:
        """Return a display-only account snapshot; failures remain normal data."""

        try:
            with MT5_API_LOCK:
                mt5 = self._initialize()
                account = mt5.account_info()
                terminal = mt5.terminal_info()
                if account is None or terminal is None:
                    raise RuntimeError(f"MT5 未返回账户信息 {mt5.last_error()}")
                account_data = _namedtuple_dict(account)
                terminal_data = _namedtuple_dict(terminal)
                trade_mode = account_data.get("trade_mode")
                margin_mode = account_data.get("margin_mode")
                demo_value = getattr(mt5, "ACCOUNT_TRADE_MODE_DEMO", 0)
                contest_value = getattr(mt5, "ACCOUNT_TRADE_MODE_CONTEST", 1)
                real_value = getattr(mt5, "ACCOUNT_TRADE_MODE_REAL", 2)
                hedging_value = getattr(mt5, "ACCOUNT_MARGIN_MODE_RETAIL_HEDGING", 2)
                account_kind = {
                    demo_value: "demo",
                    contest_value: "contest",
                    real_value: "real",
                }.get(trade_mode, "unknown")
                position_mode = "hedging" if margin_mode == hedging_value else "netting"
                return {
                    "available": True,
                    "connected": True,
                    "read_only": True,
                    "execution_enabled": False,
                    "login": account_data.get("login"),
                    "server": account_data.get("server") or terminal_data.get("name"),
                    "company": account_data.get("company"),
                    "currency": account_data.get("currency"),
                    "balance": account_data.get("balance"),
                    "equity": account_data.get("equity"),
                    "account_kind": account_kind,
                    "position_mode": position_mode,
                    "trade_allowed": bool(
                        account_data.get("trade_allowed", True)
                        and terminal_data.get("trade_allowed", False)
                    ),
                    "trade_expert": bool(terminal_data.get("trade_expert", False)),
                    "message": "只读连接；阶段 4 不会发送订单",
                }
        except Exception as exc:  # noqa: BLE001 - status endpoint returns display state
            return {
                "available": False,
                "connected": False,
                "read_only": True,
                "execution_enabled": False,
                "login": None,
                "server": None,
                "company": None,
                "currency": None,
                "balance": None,
                "equity": None,
                "account_kind": "unknown",
                "position_mode": "unknown",
                "trade_allowed": False,
                "trade_expert": False,
                "message": str(exc),
            }

    def _symbol_info(self, symbol: str) -> dict[str, Any]:
        with MT5_API_LOCK:
            mt5 = self._initialize()
            mt5.symbol_select(symbol, True)
            info = mt5.symbol_info(symbol)
            if info is None:
                raise RuntimeError(f"MT5 无法读取 {symbol} 合约信息 {mt5.last_error()}")
            return _namedtuple_dict(info)

    def instrument_constraints(self, symbol: str) -> InstrumentConstraintsV1:
        info = self._symbol_info(symbol)
        point = float(info.get("point") or 0.0)
        if point <= 0.0:
            raise RuntimeError(f"MT5 返回的 {symbol} point 无效")
        stops_level = max(0, int(info.get("trade_stops_level") or 0))
        return InstrumentConstraintsV1(
            tick=point,
            min_stop_distance=stops_level * point,
            min_pending_distance=stops_level * point,
            supports_stop_limit=False,
        )

    def fetch_series(
        self,
        symbol: str,
        timeframe: str,
        count: int,
    ) -> TimeframeSeriesV1:
        try:
            source_timeframe = _SOURCE_TIMEFRAMES[timeframe]
        except KeyError as exc:
            raise ValueError(f"unsupported trading timeframe: {timeframe}") from exc
        if type(count) is not int or count < 1:
            raise ValueError("count must be a positive integer")
        point = self.instrument_constraints(symbol).tick
        bars = get_source("mt5").fetch_bars(
            symbol,
            source_timeframe,
            count,
            drop_forming=True,
        )
        return TimeframeSeriesV1(
            symbol=symbol,
            timeframe=timeframe,
            tick=point,
            bars=tuple(
                ClosedBarV1(
                    timestamp=int(bar.ts),
                    open=float(bar.open),
                    high=float(bar.high),
                    low=float(bar.low),
                    close=float(bar.close),
                    volume=max(0.0, float(bar.volume)),
                )
                for bar in bars
            ),
        )


mt5_read_only_market = MT5ReadOnlyMarket()
