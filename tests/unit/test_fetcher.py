"""
tests/unit/test_fetcher.py — MT5DataFetcher 边界条件单元测试

验证需求：
  - Req 2.2: mt5.initialize() 返回 False 时抛出 ConnectionError
  - Req 2.7: symbol 不可用（copy_rates_from_pos 返回 None 或空列表）时记录 WARNING 并返回空 DataFrame
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch


# ── 辅助函数 ──────────────────────────────────────────────────────────────────

def _make_mt5_mock(**kwargs) -> MagicMock:
    """创建一个预配置的 mt5 MagicMock，接受任意关键字覆盖默认值。"""
    mock = MagicMock()
    # 默认值：成功连接、copy_rates_from_pos 返回 None
    mock.initialize.return_value = kwargs.get("initialize", True)
    mock.last_error.return_value = kwargs.get("last_error", (0, "OK"))
    mock.copy_rates_from_pos.return_value = kwargs.get("copy_rates_from_pos", None)
    mock.shutdown.return_value = None
    return mock


V2_SCHEMA = ("time", "open", "high", "low", "close", "volume")


# ── 测试 1：mt5.initialize() 返回 False → 抛出 ConnectionError ────────────────

class TestConnectRaisesOnFailure:
    """Req 2.2: 连接失败时应抛出 ConnectionError。"""

    def test_connection_error_when_initialize_false(self):
        """当 mt5.initialize() 返回 False 时，connect() 必须抛出 ConnectionError。"""
        mt5_mock = _make_mt5_mock(initialize=False, last_error=(1, "Test error"))

        with patch("data_pipeline.fetcher.mt5", mt5_mock), \
             patch("data_pipeline.fetcher._MT5_AVAILABLE", True):

            from data_pipeline.fetcher import MT5DataFetcher
            fetcher = MT5DataFetcher()

            with pytest.raises(ConnectionError) as exc_info:
                fetcher.connect()

        # 错误消息应包含 MT5 错误详情
        assert "MT5 connection failed" in str(exc_info.value)

    def test_connection_error_message_contains_last_error(self):
        """ConnectionError 消息应包含 mt5.last_error() 返回的错误信息。"""
        error_tuple = (5, "Terminal not found")
        mt5_mock = _make_mt5_mock(initialize=False, last_error=error_tuple)

        with patch("data_pipeline.fetcher.mt5", mt5_mock), \
             patch("data_pipeline.fetcher._MT5_AVAILABLE", True):

            from data_pipeline.fetcher import MT5DataFetcher
            fetcher = MT5DataFetcher()

            with pytest.raises(ConnectionError) as exc_info:
                fetcher.connect()

        assert "Terminal not found" in str(exc_info.value) or \
               str(error_tuple) in str(exc_info.value)


# ── 测试 2：空 MT5 结果 → 固定 V2 schema 的空 DataFrame ─────────────────────

@pytest.mark.parametrize(
    ("rates", "symbol", "timeframe", "count"),
    [
        (None, "NOSUCHSYMBOL", 1, 100),
        (None, "XAUUSD", 16385, 500),
        ([], "NOSUCHSYMBOL", 1, 100),
        (np.array([]), "EURUSD", 16385, 200),
        ([], "US500", 16408, 1000),
    ],
    ids=("none-m1", "none-h1", "empty-list-m1", "empty-array-h1", "empty-list-d1"),
)
def test_empty_mt5_results_return_exact_v2_schema(
    rates,
    symbol: str,
    timeframe: int,
    count: int,
) -> None:
    mt5_mock = _make_mt5_mock(copy_rates_from_pos=rates)

    with (
        patch("data_pipeline.fetcher.mt5", mt5_mock),
        patch("data_pipeline.fetcher._MT5_AVAILABLE", True),
        patch("data_pipeline.kline_cache.KlineCache.get", return_value=None),
    ):
        from data_pipeline.fetcher import MT5DataFetcher

        fetcher = MT5DataFetcher()
        fetcher.connect()
        frame = fetcher.fetch(symbol, timeframe, count)

    assert isinstance(frame, pd.DataFrame)
    assert frame.empty
    assert tuple(frame.columns) == V2_SCHEMA
    assert "tick_volume" not in frame.columns
    mt5_mock.copy_rates_from_pos.assert_called_once_with(
        symbol, timeframe, 1, count
    )
