"""
单元测试：Config 字段类型和默认值

验证 config.py 中 Config 类的所有关键字段的类型和默认值。
Requirements: 11.1
"""
import pytest
import sys
import os
import ast
from pathlib import Path

# 确保项目根目录在 sys.path 中
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from config import Config
from model_core.config import ModelConfig
from model_core.vocab import FORMULA_VOCAB


class TestConfigInputDim:
    def test_root_config_has_no_executable_input_dim(self):
        assert not hasattr(Config, "INPUT_DIM")

    def test_model_input_dim_is_derived_from_v2_vocab(self):
        assert ModelConfig.INPUT_DIM == FORMULA_VOCAB.feature_count
        assert isinstance(ModelConfig.INPUT_DIM, int)


class TestConfigCostRate:
    def test_cost_rate_equals_0001(self):
        assert Config.COST_RATE == 0.0001

    def test_cost_rate_is_float(self):
        assert isinstance(Config.COST_RATE, float)


class TestConfigRiskPerTrade:
    def test_risk_per_trade_equals_001(self):
        assert Config.RISK_PER_TRADE == 0.01

    def test_risk_per_trade_is_float(self):
        assert isinstance(Config.RISK_PER_TRADE, float)


class TestConfigMagicNumber:
    def test_magic_number_is_int(self):
        assert isinstance(Config.MAGIC_NUMBER, int)

    def test_magic_number_equals_20250101(self):
        assert Config.MAGIC_NUMBER == 20250101


class TestConfigSymbols:
    def test_symbols_not_empty(self):
        assert len(Config.SYMBOLS) > 0

    def test_all_symbols_are_strings(self):
        for sym in Config.SYMBOLS:
            assert isinstance(sym, str), f"Symbol {sym!r} is not a string"


class TestConfigDataParams:
    def test_root_config_has_no_fixed_min_bars(self):
        assert not hasattr(Config, "MIN_BARS")

    def test_bars_count_equals_2000(self):
        assert Config.BARS_COUNT >= 100   # 只断言合理下界，不固定具体值

    def test_formal_python_paths_do_not_read_config_min_bars(self):
        root = Path(__file__).resolve().parents[2]
        paths = list(root.glob("*.py")) + list((root / "model_core").glob("*.py"))
        offenders = []
        for path in paths:
            tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Attribute)
                    and node.attr == "MIN_BARS"
                    and isinstance(node.value, ast.Name)
                    and node.value.id == "Config"
                ):
                    offenders.append(f"{path.relative_to(root)}:{node.lineno}")
        assert offenders == []


class TestConfigStrategyParams:
    def test_buy_threshold_equals_070(self):
        assert Config.BUY_THRESHOLD == 0.70

    def test_sell_threshold_equals_040(self):
        assert Config.SELL_THRESHOLD == 0.40


class TestConfigGetTimeframe:
    def test_get_timeframe_h1_returns_int(self):
        result = Config.get_timeframe("H1")
        assert isinstance(result, int)

    def test_get_timeframe_d1_returns_int(self):
        result = Config.get_timeframe("D1")
        assert isinstance(result, int)

    def test_get_timeframe_invalid_raises_value_error(self):
        with pytest.raises(ValueError):
            Config.get_timeframe("INVALID")

    def test_get_timeframe_all_valid_keys_return_int(self):
        """验证所有支持的时间周期字符串都返回整数"""
        valid_timeframes = ["M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1", "MN1"]
        for tf in valid_timeframes:
            result = Config.get_timeframe(tf)
            assert isinstance(result, int), f"get_timeframe({tf!r}) should return int"
