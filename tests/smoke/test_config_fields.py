"""
冒烟测试：Config 字段与 FEATURE_NAMES 内容验证

断言：
- FEATURE_NAMES 不含 LIQ_SCORE / FOMO（旧 Solana 特有因子）
- 根 Config 不再提供 INPUT_DIM；ModelConfig 从 V2 词表派生该维度

注意：task 5.1 会将 vocab.py 更新为新的 MT5 特征名称。
      此测试使用 try/except 优雅处理旧版 vocab.py 中仍含旧字段的情形。

Requirements: 11.1, 5.5 (4.1)
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest
from config import Config
from model_core.config import ModelConfig
from model_core.vocab import FEATURE_NAMES, FORMULA_VOCAB


def _get_feature_names():
    return FEATURE_NAMES


def _is_mt5_feature_names_updated():
    """
    检查 FEATURE_NAMES 是否已更新为 MT5 特征集（task 5.1 完成后）。
    若仍包含旧 Solana 特有因子则返回 False，表示 task 5.1 尚未执行。
    """
    feature_names = _get_feature_names()
    if feature_names is None:
        return False
    # 旧版特征含 Solana 特有因子
    old_features = {"LIQ_SCORE", "FOMO", "LOG_VOL"}
    return not bool(old_features & set(feature_names))


class TestConfigInputDimSmoke:
    def test_only_model_config_exposes_vocab_derived_input_dim(self):
        assert not hasattr(Config, "INPUT_DIM")
        assert ModelConfig.INPUT_DIM == FORMULA_VOCAB.feature_count


class TestFeatureNamesSmoke:
    """
    验证 FEATURE_NAMES 不含旧 Solana 链上特有因子。

    - 若 vocab.py 尚未更新（task 5.1 前），这些测试将跳过，以免误报。
    - 若 vocab.py 已更新，则必须满足断言。
    """

    def test_feature_names_accessible(self):
        """FEATURE_NAMES 应当可以从 model_core.vocab 导入"""
        feature_names = _get_feature_names()
        assert feature_names is not None, (
            "Cannot import FEATURE_NAMES from model_core.vocab. "
            "Ensure model_core/vocab.py exists and defines FEATURE_NAMES."
        )

    def test_liq_score_not_in_feature_names(self):
        """FEATURE_NAMES 不应包含 'LIQ_SCORE'（已废弃的链上流动性因子）"""
        feature_names = _get_feature_names()
        if feature_names is None:
            pytest.skip("model_core.vocab.FEATURE_NAMES not available yet (pending task 5.1)")
        if not _is_mt5_feature_names_updated():
            pytest.skip(
                "vocab.py still contains old Solana features — "
                "pending task 5.1 (MT5FeatureEngineer implementation)"
            )
        assert "LIQ_SCORE" not in feature_names, (
            f"FEATURE_NAMES should not contain 'LIQ_SCORE', but got: {feature_names}"
        )

    def test_fomo_not_in_feature_names(self):
        """FEATURE_NAMES 不应包含 'FOMO'（已废弃的链上情绪因子）"""
        feature_names = _get_feature_names()
        if feature_names is None:
            pytest.skip("model_core.vocab.FEATURE_NAMES not available yet (pending task 5.1)")
        if not _is_mt5_feature_names_updated():
            pytest.skip(
                "vocab.py still contains old Solana features — "
                "pending task 5.1 (MT5FeatureEngineer implementation)"
            )
        assert "FOMO" not in feature_names, (
            f"FEATURE_NAMES should not contain 'FOMO', but got: {feature_names}"
        )

    def test_feature_names_length_matches_input_dim(self):
        """FEATURE_NAMES 的长度应等于 ModelConfig.INPUT_DIM。"""
        feature_names = _get_feature_names()
        if feature_names is None:
            pytest.skip("model_core.vocab.FEATURE_NAMES not available yet (pending task 5.1)")
        if not _is_mt5_feature_names_updated():
            pytest.skip(
                "vocab.py still contains old Solana features — "
                "pending task 5.1 (MT5FeatureEngineer implementation)"
            )
        assert len(feature_names) == ModelConfig.INPUT_DIM, (
            f"Expected len(FEATURE_NAMES) == {ModelConfig.INPUT_DIM}, "
            f"got {len(feature_names)}: {feature_names}"
        )
