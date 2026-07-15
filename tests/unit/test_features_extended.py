"""
单元测试：MT5FeatureEngineer 扩展特征验证

覆盖 compute_features 的形状、数值安全性、各特征的值域与前缀约束。

需求：F1.1~F1.7, F4.1~F4.4
"""
import ast
from pathlib import Path

import pytest
import torch
import sys
import os

# 确保项目根目录在 sys.path 中
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import model_core.features as features_module
from model_core.features import FEATURE_NAMES, FEATURE_REGISTRY, MT5FeatureEngineer
from model_core.registry import FeatureSpec, RegistrationError, Registry


# ─── 测试用 OHLCV fixture ─────────────────────────────────────────────────────

def _make_raw_dict(N: int = 3, T: int = 50, seed: int = 42) -> dict:
    """
    生成合法的随机 OHLCV 字典。
    - close, open : rand(N, T) + 1.0  → 正值，范围 [1, 2)
    - high        : max(close, open) + rand * 0.5  → high >= close, open
    - low         : min(close, open) - rand * 0.5  → low  <= close, open（但 > 0）
    - volume      : rand(N, T) * 100 + 1.0  → 正值
    """
    torch.manual_seed(seed)
    close  = torch.rand(N, T) + 1.0
    open_  = torch.rand(N, T) + 1.0
    noise  = torch.rand(N, T) * 0.5
    high   = torch.maximum(close, open_) + noise
    low    = (torch.minimum(close, open_) - noise).clamp(min=1e-3)
    volume = torch.rand(N, T) * 100.0 + 1.0
    return {
        "close":  close,
        "open":   open_,
        "high":   high,
        "low":    low,
        "volume": volume,
    }


# ─── 1. 输出形状 ──────────────────────────────────────────────────────────────

class TestComputeFeaturesShape:
    """compute_features 输出形状应为 [N, F, T]。"""

    def test_output_shape_default(self):
        raw = _make_raw_dict(N=3, T=50)
        out = MT5FeatureEngineer.compute_features(raw)
        assert out.shape == (3, len(FEATURE_NAMES), 50), (
            f"Expected shape (3, {len(FEATURE_NAMES)}, 50), got {tuple(out.shape)}"
        )

    def test_output_ndim(self):
        raw = _make_raw_dict(N=3, T=50)
        out = MT5FeatureEngineer.compute_features(raw)
        assert out.ndim == 3

    def test_feature_dim_matches_registry(self):
        raw = _make_raw_dict(N=3, T=50)
        out = MT5FeatureEngineer.compute_features(raw)
        assert out.shape[1] == len(FEATURE_REGISTRY.feature_names)

    def test_time_dim_preserved(self):
        """T 维度应与输入完全一致（需求 F1.1）"""
        raw = _make_raw_dict(N=3, T=50)
        out = MT5FeatureEngineer.compute_features(raw)
        assert out.shape[2] == 50


# ─── 2. 数值安全：无 NaN / Inf ────────────────────────────────────────────────

class TestNoNanInf:
    """输出中不应含有 NaN 或 Inf（需求 F4.3~F4.6）"""

    def test_no_nan(self):
        raw = _make_raw_dict(N=3, T=50)
        out = MT5FeatureEngineer.compute_features(raw)
        assert not torch.isnan(out).any(), "Output contains NaN values"

    def test_no_inf(self):
        raw = _make_raw_dict(N=3, T=50)
        out = MT5FeatureEngineer.compute_features(raw)
        assert not torch.isinf(out).any(), "Output contains Inf values"

    def test_no_nan_inf_combined(self):
        raw = _make_raw_dict(N=3, T=50)
        out = MT5FeatureEngineer.compute_features(raw)
        assert torch.isfinite(out).all(), "Output contains non-finite values (NaN or Inf)"


# ─── 3. PRESSURE（索引 3）值域 ∈ [-1, 1] ─────────────────────────────────────

class TestPressureRange:
    """PRESSURE（索引 12）所有值应被 clamp 至 [-1, 1]（需求 F4.1）"""

    def test_pressure_leq_1(self):
        raw = _make_raw_dict(N=3, T=50)
        out = MT5FeatureEngineer.compute_features(raw)
        pressure = out[:, FEATURE_NAMES.index("PRESSURE"), :]
        assert (pressure <= 1.0).all(), (
            f"PRESSURE has values > 1.0, max={pressure.max().item():.4f}"
        )

    def test_pressure_geq_neg1(self):
        raw = _make_raw_dict(N=3, T=50)
        out = MT5FeatureEngineer.compute_features(raw)
        pressure = out[:, FEATURE_NAMES.index("PRESSURE"), :]
        assert (pressure >= -1.0).all(), (
            f"PRESSURE has values < -1.0, min={pressure.min().item():.4f}"
        )

    def test_pressure_range_strict(self):
        """一次性验证 [-1, 1] 双侧边界（需求 F4.1）"""
        raw = _make_raw_dict(N=3, T=50)
        out = MT5FeatureEngineer.compute_features(raw)
        pressure = out[:, FEATURE_NAMES.index("PRESSURE"), :]
        assert pressure.abs().max().item() <= 1.0 + 1e-6, (
            "PRESSURE violates [-1, 1] bound"
        )


# ─── 4. ATR 原始值非负（在 log1p 前）────────────────────────────────────────

class TestAtrRawNonNegative:
    """_atr 原始输出（log1p 压缩前）应全部非负（需求 F1.3）"""

    def test_atr_raw_nonnegative(self):
        raw = _make_raw_dict(N=3, T=50)
        close  = raw["close"].float()
        high   = raw["high"].float()
        low    = raw["low"].float()
        atr_raw = MT5FeatureEngineer._atr(close, high, low)
        assert (atr_raw >= 0).all(), (
            f"ATR raw has negative values, min={atr_raw.min().item():.6f}"
        )

    def test_atr_raw_shape(self):
        raw = _make_raw_dict(N=3, T=50)
        close  = raw["close"].float()
        high   = raw["high"].float()
        low    = raw["low"].float()
        atr_raw = MT5FeatureEngineer._atr(close, high, low)
        assert atr_raw.shape == (3, 50)

    def test_atr_raw_no_nan(self):
        raw = _make_raw_dict(N=3, T=50)
        close  = raw["close"].float()
        high   = raw["high"].float()
        low    = raw["low"].float()
        atr_raw = MT5FeatureEngineer._atr(close, high, low)
        assert not torch.isnan(atr_raw).any(), "ATR raw contains NaN"


# ─── 5. RET20（索引 8）前 20 个位置应为 0 ────────────────────────────────────

class TestRet20PrefixZero:
    """RET20 原始输出的前 20 个时间步应等于 0（需求 F1.5）"""

    def test_ret20_raw_first20_are_zero(self):
        raw = _make_raw_dict(N=3, T=50)
        close  = raw["close"].float()
        ret20_raw = MT5FeatureEngineer._ret20(close)
        prefix = ret20_raw[:, :20]
        assert (prefix == 0.0).all(), (
            f"RET20 raw first 20 positions are not all zero; "
            f"max abs = {prefix.abs().max().item():.6f}"
        )

    def test_ret20_raw_shape(self):
        raw = _make_raw_dict(N=3, T=50)
        close = raw["close"].float()
        ret20_raw = MT5FeatureEngineer._ret20(close)
        assert ret20_raw.shape == (3, 50)

    def test_ret20_after_position20_nonzero(self):
        """位置 20 之后至少部分值应非零（验证计算逻辑未全零化）"""
        raw = _make_raw_dict(N=3, T=50)
        close = raw["close"].float()
        ret20_raw = MT5FeatureEngineer._ret20(close)
        suffix = ret20_raw[:, 20:]
        assert suffix.abs().max().item() > 0.0, (
            "RET20 suffix (positions 20+) is unexpectedly all zero"
        )


class TestFeatureLookbacks:
    def test_input_dim_is_only_derived_after_registry_construction(self):
        source = Path(features_module.__file__).read_text(encoding="utf-8-sig")
        tree = ast.parse(source)
        engineer = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "MT5FeatureEngineer"
        )
        class_body_assignments = []
        for statement in engineer.body:
            if isinstance(statement, ast.Assign):
                class_body_assignments.extend(
                    target
                    for target in statement.targets
                    if isinstance(target, ast.Name) and target.id == "INPUT_DIM"
                )
            elif (
                isinstance(statement, ast.AnnAssign)
                and isinstance(statement.target, ast.Name)
                and statement.target.id == "INPUT_DIM"
            ):
                class_body_assignments.append(statement.target)

        assert class_body_assignments == [], (
            "MT5FeatureEngineer must not contain a class-body INPUT_DIM literal"
        )
        assert MT5FeatureEngineer.INPUT_DIM == len(FEATURE_REGISTRY.feature_names)

    def test_all_registered_features_have_positive_integer_lookback(self):
        assert FEATURE_REGISTRY.feature_specs
        for spec in FEATURE_REGISTRY.feature_specs:
            assert isinstance(getattr(spec, "lookback", None), int)
            assert not isinstance(spec.lookback, bool)
            assert spec.lookback >= 1
        assert getattr(features_module, "MAX_FEATURE_LOOKBACK", 0) >= 200

    @pytest.mark.parametrize("lookback", [None, True, False, 0, -1])
    def test_invalid_lookback_does_not_mutate_registry(self, lookback):
        registry = Registry()
        before = registry.feature_specs

        def compute(raw: dict) -> torch.Tensor:
            return raw["close"]

        with pytest.raises(RegistrationError):
            registry.register_feature(
                FeatureSpec(
                    name="TEST_FEATURE",
                    category="test",
                    compute=compute,
                    lookback=lookback,
                )
            )
        assert registry.feature_specs == before

    def test_missing_lookback_cannot_mutate_registry(self):
        registry = Registry()

        def compute(raw: dict) -> torch.Tensor:
            return raw["close"]

        with pytest.raises(TypeError):
            FeatureSpec(name="TEST_FEATURE", category="test", compute=compute)
        assert registry.feature_specs == ()

    @staticmethod
    def _spec(name):
        return next(
            spec for spec in FEATURE_REGISTRY.feature_specs if spec.name == name
        )

    @pytest.mark.parametrize(
        ("name", "lookback"),
        [
            ("ULT_OSC", 29),
            ("OBV_SLOPE", 220),
            ("AD_LINE_SLOPE", 219),
            ("SUPERTREND_DIR", 400),
            ("SAR_DIST", 545),
        ],
    )
    def test_cumulative_and_recursive_feature_lookbacks(self, name, lookback):
        assert self._spec(name).lookback == lookback

    @pytest.mark.parametrize(
        ("name", "lookback"),
        [
            ("ULT_OSC", 29),
            ("OBV_SLOPE", 220),
            ("AD_LINE_SLOPE", 219),
            ("SAR_DIST", 545),
        ],
    )
    def test_declared_feature_window_boundary(self, name, lookback):
        spec = self._spec(name)
        raw = _make_raw_dict(N=1, T=lookback + 1, seed=97)
        target = lookback
        baseline = spec.compute(raw)[0, target]

        def variants(index):
            for field in raw:
                changed = {key: value.clone() for key, value in raw.items()}
                changed[field][:, index] = changed[field][:, index] * 7.0 + 11.0
                yield changed

        for changed in variants(target - lookback):
            torch.testing.assert_close(
                spec.compute(changed)[0, target], baseline, rtol=0, atol=0
            )
        assert any(
            not torch.equal(spec.compute(changed)[0, target], baseline)
            for changed in variants(target - lookback + 1)
        ), f"{name} ignores the earliest bar inside its declared window"

    def test_supertrend_declared_window_boundary(self):
        spec = self._spec("SUPERTREND_DIR")
        lookback = 400
        target = lookback

        outside_base = {
            "open": torch.full((1, target + 1), 100.0),
            "high": torch.full((1, target + 1), 101.0),
            "low": torch.full((1, target + 1), 99.0),
            "close": torch.full((1, target + 1), 100.0),
            "volume": torch.ones(1, target + 1),
        }
        outside = {key: value.clone() for key, value in outside_base.items()}
        outside["high"][:, 0] = 110.0
        outside["low"][:, 0] = 110.0
        torch.testing.assert_close(
            spec.compute(outside)[0, target],
            spec.compute(outside_base)[0, target],
            rtol=0,
            atol=0,
        )

        inside_base = {key: value.clone() for key, value in outside_base.items()}
        inside_base["close"][:, 16] = 96.0
        inside = {key: value.clone() for key, value in inside_base.items()}
        inside["close"][:, 1] = 1_000.0
        assert not torch.equal(
            spec.compute(inside)[0, target],
            spec.compute(inside_base)[0, target],
        )


class TestWilliamsR:
    def test_exact_approved_negative_unit_scale(self):
        high = torch.full((1, 14), 10.0, dtype=torch.float64)
        low = torch.full((1, 14), 5.0, dtype=torch.float64)
        close_7 = torch.full((1, 14), 7.0, dtype=torch.float64)
        close_9 = torch.full((1, 14), 9.0, dtype=torch.float64)

        at_7 = MT5FeatureEngineer._willr(close_7, high, low, 14)
        at_9 = MT5FeatureEngineer._willr(close_9, high, low, 14)

        torch.testing.assert_close(
            at_7, torch.full_like(at_7, -0.6), rtol=0, atol=1e-7
        )
        torch.testing.assert_close(
            at_9, torch.full_like(at_9, -0.2), rtol=0, atol=1e-7
        )
        assert not torch.equal(at_7, at_9)

    def test_declared_14_bar_window_has_exact_boundary(self):
        length = 15
        target = length - 1
        close = torch.full((1, length), 7.0)
        high = torch.full((1, length), 10.0)
        low = torch.full((1, length), 5.0)
        baseline = MT5FeatureEngineer._willr(close, high, low, 14)[0, target]

        outside = high.clone()
        outside[:, 0] = 100.0
        torch.testing.assert_close(
            MT5FeatureEngineer._willr(close, outside, low, 14)[0, target],
            baseline,
            rtol=0,
            atol=0,
        )

        inside = high.clone()
        inside[:, 1] = 20.0
        assert not torch.equal(
            MT5FeatureEngineer._willr(close, inside, low, 14)[0, target],
            baseline,
        )
        spec = next(
            spec for spec in FEATURE_REGISTRY.feature_specs
            if spec.name == "WILLR_14"
        )
        assert spec.lookback == 14

    def test_zero_range_is_finite_zero(self):
        price = torch.full((2, 5), 7.0)
        actual = MT5FeatureEngineer._willr(price, price, price, 14)
        torch.testing.assert_close(actual, torch.zeros_like(actual), rtol=0, atol=0)


class TestFeatureRegistryHardeningAndShortAxes:
    def test_non_callable_feature_is_rejected_atomically(self):
        registry = Registry()
        before = (registry.feature_specs, registry.feature_names)
        with pytest.raises(RegistrationError) as error:
            registry.register_feature(
                FeatureSpec(
                    name="NOT_CALLABLE",
                    category="test",
                    compute=42,
                    lookback=1,
                )
            )
        assert type(error.value) is not RegistrationError
        assert (registry.feature_specs, registry.feature_names) == before

    def test_local_registry_freeze_rejects_feature_atomically(self):
        registry = Registry()
        registry.freeze()
        before = (registry.feature_specs, registry.feature_names)
        with pytest.raises(RegistrationError) as error:
            registry.register_feature(
                FeatureSpec(
                    name="AFTER_FREEZE",
                    category="test",
                    compute=lambda raw: raw["close"],
                    lookback=1,
                )
            )
        assert type(error.value) is not RegistrationError
        assert (registry.feature_specs, registry.feature_names) == before

    def test_global_feature_registry_is_frozen(self):
        assert FEATURE_REGISTRY.is_frozen
        before = (FEATURE_REGISTRY.feature_specs, FEATURE_REGISTRY.feature_names)
        with pytest.raises(RegistrationError):
            FEATURE_REGISTRY.register_feature(
                FeatureSpec(
                    name="GLOBAL_AFTER_FREEZE",
                    category="test",
                    compute=lambda raw: raw["close"],
                    lookback=1,
                )
            )
        assert (FEATURE_REGISTRY.feature_specs, FEATURE_REGISTRY.feature_names) == before

    @pytest.mark.parametrize("length", [1, 19])
    def test_every_feature_preserves_nonempty_short_axis(self, length):
        raw = _make_raw_dict(N=2, T=length)
        for spec in FEATURE_REGISTRY.feature_specs:
            assert spec.compute(raw).shape == (2, length), spec.name
        assert MT5FeatureEngineer.compute_features(raw).shape == (
            2,
            len(FEATURE_REGISTRY.feature_names),
            length,
        )

    def test_every_feature_rejects_empty_time_axis_consistently(self):
        raw = _make_raw_dict(N=2, T=0)
        for spec in FEATURE_REGISTRY.feature_specs:
            with pytest.raises(ValueError, match=r"non-empty.*\[N,T\]"):
                spec.compute(raw)
        with pytest.raises(ValueError, match=r"non-empty.*\[N,T\]"):
            MT5FeatureEngineer.compute_features(raw)
