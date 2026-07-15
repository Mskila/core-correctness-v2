"""
单元测试：时序算子（TS_MEAN / TS_STD / TS_RANK / TS_CORR_10）及新增趋势算子

验证：
- 输出形状均为 [N, T]
- TS_RANK_5/10/20 值域 ∈ [0, 1)
- TS_CORR_10 在常数输入时输出 0
- 所有算子对边界值（全零、极大值 1e8）无 NaN / Inf
- OPS_CONFIG 数量与 V2 operator registry 动态保持一致

需求：F2.1~F2.6
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import model_core.ops as ops_module
from model_core.ops import (
    OPERATOR_REGISTRY,
    OPS_CONFIG,
    _ts_corr_10,
    _ts_mean,
    _ts_rank,
    _ts_std,
)
from model_core.registry import OperatorSpec, RegistrationError, Registry
from model_core.vm import StackVM
from model_core.vocab import FORMULA_VOCAB, VOCAB_VERSION

# ── 常量 ────────────────────────────────────────────────────────────────────────
N, T = 4, 30

# 原始 10 个时序算子（索引 12~21），不含新增的趋势算子
TS_OPS = OPS_CONFIG[12:22]


# ── 辅助：随机正态输入 ───────────────────────────────────────────────────────────
def rand_input() -> torch.Tensor:
    torch.manual_seed(42)
    return torch.randn(N, T)


# ── 1. OPS_CONFIG 长度验证 ───────────────────────────────────────────────────────
class TestOpsConfigLength:
    def test_ops_config_matches_registry(self):
        assert len(OPS_CONFIG) == len(OPERATOR_REGISTRY.operator_names)

    def test_ops_config_is_an_immutable_registry_snapshot(self):
        expected = tuple(
            (spec.name, spec.transform, spec.arity)
            for spec in OPERATOR_REGISTRY.operator_specs
        )
        assert isinstance(OPS_CONFIG, tuple)
        assert OPS_CONFIG == expected

        feature = torch.tensor([[[1.0, 2.0, 4.0, 8.0]]])
        neg_token = FORMULA_VOCAB.operator_offset + 4
        before = StackVM().execute([0, neg_token], feature)
        version_before = VOCAB_VERSION
        original = OPS_CONFIG[4]
        try:
            with pytest.raises(TypeError):
                OPS_CONFIG[4] = ("NEG", lambda x: x, 1)
        finally:
            if isinstance(OPS_CONFIG, list):
                OPS_CONFIG[4] = original

        after = StackVM().execute([0, neg_token], feature)
        assert VOCAB_VERSION == version_before == FORMULA_VOCAB.version
        torch.testing.assert_close(after, before, rtol=0, atol=0)

    def test_new_ops_count_equals_10(self):
        """时序算子（索引 12-21）共 10 个"""
        assert len(TS_OPS) == 10

    def test_ts_op_names(self):
        """验证 10 个新算子名称正确"""
        expected_names = [
            "TS_MEAN_5", "TS_MEAN_10", "TS_MEAN_20",
            "TS_STD_5",  "TS_STD_10",  "TS_STD_20",
            "TS_RANK_5", "TS_RANK_10", "TS_RANK_20",
            "TS_CORR_10",
        ]
        actual_names = [name for name, _, _ in TS_OPS]
        assert actual_names == expected_names


# ── 2. 输出形状 [N, T] ──────────────────────────────────────────────────────────
class TestOutputShape:
    """全部 10 个新算子：输入 [N, T] → 输出 [N, T]"""

    def _invoke(self, name, fn, arity):
        x = rand_input()
        if arity == 1:
            return fn(x)
        else:  # arity == 2（TS_CORR_10）
            y = rand_input() + 0.1  # 略加偏移，避免 x==y 完全相关
            return fn(x, y)

    @pytest.mark.parametrize("name,fn,arity", TS_OPS, ids=[n for n, _, _ in TS_OPS])
    def test_output_shape(self, name, fn, arity):
        out = self._invoke(name, fn, arity)
        assert out.shape == (N, T), (
            f"{name}: 期望形状 ({N}, {T})，实际 {tuple(out.shape)}"
        )


# ── 3. TS_RANK 值域 ∈ [0, 1) ────────────────────────────────────────────────────
class TestTsRankRange:
    """TS_RANK_5/10/20 输出所有值满足 0 ≤ v < 1"""

    @pytest.mark.parametrize("d", [5, 10, 20])
    def test_rank_lower_bound(self, d):
        x = rand_input()
        out = _ts_rank(x, d)
        assert (out >= 0.0).all(), f"_ts_rank(x, {d}) 存在负值"

    @pytest.mark.parametrize("d", [5, 10, 20])
    def test_rank_upper_bound_strict(self, d):
        x = rand_input()
        out = _ts_rank(x, d)
        assert (out < 1.0).all(), (
            f"_ts_rank(x, {d}) 存在 ≥ 1.0 的值（最大值={out.max().item():.6f}）"
        )

    @pytest.mark.parametrize("name,fn,arity", [
        (n, f, a) for n, f, a in TS_OPS if n.startswith("TS_RANK")
    ], ids=["TS_RANK_5", "TS_RANK_10", "TS_RANK_20"])
    def test_rank_range_via_ops_config(self, name, fn, arity):
        """通过 OPS_CONFIG 中的 lambda 验证同一约束"""
        x = rand_input()
        out = fn(x)
        assert (out >= 0.0).all() and (out < 1.0).all(), (
            f"{name} 值域越界：min={out.min().item():.6f}, max={out.max().item():.6f}"
        )


# ── 4. TS_CORR_10 常数输入输出 0 ────────────────────────────────────────────────
class TestTsCorr10Constant:
    """当 x 或 y 在整个序列中为常数时，TS_CORR_10 在窗口完全填满后（t >= 9）输出应为 0。

    注意：_ts_corr_10 使用 d=10 的因果滑动窗口，左补零填充。
    在前 9 个时间步（t=0..8），窗口内混有零值和真实常数，std 不为零，
    因此 mask (std < 1e-6) 不生效，输出可能非零。
    从第 10 步（t=9）起，窗口完全由真实常数填满，std=0 触发 mask，输出为 0。
    """

    # d=10，满窗口需要 10 个真实值，从 t=9 开始（0-indexed）
    _FULL_WINDOW_START = 9  # 即 T 维度索引 9 起（第 10 个时间步）

    def test_constant_x_outputs_zero_after_warmup(self):
        """x 为常数时，满窗口之后（t >= 9）输出应全为 0"""
        x = torch.ones(N, T) * 3.14
        y = rand_input()
        out = _ts_corr_10(x, y)
        warmed = out[:, self._FULL_WINDOW_START:]
        assert torch.allclose(warmed, torch.zeros_like(warmed), atol=1e-6), (
            f"x 为常数时 t>={self._FULL_WINDOW_START} 输出应为 0，"
            f"实际最大绝对值={warmed.abs().max().item():.2e}"
        )

    def test_constant_y_outputs_zero_after_warmup(self):
        """y 为常数时，满窗口之后（t >= 9）输出应全为 0"""
        x = rand_input()
        y = torch.ones(N, T) * -2.71
        out = _ts_corr_10(x, y)
        warmed = out[:, self._FULL_WINDOW_START:]
        assert torch.allclose(warmed, torch.zeros_like(warmed), atol=1e-6), (
            f"y 为常数时 t>={self._FULL_WINDOW_START} 输出应为 0，"
            f"实际最大绝对值={warmed.abs().max().item():.2e}"
        )

    def test_both_constant_outputs_zero_after_warmup(self):
        """x 和 y 均为常数时，满窗口之后（t >= 9）输出应全为 0"""
        x = torch.full((N, T), 1.0)
        y = torch.full((N, T), 2.0)
        out = _ts_corr_10(x, y)
        warmed = out[:, self._FULL_WINDOW_START:]
        assert torch.allclose(warmed, torch.zeros_like(warmed), atol=1e-6)

    def test_corr_10_via_ops_config_constant_after_warmup(self):
        """通过 OPS_CONFIG 中的 TS_CORR_10 条目验证常数输入的满窗口行为"""
        _, fn, _ = next((name, fn, a) for name, fn, a in TS_OPS if name == "TS_CORR_10")
        x = torch.ones(N, T)
        y = rand_input()
        out = fn(x, y)
        warmed = out[:, self._FULL_WINDOW_START:]
        assert torch.allclose(warmed, torch.zeros_like(warmed), atol=1e-6)

    def test_long_constant_input_fully_zero(self):
        """使用更长序列（T=50）确保满窗口大部分为零"""
        T_long = 50
        x = torch.full((N, T_long), 7.0)
        y = torch.randn(N, T_long)
        out = _ts_corr_10(x, y)
        # 从 t=9 起所有位置应为 0
        warmed = out[:, self._FULL_WINDOW_START:]
        assert torch.allclose(warmed, torch.zeros_like(warmed), atol=1e-6), (
            f"长序列中 x 为常数时，满窗口部分应全为 0，"
            f"实际最大绝对值={warmed.abs().max().item():.2e}"
        )


# ── 5. 无 NaN / Inf（边界值测试）───────────────────────────────────────────────
class TestNoNanInf:
    """全部 10 个算子对边界输入（全零、极大值 1e8）不产生 NaN / Inf"""

    @staticmethod
    def _check(name, out):
        assert not torch.isnan(out).any(), f"{name}: 输出包含 NaN"
        assert not torch.isinf(out).any(), f"{name}: 输出包含 Inf"

    def _invoke(self, fn, arity, x):
        if arity == 1:
            return fn(x)
        else:
            return fn(x, x.clone())  # TS_CORR_10：x==y → 常数窗口 → 输出 0（已 mask）

    @pytest.mark.parametrize("name,fn,arity", TS_OPS, ids=[n for n, _, _ in TS_OPS])
    def test_zero_input(self, name, fn, arity):
        x = torch.zeros(N, T)
        out = self._invoke(fn, arity, x)
        self._check(name, out)

    @pytest.mark.parametrize("name,fn,arity", TS_OPS, ids=[n for n, _, _ in TS_OPS])
    def test_large_input(self, name, fn, arity):
        x = torch.full((N, T), 1e8)
        out = self._invoke(fn, arity, x)
        self._check(name, out)

    @pytest.mark.parametrize("name,fn,arity", TS_OPS, ids=[n for n, _, _ in TS_OPS])
    def test_random_input(self, name, fn, arity):
        """随机正态输入亦无 NaN / Inf"""
        x = rand_input()
        out = self._invoke(fn, arity, x)
        self._check(name, out)


# ── 6. 辅助函数单元测试 ─────────────────────────────────────────────────────────
class TestHelperFunctions:
    """直接测试 _ts_mean / _ts_std / _ts_rank / _ts_corr_10 的基本行为"""

    def test_ts_mean_shape(self):
        x = rand_input()
        assert _ts_mean(x, 5).shape == (N, T)
        assert _ts_mean(x, 10).shape == (N, T)
        assert _ts_mean(x, 20).shape == (N, T)

    def test_ts_std_non_negative(self):
        """滑动标准差加了 1e-6 下界，输出应 ≥ 1e-6"""
        x = rand_input()
        for d in (5, 10, 20):
            out = _ts_std(x, d)
            assert (out >= 1e-7).all(), f"_ts_std(x, {d}) 存在 < 1e-7 的值"

    def test_ts_std_constant_near_eps(self):
        """全常数输入的标准差：窗口填满（t >= d-1）后应约等于 1e-6（仅来自下界偏移）。

        _ts_rolling 对长度 d 的窗口左补 d-1 个零，所以：
        - t=0..d-2：窗口包含零和真实常数，std > 0
        - t=d-1 起：窗口全为真实常数，std ≈ 0，加上 1e-6 下界后 ≈ 1e-6
        """
        x = torch.ones(N, T) * 5.0
        for d in (5, 10, 20):
            out = _ts_std(x, d)
            # 只检查满窗口部分（t >= d-1）
            warmed = out[:, d - 1:]
            assert torch.allclose(warmed, torch.full_like(warmed, 1e-6), atol=1e-7), (
                f"_ts_std 常数输入（满窗口 t>={d-1}）应接近 1e-6，实际最大={warmed.max().item():.2e}"
            )

    def test_ts_corr_10_range(self):
        """正常随机输入时，相关系数值域应在 [-1, 1]"""
        torch.manual_seed(0)
        x = torch.randn(N, T)
        y = torch.randn(N, T)
        out = _ts_corr_10(x, y)
        assert (out >= -1.0 - 1e-5).all() and (out <= 1.0 + 1e-5).all(), (
            f"TS_CORR_10 值域越界：min={out.min().item():.6f}, max={out.max().item():.6f}"
        )


class TestProductFiveFormula:
    def test_registry_transform_matches_explicit_causal_compounding(self):
        cases = (
            torch.tensor(
                [
                    [0.10, -0.20, 0.05, 0.30, -0.10, 0.25, -0.15, 0.40],
                    [0.20, -1.20, 0.15, -0.30, 0.40, -0.05, 0.25, -0.10],
                ],
                dtype=torch.float32,
            ),
            torch.tensor(
                [[-0.99, -0.99, 0.10, -0.20, 0.05]],
                dtype=torch.float32,
            ),
            torch.tensor(
                [[9.05984497, -0.998884857, -0.879774213, 10.1290340,
                  89.9148178]],
                dtype=torch.float32,
            ),
        )
        product = next(
            spec.transform
            for spec in OPERATOR_REGISTRY.operator_specs
            if spec.name == "PRODUCT_5"
        )

        for x in cases:
            actual = product(x)
            safe = x.clamp_min(-0.999)
            expected = torch.stack(
                [
                    torch.prod(
                        1.0 + safe[:, max(0, end - 4):end + 1], dim=1
                    )
                    - 1.0
                    for end in range(x.shape[1])
                ],
                dim=1,
            )

            torch.testing.assert_close(actual, expected, rtol=0, atol=2.0e-7)

    @pytest.mark.parametrize(
        ("dtype", "atol"),
        [(torch.float32, 2.0e-7), (torch.float64, 2.0e-15)],
    )
    def test_registry_transform_preserves_tensor_contract_and_true_append(
        self, dtype, atol
    ):
        prefix = torch.tensor(
            [
                [0.10, -0.20, 0.05, 0.30, -0.10, 0.25, -0.15, 0.40],
                [0.20, -0.35, 0.15, -0.30, 0.40, -0.05, 0.25, -0.10],
            ],
            dtype=dtype,
        )
        future = torch.tensor(
            [[0.20, -0.05, 0.10, -0.25], [-0.15, 0.30, -0.20, 0.05]],
            dtype=dtype,
        )
        short = prefix.clone().requires_grad_()
        long = torch.cat((prefix, future), dim=1).requires_grad_()
        product = next(
            spec.transform
            for spec in OPERATOR_REGISTRY.operator_specs
            if spec.name == "PRODUCT_5"
        )

        def explicit_compounding(x):
            safe = x.clamp_min(-0.999)
            return torch.stack(
                [
                    torch.prod(
                        1.0 + safe[:, max(0, end - 4):end + 1], dim=1
                    )
                    - 1.0
                    for end in range(x.shape[1])
                ],
                dim=1,
            )

        short_output = product(short)
        long_output = product(long)
        for output, operand in ((short_output, short), (long_output, long)):
            assert output.shape == operand.shape
            assert output.dtype == operand.dtype
            assert output.device == operand.device
            torch.testing.assert_close(
                output, explicit_compounding(operand), rtol=0, atol=atol
            )
        assert torch.equal(short_output, long_output[:, :prefix.shape[1]])

        short_output.sum().backward()
        long_output[:, :prefix.shape[1]].sum().backward()
        assert short.grad is not None and torch.isfinite(short.grad).all()
        assert long.grad is not None and torch.isfinite(long.grad).all()
        assert torch.count_nonzero(short.grad).item() == short.numel()
        assert torch.equal(short.grad, long.grad[:, :prefix.shape[1]])
        torch.testing.assert_close(
            long.grad[:, prefix.shape[1]:],
            torch.zeros_like(long.grad[:, prefix.shape[1]:]),
            rtol=0,
            atol=0,
        )


class TestOperatorLookbacks:
    @staticmethod
    def _spec(name):
        return next(
            spec for spec in OPERATOR_REGISTRY.operator_specs if spec.name == name
        )

    @pytest.mark.parametrize(
        ("name", "expected_window"),
        [("EMA_5", 35), ("EMA_20", 139)],
    )
    def test_ema_declaration_matches_effective_kernel(
        self, name, expected_window
    ):
        spec = self._spec(name)
        assert spec.lookback == expected_window, (
            f"{name} declares {spec.lookback} bars but its 1e-6-truncated "
            f"EMA kernel requires {expected_window}"
        )

    @pytest.mark.parametrize(
        ("span", "expected_window"),
        [(5, 35), (20, 139)],
    )
    def test_ema_window_helper_is_the_registration_source(
        self, span, expected_window
    ):
        helper = getattr(ops_module, "_ema_effective_window", None)
        assert callable(helper), "EMA implementation has no shared window helper"
        assert helper(span) == expected_window

    @pytest.mark.parametrize(
        ("name", "window"),
        [("EMA_5", 35), ("EMA_20", 139)],
    )
    def test_ema_dependency_stops_at_declared_window(self, name, window):
        spec = self._spec(name)
        target = window
        base = torch.zeros(1, window + 1)

        outside = base.clone()
        outside[0, target - window] = 10_000.0
        inside = base.clone()
        inside[0, target - window + 1] = 10_000.0

        baseline_value = spec.transform(base)[0, target]
        outside_value = spec.transform(outside)[0, target]
        inside_value = spec.transform(inside)[0, target]

        torch.testing.assert_close(
            outside_value,
            baseline_value,
            rtol=0,
            atol=0,
            msg=f"{name} still depends on a value older than {window} bars",
        )
        assert not torch.isclose(
            inside_value,
            baseline_value,
            rtol=0,
            atol=1e-6,
        ), f"{name} must still depend on the earliest value inside its window"

    def test_all_registered_operators_have_positive_integer_lookback(self):
        assert OPERATOR_REGISTRY.operator_specs
        for spec in OPERATOR_REGISTRY.operator_specs:
            assert isinstance(getattr(spec, "lookback", None), int)
            assert not isinstance(spec.lookback, bool)
            assert spec.lookback >= 1
        assert getattr(ops_module, "MAX_OPERATOR_LOOKBACK", 0) >= 200

    @pytest.mark.parametrize("lookback", [None, True, False, 0, -1])
    def test_invalid_lookback_does_not_mutate_registry(self, lookback):
        registry = Registry()
        before = registry.operator_specs

        def transform(x: torch.Tensor) -> torch.Tensor:
            return x

        with pytest.raises(RegistrationError):
            registry.register_operator(
                OperatorSpec(
                    name="TEST_OPERATOR",
                    arity=1,
                    transform=transform,
                    lookback=lookback,
                )
            )
        assert registry.operator_specs == before

    def test_missing_lookback_cannot_mutate_registry(self):
        registry = Registry()

        def transform(x: torch.Tensor) -> torch.Tensor:
            return x

        with pytest.raises(TypeError):
            OperatorSpec(name="TEST_OPERATOR", arity=1, transform=transform)
        assert registry.operator_specs == ()

    def test_scale_declared_window_is_exactly_enforced(self):
        spec = self._spec("SCALE")
        assert spec.lookback == 200
        target = spec.lookback
        base = torch.ones(1, target + 1)
        outside = base.clone()
        outside[:, target - spec.lookback] = 10_000.0
        inside = base.clone()
        inside[:, target - spec.lookback + 1] = 10_000.0

        baseline = spec.transform(base)[0, target]
        torch.testing.assert_close(
            spec.transform(outside)[0, target], baseline, rtol=0, atol=0
        )
        assert not torch.equal(spec.transform(inside)[0, target], baseline)


class TestRegistryHardeningAndShortAxes:
    def test_non_callable_operator_is_rejected_atomically(self):
        registry = Registry()
        before = (registry.operator_specs, registry.operator_names)
        with pytest.raises(RegistrationError) as error:
            registry.register_operator(
                OperatorSpec(
                    name="NOT_CALLABLE",
                    arity=1,
                    transform=42,
                    lookback=1,
                )
            )
        assert type(error.value) is not RegistrationError
        assert (registry.operator_specs, registry.operator_names) == before

    def test_local_registry_freeze_rejects_operator_atomically(self):
        registry = Registry()
        registry.freeze()
        before = (registry.operator_specs, registry.operator_names)
        with pytest.raises(RegistrationError) as error:
            registry.register_operator(
                OperatorSpec(
                    name="AFTER_FREEZE",
                    arity=1,
                    transform=lambda x: x,
                    lookback=1,
                )
            )
        assert type(error.value) is not RegistrationError
        assert (registry.operator_specs, registry.operator_names) == before

    def test_global_operator_registry_is_frozen(self):
        assert OPERATOR_REGISTRY.is_frozen
        before = (OPERATOR_REGISTRY.operator_specs, OPERATOR_REGISTRY.operator_names)
        with pytest.raises(RegistrationError):
            OPERATOR_REGISTRY.register_operator(
                OperatorSpec(
                    name="GLOBAL_AFTER_FREEZE",
                    arity=1,
                    transform=lambda x: x,
                    lookback=1,
                )
            )
        assert (OPERATOR_REGISTRY.operator_specs, OPERATOR_REGISTRY.operator_names) == before

    @pytest.mark.parametrize("length", [1, 19])
    def test_every_operator_preserves_nonempty_short_axis(self, length):
        for spec in OPERATOR_REGISTRY.operator_specs:
            operands = [torch.ones(2, length) for _ in range(spec.arity)]
            assert spec.transform(*operands).shape == (2, length), spec.name

    def test_every_operator_rejects_empty_time_axis_consistently(self):
        for spec in OPERATOR_REGISTRY.operator_specs:
            operands = [torch.ones(2, 0) for _ in range(spec.arity)]
            with pytest.raises(ValueError, match=r"non-empty.*\[N,T\]"):
                spec.transform(*operands)
