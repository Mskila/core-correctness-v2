"""
model_core/backtest.py — MT5 回测评估器（组合级多目标 Reward）

评分框架（5品种组合版）：
  final_score =
      0.35 * portfolio_sortino          # 组合整体风险调整收益
    + 0.20 * portfolio_calmar           # 组合整体回撤控制
    + 0.15 * ts_ic_stability            # 时序IC稳定性（比横截面IC更重要）
    + 0.10 * symbol_consistency         # 品种一致性（防止单品种拖累）
    + 0.10 * cost_stress                # 成本压力测试（2x成本下仍盈利）
    + 0.10 * turnover_quality           # 换手率质量（交易频率奖励）
    - complexity_penalty                # 公式长度惩罚
    - correlation_penalty               # 因子相关性惩罚（由 engine 施加）

symbol_consistency 规则：
  - N 个品种中至少 ceil(N*0.6) 个 Sortino > 0 → 正分
  - 任何品种 Sortino < -2.0 → 重惩罚
  - 全部品种 Sortino > 0 → 额外奖励
"""
import math
import sys
from dataclasses import dataclass

import torch
from torch import Tensor

from config import Config
from .config import ModelConfig
from .execution import ExecutionResult, performance_metrics, run_execution
from .reward import apply_oos_gate, target_bars_per_trade
from .semantics import DataValidationError
from .walk_forward import MIN_EXECUTION_SEGMENT_OBSERVATIONS

_SORTINO_CLIP        = 20.0
_TURNOVER_EVENT_THRESHOLD = 0.0
_FOLD_INDEX_MIN = -sys.maxsize - 1
_FOLD_INDEX_MAX = sys.maxsize


@dataclass(frozen=True, slots=True)
class PreparedFold:
    source_shape: tuple[int, int]
    dtype: torch.dtype
    device: torch.device
    train_start: int
    train_end: int
    val_start: int
    val_end: int
    train_static: tuple[Tensor, Tensor, Tensor]
    val_static: tuple[Tensor, Tensor, Tensor]


class _ScaleBoundedPearsonIC(torch.autograd.Function):
    """Pearson IC with an exact forward and scale-bounded surrogate backward."""

    @staticmethod
    def forward(ctx, x: Tensor, y: Tensor) -> Tensor:
        work_dtype = (
            torch.float64
            if x.dtype in {torch.float16, torch.bfloat16, torch.float32}
            else x.dtype
        )
        x_work = x.to(work_dtype)
        y_work = y.to(work_dtype)
        x_scaled = x_work / x_work.abs().max()
        y_scaled = y_work / y_work.abs().max()
        x_centered = x_scaled - x_scaled.mean()
        y_centered = y_scaled - y_scaled.mean()
        x_norm = torch.linalg.vector_norm(x_centered)
        y_norm = torch.linalg.vector_norm(y_centered)
        correlation = (x_centered * y_centered).sum() / (x_norm * y_norm)
        ctx.save_for_backward(x_centered, y_centered, x_norm, y_norm, correlation)
        ctx.x_dtype = x.dtype
        ctx.y_dtype = y.dtype
        return correlation.to(dtype=x.dtype)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor, Tensor]:
        x_centered, y_centered, x_norm, y_norm, correlation = ctx.saved_tensors
        upstream = grad_output.to(dtype=x_centered.dtype)
        grad_x = (
            y_centered / (x_norm * y_norm)
            - correlation * x_centered / x_norm.square()
        )
        grad_y = (
            x_centered / (x_norm * y_norm)
            - correlation * y_centered / y_norm.square()
        )
        return (
            (upstream * grad_x).to(dtype=ctx.x_dtype),
            (upstream * grad_y).to(dtype=ctx.y_dtype),
        )


def compute_ic_metrics(
    factors: Tensor,
    target_ret: Tensor,
    target_valid: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return mean same-index Pearson IC and cross-symbol stability.

    Centered vectors are normalized by their own magnitudes before norms are
    taken, so finite proportional inputs are scale invariant. Only exact zero
    centered magnitude is unscorable. With one IC, or no cross-symbol
    dispersion, stability is the bounded mean IC itself.
    """
    if target_ret.shape != factors.shape or target_valid.shape != factors.shape:
        raise ValueError("factors, target_ret and target_valid must share shape")
    if target_valid.dtype is not torch.bool:
        raise ValueError("target_valid must be a boolean mask")

    ic_values = []
    for symbol in range(factors.shape[0]):
        valid = target_valid[symbol]
        x = factors[symbol, valid]
        y = target_ret[symbol, valid]
        if x.numel() < 2:
            continue
        work_dtype = (
            torch.float64
            if x.dtype in {torch.float16, torch.bfloat16, torch.float32}
            else x.dtype
        )
        x_work = x.to(work_dtype)
        y_work = y.to(work_dtype)
        if not bool(torch.isfinite(x_work).all()) or not bool(
            torch.isfinite(y_work).all()
        ):
            raise DataValidationError("IC inputs must be finite")
        x_scale = x_work.abs().max()
        y_scale = y_work.abs().max()
        if bool(x_scale == 0) or bool(y_scale == 0):
            continue
        x_scaled = x_work / x_scale
        y_scaled = y_work / y_scale
        x_unit = x_scaled - x_scaled.mean()
        y_unit = y_scaled - y_scaled.mean()
        x_norm = torch.linalg.vector_norm(x_unit)
        y_norm = torch.linalg.vector_norm(y_unit)
        if bool(x_norm == 0) or bool(y_norm == 0):
            continue
        correlation = _ScaleBoundedPearsonIC.apply(x, y)
        if not bool(torch.isfinite(correlation)):
            raise DataValidationError("non-finite IC rejected after stable normalization")
        ic_values.append(correlation)

    if not ic_values:
        zero = factors.new_zeros(())
        return zero, zero

    ic_tensor = torch.stack(ic_values)
    mean_ic = ic_tensor.mean()
    if ic_tensor.numel() < 2:
        return mean_ic, mean_ic
    dispersion = ic_tensor.std(unbiased=False)
    stability = (
        mean_ic
        if bool(dispersion == 0)
        else torch.clamp(mean_ic / dispersion, -3.0, 3.0)
    )
    return mean_ic, stability


class MT5Backtest:
    """MT5 组合级回测评估器。"""

    def __init__(
        self,
        cost_rate: float = 0.0001,
        *,
        timeframe: str = "H1",
        target_trades_per_day: float = 2.0,
        oos_gate_scale: float = 0.5,
    ):
        self.cost_rate = cost_rate
        self.timeframe = timeframe
        self.target_trades_per_day = target_trades_per_day
        self.target_bars_per_trade = target_bars_per_trade(
            timeframe,
            target_trades_per_day,
        )
        self.oos_gate_scale = oos_gate_scale

    @staticmethod
    def _promoted_finite_metric(
        name: str, value: float, reference: Tensor
    ) -> Tensor:
        if not math.isfinite(value):
            raise DataValidationError(
                f"public backtest metric must be finite: field={name} actual={value!r}"
            )
        return torch.tensor(value, dtype=torch.float64, device=reference.device)

    @staticmethod
    def _require_finite_public_score(score: Tensor) -> Tensor:
        if score.shape != torch.Size([]) or not bool(torch.isfinite(score).item()):
            raise DataValidationError(
                "public backtest score must be a finite scalar in the promoted domain"
            )
        return score

    # ──────────────────────────────────────────────────────────────────────
    # 基础统计
    # ──────────────────────────────────────────────────────────────────────

    def _sortino(
        self, pnl: Tensor, periods_per_year: float, eps: float = 1e-8
    ) -> Tensor:
        flat     = pnl.reshape(-1)
        mean_pnl = flat.mean()
        downside = flat[flat < 0]
        raw_std  = downside.std(unbiased=False) if downside.numel() > 0 \
                   else torch.tensor(0.0, dtype=flat.dtype, device=flat.device)
        # P0b 修复：下行标准差地板改为全序列 std 的 20%，防止稀疏 PnL 靠极小分母刷高分。
        # 原来 floor=|mean_pnl| 对稀疏序列趋近于零，导致 Sortino 爆炸。
        full_std       = flat.std(unbiased=False).clamp(min=eps)
        floor          = torch.clamp(full_std * 0.2, min=eps)
        downside_std   = torch.clamp(raw_std, min=floor)
        sortino        = mean_pnl / downside_std * math.sqrt(periods_per_year)
        return torch.clamp(sortino, -_SORTINO_CLIP, _SORTINO_CLIP)

    def _calmar(
        self, pnl: Tensor, periods_per_year: float, eps: float = 1e-8
    ) -> Tensor:
        """Calmar = annualized_return / max_drawdown（截断到 [-10, 10]）。"""
        flat      = pnl.reshape(-1)
        ann_ret   = flat.mean() * periods_per_year
        cum       = torch.cumsum(flat, dim=0)
        peak      = torch.cummax(cum, dim=0).values
        drawdown  = (peak - cum).max()
        drawdown  = torch.clamp(drawdown, min=eps)
        calmar    = ann_ret / drawdown
        return torch.clamp(calmar, -10.0, 10.0)

    # ──────────────────────────────────────────────────────────────────────
    # 组合级评分组件
    # ──────────────────────────────────────────────────────────────────────

    def _ts_ic_stability(
        self,
        factors: Tensor,
        target_ret: Tensor,
        target_valid: Tensor,
    ) -> float:
        """时序 IC 稳定性：同一有效索引上的 factor[t] 与 target_ret[t]。

        比横截面 IC 更适合 5 品种宇宙（横截面 N=5 统计意义弱）。

        Returns:
            float，约 [-1, 1]，正值代表因子有预测力。
        """
        _, stability = compute_ic_metrics(factors, target_ret, target_valid)
        return float(stability.item())

    def _symbol_consistency(
        self,
        per_symbol_sortino: list[float],
        per_symbol_trade_count: list[int] | None = None,
        eval_bars: int = 0,
    ) -> float:
        """品种一致性惩罚/奖励。

        规则（优先级从高到低）：
        1. 无交易品种超过 40%：重惩罚 -3.0
        2. P0a 新增：有交易的品种中，交易笔数 < eval_bars/100 (约每100bar少于1笔)
           视为"稀疏有效"，等同无效。防止3~6笔偶发交易刷高 Sortino。
        3. 任何品种 Sortino < -2.0：重惩罚 -2.0
        4. 有效品种中正收益比例决定奖惩
        """
        N = len(per_symbol_sortino)
        if N == 0:
            return 0.0

        # 最小有效交易数：每 100 bar 至少 1 笔，下限 5 笔
        min_trades = max(5, eval_bars // 100) if eval_bars > 0 else 5

        # 重新判定"活跃"品种（必须交易数 >= min_trades）
        if per_symbol_trade_count is not None:
            n_inactive = sum(1 for c in per_symbol_trade_count if c < min_trades)
            inactive_ratio = n_inactive / N
            if inactive_ratio > 0.4:
                return -3.0
        else:
            n_inactive = 0
            inactive_ratio = 0.0

        if any(s < -2.0 for s in per_symbol_sortino):
            return -2.0

        if per_symbol_trade_count is not None:
            active_sortinos = [
                s for s, c in zip(per_symbol_sortino, per_symbol_trade_count)
                if c >= min_trades
            ]
        else:
            active_sortinos = per_symbol_sortino

        if not active_sortinos:
            return -3.0

        n_positive = sum(1 for s in active_sortinos if s > 0)
        ratio = n_positive / len(active_sortinos)

        if ratio < 0.6:
            score = (ratio - 0.6) / 0.6 * 1.0
        else:
            score = (ratio - 0.6) / 0.4 * 1.0

        if ratio == 1.0:
            score += 0.5

        return float(score)

    def _run_execution(
        self,
        factors: Tensor,
        target_ret: Tensor,
        target_valid: Tensor,
        bar_time_ns: Tensor,
        *,
        cost_rate: float | None = None,
    ) -> ExecutionResult:
        """Delegate scoring execution to the shared V2 implementation."""
        return run_execution(
            factors=factors,
            target_ret=target_ret,
            target_valid=target_valid,
            bar_time_ns=bar_time_ns,
            cost_rate=self.cost_rate if cost_rate is None else cost_rate,
            min_exposure=Config.MIN_TRADE_EXPOSURE,
        )

    def _cost_stress(
        self,
        factors: Tensor,
        target_ret: Tensor,
        target_valid: Tensor,
        bar_time_ns: Tensor,
        stress_mult: float = 2.0,
    ) -> float:
        """Re-run shared execution at a stressed cost rate."""
        stressed = self._run_execution(
            factors,
            target_ret,
            target_valid,
            bar_time_ns,
            cost_rate=self.cost_rate * stress_mult,
        )
        return float(max(-5.0, min(5.0, performance_metrics(stressed).sortino)))

    @staticmethod
    def _turnover_activity(
        result: ExecutionResult,
    ) -> tuple[int, list[int], list[int]]:
        """Count activity from shared turnover and runs from explicit signs.

        Shared execution already applies the neutral exposure band, so any
        published turnover above zero is an activity event. Final liquidation
        remains separate and is not reconstructed here.
        """
        total_bars = 0
        event_counts = []
        all_runs = []
        result_position = result._borrow_tensor("position")
        result_valid = result._borrow_tensor("target_valid")
        result_turnover = result._borrow_tensor("turnover")
        for symbol in range(result_position.shape[0]):
            valid = result_valid[symbol]
            positions = result_position[symbol, valid]
            turnover = result_turnover[symbol, valid]
            total_bars += positions.numel()
            event_counts.append(
                int((turnover > _TURNOVER_EVENT_THRESHOLD).sum().item())
            )

            current_direction = 0
            current_length = 0
            for position in positions.tolist():
                direction = 1 if position > 0.0 else -1 if position < 0.0 else 0
                if direction == current_direction and direction != 0:
                    current_length += 1
                else:
                    if current_length:
                        all_runs.append(current_length)
                    current_direction = direction
                    current_length = 1 if direction else 0
            if current_length:
                all_runs.append(current_length)
        return total_bars, event_counts, all_runs

    def _turnover_quality(
        self, activity: tuple[int, list[int], list[int]]
    ) -> float:
        """Score activity against the configured per-day target."""
        total_bars, event_counts, all_runs = activity
        total_trades = sum(event_counts)

        target_trades = total_bars / self.target_bars_per_trade
        actual_ratio  = total_trades / max(target_trades, 1.0)

        if actual_ratio <= 0:
            freq_score = -2.0
        elif actual_ratio < 0.05:
            freq_score = -2.0 + actual_ratio / 0.05
        elif actual_ratio < 0.5:
            freq_score = -1.0 + (actual_ratio - 0.05) / 0.45
        elif actual_ratio <= 2.0:
            log_r = math.log(actual_ratio) / math.log(2.0)
            freq_score = 1.0 * math.exp(-0.5 * log_r ** 2)
        elif actual_ratio <= 8.0:
            freq_score = 0.5 - (actual_ratio - 2.0) / 6.0 * 1.5
        else:
            freq_score = -2.0

        hold_bonus = 0.0
        if all_runs:
            avg_hold = sum(all_runs) / len(all_runs)
            hold_bonus = min(0.3, math.log(max(avg_hold, 1.0)) / math.log(30.0) * 0.3)

        return float(freq_score + hold_bonus)

    def _beta_neutral_penalty(
        self, position: Tensor, target_valid: Tensor
    ) -> float:
        """Beta 中性惩罚：多空比例严重失衡时扣分。

        因子输出 >85% 同方向时，说明不是 alpha 因子而是 beta 因子
        （如 index 组的 TS_RANK 连续使用导致恒正输出）。

        Returns:
            float，惩罚值（负数或零）
        """
        flat = position[target_valid]
        long_ratio = (flat > 0.05).float().mean().item()
        short_ratio = (flat < -0.05).float().mean().item()
        max_ratio = max(long_ratio, short_ratio)
        if max_ratio > 0.85:
            # 超过 85% 同方向，重罚
            excess = (max_ratio - 0.85) / 0.15  # 0~1
            return -2.0 * excess  # 最多 -2.0
        elif max_ratio > 0.70:
            # 70-85% 轻度失衡，轻罚
            excess = (max_ratio - 0.70) / 0.15  # 0~1
            return -0.5 * excess  # 最多 -0.5
        return 0.0

    def _half_consistency_bonus(
        self,
        pnl: Tensor,
        target_valid: Tensor,
        periods_per_year: float,
    ) -> float:
        """前后一致性奖励：前半段和后半段 Sortino 同号时加分。

        防止因子只在某一段市场环境（如牛市）有效。

        Returns:
            float，奖励/惩罚值
        """
        if int(target_valid.sum().item()) < 20:
            return 0.0
        midpoint = pnl.shape[1] // 2
        first_half = pnl[:, :midpoint][target_valid[:, :midpoint]]
        second_half = pnl[:, midpoint:][target_valid[:, midpoint:]]
        if first_half.numel() == 0 or second_half.numel() == 0:
            return 0.0
        s1 = self._sortino(first_half, periods_per_year).item()
        s2 = self._sortino(second_half, periods_per_year).item()
        if s1 > 0 and s2 > 0:
            return 0.5  # 前后都赚钱，奖励
        elif s1 * s2 < 0:
            return -1.0  # 前后相反，重罚（如 index 组的 beta 因子）
        return 0.0  # 一正一零或两零，不奖不罚

    def _exposure_penalty(
        self, position: Tensor, target_valid: Tensor
    ) -> float:
        """在场时间惩罚（仅下限，无上限）：收益优先模式。

        只惩罚极稀疏交易（<10%在场），不惩罚高在场时间。
        高在场时间（满仓趋势跟踪）是外汇市场最赚钱的形态之一，不应受罚。
        """
        flat = position[target_valid].abs()
        exposure = flat.mean().item()   # 连续仓位：均值即平均持仓量
        if exposure < 0.10:
            # 极稀疏：平均持仓 < 10% → 线性惩罚 [-2, 0)
            return float((exposure / 0.10 - 1.0) * 2.0)
        return 0.0

    def _turnover_penalty(self, turnover: Tensor) -> Tensor:
        """梯度式换手率惩罚。"""
        mean_to = turnover.mean()
        penalty = torch.clamp(
            (mean_to - 0.2) * 3.0,
            min=0.0,
            max=3.0,
        )
        return -penalty

    # ──────────────────────────────────────────────────────────────────────
    # Walk-Forward 辅助接口
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _validate_fold_request(
        factors: Tensor,
        target_ret: Tensor,
        target_valid: Tensor,
        bar_time_ns: Tensor,
        train_start: int,
        train_end: int,
        val_start: int,
        val_end: int,
    ) -> None:
        inputs = (
            ("factors", factors),
            ("target_ret", target_ret),
            ("target_valid", target_valid),
            ("bar_time_ns", bar_time_ns),
        )
        for field, value in inputs:
            if not isinstance(value, torch.Tensor):
                raise DataValidationError(
                    "evaluate_fold invalid input: "
                    f"field={field}; expected=torch.Tensor; "
                    f"actual={type(value).__name__}"
                )
            if value.ndim != 2:
                raise DataValidationError(
                    "evaluate_fold invalid input: "
                    f"field={field}; expected=rank 2; actual=rank {value.ndim}"
                )

        shapes = [tuple(value.shape) for _, value in inputs]
        common_shape = max(shapes, key=shapes.count)
        for (field, _), shape in zip(inputs, shapes):
            if shape != common_shape:
                raise DataValidationError(
                    "evaluate_fold invalid input: "
                    f"field={field}; expected=shape {common_shape}; "
                    f"actual=shape {shape}"
                )

        supported_float_dtypes = {
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
        }
        for field, value in (("factors", factors), ("target_ret", target_ret)):
            if (
                not value.is_floating_point()
                or value.dtype not in supported_float_dtypes
            ):
                raise DataValidationError(
                    "evaluate_fold invalid input: "
                    f"field={field}; expected=floating tensor; actual={value.dtype}"
                )
        if target_ret.dtype != factors.dtype:
            raise DataValidationError(
                "evaluate_fold invalid input: "
                f"field=target_ret; expected={factors.dtype}; actual={target_ret.dtype}"
            )
        if target_valid.dtype is not torch.bool:
            raise DataValidationError(
                "evaluate_fold invalid input: "
                f"field=target_valid; expected=torch.bool; actual={target_valid.dtype}"
            )
        if bar_time_ns.dtype is not torch.int64:
            raise DataValidationError(
                "evaluate_fold invalid input: "
                f"field=bar_time_ns; expected=torch.int64; actual={bar_time_ns.dtype}"
            )

        expected_device = factors.device
        for field, value in inputs[1:]:
            if value.device != expected_device:
                raise DataValidationError(
                    "evaluate_fold invalid input: "
                    f"field={field}; expected=device {expected_device}; "
                    f"actual=device {value.device}"
                )
        if expected_device.type not in {"cpu", "cuda"}:
            raise DataValidationError(
                "evaluate_fold invalid input: "
                "field=factors; expected=device type cpu or cuda; "
                f"actual=device {expected_device}"
            )

        bounds = (
            ("train_start", train_start),
            ("train_end", train_end),
            ("val_start", val_start),
            ("val_end", val_end),
        )
        for field, value in bounds:
            if type(value) is not int:
                raise ValueError(
                    "evaluate_fold invalid boundary: "
                    f"field={field}; expected=exact built-in int; "
                    f"actual_type={type(value).__name__}"
                )
        for field, value in bounds:
            if value < _FOLD_INDEX_MIN or value > _FOLD_INDEX_MAX:
                raise ValueError(
                    "evaluate_fold invalid boundary: "
                    f"field={field}; expected=exact built-in int within "
                    "operational bounds; actual=out of operational bounds"
                )
        valid_order = (
            0 <= train_start < train_end
            and train_end < val_start < val_end
        )
        if not valid_order:
            actual_bounds = tuple(value for _, value in bounds)
            raise ValueError(
                "evaluate_fold invalid boundary: field=fold_bounds; "
                "expected=0 <= train_start < train_end < "
                "val_start < val_end; "
                f"actual={actual_bounds}"
            )

        segment_lengths = (
            ("train_bars", train_end - train_start),
            ("val_bars", val_end - val_start),
        )
        for field, length in segment_lengths:
            if length < MIN_EXECUTION_SEGMENT_OBSERVATIONS:
                raise ValueError(
                    "evaluate_fold invalid boundary: "
                    f"field={field}; expected=at least "
                    f"{MIN_EXECUTION_SEGMENT_OBSERVATIONS} observations; "
                    f"actual={length}"
                )

        common_length = common_shape[1]
        for field, end in (("train_end", train_end), ("val_end", val_end)):
            if end + 2 > common_length:
                raise ValueError(
                    "evaluate_fold invalid boundary: "
                    f"field={field}; expected={field} + 2 <= common length "
                    f"{common_length}; actual={end}"
                )

    @staticmethod
    def _fold_segment(
        factors: Tensor,
        target_ret: Tensor,
        target_valid: Tensor,
        bar_time_ns: Tensor,
        start: int,
        end: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Isolate one fold segment while retaining only its exit timestamps."""
        if start < 0 or end <= start or end + 2 > factors.shape[1]:
            raise ValueError("fold segment must leave two exit timestamps")
        segment_factors = factors[:, start:end]
        segment_target = target_ret[:, start:end]
        segment_valid = target_valid[:, start:end]
        padding = torch.zeros_like(segment_factors[:, :2])
        valid_padding = torch.zeros_like(segment_valid[:, :2])
        return (
            torch.cat([segment_factors, padding], dim=1),
            torch.cat([segment_target, padding], dim=1),
            torch.cat([segment_valid, valid_padding], dim=1),
            bar_time_ns[:, start:end + 2].clone(),
        )

    def evaluate_fold(
        self,
        factors:     Tensor,
        target_ret:  Tensor,
        target_valid: Tensor,
        bar_time_ns: Tensor,
        train_start: int,
        train_end:   int,
        val_start:   int,
        val_end:     int,
    ) -> tuple[Tensor, Tensor]:
        """在指定训练/验证切片上计算组合多目标得分。

        train_score：用于 REINFORCE 梯度更新（in-sample 多目标）。
        val_score：用于选冠军，加入符号无关的加法 OOS Sortino 门控。
        """
        self._validate_fold_request(
            factors,
            target_ret,
            target_valid,
            bar_time_ns,
            train_start,
            train_end,
            val_start,
            val_end,
        )
        train_values = self._fold_segment(
            factors,
            target_ret,
            target_valid,
            bar_time_ns,
            train_start,
            train_end,
        )
        val_values = self._fold_segment(
            factors,
            target_ret,
            target_valid,
            bar_time_ns,
            val_start,
            val_end,
        )
        train_result = self._run_execution(*train_values)
        val_result = self._run_execution(*val_values)
        train_score = self._multi_objective(
            *train_values, train_result, eval_bars=train_end - train_start
        ) + self._turnover_penalty(
            train_result._borrow_tensor("turnover")[
                train_result._borrow_tensor("target_valid")
            ]
        )
        base_val = self._multi_objective(
            *val_values, val_result, eval_bars=val_end - val_start
        )
        oos_sor = performance_metrics(val_result).sortino
        gate = apply_oos_gate(
            float(base_val.item()),
            oos_sor,
            scale=self.oos_gate_scale,
        )
        val_score = base_val.new_tensor(gate.final)

        return (
            self._require_finite_public_score(train_score),
            self._require_finite_public_score(val_score),
        )

    def prepare_fold(
        self,
        *,
        target_ret: Tensor,
        target_valid: Tensor,
        bar_time_ns: Tensor,
        train_start: int,
        train_end: int,
        val_start: int,
        val_end: int,
    ) -> PreparedFold:
        """Precompute formula-independent fold tensors once per training run."""
        self._validate_fold_request(
            target_ret,
            target_ret,
            target_valid,
            bar_time_ns,
            train_start,
            train_end,
            val_start,
            val_end,
        )

        def prepare_static(start: int, end: int) -> tuple[Tensor, Tensor, Tensor]:
            segment_target = target_ret[:, start:end]
            segment_valid = target_valid[:, start:end]
            return (
                torch.cat(
                    [segment_target, torch.zeros_like(segment_target[:, :2])],
                    dim=1,
                ),
                torch.cat(
                    [segment_valid, torch.zeros_like(segment_valid[:, :2])],
                    dim=1,
                ),
                bar_time_ns[:, start:end + 2].clone(),
            )

        return PreparedFold(
            source_shape=tuple(target_ret.shape),
            dtype=target_ret.dtype,
            device=target_ret.device,
            train_start=train_start,
            train_end=train_end,
            val_start=val_start,
            val_end=val_end,
            train_static=prepare_static(train_start, train_end),
            val_static=prepare_static(val_start, val_end),
        )

    def evaluate_prepared_fold(
        self,
        factors: Tensor,
        prepared: PreparedFold,
    ) -> tuple[Tensor, Tensor]:
        """Evaluate a factor against immutable, precomputed fold context."""
        if (
            not isinstance(factors, Tensor)
            or tuple(factors.shape) != prepared.source_shape
            or factors.dtype != prepared.dtype
            or factors.device != prepared.device
        ):
            raise DataValidationError(
                "prepared fold factor mismatch: expected="
                f"shape {prepared.source_shape}, dtype {prepared.dtype}, "
                f"device {prepared.device}"
            )

        def factor_segment(start: int, end: int) -> Tensor:
            segment = factors[:, start:end]
            return torch.cat(
                [segment, torch.zeros_like(segment[:, :2])],
                dim=1,
            )

        train_values = (factor_segment(prepared.train_start, prepared.train_end),) + prepared.train_static
        val_values = (factor_segment(prepared.val_start, prepared.val_end),) + prepared.val_static
        train_result = self._run_execution(*train_values)
        val_result = self._run_execution(*val_values)
        train_score = self._multi_objective(
            *train_values,
            train_result,
            eval_bars=prepared.train_end - prepared.train_start,
        ) + self._turnover_penalty(
            train_result._borrow_tensor("turnover")[
                train_result._borrow_tensor("target_valid")
            ]
        )
        base_val = self._multi_objective(
            *val_values,
            val_result,
            eval_bars=prepared.val_end - prepared.val_start,
        )
        oos_sortino = performance_metrics(val_result).sortino
        gate = apply_oos_gate(
            float(base_val.item()),
            oos_sortino,
            scale=self.oos_gate_scale,
        )
        val_score = base_val.new_tensor(gate.final)
        return (
            self._require_finite_public_score(train_score),
            self._require_finite_public_score(val_score),
        )

    def _reversal_bonus(
        self, factors: Tensor, target_valid: Tensor
    ) -> Tensor:
        """反转奖励：鼓励因子有低/负自相关（均值回归特征）。

        计算每个品种的 lag-1 自相关系数，越接近 0 或负值 = 越好。
        高度正自相关（>0.5）= 趋势跟踪，减分。

        单品种模式：直接返回标量。
        """
        N = factors.shape[0]
        scores = []
        for n in range(N):
            values = factors[n, target_valid[n]]
            x = values[:-1]
            y = values[1:]
            xm = x - x.mean(); ym = y - y.mean()
            sx = (xm**2).mean().sqrt(); sy = (ym**2).mean().sqrt()
            ac1 = (
                (xm * ym).mean() / (sx * sy + 1e-8)
                if sx > 1e-6 and sy > 1e-6
                else xm.new_zeros(())
            )
            # 奖励低自相关：bonus = 1 - |ac1|, 负自相关额外加分
            bonus = 1.0 - torch.abs(ac1)
            if ac1 < 0:
                bonus = bonus + 0.5  # 负自相关（真正反转）额外加分
            bonus = torch.clamp(bonus, -1.0, 2.0)
            scores.append(bonus)
        return torch.stack(scores).mean()

    def _symmetry_check(
        self, position: Tensor, target_valid: Tensor
    ) -> Tensor:
        """多空对称性检查：奖励 50/50 多空分布。

        均值回归策略应该在多空之间大致平衡，
        过度偏向某一侧 = 趋势跟踪特征，应惩罚。
        """
        valid_position = position[target_valid]
        long_ratio  = (valid_position > 0).float().mean()
        short_ratio = (valid_position < 0).float().mean()
        # 理想值：long_ratio ≈ 0.5, short_ratio ≈ 0.5
        # 偏差：|long_ratio - 0.5| + |short_ratio - 0.5|
        deviation = torch.abs(long_ratio - 0.5) + torch.abs(short_ratio - 0.5)
        # 偏差 0 → 奖励 1.0; 偏差 1.0 → 奖励 -1.0
        bonus = 1.0 - 2.0 * deviation
        return torch.clamp(bonus, -1.0, 1.0)

    def _multi_objective(
        self,
        factors:    Tensor,
        target_ret: Tensor,
        target_valid: Tensor,
        bar_time_ns: Tensor,
        result: ExecutionResult,
        eval_bars:  int = 0,
    ) -> Tensor:
        """收益优先的多目标评分（2026-07-04 重构）。

        核心改变：加入年化绝对收益项（权重 0.40），这是最主要的优化目标。
        Sortino/Calmar 权重大幅下调，仅作为风险调整辅助。
        clamp 上限放开（Sortino 40→20 保持，收益无上限）。

        N=1 单品种模式权重略有不同（无 symbol_consistency/cost_stress）。

        2026-07-08: 新增 forex 模式 — 偏向均值回归策略。
        """
        N = factors.shape[0]
        metrics = performance_metrics(result)
        position = result._borrow_tensor("position")
        pnl = result._borrow_tensor("net_pnl")
        ann_ret = self._promoted_finite_metric(
            "annualized_return", metrics.annualized_return, factors
        )
        port_sortino = self._promoted_finite_metric(
            "sortino", metrics.sortino, factors
        )
        port_calmar = self._promoted_finite_metric(
            "calmar", metrics.calmar, factors
        )
        _, ts_ic = compute_ic_metrics(factors, target_ret, target_valid)
        turnover_activity = self._turnover_activity(result)
        tq = self._turnover_quality(turnover_activity)
        exp_pen = self._exposure_penalty(position, target_valid)

        if N == 1:
            beta_pen = self._beta_neutral_penalty(position, target_valid)
            consist = self._half_consistency_bonus(
                pnl, target_valid, metrics.periods_per_year
            )

            if ModelConfig.REWARD_MODE == "forex":
                # Forex 均值回归模式：
                #   - 降年化收益权重 (0.80→0.25)：外汇趋势弱，避免奖励虚假趋势
                #   - 提 IC 权重 (0.03→0.25)：信号质量是核心
                #   - 新增反转奖励 (0.20)：奖励低/负因子自相关
                #   - 新增对称检查 (0.15)：奖励 50/50 多空平衡
                rev_bonus = self._reversal_bonus(factors, target_valid)
                sym_bonus = self._symmetry_check(position, target_valid)
                return (
                    0.25 * ann_ret           # 年化收益（降权，外汇趋势噪声大）
                    + 0.05 * port_sortino    # 风险调整辅助
                    + 0.05 * port_calmar     # 回撤控制
                    + 0.25 * ts_ic           # 信号质量（大幅提权）
                    + 0.20 * rev_bonus       # 反转奖励（核心：反趋势）
                    + 0.15 * sym_bonus       # 多空对称（均值回归特征）
                    + 0.05 * tq              # 交易频率质量
                    + exp_pen                # 稀疏惩罚
                    + beta_pen               # Beta 中性惩罚
                    + consist                # 前后一致性奖惩
                )

            if ModelConfig.REWARD_MODE == "ftmo":
                # FTMO 专属：年化收益 0.80，Calmar 0.10（控制 MDD 贴近 10% 上限）
                return (
                    0.80 * ann_ret           # 主目标：年化绝对收益（FTMO 加权）
                    + 0.05 * port_sortino    # 风险调整辅助（降权）
                    + 0.10 * port_calmar     # 回撤控制（保持，对齐 10% Max Loss）
                    + 0.03 * ts_ic           # IC 预测方向（降权）
                    + 0.02 * tq              # 交易频率质量（降权）
                    + exp_pen                # 稀疏惩罚
                    + beta_pen               # Beta 中性惩罚
                    + consist                # 前后一致性奖惩
                )
            return (
                0.60 * ann_ret           # 主目标：年化绝对收益
                + 0.15 * port_sortino    # 风险调整辅助
                + 0.10 * port_calmar     # 回撤控制辅助
                + 0.10 * ts_ic           # IC 预测方向
                + 0.05 * tq              # 交易频率质量
                + exp_pen                # 稀疏惩罚
                + beta_pen               # Beta 中性惩罚
                + consist                # 前后一致性奖惩
            )

        result_position = result._borrow_tensor("position")
        result_turnover = result._borrow_tensor("turnover")
        result_gross = result._borrow_tensor("gross_pnl")
        result_cost = result._borrow_tensor("cost")
        result_net = result._borrow_tensor("net_pnl")
        result_valid = result._borrow_tensor("target_valid")
        result_time = result._borrow_tensor("bar_time_ns")
        result_liquidation = result._borrow_tensor("final_liquidation_cost")
        per_sym_sortino     = []
        per_sym_trade_count = []
        for n in range(N):
            symbol_result = ExecutionResult(
                position=result_position[n:n + 1],
                turnover=result_turnover[n:n + 1],
                gross_pnl=result_gross[n:n + 1],
                cost=result_cost[n:n + 1],
                net_pnl=result_net[n:n + 1],
                target_valid=result_valid[n:n + 1],
                bar_time_ns=result_time[n:n + 1],
                final_liquidation_cost=result_liquidation[n:n + 1],
            )
            per_sym_sortino.append(performance_metrics(symbol_result).sortino)
            per_sym_trade_count.append(turnover_activity[1][n])

        sym_cons = self._symbol_consistency(
            per_sym_sortino, per_sym_trade_count, eval_bars=eval_bars
        )
        cost_s = self._cost_stress(
            factors, target_ret, target_valid, bar_time_ns
        )
        beta_pen = self._beta_neutral_penalty(position, target_valid)
        consist = self._half_consistency_bonus(
            pnl, target_valid, metrics.periods_per_year
        )

        if ModelConfig.REWARD_MODE == "ftmo":
            # FTMO 专属：年化收益 0.75（提权），Calmar 0.10（对齐 10% Max Loss）
            return (
                0.75 * ann_ret               # 主目标：年化绝对收益（FTMO 加权）
                + 0.05 * port_sortino        # 风险调整辅助（降权）
                + 0.10 * port_calmar         # 回撤控制（提权，控制 MDD）
                + 0.02 * ts_ic               # IC 预测方向（降权）
                + 0.03 * sym_cons            # 品种一致性（降权）
                + 0.02 * cost_s              # 成本压力测试（降权）
                + 0.03 * tq                  # 交易频率质量（降权）
                + exp_pen                    # 稀疏惩罚
                + beta_pen                   # Beta 中性惩罚
                + consist                    # 前后一致性奖惩
            )

        return (
            0.60 * ann_ret               # 主目标：年化绝对收益
            + 0.10 * port_sortino        # 风险调整辅助
            + 0.05 * port_calmar         # 回撤控制辅助
            + 0.10 * ts_ic               # IC 预测方向
            + 0.05 * sym_cons            # 品种一致性
            + 0.05 * cost_s              # 成本压力测试
            + 0.05 * tq                  # 交易频率质量
            + exp_pen                    # 稀疏惩罚
            + beta_pen                   # Beta 中性惩罚
            + consist                    # 前后一致性奖惩
        )

    # ──────────────────────────────────────────────────────────────────────
    # 公开接口（单片段诊断，不代表样本外）
    # ──────────────────────────────────────────────────────────────────────

    def evaluate_segment(
        self,
        factors:    Tensor,
        target_ret: Tensor,
        target_valid: Tensor,
        bar_time_ns: Tensor,
    ) -> Tensor:
        """Score one explicitly supplied segment using shared V2 execution."""
        result = self._run_execution(
            factors, target_ret, target_valid, bar_time_ns
        )
        score = self._multi_objective(
            factors,
            target_ret,
            target_valid,
            bar_time_ns,
            result,
            eval_bars=int(target_valid.sum(dim=1).max().item()),
        ) + self._turnover_penalty(
            result._borrow_tensor("turnover")[
                result._borrow_tensor("target_valid")
            ]
        )
        return self._require_finite_public_score(score)
