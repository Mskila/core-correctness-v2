"""Formula-evaluation boundary used by serial and future parallel evaluators."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import torch

from .backtest import MT5Backtest, PreparedFold, compute_ic_metrics
from .vm import FormulaEvaluationError, StackVM
from .walk_forward import WalkForwardFold
from strategy_manager.signal import compute_target_positions_stateless


def evaluate_training_formula(vm, formula, features, *, step: int, index: int):
    """Evaluate one formula and reject the legacy ``None`` failure sentinel."""
    attributes = getattr(vm, "__dict__", {})
    has_execute_override = type(attributes) is dict and "execute" in attributes
    evaluate = getattr(vm, "evaluate", None)
    if callable(evaluate) and not has_execute_override:
        return evaluate(formula, features)
    result = vm.execute(formula, features)
    if result is None:
        raise RuntimeError(
            "training program evaluation failed: vm.execute returned None; "
            f"step={step} formula_index={index} formula_length={len(formula)}"
        )
    return result


@dataclass(frozen=True, slots=True)
class RawFormulaError:
    """Structured expected rejection returned by a raw evaluator."""

    kind: str
    token_index: int
    operator: str | None
    detail: str


@dataclass(frozen=True, slots=True)
class RawFormulaEvaluation:
    """Formula-local output with no serial decision-layer mutations."""

    formula_index: int
    formula: tuple[int, ...]
    factor: torch.Tensor | None
    fold_train_scores: tuple[torch.Tensor, ...]
    fold_val_scores: tuple[torch.Tensor, ...]
    fold_ics: tuple[float, ...]
    selection_ic: float | None
    selection_ic_stability: float | None
    exposure: float | None
    constant: bool
    error: RawFormulaError | None


@dataclass(frozen=True, slots=True)
class RawEvaluationContext:
    """Read-only-by-contract tensors and scalar configuration for one run."""

    features: torch.Tensor
    target_ret: torch.Tensor
    target_valid: torch.Tensor
    bar_time_ns: torch.Tensor
    folds: tuple[WalkForwardFold, ...]
    selection_index: torch.Tensor
    cost_rate: float
    timeframe: str
    target_trades_per_day: float
    oos_gate_scale: float

    def validate(self) -> None:
        for name in (
            "features", "target_ret", "target_valid", "bar_time_ns",
            "selection_index",
        ):
            value = getattr(self, name)
            if type(value) is not torch.Tensor:
                raise TypeError(f"{name} must be an exact Tensor")
            if value.device.type != "cpu":
                raise ValueError(f"{name} must be on CPU for CPU evaluation")
        if self.features.ndim != 3:
            raise ValueError("features must have shape [symbol, feature, time]")
        source_shape = tuple(self.target_ret.shape)
        if (
            self.target_ret.ndim != 2
            or tuple(self.target_valid.shape) != source_shape
            or tuple(self.bar_time_ns.shape) != source_shape
            or self.features.shape[0] != source_shape[0]
            or self.features.shape[-1] != source_shape[1]
        ):
            raise ValueError("raw evaluation tensors have incompatible shapes")
        if self.features.dtype != self.target_ret.dtype:
            raise ValueError("features and target_ret must have the same dtype")
        if self.target_valid.dtype is not torch.bool:
            raise ValueError("target_valid must have dtype bool")
        if self.selection_index.dtype is not torch.long or self.selection_index.ndim != 1:
            raise ValueError("selection_index must be a one-dimensional long Tensor")
        if not self.folds or any(type(fold) is not WalkForwardFold for fold in self.folds):
            raise TypeError("folds must be a non-empty exact WalkForwardFold tuple")


class FormulaEvaluator(Protocol):
    """Batch evaluator interface shared by reference and parallel CPU paths."""

    def evaluate_batch(
        self,
        formulas: Sequence[Sequence[int]],
        *,
        step: int,
    ) -> list[RawFormulaEvaluation]: ...

    def close(self) -> None: ...


def _ic_pair(
    factor: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[float, float]:
    mean, stability = compute_ic_metrics(factor, target, valid)
    return float(mean.item()), float(stability.item())


def evaluate_raw_formula(
    *,
    formula_index: int,
    formula: Sequence[int],
    step: int,
    context: RawEvaluationContext,
    vm: StackVM,
    backtest: MT5Backtest,
    prepared_folds: tuple[PreparedFold, ...],
) -> RawFormulaEvaluation:
    """Evaluate only formula-local state; unexpected failures propagate."""
    tokens = tuple(int(token) for token in formula)
    try:
        factor = evaluate_training_formula(
            vm,
            tokens,
            context.features,
            step=step,
            index=formula_index,
        )
    except FormulaEvaluationError as failure:
        return RawFormulaEvaluation(
            formula_index=formula_index,
            formula=tokens,
            factor=None,
            fold_train_scores=(),
            fold_val_scores=(),
            fold_ics=(),
            selection_ic=None,
            selection_ic_stability=None,
            exposure=None,
            constant=False,
            error=RawFormulaError(
                kind=failure.error_kind.value,
                token_index=failure.token_index,
                operator=failure.operator,
                detail=failure.detail,
            ),
        )

    flat = factor.reshape(-1)
    constant = not bool((flat != flat[0]).any())
    if constant:
        return RawFormulaEvaluation(
            formula_index=formula_index,
            formula=tokens,
            factor=factor,
            fold_train_scores=(),
            fold_val_scores=(),
            fold_ics=(),
            selection_ic=None,
            selection_ic_stability=None,
            exposure=None,
            constant=True,
            error=None,
        )

    fold_train_scores: list[torch.Tensor] = []
    fold_val_scores: list[torch.Tensor] = []
    fold_ics: list[float] = []
    with torch.no_grad():
        for fold, prepared in zip(context.folds, prepared_folds):
            train_score, val_score = backtest.evaluate_prepared_fold(factor, prepared)
            fold_train_scores.append(train_score)
            fold_val_scores.append(val_score)
            fold_ic, _ = _ic_pair(
                factor[:, fold.train_start:fold.train_end],
                context.target_ret[:, fold.train_start:fold.train_end],
                context.target_valid[:, fold.train_start:fold.train_end],
            )
            fold_ics.append(fold_ic)

        selection_factor = torch.index_select(
            factor,
            dim=-1,
            index=context.selection_index,
        )
        selection_target = torch.index_select(
            context.target_ret,
            dim=-1,
            index=context.selection_index,
        )
        selection_valid = torch.index_select(
            context.target_valid,
            dim=-1,
            index=context.selection_index,
        )
        selection_ic, selection_stability = _ic_pair(
            selection_factor,
            selection_target,
            selection_valid,
        )
        exposure = float(
            compute_target_positions_stateless(selection_factor).abs().mean().item()
        )

    return RawFormulaEvaluation(
        formula_index=formula_index,
        formula=tokens,
        factor=factor,
        fold_train_scores=tuple(fold_train_scores),
        fold_val_scores=tuple(fold_val_scores),
        fold_ics=tuple(fold_ics),
        selection_ic=selection_ic,
        selection_ic_stability=selection_stability,
        exposure=exposure,
        constant=False,
        error=None,
    )


class ReferenceCpuEvaluator:
    """Exact serial reference implementation of raw formula evaluation."""

    def __init__(
        self,
        context: RawEvaluationContext,
        *,
        vm: StackVM | None = None,
        backtest: MT5Backtest | None = None,
        prepared_folds: tuple[PreparedFold, ...] | None = None,
    ) -> None:
        if type(context) is not RawEvaluationContext:
            raise TypeError("context must be an exact RawEvaluationContext")
        context.validate()
        self.context = context
        self.vm = StackVM() if vm is None else vm
        self.backtest = (
            MT5Backtest(
                cost_rate=context.cost_rate,
                timeframe=context.timeframe,
                target_trades_per_day=context.target_trades_per_day,
                oos_gate_scale=context.oos_gate_scale,
            )
            if backtest is None
            else backtest
        )
        self.prepared_folds = (
            tuple(
                self.backtest.prepare_fold(
                    target_ret=context.target_ret,
                    target_valid=context.target_valid,
                    bar_time_ns=context.bar_time_ns,
                    train_start=fold.train_start,
                    train_end=fold.train_end,
                    val_start=fold.val_start,
                    val_end=fold.val_end,
                )
                for fold in context.folds
            )
            if prepared_folds is None
            else prepared_folds
        )
        if len(self.prepared_folds) != len(context.folds):
            raise ValueError("prepared_folds must match the walk-forward folds")

    def evaluate_batch(
        self,
        formulas: Sequence[Sequence[int]],
        *,
        step: int,
    ) -> list[RawFormulaEvaluation]:
        return [
            evaluate_raw_formula(
                formula_index=index,
                formula=formula,
                step=step,
                context=self.context,
                vm=self.vm,
                backtest=self.backtest,
                prepared_folds=self.prepared_folds,
            )
            for index, formula in enumerate(formulas)
        ]

    def close(self) -> None:
        return None


__all__ = [
    "FormulaEvaluator",
    "RawEvaluationContext",
    "RawFormulaError",
    "RawFormulaEvaluation",
    "ReferenceCpuEvaluator",
    "evaluate_raw_formula",
    "evaluate_training_formula",
]
