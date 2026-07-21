from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

import torch

from .causal import causal_rolling_zscore
from .ops import DomainError, OPS_CONFIG, ShapeError
from .vocab import FORMULA_VOCAB


class FormulaDomain(str, Enum):
    SIGNED = "signed"
    NONNEGATIVE = "nonnegative"
    NONPOSITIVE = "nonpositive"
    ZERO = "zero"
    UNKNOWN = "unknown"


class FormulaErrorKind(str, Enum):
    INVALID_FORMULA = "invalid_formula"
    DOMAIN_ERROR = "domain_error"
    NONFINITE = "nonfinite"
    SHAPE_ERROR = "shape_error"
    OPERATOR_ERROR = "operator_error"


class FormulaEvaluationError(RuntimeError):
    """A bounded, auditable failure at one postfix token."""

    def __init__(
        self,
        *,
        formula: Iterable[int],
        token_index: int,
        operator: str | None,
        error_kind: FormulaErrorKind,
        detail: str,
    ) -> None:
        self.formula = tuple(int(token) for token in formula)
        self.token_index = int(token_index)
        self.operator = operator
        self.error_kind = error_kind
        self.detail = str(detail)[:500]
        super().__init__(
            f"formula evaluation failed: kind={error_kind.value} "
            f"token_index={self.token_index} operator={operator!r} "
            f"formula={self.formula!r} detail={self.detail}"
        )


@dataclass(frozen=True, slots=True)
class _AbstractValue:
    domain: FormulaDomain
    one_sided_chain: int = 0


@dataclass(frozen=True, slots=True)
class FormulaAnalysis:
    final_domain: FormulaDomain
    final_stack_depth: int
    violations: tuple[str, ...]


_NONNEGATIVE_OUTPUTS = {
    "ABS",
    "TS_STD_5",
    "TS_STD_10",
    "TS_STD_20",
    "TS_RANK_5",
    "TS_RANK_10",
    "TS_RANK_20",
    "TS_QUANTILE_10",
    "TS_ARG_MAX_5",
    "TS_ARG_MIN_5",
    "GT",
    "LT",
    "AND",
    "OR",
}
_SIGNED_OUTPUTS = {
    "TS_CORR_10",
    "COVARIANCE_10",
    "TS_SKEW_10",
    "TS_ZSCORE_10",
    "TS_ZSCORE_20",
    "DELTA",
    "DELTA_5",
    "MOMENTUM_5",
    "MOMENTUM_10",
}
_DOMAIN_PRESERVING = {
    "SIGN",
    "JUMP",
    "DECAY",
    "DELAY1",
    "MAX3",
    "TS_MEAN_5",
    "TS_MEAN_10",
    "TS_MEAN_20",
    "TS_MAX_10",
    "TS_MIN_10",
    "WMA",
    "DELAY4",
    "EMA_5",
    "EMA_20",
    "TS_MIN_20",
    "TS_MAX_20",
    "DECAY_LINEAR_5",
    "SCALE",
    "PRODUCT_5",
    "SIGNED_POWER_2",
    "TS_DECAY_EXP_5",
    "TS_SUM_5",
    "TS_SUM_10",
    "TS_SUM_20",
    "POWER",
    "SIGNED_LOG",
    "SQRT",
    "WINSORIZE",
    "CLIP",
    "SIGMOID",
    "TANH_SQUASH",
}


def _negate(domain: FormulaDomain) -> FormulaDomain:
    return {
        FormulaDomain.NONNEGATIVE: FormulaDomain.NONPOSITIVE,
        FormulaDomain.NONPOSITIVE: FormulaDomain.NONNEGATIVE,
    }.get(domain, domain)


def _join(left: FormulaDomain, right: FormulaDomain) -> FormulaDomain:
    if left is right:
        return left
    if left is FormulaDomain.ZERO:
        return right
    if right is FormulaDomain.ZERO:
        return left
    if FormulaDomain.UNKNOWN in {left, right}:
        return FormulaDomain.UNKNOWN
    return FormulaDomain.SIGNED


def _binary_domain(
    name: str,
    left: FormulaDomain,
    right: FormulaDomain,
) -> FormulaDomain:
    if name in {"MIN", "MAX"}:
        return _join(left, right)
    if name == "ADD":
        if left is FormulaDomain.ZERO:
            return right
        if right is FormulaDomain.ZERO:
            return left
        if left is right and left in {
            FormulaDomain.NONNEGATIVE,
            FormulaDomain.NONPOSITIVE,
        }:
            return left
        return _join(left, right)
    if name == "SUB":
        return _binary_domain("ADD", left, _negate(right))
    if name in {"MUL", "DIV"}:
        if FormulaDomain.ZERO in {left, right}:
            return FormulaDomain.ZERO
        if FormulaDomain.UNKNOWN in {left, right}:
            return FormulaDomain.UNKNOWN
        if FormulaDomain.SIGNED in {left, right}:
            return FormulaDomain.SIGNED
        return (
            FormulaDomain.NONNEGATIVE
            if left is right
            else FormulaDomain.NONPOSITIVE
        )
    return FormulaDomain.UNKNOWN


def _transfer(name: str, operands: list[_AbstractValue]) -> _AbstractValue:
    if name in _NONNEGATIVE_OUTPUTS:
        chain = max((value.one_sided_chain for value in operands), default=0) + 1
        return _AbstractValue(FormulaDomain.NONNEGATIVE, chain)
    if name in _SIGNED_OUTPUTS:
        if all(value.domain is FormulaDomain.ZERO for value in operands):
            return _AbstractValue(FormulaDomain.ZERO)
        return _AbstractValue(FormulaDomain.SIGNED)
    if name == "NEG":
        value = operands[0]
        return _AbstractValue(_negate(value.domain), value.one_sided_chain)
    if name in _DOMAIN_PRESERVING:
        value = operands[0]
        chain = value.one_sided_chain + int(
            value.domain
            in {FormulaDomain.NONNEGATIVE, FormulaDomain.NONPOSITIVE}
        )
        return _AbstractValue(value.domain, chain)
    if name in {"ADD", "SUB", "MUL", "DIV", "MIN", "MAX"}:
        domain = _binary_domain(name, operands[0].domain, operands[1].domain)
        chain = (
            max(value.one_sided_chain for value in operands) + 1
            if domain in {FormulaDomain.NONNEGATIVE, FormulaDomain.NONPOSITIVE}
            else 0
        )
        return _AbstractValue(domain, chain)
    if name in {"GATE", "IF_GT"}:
        domain = _join(operands[-2].domain, operands[-1].domain)
        chain = (
            max(operands[-2].one_sided_chain, operands[-1].one_sided_chain)
            if domain in {FormulaDomain.NONNEGATIVE, FormulaDomain.NONPOSITIVE}
            else 0
        )
        return _AbstractValue(domain, chain)
    return _AbstractValue(FormulaDomain.UNKNOWN)


def analyze_formula_structure(
    formula_tokens: Iterable[int],
    vocab_names: tuple[str, ...] = FORMULA_VOCAB.token_names,
) -> FormulaAnalysis:
    """Interpret postfix stack shape and value domains without executing tensors."""
    tokens = tuple(int(token) for token in formula_tokens)
    operator_offset = FORMULA_VOCAB.operator_offset
    arity_by_name = {
        name: arity for name, _transform, arity in OPS_CONFIG
    }
    stack: list[_AbstractValue] = []
    violations: list[str] = []
    for index, token in enumerate(tokens):
        if token < 0 or token >= len(vocab_names):
            violations.append(f"token {index} is outside the vocabulary: {token}")
            continue
        if token < operator_offset:
            stack.append(_AbstractValue(FormulaDomain.SIGNED))
            continue
        name = vocab_names[token]
        arity = arity_by_name.get(name)
        if arity is None:
            violations.append(f"token {index} has no registered operator: {name}")
            continue
        if len(stack) < arity:
            violations.append(
                f"token {index} operator {name} requires {arity} operands; "
                f"stack depth is {len(stack)}"
            )
            continue
        operands = stack[-arity:]
        del stack[-arity:]
        stack.append(_transfer(name, operands))

    if len(stack) != 1:
        violations.append(
            f"postfix formula must finish with stack depth 1; actual={len(stack)}"
        )
    final = stack[0] if len(stack) == 1 else _AbstractValue(FormulaDomain.UNKNOWN)
    if (
        final.domain in {FormulaDomain.NONNEGATIVE, FormulaDomain.NONPOSITIVE}
        and final.one_sided_chain >= 2
    ):
        violations.append(
            "postfix domain analysis found a terminal one-sided chain: "
            f"domain={final.domain.value} chain={final.one_sided_chain}"
        )
    return FormulaAnalysis(
        final_domain=final.domain,
        final_stack_depth=len(stack),
        violations=tuple(violations),
    )


def validate_formula_structure(
    formula_tokens: list[int],
    vocab_names: tuple[str, ...],
) -> list[str]:
    """Compatibility wrapper returning postfix abstract-interpreter violations."""
    return list(analyze_formula_structure(formula_tokens, vocab_names).violations)


def is_positive_only_op(token_name: str) -> bool:
    return token_name in _NONNEGATIVE_OUTPUTS


def is_infected_propagating(token_name: str) -> bool:
    return token_name in _DOMAIN_PRESERVING or token_name in _NONNEGATIVE_OUTPUTS


def is_sign_restoring(token_name: str) -> bool:
    return token_name in _SIGNED_OUTPUTS or token_name in {"SUB", "NEG", "DIV"}


class StackVM:
    def __init__(self) -> None:
        self.feat_offset = FORMULA_VOCAB.operator_offset
        self.op_map = {
            index + self.feat_offset: config[1]
            for index, config in enumerate(OPS_CONFIG)
        }
        self.arity_map = {
            index + self.feat_offset: config[2]
            for index, config in enumerate(OPS_CONFIG)
        }
        self.operator_names = {
            index + self.feat_offset: config[0]
            for index, config in enumerate(OPS_CONFIG)
        }
        self.positive_only_ids = {
            token
            for token, name in self.operator_names.items()
            if is_positive_only_op(name)
        }
        self.last_error: FormulaEvaluationError | None = None

    @staticmethod
    def _error(
        formula: tuple[int, ...],
        token_index: int,
        operator: str | None,
        kind: FormulaErrorKind,
        detail: str,
    ) -> FormulaEvaluationError:
        return FormulaEvaluationError(
            formula=formula,
            token_index=token_index,
            operator=operator,
            error_kind=kind,
            detail=detail,
        )

    @staticmethod
    def _normalize_output(x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(causal_rolling_zscore(x, 200), -3.0, 3.0)

    def evaluate(self, formula_tokens, feat_tensor: torch.Tensor) -> torch.Tensor:
        try:
            formula = tuple(int(token) for token in formula_tokens)
        except (TypeError, ValueError, OverflowError) as exc:
            raise FormulaEvaluationError(
                formula=(),
                token_index=-1,
                operator=None,
                error_kind=FormulaErrorKind.INVALID_FORMULA,
                detail=f"formula tokens must be exact integers: {exc}",
            ) from exc
        if not formula:
            raise self._error(
                formula, -1, None, FormulaErrorKind.INVALID_FORMULA,
                "formula must not be empty",
            )
        if not isinstance(feat_tensor, torch.Tensor) or feat_tensor.ndim != 3:
            raise self._error(
                formula, -1, None, FormulaErrorKind.SHAPE_ERROR,
                "feature tensor must have shape [symbols, features, time]",
            )
        if feat_tensor.shape[2] == 0:
            raise self._error(
                formula, -1, None, FormulaErrorKind.SHAPE_ERROR,
                "feature tensor time axis must be non-empty",
            )

        stack: list[torch.Tensor] = []
        for index, token in enumerate(formula):
            if token < 0:
                raise self._error(
                    formula, index, None, FormulaErrorKind.INVALID_FORMULA,
                    f"negative token id: {token}",
                )
            if token < self.feat_offset:
                if token >= feat_tensor.shape[1]:
                    raise self._error(
                        formula, index, None, FormulaErrorKind.INVALID_FORMULA,
                        f"feature token {token} exceeds available feature count",
                    )
                value = feat_tensor[:, token, :]
                if not bool(torch.isfinite(value).all()):
                    raise self._error(
                        formula, index, FORMULA_VOCAB.token_names[token],
                        FormulaErrorKind.NONFINITE,
                        "feature operand contains NaN or infinity",
                    )
                stack.append(value)
                continue

            operator = self.operator_names.get(token)
            if operator is None:
                raise self._error(
                    formula, index, None, FormulaErrorKind.INVALID_FORMULA,
                    f"unknown operator token: {token}",
                )
            arity = self.arity_map[token]
            if len(stack) < arity:
                raise self._error(
                    formula, index, operator, FormulaErrorKind.INVALID_FORMULA,
                    f"requires {arity} operands; stack depth is {len(stack)}",
                )
            operands = stack[-arity:]
            del stack[-arity:]
            try:
                result = self.op_map[token](*operands)
            except DomainError as exc:
                raise self._error(
                    formula, index, operator, FormulaErrorKind.DOMAIN_ERROR, str(exc)
                ) from exc
            except ShapeError as exc:
                raise self._error(
                    formula, index, operator, FormulaErrorKind.SHAPE_ERROR, str(exc)
                ) from exc
            except Exception as exc:
                raise self._error(
                    formula,
                    index,
                    operator,
                    FormulaErrorKind.OPERATOR_ERROR,
                    f"{type(exc).__name__}: {exc}",
                ) from exc
            if not isinstance(result, torch.Tensor) or result.shape != operands[0].shape:
                raise self._error(
                    formula, index, operator, FormulaErrorKind.SHAPE_ERROR,
                    "operator output must match operand tensor shape",
                )
            if not bool(torch.isfinite(result).all()):
                raise self._error(
                    formula, index, operator, FormulaErrorKind.NONFINITE,
                    "operator produced NaN or infinity",
                )
            stack.append(result)

        if len(stack) != 1:
            raise self._error(
                formula,
                len(formula),
                None,
                FormulaErrorKind.INVALID_FORMULA,
                f"final stack depth must be 1; actual={len(stack)}",
            )
        analysis = analyze_formula_structure(formula)
        one_sided = [
            violation
            for violation in analysis.violations
            if "terminal one-sided chain" in violation
        ]
        if one_sided:
            final_token = formula[-1]
            raise self._error(
                formula,
                len(formula) - 1,
                self.operator_names.get(final_token),
                FormulaErrorKind.INVALID_FORMULA,
                one_sided[0],
            )
        try:
            normalized = self._normalize_output(stack[0])
        except Exception as exc:
            raise self._error(
                formula,
                len(formula) - 1,
                "NORMALIZE",
                FormulaErrorKind.OPERATOR_ERROR,
                f"{type(exc).__name__}: {exc}",
            ) from exc
        if not bool(torch.isfinite(normalized).all()):
            raise self._error(
                formula,
                len(formula) - 1,
                "NORMALIZE",
                FormulaErrorKind.NONFINITE,
                "normalization produced NaN or infinity",
            )
        self.last_error = None
        return normalized

    def execute(self, formula_tokens, feat_tensor: torch.Tensor):
        """Legacy thin adapter; use evaluate() to preserve structured failures."""
        try:
            return self.evaluate(formula_tokens, feat_tensor)
        except FormulaEvaluationError as exc:
            self.last_error = exc
            return None
