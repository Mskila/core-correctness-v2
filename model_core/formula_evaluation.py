"""Formula-evaluation boundary used by serial and future parallel evaluators."""
from __future__ import annotations


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


__all__ = ["evaluate_training_formula"]

