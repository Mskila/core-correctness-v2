"""Main-process-only mutation and publication of evaluated formula decisions."""
from __future__ import annotations

import traceback


def commit_pending_actions(
    engine,
    pending_actions: list[tuple],
    *,
    publish_strategy: bool = True,
) -> list[str]:
    """Apply one validated batch of decisions, or restore all decision state."""
    before_best_score = engine.best_score
    before_best_formula = engine.best_formula
    before_best_snapshot = engine._best_snapshot
    before_best_update_step = engine._best_update_step
    before_stagnation_steps = engine._stagnation_steps
    before_factor_pool = list(engine.factor_pool)
    before_factor_counter = engine._factor_pool_counter
    before_elite_pool = list(engine._elite_pool)
    before_elite_counter = engine._elite_counter
    messages: list[str] = []
    has_winner = False
    action = snapshot = buffered_factor = None
    try:
        for action in pending_actions:
            if action[0] == "elite":
                _, final_val, formula, action_step = action
                engine._update_elite_pool(final_val, formula, action_step)
                continue
            if len(action) == 10:
                (
                    _,
                    final_val,
                    formula,
                    snapshot,
                    action_step,
                    buffered_factor,
                    old_best,
                    ic_value,
                    exposure,
                    fold_evidence,
                ) = action
            else:
                (
                    _,
                    final_val,
                    formula,
                    snapshot,
                    action_step,
                    buffered_factor,
                    old_best,
                    ic_value,
                    exposure,
                ) = action
                fold_evidence = None
            engine.best_score = final_val
            engine.best_formula = formula
            if fold_evidence is not None:
                engine.best_metrics = {
                    "validation_score": final_val,
                    "fold_evidence": fold_evidence,
                }
            engine._best_snapshot = snapshot
            engine._best_update_step = action_step
            engine._stagnation_steps = 0
            engine._update_factor_pool(final_val, buffered_factor)
            has_winner = True
            messages.append(
                f"[!] 新最优 @ 第{action_step}步: 验证={final_val:.3f} "
                f"(原 {old_best:.3f}，+{final_val-old_best:.3f}) "
                f"IC={ic_value:.4f} 暴露度={exposure:.1%} | "
                f"{formula}\n    {engine._decode_formula(formula)}"
            )
        if has_winner and publish_strategy:
            engine._save_strategy_live()
    except BaseException as failure:
        engine.best_score = before_best_score
        engine.best_formula = before_best_formula
        engine._best_snapshot = before_best_snapshot
        engine._best_update_step = before_best_update_step
        engine._stagnation_steps = before_stagnation_steps
        engine.factor_pool = before_factor_pool
        engine._factor_pool_counter = before_factor_counter
        engine._elite_pool = before_elite_pool
        engine._elite_counter = before_elite_counter
        pending_actions.clear()
        messages.clear()
        action = snapshot = buffered_factor = None
        traceback.clear_frames(failure.__traceback__)
        raise
    pending_actions.clear()
    action = snapshot = buffered_factor = None
    return messages


__all__ = ["commit_pending_actions"]
