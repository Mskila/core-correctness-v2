"""Persistent spawn-process raw formula evaluator for PERF-02."""
from __future__ import annotations

import concurrent.futures
import multiprocessing
import os
from collections.abc import Sequence

import torch

from .formula_evaluation import (
    RawEvaluationContext,
    RawFormulaEvaluation,
    ReferenceCpuEvaluator,
)


_WORKER_EVALUATOR: ReferenceCpuEvaluator | None = None


def _initialize_worker(context: RawEvaluationContext, torch_threads: int) -> None:
    global _WORKER_EVALUATOR
    torch.set_num_threads(torch_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    _WORKER_EVALUATOR = ReferenceCpuEvaluator(context)


def _evaluate_in_worker(
    formula_index: int,
    formula: tuple[int, ...],
    step: int,
) -> tuple[int, RawFormulaEvaluation]:
    evaluator = _WORKER_EVALUATOR
    if evaluator is None:
        raise RuntimeError("parallel evaluator worker was not initialized")
    result = evaluator.evaluate_batch((formula,), step=step)[0]
    if result.formula_index != 0:
        raise RuntimeError("worker-local formula index contract was violated")
    return os.getpid(), RawFormulaEvaluation(
        formula_index=formula_index,
        formula=result.formula,
        factor=result.factor,
        fold_train_scores=result.fold_train_scores,
        fold_val_scores=result.fold_val_scores,
        fold_ics=result.fold_ics,
        selection_ic=result.selection_ic,
        selection_ic_stability=result.selection_ic_stability,
        exposure=result.exposure,
        constant=result.constant,
        error=result.error,
    )


def _shared_context(context: RawEvaluationContext) -> RawEvaluationContext:
    def shared(value: torch.Tensor) -> torch.Tensor:
        copied = value.detach().clone().contiguous()
        copied.share_memory_()
        return copied

    return RawEvaluationContext(
        features=shared(context.features),
        target_ret=shared(context.target_ret),
        target_valid=shared(context.target_valid),
        bar_time_ns=shared(context.bar_time_ns),
        folds=context.folds,
        selection_index=shared(context.selection_index),
        cost_rate=context.cost_rate,
        timeframe=context.timeframe,
        target_trades_per_day=context.target_trades_per_day,
        oos_gate_scale=context.oos_gate_scale,
    )


class ParallelCpuEvaluator:
    """Persist one spawn pool and return only complete index-sorted batches."""

    def __init__(
        self,
        context: RawEvaluationContext,
        *,
        workers: int,
        timeout_seconds: float = 300.0,
        torch_threads: int = 1,
        start_method: str = "spawn",
    ) -> None:
        if type(context) is not RawEvaluationContext:
            raise TypeError("context must be an exact RawEvaluationContext")
        context.validate()
        if type(workers) is not int or workers < 1:
            raise ValueError("workers must be a positive built-in int")
        if type(torch_threads) is not int or torch_threads < 1:
            raise ValueError("torch_threads must be a positive built-in int")
        if type(timeout_seconds) not in (int, float) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if start_method not in ("spawn", "forkserver"):
            raise ValueError("start_method must be 'spawn' or 'forkserver'")
        if start_method not in multiprocessing.get_all_start_methods():
            raise ValueError(f"start_method is unavailable: {start_method}")
        self.workers = workers
        self.timeout_seconds = float(timeout_seconds)
        self._worker_pids: tuple[int, ...] = ()
        self._closed = False
        self._executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context(start_method),
            initializer=_initialize_worker,
            initargs=(_shared_context(context), torch_threads),
        )

    @property
    def worker_pids(self) -> tuple[int, ...]:
        return self._worker_pids

    @property
    def closed(self) -> bool:
        return self._closed

    def evaluate_batch(
        self,
        formulas: Sequence[Sequence[int]],
        *,
        step: int,
    ) -> list[RawFormulaEvaluation]:
        if self._closed:
            raise RuntimeError("parallel evaluator is closed")
        futures = [
            self._executor.submit(
                _evaluate_in_worker,
                index,
                tuple(int(token) for token in formula),
                step,
            )
            for index, formula in enumerate(formulas)
        ]
        try:
            done, pending = concurrent.futures.wait(
                futures,
                timeout=self.timeout_seconds,
                return_when=concurrent.futures.ALL_COMPLETED,
            )
        except BaseException:
            self._abort_pool()
            raise
        if pending:
            self._abort_pool()
            raise TimeoutError(
                "parallel raw evaluation batch timed out: "
                f"completed={len(done)} pending={len(pending)}"
            )
        try:
            completed = [future.result() for future in futures]
        except BaseException:
            self._abort_pool()
            raise
        self._worker_pids = tuple(sorted({pid for pid, _ in completed}))
        results = sorted((result for _, result in completed), key=lambda item: item.formula_index)
        if [item.formula_index for item in results] != list(range(len(formulas))):
            self._abort_pool()
            raise RuntimeError("parallel raw evaluation returned an incomplete batch")
        return results

    def _abort_pool(self) -> None:
        if self._closed:
            return
        processes = tuple(getattr(self._executor, "_processes", {}).values())
        self._executor.shutdown(wait=False, cancel_futures=True)
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=5.0)
            if process.is_alive() and hasattr(process, "kill"):
                process.kill()
                process.join(timeout=5.0)
        self._closed = True
        self._worker_pids = ()

    def close(self) -> None:
        if self._closed:
            return
        self._executor.shutdown(wait=True, cancel_futures=True)
        self._closed = True
        self._worker_pids = ()

    def __enter__(self) -> "ParallelCpuEvaluator":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if exc_type is None:
            self.close()
        else:
            self._abort_pool()

    def __del__(self) -> None:
        try:
            self._abort_pool()
        except BaseException:
            pass


__all__ = ["ParallelCpuEvaluator"]
