# PERF-01 single-CPU optimization contract

PERF-01 improves the existing serial CPU path without changing formula, VM,
execution, reward, walk-forward, holdout, checkpoint, or resume semantics.
`optimized_cpu` is not introduced as a new default mode; the optimized path is
the same supported CPU path and remains guarded by exact-parity tests.

## Implemented boundaries

- Each sampled formula is executed by `StackVM` once per batch. The validated
  factor is retained only until scoring completes and is then released.
- Walk-forward target, validity, and timestamp segments are prepared once per
  training run. `evaluate_fold` remains the reference/compatibility evaluator.
- Transformer causal masks are cached by sequence length, device, and runtime
  dtype.
- Trusted core consumers borrow `ExecutionResult` tensors under a read-only
  contract. Public attribute access and `copy_tensor()` remain defensive.
- Device-created scalar fallbacks inherit dtype and device from their reference
  tensor.
- Factor-pool storage remains exact because correlation filtering needs full
  factors, but is strictly bounded by `FACTOR_TOP_K`.
- Detailed terminal logging is emitted on the first step, every 10 steps, and
  the final step. Every training-history series is still updated every step.
- Transaction snapshot buffers were not changed: profiling did not establish
  them as the dominant cost, and changing rollback state would increase risk.

## Benchmark

Run from the repository root:

```powershell
python -m benchmarks.perf01 --iterations 5 --output perf01.json
```

The benchmark is informational rather than a required CI threshold. It reports
formula throughput, median/p95 step and stage timing, retained factor bytes,
Python allocation peak, and an exact factor digest. On the 2026-07-22 Windows
CPU verification environment (PyTorch 2.13.0+cpu, six CORE-00 valid formulas),
the measured reference/optimized throughput was approximately 645/1722
formulas per second (2.67x), with `torch.equal` parity.

## Required evidence

- `tests/unit/test_alphagpt_performance.py`
- `tests/unit/test_training_alignment.py`
- `tests/unit/test_deterministic_resume.py`
- execution/backtest unit and property suites
- CORE-00 reference trace and required integration gates

Old checkpoints and strategies remain preserved and are never silently
converted or deleted.
