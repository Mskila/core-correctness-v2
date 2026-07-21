# PERF-02 exact-parity parallel CPU evaluation

PERF-02 parallelizes only formula-local raw evaluation. Sampling and every
state-dependent decision remain in the main process and run in original
`formula_index` order.

## Architecture and ownership

- `RawEvaluationContext` contains CPU features, targets, validity masks,
  timestamps, folds, and the selection index. Parallel evaluators clone these
  tensors into shared memory before workers start, so later caller or worker
  mutations cannot alter the caller's source tensors.
- `RawFormulaEvaluation` contains the formula index, exact factor, fold base
  scores and IC values, selection IC/stability, exposure, constant status, and
  structured expected errors. It never contains correlation penalties, pool
  state, best state, model/optimizer/RNG state, checkpoints, or file handles.
- `ReferenceCpuEvaluator` and `ParallelCpuEvaluator` call the same pure raw
  evaluation function. The existing `model_core/evaluator.py` remains the
  unrelated feature-effectiveness evaluator; the CORE-08 formula boundary in
  `model_core/formula_evaluation.py` is extended instead of overwriting it.
- `ParallelCpuEvaluator` owns one persistent `spawn` process pool per training
  call. Workers use a controlled PyTorch thread count. Linux `forkserver` is
  also accepted and parity-tested when the platform provides it.
- The main process rejects incomplete batches, sorts complete results by
  `formula_index`, then applies IC gates, repetition/correlation penalties,
  shadow factor pool, elite/best updates, pending actions, optimization, history,
  checkpoints, and artifact publication serially.

## Failure and lifecycle contract

Timeout, worker crash, unexpected worker exception, or main-process interrupt
fails the whole raw batch. The pool cancels pending futures, terminates and joins
workers, and the enclosing batch transaction restores model, optimizer, RNG,
pools, history, and artifacts. No partial results enter optimization. The
training service closes the evaluator in a `finally` block, and normal training
also closes it before final publication.

The default remains `ModelConfig.EVALUATION_WORKERS = 1` (reference CPU). A
parallel worker count is enabled only by explicit configuration after benchmark
and parity gates; it is not derived from the machine's logical-core count.

## Benchmark

```powershell
python -m benchmarks.perf02 --iterations 3 --workers 1 2 4 8 --output perf02.json
```

The benchmark reports executor construction, cold-start batch time,
steady-state median/p95 end-to-end raw batch time, formulas/second, exact
digest, and a recommendation chosen only from the
measured 1/2/4/8 candidates. It is informational rather than a CI threshold.

On the 2026-07-22 Windows CPU verification host (PyTorch 2.13.0+cpu, 48
formula evaluations), all variants produced digest
`a776973ba25ff01fc2a1fc257e5de62f629d8cb4b21c4d101d4cd21c2d1fa07f`.
Measured steady-state throughput was approximately 199/317/456/668 formulas
per second for 1/2/4/8 workers, so this host recommended 8 workers (about 3.36x
reference). The report keeps cold-start time separate; on short runs, process
startup can outweigh the steady-state gain.

## Required parity evidence

- raw output equality and deterministic index ordering;
- complete engine trace equality for workers 1/2/4/8, including history, best,
  elite, factor pool, model, optimizer, Python/NumPy/PyTorch RNG;
- timeout, actual worker exit, main interrupt, persistent reuse, input isolation,
  and no-orphan cleanup;
- Windows spawn and Linux spawn/forkserver where available;
- existing deterministic checkpoint/resume and complete core gates.

No GPU, VM semantic, reward semantic, execution semantic, holdout semantic,
checkpoint schema, legacy artifact compatibility, or serial decision change is
part of PERF-02. Old artifacts remain preserved and are never silently migrated
or deleted.
