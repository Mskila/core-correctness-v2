# CORE-08 module boundaries, process lifecycle, and error telemetry

CORE-08 keeps the corrected training mathematics unchanged while separating
responsibilities that need independent tests and future replacement.

## Boundaries

- `model_core.sampling`: grammar-constrained formula sampling.
- `model_core.formula_evaluation`: one-formula VM evaluation and fail-closed
  rejection of the legacy `None` sentinel.
- `model_core.serial_decision`: ordered main-process updates to best, elite, and
  factor-pool state.
- `model_core.batch_transaction`: the transaction entry boundary used before
  the first stochastic draw in a batch. The existing compatibility method on
  `AlphaEngine` remains during migration.
- `model_core.artifact_publication`: training-history publication boundary.
- `model_core.checkpoint_codec`: checkpoint save/load boundary.
- `backtest_viz.output_publication`: atomic report/chart output boundary.

Compatibility methods remain importable so existing callers do not need a
flag day migration. Sampling, formula evaluation, serial decision, and batch
transaction construction are already invoked through the extracted modules.

## Process lifecycle

Training and backtest subprocesses are created in an addressable process group.
Stop is synchronous and follows this sequence:

1. terminate the process tree;
2. wait for the configured timeout;
3. kill the process tree only after timeout;
4. wait again and require a real exit code.

The manager refreshes the job after the wait, closes the log, and exposes only
the terminal `completed`, `failed`, or `stopped` states for finished jobs. A
failure to stop raises `ProcessLifecycleError`; the manager does not claim that
a still-running process stopped.

## Error telemetry

`ErrorTelemetry` records an explicit category counter and a bounded diagnostic
tail. Manager status responses expose this as `errors`. Message count and
message length are bounded; errors are not converted into training decisions.
Lifecycle failures remain fail-closed.

Formula evaluation already records per-category counts and bounded examples in
training history. Batch transactions already restore model, optimizer, RNG,
best/elite/factor pools, history, and transaction-owned artifacts after a
mid-batch failure. CORE-08 retains those exact rollback contracts.

## Required evidence

Run:

```powershell
python -m pytest tests/core/test_core08_boundaries_lifecycle.py -q
python -m pytest tests/property/test_prop_engine.py -q
python -m pytest tests/unit/test_training_alignment.py -k "batch or formula or transaction" -q
python -m pytest tests/core/test_core00_baseline.py::test_short_reference_trace_repeats_exactly_for_same_seed -q
```

The reference trace digest must remain unchanged. Security and network tests
are outside this work package.
