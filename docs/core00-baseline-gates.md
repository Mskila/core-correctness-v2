# CORE-00 baseline and gates

CORE-00 adds characterization assets only. It does not change training,
walk-forward, backtest, checkpoint, strategy-artifact, model, sampler, or VM
production semantics.

## Assets

- `tests/fixtures/core00_ohlcv.json`: fixed local, single-symbol OHLCV data with
  decimal prices, a trend/reversal boundary, zero volume, one explicit time gap,
  and two exit timestamps.
- `tests/fixtures/core00_formula_corpus.json`: legal formula shapes plus isolated
  structural boundaries and known-defect examples. Only formulas classified as
  `valid` carry output digests.
- `tests/fixtures/core00_known_defects.json`: the frozen 30-item review register.
  Every item is marked `do-not-goldenize`; it is not a correctness oracle.
- `tests/support/core00.py`: deterministic fixture loader and two-step reference
  trace, including sampled tokens, factor digests, fold raw scores, final rewards,
  best/elite/factor pools, and model/optimizer/RNG digests.
- `benchmarks/core00.py`: optional stage benchmark. It records samples without
  thresholds and is not a required CI gate.

## Commands

```powershell
python -m pytest tests/core/test_core00_baseline.py -q
python -m pytest -m "core or deterministic or integration" -q
python -m benchmarks.core00 --iterations 3 --output core00-benchmark.json
```

The benchmark JSON records machine, Python, PyTorch, OS, CPU thread count,
dataset fingerprint, commit SHA, and nanosecond samples for sampler, VM, fold
scoring, decision, backward, transaction, logging, and checkpoint stages.

## Scope guard

CORE-00 deliberately does not correct any F-series issue. In particular, the
TS decay and non-finite examples have no expected output digest. Legacy
checkpoint and strategy files remain untouched; later work packages must reject
incompatible artifacts without deleting, relabeling, or silently adapting them.
