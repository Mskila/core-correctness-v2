# CORE-04 operator, postfix, VM, and feature semantics

## Operator and postfix semantics

`TS_DECAY_EXP_5` uses increasing chronological weights
`[1, 2, 4, 8, 16] / 31`, so the newest observation has the greatest weight.

Formula validation is a postfix abstract interpretation. Each feature pushes a
value domain and each registered operator pops its declared arity, applies a
domain transfer, and pushes the result. Stack underflow, unknown tokens, a final
stack depth other than one, and terminal repeated one-sided transformations are
reported as explicit violations. `TS_STD_*` and `TS_QUANTILE_10` produce
nonnegative domains; they are not sign-restoring operators.

## VM error model

`StackVM.evaluate()` either returns one finite `[symbols, time]` tensor or raises
`FormulaEvaluationError`. The error publishes the formula, token index,
operator, bounded detail, and one of these categories:

- `invalid_formula`
- `domain_error`
- `nonfinite`
- `shape_error`
- `operator_error`

Intermediate NaN or infinity is never converted into a signal. Operators no
longer use a blanket `nan_to_num` fallback. `StackVM.execute()` remains only as
a legacy adapter: it returns `None` and retains the structured reason in
`last_error`.

Training records per-step category counts and bounded error samples. Rejected
formulas receive the rejection sentinel and cannot update scoring evidence,
best strategy, elite state, or the factor pool. Unexpected exceptions still
roll back the entire batch.

## Feature and volume metadata

Every registered feature has auditable name, category, definition,
implementation, and volume-input metadata in `FEATURE_METADATA_BY_NAME`.
Volume-derived features consume `dataset.volume`.

Canonical datasets publish `volume_type` as one of `tick`, `real`, `quote`, or
`base`, and include it in the data fingerprint. Source columns named
`tick_volume`, `real_volume`, `quote_volume`, or `base_volume` are inferred;
the generic historical `volume` column defaults to `tick` unless the caller
declares another type.

## Compatibility

CORE semantics are version `4`, the vocabulary schema tag is
`5.1-core04-postfix-vm`, the dataset schema is `ohlcv-v4`, and data canonicalization is
`float32-le-ns-gap-volume-v2`. Pre-CORE-04 checkpoints and strategies remain
stored but are incompatible with formal resume or replay and must be retrained.
