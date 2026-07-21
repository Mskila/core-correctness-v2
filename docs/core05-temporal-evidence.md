# CORE-05 temporal isolation and evidence lifecycle

CORE-05 separates model selection from the one independent final conclusion.

## Three data layers

- `search-train` drives parameter updates.
- `search-validation` drives candidate ranking and strategy selection.
- `final-untouched-holdout` is unavailable to training and live Web progress.

`ExperimentProtocol` enforces this boundary. The final evaluator cannot load the
holdout before a strategy fingerprint is frozen. Loading claims the holdout for
that experiment, and a second claim is rejected. Any candidate evaluation after
freeze is also rejected with an instruction to create a new experiment identity.
The final dataset must start strictly after the search dataset ends.

## Evidence names

Newly published training strategies use `strategy-v3` and contain:

- `selection_evidence`: expanding-window folds, evaluated-candidate count,
  random seed, fold count, protocol name, minimum observations, and minimum
  trade-event requirement;
- `final_oos_evidence`: `null` until the frozen strategy is evaluated once on
  its registered final holdout, then the exact dataset identity, time range,
  metrics, experiment identity, frozen strategy fingerprint, and timestamp.

Legacy `strategy-v2.fold_evidence` remains readable only as selection evidence.
It is never relabelled as final or independent OOS evidence.

## Statistical minimum

The selection protocol rejects folds below 200 observations. The hashed training
configuration records the minimum fold size, required metrics (`ic`,
`net_return`, `sortino`), and the minimum trade-event count. The lower two-point
limit remains only inside the execution kernel for isolated numerical testing;
it cannot be selected by the walk-forward protocol.

## Reports

Backtest reports include `evidence_scope` and distinguish:

- `internal_selection_replay` for training-data replay;
- `independent_out_of_sample_evaluation` for a strict future dataset not
  registered as the final holdout;
- `independent_final_oos` only when the test dataset equals the strategy's
  persisted `final_oos_evidence.dataset` identity.

Internal validation metrics are never described as final OOS.
