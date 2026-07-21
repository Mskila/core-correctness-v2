# CORE-03 execution, portfolio, ledger, and trade accounting

## Return and cost units

The shared execution path accepts asset log returns and converts them with
`asset_simple = expm1(asset_log_return)`. It then computes
`gross_simple = position * asset_simple`, subtracts turnover and final
liquidation costs as equity fractions, and publishes the authoritative
`net_log_return = log1p(gross_simple - cost)`.

If the simple portfolio return is less than or equal to `-1`, execution is
insolvent and fails closed. Ordinary finite positions, costs, and returns must
remain representable through every arithmetic stage.

## Portfolio metrics

Single-symbol metrics consume that symbol's net log-return path. Multi-symbol
metrics group simultaneous exits and form an equal-weight simple-return
portfolio before converting back to log return. Symbol rows are never flattened
or concatenated. Annualization uses the actual first-entry to final-exit
timestamp span, including irregular calendar months and leap days.

## Ledger and trade definitions

Every ledger row copies the shared gross simple return, equity-fraction cost,
and net log return. Reconciliation therefore checks
`expm1(net_pnl) == gross_pnl - cost` within the published dtype's arithmetic
tolerance.

Reports distinguish:

- `turnover_events`: non-zero position changes in the sampled path;
- `entries`, `exits`, and `reversals`: economic position lifecycle events;
- `liquidation_events`: terminal forced closes;
- `display_trades`: directionally grouped display objects;
- `n_trades`: explicitly defined as `display_trades`.

`EXECUTION_SEMANTICS_VERSION` is
`tanh-threshold-cost-liquidate-v3`. Older reports and strategies remain stored
but are pre-CORE-03 artifacts and must be regenerated under the new version.
