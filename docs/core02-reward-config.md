# CORE-02 reward gates and authoritative training configuration

CORE-02 closes F-002 and F-003 and establishes the configuration dependency
required by later core work.

## Reward contract

- OOS Sortino is applied as a bounded additive adjustment. Better OOS Sortino
  cannot reduce the final score, including when the base score is negative.
- IC is also additive. Positive IC produces a non-negative adjustment and
  negative IC produces a non-positive adjustment for negative, zero, and
  positive rewards. Non-finite gate inputs fail closed.
- Gate evaluation records `base`, `adjustment`, and `final` at debug log level.
- The existing `1.15` and `0.75` values define positive and negative adjustment
  strengths; they no longer multiply the signed reward directly.

## Configuration and timeframe contract

`ModelConfig.training_config_snapshot` is the sole exporter of the immutable,
identity-hashed training configuration. The inert duplicate model settings were
removed from the root `Config` class. The snapshot includes all gate scales and
floors, the neutral exposure band, and the timeframe reward target.

The activity target is two trades per day, preserving the existing H1 target of
one event per 12 bars. The corresponding targets are 720 M1 bars, 12 H1 bars,
3 H4 bars, and 0.5 D1 bars per target trade.

`min_exposure` must be finite and satisfy `0 <= min_exposure < 1`.

## Version and migration

`CORE_SEMANTICS_VERSION` is `3`. Checkpoints, elite/best state, strategies, and
other artifacts created under earlier reward semantics remain on disk but are
rejected by current identity validation and cannot be resumed automatically.
