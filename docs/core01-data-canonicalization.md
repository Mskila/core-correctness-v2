# CORE-01 data canonicalization and migration

CORE-01 closes F-001, F-006 (Parquet dependency portion), F-023, and F-027.

## Canonical data contract

- All OHLCV values are validated as finite numeric inputs and then quantized to
  little-endian float32. Exact float64-to-float32-to-float64 reversal is not a
  validity requirement.
- Dataset fingerprints hash the canonical float32 bytes, UTC int64 nanosecond
  timestamps, and canonicalization metadata.
- Numeric timestamps require an explicit `s`, `ms`, `us`, or `ns` declaration.
  Datetime/string timestamps are normalized to UTC nanoseconds. Dataset identity
  records the canonical unit as `ns`.
- Gap policy is part of dataset identity. `reject` fails on a large gap;
  `segment` creates segment IDs and invalidates labels crossing a segment;
  `explicitly-allowed` retains one segment by explicit caller choice.
- MT5 and Parquet managers default to the same `segment` policy. Parquet numeric
  time columns must pass `numeric_time_unit` to `ParquetDataManager` or
  `inspect_parquet_file`.

## Version and migration

- `DATA_SCHEMA_VERSION`: `ohlcv-v2` -> `ohlcv-v3`
- `DATA_CANONICALIZATION_VERSION`: `float32-le-ns-gap-v1`
- `DatasetIdentity` adds `canonicalization_version`, `time_unit`, and
  `gap_policy`.

Pre-CORE-01 identities, checkpoints, strategies, and reports remain available
as historical files but cannot be loaded into the new identity contract. They
must not be edited, relabeled, deleted, or silently upgraded. New training must
start from canonicalized data and produces a new identity/fingerprint.

## Parquet dependency

The core requirements pin `pyarrow==25.0.0`. The CORE-01 gate performs a real
Parquet write/read through pandas and verifies MT5/Parquet tensor, target, time,
and identity parity for the same declared source data.
