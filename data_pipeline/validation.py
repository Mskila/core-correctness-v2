"""Canonical OHLCV validation and stable dataset identity."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import re
from typing import Mapping
import warnings

import numpy as np
import pandas as pd

from model_core.semantics import (
    DATA_CANONICALIZATION_VERSION,
    DATA_SCHEMA_VERSION,
    DataValidationError,
)


_CANONICAL_COLUMNS = ("time", "open", "high", "low", "close", "volume")
_VALUE_COLUMNS = ("open", "high", "low", "close", "volume")
_TIME_UNIT_NS = {
    "s": 1_000_000_000,
    "ms": 1_000_000,
    "us": 1_000,
    "ns": 1,
}
_TIMEFRAME_BY_MT5_VALUE = {
    1: "M1",
    5: "M5",
    15: "M15",
    30: "M30",
    16385: "H1",
    16388: "H4",
    16408: "D1",
    32769: "W1",
    49153: "MN1",
}
_TIMEFRAME_NS = {
    "M1": 60 * 1_000_000_000,
    "M5": 5 * 60 * 1_000_000_000,
    "M15": 15 * 60 * 1_000_000_000,
    "M30": 30 * 60 * 1_000_000_000,
    "H1": 60 * 60 * 1_000_000_000,
    "H4": 4 * 60 * 60 * 1_000_000_000,
    "D1": 24 * 60 * 60 * 1_000_000_000,
    "W1": 7 * 24 * 60 * 60 * 1_000_000_000,
}
_CANONICAL_TIMEFRAMES = frozenset({*_TIMEFRAME_NS, "MN1"})
_GAP_POLICIES = frozenset({"reject", "segment", "explicitly-allowed"})
_DATASET_IDENTITY_FIELDS = (
    "schema_version",
    "canonicalization_version",
    "time_unit",
    "gap_policy",
    "symbol",
    "timeframe",
    "start_time_ns",
    "end_time_ns",
    "bars",
    "data_fingerprint",
    "time_fingerprint",
)
_SHA256_HEX_PATTERN = re.compile(r"[0-9a-f]{64}")
_MISSING_IDENTITY_FIELD = object()


def _identity_actual(value: object) -> str:
    return f"{value!r} (type={type(value).__name__})"


def _raise_identity_field_error(
    field_name: str,
    *,
    expected: str,
    actual: object,
) -> None:
    raise DataValidationError(
        "invalid dataset identity: "
        f"field={field_name}; expected={expected}; actual={_identity_actual(actual)}"
    )


def _validate_identity_schema(value: Mapping[object, object]) -> None:
    expected = set(_DATASET_IDENTITY_FIELDS)
    actual = set(value)
    missing = sorted(expected - actual)
    extra = sorted(repr(key) for key in actual - expected)
    if missing or extra:
        actual_fields = sorted(repr(key) for key in actual)
        raise DataValidationError(
            "invalid dataset identity schema: "
            f"expected={list(_DATASET_IDENTITY_FIELDS)!r}; "
            f"actual={actual_fields!r}; missing={missing!r}; extra={extra!r}"
        )


def _validate_dataset_identity_values(value: Mapping[str, object]) -> None:
    schema_version = value["schema_version"]
    if type(schema_version) is not str or schema_version != DATA_SCHEMA_VERSION:
        _raise_identity_field_error(
            "schema_version",
            expected=f"exact str {DATA_SCHEMA_VERSION!r}",
            actual=schema_version,
        )

    canonicalization_version = value["canonicalization_version"]
    if (
        type(canonicalization_version) is not str
        or canonicalization_version != DATA_CANONICALIZATION_VERSION
    ):
        _raise_identity_field_error(
            "canonicalization_version",
            expected=f"exact str {DATA_CANONICALIZATION_VERSION!r}",
            actual=canonicalization_version,
        )

    time_unit = value["time_unit"]
    if type(time_unit) is not str or time_unit != "ns":
        _raise_identity_field_error(
            "time_unit", expected="exact str 'ns'", actual=time_unit
        )

    gap_policy = value["gap_policy"]
    if type(gap_policy) is not str or gap_policy not in _GAP_POLICIES:
        _raise_identity_field_error(
            "gap_policy",
            expected=f"exact str in {sorted(_GAP_POLICIES)!r}",
            actual=gap_policy,
        )

    symbol = value["symbol"]
    if type(symbol) is not str or not symbol.strip():
        _raise_identity_field_error(
            "symbol",
            expected="non-empty non-whitespace exact str",
            actual=symbol,
        )

    timeframe = value["timeframe"]
    if type(timeframe) is not str or timeframe not in _CANONICAL_TIMEFRAMES:
        _raise_identity_field_error(
            "timeframe",
            expected=f"canonical exact str in {sorted(_CANONICAL_TIMEFRAMES)!r}",
            actual=timeframe,
        )

    for field_name in ("start_time_ns", "end_time_ns"):
        timestamp = value[field_name]
        if type(timestamp) is not int:
            _raise_identity_field_error(
                field_name,
                expected="exact int (bool excluded)",
                actual=timestamp,
            )

    start_time_ns = value["start_time_ns"]
    end_time_ns = value["end_time_ns"]
    if start_time_ns > end_time_ns:  # type: ignore[operator]
        raise DataValidationError(
            "invalid dataset identity: "
            "field=start_time_ns; expected=<= end_time_ns; "
            f"actual={start_time_ns!r} > {end_time_ns!r}; "
            "field=end_time_ns; expected=>= start_time_ns; "
            f"actual={end_time_ns!r} < {start_time_ns!r}"
        )

    bars = value["bars"]
    if type(bars) is not int or bars <= 0:  # type: ignore[operator]
        _raise_identity_field_error(
            "bars",
            expected="positive exact int (bool excluded)",
            actual=bars,
        )

    for field_name in ("data_fingerprint", "time_fingerprint"):
        fingerprint = value[field_name]
        if (
            type(fingerprint) is not str
            or _SHA256_HEX_PATTERN.fullmatch(fingerprint) is None
        ):
            _raise_identity_field_error(
                field_name,
                expected="64-character lowercase hexadecimal SHA-256 exact str",
                actual=fingerprint,
            )


@dataclass(frozen=True, init=False)
class DatasetIdentity:
    schema_version: str
    canonicalization_version: str
    time_unit: str
    gap_policy: str
    symbol: str
    timeframe: str
    start_time_ns: int
    end_time_ns: int
    bars: int
    data_fingerprint: str
    time_fingerprint: str

    def __init__(
        self,
        schema_version: object = _MISSING_IDENTITY_FIELD,
        canonicalization_version: object = _MISSING_IDENTITY_FIELD,
        time_unit: object = _MISSING_IDENTITY_FIELD,
        gap_policy: object = _MISSING_IDENTITY_FIELD,
        symbol: object = _MISSING_IDENTITY_FIELD,
        timeframe: object = _MISSING_IDENTITY_FIELD,
        start_time_ns: object = _MISSING_IDENTITY_FIELD,
        end_time_ns: object = _MISSING_IDENTITY_FIELD,
        bars: object = _MISSING_IDENTITY_FIELD,
        data_fingerprint: object = _MISSING_IDENTITY_FIELD,
        time_fingerprint: object = _MISSING_IDENTITY_FIELD,
        **extra_fields: object,
    ) -> None:
        supplied = {
            field_name: field_value
            for field_name, field_value in zip(
                _DATASET_IDENTITY_FIELDS,
                (
                    schema_version,
                    canonicalization_version,
                    time_unit,
                    gap_policy,
                    symbol,
                    timeframe,
                    start_time_ns,
                    end_time_ns,
                    bars,
                    data_fingerprint,
                    time_fingerprint,
                ),
                strict=True,
            )
            if field_value is not _MISSING_IDENTITY_FIELD
        }
        supplied.update(extra_fields)
        _validate_identity_schema(supplied)
        _validate_dataset_identity_values(supplied)
        for field_name in _DATASET_IDENTITY_FIELDS:
            object.__setattr__(self, field_name, supplied[field_name])

    def to_dict(self) -> dict[str, str | int]:
        state: dict[str, object] = dict(vars(self))
        _validate_identity_schema(state)
        _validate_dataset_identity_values(state)
        return {
            field_name: state[field_name]  # type: ignore[misc]
            for field_name in _DATASET_IDENTITY_FIELDS
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "DatasetIdentity":
        if not isinstance(value, Mapping):
            _raise_identity_field_error(
                "identity",
                expected="Mapping with exact DatasetIdentity fields",
                actual=value,
            )
        _validate_identity_schema(value)
        payload: dict[str, object] = {}
        for field_name in _DATASET_IDENTITY_FIELDS:
            try:
                payload[field_name] = value[field_name]
            except (KeyError, TypeError, ValueError) as exc:
                raise DataValidationError(
                    "invalid dataset identity: "
                    f"field={field_name}; expected=successful Mapping lookup; "
                    f"actual={_identity_actual(exc)}"
                ) from exc
        return cls(**payload)  # type: ignore[arg-type]


@dataclass(frozen=True, init=False)
class CanonicalDataset:
    _frame: pd.DataFrame = field(repr=False, compare=False)
    identity: DatasetIdentity
    gap_count: int
    _segment_ids: np.ndarray = field(repr=False, compare=False)

    def __init__(
        self,
        frame: pd.DataFrame,
        identity: DatasetIdentity,
        gap_count: int,
        segment_ids: np.ndarray,
    ) -> None:
        object.__setattr__(self, "_frame", frame.copy(deep=True))
        object.__setattr__(self, "identity", identity)
        object.__setattr__(self, "gap_count", gap_count)
        object.__setattr__(
            self,
            "_segment_ids",
            np.asarray(segment_ids, dtype=np.int64).copy(),
        )

    @property
    def frame(self) -> pd.DataFrame:
        """Return a defensive copy so content cannot diverge from identity."""
        return self._frame.copy(deep=True)

    @property
    def segment_ids(self) -> np.ndarray:
        """Return defensive per-bar segment identifiers."""
        return self._segment_ids.copy()


def normalize_timeframe_name(value: str | int) -> str:
    """Return the canonical MT5 timeframe name."""
    if isinstance(value, bool):
        raise DataValidationError(f"unknown timeframe: {value!r}")
    if isinstance(value, str):
        normalized = value.strip().upper()
        if normalized in {*_TIMEFRAME_NS, "MN1"}:
            return normalized
    elif isinstance(value, int) and value in _TIMEFRAME_BY_MT5_VALUE:
        return _TIMEFRAME_BY_MT5_VALUE[value]
    raise DataValidationError(f"unknown timeframe: {value!r}")


def assert_minimum_bars(actual: int, required: int, *, context: str) -> None:
    """Fail closed when a caller's explicit bar requirement is not met."""
    if actual < required:
        raise DataValidationError(
            f"{context}: insufficient bars; expected at least {required}, actual {actual}"
        )


def _contains_bool(values: pd.Series) -> bool:
    return any(
        isinstance(value, (bool, np.bool_))
        for value in values.to_numpy(copy=False)
    )


def _integer_numeric_timestamps(values: pd.Series) -> list[int]:
    if _contains_bool(values):
        raise DataValidationError("numeric timestamps cannot contain boolean values")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            numeric = pd.to_numeric(values, errors="raise")
    except Exception as exc:
        raise DataValidationError(f"invalid numeric timestamp: {exc}") from exc

    array = numeric.to_numpy(copy=False)
    if np.iscomplexobj(array):
        raise DataValidationError("numeric timestamps must be real integers")
    if pd.api.types.is_float_dtype(numeric.dtype):
        numpy_dtype = np.dtype(
            getattr(numeric.dtype, "numpy_dtype", numeric.dtype)
        )
        array = numeric.to_numpy(
            dtype=numpy_dtype,
            na_value=np.nan,
            copy=False,
        )
        if not np.isfinite(array).all():
            raise DataValidationError("invalid time: non-finite timestamp")
        if not np.equal(array, np.trunc(array)).all():
            raise DataValidationError("numeric timestamps must be integer values")
        precision_bits = np.finfo(numpy_dtype).nmant + 1
        lossless_integer_limit = np.array(2**precision_bits, dtype=numpy_dtype)
        if (np.abs(array) > lossless_integer_limit).any():
            raise DataValidationError(
                "floating numeric timestamps exceed lossless integer precision "
                f"for source dtype {numeric.dtype}"
            )
    try:
        return [int(value) for value in array]
    except (OverflowError, TypeError, ValueError) as exc:
        raise DataValidationError(f"invalid numeric timestamp: {exc}") from exc


def _scaled_time_ns(value: int, unit: str) -> int | None:
    scaled = value * _TIME_UNIT_NS[unit]
    if scaled < pd.Timestamp.min.value or scaled > pd.Timestamp.max.value:
        return None
    return scaled


def _normalize_numeric_time_unit(value: object) -> str:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _TIME_UNIT_NS:
            return normalized
    raise DataValidationError(
        "numeric time unit must be one of s, ms, us or ns"
    )


def _normalize_gap_policy(value: object) -> str:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _GAP_POLICIES:
            return normalized
    raise DataValidationError(
        "gap policy must be one of reject, segment or explicitly-allowed"
    )


def _numeric_time_to_utc(values: pd.Series, *, unit: str) -> pd.Series:
    integers = _integer_numeric_timestamps(values)
    if not integers:
        return pd.Series(pd.to_datetime([], utc=True)).astype(
            "datetime64[ns, UTC]"
        )
    if len(set(integers)) != len(integers):
        raise DataValidationError("duplicate timestamp")

    scaled = [_scaled_time_ns(value, unit) for value in integers]
    if any(value is None for value in scaled):
        raise DataValidationError(
            f"numeric timestamp is out of range for declared unit {unit}"
        )
    scaled_ns = [int(value) for value in scaled if value is not None]
    return pd.Series(pd.to_datetime(scaled_ns, unit="ns", utc=True)).astype(
        "datetime64[ns, UTC]"
    )


def _to_utc_time(
    values: pd.Series,
    *,
    numeric_time_unit: str | None,
) -> pd.Series:
    if pd.api.types.is_numeric_dtype(values.dtype):
        if numeric_time_unit is None:
            raise DataValidationError(
                "numeric timestamp unit provenance is required; pass "
                "numeric_time_unit as s, ms, us or ns"
            )
        converted = _numeric_time_to_utc(values, unit=numeric_time_unit)
    else:
        if pd.api.types.is_object_dtype(values.dtype) and any(
            pd.api.types.is_number(value)
            or isinstance(value, (bool, np.bool_))
            for value in values.to_numpy(copy=False)
        ):
            raise DataValidationError(
                "object time values cannot contain numeric values"
            )
        try:
            converted = pd.Series(
                pd.to_datetime(values, utc=True, errors="coerce")
            )
            if converted.isna().any():
                raise DataValidationError(
                    "invalid time: timestamp cannot be converted to UTC"
                )
            return converted.astype("datetime64[ns, UTC]")
        except DataValidationError:
            raise
        except Exception as exc:
            raise DataValidationError(
                "invalid time: timestamp is out of range for UTC nanoseconds: "
                f"{exc}"
            ) from exc
    if converted.isna().any():
        raise DataValidationError("invalid time: timestamp cannot be converted to UTC")
    return converted.astype("datetime64[ns, UTC]")


def _contains_complex(values: pd.Series) -> bool:
    array = values.to_numpy(copy=False)
    if np.iscomplexobj(array):
        return True
    return any(
        isinstance(value, (complex, np.complexfloating)) for value in array
    )


def _integral_source_value(value: object) -> int | None:
    """Return an exact source integer without relying on its concrete type."""
    try:
        integer = int(value)  # type: ignore[arg-type]
        return integer if bool(value == integer) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _coerce_value_column(values: pd.Series, *, field: str) -> pd.Series:
    if _contains_bool(values):
        raise DataValidationError(f"boolean OHLCV value: field={field}")
    source_values = values.to_numpy(dtype=object, copy=False)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            numeric = pd.to_numeric(values, errors="raise")
            converted = numeric.astype("float64")
    except Exception as exc:
        raise DataValidationError(
            f"invalid numeric OHLCV value: field={field}"
        ) from exc
    array = converted.to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(array).all():
        raise DataValidationError(f"non-finite OHLCV value: field={field}")
    for source, float_value in zip(source_values, array, strict=True):
        source_integer = _integral_source_value(source)
        if source_integer is not None and int(float_value) != source_integer:
            raise DataValidationError(
                "integer OHLCV value must round-trip exactly through float64: "
                f"field={field}"
            )
    return converted


def float32_ohlcv_arrays(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """Return finite canonical float32 values without requiring exact reversal."""
    converted: dict[str, np.ndarray] = {}
    for field in _VALUE_COLUMNS:
        source = frame[field].to_numpy(dtype=np.float64, copy=False)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            values = source.astype(np.float32, copy=True)
        if not np.isfinite(values).all():
            raise DataValidationError(
                "OHLCV values must remain finite after float32 conversion: "
                f"field={field}"
            )
        converted[field] = values.astype("<f4", copy=False)
    return converted


def _fingerprints(
    *,
    frame: pd.DataFrame,
    symbol: str,
    timeframe: str,
    gap_policy: str,
) -> tuple[str, str, np.ndarray]:
    time_ns = frame["time"].astype("int64").to_numpy(dtype="<i8", copy=True)
    value_bytes = frame.loc[:, _VALUE_COLUMNS].to_numpy(dtype="<f4", copy=True)
    metadata = {
        "bars": len(frame),
        "end_time_ns": int(time_ns[-1]),
        "schema_version": DATA_SCHEMA_VERSION,
        "canonicalization_version": DATA_CANONICALIZATION_VERSION,
        "time_unit": "ns",
        "gap_policy": gap_policy,
        "start_time_ns": int(time_ns[0]),
        "symbol": symbol,
        "timeframe": timeframe,
    }
    metadata_bytes = json.dumps(
        metadata,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    time_bytes = time_ns.tobytes(order="C")

    data_hash = hashlib.sha256()
    data_hash.update(metadata_bytes)
    data_hash.update(time_bytes)
    data_hash.update(value_bytes.tobytes(order="C"))
    return data_hash.hexdigest(), hashlib.sha256(time_bytes).hexdigest(), time_ns


def canonicalize_ohlcv(
    frame: pd.DataFrame,
    *,
    symbol: str,
    timeframe: str | int,
    numeric_time_unit: str | None = None,
    gap_policy: str = "segment",
) -> CanonicalDataset:
    """Validate and canonicalize an OHLCV frame without filling missing bars."""
    if not isinstance(frame, pd.DataFrame):
        raise DataValidationError("OHLCV input must be a pandas DataFrame")

    canonical_timeframe = normalize_timeframe_name(timeframe)
    canonical_gap_policy = _normalize_gap_policy(gap_policy)
    canonical_numeric_time_unit = (
        _normalize_numeric_time_unit(numeric_time_unit)
        if numeric_time_unit is not None
        else None
    )
    result = frame.copy()
    result.columns = [str(column).strip().lower() for column in result.columns]
    if result.columns.duplicated().any():
        raise DataValidationError("duplicate columns after lower-case normalization")
    if "volume" not in result.columns and "tick_volume" in result.columns:
        result = result.rename(columns={"tick_volume": "volume"})

    missing = [column for column in _CANONICAL_COLUMNS if column not in result.columns]
    if missing:
        raise DataValidationError(f"missing columns: {missing}")

    result = result.loc[:, _CANONICAL_COLUMNS].copy()
    for column in _VALUE_COLUMNS:
        if _contains_complex(result[column]):
            raise DataValidationError(f"complex OHLCV value: field={column}")

    converted_time = _to_utc_time(
        result["time"],
        numeric_time_unit=canonical_numeric_time_unit,
    )
    result["time"] = converted_time.array
    if result["time"].duplicated().any():
        raise DataValidationError("duplicate timestamp")

    if result.empty:
        raise DataValidationError("OHLCV data has no bars")

    original_time_ns = result["time"].astype("int64").to_numpy(
        dtype=np.int64,
        copy=False,
    )
    if any(
        int(current) <= int(previous)
        for previous, current in zip(original_time_ns, original_time_ns[1:])
    ):
        raise DataValidationError("time must be strictly increasing")
    result = result.reset_index(drop=True)

    for column in _VALUE_COLUMNS:
        result[column] = _coerce_value_column(result[column], field=column)
    converted_values = float32_ohlcv_arrays(result)
    for column in _VALUE_COLUMNS:
        result[column] = converted_values[column]

    if (result.loc[:, ("open", "high", "low", "close")] <= 0.0).any().any():
        raise DataValidationError("OHLC prices must be positive")
    if (result["volume"] < 0.0).any():
        raise DataValidationError("volume must be non-negative")

    highest_component = result.loc[:, ("open", "close", "low")].max(axis=1)
    lowest_component = result.loc[:, ("open", "close", "high")].min(axis=1)
    if (result["high"] < highest_component).any() or (
        result["low"] > lowest_component
    ).any():
        raise DataValidationError("invalid OHLC containment")

    time_ns = result["time"].astype("int64").to_numpy(dtype=np.int64, copy=False)
    deltas = [
        int(current) - int(previous)
        for previous, current in zip(time_ns, time_ns[1:])
    ]
    if any(delta <= 0 for delta in deltas):
        raise DataValidationError("time must be strictly increasing")

    nominal_ns = _TIMEFRAME_NS.get(canonical_timeframe)
    segment_ids = np.zeros(len(result), dtype=np.int64)
    if nominal_ns is None:
        gap_count = 0
    else:
        if any(delta < nominal_ns for delta in deltas):
            raise DataValidationError(
                f"bar spacing is shorter than timeframe {canonical_timeframe}"
            )
        gap_indices = [
            index
            for index, delta in enumerate(deltas, start=1)
            if delta > nominal_ns
        ]
        gap_count = len(gap_indices)
        if gap_count and canonical_gap_policy == "reject":
            raise DataValidationError(
                f"gap policy reject forbids {gap_count} large time gap(s)"
            )
        if canonical_gap_policy == "segment":
            for index in gap_indices:
                segment_ids[index:] += 1

    data_fingerprint, time_fingerprint, time_ns = _fingerprints(
        frame=result,
        symbol=str(symbol),
        timeframe=canonical_timeframe,
        gap_policy=canonical_gap_policy,
    )
    identity = DatasetIdentity(
        schema_version=DATA_SCHEMA_VERSION,
        canonicalization_version=DATA_CANONICALIZATION_VERSION,
        time_unit="ns",
        gap_policy=canonical_gap_policy,
        symbol=str(symbol),
        timeframe=canonical_timeframe,
        start_time_ns=int(time_ns[0]),
        end_time_ns=int(time_ns[-1]),
        bars=len(result),
        data_fingerprint=data_fingerprint,
        time_fingerprint=time_fingerprint,
    )
    return CanonicalDataset(
        frame=result,
        identity=identity,
        gap_count=gap_count,
        segment_ids=segment_ids,
    )
