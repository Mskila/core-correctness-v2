"""Canonical OHLCV validation and stable dataset identity."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Mapping
import warnings

import numpy as np
import pandas as pd

from model_core.semantics import DATA_SCHEMA_VERSION, DataValidationError


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


@dataclass(frozen=True)
class DatasetIdentity:
    schema_version: str
    symbol: str
    timeframe: str
    start_time_ns: int
    end_time_ns: int
    bars: int
    data_fingerprint: str
    time_fingerprint: str

    def to_dict(self) -> dict[str, str | int]:
        return {
            "schema_version": self.schema_version,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "start_time_ns": self.start_time_ns,
            "end_time_ns": self.end_time_ns,
            "bars": self.bars,
            "data_fingerprint": self.data_fingerprint,
            "time_fingerprint": self.time_fingerprint,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "DatasetIdentity":
        try:
            return cls(
                schema_version=str(value["schema_version"]),
                symbol=str(value["symbol"]),
                timeframe=normalize_timeframe_name(value["timeframe"]),  # type: ignore[arg-type]
                start_time_ns=int(value["start_time_ns"]),
                end_time_ns=int(value["end_time_ns"]),
                bars=int(value["bars"]),
                data_fingerprint=str(value["data_fingerprint"]),
                time_fingerprint=str(value["time_fingerprint"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise DataValidationError(f"invalid dataset identity: {exc}") from exc


@dataclass(frozen=True, init=False)
class CanonicalDataset:
    _frame: pd.DataFrame = field(repr=False, compare=False)
    identity: DatasetIdentity
    gap_count: int

    def __init__(
        self,
        frame: pd.DataFrame,
        identity: DatasetIdentity,
        gap_count: int,
    ) -> None:
        object.__setattr__(self, "_frame", frame.copy(deep=True))
        object.__setattr__(self, "identity", identity)
        object.__setattr__(self, "gap_count", gap_count)

    @property
    def frame(self) -> pd.DataFrame:
        """Return a defensive copy so content cannot diverge from identity."""
        return self._frame.copy(deep=True)


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
    """Convert canonical OHLCV values to float32 without silent value loss."""
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
        if np.any((source != 0.0) & (values == 0.0)):
            raise DataValidationError(
                "OHLCV values must remain non-zero after float32 conversion: "
                f"field={field}"
            )
        if not np.equal(values.astype(np.float64), source).all():
            raise DataValidationError(
                "OHLCV values must round-trip exactly through float32 conversion: "
                f"field={field}"
            )
        converted[field] = values
    return converted


def _fingerprints(
    *,
    frame: pd.DataFrame,
    symbol: str,
    timeframe: str,
) -> tuple[str, str, np.ndarray]:
    time_ns = frame["time"].astype("int64").to_numpy(dtype="<i8", copy=True)
    value_bytes = frame.loc[:, _VALUE_COLUMNS].to_numpy(dtype="<f8", copy=True)
    metadata = {
        "bars": len(frame),
        "end_time_ns": int(time_ns[-1]),
        "schema_version": DATA_SCHEMA_VERSION,
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
) -> CanonicalDataset:
    """Validate and canonicalize an OHLCV frame without filling missing bars."""
    if not isinstance(frame, pd.DataFrame):
        raise DataValidationError("OHLCV input must be a pandas DataFrame")

    canonical_timeframe = normalize_timeframe_name(timeframe)
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
    result = result.sort_values("time", kind="mergesort").reset_index(drop=True)

    if result.empty:
        raise DataValidationError("OHLCV data has no bars")

    for column in _VALUE_COLUMNS:
        result[column] = _coerce_value_column(result[column], field=column)
    values = result.loc[:, _VALUE_COLUMNS].to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(values).all():
        raise DataValidationError("non-finite OHLCV value")

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
    if nominal_ns is None:
        gap_count = 0
    else:
        if any(delta < nominal_ns for delta in deltas):
            raise DataValidationError(
                f"bar spacing is shorter than timeframe {canonical_timeframe}"
            )
        gap_count = sum(delta > nominal_ns for delta in deltas)

    data_fingerprint, time_fingerprint, time_ns = _fingerprints(
        frame=result,
        symbol=str(symbol),
        timeframe=canonical_timeframe,
    )
    identity = DatasetIdentity(
        schema_version=DATA_SCHEMA_VERSION,
        symbol=str(symbol),
        timeframe=canonical_timeframe,
        start_time_ns=int(time_ns[0]),
        end_time_ns=int(time_ns[-1]),
        bars=len(result),
        data_fingerprint=data_fingerprint,
        time_fingerprint=time_fingerprint,
    )
    return CanonicalDataset(frame=result, identity=identity, gap_count=gap_count)
