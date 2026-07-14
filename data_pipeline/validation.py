"""Canonical OHLCV validation and stable dataset identity."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Mapping

import numpy as np
import pandas as pd

from model_core.semantics import DATA_SCHEMA_VERSION, DataValidationError


_CANONICAL_COLUMNS = ("time", "open", "high", "low", "close", "volume")
_VALUE_COLUMNS = ("open", "high", "low", "close", "volume")
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


@dataclass(frozen=True)
class CanonicalDataset:
    frame: pd.DataFrame
    identity: DatasetIdentity
    gap_count: int


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


def _convert_numeric_time(
    numeric: pd.Series,
    *,
    unit: str,
) -> pd.Series | None:
    try:
        converted = pd.Series(
            pd.to_datetime(numeric, unit=unit, utc=True, errors="coerce")
        )
        if converted.isna().any():
            return None
        return converted.astype("datetime64[ns, UTC]")
    except (OverflowError, ValueError, pd.errors.OutOfBoundsDatetime):
        return None


def _numeric_time_to_utc(
    values: pd.Series,
    *,
    timeframe: str,
) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    numeric_array = numeric.to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(numeric_array).all():
        raise DataValidationError("invalid time: non-finite timestamp")

    absolute = np.abs(numeric_array)
    max_abs = float(np.max(absolute)) if len(numeric_array) else 0.0
    inferred = np.select(
        [
            absolute < 100_000_000_000,
            absolute < 100_000_000_000_000,
            absolute < 100_000_000_000_000_000,
        ],
        ["s", "ms", "us"],
        default="ns",
    )
    nonzero = absolute > 0
    inferred_units = set(inferred[nonzero].tolist())
    positive = absolute[nonzero]
    crosses_near_boundary = (
        len(positive) == 0
        or float(np.max(positive) / np.min(positive)) <= 10.0
    )
    if len(inferred_units) > 1 and not crosses_near_boundary:
        counts = {
            unit: int(np.count_nonzero(inferred[nonzero] == unit))
            for unit in sorted(inferred_units)
        }
        raise DataValidationError(
            "mixed numeric timestamp units: "
            f"inferred {counts}; values must use one epoch unit"
        )

    nominal_ns = _TIMEFRAME_NS.get(timeframe)
    if nominal_ns is not None and len(numeric_array) > 1:
        candidates: list[tuple[str, pd.Series, int]] = []
        for unit in ("s", "ms", "us", "ns"):
            converted = _convert_numeric_time(numeric, unit=unit)
            if converted is None:
                continue
            converted_ns = np.sort(
                converted.astype("int64").to_numpy(dtype=np.int64, copy=False)
            )
            deltas = np.diff(converted_ns)
            if (deltas <= 0).any() or (deltas < nominal_ns).any():
                continue
            exact_cadence = int(np.count_nonzero(deltas == nominal_ns))
            candidates.append((unit, converted, exact_cadence))

        if candidates:
            best_score = max(candidate[2] for candidate in candidates)
            best = [candidate for candidate in candidates if candidate[2] == best_score]
            if best_score > 0 and len(best) == 1:
                return best[0][1]
            if len(candidates) == 1:
                return candidates[0][1]
            raise DataValidationError(
                "ambiguous numeric timestamp unit: cadence does not uniquely "
                f"identify one epoch unit for timeframe {timeframe}"
            )

    if max_abs < 100_000_000_000:
        unit = "s"
    elif max_abs < 100_000_000_000_000:
        unit = "ms"
    elif max_abs < 100_000_000_000_000_000:
        unit = "us"
    else:
        unit = "ns"
    converted = _convert_numeric_time(numeric, unit=unit)
    if converted is None:
        raise DataValidationError("invalid time: timestamp cannot be converted to UTC")
    return converted


def _to_utc_time(values: pd.Series, *, timeframe: str) -> pd.Series:
    if pd.api.types.is_numeric_dtype(values.dtype):
        converted = _numeric_time_to_utc(values, timeframe=timeframe)
    else:
        converted = pd.Series(pd.to_datetime(values, utc=True, errors="coerce"))
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
        if field != "volume" and np.any((source != 0.0) & (values == 0.0)):
            raise DataValidationError(
                "OHLCV values must remain non-zero after float32 conversion: "
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
) -> CanonicalDataset:
    """Validate and canonicalize an OHLCV frame without filling missing bars."""
    if not isinstance(frame, pd.DataFrame):
        raise DataValidationError("OHLCV input must be a pandas DataFrame")

    canonical_timeframe = normalize_timeframe_name(timeframe)
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
    for column in _CANONICAL_COLUMNS:
        if _contains_complex(result[column]):
            raise DataValidationError(f"complex OHLCV value: field={column}")

    result["time"] = _to_utc_time(
        result["time"],
        timeframe=canonical_timeframe,
    )
    if result["time"].duplicated().any():
        raise DataValidationError("duplicate timestamp")
    result = result.sort_values("time", kind="mergesort").reset_index(drop=True)

    if result.empty:
        raise DataValidationError("OHLCV data has no bars")

    for column in _VALUE_COLUMNS:
        result[column] = pd.to_numeric(result[column], errors="coerce").astype(
            "float64"
        )
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
    deltas = np.diff(time_ns)
    if (deltas <= 0).any():
        raise DataValidationError("time must be strictly increasing")

    nominal_ns = _TIMEFRAME_NS.get(canonical_timeframe)
    if nominal_ns is None:
        gap_count = 0
    else:
        if (deltas < nominal_ns).any():
            raise DataValidationError(
                f"bar spacing is shorter than timeframe {canonical_timeframe}"
            )
        gap_count = int(np.count_nonzero(deltas > nominal_ns))

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
