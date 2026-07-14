"""Canonical OHLCV validation and stable dataset identity."""

from __future__ import annotations

from collections import deque
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
    if np.issubdtype(array.dtype, np.floating):
        if not np.isfinite(array).all():
            raise DataValidationError("invalid time: non-finite timestamp")
        if not np.equal(array, np.trunc(array)).all():
            raise DataValidationError("numeric timestamps must be integer values")
        if (np.abs(array) > 2**53).any():
            raise DataValidationError(
                "floating numeric timestamps must be lossless integers"
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


def _complete_row_quotient_matching(
    candidates_by_row: dict[int, set[int]],
) -> dict[int, int] | None:
    """Match every row to a distinct cadence quotient, if possible."""
    matched_row: dict[int, int] = {}
    matched_quotient: dict[int, int] = {}
    for start_row in sorted(
        candidates_by_row,
        key=lambda row: len(candidates_by_row[row]),
    ):
        pending = deque([start_row])
        seen_rows = {start_row}
        seen_quotients: set[int] = set()
        previous_row_by_quotient: dict[int, int] = {}
        free_quotient: int | None = None
        while pending and free_quotient is None:
            row = pending.popleft()
            for quotient in sorted(candidates_by_row[row]):
                if quotient in seen_quotients:
                    continue
                seen_quotients.add(quotient)
                previous_row_by_quotient[quotient] = row
                owner = matched_quotient.get(quotient)
                if owner is None:
                    free_quotient = quotient
                    break
                if owner not in seen_rows:
                    seen_rows.add(owner)
                    pending.append(owner)
        if free_quotient is None:
            return None

        quotient = free_quotient
        while True:
            row = previous_row_by_quotient[quotient]
            previous_quotient = matched_row.get(row)
            matched_row[row] = quotient
            matched_quotient[quotient] = row
            if previous_quotient is None:
                break
            quotient = previous_quotient
    return matched_row


def _cadence_quotient_score(quotients: list[int]) -> tuple[int, int]:
    ordered = sorted(quotients)
    if len(set(ordered)) != len(ordered):
        return (-1, 0)
    exact_count = sum(
        current - previous == 1
        for previous, current in zip(ordered, ordered[1:])
    )
    return exact_count, -(ordered[-1] - ordered[0])


def _has_single_unit_regular_run(
    candidates_by_row: dict[int, list[tuple[int, int]]],
) -> bool:
    candidates_by_unit: dict[int, list[tuple[int, int]]] = {}
    for row, options in candidates_by_row.items():
        for quotient, unit_bit in options:
            if unit_bit:
                candidates_by_unit.setdefault(unit_bit, []).append((quotient, row))
    for candidates in candidates_by_unit.values():
        ordered = sorted(candidates)
        for index in range(1, len(ordered) - 1):
            previous_quotient, previous_row = ordered[index - 1]
            current_quotient, current_row = ordered[index]
            following_quotient, following_row = ordered[index + 1]
            cadence_multiple = current_quotient - previous_quotient
            if (
                cadence_multiple > 0
                and following_quotient - current_quotient == cadence_multiple
                and len({previous_row, current_row, following_row}) == 3
            ):
                return True
    return False


def _has_better_single_row_mixed_lattice(
    candidates_by_row: dict[int, list[tuple[int, int]]],
    *,
    common_units: int,
) -> bool:
    nonzero_rows = sum(
        any(unit_bit for _quotient, unit_bit in options)
        for options in candidates_by_row.values()
    )
    if nonzero_rows < 2:
        return False
    unit_bit = 1
    while unit_bit <= common_units:
        if not common_units & unit_bit:
            unit_bit <<= 1
            continue
        baseline_by_row: dict[int, int] = {}
        for row, options in candidates_by_row.items():
            matching = {
                quotient
                for quotient, option_unit in options
                if option_unit in (0, unit_bit)
            }
            if len(matching) != 1:
                break
            baseline_by_row[row] = matching.pop()
        else:
            baseline_score = _cadence_quotient_score(list(baseline_by_row.values()))
            for row, options in candidates_by_row.items():
                for quotient, option_unit in options:
                    if option_unit in (0, unit_bit):
                        continue
                    alternative = [
                        quotient if candidate_row == row else baseline
                        for candidate_row, baseline in baseline_by_row.items()
                    ]
                    if _cadence_quotient_score(alternative) > baseline_score:
                        return True
        unit_bit <<= 1
    return False


def _has_mixed_unit_lattice_evidence(
    values: list[int],
    *,
    nominal_ns: int,
    selected_unit: str | None = None,
) -> bool:
    """Detect mixed-unit evidence across each complete cadence lattice."""
    units = tuple(_TIME_UNIT_NS)
    lattice: dict[int, dict[int, list[tuple[int, int]]]] = {}
    for row_index, value in enumerate(values):
        for unit_index, unit in enumerate(units):
            timestamp_ns = _scaled_time_ns(value, unit)
            if timestamp_ns is None:
                continue
            remainder = timestamp_ns % nominal_ns
            quotient = (timestamp_ns - remainder) // nominal_ns
            unit_bit = 0 if value == 0 else 1 << unit_index
            lattice.setdefault(remainder, {}).setdefault(row_index, []).append(
                (quotient, unit_bit)
            )

    all_units_mask = (1 << len(units)) - 1
    for candidates_by_row in lattice.values():
        if len(candidates_by_row) != len(values):
            continue
        common_units = all_units_mask
        has_nonzero_row = False
        for options in candidates_by_row.values():
            row_units = 0
            for _quotient, unit_bit in options:
                row_units |= unit_bit
            if row_units:
                common_units &= row_units
                has_nonzero_row = True
        if not has_nonzero_row:
            continue
        if common_units:
            if selected_unit is None:
                continue
            selected_bit = 1 << units.index(selected_unit)
            if common_units & selected_bit and _has_better_single_row_mixed_lattice(
                candidates_by_row,
                common_units=selected_bit,
            ):
                return True
            continue
        quotient_candidates = {
            row: {quotient for quotient, _unit_bit in options}
            for row, options in candidates_by_row.items()
        }
        matching = _complete_row_quotient_matching(quotient_candidates)
        if matching is not None and _has_single_unit_regular_run(candidates_by_row):
            return True
    return False


def _best_mixed_unit_cadence(
    sorted_values: list[int],
    *,
    nominal_ns: int,
) -> int:
    units = tuple(_TIME_UNIT_NS)
    scaled = [
        [_scaled_time_ns(value, unit) for unit in units]
        for value in sorted_values
    ]
    states: dict[tuple[int, int], int] = {}
    for unit_index, timestamp_ns in enumerate(scaled[0]):
        if timestamp_ns is None:
            continue
        mask = 0 if sorted_values[0] == 0 else 1 << unit_index
        states[(unit_index, mask)] = 0

    for row_index in range(1, len(sorted_values)):
        next_states: dict[tuple[int, int], int] = {}
        for unit_index, timestamp_ns in enumerate(scaled[row_index]):
            if timestamp_ns is None:
                continue
            unit_bit = 0 if sorted_values[row_index] == 0 else 1 << unit_index
            for (previous_unit, mask), exact_count in states.items():
                previous_ns = scaled[row_index - 1][previous_unit]
                if previous_ns is None:
                    continue
                delta = timestamp_ns - previous_ns
                if delta < nominal_ns:
                    continue
                key = (unit_index, mask | unit_bit)
                score = exact_count + int(delta == nominal_ns)
                next_states[key] = max(next_states.get(key, -1), score)
        states = next_states
        if not states:
            return -1

    return max(
        (
            score
            for (_unit, mask), score in states.items()
            if mask.bit_count() > 1
        ),
        default=-1,
    )


def _has_mixed_cadence_multiple_run(
    values: list[int],
    *,
    nominal_ns: int,
) -> bool:
    """Detect a regular three-bar cadence that requires mixed epoch units."""
    units = tuple(_TIME_UNIT_NS)
    lattice: dict[int, dict[int, list[tuple[int, int]]]] = {}
    for row_index, value in enumerate(values):
        for unit_index, unit in enumerate(units):
            timestamp_ns = _scaled_time_ns(value, unit)
            if timestamp_ns is None:
                continue
            remainder = timestamp_ns % nominal_ns
            quotient = (timestamp_ns - remainder) // nominal_ns
            unit_bit = 0 if value == 0 else 1 << unit_index
            lattice.setdefault(remainder, {}).setdefault(quotient, []).append(
                (row_index, unit_bit)
            )

    for candidates_by_quotient in lattice.values():
        quotients = sorted(candidates_by_quotient)
        for index in range(1, len(quotients) - 1):
            previous = quotients[index - 1]
            current = quotients[index]
            following = quotients[index + 1]
            cadence_multiple = current - previous
            if cadence_multiple <= 0 or following - current != cadence_multiple:
                continue
            for left_row, left_unit in candidates_by_quotient[previous]:
                for middle_row, middle_unit in candidates_by_quotient[current]:
                    if middle_row == left_row:
                        continue
                    for right_row, right_unit in candidates_by_quotient[following]:
                        if right_row in (left_row, middle_row):
                            continue
                        unit_mask = left_unit | middle_unit | right_unit
                        if unit_mask.bit_count() > 1:
                            return True
    return False


def _select_pure_cadence_multiple_candidate(
    candidates: list[tuple[str, list[int], int, bool, bool]],
    *,
    nominal_ns: int,
) -> tuple[str, list[int], int, bool, bool] | None:
    """Select the finest internally aligned pure-unit cadence candidate."""
    aligned: list[tuple[int, tuple[str, list[int], int, bool, bool]]] = []
    plausible_spans: list[int] = []
    for candidate in candidates:
        ordered_ns = sorted(candidate[1])
        if len(ordered_ns) < 2:
            continue
        deltas = [
            current - previous
            for previous, current in zip(ordered_ns, ordered_ns[1:])
        ]
        span = ordered_ns[-1] - ordered_ns[0]
        if candidate[3] or candidate[4]:
            plausible_spans.append(span)
        if all(
            delta >= nominal_ns and delta % nominal_ns == 0
            for delta in deltas
        ):
            aligned.append((span, candidate))

    if not aligned:
        return None
    aligned.sort(key=lambda item: item[0])
    selected_span, selected = aligned[0]
    if any(span < selected_span for span in plausible_spans):
        return None
    return selected


def _numeric_time_to_utc(
    values: pd.Series,
    *,
    timeframe: str,
) -> pd.Series:
    integers = _integer_numeric_timestamps(values)
    if not integers:
        return pd.Series(pd.to_datetime([], utc=True)).astype(
            "datetime64[ns, UTC]"
        )
    if len(set(integers)) != len(integers):
        raise DataValidationError("duplicate timestamp")

    sorted_values = sorted(integers)
    nominal_ns = _TIMEFRAME_NS.get(timeframe)
    candidates: list[tuple[str, list[int], int, bool, bool]] = []
    for unit in _TIME_UNIT_NS:
        scaled = [_scaled_time_ns(value, unit) for value in integers]
        if any(value is None for value in scaled):
            continue
        scaled_ns = [int(value) for value in scaled if value is not None]
        ordered_ns = sorted(scaled_ns)
        deltas = [
            current - previous
            for previous, current in zip(ordered_ns, ordered_ns[1:])
        ]
        exact_count = (
            sum(delta == nominal_ns for delta in deltas)
            if nominal_ns is not None
            else 0
        )
        spacing_valid = nominal_ns is not None and bool(
            all(delta >= nominal_ns for delta in deltas)
        )
        span_aligned = (
            nominal_ns is not None
            and len(ordered_ns) > 1
            and ordered_ns[-1] - ordered_ns[0] >= nominal_ns
            and (ordered_ns[-1] - ordered_ns[0]) % nominal_ns == 0
        )
        candidates.append(
            (unit, scaled_ns, exact_count, spacing_valid, span_aligned)
        )

    if not candidates:
        raise DataValidationError("invalid time: timestamp cannot be converted to UTC")
    best_exact = max(candidate[2] for candidate in candidates)
    if best_exact > 0:
        plausible = [candidate for candidate in candidates if candidate[2] == best_exact]
    else:
        plausible = [
            candidate for candidate in candidates if candidate[3] or candidate[4]
        ]
    if len(plausible) != 1:
        if nominal_ns is not None:
            mixed_lattice_evidence = _has_mixed_unit_lattice_evidence(
                integers,
                nominal_ns=nominal_ns,
            )
            mixed_multiple_run = _has_mixed_cadence_multiple_run(
                integers,
                nominal_ns=nominal_ns,
            )
            if mixed_lattice_evidence or mixed_multiple_run:
                raise DataValidationError(
                    "mixed numeric timestamp units: values require multiple "
                    "epoch units on one cadence lattice"
                )
            if best_exact == 0:
                pure_candidate = _select_pure_cadence_multiple_candidate(
                    candidates,
                    nominal_ns=nominal_ns,
                )
                if pure_candidate is not None:
                    plausible = [pure_candidate]
        if len(plausible) == 1:
            selected = plausible[0]
        else:
            raise DataValidationError(
                "ambiguous numeric timestamp unit: cadence does not uniquely "
                f"identify one epoch unit for timeframe {timeframe}"
            )
    else:
        selected = plausible[0]
    if nominal_ns is not None and len(sorted_values) > 1:
        mixed_exact = _best_mixed_unit_cadence(
            sorted_values,
            nominal_ns=nominal_ns,
        )
        mixed_multiple_run = _has_mixed_cadence_multiple_run(
            integers,
            nominal_ns=nominal_ns,
        )
        mixed_lattice_evidence = _has_mixed_unit_lattice_evidence(
            integers,
            nominal_ns=nominal_ns,
            selected_unit=selected[0],
        )
        if (
            mixed_exact > selected[2]
            or mixed_multiple_run
            or mixed_lattice_evidence
        ):
            raise DataValidationError(
                "mixed numeric timestamp units: values align better under "
                "multiple epoch units"
            )

    return pd.Series(pd.to_datetime(selected[1], unit="ns", utc=True)).astype(
        "datetime64[ns, UTC]"
    )


def _to_utc_time(values: pd.Series, *, timeframe: str) -> pd.Series:
    if pd.api.types.is_numeric_dtype(values.dtype):
        converted = _numeric_time_to_utc(values, timeframe=timeframe)
    else:
        try:
            converted = pd.Series(
                pd.to_datetime(values, utc=True, errors="coerce")
            )
        except Exception as exc:
            raise DataValidationError(f"invalid time: {exc}") from exc
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


def _coerce_value_column(values: pd.Series, *, field: str) -> pd.Series:
    if _contains_bool(values):
        raise DataValidationError(f"boolean OHLCV value: field={field}")
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

    converted_time = _to_utc_time(
        result["time"],
        timeframe=canonical_timeframe,
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
