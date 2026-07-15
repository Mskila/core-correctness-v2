from __future__ import annotations

from dataclasses import FrozenInstanceError
from decimal import Decimal
from fractions import Fraction
import warnings

import numpy as np
import pandas as pd
import pytest

from data_pipeline.validation import (
    DatasetIdentity,
    assert_minimum_bars,
    canonicalize_ohlcv,
    float32_ohlcv_arrays,
    normalize_timeframe_name,
)
from model_core.semantics import DATA_SCHEMA_VERSION, DataValidationError


def valid_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": pd.date_range("2026-01-01", periods=6, freq="1h", tz="UTC"),
            "open": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
            "high": [10.5, 11.5, 12.5, 13.5, 14.5, 15.5],
            "low": [9.5, 10.5, 11.5, 12.5, 13.5, 14.5],
            "close": [10.2, 11.2, 12.2, 13.2, 14.2, 15.2],
            "volume": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
        }
    )


_TEST_TIME_UNIT_NS = {
    "s": 1_000_000_000,
    "ms": 1_000_000,
    "us": 1_000,
    "ns": 1,
}
_TEST_H1_NS = 3_600_000_000_000
_UNIQUE_PURE_NUMERIC_BASE = {
    "s": -999_963,
    "ms": 1_700_000_000_018,
    "us": 1_700_000_000_000_035,
    "ns": 1_700_000_000_000_000_001,
}
_UNIQUE_PURE_NUMERIC_TIMES = {
    "s": [-999_963, 2_603_637, 2_607_237, 2_632_437, 2_679_237, 2_783_637],
    "ms": [
        1_700_000_000_018,
        1_703_603_600_018,
        1_703_607_200_018,
        1_703_632_400_018,
        1_703_679_200_018,
        1_703_783_600_018,
    ],
    "us": [
        1_700_000_000_000_035,
        1_703_603_600_000_035,
        1_703_607_200_000_035,
        1_703_632_400_000_035,
        1_703_679_200_000_035,
        1_703_783_600_000_035,
    ],
    "ns": [
        1_700_000_000_000_000_001,
        1_703_600_000_000_000_001,
        1_703_603_600_000_000_001,
        1_703_628_800_000_000_001,
        1_703_675_600_000_000_001,
        1_703_780_000_000_000_001,
    ],
}


def _frame_with_numeric_times(timestamps: list[int]) -> pd.DataFrame:
    opens = 10.0 + np.arange(len(timestamps), dtype=np.float64)
    return pd.DataFrame(
        {
            "time": timestamps,
            "open": opens,
            "high": opens + 1.0,
            "low": opens - 1.0,
            "close": opens + 0.25,
            "volume": 100.0 + np.arange(len(timestamps), dtype=np.float64),
        }
    )


def _encode_semantic_times(offsets: list[int], units: list[str]) -> list[int]:
    base_ns = 1_700_000_000_000_000_000
    return [
        (base_ns + offset * _TEST_H1_NS) // _TEST_TIME_UNIT_NS[unit]
        for offset, unit in zip(offsets, units)
    ]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda df: df.drop(columns=["high"]), "missing columns"),
        (
            lambda df: df.assign(open=[float("nan"), 11, 12, 13, 14, 15]),
            "non-finite",
        ),
        (
            lambda df: df.assign(volume=[100, 101, 102, float("inf"), 104, 105]),
            "non-finite",
        ),
        (
            lambda df: df.assign(low=[11, 10.5, 11.5, 12.5, 13.5, 14.5]),
            "invalid OHLC",
        ),
        (
            lambda df: pd.concat([df, df.iloc[[0]]], ignore_index=True),
            "duplicate timestamp",
        ),
    ],
)
def test_invalid_ohlcv_is_rejected(mutation, message: str) -> None:
    with pytest.raises(DataValidationError, match=message):
        canonicalize_ohlcv(mutation(valid_frame()), symbol="EURUSD", timeframe="H1")


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("open", 0.0, "positive"),
        ("close", -1.0, "positive"),
        ("volume", -1.0, "non-negative"),
    ],
)
def test_price_and_volume_domains_are_enforced(
    column: str, value: float, message: str
) -> None:
    frame = valid_frame()
    frame.loc[2, column] = value
    with pytest.raises(DataValidationError, match=message):
        canonicalize_ohlcv(frame, symbol="EURUSD", timeframe="H1")


def test_canonical_output_maps_tick_volume_and_has_fixed_schema() -> None:
    frame = valid_frame().rename(columns={"volume": "TICK_VOLUME"})
    frame.columns = [column.upper() for column in frame.columns]

    result = canonicalize_ohlcv(frame, symbol="EURUSD", timeframe="h1")

    assert list(result.frame.columns) == [
        "time",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]
    assert str(result.frame["time"].dtype) == "datetime64[ns, UTC]"
    assert all(result.frame[column].dtype == np.dtype("float64") for column in result.frame.columns[1:])
    assert result.frame["volume"].tolist() == valid_frame()["volume"].tolist()
    assert result.identity.timeframe == "H1"


def test_existing_volume_takes_precedence_over_tick_volume() -> None:
    frame = valid_frame().assign(tick_volume=[900, 901, 902, 903, 904, 905])

    result = canonicalize_ohlcv(frame, symbol="EURUSD", timeframe="H1")

    assert result.frame["volume"].tolist() == valid_frame()["volume"].tolist()


def test_time_is_stably_sorted_without_changing_bar_contents() -> None:
    frame = valid_frame().iloc[[2, 0, 1, 5, 3, 4]].copy()
    expected = valid_frame().reset_index(drop=True)
    expected["time"] = expected["time"].astype("datetime64[ns, UTC]")

    result = canonicalize_ohlcv(frame, symbol="EURUSD", timeframe="H1")

    pd.testing.assert_frame_equal(result.frame, expected)


def test_numeric_time_preserves_rows_with_nonmonotonic_input_index() -> None:
    frame = pd.DataFrame(
        {
            "time": _UNIQUE_PURE_NUMERIC_TIMES["s"][:3],
            "open": [10.0, 20.0, 30.0],
            "high": [11.0, 21.0, 31.0],
            "low": [9.0, 19.0, 29.0],
            "close": [10.25, 20.25, 30.25],
            "volume": [100.0, 200.0, 300.0],
        }
    ).iloc[[2, 0, 1]]
    reset = frame.reset_index(drop=True)

    retained_index = canonicalize_ohlcv(
        frame,
        symbol="EURUSD",
        timeframe="H1",
        numeric_time_unit="s",
    )
    reset_index = canonicalize_ohlcv(
        reset,
        symbol="EURUSD",
        timeframe="H1",
        numeric_time_unit="s",
    )

    pd.testing.assert_frame_equal(retained_index.frame, reset_index.frame)
    assert retained_index.identity == reset_index.identity


def _time_values_for_index_invariance(kind: str) -> list[object]:
    if kind in _UNIQUE_PURE_NUMERIC_TIMES:
        return _UNIQUE_PURE_NUMERIC_TIMES[kind]
    seconds = [1_700_000_000 + index * 3_600 for index in range(6)]
    timestamps = pd.to_datetime(seconds, unit="s", utc=True)
    if kind == "datetime":
        return timestamps.tolist()
    if kind == "string":
        return [value.isoformat() for value in timestamps]
    raise AssertionError(f"unknown test time kind: {kind}")


@pytest.mark.parametrize(
    ("unit", "legacy"),
    [
        ("s", [1_700_000_000 + index * 3_600 for index in range(6)]),
        ("ms", [(1_700_000_000 + index * 3_600) * 1_000 for index in range(6)]),
        ("us", [(1_700_000_000 + index * 3_600) * 1_000_000 for index in range(6)]),
    ],
)
def test_numeric_index_vectors_round_trip_with_declared_source_unit(
    unit: str,
    legacy: list[int],
) -> None:
    dataset = canonicalize_ohlcv(
        _frame_with_numeric_times(legacy),
        symbol="EURUSD",
        timeframe="H1",
        numeric_time_unit=unit,
    )

    assert dataset.frame["time"].astype("int64").tolist() == sorted(
        value * _TEST_TIME_UNIT_NS[unit] for value in legacy
    )


@pytest.mark.parametrize("time_kind", ["s", "ms", "us", "ns", "datetime", "string"])
@pytest.mark.parametrize("index_kind", ["unique", "duplicate", "string", "multi"])
def test_canonicalization_is_independent_of_input_index_labels(
    time_kind: str,
    index_kind: str,
) -> None:
    frame = valid_frame()
    frame["time"] = _time_values_for_index_invariance(time_kind)
    frame = frame.iloc[[4, 1, 5, 0, 3, 2]].copy()
    if index_kind == "duplicate":
        frame.index = [7, 7, 2, 2, 9, 9]
    elif index_kind == "string":
        frame.index = ["echo", "bravo", "foxtrot", "alpha", "delta", "charlie"]
    elif index_kind == "multi":
        frame.index = pd.MultiIndex.from_tuples(
            [("z", 2), ("a", 1), ("z", 2), ("b", 0), ("a", 1), ("c", 3)]
        )
    expected_input = frame.copy(deep=True)
    expected_index = frame.index.copy()

    numeric_time_unit = time_kind if time_kind in _TEST_TIME_UNIT_NS else None
    retained_index = canonicalize_ohlcv(
        frame,
        symbol="EURUSD",
        timeframe="H1",
        numeric_time_unit=numeric_time_unit,
    )
    reset_index = canonicalize_ohlcv(
        frame.reset_index(drop=True),
        symbol="EURUSD",
        timeframe="H1",
        numeric_time_unit=numeric_time_unit,
    )

    pd.testing.assert_frame_equal(retained_index.frame, reset_index.frame)
    assert retained_index.identity == reset_index.identity
    pd.testing.assert_frame_equal(frame, expected_input)
    assert frame.index.equals(expected_index)


def test_unsorted_rows_keep_time_ohlcv_and_volume_bound_together() -> None:
    frame = valid_frame()
    frame["time"] = _time_values_for_index_invariance("ms")
    frame["open"] = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    frame["high"] = frame["open"] + 1.0
    frame["low"] = frame["open"] - 1.0
    frame["close"] = frame["open"] + 0.25
    frame["volume"] = frame["open"] * 10.0
    frame = frame.iloc[[5, 2, 0, 4, 1, 3]].copy()
    frame.index = ["f", "c", "a", "e", "b", "d"]
    expected_input = frame.copy(deep=True)

    dataset = canonicalize_ohlcv(
        frame,
        symbol="EURUSD",
        timeframe="H1",
        numeric_time_unit="ms",
    )

    assert list(
        zip(
            dataset.frame["open"].tolist(),
            dataset.frame["high"].tolist(),
            dataset.frame["low"].tolist(),
            dataset.frame["close"].tolist(),
            dataset.frame["volume"].tolist(),
        )
    ) == [
        (10.0, 11.0, 9.0, 10.25, 100.0),
        (20.0, 21.0, 19.0, 20.25, 200.0),
        (30.0, 31.0, 29.0, 30.25, 300.0),
        (40.0, 41.0, 39.0, 40.25, 400.0),
        (50.0, 51.0, 49.0, 50.25, 500.0),
        (60.0, 61.0, 59.0, 60.25, 600.0),
    ]
    pd.testing.assert_frame_equal(frame, expected_input)


def test_bar_spacing_shorter_than_timeframe_is_rejected() -> None:
    frame = valid_frame()
    frame.loc[1, "time"] = frame.loc[0, "time"] + pd.Timedelta(minutes=30)

    with pytest.raises(DataValidationError, match="shorter than timeframe"):
        canonicalize_ohlcv(frame, symbol="EURUSD", timeframe="H1")


def test_mixed_numeric_timestamp_units_are_rejected() -> None:
    frame = valid_frame().iloc[:3].copy()
    frame["time"] = [
        1_700_000_000,
        1_700_003_600_000,
        1_700_007_200_000,
    ]

    with pytest.raises(DataValidationError, match=r"provenance|unit.*required"):
        canonicalize_ohlcv(frame, symbol="EURUSD", timeframe="H1")


@pytest.mark.parametrize(
    ("unit", "base", "step", "scale"),
    [
        ("s", 1_700_000_000, 3_600, 1_000_000_000),
        ("ms", 1_700_000_000_000, 3_600_000, 1_000_000),
        ("us", 1_700_000_000_000_000, 3_600_000_000, 1_000),
        ("ns", 1_700_000_000_000_000_000, 3_600_000_000_000, 1),
    ],
)
@pytest.mark.parametrize(
    ("offsets", "expected_gap_count"),
    [
        ([0, 1, 2, 3, 4, 5], 0),
        ([0, 1, 3, 4, 8, 11], 3),
    ],
    ids=["continuous", "legal-gaps"],
)
def test_explicit_numeric_time_unit_round_trips_real_cadence(
    unit: str,
    base: int,
    step: int,
    scale: int,
    offsets: list[int],
    expected_gap_count: int,
) -> None:
    timestamps = [base + step * offset for offset in offsets]

    dataset = canonicalize_ohlcv(
        _frame_with_numeric_times(timestamps),
        symbol="EURUSD",
        timeframe="H1",
        numeric_time_unit=unit,
    )

    assert dataset.frame["time"].astype("int64").tolist() == sorted(
        value * scale for value in timestamps
    )
    assert dataset.gap_count == expected_gap_count


@pytest.mark.parametrize(
    "timestamps",
    [
        [1_700_000_000 + 3_600 * index for index in range(6)],
        [3_600, 10_800_000, 14_400_000],
        [-568_800_000, -565_200_000, -432_000_000, -255_600, -205_200_000],
    ],
    ids=["uniform-seconds", "three-row-mixed", "isolated-mixed"],
)
def test_untagged_numeric_time_requires_trusted_unit_provenance(
    timestamps: list[int],
) -> None:
    with pytest.raises(DataValidationError, match=r"provenance|unit.*required"):
        canonicalize_ohlcv(
            _frame_with_numeric_times(timestamps),
            symbol="EURUSD",
            timeframe="H1",
        )


@pytest.mark.parametrize("unit", ["minutes", "", True, 1])
def test_numeric_time_unit_rejects_unknown_values(unit: object) -> None:
    with pytest.raises(DataValidationError, match="numeric time unit"):
        canonicalize_ohlcv(
            _frame_with_numeric_times([1_700_000_000 + 3_600 * i for i in range(3)]),
            symbol="EURUSD",
            timeframe="H1",
            numeric_time_unit=unit,  # type: ignore[arg-type]
        )


def test_explicit_numeric_time_unit_still_rejects_out_of_range_values() -> None:
    timestamps = [pd.Timestamp.max.value - 3_600_000_000_000, pd.Timestamp.max.value + 1]

    with pytest.raises(DataValidationError, match=r"range|convert"):
        canonicalize_ohlcv(
            _frame_with_numeric_times(timestamps),
            symbol="EURUSD",
            timeframe="H1",
            numeric_time_unit="ns",
        )


def test_incorrect_declared_numeric_unit_still_fails_spacing_validation() -> None:
    seconds = [1_700_000_000 + 3_600 * index for index in range(3)]

    with pytest.raises(DataValidationError, match="shorter than timeframe"):
        canonicalize_ohlcv(
            _frame_with_numeric_times(seconds),
            symbol="EURUSD",
            timeframe="H1",
            numeric_time_unit="ms",
        )


def test_three_bar_mixed_numeric_timestamp_units_near_epoch_are_rejected() -> None:
    frame = _frame_with_numeric_times([3_600, 10_800_000, 14_400_000])

    with pytest.raises(DataValidationError, match=r"provenance|unit.*required"):
        canonicalize_ohlcv(frame, symbol="EURUSD", timeframe="H1")


@pytest.mark.parametrize(
    "timestamps",
    [
        [3_600, 10_800_000, 14_400_000],
        [-568_800_000, -565_200_000, -432_000_000, -255_600, -205_200_000],
        [
            -1_800_000_000_000,
            -1_713_600_000_000,
            -1_674_000,
            -1_605_600_000_000,
            -1_526_400_000,
            -1_458_000_000,
        ],
        [
            1_700_000_000,
            1_700_010_800_000_000,
            1_700_014_400_000_000,
            1_700_028_800_000_000,
            1_700_039_600_000_000,
        ],
    ],
    ids=["three-bar", "single-isolated", "three-isolated", "isolated-leading"],
)
def test_known_mixed_vectors_require_unit_provenance(
    timestamps: list[int],
) -> None:
    with pytest.raises(DataValidationError, match=r"provenance|unit.*required"):
        canonicalize_ohlcv(
            _frame_with_numeric_times(timestamps),
            symbol="EURUSD",
            timeframe="H1",
        )


@pytest.mark.parametrize(
    ("unit", "timestamps"),
    [
        ("s", [0, -7_200, -3_600]),
        ("ms", [1_700_000_000_001, 1_800_000_800_001, 1_800_004_400_001]),
        (
            "us",
            [
                1_700_000_000_000_001,
                1_700_010_800_000_001,
                1_700_025_200_000_001,
                1_700_064_800_000_001,
            ],
        ),
        (
            "ns",
            [
                1_700_025_200_000_000_123,
                1_700_000_000_000_000_123,
                1_700_064_800_000_000_123,
                1_700_010_800_000_000_123,
            ],
        ),
    ],
)
def test_four_numeric_controls_round_trip_with_declared_unit(
    unit: str,
    timestamps: list[int],
) -> None:
    dataset = canonicalize_ohlcv(
        _frame_with_numeric_times(timestamps),
        symbol="EURUSD",
        timeframe="H1",
        numeric_time_unit=unit,
    )

    expected = sorted(value * _TEST_TIME_UNIT_NS[unit] for value in timestamps)
    assert dataset.frame["time"].astype("int64").tolist() == expected


@pytest.mark.parametrize("unit", [" S ", "MS", "Us", "nS"])
def test_numeric_time_unit_is_case_and_whitespace_normalized(unit: str) -> None:
    normalized = unit.strip().lower()
    timestamps = _UNIQUE_PURE_NUMERIC_TIMES[normalized]

    dataset = canonicalize_ohlcv(
        _frame_with_numeric_times(timestamps),
        symbol="EURUSD",
        timeframe="H1",
        numeric_time_unit=unit,
    )

    assert dataset.frame["time"].astype("int64").tolist() == sorted(
        value * _TEST_TIME_UNIT_NS[normalized] for value in timestamps
    )


@pytest.mark.parametrize(
    "timestamps",
    [
        [-568_800_000, -565_200_000, -432_000_000, -255_600, -205_200_000],
        [
            -1_800_000_000_000,
            -1_713_600_000_000,
            -1_674_000,
            -1_605_600_000_000,
            -1_526_400_000,
            -1_458_000_000,
        ],
    ],
    ids=["single-isolated-no-span-ratio", "three-isolated-two-units"],
)
def test_mixed_numeric_timestamp_units_do_not_depend_on_isolated_limits(
    timestamps: list[int],
) -> None:
    with pytest.raises(DataValidationError, match=r"provenance|unit.*required"):
        canonicalize_ohlcv(
            _frame_with_numeric_times(timestamps),
            symbol="EURUSD",
            timeframe="H1",
        )


def test_mixed_numeric_timestamp_units_at_double_cadence_are_rejected() -> None:
    frame = valid_frame().iloc[:3].copy()
    frame["time"] = [
        1_700_000_000,
        1_700_007_200_000,
        1_700_014_400_000,
    ]

    with pytest.raises(DataValidationError, match=r"provenance|unit.*required"):
        canonicalize_ohlcv(frame, symbol="EURUSD", timeframe="H1")


def test_mixed_numeric_timestamp_units_with_lattice_insertion_are_rejected() -> None:
    timestamps = [
        1_700_000_000,
        1_700_007_200_000,
        1_700_010_800_000,
        1_700_014_400_000,
    ]
    selected_ns = sorted(value * _TEST_TIME_UNIT_NS["ms"] for value in timestamps)
    mixed_ns = sorted(
        [
            timestamps[0] * _TEST_TIME_UNIT_NS["s"],
            *(value * _TEST_TIME_UNIT_NS["ms"] for value in timestamps[1:]),
        ]
    )
    selected_exact = sum(
        current - previous == _TEST_H1_NS
        for previous, current in zip(selected_ns, selected_ns[1:])
    )
    mixed_exact = sum(
        current - previous == _TEST_H1_NS
        for previous, current in zip(mixed_ns, mixed_ns[1:])
    )
    assert selected_exact == mixed_exact == 2

    with pytest.raises(DataValidationError, match=r"provenance|unit.*required"):
        canonicalize_ohlcv(
            _frame_with_numeric_times(timestamps),
            symbol="EURUSD",
            timeframe="H1",
        )


def test_isolated_leading_mixed_numeric_timestamp_unit_is_rejected() -> None:
    timestamps = [
        1_700_000_000,
        1_700_010_800_000_000,
        1_700_014_400_000_000,
        1_700_028_800_000_000,
        1_700_039_600_000_000,
    ]

    with pytest.raises(DataValidationError, match=r"provenance|unit.*required"):
        canonicalize_ohlcv(
            _frame_with_numeric_times(timestamps),
            symbol="EURUSD",
            timeframe="H1",
        )


_ISOLATED_MIXED_UNIT_PAIRS = [
    (isolated, dominant)
    for isolated in _TEST_TIME_UNIT_NS
    for dominant in _TEST_TIME_UNIT_NS
    if isolated != dominant
]


@pytest.mark.parametrize(
    ("isolated_unit", "dominant_unit"),
    _ISOLATED_MIXED_UNIT_PAIRS,
    ids=[
        f"{isolated}-into-{dominant}"
        for isolated, dominant in _ISOLATED_MIXED_UNIT_PAIRS
    ],
)
@pytest.mark.parametrize(
    "isolated_position",
    [0, 2, 4],
    ids=["leading", "middle", "trailing"],
)
def test_isolated_mixed_unit_pair_positions_are_rejected(
    isolated_unit: str,
    dominant_unit: str,
    isolated_position: int,
) -> None:
    units = [dominant_unit] * 5
    units[isolated_position] = isolated_unit
    timestamps = _encode_semantic_times([0, 3, 4, 8, 11], units)

    with pytest.raises(DataValidationError, match=r"provenance|unit.*required"):
        canonicalize_ohlcv(
            _frame_with_numeric_times(timestamps),
            symbol="EURUSD",
            timeframe="H1",
        )


def test_isolated_mixed_unit_rejection_is_input_order_independent() -> None:
    timestamps = _encode_semantic_times(
        [0, 3, 4, 8, 11],
        ["s", "us", "us", "us", "us"],
    )
    frame = _frame_with_numeric_times(timestamps).iloc[[3, 0, 4, 1, 2]]

    with pytest.raises(DataValidationError, match=r"provenance|unit.*required"):
        canonicalize_ohlcv(frame, symbol="EURUSD", timeframe="H1")


def test_two_isolated_mixed_unit_rows_are_rejected() -> None:
    timestamps = _encode_semantic_times(
        [0, 3, 4, 8, 11],
        ["s", "us", "us", "us", "ms"],
    )

    with pytest.raises(DataValidationError, match=r"provenance|unit.*required"):
        canonicalize_ohlcv(
            _frame_with_numeric_times(timestamps),
            symbol="EURUSD",
            timeframe="H1",
        )


@pytest.mark.parametrize("unit", list(_TEST_TIME_UNIT_NS))
@pytest.mark.parametrize(
    "base_ns",
    [
        1_700_000_000_000_000_000,
        -24 * _TEST_H1_NS,
        0,
    ],
    ids=["modern", "negative", "epoch-zero"],
)
def test_epoch_origin_matrix_round_trips_declared_unit(
    unit: str,
    base_ns: int,
) -> None:
    semantic_ns = [
        base_ns + offset * _TEST_H1_NS
        for offset in [0, 3, 4, 8, 11]
    ]
    timestamps = [
        timestamp_ns // _TEST_TIME_UNIT_NS[unit]
        for timestamp_ns in semantic_ns
    ]
    frame = _frame_with_numeric_times(timestamps).iloc[[3, 0, 4, 1, 2]]

    result = canonicalize_ohlcv(
        frame,
        symbol="EURUSD",
        timeframe="H1",
        numeric_time_unit=unit,
    )

    assert result.frame["time"].astype("int64").tolist() == semantic_ns
    assert result.gap_count == 3


def test_mixed_epoch_unit_lattice_insertion_matrix_is_rejected() -> None:
    units = tuple(_TEST_TIME_UNIT_NS)
    accepted: list[tuple[object, ...]] = []
    wrong_errors: list[tuple[object, ...]] = []
    cases = 0
    for minority_unit in units:
        for majority_unit in units:
            if minority_unit == majority_unit:
                continue
            for length in range(4, 9):
                for multiplier in (1, 2, 4, 7):
                    offsets = [0, 2 * multiplier, 3 * multiplier, 4 * multiplier]
                    offsets.extend(
                        (5 + index) * multiplier for index in range(length - 4)
                    )
                    encoded = _encode_semantic_times(
                        offsets,
                        [minority_unit, *([majority_unit] * (length - 1))],
                    )
                    for reverse in (False, True):
                        cases += 1
                        frame = _frame_with_numeric_times(encoded)
                        if reverse:
                            frame = frame.iloc[::-1]
                        try:
                            canonicalize_ohlcv(
                                frame,
                                symbol="EURUSD",
                                timeframe="H1",
                            )
                        except DataValidationError as exc:
                            if not (
                                "provenance" in str(exc)
                                or "unit provenance is required" in str(exc)
                            ):
                                wrong_errors.append(
                                    (
                                        minority_unit,
                                        majority_unit,
                                        length,
                                        multiplier,
                                        reverse,
                                        str(exc),
                                    )
                                )
                        else:
                            accepted.append(
                                (
                                    minority_unit,
                                    majority_unit,
                                    length,
                                    multiplier,
                                    reverse,
                                )
                            )

    assert cases == 480
    assert not accepted, f"accepted mixed-unit cases: {len(accepted)}/480; {accepted[:3]}"
    assert not wrong_errors, f"wrong domain errors: {wrong_errors[:3]}"


@pytest.mark.parametrize(
    "units",
    [
        ["s", "ms", "us", "us", "us"],
        ["ns", "us", "ms", "ms", "ms"],
        ["ms", "s", "us", "us", "us"],
        ["us", "ns", "s", "s", "s"],
    ],
)
def test_three_epoch_unit_lattice_insertions_are_rejected(units: list[str]) -> None:
    encoded = _encode_semantic_times([0, 2, 3, 4, 5], units)
    frame = _frame_with_numeric_times(encoded).iloc[[3, 0, 4, 1, 2]]

    with pytest.raises(DataValidationError, match=r"provenance|unit.*required"):
        canonicalize_ohlcv(frame, symbol="EURUSD", timeframe="H1")


@pytest.mark.parametrize(
    ("units", "gap_multiplier"),
    [
        (("s", "ms", "ms"), 3),
        (("s", "ms", "us"), 5),
        (("ms", "s", "ms"), 2),
        (("ms", "ms", "s"), 4),
        (("ms", "us", "us"), 6),
    ],
)
def test_mixed_numeric_timestamp_unit_combinations_at_cadence_multiples_are_rejected(
    units: tuple[str, str, str],
    gap_multiplier: int,
) -> None:
    unit_ns = {"s": 1_000_000_000, "ms": 1_000_000, "us": 1_000}
    base_ns = 1_700_000_000_000_000_000
    nominal_ns = 3_600_000_000_000
    semantic_ns = [
        base_ns + index * gap_multiplier * nominal_ns for index in range(3)
    ]
    encoded = [
        timestamp_ns // unit_ns[unit]
        for timestamp_ns, unit in zip(semantic_ns, units)
    ]
    frame = valid_frame().iloc[:3].copy()
    frame["time"] = encoded

    with pytest.raises(DataValidationError, match=r"provenance|unit.*required"):
        canonicalize_ohlcv(frame, symbol="EURUSD", timeframe="H1")


@pytest.mark.parametrize(
    ("unit", "unit_ns", "gap_multipliers"),
    [
        ("s", 1_000_000_000, (2, 5)),
        ("ms", 1_000_000, (3, 7)),
        ("us", 1_000, (4, 9)),
        ("ns", 1, (2, 11)),
    ],
    ids=["seconds", "milliseconds", "microseconds", "nanoseconds"],
)
def test_cadence_multiple_matrix_round_trips_declared_unit(
    unit: str,
    unit_ns: int,
    gap_multipliers: tuple[int, int],
) -> None:
    base_ns = 1_700_000_000_000_000_000
    nominal_ns = 3_600_000_000_000
    semantic_ns = [base_ns]
    for multiplier in gap_multipliers:
        semantic_ns.append(semantic_ns[-1] + multiplier * nominal_ns)
    encoded = [timestamp_ns // unit_ns for timestamp_ns in semantic_ns]
    frame = valid_frame().iloc[:3].copy()
    frame["time"] = encoded[::-1]

    dataset = canonicalize_ohlcv(
        frame,
        symbol="EURUSD",
        timeframe="H1",
        numeric_time_unit=unit,
    )

    assert dataset.frame["time"].astype("int64").tolist() == semantic_ns
    assert dataset.identity.start_time_ns == semantic_ns[0]
    assert dataset.identity.end_time_ns == semantic_ns[-1]
    assert dataset.gap_count == 2


@pytest.mark.parametrize(
    ("unit", "unit_ns"),
    [
        ("ms", 1_000_000),
        ("us", 1_000),
        ("ns", 1),
    ],
)
def test_subsecond_negative_triple_cadence_round_trips_declared_unit(
    unit: str,
    unit_ns: int,
) -> None:
    semantic_ns = [-6 * _TEST_H1_NS, -3 * _TEST_H1_NS, 0]
    encoded = [timestamp_ns // unit_ns for timestamp_ns in semantic_ns]

    dataset = canonicalize_ohlcv(
        _frame_with_numeric_times(encoded),
        symbol="EURUSD",
        timeframe="H1",
        numeric_time_unit=unit,
    )

    assert dataset.frame["time"].astype("int64").tolist() == semantic_ns


def test_numeric_epoch_matrix_round_trips_generator_declared_unit() -> None:
    failures: list[tuple[object, ...]] = []
    cases = 0
    successful_cases = 0
    for unit, unit_ns in _TEST_TIME_UNIT_NS.items():
        for length in range(3, 9):
            gap_patterns = [
                [multiplier] * (length - 1) for multiplier in (2, 3, 4, 7)
            ]
            gap_patterns.append(
                [(2, 3, 4, 7)[index % 4] for index in range(length - 1)]
            )
            for gaps in gap_patterns:
                offsets = [0]
                for gap in gaps:
                    offsets.append(offsets[-1] + gap)
                bases = {
                    "negative": -(offsets[-1] + 5) * _TEST_H1_NS,
                    "epoch_zero": 0,
                    "cross_epoch": -offsets[length // 2] * _TEST_H1_NS,
                    "positive": 1_700_000_000_000_000_000,
                }
                for base_name, base_ns in bases.items():
                    semantic_ns = [
                        base_ns + offset * _TEST_H1_NS for offset in offsets
                    ]
                    encoded = [value // unit_ns for value in semantic_ns]
                    orders = [
                        list(range(length)),
                        list(reversed(range(length))),
                        list(range(1, length)) + [0],
                    ]
                    expected_identity = None
                    for order_name, order in zip(
                        ("forward", "reverse", "rotate"),
                        orders,
                    ):
                        cases += 1
                        frame = _frame_with_numeric_times(encoded).iloc[order]
                        try:
                            dataset = canonicalize_ohlcv(
                                frame,
                                symbol="EURUSD",
                                timeframe="H1",
                                numeric_time_unit=unit,
                            )
                            assert (
                                dataset.frame["time"].astype("int64").tolist()
                                == semantic_ns
                            )
                            assert dataset.identity.start_time_ns == semantic_ns[0]
                            assert dataset.identity.end_time_ns == semantic_ns[-1]
                            assert dataset.gap_count == len(gaps)
                            successful_cases += 1
                            if expected_identity is None:
                                expected_identity = dataset.identity
                            else:
                                assert dataset.identity == expected_identity
                        except (AssertionError, DataValidationError) as exc:
                            failures.append(
                                (
                                    unit,
                                    length,
                                    tuple(gaps),
                                    base_name,
                                    order_name,
                                    str(exc),
                                )
                            )

    assert cases == 1_440
    assert successful_cases == 1_440
    assert not failures, (
        f"declared-unit round-trip mismatches: {len(failures)}/1440; {failures[:5]}"
    )


@pytest.mark.parametrize(
    ("unit", "unit_ns"),
    list(_TEST_TIME_UNIT_NS.items()),
)
def test_long_irregular_numeric_sequences_round_trip_declared_unit(
    unit: str,
    unit_ns: int,
) -> None:
    base_ns = 1_700_000_000_000_000_000
    gaps = [1, 2, 7, 1, 11, 3, 4]
    offsets = [0]
    for gap in gaps:
        offsets.append(offsets[-1] + gap)
    semantic_ns = [base_ns + offset * _TEST_H1_NS for offset in offsets]
    encoded = [value // unit_ns for value in semantic_ns]
    frame = _frame_with_numeric_times(encoded).iloc[[7, 2, 0, 5, 1, 6, 3, 4]]

    dataset = canonicalize_ohlcv(
        frame,
        symbol="EURUSD",
        timeframe="H1",
        numeric_time_unit=unit,
    )

    assert dataset.frame["time"].astype("int64").tolist() == semantic_ns
    assert dataset.gap_count == sum(gap > 1 for gap in gaps)


def test_consistent_early_milliseconds_can_cross_inference_boundary() -> None:
    frame = valid_frame().iloc[:3].copy()
    milliseconds = [
        99_999_999_000,
        100_003_599_000,
        100_007_199_000,
    ]
    frame["time"] = milliseconds

    result = canonicalize_ohlcv(
        frame,
        symbol="EURUSD",
        timeframe="H1",
        numeric_time_unit="ms",
    )

    expected = pd.to_datetime(milliseconds, unit="ms", utc=True).astype(
        "datetime64[ns, UTC]"
    )
    assert result.frame["time"].tolist() == expected.tolist()
    assert result.gap_count == 0


def test_epoch_zero_is_unit_neutral_for_consistent_early_milliseconds() -> None:
    frame = valid_frame().iloc[:3].copy()
    milliseconds = [0, 99_999_999_000, 100_003_599_000]
    frame["time"] = milliseconds

    result = canonicalize_ohlcv(
        frame,
        symbol="EURUSD",
        timeframe="H1",
        numeric_time_unit="ms",
    )

    expected = pd.to_datetime(milliseconds, unit="ms", utc=True).astype(
        "datetime64[ns, UTC]"
    )
    assert str(result.frame["time"].dtype) == "datetime64[ns, UTC]"
    assert result.frame["time"].astype("int64").tolist() == expected.astype(
        "int64"
    ).tolist()
    assert result.identity.start_time_ns == 0
    assert result.identity.end_time_ns == int(expected.astype("int64")[-1])
    assert result.gap_count == 1
    assert len(result.frame) == 3


@pytest.mark.parametrize(
    "milliseconds",
    [
        [0, 3_600_000, 7_200_000],
        [-7_200_000, -3_600_000, 0],
    ],
)
def test_epoch_milliseconds_with_a_mixed_witness_fail_closed(
    milliseconds: list[int],
) -> None:
    frame = valid_frame().iloc[:3].copy()
    frame["time"] = milliseconds

    with pytest.raises(DataValidationError, match=r"provenance|unit.*required"):
        canonicalize_ohlcv(frame, symbol="EURUSD", timeframe="H1")


def test_numeric_time_unit_preserves_epoch_seconds_with_h1_cadence() -> None:
    frame = valid_frame().iloc[:3].copy()
    seconds = [0, 3_600, 7_200]
    frame["time"] = seconds

    result = canonicalize_ohlcv(
        frame,
        symbol="EURUSD",
        timeframe="H1",
        numeric_time_unit="s",
    )

    expected = pd.to_datetime(seconds, unit="s", utc=True).astype(
        "datetime64[ns, UTC]"
    )
    assert result.frame["time"].astype("int64").tolist() == expected.astype(
        "int64"
    ).tolist()
    assert result.gap_count == 0


def test_ambiguous_epoch_numeric_time_unit_is_rejected() -> None:
    frame = valid_frame().iloc[:3].copy()
    frame["time"] = [0, 1_800_000, 3_600_000]

    with pytest.raises(DataValidationError, match=r"provenance|unit.*required"):
        canonicalize_ohlcv(frame, symbol="EURUSD", timeframe="H1")


def test_long_gap_numeric_times_with_a_valid_mixed_interpretation_are_ambiguous() -> None:
    frame = valid_frame().iloc[:3].copy()
    milliseconds = [1_000_000, 100_000_000_000, 100_003_600_000]
    frame["time"] = milliseconds

    with pytest.raises(DataValidationError, match=r"provenance|unit.*required"):
        canonicalize_ohlcv(frame, symbol="EURUSD", timeframe="H1")


def test_uniquely_identified_milliseconds_allow_a_real_long_gap() -> None:
    frame = valid_frame().iloc[:3].copy()
    milliseconds = [1_700_000_000_001, 1_800_000_800_001, 1_800_004_400_001]
    frame["time"] = milliseconds

    result = canonicalize_ohlcv(
        frame,
        symbol="EURUSD",
        timeframe="H1",
        numeric_time_unit="ms",
    )

    expected_ns = (
        pd.to_datetime(milliseconds, unit="ms", utc=True)
        .astype("datetime64[ns, UTC]")
        .astype("int64")
        .tolist()
    )
    assert result.frame["time"].astype("int64").tolist() == expected_ns
    assert result.identity.start_time_ns == expected_ns[0]
    assert result.identity.end_time_ns == expected_ns[-1]
    assert result.gap_count == 1


@pytest.mark.parametrize(
    ("timestamps", "unit"),
    [
        ([0.0, 3_600.5, 7_200.0], "s"),
        (
            [1.0e18, 1.0e18 + 3_600_000_000_000, 1.0e18 + 7_200_000_000_000],
            "ns",
        ),
    ],
    ids=["fractional", "lossy-float-ns"],
)
def test_float_numeric_timestamps_must_be_lossless_integers(
    timestamps: list[float],
    unit: str,
) -> None:
    frame = valid_frame().iloc[:3].copy()
    frame["time"] = timestamps

    with pytest.raises(DataValidationError, match=r"integer|lossless"):
        canonicalize_ohlcv(
            frame,
            symbol="EURUSD",
            timeframe="H1",
            numeric_time_unit=unit,
        )


@pytest.mark.parametrize(
    ("dtype", "timestamps", "unit", "timeframe"),
    [
        (np.float16, [3_601, 7_201, 10_801], "s", "H1"),
        (
            np.float32,
            [1_700_000_001 + 86_400 * index for index in range(3)],
            "s",
            "D1",
        ),
        (
            "Float32",
            [1_700_000_001 + 86_400 * index for index in range(3)],
            "s",
            "D1",
        ),
        (
            np.float64,
            [2**53 + 1 + 3_600_000_000_000 * index for index in range(3)],
            "ns",
            "H1",
        ),
        (
            "Float64",
            [2**53 + 1 + 3_600_000_000_000 * index for index in range(3)],
            "ns",
            "H1",
        ),
    ],
    ids=[
        "numpy-float16",
        "numpy-float32",
        "pandas-Float32",
        "numpy-float64",
        "pandas-Float64",
    ],
)
def test_float_timestamp_dtype_must_guarantee_consecutive_integer_identity(
    dtype: object,
    timestamps: list[int],
    unit: str,
    timeframe: str,
) -> None:
    frame = valid_frame().iloc[:3].copy()
    frame["time"] = pd.Series(timestamps, dtype=dtype)

    with pytest.raises(DataValidationError, match=r"lossless|precision"):
        canonicalize_ohlcv(
            frame,
            symbol="EURUSD",
            timeframe=timeframe,
            numeric_time_unit=unit,
        )


@pytest.mark.parametrize(
    ("dtype", "timestamps", "unit"),
    [
        (np.float16, [1_920, 1_980, 2_040], "s"),
        (np.float32, [2**24 - 120, 2**24 - 60, 2**24], "s"),
        ("Float32", [2**24 - 120, 2**24 - 60, 2**24], "s"),
        (
            np.float64,
            [2**53 - 120_000_000_000, 2**53 - 60_000_000_000, 2**53],
            "ns",
        ),
        (
            "Float64",
            [2**53 - 120_000_000_000, 2**53 - 60_000_000_000, 2**53],
            "ns",
        ),
    ],
    ids=[
        "numpy-float16",
        "numpy-float32",
        "pandas-Float32",
        "numpy-float64",
        "pandas-Float64",
    ],
)
def test_float_timestamp_values_within_dtype_precision_remain_valid(
    dtype: object,
    timestamps: list[int],
    unit: str,
) -> None:
    frame = valid_frame().iloc[:3].copy()
    frame["time"] = pd.Series(timestamps, dtype=dtype)

    result = canonicalize_ohlcv(
        frame,
        symbol="EURUSD",
        timeframe="M1",
        numeric_time_unit=unit,
    )

    scale = _TEST_TIME_UNIT_NS[unit]
    assert result.frame["time"].astype("int64").tolist() == [
        timestamp * scale for timestamp in timestamps
    ]


@pytest.mark.parametrize(
    "dtype",
    [np.float16, np.float32, "Float32", np.float64, "Float64"],
    ids=[
        "numpy-float16",
        "numpy-float32",
        "pandas-Float32",
        "numpy-float64",
        "pandas-Float64",
    ],
)
@pytest.mark.parametrize(
    ("timestamps", "message"),
    [([0.0, 60.5, 120.0], "integer"), ([0.0, 60.0, float("inf")], "finite")],
    ids=["fractional", "nonfinite"],
)
def test_float_timestamp_dtype_still_rejects_fractional_and_nonfinite_values(
    dtype: object,
    timestamps: list[float],
    message: str,
) -> None:
    frame = valid_frame().iloc[:3].copy()
    frame["time"] = pd.Series(timestamps, dtype=dtype)

    with pytest.raises(DataValidationError, match=message):
        canonicalize_ohlcv(
            frame,
            symbol="EURUSD",
            timeframe="M1",
            numeric_time_unit="s",
        )


def test_out_of_range_integer_numeric_timestamps_are_rejected() -> None:
    frame = valid_frame().iloc[:3].copy()
    start = pd.Timestamp.max.value + 1
    frame["time"] = [start, start + 3_600_000_000_000, start + 7_200_000_000_000]

    with pytest.raises(DataValidationError, match=r"range|convert"):
        canonicalize_ohlcv(
            frame,
            symbol="EURUSD",
            timeframe="H1",
            numeric_time_unit="ns",
        )


@pytest.mark.parametrize(
    ("case", "timestamps"),
    [
        ("python-seconds", [1_700_000_000, 1_700_003_600, 1_700_007_200]),
        (
            "python-milliseconds",
            [1_700_000_000_000, 1_700_003_600_000, 1_700_007_200_000],
        ),
        (
            "python-microseconds",
            [
                1_700_000_000_000_000,
                1_700_003_600_000_000,
                1_700_007_200_000_000,
            ],
        ),
        (
            "python-nanoseconds",
            [
                1_700_000_000_000_000_000,
                1_700_003_600_000_000_000,
                1_700_007_200_000_000_000,
            ],
        ),
        (
            "mixed-units",
            [1_700_000_000, 1_700_003_600_000, 1_700_007_200_000_000_000],
        ),
        (
            "numpy-integer",
            [np.int64(1_700_000_000), np.int64(1_700_003_600), np.int64(1_700_007_200)],
        ),
        (
            "numpy-floating",
            [np.float64(1_700_000_000), np.float64(1_700_003_600), np.float64(1_700_007_200)],
        ),
        ("boolean", [True, False, True]),
        ("numpy-boolean", [np.bool_(True), np.bool_(False), np.bool_(True)]),
        ("complex", [1_700_000_000 + 0j, 1_700_003_600 + 0j, 1_700_007_200 + 0j]),
        (
            "mixed-datetime-and-numeric",
            [
                pd.Timestamp("2026-01-01T00:00:00Z"),
                1_767_226_800_000_000_000,
                "2026-01-01T02:00:00Z",
            ],
        ),
    ],
)
def test_object_time_rejects_numeric_boolean_and_complex_values(
    case: str,
    timestamps: list[object],
) -> None:
    del case
    frame = valid_frame().iloc[:3].copy()
    frame["time"] = pd.Series(timestamps, dtype="object")

    with pytest.raises(
        DataValidationError,
        match=r"object.*time.*numeric|numeric.*object.*time",
    ):
        canonicalize_ohlcv(frame, symbol="EURUSD", timeframe="H1")


@pytest.mark.parametrize(
    "timestamps",
    [
        pd.Series(
            [
                "2026-01-01T00:00:00Z",
                "2026-01-01T01:00:00Z",
                "2026-01-01T02:00:00Z",
            ],
            dtype="object",
        ),
        pd.Series(
            pd.date_range("2026-01-01", periods=3, freq="1h", tz="UTC").tolist(),
            dtype="object",
        ),
        pd.Series(
            np.array(
                [
                    "2026-01-01T00:00:00",
                    "2026-01-01T01:00:00",
                    "2026-01-01T02:00:00",
                ],
                dtype="datetime64[s]",
            )
        ),
    ],
    ids=["datetime-strings", "timestamp-objects", "datetime64"],
)
def test_datetime_like_time_values_remain_valid(timestamps: pd.Series) -> None:
    frame = valid_frame().iloc[:3].copy()
    frame["time"] = timestamps

    result = canonicalize_ohlcv(frame, symbol="EURUSD", timeframe="H1")

    expected = pd.date_range("2026-01-01", periods=3, freq="1h", tz="UTC")
    assert result.frame["time"].tolist() == expected.tolist()


@pytest.mark.parametrize(
    "timestamps",
    [
        pd.Series(
            np.array(
                [
                    "3000-01-01T00:00:00",
                    "3000-01-01T01:00:00",
                    "3000-01-01T02:00:00",
                ],
                dtype="datetime64[us]",
            )
        ),
        pd.Series(
            np.array(
                [
                    "1600-01-01T00:00:00",
                    "1600-01-01T01:00:00",
                    "1600-01-01T02:00:00",
                ],
                dtype="datetime64[us]",
            )
        ),
        pd.Series(
            pd.array(
                [
                    "3000-01-01T00:00:00Z",
                    "3000-01-01T01:00:00Z",
                    "3000-01-01T02:00:00Z",
                ],
                dtype="datetime64[us, UTC]",
            )
        ),
        pd.Series(
            [
                "1600-01-01T00:00:00Z",
                "1600-01-01T01:00:00Z",
                "1600-01-01T02:00:00Z",
            ],
            dtype="object",
        ),
    ],
    ids=["upper-naive-us", "lower-naive-us", "upper-utc-us", "lower-strings"],
)
def test_datetime_like_values_outside_utc_ns_range_raise_domain_error(
    timestamps: pd.Series,
) -> None:
    frame = valid_frame().iloc[:3].copy()
    frame["time"] = timestamps

    with pytest.raises(DataValidationError, match=r"invalid time|out of range"):
        canonicalize_ohlcv(frame, symbol="EURUSD", timeframe="H1")


@pytest.mark.parametrize(
    "timestamps",
    [
        pd.Series(
            np.array(
                [
                    "2026-01-01T00:00:00",
                    "2026-01-01T01:00:00",
                    "2026-01-01T02:00:00",
                ],
                dtype="datetime64[us]",
            )
        ),
        pd.Series(
            pd.array(
                [
                    "2026-01-01T00:00:00Z",
                    "2026-01-01T01:00:00Z",
                    "2026-01-01T02:00:00Z",
                ],
                dtype="datetime64[us, UTC]",
            )
        ),
    ],
    ids=["naive-us", "utc-us"],
)
def test_in_range_microsecond_datetime_values_convert_to_utc_ns(
    timestamps: pd.Series,
) -> None:
    frame = valid_frame().iloc[:3].copy()
    frame["time"] = timestamps

    result = canonicalize_ohlcv(frame, symbol="EURUSD", timeframe="H1")

    expected = pd.date_range("2026-01-01", periods=3, freq="1h", tz="UTC")
    assert str(result.frame["time"].dtype) == "datetime64[ns, UTC]"
    assert result.frame["time"].tolist() == expected.tolist()


def test_extreme_in_range_ns_gap_does_not_overflow_spacing_check() -> None:
    frame = valid_frame().iloc[:2].copy()
    timestamps = [pd.Timestamp.min.value, pd.Timestamp.max.value]
    frame["time"] = timestamps

    result = canonicalize_ohlcv(
        frame,
        symbol="EURUSD",
        timeframe="H1",
        numeric_time_unit="ns",
    )

    assert result.frame["time"].astype("int64").tolist() == timestamps
    assert result.identity.start_time_ns == timestamps[0]
    assert result.identity.end_time_ns == timestamps[-1]
    assert result.gap_count == 1


@pytest.mark.parametrize(
    ("unit", "timestamps", "expected_ns"),
    [
        ("s", [-7_200, -3_600, 0], 1_000_000_000),
        (
            "ms",
            [1_700_000_000_001, 1_800_000_800_001, 1_800_004_400_001],
            1_000_000,
        ),
        (
            "us",
            [
                1_700_000_000_000_018,
                1_703_603_600_000_018,
                1_703_607_200_000_018,
            ],
            1_000,
        ),
        (
            "ns",
            [
                1_700_000_000_000_000_001,
                1_703_600_000_000_000_001,
                1_703_603_600_000_000_001,
            ],
            1,
        ),
    ],
)
def test_integer_numeric_timestamp_units_have_stable_utc_identity(
    unit: str,
    timestamps: list[int],
    expected_ns: int,
) -> None:
    frame = valid_frame().iloc[:3].copy()
    frame["time"] = timestamps[::-1]

    result = canonicalize_ohlcv(
        frame,
        symbol="EURUSD",
        timeframe="H1",
        numeric_time_unit=unit,
    )

    expected = [value * expected_ns for value in timestamps]
    assert result.frame["time"].astype("int64").tolist() == expected
    assert result.identity.start_time_ns == expected[0]
    assert result.identity.end_time_ns == expected[-1]


@pytest.mark.parametrize(
    "timestamps",
    [
        [1_700_000_000, 1_700_003_600, 1_700_007_200],
        [1_700_000_000_000, 1_700_003_600_000, 1_700_007_200_000],
        [
            1_700_000_000_000_000,
            1_700_003_600_000_000,
            1_700_007_200_000_000,
        ],
    ],
    ids=["seconds", "milliseconds", "microseconds"],
)
def test_untagged_identity_vectors_are_rejected_without_unit_provenance(
    timestamps: list[int],
) -> None:
    with pytest.raises(DataValidationError, match=r"provenance|unit.*required"):
        canonicalize_ohlcv(
            _frame_with_numeric_times(timestamps),
            symbol="EURUSD",
            timeframe="H1",
        )


def test_long_gap_is_counted_without_synthesizing_bars() -> None:
    frame = valid_frame().drop(index=[2, 3]).reset_index(drop=True)

    result = canonicalize_ohlcv(frame, symbol="EURUSD", timeframe="H1")

    assert result.gap_count == 1
    assert len(result.frame) == len(frame)
    assert result.frame["time"].tolist() == frame["time"].tolist()


def test_monthly_timeframe_requires_only_unique_increasing_time() -> None:
    frame = valid_frame()

    result = canonicalize_ohlcv(frame, symbol="EURUSD", timeframe="MN1")

    assert result.gap_count == 0
    assert len(result.frame) == len(frame)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("m1", "M1"),
        ("M5", "M5"),
        (15, "M15"),
        (30, "M30"),
        (16385, "H1"),
        (16388, "H4"),
        (16408, "D1"),
        (32769, "W1"),
        (49153, "MN1"),
    ],
)
def test_timeframe_names_are_normalized(value: str | int, expected: str) -> None:
    assert normalize_timeframe_name(value) == expected


@pytest.mark.parametrize("value", ["H2", 2, True, ""])
def test_unknown_timeframe_is_rejected(value: object) -> None:
    with pytest.raises(DataValidationError, match="unknown timeframe"):
        normalize_timeframe_name(value)  # type: ignore[arg-type]


def test_fingerprint_is_index_and_timezone_independent_and_content_sensitive() -> None:
    left_frame = valid_frame()
    right_frame = valid_frame().copy()
    right_frame.index = pd.Index([91, 82, 73, 64, 55, 46])
    right_frame["time"] = right_frame["time"].dt.tz_convert("Asia/Shanghai")
    changed = valid_frame()
    changed.loc[2, "close"] += 0.01

    left = canonicalize_ohlcv(left_frame, symbol="EURUSD", timeframe="H1")
    right = canonicalize_ohlcv(right_frame, symbol="EURUSD", timeframe="H1")
    third = canonicalize_ohlcv(changed, symbol="EURUSD", timeframe="H1")

    assert left.identity == right.identity
    assert left.identity.data_fingerprint != third.identity.data_fingerprint
    assert left.identity.time_fingerprint == third.identity.time_fingerprint


def test_identity_round_trip_is_stable_and_frozen() -> None:
    identity = canonicalize_ohlcv(
        valid_frame(), symbol="EURUSD", timeframe="H1"
    ).identity

    assert identity.schema_version == DATA_SCHEMA_VERSION
    assert DatasetIdentity.from_dict(identity.to_dict()) == identity
    with pytest.raises(FrozenInstanceError):
        identity.bars = 99  # type: ignore[misc]


def test_assert_minimum_bars_reports_expected_and_actual() -> None:
    assert_minimum_bars(6, 6, context="unit fixture")

    with pytest.raises(
        DataValidationError,
        match=r"unit fixture.*expected.*6.*actual.*5",
    ):
        assert_minimum_bars(5, 6, context="unit fixture")


@pytest.mark.parametrize("field", ["open", "volume"])
def test_boolean_ohlcv_values_are_rejected_before_numeric_coercion(
    field: str,
) -> None:
    frame = valid_frame()
    frame[field] = True

    with pytest.raises(
        DataValidationError,
        match=rf"boolean.*field={field}",
    ):
        canonicalize_ohlcv(frame, symbol="EURUSD", timeframe="H1")


@pytest.mark.parametrize(
    "invalid",
    [Decimal("sNaN"), Fraction(1, 1), object(), "not-a-number"],
    ids=["signaling-decimal", "fraction", "object", "text"],
)
def test_unsafe_object_values_raise_domain_error(invalid: object) -> None:
    frame = valid_frame()
    frame["volume"] = pd.Series([invalid] * len(frame), dtype="object")

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(
            DataValidationError,
            match=r"invalid numeric OHLCV value: field=volume",
        ):
            canonicalize_ohlcv(frame, symbol="EURUSD", timeframe="H1")


@pytest.mark.parametrize("field", ["open", "high", "low", "close", "volume"])
def test_canonicalization_rejects_lossy_integer_to_float64(field: str) -> None:
    exact = 2**53
    lossy = exact + 1
    frame = valid_frame()
    frame["open"] = pd.Series([exact + 2] * len(frame), dtype="int64")
    frame["high"] = pd.Series([exact + 4] * len(frame), dtype="int64")
    frame["low"] = pd.Series([exact] * len(frame), dtype="int64")
    frame["close"] = pd.Series([exact + 2] * len(frame), dtype="int64")
    frame["volume"] = pd.Series([100] * len(frame), dtype="int64")
    frame[field] = pd.Series([lossy] * len(frame), dtype="int64")

    with pytest.raises(DataValidationError, match=rf"float64.*field={field}"):
        canonicalize_ohlcv(frame, symbol="EURUSD", timeframe="H1")


@pytest.mark.parametrize(
    ("values", "dtype"),
    [
        ([int(2**53 + 1)] * 6, "object"),
        ([np.int64(2**53 + 1)] * 6, "object"),
        ([np.uint64(2**53 + 1)] * 6, "object"),
        ([2**53 + 1] * 6, "Int64"),
        ([2**53 + 1] * 6, "UInt64"),
    ],
    ids=["python-int", "numpy-int64", "numpy-uint64", "pandas-Int64", "pandas-UInt64"],
)
def test_integer_source_kinds_share_the_lossless_float64_guard(
    values: list[object],
    dtype: str,
) -> None:
    frame = valid_frame()
    frame["volume"] = pd.Series(values, dtype=dtype)

    with pytest.raises(DataValidationError, match=r"float64.*field=volume"):
        canonicalize_ohlcv(frame, symbol="EURUSD", timeframe="H1")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("open", Decimal(2**53 + 1)),
        ("high", Decimal(2**53 + 3)),
        ("low", Decimal(2**53 + 1)),
        ("close", Decimal(2**53 + 1)),
        ("volume", Decimal(2**53 + 1)),
        ("volume", Decimal(2**53 + 3)),
    ],
)
def test_integral_decimal_values_must_round_trip_exactly_through_float64(
    field: str,
    value: Decimal,
) -> None:
    exact = 2**53
    frame = valid_frame()
    frame["open"] = [exact + 2] * len(frame)
    frame["high"] = [exact + 4] * len(frame)
    frame["low"] = [exact] * len(frame)
    frame["close"] = [exact + 2] * len(frame)
    frame[field] = pd.Series([value] * len(frame), dtype="object")

    with pytest.raises(DataValidationError, match=rf"float64.*field={field}"):
        canonicalize_ohlcv(frame, symbol="EURUSD", timeframe="H1")


@pytest.mark.parametrize(
    "value",
    [
        Decimal(100),
        Decimal(2**53),
        Decimal(2**53 + 2),
        np.longdouble(str(2**53)),
    ],
    ids=[
        "ordinary-decimal",
        "decimal-2**53",
        "decimal-exact-next",
        "numpy-longdouble",
    ],
)
def test_exact_integral_non_builtin_numeric_values_remain_exact(
    value: object,
) -> None:
    frame = valid_frame()
    frame["volume"] = pd.Series([value] * len(frame), dtype="object")

    result = canonicalize_ohlcv(frame, symbol="EURUSD", timeframe="H1")

    assert result.frame["volume"].tolist() == [float(value)] * len(frame)


def test_non_integral_decimal_behavior_is_unchanged() -> None:
    frame = valid_frame()
    frame["volume"] = pd.Series([Decimal("100.25")] * len(frame), dtype="object")

    result = canonicalize_ohlcv(frame, symbol="EURUSD", timeframe="H1")

    assert result.frame["volume"].tolist() == [100.25] * len(frame)


def test_negative_lossy_integral_decimal_fails_closed() -> None:
    frame = valid_frame()
    value = Decimal(-(2**53 + 1))
    frame["volume"] = pd.Series([value] * len(frame), dtype="object")

    with pytest.raises(DataValidationError, match=r"float64.*field=volume"):
        canonicalize_ohlcv(frame, symbol="EURUSD", timeframe="H1")


def test_distinct_lossy_integral_sources_cannot_share_a_canonical_identity() -> None:
    values = [Decimal(2**54 + 1), Decimal(2**54 + 2)]
    assert float(values[0]) == float(values[1])

    for value in values:
        frame = valid_frame()
        frame["volume"] = pd.Series([value] * len(frame), dtype="object")
        with pytest.raises(DataValidationError, match=r"float64.*field=volume"):
            canonicalize_ohlcv(frame, symbol="EURUSD", timeframe="H1")


@pytest.mark.parametrize("value", [2**53, 2**53 + 2])
def test_exactly_representable_large_integers_remain_exact(value: int) -> None:
    frame = valid_frame()
    frame["open"] = pd.Series([value] * len(frame), dtype="int64")
    frame["high"] = pd.Series([value + 2] * len(frame), dtype="int64")
    frame["low"] = pd.Series([value] * len(frame), dtype="int64")
    frame["close"] = pd.Series([value] * len(frame), dtype="int64")
    frame["volume"] = pd.Series([value] * len(frame), dtype="uint64")

    result = canonicalize_ohlcv(frame, symbol="EURUSD", timeframe="H1")

    for field in ["open", "low", "close", "volume"]:
        assert result.frame[field].tolist() == [float(value)] * len(frame)


@pytest.mark.parametrize(
    ("value", "dtype"),
    [(np.iinfo(np.int64).max, "int64"), (np.iinfo(np.uint64).max, "uint64")],
    ids=["int64-max", "uint64-max"],
)
def test_integer_endpoints_that_lose_float64_precision_are_rejected(
    value: int,
    dtype: str,
) -> None:
    frame = valid_frame()
    frame["volume"] = pd.Series([value] * len(frame), dtype=dtype)

    with pytest.raises(DataValidationError, match=r"float64.*field=volume"):
        canonicalize_ohlcv(frame, symbol="EURUSD", timeframe="H1")


def _float32_exact_canonical_frame() -> pd.DataFrame:
    frame = valid_frame()
    frame["close"] = frame["open"] + 0.25
    return canonicalize_ohlcv(
        frame, symbol="EURUSD", timeframe="H1"
    ).frame


def test_zero_and_negative_zero_volume_remain_valid_float32() -> None:
    canonical = _float32_exact_canonical_frame()
    canonical["volume"] = [0.0, -0.0, 0.0, -0.0, 0.0, -0.0]

    volume = float32_ohlcv_arrays(canonical)["volume"]

    assert np.equal(volume, 0.0).all()
    assert np.signbit(volume).tolist() == [False, True, False, True, False, True]


@pytest.mark.parametrize("field", ["open", "high", "low", "close", "volume"])
@pytest.mark.parametrize(
    ("value", "message"),
    [(1.0e-50, "non-zero"), (1.0e40, "finite")],
    ids=["underflow", "overflow"],
)
def test_float32_conversion_rejects_each_field_without_silent_change(
    field: str,
    value: float,
    message: str,
) -> None:
    canonical = _float32_exact_canonical_frame()
    canonical[field] = value

    with pytest.raises(
        DataValidationError,
        match=rf"{message}.*field={field}",
    ):
        float32_ohlcv_arrays(canonical)


@pytest.mark.parametrize("field", ["open", "high", "low", "close", "volume"])
def test_float32_conversion_rejects_each_lossy_finite_round_trip(
    field: str,
) -> None:
    canonical = _float32_exact_canonical_frame()
    canonical[field] = np.array(
        [16_777_216.0, 16_777_217.0, 16_777_218.0, 16_777_220.0, 16_777_222.0, 16_777_224.0],
        dtype=np.float64,
    )

    with pytest.raises(
        DataValidationError,
        match=rf"float32.*field={field}",
    ):
        float32_ohlcv_arrays(canonical)


@pytest.mark.parametrize("field", ["open", "high", "low", "close", "volume"])
def test_float32_conversion_accepts_exactly_representable_values(
    field: str,
) -> None:
    canonical = _float32_exact_canonical_frame()
    expected = np.array(
        [16_777_216.0, 16_777_218.0, 16_777_220.0, 16_777_222.0, 16_777_224.0, 16_777_226.0],
        dtype=np.float64,
    )
    canonical[field] = expected

    converted = float32_ohlcv_arrays(canonical)[field]

    np.testing.assert_array_equal(converted.astype(np.float64), expected)


def test_canonical_dataset_frame_cannot_be_mutated_out_of_identity() -> None:
    source = valid_frame()
    dataset = canonicalize_ohlcv(source, symbol="EURUSD", timeframe="H1")
    expected = dataset.frame.copy(deep=True)
    identity = dataset.identity

    source.loc[0, "close"] += 10.0
    exposed = dataset.frame
    exposed.loc[0, "close"] += 1.0
    column = dataset.frame["close"]
    column.iloc[1] += 1.0
    for array, index in (
        (dataset.frame["close"].values, 2),
        (dataset.frame["close"].to_numpy(copy=False), 3),
    ):
        try:
            array[index] += 1.0
        except ValueError as exc:
            assert "read-only" in str(exc)
    sliced = dataset.frame.iloc[:2]
    sliced.loc[sliced.index[0], "close"] += 1.0

    pd.testing.assert_frame_equal(dataset.frame, expected)
    assert dataset.identity == identity


def test_complex_ohlcv_is_rejected_before_float_coercion() -> None:
    frame = valid_frame()
    frame["open"] = frame["open"].astype("complex128") + 1j

    with warnings.catch_warnings():
        warnings.simplefilter("error", np.exceptions.ComplexWarning)
        with pytest.raises(DataValidationError, match=r"complex.*field=open"):
            canonicalize_ohlcv(frame, symbol="EURUSD", timeframe="H1")
