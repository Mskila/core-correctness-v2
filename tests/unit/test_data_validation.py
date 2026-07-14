from __future__ import annotations

from dataclasses import FrozenInstanceError
from decimal import Decimal
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

    with pytest.raises(DataValidationError, match="mixed numeric timestamp units"):
        canonicalize_ohlcv(frame, symbol="EURUSD", timeframe="H1")


def test_consistent_early_milliseconds_can_cross_inference_boundary() -> None:
    frame = valid_frame().iloc[:3].copy()
    milliseconds = [
        99_999_999_000,
        100_003_599_000,
        100_007_199_000,
    ]
    frame["time"] = milliseconds

    result = canonicalize_ohlcv(frame, symbol="EURUSD", timeframe="H1")

    expected = pd.to_datetime(milliseconds, unit="ms", utc=True).astype(
        "datetime64[ns, UTC]"
    )
    assert result.frame["time"].tolist() == expected.tolist()
    assert result.gap_count == 0


def test_epoch_zero_is_unit_neutral_for_consistent_early_milliseconds() -> None:
    frame = valid_frame().iloc[:3].copy()
    milliseconds = [0, 99_999_999_000, 100_003_599_000]
    frame["time"] = milliseconds

    result = canonicalize_ohlcv(frame, symbol="EURUSD", timeframe="H1")

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
def test_numeric_time_unit_uses_h1_cadence_for_epoch_milliseconds(
    milliseconds: list[int],
) -> None:
    frame = valid_frame().iloc[:3].copy()
    frame["time"] = milliseconds

    result = canonicalize_ohlcv(frame, symbol="EURUSD", timeframe="H1")

    expected = pd.to_datetime(milliseconds, unit="ms", utc=True).astype(
        "datetime64[ns, UTC]"
    )
    expected_ns = sorted(expected.astype("int64").tolist())
    assert result.frame["time"].astype("int64").tolist() == expected_ns
    assert result.identity.start_time_ns == expected_ns[0]
    assert result.identity.end_time_ns == expected_ns[-1]
    assert result.gap_count == 0


def test_numeric_time_unit_preserves_epoch_seconds_with_h1_cadence() -> None:
    frame = valid_frame().iloc[:3].copy()
    seconds = [0, 3_600, 7_200]
    frame["time"] = seconds

    result = canonicalize_ohlcv(frame, symbol="EURUSD", timeframe="H1")

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

    with pytest.raises(DataValidationError, match="ambiguous numeric timestamp"):
        canonicalize_ohlcv(frame, symbol="EURUSD", timeframe="H1")


def test_consistent_milliseconds_with_long_gap_crossing_threshold_are_valid() -> None:
    frame = valid_frame().iloc[:3].copy()
    milliseconds = [1_000_000, 100_000_000_000, 100_003_600_000]
    frame["time"] = milliseconds

    result = canonicalize_ohlcv(frame, symbol="EURUSD", timeframe="H1")

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
    "timestamps",
    [
        [0.0, 3_600.5, 7_200.0],
        [1.0e18, 1.0e18 + 3_600_000_000_000, 1.0e18 + 7_200_000_000_000],
    ],
    ids=["fractional", "lossy-float-ns"],
)
def test_float_numeric_timestamps_must_be_lossless_integers(
    timestamps: list[float],
) -> None:
    frame = valid_frame().iloc[:3].copy()
    frame["time"] = timestamps

    with pytest.raises(DataValidationError, match=r"integer|lossless"):
        canonicalize_ohlcv(frame, symbol="EURUSD", timeframe="H1")


def test_out_of_range_integer_numeric_timestamps_are_rejected() -> None:
    frame = valid_frame().iloc[:3].copy()
    start = pd.Timestamp.max.value + 1
    frame["time"] = [start, start + 3_600_000_000_000, start + 7_200_000_000_000]

    with pytest.raises(DataValidationError, match="cannot be converted"):
        canonicalize_ohlcv(frame, symbol="EURUSD", timeframe="H1")


def test_extreme_in_range_ns_gap_does_not_overflow_spacing_check() -> None:
    frame = valid_frame().iloc[:2].copy()
    timestamps = [pd.Timestamp.min.value, pd.Timestamp.max.value]
    frame["time"] = timestamps

    result = canonicalize_ohlcv(frame, symbol="EURUSD", timeframe="H1")

    assert result.frame["time"].astype("int64").tolist() == timestamps
    assert result.identity.start_time_ns == timestamps[0]
    assert result.identity.end_time_ns == timestamps[-1]
    assert result.gap_count == 1


@pytest.mark.parametrize(
    ("unit", "timestamps", "expected_ns"),
    [
        ("s", [1_700_000_000, 1_700_003_600, 1_700_007_200], 1_000_000_000),
        (
            "ms",
            [1_700_000_000_000, 1_700_003_600_000, 1_700_007_200_000],
            1_000_000,
        ),
        (
            "us",
            [
                1_700_000_000_000_000,
                1_700_003_600_000_000,
                1_700_007_200_000_000,
            ],
            1_000,
        ),
        (
            "ns",
            [
                1_700_000_000_000_000_000,
                1_700_003_600_000_000_000,
                1_700_007_200_000_000_000,
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

    result = canonicalize_ohlcv(frame, symbol="EURUSD", timeframe="H1")

    expected = [value * expected_ns for value in timestamps]
    assert result.frame["time"].astype("int64").tolist() == expected
    assert result.identity.start_time_ns == expected[0]
    assert result.identity.end_time_ns == expected[-1]


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
    [Decimal("sNaN"), object(), "not-a-number"],
    ids=["signaling-decimal", "object", "text"],
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


def test_zero_and_negative_zero_volume_remain_valid_float32() -> None:
    frame = valid_frame()
    frame["volume"] = [0.0, -0.0, 0.0, -0.0, 0.0, -0.0]

    dataset = canonicalize_ohlcv(frame, symbol="EURUSD", timeframe="H1")
    volume = float32_ohlcv_arrays(dataset.frame)["volume"]

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
    canonical = canonicalize_ohlcv(
        valid_frame(), symbol="EURUSD", timeframe="H1"
    ).frame
    canonical[field] = value

    with pytest.raises(
        DataValidationError,
        match=rf"{message}.*field={field}",
    ):
        float32_ohlcv_arrays(canonical)


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
