from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

from data_pipeline.validation import (
    DatasetIdentity,
    assert_minimum_bars,
    canonicalize_ohlcv,
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
