from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow
import pytest
import torch

from data_pipeline.data_manager import MT5DataManager, compute_forward_open_returns
from data_pipeline.parquet_manager import ParquetDataManager
from data_pipeline.validation import (
    DatasetIdentity,
    canonicalize_ohlcv,
    float32_ohlcv_arrays,
)
from model_core.semantics import (
    DATA_CANONICALIZATION_VERSION,
    DATA_SCHEMA_VERSION,
    DataValidationError,
)


pytestmark = pytest.mark.core


def market_frame(*, time: object | None = None) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": time if time is not None else pd.date_range(
                "2026-01-01", periods=6, freq="1h", tz="UTC"
            ),
            "open": [1.1, 1.23456, 100.1, 100.09999, 100.10001, 99.95],
            "high": [1.2, 1.3, 100.2, 100.2, 100.2, 100.0],
            "low": [1.0, 1.2, 100.0, 100.0, 100.0, 99.9],
            "close": [1.15, 1.25, 100.11, 100.09, 100.12, 99.97],
            "volume": [100.1, 0.0, 102.25, 103.5, 104.75, 105.125],
        }
    )


def test_normal_market_decimals_quantize_to_finite_little_endian_float32() -> None:
    dataset = canonicalize_ohlcv(
        market_frame(), symbol="CORE01", timeframe="H1", gap_policy="segment"
    )
    arrays = float32_ohlcv_arrays(dataset.frame)

    assert all(values.dtype == np.dtype("float32") for values in arrays.values())
    assert all(values.astype("<f4", copy=False).dtype.byteorder in ("<", "=") for values in arrays.values())
    assert all(np.isfinite(values).all() for values in arrays.values())
    assert arrays["open"].tolist() == np.asarray(
        market_frame()["open"], dtype=np.float32
    ).tolist()
    returns, valid = compute_forward_open_returns(
        torch.from_numpy(arrays["open"][None, :].copy())
    )
    assert bool((returns[valid] > 0).any())
    assert bool((returns[valid] < 0).any())
    assert dataset.identity.schema_version == DATA_SCHEMA_VERSION
    assert dataset.identity.canonicalization_version == DATA_CANONICALIZATION_VERSION
    assert dataset.identity.time_unit == "ns"
    assert dataset.identity.gap_policy == "segment"


@pytest.mark.parametrize("value", [float(np.finfo(np.float32).max) * 2.0, np.inf, np.nan])
def test_nonfinite_or_float32_overflow_is_rejected(value: float) -> None:
    frame = market_frame()
    frame.loc[2, "volume"] = value
    with pytest.raises(DataValidationError, match=r"finite|float32"):
        canonicalize_ohlcv(frame, symbol="CORE01", timeframe="H1")


def test_fingerprint_uses_quantized_float32_not_source_float64_noise() -> None:
    left = market_frame()
    right = market_frame()
    baseline = np.float32(left.loc[2, "open"])
    right.loc[2, "open"] = np.nextafter(
        float(baseline), float("inf"), dtype=np.float64
    )
    assert left.loc[2, "open"] != right.loc[2, "open"]
    assert np.float32(left.loc[2, "open"]) == np.float32(right.loc[2, "open"])

    left_identity = canonicalize_ohlcv(
        left, symbol="CORE01", timeframe="H1"
    ).identity
    right_identity = canonicalize_ohlcv(
        right, symbol="CORE01", timeframe="H1"
    ).identity

    assert left_identity == right_identity


@pytest.mark.parametrize(
    ("unit", "scale"),
    [("s", 1), ("ms", 1_000), ("us", 1_000_000), ("ns", 1_000_000_000)],
)
def test_declared_numeric_units_canonicalize_to_same_ns_identity(
    unit: str, scale: int
) -> None:
    seconds = np.asarray([1_767_225_600 + 3_600 * i for i in range(6)], dtype=np.int64)
    frame = market_frame(time=(seconds * scale).tolist())
    dataset = canonicalize_ohlcv(
        frame,
        symbol="CORE01",
        timeframe="H1",
        numeric_time_unit=unit,
        gap_policy="segment",
    )
    reference = canonicalize_ohlcv(
        market_frame(time=seconds.tolist()),
        symbol="CORE01",
        timeframe="H1",
        numeric_time_unit="s",
        gap_policy="segment",
    )

    assert dataset.frame["time"].astype("int64").tolist() == (
        seconds * 1_000_000_000
    ).tolist()
    assert dataset.identity == reference.identity


def test_untagged_mixed_nonmonotonic_and_duplicate_times_fail_closed() -> None:
    mixed = market_frame(time=[1_700_000_000, 1_700_003_600_000, 1_700_007_200, 1_700_010_800, 1_700_014_400, 1_700_018_000])
    with pytest.raises(DataValidationError, match=r"provenance|unit"):
        canonicalize_ohlcv(mixed, symbol="CORE01", timeframe="H1")

    nonmonotonic = market_frame().iloc[[0, 2, 1, 3, 4, 5]].reset_index(drop=True)
    with pytest.raises(DataValidationError, match="strictly increasing"):
        canonicalize_ohlcv(nonmonotonic, symbol="CORE01", timeframe="H1")

    duplicate = market_frame()
    duplicate.loc[2, "time"] = duplicate.loc[1, "time"]
    with pytest.raises(DataValidationError, match="duplicate timestamp"):
        canonicalize_ohlcv(duplicate, symbol="CORE01", timeframe="H1")


def test_gap_policy_reject_segment_and_explicitly_allowed_are_distinct() -> None:
    frame = market_frame()
    frame.loc[3:, "time"] = frame.loc[3:, "time"] + pd.Timedelta(days=2)

    with pytest.raises(DataValidationError, match="gap policy reject"):
        canonicalize_ohlcv(
            frame, symbol="CORE01", timeframe="H1", gap_policy="reject"
        )
    segmented = canonicalize_ohlcv(
        frame, symbol="CORE01", timeframe="H1", gap_policy="segment"
    )
    allowed = canonicalize_ohlcv(
        frame,
        symbol="CORE01",
        timeframe="H1",
        gap_policy="explicitly-allowed",
    )

    assert segmented.gap_count == allowed.gap_count == 1
    assert segmented.segment_ids.tolist() == [0, 0, 0, 1, 1, 1]
    assert allowed.segment_ids.tolist() == [0, 0, 0, 0, 0, 0]
    assert segmented.identity.data_fingerprint != allowed.identity.data_fingerprint

    values, valid = compute_forward_open_returns(
        torch.from_numpy(frame["open"].to_numpy(dtype=np.float32)[None, :].copy()),
        torch.from_numpy(segmented.segment_ids[None, :].copy()),
    )
    assert valid.tolist() == [[True, False, True, True, False, False]]
    assert values[0, 1].item() == 0.0


def test_numeric_parquet_requires_unit_and_matches_mt5_identity_and_tensors(
    tmp_path: Path,
) -> None:
    seconds = [
        1_767_225_600 + 3_600 * offset for offset in (0, 1, 2, 51, 52, 53)
    ]
    frame = market_frame(time=seconds)
    path = tmp_path / "CORE01_H1.parquet"
    frame.to_parquet(path, index=False)

    with pytest.raises(DataValidationError, match=r"provenance|unit"):
        ParquetDataManager(path).load()
    parquet = ParquetDataManager(path, numeric_time_unit="s", gap_policy="segment")
    parquet.load()

    class Fetcher:
        @staticmethod
        def fetch(symbol: str, timeframe: object, bars: int) -> pd.DataFrame:
            return frame.rename(columns={"volume": "tick_volume"})

    mt5 = MT5DataManager(Fetcher(), gap_policy="segment")
    mt5.load(["CORE01"])

    assert parquet.data_identities == mt5.data_identities
    assert torch.equal(parquet.bar_time, mt5.bar_time)
    assert torch.equal(parquet.target_ret, mt5.target_ret)
    assert torch.equal(parquet.target_valid, mt5.target_valid)
    assert torch.equal(parquet.segment_ids, mt5.segment_ids)
    assert parquet.segment_ids.tolist() == [[0, 0, 0, 1, 1, 1]]
    for field in ("open", "high", "low", "close", "volume"):
        assert torch.equal(parquet.raw_dict[field], mt5.raw_dict[field])


def test_core_requirements_pin_a_parquet_engine() -> None:
    requirements = (Path(__file__).resolve().parents[2] / "requirements.txt").read_text(
        encoding="utf-8"
    )
    assert "pyarrow==25.0.0" in requirements
    assert pyarrow.__version__ == "25.0.0"


def test_pre_core01_dataset_identity_is_rejected_not_silently_migrated() -> None:
    legacy = {
        "schema_version": "ohlcv-v2",
        "symbol": "CORE01",
        "timeframe": "H1",
        "start_time_ns": 1,
        "end_time_ns": 2,
        "bars": 2,
        "data_fingerprint": "a" * 64,
        "time_fingerprint": "b" * 64,
    }

    with pytest.raises(DataValidationError, match=r"missing=.*canonicalization_version"):
        DatasetIdentity.from_dict(legacy)
