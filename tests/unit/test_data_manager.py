from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import torch

from data_pipeline.data_manager import MT5DataManager, compute_forward_open_returns
from data_pipeline.parquet_manager import ParquetDataManager, inspect_parquet_file
from data_pipeline.single_symbol_manager import SingleSymbolDataManager
from model_core.semantics import DataValidationError, DatasetAlignmentError


def _make_ohlcv_df(
    *,
    start: str = "2026-01-01 00:00:00",
    periods: int = 6,
    base_price: float = 100.0,
) -> pd.DataFrame:
    opens = base_price + np.arange(periods, dtype=np.float64)
    return pd.DataFrame(
        {
            "time": pd.date_range(start, periods=periods, freq="1h", tz="UTC"),
            "open": opens,
            "high": opens + 1.0,
            "low": opens - 1.0,
            "close": opens + 0.25,
            "tick_volume": 1_000.0 + np.arange(periods, dtype=np.float64),
        }
    )


def _make_mock_fetcher(return_map: dict[str, pd.DataFrame]) -> MagicMock:
    fetcher = MagicMock()
    fetcher.fetch.side_effect = lambda symbol, timeframe, count: return_map[symbol].copy()
    return fetcher


def _write_parquet(path: Path, frame: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return path


def test_forward_open_returns_use_t1_to_t2_and_mask_tail() -> None:
    opens = torch.tensor([[10.0, 11.0, 12.0, 15.0, 18.0]])

    returns, valid = compute_forward_open_returns(opens)

    expected = torch.tensor(
        [[
            torch.log(torch.tensor(12.0 / 11.0)),
            torch.log(torch.tensor(15.0 / 12.0)),
            torch.log(torch.tensor(18.0 / 15.0)),
            0.0,
            0.0,
        ]]
    )
    assert torch.allclose(returns, expected)
    assert valid.dtype == torch.bool
    assert valid.tolist() == [[True, True, True, False, False]]
    assert torch.equal(returns[:, -2:], torch.zeros((1, 2)))


@pytest.mark.parametrize(
    "opens",
    [
        torch.tensor([10.0, 11.0, 12.0]),
        torch.tensor([[10.0, 11.0]]),
        torch.tensor([[10.0, 0.0, 12.0]]),
        torch.tensor([[10.0, float("nan"), 12.0]]),
        torch.tensor([[10, 11, 12]]),
    ],
)
def test_forward_open_returns_reject_invalid_input(opens: torch.Tensor) -> None:
    with pytest.raises(DataValidationError):
        compute_forward_open_returns(opens)


def test_mt5_manager_uses_only_real_intersection_and_exposes_v2_shapes() -> None:
    frames = {
        "LEFT": _make_ohlcv_df(start="2026-01-01 00:00:00", base_price=100.0),
        "RIGHT": _make_ohlcv_df(start="2026-01-01 02:00:00", base_price=200.0),
    }
    manager = MT5DataManager(_make_mock_fetcher(frames))

    manager.load(["LEFT", "RIGHT"])

    expected_time = pd.date_range(
        "2026-01-01 02:00:00", periods=4, freq="1h", tz="UTC"
    ).astype("datetime64[ns, UTC]").astype("int64")
    assert manager.raw_dict["open"].shape == (2, 4)
    assert manager.raw_dict["open"][0].tolist() == [102.0, 103.0, 104.0, 105.0]
    assert manager.raw_dict["open"][1].tolist() == [200.0, 201.0, 202.0, 203.0]
    assert manager.bar_time.shape == manager.target_ret.shape == (2, 4)
    assert manager.bar_time.dtype == torch.int64
    assert manager.target_valid.dtype == torch.bool
    assert torch.equal(
        manager.bar_time,
        torch.tensor(np.tile(expected_time, (2, 1)), dtype=torch.int64),
    )
    assert manager.target_valid[:, :-2].all()
    assert not manager.target_valid[:, -2:].any()
    assert len(manager.data_identities) == 2
    assert [identity.symbol for identity in manager.data_identities] == manager.symbols
    assert all(identity.bars == 4 for identity in manager.data_identities)


def test_mt5_manager_rejects_insufficient_intersection_with_coverage() -> None:
    frames = {
        "LEFT": _make_ohlcv_df(
            start="2026-01-01 00:00:00", periods=4, base_price=100.0
        ),
        "RIGHT": _make_ohlcv_df(
            start="2026-01-01 02:00:00", periods=4, base_price=200.0
        ),
    }
    manager = MT5DataManager(_make_mock_fetcher(frames))

    with pytest.raises(DatasetAlignmentError) as exc_info:
        manager.load(["LEFT", "RIGHT"])

    message = str(exc_info.value)
    assert "intersection" in message
    assert "actual 2" in message
    assert "LEFT" in message and "bars=4" in message
    assert "RIGHT" in message and "2026-01-01" in message


def test_mt5_manager_canonicalizes_each_input_before_alignment() -> None:
    duplicate = pd.concat(
        [_make_ohlcv_df(periods=4), _make_ohlcv_df(periods=4).iloc[[0]]],
        ignore_index=True,
    )
    manager = MT5DataManager(_make_mock_fetcher({"EURUSD": duplicate}))

    with pytest.raises(DataValidationError, match="duplicate timestamp"):
        manager.load(["EURUSD"])


@pytest.mark.parametrize(
    "property_name",
    ["raw_dict", "feat_tensor", "target_ret", "target_valid", "bar_time", "data_identities"],
)
def test_mt5_manager_properties_fail_clearly_before_load(property_name: str) -> None:
    manager = MT5DataManager(MagicMock())

    with pytest.raises(RuntimeError, match=r"Data not loaded.*load"):
        getattr(manager, property_name)


def test_mt5_manager_fingerprints_are_stable_across_reload() -> None:
    frames = {"EURUSD": _make_ohlcv_df()}
    manager = MT5DataManager(_make_mock_fetcher(frames))

    manager.load(["EURUSD"])
    first = manager.data_identities
    manager.load(["EURUSD"])

    assert manager.data_identities == first


def test_parquet_manager_exposes_same_v2_contract(tmp_path: Path) -> None:
    path = _write_parquet(tmp_path / "EURUSD_H1.parquet", _make_ohlcv_df())
    manager = ParquetDataManager(path, required_bars=6)

    manager.load()

    assert manager.target_ret.shape == (1, 6)
    assert manager.target_valid.shape == (1, 6)
    assert manager.target_valid.dtype == torch.bool
    assert manager.target_valid.tolist() == [[True, True, True, True, False, False]]
    assert manager.bar_time.shape == (1, 6)
    assert manager.bar_time.dtype == torch.int64
    assert len(manager.data_identities) == 1
    assert manager.data_identities[0].bars == 6


def test_parquet_required_bars_is_checked_only_when_explicit(tmp_path: Path) -> None:
    path = _write_parquet(
        tmp_path / "EURUSD_H1.parquet", _make_ohlcv_df(periods=3)
    )

    permissive = ParquetDataManager(path)
    permissive.load()
    assert permissive.target_ret.shape == (1, 3)

    strict = ParquetDataManager(path, required_bars=4)
    with pytest.raises(DataValidationError, match=r"expected.*4.*actual.*3"):
        strict.load()


def test_parquet_validation_precedes_any_row_count_check(tmp_path: Path) -> None:
    frame = pd.concat(
        [_make_ohlcv_df(periods=6), _make_ohlcv_df(periods=6).iloc[[0]]],
        ignore_index=True,
    )
    path = _write_parquet(tmp_path / "EURUSD_H1.parquet", frame)

    with pytest.raises(DataValidationError, match="duplicate timestamp"):
        inspect_parquet_file(path)
    with pytest.raises(DataValidationError, match="duplicate timestamp"):
        ParquetDataManager(path, required_bars=7).load()


def test_parquet_inspection_uses_timestamp_span_not_h1_bar_constant(
    tmp_path: Path,
) -> None:
    frame = _make_ohlcv_df(periods=2)
    frame["time"] = pd.to_datetime(
        ["2025-01-01 00:00:00Z", "2026-01-01 00:00:00Z"], utc=True
    )
    path = _write_parquet(tmp_path / "EURUSD_H1.parquet", frame)

    info = inspect_parquet_file(path)

    assert info["bars"] == 2
    assert info["years_h1"] == pytest.approx(365 / 365.2425, abs=0.01)


def test_parquet_fingerprint_is_path_independent(tmp_path: Path) -> None:
    frame = _make_ohlcv_df()
    left_path = _write_parquet(tmp_path / "left" / "EURUSD_H1.parquet", frame)
    right_path = _write_parquet(tmp_path / "right" / "EURUSD_H1.parquet", frame)
    left = ParquetDataManager(left_path)
    right = ParquetDataManager(right_path)

    left.load()
    right.load()

    assert left.data_identities == right.data_identities


def test_single_symbol_manager_forwards_mask_time_and_identity(tmp_path: Path) -> None:
    path = _write_parquet(tmp_path / "EURUSD_H1.parquet", _make_ohlcv_df())
    multi = ParquetDataManager(path)
    multi.load()

    single = SingleSymbolDataManager(multi, "EURUSD")

    assert single.target_ret.shape == (1, 6)
    assert single.target_valid.shape == (1, 6)
    assert single.bar_time.shape == (1, 6)
    assert torch.equal(single.target_valid, multi.target_valid)
    assert torch.equal(single.bar_time, multi.bar_time)
    assert single.data_identity == multi.data_identities[0]
