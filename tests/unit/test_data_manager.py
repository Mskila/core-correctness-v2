from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import torch

from config import Config
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


def _make_mt5_seconds_df(offsets: list[int]) -> pd.DataFrame:
    frame = _make_ohlcv_df(periods=len(offsets))
    frame["time"] = np.array(
        [1_700_000_000 + 3_600 * offset for offset in offsets],
        dtype=np.int64,
    )
    return frame


def _make_mock_fetcher(return_map: dict[str, pd.DataFrame]) -> MagicMock:
    fetcher = MagicMock()
    fetcher.fetch.side_effect = lambda symbol, timeframe, count: return_map[symbol].copy()
    return fetcher


def _write_parquet(path: Path, frame: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return path


def _make_float32_round_trip_frame(*, lossy_open: bool) -> pd.DataFrame:
    opens = np.array(
        (
            [16_777_216.0, 16_777_217.0, 16_777_218.0, 16_777_220.0]
            if lossy_open
            else [16_777_216.0, 16_777_218.0, 16_777_220.0, 16_777_222.0]
        ),
        dtype=np.float64,
    )
    return pd.DataFrame(
        {
            "time": pd.date_range("2026-01-01", periods=4, freq="1h", tz="UTC"),
            "open": opens,
            "high": np.full(4, 16_777_224.0, dtype=np.float64),
            "low": np.full(4, 16_777_214.0, dtype=np.float64),
            "close": np.array(
                [16_777_216.0, 16_777_218.0, 16_777_220.0, 16_777_222.0],
                dtype=np.float64,
            ),
            "tick_volume": np.array([100.0, 102.0, 104.0, 106.0], dtype=np.float64),
        }
    )


def _assert_public_tensors_are_defensive_copies(manager) -> None:
    expected_raw = {key: value.clone() for key, value in manager.raw_dict.items()}
    expected_target = manager.target_ret.clone()
    expected_valid = manager.target_valid.clone()
    expected_time = manager.bar_time.clone()
    expected_features = manager.feat_tensor.clone()

    for value in manager.raw_dict.values():
        value.fill_(-123)
    manager.target_ret.fill_(123)
    manager.target_valid.logical_not_()
    manager.bar_time.fill_(0)
    manager.feat_tensor.fill_(123)

    for key, expected in expected_raw.items():
        assert torch.equal(manager.raw_dict[key], expected)
    assert torch.equal(manager.target_ret, expected_target)
    assert torch.equal(manager.target_valid, expected_valid)
    assert torch.equal(manager.bar_time, expected_time)
    torch.testing.assert_close(
        manager.feat_tensor,
        expected_features,
        equal_nan=True,
    )


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


def test_forward_open_returns_are_stable_and_differentiable_for_extreme_ratio() -> None:
    opens = torch.tensor(
        [[1.0, 1.0e-30, 1.0e10]],
        dtype=torch.float32,
        requires_grad=True,
    )

    returns, valid = compute_forward_open_returns(opens)

    expected = math.log(1.0e10) - math.log(1.0e-30)
    assert returns.dtype == opens.dtype
    assert returns.device == opens.device
    assert returns[0, 0].item() == pytest.approx(expected, rel=1.0e-6)
    assert torch.isfinite(returns[valid]).all()
    returns[valid].sum().backward()
    assert opens.grad is not None
    assert torch.isfinite(opens.grad).all()


def test_forward_open_returns_preserve_adjacent_large_float32_prices() -> None:
    opens = torch.tensor(
        [[1.0, 16_777_215.0, 16_777_216.0]],
        dtype=torch.float32,
        requires_grad=True,
    )

    returns, valid = compute_forward_open_returns(opens)

    expected = math.log(16_777_216.0 / 16_777_215.0)
    assert returns[0, 0].item() == pytest.approx(expected, rel=1.0e-6)
    returns[valid].sum().backward()
    assert opens.grad is not None
    assert torch.isfinite(opens.grad).all()


def test_forward_open_returns_never_expose_non_finite_local_gradients() -> None:
    smallest = torch.nextafter(
        torch.tensor(0.0, dtype=torch.float32),
        torch.tensor(1.0, dtype=torch.float32),
    ).item()
    opens = torch.tensor(
        [[1.0, smallest, 1.0]],
        dtype=torch.float32,
        requires_grad=True,
    )

    try:
        returns, valid = compute_forward_open_returns(opens)
    except DataValidationError as exc:
        assert "gradient" in str(exc)
        return

    returns[valid].sum().backward()
    assert opens.grad is not None
    assert torch.isfinite(opens.grad).all()


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


@pytest.mark.parametrize(
    "offsets",
    [[0, 1, 2, 3, 4, 5], [0, 1, 3, 4, 8, 11]],
    ids=["continuous", "legal-gaps"],
)
def test_mt5_manager_loads_declared_unix_seconds(offsets: list[int]) -> None:
    frame = _make_mt5_seconds_df(offsets)
    manager = MT5DataManager(_make_mock_fetcher({"EURUSD": frame}))

    manager.load(["EURUSD"])

    expected_ns = torch.tensor(
        [[(1_700_000_000 + 3_600 * offset) * 1_000_000_000 for offset in offsets]],
        dtype=torch.int64,
    )
    assert torch.equal(manager.bar_time, expected_ns)
    assert manager.raw_dict["open"].shape == (1, len(offsets))
    assert manager.target_valid[:, -2:].logical_not().all()


def test_parquet_inspect_and_load_use_declared_unix_seconds(tmp_path: Path) -> None:
    frame = _make_mt5_seconds_df([0, 1, 3, 4, 8, 11])
    path = _write_parquet(tmp_path / "EURUSD_H1.parquet", frame)

    info = inspect_parquet_file(path)
    manager = ParquetDataManager(path)
    manager.load()

    expected_ns = torch.tensor(
        [[
            (1_700_000_000 + 3_600 * offset) * 1_000_000_000
            for offset in [0, 1, 3, 4, 8, 11]
        ]],
        dtype=torch.int64,
    )
    assert info["valid"] is True
    assert info["bars"] == 6
    assert torch.equal(manager.bar_time, expected_ns)


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


def test_mt5_load_failure_after_tensor_conversion_leaves_manager_unloaded() -> None:
    frame = _make_ohlcv_df(periods=4)
    frame["open"] = 1.0e40
    frame["high"] = 1.1e40
    frame["low"] = 0.9e40
    frame["close"] = 1.05e40
    manager = MT5DataManager(_make_mock_fetcher({"BIG": frame}))

    with pytest.raises(DataValidationError, match="finite|float32"):
        manager.load(["BIG"])

    assert manager.symbols == []
    for property_name in (
        "raw_dict",
        "feat_tensor",
        "target_ret",
        "target_valid",
        "bar_time",
        "data_identities",
    ):
        with pytest.raises(RuntimeError, match=r"Data not loaded.*load"):
            getattr(manager, property_name)


def test_mt5_empty_load_invalidates_previously_loaded_state() -> None:
    manager = MT5DataManager(
        _make_mock_fetcher({"OK": _make_ohlcv_df(periods=4)})
    )
    manager.load(["OK"])

    with pytest.raises(DataValidationError, match="at least one symbol"):
        manager.load([])

    assert manager.symbols == []
    for property_name in (
        "raw_dict",
        "feat_tensor",
        "target_ret",
        "target_valid",
        "bar_time",
        "data_identities",
    ):
        with pytest.raises(RuntimeError, match=r"Data not loaded.*load"):
            getattr(manager, property_name)

    manager.reload()
    assert manager.symbols == ["OK"]


def test_mt5_duplicate_symbols_fail_before_fetch_and_clear_loaded_state() -> None:
    fetcher = _make_mock_fetcher(
        {
            "OK": _make_ohlcv_df(periods=4),
            "DUP": _make_ohlcv_df(periods=4),
        }
    )
    manager = MT5DataManager(fetcher)
    manager.load(["OK"])
    fetcher.fetch.reset_mock()

    with pytest.raises(DataValidationError, match="duplicate symbols"):
        manager.load(["DUP", "DUP"])

    fetcher.fetch.assert_not_called()
    assert manager.symbols == []
    for property_name in (
        "raw_dict",
        "feat_tensor",
        "target_ret",
        "target_valid",
        "bar_time",
        "data_identities",
    ):
        with pytest.raises(RuntimeError, match=r"Data not loaded.*load"):
            getattr(manager, property_name)
    manager.reload()
    assert manager.symbols == ["OK"]


def test_mt5_default_reload_resolves_current_config_symbols(monkeypatch) -> None:
    fetcher = _make_mock_fetcher(
        {
            "A": _make_ohlcv_df(periods=4, base_price=100.0),
            "B": _make_ohlcv_df(periods=4, base_price=200.0),
        }
    )
    manager = MT5DataManager(fetcher)
    monkeypatch.setattr(Config, "SYMBOLS", ["A"])
    manager.load()

    monkeypatch.setattr(Config, "SYMBOLS", ["B"])
    fetcher.fetch.reset_mock()
    manager.reload()

    assert manager.symbols == ["B"]
    assert manager.raw_dict["open"][0, 0].item() == 200.0
    assert fetcher.fetch.call_args.args[0] == "B"


def test_mt5_explicit_reload_ignores_later_config_changes(monkeypatch) -> None:
    fetcher = _make_mock_fetcher(
        {
            "A": _make_ohlcv_df(periods=4, base_price=100.0),
            "B": _make_ohlcv_df(periods=4, base_price=200.0),
        }
    )
    manager = MT5DataManager(fetcher)
    monkeypatch.setattr(Config, "SYMBOLS", ["B"])
    manager.load(["A"])

    fetcher.fetch.reset_mock()
    manager.reload()

    assert manager.symbols == ["A"]
    assert fetcher.fetch.call_args.args[0] == "A"


def test_mt5_failed_explicit_selection_is_not_committed(monkeypatch) -> None:
    invalid = pd.concat(
        [_make_ohlcv_df(periods=4), _make_ohlcv_df(periods=4).iloc[[0]]],
        ignore_index=True,
    )
    frames = {
        "DEFAULT": _make_ohlcv_df(periods=4, base_price=300.0),
        "GOOD": _make_ohlcv_df(periods=4, base_price=100.0),
        "BAD": invalid,
    }
    monkeypatch.setattr(Config, "SYMBOLS", ["DEFAULT"])
    manager = MT5DataManager(_make_mock_fetcher(frames))
    manager.load(["GOOD"])

    with pytest.raises(DataValidationError, match="duplicate timestamp"):
        manager.load(["BAD"])
    assert manager.symbols == []

    frames["BAD"] = _make_ohlcv_df(periods=4, base_price=200.0)
    manager.reload()
    assert manager.symbols == ["GOOD"]

    first_failure = MT5DataManager(_make_mock_fetcher(frames))
    frames["BAD"] = invalid
    with pytest.raises(DataValidationError, match="duplicate timestamp"):
        first_failure.load(["BAD"])
    first_failure.reload()
    assert first_failure.symbols == ["DEFAULT"]


def test_mt5_failed_default_selection_does_not_replace_explicit_request(
    monkeypatch,
) -> None:
    invalid = pd.concat(
        [_make_ohlcv_df(periods=4), _make_ohlcv_df(periods=4).iloc[[0]]],
        ignore_index=True,
    )
    frames = {
        "GOOD": _make_ohlcv_df(periods=4, base_price=100.0),
        "BAD": invalid,
        "DEFAULT": _make_ohlcv_df(periods=4, base_price=300.0),
    }
    manager = MT5DataManager(_make_mock_fetcher(frames))
    manager.load(["GOOD"])
    monkeypatch.setattr(Config, "SYMBOLS", ["BAD"])

    with pytest.raises(DataValidationError, match="duplicate timestamp"):
        manager.load()
    assert manager.symbols == []

    monkeypatch.setattr(Config, "SYMBOLS", ["DEFAULT"])
    manager.reload()
    assert manager.symbols == ["GOOD"]


def test_mt5_boolean_data_failure_clears_state_without_replacing_request() -> None:
    bad = _make_ohlcv_df(periods=4)
    bad["tick_volume"] = True
    frames = {
        "GOOD": _make_ohlcv_df(periods=4),
        "BAD": bad,
    }
    manager = MT5DataManager(_make_mock_fetcher(frames))
    manager.load(["GOOD"])

    with pytest.raises(DataValidationError, match=r"boolean.*field=volume"):
        manager.load(["BAD"])
    assert manager.symbols == []

    manager.reload()
    assert manager.symbols == ["GOOD"]


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


def test_mt5_public_tensors_are_defensive_copies() -> None:
    manager = MT5DataManager(
        _make_mock_fetcher({"EURUSD": _make_ohlcv_df(periods=64)})
    )
    manager.load(["EURUSD"])

    _assert_public_tensors_are_defensive_copies(manager)


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


def test_parquet_failed_reload_invalidates_previously_loaded_state(
    tmp_path: Path,
) -> None:
    path = _write_parquet(tmp_path / "EURUSD_H1.parquet", _make_ohlcv_df())
    manager = ParquetDataManager(path)
    manager.load()
    duplicate = pd.concat(
        [_make_ohlcv_df(), _make_ohlcv_df().iloc[[0]]],
        ignore_index=True,
    )
    _write_parquet(path, duplicate)

    with pytest.raises(DataValidationError, match="duplicate timestamp"):
        manager.load()

    for property_name in (
        "raw_dict",
        "feat_tensor",
        "target_ret",
        "target_valid",
        "bar_time",
        "data_identities",
    ):
        with pytest.raises(RuntimeError, match=r"Data not loaded.*load"):
            getattr(manager, property_name)


def test_parquet_float32_overflow_fails_closed_after_reload(
    tmp_path: Path,
) -> None:
    path = _write_parquet(
        tmp_path / "EURUSD_H1.parquet",
        _make_ohlcv_df(periods=4),
    )
    manager = ParquetDataManager(path)
    manager.load()
    overflow = _make_ohlcv_df(periods=4)
    overflow["tick_volume"] = np.full(4, 1.0e40, dtype=np.float64)
    _write_parquet(path, overflow)

    with pytest.raises(DataValidationError, match=r"field=volume"):
        inspect_parquet_file(path)
    with pytest.raises(DataValidationError, match=r"field=volume"):
        manager.load()

    for property_name in (
        "raw_dict",
        "feat_tensor",
        "target_ret",
        "target_valid",
        "bar_time",
        "data_identities",
    ):
        with pytest.raises(RuntimeError, match=r"Data not loaded.*load"):
            getattr(manager, property_name)


def test_float32_underflow_is_rejected_by_inspect_and_managers(
    tmp_path: Path,
) -> None:
    underflow = _make_ohlcv_df(periods=4)
    for field in ("open", "high", "low", "close"):
        underflow[field] = np.full(4, 1.0e-50, dtype=np.float64)
    path = _write_parquet(tmp_path / "EURUSD_H1.parquet", underflow)

    with pytest.raises(DataValidationError, match=r"float32.*field=open"):
        inspect_parquet_file(path)
    with pytest.raises(DataValidationError, match=r"float32.*field=open"):
        ParquetDataManager(path).load()

    mt5 = MT5DataManager(_make_mock_fetcher({"EURUSD": underflow}))
    with pytest.raises(DataValidationError, match=r"float32.*field=open"):
        mt5.load(["EURUSD"])


def test_nonzero_volume_float32_underflow_is_rejected_everywhere(
    tmp_path: Path,
) -> None:
    underflow = _make_ohlcv_df(periods=4)
    underflow["tick_volume"] = np.full(4, 1.0e-50, dtype=np.float64)
    path = _write_parquet(tmp_path / "EURUSD_H1.parquet", underflow)

    with pytest.raises(DataValidationError, match=r"float32.*field=volume"):
        inspect_parquet_file(path)
    with pytest.raises(DataValidationError, match=r"float32.*field=volume"):
        ParquetDataManager(path).load()

    mt5 = MT5DataManager(_make_mock_fetcher({"EURUSD": underflow}))
    with pytest.raises(DataValidationError, match=r"float32.*field=volume"):
        mt5.load(["EURUSD"])


def test_lossy_finite_float32_inputs_fail_closed_in_consumer_loads(
    tmp_path: Path,
) -> None:
    frame = _make_float32_round_trip_frame(lossy_open=True)
    path = _write_parquet(tmp_path / "LOSSY_H1.parquet", frame)
    parquet = ParquetDataManager(path)
    mt5 = MT5DataManager(_make_mock_fetcher({"LOSSY": frame}))

    with pytest.raises(DataValidationError, match=r"float32.*field=open"):
        parquet.load()
    with pytest.raises(DataValidationError, match=r"float32.*field=open"):
        mt5.load(["LOSSY"])

    for manager in (parquet, mt5):
        for property_name in (
            "raw_dict",
            "feat_tensor",
            "target_ret",
            "target_valid",
            "bar_time",
            "data_identities",
        ):
            with pytest.raises(RuntimeError, match=r"Data not loaded.*load"):
                getattr(manager, property_name)


def test_exact_float32_inputs_reach_consumer_label_paths(tmp_path: Path) -> None:
    frame = _make_float32_round_trip_frame(lossy_open=False)
    path = _write_parquet(tmp_path / "EXACT_H1.parquet", frame)
    parquet = ParquetDataManager(path)
    mt5 = MT5DataManager(_make_mock_fetcher({"EXACT": frame}))

    parquet.load()
    mt5.load(["EXACT"])

    expected_open = torch.tensor(
        [[16_777_216.0, 16_777_218.0, 16_777_220.0, 16_777_222.0]],
        dtype=torch.float32,
    )
    for manager in (parquet, mt5):
        assert torch.equal(manager.raw_dict["open"], expected_open)
        assert manager.target_ret.shape == (1, 4)
        assert manager.target_valid.tolist() == [[True, True, False, False]]
        assert torch.isfinite(manager.target_ret[manager.target_valid]).all()


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


def test_parquet_public_tensors_are_defensive_copies(tmp_path: Path) -> None:
    path = _write_parquet(
        tmp_path / "EURUSD_H1.parquet", _make_ohlcv_df(periods=64)
    )
    manager = ParquetDataManager(path)
    manager.load()

    _assert_public_tensors_are_defensive_copies(manager)


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


def test_single_symbol_public_tensors_are_defensive_copies(tmp_path: Path) -> None:
    path = _write_parquet(
        tmp_path / "EURUSD_H1.parquet", _make_ohlcv_df(periods=64)
    )
    multi = ParquetDataManager(path)
    multi.load()
    single = SingleSymbolDataManager(multi, "EURUSD")

    _assert_public_tensors_are_defensive_copies(single)


def test_single_symbol_manager_rebinds_by_symbol_after_reorder_and_shrink() -> None:
    frames = {
        "LEFT": _make_ohlcv_df(periods=4, base_price=100.0),
        "RIGHT": _make_ohlcv_df(periods=4, base_price=200.0),
    }
    multi = MT5DataManager(_make_mock_fetcher(frames))
    multi.load(["LEFT", "RIGHT"])
    single = SingleSymbolDataManager(multi, "RIGHT")
    assert single.raw_dict["open"][0, 0].item() == 200.0
    assert single.data_identity.symbol == "RIGHT"

    multi.load(["RIGHT", "LEFT"])
    assert single.raw_dict["open"][0, 0].item() == 200.0
    assert torch.equal(single.target_ret, multi.target_ret[0:1])
    assert torch.equal(single.target_valid, multi.target_valid[0:1])
    assert torch.equal(single.bar_time, multi.bar_time[0:1])
    assert single.data_identity == multi.data_identities[0]

    multi.load(["RIGHT"])
    assert single.raw_dict["open"].shape == (1, 4)
    assert single.raw_dict["open"][0, 0].item() == 200.0
    assert single.data_identity.symbol == "RIGHT"


@pytest.mark.parametrize(
    "property_name",
    [
        "raw_dict",
        "feat_tensor",
        "target_ret",
        "target_valid",
        "bar_time",
        "data_identity",
    ],
)
def test_single_symbol_manager_rejects_access_after_symbol_is_removed(
    property_name: str,
) -> None:
    frames = {
        "LEFT": _make_ohlcv_df(periods=4, base_price=100.0),
        "RIGHT": _make_ohlcv_df(periods=4, base_price=200.0),
    }
    multi = MT5DataManager(_make_mock_fetcher(frames))
    multi.load(["LEFT", "RIGHT"])
    single = SingleSymbolDataManager(multi, "RIGHT")
    multi.load(["LEFT"])

    with pytest.raises(DataValidationError, match=r"RIGHT.*not available"):
        getattr(single, property_name)
