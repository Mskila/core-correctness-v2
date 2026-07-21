from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pyarrow.lib import ArrowInvalid

import data_pipeline.kline_cache as kline_cache_module
from data_pipeline.fetcher import MT5DataFetcher
from data_pipeline.kline_cache import KlineCache
from data_pipeline.validation import canonicalize_ohlcv
from model_core.semantics import DataValidationError


RATE_DTYPE = np.dtype(
    [
        ("time", "<i8"),
        ("open", "<f8"),
        ("high", "<f8"),
        ("low", "<f8"),
        ("close", "<f8"),
        ("tick_volume", "<i8"),
    ]
)


class FakeMT5:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int, int, int]] = []
        self.forming_time = 1_700_010_800

    def copy_rates_from_pos(
        self, symbol: str, timeframe: int, start_pos: int, count: int
    ) -> np.ndarray:
        self.calls.append((symbol, timeframe, start_pos, count))
        rows = [
            (self.forming_time, 13.0, 14.0, 12.0, 13.5, 103),
            (self.forming_time - 10_800, 10.0, 11.0, 9.0, 10.5, 100),
            (self.forming_time - 7_200, 11.0, 12.0, 10.0, 11.5, 101),
            (self.forming_time - 3_600, 12.0, 13.0, 11.0, 12.5, 102),
        ]
        return np.array(rows[start_pos : start_pos + count], dtype=RATE_DTYPE)

    def last_error(self) -> tuple[int, str]:
        return (0, "OK")


def test_direct_fallback_uses_closed_bar_adapter_and_never_returns_position_zero(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeMT5()
    monkeypatch.setenv("KLINE_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr("data_pipeline.fetcher.mt5", fake)
    monkeypatch.setattr("data_pipeline.fetcher._MT5_AVAILABLE", True)
    monkeypatch.setattr(
        KlineCache,
        "get",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            kline_cache_module.CacheReadError("cache unavailable")
        ),
    )
    fetcher = MT5DataFetcher()
    fetcher._mt5_initialized = True

    actual = fetcher.fetch("EURUSD", 16385, 3)

    assert fake.calls == [("EURUSD", 16385, 1, 3)]
    actual_seconds = (actual["time"].astype("int64") // 1_000_000_000).tolist()
    assert fake.forming_time not in actual_seconds
    assert len(actual) == 3
    assert actual.columns.tolist() == ["time", "open", "high", "low", "close", "volume"]
    assert str(actual["time"].dtype) == "datetime64[ns, UTC]"


def test_cache_data_validation_error_is_never_swallowed_by_direct_fallback(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeMT5()
    monkeypatch.setenv("KLINE_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr("data_pipeline.fetcher.mt5", fake)
    monkeypatch.setattr("data_pipeline.fetcher._MT5_AVAILABLE", True)
    monkeypatch.setattr(
        KlineCache,
        "get",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            DataValidationError("invalid OHLC containment")
        ),
    )
    fetcher = MT5DataFetcher()
    fetcher._mt5_initialized = True

    with pytest.raises(DataValidationError, match="invalid OHLC"):
        fetcher.fetch("EURUSD", 16385, 3)

    assert fake.calls == []


def test_direct_fallback_canonicalizes_and_rejects_invalid_closed_rates(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeMT5()
    invalid = np.array(
        [
            (1_700_000_000, 10.0, 11.0, 9.0, 10.5, 100),
            (1_700_000_000, 11.0, 12.0, 10.0, 11.5, 101),
        ],
        dtype=RATE_DTYPE,
    )
    fake.copy_rates_from_pos = lambda symbol, timeframe, start_pos, count: (
        fake.calls.append((symbol, timeframe, start_pos, count)) or invalid
    )
    monkeypatch.setenv("KLINE_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr("data_pipeline.fetcher.mt5", fake)
    monkeypatch.setattr("data_pipeline.fetcher._MT5_AVAILABLE", True)
    monkeypatch.setattr(
        KlineCache,
        "get",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            kline_cache_module.CacheReadError("cache unavailable")
        ),
    )
    fetcher = MT5DataFetcher()
    fetcher._mt5_initialized = True

    with pytest.raises(DataValidationError, match="duplicate timestamp"):
        fetcher.fetch("EURUSD", 16385, 2)

    assert fake.calls == [("EURUSD", 16385, 1, 2)]


@pytest.mark.parametrize("error", [KeyError("programmer bug"), TypeError("bad type")])
def test_non_io_cache_errors_propagate_without_direct_fallback(
    tmp_path, monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    fake = FakeMT5()
    monkeypatch.setenv("KLINE_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr("data_pipeline.fetcher.mt5", fake)
    monkeypatch.setattr("data_pipeline.fetcher._MT5_AVAILABLE", True)
    monkeypatch.setattr(
        KlineCache,
        "get",
        lambda *args, **kwargs: (_ for _ in ()).throw(error),
    )
    fetcher = MT5DataFetcher()
    fetcher._mt5_initialized = True

    with pytest.raises(type(error), match=str(error)):
        fetcher.fetch("EURUSD", 16385, 3)

    assert fake.calls == []


class PayloadMT5:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = rows
        self.calls: list[tuple[str, int, int, int]] = []

    def copy_rates_from_pos(self, symbol, timeframe, start_pos, count):
        self.calls.append((symbol, timeframe, start_pos, count))
        return self.rows[start_pos : start_pos + count]

    def last_error(self):
        return (0, "OK")


def valid_payload_row(timestamp: object, price: float = 10.0) -> dict[str, object]:
    return {
        "time": timestamp,
        "open": price,
        "high": price + 1.0,
        "low": price - 1.0,
        "close": price + 0.5,
        "tick_volume": 100,
    }


@pytest.mark.parametrize(
    "closed_rows",
    [
        [
            {
                key: value
                for key, value in valid_payload_row(1_700_000_000).items()
                if key != "tick_volume"
            }
        ],
        [
            {
                **valid_payload_row(1_700_000_000),
                "high": 9.0,
            }
        ],
        [
            valid_payload_row(1_700_000_000),
            valid_payload_row(1_700_000_000, 11.0),
        ],
        [valid_payload_row("not-a-time")],
        [valid_payload_row(10**30)],
    ],
    ids=["missing-column", "invalid-ohlc", "duplicate-time", "invalid-time-type", "time-range"],
)
def test_every_malformed_direct_payload_raises_data_validation_error(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    closed_rows: list[dict[str, object]],
) -> None:
    forming = valid_payload_row(1_700_010_800, 20.0)
    fake = PayloadMT5([forming, *closed_rows])
    monkeypatch.setenv("KLINE_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr("data_pipeline.fetcher.mt5", fake)
    monkeypatch.setattr("data_pipeline.fetcher._MT5_AVAILABLE", True)
    monkeypatch.setattr(
        KlineCache,
        "get",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            kline_cache_module.CacheReadError("cache unavailable")
        ),
    )
    fetcher = MT5DataFetcher()
    fetcher._mt5_initialized = True

    with pytest.raises(DataValidationError):
        fetcher.fetch("EURUSD", 16385, len(closed_rows))

    assert fake.calls == [("EURUSD", 16385, 1, len(closed_rows))]


def test_empty_cache_result_uses_canonical_closed_direct_fallback(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeMT5()
    monkeypatch.setenv("KLINE_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr("data_pipeline.fetcher.mt5", fake)
    monkeypatch.setattr("data_pipeline.fetcher._MT5_AVAILABLE", True)
    monkeypatch.setattr(
        KlineCache,
        "get",
        lambda *args, **kwargs: pd.DataFrame(),
    )
    fetcher = MT5DataFetcher()
    fetcher._mt5_initialized = True

    actual = fetcher.fetch("EURUSD", 16385, 3)

    assert fake.calls == [("EURUSD", 16385, 1, 3)]
    assert actual.columns.tolist() == ["time", "open", "high", "low", "close", "volume"]


def test_direct_scalar_payload_is_reported_as_data_validation_error(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeMT5()
    fake.copy_rates_from_pos = lambda *args: 123
    monkeypatch.setenv("KLINE_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr("data_pipeline.fetcher.mt5", fake)
    monkeypatch.setattr("data_pipeline.fetcher._MT5_AVAILABLE", True)
    monkeypatch.setattr(
        KlineCache,
        "get",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            kline_cache_module.CacheReadError("cache unavailable")
        ),
    )
    fetcher = MT5DataFetcher()
    fetcher._mt5_initialized = True

    with pytest.raises(DataValidationError, match="malformed MT5 rates payload"):
        fetcher.fetch("EURUSD", 16385, 3)


def test_offline_fetcher_rejects_unversioned_numeric_legacy_cache_without_mutation(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = KlineCache(tmp_path, timeframe=16385)
    path = cache._cache_path("EURUSD")
    pd.DataFrame(
        [
            (1_700_000_000 + index * 3_600, 10.0 + index, 11.0 + index,
             9.0 + index, 10.5 + index, 100 + index)
            for index in range(4)
        ],
        columns=["time", "open", "high", "low", "close", "tick_volume"],
    ).to_parquet(path, index=False)
    before = path.read_bytes()
    monkeypatch.setattr("data_pipeline.kline_cache._default_cache_dir", lambda: tmp_path)

    with pytest.raises(DataValidationError, match=r"provenance|unit"):
        MT5DataFetcher(offline=True).fetch("EURUSD", 16385, 4)
    assert path.read_bytes() == before


def test_proven_cache_read_error_allows_closed_direct_fallback(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeMT5()
    monkeypatch.setenv("KLINE_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr("data_pipeline.fetcher.mt5", fake)
    monkeypatch.setattr("data_pipeline.fetcher._MT5_AVAILABLE", True)
    monkeypatch.setattr(
        KlineCache,
        "get",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            kline_cache_module.CacheReadError("read failed")
        ),
    )
    fetcher = MT5DataFetcher()
    fetcher._mt5_initialized = True

    actual = fetcher.fetch("EURUSD", 16385, 3)

    assert len(actual) == 3
    assert fake.calls == [("EURUSD", 16385, 1, 3)]


def test_actual_corrupt_parquet_falls_back_to_canonical_closed_rates(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = KlineCache(tmp_path, timeframe=16385)
    cache._cache_path("EURUSD").write_bytes(b"not parquet")
    fake = FakeMT5()
    monkeypatch.setattr("data_pipeline.kline_cache._default_cache_dir", lambda: tmp_path)
    monkeypatch.setattr("data_pipeline.fetcher.mt5", fake)
    monkeypatch.setattr("data_pipeline.fetcher._MT5_AVAILABLE", True)
    fetcher = MT5DataFetcher()
    fetcher._mt5_initialized = True

    actual = fetcher.fetch("EURUSD", 16385, 3)

    assert actual.columns.tolist() == ["time", "open", "high", "low", "close", "volume"]
    assert str(actual["time"].dtype) == "datetime64[ns, UTC]"
    assert len(actual) == 3
    assert fake.calls == [("EURUSD", 16385, 1, 3)]


@pytest.mark.parametrize("backend_error_type", [OSError, ArrowInvalid])
def test_backend_parquet_read_error_uses_closed_direct_fallback(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    backend_error_type: type[Exception],
) -> None:
    cache = KlineCache(tmp_path, timeframe=16385)
    cache._cache_path("EURUSD").write_bytes(b"placeholder")
    backend_error = backend_error_type("backend read failed")
    fake = FakeMT5()
    monkeypatch.setattr("data_pipeline.kline_cache._default_cache_dir", lambda: tmp_path)
    monkeypatch.setattr(
        "data_pipeline.kline_cache.pd.read_parquet",
        lambda *args, **kwargs: (_ for _ in ()).throw(backend_error),
    )
    monkeypatch.setattr("data_pipeline.fetcher.mt5", fake)
    monkeypatch.setattr("data_pipeline.fetcher._MT5_AVAILABLE", True)
    fetcher = MT5DataFetcher()
    fetcher._mt5_initialized = True

    actual = fetcher.fetch("EURUSD", 16385, 3)

    assert actual.columns.tolist() == ["time", "open", "high", "low", "close", "volume"]
    assert len(actual) == 3
    assert fake.calls == [("EURUSD", 16385, 1, 3)]


@pytest.mark.parametrize(
    "programming_error_type",
    [KeyError, TypeError, AssertionError, RuntimeError, ImportError],
)
def test_programming_or_configuration_read_error_never_falls_back_to_mt5(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    programming_error_type: type[Exception],
) -> None:
    cache = KlineCache(tmp_path, timeframe=16385)
    cache._cache_path("EURUSD").write_bytes(b"placeholder")
    programming_error = programming_error_type("programmer bug")
    fake = FakeMT5()
    monkeypatch.setattr("data_pipeline.kline_cache._default_cache_dir", lambda: tmp_path)
    monkeypatch.setattr(
        "data_pipeline.kline_cache.pd.read_parquet",
        lambda *args, **kwargs: (_ for _ in ()).throw(programming_error),
    )
    monkeypatch.setattr("data_pipeline.fetcher.mt5", fake)
    monkeypatch.setattr("data_pipeline.fetcher._MT5_AVAILABLE", True)
    fetcher = MT5DataFetcher()
    fetcher._mt5_initialized = True

    with pytest.raises(programming_error_type) as caught:
        fetcher.fetch("EURUSD", 16385, 3)

    assert caught.value is programming_error
    assert fake.calls == []


def test_malformed_readable_cache_does_not_fall_back_to_direct_mt5(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = KlineCache(tmp_path, timeframe=16385)
    path = cache._cache_path("EURUSD")
    invalid = pd.DataFrame(
        [(1_700_000_000, 10.0, 9.0, 11.0, 10.5, 100)],
        columns=["time", "open", "high", "low", "close", "tick_volume"],
    )
    invalid.to_parquet(path, index=False)
    fake = FakeMT5()
    monkeypatch.setattr("data_pipeline.kline_cache._default_cache_dir", lambda: tmp_path)
    monkeypatch.setattr("data_pipeline.fetcher.mt5", fake)
    monkeypatch.setattr("data_pipeline.fetcher._MT5_AVAILABLE", True)
    fetcher = MT5DataFetcher()
    fetcher._mt5_initialized = True

    with pytest.raises(DataValidationError):
        fetcher.fetch("EURUSD", 16385, 3)

    assert fake.calls == []


@pytest.mark.parametrize("failure_source", ["incremental", "atomic-write"])
def test_cache_update_or_write_oserror_propagates_without_direct_retry(
    tmp_path, monkeypatch: pytest.MonkeyPatch, failure_source: str
) -> None:
    cache = KlineCache(tmp_path, timeframe=16385)
    path = cache._cache_path("EURUSD")
    initial = canonicalize_ohlcv(
        pd.DataFrame(
            [
                (1_700_000_000 + index * 3_600, 10.0 + index, 11.0 + index,
                 9.0 + index, 10.5 + index, 100 + index)
                for index in range(6)
            ],
            columns=["time", "open", "high", "low", "close", "tick_volume"],
        ),
        symbol="EURUSD",
        timeframe=16385,
        numeric_time_unit="s",
    )
    cache._atomic_write("EURUSD", initial)
    metadata_path = cache._metadata_path("EURUSD")
    parquet_before = path.read_bytes()
    metadata_before = metadata_path.read_bytes()
    fake = FakeMT5()
    if failure_source == "atomic-write":
        revision_rates = np.array(
            [
                (1_700_025_200, 30.0, 31.0, 29.0, 30.5, 500),
                (1_700_018_000, 99.0, 100.0, 98.0, 99.5, 999),
                (1_700_021_600, 16.0, 17.0, 15.0, 16.5, 106),
            ],
            dtype=RATE_DTYPE,
        )

        def revised_copy(symbol, timeframe, start_pos, count):
            fake.calls.append((symbol, timeframe, start_pos, count))
            return revision_rates[start_pos : start_pos + count]

        fake.copy_rates_from_pos = revised_copy
    monkeypatch.setattr("data_pipeline.kline_cache._default_cache_dir", lambda: tmp_path)
    monkeypatch.setattr("data_pipeline.kline_cache.mt5", fake)
    monkeypatch.setattr("data_pipeline.kline_cache._MT5_AVAILABLE", True)
    monkeypatch.setattr("data_pipeline.fetcher.mt5", fake)
    monkeypatch.setattr("data_pipeline.fetcher._MT5_AVAILABLE", True)
    if failure_source == "incremental":
        monkeypatch.setattr(
            KlineCache,
            "_incremental_update",
            lambda *args, **kwargs: (_ for _ in ()).throw(OSError("incremental failed")),
        )
    else:
        monkeypatch.setattr(
            KlineCache,
            "_atomic_write",
            lambda *args, **kwargs: (_ for _ in ()).throw(OSError("atomic failed")),
        )
    fetcher = MT5DataFetcher()
    fetcher._mt5_initialized = True

    with pytest.raises(OSError, match="incremental failed|atomic failed"):
        fetcher.fetch("EURUSD", 16385, 3)

    expected_calls = 0 if failure_source == "incremental" else 1
    assert len(fake.calls) == expected_calls
    assert path.read_bytes() == parquet_before
    assert metadata_path.read_bytes() == metadata_before


@pytest.mark.parametrize("rates_value", [None, []])
def test_empty_direct_result_uses_canonical_public_schema(
    tmp_path, monkeypatch: pytest.MonkeyPatch, rates_value
) -> None:
    fake = FakeMT5()
    fake.copy_rates_from_pos = lambda *args: rates_value
    monkeypatch.setenv("KLINE_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr("data_pipeline.fetcher.mt5", fake)
    monkeypatch.setattr("data_pipeline.fetcher._MT5_AVAILABLE", True)
    monkeypatch.setattr(KlineCache, "get", lambda *args, **kwargs: None)
    fetcher = MT5DataFetcher()
    fetcher._mt5_initialized = True

    actual = fetcher.fetch("NOSYMBOL", 16385, 3)

    assert actual.columns.tolist() == ["time", "open", "high", "low", "close", "volume"]
    assert str(actual["time"].dtype) == "datetime64[ns, UTC]"
