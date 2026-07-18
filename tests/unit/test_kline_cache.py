from __future__ import annotations

import builtins
import errno
import inspect
import importlib.util
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import shutil
import threading
import time
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from pyarrow.lib import ArrowInvalid

import data_pipeline.kline_cache as kline_cache_module
from data_pipeline.kline_cache import CacheRecoveryError as ExpectedCacheRecoveryError
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


def rates(rows: list[tuple[int, float, float, float, float, int]]) -> np.ndarray:
    return np.array(rows, dtype=RATE_DTYPE)


def valid_rows(start: int, count: int) -> list[tuple[int, float, float, float, float, int]]:
    return [
        (start + index * 3_600, price, price + 1.0, price - 1.0, price + 0.5, 100 + index)
        for index in range(count)
        for price in [10.0 + index]
    ]


class FakeMT5:
    def __init__(self, all_rates: np.ndarray) -> None:
        self.all_rates = all_rates
        self.calls: list[tuple[str, int, int, int]] = []

    def copy_rates_from_pos(
        self, symbol: str, timeframe: int, start_pos: int, count: int
    ) -> np.ndarray:
        self.calls.append((symbol, timeframe, start_pos, count))
        return self.all_rates[start_pos : start_pos + count]


def raw_frame(rows: list[tuple[int, float, float, float, float, int]]) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=["time", "open", "high", "low", "close", "tick_volume"],
    )


def epoch_seconds(frame: pd.DataFrame) -> list[int]:
    return (frame["time"].astype("int64") // 1_000_000_000).tolist()


def run_worker_capturing_baseexceptions(
    worker,
    errors: queue.Queue[BaseException],
) -> None:
    try:
        worker()
    except BaseException as exc:
        errors.put(exc)


def join_worker_without_errors(
    worker: threading.Thread,
    errors: queue.Queue[BaseException],
) -> None:
    worker.join(10)
    assert not worker.is_alive()
    assert errors.empty(), list(errors.queue)


def test_full_download_requests_only_closed_bars_and_writes_identity_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    forming_time = 1_700_000_000
    closed = rates(valid_rows(forming_time - 3 * 3_600, 3))
    forming = rates([(forming_time, 20.0, 21.0, 19.0, 20.5, 999)])
    fake = FakeMT5(np.concatenate([forming, closed]))
    monkeypatch.setattr("data_pipeline.kline_cache.mt5", fake)
    monkeypatch.setattr("data_pipeline.kline_cache._MT5_AVAILABLE", True)
    cache = KlineCache(tmp_path, timeframe=16385, bars_count=3)

    actual = cache._full_download("EURUSD")

    assert actual is not None
    assert fake.calls == [("EURUSD", 16385, 1, 3)]
    assert forming_time not in epoch_seconds(actual)
    metadata = json.loads(cache._metadata_path("EURUSD").read_text(encoding="utf-8"))
    assert metadata["data_fingerprint"]
    assert metadata["start_time_ns"] == int(actual["time"].iloc[0].value)
    assert metadata["end_time_ns"] == int(actual["time"].iloc[-1].value)
    assert metadata["bars"] == 3
    assert metadata["schema_version"] == "ohlcv-v2"
    assert metadata["gap_count"] == 0


def test_incremental_remote_tail_revises_matching_timestamp_and_adds_closed_bar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    start = 1_700_000_000
    local_rows = valid_rows(start, 6)
    local = raw_frame(local_rows)
    revised_tail = valid_rows(start + 3 * 3_600, 4)
    revised_tail[1] = (
        revised_tail[1][0],
        99.0,
        100.0,
        98.0,
        99.5,
        999,
    )
    forming = rates([(start + 7 * 3_600, 30.0, 31.0, 29.0, 30.5, 500)])
    fake = FakeMT5(np.concatenate([forming, rates(revised_tail)]))
    monkeypatch.setattr("data_pipeline.kline_cache.mt5", fake)
    monkeypatch.setattr("data_pipeline.kline_cache._MT5_AVAILABLE", True)
    monkeypatch.setattr("data_pipeline.kline_cache.Config.EXECUTION_LAG_BARS", 1)
    cache = KlineCache(tmp_path, timeframe=16385, bars_count=20)

    actual = cache._incremental_update("EURUSD", local)

    assert fake.calls == [("EURUSD", 16385, 1, 5)]
    seconds = epoch_seconds(actual)
    assert seconds == sorted(set(seconds))
    revised_time = revised_tail[1][0]
    revised = actual.loc[np.array(seconds) == revised_time].iloc[0]
    assert revised["open"] == pytest.approx(99.0)
    assert revised["close"] == pytest.approx(99.5)
    assert seconds[-1] == start + 6 * 3_600


@pytest.mark.parametrize("invalid_kind", ["ohlc", "duplicate"])
def test_invalid_remote_tail_leaves_existing_cache_bytes_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invalid_kind: str,
) -> None:
    start = 1_700_000_000
    local = raw_frame(valid_rows(start, 6))
    cache = KlineCache(tmp_path, timeframe=16385, bars_count=20)
    path = cache._cache_path("EURUSD")
    local_dataset = canonicalize_ohlcv(
        local, symbol="EURUSD", timeframe="H1", numeric_time_unit="s"
    )
    cache._atomic_write("EURUSD", local_dataset)
    metadata_path = cache._metadata_path("EURUSD")
    parquet_before = path.read_bytes()
    metadata_before = metadata_path.read_bytes()

    remote_rows = valid_rows(start + 4 * 3_600, 2)
    if invalid_kind == "ohlc":
        remote_rows[0] = (remote_rows[0][0], 20.0, 19.0, 18.0, 20.5, 100)
    else:
        remote_rows[1] = (remote_rows[0][0], 21.0, 22.0, 20.0, 21.5, 101)
    forming = rates([(start + 7 * 3_600, 30.0, 31.0, 29.0, 30.5, 500)])
    fake = FakeMT5(np.concatenate([forming, rates(remote_rows)]))
    monkeypatch.setattr("data_pipeline.kline_cache.mt5", fake)
    monkeypatch.setattr("data_pipeline.kline_cache._MT5_AVAILABLE", True)

    with pytest.raises(DataValidationError):
        cache._incremental_update("EURUSD", local)

    assert path.read_bytes() == parquet_before
    assert metadata_path.read_bytes() == metadata_before


@pytest.mark.parametrize(
    ("local_indexes", "remote_indexes", "expected_indexes"),
    [
        (list(range(10)), [5, 6, 7], list(range(10))),
        (list(range(3, 10)), [0, 1, 2], list(range(10))),
        (list(range(10)), [7, 8, 9], list(range(10))),
        (list(range(10)), [5, 7], list(range(10))),
        (list(range(10)), [8, 9, 10], list(range(11))),
    ],
)
def test_incremental_update_is_timestamp_upsert_without_partial_tail_loss(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    local_indexes: list[int],
    remote_indexes: list[int],
    expected_indexes: list[int],
) -> None:
    start = 1_700_000_000

    def row(index: int, *, revised: bool) -> tuple[int, float, float, float, float, int]:
        price = (100.0 if revised else 10.0) + index
        return (
            start + index * 3_600,
            price,
            price + 1.0,
            price - 1.0,
            price + 0.5,
            (900 if revised else 100) + index,
        )

    local = raw_frame([row(index, revised=False) for index in local_indexes])
    remote_rows = [row(index, revised=True) for index in remote_indexes]
    forming = rates([(start + 20 * 3_600, 500.0, 501.0, 499.0, 500.5, 1)])
    fake = FakeMT5(np.concatenate([forming, rates(remote_rows)]))
    monkeypatch.setattr("data_pipeline.kline_cache.mt5", fake)
    monkeypatch.setattr("data_pipeline.kline_cache._MT5_AVAILABLE", True)
    monkeypatch.setattr("data_pipeline.kline_cache.Config.EXECUTION_LAG_BARS", 1)
    cache = KlineCache(tmp_path, timeframe=16385, bars_count=20)

    actual = cache._incremental_update("EURUSD", local)

    assert fake.calls == [("EURUSD", 16385, 1, 5)]
    assert epoch_seconds(actual) == [start + index * 3_600 for index in expected_indexes]
    for index in remote_indexes:
        revised = actual.loc[
            actual["time"] == pd.Timestamp(start + index * 3_600, unit="s", tz="UTC")
        ].iloc[0]
        assert revised["open"] == pytest.approx(100.0 + index)
    for index in set(local_indexes) - set(remote_indexes):
        retained = actual.loc[
            actual["time"] == pd.Timestamp(start + index * 3_600, unit="s", tz="UTC")
        ].iloc[0]
        assert retained["open"] == pytest.approx(10.0 + index)


def test_configless_import_uses_stable_incremental_revision_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = Path(kline_cache_module.__file__)
    spec = importlib.util.spec_from_file_location("_test_kline_cache_no_config", source)
    assert spec is not None and spec.loader is not None
    isolated = importlib.util.module_from_spec(spec)
    real_import = builtins.__import__

    def import_without_config(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "config":
            raise ImportError("config intentionally unavailable")
        return real_import(name, globals, locals, fromlist, level)

    with monkeypatch.context() as context:
        context.setattr(builtins, "__import__", import_without_config)
        spec.loader.exec_module(isolated)

    start = 1_700_000_000
    local = raw_frame(valid_rows(start, 6))
    forming = rates([(start + 7 * 3_600, 30.0, 31.0, 29.0, 30.5, 500)])
    remote = rates(valid_rows(start + 4 * 3_600, 2))
    fake = FakeMT5(np.concatenate([forming, remote]))
    isolated.mt5 = fake
    isolated._MT5_AVAILABLE = True
    cache = isolated.KlineCache(tmp_path, timeframe=16385, bars_count=20)

    cache._incremental_update("EURUSD", local)

    assert fake.calls == [("EURUSD", 16385, 1, 5)]


def canonical_dataset(
    start: int,
    count: int,
    *,
    price_offset: float = 0.0,
    symbol: str = "EURUSD",
):
    rows = valid_rows(start, count)
    if price_offset:
        rows = [
            (time, open_ + price_offset, high + price_offset, low + price_offset,
             close + price_offset, volume)
            for time, open_, high, low, close, volume in rows
        ]
    return canonicalize_ohlcv(
        raw_frame(rows), symbol=symbol, timeframe="H1", numeric_time_unit="s"
    )


def seeded_cache(cache: KlineCache, dataset) -> tuple[Path, Path, bytes, bytes]:
    path = cache._cache_path("EURUSD")
    metadata_path = cache._metadata_path("EURUSD")
    dataset.frame.to_parquet(path, index=False)
    metadata_path.write_text('{"original":true}', encoding="utf-8")
    return path, metadata_path, path.read_bytes(), metadata_path.read_bytes()


def assert_cache_pair_matches(
    cache: KlineCache,
    symbol: str,
    expected=None,
) -> None:
    frame = pd.read_parquet(cache._cache_path(symbol))
    actual = canonicalize_ohlcv(frame, symbol=symbol, timeframe="H1")
    metadata = json.loads(cache._metadata_path(symbol).read_text(encoding="utf-8"))
    assert metadata == {**actual.identity.to_dict(), "gap_count": actual.gap_count}
    if expected is not None:
        assert actual.identity == expected.identity
        pd.testing.assert_frame_equal(actual.frame, expected.frame)


def _cross_process_atomic_writer(
    cache_dir: str,
    role: str,
    price_offset: float,
    release_first_writer,
    messages,
) -> None:
    """Spawn-compatible offline worker used to exercise the real file lock."""
    cache = KlineCache(cache_dir, timeframe=16385)
    dataset = canonical_dataset(
        1_700_000_000,
        3,
        price_offset=price_offset,
    )
    path = cache._cache_path("EURUSD")
    real_replace = kline_cache_module.os.replace
    observed_parquet_commit = False

    def observe_replace(source, destination):
        nonlocal observed_parquet_commit
        result = real_replace(source, destination)
        if not observed_parquet_commit and Path(destination) == path:
            observed_parquet_commit = True
            messages.put(f"{role}:parquet")
            if role == "A" and not release_first_writer.wait(15):
                raise RuntimeError("timed out waiting to release first writer")
        return result

    kline_cache_module.os.replace = observe_replace
    messages.put(f"{role}:attempt")
    try:
        cache._atomic_write("EURUSD", dataset)
    except BaseException as exc:
        messages.put(f"{role}:error:{type(exc).__name__}:{exc}")
        raise
    else:
        messages.put(f"{role}:done")


def _hard_exit_atomic_writer(
    cache_dir: str,
    boundary: str,
    price_offset: float,
) -> None:
    """Spawn worker that dies immediately after a selected publication boundary."""
    cache = KlineCache(cache_dir, timeframe=16385)
    dataset = canonical_dataset(
        1_700_000_000,
        3,
        price_offset=price_offset,
    )
    parquet_path = cache._cache_path("EURUSD")
    metadata_path = cache._metadata_path("EURUSD")
    journal_path, _ = cache._transaction_paths(parquet_path, metadata_path)
    selected = parquet_path if boundary == "parquet" else metadata_path
    real_replace = kline_cache_module.os.replace
    real_rename = kline_cache_module.os.rename
    real_unlink = Path.unlink

    def exit_after_boundary(source, destination):
        result = real_replace(source, destination)
        if boundary in {"parquet", "metadata"} and Path(destination) == selected:
            os._exit(73)
        return result

    def exit_after_journal_publication(source, destination):
        result = real_rename(source, destination)
        if boundary == "journal" and Path(destination) == journal_path:
            os._exit(73)
        return result

    def exit_after_journal_removal(self, *args, **kwargs):
        result = real_unlink(self, *args, **kwargs)
        if boundary == "journal-removal" and self == journal_path:
            os._exit(73)
        return result

    kline_cache_module.os.replace = exit_after_boundary
    kline_cache_module.os.rename = exit_after_journal_publication
    Path.unlink = exit_after_journal_removal
    cache._atomic_write("EURUSD", dataset)


def _hold_cache_lock(cache_dir: str, ready, release) -> None:
    cache = KlineCache(cache_dir, timeframe=16385)
    with kline_cache_module._cache_transaction_lock(cache._cache_path("EURUSD")):
        ready.set()
        if not release.wait(30):
            raise RuntimeError("timed out waiting to release held cache lock")


def _timed_atomic_writer(cache_dir: str, messages) -> None:
    cache = KlineCache(cache_dir, timeframe=16385)
    dataset = canonical_dataset(1_700_000_000, 3, price_offset=20.0)
    started = time.monotonic()
    messages.put(("started", "", "", 0.0))
    try:
        cache._atomic_write("EURUSD", dataset)
    except BaseException as exc:
        messages.put(("error", type(exc).__name__, str(exc), time.monotonic() - started))
    else:
        messages.put(("ok", "", "", time.monotonic() - started))


@pytest.mark.parametrize("initial_state", ["empty", "metadata-only"])
@pytest.mark.parametrize(
    ("rollback_error_type", "rollback_message"),
    [
        (KeyboardInterrupt, "ROLLBACK unlink interrupted"),
        (SystemExit, "ROLLBACK unlink exited"),
        (OSError, "ROLLBACK unlink persistent"),
    ],
    ids=["keyboard-interrupt", "system-exit", "ordinary-oserror"],
)
def test_first_write_rollback_failure_reports_primary_and_recoverable_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    initial_state: str,
    rollback_error_type: type[BaseException],
    rollback_message: str,
) -> None:
    cache = KlineCache(tmp_path, timeframe=16385)
    new = canonical_dataset(1_700_000_000, 3, price_offset=20.0)
    path = cache._cache_path("EURUSD")
    metadata_path = cache._metadata_path("EURUSD")
    metadata_before = b""
    if initial_state == "metadata-only":
        metadata_path.write_bytes(b'{"original":true}')
        metadata_before = metadata_path.read_bytes()
    primary = OSError("PRIMARY metadata replace")
    rollback_error = rollback_error_type(rollback_message)
    real_replace = kline_cache_module.os.replace
    real_unlink = Path.unlink
    replace_calls = 0
    bytes_at_failed_unlink: list[bytes] = []

    def fail_metadata_commit(source, destination):
        nonlocal replace_calls
        replace_calls += 1
        if replace_calls == 2:
            raise primary
        return real_replace(source, destination)

    def fail_new_parquet_removal(self, *args, **kwargs):
        if self == path:
            bytes_at_failed_unlink.append(self.read_bytes())
            raise rollback_error
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr("data_pipeline.kline_cache.os.replace", fail_metadata_commit)
    monkeypatch.setattr(Path, "unlink", fail_new_parquet_removal)

    with pytest.raises(
        ExpectedCacheRecoveryError, match="cache recovery failed"
    ) as caught:
        cache._atomic_write("EURUSD", new)

    assert caught.value.__cause__ is primary
    assert rollback_error_type.__name__ in str(caught.value)
    assert rollback_message in str(caught.value)
    assert "retained evidence at " in str(caught.value)
    assert "retained evidence at ;" not in str(caught.value)
    assert bytes_at_failed_unlink and bytes_at_failed_unlink[0]
    assert path.exists()
    assert metadata_path.exists() == (initial_state == "metadata-only")
    if initial_state == "metadata-only":
        assert metadata_path.read_bytes() == metadata_before
    evidence_field = str(caught.value).split("retained evidence at ", 1)[1]
    evidence_field = evidence_field.split("; errors:", 1)[0]
    evidence = [Path(value) for value in evidence_field.split(", ") if value]
    assert evidence and all(candidate.exists() for candidate in evidence)
    assert any(
        candidate.read_bytes() == bytes_at_failed_unlink[0]
        for candidate in evidence
    )
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("initial_state", ["empty", "metadata-only"])
def test_first_write_rollback_success_restores_absence_and_reraises_exact_primary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    initial_state: str,
) -> None:
    cache = KlineCache(tmp_path, timeframe=16385)
    new = canonical_dataset(1_700_000_000, 3, price_offset=20.0)
    path = cache._cache_path("EURUSD")
    metadata_path = cache._metadata_path("EURUSD")
    metadata_before = b""
    if initial_state == "metadata-only":
        metadata_path.write_bytes(b'{"original":true}')
        metadata_before = metadata_path.read_bytes()
    primary = OSError("PRIMARY metadata replace")
    real_replace = kline_cache_module.os.replace
    replace_calls = 0

    def fail_metadata_commit(source, destination):
        nonlocal replace_calls
        replace_calls += 1
        if replace_calls == 2:
            raise primary
        return real_replace(source, destination)

    monkeypatch.setattr("data_pipeline.kline_cache.os.replace", fail_metadata_commit)

    with pytest.raises(OSError) as caught:
        cache._atomic_write("EURUSD", new)

    assert caught.value is primary
    assert not path.exists()
    assert metadata_path.exists() == (initial_state == "metadata-only")
    if initial_state == "metadata-only":
        assert metadata_path.read_bytes() == metadata_before
    assert not list(tmp_path.glob("*.recovery"))


def test_parquet_only_second_replace_failure_restores_old_parquet_and_absent_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = KlineCache(tmp_path, timeframe=16385)
    old = canonical_dataset(1_700_000_000, 3)
    new = canonical_dataset(1_700_000_000, 3, price_offset=20.0)
    path = cache._cache_path("EURUSD")
    metadata_path = cache._metadata_path("EURUSD")
    old.frame.to_parquet(path, index=False)
    parquet_before = path.read_bytes()
    primary = OSError("PRIMARY metadata replace")
    real_replace = kline_cache_module.os.replace
    replace_calls = 0

    def fail_metadata_commit(source, destination):
        nonlocal replace_calls
        replace_calls += 1
        if replace_calls == 2:
            raise primary
        return real_replace(source, destination)

    monkeypatch.setattr("data_pipeline.kline_cache.os.replace", fail_metadata_commit)

    with pytest.raises(OSError) as caught:
        cache._atomic_write("EURUSD", new)

    assert caught.value is primary
    assert path.read_bytes() == parquet_before
    assert not metadata_path.exists()
    assert not list(tmp_path.glob("*.recovery"))


def test_successful_first_write_creates_complete_pair_without_recovery_artifacts(
    tmp_path: Path,
) -> None:
    cache = KlineCache(tmp_path, timeframe=16385)
    new = canonical_dataset(1_700_000_000, 3, price_offset=20.0)

    cache._atomic_write("EURUSD", new)

    pd.testing.assert_frame_equal(pd.read_parquet(cache._cache_path("EURUSD")), new.frame)
    assert cache._metadata_path("EURUSD").exists()
    assert not list(tmp_path.glob("*.recovery"))


@pytest.mark.parametrize("failure_stage", ["serialization", "metadata_write"])
def test_atomic_write_precommit_failures_leave_both_originals_byte_identical(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    cache = KlineCache(tmp_path, timeframe=16385)
    old = canonical_dataset(1_700_000_000, 3)
    new = canonical_dataset(1_700_000_000, 3, price_offset=20.0)
    path, metadata_path, parquet_before, metadata_before = seeded_cache(cache, old)
    if failure_stage == "serialization":
        monkeypatch.setattr(
            pd.DataFrame,
            "to_parquet",
            lambda *args, **kwargs: (_ for _ in ()).throw(OSError("serialize")),
        )
    else:
        original_write_text = Path.write_text

        def fail_temp_metadata(self, *args, **kwargs):
            if self.suffix == ".tmp":
                raise OSError("metadata write")
            return original_write_text(self, *args, **kwargs)

        monkeypatch.setattr(Path, "write_text", fail_temp_metadata)

    with pytest.raises(OSError):
        cache._atomic_write("EURUSD", new)

    assert path.read_bytes() == parquet_before
    assert metadata_path.read_bytes() == metadata_before


@pytest.mark.parametrize("copy_call", [1, 2])
def test_atomic_write_backup_copy_failures_leave_both_originals_byte_identical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, copy_call: int
) -> None:
    cache = KlineCache(tmp_path, timeframe=16385)
    old = canonical_dataset(1_700_000_000, 3)
    new = canonical_dataset(1_700_000_000, 3, price_offset=20.0)
    path, metadata_path, parquet_before, metadata_before = seeded_cache(cache, old)
    real_copy2 = shutil.copy2
    calls = 0

    def fail_selected_copy(source, destination, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == copy_call:
            raise OSError(f"backup copy {copy_call}")
        return real_copy2(source, destination, *args, **kwargs)

    monkeypatch.setattr("data_pipeline.kline_cache.shutil.copy2", fail_selected_copy)

    with pytest.raises(OSError, match="backup copy"):
        cache._atomic_write("EURUSD", new)

    assert path.read_bytes() == parquet_before
    assert metadata_path.read_bytes() == metadata_before


@pytest.mark.parametrize("replace_call", [1, 2])
def test_atomic_write_replace_failures_restore_both_originals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, replace_call: int
) -> None:
    cache = KlineCache(tmp_path, timeframe=16385)
    old = canonical_dataset(1_700_000_000, 3)
    new = canonical_dataset(1_700_000_000, 3, price_offset=20.0)
    path, metadata_path, parquet_before, metadata_before = seeded_cache(cache, old)
    real_replace = kline_cache_module.os.replace
    calls = 0

    def fail_selected_replace(source, destination):
        nonlocal calls
        calls += 1
        if calls == replace_call:
            raise OSError(f"replace {replace_call}")
        return real_replace(source, destination)

    monkeypatch.setattr("data_pipeline.kline_cache.os.replace", fail_selected_replace)

    with pytest.raises(OSError, match=f"replace {replace_call}"):
        cache._atomic_write("EURUSD", new)

    assert path.read_bytes() == parquet_before
    assert metadata_path.read_bytes() == metadata_before


def test_atomic_write_commit_interruption_restores_pair_and_reraises_exact_object(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = KlineCache(tmp_path, timeframe=16385)
    old = canonical_dataset(1_700_000_000, 3)
    new = canonical_dataset(1_700_000_000, 3, price_offset=20.0)
    path, metadata_path, parquet_before, metadata_before = seeded_cache(cache, old)
    real_replace = kline_cache_module.os.replace
    interruption = KeyboardInterrupt("metadata commit interrupted")
    calls = 0

    def interrupt_metadata_commit(source, destination):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise interruption
        return real_replace(source, destination)

    monkeypatch.setattr(
        "data_pipeline.kline_cache.os.replace", interrupt_metadata_commit
    )

    with pytest.raises(KeyboardInterrupt) as caught:
        cache._atomic_write("EURUSD", new)

    assert caught.value is interruption
    assert path.read_bytes() == parquet_before
    assert metadata_path.read_bytes() == metadata_before
    assert not list(tmp_path.glob("*.tmp"))
    assert not list(tmp_path.glob("*.recovery"))


def test_atomic_write_failed_interruption_rollback_preserves_exact_cause_and_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = KlineCache(tmp_path, timeframe=16385)
    old = canonical_dataset(1_700_000_000, 3)
    new = canonical_dataset(1_700_000_000, 3, price_offset=20.0)
    path, metadata_path, parquet_before, metadata_before = seeded_cache(cache, old)
    real_replace = kline_cache_module.os.replace
    interruption = KeyboardInterrupt("metadata commit interrupted")
    calls = 0

    def interrupt_commit_and_fail_rollback(source, destination):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise interruption
        if calls >= 3:
            raise OSError(f"persistent rollback replace {calls}")
        return real_replace(source, destination)

    monkeypatch.setattr(
        "data_pipeline.kline_cache.os.replace", interrupt_commit_and_fail_rollback
    )

    with pytest.raises(
        ExpectedCacheRecoveryError, match="cache recovery failed"
    ) as caught:
        cache._atomic_write("EURUSD", new)

    assert caught.value.__cause__ is interruption
    recovery = list(tmp_path.glob("*.recovery"))
    assert recovery
    assert any(candidate.read_bytes() == parquet_before for candidate in recovery)
    assert metadata_path.read_bytes() == metadata_before
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize(
    "rollback_error",
    [
        KeyboardInterrupt("ROLLBACK interrupted"),
        SystemExit("ROLLBACK exited"),
    ],
    ids=["keyboard-interrupt", "system-exit"],
)
def test_atomic_write_rollback_baseexception_is_reported_without_masking_primary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    rollback_error: BaseException,
) -> None:
    cache = KlineCache(tmp_path, timeframe=16385)
    old = canonical_dataset(1_700_000_000, 3)
    new = canonical_dataset(1_700_000_000, 3, price_offset=20.0)
    path, metadata_path, parquet_before, metadata_before = seeded_cache(cache, old)
    real_replace = kline_cache_module.os.replace
    primary = OSError("PRIMARY metadata replace")
    calls = 0

    def fail_metadata_commit_and_rollback(source, destination):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise primary
        if calls >= 3:
            raise rollback_error
        return real_replace(source, destination)

    monkeypatch.setattr(
        "data_pipeline.kline_cache.os.replace", fail_metadata_commit_and_rollback
    )

    with pytest.raises(
        ExpectedCacheRecoveryError, match="cache recovery failed"
    ) as caught:
        cache._atomic_write("EURUSD", new)

    assert type(caught.value) is ExpectedCacheRecoveryError
    assert caught.value.__cause__ is primary
    assert type(rollback_error).__name__ in str(caught.value)
    assert str(rollback_error) in str(caught.value)
    assert path.read_bytes() != parquet_before
    pd.testing.assert_frame_equal(pd.read_parquet(path), new.frame)
    assert metadata_path.read_bytes() == metadata_before
    recovery = list(tmp_path.glob("*.recovery"))
    assert recovery
    assert any(candidate.read_bytes() == parquet_before for candidate in recovery)
    assert not list(tmp_path.glob("*.tmp"))


def test_atomic_write_retries_one_shot_rollback_failure_and_restores_originals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = KlineCache(tmp_path, timeframe=16385)
    old = canonical_dataset(1_700_000_000, 3)
    new = canonical_dataset(1_700_000_000, 3, price_offset=20.0)
    path, metadata_path, parquet_before, metadata_before = seeded_cache(cache, old)
    real_replace = kline_cache_module.os.replace
    calls = 0

    def fail_commit_and_first_rollback(source, destination):
        nonlocal calls
        calls += 1
        if calls in {2, 3}:
            raise OSError(f"replace call {calls}")
        return real_replace(source, destination)

    monkeypatch.setattr(
        "data_pipeline.kline_cache.os.replace", fail_commit_and_first_rollback
    )

    with pytest.raises(OSError, match="replace call 2"):
        cache._atomic_write("EURUSD", new)

    assert calls >= 4
    assert path.read_bytes() == parquet_before
    assert metadata_path.read_bytes() == metadata_before
    assert not list(tmp_path.glob("*.recovery"))


def test_atomic_write_persistent_rollback_failure_preserves_recovery_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = KlineCache(tmp_path, timeframe=16385)
    old = canonical_dataset(1_700_000_000, 3)
    new = canonical_dataset(1_700_000_000, 3, price_offset=20.0)
    path, metadata_path, parquet_before, metadata_before = seeded_cache(cache, old)
    real_replace = kline_cache_module.os.replace
    calls = 0

    def fail_commit_and_all_rollback(source, destination):
        nonlocal calls
        calls += 1
        if calls >= 2:
            raise OSError(f"persistent replace call {calls}")
        return real_replace(source, destination)

    monkeypatch.setattr(
        "data_pipeline.kline_cache.os.replace", fail_commit_and_all_rollback
    )

    with pytest.raises(
        ExpectedCacheRecoveryError, match="cache recovery failed"
    ) as caught:
        cache._atomic_write("EURUSD", new)

    assert type(caught.value) is ExpectedCacheRecoveryError
    assert type(caught.value.__cause__) is OSError
    assert str(caught.value.__cause__) == "persistent replace call 2"
    recovery = list(tmp_path.glob("*.recovery"))
    assert recovery
    assert any(candidate.read_bytes() == parquet_before for candidate in recovery)
    assert metadata_path.read_bytes() == metadata_before


def test_successful_atomic_commit_ignores_recovery_cleanup_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = KlineCache(tmp_path, timeframe=16385)
    old = canonical_dataset(1_700_000_000, 3)
    new = canonical_dataset(1_700_000_000, 3, price_offset=20.0)
    path, metadata_path, _, _ = seeded_cache(cache, old)
    real_unlink = Path.unlink
    cleanup_attempts: list[Path] = []

    def deny_recovery_cleanup(self, *args, **kwargs):
        if self.suffix == ".recovery":
            cleanup_attempts.append(self)
            raise PermissionError("recovery cleanup denied")
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", deny_recovery_cleanup)

    cache._atomic_write("EURUSD", new)

    pd.testing.assert_frame_equal(pd.read_parquet(path), new.frame)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["data_fingerprint"] == new.identity.data_fingerprint
    assert metadata["time_fingerprint"] == new.identity.time_fingerprint
    assert cleanup_attempts
    assert list(tmp_path.glob("*.recovery"))


def test_successful_atomic_commit_ignores_recovery_cleanup_keyboard_interrupt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = KlineCache(tmp_path, timeframe=16385)
    old = canonical_dataset(1_700_000_000, 3)
    new = canonical_dataset(1_700_000_000, 3, price_offset=20.0)
    path, metadata_path, _, _ = seeded_cache(cache, old)
    real_unlink = Path.unlink
    cleanup_interruption = KeyboardInterrupt("recovery cleanup interrupted")
    cleanup_attempts: list[Path] = []

    def interrupt_recovery_cleanup(self, *args, **kwargs):
        if self.suffix == ".recovery":
            cleanup_attempts.append(self)
            raise cleanup_interruption
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", interrupt_recovery_cleanup)

    cache._atomic_write("EURUSD", new)

    pd.testing.assert_frame_equal(pd.read_parquet(path), new.frame)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["data_fingerprint"] == new.identity.data_fingerprint
    assert metadata["time_fingerprint"] == new.identity.time_fingerprint
    assert cleanup_attempts
    assert list(tmp_path.glob("*.recovery"))


def test_serialization_error_is_not_masked_by_temp_cleanup_keyboard_interrupt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = KlineCache(tmp_path, timeframe=16385)
    old = canonical_dataset(1_700_000_000, 3)
    new = canonical_dataset(1_700_000_000, 3, price_offset=20.0)
    path, metadata_path, parquet_before, metadata_before = seeded_cache(cache, old)
    primary = OSError("serialization failed after partial write")
    cleanup_interruption = KeyboardInterrupt("temporary cleanup interrupted")
    real_unlink = Path.unlink

    def partial_then_fail(self, target, *args, **kwargs):
        Path(target).write_bytes(b"partial")
        raise primary

    def interrupt_temp_cleanup(self, *args, **kwargs):
        if self.suffix == ".tmp":
            raise cleanup_interruption
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_parquet", partial_then_fail)
    monkeypatch.setattr(Path, "unlink", interrupt_temp_cleanup)

    with pytest.raises(OSError) as caught:
        cache._atomic_write("EURUSD", new)

    assert caught.value is primary
    assert caught.value.__cause__ is None
    assert path.read_bytes() == parquet_before
    assert metadata_path.read_bytes() == metadata_before
    assert list(tmp_path.glob("*.tmp"))


def test_commit_interruption_is_not_masked_by_recovery_cleanup_system_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = KlineCache(tmp_path, timeframe=16385)
    old = canonical_dataset(1_700_000_000, 3)
    new = canonical_dataset(1_700_000_000, 3, price_offset=20.0)
    path, metadata_path, parquet_before, metadata_before = seeded_cache(cache, old)
    real_replace = kline_cache_module.os.replace
    real_unlink = Path.unlink
    primary = KeyboardInterrupt("metadata commit interrupted")
    cleanup_interruption = SystemExit("recovery cleanup interrupted")
    replace_calls = 0

    def interrupt_metadata_commit(source, destination):
        nonlocal replace_calls
        replace_calls += 1
        if replace_calls == 2:
            raise primary
        return real_replace(source, destination)

    def interrupt_recovery_cleanup(self, *args, **kwargs):
        if self.suffix == ".recovery":
            raise cleanup_interruption
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(
        "data_pipeline.kline_cache.os.replace", interrupt_metadata_commit
    )
    monkeypatch.setattr(Path, "unlink", interrupt_recovery_cleanup)

    with pytest.raises(KeyboardInterrupt) as caught:
        cache._atomic_write("EURUSD", new)

    assert caught.value is primary
    assert path.read_bytes() == parquet_before
    assert metadata_path.read_bytes() == metadata_before
    assert list(tmp_path.glob("*.recovery"))


@pytest.mark.parametrize("failure_stage", ["serialization", "commit"])
def test_cleanup_failure_never_masks_primary_atomic_write_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    cache = KlineCache(tmp_path, timeframe=16385)
    old = canonical_dataset(1_700_000_000, 3)
    new = canonical_dataset(1_700_000_000, 3, price_offset=20.0)
    path, metadata_path, parquet_before, metadata_before = seeded_cache(cache, old)
    primary = OSError(f"primary {failure_stage} failure")
    real_unlink = Path.unlink
    cleanup_attempts: list[Path] = []

    def deny_selected_cleanup(self, *args, **kwargs):
        denied_suffix = ".tmp" if failure_stage == "serialization" else ".recovery"
        if self.suffix == denied_suffix:
            cleanup_attempts.append(self)
            raise PermissionError("cleanup denied")
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", deny_selected_cleanup)
    if failure_stage == "serialization":
        monkeypatch.setattr(
            pd.DataFrame,
            "to_parquet",
            lambda *args, **kwargs: (_ for _ in ()).throw(primary),
        )
    else:
        real_replace = kline_cache_module.os.replace
        replace_calls = 0

        def fail_metadata_commit(source, destination):
            nonlocal replace_calls
            replace_calls += 1
            if replace_calls == 2:
                raise primary
            return real_replace(source, destination)

        monkeypatch.setattr(
            "data_pipeline.kline_cache.os.replace", fail_metadata_commit
        )

    with pytest.raises(OSError) as caught:
        cache._atomic_write("EURUSD", new)

    assert caught.value is primary
    assert cleanup_attempts
    assert path.read_bytes() == parquet_before
    assert metadata_path.read_bytes() == metadata_before


@pytest.mark.parametrize("read_mode", ["get", "force-refresh", "read-local"])
def test_every_public_local_read_rejects_metadata_free_legacy_cache_without_mutating_disk(
    tmp_path: Path, read_mode: str
) -> None:
    cache = KlineCache(tmp_path, timeframe=16385)
    path = cache._cache_path("EURUSD")
    raw_frame(valid_rows(1_700_000_000, 4)).to_parquet(path, index=False)
    before = path.read_bytes()

    with pytest.raises(ExpectedCacheRecoveryError, match="cache pair"):
        if read_mode == "get":
            cache.get("EURUSD", mt5_connected=False)
        elif read_mode == "force-refresh":
            cache.get("EURUSD", force_refresh=True, mt5_connected=False)
        else:
            cache.read_local("EURUSD")

    assert path.read_bytes() == before
    assert not cache._metadata_path("EURUSD").exists()


def _public_cache_read(cache: KlineCache, operation: str):
    if operation == "read_local":
        return cache.read_local("EURUSD")
    if operation == "get_offline":
        return cache.get("EURUSD", mt5_connected=False)
    if operation == "list_cached":
        return cache.list_cached()
    raise AssertionError(operation)


def _metadata_damage_cases(dataset) -> list[tuple[str, object]]:
    valid = {**dataset.identity.to_dict(), "gap_count": dataset.gap_count}
    cases: list[tuple[str, object]] = [
        ("invalid-json", "{"),
        ("non-object", []),
        ("unknown-field", {**valid, "unknown": "evidence"}),
    ]
    for field in valid:
        missing = dict(valid)
        del missing[field]
        cases.append((f"missing-{field}", missing))
    mismatches = {
        "schema_version": "ohlcv-v999",
        "symbol": "GBPUSD",
        "timeframe": "M30",
        "start_time_ns": valid["start_time_ns"] + 1,
        "end_time_ns": valid["end_time_ns"] + 1,
        "bars": valid["bars"] + 1,
        "data_fingerprint": "0" * 64,
        "time_fingerprint": "1" * 64,
        "gap_count": valid["gap_count"] + 1,
    }
    for field, value in mismatches.items():
        cases.append((f"mismatch-{field}", {**valid, field: value}))
    return cases


@pytest.mark.parametrize("operation", ["read_local", "get_offline", "list_cached"])
def test_public_cache_reads_reject_missing_metadata_without_changing_evidence_or_aliasing_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    cache = KlineCache(tmp_path, timeframe=16385)
    dataset = canonical_dataset(1_700_000_000, 3)
    path = cache._cache_path("EURUSD")
    dataset.frame.to_parquet(path, index=False)
    parquet_before = path.read_bytes()
    real_lock = kline_cache_module._cache_transaction_lock
    locked_targets: list[Path] = []

    @kline_cache_module.contextmanager
    def recording_lock(target: Path):
        locked_targets.append(target)
        with real_lock(target):
            yield

    monkeypatch.setattr(kline_cache_module, "_cache_transaction_lock", recording_lock)

    with pytest.raises(ExpectedCacheRecoveryError, match="cache pair"):
        _public_cache_read(cache, operation)

    assert path.read_bytes() == parquet_before
    assert not cache._metadata_path("EURUSD").exists()
    assert locked_targets == [path]


def test_online_get_rejects_metadata_free_parquet_without_fetch_or_metadata_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = KlineCache(tmp_path, timeframe=16385)
    dataset = canonical_dataset(1_700_000_000, 3)
    path = cache._cache_path("EURUSD")
    dataset.frame.to_parquet(path, index=False)
    before = path.read_bytes()
    fake = FakeMT5(rates(valid_rows(1_700_020_000, 3)))
    write = MagicMock()
    monkeypatch.setattr(kline_cache_module, "mt5", fake)
    monkeypatch.setattr(kline_cache_module, "_MT5_AVAILABLE", True)
    monkeypatch.setattr(cache, "_atomic_write", write)

    with pytest.raises(ExpectedCacheRecoveryError, match="cache pair"):
        cache.get("EURUSD", mt5_connected=True)

    assert fake.calls == []
    write.assert_not_called()
    assert path.read_bytes() == before
    assert not cache._metadata_path("EURUSD").exists()


def test_list_cached_rejects_metadata_only_target_without_inventing_parquet(
    tmp_path: Path,
) -> None:
    cache = KlineCache(tmp_path, timeframe=16385)
    metadata_path = cache._metadata_path("EURUSD")
    payload = b'{"retained":"metadata-only"}'
    metadata_path.write_bytes(payload)

    with pytest.raises(ExpectedCacheRecoveryError, match="cache pair"):
        cache.list_cached()

    assert metadata_path.read_bytes() == payload
    assert not cache._cache_path("EURUSD").exists()


@pytest.mark.parametrize("operation", ["read_local", "get_offline", "list_cached"])
@pytest.mark.parametrize(
    "case_index",
    range(21),
    ids=[
        "invalid-json", "non-object", "unknown-field",
        "missing-schema", "missing-symbol", "missing-timeframe",
        "missing-start", "missing-end", "missing-bars", "missing-data-fingerprint",
        "missing-time-fingerprint", "missing-gap-count", "wrong-schema",
        "wrong-symbol", "wrong-timeframe", "wrong-start", "wrong-end", "wrong-bars",
        "wrong-data-fingerprint", "wrong-time-fingerprint", "wrong-gap-count",
    ],
)
def test_public_cache_reads_reject_every_incomplete_or_mismatched_metadata_form(
    tmp_path: Path,
    operation: str,
    case_index: int,
) -> None:
    cache = KlineCache(tmp_path, timeframe=16385)
    dataset = canonical_dataset(1_700_000_000, 3)
    cache._atomic_write("EURUSD", dataset)
    path = cache._cache_path("EURUSD")
    metadata_path = cache._metadata_path("EURUSD")
    case_name, damaged = _metadata_damage_cases(dataset)[case_index]
    if case_name == "invalid-json":
        metadata_path.write_text(damaged, encoding="utf-8")
    else:
        metadata_path.write_text(json.dumps(damaged), encoding="utf-8")
    parquet_before = path.read_bytes()
    metadata_before = metadata_path.read_bytes()

    with pytest.raises(ExpectedCacheRecoveryError, match="cache pair"):
        _public_cache_read(cache, operation)

    assert path.read_bytes() == parquet_before
    assert metadata_path.read_bytes() == metadata_before


def test_valid_and_recovered_valid_pairs_are_readable_and_listed_once(
    tmp_path: Path,
) -> None:
    cache = KlineCache(tmp_path, timeframe=16385)
    dataset = canonical_dataset(1_700_000_000, 3)
    cache._atomic_write("EURUSD", dataset)
    path = cache._cache_path("EURUSD")
    metadata_path = cache._metadata_path("EURUSD")
    _, recoveries = cache._transaction_paths(path, metadata_path)

    first = cache.read_local("EURUSD")
    assert first is not None
    pd.testing.assert_frame_equal(first, dataset.frame)
    assert [entry["file"] for entry in cache.list_cached()] == [path.name]

    for target, recovery in recoveries.items():
        shutil.copy2(target, recovery)
    recovered = cache.read_local("EURUSD")
    assert recovered is not None
    pd.testing.assert_frame_equal(recovered, dataset.frame)
    assert [entry["file"] for entry in cache.list_cached()] == [path.name]


def test_update_all_reraises_exact_cache_recovery_error_object(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = KlineCache(tmp_path, timeframe=16385)
    sentinel = ExpectedCacheRecoveryError("sentinel recovery")
    monkeypatch.setattr(
        cache,
        "_read_local_with_recovery",
        lambda _symbol: (_ for _ in ()).throw(sentinel),
    )

    with pytest.raises(ExpectedCacheRecoveryError) as caught:
        cache.update_all(["EURUSD"], mt5_connected=False)

    assert caught.value is sentinel


def test_update_all_keeps_documented_minus_one_for_ordinary_symbol_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = KlineCache(tmp_path, timeframe=16385)
    ordinary = RuntimeError("ordinary update failure")
    monkeypatch.setattr(
        cache,
        "_read_local_with_recovery",
        lambda _symbol: (_ for _ in ()).throw(ordinary),
    )

    assert cache.update_all(["EURUSD"], mt5_connected=False) == {"EURUSD": -1}


@pytest.mark.parametrize("symbols", [["EURUSD"], ["EURUSD", "GBPUSD", "USDJPY"]])
def test_update_all_offline_reads_each_validated_pair_exactly_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    symbols: list[str],
) -> None:
    cache = KlineCache(tmp_path, timeframe=16385)
    for index, symbol in enumerate(symbols):
        cache._atomic_write(
            symbol,
            canonical_dataset(
                1_700_000_000,
                3,
                symbol=symbol,
                price_offset=index * 10.0,
            ),
        )
    real_read_parquet = pd.read_parquet
    reads: list[Path] = []

    def count_pair_read(path, *args, **kwargs):
        reads.append(Path(path))
        return real_read_parquet(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", count_pair_read)

    assert cache.update_all(symbols, mt5_connected=False) == {
        symbol: 0 for symbol in symbols
    }
    assert reads == [cache._cache_path(symbol) for symbol in symbols]


def test_invalid_legacy_local_cache_fails_closed_without_mutating_disk(
    tmp_path: Path,
) -> None:
    cache = KlineCache(tmp_path, timeframe=16385)
    path = cache._cache_path("EURUSD")
    invalid = raw_frame(valid_rows(1_700_000_000, 4))
    invalid.loc[1, "high"] = invalid.loc[1, "low"]
    invalid.to_parquet(path, index=False)
    before = path.read_bytes()

    with pytest.raises(ExpectedCacheRecoveryError, match="cache pair"):
        cache.read_local("EURUSD")

    assert path.read_bytes() == before


@pytest.mark.parametrize("backend_error_type", [OSError, ArrowInvalid])
def test_local_parquet_backend_read_error_preserves_exact_cause(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    backend_error_type: type[Exception],
) -> None:
    cache = KlineCache(tmp_path, timeframe=16385)
    cache._atomic_write("EURUSD", canonical_dataset(1_700_000_000, 3))
    backend_error = backend_error_type("read failed")
    monkeypatch.setattr(
        "data_pipeline.kline_cache.pd.read_parquet",
        lambda *args, **kwargs: (_ for _ in ()).throw(backend_error),
    )

    with pytest.raises(ExpectedCacheRecoveryError, match="cache pair") as caught:
        cache.read_local("EURUSD")

    assert isinstance(caught.value, kline_cache_module.CacheReadError)
    assert type(caught.value.__cause__) is kline_cache_module.CacheReadError
    assert caught.value.__cause__.__cause__ is backend_error


@pytest.mark.parametrize(
    "programming_error_type",
    [KeyError, TypeError, AssertionError, RuntimeError, ImportError],
)
def test_local_parquet_programming_or_configuration_error_propagates_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    programming_error_type: type[Exception],
) -> None:
    cache = KlineCache(tmp_path, timeframe=16385)
    cache._atomic_write("EURUSD", canonical_dataset(1_700_000_000, 3))
    programming_error = programming_error_type("programmer bug")
    monkeypatch.setattr(
        "data_pipeline.kline_cache.pd.read_parquet",
        lambda *args, **kwargs: (_ for _ in ()).throw(programming_error),
    )

    with pytest.raises(programming_error_type) as caught:
        cache.read_local("EURUSD")

    assert caught.value is programming_error
    assert not isinstance(caught.value, kline_cache_module.CacheReadError)


def test_kline_cache_import_without_pyarrow_does_not_classify_engine_import_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = Path(kline_cache_module.__file__)
    spec = importlib.util.spec_from_file_location("_test_kline_cache_no_pyarrow", source)
    assert spec is not None and spec.loader is not None
    isolated = importlib.util.module_from_spec(spec)
    real_import = builtins.__import__

    def import_without_pyarrow(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "pyarrow.lib":
            raise ImportError("pyarrow intentionally unavailable")
        return real_import(name, globals, locals, fromlist, level)

    with monkeypatch.context() as context:
        context.setattr(builtins, "__import__", import_without_pyarrow)
        spec.loader.exec_module(isolated)

    KlineCache(tmp_path, timeframe=16385)._atomic_write(
        "EURUSD", canonical_dataset(1_700_000_000, 3)
    )
    cache = isolated.KlineCache(tmp_path, timeframe=16385)
    engine_error = ImportError("parquet engine missing")
    monkeypatch.setattr(
        isolated.pd,
        "read_parquet",
        lambda *args, **kwargs: (_ for _ in ()).throw(engine_error),
    )

    with pytest.raises(ImportError) as caught:
        cache.read_local("EURUSD")

    assert caught.value is engine_error


def test_actual_corrupt_parquet_is_cache_read_error_with_backend_cause(
    tmp_path: Path,
) -> None:
    cache = KlineCache(tmp_path, timeframe=16385)
    cache._atomic_write("EURUSD", canonical_dataset(1_700_000_000, 3))
    cache._cache_path("EURUSD").write_bytes(b"not parquet")

    with pytest.raises(ExpectedCacheRecoveryError, match="cache pair") as caught:
        cache.read_local("EURUSD")

    assert isinstance(caught.value, kline_cache_module.CacheReadError)
    assert type(caught.value.__cause__) is kline_cache_module.CacheReadError
    assert type(caught.value.__cause__.__cause__).__name__ == "ArrowInvalid"


def test_identical_remote_tail_is_noop_with_zero_cache_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    start = 1_700_000_000
    local_rows = valid_rows(start, 10)
    local = raw_frame(local_rows)
    forming = rates([(start + 10 * 3_600, 30.0, 31.0, 29.0, 30.5, 500)])
    identical_tail = rates(local_rows[5:])
    fake = FakeMT5(np.concatenate([forming, identical_tail]))
    monkeypatch.setattr("data_pipeline.kline_cache.mt5", fake)
    monkeypatch.setattr("data_pipeline.kline_cache._MT5_AVAILABLE", True)
    cache = KlineCache(tmp_path, timeframe=16385)
    write = MagicMock()
    monkeypatch.setattr(cache, "_atomic_write", write)

    actual = cache._incremental_update("EURUSD", local)

    expected = canonicalize_ohlcv(
        local, symbol="EURUSD", timeframe="H1", numeric_time_unit="s"
    ).frame
    pd.testing.assert_frame_equal(actual, expected)
    write.assert_not_called()


@pytest.mark.parametrize("change", ["revision", "addition"])
def test_real_remote_change_writes_once_from_exact_merged_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    start = 1_700_000_000
    local = raw_frame(valid_rows(start, 10))
    remote_rows = valid_rows(start + 5 * 3_600, 5)
    if change == "revision":
        timestamp, _, _, _, _, volume = remote_rows[2]
        remote_rows[2] = (timestamp, 99.0, 100.0, 98.0, 99.5, volume)
    else:
        remote_rows = valid_rows(start + 6 * 3_600, 5)
    forming = rates([(start + 12 * 3_600, 30.0, 31.0, 29.0, 30.5, 500)])
    fake = FakeMT5(np.concatenate([forming, rates(remote_rows)]))
    monkeypatch.setattr("data_pipeline.kline_cache.mt5", fake)
    monkeypatch.setattr("data_pipeline.kline_cache._MT5_AVAILABLE", True)
    cache = KlineCache(tmp_path, timeframe=16385)
    write = MagicMock()
    monkeypatch.setattr(cache, "_atomic_write", write)

    actual = cache._incremental_update("EURUSD", local)

    write.assert_called_once()
    assert write.call_args.args[0] == "EURUSD"
    assert write.call_args.kwargs == {"_lock_held": True}
    written = write.call_args.args[1]
    pd.testing.assert_frame_equal(written.frame, actual)
    recanonicalized = canonicalize_ohlcv(actual, symbol="EURUSD", timeframe="H1")
    assert written.identity == recanonicalized.identity
    assert written.gap_count == recanonicalized.gap_count


def test_atomic_write_same_target_threads_never_interleave_commits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_cache = KlineCache(tmp_path, timeframe=16385)
    second_cache = KlineCache(tmp_path, timeframe=16385)
    first = canonical_dataset(1_700_000_000, 3, price_offset=10.0)
    second = canonical_dataset(1_700_000_000, 3, price_offset=20.0)
    path = first_cache._cache_path("EURUSD")
    real_replace = kline_cache_module.os.replace
    first_parquet_committed = threading.Event()
    release_first = threading.Event()
    second_attempting = threading.Event()
    second_parquet_committed = threading.Event()
    errors: list[BaseException] = []

    def observe_replace(source, destination):
        result = real_replace(source, destination)
        if Path(destination) == path:
            if threading.current_thread().name == "writer-a":
                first_parquet_committed.set()
                if not release_first.wait(10):
                    raise RuntimeError("timed out waiting to release writer A")
            elif threading.current_thread().name == "writer-b":
                second_parquet_committed.set()
        return result

    def write(cache: KlineCache, dataset, *, attempting=None) -> None:
        if attempting is not None:
            attempting.set()
        try:
            cache._atomic_write("EURUSD", dataset)
        except BaseException as exc:
            errors.append(exc)

    monkeypatch.setattr("data_pipeline.kline_cache.os.replace", observe_replace)
    writer_a = threading.Thread(
        target=write,
        name="writer-a",
        args=(first_cache, first),
    )
    writer_b = threading.Thread(
        target=write,
        name="writer-b",
        args=(second_cache, second),
        kwargs={"attempting": second_attempting},
    )
    writer_a.start()
    assert first_parquet_committed.wait(10)
    writer_b.start()
    assert second_attempting.wait(10)
    interleaved = second_parquet_committed.wait(1)
    release_first.set()
    writer_a.join(10)
    writer_b.join(10)

    assert not writer_a.is_alive()
    assert not writer_b.is_alive()
    assert not interleaved
    assert errors == []
    assert_cache_pair_matches(first_cache, "EURUSD", expected=second)


def test_atomic_write_failed_rollback_cannot_overwrite_successful_concurrent_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failing_cache = KlineCache(tmp_path, timeframe=16385)
    successful_cache = KlineCache(tmp_path, timeframe=16385)
    original = canonical_dataset(1_700_000_000, 3)
    failing = canonical_dataset(1_700_000_000, 3, price_offset=10.0)
    successful = canonical_dataset(1_700_000_000, 3, price_offset=20.0)
    failing_cache._atomic_write("EURUSD", original)
    path = failing_cache._cache_path("EURUSD")
    real_replace = kline_cache_module.os.replace
    failing_parquet_committed = threading.Event()
    allow_metadata_failure = threading.Event()
    successful_attempting = threading.Event()
    successful_done = threading.Event()
    primary = OSError("forced metadata commit failure")
    failing_replace_calls = 0
    errors: dict[str, BaseException] = {}

    def fail_selected_replace(source, destination):
        nonlocal failing_replace_calls
        if threading.current_thread().name != "failing-writer":
            return real_replace(source, destination)
        failing_replace_calls += 1
        if failing_replace_calls == 1:
            result = real_replace(source, destination)
            assert Path(destination) == path
            failing_parquet_committed.set()
            if not allow_metadata_failure.wait(10):
                raise RuntimeError("timed out waiting to fail metadata commit")
            return result
        if failing_replace_calls == 2:
            raise primary
        return real_replace(source, destination)

    def failing_write() -> None:
        try:
            failing_cache._atomic_write("EURUSD", failing)
        except BaseException as exc:
            errors["failing"] = exc

    def successful_write() -> None:
        successful_attempting.set()
        try:
            successful_cache._atomic_write("EURUSD", successful)
        except BaseException as exc:
            errors["successful"] = exc
        finally:
            successful_done.set()

    monkeypatch.setattr(
        "data_pipeline.kline_cache.os.replace",
        fail_selected_replace,
    )
    first_thread = threading.Thread(target=failing_write, name="failing-writer")
    second_thread = threading.Thread(target=successful_write, name="successful-writer")
    first_thread.start()
    assert failing_parquet_committed.wait(10)
    second_thread.start()
    assert successful_attempting.wait(10)
    success_interleaved = successful_done.wait(1)
    allow_metadata_failure.set()
    first_thread.join(10)
    second_thread.join(10)

    assert not first_thread.is_alive()
    assert not second_thread.is_alive()
    assert not success_interleaved
    assert errors == {"failing": primary}
    assert_cache_pair_matches(successful_cache, "EURUSD", expected=successful)


def test_atomic_write_releases_same_target_lock_after_baseexception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = KlineCache(tmp_path, timeframe=16385)
    interrupted = canonical_dataset(1_700_000_000, 3, price_offset=10.0)
    successful = canonical_dataset(1_700_000_000, 3, price_offset=20.0)
    interruption = KeyboardInterrupt("serialization interrupted")
    real_to_parquet = pd.DataFrame.to_parquet
    interrupted_errors: queue.Queue[BaseException] = queue.Queue()
    successful_errors: queue.Queue[BaseException] = queue.Queue()

    def interrupt_first_writer(self, *args, **kwargs):
        if threading.current_thread().name == "interrupted-writer":
            raise interruption
        return real_to_parquet(self, *args, **kwargs)

    def interrupted_write() -> None:
        cache._atomic_write("EURUSD", interrupted)

    def successful_write() -> None:
        cache._atomic_write("EURUSD", successful)

    monkeypatch.setattr(pd.DataFrame, "to_parquet", interrupt_first_writer)
    first_thread = threading.Thread(
        target=run_worker_capturing_baseexceptions,
        name="interrupted-writer",
        args=(interrupted_write, interrupted_errors),
    )
    first_thread.start()
    first_thread.join(10)
    assert not first_thread.is_alive()
    assert interrupted_errors.get_nowait() is interruption
    assert interrupted_errors.empty()

    second_thread = threading.Thread(
        target=run_worker_capturing_baseexceptions,
        name="successful-writer",
        args=(successful_write, successful_errors),
    )
    second_thread.start()
    join_worker_without_errors(second_thread, successful_errors)
    assert_cache_pair_matches(cache, "EURUSD", expected=successful)


def test_worker_assertion_kills_post_commit_release_exception_mutation(
    tmp_path: Path,
) -> None:
    guarded_source = inspect.getsource(
        test_atomic_write_releases_same_target_lock_after_baseexception
    )
    assert "target=cache._atomic_write" not in guarded_source
    assert guarded_source.count("target=run_worker_capturing_baseexceptions") == 2

    cache = KlineCache(tmp_path, timeframe=16385)
    successful = canonical_dataset(1_700_000_000, 3, price_offset=20.0)
    mutation = OSError("post-commit release mutation")
    errors: queue.Queue[BaseException] = queue.Queue()

    def mutated_successful_write() -> None:
        cache._atomic_write("EURUSD", successful)
        raise mutation

    worker = threading.Thread(
        target=run_worker_capturing_baseexceptions,
        name="successful-writer",
        args=(mutated_successful_write, errors),
    )
    worker.start()

    with pytest.raises(AssertionError, match="post-commit release mutation"):
        join_worker_without_errors(worker, errors)

    assert errors.get_nowait() is mutation
    assert errors.empty()
    assert_cache_pair_matches(cache, "EURUSD", expected=successful)


def test_atomic_write_different_symbols_are_not_globally_serialized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = KlineCache(tmp_path, timeframe=16385)
    eurusd = canonical_dataset(1_700_000_000, 3, symbol="EURUSD")
    gbpusd = canonical_dataset(
        1_700_000_000,
        3,
        price_offset=20.0,
        symbol="GBPUSD",
    )
    eurusd_path = cache._cache_path("EURUSD")
    real_replace = kline_cache_module.os.replace
    eurusd_parquet_committed = threading.Event()
    release_eurusd = threading.Event()
    gbpusd_done = threading.Event()
    errors: list[BaseException] = []

    def observe_replace(source, destination):
        result = real_replace(source, destination)
        if (
            threading.current_thread().name == "eurusd-writer"
            and Path(destination) == eurusd_path
        ):
            eurusd_parquet_committed.set()
            if not release_eurusd.wait(10):
                raise RuntimeError("timed out waiting to release EURUSD writer")
        return result

    def write(symbol: str, dataset) -> None:
        try:
            cache._atomic_write(symbol, dataset)
        except BaseException as exc:
            errors.append(exc)
        finally:
            if symbol == "GBPUSD":
                gbpusd_done.set()

    monkeypatch.setattr("data_pipeline.kline_cache.os.replace", observe_replace)
    eurusd_thread = threading.Thread(
        target=write,
        name="eurusd-writer",
        args=("EURUSD", eurusd),
    )
    gbpusd_thread = threading.Thread(
        target=write,
        name="gbpusd-writer",
        args=("GBPUSD", gbpusd),
    )
    eurusd_thread.start()
    assert eurusd_parquet_committed.wait(10)
    gbpusd_thread.start()
    different_target_completed = gbpusd_done.wait(10)
    release_eurusd.set()
    eurusd_thread.join(10)
    gbpusd_thread.join(10)

    assert different_target_completed
    assert not eurusd_thread.is_alive()
    assert not gbpusd_thread.is_alive()
    assert errors == []
    assert_cache_pair_matches(cache, "EURUSD", expected=eurusd)
    assert_cache_pair_matches(cache, "GBPUSD", expected=gbpusd)


def test_atomic_write_same_target_is_serialized_across_spawned_processes(
    tmp_path: Path,
) -> None:
    context = mp.get_context("spawn")
    release_first_writer = context.Event()
    messages = context.Queue()
    first = context.Process(
        target=_cross_process_atomic_writer,
        args=(str(tmp_path), "A", 10.0, release_first_writer, messages),
    )
    second = context.Process(
        target=_cross_process_atomic_writer,
        args=(str(tmp_path), "B", 20.0, release_first_writer, messages),
    )
    unexpected_while_first_held: str | None = None
    remaining: list[str] = []
    try:
        first.start()
        assert messages.get(timeout=15) == "A:attempt"
        assert messages.get(timeout=15) == "A:parquet"
        second.start()
        assert messages.get(timeout=15) == "B:attempt"
        try:
            unexpected_while_first_held = messages.get(timeout=2)
        except queue.Empty:
            pass
    finally:
        release_first_writer.set()
        first.join(20)
        second.join(20)
        if first.is_alive():
            first.terminate()
            first.join(5)
        if second.is_alive():
            second.terminate()
            second.join(5)
        while True:
            try:
                remaining.append(messages.get_nowait())
            except queue.Empty:
                break
        messages.close()
        messages.join_thread()

    assert unexpected_while_first_held is None
    assert first.exitcode == 0
    assert second.exitcode == 0
    assert set(remaining) == {"A:done", "B:parquet", "B:done"}
    cache = KlineCache(tmp_path, timeframe=16385)
    expected = canonical_dataset(1_700_000_000, 3, price_offset=20.0)
    assert_cache_pair_matches(cache, "EURUSD", expected=expected)


def test_atomic_write_acquire_error_propagates_exactly_and_releases_thread_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = KlineCache(tmp_path, timeframe=16385)
    first = canonical_dataset(1_700_000_000, 3, price_offset=10.0)
    second = canonical_dataset(1_700_000_000, 3, price_offset=20.0)
    acquire_error = OSError("forced process lock acquire failure")
    real_acquire = kline_cache_module._acquire_process_lock
    acquire_calls = 0

    def fail_first_acquire(handle) -> None:
        nonlocal acquire_calls
        acquire_calls += 1
        if acquire_calls == 1:
            raise acquire_error
        real_acquire(handle)

    monkeypatch.setattr(kline_cache_module, "_acquire_process_lock", fail_first_acquire)
    with pytest.raises(OSError) as caught:
        cache._atomic_write("EURUSD", first)
    assert caught.value is acquire_error

    errors: list[BaseException] = []

    def write_after_failure() -> None:
        try:
            cache._atomic_write("EURUSD", second)
        except BaseException as exc:
            errors.append(exc)

    writer = threading.Thread(target=write_after_failure)
    writer.start()
    writer.join(10)
    assert not writer.is_alive()
    assert errors == []
    assert_cache_pair_matches(cache, "EURUSD", expected=second)


def test_atomic_write_release_error_is_observable_without_rolling_back_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = KlineCache(tmp_path, timeframe=16385)
    first = canonical_dataset(1_700_000_000, 3, price_offset=10.0)
    second = canonical_dataset(1_700_000_000, 3, price_offset=20.0)
    release_error = OSError("forced process lock release failure")
    real_release = kline_cache_module._release_process_lock
    release_calls = 0

    def fail_first_release(handle) -> None:
        nonlocal release_calls
        release_calls += 1
        if release_calls == 1:
            raise release_error
        real_release(handle)

    monkeypatch.setattr(kline_cache_module, "_release_process_lock", fail_first_release)
    with pytest.raises(OSError) as caught:
        cache._atomic_write("EURUSD", first)
    assert caught.value is release_error
    assert_cache_pair_matches(cache, "EURUSD", expected=first)

    errors: list[BaseException] = []

    def write_after_failure() -> None:
        try:
            cache._atomic_write("EURUSD", second)
        except BaseException as exc:
            errors.append(exc)

    writer = threading.Thread(target=write_after_failure)
    writer.start()
    writer.join(10)
    assert not writer.is_alive()
    assert errors == []
    assert_cache_pair_matches(cache, "EURUSD", expected=second)


def test_atomic_write_primary_baseexception_retains_release_failure_diagnostic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = KlineCache(tmp_path, timeframe=16385)
    interrupted = canonical_dataset(1_700_000_000, 3, price_offset=10.0)
    successful = canonical_dataset(1_700_000_000, 3, price_offset=20.0)
    primary = KeyboardInterrupt("forced transaction interruption")
    release_error = OSError("forced release failure after primary")
    real_to_parquet = pd.DataFrame.to_parquet
    real_release = kline_cache_module._release_process_lock
    serialization_calls = 0
    release_calls = 0

    def fail_first_serialization(self, *args, **kwargs):
        nonlocal serialization_calls
        serialization_calls += 1
        if serialization_calls == 1:
            raise primary
        return real_to_parquet(self, *args, **kwargs)

    def fail_first_release(handle) -> None:
        nonlocal release_calls
        release_calls += 1
        if release_calls == 1:
            raise release_error
        real_release(handle)

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fail_first_serialization)
    monkeypatch.setattr(kline_cache_module, "_release_process_lock", fail_first_release)
    with pytest.raises(KeyboardInterrupt) as caught:
        cache._atomic_write("EURUSD", interrupted)
    assert caught.value is primary
    notes = getattr(caught.value, "__notes__", [])
    assert len(notes) == 1
    assert "cache transaction lock release error" in notes[0]
    assert type(release_error).__name__ in notes[0]
    assert str(release_error) in notes[0]

    errors: list[BaseException] = []

    def write_after_failure() -> None:
        try:
            cache._atomic_write("EURUSD", successful)
        except BaseException as exc:
            errors.append(exc)

    writer = threading.Thread(target=write_after_failure)
    writer.start()
    writer.join(10)
    assert not writer.is_alive()
    assert errors == []
    assert_cache_pair_matches(cache, "EURUSD", expected=successful)


def test_incremental_update_serializes_read_fetch_merge_and_pair_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start = 1_700_000_000
    cache = KlineCache(tmp_path, timeframe=16385)
    initial = canonical_dataset(start, 10)
    cache._atomic_write("EURUSD", initial)
    stale = initial.frame.copy()
    writer_b_started = threading.Event()
    writer_a_done = threading.Event()
    errors: list[BaseException] = []

    remote_a = valid_rows(start + 6 * 3_600, 5)
    remote_b = valid_rows(start + 5 * 3_600, 5)
    timestamp, _, _, _, _, volume = remote_b[0]
    remote_b[0] = (timestamp, 99.0, 100.0, 98.0, 99.5, volume)

    def coordinated_copy(_module, _symbol, _timeframe, _count):
        if threading.current_thread().name == "writer-a":
            assert writer_b_started.wait(10)
            return rates(remote_a)
        assert writer_a_done.wait(10)
        return rates(remote_b)

    monkeypatch.setattr(kline_cache_module, "_copy_closed_rates", coordinated_copy)
    monkeypatch.setattr(kline_cache_module, "_MT5_AVAILABLE", True)
    monkeypatch.setattr(kline_cache_module, "mt5", object())

    def update_a() -> None:
        try:
            cache._incremental_update("EURUSD", stale)
        except BaseException as exc:
            errors.append(exc)
        finally:
            writer_a_done.set()

    def update_b() -> None:
        writer_b_started.set()
        try:
            cache._incremental_update("EURUSD", stale)
        except BaseException as exc:
            errors.append(exc)

    first = threading.Thread(target=update_a, name="writer-a")
    second = threading.Thread(target=update_b, name="writer-b")
    first.start()
    second.start()
    first.join(20)
    second.join(20)

    assert not first.is_alive()
    assert not second.is_alive()
    assert errors == []
    actual = cache.read_local("EURUSD")
    assert actual is not None
    assert epoch_seconds(actual) == [start + index * 3_600 for index in range(11)]
    revised = actual.loc[
        actual["time"] == pd.Timestamp(start + 5 * 3_600, unit="s", tz="UTC")
    ].iloc[0]
    assert revised["open"] == pytest.approx(99.0)
    assert_cache_pair_matches(cache, "EURUSD")
    assert not list(tmp_path.glob("*.tmp"))
    assert not list(tmp_path.glob("*.recovery"))
    assert not list(tmp_path.glob("*.transaction.json"))


@pytest.mark.parametrize("initial_state", ["first-create", "existing-pair"])
@pytest.mark.parametrize(
    "boundary",
    ["journal", "parquet", "metadata", "journal-removal"],
)
def test_hard_exit_at_each_publication_boundary_recovers_complete_pair(
    tmp_path: Path,
    initial_state: str,
    boundary: str,
) -> None:
    cache = KlineCache(tmp_path, timeframe=16385)
    old = canonical_dataset(1_700_000_000, 3)
    new = canonical_dataset(1_700_000_000, 3, price_offset=20.0)
    if initial_state == "existing-pair":
        cache._atomic_write("EURUSD", old)

    context = mp.get_context("spawn")
    child = context.Process(
        target=_hard_exit_atomic_writer,
        args=(str(tmp_path), boundary, 20.0),
    )
    child.start()
    child.join(30)
    if child.is_alive():
        child.terminate()
        child.join(5)
    assert child.exitcode == 73

    fresh_cache = KlineCache(tmp_path, timeframe=16385)
    recovered = fresh_cache.read_local("EURUSD")
    if boundary == "journal-removal":
        assert recovered is not None
        assert_cache_pair_matches(fresh_cache, "EURUSD", expected=new)
    elif initial_state == "existing-pair":
        assert recovered is not None
        assert_cache_pair_matches(fresh_cache, "EURUSD", expected=old)
    else:
        assert recovered is None
        assert not fresh_cache._cache_path("EURUSD").exists()
        assert not fresh_cache._metadata_path("EURUSD").exists()
    residue = [
        candidate
        for candidate in tmp_path.iterdir()
        if candidate.name.endswith((".tmp", ".recovery", ".transaction.json"))
    ]
    assert residue == []


@pytest.mark.parametrize("damage", ["metadata-mismatch", "corrupt-parquet"])
def test_orphaned_recoveries_without_journal_fail_closed_for_unproven_pair(
    tmp_path: Path,
    damage: str,
) -> None:
    cache = KlineCache(tmp_path, timeframe=16385)
    dataset = canonical_dataset(1_700_000_000, 3)
    cache._atomic_write("EURUSD", dataset)
    path = cache._cache_path("EURUSD")
    metadata_path = cache._metadata_path("EURUSD")
    journal, recoveries = cache._transaction_paths(path, metadata_path)
    assert not journal.exists()
    for target, recovery in recoveries.items():
        shutil.copy2(target, recovery)
    recovery_before = {
        candidate: candidate.read_bytes() for candidate in recoveries.values()
    }

    if damage == "metadata-mismatch":
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["gap_count"] += 1
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    else:
        path.write_bytes(b"not parquet")

    with pytest.raises(
        ExpectedCacheRecoveryError,
        match="current cache pair is not provably complete",
    ):
        cache.read_local("EURUSD")

    assert {
        candidate: candidate.read_bytes() for candidate in recoveries.values()
    } == recovery_before


def test_list_cached_surfaces_unproven_pair_and_preserves_recovery_evidence(
    tmp_path: Path,
) -> None:
    cache = KlineCache(tmp_path, timeframe=16385)
    dataset = canonical_dataset(1_700_000_000, 3)
    cache._atomic_write("EURUSD", dataset)
    path = cache._cache_path("EURUSD")
    metadata_path = cache._metadata_path("EURUSD")
    journal, recoveries = cache._transaction_paths(path, metadata_path)
    assert not journal.exists()
    for target, recovery in recoveries.items():
        shutil.copy2(target, recovery)
    recovery_before = {
        candidate: candidate.read_bytes() for candidate in recoveries.values()
    }
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["gap_count"] += 1
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(
        ExpectedCacheRecoveryError,
        match="current cache pair is not provably complete",
    ) as caught:
        cache.list_cached()

    assert type(caught.value) is ExpectedCacheRecoveryError
    assert all(str(candidate) in str(caught.value) for candidate in recoveries.values())
    assert {
        candidate: candidate.read_bytes() for candidate in recoveries.values()
    } == recovery_before


def test_list_cached_returns_valid_entries_normally(tmp_path: Path) -> None:
    cache = KlineCache(tmp_path, timeframe=16385)
    dataset = canonical_dataset(1_700_000_000, 3)
    cache._atomic_write("EURUSD", dataset)

    assert cache.list_cached() == [
        {
            "file": "EURUSD_H1.parquet",
            "bars": 3,
            "last_bar": str(dataset.frame["time"].iloc[-1]),
            "size_kb": round(cache._cache_path("EURUSD").stat().st_size / 1024, 1),
        }
    ]


def test_list_cached_discovers_valid_symbol_with_transaction_looking_suffix(
    tmp_path: Path,
) -> None:
    cache = KlineCache(tmp_path, timeframe=16385)
    symbol = "EURUSD_H1.parquet.recovery"
    dataset = canonical_dataset(1_700_000_000, 3, symbol=symbol)

    cache._atomic_write(symbol, dataset)

    path = cache._cache_path(symbol)
    assert path.exists()
    assert cache._metadata_path(symbol).exists()
    assert cache.list_cached() == [
        {
            "file": path.name,
            "bars": 3,
            "last_bar": str(dataset.frame["time"].iloc[-1]),
            "size_kb": round(path.stat().st_size / 1024, 1),
        }
    ]


@pytest.mark.parametrize("evidence_kind", ["recovery", "journal"])
def test_list_cached_suffix_symbol_absent_primary_evidence_fails_closed_without_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    evidence_kind: str,
) -> None:
    cache = KlineCache(tmp_path, timeframe=16385)
    symbol = "EURUSD_H1.parquet.recovery"
    dataset = canonical_dataset(1_700_000_000, 3)
    cache._atomic_write(symbol, dataset)
    path = cache._cache_path(symbol)
    metadata_path = cache._metadata_path(symbol)
    journal, recoveries = cache._transaction_paths(path, metadata_path)

    if evidence_kind == "recovery":
        evidence = recoveries[path]
        shutil.copy2(path, evidence)
    else:
        evidence = journal
        evidence.write_bytes(b"genuine-but-invalid-journal-evidence")
    before = evidence.read_bytes()
    path.unlink()
    metadata_path.unlink()

    real_lock = kline_cache_module._cache_transaction_lock
    locked_targets: list[Path] = []

    @kline_cache_module.contextmanager
    def recording_lock(target: Path):
        locked_targets.append(target)
        with real_lock(target):
            yield

    monkeypatch.setattr(kline_cache_module, "_cache_transaction_lock", recording_lock)

    with pytest.raises(ExpectedCacheRecoveryError) as caught:
        cache.list_cached()

    assert type(caught.value) is ExpectedCacheRecoveryError
    assert str(evidence) in str(caught.value)
    assert evidence.read_bytes() == before
    assert locked_targets == [path]
    assert not (tmp_path / ".EURUSD_H1.parquet.lock").exists()


def test_list_cached_propagates_pair_read_failure_instead_of_broad_swallowing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = KlineCache(tmp_path, timeframe=16385)
    cache._atomic_write("EURUSD", canonical_dataset(1_700_000_000, 3))

    def fail_display_read(*args, **kwargs):
        raise ArrowInvalid("display inspection failed")

    monkeypatch.setattr(pd, "read_parquet", fail_display_read)

    with pytest.raises(ExpectedCacheRecoveryError, match="cache pair") as caught:
        cache.list_cached()

    assert type(caught.value.__cause__) is kline_cache_module.CacheReadError


def test_list_cached_propagates_exact_final_stat_oserror_after_pair_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = KlineCache(tmp_path, timeframe=16385)
    cache._atomic_write("EURUSD", canonical_dataset(1_700_000_000, 3))
    target = cache._cache_path("EURUSD")
    sentinel = OSError("final display stat failed")
    real_stat = Path.stat
    target_calls = 0

    def fail_third_target_stat(path: Path, *args, **kwargs):
        nonlocal target_calls
        if path == target:
            target_calls += 1
            if target_calls == 3:
                raise sentinel
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", fail_third_target_stat)

    with pytest.raises(OSError) as caught:
        cache.list_cached()

    assert target_calls == 3
    assert caught.value is sentinel


def test_list_cached_propagates_exact_cache_directory_iteration_oserror(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = KlineCache(tmp_path, timeframe=16385)
    sentinel = OSError("cache directory iteration failed")
    real_iterdir = Path.iterdir

    def fail_cache_directory_iteration(path: Path):
        if path == tmp_path:
            raise sentinel
        return real_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", fail_cache_directory_iteration)

    with pytest.raises(OSError) as caught:
        cache.list_cached()

    assert caught.value is sentinel


@pytest.mark.parametrize(
    "evidence_kind",
    ["parquet-recovery", "metadata-recovery", "both-recoveries"],
)
def test_list_cached_discovers_orphan_recovery_when_primary_pair_is_absent(
    tmp_path: Path,
    evidence_kind: str,
) -> None:
    cache = KlineCache(tmp_path, timeframe=16385)
    path = cache._cache_path("EURUSD")
    metadata_path = cache._metadata_path("EURUSD")
    _, recoveries = cache._transaction_paths(path, metadata_path)
    evidence = {
        "parquet-recovery": [recoveries[path]],
        "metadata-recovery": [recoveries[metadata_path]],
        "both-recoveries": list(recoveries.values()),
    }[evidence_kind]
    before: dict[Path, bytes] = {}
    for index, candidate in enumerate(evidence):
        payload = f"retained-evidence-{index}".encode()
        candidate.write_bytes(payload)
        before[candidate] = payload

    with pytest.raises(ExpectedCacheRecoveryError) as caught:
        cache.list_cached()

    assert type(caught.value) is ExpectedCacheRecoveryError
    assert all(str(candidate) in str(caught.value) for candidate in evidence)
    assert {candidate: candidate.read_bytes() for candidate in evidence} == before


def test_list_cached_discovers_fresh_process_first_create_journal_without_primary(
    tmp_path: Path,
) -> None:
    context = mp.get_context("spawn")
    child = context.Process(
        target=_hard_exit_atomic_writer,
        args=(str(tmp_path), "journal", 20.0),
    )
    child.start()
    child.join(30)
    if child.is_alive():
        child.terminate()
        child.join(5)
    assert child.exitcode == 73

    cache = KlineCache(tmp_path, timeframe=16385)
    path = cache._cache_path("EURUSD")
    metadata_path = cache._metadata_path("EURUSD")
    journal, _ = cache._transaction_paths(path, metadata_path)
    assert journal.exists()
    assert not path.exists()
    assert not metadata_path.exists()

    assert cache.list_cached() == []
    assert not journal.exists()
    assert not path.exists()
    assert not metadata_path.exists()


def test_list_cached_deduplicates_artifacts_and_locks_canonical_target_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = KlineCache(tmp_path, timeframe=16385)
    dataset = canonical_dataset(1_700_000_000, 3)
    cache._atomic_write("EURUSD", dataset)
    path = cache._cache_path("EURUSD")
    metadata_path = cache._metadata_path("EURUSD")
    _, recoveries = cache._transaction_paths(path, metadata_path)
    for target, recovery in recoveries.items():
        shutil.copy2(target, recovery)

    real_lock = kline_cache_module._cache_transaction_lock
    locked_targets: list[Path] = []

    @kline_cache_module.contextmanager
    def recording_lock(target: Path):
        locked_targets.append(target)
        with real_lock(target):
            yield

    monkeypatch.setattr(kline_cache_module, "_cache_transaction_lock", recording_lock)

    assert cache.list_cached()[0]["file"] == path.name
    assert locked_targets == [path]


def test_list_cached_ignores_unrelated_hidden_files(tmp_path: Path) -> None:
    cache = KlineCache(tmp_path, timeframe=16385)
    unrelated = {
        ".notes.recovery": b"notes",
        ".EURUSD_H1.parquet.recovery.extra": b"extra",
        ".EURUSD_H1.metadata.json.recovery.extra": b"extra-metadata",
        ".EURUSD_H1.parquet.transaction.json.backup": b"backup",
        ".EURUSD_M30.parquet.recovery": b"wrong-timeframe",
    }
    for name, payload in unrelated.items():
        (tmp_path / name).write_bytes(payload)

    assert cache.list_cached() == []
    assert {name: (tmp_path / name).read_bytes() for name in unrelated} == unrelated


def test_list_cached_ignores_double_hidden_parquet_recovery_byte_exactly(
    tmp_path: Path,
) -> None:
    cache = KlineCache(tmp_path, timeframe=16385)
    evidence = tmp_path / "..EURUSD_H1.parquet.recovery"
    payload = b"unrelated-evidence"
    evidence.write_bytes(payload)

    assert cache.list_cached() == []
    assert evidence.read_bytes() == payload


@pytest.mark.parametrize(
    "name",
    [
        "..EURUSD_H1.parquet.transaction.json",
        "...EURUSD_H1.parquet.transaction.json",
        "..EURUSD_H1.parquet.recovery",
        "....EURUSD_H1.parquet.recovery",
        "..EURUSD_H1.metadata.json.recovery",
        "...EURUSD_H1.metadata.json.recovery",
    ],
    ids=[
        "double-hidden-journal",
        "multiple-hidden-journal",
        "double-hidden-parquet-recovery",
        "multiple-hidden-parquet-recovery",
        "double-hidden-metadata-recovery",
        "multiple-hidden-metadata-recovery",
    ],
)
def test_list_cached_ignores_double_and_multiple_hidden_transaction_names(
    tmp_path: Path,
    name: str,
) -> None:
    cache = KlineCache(tmp_path, timeframe=16385)
    evidence = tmp_path / name
    payload = f"unrelated:{name}".encode()
    evidence.write_bytes(payload)

    assert cache.list_cached() == []
    assert evidence.read_bytes() == payload


@pytest.mark.parametrize(
    "name",
    [
        "._H1.parquet.transaction.json",
        ".._H1.parquet.recovery",
        ".._H1.metadata.json.recovery",
        "..._H1.parquet.transaction.json",
        "...._H1.parquet.recovery",
        "..._H1.metadata.json.recovery",
        "...._H1.metadata.json.recovery",
    ],
    ids=[
        "empty-symbol-journal",
        "hidden-symbol-parquet-recovery",
        "hidden-symbol-metadata-recovery",
        "multiple-hidden-empty-symbol-journal",
        "multiple-hidden-empty-symbol-parquet-recovery",
        "multiple-hidden-empty-symbol-metadata-recovery",
        "many-hidden-empty-symbol-metadata-recovery",
    ],
)
def test_list_cached_ignores_noncanonical_derived_targets(
    tmp_path: Path,
    name: str,
) -> None:
    cache = KlineCache(tmp_path, timeframe=16385)
    evidence = tmp_path / name
    payload = f"malformed:{name}".encode()
    evidence.write_bytes(payload)

    assert cache.list_cached() == []
    assert evidence.read_bytes() == payload


def test_malformed_recovery_names_do_not_change_valid_listing_or_lock_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = KlineCache(tmp_path, timeframe=16385)
    dataset = canonical_dataset(1_700_000_000, 3)
    cache._atomic_write("EURUSD", dataset)
    malformed = {
        "..GBPUSD_H1.parquet.recovery": b"double-hidden",
        "...GBPUSD_H1.parquet.recovery": b"multiple-hidden",
        "..GBPUSD_H1.metadata.json.recovery": b"metadata",
    }
    for name, payload in malformed.items():
        (tmp_path / name).write_bytes(payload)

    real_lock = kline_cache_module._cache_transaction_lock
    locked_targets: list[Path] = []

    @kline_cache_module.contextmanager
    def recording_lock(target: Path):
        locked_targets.append(target)
        with real_lock(target):
            yield

    monkeypatch.setattr(kline_cache_module, "_cache_transaction_lock", recording_lock)

    listing = cache.list_cached()

    assert [entry["file"] for entry in listing] == ["EURUSD_H1.parquet"]
    assert locked_targets == [cache._cache_path("EURUSD")]
    assert {name: (tmp_path / name).read_bytes() for name in malformed} == malformed


def test_valid_recovery_error_still_propagates_with_malformed_names_present(
    tmp_path: Path,
) -> None:
    cache = KlineCache(tmp_path, timeframe=16385)
    path = cache._cache_path("EURUSD")
    metadata_path = cache._metadata_path("EURUSD")
    _, recoveries = cache._transaction_paths(path, metadata_path)
    valid_evidence = recoveries[path]
    valid_payload = b"genuine-recovery-evidence"
    valid_evidence.write_bytes(valid_payload)
    malformed = tmp_path / "..GBPUSD_H1.parquet.recovery"
    malformed_payload = b"unrelated-malformed-evidence"
    malformed.write_bytes(malformed_payload)

    with pytest.raises(ExpectedCacheRecoveryError) as caught:
        cache.list_cached()

    assert type(caught.value) is ExpectedCacheRecoveryError
    assert str(valid_evidence) in str(caught.value)
    assert valid_evidence.read_bytes() == valid_payload
    assert malformed.read_bytes() == malformed_payload


@pytest.mark.parametrize(
    ("wrong_symbol", "wrong_timeframe"),
    [("GBPUSD", "H1"), ("EURUSD", "M30")],
    ids=["wrong-symbol", "wrong-timeframe"],
)
def test_orphaned_recoveries_without_journal_reject_self_consistent_wrong_target(
    tmp_path: Path,
    wrong_symbol: str,
    wrong_timeframe: str,
) -> None:
    cache = KlineCache(tmp_path, timeframe=16385)
    requested = canonical_dataset(1_700_000_000, 3)
    cache._atomic_write("EURUSD", requested)
    path = cache._cache_path("EURUSD")
    metadata_path = cache._metadata_path("EURUSD")
    journal, recoveries = cache._transaction_paths(path, metadata_path)
    assert not journal.exists()
    for target, recovery in recoveries.items():
        shutil.copy2(target, recovery)
    recovery_before = {
        candidate: candidate.read_bytes() for candidate in recoveries.values()
    }

    misleading = canonicalize_ohlcv(
        pd.read_parquet(path),
        symbol=wrong_symbol,
        timeframe=wrong_timeframe,
    )
    metadata_path.write_text(
        json.dumps(cache._metadata(misleading), sort_keys=True),
        encoding="utf-8",
    )
    misleading_metadata = metadata_path.read_bytes()

    fresh_cache = KlineCache(tmp_path, timeframe=16385)
    with pytest.raises(
        ExpectedCacheRecoveryError,
        match="current cache pair is not provably complete",
    ):
        fresh_cache.read_local("EURUSD")

    assert metadata_path.read_bytes() == misleading_metadata
    assert {
        candidate: candidate.read_bytes() for candidate in recoveries.values()
    } == recovery_before


@pytest.mark.skipif(os.name != "nt", reason="real msvcrt contention contract")
def test_windows_process_lock_waits_beyond_ten_seconds_until_release(
    tmp_path: Path,
) -> None:
    context = mp.get_context("spawn")
    ready = context.Event()
    release = context.Event()
    messages = context.Queue()
    holder = context.Process(target=_hold_cache_lock, args=(str(tmp_path), ready, release))
    contender = context.Process(target=_timed_atomic_writer, args=(str(tmp_path), messages))
    try:
        holder.start()
        assert ready.wait(15)
        contender.start()
        assert messages.get(timeout=15)[0] == "started"
        contender.join(10.5)
        assert contender.is_alive(), "contender stopped before the holder released the lock"
        release.set()
        holder.join(15)
        contender.join(15)
        outcome = messages.get(timeout=5)
    finally:
        release.set()
        for process in (holder, contender):
            if process.is_alive():
                process.terminate()
            process.join(5)
        messages.close()
        messages.join_thread()

    assert holder.exitcode == 0
    assert contender.exitcode == 0
    assert outcome[0] == "ok"
    assert outcome[3] >= 10.0
    assert_cache_pair_matches(KlineCache(tmp_path, timeframe=16385), "EURUSD")


@pytest.mark.skipif(os.name != "nt", reason="msvcrt retry taxonomy is Windows-only")
def test_windows_process_lock_retries_only_genuine_contention(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import msvcrt

    lock_file = tmp_path / "retry.lock"
    lock_file.write_bytes(b"\0")
    calls = 0

    def contend_repeatedly(_fd, mode, _length):
        nonlocal calls
        assert mode == msvcrt.LK_NBLCK
        calls += 1
        if calls <= 25:
            raise OSError(errno.EACCES, "sharing violation")

    monkeypatch.setattr(msvcrt, "locking", contend_repeatedly)
    monkeypatch.setattr(kline_cache_module.time, "sleep", lambda _delay: None)
    with lock_file.open("r+b") as handle:
        kline_cache_module._acquire_process_lock(handle)
    assert calls == 26

    unrelated = OSError(errno.EINVAL, "unrelated acquire failure")
    monkeypatch.setattr(msvcrt, "locking", lambda *_args: (_ for _ in ()).throw(unrelated))
    with lock_file.open("r+b") as handle, pytest.raises(OSError) as caught:
        kline_cache_module._acquire_process_lock(handle)
    assert caught.value is unrelated


def test_target_lock_registry_returns_to_baseline_after_100k_unique_targets(
    tmp_path: Path,
) -> None:
    baseline = len(kline_cache_module._TARGET_THREAD_LOCKS)
    for index in range(100_000):
        key, _ = kline_cache_module._normalized_cache_target(
            tmp_path / f"symbol-{index}.parquet"
        )
        entry = kline_cache_module._reserve_target_thread_lock(key)
        kline_cache_module._release_target_thread_lock(key, entry)
    assert len(kline_cache_module._TARGET_THREAD_LOCKS) == baseline


def test_target_lock_registry_keeps_one_entry_for_holder_and_waiter(
    tmp_path: Path,
) -> None:
    baseline = len(kline_cache_module._TARGET_THREAD_LOCKS)
    target = tmp_path / "EURUSD_H1.parquet"
    holder_ready = threading.Event()
    release_holder = threading.Event()
    waiter_started = threading.Event()
    waiter_done = threading.Event()
    thread_errors: queue.Queue[BaseException] = queue.Queue()

    def holder() -> None:
        with kline_cache_module._cache_transaction_lock(target):
            holder_ready.set()
            assert release_holder.wait(10)

    def waiter() -> None:
        waiter_started.set()
        with kline_cache_module._cache_transaction_lock(target):
            waiter_done.set()

    def capture_worker_error(worker) -> None:
        try:
            worker()
        except BaseException as exc:
            thread_errors.put(exc)

    first = threading.Thread(target=capture_worker_error, args=(holder,))
    second = threading.Thread(target=capture_worker_error, args=(waiter,))
    try:
        first.start()
        assert holder_ready.wait(10)
        second.start()
        assert waiter_started.wait(10)
        time.sleep(0.05)
        assert len(kline_cache_module._TARGET_THREAD_LOCKS) == baseline + 1
        assert not waiter_done.is_set()
    finally:
        release_holder.set()
        first.join(10)
        second.join(10)
    assert not first.is_alive()
    assert not second.is_alive()
    errors: list[BaseException] = []
    while not thread_errors.empty():
        errors.append(thread_errors.get_nowait())
    assert errors == []
    assert waiter_done.is_set()
    assert len(kline_cache_module._TARGET_THREAD_LOCKS) == baseline


def test_normalized_cache_target_coalesces_path_aliases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "EURUSD_H1.parquet"
    monkeypatch.chdir(tmp_path.parent)
    relative = Path(tmp_path.name) / target.name
    absolute_key, _ = kline_cache_module._normalized_cache_target(target)
    relative_key, _ = kline_cache_module._normalized_cache_target(relative)
    assert relative_key == absolute_key

    alias_dir = tmp_path.parent / f"{tmp_path.name}-alias"
    try:
        os.symlink(tmp_path, alias_dir, target_is_directory=True)
    except OSError:
        return
    try:
        alias_key, _ = kline_cache_module._normalized_cache_target(alias_dir / target.name)
        assert alias_key == absolute_key
    finally:
        alias_dir.rmdir()
