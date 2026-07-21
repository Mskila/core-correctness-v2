"""
data_pipeline/kline_cache.py — 本地 K 线缓存管理器

设计：
  - 每个品种存一个 Parquet 文件：D:/K线数据/{symbol}_H1.parquet
  - 列：time(UTC), open, high, low, close, volume
  - 首次：从 MT5 拉取仅闭合历史（BARS_COUNT 根）并规范化写入
  - 后续：重拉闭合尾部并按时间戳 upsert；远端修订覆盖同时间戳本地值
  - 无 MT5 连接时：直接读本地文件（供查询/分析使用）

用法：
    cache = KlineCache()
    df = cache.get(symbol)          # 优先读本地，按需增量更新
    df = cache.get(symbol, force_refresh=True)  # 强制重拉全量
    cache.update_all(symbols)       # 批量更新
"""
from __future__ import annotations

from contextlib import contextmanager
import errno
import json
import os
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import BinaryIO, Iterator, Optional

import pandas as pd
from loguru import logger

from data_pipeline.validation import CanonicalDataset, canonicalize_ohlcv
from model_core.semantics import DataValidationError

_PARQUET_READ_ERRORS: tuple[type[BaseException], ...] = (OSError,)
try:
    from pyarrow.lib import ArrowException as _ArrowException
except ImportError:
    pass
else:
    _PARQUET_READ_ERRORS += (_ArrowException,)

class _TargetThreadLock:
    __slots__ = ("lock", "users")

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.users = 0


_TARGET_THREAD_LOCKS: dict[str, _TargetThreadLock] = {}
_TARGET_THREAD_LOCKS_GUARD = threading.Lock()


def _normalized_cache_target(target: Path) -> tuple[str, Path]:
    absolute = os.path.abspath(os.fspath(target))
    canonical = os.path.realpath(absolute)
    normalized = os.path.normcase(os.path.normpath(canonical))
    return normalized, Path(canonical)


def _reserve_target_thread_lock(normalized_target: str) -> _TargetThreadLock:
    with _TARGET_THREAD_LOCKS_GUARD:
        entry = _TARGET_THREAD_LOCKS.get(normalized_target)
        if entry is None:
            entry = _TargetThreadLock()
            _TARGET_THREAD_LOCKS[normalized_target] = entry
        entry.users += 1
        return entry


def _release_target_thread_lock(
    normalized_target: str,
    entry: _TargetThreadLock,
) -> None:
    with _TARGET_THREAD_LOCKS_GUARD:
        current = _TARGET_THREAD_LOCKS.get(normalized_target)
        if current is not entry or entry.users < 1:
            raise RuntimeError("cache target thread-lock registry accounting corrupted")
        entry.users -= 1
        if entry.users == 0:
            del _TARGET_THREAD_LOCKS[normalized_target]


def _acquire_process_lock(handle: BinaryIO) -> None:
    handle.seek(0)
    if os.name == "nt":
        import msvcrt

        contention_errnos = {errno.EACCES, errno.EAGAIN, errno.EDEADLK}
        contention_winerrors = {32, 33, 36}
        while True:
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                break
            except OSError as exc:
                if (
                    exc.errno not in contention_errnos
                    and getattr(exc, "winerror", None) not in contention_winerrors
                ):
                    raise
                time.sleep(0.05)
        return

    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)


def _release_process_lock(handle: BinaryIO) -> None:
    handle.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        return

    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def _cache_transaction_lock(target: Path) -> Iterator[None]:
    normalized_target, canonical_target = _normalized_cache_target(target)
    thread_entry = _reserve_target_thread_lock(normalized_target)
    thread_acquired = False
    process_handle: BinaryIO | None = None
    process_acquired = False
    primary_error: BaseException | None = None
    release_errors: list[BaseException] = []

    try:
        thread_entry.lock.acquire()
        thread_acquired = True
        lock_path = canonical_target.with_name(f".{canonical_target.name}.lock")
        process_handle = lock_path.open("a+b")
        process_handle.seek(0, os.SEEK_END)
        if process_handle.tell() == 0:
            process_handle.write(b"\0")
            process_handle.flush()
        _acquire_process_lock(process_handle)
        process_acquired = True
        yield
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        if process_acquired and process_handle is not None:
            try:
                _release_process_lock(process_handle)
            except BaseException as exc:
                release_errors.append(exc)
        if process_handle is not None:
            try:
                process_handle.close()
            except BaseException as exc:
                release_errors.append(exc)
        if thread_acquired:
            try:
                thread_entry.lock.release()
            except BaseException as exc:
                release_errors.append(exc)
        try:
            _release_target_thread_lock(normalized_target, thread_entry)
        except BaseException as exc:
            release_errors.append(exc)
        if release_errors:
            details = "; ".join(
                f"{type(exc).__name__}: {exc}" for exc in release_errors
            )
            if primary_error is None:
                first_release_error = release_errors[0]
                for extra_error in release_errors[1:]:
                    first_release_error.add_note(
                        "additional cache transaction lock release error: "
                        f"{type(extra_error).__name__}: {extra_error}"
                    )
                raise first_release_error
            primary_error.add_note(
                f"cache transaction lock release error for {canonical_target}: {details}"
            )
            logger.error(
                f"[Cache] transaction lock release failed for {canonical_target}: "
                f"{details}"
            )

try:
    import MetaTrader5 as mt5
    _MT5_AVAILABLE = True
except ImportError:
    _MT5_AVAILABLE = False
    mt5 = None

try:
    from config import Config
    _TIMEFRAME = Config.TIMEFRAME
    _BARS_COUNT = Config.BARS_COUNT
    _FALLBACK_EXECUTION_LAG_BARS = 1
except ImportError:
    Config = None  # type: ignore[assignment]
    _TIMEFRAME = 16385   # H1
    _BARS_COUNT = 12000
    _FALLBACK_EXECUTION_LAG_BARS = 1


def _default_cache_dir() -> Path:
    try:
        from config import Config
        return Path(getattr(Config, "KLINE_CACHE_DIR", r"D:\K线数据"))
    except ImportError:
        return Path(r"D:\K线数据")

_COLUMNS = ["time", "open", "high", "low", "close", "tick_volume"]


def _copy_closed_rates(mt5_module, symbol: str, timeframe: int, count: int):
    """Copy only completed rates; MT5 position zero is the forming bar."""
    return mt5_module.copy_rates_from_pos(symbol, timeframe, 1, count)


def _closed_rates_are_empty(rates) -> bool:
    if rates is None:
        return True
    try:
        return len(rates) == 0
    except (OverflowError, TypeError) as exc:
        raise DataValidationError(f"malformed MT5 rates payload: {exc}") from exc


def _canonicalize_closed_rates(
    rates, *, symbol: str, timeframe: int
) -> CanonicalDataset:
    """Convert an untrusted MT5 payload into the single canonical OHLCV form."""
    try:
        frame = pd.DataFrame(rates)
    except Exception as exc:
        raise DataValidationError(f"malformed MT5 rates payload: {exc}") from exc
    missing = [column for column in _COLUMNS if column not in frame.columns]
    if missing:
        raise DataValidationError(f"missing columns: {missing}")
    try:
        return canonicalize_ohlcv(
            frame.loc[:, _COLUMNS],
            symbol=symbol,
            timeframe=timeframe,
            numeric_time_unit="s",
        )
    except DataValidationError:
        raise
    except (KeyError, OverflowError, TypeError, ValueError) as exc:
        raise DataValidationError(f"malformed MT5 rates payload: {exc}") from exc


def _revision_bar_count() -> int:
    lag = (
        getattr(Config, "EXECUTION_LAG_BARS", _FALLBACK_EXECUTION_LAG_BARS)
        if Config is not None
        else _FALLBACK_EXECUTION_LAG_BARS
    )
    return max(5, int(lag) + 2)


class CacheReadError(OSError):
    """The local parquet could not be read from storage."""


class CacheRecoveryError(RuntimeError):
    """A cache update failed and automatic restoration could not complete."""


class _CachePairReadRecoveryError(CacheRecoveryError, CacheReadError):
    """An untrusted pair cannot be read and a fetcher may use direct closed bars."""


class _CachePairValidationRecoveryError(CacheRecoveryError, DataValidationError):
    """A readable pair violates canonical data validation and must fail closed."""


def _empty_canonical_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": pd.Series(dtype="datetime64[ns, UTC]"),
            "open": pd.Series(dtype="float64"),
            "high": pd.Series(dtype="float64"),
            "low": pd.Series(dtype="float64"),
            "close": pd.Series(dtype="float64"),
            "volume": pd.Series(dtype="float64"),
        }
    )


def _fsync_file(path: Path) -> None:
    with path.open("r+b") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        if os.name == "nt":
            return
        raise
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class KlineCache:
    """本地 K 线缓存管理器，支持增量更新。"""

    def __init__(
        self,
        cache_dir:  str | Path | None = None,
        timeframe:  int         = _TIMEFRAME,
        bars_count: int         = _BARS_COUNT,
    ) -> None:
        self.cache_dir  = Path(cache_dir) if cache_dir is not None else _default_cache_dir()
        self.timeframe  = timeframe
        self.bars_count = bars_count
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, symbol: str) -> Path:
        tf_name = {16385: "H1", 16388: "H4", 16408: "D1", 1: "M1", 5: "M5"}.get(
            self.timeframe, f"TF{self.timeframe}"
        )
        return self.cache_dir / f"{symbol}_{tf_name}.parquet"

    def _metadata_path(self, symbol: str) -> Path:
        return self._cache_path(symbol).with_suffix(".metadata.json")

    def _read_local_dataset(self, symbol: str, path: Path) -> CanonicalDataset:
        try:
            frame = pd.read_parquet(path)
        except _PARQUET_READ_ERRORS as exc:
            raise CacheReadError(f"cache read failed for {path}: {exc}") from exc
        return canonicalize_ohlcv(
            frame,
            symbol=symbol,
            timeframe=self.timeframe,
        )

    def _read_validated_pair_locked(
        self,
        symbol: str,
        path: Path,
        metadata_path: Path,
    ) -> CanonicalDataset | None:
        """Read and prove one complete cache pair while its target lock is held."""
        if not path.exists() and not metadata_path.exists():
            return None
        if not path.exists():
            raise _CachePairReadRecoveryError(
                f"cache pair validation failed for {path}: "
                f"cache pair is incomplete; missing {path}"
            )
        try:
            dataset = self._read_local_dataset(symbol, path)
        except CacheReadError as exc:
            raise _CachePairReadRecoveryError(
                f"cache pair validation failed for {path}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        except DataValidationError as exc:
            raise _CachePairValidationRecoveryError(
                f"cache pair validation failed for {path}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if not metadata_path.exists():
            raise _CachePairReadRecoveryError(
                f"cache pair validation failed for {path}: "
                f"cache pair is incomplete; missing {metadata_path}"
            )
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if type(metadata) is not dict:
                raise ValueError("cache pair metadata must be a JSON object")
            expected = self._metadata(dataset)
            if metadata.keys() != expected.keys():
                missing = sorted(expected.keys() - metadata.keys())
                unknown = sorted(metadata.keys() - expected.keys())
                raise ValueError(
                    "cache pair metadata fields mismatch; "
                    f"missing={missing} unknown={unknown}"
                )
            mismatched = [
                field
                for field, expected_value in expected.items()
                if type(metadata[field]) is not type(expected_value)
                or metadata[field] != expected_value
            ]
            if mismatched:
                raise ValueError(
                    f"cache pair metadata identity mismatch: {mismatched}"
                )
            return dataset
        except (
            DataValidationError,
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            ValueError,
        ) as exc:
            raise CacheRecoveryError(
                f"cache pair validation failed for {path}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    def _read_local_with_recovery(self, symbol: str) -> Optional[pd.DataFrame]:
        path = self._cache_path(symbol)
        metadata_path = self._metadata_path(symbol)
        with _cache_transaction_lock(path):
            self._recover_abandoned_transaction(
                path, metadata_path, expected_symbol=symbol
            )
            dataset = self._read_validated_pair_locked(symbol, path, metadata_path)
            return None if dataset is None else dataset.frame

    def _canonicalize_rates(self, symbol: str, rates) -> CanonicalDataset:
        return _canonicalize_closed_rates(
            rates,
            symbol=symbol,
            timeframe=self.timeframe,
        )

    @staticmethod
    def _metadata(dataset: CanonicalDataset) -> dict[str, str | int]:
        return {**dataset.identity.to_dict(), "gap_count": dataset.gap_count}

    def _atomic_write(
        self,
        symbol: str,
        dataset: CanonicalDataset,
        *,
        _lock_held: bool = False,
    ) -> None:
        path = self._cache_path(symbol)
        metadata_path = self._metadata_path(symbol)
        if _lock_held:
            self._atomic_write_transaction(path, metadata_path, dataset)
            return
        with _cache_transaction_lock(path):
            self._recover_abandoned_transaction(
                path, metadata_path, expected_symbol=symbol
            )
            self._atomic_write_transaction(path, metadata_path, dataset)

    @staticmethod
    def _transaction_paths(
        path: Path,
        metadata_path: Path,
    ) -> tuple[Path, dict[Path, Path]]:
        journal = path.with_name(f".{path.name}.transaction.json")
        recoveries = {
            path: path.with_name(f".{path.name}.recovery"),
            metadata_path: metadata_path.with_name(f".{metadata_path.name}.recovery"),
        }
        return journal, recoveries

    def _recover_abandoned_transaction(
        self,
        path: Path,
        metadata_path: Path,
        *,
        expected_symbol: str,
    ) -> None:
        journal, recoveries = self._transaction_paths(path, metadata_path)
        if not journal.exists():
            orphaned_recoveries = [
                candidate for candidate in recoveries.values() if candidate.exists()
            ]
            if not orphaned_recoveries:
                return
            try:
                if self._read_validated_pair_locked(
                    expected_symbol, path, metadata_path
                ) is None:
                    raise ValueError("current cache pair is absent")
            except Exception as exc:
                locations = ", ".join(str(candidate) for candidate in orphaned_recoveries)
                raise CacheRecoveryError(
                    "cache recovery failed; retained orphaned recovery evidence at "
                    f"{locations}; errors: current cache pair is not provably complete: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            cleanup_errors: list[str] = []
            for candidate in orphaned_recoveries:
                try:
                    candidate.unlink()
                except BaseException as exc:
                    cleanup_errors.append(
                        f"{candidate}: {type(exc).__name__}: {exc}"
                    )
            if cleanup_errors:
                raise CacheRecoveryError(
                    "cache recovery failed; retained orphaned recovery evidence; "
                    f"errors: {'; '.join(cleanup_errors)}"
                )
            _fsync_directory(path.parent)
            return

        try:
            payload = json.loads(journal.read_text(encoding="utf-8"))
            if payload.get("version") != 1:
                raise ValueError("unsupported transaction journal version")
            if payload.get("parquet") != path.name:
                raise ValueError("transaction parquet target mismatch")
            if payload.get("metadata") != metadata_path.name:
                raise ValueError("transaction metadata target mismatch")
            existed = {
                path: payload["parquet_existed"],
                metadata_path: payload["metadata_existed"],
            }
            if any(type(value) is not bool for value in existed.values()):
                raise ValueError("transaction existence flags must be booleans")
            temporary = [
                path.parent / payload["parquet_temp"],
                path.parent / payload["metadata_temp"],
            ]
            if any(candidate.parent != path.parent for candidate in temporary):
                raise ValueError("transaction temporary path escaped cache directory")
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise CacheRecoveryError(
                f"cache recovery failed; retained evidence at {journal}; "
                f"errors: invalid transaction journal: {type(exc).__name__}: {exc}"
            ) from exc

        errors: list[str] = []
        evidence: list[Path] = [journal]

        def cleanup(candidate: Path, *, kind: str) -> bool:
            try:
                candidate.unlink(missing_ok=True)
                return True
            except BaseException as exc:
                logger.warning(
                    f"[Cache] 无法清理 {kind} 文件 {candidate}；保留供后续诊断: {exc}"
                )
                return False

        def restore(target: Path, recovery: Path) -> BaseException | None:
            last_error: BaseException | None = None
            for _ in range(2):
                descriptor, name = tempfile.mkstemp(
                    dir=target.parent,
                    prefix=f".{target.name}.",
                    suffix=".tmp",
                )
                os.close(descriptor)
                restore_temp = Path(name)
                try:
                    shutil.copy2(recovery, restore_temp)
                    _fsync_file(restore_temp)
                    os.replace(restore_temp, target)
                    _fsync_directory(target.parent)
                    return None
                except BaseException as exc:
                    last_error = exc
                finally:
                    cleanup(restore_temp, kind="rollback temporary")
            return last_error

        for target in (metadata_path, path):
            recovery = recoveries[target]
            if existed[target]:
                if not recovery.exists():
                    errors.append(f"{target}: missing recovery copy {recovery}")
                    continue
                evidence.append(recovery)
                restore_error = restore(target, recovery)
                if restore_error is not None:
                    errors.append(
                        f"{target}: {type(restore_error).__name__}: {restore_error}"
                    )
            elif target.exists():
                try:
                    shutil.copy2(target, recovery)
                    _fsync_file(recovery)
                    evidence.append(recovery)
                    target.unlink()
                    _fsync_directory(target.parent)
                except BaseException as exc:
                    errors.append(f"{target}: {type(exc).__name__}: {exc}")

        if errors:
            locations = ", ".join(
                str(candidate)
                for candidate in dict.fromkeys(evidence)
                if candidate.exists()
            )
            raise CacheRecoveryError(
                "cache recovery failed; retained evidence at "
                f"{locations}; errors: {'; '.join(errors)}"
            )

        try:
            journal.unlink()
            _fsync_directory(journal.parent)
        except BaseException as exc:
            raise CacheRecoveryError(
                f"cache recovery failed; retained evidence at {journal}; "
                f"errors: journal cleanup {type(exc).__name__}: {exc}"
            ) from exc
        for candidate in [*temporary, *recoveries.values()]:
            cleanup(candidate, kind="transaction recovery")

    def _atomic_write_transaction(
        self,
        path: Path,
        metadata_path: Path,
        dataset: CanonicalDataset,
    ) -> None:
        temporary: list[Path] = []
        journal, recoveries = self._transaction_paths(path, metadata_path)
        journal_published = False
        commit_succeeded = False

        def temporary_path(target: Path, *, suffix: str = ".tmp") -> Path:
            descriptor, name = tempfile.mkstemp(
                dir=target.parent,
                prefix=f".{target.name}.",
                suffix=suffix,
            )
            os.close(descriptor)
            result = Path(name)
            if suffix == ".tmp":
                temporary.append(result)
            return result

        def cleanup(candidate: Path, *, kind: str) -> bool:
            try:
                candidate.unlink(missing_ok=True)
                return True
            except BaseException as exc:
                logger.warning(
                    f"[Cache] 无法清理 {kind} 文件 {candidate}；保留供后续诊断: {exc}"
                )
                return False

        parquet_temp = temporary_path(path)
        metadata_temp = temporary_path(metadata_path)
        try:
            dataset.frame.to_parquet(parquet_temp, index=False)
            metadata_temp.write_text(
                json.dumps(
                    self._metadata(dataset),
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            _fsync_file(parquet_temp)
            _fsync_file(metadata_temp)
            existed = {path: path.exists(), metadata_path: metadata_path.exists()}
            for target in (path, metadata_path):
                if existed[target]:
                    recovery = recoveries[target]
                    try:
                        shutil.copy2(target, recovery)
                        _fsync_file(recovery)
                    except Exception:
                        cleanup(recovery, kind="incomplete recovery")
                        raise
            descriptor, journal_temp_name = tempfile.mkstemp(
                dir=path.parent,
                prefix=f".{path.name}.journal.",
                suffix=".tmp",
            )
            os.close(descriptor)
            journal_temp = Path(journal_temp_name)
            temporary.append(journal_temp)
            journal_temp.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "parquet": path.name,
                        "metadata": metadata_path.name,
                        "parquet_existed": existed[path],
                        "metadata_existed": existed[metadata_path],
                        "parquet_temp": parquet_temp.name,
                        "metadata_temp": metadata_temp.name,
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            _fsync_file(journal_temp)
            os.rename(journal_temp, journal)
            temporary.remove(journal_temp)
            journal_published = True
            _fsync_directory(path.parent)
            try:
                os.replace(parquet_temp, path)
                temporary.remove(parquet_temp)
                _fsync_directory(path.parent)
                os.replace(metadata_temp, metadata_path)
                temporary.remove(metadata_temp)
                _fsync_directory(path.parent)
                commit_succeeded = True
            except BaseException as commit_error:
                try:
                    self._recover_abandoned_transaction(
                        path,
                        metadata_path,
                        expected_symbol=dataset.identity.symbol,
                    )
                except CacheRecoveryError as recovery_error:
                    raise CacheRecoveryError(
                        str(recovery_error)
                    ) from commit_error
                raise
        finally:
            for candidate in temporary:
                cleanup(candidate, kind="temporary")
            if commit_succeeded and journal_published:
                try:
                    journal.unlink()
                    _fsync_directory(journal.parent)
                except BaseException as exc:
                    raise CacheRecoveryError(
                        "cache commit publication completed but transaction journal "
                        f"cleanup failed at {journal}: {type(exc).__name__}: {exc}"
                    ) from exc
                else:
                    for recovery in recoveries.values():
                        cleanup(recovery, kind="recovery")
            elif not journal_published:
                for recovery in recoveries.values():
                    cleanup(recovery, kind="incomplete recovery")

    # ── 公开接口 ─────────────────────────────────────────────────────────

    def get(
        self,
        symbol:        str,
        force_refresh: bool = False,
        mt5_connected: bool = True,
    ) -> Optional[pd.DataFrame]:
        """获取品种 K 线数据（本地优先，自动增量更新）。

        Args:
            symbol:        MT5 品种名
            force_refresh: True = 忽略本地缓存，强制重拉全量
            mt5_connected: False = 只读本地，不尝试连接 MT5

        Returns:
            规范化 DataFrame（time UTC, open, high, low, close, volume），
            或 None（本地无数据且无 MT5 连接时）。
        """
        if force_refresh and mt5_connected and _MT5_AVAILABLE:
            return self._full_download(symbol)
        if mt5_connected and _MT5_AVAILABLE:
            return self._incremental_update(symbol, None)

        local_df = self._read_local_with_recovery(symbol)
        if local_df is None:
            logger.warning(f"[Cache] {symbol}: 无本地缓存且无 MT5 连接")
            return None
        logger.info(f"[Cache] {symbol}: 读取本地缓存（无 MT5 连接）")
        return local_df

    def update_all(
        self,
        symbols: list[str],
        mt5_connected: bool = True,
    ) -> dict[str, int]:
        """批量更新多个品种，返回 {symbol: new_bars_added}。"""
        results = {}
        for sym in symbols:
            try:
                before = 0
                before_frame = self._read_local_with_recovery(sym)
                if before_frame is not None:
                    before = len(before_frame)
                if mt5_connected:
                    df = self.get(sym, mt5_connected=True)
                else:
                    df = before_frame
                    if df is None:
                        logger.warning(f"[Cache] {sym}: 无本地缓存且无 MT5 连接")
                    else:
                        logger.info(f"[Cache] {sym}: 读取本地缓存（无 MT5 连接）")
                after = len(df) if df is not None else 0
                added = max(0, after - before)
                results[sym] = added
                logger.info(f"[Cache] {sym}: {after} bars total, +{added} new")
            except CacheRecoveryError:
                raise
            except Exception as exc:
                logger.error(f"[Cache] {sym}: update failed: {exc}")
                results[sym] = -1
        return results

    def _list_cached_targets(self) -> list[Path]:
        """Discover canonical cache targets from primary and transaction artifacts."""
        timeframe_suffix = self._cache_path("").name
        targets: dict[str, Path] = {}
        for artifact in self.cache_dir.iterdir():
            name = artifact.name
            target_name: str | None = None
            expected_artifact_name: str | None = None
            if not name.startswith(".") and name.endswith(".parquet"):
                target_name = name
            elif not name.startswith(".") and name.endswith(".metadata.json"):
                target_name = name[: -len(".metadata.json")] + ".parquet"
            elif (
                name.startswith(".")
                and not name.startswith("..")
                and name.endswith(".parquet.transaction.json")
            ):
                target_name = name[1 : -len(".transaction.json")]
                expected_artifact_name = f".{target_name}.transaction.json"
            elif (
                name.startswith(".")
                and not name.startswith("..")
                and name.endswith(".parquet.recovery")
            ):
                target_name = name[1 : -len(".recovery")]
                expected_artifact_name = f".{target_name}.recovery"
            elif (
                name.startswith(".")
                and not name.startswith("..")
                and name.endswith(".metadata.json.recovery")
            ):
                target_name = name[1 : -len(".metadata.json.recovery")] + ".parquet"
                expected_artifact_name = (
                    f".{target_name[:-len('.parquet')]}.metadata.json.recovery"
                )

            if target_name is None or not target_name.endswith(timeframe_suffix):
                continue
            symbol = target_name[: -len(timeframe_suffix)]
            if (
                not symbol
                or symbol.startswith(".")
                or (expected_artifact_name is not None and name != expected_artifact_name)
            ):
                continue
            target = self._cache_path(symbol)
            if target.name != target_name:
                continue
            normalized_target, canonical_target = _normalized_cache_target(target)
            targets.setdefault(normalized_target, canonical_target)
        return sorted(targets.values(), key=lambda candidate: candidate.name)

    def list_cached(self) -> list[dict]:
        """列出所有已缓存的品种和数据量。"""
        out = []
        for p in self._list_cached_targets():
            metadata_path = p.with_suffix(".metadata.json")
            with _cache_transaction_lock(p):
                self._recover_abandoned_transaction(
                    p,
                    metadata_path,
                    expected_symbol=p.stem.rsplit("_", 1)[0],
                )
                dataset = self._read_validated_pair_locked(
                    p.stem.rsplit("_", 1)[0], p, metadata_path
                )
                if dataset is None:
                    continue
                df = dataset.frame
                entry = {
                    "file": p.name,
                    "bars": len(df),
                    "last_bar": str(df["time"].iloc[-1]),
                    "size_kb": round(p.stat().st_size / 1024, 1),
                }
            out.append(entry)
        return out

    def read_local(self, symbol: str) -> Optional[pd.DataFrame]:
        """直接读本地文件，不尝试 MT5（离线使用）。"""
        return self.get(symbol, mt5_connected=False)

    # ── 内部方法 ─────────────────────────────────────────────────────────

    def _full_download(self, symbol: str) -> Optional[pd.DataFrame]:
        """从 MT5 下载全量历史数据并保存。"""
        path = self._cache_path(symbol)
        metadata_path = self._metadata_path(symbol)
        with _cache_transaction_lock(path):
            self._recover_abandoned_transaction(
                path, metadata_path, expected_symbol=symbol
            )
            return self._full_download_locked(symbol, path, metadata_path)

    def _full_download_locked(
        self,
        symbol: str,
        path: Path,
        metadata_path: Path,
    ) -> Optional[pd.DataFrame]:
        if not _MT5_AVAILABLE or mt5 is None:
            return None
        logger.info(f"[Cache] {symbol}: 全量下载（{self.bars_count} bars）...")
        t0 = time.time()
        rates = _copy_closed_rates(mt5, symbol, self.timeframe, self.bars_count)
        if _closed_rates_are_empty(rates):
            logger.warning(f"[Cache] {symbol}: MT5 返回空数据")
            return None
        dataset = self._canonicalize_rates(symbol, rates)
        df = dataset.frame
        self._atomic_write(symbol, dataset, _lock_held=True)
        elapsed = time.time() - t0
        logger.info(
            f"[Cache] {symbol}: {len(df)} bars 已保存 → {path.name}  ({elapsed:.1f}s)"
        )
        return df

    def _incremental_update(
        self,
        symbol: str,
        local_df: pd.DataFrame | None,
    ) -> Optional[pd.DataFrame]:
        """按时间戳 upsert 远端闭合尾部，同时保留响应未包含的本地行。"""
        path = self._cache_path(symbol)
        metadata_path = self._metadata_path(symbol)
        with _cache_transaction_lock(path):
            self._recover_abandoned_transaction(
                path, metadata_path, expected_symbol=symbol
            )
            if path.exists() or metadata_path.exists():
                current = self._read_validated_pair_locked(
                    symbol, path, metadata_path
                )
                if current is None:
                    raise RuntimeError("cache pair disappeared while target lock was held")
                current_frame = current.frame
            elif local_df is not None:
                current_frame = canonicalize_ohlcv(
                    local_df,
                    symbol=symbol,
                    timeframe=self.timeframe,
                    numeric_time_unit="s",
                ).frame
            else:
                downloaded = self._full_download_locked(symbol, path, metadata_path)
                return downloaded

            if not _MT5_AVAILABLE or mt5 is None:
                return current_frame
            if current_frame.empty:
                downloaded = self._full_download_locked(symbol, path, metadata_path)
                return downloaded if downloaded is not None else current_frame

            revision_bars = _revision_bar_count()
            rates = _copy_closed_rates(mt5, symbol, self.timeframe, revision_bars)
            if _closed_rates_are_empty(rates):
                return current_frame

            remote = self._canonicalize_rates(symbol, rates)
            local = canonicalize_ohlcv(
                current_frame,
                symbol=symbol,
                timeframe=self.timeframe,
            )
            remote_frame = remote.frame
            local_frame = local.frame
            retained_local = local_frame[
                ~local_frame["time"].isin(remote_frame["time"])
            ]
            combined = pd.concat([retained_local, remote_frame], ignore_index=True)
            combined = combined.sort_values("time", kind="mergesort").reset_index(
                drop=True
            )
            merged = canonicalize_ohlcv(
                combined,
                symbol=symbol,
                timeframe=self.timeframe,
            )
            if merged.identity == local.identity:
                logger.debug(f"[Cache] {symbol}: 闭合尾部无新增或修订，跳过写盘")
                return local_frame
            self._atomic_write(symbol, merged, _lock_held=True)
            merged_frame = merged.frame
            added = max(0, len(merged_frame) - len(local_frame))
            logger.info(f"[Cache] {symbol}: +{added} 新 bar，共 {len(merged_frame)} bars")
            return merged_frame
