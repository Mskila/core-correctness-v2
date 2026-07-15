import copy
import contextvars
import ctypes
import hashlib
import heapq
import inspect
import json
import math
import os
import pathlib
import random
import sys
import tempfile
import traceback
from ctypes import wintypes

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from tqdm import tqdm

from data_pipeline.validation import assert_minimum_bars
from .config import ModelConfig
from .alphagpt import AlphaGPT, NewtonSchulzLowRankDecay, StableRankMonitor
from .vm import StackVM
from .backtest import MT5Backtest, compute_ic_metrics
from .semantics import (
    LABEL_LOOKAHEAD_BARS,
    DataValidationError,
    InsufficientWalkForwardDataError,
)
from .vocab import FORMULA_VOCAB, VOCAB_VERSION, VocabVersionMismatchError  # task 12.2
from .walk_forward import (
    WalkForwardFold,
    _safe_diagnostic_value,
    build_walk_forward_folds,
    formula_warmup_bars,
    required_training_bars,
)


_CONFIGURED_N_FOLDS = object()
_ACTIVE_ARTIFACT_PUBLICATION = contextvars.ContextVar(
    "active_artifact_publication", default=None
)

# Artifact observation and rollback retain at most one fixed-size Python payload.
# The bound is independent of checkpoint/history/strategy size.
_ARTIFACT_IO_CHUNK_SIZE = 1024 * 1024
_ABSENT_ARTIFACT_VERSION = (False, 0, None)


def _artifact_stream_version(fp) -> tuple[bool, int, str | None]:
    digest = hashlib.sha256()
    size = 0
    while True:
        chunk = fp.read(_ARTIFACT_IO_CHUNK_SIZE)
        if not chunk:
            break
        size += len(chunk)
        digest.update(chunk)
    return True, size, digest.hexdigest()


def _artifact_open_file_identity(fp) -> tuple[int, int, int]:
    """Return constant-size identity for the same open file being observed."""
    if os.name == "nt":
        import msvcrt

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        return _windows_file_identity(
            kernel32, msvcrt.get_osfhandle(fp.fileno())
        )
    stat_result = os.fstat(fp.fileno())
    return int(stat_result.st_dev), 0, int(stat_result.st_ino)


def _artifact_path_observation(
    path: pathlib.Path,
) -> tuple[tuple[bool, int, str | None], tuple[int, int, int] | None]:
    try:
        with open(path, "rb") as fp:
            identity = _artifact_open_file_identity(fp)
            return _artifact_stream_version(fp), identity
    except FileNotFoundError:
        return _ABSENT_ARTIFACT_VERSION, None


def _artifact_path_version(path: pathlib.Path) -> tuple[bool, int, str | None]:
    return _artifact_path_observation(path)[0]


def _snapshot_artifact_to_owned_backup(
    path: pathlib.Path,
) -> tuple[
    tuple[bool, int, str | None],
    pathlib.Path | None,
    tuple[int, int, int] | None,
]:
    try:
        source = open(path, "rb")
    except FileNotFoundError:
        return _ABSENT_ARTIFACT_VERSION, None, None
    descriptor = None
    backup = None
    try:
        identity = _artifact_open_file_identity(source)
        descriptor, backup_name = tempfile.mkstemp(
            prefix=f".{path.name}.transaction-backup.", dir=path.parent
        )
        backup = pathlib.Path(backup_name)
        digest = hashlib.sha256()
        size = 0
        with source, os.fdopen(descriptor, "wb") as destination:
            descriptor = None
            while True:
                chunk = source.read(_ARTIFACT_IO_CHUNK_SIZE)
                if not chunk:
                    break
                destination.write(chunk)
                digest.update(chunk)
                size += len(chunk)
            destination.flush()
            os.fsync(destination.fileno())
        return (True, size, digest.hexdigest()), backup, identity
    except BaseException:
        source.close()
        if descriptor is not None:
            os.close(descriptor)
        if backup is not None:
            backup.unlink(missing_ok=True)
        raise


def _copy_artifact_path(source: pathlib.Path, destination: pathlib.Path) -> None:
    with open(source, "rb") as source_fp, open(destination, "wb") as destination_fp:
        while True:
            chunk = source_fp.read(_ARTIFACT_IO_CHUNK_SIZE)
            if not chunk:
                break
            destination_fp.write(chunk)
        destination_fp.flush()
        os.fsync(destination_fp.fileno())


def _coerce_artifact_version(value) -> tuple[bool, int, str | None]:
    """Accept old in-process byte receipts without retaining them in transaction state."""
    if isinstance(value, tuple) and len(value) == 3:
        existed, size, digest = value
        return bool(existed), int(size), digest
    if isinstance(value, tuple) and len(value) == 2:
        existed, payload = value
        if not existed:
            return _ABSENT_ARTIFACT_VERSION
        if isinstance(payload, (bytes, bytearray, memoryview)):
            materialized = bytes(payload)
            return True, len(materialized), hashlib.sha256(materialized).hexdigest()
    raise TypeError("invalid artifact publication version receipt")

# P3：冠军在场时间稳健性校验所需；共享执行依赖缺失时导入直接失败。
from strategy_manager.signal import compute_target_positions_stateless

try:
    from config import Config as _RootConfig
    _STRATEGY_FILE  = _RootConfig.STRATEGY_FILE
    _CHECKPOINT_DIR = pathlib.Path(getattr(_RootConfig, 'CHECKPOINT_DIR', 'checkpoints'))
except ImportError:
    _STRATEGY_FILE  = "best_mt5_strategy.json"
    _CHECKPOINT_DIR = pathlib.Path("checkpoints")


def _strategy_file_for_symbol(symbol: str | None) -> str:
    """返回该品种对应的策略文件路径。

    单品种训练时使用 strategies/best_{symbol}.json，
    多品种/未指定品种时回退到默认路径。
    """
    if symbol:
        return str(pathlib.Path("strategies") / f"best_{symbol}.json")
    return _STRATEGY_FILE


def _atomic_json_replace(
    target: pathlib.Path,
    payload: object,
    *,
    indent: int | None = None,
    ensure_ascii: bool = True,
    publisher=None,
) -> None:
    """Publish JSON without exposing a truncated or partially written target."""
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    os.close(descriptor)
    temporary = pathlib.Path(temporary_name)
    fp = None
    try:
        fp = open(temporary, "w")
        try:
            json.dump(
                payload,
                fp,
                indent=indent,
                ensure_ascii=ensure_ascii,
            )
            fp.flush()
        except BaseException as failure:
            try:
                fp.close()
            except BaseException as close_failure:
                failure.add_note(
                    "temporary strategy close also failed: "
                    f"{type(close_failure).__name__}: {close_failure}"
                )
            fp = None
            raise
        fp.close()
        fp = None
        if publisher is None:
            temporary.replace(target)
        else:
            publisher(temporary, target)
    except BaseException as failure:
        if fp is not None:
            try:
                fp.close()
            except BaseException as close_failure:
                failure.add_note(
                    "temporary strategy close also failed: "
                    f"{type(close_failure).__name__}: {close_failure}"
                )
        try:
            temporary.unlink(missing_ok=True)
        except BaseException as cleanup_failure:
            failure.add_note(
                "temporary strategy cleanup also failed: "
                f"{type(cleanup_failure).__name__}: {cleanup_failure}"
            )
        raise


def _windows_open_exclusive(
    path: pathlib.Path,
    *,
    creation_disposition: int = 3,
    desired_access: int = 0x80000000 | 0x40000000 | 0x00010000,
    share_mode: int = 0,
):
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    )
    create_file.restype = wintypes.HANDLE
    kernel32.GetFileSizeEx.argtypes = (
        wintypes.HANDLE,
        ctypes.POINTER(ctypes.c_longlong),
    )
    kernel32.GetFileSizeEx.restype = wintypes.BOOL
    kernel32.SetFilePointerEx.argtypes = (
        wintypes.HANDLE,
        ctypes.c_longlong,
        ctypes.POINTER(ctypes.c_longlong),
        wintypes.DWORD,
    )
    kernel32.SetFilePointerEx.restype = wintypes.BOOL
    kernel32.ReadFile.argtypes = (
        wintypes.HANDLE,
        wintypes.LPVOID,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
        wintypes.LPVOID,
    )
    kernel32.ReadFile.restype = wintypes.BOOL
    kernel32.WriteFile.argtypes = kernel32.ReadFile.argtypes
    kernel32.WriteFile.restype = wintypes.BOOL
    kernel32.SetEndOfFile.argtypes = (wintypes.HANDLE,)
    kernel32.SetEndOfFile.restype = wintypes.BOOL
    kernel32.FlushFileBuffers.argtypes = (wintypes.HANDLE,)
    kernel32.FlushFileBuffers.restype = wintypes.BOOL
    kernel32.SetFileInformationByHandle.argtypes = (
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.LPVOID,
        wintypes.DWORD,
    )
    kernel32.SetFileInformationByHandle.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel32.CloseHandle.restype = wintypes.BOOL
    handle = create_file(
        str(path),
        desired_access,
        share_mode,
        None,
        creation_disposition,
        0x00000080,  # FILE_ATTRIBUTE_NORMAL
        None,
    )
    if handle == wintypes.HANDLE(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    return kernel32, handle


def _windows_handle_version(kernel32, handle) -> tuple[bool, int, str | None]:
    size = ctypes.c_longlong()
    if not kernel32.GetFileSizeEx(handle, ctypes.byref(size)):
        raise ctypes.WinError(ctypes.get_last_error())
    if size.value < 0:
        raise OSError("artifact has a negative size during rollback CAS")
    if not kernel32.SetFilePointerEx(handle, 0, None, 0):
        raise ctypes.WinError(ctypes.get_last_error())
    digest = hashlib.sha256()
    remaining = size.value
    while remaining:
        requested = min(_ARTIFACT_IO_CHUNK_SIZE, remaining)
        buffer = ctypes.create_string_buffer(requested)
        read = wintypes.DWORD()
        if not kernel32.ReadFile(
            handle, buffer, requested, ctypes.byref(read), None
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        if read.value != requested:
            raise OSError(
                f"short artifact read during rollback CAS: expected={requested} "
                f"actual={read.value}"
            )
        digest.update(buffer.raw[: read.value])
        remaining -= read.value
    return True, size.value, digest.hexdigest()


def _windows_write_handle(kernel32, handle, payload: bytes) -> None:
    if len(payload) > 0xFFFFFFFF:
        raise OSError("artifact is too large for bounded rollback CAS")
    if not kernel32.SetFilePointerEx(handle, 0, None, 0):
        raise ctypes.WinError(ctypes.get_last_error())
    if payload:
        buffer = ctypes.create_string_buffer(payload)
        written = wintypes.DWORD()
        if not kernel32.WriteFile(
            handle, buffer, len(payload), ctypes.byref(written), None
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        if written.value != len(payload):
            raise OSError(
                f"short artifact write during rollback CAS: expected={len(payload)} "
                f"actual={written.value}"
            )
    if not kernel32.SetEndOfFile(handle):
        raise ctypes.WinError(ctypes.get_last_error())
    if not kernel32.FlushFileBuffers(handle):
        raise ctypes.WinError(ctypes.get_last_error())


def _windows_copy_artifact_to_handle(
    source: pathlib.Path, kernel32, handle
) -> None:
    if not kernel32.SetFilePointerEx(handle, 0, None, 0):
        raise ctypes.WinError(ctypes.get_last_error())
    with open(source, "rb") as source_fp:
        while True:
            chunk = source_fp.read(_ARTIFACT_IO_CHUNK_SIZE)
            if not chunk:
                break
            buffer = ctypes.create_string_buffer(chunk)
            written = wintypes.DWORD()
            if not kernel32.WriteFile(
                handle, buffer, len(chunk), ctypes.byref(written), None
            ):
                raise ctypes.WinError(ctypes.get_last_error())
            if written.value != len(chunk):
                raise OSError(
                    f"short artifact write during rollback: expected={len(chunk)} "
                    f"actual={written.value}"
                )
    if not kernel32.SetEndOfFile(handle):
        raise ctypes.WinError(ctypes.get_last_error())
    if not kernel32.FlushFileBuffers(handle):
        raise ctypes.WinError(ctypes.get_last_error())


def _windows_delete_handle(kernel32, handle) -> None:
    class _FileDispositionInfo(ctypes.Structure):
        _fields_ = [("DeleteFile", wintypes.BOOL)]

    disposition = _FileDispositionInfo(True)
    if not kernel32.SetFileInformationByHandle(
        handle,
        4,  # FileDispositionInfo
        ctypes.byref(disposition),
        ctypes.sizeof(disposition),
    ):
        raise ctypes.WinError(ctypes.get_last_error())


def _windows_unlink_handle(kernel32, handle) -> None:
    class _FileDispositionInfoEx(ctypes.Structure):
        _fields_ = [("Flags", wintypes.DWORD)]

    disposition = _FileDispositionInfoEx(
        0x00000001 | 0x00000002 | 0x00000010
    )  # DELETE | POSIX_SEMANTICS | IGNORE_READONLY_ATTRIBUTE
    if not kernel32.SetFileInformationByHandle(
        handle,
        21,  # FileDispositionInfoEx
        ctypes.byref(disposition),
        ctypes.sizeof(disposition),
    ):
        raise ctypes.WinError(ctypes.get_last_error())


def _windows_handle_is_linked(kernel32, handle) -> bool:
    class _FileStandardInfo(ctypes.Structure):
        _fields_ = [
            ("AllocationSize", ctypes.c_longlong),
            ("EndOfFile", ctypes.c_longlong),
            ("NumberOfLinks", wintypes.DWORD),
            ("DeletePending", wintypes.BOOLEAN),
            ("Directory", wintypes.BOOLEAN),
        ]

    get_info = kernel32.GetFileInformationByHandleEx
    get_info.argtypes = (
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.LPVOID,
        wintypes.DWORD,
    )
    get_info.restype = wintypes.BOOL
    info = _FileStandardInfo()
    if not get_info(handle, 1, ctypes.byref(info), ctypes.sizeof(info)):
        raise ctypes.WinError(ctypes.get_last_error())
    return bool(info.NumberOfLinks) and not bool(info.DeletePending)


def _windows_file_identity(kernel32, handle) -> tuple[int, int, int]:
    class _ByHandleFileInformation(ctypes.Structure):
        _fields_ = [
            ("FileAttributes", wintypes.DWORD),
            ("CreationTime", wintypes.FILETIME),
            ("LastAccessTime", wintypes.FILETIME),
            ("LastWriteTime", wintypes.FILETIME),
            ("VolumeSerialNumber", wintypes.DWORD),
            ("FileSizeHigh", wintypes.DWORD),
            ("FileSizeLow", wintypes.DWORD),
            ("NumberOfLinks", wintypes.DWORD),
            ("FileIndexHigh", wintypes.DWORD),
            ("FileIndexLow", wintypes.DWORD),
        ]

    get_info = kernel32.GetFileInformationByHandle
    get_info.argtypes = (
        wintypes.HANDLE,
        ctypes.POINTER(_ByHandleFileInformation),
    )
    get_info.restype = wintypes.BOOL
    info = _ByHandleFileInformation()
    if not get_info(handle, ctypes.byref(info)):
        raise ctypes.WinError(ctypes.get_last_error())
    return (
        int(info.VolumeSerialNumber),
        int(info.FileIndexHigh),
        int(info.FileIndexLow),
    )


def _windows_handles_same_file(
    left_kernel32, left_handle, right_kernel32, right_handle
) -> bool:
    return _windows_file_identity(
        left_kernel32, left_handle
    ) == _windows_file_identity(right_kernel32, right_handle)


def _windows_path_identity(path: pathlib.Path) -> tuple[int, int, int]:
    kernel32, handle = _windows_open_exclusive(
        path,
        desired_access=0x80000000,
        share_mode=0x00000001 | 0x00000002 | 0x00000004,
    )
    try:
        return _windows_file_identity(kernel32, handle)
    finally:
        if not kernel32.CloseHandle(handle):
            raise ctypes.WinError(ctypes.get_last_error())


def _windows_rename_handle_no_replace(kernel32, handle, target: pathlib.Path) -> None:
    target_name = str(target.resolve())

    class _FileRenameInfo(ctypes.Structure):
        _fields_ = [
            ("ReplaceIfExists", wintypes.BOOLEAN),
            ("RootDirectory", wintypes.HANDLE),
            ("FileNameLength", wintypes.DWORD),
            ("FileName", wintypes.WCHAR * (len(target_name) + 1)),
        ]

    info = _FileRenameInfo()
    info.ReplaceIfExists = False
    info.RootDirectory = None
    info.FileNameLength = len(target_name.encode("utf-16-le"))
    info.FileName = target_name
    if not kernel32.SetFileInformationByHandle(
        handle,
        3,  # FileRenameInfo
        ctypes.byref(info),
        ctypes.sizeof(info),
    ):
        raise ctypes.WinError(ctypes.get_last_error())


def _windows_replace_file(
    target: pathlib.Path,
    replacement: pathlib.Path,
    backup: pathlib.Path | None = None,
):
    receipt_root = None
    backup_path = None
    if backup is not None:
        receipt_root = pathlib.Path(backup)
        try:
            receipt_root.mkdir()
        except FileExistsError:
            raise RuntimeError(
                f"artifact publication receipt conflict: {str(receipt_root)[:160]}"
            ) from None
        backup_path = receipt_root / "displaced"
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    replace_file = kernel32.ReplaceFileW
    replace_file.argtypes = (
        wintypes.LPCWSTR,
        wintypes.LPCWSTR,
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.LPVOID,
    )
    replace_file.restype = wintypes.BOOL
    if not replace_file(
        str(target),
        str(replacement),
        None if backup_path is None else str(backup_path),
        0,
        None,
        None,
    ):
        failure = ctypes.WinError(ctypes.get_last_error())
        if receipt_root is not None:
            try:
                receipt_root.rmdir()
            except BaseException as cleanup_failure:
                failure.add_note(
                    "publication receipt reservation cleanup also failed: "
                    f"{type(cleanup_failure).__name__}: {cleanup_failure}"
                )
        raise failure
    if backup is None:
        return None
    try:
        receipt_kernel32, receipt_handle = _windows_open_exclusive(
            backup_path,
            desired_access=0x80000000 | 0x00010000,
            share_mode=0x00000001 | 0x00000004,
        )
    except BaseException as failure:
        failure._artifact_kernel_replace_succeeded = True
        raise
    try:
        published_kernel32, published_handle = _windows_open_exclusive(
            target,
            desired_access=0x80000000 | 0x00010000,
            share_mode=0x00000001 | 0x00000002 | 0x00000004,
        )
    except BaseException as failure:
        failure._artifact_kernel_replace_succeeded = True
        try:
            if not receipt_kernel32.CloseHandle(receipt_handle):
                raise ctypes.WinError(ctypes.get_last_error())
        except BaseException as close_failure:
            failure.add_note(
                "post-publication displaced handle close also failed: "
                f"{type(close_failure).__name__}: {close_failure}"
            )
        raise
    return (
        receipt_kernel32,
        receipt_handle,
        published_kernel32,
        published_handle,
    )


def _repetition_penalty(formula: list[int]) -> float:
    if not formula:
        return 0.0
    penalty, count = 0.0, 1
    for i in range(1, len(formula)):
        if formula[i] == formula[i - 1]:
            count += 1
            if count >= 2:
                penalty += 0.3
        else:
            count = 1
    return penalty


def _has_exact_variation(values: torch.Tensor) -> bool:
    """Return whether finite values differ at their represented precision."""
    flat = values.reshape(-1)
    return bool((flat != flat[0]).any())


class _ArtifactPublicationReceipt:
    """Exact versions explicitly published by one transaction callback."""

    def __init__(self, result, claims):
        self._artifact_publication_result = result
        self._artifact_publication_claims = dict(claims)


class _BatchTransaction:
    """Rollback one training batch across memory and its public artifacts."""

    _STATE_FIELDS = (
        "best_score",
        "best_formula",
        "_best_snapshot",
        "_best_update_step",
        "_stagnation_steps",
        "factor_pool",
        "_factor_pool_counter",
        "_elite_pool",
        "_elite_counter",
        "_reward_ema",
        "_reward_ema_step",
        "training_history",
        "_restart_count",
        "_low_entropy_streak",
    )

    @staticmethod
    def _artifact_version(path: pathlib.Path) -> tuple[bool, int, str | None]:
        return _artifact_path_version(path)

    @staticmethod
    def _artifact_observation(path: pathlib.Path):
        return _artifact_path_observation(path)

    def __init__(self, engine: "AlphaEngine", artifact_paths: list[pathlib.Path]):
        self.engine = engine
        self.model_state = copy.deepcopy(engine.model.state_dict())
        self.optimizer = engine.opt
        self.optimizer_state = copy.deepcopy(engine.opt.state_dict())
        self.gradients = [
            None if parameter.grad is None else parameter.grad.detach().clone()
            for parameter in engine.model.parameters()
        ]
        self.state = {
            name: copy.deepcopy(getattr(engine, name)) for name in self._STATE_FIELDS
        }
        self.artifacts: dict[pathlib.Path, tuple[bool, int, str | None]] = {}
        self.artifact_identities: dict[
            pathlib.Path, tuple[int, int, int] | None
        ] = {}
        self._initial_engine_owned_artifacts: set[pathlib.Path] = set()
        self._artifact_backups: dict[pathlib.Path, pathlib.Path | None] = {}
        ownership_records = engine._artifact_ownership_records()
        try:
            for path in dict.fromkeys(artifact_paths):
                version, backup, identity = _snapshot_artifact_to_owned_backup(path)
                self.artifacts[path] = version
                self.artifact_identities[path] = identity
                self._artifact_backups[path] = backup
                ownership_key = engine._artifact_ownership_key(path)
                if (
                    version[0]
                    and identity is not None
                    and ownership_records.get(ownership_key) == (version, identity)
                ):
                    self._initial_engine_owned_artifacts.add(path)
        except BaseException:
            for backup in self._artifact_backups.values():
                if backup is not None:
                    backup.unlink(missing_ok=True)
            raise
        self.owned_artifacts: dict[
            pathlib.Path,
            tuple[tuple[bool, int, str | None], tuple[int, int, int] | None],
        ] = {}
        self._publication_identities: dict[
            pathlib.Path, tuple[int, int, int]
        ] = {}
        self._publication_expected_identities: dict[
            pathlib.Path, tuple[int, int, int] | None
        ] = {}
        self._last_observed_identities: dict[
            pathlib.Path, tuple[int, int, int] | None
        ] = {}
        self.publication_conflicts: list[str] = []
        self.python_rng_state = random.getstate()
        self.numpy_rng_state = np.random.get_state()
        self.torch_cpu_rng_state = torch.get_rng_state().clone()
        self.torch_cuda_rng_state = (
            [state.clone() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_initialized()
            else None
        )
        sampler = getattr(engine, "sampler", None)
        state_dict = getattr(sampler, "state_dict", None)
        load_state_dict = getattr(sampler, "load_state_dict", None)
        getstate = getattr(sampler, "getstate", None)
        setstate = getattr(sampler, "setstate", None)
        if callable(state_dict) and callable(load_state_dict):
            self.sampler_state = ("state_dict", copy.deepcopy(state_dict()))
        elif callable(getstate) and callable(setstate):
            self.sampler_state = ("getstate", copy.deepcopy(getstate()))
        else:
            self.sampler_state = None
        self.active = True

    def _current_artifacts(
        self,
        paths: tuple[pathlib.Path, ...] | list[pathlib.Path] | None = None,
    ) -> dict[pathlib.Path, tuple[bool, int, str | None]]:
        versions = {}
        identities = {}
        for path in (self.artifacts if paths is None else paths):
            version, identity = self._artifact_observation(path)
            versions[path] = version
            identities[path] = identity
        self._last_observed_identities = identities
        return versions

    def _replace_artifact_snapshot(
        self, path: pathlib.Path
    ) -> tuple[bool, int, str | None]:
        version, backup, identity = _snapshot_artifact_to_owned_backup(path)
        previous = self._artifact_backups.get(path)
        self._artifact_backups[path] = backup
        self.artifacts[path] = version
        self.artifact_identities[path] = identity
        if previous is not None:
            previous.unlink(missing_ok=True)
        return version

    def _record_claimed_publications(
        self,
        before: dict[pathlib.Path, tuple[bool, int, str | None]],
        claims: dict[pathlib.Path, object],
    ) -> None:
        after = self._current_artifacts(list(before))
        after_identities = dict(self._last_observed_identities)
        for path, version in after.items():
            raw_claim = claims.get(path)
            claimed = (
                None
                if raw_claim is None
                else _coerce_artifact_version(raw_claim)
            )
            claimed_identity = self._publication_identities.get(path)
            observed_identity = after_identities.get(path)
            identity_matches = claimed is not None and (
                (
                    claimed[0]
                    and claimed_identity is not None
                    and observed_identity == claimed_identity
                )
                or (
                    not claimed[0]
                    and claimed_identity is None
                    and observed_identity is None
                )
            )
            if claimed is not None and version == claimed and identity_matches:
                self.owned_artifacts[path] = (version, claimed_identity)
            elif claimed is not None:
                self.publication_conflicts.append(
                    f"{str(path)[:160]} exact publication receipt no longer current"
                )
                self._replace_artifact_snapshot(path)
                self.owned_artifacts.pop(path, None)
        self._publication_identities.clear()

    def _replace_publication_temp(
        self,
        temporary: pathlib.Path,
        target: pathlib.Path,
        expected: tuple[bool, int, str | None],
    ) -> None:
        hook = getattr(self, "_publication_cas_interleave_hook", None)
        if callable(hook):
            hook(target)
        published, candidate_identity = self._artifact_observation(temporary)
        if os.name != "nt":
            current, current_identity = self._artifact_observation(target)
            expected_identity = self._publication_expected_identities.get(target)
            if current != expected or current_identity != expected_identity:
                raise RuntimeError(
                    f"artifact publication conflict: {str(target)[:160]}"
                )
            latest, latest_identity = self._artifact_observation(target)
            if latest != current or latest_identity != current_identity:
                raise RuntimeError(
                    f"artifact publication conflict: {str(target)[:160]}"
                )
            temporary.replace(target)
            if candidate_identity is None:
                raise RuntimeError(
                    f"artifact publication identity unavailable: {str(target)[:160]}"
                )
            self._publication_identities[target] = candidate_identity
            return

        existed, _original_size, _original_digest = expected
        expected_identity = self._publication_expected_identities.get(target)
        created_placeholder = not existed
        candidate_kernel32 = None
        candidate_handle = None
        candidate_identity = None
        receipt_kernel32 = None
        receipt_handle = None
        receipt = temporary.with_name(f"{temporary.name}.receipt")
        try:
            kernel32, handle = _windows_open_exclusive(
                target,
                creation_disposition=3 if existed else 1,
                desired_access=0x80000000 | 0x00010000,
                share_mode=0x00000001 | 0x00000004,
            )
        except OSError as failure:
            current = self._artifact_version(target)
            if current != expected:
                raise RuntimeError(
                    f"artifact publication conflict: {str(target)[:160]}"
                ) from None
            raise failure
        try:
            locked_identity = _windows_file_identity(kernel32, handle)
            if existed and locked_identity != expected_identity:
                raise RuntimeError(
                    f"artifact publication conflict: {str(target)[:160]}"
                )
            locked = (
                _windows_handle_version(kernel32, handle)
                if existed
                else _ABSENT_ARTIFACT_VERSION
            )
            if locked != expected:
                raise RuntimeError(
                    f"artifact publication conflict: {str(target)[:160]}"
                )
            locked_hook = getattr(
                self, "_publication_locked_interleave_hook", None
            )
            if callable(locked_hook):
                locked_hook(target)
            if not _windows_handle_is_linked(kernel32, handle):
                raise RuntimeError(
                    f"artifact publication conflict: {str(target)[:160]}"
                )
            if receipt.exists():
                raise RuntimeError(
                    f"artifact publication receipt conflict: {str(receipt)[:160]}"
                )
            candidate_kernel32, candidate_handle = _windows_open_exclusive(
                temporary,
                desired_access=0x80000000 | 0x00010000,
                share_mode=0x00000001 | 0x00000002 | 0x00000004,
            )
            candidate_identity = _windows_file_identity(
                candidate_kernel32, candidate_handle
            )
            if not candidate_kernel32.CloseHandle(candidate_handle):
                raise ctypes.WinError(ctypes.get_last_error())
            candidate_handle = None
            (
                receipt_kernel32,
                receipt_handle,
                candidate_kernel32,
                candidate_handle,
            ) = _windows_replace_file(target, temporary, receipt)
            self._publication_identities[target] = candidate_identity
            if not _windows_handles_same_file(
                kernel32,
                handle,
                receipt_kernel32,
                receipt_handle,
            ):
                conflict = RuntimeError(
                    f"artifact publication conflict: {str(target)[:160]}"
                )
                candidate_is_current = (
                    _windows_file_identity(candidate_kernel32, candidate_handle)
                    == candidate_identity
                )
                if candidate_is_current:
                    _windows_unlink_handle(candidate_kernel32, candidate_handle)
                if not candidate_kernel32.CloseHandle(candidate_handle):
                    raise ctypes.WinError(ctypes.get_last_error())
                candidate_handle = None
                if candidate_is_current:
                    try:
                        _windows_rename_handle_no_replace(
                            receipt_kernel32, receipt_handle, target
                        )
                    except OSError as recovery_failure:
                        if getattr(recovery_failure, "winerror", None) not in (80, 183):
                            conflict.add_note(
                                "publication receipt recovery also failed: "
                                f"{type(recovery_failure).__name__}: {recovery_failure}"
                            )
                            raise conflict
                        _windows_delete_handle(receipt_kernel32, receipt_handle)
                else:
                    _windows_delete_handle(receipt_kernel32, receipt_handle)
                raise conflict
            _windows_delete_handle(receipt_kernel32, receipt_handle)
        except BaseException as failure:
            if getattr(failure, "_artifact_kernel_replace_succeeded", False):
                recovery_kernel32 = None
                recovery_handle = None
                current_identified = False
                candidate_is_current = False
                try:
                    recovery_kernel32, recovery_handle = _windows_open_exclusive(
                        target,
                        desired_access=0x80000000 | 0x00010000,
                        share_mode=0x00000001 | 0x00000002 | 0x00000004,
                    )
                    candidate_is_current = (
                        _windows_file_identity(recovery_kernel32, recovery_handle)
                        == candidate_identity
                    )
                    current_identified = True
                    if candidate_is_current:
                        _windows_unlink_handle(recovery_kernel32, recovery_handle)
                except BaseException as recovery_failure:
                    failure.add_note(
                        "post-publication candidate recovery also failed: "
                        f"{type(recovery_failure).__name__}: {recovery_failure}"
                    )
                finally:
                    if recovery_handle is not None:
                        try:
                            if not recovery_kernel32.CloseHandle(recovery_handle):
                                raise ctypes.WinError(ctypes.get_last_error())
                        except BaseException as close_failure:
                            failure.add_note(
                                "publication recovery handle close also failed: "
                                f"{type(close_failure).__name__}: {close_failure}"
                            )
                if current_identified:
                    try:
                        if candidate_is_current and existed:
                            _windows_rename_handle_no_replace(
                                kernel32, handle, target
                            )
                        else:
                            _windows_delete_handle(kernel32, handle)
                    except BaseException as recovery_failure:
                        failure.add_note(
                            "post-publication receipt recovery also failed: "
                            f"{type(recovery_failure).__name__}: {recovery_failure}"
                        )
                try:
                    if not kernel32.CloseHandle(handle):
                        raise ctypes.WinError(ctypes.get_last_error())
                except BaseException as close_failure:
                    failure.add_note(
                        "publication CAS handle close also failed: "
                        f"{type(close_failure).__name__}: {close_failure}"
                    )
                handle = None
            for close_kernel32, close_handle, label in (
                (receipt_kernel32, receipt_handle, "receipt"),
                (candidate_kernel32, candidate_handle, "candidate"),
            ):
                if close_handle is None:
                    continue
                try:
                    if not close_kernel32.CloseHandle(close_handle):
                        raise ctypes.WinError(ctypes.get_last_error())
                except BaseException as close_failure:
                    failure.add_note(
                        f"publication {label} handle close also failed: "
                        f"{type(close_failure).__name__}: {close_failure}"
                    )
            if created_placeholder and handle is not None:
                try:
                    _windows_delete_handle(kernel32, handle)
                except BaseException as cleanup_failure:
                    failure.add_note(
                        "publication placeholder cleanup also failed: "
                        f"{type(cleanup_failure).__name__}: {cleanup_failure}"
                    )
            if handle is not None:
                try:
                    if not kernel32.CloseHandle(handle):
                        raise ctypes.WinError(ctypes.get_last_error())
                except BaseException as close_failure:
                    failure.add_note(
                        "publication CAS handle close also failed: "
                        f"{type(close_failure).__name__}: {close_failure}"
                    )
            if receipt.is_dir():
                try:
                    receipt.rmdir()
                except BaseException as cleanup_failure:
                    failure.add_note(
                        "publication receipt reservation cleanup also failed: "
                        f"{type(cleanup_failure).__name__}: {cleanup_failure}"
                    )
            raise
        else:
            close_failures: list[BaseException] = []
            for close_kernel32, close_handle, label in (
                (receipt_kernel32, receipt_handle, "receipt"),
                (candidate_kernel32, candidate_handle, "candidate"),
                (kernel32, handle, "CAS"),
            ):
                if close_handle is None:
                    continue
                try:
                    if not close_kernel32.CloseHandle(close_handle):
                        raise ctypes.WinError(ctypes.get_last_error())
                except BaseException as close_failure:
                    close_failures.append(close_failure)
            try:
                receipt.rmdir()
            except BaseException as cleanup_failure:
                close_failures.append(cleanup_failure)
            if close_failures:
                primary = close_failures[0]
                primary._artifact_publication_claims = {
                    target: published
                }
                for additional in close_failures[1:8]:
                    primary.add_note(
                        "additional publication cleanup failure: "
                        f"{type(additional).__name__}: {additional}"
                    )
                raise primary

    def run(self, operation, /, *args, **kwargs):
        if not self.active:
            raise RuntimeError("batch transaction is no longer active")
        try:
            result = operation(*args, **kwargs)
        except BaseException as failure:
            try:
                self.rollback()
            except BaseException as rollback_failure:
                failure.add_note(
                    "batch rollback also failed: "
                    f"{type(rollback_failure).__name__}: {rollback_failure}"
                )
            traceback.clear_frames(failure.__traceback__)
            raise
        return result

    def run_artifact(
        self,
        operation,
        affected_paths: tuple[pathlib.Path, ...] | list[pathlib.Path],
        /,
        *args,
        **kwargs,
    ):
        """Run one publication and own only exact declared receipt versions."""
        if not self.active:
            raise RuntimeError("batch transaction is no longer active")
        paths = tuple(dict.fromkeys(pathlib.Path(path) for path in affected_paths))
        undeclared = [path for path in paths if path not in self.artifacts]
        if undeclared:
            raise ValueError(
                "artifact publication contains undeclared transaction paths: "
                + ", ".join(str(path) for path in undeclared)
            )
        before = self._current_artifacts(paths)
        before_identities = dict(self._last_observed_identities)
        self._publication_identities.clear()
        for path, version in tuple(before.items()):
            owned = self.owned_artifacts.get(path)
            if owned is None:
                if version != self.artifacts[path]:
                    before[path] = self._replace_artifact_snapshot(path)
                    before_identities[path] = self.artifact_identities[path]
            elif (
                version != owned[0]
                or (
                    owned[1] is not None
                    and before_identities.get(path) != owned[1]
                )
            ):
                before[path] = self._replace_artifact_snapshot(path)
                before_identities[path] = self.artifact_identities[path]
                self.owned_artifacts.pop(path, None)
        self._publication_expected_identities = before_identities
        publication_token = _ACTIVE_ARTIFACT_PUBLICATION.set((self, before))
        try:
            result = operation(*args, **kwargs)
        except BaseException as failure:
            _ACTIVE_ARTIFACT_PUBLICATION.reset(publication_token)
            try:
                claims = getattr(failure, "_artifact_publication_claims", {})
                self._record_claimed_publications(before, dict(claims))
                self.rollback()
            except BaseException as rollback_failure:
                failure.add_note(
                    "batch rollback also failed: "
                    f"{type(rollback_failure).__name__}: {rollback_failure}"
                )
            traceback.clear_frames(failure.__traceback__)
            raise
        _ACTIVE_ARTIFACT_PUBLICATION.reset(publication_token)
        claims = getattr(result, "_artifact_publication_claims", {})
        self._record_claimed_publications(before, dict(claims))
        return getattr(result, "_artifact_publication_result", result)

    def _release(self) -> None:
        self.state.clear()
        self.artifacts.clear()
        self.artifact_identities.clear()
        self._initial_engine_owned_artifacts.clear()
        self._artifact_backups.clear()
        self.owned_artifacts.clear()
        self._publication_identities.clear()
        self._publication_expected_identities.clear()
        self._last_observed_identities.clear()
        self.publication_conflicts.clear()
        self.model_state.clear()
        self.optimizer = None
        self.optimizer_state.clear()
        self.gradients.clear()
        self.python_rng_state = None
        self.numpy_rng_state = None
        self.torch_cpu_rng_state = None
        self.torch_cuda_rng_state = None
        self.sampler_state = None

    def commit(self) -> None:
        failures: list[BaseException] = []
        for backup in self._artifact_backups.values():
            if backup is None:
                continue
            try:
                backup.unlink(missing_ok=True)
            except BaseException as failure:
                failures.append(failure)
        self.active = False
        self._release()
        if failures:
            primary = failures[0]
            for additional in failures[1:8]:
                primary.add_note(
                    "additional transaction backup cleanup failure: "
                    f"{type(additional).__name__}: {additional}"
                )
            raise primary

    def rollback(self) -> None:
        if not self.active:
            return
        failures: list[BaseException] = []
        conflicts: list[str] = list(self.publication_conflicts[:8])

        def attempt(operation) -> None:
            try:
                operation()
            except BaseException as failure:
                failures.append(failure)

        def describe_version(version: tuple[bool, int, str | None]) -> str:
            existed, size, digest = version
            if not existed:
                return "absent"
            return f"sha256={(digest or '')[:16]},bytes={size}"

        def restore_artifact(
            path: pathlib.Path,
            original: tuple[bool, int, str | None],
        ) -> None:
            ownership_records = self.engine._artifact_ownership_records()
            ownership_key = self.engine._artifact_ownership_key(path)
            ownership_records.pop(ownership_key, None)
            current, current_identity = self._artifact_observation(path)

            def reconcile_restored_ownership(
                restored_version: tuple[bool, int, str | None],
                restored_identity: tuple[int, int, int] | None,
            ) -> None:
                if not original[0] or path not in self._initial_engine_owned_artifacts:
                    return
                if restored_version != original or restored_identity is None:
                    raise RuntimeError(
                        f"rollback ownership reconciliation failed: {str(path)[:160]}"
                    )
                ownership_records[ownership_key] = (
                    restored_version,
                    restored_identity,
                )

            if current == original:
                if (
                    path in self._initial_engine_owned_artifacts
                    and current_identity != self.artifact_identities[path]
                ):
                    conflicts.append(
                        f"{str(path)[:160]} current=file-identity-changed"
                    )
                    return
                reconcile_restored_ownership(current, current_identity)
                return
            owned = self.owned_artifacts.get(path)
            if owned is None or current != owned[0]:
                conflicts.append(
                    f"{str(path)[:160]} current={describe_version(current)}"
                )
                return
            _owned_version, owned_identity = owned
            if os.name != "nt" and (
                owned_identity is None or current_identity != owned_identity
            ):
                conflicts.append(
                    f"{str(path)[:160]} current=file-identity-changed"
                )
                return
            existed, _size, _digest = original
            backup = self._artifact_backups.get(path)
            if existed and backup is None:
                raise RuntimeError(
                    f"transaction backup missing for artifact: {str(path)[:160]}"
                )
            if os.name == "nt":
                try:
                    kernel32, handle = _windows_open_exclusive(path)
                except OSError as failure:
                    conflicts.append(
                        f"{str(path)[:160]} exclusive-open="
                        f"{type(failure).__name__}:{str(failure)[:80]}"
                    )
                    return
                try:
                    if (
                        owned_identity is not None
                        and _windows_file_identity(kernel32, handle)
                        != owned_identity
                    ):
                        conflicts.append(
                            f"{str(path)[:160]} current=file-identity-changed"
                        )
                        return
                    locked_current = _windows_handle_version(kernel32, handle)
                    if locked_current != current:
                        conflicts.append(
                            f"{str(path)[:160]} current="
                            f"{describe_version(locked_current)}"
                        )
                        return
                    hook = getattr(self, "_cas_interleave_hook", None)
                    if callable(hook):
                        hook(path, kernel32, handle)
                    locked_latest = _windows_handle_version(kernel32, handle)
                    if locked_latest != current:
                        conflicts.append(
                            f"{str(path)[:160]} current="
                            f"{describe_version(locked_latest)}"
                        )
                        return
                    if existed:
                        _windows_copy_artifact_to_handle(backup, kernel32, handle)
                    else:
                        _windows_delete_handle(kernel32, handle)
                    restored_version = _windows_handle_version(kernel32, handle)
                    restored_identity = _windows_file_identity(kernel32, handle)
                    reconcile_restored_ownership(
                        restored_version, restored_identity
                    )
                finally:
                    try:
                        closed = kernel32.CloseHandle(handle)
                    except BaseException:
                        ownership_records.pop(ownership_key, None)
                        raise
                    if not closed:
                        ownership_records.pop(ownership_key, None)
                        raise ctypes.WinError(ctypes.get_last_error())
                return
            if not existed:
                latest, latest_identity = self._artifact_observation(path)
                if latest != current or latest_identity != owned_identity:
                    conflicts.append(
                        f"{str(path)[:160]} current=file-identity-changed"
                    )
                    return
                path.unlink(missing_ok=True)
                reconcile_restored_ownership(_ABSENT_ARTIFACT_VERSION, None)
                return
            path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temp_name = tempfile.mkstemp(
                prefix=f".{path.name}.rollback.", suffix=".tmp", dir=path.parent
            )
            os.close(descriptor)
            rollback_temp = pathlib.Path(temp_name)
            try:
                _copy_artifact_path(backup, rollback_temp)
                latest, latest_identity = self._artifact_observation(path)
                if latest != current or latest_identity != owned_identity:
                    conflicts.append(
                        f"{str(path)[:160]} current=file-identity-changed"
                    )
                    return
                restored_version, restored_identity = _artifact_path_observation(
                    rollback_temp
                )
                rollback_temp.replace(path)
                reconcile_restored_ownership(
                    restored_version, restored_identity
                )
            finally:
                rollback_temp.unlink(missing_ok=True)

        try:
            attempt(lambda: random.setstate(self.python_rng_state))
            attempt(lambda: np.random.set_state(self.numpy_rng_state))
            attempt(lambda: torch.set_rng_state(self.torch_cpu_rng_state))
            if self.torch_cuda_rng_state is not None:
                attempt(
                    lambda: torch.cuda.set_rng_state_all(self.torch_cuda_rng_state)
                )
            if self.sampler_state is not None:
                protocol, sampler_state = self.sampler_state
                if protocol == "state_dict":
                    attempt(
                        lambda: self.engine.sampler.load_state_dict(
                            copy.deepcopy(sampler_state)
                        )
                    )
                else:
                    attempt(
                        lambda: self.engine.sampler.setstate(
                            copy.deepcopy(sampler_state)
                        )
                    )
            attempt(lambda: self.engine.model.load_state_dict(self.model_state))

            def restore_optimizer() -> None:
                self.engine.opt = self.optimizer
                self.engine.opt.load_state_dict(self.optimizer_state)

            attempt(restore_optimizer)

            def restore_gradients() -> None:
                for parameter, gradient in zip(
                    self.engine.model.parameters(), self.gradients
                ):
                    parameter.grad = None if gradient is None else gradient.clone()

            attempt(restore_gradients)
            for name, value in self.state.items():
                attempt(
                    lambda name=name, value=value: setattr(
                        self.engine, name, copy.deepcopy(value)
                    )
                )
            for path, original in self.artifacts.items():
                attempt(
                    lambda path=path, original=original: restore_artifact(
                        path, original
                    )
                )
            for backup in self._artifact_backups.values():
                if backup is not None:
                    attempt(lambda backup=backup: backup.unlink(missing_ok=True))
            if conflicts:
                failures.append(
                    RuntimeError(
                        "batch rollback conflict: concurrent artifact version preserved; "
                        + "; ".join(sorted(conflicts)[:8])
                    )
                )
        finally:
            self.active = False
            self._release()
        if failures:
            primary = failures[0]
            for additional in failures[1:8]:
                primary.add_note(
                    "additional rollback failure: "
                    f"{type(additional).__name__}: {additional}"
                )
            raise primary


def _replace_artifact_publication_temp(
    temporary: pathlib.Path, target: pathlib.Path
) -> None:
    active = _ACTIVE_ARTIFACT_PUBLICATION.get()
    if active is None:
        temporary.replace(target)
        return
    transaction, before = active
    if target not in before:
        raise ValueError(f"undeclared artifact publication target: {target}")
    transaction._replace_publication_temp(temporary, target, before[target])


def _artifact_publication_result(result, claims):
    if _ACTIVE_ARTIFACT_PUBLICATION.get() is None:
        return result
    return _ArtifactPublicationReceipt(result, claims)


class _StableCategoricalEntropy(torch.autograd.Function):
    """Exact categorical entropy with a scale-bounded surrogate backward."""

    @staticmethod
    def forward(ctx, logits: torch.Tensor) -> torch.Tensor:
        finite = torch.isfinite(logits)
        if not bool(finite.any(dim=-1).all()):
            raise ValueError("categorical logits must contain finite support")
        safe_logits = torch.where(finite, logits, torch.zeros_like(logits))
        scale = safe_logits.abs().amax(dim=-1, keepdim=True).clamp_min(1.0)
        bounded_logits = (safe_logits / scale).masked_fill(~finite, -torch.inf)
        probabilities = torch.softmax(bounded_logits, dim=-1)
        ctx.save_for_backward(probabilities, scale, finite)
        return Categorical(logits=logits).entropy()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor]:
        probabilities, scale, finite = ctx.saved_tensors
        log_probabilities = torch.where(
            probabilities > 0,
            probabilities.log(),
            torch.zeros_like(probabilities),
        )
        entropy = -torch.special.xlogy(probabilities, probabilities).sum(
            dim=-1, keepdim=True
        )
        entropy_gradient = (
            -probabilities * (log_probabilities + entropy) / scale
        ).masked_fill(~finite, 0.0)
        return (grad_output.unsqueeze(-1) * entropy_gradient,)


def _scale_invariant_centered(values: torch.Tensor) -> torch.Tensor:
    """Center and unit-normalize a known nonconstant one-dimensional tensor."""
    scaled = values / values.abs().amax()
    translated = scaled - scaled[0]
    translated = translated / translated.abs().amax()
    centered = translated - translated.mean()
    return centered / centered.norm()


# ─────────────────────────────────────────────────────────────────────────────
# ConstrainedSampler — 保证 100% 合法公式
# ─────────────────────────────────────────────────────────────────────────────

class ConstrainedSampler:
    def __init__(self, vocab_size: int, feat_offset: int, arity_map: dict[int, int],
                 positive_only_ids: set[int] | None = None):
        self.vocab_size  = vocab_size
        self.feat_offset = feat_offset
        self.arity_map   = arity_map
        self.delta: dict[int, int] = {}
        for tid in range(vocab_size):
            if tid < feat_offset:
                self.delta[tid] = 1
            else:
                a = arity_map.get(tid, 1)
                self.delta[tid] = 1 - a
        # 恒正算子 token id 集合（用于算子链约束）
        self.positive_only_ids = positive_only_ids or set()
        # 构建感染传播/恢复算子 id 集合
        from .vm import INFECTED_PROPAGATING_OPS, SIGN_RESTORE_OPS
        from .ops import OPS_CONFIG as _ops
        self.infected_propagating_ids = set()
        self.sign_restore_ids = set()
        for i, cfg in enumerate(_ops):
            tid = i + feat_offset
            if cfg[0] in INFECTED_PROPAGATING_OPS:
                self.infected_propagating_ids.add(tid)
            if cfg[0] in SIGN_RESTORE_OPS:
                self.sign_restore_ids.add(tid)

    def valid_mask(self, stack_depth: int, step_idx: int,
                   total_steps: int, device: torch.device,
                   prev_token: int | None = None,
                   infected_chain_len: int = 0) -> torch.Tensor:
        remaining = total_steps - step_idx
        mask = torch.ones(self.vocab_size, dtype=torch.bool, device=device)
        for tid in range(self.vocab_size):
            d         = self.delta[tid]
            new_depth = stack_depth + d
            if new_depth < 1:
                mask[tid] = False;  continue
            min_future = new_depth + (remaining - 1) * (-2)
            max_future = new_depth + (remaining - 1) * 1
            if 1 < min_future or 1 > max_future:
                mask[tid] = False
            # ── 算子链约束（感染模型）──────────────────────────────
            # 如果已感染且感染链 >= 2，禁止再使用传播算子
            # （允许恢复算子和非传播算子如 ADD/SUB/MUL）
            if infected_chain_len >= 2 and tid in self.infected_propagating_ids:
                mask[tid] = False
            # 如果已感染且感染链 >= 3，禁止所有算子（强制恢复或结束）
            # 实际上不禁止恢复算子，只禁止传播和恒正算子
            if infected_chain_len >= 3:
                if tid in self.infected_propagating_ids or tid in self.positive_only_ids:
                    mask[tid] = False
        if not mask.any():
            for tid in range(self.vocab_size):
                if stack_depth + self.delta[tid] >= 1:
                    mask[tid] = True
        return mask

    def apply_mask_to_logits(self, logits: torch.Tensor, stack_depths: list[int],
                              step_idx: int, total_steps: int,
                              prev_tokens: list[int | None] | None = None,
                              infected_chain_lens: list[int] | None = None) -> torch.Tensor:
        masked = logits.clone()
        device = logits.device
        for b, depth in enumerate(stack_depths):
            prev_t = prev_tokens[b] if prev_tokens else None
            icl = infected_chain_lens[b] if infected_chain_lens else 0
            vmask = self.valid_mask(depth, step_idx, total_steps, device,
                                    prev_token=prev_t, infected_chain_len=icl)
            masked[b][~vmask] = -1e9
        return masked

    def update_infection(self, token: int, infected_chain_len: int) -> int:
        """更新感染链长度，返回新的感染链长度。"""
        if token in self.positive_only_ids:
            return infected_chain_len + 1
        elif token in self.sign_restore_ids:
            return 0
        elif token in self.infected_propagating_ids:
            if infected_chain_len > 0:
                return infected_chain_len + 1
            return 0
        return infected_chain_len  # 非传播/非恢复算子，不改变状态


# ─────────────────────────────────────────────────────────────────────────────
# AlphaEngine — __init__ 与静态辅助方法
# ─────────────────────────────────────────────────────────────────────────────

class AlphaEngine:
    @staticmethod
    def _validate_n_folds(n_folds: object) -> int:
        expected = ModelConfig.WF_N_BLOCKS
        if type(expected) is not int or expected < 2:
            raise InsufficientWalkForwardDataError(
                "invalid walk-forward configuration: ModelConfig.WF_N_BLOCKS "
                "expected=exact built-in int >=2; "
                f"actual={AlphaEngine._bounded_actual(expected)}"
            )
        if type(n_folds) is not int or n_folds != expected:
            raise InsufficientWalkForwardDataError(
                "invalid walk-forward configuration: n_folds "
                f"expected={expected}; actual={AlphaEngine._bounded_actual(n_folds)}"
            )
        return expected

    @staticmethod
    def _bounded_actual(value: object, *, limit: int = 80) -> str:
        return _safe_diagnostic_value(value, limit=limit)

    @staticmethod
    def _validate_training_tensors(data_manager) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        tensors: dict[str, torch.Tensor] = {}
        missing = object()
        for name in ("feat_tensor", "target_ret", "target_valid", "bar_time"):
            if inspect.getattr_static(data_manager, name, missing) is missing:
                raise DataValidationError(
                    "AlphaEngine training data contract violation: "
                    f"{name} expected torch.Tensor; actual=missing"
                )
            value = getattr(data_manager, name)
            if not isinstance(value, torch.Tensor):
                raise DataValidationError(
                    "AlphaEngine training data contract violation: "
                    f"{name} expected torch.Tensor; actual={type(value).__name__}"
                )
            tensors[name] = value

        feat_tensor = tensors["feat_tensor"]
        target_ret = tensors["target_ret"]
        target_valid = tensors["target_valid"]
        bar_time = tensors["bar_time"]

        if feat_tensor.ndim != 3:
            raise DataValidationError(
                "AlphaEngine training data contract violation: feat_tensor expected "
                f"rank 3 [N,F,T]; actual shape={tuple(feat_tensor.shape)}"
            )
        for name, value in (
            ("target_ret", target_ret),
            ("target_valid", target_valid),
            ("bar_time", bar_time),
        ):
            if value.ndim != 2:
                raise DataValidationError(
                    "AlphaEngine training data contract violation: "
                    f"{name} expected rank 2 [N,T]; actual shape={tuple(value.shape)}"
                )

        symbols, bars = target_ret.shape
        if symbols < 1 or bars < 1 or feat_tensor.shape[1] < 1:
            raise DataValidationError(
                "AlphaEngine training data contract violation: expected positive N/F/T; "
                f"actual feat_tensor shape={tuple(feat_tensor.shape)} "
                f"target_ret shape={tuple(target_ret.shape)}"
            )
        expected_shape = (symbols, bars)
        for name, value in (("target_valid", target_valid), ("bar_time", bar_time)):
            if tuple(value.shape) != expected_shape:
                raise DataValidationError(
                    "AlphaEngine training data contract violation: "
                    f"{name} expected shape={expected_shape}; actual shape={tuple(value.shape)}"
                )
        if (feat_tensor.shape[0], feat_tensor.shape[2]) != expected_shape:
            raise DataValidationError(
                "AlphaEngine training data contract violation: feat_tensor expected "
                f"N/T={expected_shape}; actual N/T="
                f"{(feat_tensor.shape[0], feat_tensor.shape[2])}"
            )

        if not feat_tensor.is_floating_point():
            raise DataValidationError(
                "AlphaEngine training data contract violation: feat_tensor expected "
                f"floating dtype; actual={feat_tensor.dtype}"
            )
        if not target_ret.is_floating_point():
            raise DataValidationError(
                "AlphaEngine training data contract violation: target_ret expected "
                f"floating dtype; actual={target_ret.dtype}"
            )
        if target_ret.dtype != feat_tensor.dtype:
            raise DataValidationError(
                "AlphaEngine training data contract violation: target_ret expected dtype="
                f"{feat_tensor.dtype} matching feat_tensor; actual={target_ret.dtype}"
            )
        if target_valid.dtype != torch.bool:
            raise DataValidationError(
                "AlphaEngine training data contract violation: target_valid expected "
                f"torch.bool; actual={target_valid.dtype}"
            )
        if bar_time.dtype != torch.int64:
            raise DataValidationError(
                "AlphaEngine training data contract violation: bar_time expected "
                f"torch.int64 UTC nanoseconds; actual={bar_time.dtype}"
            )

        devices = {
            name: value.device
            for name, value in (
                ("feat_tensor", feat_tensor),
                ("target_ret", target_ret),
                ("target_valid", target_valid),
                ("bar_time", bar_time),
            )
        }
        if len(set(devices.values())) != 1 or feat_tensor.device.type == "meta":
            actual = ", ".join(f"{name}={device}" for name, device in devices.items())
            raise DataValidationError(
                "AlphaEngine training data contract violation: tensor devices must "
                f"match before transfer to configured device {ModelConfig.DEVICE}; actual {actual}"
            )

        return feat_tensor, target_ret, target_valid, bar_time

    def __init__(self, data_manager=None, use_lord_regularization=True,
                 lord_decay_rate=1e-3, lord_num_iterations=5,
                 n_folds: object = _CONFIGURED_N_FOLDS,
                 target_symbol: str | None = None):
        self.data_manager  = data_manager
        configured_n_folds = (
            ModelConfig.WF_N_BLOCKS
            if n_folds is _CONFIGURED_N_FOLDS
            else n_folds
        )
        self.n_folds       = self._validate_n_folds(configured_n_folds)
        self.target_symbol = target_symbol   # None = 多品种模式，str = 单品种模式
        self.model   = AlphaGPT().to(ModelConfig.DEVICE)
        self.opt     = torch.optim.AdamW(self.model.parameters(), lr=1e-3)

        self.use_lord = use_lord_regularization
        if self.use_lord:
            self.lord_opt = NewtonSchulzLowRankDecay(
                self.model.named_parameters(),
                decay_rate=lord_decay_rate,
                num_iterations=lord_num_iterations,
                target_keywords=["attention", "qk_norm"],
            )
            self.rank_monitor = StableRankMonitor(
                self.model, target_keywords=["in_proj", "out_proj", "qk_norm"]
            )
        else:
            self.lord_opt = None
            self.rank_monitor = None

        self.vm = StackVM()
        self.bt = MT5Backtest()

        from .vocab import FORMULA_VOCAB as _v
        self.sampler = ConstrainedSampler(
            vocab_size=_v.size, feat_offset=_v.operator_offset,
            arity_map=self.vm.arity_map,
            positive_only_ids=self.vm.positive_only_ids
        )

        self.best_score   = -float('inf')
        self.best_formula = None
        self._best_snapshot: dict | None = None

        self.training_history = {
            'step': [], 'avg_reward': [], 'best_score': [], 'val_score': [], 'stable_rank': []
        }
        self._restart_count      = 0
        self.factor_pool: list[tuple[float, int, torch.Tensor]] = []
        self._factor_pool_counter = 0

        # Elite Replay pool: (val_score, counter, formula_tokens, birth_step)
        self._elite_pool: list[tuple[float, int, list[int], int]] = []
        self._elite_counter = 0

        # 自适应噪声：记录 best 刷新步数
        self._best_update_step = 0
        self._stagnation_steps = 0

        # Fix 3: EMA reward baseline
        self._reward_ema: float | None = None
        self._reward_ema_step: int = 0
        self._low_entropy_streak: int = 0
        self._owned_public_artifacts: dict[
            str,
            tuple[
                tuple[bool, int, str | None],
                tuple[int, int, int],
            ],
        ] = {}

    # ── IC computation ────────────────────────────────────────────────────────

    @staticmethod
    def _compute_ic_components(
        factor: torch.Tensor,
        target_ret: torch.Tensor,
        target_valid: torch.Tensor,
                    ) -> tuple[torch.Tensor, torch.Tensor]:
        """时序 IC：同一有效索引上的 factor[t] 与 target_ret[t]。

        对 5 品种宇宙，时序 IC 比横截面 IC 统计意义更强。
        """
        return compute_ic_metrics(factor, target_ret, target_valid)

    @staticmethod
    def _compute_ic(
        factor: torch.Tensor,
        target_ret: torch.Tensor,
        target_valid: torch.Tensor,
    ) -> float:
        """Return mean same-index IC under the shared valid-label mask."""
        ic_mean, _ = AlphaEngine._compute_ic_components(
            factor, target_ret, target_valid
        )
        return float(ic_mean.item())

    @staticmethod
    def _compute_ic_stability(
        factor: torch.Tensor,
        target_ret: torch.Tensor,
        target_valid: torch.Tensor,
    ) -> float:
        """Return cross-symbol time-series IC stability under the same mask."""
        _, ic_stability = AlphaEngine._compute_ic_components(
            factor, target_ret, target_valid
        )
        return float(ic_stability.item())

    @staticmethod
    def _entropy_floor_loss(mean_entropy: torch.Tensor) -> torch.Tensor:
        """Return the configured entropy floor without detaching its graph."""
        if not ModelConfig.ENTROPY_FLOOR:
            return mean_entropy * 0.0
        return ModelConfig.ENTROPY_FLOOR_LAMBDA * torch.relu(
            mean_entropy.new_tensor(ModelConfig.ENTROPY_FLOOR_THRESH)
            - mean_entropy
        )

    @staticmethod
    def _stable_categorical_entropy(logits: torch.Tensor) -> torch.Tensor:
        """Return exact entropy values with a bounded, finite surrogate gradient.

        Softmax entropy has the right forward semantics, but its float32 backward
        either vanishes after saturation or overflows while centering very large
        opposing logits.  The straight-through surrogate evaluates entropy on
        logits scaled into [-1, 1] per row.  Its explicit backward preserves a
        directional gradient even at the finite float32 limits, while the
        forward leaves sampling and reported entropy unchanged.
        """
        return _StableCategoricalEntropy.apply(logits)

    # ── IC gate: direction-based, dimension-agnostic ──────────────────────────

    @staticmethod
    def _apply_ic_gate(reward: torch.Tensor, ic_mean) -> torch.Tensor:
        """IC 门控：用 IC 符号而非量值调整 reward，完全规避量纲问题。
        IC > thresh  → reward × IC_GATE_MULT  (正向预测，奖励)
        IC < -thresh → reward × IC_NEG_MULT   (反向预测，惩罚)
        |IC| ≤ thresh→ 不修改                  (噪声区)
        """
        ic_val = ic_mean.item() if isinstance(ic_mean, torch.Tensor) else float(ic_mean)
        t = ModelConfig.IC_GATE_THRESH
        if ic_val > t:
            return reward * ModelConfig.IC_GATE_MULT
        elif ic_val < -t:
            return reward * ModelConfig.IC_NEG_MULT
        return reward


    # ── Elite pool ────────────────────────────────────────────────────────────

    @staticmethod
    def _dedup_elite_pool(
        pool: list[tuple[float, int, list[int], int]]
    ) -> list[tuple[float, int, list[int], int]]:
        """对精英池去重：相同 tokens 只保留得分最高的一条，重建最小堆。"""
        best: dict[str, tuple[float, int, list[int], int]] = {}
        for sc, cnt, toks, birth in pool:
            key = str(toks)
            if key not in best or sc > best[key][0]:
                best[key] = (sc, cnt, toks, birth)
        deduped = list(best.values())
        heapq.heapify(deduped)
        return deduped

    @staticmethod
    def _push_elite_pool(
        pool: list[tuple[float, int, list[int], int]],
        counter: int,
        val_score: float,
        formula: list[int],
        step: int,
    ) -> int:
        """Apply one elite update to an explicit pool and return its counter."""
        k = ModelConfig.ELITE_POOL_SIZE
        for idx, (sc, _cnt, toks, _birth) in enumerate(pool):
            if toks == formula:
                if val_score <= sc:
                    return counter
                pool[idx] = pool[-1]
                pool.pop()
                heapq.heapify(pool)
                break
        entry = (val_score, counter, list(formula), step)
        counter += 1
        if len(pool) < k:
            heapq.heappush(pool, entry)
        elif val_score > pool[0][0]:
            heapq.heapreplace(pool, entry)
        return counter

    def _update_elite_pool(self, val_score: float, formula: list[int], step: int = 0) -> None:
        """维护精英公式池（最小堆，Top-ELITE_POOL_SIZE 个历史最优公式，自动去重）。

        去重逻辑：若 formula 已在池中，只在新得分更高时原地更新，不插入重复副本。
        这防止了单一公式垄断 elite pool，保持多样性。
        新增：记录 birth_step 用于 elite decay。
        """
        self._elite_counter = self._push_elite_pool(
            self._elite_pool, self._elite_counter, val_score, formula, step
        )

    # ── Factor pool ───────────────────────────────────────────────────────────

    @staticmethod
    def _push_factor_pool(
        pool: list[tuple[float, int, torch.Tensor]],
        counter: int,
        val_score: float,
        factor: torch.Tensor,
    ) -> int:
        k     = ModelConfig.FACTOR_TOP_K
        f_gpu = factor.detach()
        entry = (val_score, counter, f_gpu)
        counter += 1
        if len(pool) < k:
            heapq.heappush(pool, entry)
        elif val_score > pool[0][0]:
            heapq.heapreplace(pool, entry)
        return counter

    def _update_factor_pool(self, val_score: float, factor: torch.Tensor) -> None:
        self._factor_pool_counter = self._push_factor_pool(
            self.factor_pool,
            self._factor_pool_counter,
            val_score,
            factor,
        )

    def _begin_batch_transaction(self, next_step: int) -> _BatchTransaction:
        strategy_path = pathlib.Path(_strategy_file_for_symbol(self.target_symbol))
        history_path = pathlib.Path(
            f"training_history_{self.target_symbol}.json"
            if self.target_symbol else "training_history.json"
        )
        sym_tag = f"_{self.target_symbol}" if self.target_symbol else ""
        checkpoint_path = _CHECKPOINT_DIR / f"ckpt{sym_tag}_step_{next_step:04d}.pt"
        published_paths = [strategy_path, history_path, checkpoint_path]
        temporary_paths = [
            path.with_name(f".{path.name}.tmp") for path in published_paths
        ]
        return _BatchTransaction(self, published_paths + temporary_paths)

    def _commit_pending_actions(
        self,
        pending_actions: list[tuple],
        *,
        publish_strategy: bool = True,
    ) -> list[str]:
        """Replay one validated batch transaction and publish its final winner once."""
        before_best_score = self.best_score
        before_best_formula = self.best_formula
        before_best_snapshot = self._best_snapshot
        before_best_update_step = self._best_update_step
        before_stagnation_steps = self._stagnation_steps
        before_factor_pool = list(self.factor_pool)
        before_factor_counter = self._factor_pool_counter
        before_elite_pool = list(self._elite_pool)
        before_elite_counter = self._elite_counter
        messages: list[str] = []
        has_winner = False
        action = snapshot = buffered_factor = None
        try:
            for action in pending_actions:
                if action[0] == "elite":
                    _, final_val, fml, action_step = action
                    self._update_elite_pool(final_val, fml, action_step)
                    continue
                (
                    _, final_val, fml, snapshot, action_step, buffered_factor,
                    old_best, ic_i, exposure,
                ) = action
                self.best_score = final_val
                self.best_formula = fml
                self._best_snapshot = snapshot
                self._best_update_step = action_step
                self._stagnation_steps = 0
                self._update_factor_pool(final_val, buffered_factor)
                has_winner = True
                messages.append(
                    f"[!] 新最优 @ 第{action_step}步: 验证={final_val:.3f} "
                    f"(原 {old_best:.3f}，+{final_val-old_best:.3f}) "
                    f"IC={ic_i:.4f} 暴露度={exposure:.1%} | "
                    f"{fml}\n    {self._decode_formula(fml)}"
                )
            if has_winner and publish_strategy:
                self._save_strategy_live()
        except BaseException as failure:
            self.best_score = before_best_score
            self.best_formula = before_best_formula
            self._best_snapshot = before_best_snapshot
            self._best_update_step = before_best_update_step
            self._stagnation_steps = before_stagnation_steps
            self.factor_pool = before_factor_pool
            self._factor_pool_counter = before_factor_counter
            self._elite_pool = before_elite_pool
            self._elite_counter = before_elite_counter
            pending_actions.clear()
            messages.clear()
            action = snapshot = buffered_factor = None
            traceback.clear_frames(failure.__traceback__)
            raise
        pending_actions.clear()
        action = snapshot = buffered_factor = None
        return messages

    def _apply_corr_penalty(
        self,
        reward: torch.Tensor,
        factor: torch.Tensor,
        *,
        factor_pool: list[tuple[float, int, torch.Tensor]] | None = None,
    ) -> torch.Tensor:
        pool = self.factor_pool if factor_pool is None else factor_pool
        if not pool:
            return reward
        f_flat = factor.detach().reshape(-1)
        if not _has_exact_variation(f_flat):
            return reward
        pool_factors = [f.reshape(-1) for _, _cnt, f in pool]
        if any(pool_factor.shape != f_flat.shape for pool_factor in pool_factors):
            raise ValueError("factor pool entries must be shape-compatible")
        common_dtype = f_flat.dtype
        for pool_factor in pool_factors:
            common_dtype = torch.promote_types(common_dtype, pool_factor.dtype)
        if common_dtype in (torch.float16, torch.bfloat16):
            common_dtype = torch.float32
        f_normalized = _scale_invariant_centered(f_flat.to(dtype=common_dtype))
        pool_normalized = [
            _scale_invariant_centered(pool_factor.to(dtype=common_dtype))
            for pool_factor in pool_factors
            if _has_exact_variation(pool_factor)
        ]
        if not pool_normalized:
            return reward
        corr = (torch.stack(pool_normalized, dim=0) * f_normalized).sum(dim=1).abs()
        if (corr > ModelConfig.CORR_THRESHOLD).any():
            reward = reward * ModelConfig.CORR_PENALTY
        return reward

    @staticmethod
    def _fold_selection_index(
        total_bars: int,
        folds: list[WalkForwardFold],
        *,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Return the deterministic union of explicit train/validation indices."""
        intervals = []
        for fold in folds:
            for start, end in (
                (fold.train_start, fold.train_end),
                (fold.val_start, fold.val_end),
            ):
                if start < 0 or end > total_bars or start >= end:
                    raise ValueError("walk-forward fold bounds are invalid")
                intervals.append((start, end))
        if not intervals:
            raise ValueError("walk-forward folds select no factor bars")
        intervals.sort()
        merged = []
        for start, end in intervals:
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        return torch.cat(
            [
                torch.arange(start, end, dtype=torch.long, device=device)
                for start, end in merged
            ]
        )

    @staticmethod
    def _select_fold_bars(
        values: torch.Tensor, selection_index: torch.Tensor
    ) -> torch.Tensor:
        return torch.index_select(
            values, dim=-1, index=selection_index.to(values.device)
        )

    def _distribution_stats(self, prev_dist=None):
        """计算模型初始位置（zero prefix）token 分布的细化指标，用于判断 H 不变时
        分布是否真的在变化。
        """
        vocab_size = FORMULA_VOCAB.size
        with torch.no_grad():
            inp = torch.zeros((1, 1), dtype=torch.long,
                              device=ModelConfig.DEVICE)
            logits, _, _ = self.model(inp)
            logits = self.sampler.apply_mask_to_logits(
                logits, [0], 0, ModelConfig.MAX_FORMULA_LEN
            )
            dist = F.softmax(logits, dim=-1).squeeze(0)
            ent = -(dist * torch.log(dist + 1e-12)).sum().item()
            log_v = math.log(vocab_size)
            kl_uniform = log_v - ent
            top1 = dist.max().item()
            top5 = dist.topk(5, dim=-1).values.sum().item()
            eff_vocab = math.exp(ent)
            prob_std = dist.std(unbiased=False).item()
            kl_prev = 0.0
            if prev_dist is not None:
                kl_prev = (
                    dist * (torch.log(dist + 1e-12) -
                            torch.log(prev_dist.to(dist.device) + 1e-12))
                ).sum().item()
        return {
            'dist': dist.cpu(),
            'entropy': ent,
            'kl_uniform': kl_uniform,
            'top1_prob': top1,
            'top5_prob': top5,
            'eff_vocab': eff_vocab,
            'prob_std': prob_std,
            'kl_prev': kl_prev,
        }

    def _apply_adaptive_restart(self, step: int, ent_val: float) -> None:
        """Apply entropy-collapse recovery inside the current batch boundary."""
        if ent_val < ModelConfig.ENTROPY_COLLAPSE_THRESH:
            self._low_entropy_streak += 1
        else:
            self._low_entropy_streak = 0
        if self._low_entropy_streak < ModelConfig.ENTROPY_COLLAPSE_STEPS:
            return

        self._stagnation_steps = step - self._best_update_step
        stagnation_ratio = self._stagnation_steps / max(
            1, ModelConfig.STAGNATION_WINDOW
        )
        base_noise = ModelConfig.RESTART_NOISE
        if ModelConfig.ADAPTIVE_NOISE:
            raw_noise = (
                base_noise
                + ModelConfig.NOISE_BOOST_FACTOR
                * 0.1
                * min(stagnation_ratio, 3.0)
            )
            noise = max(
                ModelConfig.NOISE_MIN,
                min(ModelConfig.NOISE_MAX, raw_noise),
            )
        else:
            noise = base_noise

        max_r = ModelConfig.MAX_RESTARTS
        if self._restart_count < max_r:
            self._restart_count += 1
            self._low_entropy_streak = 0
            do_full_reset = (
                self._restart_count % ModelConfig.FULL_RESET_EVERY == 0
                or ent_val < 0.3
            )
            if do_full_reset:
                for layer in self.model.modules():
                    if hasattr(layer, 'reset_parameters'):
                        layer.reset_parameters()
                tqdm.write(
                    f"[重启 {self._restart_count}/{max_r} @ 第{step}步] "
                    "模式=完全重置（脱离最优快照吸引子） "
                    f"停滞={self._stagnation_steps} 熵={ent_val:.3f}"
                )
            elif self._best_snapshot is not None:
                self.model.load_state_dict(self._best_snapshot)
                with torch.no_grad():
                    perturbed_layers = []
                    for name, parameter in self.model.named_parameters():
                        if ModelConfig.PARTIAL_RESET:
                            selected = any(
                                key in name
                                for key in ModelConfig.PARTIAL_RESET_LAYERS
                            )
                        else:
                            selected = (
                                'ffn' in name
                                or 'attention' in name
                                or name.startswith('blocks')
                            )
                        if selected:
                            parameter.add_(torch.randn_like(parameter) * noise)
                            perturbed_layers.append(name)
                tqdm.write(
                    f"[重启 {self._restart_count}/{max_r} @ 第{step}步] "
                    f"模式={'部分层' if ModelConfig.PARTIAL_RESET else 'FFN/注意力'} "
                    f"噪声={noise:.4f}(基准={base_noise:.3f}，"
                    f"比率={stagnation_ratio:.2f}) "
                    f"停滞={self._stagnation_steps} 熵={ent_val:.3f} "
                    f"扰动层数={len(perturbed_layers)}"
                )
            else:
                with torch.no_grad():
                    for parameter in self.model.parameters():
                        parameter.add_(torch.randn_like(parameter) * noise)
                tqdm.write(
                    f"[重启 {self._restart_count}/{max_r} @ 第{step}步] "
                    f"模式=全参数 噪声={noise:.4f}(基准={base_noise:.3f}，"
                    f"比率={stagnation_ratio:.2f}) "
                    f"停滞={self._stagnation_steps} 熵={ent_val:.3f} | 无最优快照"
                )
            self._replace_optimizer_for_restart()
            return

        self._low_entropy_streak = 0
        hard_noise = min(ModelConfig.NOISE_MAX, noise * 2.0)
        if self._best_snapshot is not None:
            self.model.load_state_dict(self._best_snapshot)
        with torch.no_grad():
            for parameter in self.model.parameters():
                parameter.add_(torch.randn_like(parameter) * hard_noise)
        self._replace_optimizer_for_restart()
        tqdm.write(
            f"[强重启 @ 第{step}步] 已达最大重启次数={max_r} "
            f"熵={ent_val:.3f} 强噪声={hard_noise:.4f} 继续训练，不提前停止"
        )

    def _replace_optimizer_for_restart(self) -> None:
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=1e-3)

    # ── Main training loop ────────────────────────────────────────────────────

    def train(self, start_step: int = 0, end_step: int | None = None,
              migration_hook=None, verbose_header: bool = True):
        if self.data_manager is None:
            raise RuntimeError("AlphaEngine requires a data_manager.")

        feat, t_ret, t_valid, bar_time_ns = self._validate_training_tensors(
            self.data_manager
        )
        n_folds = self._validate_n_folds(self.n_folds)

        if end_step is None:
            end_step = ModelConfig.TRAIN_STEPS

        if verbose_header:
            print("开始 Alpha 因子挖掘训练" +
                  ("（含 LoRD 正则化）..." if self.use_lord else "..."))
            print(f"   策略熵: 坍塌阈值={ModelConfig.ENTROPY_COLLAPSE_THRESH}  "
                  f"系数上限={ModelConfig.ENTROPY_COEFF_MAX}  "
                  f"连续坍塌步数={ModelConfig.ENTROPY_COLLAPSE_STEPS}")
            print(f"   精英回放: 比例={ModelConfig.ELITE_REPLAY_FRAC}  "
                  f"池大小={ModelConfig.ELITE_POOL_SIZE}")
            print(f"   IC门控: 阈值±{ModelConfig.IC_GATE_THRESH}  "
                  f"正向×{ModelConfig.IC_GATE_MULT}  负向×{ModelConfig.IC_NEG_MULT}")
            print(f"   最大重启: {ModelConfig.MAX_RESTARTS}  "
                  f"噪声={ModelConfig.RESTART_NOISE}")

        T = t_ret.shape[1]
        formula_warmup = formula_warmup_bars(ModelConfig.MAX_FORMULA_LEN)
        required_bars = required_training_bars(
            warmup_bars=formula_warmup,
            label_lookahead=LABEL_LOOKAHEAD_BARS,
            n_blocks=n_folds,
            min_fold_bars=ModelConfig.WF_MIN_FOLD_BARS,
            configured_gap=ModelConfig.WF_GAP,
        )
        assert_minimum_bars(
            T,
            required_bars,
            context="AlphaEngine training dataset",
        )
        folds = build_walk_forward_folds(
            total_bars=T,
            n_blocks=n_folds,
            configured_gap=ModelConfig.WF_GAP,
            min_fold_bars=ModelConfig.WF_MIN_FOLD_BARS,
            warmup_bars=formula_warmup,
            label_lookahead=LABEL_LOOKAHEAD_BARS,
        )
        if verbose_header:
            print(f"   滚动验证: {len(folds)} 折  共 {T} 根K线")
            for k, fold in enumerate(folds):
                print(
                    f"  第{k+1}折: 训练[{fold.train_start},{fold.train_end}) "
                    f"间隔={fold.effective_gap} "
                    f"验证[{fold.val_start},{fold.val_end})"
                )

        # 因果安全：features.py 的 _robust_norm 已改为滚动因果实现
        # 每个 t 的归一化参数只用 [t-w+1..t]，walk-forward 折叠切片无泄露
        feat = feat.to(ModelConfig.DEVICE)
        t_ret = t_ret.to(ModelConfig.DEVICE)
        t_valid = t_valid.to(ModelConfig.DEVICE)
        bar_time_ns = bar_time_ns.to(ModelConfig.DEVICE)
        selection_index = self._fold_selection_index(
            T, folds, device=ModelConfig.DEVICE
        )
        selection_target = self._select_fold_bars(t_ret, selection_index)
        selection_valid = self._select_fold_bars(t_valid, selection_index)
        bs      = ModelConfig.BATCH_SIZE
        n_elite = max(1, int(bs * ModelConfig.ELITE_REPLAY_FRAC))
        n_new   = bs - n_elite

        remaining = end_step - start_step
        if remaining <= 0:
            print(f"[训练] 起始步 {start_step} 已达目标步 {end_step}，无需继续训练。")
            return

        # 非交互/重定向输出时关闭 tqdm 进度条，避免进度条刷屏把自定义日志淹掉。
        # tqdm.write 仍然可用，详细 step 日志会继续输出。
        pbar               = tqdm(range(start_step, end_step),
                                  total=end_step,
                                  initial=start_step,
                                  disable=not sys.stderr.isatty(),
                                  leave=False,
                                  mininterval=5.0)
        prev_init_dist     = None  # 用于计算相邻步分布差异 KL

        for step in pbar:
            # The transaction starts before every stochastic draw belonging to
            # this batch, so a failed attempt can be retried deterministically.
            transaction = self._begin_batch_transaction(step + 1)
            # ── Part A: Sample n_new new formulas ────────────────────
            inp_new = torch.zeros((n_new, 1), dtype=torch.long,
                                  device=ModelConfig.DEVICE)
            lp_new, tok_new, ent_new = [], [], []
            sd_new = [0] * n_new
            prev_tokens_new: list[int | None] = [None] * n_new
            infected_chain_new: list[int] = [0] * n_new

            for si in range(ModelConfig.MAX_FORMULA_LEN):
                lg, _, _ = transaction.run(self.model, inp_new)
                lg = transaction.run(
                    self.sampler.apply_mask_to_logits,
                    lg, sd_new, si, ModelConfig.MAX_FORMULA_LEN,
                    prev_tokens=prev_tokens_new,
                    infected_chain_lens=infected_chain_new,
                )
                d  = Categorical(logits=lg)
                a  = transaction.run(d.sample)
                lp_new.append(d.log_prob(a))
                tok_new.append(a)
                ent_new.append(
                    transaction.run(self._stable_categorical_entropy, lg)
                )
                inp_new = torch.cat([inp_new, a.unsqueeze(1)], dim=1)
                for b in range(n_new):
                    sd_new[b] += self.sampler.delta[a[b].item()]
                    prev_tokens_new[b] = a[b].item()
                    infected_chain_new[b] = transaction.run(
                        self.sampler.update_infection,
                        a[b].item(), infected_chain_new[b],
                    )

            seqs_new = torch.stack(tok_new, dim=1)


            # ── Part B: Elite Replay ─────────────────────────────────
            elite_formulas: list[list[int]] = []
            if self._elite_pool and n_elite > 0:
                ps = []
                pt = []
                weights = []
                for sc, cnt, toks, birth in self._elite_pool:
                    age = max(0, step - birth)
                    decay = 1.0
                    if ModelConfig.ELITE_DECAY:
                        half = max(1, ModelConfig.ELITE_DECAY_HALF_LIFE)
                        decay = 0.5 ** (age / half)
                    ps.append(sc)
                    pt.append(toks)
                    weights.append(decay)
                ps_min  = min(ps)
                ps_max  = max(ps)
                # 软温度采样：避免最高分公式垄断
                # 先归一到 [0,1]，再除以温度 T=0.5 后做 softmax
                # T<1 → 高分公式仍被偏好，但不再独占
                if ps_max > ps_min:
                    normalized = [(s - ps_min) / (ps_max - ps_min + 1e-8) for s in ps]
                else:
                    normalized = [1.0] * len(ps)
                temp = 0.5
                exp_s = [weights[i] * (2.0 ** (normalized[i] / temp)) for i in range(len(ps))]
                exp_sum = sum(exp_s)
                probs = [e / exp_sum for e in exp_s]
                idx_e = transaction.run(
                    random.choices,
                    range(len(self._elite_pool)),
                    weights=probs,
                    k=n_elite,
                )
                elite_formulas = [pt[i] for i in idx_e]

                # 详细日志：Elite Replay 衰减状态（每 100 步打印一次）
                if step % 100 == 0:
                    avg_decay = sum(weights) / len(weights)
                    max_age = max(max(0, step - birth) for _, _, _, birth in self._elite_pool)
                    age_list = sorted([max(0, step - birth) for _, _, _, birth in self._elite_pool])
                    tqdm.write(
                        f"[精英回放 @ 第{step}步] 池大小={len(self._elite_pool)} "
                        f"平均衰减={avg_decay:.3f} 最大龄期={max_age} 龄期列表={age_list} "
                        f"抽样分数=[{', '.join(f'{ps[i]:.3f}' for i in idx_e[:3])}...]"
                    )
            else:
                elite_formulas = seqs_new[:n_elite].tolist()

            lp_elite, ent_elite = [], []
            if elite_formulas:
                ne     = len(elite_formulas)
                inp_e  = torch.zeros((ne, 1), dtype=torch.long,
                                     device=ModelConfig.DEVICE)
                sd_e   = [0] * ne
                prev_tokens_elite: list[int | None] = [None] * ne
                infected_chain_elite: list[int] = [0] * ne
                tok_e_t = torch.tensor(elite_formulas, dtype=torch.long,
                                       device=ModelConfig.DEVICE)
                for si in range(ModelConfig.MAX_FORMULA_LEN):
                    lg_e, _, _ = transaction.run(self.model, inp_e)
                    lg_e = transaction.run(
                        self.sampler.apply_mask_to_logits,
                        lg_e, sd_e, si, ModelConfig.MAX_FORMULA_LEN,
                        prev_tokens=prev_tokens_elite,
                        infected_chain_lens=infected_chain_elite,
                    )
                    d_e  = Categorical(logits=lg_e)
                    tk   = tok_e_t[:, si]
                    lp_elite.append(d_e.log_prob(tk))
                    ent_elite.append(
                        transaction.run(self._stable_categorical_entropy, lg_e)
                    )
                    inp_e = torch.cat([inp_e, tk.unsqueeze(1)], dim=1)
                    for b in range(ne):
                        sd_e[b] += self.sampler.delta[tk[b].item()]
                        prev_tokens_elite[b] = tk[b].item()
                        infected_chain_elite[b] = transaction.run(
                            self.sampler.update_infection,
                            tk[b].item(), infected_chain_elite[b],
                        )


            # ── Part C: Evaluate all formulas ────────────────────────
            all_fmls = seqs_new.tolist() + elite_formulas
            tot      = len(all_fmls)
            # Fold scorers intentionally return promoted metrics independently
            # of the training tensor dtype.  Accumulate in float64 so every
            # finite supported scorer result remains representable.
            score_dtype = torch.float64
            rewards = torch.zeros(
                tot, device=ModelConfig.DEVICE, dtype=score_dtype
            )
            val_scores = torch.zeros(
                tot, device=ModelConfig.DEVICE, dtype=score_dtype
            )

            ok_cnt = none_cnt = const_cnt = 0
            step_max_val = -float('inf');  step_best_f = None
            bic, bis, bsor = [], [], []

            for i, fml in enumerate(all_fmls):
                try:
                    with torch.no_grad():
                        res = transaction.run(self.vm.execute, fml, feat)
                except BaseException as failure:
                    traceback.clear_frames(failure.__traceback__)
                    raise
                if res is None:
                    raise RuntimeError(
                        "training program evaluation failed: vm.execute returned None; "
                        f"step={step} formula_index={i} formula_length={len(fml)}"
                    )
                del res

            shadow_best_score = self.best_score
            shadow_factor_pool = list(self.factor_pool)
            shadow_factor_counter = self._factor_pool_counter
            shadow_elite_pool = list(self._elite_pool)
            shadow_elite_counter = self._elite_counter
            pending_actions: list[tuple] = []
            buffered_factor = snapshot = pos_check = None

            try:
                for i, fml in enumerate(all_fmls):
                    res = None
                    selection_factor = None
                    with torch.no_grad():
                        res = transaction.run(self.vm.execute, fml, feat)
                    if res is None:
                        raise RuntimeError(
                            "training program evaluation failed: vm.execute returned None; "
                            f"step={step} formula_index={i} formula_length={len(fml)}"
                        )
                    selection_factor = self._select_fold_bars(res, selection_index)
                    if not _has_exact_variation(selection_factor):
                        rewards[i] = val_scores[i] = -2.0
                        const_cnt += 1
                    else:
                        ok_cnt += 1

                        with torch.no_grad():
                            fold_tr, fold_vl, fold_ic = [], [], []
                            for fold in folds:
                                tr_sc, vl_sc = transaction.run(
                                    self.bt.evaluate_fold,
                                    factors=res,
                                    target_ret=t_ret,
                                    target_valid=t_valid,
                                    bar_time_ns=bar_time_ns,
                                    train_start=fold.train_start,
                                    train_end=fold.train_end,
                                    val_start=fold.val_start,
                                    val_end=fold.val_end,
                                )
                                ic_m = AlphaEngine._compute_ic(
                                    res[:, fold.train_start:fold.train_end],
                                    t_ret[:, fold.train_start:fold.train_end],
                                    t_valid[:, fold.train_start:fold.train_end],
                                )
                                tr_adj = AlphaEngine._apply_ic_gate(tr_sc, ic_m)
                                fold_tr.append(ModelConfig.REWARD_ALPHA * tr_adj)
                                fold_vl.append(vl_sc)
                                fold_ic.append(ic_m)
                            train_score = torch.stack(fold_tr).mean()
                            val_score = torch.stack(fold_vl).mean()
                            ic_i = sum(fold_ic) / len(fold_ic)
                            ic_selection = AlphaEngine._compute_ic(
                                selection_factor, selection_target, selection_valid
                            )
                            ic_stab_selection = AlphaEngine._compute_ic_stability(
                                selection_factor, selection_target, selection_valid
                            )

                        rewards[i]    = train_score
                        val_scores[i] = val_score
                        bic.append(ic_selection);  bis.append(ic_stab_selection)
                        bsor.append(val_score.item())

                        # Score against a shadow pool so successful-batch ordering
                        # is preserved without committing state during evaluation.
                        rp = _repetition_penalty(fml)
                        if rp > 0:
                            rewards[i]    -= rp
                            val_scores[i] -= rp
                        rewards[i] = self._apply_corr_penalty(
                            rewards[i], selection_factor,
                            factor_pool=shadow_factor_pool,
                        )
                        val_scores[i] = self._apply_corr_penalty(
                            val_scores[i], selection_factor,
                            factor_pool=shadow_factor_pool,
                        )

                        final_val = val_scores[i].item()
                        if final_val > step_max_val:
                            step_max_val = final_val;  step_best_f = fml

                        if final_val > shadow_best_score:
                            train_val = rewards[i].item()
                            if train_val > 0.5 and final_val < train_val * 0.5:
                                tqdm.write(
                                    f"[过拟合跳过 @ 第{step}步] 验证={final_val:.3f} "
                                    f"训练={train_val:.3f} 比值={final_val/train_val:.2f} | 样本外表现过差"
                                )
                            else:
                                pos_check = compute_target_positions_stateless(selection_factor)
                                exposure = pos_check.abs().mean().item()
                                if exposure < 0.05:
                                    tqdm.write(
                                        f"[稀疏跳过 @ 第{step}步] 验证={final_val:.3f} "
                                        f"IC={ic_i:.4f} 暴露度={exposure:.1%} | 仓位过稀疏，不更新最优"
                                    )
                                else:
                                    old_best = shadow_best_score
                                    snapshot = copy.deepcopy(self.model.state_dict())
                                    buffered_factor = selection_factor.detach()
                                    pending_actions.append(
                                        (
                                            "best", final_val, list(fml), snapshot,
                                            step, buffered_factor, old_best, ic_i, exposure,
                                        )
                                    )
                                    shadow_best_score = final_val
                                    shadow_factor_counter = self._push_factor_pool(
                                        shadow_factor_pool,
                                        shadow_factor_counter,
                                        final_val,
                                        buffered_factor,
                                    )

                        shadow_elite_counter = self._push_elite_pool(
                            shadow_elite_pool,
                            shadow_elite_counter,
                            final_val,
                            fml,
                            step,
                        )
                        pending_actions.append(("elite", final_val, list(fml), step))
                    del selection_factor, res
            except BaseException as failure:
                if transaction.active:
                    try:
                        transaction.rollback()
                    except BaseException as rollback_failure:
                        failure.add_note(
                            "batch rollback also failed: "
                            f"{type(rollback_failure).__name__}: {rollback_failure}"
                        )
                pending_actions.clear()
                shadow_factor_pool.clear()
                shadow_elite_pool.clear()
                del selection_factor, res
                del buffered_factor, snapshot, pos_check
                traceback.clear_frames(failure.__traceback__)
                raise

            # ── Part D: REINFORCE gradient update ────────────────────
            # Fix 3: EMA baseline 替代 batch mean，避免全负 batch 的相对优选问题
            batch_mean = rewards.mean().item()
            batch_std = rewards.std().clamp(min=rewards.new_tensor(0.1))
            if ModelConfig.REWARD_EMA_BASELINE and self._reward_ema_step >= ModelConfig.REWARD_EMA_WARMUP:
                baseline = self._reward_ema
                adv = (rewards - baseline) / (batch_std + 1e-5)
            else:
                adv = (rewards - batch_mean) / (batch_std + 1e-5)
            # Prepare the next EMA without publishing it until the loss graph
            # has been built successfully.
            if self._reward_ema is None:
                next_reward_ema = batch_mean
            else:
                next_reward_ema = (
                    ModelConfig.REWARD_EMA_DECAY * self._reward_ema
                    + (1.0 - ModelConfig.REWARD_EMA_DECAY) * batch_mean
                )
            next_reward_ema_step = self._reward_ema_step + 1
            adv_new   = adv[:n_new]
            adv_elite = adv[n_new:]

            policy_loss = torch.zeros(
                1, device=ModelConfig.DEVICE, dtype=rewards.dtype
            )
            for ti in range(len(lp_new)):
                policy_loss = policy_loss + (
                    -lp_new[ti].to(dtype=rewards.dtype) * adv_new
                ).mean()
            if lp_elite and adv_elite.shape[0] > 0:
                for ti in range(len(lp_elite)):
                    lpe = lp_elite[ti]
                    if lpe.shape[0] == adv_elite.shape[0]:
                        policy_loss = policy_loss + (
                            -lpe.to(dtype=rewards.dtype)
                            * adv_elite
                            * ModelConfig.ELITE_REWARD_SCALE
                        ).mean()

            if ent_new:
                mean_ent_new = torch.stack(ent_new).mean()
            else:
                mean_ent_new = torch.zeros(1, device=ModelConfig.DEVICE)
            if ent_elite:
                mean_ent_elite = torch.stack(ent_elite).mean()
                mean_ent = (
                    mean_ent_new * n_new + mean_ent_elite * n_elite
                ) / (n_new + n_elite)
            else:
                mean_ent = mean_ent_new
            ent_val   = mean_ent.item()
            ent_coeff = ModelConfig.ENTROPY_COEFF_MAX / (
                (1.0 + ent_val) ** ModelConfig.ENTROPY_COEFF_POWER
            )
            ent_floor_loss = self._entropy_floor_loss(mean_ent)
            loss = (
                policy_loss
                - ent_coeff * mean_ent.to(dtype=rewards.dtype)
                + ent_floor_loss.to(dtype=rewards.dtype)
            )

            self._reward_ema = next_reward_ema
            self._reward_ema_step = next_reward_ema_step
            self.opt.zero_grad()
            transaction.run(loss.backward)
            transaction.run(
                torch.nn.utils.clip_grad_norm_,
                self.model.parameters(),
                max_norm=1.0,
            )
            transaction.run(self.opt.step)
            if self.use_lord:
                transaction.run(self.lord_opt.step)

            # ── Part D2: 分布细化指标 ────────────────────────────────
            dst = transaction.run(self._distribution_stats, prev_init_dist)
            prev_init_dist = dst['dist']
            with torch.no_grad():
                uniq_tokens = seqs_new.unique().numel()
                uniq_fmls   = torch.unique(seqs_new, dim=0).shape[0]
                fml_div     = uniq_fmls / max(1, n_new)

            # Internal computation and optimizer mutation succeeded.  Only
            # now publish the prepared best/pool transaction and strategy.
            commit_messages = transaction.run(
                self._commit_pending_actions,
                pending_actions,
                publish_strategy=False,
            )
            if commit_messages:
                strategy_path = pathlib.Path(
                    _strategy_file_for_symbol(self.target_symbol)
                )
                transaction.run_artifact(
                    self._save_strategy_live,
                    [
                        strategy_path,
                        strategy_path.with_name(f".{strategy_path.name}.tmp"),
                    ],
                )
            for message in commit_messages:
                tqdm.write(message)
            commit_messages.clear()

            # ── Part E: Logging & history ────────────────────────────
            avg_rew = rewards.mean().item()
            avg_val = val_scores.mean().item()
            bim  = sum(bic)  / len(bic)  if bic  else 0.0
            bis_ = sum(bis)  / len(bis)  if bis  else 0.0
            bsor_= sum(bsor) / len(bsor) if bsor else 0.0

            self._stagnation_steps = step - self._best_update_step
            tqdm.write(
                f"[{step+1}/{end_step}] "
                f"新公式={n_new} 精英={n_elite} | "
                f"有效={ok_cnt} 无效={none_cnt} 常数={const_cnt} | "
                f"奖励={avg_rew:.3f} 验证={avg_val:.3f} | "
                f"IC={bim:.4f} | 熵={ent_val:.3f}(系数={ent_coeff:.3f}) | "
                f"最优={self.best_score:.3f} 停滞={self._stagnation_steps} "
                f"精英池={len(self._elite_pool)} 重启={self._restart_count}"
            )
            tqdm.write(
                f"   分布: 初始熵={dst['entropy']:.3f} KL均匀={dst['kl_uniform']:.3f} "
                f"KL上步={dst['kl_prev']:.4f} 最高概率={dst['top1_prob']:.3f} "
                f"前五概率={dst['top5_prob']:.3f} 有效词汇={dst['eff_vocab']:.2f} "
                f"标准差={dst['prob_std']:.4f} | "
                f"本批: 唯一符号={uniq_tokens}/{FORMULA_VOCAB.size} "
                f"唯一公式={uniq_fmls}/{n_new} 多样性={fml_div:.2f}"
            )
            pbar.set_postfix({
                '验证': f"{avg_val:.3f}", '最优': f"{self.best_score:.3f}",
                '熵':   f"{ent_val:.2f}", 'IC':   f"{bim:.4f}",
                '停滞': f"{self._stagnation_steps}",
                '初始熵':  f"{dst['entropy']:.2f}",
                'KL上步': f"{dst['kl_prev']:.3f}",
            })

            if self.use_lord and step % 10 == 0:
                sr = self.rank_monitor.compute()
                self.training_history['stable_rank'].append(sr)

            self.training_history['step'].append(step)
            self.training_history['avg_reward'].append(avg_rew)
            self.training_history['val_score'].append(avg_val)
            self.training_history['best_score'].append(self.best_score)
            self.training_history.setdefault('entropy', []).append(ent_val)
            self.training_history.setdefault('ic_mean', []).append(bim)
            self.training_history.setdefault('ic_stability', []).append(bis_)
            self.training_history.setdefault('sortino', []).append(bsor_)
            self.training_history.setdefault('elite_pool_size', []).append(
                len(self._elite_pool))
            self.training_history.setdefault('init_entropy', []).append(dst['entropy'])
            self.training_history.setdefault('kl_uniform', []).append(dst['kl_uniform'])
            self.training_history.setdefault('kl_prev', []).append(dst['kl_prev'])
            self.training_history.setdefault('top1_prob', []).append(dst['top1_prob'])
            self.training_history.setdefault('eff_vocab', []).append(dst['eff_vocab'])
            self.training_history.setdefault('batch_uniq_tokens', []).append(uniq_tokens)
            self.training_history.setdefault('batch_uniq_fmls', []).append(uniq_fmls)
            self.training_history.setdefault('batch_fml_div', []).append(fml_div)

            history_path = pathlib.Path(
                f"training_history_{self.target_symbol}.json"
                if self.target_symbol else "training_history.json"
            )
            transaction.run_artifact(
                self._save_training_history_live,
                [
                    history_path,
                    history_path.with_name(f".{history_path.name}.tmp"),
                ],
            )

            # ── Part F: Entropy collapse detection & restart ─────────
            transaction.run(self._apply_adaptive_restart, step, ent_val)

            # A successful-step checkpoint represents the complete supported
            # post-restart state and is still inside the rollback boundary.
            if (step + 1) % 20 == 0 or (step + 1) == end_step:
                sym_tag = f"_{self.target_symbol}" if self.target_symbol else ""
                checkpoint_path = (
                    _CHECKPOINT_DIR / f"ckpt{sym_tag}_step_{step + 1:04d}.pt"
                )
                ckpt = transaction.run_artifact(
                    self.save_checkpoint,
                    [
                        checkpoint_path,
                        checkpoint_path.with_name(
                            f".{checkpoint_path.name}.tmp"
                        ),
                    ],
                    step + 1,
                )
                tqdm.write(f"[检查点] → {ckpt} (最优={self.best_score:.3f})")

            transaction.commit()

            # ── Part G: Migration hook（多岛训练时交换精英）────────────
            # The hook remains outside the rollback contract because its
            # external side effects cannot be reversed by this engine.
            if migration_hook is not None and (step + 1) % ModelConfig.MIGRATION_INTERVAL == 0:
                tqdm.write(f"[迁移钩子 @ 第{step+1}步] 调用已注册钩子")
                migration_hook(self, step + 1)

        # ── End of training ──────────────────────────────────────────
        # 仅当跑满最终步时才保存最终 strategy 和历史
        if end_step == ModelConfig.TRAIN_STEPS:
            if self.best_formula is not None:
                from .vocab import VOCAB_VERSION
                strategy_data = {
                    "vocab_version": VOCAB_VERSION,
                    "symbol": self.target_symbol,
                    "formula": self.best_formula,
                    "best_score": self.best_score,
                }
                save_path = _strategy_file_for_symbol(self.target_symbol)
                _atomic_json_replace(
                    pathlib.Path(save_path),
                    strategy_data,
                    indent=2,
                    publisher=self._publish_owned_artifact_temp,
                )

            sym_tag = f"[{self.target_symbol}] " if self.target_symbol else ""
            self.training_history.pop('_low_entropy_streak', None)
            hist_path = (
                f"training_history_{self.target_symbol}.json"
                if self.target_symbol else "training_history.json"
            )
            _atomic_json_replace(
                pathlib.Path(hist_path), self.training_history
            )

            print(f"\n[完成] {sym_tag}训练结束！")
            print(f"  最优验证分数 : {self.best_score:.4f}")
            print(f"  最优公式令牌 : {self.best_formula}")
            print(f"  可读公式     : {self._decode_formula(self.best_formula)}")
            print(f"  精英池大小   : {len(self._elite_pool)}")
            print(f"  精英衰减     : 启用={ModelConfig.ELITE_DECAY}，半衰期={ModelConfig.ELITE_DECAY_HALF_LIFE}")
            print(f"  自适应噪声   : 启用={ModelConfig.ADAPTIVE_NOISE}，范围=[{ModelConfig.NOISE_MIN}, {ModelConfig.NOISE_MAX}]")
            print(f"  部分层重置   : 启用={ModelConfig.PARTIAL_RESET}，层={ModelConfig.PARTIAL_RESET_LAYERS}")
            print(f"  重启次数     : {self._restart_count}")
            print(f"  策略已保存   : {save_path}")


    # ── 实时保存最优公式（防进程意外退出丢失）────────────────────────────────
    @staticmethod
    def _artifact_ownership_key(path: pathlib.Path) -> str:
        return os.path.normcase(str(path.resolve()))

    def _artifact_ownership_records(self):
        records = getattr(self, "_owned_public_artifacts", None)
        if records is None:
            records = {}
            self._owned_public_artifacts = records
        return records

    def _assert_artifact_owned_or_absent(self, path: pathlib.Path) -> None:
        path = pathlib.Path(path)
        version, identity = _artifact_path_observation(path)
        key = self._artifact_ownership_key(path)
        records = self._artifact_ownership_records()
        if not version[0]:
            records.pop(key, None)
            return
        if identity is None or records.get(key) != (version, identity):
            raise RuntimeError(
                f"artifact publication conflict: unowned existing artifact "
                f"{str(path)[:160]}"
            )

    def _record_owned_artifact(self, path: pathlib.Path) -> None:
        path = pathlib.Path(path)
        version, identity = _artifact_path_observation(path)
        if not version[0] or identity is None:
            raise RuntimeError(
                f"artifact publication failed to establish ownership: "
                f"{str(path)[:160]}"
            )
        self._artifact_ownership_records()[self._artifact_ownership_key(path)] = (
            version,
            identity,
        )

    def _publish_owned_artifact_temp(
        self, temporary: pathlib.Path, target: pathlib.Path
    ) -> None:
        self._assert_artifact_owned_or_absent(target)
        _replace_artifact_publication_temp(temporary, target)
        self._record_owned_artifact(target)

    def _save_training_history_live(self) -> _ArtifactPublicationReceipt | None:
        """周期性写入训练曲线 JSON，供 Web UI 实时展示。"""
        if not self.target_symbol:
            return _artifact_publication_result(None, {})
        hist_path = pathlib.Path(f"training_history_{self.target_symbol}.json")
        temporary = hist_path.with_name(f".{hist_path.name}.tmp")
        try:
            payload = {
                k: v for k, v in self.training_history.items()
                if k != "_low_entropy_streak"
            }
            with open(temporary, "w", encoding="utf-8") as fp:
                json.dump(payload, fp)
                fp.flush()
            published = _artifact_path_version(temporary)
            _replace_artifact_publication_temp(temporary, hist_path)
            return _artifact_publication_result(
                None,
                {
                    hist_path: published,
                    temporary: _ABSENT_ARTIFACT_VERSION,
                },
            )
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    def _save_strategy_live(self) -> _ArtifactPublicationReceipt | None:
        """每次 best_formula 更新时立即保存 strategy json。
        即使训练中途进程被杀（OOM/终端回收/Ctrl+C），也能保留最新最优公式。
        """
        if self.best_formula is None:
            return _artifact_publication_result(None, {})
        from .vocab import VOCAB_VERSION
        save_path = _strategy_file_for_symbol(self.target_symbol)
        p = pathlib.Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self._assert_artifact_owned_or_absent(p)

        strategy_data = {
            "vocab_version": VOCAB_VERSION,
            "symbol": self.target_symbol,
            "formula": self.best_formula,
            "best_score": self.best_score,
            "formula_decoded": self._decode_formula(self.best_formula),
        }
        # 保留训练数据路径等元数据，避免 live 保存把 data_file 冲掉
        for key in ("timeframe", "data_file", "mode", "train_steps"):
            val = getattr(self, key, None)
            if val is not None:
                strategy_data[key] = val

        temporary = p.with_name(f".{p.name}.tmp")
        try:
            with open(temporary, "w", encoding="utf-8") as fp:
                json.dump(strategy_data, fp, indent=2, ensure_ascii=False)
                fp.flush()
            published = _artifact_path_version(temporary)
            self._publish_owned_artifact_temp(temporary, p)
            return _artifact_publication_result(
                None,
                {
                    p: published,
                    temporary: _ABSENT_ARTIFACT_VERSION,
                },
            )
        except BaseException:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            raise

    # ── Checkpoint save / load ────────────────────────────────────────────────

    def save_checkpoint(self, step: int, path: str | None = None) -> str:
        _CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        if path is None:
            sym_tag = f"_{self.target_symbol}" if self.target_symbol else ""
            path = str(_CHECKPOINT_DIR / f"ckpt{sym_tag}_step_{step:04d}.pt")
        ckpt = {
            "step":                 step,
            "vocab_version":        VOCAB_VERSION,   # task 12.2: 版本校验所需
            "model_state_dict":     self.model.state_dict(),
            "optimizer_state_dict": self.opt.state_dict(),
            "best_score":           self.best_score,
            "best_formula":         self.best_formula,
            "best_snapshot":        self._best_snapshot,
            "factor_pool":          self.factor_pool,
            "factor_pool_counter":  self._factor_pool_counter,
            "elite_pool":           self._elite_pool,
            "elite_counter":        self._elite_counter,
            "restart_count":        self._restart_count,
            "training_history":     {
                k: v for k, v in self.training_history.items()
                if k != '_low_entropy_streak'
            },
        }
        target = pathlib.Path(path)
        temporary = target.with_name(f".{target.name}.tmp")
        self._assert_artifact_owned_or_absent(target)
        try:
            torch.save(ckpt, temporary)
            published = _artifact_path_version(temporary)
            self._publish_owned_artifact_temp(temporary, target)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        return _artifact_publication_result(
            path,
            {
                target: published,
                temporary: _ABSENT_ARTIFACT_VERSION,
            },
        )

    def load_checkpoint(self, path: str) -> int:
        ckpt = torch.load(path, map_location=ModelConfig.DEVICE)

        # ── Task 12.2：版本校验（R3.7）──────────────────────────────────────
        # 从 checkpoint 读取 vocab_version；若字段缺失（旧版 checkpoint），视为
        # 版本不匹配并抛错——拒绝加载、不消费任何 token。
        artifact_version = ckpt.get("vocab_version")
        if artifact_version is None:
            raise VocabVersionMismatchError(
                f"checkpoint '{path}' 不含 vocab_version 字段（旧版产物），"
                f"当前词表版本 {FORMULA_VOCAB.version!r}；需重新训练后加载"
            )
        # verify() 版本不匹配时抛 VocabVersionMismatchError，拒绝加载
        FORMULA_VOCAB.verify(artifact_version)
        # ── 版本校验通过，继续加载 ────────────────────────────────────────

        self.model.load_state_dict(ckpt["model_state_dict"])
        self.opt.load_state_dict(ckpt["optimizer_state_dict"])
        self.best_score          = ckpt.get("best_score",  -float('inf'))
        self.best_formula        = ckpt.get("best_formula", None)
        self._best_snapshot      = ckpt.get("best_snapshot", None)
        self.factor_pool         = ckpt.get("factor_pool", [])
        self._factor_pool_counter = ckpt.get("factor_pool_counter", 0)
        self._elite_pool         = ckpt.get("elite_pool", [])
        self._elite_counter      = ckpt.get("elite_counter", 0)
        self._restart_count      = ckpt.get("restart_count", 0)
        for k, v in ckpt.get("training_history", {}).items():
            self.training_history[k] = v

        # 清理 elite pool 中的重复条目（保留各公式的最高分版本）
        self._elite_pool = self._dedup_elite_pool(self._elite_pool)

        completed = ckpt.get("step", 0)
        tqdm.write(f"[检查点] 已从 {path} 恢复。"
                   f" 当前步={completed}  最优={self.best_score:.4f}"
                   f"  精英池={len(self._elite_pool)}（去重后）")
        return completed

    # ── Decode formula tokens to readable string ──────────────────────────────

    def _decode_formula(self, tokens: list[int] | None) -> str:
        if tokens is None:
            return "无"
        from .vocab import FORMULA_VOCAB
        names = FORMULA_VOCAB.token_names
        return " -> ".join(names[t] if 0 <= t < len(names) else f"?{t}"
                           for t in tokens)
