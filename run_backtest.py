"""Strict V2 command-line replay and out-of-sample backtest entrypoint."""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import msvcrt
import os
from pathlib import Path
import re
import stat
import struct
from typing import BinaryIO, Sequence
from uuid import uuid4

import torch

from backtest_viz import BacktestEngine
from data_pipeline.parquet_manager import ParquetDataManager
from model_core.artifacts import (
    BacktestMode,
    StrategyArtifact,
    validate_backtest_dataset,
)
from model_core.execution import classify_position_events
from model_core.semantics import (
    REPORT_SCHEMA_VERSION,
    STRATEGY_SCHEMA_VERSION,
    ArtifactCompatibilityError,
)


DEFAULT_COMMISSION_PCT = 0.02
DEFAULT_SLIPPAGE_PCT = 0.01
DEFAULT_OUTPUT_DIR = "backtest_output"
SIGNATURE_READ_CHUNK_SIZE = 1024 * 1024
MODE_LABELS = {
    BacktestMode.IN_SAMPLE_REPLAY: "样本内复盘",
    BacktestMode.OUT_OF_SAMPLE_BACKTEST: "独立样本外回测",
}


def _nonnegative_float(raw: str) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a finite non-negative number") from exc
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError("must be a finite non-negative number")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run an explicit V2 in-sample replay or independent OOS backtest."
    )
    parser.add_argument("--strategy-file", required=True)
    parser.add_argument("--data-file", required=True)
    parser.add_argument(
        "--mode",
        required=True,
        choices=[mode.value for mode in BacktestMode],
    )
    parser.add_argument("--commission", type=_nonnegative_float, default=DEFAULT_COMMISSION_PCT)
    parser.add_argument("--slippage", type=_nonnegative_float, default=DEFAULT_SLIPPAGE_PCT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser


def load_strategy(path: str | Path) -> StrategyArtifact:
    """Load only the exact current StrategyArtifact dictionary schema."""
    strategy_path = Path(path)
    if not strategy_path.is_file():
        raise FileNotFoundError(f"strategy file does not exist: {strategy_path}")
    try:
        payload = json.loads(strategy_path.read_text(encoding="utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactCompatibilityError(f"invalid strategy JSON: {strategy_path}") from exc
    actual_schema = payload.get("schema_version") if type(payload) is dict else None
    if actual_schema != STRATEGY_SCHEMA_VERSION:
        raise ArtifactCompatibilityError(
            "pre-core-fix/incompatible strategy schema: "
            f"expected={STRATEGY_SCHEMA_VERSION!r} actual={actual_schema!r}; "
            "use the read-only legacy audit tool or retrain from scratch"
        )
    return StrategyArtifact.from_dict(payload)


def _cycle_safe_secondary(
    primary: BaseException,
    candidate: BaseException,
) -> BaseException | None:
    """Return a safe explicit cause without invoking exception overrides."""
    if candidate is primary:
        return None

    candidate_cause = BaseException.__getattribute__(candidate, "__cause__")
    candidate_context = BaseException.__getattribute__(candidate, "__context__")
    if candidate_cause is None and candidate_context is primary:
        BaseException.__setattr__(candidate, "__context__", None)

    visiting: set[int] = set()
    complete: set[int] = set()
    stack: list[tuple[BaseException, bool]] = [(candidate, False)]
    examined = 0
    while stack:
        node, expanded = stack.pop()
        if node is primary:
            return None
        identity = id(node)
        if expanded:
            visiting.discard(identity)
            complete.add(identity)
            continue
        if identity in complete:
            continue
        if identity in visiting:
            return None
        examined += 1
        if examined > 256:
            return None
        visiting.add(identity)
        stack.append((node, True))
        cause = BaseException.__getattribute__(node, "__cause__")
        context = BaseException.__getattribute__(node, "__context__")
        if context is not None:
            stack.append((context, False))
        if cause is not None:
            stack.append((cause, False))
    return candidate


def _add_fixed_note(error: BaseException, note: str) -> None:
    """Attach a bounded note without invoking exception subclass protocols."""
    BaseException.add_note(error, note)


def _move_no_replace(source: Path, destination: Path) -> None:
    """Atomically move on Windows and fail when destination already exists."""
    os.rename(source, destination)


def _move_to_recovery(
    source: Path,
    *,
    recovery_root: Path,
    label: str,
) -> list[BaseException]:
    """Retain a path at a bounded content-addressed recovery location."""
    errors: list[BaseException] = []
    try:
        signature = _file_signature(source)
        recovery = recovery_root / (
            f".{label}.recovery-{signature[3].hex()[:16]}-"
            f"{signature[0]:x}-{signature[1]:x}"
        )
        try:
            _move_no_replace(source, recovery)
        except FileExistsError:
            if _file_signature(recovery) != signature:
                errors.append(FileExistsError("recovery destination is occupied"))
    except BaseException as exc:
        errors.append(exc)
    return errors


def _move_to_identity_recovery(
    source: Path,
    *,
    recovery_root: Path,
    label: str,
    identity: tuple[int, int],
    signature: tuple[int, int, int, bytes, int] | None = None,
) -> list[BaseException]:
    """Retain a staged object without reopening its replaceable pathname."""
    suffix = (
        f"{signature[3].hex()[:16]}-{identity[0]:x}-{identity[1]:x}"
        if signature is not None
        else f"{identity[0]:x}-{identity[1]:x}"
    )
    recovery = recovery_root / f".{label}.recovery-{suffix}"
    try:
        _move_no_replace(source, recovery)
    except BaseException as exc:
        return [exc]
    return []


def _cleanup_owned_temporary(
    temporary: Path,
    owned_identity: tuple[int, int],
    *,
    recovery_root: Path,
    retain_owned: bool,
    owned_label: str = "report-temp",
    foreign_label: str = "foreign-report-temp",
    owned_signature: tuple[int, int, int, bytes, int] | None = None,
) -> list[BaseException]:
    """Clean a reserved stage without reopening it for data or durability."""
    if not temporary.exists():
        return []
    cleanup_path = temporary.with_name(f"{temporary.name}.cleanup")
    errors: list[BaseException] = []
    try:
        _move_no_replace(temporary, cleanup_path)
    except BaseException as exc:
        errors.append(exc)
        cleanup_identity = _identity_if_present(cleanup_path)
        if cleanup_identity is not None:
            errors.extend(
                _move_to_identity_recovery(
                    cleanup_path,
                    recovery_root=recovery_root,
                    label="foreign-report-temp-cleanup",
                    identity=cleanup_identity,
                )
            )
        temporary_identity = _identity_if_present(temporary)
        if temporary_identity is not None:
            if temporary_identity == owned_identity and not retain_owned:
                try:
                    _delete_owned_file(temporary, owned_identity)
                except BaseException as cleanup_error:
                    errors.append(cleanup_error)
            else:
                errors.extend(
                    _move_to_identity_recovery(
                        temporary,
                        recovery_root=recovery_root,
                        label=(
                            owned_label
                            if temporary_identity == owned_identity
                            else foreign_label
                        ),
                        identity=temporary_identity,
                        signature=(
                            owned_signature
                            if temporary_identity == owned_identity
                            else None
                        ),
                    )
                )
        return errors

    moved_identity = _identity_if_present(cleanup_path)
    if moved_identity is None:
        errors.append(FileNotFoundError("staged cleanup object disappeared"))
        return errors
    if moved_identity == owned_identity and retain_owned:
        errors.extend(
            _move_to_identity_recovery(
                cleanup_path,
                recovery_root=recovery_root,
                label=owned_label,
                identity=owned_identity,
                signature=owned_signature,
            )
        )
    elif moved_identity == owned_identity:
        try:
            _delete_owned_file(cleanup_path, owned_identity)
        except BaseException as exc:
            errors.append(exc)
    else:
        errors.extend(
            _move_to_identity_recovery(
                cleanup_path,
                recovery_root=recovery_root,
                label=foreign_label,
                identity=moved_identity,
            )
        )
    return errors


def _owned_stream_signature(
    stream: BinaryIO,
    owned_identity: tuple[int, int],
    *,
    digest: bytes | None = None,
    expected_size: int | None = None,
) -> tuple[int, int, int, bytes, int]:
    """Derive stage content and metadata only through its reserved stream."""
    current = os.fstat(stream.fileno())
    if expected_size is not None and current.st_size != expected_size:
        raise OSError("reserved stream size does not match completed write")
    if digest is None:
        stream.seek(0)
        hasher = hashlib.sha256()
        while chunk := stream.read(1024 * 1024):
            hasher.update(chunk)
        digest = hasher.digest()
    return (
        owned_identity[0],
        owned_identity[1],
        current.st_size,
        digest,
        current.st_mtime_ns,
    )


def _write_all(stream: BinaryIO, payload: bytes | memoryview) -> int:
    """Write a complete payload without reopening the reserved stream path."""
    remaining = memoryview(payload).cast("B")
    total = len(remaining)
    offset = 0
    while offset < total:
        accepted = stream.write(remaining[offset:])
        pending = total - offset
        if (
            isinstance(accepted, bool)
            or not isinstance(accepted, int)
            or accepted <= 0
            or accepted > pending
        ):
            raise OSError("invalid reserved stream write progress")
        offset += accepted
    return total


class _WriteAllStream:
    """Make each renderer write complete on the exact reserved stream."""

    def __init__(self, stream: BinaryIO) -> None:
        self._stream = stream

    def write(self, payload: bytes | memoryview) -> int:
        return _write_all(self._stream, payload)

    def __getattr__(self, name: str) -> object:
        return getattr(self._stream, name)


def _write_report_atomic(
    report: dict[str, object],
    path: Path,
) -> tuple[int, int, int, bytes, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    ownership: list[tuple[int, int]] = []
    owned_identity: tuple[int, int] | None = None
    owned_signature: tuple[int, int, int, bytes, int] | None = None
    write_complete = False
    stage_published = False
    stream: BinaryIO | None = None
    try:
        stream = _reserve_owned_file(temporary, ownership)
        owned_identity = ownership[0]
        payload = (
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        ).encode("utf-8")
        # The stream.write operations stay inside _write_all after reservation.
        _write_all(stream, payload)
        stream.flush()
        os.fsync(stream.fileno())
        owned_signature = _owned_stream_signature(
            stream,
            owned_identity,
            digest=hashlib.sha256(payload).digest(),
            expected_size=len(payload),
        )
        write_complete = True
        _publish_no_replace(temporary, path)
        stage_published = True
        if _directory_identity(path) != owned_identity:
            raise FileExistsError("report stage ownership changed during publication")
        stream_to_close = stream
        stream = None
        stream_to_close.close()
        cleanup_errors = _cleanup_owned_temporary(
            temporary,
            owned_identity,
            recovery_root=(
                path.parent.parent
                if path.parent.name.startswith(".alphamaster-backtest-")
                else path.parent
            ),
            retain_owned=False,
            owned_signature=owned_signature,
        )
        if cleanup_errors:
            raise cleanup_errors[0]
        return owned_signature
    except BaseException as primary_error:
        cleanup_errors: list[BaseException] = []
        if stream is not None:
            try:
                stream.close()
            except BaseException as exc:
                cleanup_errors.append(exc)
        if owned_identity is None and ownership:
            owned_identity = ownership[0]
        if owned_identity is not None:
            if stage_published:
                try:
                    _delete_owned_file(path, owned_identity)
                except BaseException as exc:
                    cleanup_errors.append(exc)
            cleanup_errors.extend(
                _cleanup_owned_temporary(
                    temporary,
                    owned_identity,
                    recovery_root=(
                        path.parent.parent
                        if path.parent.name.startswith(".alphamaster-backtest-")
                        else path.parent
                    ),
                    retain_owned=write_complete,
                    owned_signature=owned_signature,
                )
            )
        if cleanup_errors:
            _raise_primary_after_cleanup(
                primary_error,
                cleanup_errors,
                "secondary owned temporary cleanup failure",
            )
        raise


def _write_equity_chart(result, destination: object, mode_label: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    available = {font.name for font in font_manager.fontManager.ttflist}
    for candidate in ("Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC"):
        if candidate in available:
            plt.rcParams["font.sans-serif"] = [candidate, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False

    figure, axis = plt.subplots(figsize=(12, 5), dpi=110)
    axis.plot(result.cum_pnl, linewidth=1.5)
    axis.set_title(f"{mode_label} · {result.symbol} · V2 shared execution")
    axis.set_xlabel("bar")
    axis.set_ylabel("cumulative net log return")
    axis.grid(alpha=0.25)
    figure.savefig(destination, bbox_inches="tight", format="png")
    plt.close(figure)


def _write_owned_chart(
    result: object,
    path: Path,
    mode_label: str,
) -> tuple[int, int, int, bytes, int]:
    """Reserve chart identity before invoking the fallible renderer."""
    ownership: list[tuple[int, int]] = []
    owned_identity: tuple[int, int] | None = None
    owned_signature: tuple[int, int, int, bytes, int] | None = None
    stream: BinaryIO | None = None
    try:
        stream = _reserve_owned_file(path, ownership)
        owned_identity = ownership[0]
        _write_equity_chart(result, _WriteAllStream(stream), mode_label)
        stream.flush()
        os.fsync(stream.fileno())
        owned_signature = _owned_stream_signature(stream, owned_identity)
        if _directory_identity(path) != owned_identity:
            raise FileExistsError("chart stage ownership changed during rendering")
        stream_to_close = stream
        stream = None
        stream_to_close.close()
        return owned_signature
    except BaseException as primary_error:
        close_errors: list[BaseException] = []
        if stream is not None:
            try:
                stream.close()
            except BaseException as exc:
                close_errors.append(exc)
        if owned_identity is None and ownership:
            owned_identity = ownership[0]
        cleanup_errors = (
            _cleanup_owned_temporary(
                path,
                owned_identity,
                recovery_root=path.parent.parent,
                retain_owned=False,
                owned_label="chart-stage",
                foreign_label="foreign-chart-stage",
                owned_signature=owned_signature,
            )
            if owned_identity is not None
            else []
        )
        cleanup_errors = close_errors + cleanup_errors
        _raise_primary_after_cleanup(
            primary_error,
            cleanup_errors,
            "secondary chart stage cleanup failure",
        )
        raise


def _publish_no_replace(staged_path: Path, final_path: Path) -> None:
    """Atomically publish a complete staged file only when final is absent."""
    os.link(staged_path, final_path)


def _restore_no_replace(backup_path: Path, final_path: Path) -> None:
    """Atomically restore an owned backup without replacing a competitor."""
    os.link(backup_path, final_path)


def _identity_from_stat(current: os.stat_result) -> tuple[int, int]:
    """Normalize path snapshots to the native Windows handle identity shape."""
    if os.name == "nt":
        return current.st_dev & 0xFFFFFFFF, current.st_ino
    return current.st_dev, current.st_ino


def _directory_identity(path: Path) -> tuple[int, int]:
    """Identify one directory entry without following a replacement link."""
    return _identity_from_stat(path.lstat())


def _identity_if_present(path: Path) -> tuple[int, int] | None:
    try:
        return _directory_identity(path)
    except FileNotFoundError:
        return None


def _ordinary_final_identity_if_present(path: Path) -> tuple[int, int] | None:
    """Identify an occupied final without following a reparse entry."""
    try:
        current = path.lstat()
    except FileNotFoundError:
        return None
    reparse_attribute = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    if (
        not stat.S_ISREG(current.st_mode)
        or bool(getattr(current, "st_file_attributes", 0) & reparse_attribute)
        or bool(getattr(current, "st_reparse_tag", 0))
    ):
        raise OSError("output target is not an ordinary regular file")
    return _identity_from_stat(current)


def _open_owned_read_stream(
    path: Path,
    expected_identity: tuple[int, int],
) -> BinaryIO:
    """Open the exact ordinary Windows file while denying replacement."""
    if os.name != "nt":
        raise OSError("owned signature reads require Windows handle semantics")
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateFileW.argtypes = (
        ctypes.c_wchar_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
    )
    kernel32.CreateFileW.restype = ctypes.c_void_p
    handle = kernel32.CreateFileW(
        str(path),
        0x80000000,
        0x00000001,
        None,
        3,
        0x00000080 | 0x00200000,
        None,
    )
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        current_identity = _windows_handle_identity(int(handle))
        if current_identity != expected_identity:
            raise FileExistsError("signature path no longer names the approved file")
        descriptor = msvcrt.open_osfhandle(int(handle), os.O_RDONLY | os.O_BINARY)
    except BaseException:
        kernel32.CloseHandle(ctypes.c_void_p(handle))
        raise
    return os.fdopen(descriptor, "rb", buffering=0)


def _file_signature(
    path: Path,
    expected_identity: tuple[int, int] | None = None,
    *,
    generated_at_out: list[str] | None = None,
) -> tuple[int, int, int, bytes, int]:
    """Hash one exact ordinary file using bounded locked-handle reads."""
    expected = (
        _ordinary_final_identity_if_present(path)
        if expected_identity is None
        else expected_identity
    )
    if expected is None:
        raise FileNotFoundError(path)
    hasher = hashlib.sha256()
    generated_window = b""
    generated_at: str | None = None
    with _open_owned_read_stream(path, expected) as stream:
        before = os.fstat(stream.fileno())
        if _identity_from_stat(before) != expected:
            raise FileExistsError("signature stream identity changed before read")
        while True:
            chunk = stream.read(SIGNATURE_READ_CHUNK_SIZE)
            if not chunk:
                break
            hasher.update(chunk)
            if generated_at_out is not None and generated_at is None:
                generated_window += chunk
                match = re.search(
                    rb'"generated_at"\s*:\s*"([^"\\]{1,256})"',
                    generated_window,
                )
                if match is not None:
                    generated_at = match.group(1).decode("ascii")
                else:
                    generated_window = generated_window[-1024:]
            if len(chunk) > SIGNATURE_READ_CHUNK_SIZE:
                raise OSError("signature reader exceeded bounded chunk size")
        after = os.fstat(stream.fileno())
    before_metadata = (
        _identity_from_stat(before),
        before.st_size,
        before.st_mtime_ns,
    )
    after_metadata = (
        _identity_from_stat(after),
        after.st_size,
        after.st_mtime_ns,
    )
    if before_metadata != after_metadata or after_metadata[0] != expected:
        raise OSError("output changed while its identity was captured")
    if generated_at_out is not None and generated_at is not None:
        generated_at_out.append(generated_at)
    return expected[0], expected[1], after.st_size, hasher.digest(), after.st_mtime_ns


def _file_metadata_matches(
    path: Path,
    signature: tuple[int, int, int, bytes, int],
) -> bool:
    """Revalidate an already-hashed file without rescanning its content."""
    expected = signature[:2]
    try:
        if _ordinary_final_identity_if_present(path) != expected:
            return False
        with _open_owned_read_stream(path, expected) as stream:
            current = os.fstat(stream.fileno())
    except (OSError, ValueError):
        return False
    return (
        _identity_from_stat(current) == expected
        and current.st_size == signature[2]
        and current.st_mtime_ns == signature[4]
    )


def _windows_handle_identity(handle: int) -> tuple[int, int]:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    information = (ctypes.c_ubyte * 52)()
    if not kernel32.GetFileInformationByHandle(
        ctypes.c_void_p(handle), ctypes.byref(information)
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    values = struct.unpack("<13I", bytes(information))
    return values[7], (values[11] << 32) | values[12]


def _reserve_owned_file(
    path: Path,
    ownership: list[tuple[int, int]],
) -> BinaryIO:
    """Create and return the exact locked file stream with identity recorded."""
    if os.name == "nt":
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateFileW.restype = ctypes.c_void_p
        handle = kernel32.CreateFileW(
            str(path),
            0x80000000 | 0x40000000,
            0x00000001 | 0x00000002,
            None,
            1,
            0x00000080,
            None,
        )
        if handle == ctypes.c_void_p(-1).value:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            owned_identity = _windows_handle_identity(int(handle))
            ownership.append(owned_identity)
            descriptor = msvcrt.open_osfhandle(
                int(handle), os.O_RDWR | os.O_BINARY
            )
        except BaseException:
            kernel32.CloseHandle(ctypes.c_void_p(handle))
            raise
        return os.fdopen(descriptor, "w+b", buffering=0)

    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
    try:
        current = os.fstat(descriptor)
        ownership.append((current.st_dev, current.st_ino))
        return os.fdopen(descriptor, "w+b", buffering=0)
    except BaseException:
        os.close(descriptor)
        raise


def _reserve_owned_directory(
    path: Path,
    ownership: list[tuple[int, int]],
) -> None:
    """Atomically create a directory and capture identity from its locked handle."""
    if os.name == "nt":
        class UnicodeString(ctypes.Structure):
            _fields_ = [
                ("Length", ctypes.c_ushort),
                ("MaximumLength", ctypes.c_ushort),
                ("Buffer", ctypes.c_wchar_p),
            ]

        class ObjectAttributes(ctypes.Structure):
            _fields_ = [
                ("Length", ctypes.c_ulong),
                ("RootDirectory", ctypes.c_void_p),
                ("ObjectName", ctypes.POINTER(UnicodeString)),
                ("Attributes", ctypes.c_ulong),
                ("SecurityDescriptor", ctypes.c_void_p),
                ("SecurityQualityOfService", ctypes.c_void_p),
            ]

        class IoStatusBlock(ctypes.Structure):
            _fields_ = [("Status", ctypes.c_void_p), ("Information", ctypes.c_void_p)]

        native_path = "\\??\\" + str(path.resolve())
        buffer = ctypes.create_unicode_buffer(native_path)
        name = UnicodeString(
            len(native_path) * 2,
            (len(native_path) + 1) * 2,
            ctypes.cast(buffer, ctypes.c_wchar_p),
        )
        attributes = ObjectAttributes(
            ctypes.sizeof(ObjectAttributes),
            None,
            ctypes.pointer(name),
            0x00000040,
            None,
            None,
        )
        status_block = IoStatusBlock()
        handle = ctypes.c_void_p()
        ntdll = ctypes.WinDLL("ntdll")
        status = ntdll.NtCreateFile(
            ctypes.byref(handle),
            0x00010000 | 0x00000080 | 0x00100000,
            ctypes.byref(attributes),
            ctypes.byref(status_block),
            None,
            0x00000080,
            0x00000001 | 0x00000002,
            2,
            0x00000001 | 0x00000020,
            None,
            0,
        )
        if status != 0:
            unsigned_status = ctypes.c_ulong(status).value
            if unsigned_status == 0xC0000035:
                raise FileExistsError("owned directory candidate already exists")
            raise OSError(f"NtCreateFile directory status 0x{unsigned_status:08x}")
        try:
            native_identity = _windows_handle_identity(int(handle.value))
            ownership.append(native_identity)
        finally:
            ctypes.WinDLL("kernel32").CloseHandle(handle)
        return

    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CREAT | os.O_EXCL)
    try:
        current = os.fstat(descriptor)
        ownership.append((current.st_dev, current.st_ino))
    finally:
        os.close(descriptor)


def _open_owned_delete_handle(
    path: Path,
    owned_identity: tuple[int, int],
    *,
    directory: bool,
) -> int:
    """Open and lock the exact Windows object against rename/delete."""
    if os.name != "nt":
        raise OSError("owned cleanup requires Windows handle semantics")
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateFileW.argtypes = (
        ctypes.c_wchar_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
    )
    kernel32.CreateFileW.restype = ctypes.c_void_p
    flags = 0x00200000 | (0x02000000 if directory else 0)
    handle = kernel32.CreateFileW(
        str(path),
        0x00010000 | 0x00000080,
        0x00000001 | 0x00000002,
        None,
        3,
        flags,
        None,
    )
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        current_identity = _windows_handle_identity(int(handle))
    except BaseException:
        kernel32.CloseHandle(ctypes.c_void_p(handle))
        raise
    if current_identity != owned_identity:
        kernel32.CloseHandle(ctypes.c_void_p(handle))
        raise FileExistsError("cleanup path no longer names the owned object")
    return int(handle)


def _dispose_owned_handle(handle: int) -> None:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    disposition = ctypes.c_ubyte(1)
    if not kernel32.SetFileInformationByHandle(
        ctypes.c_void_p(handle),
        4,
        ctypes.byref(disposition),
        ctypes.sizeof(disposition),
    ):
        raise ctypes.WinError(ctypes.get_last_error())


def _close_windows_handle(handle: int) -> None:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    if not kernel32.CloseHandle(ctypes.c_void_p(handle)):
        raise ctypes.WinError(ctypes.get_last_error())


def _delete_owned_file(path: Path, owned_identity: tuple[int, int]) -> None:
    handle = _open_owned_delete_handle(path, owned_identity, directory=False)
    try:
        _dispose_owned_handle(handle)
    except BaseException:
        _close_windows_handle(handle)
        raise
    _close_windows_handle(handle)


def _delete_owned_transaction_tree(
    root: Path,
    owned_identity: tuple[int, int],
    owned_children: dict[str, tuple[int, int]],
) -> None:
    """Delete only manifest-owned children and their locked owning directory."""
    handle = _open_owned_delete_handle(root, owned_identity, directory=True)
    try:
        entries = list(os.scandir(root))
        for entry in entries:
            entry_path = Path(entry.path)
            expected_identity = owned_children.get(entry.name)
            if expected_identity is None:
                continue
            try:
                entry_stat = entry_path.lstat()
                if stat.S_ISDIR(entry_stat.st_mode) or stat.S_ISLNK(
                    entry_stat.st_mode
                ):
                    continue
                _delete_owned_file(entry_path, expected_identity)
            except BaseException:
                continue
        if list(os.scandir(root)):
            raise OSError("transaction cleanup retained unowned or changed children")
        _dispose_owned_handle(handle)
    except BaseException:
        _close_windows_handle(handle)
        raise
    _close_windows_handle(handle)


def _cleanup_owned_empty_directory(
    root: Path,
    owned_identity: tuple[int, int],
) -> list[BaseException]:
    """Remove an empty directory only while its exact object is locked."""
    try:
        handle = _open_owned_delete_handle(root, owned_identity, directory=True)
    except BaseException as exc:
        return [exc]
    try:
        if list(os.scandir(root)):
            raise OSError("owned output directory is no longer empty")
        _dispose_owned_handle(handle)
    except BaseException as exc:
        try:
            _close_windows_handle(handle)
        except BaseException:
            pass
        return [exc]
    try:
        _close_windows_handle(handle)
    except BaseException as exc:
        return [exc]
    return []


def _raise_primary_after_cleanup(
    primary_error: BaseException,
    cleanup_errors: list[BaseException],
    note: str,
) -> None:
    """Attach bounded cleanup evidence; the active handler must bare re-raise."""
    if cleanup_errors:
        _add_fixed_note(primary_error, note)
        if isinstance(primary_error, Exception):
            safe_cause = _cycle_safe_secondary(primary_error, cleanup_errors[0])
            BaseException.__setattr__(primary_error, "__cause__", safe_cause)
            BaseException.__setattr__(primary_error, "__suppress_context__", True)


def _allocate_owned_directory(
    parent: Path,
    prefix: str,
) -> tuple[Path, tuple[int, int]]:
    """Boundedly allocate a directory whose identity survives callback failure."""
    path = parent / f"{prefix}{uuid4().hex}"
    ownership: list[tuple[int, int]] = []
    try:
        _reserve_owned_directory(path, ownership)
    except BaseException as primary_error:
        cleanup_errors = (
            _cleanup_owned_empty_directory(path, ownership[0])
            if ownership
            else []
        )
        _raise_primary_after_cleanup(
            primary_error,
            cleanup_errors,
            "secondary owned directory allocation cleanup failure",
        )
        raise
    return path, ownership[0]


def _acquire_output_root(output_root: Path) -> tuple[int, int] | None:
    """Publish an exact-owned directory without replacing an existing root."""
    output_root.parent.mkdir(parents=True, exist_ok=True)
    candidate, owned_identity = _allocate_owned_directory(
        output_root.parent,
        ".alphamaster-output-",
    )
    try:
        _move_no_replace(candidate, output_root)
    except FileExistsError:
        cleanup_errors = _cleanup_owned_empty_directory(candidate, owned_identity)
        if cleanup_errors:
            raise cleanup_errors[0]
        try:
            root_stat = output_root.lstat()
        except OSError as exc:
            raise OSError(
                "pre-existing output root is not an ordinary directory"
            ) from exc
        reparse_attribute = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        if (
            not stat.S_ISDIR(root_stat.st_mode)
            or bool(getattr(root_stat, "st_file_attributes", 0) & reparse_attribute)
            or bool(getattr(root_stat, "st_reparse_tag", 0))
        ):
            raise OSError("pre-existing output root is not an ordinary directory")
        return None
    except BaseException as primary_error:
        cleanup_errors: list[BaseException] = []
        if _identity_if_present(output_root) == owned_identity:
            cleanup_errors.extend(
                _cleanup_owned_empty_directory(output_root, owned_identity)
            )
        elif _identity_if_present(candidate) == owned_identity:
            cleanup_errors.extend(
                _cleanup_owned_empty_directory(candidate, owned_identity)
            )
        _raise_primary_after_cleanup(
            primary_error,
            cleanup_errors,
            "secondary output directory acquisition cleanup failure",
        )
        raise
    return owned_identity


def _cleanup_owned_transaction(
    transaction_root: Path,
    owned_identity: tuple[int, int],
    owned_children: dict[str, tuple[int, int]] | None = None,
) -> list[BaseException]:
    """Quarantine, verify, then remove only the exact created directory."""
    cleanup_root = transaction_root.with_name(f"{transaction_root.name}.cleanup")
    errors: list[BaseException] = []
    move_error: BaseException | None = None
    try:
        _move_no_replace(transaction_root, cleanup_root)
    except BaseException as exc:
        move_error = exc

    try:
        cleanup_identity = _identity_if_present(cleanup_root)
        source_identity = _identity_if_present(transaction_root)
    except BaseException as exc:
        errors.append(exc)
        if move_error is not None:
            errors.insert(0, move_error)
        return errors

    if cleanup_identity != owned_identity:
        if cleanup_identity is not None:
            try:
                _move_no_replace(cleanup_root, transaction_root)
            except BaseException as exc:
                errors.append(exc)
        if move_error is not None:
            errors.insert(0, move_error)
        else:
            errors.insert(0, OSError("transaction directory ownership changed"))
        return errors

    if source_identity is not None:
        errors.append(FileExistsError("transaction path was replaced during cleanup"))

    try:
        if _directory_identity(cleanup_root) != owned_identity:
            raise OSError("transaction cleanup identity changed")
        _delete_owned_transaction_tree(
            cleanup_root,
            owned_identity,
            dict(owned_children or {}),
        )
    except BaseException as exc:
        errors.append(exc)
    if move_error is not None:
        errors.insert(0, move_error)
    return errors


def _publish_output_set(
    *,
    report: dict[str, object],
    result: object,
    output_dir: str | Path,
    stem: str,
    mode_label: str,
) -> Path:
    """Publish a report/chart pair without mutating the tree on failure."""
    output_root = Path(output_dir)
    output_root_identity = _acquire_output_root(output_root)
    report_path = output_root / f"{stem}.json"
    chart_path = output_root / f"{stem}.png"
    finals = (report_path, chart_path)
    initial_final_identities = {
        path: _ordinary_final_identity_if_present(path) for path in finals
    }

    try:
        transaction_root, transaction_identity = _allocate_owned_directory(
            output_root.parent,
            ".alphamaster-backtest-",
        )
    except BaseException as primary_error:
        cleanup_errors = (
            _cleanup_owned_empty_directory(output_root, output_root_identity)
            if output_root_identity is not None
            else []
        )
        _raise_primary_after_cleanup(
            primary_error,
            cleanup_errors,
            "secondary output directory cleanup failure",
        )
        raise
    staged_report = transaction_root / "report.stage"
    staged_chart = transaction_root / "chart.stage"
    publication_identities: dict[Path, tuple[int, int]] = {}
    validated_final_signatures: (
        dict[Path, tuple[int, int, int, bytes, int]] | None
    ) = None
    existing_final_signatures: (
        dict[Path, tuple[int, int, int, bytes, int]] | None
    ) = None
    rollback_root: Path | None = None
    rollback_identity: tuple[int, int] | None = None
    transaction_children: dict[str, tuple[int, int]] = {}
    rollback_children: dict[str, tuple[int, int]] = {}

    def ensure_rollback_root() -> Path:
        nonlocal rollback_root, rollback_identity
        if rollback_root is None:
            rollback_root, rollback_identity = _allocate_owned_directory(
                output_root.parent,
                ".alphamaster-rollback-",
            )
        return rollback_root

    def preserve_foreign(moved_path: Path, final_path: Path) -> list[BaseException]:
        errors: list[BaseException] = []
        try:
            signature = _file_signature(moved_path)
            recovery_path = output_root.parent / (
                f".{final_path.name}.recovery-"
                f"{signature[3].hex()[:16]}-{signature[0]:x}-{signature[1]:x}"
            )
            try:
                _restore_no_replace(moved_path, recovery_path)
            except FileExistsError:
                if _file_signature(recovery_path) != signature:
                    errors.append(
                        FileExistsError("foreign recovery path is occupied")
                    )
            try:
                _restore_no_replace(moved_path, final_path)
            except FileExistsError:
                pass
        except BaseException as exc:
            errors.append(exc)
        return errors

    def restore_output_tree() -> list[BaseException]:
        errors: list[BaseException] = []
        for final_path in finals:
            owned_final = False
            if final_path in publication_identities:
                try:
                    final_stat = final_path.stat()
                    owned_final = _identity_from_stat(
                        final_stat
                    ) == publication_identities[final_path]
                except BaseException as exc:
                    errors.append(exc)
            if owned_final:
                try:
                    quarantine = ensure_rollback_root() / (
                        f"{final_path.suffix[1:]}.rollback.quarantine"
                    )
                except BaseException as exc:
                    errors.append(exc)
                    try:
                        _delete_owned_file(
                            final_path,
                            publication_identities[final_path],
                        )
                    except BaseException as delete_error:
                        errors.append(delete_error)
                    continue
                move_error: BaseException | None = None
                try:
                    _move_no_replace(final_path, quarantine)
                except BaseException as exc:
                    move_error = exc
                    errors.append(exc)
                try:
                    quarantine_stat = quarantine.stat()
                    moved_identity = _identity_from_stat(quarantine_stat)
                    if moved_identity != publication_identities[final_path]:
                        errors.extend(preserve_foreign(quarantine, final_path))
                    else:
                        rollback_children[quarantine.name] = moved_identity
                except BaseException as exc:
                    if move_error is None:
                        errors.append(exc)
        if output_root_identity is not None:
            errors.extend(
                _cleanup_owned_empty_directory(output_root, output_root_identity)
            )
        return errors

    def cleanup_transaction() -> BaseException | None:
        errors = _cleanup_owned_transaction(
            transaction_root,
            transaction_identity,
            transaction_children,
        )
        if rollback_root is not None and rollback_identity is not None:
            errors.extend(
                _cleanup_owned_transaction(
                    rollback_root,
                    rollback_identity,
                    rollback_children,
                )
            )
        return errors[0] if errors else None

    try:
        chart_signature = _write_owned_chart(result, staged_chart, mode_label)
        transaction_children[staged_chart.name] = chart_signature[:2]
        report_to_write = report
        reusable_identities = {
            path: _ordinary_final_identity_if_present(path) for path in finals
        }
        if reusable_identities == initial_final_identities and all(
            identity is not None for identity in reusable_identities.values()
        ):
            generated_at_values: list[str] = []
            existing_final_signatures = {
                report_path: _file_signature(
                    report_path,
                    reusable_identities[report_path],
                    generated_at_out=generated_at_values,
                ),
                chart_path: _file_signature(
                    chart_path,
                    reusable_identities[chart_path],
                ),
            }
            if not all(
                _file_metadata_matches(path, existing_final_signatures[path])
                for path in finals
            ):
                raise FileExistsError("occupied output pair changed during validation")
            if generated_at_values:
                existing_generated_at = generated_at_values[0]
                try:
                    if existing_generated_at.endswith("Z"):
                        datetime.fromisoformat(
                            existing_generated_at.replace("Z", "+00:00")
                        )
                        report_to_write = dict(report)
                        report_to_write["generated_at"] = existing_generated_at
                except ValueError:
                    pass
        report_signature = _write_report_atomic(report_to_write, staged_report)
        transaction_children[staged_report.name] = report_signature[:2]
        staged_identities = {
            report_path: report_signature[:2],
            chart_path: chart_signature[:2],
        }
        staged_signatures = {
            report_path: report_signature,
            chart_path: chart_signature,
        }
        occupied = {
            path: _ordinary_final_identity_if_present(path) for path in finals
        }
        if any(identity is not None for identity in occupied.values()):
            if not all(identity is not None for identity in occupied.values()):
                raise FileExistsError("output pair is incomplete or occupied")
            if existing_final_signatures is not None:
                if any(
                    occupied[path] != existing_final_signatures[path][:2]
                    for path in finals
                ):
                    raise FileExistsError("occupied output pair changed during validation")
                final_signatures = existing_final_signatures
            else:
                final_signatures = {
                    path: _file_signature(path, occupied[path]) for path in finals
                }
            if any(
                (
                    final_signatures[path][2],
                    final_signatures[path][3],
                )
                != (
                    staged_signatures[path][2],
                    staged_signatures[path][3],
                )
                for path in finals
            ):
                raise FileExistsError("occupied output pair differs from candidate")
            if not all(
                _file_metadata_matches(path, final_signatures[path]) for path in finals
            ):
                raise FileExistsError("occupied output pair changed during validation")
            validated_final_signatures = final_signatures
        else:
            _publish_no_replace(staged_report, report_path)
            publication_identities[report_path] = staged_identities[report_path]
            _publish_no_replace(staged_chart, chart_path)
            publication_identities[chart_path] = staged_identities[chart_path]
            validated_final_signatures = staged_signatures
    except BaseException as primary_error:
        restore_errors = restore_output_tree()
        cleanup_error = cleanup_transaction()
        secondary_errors = restore_errors + (
            [cleanup_error] if cleanup_error is not None else []
        )
        _raise_primary_after_cleanup(
            primary_error,
            secondary_errors,
            "secondary rollback/cleanup failure occurred",
        )
        raise

    cleanup_error = cleanup_transaction()
    if cleanup_error is not None:
        raise cleanup_error
    if validated_final_signatures is not None and not all(
        _file_metadata_matches(path, validated_final_signatures[path])
        for path in finals
    ):
        try:
            raise FileExistsError(
                "new output pair changed after transaction cleanup"
                if publication_identities
                else (
                    "occupied output pair changed before idempotent return "
                    "after transaction cleanup"
                )
            )
        except BaseException as validation_error:
            restore_errors = restore_output_tree()
            if rollback_root is not None and rollback_identity is not None:
                restore_errors.extend(
                    _cleanup_owned_transaction(
                        rollback_root,
                        rollback_identity,
                        rollback_children,
                    )
                )
            _raise_primary_after_cleanup(
                validation_error,
                restore_errors,
                "secondary post-cleanup output restoration failure",
            )
            raise
    return report_path


def run_backtest(
    *,
    strategy_file: str | Path,
    data_file: str | Path,
    mode: str | BacktestMode,
    commission: float = DEFAULT_COMMISSION_PCT,
    slippage: float = DEFAULT_SLIPPAGE_PCT,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    if not math.isfinite(commission) or commission < 0:
        raise ValueError("commission must be finite and non-negative")
    if not math.isfinite(slippage) or slippage < 0:
        raise ValueError("slippage must be finite and non-negative")
    selected_mode = mode if isinstance(mode, BacktestMode) else BacktestMode(mode)
    strategy = load_strategy(strategy_file)

    data_path = Path(data_file)
    if not data_path.is_file():
        raise FileNotFoundError(f"data file does not exist: {data_path}")
    manager = ParquetDataManager(data_path)
    manager.load()
    test_identity = manager.data_identities[0]
    validate_backtest_dataset(strategy, test_identity, selected_mode)

    identity = strategy.run_identity.artifact_identity
    min_exposure = float(identity.training_config["neutral_band"])
    cost_rate = (commission + slippage) / 100.0
    engine = BacktestEngine(
        formula=list(strategy.formula_tokens),
        cost_rate=cost_rate,
        min_exposure=min_exposure,
    )
    result = engine.run(manager.raw_dict, manager.feat_tensor, manager.symbols)[0]

    ledger_total = sum(row.net_pnl for row in result.ledger)
    execution_total = float(
        result.execution.net_pnl[result.execution.target_valid]
        .to(dtype=torch.float64)
        .sum()
        .detach()
        .cpu()
    )
    difference = abs(ledger_total - execution_total)
    if not math.isfinite(difference) or difference > 1e-8:
        raise RuntimeError(
            "ledger reconciliation exceeded tolerance: "
            f"difference={difference!r} tolerance=1e-8"
        )

    event_counts = [
        classify_position_events(
            result.execution.position[index, result.execution.target_valid[index]]
        )
        for index in range(result.execution.position.shape[0])
    ]

    generated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    mode_label = MODE_LABELS[selected_mode]
    is_independent = selected_mode is BacktestMode.OUT_OF_SAMPLE_BACKTEST
    registered_final = (
        strategy.final_oos_evidence is not None
        and strategy.final_oos_evidence.dataset == test_identity
    )
    report: dict[str, object] = {
        "report_schema": REPORT_SCHEMA_VERSION,
        "mode": selected_mode.value,
        "mode_label": mode_label,
        "evidence_scope": {
            "classification": (
                "independent_final_oos"
                if registered_final
                else "independent_out_of_sample_evaluation"
                if is_independent
                else "internal_selection_replay"
            ),
            "participated_in_search": not is_independent,
            "registered_final_holdout": registered_final,
            "selection_metrics_are_final_oos": False,
        },
        "symbol": identity.symbol,
        "timeframe": identity.timeframe,
        "strategy_fingerprint": strategy.fingerprint,
        "artifact_fingerprint": identity.fingerprint,
        "versions": {
            "artifact_schema": strategy.schema_version,
            "strategy_schema": strategy.schema_version,
            "core_semantics": identity.core_semantics_version,
            "vocab": identity.vocab_version,
            "label_semantics": identity.label_semantics_version,
            "execution_semantics": identity.execution_semantics_version,
        },
        "training_dataset": identity.training_dataset.to_dict(),
        "test_dataset": test_identity.to_dict(),
        "training_start_time_ns": identity.training_dataset.start_time_ns,
        "training_end_time_ns": identity.training_dataset.end_time_ns,
        "test_start_time_ns": test_identity.start_time_ns,
        "test_end_time_ns": test_identity.end_time_ns,
        "cost": {
            "commission_pct": commission,
            "slippage_pct": slippage,
            "cost_rate": cost_rate,
            "total_cost_rate": cost_rate,
            "unit": "equity_fraction_simple_return",
        },
        "return_accounting": {
            "asset_input": "log_return",
            "asset_conversion": "asset_simple=expm1(asset_log_return)",
            "gross_unit": "equity_fraction_simple_return",
            "cost_unit": "equity_fraction_simple_return",
            "net_fact": "net_log_return=log1p(gross_simple_return-cost)",
            "insolvency_boundary": "gross_simple_return-cost<=-1 is invalid",
            "portfolio_weighting": "equal_weight_by_exit_timestamp",
        },
        "trade_statistics": {
            "turnover_events": sum(value.turnover_events for value in event_counts),
            "entries": sum(value.entries for value in event_counts),
            "exits": sum(value.exits for value in event_counts),
            "reversals": sum(value.reversals for value in event_counts),
            "liquidation_events": sum(
                value.liquidation_events for value in event_counts
            ),
            "display_trades": sum(value.display_trades for value in event_counts),
            "n_trades": sum(value.display_trades for value in event_counts),
            "n_trades_definition": "display_trades=entries+reversals",
            "turnover_definition": "every non-zero absolute position change",
        },
        "min_exposure": min_exposure,
        "formula_tokens": list(strategy.formula_tokens),
        "decoded_formula": strategy.decoded_formula,
        "metrics": asdict(result.metrics),
        "ledger": [asdict(row) for row in result.ledger],
        "ledger_reconciliation": {
            "ledger_net_pnl": ledger_total,
            "execution_net_pnl": execution_total,
            "absolute_difference": difference,
            "tolerance": 1e-8,
            "reconciled": True,
        },
        "generated_at": generated_at,
    }
    stem = (
        f"backtest_v2_{selected_mode.value}_{identity.symbol}_"
        f"{test_identity.data_fingerprint[:12]}_{strategy.fingerprint[:12]}"
    )
    report_path = _publish_output_set(
        report=report,
        result=result,
        output_dir=output_dir,
        stem=stem,
        mode_label=mode_label,
    )
    print(f"{mode_label}: {identity.symbol} {identity.timeframe}")
    print(
        f"net={execution_total:+.8f} sharpe={result.metrics.sharpe:+.4f} "
        f"reconciliation={difference:.3g}"
    )
    print(f"report: {report_path}")
    return report_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run_backtest(
            strategy_file=args.strategy_file,
            data_file=args.data_file,
            mode=args.mode,
            commission=args.commission,
            slippage=args.slippage,
            output_dir=args.output_dir,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
