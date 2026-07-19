"""Identity-preserving V2 training package import/export."""
from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from model_core.artifacts import StrategyArtifact, TrainingRunIdentity
from model_core.config import ModelConfig
from web.progress import (
    CHECKPOINT_DIR,
    PROJECT_ROOT,
    STRATEGIES_DIR,
    _history_body,
    generated_at_utc,
    _validate_history,
    _validate_history_checkpoint_relationship,
    checkpoint_glob,
    invalidate_checkpoint_cache,
)

PACKAGE_SCHEMA = "training-package-v2"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _checkpoint_identity(
    data: bytes,
) -> tuple[TrainingRunIdentity, int, dict[str, Any]]:
    payload = torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)
    if type(payload) is not dict:
        raise ValueError("V2 checkpoint payload must be an object")
    if payload.get("checkpoint_schema_version") != "checkpoint-v2":
        raise ValueError("incompatible V2 checkpoint schema")
    run = TrainingRunIdentity.from_dict(payload["run_identity"])
    step = payload["step"]
    if type(step) is not int:
        raise ValueError("checkpoint step must be an exact built-in integer")
    if step < 0 or step > ModelConfig.TRAIN_STEPS:
        raise ValueError("checkpoint step is outside configured training bounds")
    history = payload["training_history"]
    _validate_history(history, run, require_run_identity=False)
    _validate_history_checkpoint_relationship(history, step)
    rank_monitor = payload["rank_monitor_history"]
    if type(rank_monitor) is not list or rank_monitor != history["stable_rank"]:
        raise ValueError("checkpoint rank_monitor_history does not match training_history stable_rank")
    return run, step, payload


def _strategy_identity(data: bytes) -> tuple[StrategyArtifact, TrainingRunIdentity]:
    artifact = StrategyArtifact.from_dict(json.loads(data.decode("utf-8")))
    return artifact, artifact.run_identity


def _history_identity(data: bytes) -> tuple[TrainingRunIdentity, dict[str, Any]]:
    value = json.loads(data.decode("utf-8"))
    if type(value) is not dict or "run_identity" not in value:
        raise ValueError("V2 history lacks run_identity")
    run = TrainingRunIdentity.from_dict(value["run_identity"])
    _validate_history(value, run, require_run_identity=True)
    return run, value


def _same_run(left: TrainingRunIdentity, right: TrainingRunIdentity) -> None:
    if left != right:
        raise ValueError("mixed V2 training package run/artifact identity")


def build_training_export_zip(symbol: str) -> tuple[bytes, str]:
    ckpts = checkpoint_glob(symbol)
    if not ckpts:
        raise FileNotFoundError(f"未找到 {symbol} 的 V2 checkpoint")
    candidates: list[
        tuple[tuple[int, int, str], Path, bytes, TrainingRunIdentity, dict[str, Any]]
    ] = []
    first_error: Exception | None = None
    for path in ckpts:
        try:
            stat_before = path.stat()
            data = path.read_bytes()
            stat_after = path.stat()
            if (
                stat_before.st_mtime_ns != stat_after.st_mtime_ns
                or stat_before.st_size != stat_after.st_size
            ):
                raise ValueError("V2 export checkpoint changed while being validated")
            candidate_run, candidate_step, payload = _checkpoint_identity(data)
            if candidate_run.artifact_identity.symbol != symbol:
                raise ValueError("V2 export checkpoint symbol mismatch")
            if path.name != candidate_run.checkpoint_filename(candidate_step):
                raise ValueError("V2 export checkpoint filename is not canonical")
            candidates.append(
                (
                    (candidate_step, stat_after.st_mtime_ns, path.name),
                    path,
                    data,
                    candidate_run,
                    payload,
                )
            )
        except Exception as exc:
            if first_error is None:
                first_error = exc
            continue
    if not candidates:
        if first_error is not None:
            raise first_error
        raise ValueError(f"no identity-valid V2 checkpoint for {symbol}")
    _, checkpoint, ckpt_bytes, run, checkpoint_payload = max(
        candidates, key=lambda candidate: candidate[0]
    )
    step = checkpoint_payload["step"]

    strategies: list[tuple[Path, StrategyArtifact, bytes]] = []
    for path in STRATEGIES_DIR.glob("best_v2_*.json"):
        try:
            raw = path.read_bytes()
            artifact, strategy_run = _strategy_identity(raw)
            if strategy_run == run and path.name == run.strategy_filename():
                strategies.append((path, artifact, raw))
        except Exception:
            continue
    if not strategies:
        raise ValueError("V2 export requires strategy from the same run identity")
    strategy_path, artifact, strategy_bytes = max(
        strategies,
        key=lambda row: (generated_at_utc(row[1].generated_at), row[0].name),
    )
    history_path = PROJECT_ROOT / run.history_filename()
    if not history_path.is_file():
        raise ValueError("V2 export requires the canonical same-run history")
    history_bytes = history_path.read_bytes()
    history_run, history_payload = _history_identity(history_bytes)
    _same_run(run, history_run)
    _validate_history_checkpoint_relationship(history_payload, step)
    if _history_body(history_payload) != checkpoint_payload["training_history"]:
        raise ValueError("standalone and checkpoint histories conflict")

    files = {
        f"checkpoints/{checkpoint.name}": ckpt_bytes,
        f"strategies/{strategy_path.name}": strategy_bytes,
        history_path.name: history_bytes,
    }
    manifest = {
        "package_schema": PACKAGE_SCHEMA,
        "run_id": run.run_id,
        "artifact_fingerprint": run.artifact_identity.fingerprint,
        "step": step,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "files": {name: _sha256(data) for name, data in files.items()},
    }
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in files.items():
            zf.writestr(name, data)
        zf.writestr("manifest.json", json.dumps(manifest, sort_keys=True))
    return buf.getvalue(), f"training_v2_{symbol}_run_{run.run_id[:8]}_step_{step}.zip"


@dataclass(frozen=True)
class _PublicationReceipt:
    path: Path
    device: int
    inode: int
    size: int
    modified_ns: int
    digest: str


def _publication_receipt(path: Path, data: bytes) -> _PublicationReceipt:
    stat = path.stat()
    return _PublicationReceipt(
        path=path,
        device=stat.st_dev,
        inode=stat.st_ino,
        size=len(data),
        modified_ns=stat.st_mtime_ns,
        digest=_sha256(data),
    )


def _rollback_publication(receipt: _PublicationReceipt) -> None:
    path = receipt.path
    try:
        before = path.stat()
        if (
            before.st_dev != receipt.device
            or before.st_ino != receipt.inode
            or before.st_size != receipt.size
            or before.st_mtime_ns != receipt.modified_ns
            or _sha256(path.read_bytes()) != receipt.digest
        ):
            return
        after = path.stat()
        if (
            after.st_dev == receipt.device
            and after.st_ino == receipt.inode
            and after.st_size == receipt.size
            and after.st_mtime_ns == receipt.modified_ns
        ):
            _unlink_owned_entry(receipt)
    except FileNotFoundError:
        return


def _unlink_owned_entry(receipt: _PublicationReceipt) -> None:
    """Delete only the exact Windows file object recorded by this attempt."""
    if os.name != "nt":
        raise RuntimeError("race-safe immutable rollback requires Windows handle deletion")

    import ctypes
    import msvcrt
    from ctypes import wintypes

    delete_access = 0x00010000
    generic_read = 0x80000000
    share_all = 0x00000001 | 0x00000002 | 0x00000004
    invalid_handle = ctypes.c_void_p(-1).value
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = (
        wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
        wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE,
    )
    create_file.restype = wintypes.HANDLE
    handle = create_file(
        str(receipt.path),
        delete_access | generic_read,
        share_all,
        None,
        3,
        0x00000080,
        None,
    )
    if handle == invalid_handle:
        error = ctypes.get_last_error()
        if error in (2, 3):
            return
        raise ctypes.WinError(error)

    fd = -1
    try:
        fd = msvcrt.open_osfhandle(handle, os.O_RDONLY)
        stat = os.fstat(fd)
        if (
            stat.st_dev != receipt.device
            or stat.st_ino != receipt.inode
            or stat.st_size != receipt.size
            or stat.st_mtime_ns != receipt.modified_ns
        ):
            return
        os.lseek(fd, 0, os.SEEK_SET)
        digest = hashlib.sha256()
        while True:
            block = os.read(fd, 1024 * 1024)
            if not block:
                break
            digest.update(block)
        if digest.hexdigest() != receipt.digest:
            return

        class _FileDispositionInfo(ctypes.Structure):
            _fields_ = (("DeleteFile", wintypes.BOOL),)

        disposition = _FileDispositionInfo(True)
        set_information = kernel32.SetFileInformationByHandle
        set_information.argtypes = (
            wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD,
        )
        set_information.restype = wintypes.BOOL
        if not set_information(
            wintypes.HANDLE(msvcrt.get_osfhandle(fd)),
            4,
            ctypes.byref(disposition),
            ctypes.sizeof(disposition),
        ):
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        if fd >= 0:
            os.close(fd)
        else:
            kernel32.CloseHandle(handle)


def _publish_immutable(relative: str, data: bytes) -> _PublicationReceipt | None:
    destination = PROJECT_ROOT / Path(relative)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.read_bytes() == data:
            return None
        raise ValueError(f"immutable V2 target already exists with different bytes: {relative}")
    temporary: Path | None = None
    linked = False
    receipt: _PublicationReceipt | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.import-",
            delete=False,
        ) as fp:
            temporary = Path(fp.name)
            fp.write(data)
            fp.flush()
            os.fsync(fp.fileno())
        try:
            os.link(temporary, destination)
            linked = True
        except FileExistsError:
            if destination.read_bytes() == data:
                return None
            raise ValueError(
                f"immutable V2 target already exists with different bytes: {relative}"
            )
        receipt = _publication_receipt(destination, data)
        return receipt
    except Exception:
        if linked and receipt is not None:
            _rollback_publication(receipt)
        raise
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def import_training_package(content: bytes, filename: str, expected_symbol: str | None = None) -> dict[str, Any]:
    if not Path(filename).name.lower().endswith(".zip"):
        raise ValueError("V2 training imports require a verified .zip package")
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        if "manifest.json" not in zf.namelist():
            raise ValueError("V2 package lacks manifest.json")
        manifest = json.loads(zf.read("manifest.json"))
        if manifest.get("package_schema") != PACKAGE_SCHEMA:
            raise ValueError("unsupported V2 training package schema")
        declared = manifest.get("files")
        if not isinstance(declared, dict) or len(declared) != 3:
            raise ValueError("V2 manifest must declare checkpoint, strategy, and history")
        names = zf.namelist()
        if len(names) != len(set(names)) or set(names) != set(declared) | {"manifest.json"}:
            raise ValueError("V2 package contains undeclared or duplicate members")
        extracted = {name: zf.read(name) for name in declared}
    for name, digest in declared.items():
        if _sha256(extracted[name]) != digest:
            raise ValueError(f"V2 package hash mismatch: {name}")
    ckpt_name = next((n for n in extracted if n.startswith("checkpoints/")), None)
    strategy_name = next((n for n in extracted if n.startswith("strategies/")), None)
    history_name = next((n for n in extracted if n.startswith("training_history_v2_")), None)
    if not ckpt_name or not strategy_name or not history_name:
        raise ValueError("V2 package filenames are incomplete or noncanonical")
    run, step, checkpoint_payload = _checkpoint_identity(extracted[ckpt_name])
    artifact, strategy_run = _strategy_identity(extracted[strategy_name])
    history_run, history_payload = _history_identity(extracted[history_name])
    _same_run(run, strategy_run)
    _same_run(run, history_run)
    if manifest.get("run_id") != run.run_id or manifest.get("artifact_fingerprint") != run.artifact_identity.fingerprint:
        raise ValueError("V2 manifest identity mismatch")
    if expected_symbol and run.artifact_identity.symbol != expected_symbol:
        raise ValueError(f"package symbol mismatch: expected={expected_symbol} actual={run.artifact_identity.symbol}")
    manifest_step = manifest.get("step")
    if type(manifest_step) is not int or manifest_step != step:
        raise ValueError("V2 manifest step must exactly match the checkpoint step")
    _validate_history_checkpoint_relationship(history_payload, step)
    if _history_body(history_payload) != checkpoint_payload["training_history"]:
        raise ValueError("standalone and checkpoint histories conflict")

    expected_names = {
        f"checkpoints/{run.checkpoint_filename(step)}",
        f"strategies/{run.strategy_filename()}",
        run.history_filename(),
    }
    if set(extracted) != expected_names:
        raise ValueError("V2 package member filenames are not canonical for their run identity")

    ordered = sorted(extracted)
    for name in ordered:
        destination = PROJECT_ROOT / Path(name)
        if destination.exists() and destination.read_bytes() != extracted[name]:
            raise ValueError(f"immutable V2 target already exists with different bytes: {name}")

    created: list[_PublicationReceipt] = []
    try:
        for name in ordered:
            receipt = _publish_immutable(name, extracted[name])
            if receipt is not None:
                created.append(receipt)
    except Exception:
        for receipt in reversed(created):
            _rollback_publication(receipt)
        raise
    installed = [name.replace("\\", "/") for name in ordered]
    invalidate_checkpoint_cache()
    return {"ok": True, "symbol": run.artifact_identity.symbol, "step": step,
            "run_id": run.run_id, "installed": installed,
            "message": f"已幂等导入 V2 run {run.run_id}（step {step}）"}
