"""Atomic local persistence and append-only journal for Stage-5 execution."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

from trading_core import TradingExecutionStateV1


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRADING_STATE_DIR = PROJECT_ROOT / "trading_state"
TRADING_STATE_PATH = TRADING_STATE_DIR / "state_v1.json"
TRADING_JOURNAL_PATH = TRADING_STATE_DIR / "decision_receipts_v1.jsonl"


class TradingStateStore:
    """Serialize state atomically while retaining an append-only evidence trail."""

    def __init__(
        self,
        *,
        state_path: Path = TRADING_STATE_PATH,
        journal_path: Path = TRADING_JOURNAL_PATH,
    ) -> None:
        self.state_path = Path(state_path)
        self.journal_path = Path(journal_path)
        self._lock = threading.RLock()

    def load(self) -> TradingExecutionStateV1:
        with self._lock:
            if not self.state_path.exists():
                return TradingExecutionStateV1()
            try:
                payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"交易执行状态不可读取：{exc}") from exc
            if type(payload) is not dict:
                raise RuntimeError("交易执行状态根节点必须是对象")
            try:
                return TradingExecutionStateV1.from_payload(payload)
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(f"交易执行状态无效：{exc}") from exc

    def save(self, state: TradingExecutionStateV1) -> None:
        encoded = json.dumps(
            state.to_payload(),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        ) + "\n"
        with self._lock:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.state_path.with_name(f".{self.state_path.name}.tmp")
            try:
                with temporary.open("w", encoding="utf-8", newline="\n") as handle:
                    handle.write(encoded)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, self.state_path)
            finally:
                if temporary.exists():
                    temporary.unlink()

    def append_event(self, event: str, payload: dict[str, Any]) -> None:
        if not event:
            raise ValueError("event must not be empty")
        row = {
            "event": event,
            "recorded_at": time.time(),
            "payload": payload,
        }
        encoded = json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ) + "\n"
        with self._lock:
            self.journal_path.parent.mkdir(parents=True, exist_ok=True)
            with self.journal_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())


trading_state_store = TradingStateStore()
