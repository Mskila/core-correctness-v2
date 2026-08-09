"""Background manager for the Stage-6 research-only trading backtest."""

from __future__ import annotations

import threading
from typing import Any, Protocol

from trading_core import TradingBacktestRunner, TradingConfigV1


class BacktestRunner(Protocol):
    def run(
        self,
        config: TradingConfigV1,
        *,
        modes: tuple[str, ...],
        stop_event: threading.Event,
        progress: Any,
    ) -> dict[str, Any] | None: ...


class TradingBacktestManager:
    """Own one cooperative background research job, independent of live execution."""

    def __init__(self, runner: BacktestRunner | None = None) -> None:
        self.runner = runner or TradingBacktestRunner()
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._state = "idle"
        self._mode: str | None = None
        self._current_m15: int | None = None
        self._completed = 0
        self._total = 0
        self._error = ""
        self._report: dict[str, Any] | None = None

    def start(self, config: TradingConfigV1) -> dict[str, Any]:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("trading backtest job is already active")
            self._stop_event.clear()
            self._state = "running"
            self._mode = None
            self._current_m15 = None
            self._completed = 0
            self._total = 0
            self._error = ""
            self._report = None
            self._thread = threading.Thread(
                target=self._run,
                args=(config,),
                daemon=True,
                name="trading-backtest",
            )
            self._thread.start()
        return self.status()

    def _progress(
        self, mode: str, completed: int, total: int, current_m15: int | None
    ) -> None:
        with self._lock:
            self._mode = mode
            self._completed = completed
            self._total = total
            self._current_m15 = current_m15

    def _run(self, config: TradingConfigV1) -> None:
        try:
            report = self.runner.run(
                config,
                modes=("rules", "rules_codex"),
                stop_event=self._stop_event,
                progress=self._progress,
            )
            with self._lock:
                if self._stop_event.is_set() or report is None:
                    self._state = "stopped"
                else:
                    self._report = report
                    self._state = "completed"
        except Exception as exc:  # noqa: BLE001 - surfaced in status, no live fallback
            with self._lock:
                self._error = str(exc)
                self._state = "failed"

    def stop(self) -> dict[str, Any]:
        with self._lock:
            self._stop_event.set()
            if self._thread is not None and self._thread.is_alive():
                self._state = "stopping"
        return self.status()

    def status(self) -> dict[str, Any]:
        with self._lock:
            active = bool(self._thread is not None and self._thread.is_alive())
            return {
                "active": active,
                "state": self._state,
                "current_mode": self._mode,
                "current_m15": self._current_m15,
                "completed": self._completed,
                "total": self._total,
                "progress": self._completed / self._total if self._total else 0.0,
                "error": self._error,
                "report_path": None if self._report is None else self._report.get("report_path"),
            }

    def report(self) -> dict[str, Any]:
        with self._lock:
            if self._report is None:
                raise RuntimeError("no completed trading backtest report")
            return dict(self._report)


trading_backtest_manager = TradingBacktestManager()
