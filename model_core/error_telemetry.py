"""Small, dependency-free structured error telemetry for core services."""
from __future__ import annotations

import threading
from collections import Counter, deque
from typing import Any


def _bounded_message(value: object, limit: int) -> str:
    try:
        message = str(value)
    except BaseException:
        message = f"<{type(value).__name__}: message unavailable>"
    return message[:limit]


class ErrorTelemetry:
    """Count named failures and retain only a bounded diagnostic tail."""

    def __init__(self, *, message_limit: int = 20, message_chars: int = 240) -> None:
        if type(message_limit) is not int or message_limit < 0:
            raise ValueError("message_limit must be a non-negative built-in int")
        if type(message_chars) is not int or message_chars < 1:
            raise ValueError("message_chars must be a positive built-in int")
        self._counts: Counter[str] = Counter()
        self._messages: deque[dict[str, str]] = deque(maxlen=message_limit)
        self._message_chars = message_chars
        self._lock = threading.Lock()

    def record(self, category: str, error: object) -> None:
        if type(category) is not str or not category.strip():
            raise ValueError("error category must be a non-empty built-in string")
        row = {
            "category": category,
            "error_type": type(error).__name__,
            "message": _bounded_message(error, self._message_chars),
        }
        with self._lock:
            self._counts[category] += 1
            self._messages.append(row)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "counts": dict(sorted(self._counts.items())),
                "messages": [dict(row) for row in self._messages],
            }

