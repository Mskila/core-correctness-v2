"""Fail closed unless the required CORE-07 CI aggregate has passed."""

from __future__ import annotations

import os


def require_core_gates() -> None:
    status = os.environ.get("CORE_REQUIRED_GATES", "").strip().lower()
    if status != "passed":
        raise SystemExit(
            "release/retraining blocked: CORE_REQUIRED_GATES must be exactly 'passed'"
        )


if __name__ == "__main__":
    require_core_gates()
