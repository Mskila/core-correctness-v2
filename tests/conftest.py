from __future__ import annotations

import json
import os

import pytest


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if report.when != "call" or not report.failed:
        return
    diagnostics = {
        "seed": os.environ.get("ALPHAMASTER_CI_SEED", "42"),
        "formula": getattr(item, "formula", "unavailable"),
        "dataset_identity": getattr(item, "dataset_identity", "unavailable"),
        "trace_digest": getattr(item, "trace_digest", "unavailable"),
        "test": item.nodeid,
    }
    print("CORE_CI_DIAGNOSTICS=" + json.dumps(diagnostics, sort_keys=True))
