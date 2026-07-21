"""Read-only metadata inspection for artifacts excluded from formal workflows."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class LegacyArtifactSummary:
    path: str
    kind: str
    schema_version: str
    pre_core_fix: bool
    rank_eligible: bool = False
    formal_output_allowed: bool = False


def inspect_legacy_artifact(path: str | Path) -> LegacyArtifactSummary:
    """Inspect basic JSON metadata without migrating, ranking, or writing."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (UnicodeError, json.JSONDecodeError):
        return LegacyArtifactSummary(
            path=str(source),
            kind="unknown",
            schema_version="unversioned",
            pre_core_fix=True,
        )
    if type(payload) is not dict:
        schema = "unversioned"
        kind = "strategy" if "strategy" in source.name.lower() else "unknown"
    else:
        schema = str(
            payload.get("schema_version")
            or payload.get("checkpoint_schema_version")
            or payload.get("report_schema")
            or "unversioned"
        )
        if "formula_tokens" in payload or "strategy" in source.name.lower():
            kind = "strategy"
        elif "checkpoint_schema_version" in payload:
            kind = "checkpoint"
        elif "report_schema" in payload:
            kind = "backtest-report"
        elif "step" in payload:
            kind = "history"
        else:
            kind = "unknown"
    return LegacyArtifactSummary(
        path=str(source),
        kind=kind,
        schema_version=schema,
        pre_core_fix=True,
    )
