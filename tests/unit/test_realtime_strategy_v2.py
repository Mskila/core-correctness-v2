import json

import pytest

import web.realtime_manager as realtime
import web.app as web_app
import web.progress as progress
import web.strategy_file as strategy_file
from model_core.semantics import ArtifactCompatibilityError
from web.data_sources.base import Bar
from tests.unit.test_artifacts import strategy_artifact
from tests.unit.test_web_strategy_v2 import _timeframe_artifact, _write_artifact


class _Source:
    label = "fake"

    def supported_timeframes(self):
        return ["1h", "4h"]


def _path(tmp_path, artifact):
    path = tmp_path / artifact.run_identity.strategy_filename()
    path.write_text(json.dumps(artifact.to_dict()), encoding="utf-8")
    return path


def test_source_facing_1h_matches_canonical_h1_artifact(monkeypatch, tmp_path) -> None:
    artifact = strategy_artifact()
    path = _path(tmp_path, artifact)
    monkeypatch.setattr(realtime, "_VALID_KINDS", {"fake"})
    monkeypatch.setattr(realtime, "get_source", lambda source: _Source())
    manager = realtime.RealtimeManager()
    task = manager._add_task_internal("fake", "EURUSD", "1h", str(path), persist=False)
    assert task.timeframe == "1h"
    assert task.strategy_timeframe == "H1"
    assert task.formula == list(artifact.formula_tokens)


@pytest.mark.parametrize(("symbol", "timeframe"), [("GBPUSD", "1h"), ("EURUSD", "4h")])
def test_watch_identity_mismatch_fails_closed(monkeypatch, tmp_path, symbol, timeframe) -> None:
    path = _path(tmp_path, strategy_artifact())
    monkeypatch.setattr(realtime, "_VALID_KINDS", {"fake"})
    monkeypatch.setattr(realtime, "get_source", lambda source: _Source())
    with pytest.raises(ValueError, match="identity mismatch"):
        realtime.RealtimeManager()._add_task_internal(
            "fake", symbol, timeframe, str(path), persist=False
        )


@pytest.mark.parametrize("payload", [[0], {"formula": [0]}, {"vocab_version": "legacy", "formula": [0]}])
def test_realtime_loader_rejects_non_v2(tmp_path, payload) -> None:
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    before = path.read_bytes()
    with pytest.raises(ArtifactCompatibilityError, match="strategy artifact"):
        realtime._load_strategy_meta(str(path))
    assert path.read_bytes() == before


def test_realtime_loader_requires_canonical_immutable_basename(tmp_path) -> None:
    artifact = strategy_artifact()
    payload = json.dumps(artifact.to_dict()).encode("utf-8")
    canonical = tmp_path / artifact.run_identity.strategy_filename()
    canonical.write_bytes(payload)
    loaded = realtime._load_strategy_meta(str(canonical))
    assert loaded["fingerprint"] == artifact.fingerprint
    assert loaded["symbol"] == artifact.run_identity.artifact_identity.symbol
    assert loaded["timeframe"] == artifact.run_identity.artifact_identity.timeframe

    renamed = tmp_path / "renamed-user-copy.json"
    renamed.write_bytes(payload)
    before = renamed.read_bytes()
    with pytest.raises(ValueError, match="canonical|filename|basename"):
        realtime._load_strategy_meta(str(renamed))
    assert renamed.read_bytes() == before == payload
    assert sorted(path.name for path in tmp_path.iterdir()) == sorted(
        [canonical.name, renamed.name]
    )


def test_realtime_strategy_rows_keep_h1_and_h4_exact_paths(monkeypatch, tmp_path) -> None:
    strategies = tmp_path / "strategies"
    strategies.mkdir()
    h1_old = _write_artifact(
        strategies, _timeframe_artifact("H1", "2026-07-15T00:00:00Z")
    )
    h1_new = _write_artifact(
        strategies, _timeframe_artifact("H1", "2026-07-16T00:00:00Z")
    )
    h4 = _write_artifact(
        strategies, _timeframe_artifact("H4", "2026-07-17T00:00:00Z")
    )
    monkeypatch.setattr(progress, "STRATEGIES_DIR", strategies)
    monkeypatch.setattr(strategy_file, "STRATEGIES_DIR", strategies)
    rows = web_app.api_realtime_strategies()["strategies"]
    by_timeframe = {row["timeframe"]: row for row in rows}
    assert set(by_timeframe) == {"H1", "H4"}
    assert by_timeframe["H1"]["strategy_file"] == str(h1_new.resolve())
    assert by_timeframe["H4"]["strategy_file"] == str(h4.resolve())
    assert by_timeframe["H1"]["strategy_file"] != str(h1_old.resolve())
    assert len({row["strategy_file"] for row in rows}) == 2


def test_realtime_rows_order_generated_at_as_utc_instant(monkeypatch, tmp_path) -> None:
    strategies = tmp_path / "strategies"
    strategies.mkdir()
    earlier = _write_artifact(
        strategies, _timeframe_artifact("H1", "2026-07-16T00:00:00Z", 0)
    )
    later = _write_artifact(
        strategies, _timeframe_artifact("H1", "2026-07-16T00:00:00.500000Z", 1)
    )
    monkeypatch.setattr(progress, "STRATEGIES_DIR", strategies)
    monkeypatch.setattr(strategy_file, "STRATEGIES_DIR", strategies)
    rows = web_app.api_realtime_strategies()["strategies"]
    assert len(rows) == 1
    assert rows[0]["strategy_file"] == str(later.resolve())
    assert rows[0]["strategy_file"] != str(earlier.resolve())
    assert rows[0]["generated_at"] == "2026-07-16T00:00:00.500000Z"


def test_realtime_rows_preserve_submicrosecond_order(monkeypatch, tmp_path) -> None:
    strategies = tmp_path / "strategies"
    strategies.mkdir()
    earlier = _write_artifact(
        strategies,
        _timeframe_artifact(
            "H1", "2026-07-16T00:00:00.0000001Z", 0, "f" * 32
        ),
    )
    later = _write_artifact(
        strategies,
        _timeframe_artifact(
            "H1", "2026-07-16T00:00:00.0000002Z", 1, "0" * 32
        ),
    )
    monkeypatch.setattr(progress, "STRATEGIES_DIR", strategies)
    monkeypatch.setattr(strategy_file, "STRATEGIES_DIR", strategies)
    rows = web_app.api_realtime_strategies()["strategies"]
    assert len(rows) == 1
    assert rows[0]["strategy_file"] == str(later.resolve())
    assert rows[0]["strategy_file"] != str(earlier.resolve())


class _DemandSource(_Source):
    def __init__(self, *, short_by=0):
        self.short_by = short_by
        self.requests = []

    def fetch_bars(self, symbol, timeframe, n, drop_forming=True):
        self.requests.append(n)
        return [
            Bar(index, 1.0, 1.0, 1.0, 1.0, 1.0)
            for index in range(max(0, n - self.short_by))
        ]


def _demand_task(monkeypatch, tmp_path, manager, source):
    artifact = strategy_artifact()
    path = _path(tmp_path, artifact)
    monkeypatch.setattr(realtime, "_VALID_KINDS", {"fake"})
    monkeypatch.setattr(realtime, "get_source", lambda kind: source)
    return manager._add_task_internal("fake", "EURUSD", "1h", str(path), persist=False)


def test_realtime_fetches_formula_warmup_and_exact_count_reaches_evaluation(
    monkeypatch, tmp_path
) -> None:
    source = _DemandSource()
    manager = realtime.RealtimeManager()
    task = _demand_task(monkeypatch, tmp_path, manager, source)
    monkeypatch.setattr(realtime, "formula_warmup_bars", lambda length: 744, raising=False)
    monkeypatch.setattr(realtime, "_ensure_closed_bars", lambda bars, timeframe: bars)
    seen = []

    def evaluate(formula, raw):
        count = raw["close"].shape[1]
        seen.append(count)
        return {
            "state": "ok", "direction": "LONG", "strength": 0.5,
            "position": 0.5, "factor_value": 0.75, "bars_used": count,
            "message": "",
        }

    monkeypatch.setattr(realtime, "evaluate_signal", evaluate)
    manager._evaluate_task(task)
    assert source.requests == [744]
    assert seen == [744]
    assert task.state == "ok"
    assert task.bars_used == 744


def test_realtime_one_short_remains_insufficient(monkeypatch, tmp_path) -> None:
    source = _DemandSource(short_by=1)
    manager = realtime.RealtimeManager()
    task = _demand_task(monkeypatch, tmp_path, manager, source)
    monkeypatch.setattr(realtime, "formula_warmup_bars", lambda length: 4, raising=False)
    monkeypatch.setattr(realtime, "_ensure_closed_bars", lambda bars, timeframe: bars)

    def evaluate(formula, raw):
        count = raw["close"].shape[1]
        return {"state": "insufficient", "bars_used": count, "message": f"{count}/4"}

    monkeypatch.setattr(realtime, "evaluate_signal", evaluate)
    manager._evaluate_task(task)
    assert source.requests == [4]
    assert task.state == "insufficient"
    assert task.bars_used == 3


def test_realtime_bar_cache_refetches_for_larger_formula_demand(
    monkeypatch, tmp_path
) -> None:
    source = _DemandSource()
    manager = realtime.RealtimeManager()
    task = _demand_task(monkeypatch, tmp_path, manager, source)
    monkeypatch.setattr(realtime, "formula_warmup_bars", lambda length: {1: 3, 2: 5}[length], raising=False)
    monkeypatch.setattr(realtime, "_ensure_closed_bars", lambda bars, timeframe: bars)
    monkeypatch.setattr(
        realtime,
        "evaluate_signal",
        lambda formula, raw: {
            "state": "ok", "direction": "LONG", "strength": 0.5,
            "position": 0.5, "factor_value": 0.75,
            "bars_used": raw["close"].shape[1], "message": "",
        },
    )
    manager._evaluate_task(task)
    task.formula = [0, 1]
    manager._evaluate_task(task)
    task.formula = [0]
    manager._evaluate_task(task)
    assert source.requests == [3, 5]
    assert task.bars_used == 5
