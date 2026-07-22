import json
import subprocess
from html.parser import HTMLParser
from pathlib import Path

import pytest

import web.training_manager as training


ROOT = Path(__file__).resolve().parents[2]


class _ContractParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = set()
        self.mode_values = []
        self.numeric_unit_values = []
        self.backtest_numeric_unit_values = []
        self.text_by_id = {}
        self._active_id = None
        self._active_select = None

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if "id" in values:
            self.ids.add(values["id"])
            self._active_id = values["id"]
        if tag == "select":
            self._active_select = values.get("id")
        if tag == "option" and values.get("value"):
            if self._active_select == "numericTimeUnitSelect":
                self.numeric_unit_values.append(values["value"])
            elif self._active_select == "btNumericTimeUnitSelect":
                self.backtest_numeric_unit_values.append(values["value"])
            elif self._active_select == "btModeSelect":
                self.mode_values.append(values["value"])

    def handle_endtag(self, tag):
        if tag == "select":
            self._active_select = None
        self._active_id = None

    def handle_data(self, data):
        if self._active_id and data.strip():
            self.text_by_id[self._active_id] = self.text_by_id.get(self._active_id, "") + data.strip()


def _run_controller(commands):
    script = """
global.window = {};
global.document = {};
const api = require('./web/static/app.js');
const controller = api.createBacktestController();
const commands = JSON.parse(process.argv[1]);
const results = [];
for (const command of commands) {
  try { results.push({ok: true, value: controller[command.name](...(command.args || []))}); }
  catch (error) { results.push({ok: false, error: error.message}); }
}
console.log(JSON.stringify(results));
"""
    completed = subprocess.run(
        ["node", "-e", script, json.dumps(commands)], cwd=ROOT,
        check=True, capture_output=True, text=True,
    )
    return json.loads(completed.stdout)


def _run_actual_retrain_confirmation():
    script = """
const fs = require('fs');
const vm = require('vm');
const confirmations = [];
let requests = 0;
const context = {
  window: {
    confirm(message) { confirmations.push(message); return false; },
  },
  document: {},
  module: {exports: {}},
  console,
  fetch() { requests += 1; throw new Error('cancelled confirmation made a request'); },
};
vm.createContext(context);
vm.runInContext(fs.readFileSync('./web/static/app.js', 'utf8'), context);
vm.runInContext("selectedDataFile = 'training.parquet'", context);
Promise.resolve(vm.runInContext('retrainFromScratch()', context)).then(() => {
  console.log(JSON.stringify({confirmations, requests}));
});
"""
    completed = subprocess.run(
        ["node", "-e", script], cwd=ROOT,
        check=True, capture_output=True, encoding="utf-8",
    )
    return json.loads(completed.stdout)


def test_html_exposes_exact_modes_and_independent_data_controls() -> None:
    parser = _ContractParser()
    parser.feed((ROOT / "web/static/index.html").read_text(encoding="utf-8"))
    assert {
        "btBrowseDataBtn", "btDataCard", "btModeSelect", "numericTimeUnitSelect",
        "btNumericTimeUnitSelect",
    }.issubset(parser.ids)
    assert len(parser.mode_values) == 2
    assert set(parser.mode_values) == {
        "in_sample_replay", "out_of_sample_backtest",
    }
    assert parser.numeric_unit_values == ["s", "ms", "us", "ns"]
    assert parser.backtest_numeric_unit_values == ["s", "ms", "us", "ns"]
    assert parser.text_by_id["fromScratchHelp"] == (
        "重新训练会创建全新的 run；所有既有 checkpoint、策略、history 和分数均原样保留；"
        "新 run 不从旧产物播种，也不受旧分数约束。"
    )


def test_controller_blocks_missing_fields_and_builds_explicit_payload() -> None:
    results = _run_controller([
        {"name": "payload", "args": [{"commission_pct": 0.1, "slippage_pct": 0.2}]},
        {"name": "selectStrategy", "args": ["strategy.json"]},
        {"name": "selectData", "args": ["oos.parquet"]},
        {"name": "selectMode", "args": ["out_of_sample_backtest"]},
        {"name": "payload", "args": [{"commission_pct": 0.1, "slippage_pct": 0.2}]},
    ])
    assert results[0] == {"ok": False, "error": "strategy_file is required"}
    assert results[-1]["value"] == {
        "strategy_file": "strategy.json", "data_file": "oos.parquet",
        "mode": "out_of_sample_backtest", "commission_pct": 0.1, "slippage_pct": 0.2,
    }


def test_controller_keeps_backtest_data_independent_and_rejects_invalid_mode() -> None:
    results = _run_controller([
        {"name": "selectData", "args": ["backtest.parquet"]},
        {"name": "selectStrategy", "args": ["strategy.json"]},
        {"name": "selectMode", "args": ["oos"]},
        {"name": "state"},
    ])
    assert results[2] == {"ok": False, "error": "invalid backtest mode"}
    assert results[3]["value"]["dataFile"] == "backtest.parquet"


def test_actual_from_scratch_handler_confirms_immutable_independent_run() -> None:
    result = _run_actual_retrain_confirmation()
    assert result == {
        "confirmations": [
            "重新训练会创建一个全新的独立 run。\n"
            "所有已有 checkpoint、history、strategy、score、report 和 package 文件都会原样保留。\n"
            "新 run 不从旧产物播种，也不受旧分数下限约束。\n\n"
            "确定要开始新的训练 run 吗？"
        ],
        "requests": 0,
    }


@pytest.mark.parametrize("from_scratch", [False, True])
def test_training_start_preserves_legacy_history_and_forwards_mode(
    monkeypatch, tmp_path, from_scratch
) -> None:
    project = tmp_path / "external-project"
    logs = project / "logs"
    logs.mkdir(parents=True)
    history = project / "training_history_EURUSD.json"
    legacy = b"LEGACY-USER-BYTES-MUST-SURVIVE"
    history.write_bytes(legacy)
    commands = []

    class _Process:
        pid = 4312

        def poll(self):
            return None

    def popen(command, **kwargs):
        commands.append((list(command), kwargs))
        return _Process()

    monkeypatch.setattr(training, "PROJECT_ROOT", project)
    monkeypatch.setattr(training, "LOG_DIR", logs)
    monkeypatch.setattr(training.subprocess, "Popen", popen)
    manager = training.TrainingManager()
    try:
        manager.start(
            "input.parquet", "EURUSD", "H1", from_scratch=from_scratch,
            numeric_time_unit="ns",
        )
    finally:
        if manager._log_fp is not None:
            manager._log_fp.close()
            manager._log_fp = None

    assert history.exists()
    assert history.read_bytes() == legacy
    assert len(commands) == 1
    command, kwargs = commands[0]
    assert command.count("--from-scratch") == int(from_scratch)
    assert command.count("--data-file") == 1
    data_flag = command.index("--data-file")
    assert command[data_flag + 1] == "input.parquet"
    assert command.count("--numeric-time-unit") == 1
    unit_flag = command.index("--numeric-time-unit")
    assert command[unit_flag + 1] == "ns"
    assert kwargs["cwd"] == project
