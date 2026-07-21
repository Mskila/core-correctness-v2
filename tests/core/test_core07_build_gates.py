from __future__ import annotations

from pathlib import Path

import yaml
import pytest

from scripts.release_gate import require_core_gates


ROOT = Path(__file__).resolve().parents[2]


def _requirements(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8").lower()


def test_core_dependencies_are_locked_and_mt5_is_an_extra() -> None:
    core = _requirements("requirements.txt")
    constraints = _requirements("constraints-core.txt")
    mt5 = _requirements("requirements-mt5.txt")

    assert "metatrader5" not in core
    assert "metatrader5==" in mt5
    assert "-r requirements.txt" in mt5
    for package in ("torch", "numpy", "pandas", "pyarrow"):
        assert f"{package}==" in constraints


def test_project_declares_the_supported_python_range() -> None:
    project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'requires-python = ">=3.10,<3.13"' in project


def test_required_ci_has_all_release_gate_jobs_and_platforms() -> None:
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/core-required.yml").read_text(encoding="utf-8")
    )
    jobs = workflow["jobs"]
    assert set(jobs) == {
        "lint-type", "core-unit", "windows-core", "deterministic", "integration"
    }
    assert jobs["core-unit"]["strategy"]["matrix"]["python-version"] == [
        "3.10", "3.11", "3.12"
    ]
    assert jobs["windows-core"]["runs-on"] == "windows-latest"
    assert "matrix" in jobs["deterministic"]["strategy"]
    assert jobs["integration"]["runs-on"] == "ubuntu-latest"
    for job in jobs.values():
        assert job["timeout-minutes"] > 0


def test_pytest_required_gate_enforces_timeout_coverage_and_warning_policy() -> None:
    config = (ROOT / "pytest.ini").read_text(encoding="utf-8")
    assert "timeout = " in config
    assert "filterwarnings =" in config
    workflow = (ROOT / ".github/workflows/core-required.yml").read_text(encoding="utf-8")
    assert "--cov-fail-under=" in workflow
    assert "--cov=model_core" in workflow


def test_release_gate_requires_explicit_success_signal(monkeypatch) -> None:
    monkeypatch.delenv("CORE_REQUIRED_GATES", raising=False)
    with pytest.raises(SystemExit, match="blocked"):
        require_core_gates()
    monkeypatch.setenv("CORE_REQUIRED_GATES", "failed")
    with pytest.raises(SystemExit, match="blocked"):
        require_core_gates()
    monkeypatch.setenv("CORE_REQUIRED_GATES", "passed")
    require_core_gates()
