from __future__ import annotations

import builtins
import hashlib
import os

import train_file
from model_core.config import ModelConfig


def test_train_file_default_seed_and_from_scratch_ignore_old_strategy(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.chdir(tmp_path)
    old_strategy = tmp_path / "strategies" / "best_EURUSD.json"
    old_strategy.parent.mkdir()
    old_bytes = b"OLD-STRATEGY-WITHOUT-V2-IDENTITY\x00\xff"
    old_strategy.write_bytes(old_bytes)
    old_path = os.path.abspath(os.fspath(old_strategy))
    strategy_dir = os.path.dirname(old_path)
    before = (
        hashlib.sha256(old_bytes).hexdigest(),
        old_strategy.stat().st_size,
        old_strategy.stat().st_mtime_ns,
    )
    events = []

    class Manager:
        def __init__(self, path, *, numeric_time_unit=None):
            events.append(("init", path, numeric_time_unit))

        def load(self):
            events.append(("load",))

    sentinel = object()

    def service(manager, **kwargs):
        events.append(("service", manager, kwargs))
        return sentinel

    def guarded(value) -> bool:
        try:
            candidate = os.path.abspath(os.fspath(value))
        except TypeError:
            return False
        return candidate in {old_path, strategy_dir}

    real_open = builtins.open

    def guarded_open(file, *args, **kwargs):
        if guarded(file):
            raise AssertionError("from-scratch adapter accessed old strategy bytes")
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded_open)
    for name in (
        "open", "read_bytes", "read_text", "write_bytes", "write_text",
        "unlink", "rename", "replace", "exists", "is_file", "stat",
        "glob", "rglob", "iterdir",
    ):
        original = getattr(train_file.Path, name)

        def guarded_path(self, *args, _name=name, _original=original, **kwargs):
            if guarded(self) or (
                _name in {"rename", "replace"} and args and guarded(args[0])
            ):
                raise AssertionError(
                    f"from-scratch adapter attempted old strategy {_name}"
                )
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(train_file.Path, name, guarded_path)
    for name in ("remove", "unlink", "rename", "replace"):
        original = getattr(os, name)

        def guarded_os(path, *args, _name=name, _original=original, **kwargs):
            if guarded(path) or (
                _name in {"rename", "replace"} and args and guarded(args[0])
            ):
                raise AssertionError(
                    f"from-scratch adapter attempted old strategy {_name}"
                )
            return _original(path, *args, **kwargs)

        monkeypatch.setattr(os, name, guarded_os)

    monkeypatch.setattr(train_file, "ParquetDataManager", Manager)
    monkeypatch.setattr(train_file, "run_training_session", service)
    data_file = tmp_path / "EURUSD_H1.parquet"
    result = train_file.train_from_file(
        str(data_file), from_scratch=True, numeric_time_unit="s"
    )

    descriptor = os.open(old_strategy, os.O_RDONLY)
    try:
        after_bytes = os.read(descriptor, len(old_bytes) + 1)
    finally:
        os.close(descriptor)
    after_stat = os.stat(old_strategy)
    assert result is sentinel
    assert events[0] == ("init", str(data_file), "s")
    assert events[1] == ("load",)
    assert events[2][2]["from_scratch"] is True
    assert events[2][2]["random_seed"] == ModelConfig.RANDOM_SEED
    assert (
        hashlib.sha256(after_bytes).hexdigest(),
        after_stat.st_size,
        after_stat.st_mtime_ns,
    ) == before


def test_train_file_loads_parquet_then_delegates(monkeypatch, tmp_path) -> None:
    events = []

    class Manager:
        def __init__(self, path, *, numeric_time_unit=None):
            events.append(("init", path, numeric_time_unit))

        def load(self):
            events.append(("load",))

    sentinel = object()
    monkeypatch.setattr(train_file, "ParquetDataManager", Manager)
    monkeypatch.setattr(
        train_file,
        "run_training_session",
        lambda manager, **kwargs: events.append(("service", manager, kwargs)) or sentinel,
    )
    data_file = tmp_path / "EURUSD_H1.parquet"
    result = train_file.train_from_file(
        str(data_file), from_scratch=True, random_seed=123, numeric_time_unit="ms"
    )
    assert result is sentinel
    assert events[0] == ("init", str(data_file), "ms")
    assert events[1] == ("load",)
    assert events[2][2]["from_scratch"] is True
    assert events[2][2]["random_seed"] == 123


def test_train_file_parser_accepts_only_supported_numeric_time_units() -> None:
    parser = train_file._parser()
    args = parser.parse_args(
        ["--data-file", "XAUUSD_M15.parquet", "--numeric-time-unit", "us"]
    )
    assert args.numeric_time_unit == "us"

    import pytest

    with pytest.raises(SystemExit):
        parser.parse_args(
            ["--data-file", "XAUUSD_M15.parquet", "--numeric-time-unit", "minutes"]
        )
