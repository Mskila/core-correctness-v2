from __future__ import annotations

from types import SimpleNamespace

import web.app as web_app
import web.settings as settings_module


def test_browse_and_training_forward_and_persist_numeric_time_unit(monkeypatch) -> None:
    inspected: list[tuple[str, str | None]] = []
    saved: list[dict] = []
    started: dict = {}
    monkeypatch.setattr(web_app, "pick_parquet_file", lambda: "XAUUSD_M15.parquet")
    monkeypatch.setattr(
        web_app,
        "inspect_parquet_file",
        lambda path, *, numeric_time_unit=None: (
            inspected.append((path, numeric_time_unit))
            or {
                "valid": True,
                "data_file": path,
                "symbol": "XAUUSD",
                "timeframe": "M15",
            }
        ),
    )
    monkeypatch.setattr(web_app, "save_settings", lambda value: saved.append(value))
    monkeypatch.setattr(
        web_app.training_manager,
        "start",
        lambda **kwargs: (
            started.update(kwargs) or SimpleNamespace(to_dict=lambda: kwargs)
        ),
    )

    browsed = web_app._browse_data_file("ms")
    assert browsed["ok"] is True
    assert inspected[-1] == ("XAUUSD_M15.parquet", "ms")
    assert saved[-1] == {
        "last_data_file": "XAUUSD_M15.parquet",
        "numeric_time_unit": "ms",
    }

    response = web_app.api_training_start(
        web_app.StartTrainingRequest(
            data_file="XAUUSD_M15.parquet", numeric_time_unit="us"
        )
    )
    assert response["ok"] is True
    assert inspected[-1] == ("XAUUSD_M15.parquet", "us")
    assert started["numeric_time_unit"] == "us"
    assert saved[-1]["numeric_time_unit"] == "us"


def test_settings_default_and_round_trip_numeric_time_unit(monkeypatch, tmp_path) -> None:
    path = tmp_path / "web_settings.json"
    monkeypatch.setattr(settings_module, "SETTINGS_PATH", path)
    assert settings_module.load_settings()["numeric_time_unit"] == "s"
    assert settings_module.save_settings({"numeric_time_unit": "ns"})[
        "numeric_time_unit"
    ] == "ns"
    assert settings_module.load_settings()["numeric_time_unit"] == "ns"
    assert settings_module.save_settings({"numeric_time_unit": "minutes"})[
        "numeric_time_unit"
    ] == "s"
