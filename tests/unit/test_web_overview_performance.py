from pathlib import Path


def test_overview_computes_active_symbol_progress_once(monkeypatch) -> None:
    import web.app as app_module

    calls = []
    monkeypatch.setattr(app_module, "load_settings", lambda: {
        "last_data_file": "XAUUSD_H1.parquet", "numeric_time_unit": "s"
    })
    monkeypatch.setattr(app_module, "_inspect_parquet_cached", lambda *args, **kwargs: {
        "data_file": "XAUUSD_H1.parquet", "symbol": "XAUUSD", "valid": True
    })
    monkeypatch.setattr(app_module.training_manager, "status", lambda: {
        "active": True,
        "job": {"symbol": "XAUUSD", "state": "running"},
        "errors": {"counts": {}, "messages": []},
    })
    monkeypatch.setattr(app_module, "_attach_training_time", lambda row, **kwargs: row)

    def progress(symbol, active):
        calls.append((symbol, active))
        return {
            "symbol": symbol, "status": "in_progress", "current_step": 40,
            "train_steps": 9000, "progress_pct": 0.4, "best_score": 1.0,
            "val_score": 0.5, "formula_decoded": None,
            "has_checkpoint": True, "has_strategy": False,
        }

    monkeypatch.setattr(app_module, "_progress_with_live_step", progress)

    payload = app_module.api_overview()

    assert payload["progress"]["current_step"] == 40
    assert calls == [("XAUUSD", True)]


def test_parquet_inspection_cache_skips_unchanged_file(monkeypatch, tmp_path) -> None:
    import web.app as app_module

    source = tmp_path / "XAUUSD_H1.parquet"
    source.write_bytes(b"stable")
    calls = []
    monkeypatch.setattr(
        app_module,
        "inspect_parquet_file",
        lambda path, numeric_time_unit=None: calls.append(Path(path)) or {
            "symbol": "XAUUSD", "data_file": str(path), "valid": True
        },
    )
    app_module._data_file_info_cache.clear()

    first = app_module._inspect_parquet_cached(str(source), numeric_time_unit="s")
    second = app_module._inspect_parquet_cached(str(source), numeric_time_unit="s")

    assert first == second
    assert calls == [source]


def test_frontend_prevents_overlapping_overview_requests() -> None:
    source = (
        Path(__file__).resolve().parents[2] / "web" / "static" / "app.js"
    ).read_text(encoding="utf-8")

    assert "overviewInFlight" in source
    assert "if (overviewInFlight) return" in source
    assert "overviewInFlight = false" in source
