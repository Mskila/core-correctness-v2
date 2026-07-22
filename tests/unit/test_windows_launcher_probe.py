from scripts import windows_launcher_probe as probe


def test_wait_reports_free_port_without_polling_health(monkeypatch) -> None:
    monkeypatch.setattr(probe, "port_is_listening", lambda port: False)
    monkeypatch.setattr(
        probe,
        "health_is_ready",
        lambda port: (_ for _ in ()).throw(AssertionError("health should not be called")),
    )

    assert probe.wait_for_existing_service(8765, 15) == 0


def test_wait_recognizes_existing_alphamaster(monkeypatch) -> None:
    monkeypatch.setattr(probe, "port_is_listening", lambda port: True)
    monkeypatch.setattr(probe, "health_is_ready", lambda port: True)

    assert probe.wait_for_existing_service(8765, 15) == 2
