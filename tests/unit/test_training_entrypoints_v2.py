from __future__ import annotations

import pytest

from model_core.artifacts import ArtifactCompatibilityError
from model_core.island_engine import IslandAlphaEngine
import main
import train_ftmo_island


def test_island_training_rejected_at_entrypoint() -> None:
    with pytest.raises(ArtifactCompatibilityError, match="V2.*single"):
        IslandAlphaEngine(None)


@pytest.mark.parametrize("argv", [[], ["--group", "risk"], ["--cross-section"]])
def test_legacy_main_modes_reject_before_data_access(monkeypatch, argv) -> None:
    monkeypatch.setattr(
        main,
        "MT5DataFetcher",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("data accessed")),
    )
    with pytest.raises(ArtifactCompatibilityError, match="V2.*single"):
        main.main(argv)


def test_main_single_delegates(monkeypatch) -> None:
    events = []

    class Fetcher:
        def __init__(self, **kwargs): events.append(("fetcher", kwargs))
        def __enter__(self): return self
        def __exit__(self, *args): return False

    monkeypatch.setattr(main, "MT5DataFetcher", Fetcher)
    import train_single
    sentinel = object()
    monkeypatch.setattr(
        train_single, "train_single",
        lambda *args, **kwargs: events.append(("single", args, kwargs)) or sentinel,
    )
    result = main.main(["--single", "EURUSD", "--offline", "--random-seed", "7"])
    assert result is sentinel
    assert events[1][0] == "single"
    assert events[1][2]["random_seed"] == 7


def test_ftmo_island_rejects_before_data_access() -> None:
    with pytest.raises(ArtifactCompatibilityError, match="V2.*single"):
        train_ftmo_island.main()
