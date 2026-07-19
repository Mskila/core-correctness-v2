import torch

from model_core.execution import factor_to_position
from strategy_manager import live_signal


def test_live_signal_routes_position_through_shared_helper(monkeypatch) -> None:
    monkeypatch.setattr(live_signal, "formula_warmup_bars", lambda n: 1)
    monkeypatch.setattr(live_signal.MT5FeatureEngineer, "compute_features", lambda raw: torch.zeros(1, 1, 1))
    monkeypatch.setattr(live_signal._VM, "execute", lambda formula, feats: torch.tensor([[0.75]]))
    raw = {"close": torch.ones(1, 1)}
    result = live_signal.evaluate_signal([0], raw)
    expected = factor_to_position(torch.tensor([[0.75]]), min_exposure=live_signal.min_exposure()).item()
    assert expected != round(expected, 4)
    assert result["position"] == expected
    assert result["strength"] == abs(expected)


def test_one_bar_short_fails_closed(monkeypatch) -> None:
    monkeypatch.setattr(live_signal, "formula_warmup_bars", lambda n: 3)
    result = live_signal.evaluate_signal([0], {"close": torch.ones(1, 2)})
    assert result["state"] == "insufficient"
