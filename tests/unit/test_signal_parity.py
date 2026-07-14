import torch

from config import Config
from model_core.execution import factor_to_position
from strategy_manager import signal


def test_signal_wrapper_matches_shared_execution_elementwise() -> None:
    factors = torch.tensor([-3.0, -0.01, 0.0, 0.01, 3.0])

    expected = factor_to_position(
        factors,
        min_exposure=float(Config.MIN_TRADE_EXPOSURE),
    )
    actual = signal.compute_target_positions(factors)

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_signal_wrapper_delegates_with_config_default(monkeypatch) -> None:
    factors = torch.tensor([0.25])
    sentinel = torch.tensor([0.75])
    captured: dict[str, object] = {}

    def fake_factor_to_position(
        actual_factors: torch.Tensor,
        *,
        min_exposure: float,
    ) -> torch.Tensor:
        captured["factors"] = actual_factors
        captured["min_exposure"] = min_exposure
        return sentinel

    monkeypatch.setattr(Config, "MIN_TRADE_EXPOSURE", 0.123)
    monkeypatch.setattr(
        signal,
        "factor_to_position",
        fake_factor_to_position,
        raising=False,
    )

    actual = signal.compute_target_positions(factors)

    assert captured == {
        "factors": factors,
        "min_exposure": 0.123,
    }
    assert actual is sentinel
