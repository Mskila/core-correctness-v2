# Feature: mt5-alphagpt-refactor, Property 12: 策略信号触发买单
"""
Property-based tests for strategy_manager.runner (MT5StrategyRunner).

Property 12 Validates: Requirements 10.4

For any signal score tensor:
  - If score > Config.BUY_THRESHOLD AND symbol not in portfolio.positions
    → trader.buy() MUST be called for that symbol
  - If score <= Config.BUY_THRESHOLD
    → trader.buy() MUST NOT be called for that symbol
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Dict, List
from unittest.mock import MagicMock, patch

import torch
import pytest
from hypothesis import given, settings, strategies as st
from hypothesis.strategies import composite

from config import Config
from strategy_manager.runner import MT5StrategyRunner
from strategy_manager.portfolio import MT5PortfolioManager, Position
from model_core.execution import factor_to_position


# ── Helpers ───────────────────────────────────────────────────────────────────

# MT5 BUY_THRESHOLD constant (0.70)
_THRESHOLD = Config.BUY_THRESHOLD


@given(st.floats(min_value=-5, max_value=5, allow_nan=False, allow_infinity=False))
def test_runner_position_mapping_matches_shared_execution(value: float) -> None:
    runner = MT5StrategyRunner.__new__(MT5StrategyRunner)
    runner._data_manager = MagicMock(symbols=["EURUSD"], raw_dict={})
    runner.symbol_formulas = {"EURUSD": [0]}
    runner.vm = MagicMock()
    factor = torch.tensor([[value]], dtype=torch.float32)
    runner.vm.execute.return_value = factor
    with patch(
        "model_core.features.MT5FeatureEngineer.compute_features",
        return_value=torch.zeros(1, 1, 1),
    ), patch("model_core.walk_forward.formula_warmup_bars", return_value=1):
        actual = runner._compute_targets()
    expected = factor_to_position(
        factor,
        min_exposure=float(Config.MIN_TRADE_EXPOSURE),
    )
    torch.testing.assert_close(actual, expected.flatten())


def _make_runner(
    symbols: List[str],
    held_symbols: List[str],
) -> MT5StrategyRunner:
    """
    Construct an MT5StrategyRunner via __new__ and inject required attributes.
    Uses _reconcile_positions (replaces removed _scan_for_entries).
    """
    runner = MT5StrategyRunner.__new__(MT5StrategyRunner)
    runner.formula = [1, 2, 3]

    mock_trader = MagicMock()
    mock_account = {"equity": 10_000.0, "margin_free": 5_000.0}
    mock_trader.get_account_info.return_value = mock_account
    mock_trader.buy.return_value = True
    mock_trader.open_short.return_value = True
    mock_trader.get_positions.side_effect = lambda symbol, magic: (
        [SimpleNamespace(type=0, volume=1.0)] if symbol in held_symbols else []
    )
    runner.trader = mock_trader

    mock_portfolio = MagicMock(spec=MT5PortfolioManager)
    mock_portfolio.positions = {sym: MagicMock() for sym in held_symbols}
    mock_portfolio.get_open_count.return_value = len(held_symbols)
    # get_direction: 0 if not held, 1 if held
    mock_portfolio.get_direction.side_effect = lambda s: 1 if s in held_symbols else 0
    runner.portfolio = mock_portfolio

    mock_risk = MagicMock()
    mock_risk.calculate_lot.return_value = 0.01
    runner.risk = mock_risk

    mock_data_manager = MagicMock()
    mock_data_manager.symbols = symbols
    runner._data_manager = mock_data_manager
    runner._last_refresh = 0.0
    runner._calc_lot = MagicMock(return_value=0.01)
    runner._close_symbol_positions = MagicMock(return_value=True)
    runner._record_position_after_open = MagicMock()

    return runner


# ── Hypothesis strategies ─────────────────────────────────────────────────────

# Valid MT5 symbol strings: uppercase alphabetic, 3-8 characters
symbol_strategy = st.text(
    alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    min_size=3,
    max_size=8,
)

# Scores clearly above threshold (to avoid floating-point boundary ambiguity)
above_threshold_strategy = st.floats(
    min_value=_THRESHOLD + 1e-6,
    max_value=1.0 - 1e-9,
    allow_nan=False,
    allow_infinity=False,
)

# Scores clearly at or below threshold
below_threshold_strategy = st.floats(
    min_value=-1.0,
    max_value=_THRESHOLD,
    allow_nan=False,
    allow_infinity=False,
)


@composite
def score_scenario_strategy(draw):
    """
    Draw a scenario with:
    - n symbols (1–5, unique)
    - for each symbol: a score and whether the symbol is already held

    Returns a dict with keys:
        symbols:        list[str]
        scores:         list[float]
        held_symbols:   list[str]  (already in portfolio)
    """
    n = draw(st.integers(min_value=1, max_value=5))
    symbols = draw(
        st.lists(symbol_strategy, min_size=n, max_size=n, unique=True)
    )

    scores = []
    held_symbols = []

    for sym in symbols:
        # Randomly choose above or below threshold
        use_above = draw(st.booleans())
        if use_above:
            score = draw(above_threshold_strategy)
        else:
            score = draw(below_threshold_strategy)
        scores.append(score)

        # Randomly decide if this symbol is already held
        is_held = draw(st.booleans())
        if is_held:
            held_symbols.append(sym)

    return {
        "symbols": symbols,
        "scores": scores,
        "held_symbols": held_symbols,
    }


# ── Property 12: 策略信号触发买单 ─────────────────────────────────────────────
# Validates: Requirements 10.4


@settings(max_examples=100, deadline=None)
@given(scenario=score_scenario_strategy())
def test_property12_buy_signal_triggers_buy(scenario: dict):
    """
    Property 12: neutral band 信号触发正确动作。

    用 _reconcile_positions 替代已删除的 _scan_for_entries。
    目标仓位由外部构造传入（模拟 _compute_targets 输出），
    验证 reconcile_action 正确决定 open/close/hold。

    Validates: Requirements 10.4
    """
    from strategy_manager.signal import (
        reconcile_action, target_to_direction,
        OPEN_LONG, OPEN_SHORT, CLOSE, HOLD, REVERSE_TO_LONG, REVERSE_TO_SHORT,
    )

    symbols: List[str] = scenario["symbols"]
    raw_scores: List[float] = scenario["scores"]
    held_symbols: List[str] = scenario["held_symbols"]

    # 把 scores 转换为 neutral band 目标仓位：>0.6 → +1, <-0.6 → -1, else 0
    targets = torch.zeros(len(symbols))
    for i, s in enumerate(raw_scores):
        if s > 0.6:
            targets[i] = 1.0
        elif s < -0.6:
            targets[i] = -1.0

    runner = _make_runner(symbols, held_symbols)

    # 直接调用 _reconcile_positions，不走 StackVM
    runner._reconcile_positions(targets)

    expected_buy_syms: set[str] = set()
    expected_sell_syms: set[str] = set()
    # 验证每个品种的 reconcile 结果
    for idx, sym in enumerate(symbols):
        target  = target_to_direction(float(targets[idx].item()))
        current = 1 if sym in held_symbols else 0
        expected_action = reconcile_action(current, target)

        if expected_action in (OPEN_LONG, REVERSE_TO_LONG):
            expected_buy_syms.add(sym)
        elif expected_action in (OPEN_SHORT, REVERSE_TO_SHORT):
            expected_sell_syms.add(sym)

    buy_syms = {c.args[0] for c in runner.trader.buy.call_args_list if c.args}
    sell_syms = {c.args[0] for c in runner.trader.open_short.call_args_list if c.args}
    assert buy_syms == expected_buy_syms
    assert sell_syms == expected_sell_syms
    for call in runner.trader.buy.call_args_list + runner.trader.open_short.call_args_list:
        assert call.args[1] == pytest.approx(0.01)


def test_property_assertions_detect_suppressed_buy_call() -> None:
    runner = _make_runner(["EURUSD"], [])
    runner._reconcile_positions(torch.tensor([1.0]))
    runner.trader.buy.reset_mock()
    with pytest.raises(AssertionError):
        assert {c.args[0] for c in runner.trader.buy.call_args_list} == {"EURUSD"}


def test_property_assertions_detect_suppressed_sell_call() -> None:
    runner = _make_runner(["EURUSD"], [])
    runner._reconcile_positions(torch.tensor([-1.0]))
    runner.trader.open_short.reset_mock()
    with pytest.raises(AssertionError):
        assert {c.args[0] for c in runner.trader.open_short.call_args_list} == {"EURUSD"}
