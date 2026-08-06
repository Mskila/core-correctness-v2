"""Immutable contracts for multi-timeframe PA/Alpha decision construction.

Stage 2 deliberately stops at theoretical, non-executable order plans.  Web
configuration, Codex review, broker constraints and MT5 execution are later
boundaries.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

from pa_core import StrategyCandidate


def _finite_real(value: object, *, field: str) -> float:
    if type(value) not in {int, float}:
        raise ValueError(f"{field} must be a finite real number")
    numeric = float(cast(int | float, value))
    if not math.isfinite(numeric):
        raise ValueError(f"{field} must be a finite real number")
    return numeric


@dataclass(frozen=True, slots=True)
class ClosedBarV1:
    """One closed OHLCV bar whose timestamp is its opening Unix second."""

    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    closed: bool = True
    segment_id: str = "0"
    bridged_short_gap: bool = False

    def __post_init__(self) -> None:
        if type(self.timestamp) is not int:
            raise ValueError("timestamp must be an integer Unix second")
        if self.closed is not True:
            raise ValueError("closed bars are required")
        values = {
            field: _finite_real(getattr(self, field), field=field)
            for field in ("open", "high", "low", "close", "volume")
        }
        if values["volume"] < 0.0:
            raise ValueError("volume must be non-negative")
        if values["high"] < max(values["open"], values["close"], values["low"]):
            raise ValueError("OHLC containment is invalid")
        if values["low"] > min(values["open"], values["close"], values["high"]):
            raise ValueError("OHLC containment is invalid")
        if type(self.segment_id) is not str or not self.segment_id:
            raise ValueError("segment_id must be a non-empty string")
        if type(self.bridged_short_gap) is not bool:
            raise ValueError("bridged_short_gap must be bool")


@dataclass(frozen=True, slots=True)
class TimeframeSeriesV1:
    symbol: str
    timeframe: str
    bars: tuple[ClosedBarV1, ...]
    tick: float

    def __post_init__(self) -> None:
        if type(self.symbol) is not str or not self.symbol.strip():
            raise ValueError("symbol must be non-empty")
        if self.timeframe not in {"H1", "M15", "M5"}:
            raise ValueError("timeframe must be one of H1, M15 or M5")
        if type(self.bars) is not tuple or any(type(bar) is not ClosedBarV1 for bar in self.bars):
            raise TypeError("bars must be an exact tuple of ClosedBarV1 values")
        if any(
            self.bars[index].timestamp <= self.bars[index - 1].timestamp
            for index in range(1, len(self.bars))
        ):
            raise ValueError("bar timestamps must be strictly increasing")
        if _finite_real(self.tick, field="tick") <= 0.0:
            raise ValueError("tick must be positive")


@dataclass(frozen=True, slots=True)
class AlignedTimeframeV1:
    timeframe: str
    bars: tuple[ClosedBarV1, ...]
    last_close_timestamp: int


@dataclass(frozen=True, slots=True)
class AlignedDecisionFrameV1:
    symbol: str
    decision_close_timestamp: int
    h1: AlignedTimeframeV1
    m15: AlignedTimeframeV1
    m5: AlignedTimeframeV1


@dataclass(frozen=True, slots=True)
class LoadedAlphaStrategyV1:
    symbol: str
    timeframe: str
    formula_tokens: tuple[int, ...]
    strategy_fingerprint: str
    dataset_fingerprint: str
    best_score: float


@dataclass(frozen=True, slots=True)
class AlphaObservationV1:
    symbol: str
    timeframe: str
    bar_close_timestamp: int
    strategy_fingerprint: str
    position: float
    strength: float
    factor_value: float
    bars_used: int

    def __post_init__(self) -> None:
        position = _finite_real(self.position, field="position")
        strength = _finite_real(self.strength, field="strength")
        _finite_real(self.factor_value, field="factor_value")
        if not -1.0 <= position <= 1.0:
            raise ValueError("position must be in [-1, 1]")
        if not 0.0 <= strength <= 1.0:
            raise ValueError("strength must be in [0, 1]")
        if not math.isclose(strength, abs(position), rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("strength must equal abs(position)")
        if type(self.bars_used) is not int or self.bars_used < 1:
            raise ValueError("bars_used must be a positive integer")


@dataclass(frozen=True, slots=True)
class PACandidateOccurrenceV1:
    candidate: StrategyCandidate
    observed_at: int
    is_new: bool

    def __post_init__(self) -> None:
        if type(self.candidate) is not StrategyCandidate:
            raise TypeError("candidate must be an exact StrategyCandidate")
        if type(self.observed_at) is not int:
            raise ValueError("observed_at must be an integer Unix second")
        if type(self.is_new) is not bool:
            raise ValueError("is_new must be bool")


@dataclass(frozen=True, slots=True)
class PAObservationV1:
    symbol: str
    timeframe: str
    bar_close_timestamp: int
    direction_score: float
    barbwire: bool
    extreme_range: bool
    atr: float
    current_open: float
    current_high: float
    current_low: float
    current_close: float
    tick: float
    supports: tuple[float, ...]
    resistances: tuple[float, ...]
    candidates: tuple[PACandidateOccurrenceV1, ...]

    def __post_init__(self) -> None:
        if self.timeframe not in {"H1", "M15", "M5"}:
            raise ValueError("PA timeframe must be one of H1, M15 or M5")
        direction = _finite_real(self.direction_score, field="direction_score")
        if not -1.0 <= direction <= 1.0:
            raise ValueError("direction_score must be in [-1, 1]")
        if _finite_real(self.atr, field="atr") <= 0.0:
            raise ValueError("atr must be positive")
        if _finite_real(self.tick, field="tick") <= 0.0:
            raise ValueError("tick must be positive")
        for field in ("current_open", "current_high", "current_low", "current_close"):
            _finite_real(getattr(self, field), field=field)
        if self.current_high < max(self.current_open, self.current_close, self.current_low):
            raise ValueError("current OHLC containment is invalid")
        if self.current_low > min(self.current_open, self.current_close, self.current_high):
            raise ValueError("current OHLC containment is invalid")
        if any(type(item) is not PACandidateOccurrenceV1 for item in self.candidates):
            raise TypeError("candidates must contain PACandidateOccurrenceV1 values")


@dataclass(frozen=True, slots=True)
class CandidateFreshnessStateV1:
    seen_keys: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if type(self.seen_keys) is not frozenset or any(
            type(item) is not str for item in self.seen_keys
        ):
            raise TypeError("seen_keys must be an exact frozenset of strings")


@dataclass(frozen=True, slots=True)
class ModuleSelectionV1:
    direction_modules: frozenset[str]
    entry_modules: frozenset[str]


@dataclass(frozen=True, slots=True)
class FusionWeightsV1:
    direction: tuple[tuple[str, float], ...]
    entry: tuple[tuple[str, float], ...]


@dataclass(frozen=True, slots=True)
class FusionInputsV1:
    symbol: str
    decision_close_timestamp: int
    h1_alpha: AlphaObservationV1 | None
    h1_pa: PAObservationV1 | None
    m15_alpha: AlphaObservationV1 | None
    m15_pa: PAObservationV1 | None
    m5_pa: PAObservationV1 | None
    m5_alpha: AlphaObservationV1 | None = None


@dataclass(frozen=True, slots=True)
class FusionSignalV1:
    accepted: bool
    side: str | None
    direction_score: float | None
    entry_score: float | None
    reject_reasons: tuple[str, ...]
    direction_weights: tuple[tuple[str, float], ...]
    entry_weights: tuple[tuple[str, float], ...]
    selected_m15_candidates: tuple[StrategyCandidate, ...]


@dataclass(frozen=True, slots=True)
class InstrumentConstraintsV1:
    tick: float
    min_stop_distance: float = 0.0
    min_pending_distance: float = 0.0
    supports_stop_limit: bool = False

    def __post_init__(self) -> None:
        if _finite_real(self.tick, field="tick") <= 0.0:
            raise ValueError("tick must be positive")
        if _finite_real(self.min_stop_distance, field="min_stop_distance") < 0.0:
            raise ValueError("min_stop_distance must be non-negative")
        if _finite_real(self.min_pending_distance, field="min_pending_distance") < 0.0:
            raise ValueError("min_pending_distance must be non-negative")
        if type(self.supports_stop_limit) is not bool:
            raise ValueError("supports_stop_limit must be bool")


@dataclass(frozen=True, slots=True)
class OrderPlanCandidateV1:
    plan_id: str
    style: str
    side: str
    family: str
    setup_type: str
    order_type: str
    entry_price: float
    trigger_price: float | None
    limit_price: float | None
    stop_loss: float
    take_profit_1: float
    take_profit_2: float | None
    risk_distance: float
    take_profit_1_r: float
    take_profit_2_r: float | None
    source_trigger_at: int
    source_policy: str
    policy_origin: str
    broker_validated: bool = False
    executable: bool = False

    def __post_init__(self) -> None:
        if self.style not in {"pa_primary", "confirmation_breakout", "pullback"}:
            raise ValueError("unknown order plan style")
        if self.side not in {"long", "short"}:
            raise ValueError("side must be long or short")
        if self.order_type not in {"market", "limit", "stop", "stop_limit"}:
            raise ValueError("unknown order type")
        for field in (
            "entry_price",
            "stop_loss",
            "take_profit_1",
            "risk_distance",
            "take_profit_1_r",
        ):
            _finite_real(getattr(self, field), field=field)
        for field in ("trigger_price", "limit_price", "take_profit_2", "take_profit_2_r"):
            value = getattr(self, field)
            if value is not None:
                _finite_real(value, field=field)
        if self.risk_distance <= 0.0 or self.take_profit_1_r <= 0.0:
            raise ValueError("risk and TP1 R must be positive")
        if self.executable is not False:
            raise ValueError("Stage 2 order plans cannot be executable")


@dataclass(frozen=True, slots=True)
class OrderPlanGenerationV1:
    plans: tuple[OrderPlanCandidateV1, ...]
    reject_reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AblationWindowV1:
    latest_decision_close: int
    cutoff_close: int
    preholdout_closes: tuple[int, ...]
    holdout_closes: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class AblationVariantV1:
    name: str
    selection: ModuleSelectionV1
