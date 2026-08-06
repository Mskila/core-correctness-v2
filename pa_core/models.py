# SPDX-License-Identifier: AGPL-3.0-or-later
"""Immutable data contracts for AlphaMaster's deterministic PA formula layer.

Parts of the formula semantics are derived from PA_Agent at commit
d92ecd827fe671a589b7fdfdbba41e5e98081d87 (AGPL-3.0-or-later), Copyright
(C) 2026 PA Agent Contributors.  AlphaMaster corrections and deterministic
formalizations are documented in ``docs/pa-formula-catalog.md``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class PABar:
    """One time-ascending, closed OHLC bar."""

    timestamp: int
    open: float
    high: float
    low: float
    close: float
    closed: bool = True
    segment_id: str = "0"

    @property
    def valid_ohlc(self) -> bool:
        values = (self.open, self.high, self.low, self.close)
        return bool(
            all(type(value) in {int, float} and math.isfinite(float(value)) for value in values)
            and float(self.high) >= max(float(self.open), float(self.close))
            and float(self.low) <= min(float(self.open), float(self.close))
        )


@dataclass(frozen=True, slots=True)
class BarSeries:
    """Validated input boundary for one symbol/timeframe/data segment."""

    symbol: str
    timeframe: str
    bars: tuple[PABar, ...]
    tick: float

    def __post_init__(self) -> None:
        if not isinstance(self.symbol, str) or not self.symbol.strip():
            raise ValueError("symbol must be non-empty")
        if not isinstance(self.timeframe, str) or not self.timeframe.strip():
            raise ValueError("timeframe must be non-empty")
        if not isinstance(self.bars, tuple) or any(type(bar) is not PABar for bar in self.bars):
            raise TypeError("bars must be an exact tuple of PABar values")
        if not isinstance(self.tick, (int, float)) or not math.isfinite(float(self.tick)):
            raise ValueError("tick must be finite")
        if float(self.tick) <= 0:
            raise ValueError("tick must be positive")
        if any(not bar.closed for bar in self.bars):
            raise ValueError("PA formulas accept closed bars only")
        if any(
            self.bars[index].timestamp <= self.bars[index - 1].timestamp
            for index in range(1, len(self.bars))
        ):
            raise ValueError("bar timestamps must be strictly increasing")
        segments = {bar.segment_id for bar in self.bars}
        if len(segments) > 1:
            raise ValueError("PA formulas accept one data segment at a time")


@dataclass(frozen=True, slots=True)
class FormulaResult:
    formula_id: str
    value: Any
    valid: bool
    as_of: int
    source_start: int
    source_end: int
    confirmed_at: int
    invalid_reason: str | None
    provenance_class: str


@dataclass(frozen=True, slots=True)
class BarSnapshot:
    index: int
    as_of: int
    results: tuple[FormulaResult, ...]

    def get(self, formula_id: str) -> FormulaResult:
        for result in self.results:
            if result.formula_id == formula_id:
                return result
        raise KeyError(formula_id)

    def value(self, formula_id: str) -> Any:
        return self.get(formula_id).value


@dataclass(frozen=True, slots=True)
class Pivot:
    index: int
    timestamp: int
    kind: str
    price: float
    radius: int
    confirmed_at: int


@dataclass(frozen=True, slots=True)
class SupportResistance:
    supports: tuple[float, ...]
    resistances: tuple[float, ...]
    fallback: bool


@dataclass(frozen=True, slots=True)
class RangeState:
    high: float | None
    low: float | None
    width_atr: float | None
    price_position: float | None
    zone: str
    lookback_bars: int
    trading_range: bool
    extreme: bool


@dataclass(frozen=True, slots=True)
class DirectionState:
    direction: str
    score: int
    components: tuple[int, int, int, int, int]
    strength: float


@dataclass(frozen=True, slots=True)
class AlwaysInState:
    state: str
    strength: str
    source_window: str
    above_ratio: float
    below_ratio: float


@dataclass(frozen=True, slots=True)
class ChannelState:
    label: str
    confirmed: bool
    group_count: int
    pullback_ratio: float | None
    parallel_error: float | None
    max_residual_atr: float | None
    direction: str | None = None
    width_at_as_of: float | None = None


@dataclass(frozen=True, slots=True)
class BarbwireState:
    score: float
    candidate: bool
    components: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BreakoutEvent:
    direction: str
    level: float
    breakout_at: int
    follow_through: bool
    retest_at: int | None = None
    failure_at: int | None = None
    failed_failure_at: int | None = None


@dataclass(frozen=True, slots=True)
class HLTrigger:
    label: str
    trigger_at: int
    pullback_pivot_at: int | None


@dataclass(frozen=True, slots=True)
class HLState:
    triggers: tuple[HLTrigger, ...]
    candidate: str
    wedge_check_required: bool


@dataclass(frozen=True, slots=True)
class EnvironmentState:
    labels: tuple[str, ...]
    spike: bool
    micro_channel: bool
    climax_triggered: bool


@dataclass(frozen=True, slots=True)
class PatternCandidate:
    pattern: str
    direction: str
    confirmed_at: int
    evidence: tuple[str, ...]
    follow_through: bool = False
    diagnostic_only: bool = False
    invalidation_price: float | None = None


@dataclass(frozen=True, slots=True)
class MeasuredMove:
    kind: str
    target_price: float
    height: float
    source_start: int
    source_end: int
    completed: bool


@dataclass(frozen=True, slots=True)
class StructureSnapshot:
    as_of: int
    range_state: RangeState
    swing_structure: str
    support_resistance: SupportResistance
    direction: DirectionState
    always_in: AlwaysInState
    channel: ChannelState
    barbwire: BarbwireState
    breakout_events: tuple[BreakoutEvent, ...]
    hl_state: HLState
    environment: EnvironmentState
    patterns: tuple[PatternCandidate, ...]
    measured_moves: tuple[MeasuredMove, ...]
    results: tuple[FormulaResult, ...] = ()

    def get(self, formula_id: str) -> FormulaResult:
        for result in self.results:
            if result.formula_id == formula_id:
                return result
        raise KeyError(formula_id)


@dataclass(frozen=True, slots=True)
class StrategyCandidate:
    family: str
    direction: str
    setup_type: str
    trigger_at: int
    evidence: tuple[str, ...]
    invalidation_anchor: float
    target_anchors: tuple[float, ...]
    source_policy: str
    chase_forbidden: bool
    executable: bool = False
