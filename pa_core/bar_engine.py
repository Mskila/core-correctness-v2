# SPDX-License-Identifier: AGPL-3.0-or-later
"""Deterministic bar geometry plus ATR14/EMA20 calculations.

Derived in part from PA_Agent commit
d92ecd827fe671a589b7fdfdbba41e5e98081d87 (AGPL-3.0-or-later), Copyright
(C) 2026 PA Agent Contributors.  Time direction and EMA-gap names follow the
AlphaMaster corrections frozen in ``docs/pa-formula-catalog.md``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .models import BarSeries, BarSnapshot, FormulaResult, PABar


_FORMULA_PROVENANCE: tuple[tuple[str, str], ...] = (
    ("bar.range", "D0"),
    ("bar.body", "D0"),
    ("bar.body_ratio", "D0"),
    ("bar.upper_wick_ratio", "D0"),
    ("bar.lower_wick_ratio", "D0"),
    ("bar.close_position", "D0"),
    ("bar.direction", "D0"),
    ("bar.range_atr_ratio", "D0"),
    ("indicator.atr14", "D0"),
    ("indicator.ema20", "D0"),
    ("bar.type", "D0"),
    ("bar.overlap_prev", "D0"),
    ("bar.inside_sequence", "D0"),
    ("bar.ioi", "D0"),
    ("bar.micro_double", "D0"),
    ("bar.ema_relation", "D0"),
    ("bar.ema_gap_side", "D0-C"),
    ("bar.ema_gap_run", "D0-C"),
    ("bar.twenty_gap_bars", "D0-C"),
    ("bar.interbar_gap", "D1"),
    ("bar.breakout_prev_5", "D0"),
    ("bar.follow_through_1_2", "D0"),
)


@dataclass(frozen=True, slots=True)
class EmaState:
    period: int
    count: int
    value: float | None
    total: float

    @classmethod
    def initial(cls, period: int = 20) -> "EmaState":
        if period < 1:
            raise ValueError("period must be positive")
        return cls(period=period, count=0, value=None, total=0.0)

    def update(self, value: float) -> "EmaState":
        if not math.isfinite(float(value)):
            return type(self).initial(self.period)
        count = self.count + 1
        if count < self.period:
            return type(self)(self.period, count, None, self.total + float(value))
        if count == self.period:
            seed = (self.total + float(value)) / self.period
            return type(self)(self.period, count, seed, 0.0)
        if self.value is None:
            raise RuntimeError("EMA state lost its seeded value")
        alpha = 2.0 / (self.period + 1)
        updated = float(value) * alpha + self.value * (1.0 - alpha)
        return type(self)(self.period, count, updated, 0.0)


@dataclass(frozen=True, slots=True)
class AtrState:
    period: int
    count: int
    value: float | None
    previous_close: float | None
    total_tr: float

    @classmethod
    def initial(cls, period: int = 14) -> "AtrState":
        if period < 1:
            raise ValueError("period must be positive")
        return cls(period=period, count=0, value=None, previous_close=None, total_tr=0.0)

    def update(self, high: float, low: float, close: float) -> "AtrState":
        values = (high, low, close)
        if not all(math.isfinite(float(value)) for value in values) or high < low:
            return type(self).initial(self.period)
        tr = float(high) - float(low)
        if self.previous_close is not None:
            tr = max(
                tr,
                abs(float(high) - self.previous_close),
                abs(float(low) - self.previous_close),
            )
        count = self.count + 1
        if count < self.period:
            return type(self)(self.period, count, None, float(close), self.total_tr + tr)
        if count == self.period:
            seed = (self.total_tr + tr) / self.period
            return type(self)(self.period, count, seed, float(close), 0.0)
        if self.value is None:
            raise RuntimeError("ATR state lost its seeded value")
        updated = ((self.period - 1) * self.value + tr) / self.period
        return type(self)(self.period, count, updated, float(close), 0.0)


def _bar_valid(bar: PABar) -> bool:
    return bar.valid_ohlc


def _is_inside(current: PABar, previous: PABar) -> bool:
    return current.high <= previous.high and current.low >= previous.low


def _is_outside(current: PABar, previous: PABar) -> bool:
    return current.high >= previous.high and current.low <= previous.low


class BarFormulaEngine:
    """Pure, prefix-causal formulas for one validated :class:`BarSeries`."""

    def __init__(self, series: BarSeries):
        if type(series) is not BarSeries:
            raise TypeError("series must be an exact BarSeries")
        self.series = series
        self.atr14, self.ema20 = self._indicators()

    def _indicators(self) -> tuple[tuple[float | None, ...], tuple[float | None, ...]]:
        atr_state = AtrState.initial(14)
        ema_state = EmaState.initial(20)
        atr: list[float | None] = []
        ema: list[float | None] = []
        for bar in self.series.bars:
            if not _bar_valid(bar):
                atr_state = AtrState.initial(14)
                ema_state = EmaState.initial(20)
                atr.append(None)
                ema.append(None)
                continue
            atr_state = atr_state.update(bar.high, bar.low, bar.close)
            ema_state = ema_state.update(bar.close)
            atr.append(atr_state.value)
            ema.append(ema_state.value)
        return tuple(atr), tuple(ema)

    def evaluate(self, index: int, *, as_of_index: int | None = None) -> BarSnapshot:
        bars = self.series.bars
        if not bars:
            raise IndexError("cannot evaluate an empty series")
        if not 0 <= index < len(bars):
            raise IndexError(index)
        as_of_index = index if as_of_index is None else as_of_index
        if not index <= as_of_index < len(bars):
            raise ValueError("as_of_index must include index and remain inside the series")
        bar = bars[index]
        if not _bar_valid(bar):
            results = tuple(
                self._result(
                    formula_id,
                    None,
                    index,
                    index,
                    valid=False,
                    invalid_reason="invalid_ohlc",
                    provenance=provenance,
                )
                for formula_id, provenance in _FORMULA_PROVENANCE
            )
            return BarSnapshot(index=index, as_of=bar.timestamp, results=results)

        full_range = float(bar.high) - float(bar.low)
        body = abs(float(bar.close) - float(bar.open))
        ratio_valid = full_range > 0
        body_ratio = body / full_range if ratio_valid else None
        upper = (
            (float(bar.high) - max(float(bar.open), float(bar.close))) / full_range
            if ratio_valid
            else None
        )
        lower = (
            (min(float(bar.open), float(bar.close)) - float(bar.low)) / full_range
            if ratio_valid
            else None
        )
        close_position = (float(bar.close) - float(bar.low)) / full_range if ratio_valid else None
        direction = "bull" if bar.close > bar.open else "bear" if bar.close < bar.open else "flat"
        atr = self.atr14[index]
        range_atr = full_range / atr if atr is not None and atr > 0 else None

        values: list[FormulaResult] = [
            self._result("bar.range", full_range, index, index),
            self._result("bar.body", body, index, index),
            self._ratio_result("bar.body_ratio", body_ratio, index),
            self._ratio_result("bar.upper_wick_ratio", upper, index),
            self._ratio_result("bar.lower_wick_ratio", lower, index),
            self._ratio_result("bar.close_position", close_position, index),
            self._result("bar.direction", direction, index, index),
            self._optional_result(
                "bar.range_atr_ratio", range_atr, index, index, "atr_unavailable"
            ),
            self._optional_result("indicator.atr14", atr, index, index, "atr_warmup"),
            self._optional_result("indicator.ema20", self.ema20[index], index, index, "ema_warmup"),
        ]

        bar_type = self._bar_type(index, body_ratio, close_position)
        values.append(self._result("bar.type", bar_type, max(0, index - 1), index))
        values.append(self._overlap_result(index))
        values.append(
            self._result(
                "bar.inside_sequence", self._inside_sequence(index), max(0, index - 3), index
            )
        )
        values.append(self._result("bar.ioi", self._ioi(index), max(0, index - 3), index))
        values.append(
            self._result("bar.micro_double", self._micro_double(index), max(0, index - 1), index)
        )
        ema_relation, ema_gap = self._ema_relation(index)
        values.append(
            self._optional_result("bar.ema_relation", ema_relation, index, index, "ema_warmup")
        )
        values.append(
            self._optional_result("bar.ema_gap_side", ema_gap, index, index, "ema_warmup")
        )
        run = self._ema_gap_run(index)
        gap_start = index - run + 1 if run else index
        values.append(self._result("bar.ema_gap_run", run, gap_start, index, provenance="D0-C"))
        values.append(
            self._result("bar.twenty_gap_bars", run >= 20, gap_start, index, provenance="D0-C")
        )
        values.append(
            self._result(
                "bar.interbar_gap",
                self._interbar_gap(index),
                max(0, index - 1),
                index,
                provenance="D1",
            )
        )
        values.append(self._breakout_prev_result(index))
        values.append(self._follow_result(index, as_of_index))
        return BarSnapshot(index=index, as_of=bar.timestamp, results=tuple(values))

    def _result(
        self,
        formula_id: str,
        value: object,
        source_start_index: int,
        source_end_index: int,
        *,
        confirmed_index: int | None = None,
        valid: bool = True,
        invalid_reason: str | None = None,
        provenance: str = "D0",
    ) -> FormulaResult:
        bars = self.series.bars
        confirmed_index = source_end_index if confirmed_index is None else confirmed_index
        return FormulaResult(
            formula_id=formula_id,
            value=value,
            valid=valid,
            as_of=bars[confirmed_index].timestamp,
            source_start=bars[source_start_index].timestamp,
            source_end=bars[source_end_index].timestamp,
            confirmed_at=bars[confirmed_index].timestamp,
            invalid_reason=invalid_reason,
            provenance_class=provenance,
        )

    def _optional_result(
        self,
        formula_id: str,
        value: object | None,
        source_start: int,
        source_end: int,
        reason: str,
        *,
        provenance: str = "D0",
    ) -> FormulaResult:
        return self._result(
            formula_id,
            value,
            source_start,
            source_end,
            valid=value is not None,
            invalid_reason=None if value is not None else reason,
            provenance=provenance,
        )

    def _ratio_result(self, formula_id: str, value: float | None, index: int) -> FormulaResult:
        return self._optional_result(formula_id, value, index, index, "zero_range")

    def _bar_type(
        self,
        index: int,
        body_ratio: float | None,
        close_position: float | None,
    ) -> str:
        bar = self.series.bars[index]
        if index > 0:
            previous = self.series.bars[index - 1]
            if _is_inside(bar, previous):
                return "inside"
            if _is_outside(bar, previous):
                return "outside_bull" if bar.close >= bar.open else "outside_bear"
        if body_ratio is None or close_position is None:
            return "flat"
        if body_ratio <= 0.25:
            return "doji"
        if bar.close > bar.open and close_position >= 0.65:
            return "trend_bull"
        if bar.close < bar.open and close_position <= 0.35:
            return "trend_bear"
        return "other"

    def _overlap_result(self, index: int) -> FormulaResult:
        if index == 0:
            return self._optional_result("bar.overlap_prev", None, index, index, "no_previous_bar")
        current = self.series.bars[index]
        previous = self.series.bars[index - 1]
        shared = max(0.0, min(current.high, previous.high) - max(current.low, previous.low))
        union = max(current.high, previous.high) - min(current.low, previous.low)
        value = shared / union if union > 0 else None
        return self._optional_result("bar.overlap_prev", value, index - 1, index, "zero_union")

    def _inside_sequence(self, index: int) -> str:
        bars = self.series.bars
        if index >= 3 and all(
            _is_inside(bars[j], bars[j - 1]) for j in range(index - 2, index + 1)
        ):
            return "iii"
        if index >= 2 and all(
            _is_inside(bars[j], bars[j - 1]) for j in range(index - 1, index + 1)
        ):
            return "ii"
        return "none"

    def _ioi(self, index: int) -> bool:
        if index < 3:
            return False
        bars = self.series.bars
        return bool(
            _is_inside(bars[index - 2], bars[index - 3])
            and _is_outside(bars[index - 1], bars[index - 2])
            and _is_inside(bars[index], bars[index - 1])
        )

    def _micro_double(self, index: int) -> str:
        if index == 0:
            return "none"
        bar = self.series.bars[index]
        previous = self.series.bars[index - 1]
        atr = self.atr14[index]
        tolerance = 0.02 * atr if atr is not None and atr > 0 else 0.0
        if abs(bar.low - previous.low) <= tolerance:
            return "MDB"
        if abs(bar.high - previous.high) <= tolerance:
            return "MDT"
        return "none"

    def _ema_relation(self, index: int) -> tuple[str | None, str | None]:
        ema = self.ema20[index]
        if ema is None:
            return None, None
        bar = self.series.bars[index]
        relation = "above" if bar.close > ema else "below" if bar.close < ema else "touch"
        gap = "above" if bar.low > ema else "below" if bar.high < ema else "none"
        return relation, gap

    def _ema_gap_run(self, index: int) -> int:
        _, side = self._ema_relation(index)
        if side not in {"above", "below"}:
            return 0
        count = 0
        for cursor in range(index, -1, -1):
            _, cursor_side = self._ema_relation(cursor)
            if cursor_side != side:
                break
            count += 1
        return count

    def _interbar_gap(self, index: int) -> str:
        if index == 0:
            return "none"
        bar = self.series.bars[index]
        previous = self.series.bars[index - 1]
        if bar.low > previous.high + self.series.tick:
            return "up"
        if bar.high < previous.low - self.series.tick:
            return "down"
        return "none"

    def _breakout_prev_result(self, index: int) -> FormulaResult:
        if index < 5:
            return self._result(
                "bar.breakout_prev_5",
                "none",
                0,
                index,
                valid=False,
                invalid_reason="window_lt_5",
            )
        bar = self.series.bars[index]
        previous = self.series.bars[index - 5 : index]
        broke_high = bar.high > max(candidate.high for candidate in previous)
        broke_low = bar.low < min(candidate.low for candidate in previous)
        value = (
            "both"
            if broke_high and broke_low
            else "up"
            if broke_high
            else "down"
            if broke_low
            else "none"
        )
        return self._result("bar.breakout_prev_5", value, index - 5, index)

    def _follow_result(self, index: int, as_of_index: int) -> FormulaResult:
        bar = self.series.bars[index]
        direction = 1 if bar.close > bar.open else -1 if bar.close < bar.open else 0
        available_end = min(as_of_index, index + 2)
        if direction == 0:
            return self._result(
                "bar.follow_through_1_2",
                "pending",
                index,
                available_end,
                confirmed_index=available_end,
            )
        later = self.series.bars[index + 1 : available_end + 1]
        if not later:
            value = "pending"
        elif direction > 0 and any(candidate.close > bar.close for candidate in later):
            value = "yes"
        elif direction < 0 and any(candidate.close < bar.close for candidate in later):
            value = "yes"
        elif direction > 0 and any(candidate.close < bar.open for candidate in later):
            value = "failed"
        elif direction < 0 and any(candidate.close > bar.open for candidate in later):
            value = "failed"
        elif available_end < index + 2:
            value = "pending"
        else:
            value = "no"
        return self._result(
            "bar.follow_through_1_2",
            value,
            index,
            available_end,
            confirmed_index=available_end,
        )
