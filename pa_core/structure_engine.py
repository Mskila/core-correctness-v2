# SPDX-License-Identifier: AGPL-3.0-or-later
"""Deterministic market-structure and Price Action pattern formulas.

Direct formulas are derived from PA_Agent commit
d92ecd827fe671a589b7fdfdbba41e5e98081d87 (AGPL-3.0-or-later), Copyright
(C) 2026 PA Agent Contributors. Corrected time semantics (D0-C) and newly
formalized playbook rules (D1) follow ``docs/pa-formula-catalog.md``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from .bar_engine import BarFormulaEngine
from .models import (
    AlwaysInState,
    BarbwireState,
    BarSeries,
    BreakoutEvent,
    ChannelState,
    DirectionState,
    EnvironmentState,
    FormulaResult,
    HLState,
    HLTrigger,
    MeasuredMove,
    PatternCandidate,
    Pivot,
    RangeState,
    StructureSnapshot,
    SupportResistance,
)


def score_barbwire(
    *,
    overlap_mean_10: float | None,
    doji_inside_ratio_10: float | None,
    range_width_atr: float | None,
    width_to_avg_range: float | None,
) -> BarbwireState:
    """Apply the four frozen PA_Agent barbwire score components."""
    score = 0.0
    components: list[str] = []
    if overlap_mean_10 is not None and overlap_mean_10 >= 0.65:
        score += 0.4
        components.append("overlap")
    if doji_inside_ratio_10 is not None and doji_inside_ratio_10 >= 0.40:
        score += 0.2
        components.append("doji_inside")
    if range_width_atr is not None and range_width_atr <= 3.0:
        score += 0.2
        components.append("range_width")
    if width_to_avg_range is not None and width_to_avg_range < 0.30:
        score += 0.2
        components.append("compressed_width")
    score = round(score, 10)
    return BarbwireState(score=score, candidate=score >= 0.60, components=tuple(components))


def classify_range_flags(
    *,
    high_tests: int,
    low_tests: int,
    direction: str,
    swing_structure: str,
    ema_slope_atr: float | None,
    overlap_mean: float | None,
    direction_score: int,
) -> tuple[bool, bool]:
    """Return ordinary-range and extreme-range flags from frozen thresholds."""
    trading_range = (
        high_tests >= 2
        and low_tests >= 2
        and (direction == "neutral" or swing_structure == "mixed")
    )
    extreme = bool(
        ema_slope_atr is not None
        and overlap_mean is not None
        and abs(ema_slope_atr) <= 0.05
        and overlap_mean >= 0.70
        and abs(direction_score) <= 1
    )
    return trading_range, extreme


def classify_channel_metrics(
    *,
    group_count: int,
    pullback_ratio: float | None,
    parallel_error: float | None,
    max_residual_atr: float | None,
    direction: str | None = None,
    width_at_as_of: float | None = None,
) -> ChannelState:
    """Classify a causal channel from already-computed objective metrics."""
    stable = bool(
        parallel_error is not None
        and max_residual_atr is not None
        and parallel_error <= 0.35
        and max_residual_atr <= 0.75
    )
    label = "none"
    confirmed = False
    if group_count >= 2:
        label = "trending_tr"
    if group_count >= 3 and stable and pullback_ratio is not None:
        if pullback_ratio < 0.30:
            label = "tight_channel"
            confirmed = True
        elif pullback_ratio <= 0.50:
            label = "normal_channel"
            confirmed = True
        elif pullback_ratio <= 0.786:
            label = "broad_channel"
            confirmed = True
    return ChannelState(
        label=label,
        confirmed=confirmed,
        group_count=group_count,
        pullback_ratio=pullback_ratio,
        parallel_error=parallel_error,
        max_residual_atr=max_residual_atr,
        direction=direction,
        width_at_as_of=width_at_as_of,
    )


def _regression(points: Sequence[tuple[int, float]]) -> tuple[float, float, float] | None:
    if len(points) < 2:
        return None
    mean_x = sum(point[0] for point in points) / len(points)
    mean_y = sum(point[1] for point in points) / len(points)
    denominator = sum((point[0] - mean_x) ** 2 for point in points)
    if denominator <= 0:
        return None
    slope = sum((x - mean_x) * (y - mean_y) for x, y in points) / denominator
    intercept = mean_y - slope * mean_x
    residual = max(abs(y - (slope * x + intercept)) for x, y in points)
    return slope, intercept, residual


def _coerce_wedge_pivots(
    pivots: Sequence[Pivot | tuple[int, str, float]],
    *,
    base_timestamp: int,
    step_seconds: int,
) -> tuple[Pivot, ...]:
    coerced: list[Pivot] = []
    for value in pivots:
        if isinstance(value, Pivot):
            coerced.append(value)
            continue
        index, kind, price = value
        timestamp = base_timestamp + int(index) * step_seconds
        coerced.append(Pivot(int(index), timestamp, str(kind), float(price), 1, timestamp))
    return tuple(sorted(coerced, key=lambda pivot: (pivot.index, pivot.kind)))


def completed_pullback_ratio(
    pivots: Sequence[Pivot],
    *,
    direction: str,
    tick: float,
) -> float | None:
    """Use the most recent completed causal leg, never the oldest sorted pivot."""
    if direction not in {"bullish", "bearish"}:
        raise ValueError("direction must be bullish or bearish")
    if tick <= 0:
        raise ValueError("tick must be positive")
    for end in range(len(pivots) - 1, 1, -1):
        first, second, third = pivots[end - 2 : end + 1]
        kinds = (first.kind, second.kind, third.kind)
        if direction == "bullish" and kinds == ("low", "high", "low"):
            denominator = second.price - first.price
            return (second.price - third.price) / denominator if denominator > tick else None
        if direction == "bearish" and kinds == ("high", "low", "high"):
            denominator = first.price - second.price
            return (third.price - second.price) / denominator if denominator > tick else None
    return None


def project_measured_move(
    *,
    kind: str,
    height: float,
    anchor_price: float,
    direction: str,
    source_start: int,
    source_end: int,
) -> MeasuredMove:
    """Project one already-proven positive height from its causal anchor."""
    if direction not in {"up", "down"}:
        raise ValueError("direction must be up or down")
    if not math.isfinite(height) or height <= 0:
        raise ValueError("height must be finite and positive")
    if source_end < source_start:
        raise ValueError("measured-move source must be time ordered")
    target = anchor_price + height if direction == "up" else anchor_price - height
    return MeasuredMove(kind, target, height, source_start, source_end, True)


def detect_wedge_candidates(
    pivots: Sequence[Pivot | tuple[int, str, float]],
    *,
    atr: float,
    tick: float,
    base_timestamp: int = 0,
    step_seconds: int = 1,
    trend_direction: str = "neutral",
) -> tuple[PatternCandidate, ...]:
    """Detect the frozen three-push converging wedge candidate."""
    if not math.isfinite(atr) or atr <= 0 or tick <= 0:
        return ()
    ordered = _coerce_wedge_pivots(
        pivots,
        base_timestamp=base_timestamp,
        step_seconds=step_seconds,
    )
    outputs: list[PatternCandidate] = []
    for push_kind, direction, reverse_direction in (
        ("high", "bullish", "bearish"),
        ("low", "bearish", "bullish"),
    ):
        pushes = [pivot for pivot in ordered if pivot.kind == push_kind]
        if len(pushes) < 3:
            continue
        p1, p2, p3 = pushes[-3:]
        origins = [pivot for pivot in ordered if pivot.kind != push_kind and pivot.index < p1.index]
        opposite = [pivot for pivot in ordered if pivot.kind != push_kind]
        if not origins or len(opposite) < 3:
            continue
        p0 = origins[-1]
        if direction == "bullish":
            advances = (p1.price - p0.price, p2.price - p1.price, p3.price - p2.price)
            same_direction = all(
                newer.price > older.price for older, newer in zip(opposite[-3:], opposite[-2:])
            )
        else:
            advances = (p0.price - p1.price, p1.price - p2.price, p2.price - p3.price)
            same_direction = all(
                newer.price < older.price for older, newer in zip(opposite[-3:], opposite[-2:])
            )
        a1, a2, a3 = advances
        span = p3.index - p0.index
        if not (
            min(advances) > max(tick, 1e-12)
            and a2 <= 1.05 * a1
            and a3 <= 1.05 * a2
            and a3 / a1 <= 0.80
            and same_direction
            and 10 <= span <= 40
        ):
            continue
        high_points = [(pivot.index, pivot.price) for pivot in ordered if pivot.kind == "high"][-3:]
        low_points = [(pivot.index, pivot.price) for pivot in ordered if pivot.kind == "low"][-3:]
        high_line = _regression(high_points)
        low_line = _regression(low_points)
        if high_line is None or low_line is None:
            continue
        slope_high, intercept_high, residual_high = high_line
        slope_low, intercept_low, residual_low = low_line
        converges = (
            slope_low > slope_high > 0 if direction == "bullish" else slope_high < slope_low < 0
        )
        start = min(point[0] for point in (*high_points, *low_points))
        end = max(point[0] for point in (*high_points, *low_points))
        start_width = abs(
            (slope_high * start + intercept_high) - (slope_low * start + intercept_low)
        )
        end_width = abs((slope_high * end + intercept_high) - (slope_low * end + intercept_low))
        if not (
            converges and end_width < start_width and max(residual_high, residual_low) <= 0.75 * atr
        ):
            continue
        if span < 20:
            pattern = "wedge_pullback" if trend_direction != direction else "micro_wedge"
        else:
            pattern = "wedge_reversal" if trend_direction == direction else "wedge_pullback"
        outputs.append(
            PatternCandidate(
                pattern=pattern,
                direction=reverse_direction,
                confirmed_at=p3.confirmed_at,
                evidence=tuple(f"{pivot.kind}@{pivot.timestamp}" for pivot in (p0, p1, p2, p3)),
                diagnostic_only=pattern == "wedge_reversal",
                invalidation_price=p3.price,
            )
        )
    return tuple(outputs)


def evaluate_mtr(
    *,
    direction: str,
    swing_structure: str,
    trendline_break: bool,
    recovery_failed: bool,
    extreme_retest_failed: bool,
    confirmed_at: int,
) -> PatternCandidate:
    """Turn the four explicit MTR components into a diagnostic candidate."""
    original_trend = bool(
        (direction == "bullish" and swing_structure == "HH_HL")
        or (direction == "bearish" and swing_structure == "LL_LH")
    )
    components = (
        ("trend", original_trend),
        ("trendline_break", trendline_break),
        ("recovery_failure", recovery_failed),
        ("extreme_retest", extreme_retest_failed),
    )
    complete = all(value for _, value in components)
    output_direction = "bearish" if direction == "bullish" else "bullish"
    return PatternCandidate(
        pattern="mtr" if complete else "reversal_attempt",
        direction=output_direction,
        confirmed_at=confirmed_at,
        evidence=tuple(name for name, value in components if value),
        diagnostic_only=True,
    )


def evaluate_final_flag(
    *,
    direction_score: int,
    had_always_in_or_spike: bool,
    consolidation_bars: int,
    ema_slope_atr: float,
    overlap_mean: float,
    doji_inside_ratio: float,
    target_distance_atr: float,
    breakout_observed: bool,
    breakout_follow: bool,
    returned_to_range: bool,
    confirmed_at: int,
) -> PatternCandidate | None:
    """Evaluate the deterministic Final Flag gates at their exact boundaries."""
    if not (
        abs(direction_score) >= 3
        and had_always_in_or_spike
        and consolidation_bars >= 10
        and abs(ema_slope_atr) <= 0.10
        and overlap_mean >= 0.50
        and doji_inside_ratio >= 0.40
        and target_distance_atr <= 0.50
    ):
        return None
    if breakout_observed and breakout_follow:
        return None
    original = "bullish" if direction_score > 0 else "bearish"
    opposite = "bearish" if original == "bullish" else "bullish"
    failed = breakout_observed and not breakout_follow and returned_to_range
    return PatternCandidate(
        pattern="failed_final_flag" if failed else "final_flag",
        direction=opposite if failed else original,
        confirmed_at=confirmed_at,
        evidence=("prior_trend", "consolidation", "near_target"),
        follow_through=breakout_follow,
        diagnostic_only=True,
    )


class StructureFormulaEngine:
    """Pure causal structure formulas over a single :class:`BarSeries`."""

    def __init__(self, series: BarSeries):
        if type(series) is not BarSeries:
            raise TypeError("series must be an exact BarSeries")
        self.series = series
        self.bar_engine = BarFormulaEngine(series)

    def _as_of(self, as_of_index: int | None) -> int:
        if not self.series.bars:
            raise IndexError("cannot evaluate an empty series")
        resolved = len(self.series.bars) - 1 if as_of_index is None else as_of_index
        if not 0 <= resolved < len(self.series.bars):
            raise IndexError(resolved)
        return resolved

    def pivots(self, radius: int, *, as_of_index: int | None = None) -> tuple[Pivot, ...]:
        if radius not in {1, 2}:
            raise ValueError("pivot radius must be 1 or 2")
        as_of = self._as_of(as_of_index)
        bars = self.series.bars
        raw_by_index: dict[int, list[Pivot]] = {}
        for index in range(radius, as_of - radius + 1):
            window = bars[index - radius : index + radius + 1]
            if not all(bar.valid_ohlc for bar in window):
                continue
            bar = bars[index]
            neighbours = [*bars[index - radius : index], *bars[index + 1 : index + radius + 1]]
            candidates: list[Pivot] = []
            if all(bar.high > other.high for other in neighbours):
                candidates.append(
                    Pivot(
                        index,
                        bar.timestamp,
                        "high",
                        float(bar.high),
                        radius,
                        bars[index + radius].timestamp,
                    )
                )
            if all(bar.low < other.low for other in neighbours):
                candidates.append(
                    Pivot(
                        index,
                        bar.timestamp,
                        "low",
                        float(bar.low),
                        radius,
                        bars[index + radius].timestamp,
                    )
                )
            if candidates:
                raw_by_index[index] = candidates

        alternating: list[Pivot] = []
        for index in sorted(raw_by_index):
            candidates = raw_by_index[index]
            if len(candidates) == 2:
                if alternating:
                    wanted = "low" if alternating[-1].kind == "high" else "high"
                    candidate = next(item for item in candidates if item.kind == wanted)
                else:
                    bar = bars[index]
                    neighbours = [
                        *bars[index - radius : index],
                        *bars[index + 1 : index + radius + 1],
                    ]
                    high_prominence = bar.high - max(other.high for other in neighbours)
                    low_prominence = min(other.low for other in neighbours) - bar.low
                    wanted = "high" if high_prominence >= low_prominence else "low"
                    candidate = next(item for item in candidates if item.kind == wanted)
            else:
                candidate = candidates[0]
            if alternating and alternating[-1].kind == candidate.kind:
                previous = alternating[-1]
                more_extreme = (
                    candidate.price > previous.price
                    if candidate.kind == "high"
                    else candidate.price < previous.price
                )
                if more_extreme:
                    alternating[-1] = candidate
            else:
                alternating.append(candidate)
        return tuple(alternating)

    def swing_structure(self, *, radius: int = 2, as_of_index: int | None = None) -> str:
        pivots = self.pivots(radius, as_of_index=as_of_index)
        highs = [pivot.price for pivot in pivots if pivot.kind == "high"]
        lows = [pivot.price for pivot in pivots if pivot.kind == "low"]
        if len(highs) < 2 or len(lows) < 2:
            return "insufficient"
        hh = highs[-1] > highs[-2]
        lh = highs[-1] < highs[-2]
        hl = lows[-1] > lows[-2]
        ll = lows[-1] < lows[-2]
        if hh and hl:
            return "HH_HL"
        if ll and lh:
            return "LL_LH"
        return "mixed"

    def support_resistance(self, *, as_of_index: int | None = None) -> SupportResistance:
        as_of = self._as_of(as_of_index)
        start = max(0, as_of - 39)
        close = self.series.bars[as_of].close
        pivots = [pivot for pivot in self.pivots(1, as_of_index=as_of) if pivot.index >= start]
        supports = sorted(
            {
                round(pivot.price, 8)
                for pivot in pivots
                if pivot.kind == "low" and pivot.price < close
            },
            reverse=True,
        )
        resistances = sorted(
            {
                round(pivot.price, 8)
                for pivot in pivots
                if pivot.kind == "high" and pivot.price > close
            }
        )
        fallback = False
        if not supports:
            fallback = True
            supports = sorted(
                {
                    round(bar.low, 8)
                    for bar in self.series.bars[start : as_of + 1]
                    if bar.low < close
                },
                reverse=True,
            )
        if not resistances:
            fallback = True
            resistances = sorted(
                {
                    round(bar.high, 8)
                    for bar in self.series.bars[start : as_of + 1]
                    if bar.high > close
                }
            )
        return SupportResistance(tuple(supports[:3]), tuple(resistances[:3]), fallback)

    def _weighted_gravity(self, start: int, end: int) -> int:
        closes = [bar.close for bar in self.series.bars[start : end + 1]]
        count = len(closes)
        half = count // 2
        if half < 1 or count < 2 * half:
            return 0
        newest_first = list(reversed(closes))

        def weighted(values: Sequence[float], offset: int) -> float:
            weights = [count - (offset + index) for index in range(len(values))]
            return sum(weight * value for weight, value in zip(weights, values)) / sum(weights)

        near = weighted(newest_first[:half], 0)
        far = weighted(newest_first[half : 2 * half], half)
        atr = self.bar_engine.atr14[end]
        threshold = 0.10 * atr if atr is not None else 0.0
        return 1 if near - far > threshold else -1 if near - far < -threshold else 0

    def _mean_overlap(self, start: int, end: int) -> float | None:
        values: list[float] = []
        for index in range(max(start + 1, 1), end + 1):
            result = self.bar_engine.evaluate(index).get("bar.overlap_prev")
            if result.valid:
                values.append(float(result.value))
        return sum(values) / len(values) if values else None

    def direction(self, *, as_of_index: int | None = None) -> DirectionState:
        as_of = self._as_of(as_of_index)
        start = max(0, as_of - 7)
        atr = self.bar_engine.atr14[as_of]
        ema = self.bar_engine.ema20
        s1 = 0
        current_ema = ema[as_of]
        if current_ema is not None:
            lookback = min(10, as_of)
            older = ema[as_of - lookback] if lookback else None
            if older is not None:
                difference = current_ema - older
                threshold = 0.05 * atr if atr is not None else 0.0
                s1 = 1 if difference > threshold else -1 if difference < -threshold else 0
        s2 = self._weighted_gravity(start, as_of)
        pivots = [pivot for pivot in self.pivots(2, as_of_index=as_of) if pivot.index >= start]
        highs = [pivot.price for pivot in pivots if pivot.kind == "high"]
        lows = [pivot.price for pivot in pivots if pivot.kind == "low"]
        s3 = 0
        if len(highs) >= 2 and len(lows) >= 2:
            if highs[-1] > highs[-2] and lows[-1] > lows[-2]:
                s3 = 1
            elif highs[-1] < highs[-2] and lows[-1] < lows[-2]:
                s3 = -1
        types = [
            self.bar_engine.evaluate(index).value("bar.type") for index in range(start, as_of + 1)
        ]
        bulls = types.count("trend_bull")
        bears = types.count("trend_bear")
        if bulls and not bears:
            s4 = 1
        elif bears and not bulls:
            s4 = -1
        elif bears and bulls and bulls >= 1.5 * bears:
            s4 = 1
        elif bears and bulls and bears >= 1.5 * bulls:
            s4 = -1
        else:
            s4 = 0
        overlap = self._mean_overlap(start, as_of)
        s5 = s1 if overlap is not None and overlap < 0.45 else 0
        score = s1 + s2 + s3 + s4 + s5
        medium_start = max(0, as_of - 19)
        medium = self._weighted_gravity(medium_start, as_of)
        if medium and score and medium != (1 if score > 0 else -1) and abs(score) < 4:
            score -= 1 if score > 0 else -1
        direction = "bullish" if score >= 3 else "bearish" if score <= -3 else "neutral"
        return DirectionState(direction, score, (s1, s2, s3, s4, s5), abs(score) / 5)

    def _weighted_ema_side(self, as_of: int, window: int) -> tuple[float, float] | None:
        if as_of + 1 < window:
            return None
        start = as_of - window + 1
        total = 0.0
        above = 0.0
        below = 0.0
        for offset, index in enumerate(range(as_of, start - 1, -1)):
            ema = self.bar_engine.ema20[index]
            if ema is None:
                continue
            weight = float(window - offset)
            total += weight
            close = self.series.bars[index].close
            above += weight if close > ema else 0.0
            below += weight if close < ema else 0.0
        if total <= 0:
            return None
        return above / total, below / total

    def _ema_slope(self, as_of: int, lookback: int) -> int:
        if as_of < lookback:
            return 0
        current = self.bar_engine.ema20[as_of]
        previous = self.bar_engine.ema20[as_of - lookback]
        if current is None or previous is None:
            return 0
        atr = self.bar_engine.atr14[as_of]
        threshold = 0.05 * atr if atr is not None else 0.0
        difference = current - previous
        return 1 if difference > threshold else -1 if difference < -threshold else 0

    def always_in(self, *, as_of_index: int | None = None) -> AlwaysInState:
        as_of = self._as_of(as_of_index)
        near = self._weighted_ema_side(as_of, 8)
        background = self._weighted_ema_side(as_of, 20)
        chosen: tuple[str, str, float, float] | None = None
        if near is not None:
            above, below = near
            slope = self._ema_slope(as_of, 5)
            if above >= 0.65 and slope > 0:
                chosen = ("AIL", "near", above, below)
            elif below >= 0.65 and slope < 0:
                chosen = ("AIS", "near", above, below)
        if chosen is None and background is not None:
            above, below = background
            slope = self._ema_slope(as_of, 10)
            if above >= 0.70 and slope > 0:
                chosen = ("AIL", "background", above, below)
            elif below >= 0.70 and slope < 0:
                chosen = ("AIS", "background", above, below)
        if chosen is None:
            ratios = near or background or (0.0, 0.0)
            return AlwaysInState("neutral", "none", "none", ratios[0], ratios[1])
        state, source, above, below = chosen
        swing = self.swing_structure(radius=2, as_of_index=as_of)
        window = 8 if source == "near" else 20
        start = max(0, as_of - window + 1)
        closes = [bar.close for bar in self.series.bars[start : as_of + 1]]
        atr = self.bar_engine.atr14[as_of]
        shallow = atr is not None and max(closes) - min(closes) <= 1.5 * atr
        structure = swing == ("HH_HL" if state == "AIL" else "LL_LH")
        strength = "strong" if structure and shallow else "weak"
        return AlwaysInState(state, strength, source, above, below)

    def _pullback_ratio(self, direction: str, pivots: Sequence[Pivot]) -> float | None:
        return completed_pullback_ratio(pivots, direction=direction, tick=self.series.tick)

    @staticmethod
    def _trailing_groups(values: Sequence[float], direction: str) -> int:
        groups = 0
        for older, newer in zip(reversed(values[:-1]), reversed(values[1:])):
            valid = newer > older if direction == "bullish" else newer < older
            if not valid:
                break
            groups += 1
        return groups

    def channel(self, *, as_of_index: int | None = None) -> ChannelState:
        as_of = self._as_of(as_of_index)
        pivots = self.pivots(2, as_of_index=as_of)
        highs = [pivot for pivot in pivots if pivot.kind == "high"]
        lows = [pivot for pivot in pivots if pivot.kind == "low"]
        bull_groups = min(
            self._trailing_groups([pivot.price for pivot in highs], "bullish"),
            self._trailing_groups([pivot.price for pivot in lows], "bullish"),
        )
        bear_groups = min(
            self._trailing_groups([pivot.price for pivot in highs], "bearish"),
            self._trailing_groups([pivot.price for pivot in lows], "bearish"),
        )
        direction = (
            "bullish"
            if bull_groups >= bear_groups and bull_groups
            else "bearish"
            if bear_groups
            else None
        )
        groups = max(bull_groups, bear_groups)
        if direction is None:
            return classify_channel_metrics(
                group_count=0,
                pullback_ratio=None,
                parallel_error=None,
                max_residual_atr=None,
            )
        count = min(groups + 1, len(highs), len(lows))
        high_line = _regression([(pivot.index, pivot.price) for pivot in highs[-count:]])
        low_line = _regression([(pivot.index, pivot.price) for pivot in lows[-count:]])
        atr = self.bar_engine.atr14[as_of]
        if high_line is None or low_line is None or atr is None or atr <= 0:
            return classify_channel_metrics(
                group_count=groups,
                pullback_ratio=self._pullback_ratio(direction, pivots),
                parallel_error=None,
                max_residual_atr=None,
                direction=direction,
            )
        high_slope, high_intercept, high_residual = high_line
        low_slope, low_intercept, low_residual = low_line
        convergence = (
            low_slope > high_slope > 0 if direction == "bullish" else high_slope < low_slope < 0
        )
        parallel = abs(high_slope - low_slope) / (abs(high_slope) + abs(low_slope) + 1e-12)
        residual = max(high_residual, low_residual) / atr
        width = abs((high_slope * as_of + high_intercept) - (low_slope * as_of + low_intercept))
        if not convergence:
            parallel = max(parallel, 1.0)
        return classify_channel_metrics(
            group_count=groups,
            pullback_ratio=self._pullback_ratio(direction, pivots),
            parallel_error=parallel,
            max_residual_atr=residual,
            direction=direction,
            width_at_as_of=width,
        )

    def range_state(self, *, as_of_index: int | None = None) -> RangeState:
        as_of = self._as_of(as_of_index)
        start = max(0, as_of - 39)
        bars = [bar for bar in self.series.bars[start : as_of + 1] if bar.valid_ohlc]
        if len(bars) < 3:
            return RangeState(None, None, None, None, "unknown", len(bars), False, False)
        high = max(bar.high for bar in bars)
        low = min(bar.low for bar in bars)
        width = high - low
        close = self.series.bars[as_of].close
        position = (close - low) / width if width > 0 else None
        zone = (
            "unknown"
            if position is None
            else "lower_third"
            if position < 1 / 3
            else "upper_third"
            if position > 2 / 3
            else "middle_third"
        )
        atr = self.bar_engine.atr14[as_of]
        width_atr = width / atr if atr is not None and atr > 0 else None
        tol = max(2 * self.series.tick, 0.15 * atr) if atr is not None else 2 * self.series.tick
        pivots = [pivot for pivot in self.pivots(1, as_of_index=as_of) if pivot.index >= start]
        high_tests = sum(
            pivot.kind == "high" and abs(pivot.price - high) <= tol for pivot in pivots
        )
        low_tests = sum(pivot.kind == "low" and abs(pivot.price - low) <= tol for pivot in pivots)
        direction = self.direction(as_of_index=as_of)
        swing = self.swing_structure(radius=2, as_of_index=as_of)
        ema_slope_atr: float | None = None
        overlap: float | None = None
        if as_of >= 10 and atr is not None and atr > 0:
            current = self.bar_engine.ema20[as_of]
            older = self.bar_engine.ema20[as_of - 10]
            overlap = self._mean_overlap(max(0, as_of - 9), as_of)
            if current is not None and older is not None:
                ema_slope_atr = (current - older) / atr
        trading_range, extreme = classify_range_flags(
            high_tests=high_tests,
            low_tests=low_tests,
            direction=direction.direction,
            swing_structure=swing,
            ema_slope_atr=ema_slope_atr,
            overlap_mean=overlap,
            direction_score=direction.score,
        )
        return RangeState(high, low, width_atr, position, zone, len(bars), trading_range, extreme)

    def barbwire(self, *, as_of_index: int | None = None) -> BarbwireState:
        as_of = self._as_of(as_of_index)
        start = max(0, as_of - 9)
        overlap = self._mean_overlap(start, as_of)
        snapshots = [self.bar_engine.evaluate(index) for index in range(start, as_of + 1)]
        doji_inside = (
            sum(snapshot.value("bar.type") in {"doji", "inside"} for snapshot in snapshots) / 10
            if len(snapshots) == 10
            else None
        )
        bars = self.series.bars[start : as_of + 1]
        width_to_avg = None
        if len(bars) == 10:
            width = max(bar.high for bar in bars) - min(bar.low for bar in bars)
            average = sum(bar.high - bar.low for bar in bars) / len(bars)
            width_to_avg = width / average if average > 0 else None
        return score_barbwire(
            overlap_mean_10=overlap,
            doji_inside_ratio_10=doji_inside,
            range_width_atr=self.range_state(as_of_index=as_of).width_atr,
            width_to_avg_range=width_to_avg,
        )

    def breakout_events(self, *, as_of_index: int | None = None) -> tuple[BreakoutEvent, ...]:
        as_of = self._as_of(as_of_index)
        start = max(0, as_of - 39)
        bars = self.series.bars
        running_high: float | None = None
        running_low: float | None = None
        raw: list[tuple[str, float, int]] = []
        for index in range(start, as_of + 1):
            bar = bars[index]
            if running_high is not None and bar.close > running_high + self.series.tick:
                raw.append(("up", running_high, index))
            if running_low is not None and bar.close < running_low - self.series.tick:
                raw.append(("down", running_low, index))
            running_high = bar.high if running_high is None else max(running_high, bar.high)
            running_low = bar.low if running_low is None else min(running_low, bar.low)

        output: list[BreakoutEvent] = []
        for direction, level, breakout_index in raw:
            end = min(as_of, breakout_index + 5)
            atr = self.bar_engine.atr14[breakout_index]
            tolerance = 0.15 * atr if atr is not None else 0.0
            failure: int | None = None
            retest: int | None = None
            for index in range(breakout_index + 1, end + 1):
                bar = bars[index]
                if direction == "up":
                    if retest is None and bar.low <= level + tolerance and bar.close > level:
                        retest = index
                    if failure is None and bar.close < level - self.series.tick:
                        failure = index
                else:
                    if retest is None and bar.high >= level - tolerance and bar.close < level:
                        retest = index
                    if failure is None and bar.close > level + self.series.tick:
                        failure = index
            failed_failure: int | None = None
            if failure is not None:
                rebreak_end = min(as_of, failure + 5)
                for index in range(failure + 1, rebreak_end + 1):
                    rebreak = (
                        bars[index].close > level + self.series.tick
                        if direction == "up"
                        else bars[index].close < level - self.series.tick
                    )
                    if not rebreak:
                        continue
                    follow, confirmation = self._directional_follow(index, direction, as_of)
                    if follow:
                        failed_failure = confirmation
                        break
            follow, _ = self._directional_follow(breakout_index, direction, as_of)
            output.append(
                BreakoutEvent(
                    direction=direction,
                    level=level,
                    breakout_at=bars[breakout_index].timestamp,
                    follow_through=follow,
                    retest_at=bars[retest].timestamp if retest is not None else None,
                    failure_at=bars[failure].timestamp if failure is not None else None,
                    failed_failure_at=(
                        bars[failed_failure].timestamp if failed_failure is not None else None
                    ),
                )
            )
        return tuple(output)

    def _directional_follow(
        self,
        signal_index: int,
        direction: str,
        as_of: int,
    ) -> tuple[bool, int | None]:
        """Check event-direction follow, independent of signal candle colour."""
        bars = self.series.bars
        end = min(as_of, signal_index + 2)
        signal_close = bars[signal_index].close
        for index in range(signal_index + 1, end + 1):
            if (direction == "up" and bars[index].close > signal_close) or (
                direction == "down" and bars[index].close < signal_close
            ):
                return True, index
        return False, None

    def hl_state(
        self,
        *,
        as_of_index: int | None = None,
        background: str | None = None,
    ) -> HLState:
        as_of = self._as_of(as_of_index)
        supplied_background = background is not None
        if background is None:
            direction = self.direction(as_of_index=as_of).direction
            always = self.always_in(as_of_index=as_of).state
            background = (
                direction
                if direction != "neutral"
                else ("bullish" if always == "AIL" else "bearish" if always == "AIS" else "neutral")
            )
        if background not in {"bullish", "bearish"}:
            return HLState((), "none", False)
        pivots = self.pivots(1, as_of_index=as_of)
        trend_kind = "high" if background == "bullish" else "low"
        context_start = -1
        if not supplied_background and as_of >= 19:
            opposite_state = "AIS" if background == "bullish" else "AIL"
            for index in range(19, as_of + 1):
                if self.always_in(as_of_index=index).state == opposite_state:
                    context_start = index
        trend_pivots = [
            pivot for pivot in pivots if pivot.kind == trend_kind and pivot.index > context_start
        ]
        if not trend_pivots:
            return HLState((), "none", False)
        origin = (
            max(trend_pivots, key=lambda pivot: pivot.price)
            if background == "bullish"
            else min(trend_pivots, key=lambda pivot: pivot.price)
        )
        bars = self.series.bars
        opposite_kind = "low" if background == "bullish" else "high"
        invalidation_pivots = [
            pivot
            for pivot in pivots
            if pivot.kind == opposite_kind and context_start < pivot.index < origin.index
        ]
        invalidation = invalidation_pivots[-1] if invalidation_pivots else None
        triggers: list[HLTrigger] = []
        in_pullback = False
        waiting_second_leg = False
        pullback_pivot: Pivot | None = None
        for index in range(origin.index + 1, as_of + 1):
            bar = bars[index]
            previous = bars[index - 1]
            structure_invalid = bool(
                invalidation is not None
                and (
                    bar.close < invalidation.price - self.series.tick
                    if background == "bullish"
                    else bar.close > invalidation.price + self.series.tick
                )
            )
            if structure_invalid:
                triggers.clear()
                break
            atr = self.bar_engine.atr14[index]
            range_atr = (bar.high - bar.low) / atr if atr is not None and atr > 0 else 0.0
            strong_break = (
                background == "bullish"
                and bar.close > origin.price + self.series.tick
                or background == "bearish"
                and bar.close < origin.price - self.series.tick
            ) and range_atr >= 1.2
            if strong_break:
                follow_end = min(as_of, index + 2)
                if (
                    self.bar_engine.evaluate(index, as_of_index=follow_end).value(
                        "bar.follow_through_1_2"
                    )
                    == "yes"
                ):
                    triggers.clear()
                    in_pullback = False
                    waiting_second_leg = False
                    origin = Pivot(
                        index,
                        bar.timestamp,
                        trend_kind,
                        bar.high if background == "bullish" else bar.low,
                        1,
                        bar.timestamp,
                    )
                    confirmed_opposites = [
                        pivot
                        for pivot in self.pivots(1, as_of_index=index)
                        if pivot.kind == opposite_kind and pivot.index < index
                    ]
                    invalidation = confirmed_opposites[-1] if confirmed_opposites else None
                    continue
            starts = (
                bar.low < previous.low or bar.close < previous.close
                if background == "bullish"
                else bar.high > previous.high or bar.close > previous.close
            )
            if not in_pullback and not waiting_second_leg and starts:
                in_pullback = True
            if waiting_second_leg:
                confirmed_pivots = self.pivots(1, as_of_index=index)
                wanted = "low" if background == "bullish" else "high"
                # Compare timestamps because HLTrigger intentionally exposes no mutable index.
                last_trigger_at = triggers[-1].trigger_at
                fresh = [
                    pivot
                    for pivot in confirmed_pivots
                    if pivot.kind == wanted and pivot.timestamp > last_trigger_at
                ]
                if fresh:
                    pullback_pivot = fresh[-1]
                    in_pullback = True
                    waiting_second_leg = False
            triggers_now = (
                bar.high > previous.high + self.series.tick
                if background == "bullish"
                else bar.low < previous.low - self.series.tick
            )
            if in_pullback and triggers_now:
                number = len(triggers) + 1
                prefix = "H" if background == "bullish" else "L"
                label = f"{prefix}{min(number, 3)}"
                triggers.append(
                    HLTrigger(
                        label,
                        bar.timestamp,
                        pullback_pivot.timestamp if pullback_pivot is not None else None,
                    )
                )
                in_pullback = False
                follow_end = min(as_of, index + 2)
                follow = self.bar_engine.evaluate(index, as_of_index=follow_end).value(
                    "bar.follow_through_1_2"
                )
                waiting_second_leg = follow in {"failed", "no"}
        candidate = triggers[-1].label if triggers else "none"
        return HLState(tuple(triggers), candidate, len(triggers) >= 3)

    def _double_patterns(self, as_of: int) -> tuple[PatternCandidate, ...]:
        pivots = self.pivots(1, as_of_index=as_of)
        atr = self.bar_engine.atr14[as_of]
        if atr is None or atr <= 0:
            return ()
        tolerance = max(2 * self.series.tick, 0.10 * atr)
        output: list[PatternCandidate] = []
        for kind, pattern, direction in (
            ("high", "double_top", "bearish"),
            ("low", "double_bottom", "bullish"),
        ):
            same = [pivot for pivot in pivots if pivot.kind == kind]
            if len(same) < 2:
                continue
            first, second = same[-2:]
            opposite_kind = "low" if kind == "high" else "high"
            opposite = [
                pivot
                for pivot in pivots
                if pivot.kind == opposite_kind and first.index < pivot.index < second.index
            ]
            if not opposite or abs(first.price - second.price) > tolerance:
                continue
            neck = (
                min(pivot.price for pivot in opposite)
                if kind == "high"
                else max(pivot.price for pivot in opposite)
            )
            depth = first.price - neck if kind == "high" else neck - first.price
            if depth < 0.50 * atr:
                continue
            confirmation_index: int | None = None
            for index in range(second.index + 1, min(as_of, second.index + 2) + 1):
                bar_type = self.bar_engine.evaluate(index).value("bar.type")
                if bar_type == ("trend_bear" if kind == "high" else "trend_bull"):
                    confirmation_index = index
                    break
            if confirmation_index is None:
                continue
            invalidated = False
            continuation = "up" if kind == "high" else "down"
            for index in range(second.index + 1, as_of + 1):
                broke_extreme = (
                    self.series.bars[index].close > second.price + tolerance
                    if kind == "high"
                    else self.series.bars[index].close < second.price - tolerance
                )
                if broke_extreme and self._directional_follow(index, continuation, as_of)[0]:
                    invalidated = True
                    break
            if invalidated:
                continue
            output.append(
                PatternCandidate(
                    pattern=pattern,
                    direction=direction,
                    confirmed_at=self.series.bars[confirmation_index].timestamp,
                    evidence=(
                        f"first@{first.timestamp}",
                        f"neck@{opposite[-1].timestamp}",
                        f"second@{second.timestamp}",
                    ),
                    follow_through=True,
                    diagnostic_only=True,
                    invalidation_price=second.price,
                )
            )
        return tuple(output)

    def patterns(self, *, as_of_index: int | None = None) -> tuple[PatternCandidate, ...]:
        as_of = self._as_of(as_of_index)
        atr = self.bar_engine.atr14[as_of]
        if atr is None or atr <= 0:
            return ()
        direction = self.direction(as_of_index=as_of)
        pivots = self.pivots(1, as_of_index=as_of)
        output = list(
            detect_wedge_candidates(
                pivots,
                atr=atr,
                tick=self.series.tick,
                trend_direction=direction.direction,
            )
        )
        doubles = self._double_patterns(as_of)
        output.extend(doubles)
        opposite_break = any(
            (direction.direction == "bullish" and event.direction == "down")
            or (direction.direction == "bearish" and event.direction == "up")
            for event in self.breakout_events(as_of_index=as_of)
        )
        if doubles and direction.direction != "neutral":
            mtr = evaluate_mtr(
                direction=direction.direction,
                swing_structure=self.swing_structure(radius=2, as_of_index=as_of),
                trendline_break=opposite_break,
                recovery_failed=True,
                extreme_retest_failed=True,
                confirmed_at=max(candidate.confirmed_at for candidate in doubles),
            )
            output.append(mtr)
        final_flag = self._final_flag(as_of)
        if final_flag is not None:
            output.append(final_flag)
        return tuple(output)

    def _final_flag(self, as_of: int) -> PatternCandidate | None:
        if as_of < 19:
            return None
        bars = self.series.bars
        for consolidation_end in range(as_of, max(8, as_of - 2) - 1, -1):
            start = consolidation_end - 9
            if start <= 0:
                continue
            atr = self.bar_engine.atr14[consolidation_end]
            ema_now = self.bar_engine.ema20[consolidation_end]
            ema_old = self.bar_engine.ema20[start]
            if atr is None or atr <= 0 or ema_now is None or ema_old is None:
                continue
            overlap = self._mean_overlap(start, consolidation_end)
            if overlap is None:
                continue
            snapshots = [
                self.bar_engine.evaluate(index) for index in range(start, consolidation_end + 1)
            ]
            doji_inside = (
                sum(snapshot.value("bar.type") in {"doji", "inside"} for snapshot in snapshots) / 10
            )
            prior = self.direction(as_of_index=start - 1)
            prior_always = self.always_in(as_of_index=start - 1)
            prior_environment = self.environment(as_of_index=start - 1)
            moves = self.measured_moves(as_of_index=start - 1)
            close = bars[consolidation_end].close
            target_distance = min(
                (abs(move.target_price - close) / atr for move in moves),
                default=math.inf,
            )
            consolidation = bars[start : consolidation_end + 1]
            consolidation_high = max(bar.high for bar in consolidation)
            consolidation_low = min(bar.low for bar in consolidation)
            breakout_direction = "up" if prior.score > 0 else "down"
            breakout_index: int | None = None
            for index in range(consolidation_end + 1, as_of + 1):
                broke = (
                    bars[index].close > consolidation_high + self.series.tick
                    if breakout_direction == "up"
                    else bars[index].close < consolidation_low - self.series.tick
                )
                if broke:
                    breakout_index = index
                    break
            breakout_observed = breakout_index is not None
            breakout_follow = False
            returned = False
            confirmed_at = bars[consolidation_end].timestamp
            if breakout_index is not None:
                breakout_follow, confirmation_index = self._directional_follow(
                    breakout_index,
                    breakout_direction,
                    as_of,
                )
                returned = any(
                    consolidation_low <= bars[index].close <= consolidation_high
                    for index in range(breakout_index + 1, as_of + 1)
                )
                confirmed_at = bars[
                    confirmation_index if confirmation_index is not None else as_of
                ].timestamp
            candidate = evaluate_final_flag(
                direction_score=prior.score,
                had_always_in_or_spike=(prior_always.state != "neutral" or prior_environment.spike),
                consolidation_bars=10,
                ema_slope_atr=(ema_now - ema_old) / atr,
                overlap_mean=overlap,
                doji_inside_ratio=doji_inside,
                target_distance_atr=target_distance,
                breakout_observed=breakout_observed,
                breakout_follow=breakout_follow,
                returned_to_range=returned,
                confirmed_at=confirmed_at,
            )
            if candidate is not None:
                return candidate
        return None

    def environment(self, *, as_of_index: int | None = None) -> EnvironmentState:
        as_of = self._as_of(as_of_index)
        bars = self.series.bars
        types = [self.bar_engine.evaluate(index).value("bar.type") for index in range(as_of + 1)]

        def trailing_run(end: int) -> tuple[str | None, int, int]:
            if end < 0 or types[end] not in {"trend_bull", "trend_bear"}:
                return None, 0, end + 1
            kind = types[end]
            start_index = end
            while start_index > 0 and types[start_index - 1] == kind:
                start_index -= 1
            return kind, end - start_index + 1, start_index

        def body_overlap(left: int, right: int) -> float:
            left_low, left_high = sorted((bars[left].open, bars[left].close))
            right_low, right_high = sorted((bars[right].open, bars[right].close))
            shared = max(0.0, min(left_high, right_high) - max(left_low, right_low))
            union = max(left_high, right_high) - min(left_low, right_low)
            return shared / union if union > 0 else 1.0

        run_kind, run_count, run_start = trailing_run(as_of)
        run_overlaps = [body_overlap(index - 1, index) for index in range(run_start + 1, as_of + 1)]
        closes_extend = (
            all(
                bars[index].close > bars[index - 1].close
                if run_kind == "trend_bull"
                else bars[index].close < bars[index - 1].close
                for index in range(run_start + 1, as_of + 1)
            )
            if run_kind is not None
            else False
        )
        spike = bool(
            2 <= run_count <= 5
            and closes_extend
            and run_overlaps
            and all(value < 0.30 for value in run_overlaps)
        )
        standard_spike = bool(
            3 <= run_count <= 5
            and closes_extend
            and run_overlaps
            and all(value < 0.20 for value in run_overlaps)
        )
        climax_warning = bool(
            run_count >= 6
            and closes_extend
            and run_overlaps
            and all(value < 0.30 for value in run_overlaps)
        )

        climax_triggered = False
        prior_kind, prior_count, prior_start = trailing_run(as_of - 1)
        if prior_kind is not None and prior_count >= 2:
            prior_overlaps = [
                body_overlap(index - 1, index) for index in range(prior_start + 1, as_of)
            ]
            prior_strong = bool(
                prior_overlaps
                and all(value < 0.30 for value in prior_overlaps)
                and all(
                    bars[index].close > bars[index - 1].close
                    if prior_kind == "trend_bull"
                    else bars[index].close < bars[index - 1].close
                    for index in range(prior_start + 1, as_of)
                )
            )
            current = bars[as_of]
            body = abs(current.close - current.open)
            upper = current.high - max(current.open, current.close)
            lower = min(current.open, current.close) - current.low
            average_body = (
                sum(
                    abs(bars[index].close - bars[index].open) for index in range(prior_start, as_of)
                )
                / prior_count
            )
            reverse = (
                types[as_of] == "trend_bear"
                if prior_kind == "trend_bull"
                else types[as_of] == "trend_bull"
            )
            climax_triggered = bool(
                prior_strong
                and (max(upper, lower) > 0.50 * body or body < 0.30 * average_body or reverse)
            )

        atr = self.bar_engine.atr14[as_of]
        micro_channel = False
        if not spike and not climax_warning:
            for size in range(min(10, as_of + 1), 1, -1):
                start = as_of - size + 1
                for direction in ("bullish", "bearish"):
                    bad: list[int] = []
                    for index in range(start + 1, as_of + 1):
                        good = (
                            bars[index].high >= bars[index - 1].high
                            and bars[index].low >= bars[index - 1].low
                            if direction == "bullish"
                            else bars[index].high <= bars[index - 1].high
                            and bars[index].low <= bars[index - 1].low
                        )
                        if not good:
                            bad.append(index)
                    if size - 1 - len(bad) < size - 2 or len(bad) > 2:
                        continue
                    if any(types[index] not in {"inside", "doji"} for index in bad):
                        continue
                    adverse = 0.0
                    for index in bad:
                        if direction == "bullish":
                            adverse = max(
                                adverse,
                                bars[index - 1].low - bars[index].low,
                                bars[index - 1].high - bars[index].high,
                            )
                        else:
                            adverse = max(
                                adverse,
                                bars[index].low - bars[index - 1].low,
                                bars[index].high - bars[index - 1].high,
                            )
                    if atr is not None and adverse <= 0.25 * atr:
                        micro_channel = True
                        break
                if micro_channel:
                    break
        channel = self.channel(as_of_index=as_of)
        labels: list[str] = []
        if standard_spike:
            labels.append("standard_spike")
        elif spike:
            labels.append("spike_candidate")
        if climax_warning:
            labels.append("climax_warning")
        if climax_triggered:
            labels.append("climax_triggered")
        if micro_channel:
            labels.append("micro_channel")
        if channel.label != "none":
            labels.append(channel.label)
        range_state = self.range_state(as_of_index=as_of)
        if range_state.trading_range:
            labels.append("trading_range")
        return EnvironmentState(
            tuple(dict.fromkeys(labels)),
            spike,
            micro_channel,
            climax_triggered,
        )

    def measured_moves(self, *, as_of_index: int | None = None) -> tuple[MeasuredMove, ...]:
        as_of = self._as_of(as_of_index)
        bars = self.series.bars
        range_state = self.range_state(as_of_index=as_of)
        output: list[MeasuredMove] = []
        if range_state.high is not None and range_state.low is not None:
            height = range_state.high - range_state.low
            start = max(0, as_of - range_state.lookback_bars + 1)
            output.extend(
                (
                    MeasuredMove(
                        "range_up",
                        range_state.high + height,
                        height,
                        bars[start].timestamp,
                        bars[as_of].timestamp,
                        True,
                    ),
                    MeasuredMove(
                        "range_down",
                        range_state.low - height,
                        height,
                        bars[start].timestamp,
                        bars[as_of].timestamp,
                        True,
                    ),
                )
            )
        pivots = self.pivots(1, as_of_index=as_of)
        for end in range(len(pivots) - 1, 1, -1):
            first, second, third = pivots[end - 2 : end + 1]
            if (first.kind, second.kind, third.kind) == ("low", "high", "low"):
                height = second.price - first.price
                if height > self.series.tick:
                    output.append(
                        MeasuredMove(
                            "leg_up",
                            third.price + height,
                            height,
                            first.timestamp,
                            third.timestamp,
                            True,
                        )
                    )
                    break
            if (first.kind, second.kind, third.kind) == ("high", "low", "high"):
                height = first.price - second.price
                if height > self.series.tick:
                    output.append(
                        MeasuredMove(
                            "leg_down",
                            third.price - height,
                            height,
                            first.timestamp,
                            third.timestamp,
                            True,
                        )
                    )
                    break
        channel = self.channel(as_of_index=as_of)
        if channel.confirmed and channel.width_at_as_of is not None:
            height = channel.width_at_as_of
            direction = 1 if channel.direction == "bullish" else -1
            output.append(
                project_measured_move(
                    kind="channel_up" if direction > 0 else "channel_down",
                    height=height,
                    anchor_price=bars[as_of].close,
                    direction="up" if direction > 0 else "down",
                    source_start=bars[max(0, as_of - 39)].timestamp,
                    source_end=bars[as_of].timestamp,
                )
            )
        atr = self.bar_engine.atr14[as_of]
        if atr is not None and atr > 0:
            wedges = detect_wedge_candidates(
                pivots,
                atr=atr,
                tick=self.series.tick,
                trend_direction=self.direction(as_of_index=as_of).direction,
            )
            events = self.breakout_events(as_of_index=as_of)
            pivot_by_time = {pivot.timestamp: pivot for pivot in pivots}
            bar_index_by_time = {
                bar.timestamp: index for index, bar in enumerate(bars[: as_of + 1])
            }
            for wedge in wedges:
                wanted = "up" if wedge.direction == "bullish" else "down"
                breakout = next(
                    (
                        event
                        for event in events
                        if event.direction == wanted
                        and event.breakout_at >= wedge.confirmed_at
                        and event.follow_through
                    ),
                    None,
                )
                if breakout is None:
                    continue
                evidence_times = [int(item.rsplit("@", 1)[1]) for item in wedge.evidence]
                evidence_pivots = [
                    pivot_by_time[value] for value in evidence_times if value in pivot_by_time
                ]
                if len(evidence_pivots) < 4:
                    continue
                height = max(pivot.price for pivot in evidence_pivots) - min(
                    pivot.price for pivot in evidence_pivots
                )
                breakout_index = bar_index_by_time[breakout.breakout_at]
                output.append(
                    project_measured_move(
                        kind="wedge_up" if wanted == "up" else "wedge_down",
                        height=height,
                        anchor_price=bars[breakout_index].close,
                        direction=wanted,
                        source_start=min(evidence_times),
                        source_end=breakout.breakout_at,
                    )
                )
        return tuple(output)

    def _formula_result(
        self,
        formula_id: str,
        value: object,
        as_of: int,
        provenance: str,
        *,
        valid: bool = True,
        invalid_reason: str | None = None,
    ) -> FormulaResult:
        timestamp = self.series.bars[as_of].timestamp
        return FormulaResult(
            formula_id=formula_id,
            value=value,
            valid=valid,
            as_of=timestamp,
            source_start=self.series.bars[max(0, as_of - 39)].timestamp,
            source_end=timestamp,
            confirmed_at=timestamp,
            invalid_reason=invalid_reason,
            provenance_class=provenance,
        )

    def snapshot(self, *, as_of_index: int | None = None) -> StructureSnapshot:
        as_of = self._as_of(as_of_index)
        range_state = self.range_state(as_of_index=as_of)
        swing = self.swing_structure(radius=2, as_of_index=as_of)
        support = self.support_resistance(as_of_index=as_of)
        direction = self.direction(as_of_index=as_of)
        always = self.always_in(as_of_index=as_of)
        channel = self.channel(as_of_index=as_of)
        barbwire = self.barbwire(as_of_index=as_of)
        breakouts = self.breakout_events(as_of_index=as_of)
        hl = self.hl_state(as_of_index=as_of)
        environment = self.environment(as_of_index=as_of)
        patterns = self.patterns(as_of_index=as_of)
        moves = self.measured_moves(as_of_index=as_of)
        indicator_ready = bool(
            self.bar_engine.atr14[as_of] is not None and self.bar_engine.ema20[as_of] is not None
        )
        results = (
            self._formula_result(
                "structure.range",
                range_state,
                as_of,
                "D0/D1",
                valid=range_state.high is not None,
                invalid_reason=None if range_state.high is not None else "window_lt_3",
            ),
            self._formula_result(
                "structure.swing",
                swing,
                as_of,
                "D0",
                valid=swing != "insufficient",
                invalid_reason=None if swing != "insufficient" else "confirmed_pivots_insufficient",
            ),
            self._formula_result("structure.support_resistance", support, as_of, "D0-C"),
            self._formula_result(
                "structure.direction",
                direction,
                as_of,
                "D0",
                valid=indicator_ready and as_of >= 7,
                invalid_reason=None if indicator_ready and as_of >= 7 else "indicator_warmup",
            ),
            self._formula_result(
                "structure.always_in",
                always,
                as_of,
                "D0",
                valid=indicator_ready and as_of >= 7,
                invalid_reason=None if indicator_ready and as_of >= 7 else "indicator_warmup",
            ),
            self._formula_result(
                "structure.channel",
                channel,
                as_of,
                "D1",
                valid=channel.label != "none",
                invalid_reason=None if channel.label != "none" else "confirmed_pivots_insufficient",
            ),
            self._formula_result(
                "structure.barbwire",
                barbwire,
                as_of,
                "D0",
                valid=indicator_ready and as_of >= 9,
                invalid_reason=None if indicator_ready and as_of >= 9 else "window_lt_10",
            ),
            self._formula_result("structure.breakouts", breakouts, as_of, "D0/D1"),
            self._formula_result(
                "structure.hl",
                hl,
                as_of,
                "D0-C/D1",
                valid=indicator_ready,
                invalid_reason=None if indicator_ready else "indicator_warmup",
            ),
            self._formula_result(
                "structure.patterns",
                patterns,
                as_of,
                "D1",
                valid=indicator_ready,
                invalid_reason=None if indicator_ready else "indicator_warmup",
            ),
            self._formula_result(
                "structure.measured_moves",
                moves,
                as_of,
                "D0-C/D1",
                valid=range_state.high is not None,
                invalid_reason=None if range_state.high is not None else "window_lt_3",
            ),
        )
        return StructureSnapshot(
            as_of=self.series.bars[as_of].timestamp,
            range_state=range_state,
            swing_structure=swing,
            support_resistance=support,
            direction=direction,
            always_in=always,
            channel=channel,
            barbwire=barbwire,
            breakout_events=breakouts,
            hl_state=hl,
            environment=environment,
            patterns=patterns,
            measured_moves=moves,
            results=results,
        )
