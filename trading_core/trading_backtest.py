"""Deterministic research backtest primitives for Stage 6.

This module is deliberately broker-free.  It consumes immutable Stage-4 plans
and closed OHLCV bars and cannot import or call the Stage-5 MT5 adapter.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Event
from typing import Any, Callable, Mapping

from .decision_reviewer import (
    DecisionReviewRequestV1,
    DecisionReviewer,
    DecisionReviewV1,
    MechanicalDecisionReviewer,
)
from model_core.walk_forward import formula_warmup_bars

from .alpha_adapter import load_alpha_strategy
from .live import TradingConfigV1
from .models import (
    ClosedBarV1,
    InstrumentConstraintsV1,
    LoadedAlphaStrategyV1,
    OrderPlanCandidateV1,
    TimeframeSeriesV1,
)
from .time_alignment import bar_close_timestamp


BACKTEST_ENGINE_VERSION = "trading-backtest-v1"
BACKTEST_REPORT_VERSION = "trading-backtest-report-v1"
REVIEW_CACHE_VERSION = "decision-review-cache-v1"
REVIEW_SCHEMA_VERSION = "decision-review-v1"
REVIEW_PROMPT_CONTRACT_VERSION = "closed-world-review-v1"


@dataclass(frozen=True, slots=True)
class SevenDayWindowV1:
    start_close: int
    end_close: int
    calendar_days: int = 7
    timezone: str = "UTC"


def seven_natural_day_window(latest_m15_close: int) -> SevenDayWindowV1:
    """Return the UTC calendar window ending at the latest closed M15."""

    if type(latest_m15_close) is not int:
        raise TypeError("latest_m15_close must be an integer Unix second")
    latest = datetime.fromtimestamp(latest_m15_close, tz=timezone.utc)
    start = datetime.combine(
        (latest - timedelta(days=6)).date(), datetime.min.time(), tzinfo=timezone.utc
    )
    return SevenDayWindowV1(start_close=int(start.timestamp()), end_close=latest_m15_close)


def _canonical_bar(bar: ClosedBarV1) -> list[int | float]:
    return [bar.timestamp, bar.open, bar.high, bar.low, bar.close, bar.volume]


def canonical_dataset_identity(
    series: TimeframeSeriesV1,
    *,
    end_close: int,
    requested_warmup: int,
    evaluation_start: int | None = None,
) -> dict[str, Any]:
    """Describe and digest only bars closed no later than the replay end."""

    if requested_warmup < 0:
        raise ValueError("requested_warmup must be non-negative")
    eligible = tuple(
        bar
        for bar in series.bars
        if bar_close_timestamp(bar, series.timeframe) <= end_close
    )
    encoded = json.dumps(
        [_canonical_bar(bar) for bar in eligible],
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    first_close = bar_close_timestamp(eligible[0], series.timeframe) if eligible else None
    last_close = bar_close_timestamp(eligible[-1], series.timeframe) if eligible else None
    if evaluation_start is None:
        available_warmup = max(0, len(eligible) - 1)
    else:
        available_warmup = sum(
            bar_close_timestamp(bar, series.timeframe) < evaluation_start for bar in eligible
        )
    return {
        "symbol": series.symbol,
        "timeframe": series.timeframe,
        "timezone": "UTC",
        "requested_warmup": requested_warmup,
        "available_warmup": available_warmup,
        "bar_count": len(eligible),
        "first_close": first_close,
        "last_close": last_close,
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _review_payload(review: DecisionReviewV1) -> dict[str, Any]:
    return asdict(review)


class CachedDecisionReviewer:
    """Persistent decision-id cache; IDs already bind canonical request inputs."""

    def __init__(
        self,
        reviewer: DecisionReviewer,
        path: Path,
        *,
        reviewer_identity: Mapping[str, str] | None = None,
    ) -> None:
        self.reviewer = reviewer
        self.path = Path(path)
        self.reviewer_identity = dict(
            reviewer_identity
            or {
                "reviewer": f"{type(reviewer).__module__}.{type(reviewer).__qualname__}",
                "review_schema_version": REVIEW_SCHEMA_VERSION,
                "prompt_contract_version": REVIEW_PROMPT_CONTRACT_VERSION,
            }
        )
        self.hits = 0
        self.misses = 0

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {
                "schema_version": REVIEW_CACHE_VERSION,
                "reviewer_identity": self.reviewer_identity,
                "entries": {},
            }
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if type(payload) is not dict:
            raise ValueError("review cache must contain an object")
        if payload.get("schema_version") != REVIEW_CACHE_VERSION:
            raise ValueError("review cache schema version does not match")
        if payload.get("reviewer_identity") != self.reviewer_identity:
            raise ValueError("review cache reviewer identity does not match")
        if type(payload.get("entries")) is not dict:
            raise ValueError("review cache entries must contain an object")
        return payload

    def _save(self, payload: dict[str, Any]) -> None:
        _atomic_json(payload, self.path)

    async def review(self, request: DecisionReviewRequestV1) -> DecisionReviewV1:
        # Non-candidates pass through the fail-closed reviewer and neither invoke
        # Codex nor pollute candidate review cache statistics.
        if not request.signal.accepted or not request.plans:
            return await MechanicalDecisionReviewer().review(request)
        cache = self._load()
        entries = cache["entries"]
        stored = entries.get(request.decision_id)
        if stored is not None:
            if type(stored) is not dict or stored.get("decision_id") != request.decision_id:
                raise ValueError("cached review decision_id does not match cache key")
            review = DecisionReviewV1(**stored)
            if review.decision_id != request.decision_id:
                raise ValueError("cached review decision_id does not match request")
            self.hits += 1
            return review
        self.misses += 1
        review = await self.reviewer.review(request)
        if review.decision_id != request.decision_id:
            raise ValueError("review decision_id does not match request")
        entries[request.decision_id] = _review_payload(review)
        self._save(cache)
        return review


@dataclass(slots=True)
class _Leg:
    name: str
    volume: float
    target: float
    stop: float
    result_r: float | None = None


@dataclass(slots=True)
class _Trade:
    plan: OrderPlanCandidateV1
    decision_close: int
    filled: bool
    triggered: bool
    legs: list[_Leg]


class HistoricalExecutionSimulator:
    """Pure M5 plan fill and position lifecycle simulator."""

    def __init__(self, config: TradingConfigV1, *, spread_points: float) -> None:
        if not math.isfinite(spread_points) or spread_points < 0:
            raise ValueError("spread_points must be finite and non-negative")
        self.config = config
        self.spread_points = float(spread_points)
        self.pending: _Trade | None = None
        self.position: _Trade | None = None
        self.cooldown_until: int | None = None
        self.results: list[dict[str, Any]] = []
        self.counts: dict[str, int] = {
            "placed": 0,
            "filled": 0,
            "unfilled": 0,
            "expired_cancelled": 0,
            "tp1": 0,
            "tp2": 0,
            "sl": 0,
            "reverse_exit": 0,
            "stop_limit_triggered": 0,
        }
        self.side_distribution = {"long": 0, "short": 0}
        self.order_type_distribution = {
            "market": 0,
            "limit": 0,
            "stop": 0,
            "stop_limit": 0,
        }
        self.pa_family_distribution: dict[str, int] = {}

    def _new_trade(self, plan: OrderPlanCandidateV1, close: int, filled: bool) -> _Trade:
        legs = [_Leg("tp1", self.config.tp1_lots, plan.take_profit_1, plan.stop_loss)]
        if self.config.tp2_lots > 0 and plan.take_profit_2 is not None:
            legs.append(_Leg("tp2", self.config.tp2_lots, plan.take_profit_2, plan.stop_loss))
        return _Trade(plan, close, filled, plan.order_type != "stop_limit", legs)

    def submit(
        self,
        plan: OrderPlanCandidateV1,
        *,
        decision_close: int,
        market_price: float | None = None,
    ) -> str:
        if self.cooldown_until is not None and decision_close < self.cooldown_until:
            return "reverse_cooldown"
        if self.position is not None:
            if self.position.plan.side == plan.side:
                return "no_pyramiding"
            self._close_position_at(
                plan.entry_price if market_price is None else market_price,
                "reverse_exit",
            )
            self.cooldown_until = decision_close + 900
            return "reverse_exit"
        if self.pending is not None:
            self._expire_pending()
        trade = self._new_trade(plan, decision_close, plan.order_type == "market")
        self.counts["placed"] += 1
        self.side_distribution[plan.side] += 1
        self.order_type_distribution[plan.order_type] += 1
        self.pa_family_distribution[plan.family] = self.pa_family_distribution.get(plan.family, 0) + 1
        if trade.filled:
            self.counts["filled"] += 1
            self.position = trade
        else:
            self.pending = trade
        return "placed"

    def _expire_pending(self) -> None:
        if self.pending is not None:
            self.counts["unfilled"] += 1
            self.counts["expired_cancelled"] += 1
            self.pending = None

    def before_decision(self, decision_close: int) -> None:
        if self.pending is not None and self.pending.decision_close < decision_close:
            self._expire_pending()

    @staticmethod
    def _touch(bar: ClosedBarV1, price: float) -> bool:
        return bar.low <= price <= bar.high

    def _pending_fills(self, trade: _Trade, bar: ClosedBarV1) -> bool:
        plan = trade.plan
        if plan.order_type == "limit":
            limit = float(plan.limit_price)
            return bar.low <= limit if plan.side == "long" else bar.high >= limit
        if plan.order_type == "stop":
            trigger = float(plan.trigger_price)
            return bar.high >= trigger if plan.side == "long" else bar.low <= trigger
        if plan.order_type == "stop_limit":
            trigger = float(plan.trigger_price)
            trigger_hit = (
                bar.high >= trigger if plan.side == "long" else bar.low <= trigger
            )
            if not trade.triggered and trigger_hit:
                trade.triggered = True
                self.counts["stop_limit_triggered"] += 1
            limit = float(plan.limit_price)
            # If both levels occur in one M5 bar, assume trigger then limit fill;
            # protection is subsequently resolved adverse-first on that bar.
            limit_hit = bar.low <= limit if plan.side == "long" else bar.high >= limit
            return trade.triggered and limit_hit
        return False

    def process_bar(self, bar: ClosedBarV1, *, close_timestamp: int) -> None:
        if self.pending is not None and close_timestamp > self.pending.decision_close:
            trade = self.pending
            if self._pending_fills(trade, bar):
                self.pending = None
                trade.filled = True
                self.position = trade
                self.counts["filled"] += 1
                # Entry and protection touching on one bar is unknowable: the
                # adverse stop is evaluated first below.
        active_trade = self.position
        if active_trade is None or close_timestamp <= active_trade.decision_close:
            return
        plan = active_trade.plan
        active = [leg for leg in active_trade.legs if leg.result_r is None]
        if not active:
            return
        stop = active[0].stop
        if self._touch(bar, stop):
            self.counts["sl"] += 1
            for leg in active:
                leg.result_r = self._r_at(plan, stop)
                self._record_leg(active_trade, leg, "sl")
            self.position = None
            return
        tp1 = next((leg for leg in active if leg.name == "tp1"), None)
        if tp1 is not None and self._touch(bar, tp1.target):
            tp1.result_r = plan.take_profit_1_r
            self.counts["tp1"] += 1
            self._record_leg(active_trade, tp1, "tp1")
            for leg in active_trade.legs:
                if leg.name == "tp2" and leg.result_r is None:
                    leg.stop = (
                        plan.entry_price + self.spread_points
                        if plan.side == "long"
                        else plan.entry_price - self.spread_points
                    )
                    if self._touch(bar, leg.stop):
                        leg.result_r = self._r_at(plan, leg.stop)
                        self.counts["sl"] += 1
                        self._record_leg(active_trade, leg, "sl")
        tp2 = next(
            (leg for leg in active_trade.legs if leg.name == "tp2" and leg.result_r is None),
            None,
        )
        if tp2 is not None and self._touch(bar, tp2.target):
            tp2.result_r = float(plan.take_profit_2_r)
            self.counts["tp2"] += 1
            self._record_leg(active_trade, tp2, "tp2")
        if all(leg.result_r is not None for leg in active_trade.legs):
            self.position = None

    @staticmethod
    def _r_at(plan: OrderPlanCandidateV1, price: float) -> float:
        signed = price - plan.entry_price if plan.side == "long" else plan.entry_price - price
        return signed / plan.risk_distance

    def _record_leg(self, trade: _Trade, leg: _Leg, exit_kind: str) -> None:
        total_volume = sum(item.volume for item in trade.legs)
        if leg.result_r is None:
            raise RuntimeError("cannot record an unrealized leg")
        self.results.append(
            {
                "plan_id": trade.plan.plan_id,
                "decision_close": trade.decision_close,
                "side": trade.plan.side,
                "leg": leg.name,
                "volume": leg.volume,
                "exit": exit_kind,
                "r": leg.result_r,
                "r_contribution": float(leg.result_r) * leg.volume / total_volume,
            }
        )

    def _close_position_at(self, price: float, reason: str) -> None:
        trade = self.position
        if trade is None:
            return
        for leg in trade.legs:
            if leg.result_r is None:
                leg.result_r = self._r_at(trade.plan, price)
                self._record_leg(trade, leg, reason)
        self.counts[reason] += 1
        self.position = None

    def snapshot(self) -> dict[str, Any]:
        contributions_by_trade: dict[tuple[int, str], float] = {}
        for row in self.results:
            key = (int(row["decision_close"]), str(row["plan_id"]))
            contributions_by_trade[key] = contributions_by_trade.get(key, 0.0) + float(
                row["r_contribution"]
            )
        values = list(contributions_by_trade.values())
        return {
            **self.counts,
            "side_distribution": dict(self.side_distribution),
            "order_type_distribution": dict(self.order_type_distribution),
            "pa_family_distribution": dict(self.pa_family_distribution),
            "realized_legs": list(self.results),
            "basic_r": {
                "total": sum(values),
                "average": sum(values) / len(values) if values else 0.0,
                "wins": sum(value > 0 for value in values),
                "losses": sum(value < 0 for value in values),
                "flat": sum(value == 0 for value in values),
            },
            "open_at_end": int(self.position is not None),
        }

    def finish(self) -> dict[str, Any]:
        self._expire_pending()
        return self.snapshot()


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def write_backtest_report(
    report: dict[str, Any], output_dir: Path, *, end_close: int
) -> tuple[Path, Path]:
    """Atomically write one immutable report and the stable latest file."""

    if report.get("schema_version") != BACKTEST_REPORT_VERSION:
        raise ValueError("unsupported trading backtest report schema")
    digest = hashlib.sha256(
        json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()[:12]
    directory = Path(output_dir)
    immutable = directory / f"trading_backtest_{end_close}_{digest}.json"
    latest = directory / "latest_trading_backtest_report.json"
    _atomic_json(report, immutable)
    _atomic_json(report, latest)
    return immutable, latest


class ReplaySeriesMarket:
    """Read-only prefix view over one immutable downloaded dataset."""

    def __init__(
        self,
        series: Mapping[str, TimeframeSeriesV1],
        constraints: InstrumentConstraintsV1 | None,
    ) -> None:
        self.series = dict(series)
        self.constraints = constraints or InstrumentConstraintsV1(tick=0.01)
        self.decision_close: int | None = None
        self.evaluated_closes: set[int] = set()

    def set_decision_close(self, decision_close: int) -> None:
        if decision_close in self.evaluated_closes:
            raise RuntimeError(f"M15 close already evaluated: {decision_close}")
        self.evaluated_closes.add(decision_close)
        self.decision_close = decision_close

    def fetch_series(self, symbol: str, timeframe: str, count: int) -> TimeframeSeriesV1:
        if self.decision_close is None:
            raise RuntimeError("replay decision close is not set")
        source = self.series[timeframe]
        if source.symbol != symbol:
            raise ValueError("replay symbol mismatch")
        eligible = tuple(
            bar
            for bar in source.bars
            if bar_close_timestamp(bar, timeframe) <= self.decision_close
        )
        return TimeframeSeriesV1(
            symbol=symbol,
            timeframe=timeframe,
            tick=source.tick,
            bars=eligible[-count:],
        )

    def instrument_constraints(self, symbol: str) -> InstrumentConstraintsV1:
        if any(series.symbol != symbol for series in self.series.values()):
            raise ValueError("replay symbol mismatch")
        return self.constraints


class TradingBacktestRunner:
    """Acquire one dataset and replay it sequentially through the live kernel."""

    _PA_WARMUP = 384

    def __init__(
        self,
        market: Any | None = None,
        *,
        output_dir: Path | None = None,
        spread_points: float = 0.01,
        preview_engine_factory: Callable[..., Any] | None = None,
        strategy_resolver: Callable[[str, str], Path] | None = None,
    ) -> None:
        if market is None:
            from web.trading_mt5 import mt5_read_only_market

            market = mt5_read_only_market
        self.market = market
        self.output_dir = Path(output_dir or Path("backtest_output") / "trading")
        self.spread_points = spread_points
        self.preview_engine_factory = preview_engine_factory
        if strategy_resolver is None:
            from web.strategy_file import strategy_path_for_symbol

            strategy_resolver = strategy_path_for_symbol
        self.strategy_resolver = strategy_resolver

    @staticmethod
    def _mode_template() -> dict[str, Any]:
        return {
            "checked_m15": 0,
            "raw_trigger_points": 0,
            "raw_pa_trigger_points": 0,
            "raw_factor_trigger_points": 0,
            "qualified_candidate_points": 0,
            "codex": {
                "approved": 0,
                "rejected": 0,
                "failures": 0,
                "cache_hits": 0,
                "cache_misses": 0,
            },
            "placed": 0,
            "filled": 0,
            "unfilled": 0,
            "expired_cancelled": 0,
            "side_distribution": {"long": 0, "short": 0},
            "order_type_distribution": {
                "market": 0,
                "limit": 0,
                "stop": 0,
                "stop_limit": 0,
            },
            "pa_family_distribution": {},
            "tp1": 0,
            "tp2": 0,
            "sl": 0,
            "reverse_exit": 0,
            "stop_limit_triggered": 0,
            "realized_legs": [],
            "basic_r": {"total": 0.0, "average": 0.0, "wins": 0, "losses": 0, "flat": 0},
            "open_at_end": 0,
        }

    @staticmethod
    def _reviewer_identity() -> dict[str, str]:
        from .codex_runtime import (
            CODEX_REASONING_EFFORT,
            CODEX_REVIEW_MODEL,
            CODEX_SDK_VERSION,
            CODEX_SERVICE_TIER,
        )

        return {
            "sdk_version": CODEX_SDK_VERSION,
            "model": CODEX_REVIEW_MODEL,
            "reasoning_effort": CODEX_REASONING_EFFORT,
            "service_tier": CODEX_SERVICE_TIER,
            "review_schema_version": REVIEW_SCHEMA_VERSION,
            "cache_schema_version": REVIEW_CACHE_VERSION,
            "prompt_contract_version": REVIEW_PROMPT_CONTRACT_VERSION,
        }

    @classmethod
    def empty_report_for_test(
        cls, *, config: TradingConfigV1, end_close: int
    ) -> dict[str, Any]:
        from .fusion import resolve_fusion_weights

        weights = resolve_fusion_weights(config.module_selection())
        window = seven_natural_day_window(end_close)
        return {
            "schema_version": BACKTEST_REPORT_VERSION,
            "engine_version": BACKTEST_ENGINE_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "data": {
                "timezone": "UTC",
                "latest_m15_close": end_close,
                "window": asdict(window),
                "series": {timeframe: {} for timeframe in ("H1", "M15", "M5")},
            },
            "config": config.to_payload(),
            "module_selection": {
                "direction": list(config.direction_module_ids),
                "entry": list(config.entry_module_ids),
                "pa_families": list(config.pa_family_ids),
            },
            "strategy_fingerprints": {},
            "resolved_weights": asdict(weights),
            "reviewer_identity": cls._reviewer_identity(),
            "modes": {
                mode: {
                    **cls._mode_template(),
                    "config": replace(config, mode=mode).to_payload(),
                }
                for mode in ("rules", "rules_codex")
            },
            "assumptions": {
                "timezone": "UTC",
                "spread_cost": (
                    f"deterministic {0.0:g} price-unit placeholder; overridden by runner"
                ),
                "intrabar_order": "adverse-first when M5 ordering is unknowable",
                "execution": "research-only OHLC simulation; no broker adapter",
                "basic_r": (
                    "each leg keeps raw R; r_contribution is raw R times its configured "
                    "lot share; total/average/win-loss aggregate realized contributions "
                    "per trade"
                ),
            },
        }

    def _resolve_strategy_inputs(
        self, config: TradingConfigV1
    ) -> tuple[
        dict[str, Any], dict[str, int], dict[str, LoadedAlphaStrategyV1]
    ]:
        enabled = set(config.direction_module_ids) | set(config.entry_module_ids)
        alpha_ids = {"H1": "alpha_h1", "M15": "alpha_m15", "M5": "alpha_m5"}
        identities: dict[str, Any] = {}
        frozen: dict[str, LoadedAlphaStrategyV1] = {}
        warmups = {timeframe: self._PA_WARMUP for timeframe in ("H1", "M15", "M5")}
        for timeframe, module_id in alpha_ids.items():
            if module_id not in enabled:
                continue
            strategy = load_alpha_strategy(
                self.strategy_resolver(config.symbol, timeframe),
                expected_symbol=config.symbol,
                expected_timeframe=timeframe,
            )
            warmups[timeframe] = max(
                self._PA_WARMUP, formula_warmup_bars(len(strategy.formula_tokens))
            )
            identities[timeframe] = {
                "strategy_fingerprint": strategy.strategy_fingerprint,
                "dataset_fingerprint": strategy.dataset_fingerprint,
                "formula_token_count": len(strategy.formula_tokens),
                "formula_tokens_sha256": hashlib.sha256(
                    json.dumps(list(strategy.formula_tokens), separators=(",", ":")).encode()
                ).hexdigest(),
            }
            frozen[timeframe] = strategy
        return identities, warmups, frozen

    def _acquire(
        self, config: TradingConfigV1, warmups: Mapping[str, int]
    ) -> tuple[SevenDayWindowV1, dict[str, TimeframeSeriesV1]]:
        latest_series = self.market.fetch_series(config.symbol, "M15", 3)
        if not latest_series.bars:
            raise RuntimeError("MT5 returned no fully closed M15 bars")
        latest = bar_close_timestamp(latest_series.bars[-1], "M15")
        window = seven_natural_day_window(latest)
        evaluation_counts = {"H1": 7 * 24, "M15": 7 * 24 * 4, "M5": 7 * 24 * 12}
        series = {
            timeframe: self.market.fetch_series(
                config.symbol, timeframe, evaluation_counts[timeframe] + warmups[timeframe]
            )
            for timeframe in ("H1", "M15", "M5")
        }
        return window, series

    @staticmethod
    def _selected_plan(decision: Any) -> OrderPlanCandidateV1 | None:
        selected = decision.review.selected_plan_id
        return next((plan for plan in decision.plans if plan.plan_id == selected), None)

    def run(
        self,
        config: TradingConfigV1,
        *,
        modes: tuple[str, ...] = ("rules", "rules_codex"),
        stop_event: Event,
        progress: Callable[[str, int, int, int | None], None],
    ) -> dict[str, Any] | None:
        from .codex_runtime import OpenAICodexRuntime
        from .decision_reviewer import CodexDecisionReviewer, MechanicalDecisionReviewer
        from .fusion import resolve_fusion_weights
        from web.trading_preview import TradingPreviewEngine

        strategy_fingerprints, requested_warmups, frozen_strategies = (
            self._resolve_strategy_inputs(config)
        )
        window, series = self._acquire(config, requested_warmups)
        constraints = self.market.instrument_constraints(config.symbol)
        decision_closes = tuple(
            bar_close_timestamp(bar, "M15")
            for bar in series["M15"].bars
            if window.start_close <= bar_close_timestamp(bar, "M15") <= window.end_close
        )
        m15_close_prices = {
            bar_close_timestamp(bar, "M15"): bar.close for bar in series["M15"].bars
        }
        report = self.empty_report_for_test(config=config, end_close=window.end_close)
        report["assumptions"]["spread_cost"] = (
            f"fixed {self.spread_points:g} price units for the post-TP1 break-even buffer; "
            "plan entry/TP/SL fill at stated levels with no commission or slippage because "
            "historical OHLC has no bid/ask quotes"
        )
        report["data"]["series"] = {
            timeframe: canonical_dataset_identity(
                value,
                end_close=window.end_close,
                requested_warmup=requested_warmups[timeframe],
                evaluation_start=window.start_close,
            )
            for timeframe, value in series.items()
        }
        insufficient = [
            timeframe
            for timeframe, identity in report["data"]["series"].items()
            if identity["available_warmup"] < identity["requested_warmup"]
        ]
        if insufficient:
            detail = ", ".join(
                f"{timeframe} {report['data']['series'][timeframe]['available_warmup']}"
                f"/{report['data']['series'][timeframe]['requested_warmup']}"
                for timeframe in insufficient
            )
            raise RuntimeError(f"insufficient historical warmup bars: {detail}")
        report["strategy_fingerprints"] = strategy_fingerprints
        report["resolved_weights"] = asdict(resolve_fusion_weights(config.module_selection()))
        reviewer_identity = self._reviewer_identity()
        report["reviewer_identity"] = reviewer_identity

        m5_bars = tuple(
            (bar_close_timestamp(bar, "M5"), bar)
            for bar in series["M5"].bars
            if bar_close_timestamp(bar, "M5") <= window.end_close
        )
        for mode in modes:
            if stop_event.is_set():
                return None
            replay = ReplaySeriesMarket(series, constraints)
            cache: CachedDecisionReviewer | None = None

            def reviewer_factory(requested_mode: str) -> DecisionReviewer:
                nonlocal cache
                if requested_mode == "rules":
                    return MechanicalDecisionReviewer()
                cache = CachedDecisionReviewer(
                    CodexDecisionReviewer(OpenAICodexRuntime()),
                    self.output_dir / "review_cache" / "rules_codex_reviews.json",
                    reviewer_identity=reviewer_identity,
                )
                return cache

            factory = self.preview_engine_factory or TradingPreviewEngine
            def frozen_strategy_loader(
                symbol: str, timeframe: str
            ) -> LoadedAlphaStrategyV1:
                if symbol != config.symbol or timeframe not in frozen_strategies:
                    raise ValueError(f"frozen strategy unavailable: {symbol} {timeframe}")
                return frozen_strategies[timeframe]

            engine = factory(
                market=replay,
                reviewer_factory=reviewer_factory,
                strategy_loader=frozen_strategy_loader,
            )
            mode_config = replace(config, mode=mode)
            simulator = HistoricalExecutionSimulator(
                mode_config, spread_points=self.spread_points
            )
            stats = self._mode_template()
            stats["config"] = mode_config.to_payload()
            m5_index = 0
            previous_close: int | None = None
            for index, decision_close in enumerate(decision_closes, start=1):
                if stop_event.is_set():
                    return None
                if previous_close is not None:
                    while m5_index < len(m5_bars) and m5_bars[m5_index][0] <= decision_close:
                        close, bar = m5_bars[m5_index]
                        if close > previous_close:
                            simulator.process_bar(bar, close_timestamp=close)
                        m5_index += 1
                simulator.before_decision(decision_close)
                replay.set_decision_close(decision_close)
                decision = engine.evaluate_at(mode_config, decision_close)
                stats["checked_m15"] += 1
                pa_trigger = bool(decision.pa_evidence)
                factor_trigger = any(
                    score is not None and abs(score) > 0 for _, score in decision.alpha_scores
                )
                stats["raw_pa_trigger_points"] += int(pa_trigger)
                stats["raw_factor_trigger_points"] += int(factor_trigger)
                if pa_trigger or factor_trigger:
                    stats["raw_trigger_points"] += 1
                if decision.fusion_accepted and decision.plans:
                    stats["qualified_candidate_points"] += 1
                if mode == "rules_codex" and decision.fusion_accepted and decision.plans:
                    if decision.review.reason_code.startswith("reviewer_"):
                        stats["codex"]["failures"] += 1
                    elif decision.review.verdict == "approve":
                        stats["codex"]["approved"] += 1
                    else:
                        stats["codex"]["rejected"] += 1
                selected = self._selected_plan(decision)
                if selected is not None:
                    simulator.submit(
                        selected,
                        decision_close=decision_close,
                        market_price=m15_close_prices[decision_close],
                    )
                previous_close = decision_close
                progress(mode, index, len(decision_closes), decision_close)
            stats.update(simulator.finish())
            if cache is not None:
                stats["codex"]["cache_hits"] = cache.hits
                stats["codex"]["cache_misses"] = cache.misses
            report["modes"][mode] = stats
        immutable, latest = write_backtest_report(
            report, self.output_dir, end_close=window.end_close
        )
        return {**report, "report_path": str(immutable), "latest_report_path": str(latest)}
