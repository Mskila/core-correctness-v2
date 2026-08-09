"""Stage-4 read-only M15 trading preview orchestration."""

from __future__ import annotations

import asyncio
import threading
import time
from collections import deque
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Protocol

from model_core.walk_forward import formula_warmup_bars
from trading_core import (
    ALPHA_H1,
    ALPHA_M15,
    M5_ALPHA,
    CandidateFreshnessStateV1,
    CodexDecisionReviewer,
    DecisionReviewer,
    FusionInputsV1,
    InstrumentConstraintsV1,
    TimeframeSeriesV1,
    MechanicalDecisionReviewer,
    LoadedAlphaStrategyV1,
    OpenAICodexRuntime,
    OrderPlanGenerationV1,
    PAObservationV1,
    TradeDecisionV1,
    TradingConfigV1,
    align_decision_frame,
    bar_close_timestamp,
    build_decision_review_request,
    build_pa_observation,
    evaluate_alpha_observation,
    fuse_signals,
    generate_order_plans,
    load_alpha_strategy,
)
from web.settings import load_settings, save_settings
from web.strategy_file import strategy_path_for_symbol
from web.trading_execution import trading_execution_controller
from web.trading_mt5 import mt5_read_only_market


_PA_HISTORY_BARS = 384
_POLL_SECONDS = 10.0


class PreviewEngine(Protocol):
    def latest_m15_close(self, symbol: str) -> int:
        ...

    def evaluate_at(
        self,
        config: TradingConfigV1,
        decision_close_timestamp: int,
    ) -> TradeDecisionV1:
        ...


class DecisionConsumer(Protocol):
    def process_decision(self, decision: TradeDecisionV1) -> dict[str, Any]:
        ...


class ReadOnlyMarket(Protocol):
    def fetch_series(
        self,
        symbol: str,
        timeframe: str,
        count: int,
    ) -> TimeframeSeriesV1:
        ...

    def instrument_constraints(self, symbol: str) -> InstrumentConstraintsV1:
        ...


def _default_strategy_resolver(symbol: str, timeframe: str) -> Path:
    return strategy_path_for_symbol(symbol, timeframe)


def _filter_pa_families(
    observation: PAObservationV1 | None,
    families: frozenset[str],
) -> PAObservationV1 | None:
    if observation is None:
        return None
    return replace(
        observation,
        candidates=tuple(
            occurrence
            for occurrence in observation.candidates
            if occurrence.candidate.family in families
        ),
    )


def _candidate_evidence(timeframe: str, observation: PAObservationV1 | None) -> tuple[str, ...]:
    if observation is None:
        return ()
    return tuple(
        ":".join(
            (
                timeframe,
                occurrence.candidate.family,
                occurrence.candidate.setup_type,
                occurrence.candidate.direction,
                "new" if occurrence.is_new else "existing",
            )
        )
        for occurrence in observation.candidates
    )


class TradingPreviewEngine:
    """Build one non-executable decision from closed H1/M15/M5 MT5 bars."""

    def __init__(
        self,
        market: ReadOnlyMarket = mt5_read_only_market,
        *,
        strategy_resolver: Callable[[str, str], Path] = _default_strategy_resolver,
        reviewer_factory: Callable[[str], DecisionReviewer] | None = None,
    ) -> None:
        self.market = market
        self.strategy_resolver = strategy_resolver
        self.reviewer_factory = reviewer_factory or self._default_reviewer
        self._m15_freshness = CandidateFreshnessStateV1()
        self._m5_freshness = CandidateFreshnessStateV1()

    @staticmethod
    def _default_reviewer(mode: str) -> DecisionReviewer:
        if mode == "rules":
            return MechanicalDecisionReviewer()
        return CodexDecisionReviewer(OpenAICodexRuntime())

    def latest_m15_close(self, symbol: str) -> int:
        series = self.market.fetch_series(symbol, "M15", 3)
        if not series.bars:
            raise RuntimeError(f"MT5 未返回 {symbol} 已收盘 M15 K 线")
        return bar_close_timestamp(series.bars[-1], "M15")

    def _load_alpha_strategies(
        self,
        config: TradingConfigV1,
    ) -> tuple[dict[str, LoadedAlphaStrategyV1], list[str]]:
        enabled = set(config.direction_module_ids) | set(config.entry_module_ids)
        requested = []
        if ALPHA_H1 in enabled:
            requested.append("H1")
        if ALPHA_M15 in enabled:
            requested.append("M15")
        if M5_ALPHA in enabled:
            requested.append("M5")
        strategies: dict[str, LoadedAlphaStrategyV1] = {}
        warnings: list[str] = []
        for timeframe in requested:
            try:
                path = self.strategy_resolver(config.symbol, timeframe)
                strategies[timeframe] = load_alpha_strategy(
                    path,
                    expected_symbol=config.symbol,
                    expected_timeframe=timeframe,
                )
            except Exception as exc:  # noqa: BLE001 - missing input becomes no-trade evidence
                warnings.append(f"alpha_{timeframe.lower()}:{exc}")
        return strategies, warnings

    @staticmethod
    def _history_count(strategy: LoadedAlphaStrategyV1 | None) -> int:
        if strategy is None:
            return _PA_HISTORY_BARS
        return max(
            _PA_HISTORY_BARS,
            formula_warmup_bars(len(strategy.formula_tokens)),
        )

    def evaluate_at(
        self,
        config: TradingConfigV1,
        decision_close_timestamp: int,
    ) -> TradeDecisionV1:
        strategies, warnings = self._load_alpha_strategies(config)
        h1 = self.market.fetch_series(
            config.symbol,
            "H1",
            self._history_count(strategies.get("H1")),
        )
        m15 = self.market.fetch_series(
            config.symbol,
            "M15",
            self._history_count(strategies.get("M15")),
        )
        m5 = self.market.fetch_series(
            config.symbol,
            "M5",
            self._history_count(strategies.get("M5")),
        )
        frame = align_decision_frame(
            symbol=config.symbol,
            decision_close_timestamp=decision_close_timestamp,
            h1=h1,
            m15=m15,
            m5=m5,
        )

        def alpha(timeframe: str, series):
            strategy = strategies.get(timeframe)
            if strategy is None:
                return None
            try:
                return evaluate_alpha_observation(
                    strategy,
                    series,
                    decision_close_timestamp=decision_close_timestamp,
                )
            except Exception as exc:  # noqa: BLE001 - input error is an explicit rejection
                warnings.append(f"alpha_{timeframe.lower()}:{exc}")
                return None

        h1_alpha = alpha("H1", h1)
        m15_alpha = alpha("M15", m15)
        m5_alpha = alpha("M5", m5)

        def pa(timeframe: str, series, previous, *, emit_new: bool):
            try:
                return build_pa_observation(
                    series,
                    decision_close_timestamp=decision_close_timestamp,
                    previous=previous,
                    emit_new=emit_new,
                )
            except Exception as exc:  # noqa: BLE001 - insufficient PA history rejects safely
                warnings.append(f"pa_{timeframe.lower()}:{exc}")
                return None, previous

        h1_pa, _ = pa(
            "H1",
            h1,
            CandidateFreshnessStateV1(),
            emit_new=False,
        )
        m15_pa, next_m15_freshness = pa(
            "M15",
            m15,
            self._m15_freshness,
            emit_new=True,
        )
        m5_pa, next_m5_freshness = pa(
            "M5",
            m5,
            self._m5_freshness,
            emit_new=True,
        )
        self._m15_freshness = next_m15_freshness
        self._m5_freshness = next_m5_freshness

        families = frozenset(config.pa_family_ids)
        h1_pa = _filter_pa_families(h1_pa, families)
        m15_pa = _filter_pa_families(m15_pa, families)
        m5_pa = _filter_pa_families(m5_pa, families)
        inputs = FusionInputsV1(
            symbol=config.symbol,
            decision_close_timestamp=decision_close_timestamp,
            h1_alpha=h1_alpha,
            h1_pa=h1_pa,
            m15_alpha=m15_alpha,
            m15_pa=m15_pa,
            m5_pa=m5_pa,
            m5_alpha=m5_alpha,
        )
        signal = fuse_signals(inputs, config.module_selection())
        if m15_pa is None:
            generation = OrderPlanGenerationV1(
                plans=(),
                reject_reasons=("insufficient_m15_pa",),
            )
        else:
            generation = generate_order_plans(
                signal,
                m15_pa,
                constraints=self.market.instrument_constraints(config.symbol),
            )
        request = build_decision_review_request(inputs, signal, generation)
        reviewer = self.reviewer_factory(config.mode)
        review = asyncio.run(reviewer.review(request))
        evidence = (
            *_candidate_evidence("H1", h1_pa),
            *_candidate_evidence("M15", m15_pa),
            *_candidate_evidence("M5", m5_pa),
        )
        return TradeDecisionV1(
            decision_id=request.decision_id,
            symbol=config.symbol,
            decision_close_timestamp=decision_close_timestamp,
            h1_close_timestamp=frame.h1.last_close_timestamp,
            m15_close_timestamp=frame.m15.last_close_timestamp,
            m5_close_timestamp=frame.m5.last_close_timestamp,
            config=config,
            alpha_scores=(
                ("H1", None if h1_alpha is None else h1_alpha.position),
                ("M15", None if m15_alpha is None else m15_alpha.position),
                ("M5", None if m5_alpha is None else m5_alpha.position),
            ),
            pa_scores=(
                ("H1", None if h1_pa is None else h1_pa.direction_score),
                ("M15", None if m15_pa is None else m15_pa.direction_score),
                ("M5", None if m5_pa is None else m5_pa.direction_score),
            ),
            pa_evidence=evidence,
            input_warnings=tuple(warnings),
            direction_score=signal.direction_score,
            entry_score=signal.entry_score,
            side=signal.side,
            fusion_accepted=signal.accepted,
            fusion_reject_reasons=signal.reject_reasons,
            plan_reject_reasons=generation.reject_reasons,
            plans=generation.plans,
            review=review,
            final_action=(
                "preview_approved"
                if review.verdict == "approve" and review.selected_plan_id is not None
                else "preview_rejected"
            ),
            created_at=time.time(),
        )


class TradingPreviewManager:
    """Run the preview once for each newly observed closed M15 bar."""

    def __init__(
        self,
        engine: PreviewEngine | None = None,
        *,
        settings_loader: Callable[[], dict[str, Any]] = load_settings,
        settings_saver: Callable[[dict[str, Any]], dict[str, Any]] = save_settings,
        poll_seconds: float = _POLL_SECONDS,
        decision_consumer: DecisionConsumer | None = None,
    ) -> None:
        self.engine = engine or TradingPreviewEngine()
        self._settings_loader = settings_loader
        self._settings_saver = settings_saver
        self._poll_seconds = poll_seconds
        self._decision_consumer = decision_consumer
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._running = False
        self._processing = False
        self._loaded = False
        self._desired_config = TradingConfigV1.default()
        self._active_config: TradingConfigV1 | None = None
        self._last_m15_close: int | None = None
        self._last_error = ""
        self._decisions: deque[TradeDecisionV1] = deque(maxlen=100)
        self._execution_results: dict[str, dict[str, Any]] = {}

    def load_persisted(self) -> None:
        with self._lock:
            if self._loaded:
                return
            self._loaded = True
        try:
            payload = self._settings_loader().get("trading_config") or {}
            config = TradingConfigV1.from_payload(payload) if payload else TradingConfigV1.default()
        except Exception:
            config = TradingConfigV1.default()
        with self._lock:
            self._desired_config = config

    def config_state(self) -> dict[str, Any]:
        self.load_persisted()
        with self._lock:
            active_hash = (
                None if self._active_config is None else self._active_config.config_hash
            )
            return {
                "config": self._desired_config.to_payload(),
                "active_config_hash": active_hash,
                "pending_next_m15": active_hash != self._desired_config.config_hash,
            }

    def update_config(self, config: TradingConfigV1) -> dict[str, Any]:
        self.load_persisted()
        self._settings_saver({"trading_config": config.identity_payload()})
        with self._lock:
            self._desired_config = config
        return self.config_state()

    def start(self) -> dict[str, Any]:
        self.load_persisted()
        with self._lock:
            if self._running and self._thread and self._thread.is_alive():
                return self.status()
            self._running = True
            self._last_error = ""
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._loop,
                name="trading-preview",
                daemon=True,
            )
            self._thread.start()
        return self.status()

    def stop(self) -> dict[str, Any]:
        with self._lock:
            self._running = False
            self._stop_event.set()
        return self.status()

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            self.run_once()
            self._stop_event.wait(self._poll_seconds)
        with self._lock:
            self._running = False

    def run_once(self) -> bool:
        self.load_persisted()
        try:
            with self._lock:
                config = self._desired_config
            decision_close = self.engine.latest_m15_close(config.symbol)
        except Exception as exc:  # noqa: BLE001 - operational status, retried on next poll
            with self._lock:
                self._last_error = str(exc)
            return False
        with self._lock:
            if self._processing or decision_close == self._last_m15_close:
                return False
            self._processing = True
            self._last_m15_close = decision_close
            self._active_config = config
        try:
            decision = self.engine.evaluate_at(config, decision_close)
            execution_result = None
            if self._decision_consumer is not None:
                raw_execution = self._decision_consumer.process_decision(decision)
                execution_result = {
                    "action": raw_execution.get("action"),
                    "execution_enabled": raw_execution.get("execution_enabled", False),
                    "receipts": list(raw_execution.get("receipts") or ()),
                }
            with self._lock:
                self._decisions.append(decision)
                if execution_result is not None:
                    self._execution_results[decision.decision_id] = execution_result
                    retained = {row.decision_id for row in self._decisions}
                    self._execution_results = {
                        key: value
                        for key, value in self._execution_results.items()
                        if key in retained
                    }
                self._last_error = ""
            return True
        except Exception as exc:  # noqa: BLE001 - exactly one failed attempt per closed M15
            with self._lock:
                self._last_error = str(exc)
            return False
        finally:
            with self._lock:
                self._processing = False

    def decisions(self, *, limit: int = 20) -> list[dict[str, Any]]:
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("limit must be in [1, 100]")
        with self._lock:
            selected = list(self._decisions)[-limit:]
            results = dict(self._execution_results)
        return [
            self._decision_payload(decision, results.get(decision.decision_id))
            for decision in reversed(selected)
        ]

    @staticmethod
    def _decision_payload(
        decision: TradeDecisionV1,
        execution_result: dict[str, Any] | None,
    ) -> dict[str, Any]:
        payload = decision.to_payload()
        if execution_result is None:
            return payload
        receipts = list(execution_result.get("receipts") or ())
        confirmed = [receipt for receipt in receipts if receipt.get("confirmed")]
        tickets = [
            ticket
            for receipt in confirmed
            for ticket in (
                receipt.get("position_ticket"),
                receipt.get("order_ticket"),
            )
            if ticket is not None
        ]
        latest = receipts[-1] if receipts else {}
        action = str(execution_result.get("action") or "execution_unknown")
        payload["final_action"] = action
        payload["execution"] = {
            "enabled": bool(execution_result.get("execution_enabled")),
            "action": action,
            "confirmed": bool(confirmed) if receipts else action in {
                "decision_rejected",
                "execution_disabled",
                "no_pyramiding",
                "reverse_cooldown",
            },
            "tickets": tickets,
            "ticket": tickets[-1] if tickets else None,
            "retcode": latest.get("retcode"),
            "receipts": receipts,
        }
        return payload

    def status(self) -> dict[str, Any]:
        self.load_persisted()
        with self._lock:
            latest = (
                self._decision_payload(
                    self._decisions[-1],
                    self._execution_results.get(self._decisions[-1].decision_id),
                )
                if self._decisions
                else None
            )
            active_hash = (
                None if self._active_config is None else self._active_config.config_hash
            )
            return {
                "stage": 4,
                "read_only": True,
                "execution_enabled": False,
                "running": self._running,
                "processing": self._processing,
                "last_m15_close": self._last_m15_close,
                "last_error": self._last_error,
                "desired_config_hash": self._desired_config.config_hash,
                "active_config_hash": active_hash,
                "pending_next_m15": active_hash != self._desired_config.config_hash,
                "decision_count": len(self._decisions),
                "latest_decision": latest,
            }


trading_preview_engine = TradingPreviewEngine()
trading_preview_manager = TradingPreviewManager(
    trading_preview_engine,
    decision_consumer=trading_execution_controller,
)
