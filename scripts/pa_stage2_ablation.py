"""Run the fixed, pre-holdout Stage-2 XAUUSD fusion ablation.

This tool is read-only with respect to market data and strategy artifacts.  It
does not simulate broker fills and cannot submit orders.  Its only writes are
the explicitly requested JSON and Markdown reports.
"""

from __future__ import annotations

import argparse
import bisect
import json
import statistics
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_pipeline.validation import CanonicalDataset, canonicalize_ohlcv
from model_core.artifacts import StrategyArtifact
from model_core.walk_forward import formula_warmup_bars
from trading_core import (
    CandidateFreshnessStateV1,
    ClosedBarV1,
    FusionInputsV1,
    OrderPlanGenerationV1,
    PAObservationV1,
    TimeframeSeriesV1,
    bar_close_timestamp,
    default_ablation_variants,
    evaluate_alpha_observations_at_indices,
    evaluate_alpha_observations_causal_prefix,
    evaluate_pa_observations_at_indices,
    freeze_ablation_window,
    fuse_signals,
    generate_order_plans,
    load_alpha_strategy,
    resolve_fusion_weights,
)


@dataclass(frozen=True)
class _Bundle:
    timeframe: str
    data_path: Path
    strategy_path: Path
    canonical: CanonicalDataset
    series: TimeframeSeriesV1
    strategy: Any


def _iso(timestamp: int) -> str:
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat().replace("+00:00", "Z")


def _load_bundle(
    *,
    data_path: Path,
    strategy_path: Path,
    timeframe: str,
    tick: float,
) -> _Bundle:
    frame = pd.read_parquet(data_path)
    canonical = canonicalize_ohlcv(
        frame,
        symbol="XAUUSD",
        timeframe=timeframe,
        numeric_time_unit="s",
        gap_policy="segment",
    )
    artifact = StrategyArtifact.from_dict(
        json.loads(strategy_path.read_text(encoding="utf-8"))
    )
    expected_dataset = artifact.run_identity.artifact_identity.training_dataset
    if canonical.identity != expected_dataset:
        raise RuntimeError(
            f"{timeframe} data identity does not exactly match its strategy artifact"
        )
    strategy = load_alpha_strategy(
        strategy_path,
        expected_symbol="XAUUSD",
        expected_timeframe=timeframe,
    )
    rows = canonical.frame
    time_seconds = rows["time"].astype("int64").to_numpy() // 1_000_000_000
    bars = tuple(
        ClosedBarV1(
            timestamp=int(time_seconds[index]),
            open=float(row.open),
            high=float(row.high),
            low=float(row.low),
            close=float(row.close),
            volume=float(row.volume),
        )
        for index, row in enumerate(rows.itertuples(index=False))
    )
    return _Bundle(
        timeframe=timeframe,
        data_path=data_path,
        strategy_path=strategy_path,
        canonical=canonical,
        series=TimeframeSeriesV1(
            symbol="XAUUSD",
            timeframe=timeframe,
            bars=bars,
            tick=tick,
        ),
        strategy=strategy,
    )


def _close_times(series: TimeframeSeriesV1) -> tuple[int, ...]:
    return tuple(bar_close_timestamp(bar, series.timeframe) for bar in series.bars)


def _anchor_indices(indices: tuple[int, ...], count: int = 5) -> tuple[int, ...]:
    if len(indices) <= count:
        return indices
    positions = sorted({round(step * (len(indices) - 1) / (count - 1)) for step in range(count)})
    return tuple(indices[position] for position in positions)


def _alpha_map(
    bundle: _Bundle,
    end_indices: tuple[int, ...],
) -> tuple[dict[int, Any], dict[str, Any]]:
    started = time.perf_counter()
    observations = evaluate_alpha_observations_causal_prefix(
        bundle.strategy,
        bundle.series,
        end_indices=end_indices,
    )
    anchors = _anchor_indices(end_indices)
    rolling = evaluate_alpha_observations_at_indices(
        bundle.strategy,
        bundle.series,
        end_indices=anchors,
        batch_size=min(16, max(1, len(anchors))),
    )
    causal_by_index = {index: item for index, item in zip(end_indices, observations)}
    causal_anchors = tuple(causal_by_index[index] for index in anchors)
    if causal_anchors != rolling:
        raise RuntimeError(
            f"{bundle.timeframe} causal-prefix Alpha path does not match exact rolling live path"
        )
    return (
        {item.bar_close_timestamp: item for item in observations},
        {
            "timeframe": bundle.timeframe,
            "checked_indices": list(anchors),
            "checked_count": len(anchors),
            "exact": True,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        },
    )


def _seed_and_evaluate_pa(
    series: TimeframeSeriesV1,
    *,
    first_index: int,
    last_index: int,
    seed_bars: int = 50,
) -> tuple[tuple[int, ...], tuple[PAObservationV1 | None, ...], dict[str, Any]]:
    started = time.perf_counter()
    seed_start = max(0, first_index - seed_bars)
    seed_indices = tuple(range(seed_start, first_index))
    state = CandidateFreshnessStateV1()
    if seed_indices:
        _, state = evaluate_pa_observations_at_indices(
            series,
            end_indices=seed_indices,
            previous=state,
            emit_new=False,
        )
    indices = tuple(range(first_index, last_index + 1))
    observations, state = evaluate_pa_observations_at_indices(
        series,
        end_indices=indices,
        previous=state,
        emit_new=True,
    )
    return (
        indices,
        observations,
        {
            "timeframe": series.timeframe,
            "seed_start_index": seed_start,
            "first_index": first_index,
            "last_index": last_index,
            "evaluated_bars": len(indices),
            "valid_observations": sum(item is not None for item in observations),
            "seen_candidate_keys": len(state.seen_keys),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        },
    )


def _evaluate_h1_pa(
    series: TimeframeSeriesV1,
    indices: tuple[int, ...],
) -> tuple[dict[int, PAObservationV1 | None], dict[str, Any]]:
    started = time.perf_counter()
    observations, _ = evaluate_pa_observations_at_indices(
        series,
        end_indices=indices,
        previous=CandidateFreshnessStateV1(),
        emit_new=False,
    )
    return (
        {
            bar_close_timestamp(series.bars[index], series.timeframe): observation
            for index, observation in zip(indices, observations)
        },
        {
            "timeframe": series.timeframe,
            "evaluated_bars": len(indices),
            "valid_observations": sum(item is not None for item in observations),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        },
    )


def _new_occurrences(
    observations: tuple[PAObservationV1 | None, ...],
) -> tuple[Any, ...]:
    output = []
    seen: set[tuple[str, str, str, int, str]] = set()
    for observation in observations:
        if observation is None:
            continue
        for occurrence in observation.candidates:
            if not occurrence.is_new:
                continue
            candidate = occurrence.candidate
            key = (
                candidate.family,
                candidate.direction,
                candidate.setup_type,
                candidate.trigger_at,
                candidate.source_policy,
            )
            if key not in seen:
                seen.add(key)
                output.append(occurrence)
    return tuple(output)


def _empty_stats() -> dict[str, Any]:
    return {
        "decisions": 0,
        "accepted": 0,
        "long": 0,
        "short": 0,
        "plans": 0,
        "direction_labeled": 0,
        "direction_hits": 0,
        "direction_signed_returns": [],
        "entry_labeled": 0,
        "entry_hits": 0,
        "entry_signed_returns": [],
        "mfe_atr": [],
        "mae_atr": [],
        "reject_counts": Counter(),
        "plan_styles": Counter(),
        "order_types": Counter(),
        "families": Counter(),
    }


def _record(
    stats: dict[str, Any],
    *,
    signal,
    plans,
    direction_return: float | None,
    entry_return: float | None,
    mfe_atr: float | None,
    mae_atr: float | None,
) -> None:
    stats["decisions"] += 1
    if not signal.accepted:
        stats["reject_counts"].update(signal.reject_reasons)
        return
    stats["accepted"] += 1
    stats[signal.side] += 1
    stats["plans"] += len(plans.plans)
    if plans.reject_reasons:
        stats["reject_counts"].update(f"plan:{reason}" for reason in plans.reject_reasons)
    stats["plan_styles"].update(plan.style for plan in plans.plans)
    stats["order_types"].update(plan.order_type for plan in plans.plans)
    stats["families"].update(plan.family for plan in plans.plans)
    if direction_return is not None:
        stats["direction_labeled"] += 1
        stats["direction_hits"] += int(direction_return > 0.0)
        stats["direction_signed_returns"].append(direction_return)
    if entry_return is not None:
        stats["entry_labeled"] += 1
        stats["entry_hits"] += int(entry_return > 0.0)
        stats["entry_signed_returns"].append(entry_return)
    if mfe_atr is not None:
        stats["mfe_atr"].append(mfe_atr)
    if mae_atr is not None:
        stats["mae_atr"].append(mae_atr)


def _mean(values: list[float]) -> float | None:
    return float(statistics.fmean(values)) if values else None


def _finalize(stats: dict[str, Any]) -> dict[str, Any]:
    mean_direction_return = _mean(stats["direction_signed_returns"])
    mean_entry_return = _mean(stats["entry_signed_returns"])
    return {
        "decisions": stats["decisions"],
        "accepted": stats["accepted"],
        "acceptance_rate": (
            stats["accepted"] / stats["decisions"] if stats["decisions"] else None
        ),
        "long": stats["long"],
        "short": stats["short"],
        "plans": stats["plans"],
        "direction_labeled": stats["direction_labeled"],
        "direction_hit_rate": (
            stats["direction_hits"] / stats["direction_labeled"]
            if stats["direction_labeled"]
            else None
        ),
        "mean_direction_signed_return_bps": (
            mean_direction_return * 10_000
            if mean_direction_return is not None
            else None
        ),
        "entry_labeled": stats["entry_labeled"],
        "entry_hit_rate": (
            stats["entry_hits"] / stats["entry_labeled"]
            if stats["entry_labeled"]
            else None
        ),
        "mean_entry_signed_return_bps": (
            mean_entry_return * 10_000
            if mean_entry_return is not None
            else None
        ),
        "mean_mfe_atr": _mean(stats["mfe_atr"]),
        "mean_mae_atr": _mean(stats["mae_atr"]),
        "reject_counts": dict(stats["reject_counts"].most_common()),
        "plan_styles": dict(stats["plan_styles"].most_common()),
        "order_types": dict(stats["order_types"].most_common()),
        "families": dict(stats["families"].most_common()),
    }


def _clearly_better(default_blocks: list[dict[str, Any]], other_blocks: list[dict[str, Any]]) -> bool:
    wins = 0
    comparable = 0
    for baseline, candidate in zip(default_blocks, other_blocks):
        if min(baseline["direction_labeled"], candidate["direction_labeled"]) < 30:
            continue
        comparable += 1
        hit_better = (
            candidate["direction_hit_rate"] is not None
            and baseline["direction_hit_rate"] is not None
            and candidate["direction_hit_rate"] >= baseline["direction_hit_rate"] + 0.02
        )
        return_not_worse = (
            candidate["mean_direction_signed_return_bps"] is not None
            and baseline["mean_direction_signed_return_bps"] is not None
            and candidate["mean_direction_signed_return_bps"]
            >= baseline["mean_direction_signed_return_bps"]
        )
        adverse_not_worse = (
            candidate["mean_mae_atr"] is not None
            and baseline["mean_mae_atr"] is not None
            and candidate["mean_mae_atr"] >= baseline["mean_mae_atr"] - 0.05
        )
        wins += int(hit_better and return_not_worse and adverse_not_worse)
    return comparable >= 3 and wins >= 3


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Stage 2 XAUUSD 多周期融合消融",
        "",
        f"生成时间：`{report['generated_at']}`",
        "",
        "本报告仅使用冻结点以前的数据验证权重方向；保留七天未参与计算或排名。",
        "",
        "## 数据与窗口",
        "",
        "| 周期 | K 线 | 缺口 | 数据指纹 | 策略指纹 | 身份 |",
        "|---|---:|---:|---|---|---|",
    ]
    for timeframe in ("H1", "M15", "M5"):
        item = report["datasets"][timeframe]
        lines.append(
            f"| {timeframe} | {item['bars']} | {item['gap_count']} | "
            f"`{item['data_fingerprint'][:12]}` | `{item['strategy_fingerprint'][:12]}` | exact |"
        )
    window = report["window"]
    lines.extend(
        [
            "",
            f"- 首个预留前决策：`{window['preholdout_start']}`",
            f"- 冻结点：`{window['cutoff']}`",
            f"- 最后完整决策：`{window['latest_complete']}`",
            f"- 预留前 M15：{window['preholdout_decisions']}",
            f"- 保留七天 M15：{window['holdout_decisions']}（未读取其结果指标）",
            "",
            "## 固定消融结果",
            "",
            "| 变体 | 合格决策 | 方向命中 | 方向均值(bps) | 入场命中 | MFE/ATR | MAE/ATR | 计划数 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )

    def fmt(value: Any, digits: int = 4) -> str:
        return "—" if value is None else f"{value:.{digits}f}"

    for variant in report["variants"]:
        overall = variant["overall"]
        lines.append(
            f"| {variant['name']} | {overall['accepted']} | "
            f"{fmt(overall['direction_hit_rate'])} | "
            f"{fmt(overall['mean_direction_signed_return_bps'], 3)} | "
            f"{fmt(overall['entry_hit_rate'])} | {fmt(overall['mean_mfe_atr'])} | "
            f"{fmt(overall['mean_mae_atr'])} | {overall['plans']} |"
        )
    assessment = report["assessment"]
    lines.extend(
        [
            "",
            "## 裁决",
            "",
            f"- 是否有充分证据修改权重：`{str(assessment['weight_change_recommended']).lower()}`",
            f"- 结论：{assessment['summary_zh']}",
            "- M5 Alpha 仍默认关闭，除非固定 `with_m5_alpha` 变体满足三段一致改善门槛。",
            "- 现有 Alpha 策略训练区间与本次诊断区间重叠，因此结果不是严格独立 OOS；阶段 6 的最新七天才是最终门禁。",
            "",
            "仅供研究和复盘使用，不构成投资建议或交易指令。",
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    print("[stage2] loading and validating exact dataset identities", flush=True)
    bundles = {
        "H1": _load_bundle(
            data_path=args.h1_data,
            strategy_path=args.h1_strategy,
            timeframe="H1",
            tick=args.tick,
        ),
        "M15": _load_bundle(
            data_path=args.m15_data,
            strategy_path=args.m15_strategy,
            timeframe="M15",
            tick=args.tick,
        ),
        "M5": _load_bundle(
            data_path=args.m5_data,
            strategy_path=args.m5_strategy,
            timeframe="M5",
            tick=args.tick,
        ),
    }
    closes = {name: _close_times(bundle.series) for name, bundle in bundles.items()}
    m5_by_close = {value: index for index, value in enumerate(closes["M5"])}
    required = {
        name: formula_warmup_bars(len(bundle.strategy.formula_tokens))
        for name, bundle in bundles.items()
    }
    decision_rows: list[tuple[int, int, int, int]] = []
    for m15_index, close_timestamp in enumerate(closes["M15"]):
        if m15_index < required["M15"] - 1:
            continue
        m5_index = m5_by_close.get(close_timestamp)
        if m5_index is None or m5_index < required["M5"] - 1:
            continue
        h1_index = bisect.bisect_right(closes["H1"], close_timestamp) - 1
        if h1_index < required["H1"] - 1:
            continue
        decision_rows.append((close_timestamp, h1_index, m15_index, m5_index))
    window = freeze_ablation_window(
        tuple(row[0] for row in decision_rows), holdout_days=args.holdout_days
    )
    pre_set = set(window.preholdout_closes)
    pre_rows = tuple(row for row in decision_rows if row[0] in pre_set)
    if not pre_rows:
        raise RuntimeError("preholdout decision window is empty")
    print(
        f"[stage2] frozen preholdout={len(pre_rows)} holdout={len(window.holdout_closes)} "
        f"cutoff={_iso(window.cutoff_close)}",
        flush=True,
    )

    h1_indices = tuple(sorted({row[1] for row in pre_rows}))
    m15_indices = tuple(row[2] for row in pre_rows)
    m5_indices = tuple(row[3] for row in pre_rows)
    alpha_maps: dict[str, dict[int, Any]] = {}
    alpha_checks = []
    for timeframe, indices in (("H1", h1_indices), ("M15", m15_indices), ("M5", m5_indices)):
        print(f"[stage2] Alpha {timeframe}: causal prefix + exact rolling anchors", flush=True)
        alpha_maps[timeframe], check = _alpha_map(bundles[timeframe], indices)
        alpha_checks.append(check)

    print("[stage2] PA H1 snapshots", flush=True)
    h1_pa, h1_pa_stats = _evaluate_h1_pa(bundles["H1"].series, h1_indices)
    print("[stage2] PA M15 sequential freshness", flush=True)
    m15_eval_indices, m15_eval, m15_pa_stats = _seed_and_evaluate_pa(
        bundles["M15"].series,
        first_index=m15_indices[0],
        last_index=m15_indices[-1],
    )
    m15_pa_by_index = dict(zip(m15_eval_indices, m15_eval))
    first_m5_close = pre_rows[0][0] - 900
    first_m5_eval = bisect.bisect_right(closes["M5"], first_m5_close)
    print("[stage2] PA M5 sequential timing", flush=True)
    m5_eval_indices, m5_eval, m5_pa_stats = _seed_and_evaluate_pa(
        bundles["M5"].series,
        first_index=first_m5_eval,
        last_index=m5_indices[-1],
    )
    m5_pa_by_index = dict(zip(m5_eval_indices, m5_eval))

    variants = default_ablation_variants()
    raw_stats: dict[str, dict[str, Any]] = {
        variant.name: {
            "overall": _empty_stats(),
            "blocks": [_empty_stats() for _ in range(4)],
        }
        for variant in variants
    }
    trigger_counts: Counter[str] = Counter()
    previous_decision = pre_rows[0][0] - 900
    m5_cursor = first_m5_eval
    print("[stage2] fusing fixed variants on preholdout decisions", flush=True)
    for row_number, (decision_close, h1_index, m15_index, m5_index) in enumerate(pre_rows):
        h1_close = closes["H1"][h1_index]
        m15_pa = m15_pa_by_index.get(m15_index)
        interval_m5: list[PAObservationV1 | None] = []
        while m5_cursor <= m5_index:
            close_timestamp = closes["M5"][m5_cursor]
            if close_timestamp > previous_decision:
                interval_m5.append(m5_pa_by_index.get(m5_cursor))
            m5_cursor += 1
        latest_m5_pa = m5_pa_by_index.get(m5_index)
        if latest_m5_pa is not None:
            latest_m5_pa = replace(
                latest_m5_pa,
                candidates=_new_occurrences(tuple(interval_m5)),
            )
        if m15_pa is not None:
            trigger_counts.update(
                f"M15:{item.candidate.family}:{item.candidate.direction}"
                for item in m15_pa.candidates
                if item.is_new
            )
        if latest_m5_pa is not None:
            trigger_counts.update(
                f"M5:{item.candidate.family}:{item.candidate.direction}"
                for item in latest_m5_pa.candidates
                if item.is_new
            )
        inputs = FusionInputsV1(
            symbol="XAUUSD",
            decision_close_timestamp=decision_close,
            h1_alpha=alpha_maps["H1"].get(h1_close),
            h1_pa=h1_pa.get(h1_close),
            m15_alpha=alpha_maps["M15"].get(decision_close),
            m15_pa=m15_pa,
            m5_pa=latest_m5_pa,
            m5_alpha=alpha_maps["M5"].get(decision_close),
        )
        block = min(3, row_number * 4 // len(pre_rows))
        for variant in variants:
            signal = fuse_signals(inputs, variant.selection)
            plans = (
                generate_order_plans(signal, m15_pa)
                if m15_pa is not None
                else OrderPlanGenerationV1(
                    plans=(), reject_reasons=("missing_m15_pa",)
                )
            )
            direction_return = None
            entry_return = None
            mfe_atr = None
            mae_atr = None
            if signal.accepted and signal.side is not None and m15_pa is not None:
                sign = 1.0 if signal.side == "long" else -1.0
                current_close = bundles["M15"].series.bars[m15_index].close
                future_m15 = m15_index + 4
                if (
                    future_m15 < len(bundles["M15"].series.bars)
                    and closes["M15"][future_m15] <= window.cutoff_close
                ):
                    future_close = bundles["M15"].series.bars[future_m15].close
                    direction_return = sign * (future_close / current_close - 1.0)
                next_m15 = m15_index + 1
                if (
                    next_m15 < len(bundles["M15"].series.bars)
                    and closes["M15"][next_m15] <= window.cutoff_close
                ):
                    future_close = bundles["M15"].series.bars[next_m15].close
                    entry_return = sign * (future_close / current_close - 1.0)
                future_m5_indices = tuple(range(m5_index + 1, min(m5_index + 4, len(closes["M5"]))))
                if (
                    len(future_m5_indices) == 3
                    and closes["M5"][future_m5_indices[-1]] <= window.cutoff_close
                ):
                    future_bars = [bundles["M5"].series.bars[index] for index in future_m5_indices]
                    if signal.side == "long":
                        mfe_atr = (max(bar.high for bar in future_bars) - current_close) / m15_pa.atr
                        mae_atr = (min(bar.low for bar in future_bars) - current_close) / m15_pa.atr
                    else:
                        mfe_atr = (current_close - min(bar.low for bar in future_bars)) / m15_pa.atr
                        mae_atr = (current_close - max(bar.high for bar in future_bars)) / m15_pa.atr
            for target in (raw_stats[variant.name]["overall"], raw_stats[variant.name]["blocks"][block]):
                _record(
                    target,
                    signal=signal,
                    plans=plans,
                    direction_return=direction_return,
                    entry_return=entry_return,
                    mfe_atr=mfe_atr,
                    mae_atr=mae_atr,
                )
        previous_decision = decision_close

    variant_reports: list[dict[str, Any]] = []
    for variant in variants:
        variant_reports.append(
            {
                "name": variant.name,
                "selection": {
                    "direction_modules": sorted(variant.selection.direction_modules),
                    "entry_modules": sorted(variant.selection.entry_modules),
                },
                "weights": asdict(resolve_fusion_weights(variant.selection)),
                "overall": _finalize(raw_stats[variant.name]["overall"]),
                "blocks": [_finalize(item) for item in raw_stats[variant.name]["blocks"]],
            }
        )
    default_blocks = variant_reports[0]["blocks"]
    contradictions = [
        variant["name"]
        for variant in variant_reports[1:]
        if _clearly_better(default_blocks, variant["blocks"])
    ]
    recommend = bool(contradictions)
    assessment = {
        "weight_change_recommended": recommend,
        "contradicting_variants": contradictions,
        "rule": (
            "at least 30 labeled accepted decisions per comparable block; hit rate +2pp, "
            "mean signed return not worse and MAE/ATR not worse in at least 3 of 4 blocks"
        ),
        "summary_zh": (
            "固定消融出现跨至少三段的一致反证，必须先人工审批新权重。"
            if recommend
            else "没有达到修改默认权重的预注册反证门槛，保持已批准权重。"
        ),
    }
    report = {
        "schema_version": "stage2-ablation-v1",
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "symbol": "XAUUSD",
        "decision_timeframe": "M15",
        "policy": {
            "direction_threshold": 0.60,
            "entry_threshold": 0.65,
            "holdout_days": args.holdout_days,
            "pa_short_closure_seconds": 5_400,
            "holdout_used_for_metrics": False,
            "strict_independent_oos": False,
        },
        "datasets": {
            timeframe: {
                "data_file": bundle.data_path.name,
                "strategy_file": bundle.strategy_path.name,
                "bars": bundle.canonical.identity.bars,
                "gap_count": bundle.canonical.gap_count,
                "data_fingerprint": bundle.canonical.identity.data_fingerprint,
                "strategy_fingerprint": bundle.strategy.strategy_fingerprint,
                "identity_match": True,
            }
            for timeframe, bundle in bundles.items()
        },
        "window": {
            "preholdout_start": _iso(window.preholdout_closes[0]),
            "preholdout_end": _iso(window.preholdout_closes[-1]),
            "cutoff": _iso(window.cutoff_close),
            "latest_complete": _iso(window.latest_decision_close),
            "preholdout_decisions": len(window.preholdout_closes),
            "holdout_decisions": len(window.holdout_closes),
        },
        "alpha_parity": alpha_checks,
        "pa_evaluation": [h1_pa_stats, m15_pa_stats, m5_pa_stats],
        "raw_pa_triggers": dict(trigger_counts.most_common()),
        "variants": variant_reports,
        "assessment": assessment,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.markdown_output.write_text(_markdown(report), encoding="utf-8")
    print(
        f"[stage2] complete elapsed={report['elapsed_seconds']}s "
        f"weight_change_recommended={recommend}",
        flush=True,
    )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h1-data", type=Path, required=True)
    parser.add_argument("--m15-data", type=Path, required=True)
    parser.add_argument("--m5-data", type=Path, required=True)
    parser.add_argument("--h1-strategy", type=Path, required=True)
    parser.add_argument("--m15-strategy", type=Path, required=True)
    parser.add_argument("--m5-strategy", type=Path, required=True)
    parser.add_argument("--tick", type=float, default=0.01)
    parser.add_argument("--holdout-days", type=int, default=7)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    return parser


def main() -> int:
    run(_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
