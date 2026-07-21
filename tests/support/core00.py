"""CORE-00 characterization assets and deterministic reference trace.

This module intentionally lives under ``tests``.  It observes the frozen core
without changing production execution or treating registered defects as an
accepted correctness oracle.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch

from model_core.features import MT5FeatureEngineer
from model_core.vocab import FORMULA_VOCAB
from model_core.vm import StackVM


TRACE_SCHEMA_VERSION = "core00-reference-trace-v1"
CORE00_REFERENCE_TRACE_SHA256 = {
    ("linux", "2.5.1"): "c93f89fb55efdcc8b9c3ea59d1705208ce163e19fd3e384966a9fca0f90b5e25",
    ("win32", "2.5.1"): "69a0385e7da0e2f78beccf1cb48485f3315ae59beaaa06f80335b8d63a86879a",
    ("win32", "2.13.0"): "eac1b45fd5f979d968d6422deddc538fde2f868c8e96a5bcf9d2e84a4e9c64cc",
}
CORE00_FIXTURE_SHA256 = "2fa10bf75e68532f2177bb2b982842e7a4db470837a782a1004aeed4d0e5468e"
_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _tensor_digest(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _stable_digest(value: Any) -> str:
    digest = hashlib.sha256()

    def visit(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            digest.update(b"tensor:")
            digest.update(_tensor_digest(item).encode("ascii"))
        elif isinstance(item, np.ndarray):
            digest.update(b"ndarray:")
            digest.update(str(item.dtype).encode("ascii"))
            digest.update(str(item.shape).encode("ascii"))
            digest.update(item.tobytes())
        elif isinstance(item, dict):
            digest.update(b"dict{")
            for key in sorted(item, key=lambda candidate: repr(candidate)):
                visit(key)
                visit(item[key])
            digest.update(b"}")
        elif isinstance(item, (list, tuple)):
            digest.update(b"sequence[")
            for child in item:
                visit(child)
            digest.update(b"]")
        else:
            digest.update(type(item).__name__.encode("ascii"))
            digest.update(b":")
            digest.update(repr(item).encode("utf-8"))

    visit(value)
    return digest.hexdigest()


@dataclass(frozen=True)
class Core00Fixture:
    payload: dict[str, Any]
    metadata: dict[str, Any]
    tensors: dict[str, torch.Tensor]
    canonical_bytes: bytes
    canonical_sha256: str
    vm: StackVM

    def feature_tensor(self) -> torch.Tensor:
        return MT5FeatureEngineer.compute_features(
            {name: self.tensors[name] for name in ("open", "high", "low", "close", "volume")}
        )

    @staticmethod
    def tensor_digest(tensor: torch.Tensor) -> str:
        return _tensor_digest(tensor)


def load_ohlcv_fixture() -> Core00Fixture:
    payload = json.loads((_FIXTURES / "core00_ohlcv.json").read_text(encoding="utf-8"))
    canonical_bytes = _canonical_json(payload)
    metadata = dict(payload["metadata"])
    count = len(payload["close"])
    gap_after = set(metadata["gap_after_indices"])
    current = int(metadata["start_timestamp_ns"])
    timestamps: list[int] = []
    for index in range(count):
        timestamps.append(current)
        current += int(metadata["interval_ns"])
        if index in gap_after:
            current += int(metadata["gap_intervals"]) * int(metadata["interval_ns"])
    metadata["exit_timestamps_ns"] = [timestamps[index] for index in metadata["exit_indices"]]
    tensors = {
        name: torch.tensor(payload[name], dtype=torch.float64).unsqueeze(0)
        for name in ("open", "high", "low", "close", "volume")
    }
    tensors["time_ns"] = torch.tensor(timestamps, dtype=torch.int64).unsqueeze(0)
    return Core00Fixture(
        payload=payload,
        metadata=metadata,
        tensors=tensors,
        canonical_bytes=canonical_bytes,
        canonical_sha256=hashlib.sha256(canonical_bytes).hexdigest(),
        vm=StackVM(),
    )


def load_formula_corpus() -> list[dict[str, Any]]:
    return json.loads((_FIXTURES / "core00_formula_corpus.json").read_text(encoding="utf-8"))


def load_known_defects() -> list[dict[str, str]]:
    return json.loads((_FIXTURES / "core00_known_defects.json").read_text(encoding="utf-8"))


def resolve_formula_tokens(token_names: list[str]) -> list[int]:
    lookup = {name: index for index, name in enumerate(FORMULA_VOCAB.token_names)}
    return [lookup[name] for name in token_names]


def _fold_raw_scores(factor: torch.Tensor, close: torch.Tensor) -> list[float]:
    future_return = torch.zeros_like(close)
    future_return[:, :-2] = torch.log(close[:, 2:] / close[:, 1:-1])
    slices = (slice(6, 22), slice(22, 38))
    return [
        float((factor[:, window] * future_return[:, window]).mean().item())
        for window in slices
    ]


def _rng_digest() -> str:
    return _stable_digest(
        {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        }
    )


def trace_digest(trace: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(trace)).hexdigest()


def run_reference_trace(seed: int, steps: int, work_dir: Path) -> dict[str, Any]:
    """Run a tiny deterministic training-like characterization loop."""
    work_dir.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    fixture = load_ohlcv_fixture()
    features = fixture.feature_tensor()
    valid_entries = [item for item in load_formula_corpus() if item["classification"] == "valid"]
    logits = torch.nn.Parameter(torch.linspace(-0.2, 0.2, len(valid_entries), dtype=torch.float64))
    optimizer = torch.optim.Adam([logits], lr=0.01)
    best: dict[str, Any] | None = None
    elite_pool: list[dict[str, Any]] = []
    factor_pool: list[dict[str, Any]] = []
    step_traces: list[dict[str, Any]] = []

    for step in range(steps):
        distribution = torch.distributions.Categorical(logits=logits)
        sampled = distribution.sample((2,))
        log_probabilities = distribution.log_prob(sampled)
        token_names: list[list[str]] = []
        factor_digests: list[str] = []
        fold_scores: list[list[float]] = []
        rewards: list[float] = []
        for sampled_index in sampled.tolist():
            entry = valid_entries[sampled_index]
            names = list(entry["tokens"])
            factor = fixture.vm.execute(resolve_formula_tokens(names), features)
            if factor is None:
                raise AssertionError(f"valid CORE-00 formula failed: {entry['id']}")
            raw_scores = _fold_raw_scores(factor, fixture.tensors["close"])
            reward = float(sum(raw_scores) / len(raw_scores))
            record = {"formula": names, "reward": reward}
            token_names.append(names)
            factor_digests.append(_tensor_digest(factor))
            fold_scores.append(raw_scores)
            rewards.append(reward)
            factor_pool.append(record)
            elite_pool.append(record)
            elite_pool = sorted(elite_pool, key=lambda item: (-item["reward"], item["formula"]))[:4]
            if best is None or (reward, names) > (best["reward"], best["formula"]):
                best = record

        reward_tensor = torch.tensor(rewards, dtype=logits.dtype)
        loss = -(log_probabilities * reward_tensor).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        step_traces.append(
            {
                "step": step,
                "sampled_tokens": token_names,
                "factor_digests": factor_digests,
                "fold_raw_scores": fold_scores,
                "final_rewards": rewards,
            }
        )

    assert best is not None
    return {
        "schema_version": TRACE_SCHEMA_VERSION,
        "seed": seed,
        "steps": step_traces,
        "final_state": {
            "best_formula": best,
            "elite_pool": elite_pool,
            "factor_pool": factor_pool,
            "model_digest": _stable_digest({"logits": logits.detach()}),
            "optimizer_digest": _stable_digest(optimizer.state_dict()),
            "rng_digest": _rng_digest(),
        },
    }


__all__ = [
    "CORE00_FIXTURE_SHA256",
    "CORE00_REFERENCE_TRACE_SHA256",
    "TRACE_SCHEMA_VERSION",
    "load_formula_corpus",
    "load_known_defects",
    "load_ohlcv_fixture",
    "resolve_formula_tokens",
    "run_reference_trace",
    "trace_digest",
]
