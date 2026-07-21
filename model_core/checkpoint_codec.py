"""Checkpoint codec/publication compatibility boundary."""
from __future__ import annotations


def save_checkpoint(engine, step: int, path: str | None = None) -> str:
    if path is None:
        return engine.save_checkpoint(step)
    return engine.save_checkpoint(step, path)


def load_checkpoint(engine, path: str) -> int:
    return engine.load_checkpoint(path)


__all__ = ["load_checkpoint", "save_checkpoint"]
