"""Training artifact publication boundary."""
from __future__ import annotations


def publish_training_history(engine):
    return engine._save_training_history_live()  # noqa: SLF001


__all__ = ["publish_training_history"]
