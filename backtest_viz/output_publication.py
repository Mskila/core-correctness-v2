"""Atomic backtest report/chart publication boundary."""
from __future__ import annotations


def publish_output_set(publisher, /, *args, **kwargs):
    """Invoke the injected atomic publisher without importing the CLI module."""
    return publisher(*args, **kwargs)


__all__ = ["publish_output_set"]
