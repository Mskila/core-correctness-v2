"""Atomic training-batch transaction boundary."""
from __future__ import annotations


def begin_batch_transaction(
    engine,
    next_step: int,
    run_identity,
    *,
    revalidate=None,
    transaction_type=None,
):
    """Construct the transaction before any stochastic draw in one batch."""
    if revalidate is None or transaction_type is None:
        return engine._begin_batch_transaction(next_step, run_identity)  # noqa: SLF001
    del next_step
    identity = revalidate(
        engine,
        run_identity,
        "training batch before artifact observation",
    )
    return transaction_type(engine, [], identity)


__all__ = ["begin_batch_transaction"]
