"""Parquet adapter for strict one-symbol V2 training."""

from __future__ import annotations

import argparse
from pathlib import Path

from data_pipeline.parquet_manager import ParquetDataManager
from model_core.config import ModelConfig
from model_core.engine import AlphaEngine
from training_service import run_training_session


def train_from_file(
    data_file: str,
    *,
    from_scratch: bool = False,
    random_seed: int = ModelConfig.RANDOM_SEED,
) -> AlphaEngine:
    manager = ParquetDataManager(data_file)
    manager.load()
    return run_training_session(
        manager,
        source_path=Path(data_file).resolve(),
        from_scratch=from_scratch,
        random_seed=random_seed,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train one V2 strategy from Parquet")
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--from-scratch", action="store_true")
    parser.add_argument("--random-seed", type=int, default=ModelConfig.RANDOM_SEED)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    train_from_file(
        args.data_file,
        from_scratch=args.from_scratch,
        random_seed=args.random_seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
