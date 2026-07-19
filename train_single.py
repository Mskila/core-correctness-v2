"""MT5/cache adapter for strict one-symbol V2 training."""

from __future__ import annotations

import argparse

from data_pipeline.data_manager import MT5DataManager
from data_pipeline.fetcher import MT5DataFetcher
from model_core.config import ModelConfig
from model_core.engine import AlphaEngine
from training_service import run_training_session


def train_single(
    fetcher,
    symbol: str,
    offline: bool = True,
    *,
    from_scratch: bool = False,
    random_seed: int = ModelConfig.RANDOM_SEED,
) -> AlphaEngine:
    manager = MT5DataManager(fetcher)
    manager.load(symbols=[symbol])
    return run_training_session(
        manager,
        source_path=None,
        from_scratch=from_scratch,
        random_seed=random_seed,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train one V2 strategy from MT5 cache")
    parser.add_argument("symbol")
    parser.add_argument("--offline", action="store_true", default=True)
    parser.add_argument("--from-scratch", action="store_true")
    parser.add_argument("--random-seed", type=int, default=ModelConfig.RANDOM_SEED)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    with MT5DataFetcher(offline=args.offline) as fetcher:
        train_single(
            fetcher,
            args.symbol,
            args.offline,
            from_scratch=args.from_scratch,
            random_seed=args.random_seed,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
