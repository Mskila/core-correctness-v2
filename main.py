"""Official V2 training entrypoint."""

from __future__ import annotations

import argparse

from data_pipeline.fetcher import MT5DataFetcher
from model_core.config import ModelConfig
from model_core.semantics import ArtifactCompatibilityError


_MIGRATION = (
    "V2 supports only single-symbol training; use train_file.py, Web, "
    "or train_single.py"
)


def _reject_multi_symbol() -> None:
    raise ArtifactCompatibilityError(_MIGRATION)


def save_group_strategy(*_args, **_kwargs) -> None:
    _reject_multi_symbol()


def train_group(*_args, **_kwargs):
    _reject_multi_symbol()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AlphaMaster V2 training")
    parser.add_argument("--single", metavar="SYMBOL")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--from-scratch", action="store_true")
    parser.add_argument("--random-seed", type=int, default=ModelConfig.RANDOM_SEED)
    parser.add_argument("--group")
    parser.add_argument("--cross-section", action="store_true")
    return parser


def main(argv: list[str] | None = None):
    args = _parser().parse_args(argv)
    if args.single is None or args.group is not None or args.cross_section:
        _reject_multi_symbol()
    from train_single import train_single

    with MT5DataFetcher(offline=args.offline) as fetcher:
        return train_single(
            fetcher,
            args.single,
            args.offline,
            from_scratch=args.from_scratch,
            random_seed=args.random_seed,
        )


if __name__ == "__main__":
    main()
