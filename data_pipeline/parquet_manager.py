"""Load canonical training data from a single Parquet K-line file."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from loguru import logger

from data_pipeline.data_manager import compute_forward_open_returns
from data_pipeline.validation import (
    DatasetIdentity,
    assert_minimum_bars,
    canonicalize_ohlcv,
)
from model_core.features import MT5FeatureEngineer


_TIMEFRAMES = ("M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1", "MN1")
_PARQUET_RE = re.compile(
    rf"^(.+)_({'|'.join(_TIMEFRAMES)})\.parquet$",
    re.IGNORECASE,
)
_SECONDS_PER_YEAR = 365.2425 * 24 * 60 * 60


def parse_parquet_filename(path: str | Path) -> tuple[str, str]:
    """Parse ``{symbol}_{timeframe}.parquet``."""
    name = Path(path).name
    match = _PARQUET_RE.match(name)
    if not match:
        raise ValueError(
            f"文件名须为 {{品种}}_{{周期}}.parquet，例如 AAPL_H1.parquet；当前: {name}"
        )
    return match.group(1), match.group(2).upper()


def inspect_parquet_file(path: str | Path) -> dict[str, Any]:
    """Inspect a Parquet file only after full canonical validation."""
    parquet_path = Path(path)
    if not parquet_path.exists():
        raise FileNotFoundError(f"文件不存在: {parquet_path}")
    if parquet_path.suffix.lower() != ".parquet":
        raise ValueError("请选择 .parquet 文件")

    symbol, timeframe = parse_parquet_filename(parquet_path)
    dataset = canonicalize_ohlcv(
        pd.read_parquet(parquet_path),
        symbol=symbol,
        timeframe=timeframe,
    )
    span_seconds = (
        dataset.identity.end_time_ns - dataset.identity.start_time_ns
    ) / 1_000_000_000
    years = round(span_seconds / _SECONDS_PER_YEAR, 2)
    return {
        "data_file": str(parquet_path.resolve()),
        "filename": parquet_path.name,
        "symbol": symbol,
        "timeframe": dataset.identity.timeframe,
        "bars": dataset.identity.bars,
        "years_h1": years,
        "valid": True,
        "message": "",
    }


class ParquetDataManager:
    """Single-symbol canonical data manager backed by one Parquet file.

    ``target_ret``, ``target_valid`` and ``bar_time`` all use shape ``[1, T]``.
    ``bar_time`` contains UTC nanoseconds. A bar minimum is enforced only when
    ``required_bars`` is explicitly supplied by the caller.
    """

    def __init__(
        self,
        file_path: str | Path,
        required_bars: int | None = None,
    ) -> None:
        self.file_path = Path(file_path)
        self.symbol, self.timeframe = parse_parquet_filename(self.file_path)
        self.required_bars = required_bars
        self._clear_loaded_state()

    def _clear_loaded_state(self) -> None:
        self._raw_dict: dict[str, torch.Tensor] | None = None
        self._target_ret: torch.Tensor | None = None
        self._target_valid: torch.Tensor | None = None
        self._data_identities: tuple[DatasetIdentity, ...] | None = None

    def load(self) -> None:
        self._clear_loaded_state()
        dataset = canonicalize_ohlcv(
            pd.read_parquet(self.file_path),
            symbol=self.symbol,
            timeframe=self.timeframe,
        )
        if self.required_bars is not None:
            assert_minimum_bars(
                dataset.identity.bars,
                self.required_bars,
                context=f"Parquet dataset {self.file_path.name}",
            )

        frame = dataset.frame
        raw = {
            field: torch.tensor(
                frame[field].to_numpy(dtype=np.float64)[None, :],
                dtype=torch.float32,
            )
            for field in ("open", "high", "low", "close", "volume")
        }
        raw["time"] = torch.tensor(
            frame["time"].astype("int64").to_numpy(dtype=np.int64)[None, :],
            dtype=torch.int64,
        )

        target_ret, target_valid = compute_forward_open_returns(raw["open"])
        self._raw_dict = raw
        self._target_ret = target_ret
        self._target_valid = target_valid
        self._data_identities = (dataset.identity,)
        logger.info(
            f"[数据] 已加载 {self.symbol} {self.timeframe}，"
            f"共 {raw['open'].shape[1]} 根K线，文件 {self.file_path.name}"
        )

    @property
    def symbols(self) -> list[str]:
        return [self.symbol]

    def _ensure_loaded(self) -> None:
        if (
            self._raw_dict is None
            or self._target_ret is None
            or self._target_valid is None
            or self._data_identities is None
        ):
            raise RuntimeError("Data not loaded. Call ParquetDataManager.load() first.")

    @property
    def raw_dict(self) -> dict[str, torch.Tensor]:
        self._ensure_loaded()
        return self._raw_dict  # type: ignore[return-value]

    @property
    def feat_tensor(self) -> torch.Tensor:
        return MT5FeatureEngineer.compute_features(self.raw_dict)

    @property
    def target_ret(self) -> torch.Tensor:
        self._ensure_loaded()
        return self._target_ret  # type: ignore[return-value]

    @property
    def target_valid(self) -> torch.Tensor:
        self._ensure_loaded()
        return self._target_valid  # type: ignore[return-value]

    @property
    def bar_time(self) -> torch.Tensor:
        return self.raw_dict["time"]

    @property
    def data_identities(self) -> tuple[DatasetIdentity, ...]:
        self._ensure_loaded()
        return self._data_identities  # type: ignore[return-value]
