"""Validated MT5 OHLCV loading with strict real-timestamp alignment."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from loguru import logger

from config import Config
from data_pipeline.fetcher import MT5DataFetcher
from data_pipeline.validation import (
    DatasetIdentity,
    canonicalize_ohlcv,
    float32_ohlcv_arrays,
    normalize_timeframe_name,
)
from model_core.semantics import DataValidationError, DatasetAlignmentError


def compute_forward_open_returns(
    open_prices: torch.Tensor,
    segment_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the V2 t+1-open to t+2-open label and validity mask."""
    if open_prices.ndim != 2 or open_prices.shape[1] < 3:
        raise DataValidationError("at least 3 bars are required for t+2 labels")
    if not torch.is_floating_point(open_prices):
        raise DataValidationError("open prices must use a floating point dtype")
    if not torch.isfinite(open_prices).all():
        raise DataValidationError("open prices must be finite")
    if (open_prices <= 0).any():
        raise DataValidationError("open prices must be positive")

    work_dtype = (
        torch.float64
        if open_prices.dtype in (torch.float16, torch.bfloat16, torch.float32)
        else open_prices.dtype
    )
    work = open_prices.to(dtype=work_dtype)
    local_reciprocals = torch.reciprocal(work[:, 1:])
    if (
        not torch.isfinite(local_reciprocals).all()
        or (local_reciprocals > torch.finfo(open_prices.dtype).max).any()
    ):
        raise DataValidationError(
            "open prices cannot guarantee finite gradient in the input dtype"
        )

    previous = work[:, 1:-1]
    following = work[:, 2:]
    difference = following - previous
    close_values = torch.abs(difference) <= (
        torch.maximum(previous, following) * 0.5
    )
    forward_work = torch.empty_like(previous)
    if close_values.any():
        forward_work[close_values] = torch.log1p(
            difference[close_values] / previous[close_values]
        )
    far_values = ~close_values
    if far_values.any():
        forward_work[far_values] = (
            torch.log(following[far_values]) - torch.log(previous[far_values])
        )
    forward_returns = forward_work.to(dtype=open_prices.dtype)
    if not torch.isfinite(forward_returns).all():
        raise DataValidationError("valid forward open returns must be finite")

    values = torch.zeros_like(open_prices)
    valid = torch.zeros_like(open_prices, dtype=torch.bool)
    values[:, :-2] = forward_returns
    valid[:, :-2] = True
    if segment_ids is not None:
        if segment_ids.shape != open_prices.shape or segment_ids.dtype != torch.int64:
            raise DataValidationError(
                "segment_ids must be int64 with the same shape as open prices"
            )
        valid[:, :-2] &= segment_ids[:, 1:-1] == segment_ids[:, 2:]
        values = torch.where(valid, values, torch.zeros_like(values))
    return values, valid


class MT5DataManager:
    """Load canonical OHLCV tensors for one or more symbols.

    The retained multi-symbol view is aligned only on timestamps present in every
    source. Public V2 tensors use shapes ``[N, T]``: ``target_ret``,
    ``target_valid`` and ``bar_time``. ``bar_time`` stores UTC nanoseconds.
    """

    def __init__(self, fetcher: MT5DataFetcher, *, gap_policy: str = "segment") -> None:
        self._fetcher = fetcher
        self._gap_policy = gap_policy
        self._requested_symbols: list[str] | None = None
        self._clear_loaded_state()

    def _clear_loaded_state(self) -> None:
        self._symbols: list[str] = []
        self._raw_dict: dict[str, torch.Tensor] | None = None
        self._target_ret: torch.Tensor | None = None
        self._target_valid: torch.Tensor | None = None
        self._data_identities: tuple[DatasetIdentity, ...] | None = None
        self._segment_ids: torch.Tensor | None = None

    def load(self, symbols: list[str] | None = None) -> None:
        """Fetch, validate and strictly align requested symbols."""
        self._clear_loaded_state()
        symbol_list = list(symbols) if symbols is not None else list(Config.SYMBOLS)
        if not symbol_list:
            raise DataValidationError("at least one symbol is required")
        if any(
            not isinstance(symbol, str) or not symbol.strip()
            for symbol in symbol_list
        ):
            raise DataValidationError("symbols must be unique non-empty strings")
        if len(set(symbol_list)) != len(symbol_list):
            raise DataValidationError("duplicate symbols are not allowed")
        timeframe = normalize_timeframe_name(Config.TIMEFRAME)
        logger.info(f"Loading data for {len(symbol_list)} symbols: {symbol_list}")

        canonical_frames: dict[str, pd.DataFrame] = {}
        for symbol in symbol_list:
            raw_frame = self._fetcher.fetch(
                symbol, Config.TIMEFRAME, Config.BARS_COUNT
            )
            dataset = canonicalize_ohlcv(
                raw_frame,
                symbol=symbol,
                timeframe=timeframe,
                numeric_time_unit="s",
                gap_policy=self._gap_policy,
            )
            canonical_frames[symbol] = dataset.frame

        aligned = self._align_timelines(canonical_frames)
        aligned_datasets = {
            symbol: canonicalize_ohlcv(
                aligned[symbol],
                symbol=symbol,
                timeframe=timeframe,
                gap_policy=self._gap_policy,
            )
            for symbol in symbol_list
        }

        aligned_frames = {
            symbol: aligned_datasets[symbol].frame for symbol in symbol_list
        }
        raw_dict = self._build_raw_dict(aligned_frames, symbol_list)
        segment_ids = torch.tensor(
            np.stack([aligned_datasets[symbol].segment_ids for symbol in symbol_list]),
            dtype=torch.int64,
        )
        target_ret, target_valid = compute_forward_open_returns(
            raw_dict["open"], segment_ids
        )
        data_identities = tuple(
            aligned_datasets[symbol].identity for symbol in symbol_list
        )

        self._symbols = symbol_list
        self._raw_dict = raw_dict
        self._target_ret = target_ret
        self._target_valid = target_valid
        self._data_identities = data_identities
        self._segment_ids = segment_ids
        self._requested_symbols = None if symbols is None else list(symbol_list)
        logger.info(
            f"Data loaded. raw_dict shape: N={len(self._symbols)}, "
            f"T={self._raw_dict['open'].shape[1]}"
        )

    def reload(self) -> None:
        """Reload the same explicit symbol selection, or the configured default."""
        requested = (
            list(self._requested_symbols)
            if self._requested_symbols is not None
            else None
        )
        self.load(requested)

    @property
    def raw_dict(self) -> dict[str, torch.Tensor]:
        self._ensure_loaded()
        return {
            field: values.clone()
            for field, values in self._raw_dict.items()  # type: ignore[union-attr]
        }

    @property
    def feat_tensor(self) -> torch.Tensor:
        self._ensure_loaded()
        from model_core.features import MT5FeatureEngineer

        return MT5FeatureEngineer.compute_features(self.raw_dict)

    @property
    def target_ret(self) -> torch.Tensor:
        self._ensure_loaded()
        return self._target_ret.clone()  # type: ignore[union-attr]

    @property
    def target_valid(self) -> torch.Tensor:
        self._ensure_loaded()
        return self._target_valid.clone()  # type: ignore[union-attr]

    @property
    def bar_time(self) -> torch.Tensor:
        self._ensure_loaded()
        return self._raw_dict["time"].clone()  # type: ignore[index]

    @property
    def data_identities(self) -> tuple[DatasetIdentity, ...]:
        self._ensure_loaded()
        return self._data_identities  # type: ignore[return-value]

    @property
    def symbols(self) -> list[str]:
        return list(self._symbols)

    @property
    def segment_ids(self) -> torch.Tensor:
        self._ensure_loaded()
        return self._segment_ids.clone()  # type: ignore[union-attr]

    def _ensure_loaded(self) -> None:
        if (
            not self._symbols
            or self._raw_dict is None
            or self._target_ret is None
            or self._target_valid is None
            or self._data_identities is None
            or self._segment_ids is None
        ):
            raise RuntimeError("Data not loaded. Call MT5DataManager.load() first.")

    @staticmethod
    def _coverage_text(raw_dfs: dict[str, pd.DataFrame]) -> str:
        coverage = []
        for symbol, frame in raw_dfs.items():
            start = frame["time"].iloc[0].isoformat()
            end = frame["time"].iloc[-1].isoformat()
            coverage.append(
                f"{symbol}[start={start}, end={end}, bars={len(frame)}]"
            )
        return "; ".join(coverage)

    def _align_timelines(
        self, raw_dfs: dict[str, pd.DataFrame]
    ) -> dict[str, pd.DataFrame]:
        """Return canonical frames sliced to their exact timestamp intersection."""
        if not raw_dfs:
            raise DatasetAlignmentError("no datasets were provided for alignment")

        frames = list(raw_dfs.values())
        intersection = pd.Index(frames[0]["time"])
        for frame in frames[1:]:
            intersection = intersection.intersection(pd.Index(frame["time"]))
        intersection = intersection.sort_values()

        if len(intersection) < 3:
            raise DatasetAlignmentError(
                "timeline intersection is insufficient for t+2 labels: "
                f"expected at least 3 bars, actual {len(intersection)}; "
                f"coverage: {self._coverage_text(raw_dfs)}"
            )

        aligned: dict[str, pd.DataFrame] = {}
        for symbol, frame in raw_dfs.items():
            indexed = frame.set_index("time", drop=False)
            aligned[symbol] = indexed.loc[intersection].reset_index(drop=True)
        return aligned

    def _build_raw_dict(
        self,
        aligned: dict[str, pd.DataFrame],
        symbols: list[str],
    ) -> dict[str, torch.Tensor]:
        """Convert aligned canonical frames to ``{field: Tensor[N, T]}``."""
        fields = ("open", "high", "low", "close", "volume")
        converted = {
            symbol: float32_ohlcv_arrays(aligned[symbol]) for symbol in symbols
        }
        raw_dict = {
            field: torch.tensor(
                np.stack([converted[symbol][field] for symbol in symbols]),
                dtype=torch.float32,
            )
            for field in fields
        }
        time_rows = np.stack(
            [
                aligned[symbol]["time"].astype("int64").to_numpy(dtype=np.int64)
                for symbol in symbols
            ]
        )
        raw_dict["time"] = torch.tensor(time_rows, dtype=torch.int64)
        return raw_dict

    @staticmethod
    def _compute_target_ret(open_tensor: torch.Tensor) -> torch.Tensor:
        """Compatibility wrapper returning only the V2 label values."""
        return compute_forward_open_returns(open_tensor)[0]
