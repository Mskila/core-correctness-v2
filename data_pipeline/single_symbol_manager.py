"""
data_pipeline/single_symbol_manager.py — 单品种数据视图

将 MT5DataManager 加载的多品种数据切片为单品种视图，
供 AlphaEngine 单品种训练模式使用。

使用方式：
    with MT5DataFetcher() as fetcher:
        mgr = MT5DataManager(fetcher)
        mgr.load()
        for sym in mgr.symbols:
            single = SingleSymbolDataManager(mgr, sym)
            engine = AlphaEngine(data_manager=single)
            engine.train()
"""
from __future__ import annotations

import torch
from loguru import logger

from data_pipeline.validation import DatasetIdentity
from model_core.features import MT5FeatureEngineer
from model_core.semantics import DataValidationError


class SingleSymbolDataManager:
    """单品种数据视图，兼容 AlphaEngine 对 data_manager 的接口。

    AlphaEngine 调用的接口：
        .feat_tensor   → [1, F, T]  (N=1)
        .target_ret    → [1, T]
        .target_valid  → bool [1, T]
        .raw_dict      → {field: [1, T]}
        .symbols       → [symbol]
        .bar_time      → UTC ns, int64 [1, T]
        .data_identity → DatasetIdentity
    """

    def __init__(self, multi_manager, symbol: str) -> None:
        """
        Args:
            multi_manager: 已 load() 的 MT5DataManager 实例。
            symbol:        要切片的品种名，必须在 multi_manager.symbols 中。
        """
        if symbol not in multi_manager.symbols:
            raise DataValidationError(
                f"Symbol '{symbol}' is not available in multi_manager.symbols: "
                f"{multi_manager.symbols}"
            )
        self._multi  = multi_manager
        self._symbol = symbol

        logger.info(
            f"[SingleSymbolDataManager] symbol={symbol}  "
            f"idx={multi_manager.symbols.index(symbol)}"
        )

    def _resolve_index(self) -> int:
        """Resolve the symbol against the manager's current ordering."""
        symbols = self._multi.symbols
        try:
            return symbols.index(self._symbol)
        except ValueError as exc:
            raise DataValidationError(
                f"Symbol '{self._symbol}' is not available in current manager "
                f"symbols: {symbols}"
            ) from exc

    # ── AlphaEngine 所需接口 ──────────────────────────────────────────────

    @property
    def symbols(self) -> list[str]:
        return [self._symbol]

    @property
    def raw_dict(self) -> dict:
        idx = self._resolve_index()
        full = self._multi.raw_dict
        return {k: v[idx:idx+1] for k, v in full.items()}  # [1, T]

    @property
    def feat_tensor(self) -> torch.Tensor:
        """返回 [1, F, T] 特征张量（只含目标品种）。"""
        raw = self.raw_dict
        return MT5FeatureEngineer.compute_features(raw)   # [1, F, T]

    @property
    def target_ret(self) -> torch.Tensor:
        idx = self._resolve_index()
        full = self._multi.target_ret
        return full[idx:idx+1]   # [1, T]

    @property
    def target_valid(self) -> torch.Tensor:
        idx = self._resolve_index()
        full = self._multi.target_valid
        return full[idx:idx+1]   # bool [1, T]

    @property
    def bar_time(self) -> torch.Tensor:
        idx = self._resolve_index()
        full = self._multi.bar_time
        return full[idx:idx+1]   # [1, T]

    @property
    def data_identity(self) -> DatasetIdentity:
        idx = self._resolve_index()
        return self._multi.data_identities[idx]

    @property
    def symbol(self) -> str:
        return self._symbol
