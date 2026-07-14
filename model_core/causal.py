import torch


def causal_rolling_zscore(x: torch.Tensor, window: int = 200) -> torch.Tensor:
    """Return a causal rolling z-score for a two-dimensional ``[N, T]`` tensor.

    Warm-up windows contain only values that have actually appeared. Left-side
    padding exists solely to form windows and is excluded from the statistics.
    """
    if (
        x.ndim != 2
        or isinstance(window, bool)
        or not isinstance(window, int)
        or window < 1
    ):
        raise ValueError(
            "causal_rolling_zscore expects [N,T] and integer window >= 1"
        )

    pad = torch.zeros(
        x.shape[0], window - 1, dtype=x.dtype, device=x.device
    )
    valid_pad = torch.zeros(
        x.shape[0], window - 1, dtype=torch.bool, device=x.device
    )
    windows = torch.cat([pad, x], dim=1).unfold(1, window, 1)
    valid = torch.cat(
        [valid_pad, torch.ones_like(x, dtype=torch.bool)], dim=1
    ).unfold(1, window, 1)
    count = valid.sum(dim=-1).clamp_min(1).to(x.dtype)
    mean = (windows * valid).sum(dim=-1) / count
    centered = (windows - mean.unsqueeze(-1)) * valid
    variance = centered.square().sum(dim=-1) / count
    std = variance.sqrt()
    zscore = torch.where(
        std > 1e-6,
        (x - mean) / std.clamp_min(1e-6),
        torch.zeros_like(x),
    )
    return torch.nan_to_num(zscore, nan=0.0, posinf=0.0, neginf=0.0)
