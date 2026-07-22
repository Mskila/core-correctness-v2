from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor

from .semantics import DataValidationError


_EXECUTION_RESULT_TENSOR_FIELDS = (
    "position",
    "turnover",
    "gross_pnl",
    "cost",
    "net_pnl",
    "target_valid",
    "bar_time_ns",
    "final_liquidation_cost",
)


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    """Immutable execution snapshot with differentiable defensive tensor access."""

    position: Tensor
    turnover: Tensor
    gross_pnl: Tensor
    cost: Tensor
    net_pnl: Tensor
    target_valid: Tensor
    bar_time_ns: Tensor
    final_liquidation_cost: Tensor

    def __post_init__(self) -> None:
        for field_name in _EXECUTION_RESULT_TENSOR_FIELDS:
            value = object.__getattribute__(self, field_name)
            if isinstance(value, Tensor):
                object.__setattr__(self, field_name, value.clone())

    def __getattribute__(self, name: str) -> object:
        value = object.__getattribute__(self, name)
        if name in _EXECUTION_RESULT_TENSOR_FIELDS and isinstance(value, Tensor):
            return value.clone()
        return value

    def _borrow_tensor(self, name: str) -> Tensor:
        """Return an internal read-only-by-contract tensor for core consumers."""
        if name not in _EXECUTION_RESULT_TENSOR_FIELDS:
            raise KeyError(f"unknown execution tensor field: {name}")
        value = object.__getattribute__(self, name)
        if not isinstance(value, Tensor):
            raise TypeError(f"execution field is not a Tensor: {name}")
        return value

    def copy_tensor(self, name: str) -> Tensor:
        """Return an explicit defensive copy of a result tensor."""
        return self._borrow_tensor(name).clone()


@dataclass(frozen=True)
class PerformanceMetrics:
    observations: int
    elapsed_years: float
    periods_per_year: float
    total_return: float
    annualized_return: float
    volatility: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    win_rate: float


@dataclass(frozen=True)
class LedgerEntry:
    symbol: str
    signal_time_ns: int
    entry_time_ns: int
    exit_time_ns: int
    position: float
    gross_pnl: float
    cost: float
    net_pnl: float
    is_final_liquidation: bool


@dataclass(frozen=True)
class ReturnAccounting:
    asset_simple_return: Tensor
    gross_simple_return: Tensor
    portfolio_simple_return: Tensor
    net_log_return: Tensor


@dataclass(frozen=True)
class PositionEventCounts:
    turnover_events: int
    entries: int
    exits: int
    reversals: int
    liquidation_events: int
    display_trades: int


_SECONDS_PER_YEAR = 365.2425 * 24 * 3_600
_NANOSECONDS_PER_SECOND = 1_000_000_000
_MIN_LOG_FLOAT64 = math.log(math.ulp(0.0))
_MAX_LOG_FLOAT64 = math.log(float.fromhex("0x1.fffffffffffffp+1023"))
_SUPPORTED_DEVICE_TYPES = {"cpu", "cuda"}
_SUPPORTED_FLOAT_DTYPES = {
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
}


def _validate_supported_device(device: torch.device, *, context: str) -> None:
    if device.type not in _SUPPORTED_DEVICE_TYPES:
        raise DataValidationError(f"{context} tensors must use CPU or CUDA")


def _validate_supported_float_tensor(value: Tensor, *, name: str) -> None:
    if not value.is_floating_point():
        raise DataValidationError(f"{name} must be a floating-point tensor")
    if value.dtype not in _SUPPORTED_FLOAT_DTYPES:
        raise DataValidationError(
            f"{name} must use a supported floating-point dtype"
        )


def _cost_work_dtype(published_dtype: torch.dtype) -> torch.dtype:
    if published_dtype in {torch.float16, torch.bfloat16}:
        return torch.float32
    return torch.float64


def _validate_cost_component_product(
    working_source: Tensor,
    cost_rate: float,
    working_product: Tensor,
) -> None:
    underflowed = (
        cost_rate != 0.0
        and bool(((working_source != 0) & (working_product == 0)).any())
    )
    if underflowed:
        raise DataValidationError(
            "non-zero execution cost component underflowed in the working dtype"
        )


def _validate_additive_component_preservation(
    first_component: Tensor,
    second_component: Tensor,
    combined_value: Tensor,
    *,
    context: str,
) -> None:
    """Reject arithmetic that completely absorbs either non-zero addend."""
    first_absorbed = (first_component != 0) & (
        combined_value == second_component
    )
    second_absorbed = (second_component != 0) & (
        combined_value == first_component
    )
    if bool((first_absorbed | second_absorbed).any()):
        raise DataValidationError(
            f"each non-zero {context} component must be representable "
            "and preserved by arithmetic"
        )


def factor_to_position(factors: Tensor, *, min_exposure: float) -> Tensor:
    """Convert finite factors to continuous positions with a neutral band."""
    if not math.isfinite(min_exposure) or not 0.0 <= min_exposure < 1.0:
        raise DataValidationError("min_exposure must satisfy 0 <= min_exposure < 1")
    _validate_supported_device(factors.device, context="factors")
    _validate_supported_float_tensor(factors, name="factors")
    if not bool(torch.isfinite(factors).all()):
        raise DataValidationError("factors must contain only finite values")

    raw_position = torch.tanh(factors)
    return torch.where(
        raw_position.abs() < min_exposure,
        torch.zeros_like(raw_position),
        raw_position,
    )


def net_log_return_from_position(
    position: Tensor,
    target_log_return: Tensor,
    cost: Tensor,
) -> ReturnAccounting:
    """Account for one position path in simple-return equity units."""
    if (
        position.shape != target_log_return.shape
        or position.shape != cost.shape
        or position.device != target_log_return.device
        or position.device != cost.device
    ):
        raise DataValidationError(
            "position, target_log_return, and cost must share shape and device"
        )
    for name, value in (
        ("position", position),
        ("target_log_return", target_log_return),
        ("cost", cost),
    ):
        _validate_supported_float_tensor(value, name=name)
        if not bool(torch.isfinite(value).all()):
            raise DataValidationError(f"{name} must contain only finite values")
    if bool((cost < 0).any()):
        raise DataValidationError("cost must be a non-negative equity fraction")

    work_dtype = _cost_work_dtype(position.dtype)
    work_position = position.to(work_dtype)
    asset_simple = torch.expm1(target_log_return.to(work_dtype))
    if not bool(torch.isfinite(asset_simple).all()):
        raise DataValidationError(
            "net PnL additive component preservation failed because the "
            "asset simple return is non-finite"
        )
    gross_simple = work_position * asset_simple
    negative_cost = -cost.to(work_dtype)
    portfolio_simple = gross_simple + negative_cost
    _validate_additive_component_preservation(
        gross_simple,
        negative_cost,
        portfolio_simple,
        context="portfolio simple return",
    )
    if bool((portfolio_simple <= -1.0).any()):
        raise DataValidationError(
            "portfolio simple return reached the insolvency boundary"
        )
    net_log = torch.log1p(portfolio_simple)
    if bool(((portfolio_simple != 0.0) & (net_log == 0.0)).any()):
        raise DataValidationError(
            "non-zero portfolio simple return underflowed in log1p"
        )
    if not all(
        bool(torch.isfinite(value).all())
        for value in (gross_simple, portfolio_simple, net_log)
    ):
        raise DataValidationError("execution return accounting must remain finite")
    return ReturnAccounting(
        asset_simple_return=asset_simple,
        gross_simple_return=gross_simple,
        portfolio_simple_return=portfolio_simple,
        net_log_return=net_log,
    )


def classify_position_events(position: Tensor) -> PositionEventCounts:
    """Count economic position changes separately from display trades."""
    if position.ndim != 1 or position.numel() == 0 or not position.is_floating_point():
        raise DataValidationError("position event input must be one floating vector")
    if not bool(torch.isfinite(position).all()):
        raise DataValidationError("position event input must be finite")
    previous = torch.cat([torch.zeros_like(position[:1]), position[:-1]])
    changed = position != previous
    previous_sign = torch.sign(previous)
    current_sign = torch.sign(position)
    entries = int(((previous_sign == 0) & (current_sign != 0)).sum().item())
    exits = int(((previous_sign != 0) & (current_sign == 0)).sum().item())
    reversals = int(
        (
            (previous_sign != 0)
            & (current_sign != 0)
            & (previous_sign != current_sign)
        ).sum().item()
    )
    liquidation_events = int(current_sign[-1].item() != 0)
    return PositionEventCounts(
        turnover_events=int(changed.sum().item()),
        entries=entries,
        exits=exits + liquidation_events,
        reversals=reversals,
        liquidation_events=liquidation_events,
        display_trades=entries + reversals,
    )


def _validate_execution_inputs(
    *,
    factors: Tensor,
    target_ret: Tensor,
    target_valid: Tensor,
    bar_time_ns: Tensor,
    cost_rate: float,
) -> None:
    if factors.ndim != 2:
        raise DataValidationError("factors must have shape [symbols, time]")
    if factors.shape[0] == 0:
        raise DataValidationError("factors must contain at least one symbol")
    if target_ret.shape != factors.shape:
        raise DataValidationError("target_ret shape must match factors")
    if target_valid.shape != factors.shape or target_valid.dtype is not torch.bool:
        raise DataValidationError("target_valid must be a boolean mask matching factors")
    if bar_time_ns.shape != factors.shape:
        raise DataValidationError("bar_time_ns shape must match factors")
    if bar_time_ns.dtype is not torch.int64:
        raise DataValidationError("bar_time_ns must contain int64 nanosecond timestamps")
    devices = {
        factors.device,
        target_ret.device,
        target_valid.device,
        bar_time_ns.device,
    }
    if len(devices) != 1:
        raise DataValidationError("execution input tensors must use the same device")
    _validate_supported_device(devices.pop(), context="execution input")
    _validate_supported_float_tensor(factors, name="factors")
    _validate_supported_float_tensor(target_ret, name="target_ret")
    if factors.dtype is not target_ret.dtype:
        raise DataValidationError("factors and target_ret must use the same dtype")
    if not math.isfinite(cost_rate) or cost_rate < 0.0:
        raise DataValidationError("cost_rate must be finite and non-negative")
    if bool((target_valid.sum(dim=1) == 0).any()):
        raise DataValidationError("each symbol must have at least one valid label")

    if target_valid.shape[1] < 3 or bool(target_valid[:, -2:].any()):
        raise DataValidationError(
            "final exit timestamp is missing for a valid label"
        )
    if not bool(torch.isfinite(target_ret[target_valid]).all()):
        raise DataValidationError("valid target_ret values must be finite")


def run_execution(
    *,
    factors: Tensor,
    target_ret: Tensor,
    target_valid: Tensor,
    bar_time_ns: Tensor,
    cost_rate: float,
    min_exposure: float,
) -> ExecutionResult:
    """Run the shared differentiable execution and cost model."""
    _validate_execution_inputs(
        factors=factors,
        target_ret=target_ret,
        target_valid=target_valid,
        bar_time_ns=bar_time_ns,
        cost_rate=cost_rate,
    )
    target_valid = target_valid.clone()
    bar_time_ns = bar_time_ns.clone()

    raw_position = factor_to_position(factors, min_exposure=min_exposure)
    position = torch.where(
        target_valid,
        raw_position,
        torch.zeros_like(raw_position),
    )
    previous_position = torch.cat(
        [torch.zeros_like(position[:, :1]), position[:, :-1]],
        dim=1,
    )
    turnover = torch.where(
        target_valid,
        (position - previous_position).abs(),
        torch.zeros_like(position),
    )

    next_valid = torch.cat(
        [target_valid[:, 1:], torch.zeros_like(target_valid[:, :1])],
        dim=1,
    )
    final_valid = target_valid & ~next_valid
    cost_work_dtype = _cost_work_dtype(position.dtype)
    position_for_cost = position.to(cost_work_dtype)
    turnover_for_cost = turnover.to(cost_work_dtype)
    liquidation_position_for_cost = torch.where(
        final_valid,
        position_for_cost.abs(),
        torch.zeros_like(position_for_cost),
    )
    turnover_cost_work = turnover_for_cost * cost_rate
    liquidation_cost_by_time_work = liquidation_position_for_cost * cost_rate
    _validate_cost_component_product(
        turnover_for_cost,
        cost_rate,
        turnover_cost_work,
    )
    _validate_cost_component_product(
        liquidation_position_for_cost,
        cost_rate,
        liquidation_cost_by_time_work,
    )
    final_liquidation_cost_work = liquidation_cost_by_time_work.sum(dim=1)
    combined_cost_work = turnover_cost_work + liquidation_cost_by_time_work
    _validate_additive_component_preservation(
        turnover_cost_work,
        liquidation_cost_by_time_work,
        combined_cost_work,
        context="execution cost",
    )
    cost_work = torch.where(
        target_valid,
        combined_cost_work,
        torch.zeros_like(position_for_cost),
    )
    final_liquidation_cost = final_liquidation_cost_work
    cost = cost_work
    valid_target_log_return = torch.where(
        target_valid,
        target_ret.to(cost_work_dtype),
        torch.zeros_like(position_for_cost),
    )
    accounting = net_log_return_from_position(
        position,
        valid_target_log_return,
        cost,
    )
    gross_pnl = torch.where(
        target_valid,
        accounting.gross_simple_return,
        torch.zeros_like(position_for_cost),
    )
    net_pnl = torch.where(
        target_valid,
        accounting.net_log_return,
        torch.zeros_like(gross_pnl),
    )
    floating_fields = (
        position,
        turnover,
        gross_pnl,
        cost,
        net_pnl,
        final_liquidation_cost,
    )
    if not all(bool(torch.isfinite(value).all()) for value in floating_fields):
        raise DataValidationError(
            "execution result floating fields must contain only finite values"
        )

    return ExecutionResult(
        position=position,
        turnover=turnover,
        gross_pnl=gross_pnl,
        cost=cost,
        net_pnl=net_pnl,
        target_valid=target_valid,
        bar_time_ns=bar_time_ns,
        final_liquidation_cost=final_liquidation_cost,
    )


def _validate_time_mask(
    bar_time_ns: Tensor,
    target_valid: Tensor,
    *,
    minimum_observations: int = 2,
) -> Tensor:
    if bar_time_ns.ndim != 2 or target_valid.shape != bar_time_ns.shape:
        raise DataValidationError(
            "bar_time_ns and target_valid must have matching [symbols, time] shape"
        )
    if bar_time_ns.dtype is not torch.int64:
        raise DataValidationError("bar_time_ns must contain int64 nanosecond timestamps")
    if target_valid.dtype is not torch.bool:
        raise DataValidationError("target_valid must be a boolean mask")
    if bar_time_ns.device != target_valid.device:
        raise DataValidationError(
            "bar_time_ns and target_valid must use the same device"
        )
    _validate_supported_device(bar_time_ns.device, context="time and mask")

    valid_counts = target_valid.sum(dim=1)
    if bool((valid_counts == 0).any()):
        raise DataValidationError("each symbol must have at least one valid label")
    if int(valid_counts.sum().detach().cpu()) < minimum_observations:
        if minimum_observations == 2:
            raise DataValidationError("at least two valid observations are required")
        raise DataValidationError("not enough valid observations")

    if target_valid.shape[1] < 3 or bool(target_valid[:, -2:].any()):
        raise DataValidationError(
            "final exit timestamp is missing for a valid label"
        )
    return valid_counts


def _validate_relevant_timestamps(
    bar_time_ns: Tensor,
    target_valid: Tensor,
    *,
    start_index: int,
    missing_exit_message: str,
) -> tuple[Tensor, Tensor, Tensor]:
    """Validate the timestamp span read by a possibly segmented mask."""
    time_length = bar_time_ns.shape[1]
    if time_length <= 1:
        raise DataValidationError("first entry timestamp is missing")

    indices = torch.arange(time_length, device=target_valid.device).unsqueeze(0)
    first_signal_indices = torch.where(
        target_valid, indices, torch.full_like(indices, time_length)
    ).min(dim=1).values
    last_signal_indices = torch.where(
        target_valid, indices, torch.full_like(indices, -1)
    ).max(dim=1).values
    final_exit_indices = last_signal_indices + 2
    if bool((final_exit_indices >= time_length).any()):
        raise DataValidationError(missing_exit_message)

    first_entry_indices = first_signal_indices + 1
    first_entry_ns = bar_time_ns.gather(1, first_entry_indices[:, None]).squeeze(1)
    final_exit_ns = bar_time_ns.gather(
        1,
        final_exit_indices[:, None],
    ).squeeze(1)
    if bool((final_exit_ns <= first_entry_ns).any()):
        raise DataValidationError(
            "each symbol must have a positive entry-to-exit timestamp span"
        )

    transition_end_indices = torch.arange(
        1,
        time_length,
        device=bar_time_ns.device,
    ).unsqueeze(0)
    relevant_transitions = (
        (transition_end_indices > first_signal_indices[:, None] + start_index)
        & (transition_end_indices <= final_exit_indices[:, None])
    )
    increasing = bar_time_ns[:, 1:] > bar_time_ns[:, :-1]
    if bool((relevant_transitions & ~increasing).any()):
        raise DataValidationError(
            "bar_time_ns must be strictly increasing within each relevant prefix"
        )
    return first_signal_indices, last_signal_indices, final_exit_indices


def derive_periods_per_year(bar_time_ns: Tensor, target_valid: Tensor) -> float:
    """Derive annualization from each symbol's first entry and final exit."""
    valid_counts = _validate_time_mask(bar_time_ns, target_valid)
    first_signal_indices, _, final_exit_indices = _validate_relevant_timestamps(
        bar_time_ns,
        target_valid,
        start_index=1,
        missing_exit_message="final exit timestamp is missing",
    )

    first_entry_ns = bar_time_ns.gather(
        1, (first_signal_indices + 1)[:, None]
    ).squeeze(1)
    final_exit_ns = bar_time_ns.gather(1, final_exit_indices[:, None]).squeeze(1)

    first_entries = first_entry_ns.detach().cpu().tolist()
    final_exits = final_exit_ns.detach().cpu().tolist()
    valid_count_values = [
        int(valid_count)
        for valid_count in valid_counts.detach().cpu().tolist()
    ]
    elapsed_spans_ns = [
        int(final_exit) - int(first_entry)
        for first_entry, final_exit in zip(first_entries, final_exits)
    ]
    reference_count = valid_count_values[0]
    reference_span_ns = elapsed_spans_ns[0]
    if any(
        valid_count * reference_span_ns
        != reference_count * elapsed_span_ns
        for valid_count, elapsed_span_ns in zip(
            valid_count_values,
            elapsed_spans_ns,
        )
    ):
        raise DataValidationError(
            "all symbols must have a consistent execution cadence"
        )

    elapsed_ns = sum(elapsed_spans_ns)
    observations = sum(valid_count_values)
    elapsed_seconds = elapsed_ns / _NANOSECONDS_PER_SECOND
    periods = observations * _SECONDS_PER_YEAR / elapsed_seconds
    if not math.isfinite(periods) or periods <= 0.0:
        raise DataValidationError("periods_per_year must be finite and positive")
    return periods


def _validate_performance_result(
    *,
    net_pnl: Tensor,
    target_valid: Tensor,
    bar_time_ns: Tensor,
) -> None:
    if net_pnl.shape != target_valid.shape:
        raise DataValidationError("net_pnl shape must match target_valid")
    _validate_supported_float_tensor(net_pnl, name="net_pnl")
    if net_pnl.device != target_valid.device:
        raise DataValidationError(
            "net_pnl, target_valid, and bar_time_ns must use the same device"
        )
    _validate_time_mask(bar_time_ns, target_valid)
    _validate_relevant_timestamps(
        bar_time_ns,
        target_valid,
        start_index=1,
        missing_exit_message="final exit timestamp is missing",
    )


def _validate_ledger_result(
    *,
    bar_time_ns: Tensor,
    target_valid: Tensor,
    position: Tensor,
    gross_pnl: Tensor,
    cost: Tensor,
    net_pnl: Tensor,
) -> Tensor:
    valid_counts = _validate_time_mask(
        bar_time_ns,
        target_valid,
        minimum_observations=1,
    )
    _validate_relevant_timestamps(
        bar_time_ns,
        target_valid,
        start_index=0,
        missing_exit_message="final exit timestamp is missing for ledger",
    )
    expected_shape = target_valid.shape
    expected_device = target_valid.device
    read_fields = {
        "position": position,
        "gross_pnl": gross_pnl,
        "cost": cost,
        "net_pnl": net_pnl,
    }
    for field_name, value in read_fields.items():
        if value.shape != expected_shape:
            raise DataValidationError(
                f"{field_name} shape must match target_valid"
            )
        _validate_supported_float_tensor(value, name=field_name)
        if value.device != expected_device:
            raise DataValidationError(
                f"{field_name} must use the same device as target_valid"
            )
        if not bool(torch.isfinite(value[target_valid]).all()):
            raise DataValidationError(
                f"valid {field_name} values must be finite"
            )

    monetary_fields = {
        field_name: read_fields[field_name]
        for field_name in ("gross_pnl", "cost", "net_pnl")
    }
    if len({value.dtype for value in monetary_fields.values()}) != 1:
        raise DataValidationError(
            "ledger monetary fields must use the same dtype"
        )

    valid_gross = read_fields["gross_pnl"][target_valid]
    valid_cost = read_fields["cost"][target_valid]
    valid_net = read_fields["net_pnl"][target_valid]
    portfolio_simple = valid_gross - valid_cost
    if bool((portfolio_simple <= -1.0).any()):
        raise DataValidationError(
            "ledger portfolio simple return reached the insolvency boundary"
        )
    expected_net = torch.log1p(portfolio_simple)
    lower_bound = expected_net
    upper_bound = expected_net
    negative_infinity = torch.full_like(expected_net, -float("inf"))
    positive_infinity = torch.full_like(expected_net, float("inf"))
    for _ in range(4):
        lower_bound = torch.nextafter(lower_bound, negative_infinity)
        upper_bound = torch.nextafter(upper_bound, positive_infinity)
    consistent = (
        torch.isfinite(expected_net)
        & (valid_net >= lower_bound)
        & (valid_net <= upper_bound)
    )
    if not bool(consistent.all()):
        raise DataValidationError(
            "valid net_pnl must equal log1p(gross_pnl - cost) within dtype tolerance"
        )
    return valid_counts


def _equal_weight_portfolio_log_returns(
    result: ExecutionResult,
) -> tuple[Tensor, float, float]:
    """Aggregate simultaneous symbols by time with explicit equal weights."""
    net_pnl_by_symbol = result._borrow_tensor("net_pnl")
    target_valid = result._borrow_tensor("target_valid")
    bar_time_ns = result._borrow_tensor("bar_time_ns")
    _validate_performance_result(
        net_pnl=net_pnl_by_symbol,
        target_valid=target_valid,
        bar_time_ns=bar_time_ns,
    )
    if not bool(torch.isfinite(net_pnl_by_symbol[target_valid]).all()):
        raise DataValidationError("valid net_pnl values must be finite")
    active_time_indices = torch.nonzero(
        target_valid.any(dim=0), as_tuple=False
    ).flatten()
    active = torch.index_select(target_valid, 1, active_time_indices)
    active_entry_times = torch.index_select(
        bar_time_ns, 1, active_time_indices + 1
    )
    active_exit_times = torch.index_select(
        bar_time_ns, 1, active_time_indices + 2
    )
    column_indices = torch.arange(
        active_time_indices.numel(), device=target_valid.device
    )
    first_active_symbols = active.to(torch.int64).argmax(dim=0)
    portfolio_entry_times = active_entry_times[
        first_active_symbols, column_indices
    ]
    portfolio_exit_times = active_exit_times[
        first_active_symbols, column_indices
    ]
    if not bool(
        ((~active) | (active_entry_times == portfolio_entry_times.unsqueeze(0))).all()
    ):
        raise DataValidationError(
            "multi-symbol portfolio requires synchronized entry timestamps"
        )
    if not bool(
        ((~active) | (active_exit_times == portfolio_exit_times.unsqueeze(0))).all()
    ):
        raise DataValidationError(
            "multi-symbol portfolio requires synchronized exit timestamps"
        )
    if bool((portfolio_exit_times[1:] <= portfolio_exit_times[:-1]).any()):
        raise DataValidationError("portfolio exit timestamps must be strictly increasing")

    selected_net_pnl = torch.index_select(
        net_pnl_by_symbol, 1, active_time_indices
    ).to(torch.float64)
    symbol_simple_returns = torch.where(
        active,
        torch.expm1(selected_net_pnl),
        torch.zeros_like(selected_net_pnl),
    )
    active_counts = active.sum(dim=0)
    portfolio_simple_returns = symbol_simple_returns.sum(dim=0) / active_counts
    if bool((portfolio_simple_returns <= -1.0).any()):
        raise DataValidationError(
            "equal-weight portfolio reached the insolvency boundary"
        )
    net_pnl = torch.log1p(portfolio_simple_returns)
    first_entry_ns = int(portfolio_entry_times[0].detach().cpu())
    final_exit_ns = int(portfolio_exit_times[-1].detach().cpu())
    elapsed_seconds = (
        final_exit_ns - first_entry_ns
    ) / _NANOSECONDS_PER_SECOND
    elapsed_years = elapsed_seconds / _SECONDS_PER_YEAR
    periods_per_year = net_pnl.numel() / elapsed_years
    if not math.isfinite(periods_per_year) or periods_per_year <= 0.0:
        raise DataValidationError("portfolio periods_per_year must be finite and positive")
    return net_pnl, elapsed_years, periods_per_year


def performance_metrics(result: ExecutionResult) -> PerformanceMetrics:
    """Compute metrics from one timestamp-aligned equal-weight portfolio path."""
    net_pnl, elapsed_years, periods_per_year = (
        _equal_weight_portfolio_log_returns(result)
    )
    observations = net_pnl.numel()

    total_log_return = net_pnl.sum()
    annualized_log_return = total_log_return / elapsed_years
    aggregate_equity_logs = torch.stack(
        [total_log_return, annualized_log_return]
    )
    if not bool(torch.isfinite(aggregate_equity_logs).all()) or bool(
        (
            (aggregate_equity_logs < _MIN_LOG_FLOAT64)
            | (aggregate_equity_logs > _MAX_LOG_FLOAT64)
        ).any()
    ):
        raise DataValidationError(
            "unable to derive finite performance metrics from an "
            "unrepresentable equity path"
        )

    cumulative_log_return = torch.cumsum(net_pnl, dim=0)
    log_equity = torch.cat(
        [torch.zeros_like(cumulative_log_return[:1]), cumulative_log_return],
        dim=0,
    )
    running_peak_log = torch.cummax(log_equity, dim=0).values
    drawdown_log = log_equity - running_peak_log
    validation_flags = torch.stack(
        [
            ~torch.isfinite(cumulative_log_return).all(),
            (
                (log_equity < _MIN_LOG_FLOAT64)
                | (log_equity > _MAX_LOG_FLOAT64)
            ).any(),
            (drawdown_log < _MIN_LOG_FLOAT64).any(),
        ]
    )
    invalid_indices = torch.nonzero(validation_flags, as_tuple=False)
    if invalid_indices.numel() != 0:
        error_kind = int(invalid_indices[0, 0].detach().cpu()) % 3
        error_messages = (
            "cumulative net log return must be finite",
            "unable to derive finite performance metrics from an "
            "unrepresentable equity path",
            "unable to derive finite performance metrics from an "
            "unrepresentable equity drawdown",
        )
        raise DataValidationError(error_messages[error_kind])
    max_drawdown_tensor = (-torch.expm1(drawdown_log)).max()

    total_return_tensor = torch.expm1(total_log_return)
    annualized_return_tensor = torch.expm1(annualized_log_return)
    mean_return = net_pnl.mean()
    return_std = net_pnl.std(unbiased=False)
    downside = torch.minimum(net_pnl, torch.zeros_like(net_pnl))
    downside_scale = downside.abs().max()
    if bool(downside_scale > 0):
        downside_deviation = downside_scale * torch.sqrt(
            torch.mean((downside / downside_scale).square())
        )
    else:
        downside_deviation = torch.zeros_like(mean_return)
    critical_statistics = torch.stack(
        [mean_return, return_std, downside_deviation, max_drawdown_tensor]
    )
    if not bool(torch.isfinite(critical_statistics).all()):
        raise DataValidationError(
            "unable to derive finite performance metric intermediates"
        )

    annualization_scale = math.sqrt(periods_per_year)
    zero_risk_denominators: list[str] = []
    if bool(return_std == 0) and bool(mean_return != 0):
        zero_risk_denominators.append("volatility")
    if bool(downside_deviation == 0) and bool(mean_return != 0):
        zero_risk_denominators.append("downside deviation")
    if bool(max_drawdown_tensor == 0) and bool(annualized_return_tensor != 0):
        zero_risk_denominators.append("maximum drawdown")
    if zero_risk_denominators:
        raise DataValidationError(
            "non-zero return cannot be reported with zero risk denominator: "
            + ", ".join(zero_risk_denominators)
        )

    if bool(return_std > 0):
        sharpe_tensor = mean_return / return_std * annualization_scale
    else:
        sharpe_tensor = torch.zeros_like(mean_return)
    if bool(downside_deviation > 0):
        sortino_tensor = mean_return / downside_deviation * annualization_scale
    else:
        sortino_tensor = torch.zeros_like(mean_return)
    if bool(max_drawdown_tensor > 0):
        calmar_tensor = annualized_return_tensor / max_drawdown_tensor
    else:
        calmar_tensor = torch.zeros_like(annualized_return_tensor)

    metric_values = {
        "elapsed_years": elapsed_years,
        "periods_per_year": periods_per_year,
        "total_return": float(total_return_tensor.detach().cpu()),
        "annualized_return": float(annualized_return_tensor.detach().cpu()),
        "volatility": float((return_std * annualization_scale).detach().cpu()),
        "sharpe": float(sharpe_tensor.detach().cpu()),
        "sortino": float(sortino_tensor.detach().cpu()),
        "max_drawdown": float(max_drawdown_tensor.detach().cpu()),
        "calmar": float(calmar_tensor.detach().cpu()),
        "win_rate": float(
            (net_pnl > 0).to(torch.float64).mean().detach().cpu()
        ),
    }
    if not all(math.isfinite(value) for value in metric_values.values()):
        raise DataValidationError("unable to derive finite performance metrics")

    return PerformanceMetrics(observations=observations, **metric_values)


def build_execution_ledger(
    result: ExecutionResult,
    symbols: list[str] | tuple[str, ...],
) -> list[LedgerEntry]:
    """Copy each valid shared execution row into an auditable ledger."""
    target_valid = result._borrow_tensor("target_valid")
    bar_time_ns = result._borrow_tensor("bar_time_ns")
    position = result._borrow_tensor("position")
    gross_pnl = result._borrow_tensor("gross_pnl")
    cost = result._borrow_tensor("cost")
    net_pnl = result._borrow_tensor("net_pnl")
    _validate_ledger_result(
        bar_time_ns=bar_time_ns,
        target_valid=target_valid,
        position=position,
        gross_pnl=gross_pnl,
        cost=cost,
        net_pnl=net_pnl,
    )
    symbol_count, time_length = target_valid.shape
    if not isinstance(symbols, (list, tuple)):
        raise DataValidationError(
            "symbols must be a list or tuple of non-empty strings"
        )
    if len(symbols) != symbol_count:
        raise DataValidationError(
            f"symbols length must match execution rows: "
            f"expected {symbol_count}, actual {len(symbols)}"
        )
    if any(not isinstance(symbol, str) or symbol == "" for symbol in symbols):
        raise DataValidationError("symbols must contain only non-empty strings")

    valid_index_values = [
        torch.nonzero(target_valid[symbol_index], as_tuple=False)
        .flatten().detach().cpu().tolist()
        for symbol_index in range(symbol_count)
    ]
    bar_time_values = bar_time_ns.detach().cpu().tolist()
    position_values = position.detach().cpu().tolist()
    gross_pnl_values = gross_pnl.detach().cpu().tolist()
    cost_values = cost.detach().cpu().tolist()
    net_pnl_values = net_pnl.detach().cpu().tolist()
    target_valid_values = target_valid.detach().cpu().tolist()

    # Ledger rows are Python floats in symbol-major order.  The shared result
    # rule promotes valid published values to float64 before reduction, matching
    # Python's working precision without changing any published row value.
    ledger_net_total = sum(
        float(net_pnl_values[symbol_index][time_index])
        for symbol_index, valid_indices in enumerate(valid_index_values)
        for time_index in valid_indices
    )
    result_net_total = float(
        net_pnl[target_valid].to(torch.float64).sum().detach().cpu()
    )
    if (
        not math.isfinite(ledger_net_total)
        or not math.isfinite(result_net_total)
        or abs(ledger_net_total - result_net_total) > 1e-8
    ):
        raise DataValidationError(
            "ledger net_pnl aggregate must reconcile with the shared result "
            "within 1e-8"
        )

    ledger: list[LedgerEntry] = []
    for symbol_index, symbol in enumerate(symbols):
        valid_indices = valid_index_values[symbol_index]
        for time_index in valid_indices:
            next_is_valid = (
                time_index + 1 < time_length
                and bool(target_valid_values[symbol_index][time_index + 1])
            )
            ledger.append(
                LedgerEntry(
                    symbol=symbol,
                    signal_time_ns=int(bar_time_values[symbol_index][time_index]),
                    entry_time_ns=int(
                        bar_time_values[symbol_index][time_index + 1]
                    ),
                    exit_time_ns=int(
                        bar_time_values[symbol_index][time_index + 2]
                    ),
                    position=float(position_values[symbol_index][time_index]),
                    gross_pnl=float(gross_pnl_values[symbol_index][time_index]),
                    cost=float(cost_values[symbol_index][time_index]),
                    net_pnl=float(net_pnl_values[symbol_index][time_index]),
                    is_final_liquidation=not next_is_valid,
                )
            )
    return ledger
