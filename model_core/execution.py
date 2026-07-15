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


def _validate_cost_publication(
    working_value: Tensor,
    published_value: Tensor,
) -> None:
    underflowed = (working_value != 0) & (
        published_value.to(working_value.dtype) == 0
    )
    if bool(underflowed.any()):
        raise DataValidationError(
            "the result dtype must provide representable non-zero execution cost"
        )


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
    if not math.isfinite(min_exposure) or min_exposure < 0.0:
        raise DataValidationError("min_exposure must be finite and non-negative")
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

    invalid_seen = (~target_valid).cumsum(dim=1) > 0
    if bool((invalid_seen & target_valid).any()):
        raise DataValidationError(
            "target_valid must be one continuous prefix per symbol"
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
    final_liquidation_cost = final_liquidation_cost_work.to(position.dtype)
    cost = cost_work.to(position.dtype)
    _validate_cost_publication(final_liquidation_cost_work, final_liquidation_cost)
    _validate_cost_publication(cost_work, cost)
    _validate_additive_component_preservation(
        turnover_cost_work,
        liquidation_cost_by_time_work,
        cost.to(cost_work_dtype),
        context="execution cost",
    )
    valid_target_ret = torch.where(
        target_valid,
        target_ret,
        torch.zeros_like(target_ret),
    )
    gross_pnl = torch.where(
        target_valid,
        position * valid_target_ret,
        torch.zeros_like(position),
    )
    gross_pnl_work = gross_pnl.to(cost_work_dtype)
    negative_cost_work = -cost.to(cost_work_dtype)
    net_pnl_work = torch.where(
        target_valid,
        gross_pnl_work + negative_cost_work,
        torch.zeros_like(gross_pnl_work),
    )
    _validate_additive_component_preservation(
        gross_pnl_work,
        negative_cost_work,
        net_pnl_work,
        context="net PnL cost",
    )
    net_pnl = torch.where(
        target_valid,
        net_pnl_work.to(position.dtype),
        torch.zeros_like(position),
    )
    _validate_additive_component_preservation(
        gross_pnl_work,
        negative_cost_work,
        net_pnl.to(cost_work_dtype),
        context="net PnL cost",
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

    invalid_seen = (~target_valid).cumsum(dim=1) > 0
    if bool((invalid_seen & target_valid).any()):
        raise DataValidationError(
            "target_valid must be one continuous prefix per symbol"
        )
    return valid_counts


def _validate_relevant_timestamps(
    bar_time_ns: Tensor,
    valid_counts: Tensor,
    *,
    start_index: int,
    missing_exit_message: str,
) -> Tensor:
    """Validate only the timestamp prefix read by an execution consumer."""
    time_length = bar_time_ns.shape[1]
    if time_length <= 1:
        raise DataValidationError("first entry timestamp is missing")

    final_exit_indices = valid_counts + 1
    if bool((final_exit_indices >= time_length).any()):
        raise DataValidationError(missing_exit_message)

    first_entry_ns = bar_time_ns[:, 1]
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
        (transition_end_indices > start_index)
        & (transition_end_indices <= final_exit_indices[:, None])
    )
    increasing = bar_time_ns[:, 1:] > bar_time_ns[:, :-1]
    if bool((relevant_transitions & ~increasing).any()):
        raise DataValidationError(
            "bar_time_ns must be strictly increasing within each relevant prefix"
        )
    return final_exit_indices


def derive_periods_per_year(bar_time_ns: Tensor, target_valid: Tensor) -> float:
    """Derive annualization from each symbol's first entry and final exit."""
    valid_counts = _validate_time_mask(bar_time_ns, target_valid)
    final_exit_indices = _validate_relevant_timestamps(
        bar_time_ns,
        valid_counts,
        start_index=1,
        missing_exit_message="final exit timestamp is missing",
    )

    first_entry_ns = bar_time_ns[:, 1]
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
    valid_counts = _validate_time_mask(bar_time_ns, target_valid)
    _validate_relevant_timestamps(
        bar_time_ns,
        valid_counts,
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
        valid_counts,
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

    if len({value.dtype for value in read_fields.values()}) != 1:
        raise DataValidationError(
            "ledger read fields must use the same dtype"
        )

    valid_gross = read_fields["gross_pnl"][target_valid]
    valid_cost = read_fields["cost"][target_valid]
    valid_net = read_fields["net_pnl"][target_valid]
    expected_net = valid_gross - valid_cost
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
            "valid net_pnl must equal gross_pnl - cost within dtype tolerance"
        )
    return valid_counts


def performance_metrics(result: ExecutionResult) -> PerformanceMetrics:
    """Compute timestamp-derived metrics from valid shared net log returns.

    Total, annualized, mean, standard-deviation, and downside statistics pool
    every symbol's valid returns.  Equity paths and drawdowns instead use each
    symbol's valid prefix independently, with the worst per-symbol drawdown
    used by aggregate-annualized-return Calmar.  This avoids cross-symbol path
    concatenation, is invariant to symbol row order, and preserves one-symbol
    behavior.
    """
    net_pnl_by_symbol = result.net_pnl
    target_valid = result.target_valid
    bar_time_ns = result.bar_time_ns
    _validate_performance_result(
        net_pnl=net_pnl_by_symbol,
        target_valid=target_valid,
        bar_time_ns=bar_time_ns,
    )
    net_pnl = net_pnl_by_symbol[target_valid]
    if not bool(torch.isfinite(net_pnl).all()):
        raise DataValidationError("valid net_pnl values must be finite")
    net_pnl = net_pnl.to(torch.float64)
    periods_per_year = derive_periods_per_year(
        bar_time_ns,
        target_valid,
    )
    observations = net_pnl.numel()
    elapsed_years = observations / periods_per_year

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

    symbol_max_drawdowns: list[Tensor] = []
    symbol_validation_flags: list[Tensor] = []
    for symbol_index in range(net_pnl_by_symbol.shape[0]):
        symbol_net_pnl = net_pnl_by_symbol[
            symbol_index,
            target_valid[symbol_index],
        ].to(torch.float64)
        cumulative_log_return = torch.cumsum(symbol_net_pnl, dim=0)
        cumulative_invalid = ~torch.isfinite(cumulative_log_return).all()
        log_equity = torch.cat(
            [torch.zeros_like(cumulative_log_return[:1]), cumulative_log_return],
            dim=0,
        )
        equity_path_invalid = (
            (log_equity < _MIN_LOG_FLOAT64)
            | (log_equity > _MAX_LOG_FLOAT64)
        ).any()
        running_peak_log = torch.cummax(log_equity, dim=0).values
        drawdown_log = log_equity - running_peak_log
        drawdown_invalid = (drawdown_log < _MIN_LOG_FLOAT64).any()
        symbol_validation_flags.append(
            torch.stack(
                [cumulative_invalid, equity_path_invalid, drawdown_invalid]
            )
        )
        symbol_max_drawdowns.append((-torch.expm1(drawdown_log)).max())
    validation_flags = torch.stack(symbol_validation_flags).reshape(-1)
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
    max_drawdown_tensor = torch.stack(symbol_max_drawdowns).max()

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
    target_valid = result.target_valid
    bar_time_ns = result.bar_time_ns
    position = result.position
    gross_pnl = result.gross_pnl
    cost = result.cost
    net_pnl = result.net_pnl
    valid_counts = _validate_ledger_result(
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

    if bool((valid_counts + 1 >= time_length).any()):
        raise DataValidationError("final exit timestamp is missing for ledger")

    valid_count_values = [
        int(valid_count)
        for valid_count in valid_counts.detach().cpu().tolist()
    ]
    bar_time_values = bar_time_ns.detach().cpu().tolist()
    position_values = position.detach().cpu().tolist()
    gross_pnl_values = gross_pnl.detach().cpu().tolist()
    cost_values = cost.detach().cpu().tolist()
    net_pnl_values = net_pnl.detach().cpu().tolist()

    # Ledger rows are Python floats in symbol-major order.  The shared result
    # rule promotes valid published values to float64 before reduction, matching
    # Python's working precision without changing any published row value.
    ledger_net_total = sum(
        float(net_pnl_values[symbol_index][time_index])
        for symbol_index, valid_count in enumerate(valid_count_values)
        for time_index in range(valid_count)
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
        valid_count = valid_count_values[symbol_index]
        for time_index in range(valid_count):
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
                    is_final_liquidation=time_index == valid_count - 1,
                )
            )
    return ledger
