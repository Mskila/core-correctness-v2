"""Immutable V2 artifact identities and explicit backtest data contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import math
import re
from typing import Any, Iterator
import uuid

from data_pipeline.validation import DatasetIdentity, normalize_timeframe_name

from .semantics import (
    CORE_SEMANTICS_VERSION,
    DATA_SCHEMA_VERSION,
    EXECUTION_SEMANTICS_VERSION,
    LABEL_LOOKAHEAD_BARS,
    LABEL_SEMANTICS_VERSION,
    ArtifactCompatibilityError,
    BacktestModeError,
    DataValidationError,
)
from .vocab import FORMULA_VOCAB, VOCAB_VERSION
from .vm import validate_formula_structure


_ARTIFACT_IDENTITY_FIELDS = (
    "core_semantics_version",
    "vocab_version",
    "label_semantics_version",
    "execution_semantics_version",
    "symbol",
    "timeframe",
    "training_dataset",
    "training_config",
    "training_config_hash",
)
_STRATEGY_FIELDS = (
    "schema_version",
    "run_identity",
    "formula_tokens",
    "decoded_formula",
    "best_score",
    "fold_evidence",
    "generated_at",
    "fingerprint",
)
_FOLD_FIELDS = (
    "fold_index",
    "train_start_time_ns",
    "train_end_time_ns",
    "val_start_time_ns",
    "val_end_time_ns",
    "effective_gap",
    "validation_metrics",
)
_DATASET_IDENTITY_FIELDS = (
    "schema_version",
    "symbol",
    "timeframe",
    "start_time_ns",
    "end_time_ns",
    "bars",
    "data_fingerprint",
    "time_fingerprint",
)
_ENVIRONMENT_CONFIG_KEYS = {
    "device",
    "data_file",
    "source_path",
    "local_path",
    "output_dir",
    "checkpoint_dir",
    "strategy_dir",
    "history_dir",
}
_REQUIRED_TRAINING_CONFIG_FIELDS = {
    "model",
    "batch_size",
    "train_steps",
    "max_formula_len",
    "reward",
    "entropy",
    "elite",
    "restart",
    "noise",
    "lord",
    "walk_forward",
    "cost_rate",
    "neutral_band",
    "random_seed",
}
_CURRENT_REWARD_MODES = frozenset({"standard", "ftmo", "forex"})
_TRAINING_CONFIG_SCALAR_SCHEMA = {
    "batch_size": "positive_int",
    "train_steps": "positive_int",
    "max_formula_len": "positive_int",
    "cost_rate": "nonnegative_number",
    "neutral_band": "unit_number",
    "random_seed": "nonnegative_int",
}
_TRAINING_CONFIG_CONTAINER_SCHEMA = {
    "model": {
        "input_dim": "positive_int",
        "hidden_dim": "positive_int",
        "num_layers": "positive_int",
    },
    "reward": {
        "ic": "nonnegative_number",
        "return": "nonnegative_number",
        "alpha": "nonnegative_number",
        "mode": "current_reward_mode",
        "ic_gate_thresh": "nonnegative_number",
        "ic_gate_mult": "nonnegative_number",
        "ic_neg_mult": "nonnegative_number",
        "ema_baseline": "bool",
        "ema_decay": "unit_number",
        "ema_warmup": "nonnegative_int",
        "factor_top_k": "positive_int",
        "corr_threshold": "unit_number",
        "corr_penalty": "nonnegative_number",
        "beta_neutral_penalty": "bool",
        "half_consistency_bonus": "bool",
        "beta_neutral_thresh": "unit_number",
        "beta_neutral_light_thresh": "unit_number",
    },
    "entropy": {
        "coeff_max": "nonnegative_number",
        "coeff_power": "nonnegative_number",
        "collapse_thresh": "nonnegative_number",
        "collapse_steps": "positive_int",
        "floor_enabled": "bool",
        "floor_thresh": "nonnegative_number",
        "floor_lambda": "nonnegative_number",
    },
    "elite": {
        "size": "positive_int",
        "replay_frac": "unit_number",
        "reward_scale": "nonnegative_number",
        "decay": "bool",
        "decay_half_life": "positive_int",
    },
    "restart": {
        "max_restarts": "nonnegative_int",
        "restart_noise": "nonnegative_number",
        "stagnation_window": "positive_int",
        "full_reset_every": "positive_int",
        "partial_reset": "bool",
        "partial_reset_layers": "nonempty_string_list",
    },
    "noise": {
        "initial": "nonnegative_number",
        "boost": "nonnegative_number",
        "adaptive": "bool",
        "min": "nonnegative_number",
        "max": "nonnegative_number",
        "boost_factor": "nonnegative_number",
    },
    "lord": {
        "use_lord_regularization": "bool",
        "lord_decay_rate": "nonnegative_number",
        "lord_num_iterations": "positive_int",
    },
    "walk_forward": {
        "blocks": "exact_five_int",
        "gap": "nonnegative_int",
        "min_fold_bars": "positive_int",
        "warmup_bars": "nonnegative_int",
        "label_lookahead": "positive_int",
    },
}
_SAFE_FILENAME = re.compile(r"[^A-Za-z0-9._-]+")
_HEX_64 = re.compile(r"[0-9a-f]{64}")
_HEX_32 = re.compile(r"[0-9a-f]{32}")
_MAX_JSON_NESTING_DEPTH = 100
_MAX_ARTIFACT_JSON_BYTES = 1_000_000
_MAX_ARTIFACT_JSON_CONTAINERS = 10_000
_MAX_DIAGNOSTIC_VALUE_CHARS = 160
_MAX_DIAGNOSTIC_CAUSE_CHARS = 640
_MAX_PUBLIC_ERROR_CHARS = 1_024
_MAX_OPERATIONAL_INTEGER = int(float.fromhex("0x1.fffffffffffffp+1023"))
_APPROVED_WF_N_BLOCKS = 5
_APPROVED_WF_VALIDATION_FOLDS = _APPROVED_WF_N_BLOCKS - 1


def _bounded_text(value: str, *, quoted: bool) -> str:
    if len(value) <= _MAX_DIAGNOSTIC_VALUE_CHARS:
        return repr(value) if quoted else value
    prefix = value[:_MAX_DIAGNOSTIC_VALUE_CHARS]
    rendered = repr(prefix) if quoted else prefix
    return f"{rendered}...<truncated length={len(value)}>"


def _bounded_public_message(value: str) -> str:
    if len(value) <= _MAX_PUBLIC_ERROR_CHARS:
        return value
    suffix = f"...<truncated length={len(value)}>"
    return value[: _MAX_PUBLIC_ERROR_CHARS - len(suffix)] + suffix


def _safe_type_category(value: object) -> str:
    """Classify untrusted values without consulting dynamic class metadata."""
    value_type = type(value)
    if value_type is str:
        return "str"
    if value_type is bool:
        return "bool"
    if value_type is int:
        return "int"
    if value_type is float:
        return "float"
    if value_type is dict:
        return "dict"
    if value_type is list:
        return "list"
    if value_type is tuple:
        return "tuple"
    if value is None:
        return "NoneType"
    return "non-exact-built-in"


def _safe_diagnostic(value: object) -> str:
    """Render untrusted input without invoking arbitrary repr implementations."""
    if type(value) is str:
        return _bounded_text(value, quoted=True)
    if type(value) is int and (
        value < -_MAX_OPERATIONAL_INTEGER or value > _MAX_OPERATIONAL_INTEGER
    ):
        return "<int outside operational finite-real range>"
    if value is None or type(value) in (bool, int, float):
        return repr(value)
    if type(value) is dict:
        keys = list(value)[:4]
        rendered_keys = ", ".join(_safe_diagnostic(key) for key in keys)
        if len(value) > 4:
            rendered_keys += f", ...<{len(value) - 4} more>"
        return f"<dict keys=[{rendered_keys}]>"
    if type(value) is list:
        return f"<list length={len(value)}>"
    if type(value) is tuple:
        return f"<tuple length={len(value)}>"
    return "<non-exact-built-in>"


def _safe_diagnostic_text(value: object) -> str:
    if type(value) is str:
        return _bounded_text(value, quoted=False)
    if type(value) is int and (
        value < -_MAX_OPERATIONAL_INTEGER or value > _MAX_OPERATIONAL_INTEGER
    ):
        return "<int outside operational finite-real range>"
    if value is None or type(value) in (bool, int, float):
        return str(value)
    if type(value) in (
        ArtifactCompatibilityError,
        DataValidationError,
        RuntimeError,
        KeyError,
        TypeError,
        ValueError,
    ):
        if len(value.args) == 1 and type(value.args[0]) is str:
            message = value.args[0]
            if len(message) <= _MAX_DIAGNOSTIC_CAUSE_CHARS:
                return message
            suffix = f"...<truncated length={len(message)}>"
            return message[: _MAX_DIAGNOSTIC_CAUSE_CHARS - len(suffix)] + suffix
        return "<exception>"
    if isinstance(value, BaseException):
        return "<exception>"
    return "<non-exact-built-in>"


def _safe_exception_category(value: BaseException) -> str:
    for exact_type, category in (
        (ArtifactCompatibilityError, "ArtifactCompatibilityError"),
        (DataValidationError, "DataValidationError"),
        (RuntimeError, "RuntimeError"),
        (MemoryError, "MemoryError"),
        (KeyError, "KeyError"),
        (TypeError, "TypeError"),
        (ValueError, "ValueError"),
    ):
        if type(value) is exact_type:
            return category
    return "exception-subclass"


def _safe_exception_cause(value: BaseException) -> BaseException:
    """Select an explicit cause without invoking subclass metadata protocols."""
    cause = BaseException.__getattribute__(value, "__cause__")
    return value if cause is None else cause


def _safe_diagnostic_sequence(values: Sequence[object]) -> str:
    shown = [_safe_diagnostic(value) for value in values[:4]]
    if len(values) > 4:
        shown.append(f"...<{len(values) - 4} more>")
    return "[" + ", ".join(shown) + "]"


def _safe_diagnostic_getattr(value: object, field: str) -> object:
    try:
        return getattr(value, field, "?")
    except Exception:
        return "?"


def _is_exact_operational_integer(value: object) -> bool:
    return (
        type(value) is int
        and -_MAX_OPERATIONAL_INTEGER <= value <= _MAX_OPERATIONAL_INTEGER
    )


def _is_exact_operational_real(value: object) -> bool:
    if type(value) is int:
        return _is_exact_operational_integer(value)
    if type(value) is float:
        return math.isfinite(value)
    return False


def _validate_exact_string(
    value: object,
    *,
    field: str,
    nonempty: bool = True,
) -> str:
    if type(value) is not str or (nonempty and not value):
        expected = "non-empty exact built-in string" if nonempty else "exact built-in string"
        raise ArtifactCompatibilityError(
            f"{field} is invalid: expected={expected} "
            f"actual_type={_safe_type_category(value)} actual={_safe_diagnostic(value)}"
        )
    return value


class _FrozenDict(Mapping[str, object]):
    """Small immutable mapping used to close nested dataclass mutation holes."""

    __slots__ = ("_values",)

    def __init__(self, values: Mapping[str, object]) -> None:
        self._values = dict(values)

    def __getitem__(self, key: str) -> object:
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __repr__(self) -> str:
        return repr(self._values)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Mapping) and dict(self.items()) == dict(other.items())


def _validate_json_graph(
    value: object,
    *,
    allow_internal: bool,
    root_path: str = "$",
) -> None:
    """Reject cyclic or excessively nested JSON containers without recursion."""
    active_paths: dict[int, str] = {}
    stack: list[tuple[object, str, int, bool]] = [(value, root_path, 1, False)]
    while stack:
        current, path, depth, leaving = stack.pop()
        current_id = id(current)
        if leaving:
            active_paths.pop(current_id, None)
            continue

        is_object = (
            isinstance(current, Mapping)
            if allow_internal
            else type(current) is dict
        )
        is_array = (
            isinstance(current, (list, tuple))
            if allow_internal
            else type(current) is list
        )
        if not is_object and not is_array:
            continue
        if depth > _MAX_JSON_NESTING_DEPTH:
            raise ArtifactCompatibilityError(
                _bounded_public_message(
                    "JSON nesting depth exceeded at "
                    f"{path}: maximum={_MAX_JSON_NESTING_DEPTH} actual={depth}"
                )
            )
        first_path = active_paths.get(current_id)
        if first_path is not None:
            raise ArtifactCompatibilityError(
                _bounded_public_message(
                    f"JSON cycle detected at {path}; first_seen={first_path}"
                )
            )

        active_paths[current_id] = path
        stack.append((current, path, depth, True))
        if is_object:
            assert isinstance(current, Mapping)
            children = list(current.items())
            for key, item in reversed(children):
                if type(key) is not str:
                    raise ArtifactCompatibilityError(
                        _bounded_public_message(
                            f"JSON object key must be an exact built-in string at {path}; "
                            f"actual_type={_safe_type_category(key)}"
                        )
                    )
                child_path = f"{path}.{key}"
                child_path = _bounded_public_message(child_path)
                stack.append((item, child_path, depth + 1, False))
        else:
            assert isinstance(current, (list, tuple))
            for index in range(len(current) - 1, -1, -1):
                stack.append((current[index], f"{path}[{index}]", depth + 1, False))


def _json_value(
    value: object,
    *,
    path: str = "$",
    freeze: bool = False,
    allow_internal: bool = False,
) -> object:
    if isinstance(value, str) and type(value) is not str:
        raise ArtifactCompatibilityError(
            f"value at {path} must be an exact built-in string; "
            f"actual_type={_safe_type_category(value)}"
        )
    if value is None or type(value) in (str, bool):
        return value
    if type(value) is int:
        if not _is_exact_operational_real(value):
            raise ArtifactCompatibilityError(
                f"integer JSON value is outside the operational finite-real range at {path}"
            )
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ArtifactCompatibilityError(f"non-finite JSON value at {path}")
        return value
    is_object = isinstance(value, Mapping) if allow_internal else type(value) is dict
    if is_object:
        assert isinstance(value, Mapping)
        normalized: dict[str, object] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise ArtifactCompatibilityError(
                    f"JSON object key must be an exact built-in string at {path}; "
                    f"actual_type={_safe_type_category(key)}"
                )
            normalized[key] = _json_value(
                item,
                path=f"{path}.{key}",
                freeze=freeze,
                allow_internal=allow_internal,
            )
        return _FrozenDict(normalized) if freeze else normalized
    is_array = (
        isinstance(value, (list, tuple))
        if allow_internal
        else type(value) is list
    )
    if is_array:
        assert isinstance(value, (list, tuple))
        normalized_items = [
            _json_value(
                item,
                path=f"{path}[{index}]",
                freeze=freeze,
                allow_internal=allow_internal,
            )
            for index, item in enumerate(value)
        ]
        return tuple(normalized_items) if freeze else normalized_items
    raise ArtifactCompatibilityError(
        f"value at {path} is not JSON-compatible; "
        f"actual_type={_safe_type_category(value)}"
    )


def _plain_json(value: object) -> object:
    _validate_json_graph(value, allow_internal=False)
    return _json_value(value, freeze=False, allow_internal=False)


def _thaw_json(value: object) -> object:
    _validate_json_graph(value, allow_internal=True)
    return _json_value(value, freeze=False, allow_internal=True)


def canonical_json_bytes(value: object) -> bytes:
    """Return canonical compact UTF-8 JSON, rejecting invalid JSON values."""
    try:
        return json.dumps(
            _plain_json(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except ArtifactCompatibilityError:
        raise
    except (RecursionError, TypeError, ValueError, OverflowError) as exc:
        raise ArtifactCompatibilityError(
            _bounded_public_message(
                "cannot serialize canonical JSON: " + _safe_diagnostic_text(exc)
            )
        ) from exc


def sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


_MAPPING_PROTOCOL_ERRORS = (KeyError, RuntimeError, TypeError, ValueError)


def _snapshot_json_like(
    value: object,
    *,
    context: str,
    path: str,
    depth: int,
    active_paths: dict[int, str],
    completed_snapshots: dict[int, tuple[object, object]],
    allow_internal: bool,
) -> object:
    if isinstance(value, str) and type(value) is not str:
        raise ArtifactCompatibilityError(
            _bounded_public_message(
                f"{context} value must be an exact built-in string at {path}; "
                f"actual_type={_safe_type_category(value)}"
            )
        )
    is_mapping = isinstance(value, Mapping)
    is_list = type(value) is list
    is_internal_tuple = allow_internal and type(value) is tuple
    if isinstance(value, (list, tuple)) and not is_list and not is_internal_tuple:
        expected_array_type = (
            "exact built-in list or internal tuple"
            if allow_internal
            else "exact built-in list"
        )
        raise ArtifactCompatibilityError(
            _bounded_public_message(
                f"{context} unsupported JSON array type at {path}: "
                f"expected={expected_array_type} actual_type={_safe_type_category(value)}"
            )
        )
    if not is_mapping and not is_list and not is_internal_tuple:
        return value
    if depth > _MAX_JSON_NESTING_DEPTH:
        raise ArtifactCompatibilityError(
            _bounded_public_message(
                "JSON nesting depth exceeded at "
                f"{path}: maximum={_MAX_JSON_NESTING_DEPTH} actual={depth}"
            )
        )
    value_id = id(value)
    first_path = active_paths.get(value_id)
    if first_path is not None:
        raise ArtifactCompatibilityError(
            _bounded_public_message(
                f"JSON cycle detected at {path}; first_seen={first_path}"
            )
        )
    completed = completed_snapshots.get(value_id)
    if completed is not None and completed[0] is value:
        return completed[1]
    active_paths[value_id] = path
    try:
        if is_mapping:
            assert isinstance(value, Mapping)
            try:
                pairs = list(value.items())
            except _MAPPING_PROTOCOL_ERRORS as exc:
                raise ArtifactCompatibilityError(
                    _bounded_public_message(
                        f"{context} mapping access failed at {path}: "
                        f"cause={_safe_exception_category(exc)} "
                        f"message={_safe_diagnostic_text(exc)}"
                    )
                ) from exc
            non_exact_string_keys = [
                key for key, _ in pairs if type(key) is not str
            ]
            if non_exact_string_keys:
                raise ArtifactCompatibilityError(
                    _bounded_public_message(
                        f"{context} key must be an exact built-in string at {path}; "
                        "actual_types="
                        f"{_safe_diagnostic_sequence([_safe_type_category(key) for key in non_exact_string_keys])}"
                    )
                )
            seen_keys: set[str] = set()
            for key, _ in pairs:
                if key in seen_keys:
                    raise ArtifactCompatibilityError(
                        _bounded_public_message(
                            f"{context} duplicate exact-string key at {path}: "
                            f"key={_safe_diagnostic(key)}"
                        )
                    )
                seen_keys.add(key)
            plain = dict(pairs)
            snapshot: object = {
                key: _snapshot_json_like(
                    item,
                    context=context,
                    path=_bounded_public_message(f"{path}.{key}"),
                    depth=depth + 1,
                    active_paths=active_paths,
                    completed_snapshots=completed_snapshots,
                    allow_internal=allow_internal,
                )
                for key, item in plain.items()
            }
        else:
            assert type(value) in (list, tuple)
            snapshot = [
                _snapshot_json_like(
                    item,
                    context=context,
                    path=f"{path}[{index}]",
                    depth=depth + 1,
                    active_paths=active_paths,
                    completed_snapshots=completed_snapshots,
                    allow_internal=allow_internal,
                )
                for index, item in enumerate(value)
            ]
        completed_snapshots[value_id] = (value, snapshot)
        return snapshot
    finally:
        active_paths.pop(value_id, None)


def _require_mapping(
    value: object,
    *,
    context: str,
    root_path: str = "$",
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ArtifactCompatibilityError(
            f"{context} must be a JSON object; actual_type={_safe_type_category(value)}"
        )
    snapshot = _snapshot_json_like(
        value,
        context=context,
        path=root_path,
        depth=1,
        active_paths={},
        completed_snapshots={},
        allow_internal=isinstance(value, _FrozenDict),
    )
    assert isinstance(snapshot, dict)
    return snapshot


def _require_exact_fields(
    value: Mapping[str, object],
    fields: Sequence[str],
    *,
    context: str,
) -> None:
    non_exact_string_keys = [key for key in value if type(key) is not str]
    if non_exact_string_keys:
        raise ArtifactCompatibilityError(
            f"{context} key must be an exact built-in string; "
            "actual_types="
            f"{_safe_diagnostic_sequence([_safe_type_category(key) for key in non_exact_string_keys])}"
        )
    expected = set(fields)
    actual = set(value)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        raise ArtifactCompatibilityError(
            _bounded_public_message(
                f"invalid {context} fields: "
                f"missing={_safe_diagnostic_sequence(missing)} "
                f"unknown={_safe_diagnostic_sequence(unknown)}"
            )
        )


def _validate_hash(value: object, *, field: str) -> str:
    if type(value) is not str:
        raise ArtifactCompatibilityError(
            f"{field} must be an exact built-in string containing a lowercase "
            f"SHA-256 hex digest; actual_type={_safe_type_category(value)}"
        )
    if _HEX_64.fullmatch(value) is None:
        raise ArtifactCompatibilityError(
            f"{field} must be a lowercase SHA-256 hex digest; "
            f"actual={_safe_diagnostic(value)}"
        )
    return value


def _reject_environment_config(value: object, *, path: str = "training_config") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = key.lower()
            if (
                normalized in _ENVIRONMENT_CONFIG_KEYS
                or normalized.endswith("_path")
                or normalized.endswith("_dir")
                or "device" in normalized
            ):
                raise ArtifactCompatibilityError(
                    f"environment-specific field is forbidden in training_config: {path}.{key}"
                )
            _reject_environment_config(item, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_environment_config(item, path=f"{path}[{index}]")


def _config_value_error(path: str, *, expected: str, actual: object) -> None:
    raise ArtifactCompatibilityError(
        f"{path} is invalid: expected={expected} "
        f"actual={_safe_diagnostic(actual)}"
    )


def _validate_config_value(value: object, *, rule: str, path: str) -> None:
    if rule == "bool":
        if type(value) is not bool:
            _config_value_error(path, expected="boolean", actual=value)
        return
    if rule == "exact_five_int":
        if type(value) is not int or value != _APPROVED_WF_N_BLOCKS:
            _config_value_error(
                path,
                expected=str(_APPROVED_WF_N_BLOCKS),
                actual=value,
            )
        return
    if rule in {"positive_int", "nonnegative_int", "at_least_two_int"}:
        minimum = {
            "nonnegative_int": 0,
            "positive_int": 1,
            "at_least_two_int": 2,
        }[rule]
        if not _is_exact_operational_integer(value) or value < minimum:
            _config_value_error(
                path,
                expected=f"integer >= {minimum}",
                actual=value,
            )
        return
    if rule in {"nonnegative_number", "unit_number"}:
        if not _is_exact_operational_real(value):
            _config_value_error(path, expected="finite number", actual=value)
        assert type(value) in (int, float)
        if value < 0 or (rule == "unit_number" and value > 1):
            expected = (
                "finite number >= 0"
                if rule == "nonnegative_number"
                else "number in [0, 1]"
            )
            _config_value_error(path, expected=expected, actual=value)
        return
    if rule == "current_reward_mode":
        if type(value) is not str or value not in _CURRENT_REWARD_MODES:
            _config_value_error(
                path,
                expected="exactly one of ['standard', 'ftmo', 'forex']",
                actual=value,
            )
        return
    if rule == "nonempty_string":
        if type(value) is not str or not value:
            _config_value_error(path, expected="non-empty exact built-in string", actual=value)
        return
    if rule == "nonempty_string_list":
        if not isinstance(value, (list, tuple)) or not value:
            _config_value_error(path, expected="non-empty JSON list", actual=value)
        for index, item in enumerate(value):
            _validate_config_value(
                item,
                rule="nonempty_string",
                path=f"{path}[{index}]",
            )
        return
    raise AssertionError(f"unknown training config schema rule: {rule}")


def _validate_training_config_fields(value: Mapping[str, object]) -> None:
    actual_fields = set(value)
    missing = sorted(_REQUIRED_TRAINING_CONFIG_FIELDS - actual_fields)
    unknown = sorted(actual_fields - _REQUIRED_TRAINING_CONFIG_FIELDS)
    if missing or unknown:
        raise ArtifactCompatibilityError(
            _bounded_public_message(
                "invalid training_config behavior fields: "
                f"missing={_safe_diagnostic_sequence(missing)} "
                f"unknown={_safe_diagnostic_sequence(unknown)} "
                "expected="
                f"{_safe_diagnostic_sequence(sorted(_REQUIRED_TRAINING_CONFIG_FIELDS))}"
            )
        )
    for key, rule in _TRAINING_CONFIG_SCALAR_SCHEMA.items():
        _validate_config_value(
            value[key],
            rule=rule,
            path=f"training_config.{key}",
        )
    for container, schema in _TRAINING_CONFIG_CONTAINER_SCHEMA.items():
        item = value[container]
        if not isinstance(item, Mapping):
            _config_value_error(
                f"training_config.{container}",
                expected="non-empty JSON object",
                actual=item,
            )
        if not item:
            _config_value_error(
                f"training_config.{container}",
                expected="non-empty JSON object",
                actual=item,
            )
        actual_nested = set(item)
        expected_nested = set(schema)
        missing_nested = sorted(expected_nested - actual_nested)
        unknown_nested = sorted(actual_nested - expected_nested)
        if missing_nested or unknown_nested:
            raise ArtifactCompatibilityError(
                _bounded_public_message(
                    f"invalid training_config.{container} behavior fields: "
                    f"missing={_safe_diagnostic_sequence(missing_nested)} "
                    f"unknown={_safe_diagnostic_sequence(unknown_nested)} "
                    f"expected={_safe_diagnostic_sequence(sorted(expected_nested))} "
                    f"actual={_safe_diagnostic_sequence(sorted(actual_nested))}"
                )
            )
        for nested_key, nested_value in item.items():
            _validate_config_value(
                nested_value,
                rule=schema[nested_key],
                path=f"training_config.{container}.{nested_key}",
            )
    noise = value["noise"]
    assert isinstance(noise, Mapping)
    if (
        "min" in noise
        and "max" in noise
        and noise["min"] > noise["max"]
    ):
        raise ArtifactCompatibilityError(
            "training_config.noise range is invalid: "
            f"expected=min <= max actual={noise['min']!r} > {noise['max']!r}"
        )
    model = value["model"]
    assert isinstance(model, Mapping)
    if model["input_dim"] != FORMULA_VOCAB.feature_count:
        _config_value_error(
            "training_config.model.input_dim",
            expected=str(FORMULA_VOCAB.feature_count),
            actual=model["input_dim"],
        )
    walk_forward = value["walk_forward"]
    assert isinstance(walk_forward, Mapping)
    if walk_forward["blocks"] != _APPROVED_WF_N_BLOCKS:
        _config_value_error(
            "training_config.walk_forward.blocks",
            expected=str(_APPROVED_WF_N_BLOCKS),
            actual=walk_forward["blocks"],
        )
    if walk_forward["label_lookahead"] != LABEL_LOOKAHEAD_BARS:
        _config_value_error(
            "training_config.walk_forward.label_lookahead",
            expected=str(LABEL_LOOKAHEAD_BARS),
            actual=walk_forward["label_lookahead"],
        )


def _snapshot_dataset_identity_payload(
    value: object,
    *,
    context: str,
) -> dict[str, object]:
    if not isinstance(value, DatasetIdentity):
        raise ArtifactCompatibilityError(
            f"{context} must be a DatasetIdentity; actual_type={_safe_type_category(value)}"
        )
    payload: dict[str, object] = {}
    for field in _DATASET_IDENTITY_FIELDS:
        try:
            payload[field] = getattr(value, field)
        except Exception as exc:
            if isinstance(exc, AssertionError):
                raise
            raise ArtifactCompatibilityError(
                _bounded_public_message(
                    f"{context} attribute access failed: field={field} "
                    f"cause={_safe_exception_category(exc)} "
                    f"message={_safe_diagnostic_text(exc)}"
                )
            ) from exc
    return payload


def _dataset_identity_from_payload(
    payload: Mapping[str, object],
    *,
    context: str,
) -> DatasetIdentity:
    schema_version = payload["schema_version"]
    symbol = payload["symbol"]
    timeframe = payload["timeframe"]
    start_time_ns = payload["start_time_ns"]
    end_time_ns = payload["end_time_ns"]
    bars = payload["bars"]
    data_fingerprint = payload["data_fingerprint"]
    time_fingerprint = payload["time_fingerprint"]

    _validate_exact_string(
        schema_version,
        field=f"{context}.schema_version",
    )
    if schema_version != DATA_SCHEMA_VERSION:
        raise ArtifactCompatibilityError(
            f"{context}.schema_version mismatch: "
            f"expected={DATA_SCHEMA_VERSION!r} "
            f"actual={_safe_diagnostic(schema_version)}"
        )
    if type(symbol) is not str:
        raise ArtifactCompatibilityError(
            f"{context}.symbol must be a non-empty exact built-in string; "
            f"actual_type={_safe_type_category(symbol)}"
        )
    if not symbol.strip():
        raise ArtifactCompatibilityError(
            f"{context}.symbol must be a non-empty string; "
            f"actual={_safe_diagnostic(symbol)}"
        )
    if type(timeframe) is not str:
        raise ArtifactCompatibilityError(
            f"{context}.timeframe must be a canonical exact built-in string; "
            f"actual_type={_safe_type_category(timeframe)}"
        )
    try:
        normalized_timeframe = normalize_timeframe_name(timeframe)
    except DataValidationError as exc:
        raise ArtifactCompatibilityError(
            _bounded_public_message(
                f"{context}.timeframe is invalid; "
                f"actual={_safe_diagnostic(timeframe)} "
                f"cause={_safe_diagnostic_text(exc)}"
            )
        ) from exc
    if timeframe != normalized_timeframe:
        raise ArtifactCompatibilityError(
            f"{context}.timeframe must be canonical: "
            f"expected={_safe_diagnostic(normalized_timeframe)} "
            f"actual={_safe_diagnostic(timeframe)}"
        )
    for field, field_value in (
        ("start_time_ns", start_time_ns),
        ("end_time_ns", end_time_ns),
    ):
        if not _is_exact_operational_integer(field_value):
            raise ArtifactCompatibilityError(
                f"{context}.{field} must be an integer, not bool, within the operational range; "
                f"actual={_safe_diagnostic(field_value)}"
            )
    if not _is_exact_operational_integer(bars) or bars <= 0:
        raise ArtifactCompatibilityError(
            f"{context}.bars must be a positive integer, not bool, within the operational range; "
            f"actual={_safe_diagnostic(bars)}"
        )
    assert type(start_time_ns) is int
    assert type(end_time_ns) is int
    if start_time_ns > end_time_ns:
        raise ArtifactCompatibilityError(
            f"{context} range is reversed: "
            f"start_time_ns={start_time_ns} end_time_ns={end_time_ns}"
        )
    validated_data_fingerprint = _validate_hash(
        data_fingerprint,
        field=f"{context}.data_fingerprint",
    )
    validated_time_fingerprint = _validate_hash(
        time_fingerprint,
        field=f"{context}.time_fingerprint",
    )
    return DatasetIdentity(
        schema_version=schema_version,  # type: ignore[arg-type]
        symbol=symbol,
        timeframe=timeframe,
        start_time_ns=start_time_ns,
        end_time_ns=end_time_ns,
        bars=bars,
        data_fingerprint=validated_data_fingerprint,
        time_fingerprint=validated_time_fingerprint,
    )


def _strict_dataset_identity_from_dict(
    value: Mapping[str, object],
    *,
    context: str,
) -> DatasetIdentity:
    mapping = _require_mapping(value, context=context)
    _require_exact_fields(mapping, _DATASET_IDENTITY_FIELDS, context=context)
    return _dataset_identity_from_payload(mapping, context=context)


def _dataset_identity_payload(value: DatasetIdentity) -> dict[str, object]:
    return {
        "schema_version": value.schema_version,
        "symbol": value.symbol,
        "timeframe": value.timeframe,
        "start_time_ns": value.start_time_ns,
        "end_time_ns": value.end_time_ns,
        "bars": value.bars,
        "data_fingerprint": value.data_fingerprint,
        "time_fingerprint": value.time_fingerprint,
    }


def _validated_dataset_identity_snapshot(
    value: object,
    *,
    context: str,
) -> DatasetIdentity:
    return _dataset_identity_from_payload(
        _snapshot_dataset_identity_payload(value, context=context),
        context=context,
    )


def _validate_current_artifact_versions(value: object) -> None:
    expected_actual = (
        ("core_semantics_version", CORE_SEMANTICS_VERSION),
        ("vocab_version", VOCAB_VERSION),
        ("label_semantics_version", LABEL_SEMANTICS_VERSION),
        ("execution_semantics_version", EXECUTION_SEMANTICS_VERSION),
    )
    mismatches = [
        f"{field}: expected={_safe_diagnostic(expected)} "
        f"actual={_safe_diagnostic(getattr(value, field, None))}"
        for field, expected in expected_actual
        if getattr(value, field, None) != expected
    ]
    if mismatches:
        raise ArtifactCompatibilityError(
            _bounded_public_message(
                "artifact semantics mismatch; " + "; ".join(mismatches)
            )
        )


@dataclass(frozen=True)
class ArtifactIdentity:
    core_semantics_version: str
    vocab_version: str
    label_semantics_version: str
    execution_semantics_version: str
    symbol: str
    timeframe: str
    training_dataset: DatasetIdentity
    training_config: Mapping[str, object]
    training_config_hash: str

    def __post_init__(self) -> None:
        for field in _ARTIFACT_IDENTITY_FIELDS[:6]:
            value = getattr(self, field)
            _validate_exact_string(value, field=field)
        _validate_current_artifact_versions(self)
        dataset_snapshot = _validated_dataset_identity_snapshot(
            self.training_dataset,
            context="training_dataset",
        )
        object.__setattr__(self, "training_dataset", dataset_snapshot)
        if self.symbol != dataset_snapshot.symbol:
            raise ArtifactCompatibilityError(
                "symbol mismatch inside artifact identity: "
                f"expected={_safe_diagnostic(self.symbol)} "
                f"actual={_safe_diagnostic(dataset_snapshot.symbol)}"
            )
        if self.timeframe != dataset_snapshot.timeframe:
            raise ArtifactCompatibilityError(
                "timeframe mismatch inside artifact identity: "
                f"expected={_safe_diagnostic(self.timeframe)} "
                f"actual={_safe_diagnostic(dataset_snapshot.timeframe)}"
            )
        config = _require_mapping(
            self.training_config,
            context="training_config",
            root_path="$.training_config",
        )
        _validate_json_graph(
            config,
            allow_internal=False,
            root_path="$.training_config",
        )
        _reject_environment_config(config)
        _validate_training_config_fields(config)
        if not isinstance(config, _FrozenDict):
            canonical_json_bytes(config)
        frozen_config = _json_value(
            config,
            path="training_config",
            freeze=True,
            allow_internal=False,
        )
        object.__setattr__(self, "training_config", frozen_config)
        actual_hash = _validate_hash(
            self.training_config_hash,
            field="training_config_hash",
        )
        expected_hash = sha256_json(_thaw_json(frozen_config))
        if actual_hash != expected_hash:
            raise ArtifactCompatibilityError(
                "training_config_hash mismatch: "
                f"expected={expected_hash!r} actual={actual_hash!r}"
            )

    def _validated_payload(self) -> dict[str, object]:
        return {
            "core_semantics_version": self.core_semantics_version,
            "vocab_version": self.vocab_version,
            "label_semantics_version": self.label_semantics_version,
            "execution_semantics_version": self.execution_semantics_version,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "training_dataset": _dataset_identity_payload(self.training_dataset),
            "training_config": _thaw_json(self.training_config),
            "training_config_hash": self.training_config_hash,
        }

    def to_dict(self) -> dict[str, object]:
        snapshot = _validated_artifact_identity_snapshot(
            self,
            context="artifact_identity.to_dict",
        )
        return snapshot._validated_payload()

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "ArtifactIdentity":
        mapping = _require_mapping(value, context="artifact identity")
        _require_exact_fields(mapping, _ARTIFACT_IDENTITY_FIELDS, context="artifact identity")
        dataset_value = _require_mapping(
            mapping["training_dataset"],
            context="training_dataset",
        )
        return cls(
            core_semantics_version=mapping["core_semantics_version"],  # type: ignore[arg-type]
            vocab_version=mapping["vocab_version"],  # type: ignore[arg-type]
            label_semantics_version=mapping["label_semantics_version"],  # type: ignore[arg-type]
            execution_semantics_version=mapping["execution_semantics_version"],  # type: ignore[arg-type]
            symbol=mapping["symbol"],  # type: ignore[arg-type]
            timeframe=mapping["timeframe"],  # type: ignore[arg-type]
            training_dataset=_strict_dataset_identity_from_dict(
                dataset_value,
                context="training_dataset",
            ),
            training_config=_require_mapping(
                mapping["training_config"],
                context="training_config",
            ),
            training_config_hash=mapping["training_config_hash"],  # type: ignore[arg-type]
        )

    @property
    def fingerprint(self) -> str:
        return sha256_json(self.to_dict())


def _validated_artifact_identity_snapshot(
    value: object,
    *,
    context: str,
) -> ArtifactIdentity:
    if not isinstance(value, ArtifactIdentity):
        raise ArtifactCompatibilityError(
            f"{context} is invalid: expected=ArtifactIdentity "
            f"actual={_safe_type_category(value)}"
        )
    try:
        return ArtifactIdentity(
            core_semantics_version=value.core_semantics_version,
            vocab_version=value.vocab_version,
            label_semantics_version=value.label_semantics_version,
            execution_semantics_version=value.execution_semantics_version,
            symbol=value.symbol,
            timeframe=value.timeframe,
            training_dataset=value.training_dataset,
            training_config=value.training_config,
            training_config_hash=value.training_config_hash,
        )
    except ArtifactCompatibilityError as exc:
        raise ArtifactCompatibilityError(
            _bounded_public_message(
                f"{context} revalidation failed: "
                f"cause={_safe_diagnostic_text(exc)}"
            )
        ) from _safe_exception_cause(exc)


def verify_artifact_identity(
    expected: ArtifactIdentity,
    actual: ArtifactIdentity,
) -> None:
    expected_snapshot = _validated_artifact_identity_snapshot(
        expected,
        context="verify_artifact_identity.expected",
    )
    actual_snapshot = _validated_artifact_identity_snapshot(
        actual,
        context="verify_artifact_identity.actual",
    )
    mismatches: list[str] = []
    for field in _ARTIFACT_IDENTITY_FIELDS:
        expected_value = getattr(expected_snapshot, field)
        actual_value = getattr(actual_snapshot, field)
        if expected_value != actual_value:
            mismatches.append(
                f"{field}: expected={_safe_diagnostic(expected_value)} "
                f"actual={_safe_diagnostic(actual_value)}"
            )
    if mismatches:
        raise ArtifactCompatibilityError(
            _bounded_public_message(
                "artifact identity mismatch; " + "; ".join(mismatches)
            )
        )


def _safe_component(value: str) -> str:
    _validate_exact_string(value, field="filename identity component")
    safe = _SAFE_FILENAME.sub("_", value).strip("._")
    if not safe:
        raise ArtifactCompatibilityError(
            "identity value cannot form a safe filename component: "
            f"actual={_safe_diagnostic(value)}"
        )
    return safe


def _validate_run_id(value: object) -> str:
    if type(value) is not str:
        raise ArtifactCompatibilityError(
            "run_id is invalid: expected=UUID lowercase hex exact built-in string "
            f"actual_type={_safe_type_category(value)}"
        )
    if _HEX_32.fullmatch(value) is None:
        raise ArtifactCompatibilityError(
            "run_id is invalid: expected=UUID lowercase hex "
            f"actual={_safe_diagnostic(value)}"
        )
    try:
        uuid.UUID(hex=value)
    except ValueError as exc:
        raise ArtifactCompatibilityError(
            "run_id is invalid: expected=UUID lowercase hex "
            f"actual={_safe_diagnostic(value)}"
        ) from exc
    return value


@dataclass(frozen=True)
class TrainingRunIdentity:
    run_id: str
    artifact_identity: ArtifactIdentity

    def __post_init__(self) -> None:
        _validate_run_id(self.run_id)
        identity_snapshot = _validated_artifact_identity_snapshot(
            self.artifact_identity,
            context="TrainingRunIdentity.artifact_identity",
        )
        object.__setattr__(self, "artifact_identity", identity_snapshot)

    @classmethod
    def create(cls, artifact_identity: ArtifactIdentity) -> "TrainingRunIdentity":
        return cls(run_id=uuid.uuid4().hex, artifact_identity=artifact_identity)

    def to_dict(self) -> dict[str, object]:
        _validate_run_id(self.run_id)
        identity_snapshot = _validated_artifact_identity_snapshot(
            self.artifact_identity,
            context="TrainingRunIdentity.to_dict.artifact_identity",
        )
        return {
            "run_id": self.run_id,
            "artifact_identity": identity_snapshot._validated_payload(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "TrainingRunIdentity":
        mapping = _require_mapping(value, context="run identity")
        _require_exact_fields(
            mapping,
            ("run_id", "artifact_identity"),
            context="run identity",
        )
        return cls(
            run_id=mapping["run_id"],  # type: ignore[arg-type]
            artifact_identity=ArtifactIdentity.from_dict(
                _require_mapping(
                    mapping["artifact_identity"],
                    context="artifact_identity",
                )
            ),
        )

    def _filename_prefix(self) -> str:
        _validate_run_id(self.run_id)
        identity = _validated_artifact_identity_snapshot(
            self.artifact_identity,
            context="TrainingRunIdentity.filename.artifact_identity",
        )
        return (
            f"{_safe_component(identity.symbol)}_"
            f"{_safe_component(identity.timeframe)}_"
            f"{identity.training_dataset.data_fingerprint[:12]}_"
            f"run_{self.run_id[:8]}"
        )

    def checkpoint_filename(self, step: int) -> str:
        if type(step) is not int:
            raise ArtifactCompatibilityError(
                _bounded_public_message(
                    "checkpoint step is invalid: "
                    "expected=exact built-in integer >= 0 "
                    "actual_category=non-exact-integer"
                )
            )
        if not _is_exact_operational_integer(step):
            raise ArtifactCompatibilityError(
                "checkpoint step is outside the operational integer range; "
                f"actual={_safe_diagnostic(step)}"
            )
        if step < 0:
            raise ArtifactCompatibilityError(
                "checkpoint step must be a non-negative integer; "
                f"actual={_safe_diagnostic(step)}"
            )
        return f"ckpt_v2_{self._filename_prefix()}_step_{step}.pt"

    def strategy_filename(self) -> str:
        return f"best_v2_{self._filename_prefix()}.json"

    def history_filename(self) -> str:
        return f"training_history_v2_{self._filename_prefix()}.json"


@dataclass(frozen=True)
class FoldEvidence:
    fold_index: int
    train_start_time_ns: int
    train_end_time_ns: int
    val_start_time_ns: int
    val_end_time_ns: int
    effective_gap: int
    validation_metrics: Mapping[str, object]

    def __post_init__(self) -> None:
        for field in _FOLD_FIELDS[:-1]:
            value = getattr(self, field)
            if not _is_exact_operational_integer(value):
                raise ArtifactCompatibilityError(
                    f"{field} is invalid: expected=exact built-in integer in operational range "
                    f"actual={_safe_diagnostic(value)} "
                    f"actual_type={_safe_type_category(value)}"
                )
        if self.fold_index < 0:
            raise ArtifactCompatibilityError(
                f"fold_index is invalid: expected=integer >= 0 actual={self.fold_index!r}"
            )
        if self.effective_gap < LABEL_LOOKAHEAD_BARS:
            raise ArtifactCompatibilityError(
                "effective_gap is invalid: "
                f"expected=integer >= {LABEL_LOOKAHEAD_BARS} "
                f"actual={self.effective_gap!r}"
            )
        if self.train_start_time_ns > self.train_end_time_ns:
            raise ArtifactCompatibilityError("invalid fold training time range")
        if self.val_start_time_ns > self.val_end_time_ns:
            raise ArtifactCompatibilityError("invalid fold validation time range")
        if self.train_end_time_ns >= self.val_start_time_ns:
            raise ArtifactCompatibilityError(
                "invalid fold lineage: train_end_time_ns must precede "
                "val_start_time_ns; "
                "expected=train_end_time_ns < val_start_time_ns "
                f"actual={self.train_end_time_ns} >= {self.val_start_time_ns}"
            )
        metrics = _require_mapping(
            self.validation_metrics,
            context="validation_metrics",
        )
        if not metrics:
            raise ArtifactCompatibilityError(
                "validation_metrics is invalid: "
                "expected=non-empty mapping of finite real numbers actual={}"
            )
        for key, metric in metrics.items():
            if not isinstance(key, str) or not key:
                raise ArtifactCompatibilityError(
                    "validation_metrics key is invalid: "
                    f"expected=non-empty string actual={_safe_diagnostic(key)}"
                )
            if (
                isinstance(metric, bool)
                or not _is_exact_operational_real(metric)
            ):
                raise ArtifactCompatibilityError(
                    f"validation_metrics.{key} is invalid: "
                    f"expected=finite real number actual={_safe_diagnostic(metric)}"
                )
        _validate_json_graph(
            metrics,
            allow_internal=True,
            root_path="$.validation_metrics",
        )
        object.__setattr__(
            self,
            "validation_metrics",
            _json_value(
                metrics,
                path="validation_metrics",
                freeze=True,
                allow_internal=True,
            ),
        )

    def to_dict(self) -> dict[str, object]:
        snapshot = FoldEvidence(
            fold_index=self.fold_index,
            train_start_time_ns=self.train_start_time_ns,
            train_end_time_ns=self.train_end_time_ns,
            val_start_time_ns=self.val_start_time_ns,
            val_end_time_ns=self.val_end_time_ns,
            effective_gap=self.effective_gap,
            validation_metrics=self.validation_metrics,
        )
        return _fold_evidence_payload(snapshot)

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "FoldEvidence":
        mapping = _require_mapping(value, context="fold evidence")
        _require_exact_fields(mapping, _FOLD_FIELDS, context="fold evidence")
        return cls(
            fold_index=mapping["fold_index"],  # type: ignore[arg-type]
            train_start_time_ns=mapping["train_start_time_ns"],  # type: ignore[arg-type]
            train_end_time_ns=mapping["train_end_time_ns"],  # type: ignore[arg-type]
            val_start_time_ns=mapping["val_start_time_ns"],  # type: ignore[arg-type]
            val_end_time_ns=mapping["val_end_time_ns"],  # type: ignore[arg-type]
            effective_gap=mapping["effective_gap"],  # type: ignore[arg-type]
            validation_metrics=_require_mapping(
                mapping["validation_metrics"],
                context="validation_metrics",
            ),
        )


def _fold_evidence_payload(value: FoldEvidence) -> dict[str, object]:
    return {
        "fold_index": value.fold_index,
        "train_start_time_ns": value.train_start_time_ns,
        "train_end_time_ns": value.train_end_time_ns,
        "val_start_time_ns": value.val_start_time_ns,
        "val_end_time_ns": value.val_end_time_ns,
        "effective_gap": value.effective_gap,
        "validation_metrics": _thaw_json(value.validation_metrics),
    }


def _validate_generated_at(value: object) -> str:
    if type(value) is not str:
        raise ArtifactCompatibilityError(
            "generated_at must be a UTC timestamp exact built-in string; "
            f"actual_type={_safe_type_category(value)}"
        )
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ArtifactCompatibilityError(
            "generated_at must be a UTC timestamp string; "
            f"actual={_safe_diagnostic(value)}"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ArtifactCompatibilityError(
            f"generated_at must be UTC; actual={_safe_diagnostic(value)}"
        )
    return value


def _validate_formula_tokens(value: object) -> tuple[int, ...]:
    if type(value) is not list:
        raise ArtifactCompatibilityError("formula_tokens must be a JSON integer list")
    tokens: list[int] = []
    for index, token in enumerate(value):
        if type(token) is not int:
            raise ArtifactCompatibilityError(
                _bounded_public_message(
                    "formula token is invalid: "
                    f"index={index} expected=exact built-in integer "
                    "actual_category=non-exact-integer"
                )
            )
        if not _is_exact_operational_integer(token):
            raise ArtifactCompatibilityError(
                f"formula token is outside the operational integer range: index={index} "
                f"actual={_safe_diagnostic(token)}"
            )
        if token < 0 or token >= FORMULA_VOCAB.size:
            raise ArtifactCompatibilityError(
                f"formula token out of range: index={index} actual={token!r} "
                f"expected=0..{FORMULA_VOCAB.size - 1}"
            )
        tokens.append(token)
    if not tokens:
        raise ArtifactCompatibilityError("formula_tokens must not be empty")
    violations = validate_formula_structure(tokens, FORMULA_VOCAB.token_names)
    if violations:
        raise ArtifactCompatibilityError(
            "formula structure is incompatible: " + "; ".join(violations)
        )
    return tuple(tokens)


def _validate_decoded_formula(tokens: tuple[int, ...], value: object) -> str:
    expected = " -> ".join(FORMULA_VOCAB.token_names[token] for token in tokens)
    if type(value) is not str:
        raise ArtifactCompatibilityError(
            "decoded_formula is invalid: expected=exact built-in string "
            f"actual_type={_safe_type_category(value)}"
        )
    if value != expected:
        raise ArtifactCompatibilityError(
            "decoded_formula mismatch: "
            f"expected={_safe_diagnostic(expected)} "
            f"actual={_safe_diagnostic(value)}"
        )
    return value


def _verify_current_semantics(identity: ArtifactIdentity) -> None:
    _validate_current_artifact_versions(identity)


def _validated_strategy_run_identity(
    value: object,
) -> TrainingRunIdentity:
    if not isinstance(value, TrainingRunIdentity):
        raise ArtifactCompatibilityError(
            "run_identity must be a TrainingRunIdentity"
        )
    validated = TrainingRunIdentity.from_dict(value.to_dict())
    _verify_current_semantics(validated.artifact_identity)
    return validated


def _validated_fold_evidence(value: object, *, index: int) -> FoldEvidence:
    if not isinstance(value, FoldEvidence):
        raise ArtifactCompatibilityError(
            f"fold_evidence[{index}] must be a FoldEvidence"
        )
    return FoldEvidence.from_dict(_fold_evidence_payload(value))


def _validated_fold_evidence_tuple(
    values: Sequence[object],
) -> tuple[FoldEvidence, ...]:
    validated = tuple(
        _validated_fold_evidence(value, index=index)
        for index, value in enumerate(values)
    )
    actual_indices = tuple(fold.fold_index for fold in validated)
    expected_indices = tuple(range(len(validated)))
    if actual_indices != expected_indices:
        raise ArtifactCompatibilityError(
            _bounded_public_message(
                "fold_evidence fold_index sequence is invalid: "
                f"expected={_safe_diagnostic_sequence(expected_indices)} "
                f"actual={_safe_diagnostic_sequence(actual_indices)}"
            )
        )
    for index, (previous, current) in enumerate(
        zip(validated, validated[1:]),
        start=1,
    ):
        comparisons = (
            (
                "train_start_time_ns",
                current.train_start_time_ns == previous.train_start_time_ns,
                f"equal to {previous.train_start_time_ns}",
                current.train_start_time_ns,
            ),
            (
                "train_end_time_ns",
                current.train_end_time_ns > previous.train_end_time_ns,
                f"greater than {previous.train_end_time_ns}",
                current.train_end_time_ns,
            ),
            (
                "val_start_time_ns",
                current.val_start_time_ns > previous.val_end_time_ns,
                f"greater than {previous.val_end_time_ns}",
                current.val_start_time_ns,
            ),
            (
                "val_end_time_ns",
                current.val_end_time_ns > previous.val_end_time_ns,
                f"greater than {previous.val_end_time_ns}",
                current.val_end_time_ns,
            ),
            (
                "train_end_time_ns",
                current.train_end_time_ns >= previous.val_end_time_ns,
                f"at least {previous.val_end_time_ns}",
                current.train_end_time_ns,
            ),
        )
        for field, valid, expected, actual in comparisons:
            if not valid:
                raise ArtifactCompatibilityError(
                    f"fold_evidence[{index}].{field} breaks expanding lineage: "
                    f"expected={expected} actual={actual}"
                )
    return validated


def _validated_strategy_lineage(
    run_identity: object,
    fold_evidence: Sequence[object],
) -> tuple[TrainingRunIdentity, tuple[FoldEvidence, ...]]:
    if type(fold_evidence) not in (list, tuple):
        raise ArtifactCompatibilityError(
            "fold_evidence unsupported JSON array type: "
            f"expected=exact built-in list or tuple actual_type={_safe_type_category(fold_evidence)}"
        )
    validated_run_identity = _validated_strategy_run_identity(run_identity)
    validated_folds = _validated_fold_evidence_tuple(fold_evidence)
    artifact_identity = validated_run_identity.artifact_identity
    dataset = artifact_identity.training_dataset
    walk_forward = artifact_identity.training_config["walk_forward"]
    assert isinstance(walk_forward, Mapping)
    expected_fold_count = _APPROVED_WF_VALIDATION_FOLDS
    actual_fold_count = len(validated_folds)
    if actual_fold_count != expected_fold_count:
        raise ArtifactCompatibilityError(
            "fold_evidence count contradicts training_config.walk_forward.blocks: "
            f"expected={expected_fold_count} actual={actual_fold_count}"
        )
    expected_effective_gap = max(
        walk_forward["gap"],
        walk_forward["label_lookahead"],
    )
    for index, fold in enumerate(validated_folds):
        if fold.effective_gap != expected_effective_gap:
            raise ArtifactCompatibilityError(
                f"fold_evidence[{index}].effective_gap contradicts training_config: "
                f"expected={expected_effective_gap} actual={fold.effective_gap}"
            )
        for field in (
            "train_start_time_ns",
            "train_end_time_ns",
            "val_start_time_ns",
            "val_end_time_ns",
        ):
            actual = getattr(fold, field)
            if actual < dataset.start_time_ns or actual > dataset.end_time_ns:
                raise ArtifactCompatibilityError(
                    f"fold_evidence[{index}].{field} is outside training_dataset: "
                    f"expected={dataset.start_time_ns}..{dataset.end_time_ns} "
                    f"actual={actual!r}"
                )
    return validated_run_identity, validated_folds


@dataclass(frozen=True)
class StrategyArtifact:
    schema_version: str
    run_identity: TrainingRunIdentity
    formula_tokens: tuple[int, ...]
    decoded_formula: str
    best_score: float
    fold_evidence: tuple[FoldEvidence, ...]
    generated_at: str
    fingerprint: str

    def __post_init__(self) -> None:
        _validate_exact_string(self.schema_version, field="schema_version")
        if self.schema_version != "strategy-v2":
            raise ArtifactCompatibilityError(
                "unknown strategy schema: "
                "expected='strategy-v2' "
                f"actual={_safe_diagnostic(self.schema_version)}"
            )
        validated_run_identity, validated_folds = _validated_strategy_lineage(
            self.run_identity,
            self.fold_evidence,
        )
        object.__setattr__(self, "run_identity", validated_run_identity)
        if type(self.formula_tokens) is not tuple:
            raise ArtifactCompatibilityError("formula_tokens must be immutable")
        validated_tokens = _validate_formula_tokens(list(self.formula_tokens))
        if validated_tokens != self.formula_tokens:
            raise ArtifactCompatibilityError("formula_tokens are not canonical")
        _validate_decoded_formula(self.formula_tokens, self.decoded_formula)
        if not _is_exact_operational_real(self.best_score):
            raise ArtifactCompatibilityError(
                "best_score is invalid: "
                "expected=exact built-in operational finite real number "
                f"actual={_safe_diagnostic(self.best_score)}"
            )
        if type(self.fold_evidence) is not tuple or not all(
            isinstance(item, FoldEvidence) for item in self.fold_evidence
        ):
            raise ArtifactCompatibilityError("fold_evidence must contain FoldEvidence values")
        object.__setattr__(self, "fold_evidence", validated_folds)
        _validate_generated_at(self.generated_at)
        supplied_fingerprint = _validate_hash(self.fingerprint, field="fingerprint")
        expected_fingerprint = sha256_json(self._payload_without_fingerprint())
        if supplied_fingerprint != expected_fingerprint:
            raise ArtifactCompatibilityError(
                "strategy fingerprint mismatch: "
                f"expected={expected_fingerprint!r} actual={supplied_fingerprint!r}"
            )

    @classmethod
    def create(
        cls,
        *,
        run_identity: TrainingRunIdentity,
        formula_tokens: Sequence[int],
        decoded_formula: str,
        best_score: float,
        fold_evidence: Sequence[FoldEvidence],
        generated_at: str,
    ) -> "StrategyArtifact":
        if type(formula_tokens) not in (list, tuple):
            raise ArtifactCompatibilityError(
                "formula_tokens unsupported JSON array type: "
                f"expected=exact built-in list or tuple actual_type={_safe_type_category(formula_tokens)}"
            )
        if type(fold_evidence) not in (list, tuple):
            raise ArtifactCompatibilityError(
                "fold_evidence unsupported JSON array type: "
                f"expected=exact built-in list or tuple actual_type={_safe_type_category(fold_evidence)}"
            )
        validated_run_identity, validated_folds = _validated_strategy_lineage(
            run_identity,
            tuple(fold_evidence),
        )
        tokens = _validate_formula_tokens(list(formula_tokens))
        decoded = _validate_decoded_formula(tokens, decoded_formula)
        payload = {
            "schema_version": "strategy-v2",
            "run_identity": validated_run_identity.to_dict(),
            "formula_tokens": list(tokens),
            "decoded_formula": decoded,
            "best_score": best_score,
            "fold_evidence": [item.to_dict() for item in validated_folds],
            "generated_at": _validate_generated_at(generated_at),
        }
        return cls(
            schema_version="strategy-v2",
            run_identity=validated_run_identity,
            formula_tokens=tokens,
            decoded_formula=decoded,
            best_score=best_score,
            fold_evidence=validated_folds,
            generated_at=generated_at,
            fingerprint=sha256_json(payload),
        )

    def _payload_without_fingerprint(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "run_identity": self.run_identity.to_dict(),
            "formula_tokens": list(self.formula_tokens),
            "decoded_formula": self.decoded_formula,
            "best_score": self.best_score,
            "fold_evidence": [item.to_dict() for item in self.fold_evidence],
            "generated_at": self.generated_at,
        }

    def to_dict(self) -> dict[str, object]:
        snapshot = _validated_strategy_artifact_snapshot(
            self,
            context="StrategyArtifact.to_dict",
        )
        return {
            **snapshot._payload_without_fingerprint(),
            "fingerprint": snapshot.fingerprint,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "StrategyArtifact":
        mapping = _require_mapping(value, context="strategy artifact")
        _require_exact_fields(mapping, _STRATEGY_FIELDS, context="strategy artifact")
        _validate_exact_string(mapping["schema_version"], field="schema_version")
        if mapping["schema_version"] != "strategy-v2":
            raise ArtifactCompatibilityError(
                "unknown strategy schema: "
                "expected='strategy-v2' "
                f"actual={_safe_diagnostic(mapping['schema_version'])}"
            )
        run_identity = TrainingRunIdentity.from_dict(
            _require_mapping(mapping["run_identity"], context="run_identity")
        )
        tokens = _validate_formula_tokens(mapping["formula_tokens"])
        decoded_formula = _validate_decoded_formula(
            tokens,
            mapping["decoded_formula"],
        )
        raw_folds = mapping["fold_evidence"]
        if not isinstance(raw_folds, list):
            raise ArtifactCompatibilityError("fold_evidence must be a JSON list")
        folds = tuple(
            FoldEvidence.from_dict(
                _require_mapping(item, context=f"fold_evidence[{index}]")
            )
            for index, item in enumerate(raw_folds)
        )
        supplied_fingerprint = _validate_hash(
            mapping["fingerprint"],
            field="fingerprint",
        )
        unsigned_payload = {
            key: item for key, item in mapping.items() if key != "fingerprint"
        }
        expected_fingerprint = sha256_json(unsigned_payload)
        if supplied_fingerprint != expected_fingerprint:
            raise ArtifactCompatibilityError(
                "strategy fingerprint mismatch: "
                f"expected={expected_fingerprint!r} actual={supplied_fingerprint!r}"
            )
        return cls(
            schema_version="strategy-v2",
            run_identity=run_identity,
            formula_tokens=tokens,
            decoded_formula=decoded_formula,
            best_score=mapping["best_score"],  # type: ignore[arg-type]
            fold_evidence=folds,
            generated_at=mapping["generated_at"],  # type: ignore[arg-type]
            fingerprint=supplied_fingerprint,
        )


class _DuplicateJsonMember(ValueError):
    pass


class _ArtifactJsonParseError(ValueError):
    pass


def _artifact_json_text(raw: object) -> str:
    if type(raw) is str:
        if len(raw) > _MAX_ARTIFACT_JSON_BYTES:
            raise ArtifactCompatibilityError(
                "artifact JSON exceeds the bounded size limit: "
                f"maximum={_MAX_ARTIFACT_JSON_BYTES}"
            )
        try:
            encoded = raw.encode("utf-8")
        except UnicodeError:
            raise ArtifactCompatibilityError("invalid artifact JSON text encoding") from None
        if len(encoded) > _MAX_ARTIFACT_JSON_BYTES:
            raise ArtifactCompatibilityError(
                "artifact JSON exceeds the bounded size limit: "
                f"maximum={_MAX_ARTIFACT_JSON_BYTES}"
            )
        return raw
    if type(raw) is bytes:
        if len(raw) > _MAX_ARTIFACT_JSON_BYTES:
            raise ArtifactCompatibilityError(
                "artifact JSON exceeds the bounded size limit: "
                f"maximum={_MAX_ARTIFACT_JSON_BYTES}"
            )
        try:
            return raw.decode("utf-8-sig")
        except UnicodeError:
            raise ArtifactCompatibilityError("invalid artifact JSON text encoding") from None
    raise ArtifactCompatibilityError(
        "artifact JSON input is invalid: expected=exact built-in str or bytes "
        "actual_category=unsupported-text-input"
    )


def _validate_artifact_json_text_shape(raw: str) -> None:
    depth = 0
    containers = 0
    in_string = False
    escaped = False
    for character in raw:
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character in "{[":
            depth += 1
            containers += 1
            if depth > _MAX_JSON_NESTING_DEPTH:
                raise ArtifactCompatibilityError(
                    "artifact JSON nesting depth exceeds the bounded limit: "
                    f"maximum={_MAX_JSON_NESTING_DEPTH}"
                )
            if containers > _MAX_ARTIFACT_JSON_CONTAINERS:
                raise ArtifactCompatibilityError(
                    "artifact JSON container count exceeds the bounded limit: "
                    f"maximum={_MAX_ARTIFACT_JSON_CONTAINERS}"
                )
        elif character in "}]":
            depth -= 1


def _artifact_object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonMember
        result[key] = value
    return result


def _reject_json_constant(_value: str) -> object:
    raise _ArtifactJsonParseError


def load_artifact_json(raw: object, artifact_type: type[object]) -> object:
    """Load one supported V2 identity or artifact from duplicate-safe JSON text."""
    text = _artifact_json_text(raw)
    _validate_artifact_json_text_shape(text)
    try:
        payload = json.loads(
            text,
            object_pairs_hook=_artifact_object_pairs,
            parse_constant=_reject_json_constant,
        )
    except _DuplicateJsonMember:
        raise ArtifactCompatibilityError(
            "duplicate JSON object member is not allowed"
        ) from None
    except (
        json.JSONDecodeError,
        _ArtifactJsonParseError,
        UnicodeError,
        OverflowError,
        RecursionError,
        TypeError,
        ValueError,
        MemoryError,
    ):
        raise ArtifactCompatibilityError("invalid artifact JSON") from None
    if type(payload) is not dict:
        raise ArtifactCompatibilityError(
            "artifact JSON root is invalid: expected=object actual=non-object"
        )
    try:
        if artifact_type is DatasetIdentity:
            return _strict_dataset_identity_from_dict(
                payload,
                context="dataset identity JSON",
            )
        if artifact_type is ArtifactIdentity:
            return ArtifactIdentity.from_dict(payload)
        if artifact_type is TrainingRunIdentity:
            return TrainingRunIdentity.from_dict(payload)
        if artifact_type is FoldEvidence:
            return FoldEvidence.from_dict(payload)
        if artifact_type is StrategyArtifact:
            return StrategyArtifact.from_dict(payload)
        raise ArtifactCompatibilityError(
            "artifact JSON target type is unsupported: "
            "expected=V2 identity or artifact class"
        )
    except ArtifactCompatibilityError as exc:
        raise ArtifactCompatibilityError(
            _bounded_public_message(
                "artifact JSON validation failed: "
                f"cause={_safe_diagnostic_text(exc)}"
            )
        ) from _safe_exception_cause(exc)
    except Exception as exc:
        raise ArtifactCompatibilityError(
            _bounded_public_message(
                "artifact JSON validation failed: "
                f"cause_category={_safe_exception_category(exc)} "
                f"cause={_safe_diagnostic_text(exc)}"
            )
        ) from exc


def _validated_strategy_artifact_snapshot(
    value: object,
    *,
    context: str,
) -> StrategyArtifact:
    if not isinstance(value, StrategyArtifact):
        raise ArtifactCompatibilityError(
            f"{context} is invalid: expected=StrategyArtifact "
            f"actual={_safe_type_category(value)}"
        )
    try:
        return StrategyArtifact(
            schema_version=value.schema_version,
            run_identity=value.run_identity,
            formula_tokens=value.formula_tokens,
            decoded_formula=value.decoded_formula,
            best_score=value.best_score,
            fold_evidence=value.fold_evidence,
            generated_at=value.generated_at,
            fingerprint=value.fingerprint,
        )
    except ArtifactCompatibilityError as exc:
        raise ArtifactCompatibilityError(
            _bounded_public_message(
                f"{context} revalidation failed: "
                f"cause={_safe_diagnostic_text(exc)}"
            )
        ) from _safe_exception_cause(exc)


class BacktestMode(str, Enum):
    IN_SAMPLE_REPLAY = "in_sample_replay"
    OUT_OF_SAMPLE_BACKTEST = "out_of_sample_backtest"


@dataclass(frozen=True)
class _DatasetIdentityDiagnosticSnapshot:
    start_time_ns: object
    end_time_ns: object
    data_fingerprint: object
    time_fingerprint: object


def _backtest_context(
    *,
    mode: object,
    train: object,
    test: object,
) -> str:
    mode_value = (
        mode.value if isinstance(mode, BacktestMode) else _safe_diagnostic(mode)
    )
    train_start = _safe_diagnostic_getattr(train, "start_time_ns")
    train_end = _safe_diagnostic_getattr(train, "end_time_ns")
    test_start = _safe_diagnostic_getattr(test, "start_time_ns")
    test_end = _safe_diagnostic_getattr(test, "end_time_ns")
    train_data_fp = _safe_diagnostic_text(
        _safe_diagnostic_getattr(train, "data_fingerprint")
    )[:12]
    test_data_fp = _safe_diagnostic_text(
        _safe_diagnostic_getattr(test, "data_fingerprint")
    )[:12]
    train_time_fp = _safe_diagnostic_text(
        _safe_diagnostic_getattr(train, "time_fingerprint")
    )[:12]
    test_time_fp = _safe_diagnostic_text(
        _safe_diagnostic_getattr(test, "time_fingerprint")
    )[:12]
    return (
        f"mode={mode_value} "
        f"train_range=[{_safe_diagnostic_text(train_start)},{_safe_diagnostic_text(train_end)}] "
        f"test_range=[{_safe_diagnostic_text(test_start)},{_safe_diagnostic_text(test_end)}] "
        f"train_data_fp={train_data_fp} "
        f"test_data_fp={test_data_fp} "
        f"train_time_fp={train_time_fp} "
        f"test_time_fp={test_time_fp}"
    )


def validate_backtest_dataset(
    strategy: StrategyArtifact,
    test_identity: DatasetIdentity,
    mode: BacktestMode,
) -> None:
    if not isinstance(strategy, StrategyArtifact):
        raise BacktestModeError("strategy must be a validated StrategyArtifact")
    if not isinstance(test_identity, DatasetIdentity):
        raise BacktestModeError("test_identity must be a DatasetIdentity")
    raw_run = _safe_diagnostic_getattr(strategy, "run_identity")
    raw_artifact_identity = _safe_diagnostic_getattr(raw_run, "artifact_identity")
    raw_train = _safe_diagnostic_getattr(
        raw_artifact_identity,
        "training_dataset",
    )
    safe_raw_train = raw_train if type(raw_train) is DatasetIdentity else None
    context = _backtest_context(mode=mode, train=safe_raw_train, test=None)

    def fail(reason: str, *, cause: BaseException | None = None) -> None:
        error = BacktestModeError(
            _bounded_public_message(f"{reason}; {context}")
        )
        if cause is None:
            raise error
        raise error from cause

    def snapshot_dataset_identity(
        value: DatasetIdentity,
        *,
        label: str,
    ) -> DatasetIdentity:
        nonlocal context
        try:
            payload = _snapshot_dataset_identity_payload(
                value,
                context=f"{label}_dataset",
            )
        except ArtifactCompatibilityError as exc:
            fail(
                f"invalid {label} dataset identity: "
                f"{_safe_diagnostic_text(exc)}",
                cause=_safe_exception_cause(exc),
            )
        diagnostic_snapshot = _DatasetIdentityDiagnosticSnapshot(
            start_time_ns=payload["start_time_ns"],
            end_time_ns=payload["end_time_ns"],
            data_fingerprint=payload["data_fingerprint"],
            time_fingerprint=payload["time_fingerprint"],
        )
        if label == "test":
            context = _backtest_context(
                mode=mode,
                train=safe_raw_train,
                test=diagnostic_snapshot,
            )
        else:
            context = _backtest_context(
                mode=mode,
                train=diagnostic_snapshot,
                test=test_identity,
            )
        try:
            return _dataset_identity_from_payload(
                payload,
                context=f"{label}_dataset",
            )
        except ArtifactCompatibilityError as exc:
            fail(
                f"invalid {label} dataset identity: "
                f"{_safe_diagnostic_text(exc)}",
                cause=_safe_exception_cause(exc),
            )

    if not isinstance(mode, BacktestMode):
        fail(
            "mode must be an explicit BacktestMode; "
            f"actual={_safe_diagnostic(mode)}"
        )

    test_identity = snapshot_dataset_identity(test_identity, label="test")

    try:
        strategy = _validated_strategy_artifact_snapshot(
            strategy,
            context="validate_backtest_dataset.strategy",
        )
    except ArtifactCompatibilityError as exc:
        if type(raw_train) is DatasetIdentity:
            context = _backtest_context(
                mode=mode,
                train=raw_train,
                test=test_identity,
            )
        detail = _safe_diagnostic_text(exc)
        if "training_dataset" in detail:
            fail(
                "invalid strategy artifact; invalid training dataset identity: "
                f"cause={detail}",
                cause=_safe_exception_cause(exc),
            )
        fail(
            f"invalid strategy artifact: cause={detail}",
            cause=_safe_exception_cause(exc),
        )

    train = strategy.run_identity.artifact_identity.training_dataset
    context = _backtest_context(mode=mode, train=train, test=test_identity)

    identity = strategy.run_identity.artifact_identity
    semantic_fields = (
        ("core_semantics_version", CORE_SEMANTICS_VERSION, identity.core_semantics_version),
        ("vocab_version", VOCAB_VERSION, identity.vocab_version),
        ("label_semantics_version", LABEL_SEMANTICS_VERSION, identity.label_semantics_version),
        (
            "execution_semantics_version",
            EXECUTION_SEMANTICS_VERSION,
            identity.execution_semantics_version,
        ),
    )
    for field, expected, actual in semantic_fields:
        if expected != actual:
            fail(f"{field} mismatch: expected={expected!r} actual={actual!r}")
    for field in ("symbol", "timeframe"):
        expected = getattr(identity, field)
        actual = getattr(test_identity, field)
        if expected != actual:
            fail(f"{field} mismatch: expected={expected!r} actual={actual!r}")

    expected_fingerprint = sha256_json(strategy._payload_without_fingerprint())
    if strategy.fingerprint != expected_fingerprint:
        fail(
            "strategy fingerprint mismatch: "
            f"expected={expected_fingerprint!r} actual={strategy.fingerprint!r}"
        )

    if mode is BacktestMode.IN_SAMPLE_REPLAY:
        if test_identity.data_fingerprint != train.data_fingerprint:
            fail("replay must use exact training data")
        return

    conflicts: list[str] = []
    if test_identity.data_fingerprint == train.data_fingerprint:
        conflicts.append("data fingerprint equals training")
    if test_identity.time_fingerprint == train.time_fingerprint:
        conflicts.append("time fingerprint equals training")
    if test_identity.start_time_ns <= train.end_time_ns:
        conflicts.append("test start must be strictly later than training end")
    if conflicts:
        fail("invalid out-of-sample dataset: " + ", ".join(conflicts))
