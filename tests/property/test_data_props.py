"""Property tests for the V2 market-data contract."""

from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

from hypothesis import given, settings, strategies as st
import numpy as np
import pandas as pd
import pytest
import torch

import data_pipeline.validation as validation
from data_pipeline.data_manager import MT5DataManager, compute_forward_open_returns
from data_pipeline.validation import canonicalize_ohlcv
from model_core.semantics import DataValidationError


def _make_fake_rates(n: int = 5, *, spacing_seconds: int = 3600) -> np.ndarray:
    dtype = np.dtype(
        [
            ("time", np.int64),
            ("open", np.float64),
            ("high", np.float64),
            ("low", np.float64),
            ("close", np.float64),
            ("tick_volume", np.int64),
            ("spread", np.int32),
            ("real_volume", np.int64),
        ]
    )
    data = np.zeros(n, dtype=dtype)
    data["time"] = (
        1_700_000_000
        + np.arange(n, dtype=np.int64) * spacing_seconds
    )
    data["open"] = 1800.0 + np.arange(n, dtype=np.float64)
    data["high"] = data["open"] + 2.0
    data["low"] = data["open"] - 2.0
    data["close"] = data["open"] + 0.5
    data["tick_volume"] = 100 + np.arange(n, dtype=np.int64)
    return data


def _make_symbol_df(
    timestamps: list[int], base_price: float = 100.0
) -> pd.DataFrame:
    n = len(timestamps)
    # Alignment is the property under test; keep generated training values
    # exactly representable in float32 so the canonical downcast gate is not
    # the competing behavior under test.
    opens = base_price + np.arange(n, dtype=np.float64) / 8.0
    return pd.DataFrame(
        {
            "time": np.array(timestamps, dtype=np.int64),
            "open": opens,
            "high": opens + 2.0,
            "low": opens - 2.0,
            "close": opens + 0.5,
            "tick_volume": np.full(n, 500, dtype=np.int64),
        }
    )


symbol_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd")),
    min_size=1,
    max_size=12,
)
SUPPORTED_TIMEFRAMES = (
    ("M1", 1, 60),
    ("M5", 5, 5 * 60),
    ("M15", 15, 15 * 60),
    ("M30", 30, 30 * 60),
    ("H1", 16385, 60 * 60),
    ("H4", 16388, 4 * 60 * 60),
    ("D1", 16408, 24 * 60 * 60),
    ("W1", 32769, 7 * 24 * 60 * 60),
    ("MN1", 49153, 31 * 24 * 60 * 60),
)
timeframe_strategy = st.sampled_from(SUPPORTED_TIMEFRAMES)
INVALID_TIMEFRAMES = (0, -1, 2, 16384, 32770, 49154)
VALID_TIMEFRAME_BOUNDARIES = (
    ("M1", "M1"),
    ("H1", "H1"),
    ("MN1", "MN1"),
    (1, "M1"),
    (16385, "H1"),
    (49153, "MN1"),
)
COERCIBLE_INVALID_TIMEFRAMES = (
    1.0,
    16385.0,
    49153.0,
    "1",
    "16385",
    "49153",
    False,
    True,
)
TOO_SHORT_CADENCES = (
    ("M1", 1, 59),
    ("M5", 5, 60),
    ("M15", 15, 5 * 60),
    ("M30", 30, 15 * 60),
    ("H1", 16385, 30 * 60),
    ("H4", 16388, 60 * 60),
    ("D1", 16408, 4 * 60 * 60),
    ("W1", 32769, 24 * 60 * 60),
)


@pytest.mark.parametrize(("timeframe", "expected"), VALID_TIMEFRAME_BOUNDARIES)
def test_timeframe_normalization_accepts_only_canonical_names_or_builtin_ints(
    timeframe: str | int,
    expected: str,
) -> None:
    assert validation.normalize_timeframe_name(timeframe) == expected


@pytest.mark.parametrize("timeframe", COERCIBLE_INVALID_TIMEFRAMES)
def test_timeframe_normalization_rejects_coercible_non_contract_values(
    timeframe: object,
) -> None:
    with pytest.raises(DataValidationError) as exc_info:
        validation.normalize_timeframe_name(timeframe)  # type: ignore[arg-type]

    assert str(exc_info.value) == f"unknown timeframe: {timeframe!r}"


@settings(max_examples=100)
@given(symbol=symbol_strategy, timeframe_case=timeframe_strategy)
def test_fetcher_returns_canonical_dataframe_when_mt5_succeeds(
    symbol: str, timeframe_case: tuple[str, int, int]
) -> None:
    timeframe_name, timeframe, spacing_seconds = timeframe_case
    fake_rates = _make_fake_rates(n=5, spacing_seconds=spacing_seconds)
    mock_mt5 = MagicMock()
    mock_mt5.copy_rates_from_pos.return_value = fake_rates
    mock_mt5.initialize.return_value = True
    mock_mt5.last_error.return_value = (0, "No error")

    canonicalization_calls: list[tuple[int, str]] = []

    def record_canonicalization(
        frame: pd.DataFrame,
        *,
        symbol: str,
        timeframe: int,
        numeric_time_unit: str | None = None,
    ):
        dataset = canonicalize_ohlcv(
            frame,
            symbol=symbol,
            timeframe=timeframe,
            numeric_time_unit=numeric_time_unit,
        )
        canonicalization_calls.append((timeframe, dataset.identity.timeframe))
        return dataset

    with (
        patch("data_pipeline.fetcher.mt5", mock_mt5),
        patch("data_pipeline.fetcher._MT5_AVAILABLE", True),
        patch("data_pipeline.kline_cache.KlineCache.get", return_value=None),
        patch(
            "data_pipeline.kline_cache.canonicalize_ohlcv",
            side_effect=record_canonicalization,
        ),
    ):
        from data_pipeline.fetcher import MT5DataFetcher

        fetcher = MT5DataFetcher()
        fetcher.connect()
        frame = fetcher.fetch(symbol, timeframe, count=5)

    assert list(frame.columns) == [
        "time",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]
    assert "tick_volume" not in frame.columns
    assert len(frame) == 5
    assert isinstance(frame["time"].dtype, pd.DatetimeTZDtype)
    assert str(frame["time"].dt.tz) == "UTC"
    assert frame["time"].is_monotonic_increasing
    assert frame["time"].is_unique
    for column in ("open", "high", "low", "close", "volume"):
        assert pd.api.types.is_numeric_dtype(frame[column])
        assert np.isfinite(frame[column].to_numpy()).all()
    assert frame["volume"].tolist() == fake_rates["tick_volume"].astype(float).tolist()
    assert canonicalization_calls == [(timeframe, timeframe_name)]

    mock_mt5.copy_rates_from_pos.assert_called_once_with(symbol, timeframe, 1, 5)
    call_args = mock_mt5.copy_rates_from_pos.call_args.args
    assert call_args[2] == 1
    assert call_args[3] == 5


@pytest.mark.parametrize("timeframe", INVALID_TIMEFRAMES)
def test_fetcher_rejects_representative_invalid_timeframes(timeframe: int) -> None:
    mock_mt5 = MagicMock()
    mock_mt5.copy_rates_from_pos.return_value = _make_fake_rates(n=5)
    mock_mt5.initialize.return_value = True
    mock_mt5.last_error.return_value = (0, "No error")

    with (
        patch("data_pipeline.fetcher.mt5", mock_mt5),
        patch("data_pipeline.fetcher._MT5_AVAILABLE", True),
        patch("data_pipeline.kline_cache.KlineCache.get", return_value=None),
    ):
        from data_pipeline.fetcher import MT5DataFetcher

        fetcher = MT5DataFetcher()
        fetcher.connect()
        with pytest.raises(DataValidationError) as exc_info:
            fetcher.fetch("EURUSD", timeframe, count=5)

    assert str(exc_info.value) == f"unknown timeframe: {timeframe!r}"


@settings(max_examples=50)
@given(
    symbol=symbol_strategy,
    timeframe=st.integers(min_value=-100_000, max_value=100_000).filter(
        lambda value: value not in {case[1] for case in SUPPORTED_TIMEFRAMES}
    ),
)
def test_fetcher_rejects_unsupported_timeframe(
    symbol: str, timeframe: int
) -> None:
    mock_mt5 = MagicMock()
    mock_mt5.copy_rates_from_pos.return_value = _make_fake_rates(n=5)
    mock_mt5.initialize.return_value = True
    mock_mt5.last_error.return_value = (0, "No error")

    with (
        patch("data_pipeline.fetcher.mt5", mock_mt5),
        patch("data_pipeline.fetcher._MT5_AVAILABLE", True),
        patch("data_pipeline.kline_cache.KlineCache.get", return_value=None),
    ):
        from data_pipeline.fetcher import MT5DataFetcher

        fetcher = MT5DataFetcher()
        fetcher.connect()
        with pytest.raises(DataValidationError, match="unknown timeframe"):
            fetcher.fetch(symbol, timeframe, count=5)


@pytest.mark.parametrize(
    ("timeframe_name", "timeframe", "spacing_seconds"),
    TOO_SHORT_CADENCES,
)
def test_fetcher_rejects_cadence_shorter_than_requested_identity(
    timeframe_name: str,
    timeframe: int,
    spacing_seconds: int,
) -> None:
    mock_mt5 = MagicMock()
    mock_mt5.copy_rates_from_pos.return_value = _make_fake_rates(
        n=5,
        spacing_seconds=spacing_seconds,
    )
    mock_mt5.initialize.return_value = True
    mock_mt5.last_error.return_value = (0, "No error")
    canonicalization_arguments: list[int] = []

    def record_canonicalization(
        frame: pd.DataFrame,
        *,
        symbol: str,
        timeframe: int,
        numeric_time_unit: str | None = None,
    ):
        canonicalization_arguments.append(timeframe)
        return canonicalize_ohlcv(
            frame,
            symbol=symbol,
            timeframe=timeframe,
            numeric_time_unit=numeric_time_unit,
        )

    with (
        patch("data_pipeline.fetcher.mt5", mock_mt5),
        patch("data_pipeline.fetcher._MT5_AVAILABLE", True),
        patch("data_pipeline.kline_cache.KlineCache.get", return_value=None),
        patch(
            "data_pipeline.kline_cache.canonicalize_ohlcv",
            side_effect=record_canonicalization,
        ),
    ):
        from data_pipeline.fetcher import MT5DataFetcher

        fetcher = MT5DataFetcher()
        fetcher.connect()
        with pytest.raises(
            DataValidationError,
            match=rf"shorter than timeframe {timeframe_name}",
        ):
            fetcher.fetch("EURUSD", timeframe, count=5)

    assert canonicalization_arguments == [timeframe]
    mock_mt5.copy_rates_from_pos.assert_called_once_with(
        "EURUSD", timeframe, 1, 5
    )


@settings(max_examples=100)
@given(
    open_prices=st.lists(
        st.floats(
            min_value=0.01,
            max_value=10_000.0,
            allow_nan=False,
            allow_infinity=False,
        ),
        min_size=3,
        max_size=200,
    ),
    n_symbols=st.integers(min_value=1, max_value=3),
)
def test_forward_return_formula_and_mask_hold_for_positive_prices(
    open_prices: list[float], n_symbols: int
) -> None:
    row = torch.tensor(open_prices, dtype=torch.float64)
    opens = row.unsqueeze(0).expand(n_symbols, len(row)).clone()

    returns, valid = compute_forward_open_returns(opens)

    assert returns.shape == valid.shape == opens.shape
    for idx in range(len(open_prices) - 2):
        expected = math.log(open_prices[idx + 2] / open_prices[idx + 1])
        assert returns[0, idx].item() == pytest.approx(expected)
    assert not valid[:, -2:].any()
    assert torch.equal(returns[:, -2:], torch.zeros_like(returns[:, -2:]))


@st.composite
def multi_symbol_dfs(draw) -> dict[str, pd.DataFrame]:
    common_size = draw(st.integers(min_value=3, max_value=20))
    symbol_count = draw(st.integers(min_value=2, max_value=4))
    base_ts = 1_700_000_000
    common = [base_ts + offset * 3600 for offset in range(common_size)]
    frames: dict[str, pd.DataFrame] = {}
    for idx in range(symbol_count):
        prefix = draw(st.integers(min_value=0, max_value=3))
        suffix = draw(st.integers(min_value=0, max_value=3))
        unique_prefix = [
            base_ts - ((idx + 1) * 10 + offset + 1) * 3600
            for offset in reversed(range(prefix))
        ]
        unique_suffix = [
            base_ts + (common_size + (idx + 1) * 10 + offset) * 3600
            for offset in range(suffix)
        ]
        frame = _make_symbol_df(
            sorted(unique_prefix + common + unique_suffix),
            base_price=100.0 + idx * 100.0,
        )
        frame["time"] = pd.to_datetime(frame["time"], unit="s", utc=True)
        frames[f"SYM{idx}"] = frame
    return frames


@settings(max_examples=100)
@given(frames=multi_symbol_dfs())
def test_alignment_keeps_exact_real_intersection_without_future_fill(
    frames: dict[str, pd.DataFrame]
) -> None:
    fetcher = MagicMock()
    fetcher.fetch.side_effect = lambda symbol, timeframe, count: frames[symbol].copy()
    manager = MT5DataManager(fetcher)

    manager.load(list(frames))

    source_sets = [set(frame["time"].tolist()) for frame in frames.values()]
    expected_times = set.intersection(*source_sets)
    expected_ns = {int(value.value) for value in expected_times}
    actual_ns = set(manager.bar_time[0].tolist())
    assert actual_ns == expected_ns
    assert manager.bar_time.shape == manager.target_ret.shape
    for row in manager.bar_time.tolist():
        assert set(row) == expected_ns


@settings(max_examples=50)
@given(
    offsets=st.lists(
        st.integers(min_value=0, max_value=100),
        min_size=3,
        max_size=30,
        unique=True,
    )
)
def test_canonicalization_never_invents_timestamps(offsets: list[int]) -> None:
    timestamps = [1_700_000_000 + offset * 3600 for offset in offsets]
    frame = _make_symbol_df(sorted(timestamps))

    result = canonicalize_ohlcv(
        frame,
        symbol="EURUSD",
        timeframe="H1",
        numeric_time_unit="s",
    )

    expected_ns = {value * 1_000_000_000 for value in timestamps}
    actual_ns = set(result.frame["time"].astype("int64").tolist())
    assert actual_ns == expected_ns
    assert len(result.frame) == len(frame)


@settings(max_examples=25)
@given(
    mutation=st.sampled_from(["duplicate", "nan", "positive_infinity"]),
    periods=st.integers(min_value=3, max_value=30),
)
def test_duplicate_and_non_finite_inputs_raise_domain_errors(
    mutation: str, periods: int
) -> None:
    timestamps = [1_700_000_000 + idx * 3600 for idx in range(periods)]
    frame = _make_symbol_df(timestamps)
    if mutation == "duplicate":
        frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    elif mutation == "nan":
        frame.loc[0, "open"] = float("nan")
    else:
        frame["tick_volume"] = frame["tick_volume"].astype("float64")
        frame.loc[periods - 1, "tick_volume"] = float("inf")

    with pytest.raises(DataValidationError):
        canonicalize_ohlcv(
            frame,
            symbol="EURUSD",
            timeframe="H1",
            numeric_time_unit="s",
        )


@settings(max_examples=40)
@given(
    unit=st.sampled_from(
        [
            ("s", 1_700_000_000, 3_600, 1_000_000_000),
            ("ms", 1_700_000_000_000, 3_600_000, 1_000_000),
            ("us", 1_700_000_000_000_000, 3_600_000_000, 1_000),
            ("ns", 1_700_000_000_000_000_000, 3_600_000_000_000, 1),
        ]
    ),
    gaps=st.lists(
        st.integers(min_value=1, max_value=24),
        min_size=2,
        max_size=12,
    ),
)
def test_integer_epoch_units_preserve_exact_identity_with_long_gaps(
    unit: tuple[str, int, int, int],
    gaps: list[int],
) -> None:
    name, base, cadence, ns_factor = unit
    offsets = [0]
    for gap in gaps:
        offsets.append(offsets[-1] + gap)
    timestamps = [base + offset * cadence for offset in offsets]
    frame = _make_symbol_df(timestamps)

    dataset = canonicalize_ohlcv(
        frame,
        symbol="EURUSD",
        timeframe="H1",
        numeric_time_unit=name,
    )

    expected_ns = [timestamp * ns_factor for timestamp in timestamps]
    assert dataset.frame["time"].astype("int64").tolist() == expected_ns
    assert dataset.identity.start_time_ns == expected_ns[0]
    assert dataset.identity.end_time_ns == expected_ns[-1]
    assert dataset.gap_count == sum(gap > 1 for gap in gaps)


@st.composite
def _pure_near_epoch_multiple_case(
    draw: st.DrawFn,
) -> tuple[str, list[int], list[int], list[int], int]:
    unit, unit_ns = draw(
        st.sampled_from(
            [("s", 1_000_000_000), ("ms", 1_000_000), ("us", 1_000), ("ns", 1)]
        )
    )
    length = draw(st.integers(min_value=3, max_value=8))
    gaps = draw(
        st.lists(
            st.integers(min_value=2, max_value=32),
            min_size=length - 1,
            max_size=length - 1,
        )
    )
    offsets = [0]
    for gap in gaps:
        offsets.append(offsets[-1] + gap)
    base_mode = draw(
        st.sampled_from(("negative", "epoch_zero", "cross_epoch", "positive"))
    )
    if base_mode == "negative":
        base_ns = -(offsets[-1] + 5) * 3_600_000_000_000
    elif base_mode == "epoch_zero":
        base_ns = 0
    elif base_mode == "cross_epoch":
        base_ns = -offsets[length // 2] * 3_600_000_000_000
    else:
        base_ns = 1_700_000_000_000_000_000
    semantic_ns = [
        base_ns + offset * 3_600_000_000_000 for offset in offsets
    ]
    encoded = [timestamp_ns // unit_ns for timestamp_ns in semantic_ns]
    order = list(range(length))
    return unit, encoded, order, semantic_ns, len(gaps)


@settings(max_examples=200)
@given(case=_pure_near_epoch_multiple_case())
def test_pure_epoch_units_preserve_identity_when_all_gaps_are_multiples(
    case: tuple[str, list[int], list[int], list[int], int],
) -> None:
    unit, encoded, order, semantic_ns, gap_count = case
    frame = _make_symbol_df(encoded).iloc[order]

    dataset = canonicalize_ohlcv(
        frame,
        symbol="EURUSD",
        timeframe="H1",
        numeric_time_unit=unit,
    )

    assert dataset.frame["time"].astype("int64").tolist() == semantic_ns
    assert dataset.identity.start_time_ns == semantic_ns[0]
    assert dataset.identity.end_time_ns == semantic_ns[-1]
    assert dataset.gap_count == gap_count


@settings(max_examples=100)
@given(
    base_seconds=st.integers(min_value=946_684_800, max_value=2_000_000_000),
    gap_multiplier=st.integers(min_value=2, max_value=48),
    units=st.sampled_from(
        [
            ("s", "ms", "ms"),
            ("s", "ms", "us"),
            ("ms", "s", "ms"),
            ("ms", "ms", "s"),
            ("ms", "us", "us"),
        ]
    ),
)
def test_mixed_epoch_units_are_rejected_at_cadence_multiples(
    base_seconds: int,
    gap_multiplier: int,
    units: tuple[str, str, str],
) -> None:
    unit_ns = {"s": 1_000_000_000, "ms": 1_000_000, "us": 1_000}
    nominal_ns = 3_600_000_000_000
    base_ns = base_seconds * 1_000_000_000
    semantic_ns = [
        base_ns + index * gap_multiplier * nominal_ns for index in range(3)
    ]
    encoded = [
        timestamp_ns // unit_ns[unit]
        for timestamp_ns, unit in zip(semantic_ns, units)
    ]

    with pytest.raises(DataValidationError, match=r"provenance|unit.*required"):
        canonicalize_ohlcv(
            _make_symbol_df(encoded),
            symbol="EURUSD",
            timeframe="H1",
        )


_PROPERTY_TIME_UNIT_NS = {
    "s": 1_000_000_000,
    "ms": 1_000_000,
    "us": 1_000,
    "ns": 1,
}
_PROPERTY_UNIT_PAIRS = [
    (left, right)
    for left in _PROPERTY_TIME_UNIT_NS
    for right in _PROPERTY_TIME_UNIT_NS
    if left != right
]


@st.composite
def _mixed_lattice_insertion_case(
    draw: st.DrawFn,
) -> tuple[list[int], list[int]]:
    length = draw(st.integers(min_value=4, max_value=8))
    multiplier = draw(st.integers(min_value=1, max_value=16))
    minority_unit, majority_unit = draw(st.sampled_from(_PROPERTY_UNIT_PAIRS))
    third_unit = draw(
        st.one_of(
            st.none(),
            st.sampled_from(
                [
                    unit
                    for unit in _PROPERTY_TIME_UNIT_NS
                    if unit not in (minority_unit, majority_unit)
                ]
            ),
        )
    )
    base_seconds = draw(
        st.integers(min_value=946_684_800, max_value=2_000_000_000)
    )
    offsets = [0, 2 * multiplier, 3 * multiplier, 4 * multiplier]
    offsets.extend((5 + index) * multiplier for index in range(length - 4))
    units = [minority_unit, *([majority_unit] * (length - 1))]
    if third_unit is not None:
        units[draw(st.integers(min_value=1, max_value=length - 1))] = third_unit

    nominal_ns = 3_600_000_000_000
    base_ns = base_seconds * 1_000_000_000
    encoded = [
        (base_ns + offset * nominal_ns) // _PROPERTY_TIME_UNIT_NS[unit]
        for offset, unit in zip(offsets, units)
    ]
    order_mode = draw(st.sampled_from(("forward", "reverse", "rotate")))
    order = list(range(length))
    if order_mode == "reverse":
        order.reverse()
    elif order_mode == "rotate":
        pivot = draw(st.integers(min_value=1, max_value=length - 1))
        order = order[pivot:] + order[:pivot]
    return encoded, order


@settings(max_examples=200)
@given(case=_mixed_lattice_insertion_case())
def test_mixed_epoch_unit_lattice_insertions_are_rejected_for_arbitrary_lengths(
    case: tuple[list[int], list[int]],
) -> None:
    encoded, order = case
    frame = _make_symbol_df(encoded).iloc[order]

    with pytest.raises(DataValidationError, match=r"provenance|unit.*required"):
        canonicalize_ohlcv(frame, symbol="EURUSD", timeframe="H1")
