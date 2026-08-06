from __future__ import annotations

import json
from pathlib import Path

import pytest

from pa_core import AtrState, BarFormulaEngine, BarSeries, EmaState, PABar
from tests.support.pa_fixtures import BASE_TS, STEP_SECONDS, make_series


GOLDEN = json.loads(
    (Path(__file__).parents[1] / "fixtures" / "pa_formula_golden.json").read_text(encoding="utf-8")
)


def test_geometry_golden_cases() -> None:
    assert GOLDEN["source_commit"] == "d92ecd827fe671a589b7fdfdbba41e5e98081d87"
    assert len(GOLDEN["fixture_groups"]) == 16
    for case in GOLDEN["geometry"]:
        snapshot = BarFormulaEngine(make_series([tuple(case["row"])])).evaluate(0)
        for formula_id, expected in case["expected"].items():
            result = snapshot.get(formula_id)
            assert result.valid is (expected is not None)
            if isinstance(expected, float):
                assert result.value == pytest.approx(expected)
            else:
                assert result.value == expected


def test_atr14_ema20_full_and_incremental_are_identical() -> None:
    closes = [float(value) for value in GOLDEN["indicator_ramp"]["closes"]]
    rows = [(close, close + 1.0, close - 1.0, close) for close in closes]
    engine = BarFormulaEngine(make_series(rows))
    expected = GOLDEN["indicator_ramp"]["expected"]

    assert engine.atr14[12] is None
    assert engine.atr14[13] == pytest.approx(expected["atr14_index_13"])
    assert engine.ema20[18] is None
    assert engine.ema20[19] == pytest.approx(expected["ema20_index_19"])
    assert engine.ema20[20] == pytest.approx(expected["ema20_index_20"])

    atr_state = AtrState.initial(period=14)
    ema_state = EmaState.initial(period=20)
    incremental_atr: list[float | None] = []
    incremental_ema: list[float | None] = []
    for close in closes:
        atr_state = atr_state.update(close + 1.0, close - 1.0, close)
        ema_state = ema_state.update(close)
        incremental_atr.append(atr_state.value)
        incremental_ema.append(ema_state.value)

    assert incremental_atr == pytest.approx(engine.atr14)
    assert incremental_ema == pytest.approx(engine.ema20)


def test_inside_ii_iii_and_ioi_use_ascending_time() -> None:
    iii = make_series(
        [
            (5.0, 10.0, 0.0, 6.0),
            (5.0, 9.0, 1.0, 6.0),
            (5.0, 8.0, 2.0, 6.0),
            (5.0, 7.0, 3.0, 6.0),
        ]
    )
    iii_snapshot = BarFormulaEngine(iii).evaluate(3)
    assert iii_snapshot.value("bar.type") == "inside"
    assert iii_snapshot.value("bar.inside_sequence") == "iii"

    ioi = make_series(
        [
            (5.0, 10.0, 0.0, 6.0),
            (5.0, 9.0, 1.0, 6.0),
            (5.0, 11.0, -1.0, 6.0),
            (5.0, 10.0, 0.0, 6.0),
        ]
    )
    assert BarFormulaEngine(ioi).evaluate(3).value("bar.ioi") is True


def test_follow_through_uses_at_most_two_later_closed_bars() -> None:
    series = make_series(
        [
            (10.0, 11.0, 9.0, 10.8),
            (10.7, 10.9, 10.2, 10.7),
            (10.8, 11.3, 10.6, 11.1),
            (11.0, 20.0, 10.0, 19.0),
        ]
    )
    engine = BarFormulaEngine(series)

    pending = engine.evaluate(0, as_of_index=0).get("bar.follow_through_1_2")
    confirmed = engine.evaluate(0, as_of_index=2).get("bar.follow_through_1_2")
    after_irrelevant_future = engine.evaluate(0, as_of_index=3).get("bar.follow_through_1_2")

    assert pending.value == "pending"
    assert confirmed.value == "yes"
    assert confirmed.confirmed_at == series.bars[2].timestamp
    assert after_irrelevant_future == confirmed


def test_micro_double_gap_and_twenty_gap_bars_boundaries() -> None:
    ramp_rows = []
    for index in range(41):
        close = 100.0 + index
        ramp_rows.append((close - 0.05, close + 0.2, close - 0.1, close))
    engine = BarFormulaEngine(make_series(ramp_rows))

    assert engine.evaluate(37).value("bar.ema_gap_run") == 19
    assert engine.evaluate(37).value("bar.twenty_gap_bars") is False
    assert engine.evaluate(38).value("bar.ema_gap_run") == 20
    assert engine.evaluate(38).value("bar.twenty_gap_bars") is True
    assert engine.evaluate(39).value("bar.ema_gap_run") == 21

    gapped = make_series([(10.0, 11.0, 9.0, 10.0), (12.0, 13.0, 12.0, 12.5)])
    assert BarFormulaEngine(gapped).evaluate(1).value("bar.interbar_gap") == "up"

    repeated_low_rows = [(10.0, 11.0, 9.0, 10.5)] * 14 + [
        (10.5, 11.5, 9.5, 11.0),
        (10.8, 11.7, 9.51, 11.2),
    ]
    assert (
        BarFormulaEngine(make_series(repeated_low_rows)).evaluate(15).value("bar.micro_double")
        == "MDB"
    )


def test_interbar_gap_requires_more_than_one_tick() -> None:
    no_gap = make_series([(10.0, 11.0, 9.0, 10.0), (11.5, 12.0, 11.01, 11.5)])
    gap = make_series([(10.0, 11.0, 9.0, 10.0), (11.5, 12.0, 11.02, 11.5)])
    assert BarFormulaEngine(no_gap).evaluate(1).value("bar.interbar_gap") == "none"
    assert BarFormulaEngine(gap).evaluate(1).value("bar.interbar_gap") == "up"


def test_input_contract_rejects_time_forming_and_segment_mixing() -> None:
    valid = PABar(BASE_TS, 10.0, 11.0, 9.0, 10.5, segment_id="a")
    earlier = PABar(BASE_TS - STEP_SECONDS, 10.0, 11.0, 9.0, 10.5, segment_id="a")
    forming = PABar(
        BASE_TS + STEP_SECONDS,
        10.0,
        11.0,
        9.0,
        10.5,
        closed=False,
        segment_id="a",
    )
    other_segment = PABar(
        BASE_TS + STEP_SECONDS,
        10.0,
        11.0,
        9.0,
        10.5,
        segment_id="b",
    )

    with pytest.raises(ValueError, match="strictly increasing"):
        BarSeries("XAUUSD", "M15", (valid, earlier), 0.01)
    with pytest.raises(ValueError, match="closed bars"):
        BarSeries("XAUUSD", "M15", (valid, forming), 0.01)
    with pytest.raises(ValueError, match="one data segment"):
        BarSeries("XAUUSD", "M15", (valid, other_segment), 0.01)


def test_invalid_ohlc_returns_invalid_formula_results() -> None:
    invalid = make_series([(10.0, 9.0, 8.0, 10.5)])
    snapshot = BarFormulaEngine(invalid).evaluate(0)
    for formula_id in ("bar.range", "bar.type", "indicator.atr14", "bar.follow_through_1_2"):
        result = snapshot.get(formula_id)
        assert result.valid is False
        assert result.value is None
        assert result.invalid_reason == "invalid_ohlc"
