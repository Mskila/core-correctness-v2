from __future__ import annotations

from pa_core import BarFormulaEngine, StructureFormulaEngine
from tests.support.pa_fixtures import make_series


def pseudo_market_rows(count: int) -> list[tuple[float, float, float, float]]:
    rows: list[tuple[float, float, float, float]] = []
    close = 100.0
    for index in range(count):
        drift = ((index * 17) % 11 - 5) * 0.08 + (0.12 if index % 9 < 6 else -0.18)
        open_ = close
        close = open_ + drift
        high = max(open_, close) + 0.35 + (index % 3) * 0.04
        low = min(open_, close) - 0.30 - (index % 4) * 0.03
        rows.append((open_, high, low, close))
    return rows


def test_every_historical_snapshot_is_prefix_invariant() -> None:
    rows = pseudo_market_rows(90)
    full_series = make_series(rows)
    full_bar_engine = BarFormulaEngine(full_series)
    full_structure_engine = StructureFormulaEngine(full_series)

    for cut in (19, 27, 39, 55, 71):
        prefix_series = make_series(rows[: cut + 1])
        prefix_bar = BarFormulaEngine(prefix_series).evaluate(cut)
        full_bar = full_bar_engine.evaluate(cut, as_of_index=cut)
        assert prefix_bar == full_bar

        prefix_structure = StructureFormulaEngine(prefix_series).snapshot()
        full_structure = full_structure_engine.snapshot(as_of_index=cut)
        assert prefix_structure == full_structure
