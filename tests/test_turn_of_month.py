"""Turn-of-month rotation, which had no tests at all.

It is a REGISTERED, selectable strategy — `cross_sectional.py` maps the name
"turn_of_month" to it — and it carries the strongest claim in the repository:
an effect that "has not decayed" across four decades, confirmed on eight
held-out assets at "roughly a one-in-256 coincidence". It appears nowhere in
CLAUDE.md or README, and nothing exercised it.

The effect is real in the weak sense that it reproduces. The claims around it
do not survive this repo's own standards - the measured correction lives in the
module's own docstring and in CLAUDE.md. These tests pin the MECHANICS: the
window's shape, that it is counted in sessions, and that the strategy selects
the leg it says it does.
"""
import numpy as np
import pandas as pd
import pytest

from bipbip.core.panel import build_panel
from bipbip.strategies.turn_of_month import (TurnOfMonthRotation,
                                             turn_of_month_mask)


def _sessions(start, periods):
    return pd.DatetimeIndex(pd.bdate_range(start, periods=periods))


def test_the_window_straddles_each_month_boundary():
    idx = _sessions("2026-01-01", 60)
    m = turn_of_month_mask(idx, last_n=4, first_m=4)
    for month, grp in pd.Series(m.to_numpy(), index=idx).groupby(idx.to_period("M")):
        n = len(grp)
        if n < 8:                       # short month: entirely inside
            continue
        assert grp.iloc[:4].all(), month          # first four sessions
        assert grp.iloc[-4:].all(), month         # last four sessions
        assert not grp.iloc[4:-4].any(), month    # nothing in between


def test_the_window_is_counted_in_sessions_not_calendar_days():
    """The docstring's own claim, and the reason the mask exists at all: a
    holiday must not shift the window off the boundary it straddles.

    A calendar-day implementation keys off day-of-month, so dropping a session
    changes which dates are marked. A session-counted one does not.
    """
    full = _sessions("2026-03-02", 44)
    holed = full.delete(10)             # remove one mid-month session
    a = turn_of_month_mask(full, 4, 4)
    b = turn_of_month_mask(holed, 4, 4)
    for month in holed.to_period("M").unique():
        sel = holed[holed.to_period("M") == month]
        if len(sel) < 8:
            continue
        # The last four SESSIONS of the month are marked in both, and they are
        # the same dates, even though one index is a session shorter.
        assert list(sel[-4:]) == list(full[full.to_period("M") == month][-4:])
        assert b.loc[sel[-4:]].all()
        assert a.loc[sel[-4:]].all()


@pytest.mark.parametrize("last_n,first_m", [(1, 1), (2, 5), (6, 3), (0, 4), (4, 0)])
def test_the_window_widths_are_respected_exactly(last_n, first_m):
    """Both ends are parameters and both are used. A mask that ignored one, or
    swapped them, still looks like a turn-of-month window at the default 4/4."""
    idx = _sessions("2026-01-01", 90)
    m = pd.Series(turn_of_month_mask(idx, last_n, first_m).to_numpy(), index=idx)
    for month, grp in m.groupby(idx.to_period("M")):
        if len(grp) < last_n + first_m + 2:
            continue
        assert int(grp.sum()) == last_n + first_m, (month, int(grp.sum()))


def test_a_month_shorter_than_the_window_is_entirely_inside_it():
    """Not a contrivance: a partial month at the start or end of an archive is
    exactly this case, and an implementation that computed `size - pos - 1`
    without care could mark nothing there."""
    idx = pd.DatetimeIndex(["2026-01-28", "2026-01-29", "2026-01-30"])
    assert turn_of_month_mask(idx, 4, 4).all()


class _Ctx:
    """Minimal PortfolioContext stand-in: the strategy reads only these two."""

    def __init__(self, in_window, tradeable):
        self._w = in_window
        self.tradeable = tradeable

    def ind(self, name):
        assert name == "in_window"
        return pd.Series([self._w])


def test_it_holds_the_risk_asset_in_the_window_and_parks_outside_it():
    s = TurnOfMonthRotation(risk_symbol="SPY", park_symbol="SHY")
    assert s.target_weights(_Ctx(True, ["SPY", "SHY"])) == {"SPY": 1.0}
    assert s.target_weights(_Ctx(False, ["SPY", "SHY"])) == {"SHY": 1.0}


def test_a_missing_leg_falls_back_to_the_other_rather_than_to_cash():
    """The documented behaviour, and it matters in early history where SHY does
    not exist yet: going to cash there would silently make the backtest a
    different strategy than the one described."""
    s = TurnOfMonthRotation(risk_symbol="SPY", park_symbol="SHY")
    assert s.target_weights(_Ctx(False, ["SPY"])) == {"SPY": 1.0}
    assert s.target_weights(_Ctx(True, ["SHY"])) == {"SHY": 1.0}
    assert s.target_weights(_Ctx(True, [])) == {}


def test_weights_are_fully_invested_in_exactly_one_symbol():
    s = TurnOfMonthRotation()
    for w in (True, False):
        out = s.target_weights(_Ctx(w, ["SPY", "SHY"]))
        assert len(out) == 1 and sum(out.values()) == pytest.approx(1.0)


def test_prepare_broadcasts_one_calendar_to_every_symbol():
    """The window is a property of the DATE, not of the symbol, so every column
    must carry the same flags. A per-symbol calendar would silently let one
    symbol trade on a different schedule."""
    idx = _sessions("2026-01-01", 40)
    px = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0,
                       "close": 100.0, "volume": 1e6}, index=idx)
    panel = build_panel({"SPY": px, "SHY": px.copy()})
    ind = TurnOfMonthRotation().prepare(panel)["in_window"]
    assert list(ind.columns) == list(panel.symbols)
    assert ind.nunique(axis=1).max() == 1
