"""The most important tests in the project.

A backtest that peeks at future bars produces beautiful, worthless results.
These tests attack that directly: they mutate the future and assert the past
does not move.
"""
import numpy as np
import pandas as pd
import pytest

from bipbip.core import BacktestEngine, CashAccount, CostModel
from bipbip.core import indicators as ind
from bipbip.data import make_intraday_bars
from bipbip.strategies import get_strategy

STRATEGIES = ["buy_hold", "orb", "vwap_reversion"]


def _fresh_engine():
    return BacktestEngine(CashAccount(starting_equity=10_000.0), CostModel())


@pytest.mark.parametrize("name", STRATEGIES)
def test_future_bars_cannot_change_the_past(name):
    """Corrupt every bar after a cut point; results before it must be identical.

    If any strategy or indicator reads ahead, the equity curve before the cut
    will shift and this fails. It is the single check that keeps the whole
    project honest.
    """
    bars = make_intraday_bars(n_sessions=10, seed=5)
    cut = int(len(bars) * 0.6)

    tampered = bars.copy()
    # Violent, structural corruption - not noise that might wash out.
    tampered.iloc[cut:, tampered.columns.get_loc("open")] *= 1.75
    tampered.iloc[cut:, tampered.columns.get_loc("high")] *= 1.90
    tampered.iloc[cut:, tampered.columns.get_loc("low")] *= 1.60
    tampered.iloc[cut:, tampered.columns.get_loc("close")] *= 1.75
    tampered.iloc[cut:, tampered.columns.get_loc("volume")] *= 40.0

    base = _fresh_engine().run("SPY", bars, get_strategy(name))
    alt = _fresh_engine().run("SPY", tampered, get_strategy(name))

    cut_ts = bars.index[cut]
    a = base.equity_curve[base.equity_curve.index < cut_ts]
    b = alt.equity_curve[alt.equity_curve.index < cut_ts]

    pd.testing.assert_series_equal(a, b, check_exact=False, rtol=1e-9)


@pytest.mark.parametrize(
    "fn",
    [
        lambda df: ind.session_vwap(df),
        lambda df: ind.atr(df, 14),
        lambda df: ind.rsi(df["close"], 14),
        lambda df: ind.relative_volume(df, 20),
        lambda df: ind.opening_range(df, 30)["or_high"],
        lambda df: ind.opening_range(df, 30)["or_low"],
    ],
)
def test_indicators_are_causal(fn):
    """An indicator computed on truncated history must match the full-history
    value at every overlapping timestamp."""
    bars = make_intraday_bars(n_sessions=6, seed=9)
    cut = int(len(bars) * 0.7)

    full = fn(bars).iloc[:cut]
    truncated = fn(bars.iloc[:cut])

    pd.testing.assert_series_equal(full, truncated, check_exact=False, rtol=1e-10)


def test_strategy_never_receives_a_future_bar():
    """Assert directly on the slice handed to the strategy."""
    seen = []

    from bipbip.core.strategy import Strategy
    from bipbip.core.types import HOLD

    class Spy(Strategy):
        name = "spy"

        def on_bar(self, ctx):
            seen.append((ctx.bars.index[-1], ctx.bars.index[ctx.i], len(ctx.bars), ctx.i))
            return HOLD

    bars = make_intraday_bars(n_sessions=3, seed=1)
    _fresh_engine().run("SPY", bars, Spy())

    assert seen, "strategy was never called"
    for last_ts, current_ts, n, i in seen:
        # The last bar in the visible window IS the bar being decided on:
        # there is no bar i+1 in the slice for a strategy to read.
        assert last_ts == current_ts
        assert n == i + 1


def test_entry_fills_at_next_bar_open_not_current_close():
    """An order decided at bar i must fill at bar i+1's open, plus slippage."""
    from bipbip.core.strategy import Strategy
    from bipbip.core.types import HOLD, Intent

    class EnterOnBar5(Strategy):
        name = "enter_on_bar_5"

        def on_bar(self, ctx):
            if ctx.i == 5 and not ctx.in_position and ctx.can_open:
                return Intent(action="enter", reason="test")
            return HOLD

    bars = make_intraday_bars(n_sessions=1, seed=2)
    costs = CostModel()
    eng = BacktestEngine(CashAccount(starting_equity=10_000.0), costs)
    res = eng.run("SPY", bars, EnterOnBar5())

    assert len(res.trades) == 1
    trade = res.trades[0]
    expected = costs.fill_price("buy", float(bars["open"].iloc[6]), "SPY")

    assert trade.entry_time == bars.index[6]
    assert trade.entry_price == pytest.approx(expected)
    # And emphatically NOT the close of the bar the decision was made on.
    assert trade.entry_price != pytest.approx(float(bars["close"].iloc[5]))
