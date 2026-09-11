"""Performance statistics, which every conclusion in this project is stated in.

Mutation testing put this module at 13% — six of forty-five deliberate defects
detected. Only the Sortino correction had a test (in test_audit_fixes.py);
Sharpe, the drawdown, the annualisation, the profit factor and the
session-collapse rule that the module's own docstring calls its first rule were
all unconstrained.

Every expected value here is computed by hand or from the definition, never
from the function under test.
"""
import numpy as np
import pandas as pd
import pytest

from bipbip.core.metrics import (TRADING_DAYS, compute, daily_equity,
                                 max_drawdown)
from bipbip.core.types import BacktestResult, Trade


def _curve(values, start="2026-01-05", freq="D"):
    idx = pd.DatetimeIndex(pd.date_range(start, periods=len(values), freq=freq))
    return pd.Series([float(v) for v in values], index=idx)


def _trade(pnl, fees=0.0, minutes=30):
    t0 = pd.Timestamp("2026-01-05 10:00")
    return Trade(symbol="AAA", entry_time=t0, entry_price=100.0,
                 exit_time=t0 + pd.Timedelta(minutes=minutes),
                 exit_price=100.0 + pnl, shares=1.0, pnl=float(pnl),
                 fees=float(fees))


def _result(curve, trades=(), start=100.0, blocked=()):
    return BacktestResult(symbol="AAA", equity_curve=curve,
                          trades=list(trades), blocked=list(blocked),
                          starting_equity=start)


# --------------------------------------------------------------------------
# daily_equity — the module's stated first rule
# --------------------------------------------------------------------------

def test_a_bar_level_curve_collapses_to_one_observation_per_session():
    """The docstring's first rule: 390 marks a day inflates the observation
    count and makes a Sharpe look far more certain than it is. The value kept
    must be the session's LAST, not its first or its mean."""
    idx = pd.DatetimeIndex([
        pd.Timestamp("2026-01-05 09:30"), pd.Timestamp("2026-01-05 12:00"),
        pd.Timestamp("2026-01-05 16:00"),
        pd.Timestamp("2026-01-06 09:30"), pd.Timestamp("2026-01-06 16:00")])
    curve = pd.Series([100.0, 150.0, 110.0, 111.0, 120.0], index=idx)

    d = daily_equity(curve)

    assert len(d) == 2
    assert list(d.to_numpy()) == [110.0, 120.0]      # session ENDS


def test_an_empty_curve_collapses_to_an_empty_one_rather_than_raising():
    assert daily_equity(pd.Series(dtype="float64")).empty


# --------------------------------------------------------------------------
# max_drawdown
# --------------------------------------------------------------------------

def test_the_drawdown_is_measured_from_the_running_peak():
    """100 -> 120 -> 90 is -25% from the peak of 120, not -10% from the start.
    Measuring from the start understates every drawdown in a rising curve."""
    c = _curve([100, 120, 90, 130])
    depth, peak, trough = max_drawdown(c)
    assert depth == pytest.approx(-0.25)
    assert peak == c.index[1]
    assert trough == c.index[2]


def test_the_deepest_drawdown_wins_not_the_most_recent():
    c = _curve([100, 200, 100, 250, 200])       # -50% then -20%
    depth, _, trough = max_drawdown(c)
    assert depth == pytest.approx(-0.50)
    assert trough == c.index[2]


def test_a_curve_that_only_rises_has_no_drawdown():
    depth, peak, _ = max_drawdown(_curve([100, 110, 120]))
    assert depth == pytest.approx(0.0)
    assert peak is None


def test_an_empty_curve_has_no_drawdown_rather_than_raising():
    assert max_drawdown(pd.Series(dtype="float64")) == (0.0, None, None)


# --------------------------------------------------------------------------
# compute — the headline ratios
# --------------------------------------------------------------------------

def test_total_return_is_measured_against_starting_equity():
    m = compute(_result(_curve([100, 110, 121]), start=100.0))
    assert m["final_equity"] == pytest.approx(121.0)
    assert m["total_return_pct"] == pytest.approx(21.0)


def test_the_annualised_return_compounds_rather_than_scaling():
    """(final/start)^(252/sessions) - 1. Multiplying the total return by
    252/sessions instead would overstate a short winning sample enormously and
    is the arithmetic the docstring warns about."""
    c = _curve(np.linspace(100.0, 200.0, 127))       # ~half a year, doubled
    m = compute(_result(c, start=100.0))
    years = m["sessions"] / TRADING_DAYS
    expected = ((200.0 / 100.0) ** (1.0 / years) - 1.0) * 100.0
    assert m["annualised_return_pct"] == pytest.approx(expected)
    assert m["annualised_return_pct"] > 100.0          # compounding, not 2x


def test_sharpe_is_mean_over_sd_annualised_by_the_root_of_252():
    rng = np.random.default_rng(4)
    rets = rng.normal(0.001, 0.01, 400)
    c = _curve(100.0 * np.cumprod(1.0 + np.r_[0.0, rets]))
    m = compute(_result(c, start=100.0))

    r = pd.Series(c.to_numpy()).pct_change().dropna()
    assert m["sharpe"] == pytest.approx(
        r.mean() / r.std() * np.sqrt(TRADING_DAYS), rel=1e-9)
    assert m["annualised_vol_pct"] == pytest.approx(
        r.std() * np.sqrt(TRADING_DAYS) * 100.0, rel=1e-9)


def test_a_flat_curve_has_no_sharpe_rather_than_an_infinite_one():
    """Zero volatility must not divide. A curve that never moves is not
    infinitely good."""
    m = compute(_result(_curve([100.0] * 30), start=100.0))
    assert np.isnan(m["sharpe"])
    assert np.isnan(m["sortino"])


def test_calmar_is_the_annual_return_over_the_drawdown_depth():
    c = _curve([100, 120, 90, 130])
    m = compute(_result(c, start=100.0))
    assert m["calmar"] == pytest.approx(
        m["annualised_return_pct"] / abs(m["max_drawdown_pct"]))
    assert m["max_drawdown_pct"] == pytest.approx(-25.0)


# --------------------------------------------------------------------------
# compute — trade statistics
# --------------------------------------------------------------------------

def test_win_rate_expectancy_and_profit_factor_come_from_the_trades():
    trades = [_trade(10), _trade(20), _trade(-5), _trade(-25)]
    m = compute(_result(_curve([100, 100]), trades, start=100.0))

    assert m["trades"] == 4
    assert m["win_rate_pct"] == pytest.approx(50.0)
    assert m["avg_win"] == pytest.approx(15.0)
    assert m["avg_loss"] == pytest.approx(-15.0)
    assert m["expectancy"] == pytest.approx(0.0)
    assert m["profit_factor"] == pytest.approx(30.0 / 30.0)


def test_a_breakeven_trade_counts_as_a_loss_not_a_win():
    """`pnls > 0` for wins: exactly zero is not a win. Counting it as one
    inflates every win rate on a tape with flat exits."""
    m = compute(_result(_curve([100, 100]), [_trade(0.0), _trade(10.0)],
                        start=100.0))
    assert m["win_rate_pct"] == pytest.approx(50.0)


def test_profit_factor_is_infinite_only_when_nothing_was_lost():
    only_wins = compute(_result(_curve([100, 100]), [_trade(5), _trade(5)],
                                start=100.0))
    assert np.isinf(only_wins["profit_factor"])

    with_loss = compute(_result(_curve([100, 100]), [_trade(5), _trade(-5)],
                                start=100.0))
    assert with_loss["profit_factor"] == pytest.approx(1.0)


def test_gross_pnl_adds_the_fees_back_rather_than_subtracting_them_twice():
    """`pnl` is already net of fees, so gross = net + fees. Subtracting
    instead double-charges, which makes a losing system look worse and a
    cost-sensitivity study meaningless."""
    trades = [_trade(10.0, fees=1.0), _trade(-4.0, fees=2.0)]
    m = compute(_result(_curve([100, 100]), trades, start=100.0))
    assert m["total_fees"] == pytest.approx(3.0)
    assert m["expectancy"] == pytest.approx(3.0)          # net mean
    assert m["gross_pnl"] == pytest.approx(6.0 + 3.0)     # net sum + fees


def test_blocked_entries_are_counted_separately_from_trades():
    m = compute(_result(_curve([100, 100]), [_trade(1)], blocked=[1, 2, 3],
                        start=100.0))
    assert m["trades"] == 1
    assert m["blocked_entries"] == 3


def test_no_data_is_reported_as_no_data_rather_than_as_zero_performance():
    """An empty curve returning a filled-in metric dict would put zeros into a
    comparison table where the honest answer is that nothing ran."""
    m = compute(_result(pd.Series(dtype="float64"), start=100.0))
    assert m["sessions"] == 0 and m["trades"] == 0
    assert "sharpe" not in m

    m2 = compute(_result(_curve([100, 110]), start=0.0))
    assert m2["sessions"] == 0
