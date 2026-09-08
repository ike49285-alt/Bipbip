"""Leveraged ETF simulation."""
import numpy as np
import pandas as pd
import pytest

from bipbip.data import BarStore
from bipbip.data.leveraged import decay_report, simulate_leveraged


def _flat_then_choppy(n=500):
    """A market that ends where it started but moves every day.

    The pure test of volatility decay: the index is unchanged, so naive
    leverage would also be unchanged, while a daily-reset fund must LOSE.
    """
    alt = np.where(np.arange(n) % 2 == 0, 1.03, 1 / 1.03)
    close = 100 * np.cumprod(alt)
    idx = pd.DatetimeIndex(pd.bdate_range("2010-01-04", periods=n))
    return pd.DataFrame({"open": close, "high": close * 1.005, "low": close * 0.995,
                         "close": close, "volume": 1e7}, index=idx)


def test_daily_reset_loses_money_in_a_flat_choppy_market():
    """Volatility decay, which is what makes a leveraged fund different from
    leverage and what every naive TQQQ backtest omits."""
    bars = _flat_then_choppy()
    index_multiple = float(bars["close"].iloc[-1] / bars["close"].iloc[0])
    sim = simulate_leveraged(bars, 3.0)
    lev_multiple = float(sim["close"].iloc[-1] / sim["close"].iloc[0])

    assert 0.9 < index_multiple < 1.1, "index should end roughly flat"
    assert lev_multiple < 0.5, f"3x should decay badly in chop, got {lev_multiple:.2f}"


def test_leverage_amplifies_a_single_day():
    bars = _flat_then_choppy(50)
    sim = simulate_leveraged(bars, 3.0)
    idx_r = bars["close"].pct_change().dropna()
    lev_r = sim["close"].pct_change().dropna()
    ratio = (lev_r / idx_r).median()
    assert 2.8 < ratio < 3.0, f"daily ratio should be near 3x, got {ratio:.2f}"


def test_fund_cannot_lose_more_than_everything_in_a_day():
    """A 3x fund facing a 40% index drop would mathematically go negative."""
    idx = pd.DatetimeIndex(pd.bdate_range("2020-03-02", periods=5))
    close = np.array([100.0, 95.0, 55.0, 60.0, 62.0])
    bars = pd.DataFrame({"open": close, "high": close, "low": close,
                         "close": close, "volume": 1e7}, index=idx)
    sim = simulate_leveraged(bars, 3.0)
    assert (sim["close"] > 0).all()


def test_costs_make_leverage_lag_a_flat_index():
    """Expenses and financing are charged whether or not the market moves."""
    idx = pd.DatetimeIndex(pd.bdate_range("2015-01-05", periods=400))
    close = np.full(400, 100.0)
    bars = pd.DataFrame({"open": close, "high": close, "low": close,
                         "close": close, "volume": 1e7}, index=idx)
    sim = simulate_leveraged(bars, 3.0)
    assert float(sim["close"].iloc[-1]) < float(sim["close"].iloc[0])


def test_simulation_tracks_the_real_fund_where_they_overlap():
    """The validation that makes the simulated history usable: it must match
    the real fund over the years the real fund exists."""
    store = BarStore("data/bars")
    qqq, tqqq = store.load("QQQ", "1d"), store.load("TQQQ", "1d")
    if qqq.empty or tqqq.empty:
        pytest.skip("universe not fetched")

    sim = simulate_leveraged(qqq, 3.0)
    both = sim.join(tqqq["close"].rename("real"), how="inner").dropna()
    if len(both) < 500:
        pytest.skip("insufficient overlap")
    corr = both["close"].pct_change().corr(both["real"].pct_change())
    assert corr > 0.98, f"simulation should track the real fund, correlation {corr:.3f}"


def test_decay_report_shows_the_gap_against_naive_leverage():
    store = BarStore("data/bars")
    qqq = store.load("QQQ", "1d")
    if qqq.empty:
        pytest.skip("QQQ not fetched")
    d = decay_report(qqq, 3.0)
    # The whole point: real leverage is nowhere near index^3.
    assert d["levered_multiple"] < d["naive_multiple"] / 100
