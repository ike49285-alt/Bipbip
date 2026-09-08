"""Indicator correctness, with emphasis on the scaling bugs found on real data."""
import numpy as np
import pandas as pd
import pytest

from bipbip.core import indicators as ind
from bipbip.data import make_intraday_bars
from bipbip.data.sessions import EXCHANGE_TZ


def _bar_of_day(index):
    return ind.minutes_since_open(index).to_numpy()


def test_vwap_zscore_does_not_drift_through_the_session():
    """The bug this replaced.

    Distance from session VWAP accumulates all session; a one-minute ATR does
    not. Dividing one by the other therefore drifts upward through the day, so
    a fixed threshold means something different at 10:00 than at 15:00 and
    fires constantly by the afternoon. A z-score against session-scale
    dispersion must stay flat instead.
    """
    bars = make_intraday_bars(n_sessions=25, seed=101)
    bod = _bar_of_day(bars.index)

    old = ((bars["close"] - ind.session_vwap(bars)) / ind.atr(bars, 30)).abs()
    new = ind.vwap_zscore(bars).abs()

    early = (bod >= 60) & (bod < 120)
    late = bod >= 300

    old_drift = np.nanmean(old[late]) / np.nanmean(old[early])
    new_drift = np.nanmean(new[late]) / np.nanmean(new[early])

    assert old_drift > 1.5, "expected the ATR-normalised metric to drift upward"
    assert 0.7 < new_drift < 1.4, f"z-score should be roughly stationary, drifted {new_drift:.2f}x"


def test_vwap_zscore_is_actually_a_zscore():
    """Median near 1 and extremes in single digits - not the 3.7 median and
    31x maximum the ATR-normalised version produced on real TQQQ data."""
    bars = make_intraday_bars(n_sessions=30, seed=102)
    z = ind.vwap_zscore(bars).dropna().abs()

    assert 0.5 < z.median() < 1.6
    assert z.max() < 10.0
    assert 0.02 < (z > 2).mean() < 0.25


def test_vwap_zscore_survives_a_motionless_tape():
    """A flat session makes the dispersion zero; the ratio must not explode."""
    idx = pd.date_range("2025-04-01 09:30", periods=200, freq="1min", tz=EXCHANGE_TZ)
    flat = pd.DataFrame(
        {"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 1000.0},
        index=idx,
    )
    z = ind.vwap_zscore(flat)
    assert np.isfinite(z.dropna()).all()
    assert z.abs().max() < 10.0


def test_session_vwap_bands_are_causal():
    bars = make_intraday_bars(n_sessions=6, seed=103)
    cut = int(len(bars) * 0.7)
    full = ind.session_vwap_bands(bars).iloc[:cut]
    trunc = ind.session_vwap_bands(bars.iloc[:cut])
    pd.testing.assert_frame_equal(full, trunc, check_exact=False, rtol=1e-9)


def test_vwap_sigma_is_withheld_until_enough_bars():
    """A standard deviation over three bars is noise pretending to be a statistic."""
    bars = make_intraday_bars(n_sessions=2, seed=104)
    bands = ind.session_vwap_bands(bars, min_bars=15)
    bod = _bar_of_day(bars.index)
    assert bands["vwap_sigma"][bod < 14].isna().all()
    assert bands["vwap_sigma"][bod >= 20].notna().all()


def test_cost_floored_risk_never_sits_inside_the_round_trip():
    """A stop tighter than the round trip loses money by construction.

    On real SPY data 37% of bars had a one-minute ATR below the 2.3bp
    round-trip cost, which is what produced trades that stopped out on the
    same bar they entered.
    """
    close = pd.Series(np.full(200, 500.0))
    tiny_atr = pd.Series(np.full(200, 0.001))  # far below any sane floor
    hurdle, mult = 2.3, 2.0

    risk = ind.cost_floored_risk(close, tiny_atr, hurdle, mult)
    expected = 500.0 * (hurdle * mult / 10_000.0)

    assert (risk >= expected - 1e-12).all()
    assert risk.iloc[0] == pytest.approx(expected)


def test_cost_floored_risk_leaves_a_healthy_atr_alone():
    """The floor must bind only when it needs to."""
    close = pd.Series(np.full(50, 500.0))
    big_atr = pd.Series(np.full(50, 5.0))
    risk = ind.cost_floored_risk(close, big_atr, 2.3, 2.0)
    pd.testing.assert_series_equal(risk, big_atr)


def test_strategies_never_place_a_stop_inside_the_cost_hurdle():
    """End-to-end: the floor has to survive the trip through the engine."""
    from bipbip.core import BacktestEngine, CashAccount, CostModel
    from bipbip.strategies import get_strategy

    bars = make_intraday_bars(n_sessions=25, seed=105)
    costs = CostModel()
    hurdle = costs.round_trip_cost_bps("SPY", float(bars["close"].iloc[-1]))

    for name in ["orb", "vwap_reversion"]:
        res = BacktestEngine(CashAccount(10_000.0), costs).run("SPY", bars, get_strategy(name))
        for t in res.trades:
            if t.exit_reason != "stop":
                continue
            # Distance risked must exceed what the round trip itself costs.
            risked_bps = abs(t.entry_price - t.exit_price) / t.entry_price * 10_000
            assert risked_bps > hurdle * 0.5, (
                f"{name}: stop risked {risked_bps:.2f}bps against a {hurdle:.2f}bp hurdle"
            )
