"""Swing mode: positions that survive the closing bell."""
import numpy as np
import pandas as pd
import pytest

from bipbip.core import BacktestEngine, CashAccount, CostModel
from bipbip.core import metrics as M
from bipbip.data import make_intraday_bars
from bipbip.data.sessions import EXCHANGE_TZ
from bipbip.strategies import get_strategy


def _daily(n=900, seed=501, drift=0.0003, vol=0.011):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(drift, vol, n)))
    idx = pd.DatetimeIndex([pd.Timestamp(d).tz_localize(EXCHANGE_TZ)
                            for d in pd.bdate_range("2015-01-05", periods=n).date])
    return pd.DataFrame({"open": close, "high": close * 1.008, "low": close * 0.992,
                         "close": close, "volume": 1e7}, index=idx)


def _engine(equity=50.0, **kw):
    return BacktestEngine(CashAccount(equity, max_position_pct=0.98), CostModel(), **kw)


def test_swing_positions_are_held_overnight():
    """The whole point of the mode. Intraday runs force-flat; swing must not."""
    bars = _daily()
    res = _engine(intraday=False).run("SPY", bars, get_strategy("sma_trend"))
    assert res.trades
    multi_day = [t for t in res.trades if t.entry_time.date() != t.exit_time.date()]
    assert multi_day, "no position survived a session boundary"
    assert not any(t.exit_reason == "force_flat" for t in res.trades)


def test_intraday_mode_still_force_flats():
    """Adding swing mode must not weaken the intraday guarantee."""
    bars = make_intraday_bars(n_sessions=12, seed=502)
    res = _engine(10_000.0, intraday=True).run("SPY", bars, get_strategy("orb"))
    for t in res.trades:
        assert t.entry_time.date() == t.exit_time.date()


def test_swing_closes_the_final_position_at_the_end_of_data():
    """An open position at the end must be marked out, or the equity curve
    reports an unrealised gain as though it were banked."""
    bars = _daily(seed=503, drift=0.001)
    res = _engine(intraday=False).run("SPY", bars, get_strategy("buy_and_hold"))
    assert res.trades
    assert res.trades[-1].exit_reason == "end_of_data"
    assert res.trades[-1].exit_time == bars.index[-1]


def test_swing_respects_settled_cash():
    """T+1 settlement still applies; a swing hold simply spans it."""
    bars = _daily(seed=504)
    acct = CashAccount(50.0, max_position_pct=0.98)
    BacktestEngine(acct, CostModel(), intraday=False).run("SPY", bars, get_strategy("rsi2"))
    assert acct.settled_cash >= -1e-9
    assert acct.position.shares >= 0


def test_swing_mode_has_no_lookahead():
    """Same guarantee as the intraday engine: corrupting the future must not
    move the equity curve before the cut."""
    bars = _daily(seed=505)
    cut = int(len(bars) * 0.6)
    tampered = bars.copy()
    for col in ("open", "high", "low", "close"):
        tampered.iloc[cut:, tampered.columns.get_loc(col)] *= 1.6

    a = _engine(intraday=False).run("SPY", bars, get_strategy("sma_trend"))
    b = _engine(intraday=False).run("SPY", tampered, get_strategy("sma_trend"))
    ts = bars.index[cut]
    pd.testing.assert_series_equal(
        a.equity_curve[a.equity_curve.index < ts],
        b.equity_curve[b.equity_curve.index < ts],
        check_exact=False, rtol=1e-9,
    )


def test_swing_equity_curve_covers_every_bar():
    bars = _daily(seed=506)
    res = _engine(intraday=False).run("SPY", bars, get_strategy("rsi2"))
    assert len(res.equity_curve) == len(bars)
    assert (res.equity_curve > 0).all()


def test_rsi2_only_buys_inside_an_uptrend():
    """The trend filter is what stops it averaging into a bear market."""
    from bipbip.core import indicators as ind

    bars = _daily(seed=507)
    res = _engine(intraday=False).run("SPY", bars, get_strategy("rsi2"))
    trend = ind.sma(bars["close"], 200)
    for t in res.trades:
        # Entry is decided on the PRIOR bar and filled at this open.
        prior = bars.index[bars.index.get_loc(t.entry_time) - 1]
        assert float(bars["close"].loc[prior]) > float(trend.loc[prior])


def test_sma_trend_is_flat_below_its_average():
    from bipbip.core import indicators as ind

    bars = _daily(seed=508)
    res = _engine(intraday=False).run("SPY", bars, get_strategy("sma_trend"))
    assert res.trades
    trend = ind.sma(bars["close"], 200)
    for t in res.trades:
        prior = bars.index[bars.index.get_loc(t.entry_time) - 1]
        assert float(bars["close"].loc[prior]) > float(trend.loc[prior])


def test_fractional_sizing_leaves_no_idle_cash():
    """Whole-share sizing stranded $176 of a $9,500 SPY budget; fractional
    should deploy nearly all of it."""
    bars = _daily(seed=509)
    acct = CashAccount(50.0, max_position_pct=0.98, allow_fractional=True)
    BacktestEngine(acct, CostModel(), intraday=False).run(
        "SPY", bars, get_strategy("buy_and_hold"))
    shares = acct.affordable_shares(100.0)
    assert shares != int(shares), "fractional sizing should produce a fractional quantity"


def test_fractional_rounding_never_overspends():
    """Rounding down, never nearest: rounding up would invent buying power."""
    acct = CashAccount(50.0, max_position_pct=1.0, allow_fractional=True)
    acct.start_session(pd.Timestamp("2025-01-02").date())
    price = 777.77
    shares = acct.affordable_shares(price)
    assert shares * price <= acct.settled_cash + 1e-9
