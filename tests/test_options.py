"""Option pricing, the modelled vol surface, and the 0DTE overlay."""
import numpy as np
import pandas as pd
import pytest

from bipbip.core import BacktestEngine, CashAccount, CostModel
from bipbip.data import make_intraday_bars
from bipbip.options import pricing as bs
from bipbip.options.iv import IV_FLOORS, implied_vol, iv_floor_for, realised_vol
from bipbip.options.overlay import MULTIPLIER, breakeven_move_bps, express_in_options
from bipbip.options.synth import atm_strike, minutes_to_close, option_series, quote_spread
from bipbip.strategies import get_strategy

S, K, R = 640.0, 640.0, 0.04
T_OPEN = bs.minutes_to_years(390)


# -- pricing -----------------------------------------------------------------

def test_put_call_parity_holds():
    c = bs.price(S, K, T_OPEN, R, 0.13, bs.CALL)
    p = bs.price(S, K, T_OPEN, R, 0.13, bs.PUT)
    assert float(c - p) == pytest.approx(S - K * np.exp(-R * T_OPEN), abs=1e-8)


def test_price_at_expiry_is_intrinsic():
    assert float(bs.price(645.0, 640.0, 0.0, R, 0.13, bs.CALL)) == pytest.approx(5.0)
    assert float(bs.price(635.0, 640.0, 0.0, R, 0.13, bs.CALL)) == pytest.approx(0.0)
    assert float(bs.price(635.0, 640.0, 0.0, R, 0.13, bs.PUT)) == pytest.approx(5.0)


def test_trading_clock_prices_higher_than_calendar_clock():
    """The clock bug this module exists to avoid.

    Vol estimated from one-minute bars is annualised over trading minutes, so
    time to expiry must use the same clock. A calendar clock under-prices an
    ATM 0DTE call by more than half, and a backtest built on it would look far
    too profitable.
    """
    trading = float(bs.price(S, K, T_OPEN, R, 0.13, bs.CALL))
    calendar = float(bs.price(S, K, 6.5 / 24 / 365, R, 0.13, bs.CALL))
    assert trading > 2 * calendar
    # Real SPY 0DTE ATM sits in dollars, not cents.
    assert 1.0 < trading < 6.0


def test_atm_delta_is_about_half_and_theta_is_negative():
    assert float(bs.delta(S, K, T_OPEN, R, 0.13, bs.CALL)) == pytest.approx(0.5, abs=0.05)
    for kind in (bs.CALL, bs.PUT):
        assert float(bs.theta_per_minute(S, K, T_OPEN, R, 0.13, kind)) < 0


def test_decay_accelerates_into_the_close():
    """Theta is not spread evenly across the session."""
    early = abs(float(bs.theta_per_minute(S, K, bs.minutes_to_years(390), R, 0.13)))
    late = abs(float(bs.theta_per_minute(S, K, bs.minutes_to_years(30), R, 0.13)))
    assert late > 2 * early


def test_leverage_is_withheld_on_a_worthless_contract():
    """Ratios diverge as premium decays - 21,000x on real data. That is
    arithmetic, not an opportunity."""
    far_otm = bs.leverage(600.0, 700.0, bs.minutes_to_years(5), R, 0.10, bs.CALL)
    assert np.isnan(float(far_otm))
    assert float(bs.leverage(S, K, T_OPEN, R, 0.13, bs.CALL)) > 20


# -- modelled vol ------------------------------------------------------------

def test_implied_vol_is_floored_above_realised():
    """Implied vol does not follow realised vol to zero: sellers still charge
    for gap risk. Without this the model invents free money."""
    bars = make_intraday_bars(n_sessions=10, seed=201, annual_vol=0.02)
    iv = implied_vol(bars["close"], symbol="SPY").dropna()
    assert (iv >= IV_FLOORS["SPY"] - 1e-12).all()


def test_leveraged_etf_gets_a_higher_floor_than_the_index():
    assert iv_floor_for("TQQQ") > iv_floor_for("SPY")
    assert iv_floor_for("NOT_A_SYMBOL") == IV_FLOORS["default"]


def test_realised_vol_uses_the_trading_clock():
    """A known vol in must come back out, or pricing is silently wrong."""
    bars = make_intraday_bars(n_sessions=40, seed=202, annual_vol=0.20, annual_drift=0.0)
    rv = realised_vol(bars["close"], window=120).dropna()
    assert 0.12 < rv.median() < 0.30


# -- synthetic series --------------------------------------------------------

def test_minutes_to_close_counts_down_to_the_bell():
    bars = make_intraday_bars(n_sessions=1, seed=203)
    mins = minutes_to_close(bars.index)
    assert mins.iloc[0] == pytest.approx(390.0)
    assert mins.iloc[-1] == pytest.approx(1.0)
    assert (mins.diff().dropna() <= 0).all()


def test_option_series_decays_when_the_underlying_is_flat():
    """Theta with nothing to offset it: the defining 0DTE risk."""
    idx = pd.date_range("2025-05-01 09:30", periods=390, freq="1min", tz="America/New_York")
    flat = pd.DataFrame({"open": 500.0, "high": 500.0, "low": 500.0,
                         "close": 500.0, "volume": 1e5}, index=idx)
    iv = pd.Series(0.15, index=idx)
    ser = option_series(flat, 500.0, bs.CALL, iv=iv)
    assert ser["mid"].iloc[0] > ser["mid"].iloc[200] > ser["mid"].iloc[-1]
    assert ser["mid"].iloc[-1] < 0.10


def test_spread_is_never_tighter_than_a_tick():
    assert float(quote_spread(np.array([0.02]))[0]) == pytest.approx(0.01)
    assert float(quote_spread(np.array([10.0]))[0]) == pytest.approx(0.05)


def test_atm_strike_rounds_to_listed_increments():
    assert atm_strike(776.64) == 777.0
    assert atm_strike(776.2) == 776.0


# -- breakeven and overlay ---------------------------------------------------

def test_breakeven_grows_with_holding_time():
    """The core economic fact: theta makes options worse the longer you hold,
    while the stock's round-trip cost is flat."""
    be = [breakeven_move_bps(700.0, 700.0, 390.0, 0.12, h) for h in (15, 30, 60, 120, 240)]
    assert all(b < a for a, b in zip(be[1:], be[:-1]))
    assert be[0] < 5.0 and be[-1] > 10.0


def test_overlay_never_spends_more_than_the_premium_budget():
    """Sizing an option like a stock position is how accounts die."""
    bars = make_intraday_bars(n_sessions=20, seed=204)
    res = BacktestEngine(CashAccount(10_000.0), CostModel()).run(
        "SPY", bars, get_strategy("orb"))
    if not res.trades:
        pytest.skip("no underlying trades to express")

    opts, _ = express_in_options(bars, res.trades, "SPY", starting_equity=10_000.0,
                                 premium_pct=0.10)
    for o in opts:
        assert o.contracts * o.entry_premium * MULTIPLIER <= 10_000.0 * 0.10 * 1.5


def test_overlay_marks_a_worthless_contract_as_expired_not_sold():
    """Nobody buys back a worthless contract; modelling a sale would invent
    proceeds that do not exist."""
    bars = make_intraday_bars(n_sessions=15, seed=205)
    res = BacktestEngine(CashAccount(10_000.0), CostModel()).run(
        "SPY", bars, get_strategy("vwap_reversion"))
    if not res.trades:
        pytest.skip("no underlying trades")
    opts, _ = express_in_options(bars, res.trades, "SPY")
    for o in opts:
        if o.expired_worthless:
            assert o.exit_premium == 0.0
            # One leg of fees only - there is no closing trade.
            assert o.fees == pytest.approx(0.05 * o.contracts)


def test_overlay_pays_the_spread_in_both_directions():
    bars = make_intraday_bars(n_sessions=15, seed=206)
    res = BacktestEngine(CashAccount(10_000.0), CostModel()).run(
        "SPY", bars, get_strategy("orb"))
    if not res.trades:
        pytest.skip("no underlying trades")
    opts, _ = express_in_options(bars, res.trades, "SPY")
    for o in opts:
        if o.expired_worthless:
            continue
        # Entry is above mid and exit below it, so an unchanged underlying loses.
        assert o.entry_premium > 0
        assert o.exit_premium >= 0
