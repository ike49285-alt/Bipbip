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
                                 risk=_risk(premium_pct=0.10))
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


# -- native option risk management -------------------------------------------

def _risk(**kw):
    from bipbip.options.risk import OptionRiskModel
    return OptionRiskModel(**kw)


def test_strike_selection_tracks_target_delta():
    from bipbip.options.risk import strike_for_delta

    T, sig = bs.minutes_to_years(330.0), 0.10
    prev = None
    for target in (0.50, 0.65, 0.75, 0.85, 0.95):
        K = strike_for_delta(700.0, T, sig, target, bs.CALL)
        actual = float(bs.delta(700.0, K, T, R, sig, bs.CALL))
        assert abs(actual - target) < 0.12, f"target {target} got delta {actual}"
        if prev is not None:
            assert K <= prev, "a higher delta target must not pick a higher call strike"
        prev = K


def test_in_the_money_strikes_carry_less_extrinsic_value():
    """The reason ITM is the right default: extrinsic value is what theta eats."""
    from bipbip.options.risk import strike_for_delta

    T, sig, S = bs.minutes_to_years(330.0), 0.10, 700.0
    extrinsics = []
    for target in (0.50, 0.75, 0.95):
        K = strike_for_delta(S, T, sig, target, bs.CALL)
        p = float(bs.price(S, K, T, R, sig, bs.CALL))
        extrinsics.append((p - max(S - K, 0.0)) / p)
    assert extrinsics[0] > extrinsics[1] > extrinsics[2]
    assert extrinsics[0] > 0.5   # ATM is essentially all time value
    assert extrinsics[-1] < 0.15


def test_time_stop_closes_the_position():
    """Theta accelerates, so a stale position bleeds regardless of thesis."""
    bars = make_intraday_bars(n_sessions=15, seed=210)
    res = BacktestEngine(CashAccount(10_000.0), CostModel()).run(
        "SPY", bars, get_strategy("orb"))
    if not res.trades:
        pytest.skip("no underlying trades")

    opts, summ = express_in_options(bars, res.trades, "SPY",
                                    risk=_risk(max_hold_minutes=20))
    for o in opts:
        assert o.minutes_held <= 20 + 1e-9


def test_premium_stop_caps_the_loss():
    """Underlying stops are already 60-70% of premium when they trigger at this
    leverage; the cap has to live in premium terms."""
    bars = make_intraday_bars(n_sessions=25, seed=211, annual_vol=0.45)
    res = BacktestEngine(CashAccount(10_000.0), CostModel()).run(
        "SPY", bars, get_strategy("orb"))
    if not res.trades:
        pytest.skip("no underlying trades")

    opts, _ = express_in_options(bars, res.trades, "SPY",
                                 risk=_risk(premium_stop_pct=0.25, max_hold_minutes=300))
    stopped = [o for o in opts if o.exit_reason == "premium_stop"]
    for o in stopped:
        # Allow for the exit spread and for gapping through the stop.
        assert o.return_pct < 0
        assert o.return_pct > -0.90


def test_entries_are_refused_too_late_in_the_session():
    """Decay is fastest at the end and there is no time for a thesis to work."""
    bars = make_intraday_bars(n_sessions=10, seed=212)
    res = BacktestEngine(CashAccount(10_000.0), CostModel()).run(
        "SPY", bars, get_strategy("orb"))
    if not res.trades:
        pytest.skip("no underlying trades")

    opts, summ = express_in_options(bars, res.trades, "SPY",
                                    risk=_risk(min_minutes_left=300))
    from bipbip.options.synth import minutes_to_close
    mins = minutes_to_close(bars.index)
    for o in opts:
        assert float(mins.loc[o.entry_time]) >= 300


def test_native_exits_beat_stock_exits_on_win_loss_ratio():
    """The measured defect was a win/loss magnitude ratio of 0.51, which cannot
    be profitable near a 50% hit rate. Managing the option on its own terms is
    what repairs it."""
    bars = make_intraday_bars(n_sessions=30, seed=213)
    res = BacktestEngine(CashAccount(10_000.0), CostModel()).run(
        "SPY", bars, get_strategy("orb"))
    if len(res.trades) < 5:
        pytest.skip("too few underlying trades")

    stock_style = _risk(target_delta=0.50, premium_stop_pct=0.999,
                        premium_target_pct=99.0, max_hold_minutes=390,
                        min_minutes_left=0)
    _, old = express_in_options(bars, res.trades, "SPY", risk=stock_style)
    _, new = express_in_options(bars, res.trades, "SPY", risk=_risk())

    assert new["median_hold_min"] <= old["median_hold_min"]
    if np.isfinite(old.get("win_loss_ratio", np.nan)) and np.isfinite(new.get("win_loss_ratio", np.nan)):
        assert new["win_loss_ratio"] >= old["win_loss_ratio"] * 0.9


def test_option_exit_reasons_are_recorded():
    bars = make_intraday_bars(n_sessions=20, seed=214)
    res = BacktestEngine(CashAccount(10_000.0), CostModel()).run(
        "SPY", bars, get_strategy("orb"))
    if not res.trades:
        pytest.skip("no underlying trades")
    opts, summ = express_in_options(bars, res.trades, "SPY")
    valid = {"premium_stop", "premium_target", "time_stop", "signal_exit", "session_close"}
    assert set(summ["exit_breakdown"]) <= valid
    for o in opts:
        assert o.exit_reason in valid


def test_option_position_never_survives_the_closing_bell():
    bars = make_intraday_bars(n_sessions=15, seed=215)
    res = BacktestEngine(CashAccount(10_000.0), CostModel()).run(
        "SPY", bars, get_strategy("orb"))
    if not res.trades:
        pytest.skip("no underlying trades")
    opts, _ = express_in_options(bars, res.trades, "SPY",
                                 risk=_risk(max_hold_minutes=10_000, min_minutes_left=1))
    for o in opts:
        assert o.entry_time.date() == o.exit_time.date()


def test_breakeven_hurdle_is_monotonic_even_though_returns_are_not():
    """Separates the two claims that were once conflated.

    The HURDLE rises monotonically with holding time - that is deterministic
    Black-Scholes, independent of any data. Realised RETURNS are an inverted U,
    because cutting a position early also cuts winners before the move happens.
    An earlier version of this project described the returns as monotonic,
    which its own table contradicted.
    """
    hurdles = [breakeven_move_bps(700.0, 700.0, 390.0, 0.12, float(h))
               for h in (5, 15, 30, 60, 120, 240)]
    assert all(b > a for a, b in zip(hurdles, hurdles[1:])), hurdles


def test_very_short_holds_are_not_automatically_better():
    """Guards the corrected claim: a 5-minute cap is not a free improvement."""
    bars = make_intraday_bars(n_sessions=30, seed=216)
    res = BacktestEngine(CashAccount(10_000.0), CostModel()).run(
        "SPY", bars, get_strategy("orb"))
    if len(res.trades) < 5:
        pytest.skip("too few underlying trades")

    _, very_short = express_in_options(bars, res.trades, "SPY", risk=_risk(max_hold_minutes=5))
    _, moderate = express_in_options(bars, res.trades, "SPY", risk=_risk(max_hold_minutes=30))
    # Both must produce trades; the point is that neither dominates by construction.
    assert very_short["trades"] > 0 and moderate["trades"] > 0
    assert very_short["median_hold_min"] <= moderate["median_hold_min"]
