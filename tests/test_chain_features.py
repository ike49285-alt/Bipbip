"""Tests for chain-snapshot features.

Each case is a defect the first real TQQQ snapshot actually exhibited.
"""
import numpy as np
import pandas as pd
import pytest

from bipbip.data.chain_features import (_bs_call, _trading_years,
                                        atm_iv_from_quotes, implied_spot,
                                        snapshot_features, solve_iv)


def _chain(spot=100.0, vol=0.40, dte=7, strikes=None, noise=None):
    """A synthetic chain priced consistently, so parity holds by construction."""
    from bipbip.data.chain_features import _bs_call
    strikes = strikes if strikes is not None else np.arange(90.0, 111.0, 1.0)
    years = _trading_years(dte, pd.Timestamp("2026-09-09"))
    rows = []
    for k in strikes:
        c = _bs_call(spot, k, years, vol)
        p = c - spot + k                      # put-call parity
        for side, px in (("call", c), ("put", p)):
            px = max(px, 0.01)
            if noise:
                px += noise.get((k, side), 0.0)
            rows.append({"contractSymbol": f"X{k}{side}", "strike": float(k),
                         "lastPrice": px, "bid": px * 0.99, "ask": px * 1.01,
                         "volume": 10.0, "openInterest": 500,
                         "impliedVolatility": vol, "inTheMoney": k < spot,
                         "expiry": str((pd.Timestamp("2026-09-09")
                                        + pd.Timedelta(days=dte)).date()),
                         "side": side,
                         "fetched_at": "2026-09-09T15:40:00+00:00"})
    return pd.DataFrame(rows)


def test_trading_years_counts_sessions_not_calendar_days():
    """Scaling by 252/365 and dividing by 252 returns the calendar figure."""
    wed = pd.Timestamp("2026-09-09")
    assert _trading_years(2, wed) * 252 == pytest.approx(2)      # Wed -> Fri
    assert _trading_years(7, wed) * 252 == pytest.approx(5)      # spans a weekend
    assert _trading_years(2, wed) != pytest.approx(2 / 365)


def test_trading_years_never_returns_zero():
    same_day = pd.Timestamp("2026-09-09")
    assert _trading_years(0, same_day) > 0


def test_implied_spot_recovers_the_underlying_from_parity():
    ch = _chain(spot=100.0)
    assert implied_spot(ch) == pytest.approx(100.0, abs=0.05)


def test_implied_spot_ignores_the_illiquid_tail():
    """Far strikes vote badly; taking all of them shifted spot by 38 cents."""
    # Corrupt the deep wings, where one side is nearly worthless.
    noise = {(90.0, "put"): 0.9, (110.0, "call"): 0.9}
    ch = _chain(spot=100.0, noise=noise)
    assert implied_spot(ch) == pytest.approx(100.0, abs=0.15)


def test_solve_iv_recovers_the_volatility_it_was_priced_with():
    ch = _chain(spot=100.0, vol=0.40, dte=7)
    years = _trading_years(7, pd.Timestamp("2026-09-09"))
    row = ch[(ch.strike == 100.0) & (ch.side == "call")].iloc[0]
    got = solve_iv((row.bid + row.ask) / 2, 100.0, 100.0, years, True)
    assert got == pytest.approx(0.40, abs=0.02)


def test_call_and_put_imply_the_same_volatility():
    """Parity means one strike has one volatility; a gap means a broken spot."""
    ch = _chain(spot=100.0, vol=0.40, dte=7)
    s = implied_spot(ch)
    years = _trading_years(7, pd.Timestamp("2026-09-09"))
    c = ch[(ch.strike == 101.0) & (ch.side == "call")].iloc[0]
    p = ch[(ch.strike == 101.0) & (ch.side == "put")].iloc[0]
    cv = solve_iv((c.bid + c.ask) / 2, s, 101.0, years, True)
    pv = solve_iv((p.bid + p.ask) / 2, s, 101.0, years, False)
    assert abs(cv - pv) < 0.03, (cv, pv)


def test_atm_iv_does_not_average_a_call_against_a_put_of_different_strikes():
    """Mixing sides let skew drag ATM vol to 56% where the strikes quoted 46-50%."""
    ch = _chain(spot=100.0, vol=0.40, dte=7)
    out = atm_iv_from_quotes(ch)
    assert out["atm_iv"] == pytest.approx(0.40, abs=0.03)
    assert out["n_sides"] == 2


@pytest.mark.parametrize("dte", [1, 2, 3, 5, 7, 10])
def test_atm_iv_inverts_on_the_same_clock_the_chain_was_priced_on(dte):
    """The chain is priced on trading time, so it must be inverted on it.

    This function used dte/365 while the rest of the module used
    _trading_years, and on the real TQQQ snapshots that overstated ATM vol by
    20.4% - 70.2% against 58.3%. The old test could not see it: it ran only at
    dte=7, which is the one horizon where the two clocks nearly agree (5/252
    against 7/365, about 3% apart in T, under 2% in vol). At one to three days
    the same mismatch is 20%, so the short horizons are the ones that bite -
    and short-dated is exactly what this project prices.
    """
    ch = _chain(spot=100.0, vol=0.40, dte=dte)
    assert atm_iv_from_quotes(ch)["atm_iv"] == pytest.approx(0.40, abs=0.01)


def test_snapshot_features_derives_its_own_spot():
    ch = _chain(spot=100.0, vol=0.40, dte=7)
    f = snapshot_features(ch, realised_vol=0.30)
    assert f["spot"] == pytest.approx(100.0, abs=0.1)
    assert f["atm_iv"] == pytest.approx(0.40, abs=0.03)
    assert f["iv_premium"] == pytest.approx(0.10, abs=0.03)


def test_snapshot_features_survives_a_chain_with_no_spot_column():
    """The first collected snapshot had none; parity makes the column optional."""
    ch = _chain().drop(columns=[c for c in ("spot",) if c in _chain().columns])
    assert "spot" not in ch.columns
    f = snapshot_features(ch)
    assert np.isfinite(f["spot"])


def test_solve_iv_rejects_a_price_with_no_time_value():
    """An option quoted at or below intrinsic has no volatility to invert."""
    assert not np.isfinite(solve_iv(10.0, 110.0, 100.0, 0.02, True))


# ---------------------------------------------------------------------------
# Pricing, checked against an implementation that is not this one.
#
# Every test above builds its chain with `_chain(...)`, and that helper prices
# each strike by calling `_bs_call` - the function under test - then sets the
# put by construction as `c - spot + k`. So the IV round trip inverts the same
# formula it was priced with, and the parity test asserts an identity the
# fixture just imposed. Both pass whatever `_bs_call` does.
#
# Mutation testing showed the consequence: 96 of 133 mutants survived this
# module, including nearly every operator in the Black-Scholes expression. The
# formula is CORRECT today - it matches scipy to machine precision across the
# surface - but nothing here would notice it becoming wrong, and this is the
# code that prices any option trade the project actually places.
#
# scipy is an independent implementation of the normal CDF and a declared
# dependency, so it is used as the oracle rather than a handful of constants.
# A hand-computed reference was tried first and was wrong at the second point,
# which is its own argument for using a library that is already tested.
# ---------------------------------------------------------------------------

def _reference_call(s, k, t, vol):
    """Black-Scholes call at r=0, via scipy rather than via the module."""
    from math import log, sqrt
    from scipy.stats import norm
    if vol <= 0 or t <= 0:
        return max(0.0, s - k)
    d1 = (log(s / k) + 0.5 * vol * vol * t) / (vol * sqrt(t))
    d2 = d1 - vol * sqrt(t)
    return float(s * norm.cdf(d1) - k * norm.cdf(d2))


def _reference_put(s, k, t, vol):
    from math import log, sqrt
    from scipy.stats import norm
    if vol <= 0 or t <= 0:
        return max(0.0, k - s)
    d1 = (log(s / k) + 0.5 * vol * vol * t) / (vol * sqrt(t))
    d2 = d1 - vol * sqrt(t)
    return float(k * norm.cdf(-d2) - s * norm.cdf(-d1))


PRICING_GRID = [
    (100.0, 100.0, 1.00, 0.20),      # at the money, one year
    (100.0, 90.0, 0.50, 0.30),       # in the money, six months
    (100.0, 110.0, 0.25, 0.60),      # out of the money, high vol
    (50.0, 55.0, 2.00, 0.15),        # cheap underlying, long dated
    (200.0, 180.0, 0.08, 0.45),      # short dated, deep in the money
]


@pytest.mark.parametrize("s,k,t,vol", PRICING_GRID)
def test_the_pricing_formula_agrees_with_an_independent_implementation(s, k, t,
                                                                       vol):
    """One reference point can be matched by a formula wrong elsewhere, so the
    grid spans moneyness, maturity and volatility."""
    assert _bs_call(s, k, t, vol) == pytest.approx(_reference_call(s, k, t, vol),
                                                   rel=1e-9, abs=1e-9)


def test_zero_volatility_prices_the_intrinsic_value():
    """With no volatility there is no time value, so the call is worth exactly
    what exercising it would pay - CLAUDE.md's own no-arbitrage floor."""
    assert _bs_call(150.0, 100.0, 1.0, 0.0) == pytest.approx(50.0)
    assert _bs_call(90.0, 100.0, 1.0, 0.0) == pytest.approx(0.0)


def test_an_expired_option_is_worth_its_intrinsic_value():
    assert _bs_call(150.0, 100.0, 0.0, 0.40) == pytest.approx(50.0)
    assert _bs_call(90.0, 100.0, 0.0, 0.40) == pytest.approx(0.0)


def test_price_rises_with_volatility_and_never_falls():
    """Vega is positive everywhere. A sign error in d1 or d2 can still match a
    single reference point while getting this wrong."""
    prices = [_bs_call(100.0, 100.0, 1.0, v)
              for v in (0.05, 0.1, 0.2, 0.4, 0.8, 1.6)]
    assert all(b > a for a, b in zip(prices, prices[1:]))


def test_price_rises_with_time_and_never_falls():
    prices = [_bs_call(100.0, 100.0, t, 0.25)
              for t in (0.01, 0.1, 0.5, 1.0, 2.0)]
    assert all(b > a for a, b in zip(prices, prices[1:]))


def test_the_call_never_prices_below_intrinsic_or_above_the_underlying():
    """The two no-arbitrage bounds. An option cannot be worth less than
    exercising it, nor more than the share it delivers."""
    for s in (60.0, 100.0, 140.0):
        for k in (80.0, 100.0, 120.0):
            for vol in (0.1, 0.5, 1.5):
                c = _bs_call(s, k, 1.0, vol)
                assert c >= max(0.0, s - k) - 1e-9, (s, k, vol, c)
                assert c <= s + 1e-9, (s, k, vol, c)


@pytest.mark.parametrize("s,k,t,vol", PRICING_GRID)
def test_put_call_parity_holds_between_independently_priced_legs(s, k, t, vol):
    """Parity as a CHECK, not as a construction.

    The fixture sets the put to `c - spot + k`, so parity is true there by
    definition and constrains nothing. Here each leg is priced on its own and
    parity is then verified: C - P = S - K at r = 0.
    """
    c = _bs_call(s, k, t, vol)
    p = _reference_put(s, k, t, vol)
    assert c - p == pytest.approx(s - k, abs=1e-9)


@pytest.mark.parametrize("vol", [0.10, 0.20, 0.35, 0.60])
def test_solve_iv_inverts_a_price_this_module_did_not_produce(vol):
    """The round trip, with the loop broken.

    `test_solve_iv_recovers_the_volatility_it_was_priced_with` feeds solve_iv a
    price `_bs_call` generated, so it recovers the input whatever the formula
    does. This feeds it a price from the independent implementation instead: if
    the pricing is wrong, the recovered volatility is wrong with it.
    """
    px = _reference_call(100.0, 100.0, 1.0, vol)
    assert solve_iv(px, 100.0, 100.0, 1.0, True) == pytest.approx(vol, abs=1e-3)
