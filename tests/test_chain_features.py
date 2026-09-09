"""Tests for chain-snapshot features.

Each case is a defect the first real TQQQ snapshot actually exhibited.
"""
import numpy as np
import pandas as pd
import pytest

from bipbip.data.chain_features import (_trading_years, atm_iv_from_quotes,
                                        implied_spot, snapshot_features,
                                        solve_iv)


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
