"""Collapse an option-chain snapshot into a handful of learnable numbers.

A chain is hundreds of contracts; a model cannot consume it directly and should
not try. What carries information is the shape of the surface - what volatility
is being charged at the money, how much more is charged for downside than
upside, whether the near expiry is dearer than the far one - and how that
compares to what the underlying has actually been doing.

The last of those is the reason this file exists. Realised volatility persists
across independent weeks at 0.711 while returns persist at -0.008, so the gap
between what is implied and what is subsequently realised is the one measured
edge in this project. That gap is `iv_premium` below.

Every field is computed from a single snapshot, so a series of them is only as
long as the collector has been running. Four snapshots a session is enough to
build a usable history in weeks; it is not enough to train anything today, and
`chain_feature_frame` says so by simply returning the rows that exist.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MIN_OI = 10


def _mid(df: pd.DataFrame) -> pd.Series:
    """Mid price, but only where both sides are quoted.

    A one-sided quote has no mid; averaging against a zero would halve it and
    make an illiquid contract look cheap.
    """
    both = (df["bid"] > 0) & (df["ask"] > 0)
    return ((df["bid"] + df["ask"]) / 2.0).where(both)


def _atm_iv(df: pd.DataFrame, spot: float) -> float:
    """Implied volatility at the money: the ATM call and ATM put, averaged.

    Taking the nearest contracts without regard to side gets this wrong. Skew
    means the put is quoted above the call at the same strike, so whichever side
    happens to sit closer to spot drags the estimate with it - here that read
    56.1% where the strikes around spot were quoting 46-50%, which would have
    tripled the implied-minus-realised premium this module exists to measure.

    Both sides are taken at the single strike nearest spot, which is the
    standard construction and is skew-neutral by symmetry.
    """
    d = df[(df["openInterest"] >= MIN_OI) & df["impliedVolatility"].between(0.01, 3.0)]
    if d.empty:
        return float("nan")
    strike = float(d.iloc[(d["strike"] - spot).abs().argmin()]["strike"])
    at = d[d["strike"] == strike]
    per_side = at.groupby("side")["impliedVolatility"].mean()
    if per_side.empty:
        return float("nan")
    return float(per_side.mean())


def snapshot_features(chain: pd.DataFrame, spot: float | None = None,
                      realised_vol: float | None = None) -> dict:
    """One row of surface state from one chain snapshot.

    `spot` defaults to the price recovered from the chain's own put-call parity
    rather than a live quote, so it is the price at the instant the snapshot was
    taken instead of whenever the analysis happens to run.
    """
    if spot is None:
        spot = implied_spot(chain)
    c = chain.copy()
    c["expiry"] = pd.to_datetime(c["expiry"]).dt.tz_localize(None)
    asof = pd.Timestamp(c["fetched_at"].max()).tz_localize(None).normalize()
    c["dte"] = (c["expiry"] - asof).dt.days
    c = c[c["dte"] >= 0]
    live = c[c["openInterest"] >= MIN_OI]
    calls, puts = live[live.side == "call"], live[live.side == "put"]

    # Volatilities are inverted from the contracts' own mids on a trading-day
    # clock. Calendar time overstates short-dated volatility - a weekend carries
    # far less variance than two sessions - and realised volatility here is
    # annualised over 252 days, so the two must use the same clock to be
    # comparable at all.
    iv_near = _atm_iv_at(c, spot, True, asof)
    iv_far = _atm_iv_at(c, spot, False, asof)

    # Skew as the extra volatility charged five percent below spot over five
    # percent above it. Strike distance stands in for delta because the stored
    # chain carries no greeks; the two agree closely enough at this moneyness.
    dte_near = int(c["dte"][c["dte"] > 0].min()) if (c["dte"] > 0).any() else 1
    years = _trading_years(dte_near, asof)

    def wing(df, target, is_call):
        d = df[(df["dte"] == dte_near) & (df["bid"] > 0) & (df["ask"] > 0)]
        if d.empty:
            return float("nan")
        row = d.iloc[(d["strike"] - target).abs().argmin()]
        return solve_iv((row["bid"] + row["ask"]) / 2.0, spot,
                        float(row["strike"]), years, is_call)

    put_wing = wing(puts, spot * 0.95, False)
    call_wing = wing(calls, spot * 1.05, True)

    out = {
        "spot": spot,
        "atm_iv": iv_near,
        "dte_near": dte_near,
        "term_slope": iv_far - iv_near,
        "skew": put_wing - call_wing,
        "put_call_volume": float(puts["volume"].sum() / max(calls["volume"].sum(), 1.0)),
        "put_call_oi": float(puts["openInterest"].sum() / max(calls["openInterest"].sum(), 1)),
        "total_oi": int(live["openInterest"].sum()),
        "median_spread": float((live["ask"] - live["bid"]).div(_mid(live)).median()),
        "n_contracts": int(len(live)),
    }
    # The variance risk premium: what is being charged, less what the underlying
    # has actually been doing. Positive means options are dear relative to the
    # tape, which is the normal state and the reason selling premium is a trade
    # at all.
    out["iv_premium"] = (iv_near - realised_vol
                         if realised_vol is not None and np.isfinite(iv_near)
                         else float("nan"))
    return out


def chain_feature_frame(store, symbol: str, spot_by_date: pd.Series,
                        realised_by_date: pd.Series | None = None) -> pd.DataFrame:
    """Every stored snapshot for `symbol`, as one row each, oldest first."""
    rows = []
    for asof in store.dates(symbol):
        try:
            chain = store.load(symbol, asof)
        except Exception:
            continue
        key = pd.Timestamp(asof).normalize()
        spot = spot_by_date.get(key, np.nan)
        if not np.isfinite(spot):
            continue
        rv = None if realised_by_date is None else realised_by_date.get(key)
        rows.append({"date": key,
                     **snapshot_features(chain, float(spot),
                                         None if rv is None or not np.isfinite(rv) else float(rv))})
    return pd.DataFrame(rows).set_index("date").sort_index() if rows else pd.DataFrame()


def implied_spot(chain: pd.DataFrame, dte: int | None = None) -> float:
    """Recover the underlying's price at the moment the snapshot was taken.

    Put-call parity fixes it exactly: a call and a put on the same strike and
    expiry satisfy C - P = S - K once rates and dividends are negligible, which
    they are over a few days. Every strike with both sides quoted therefore
    votes on the spot, and the median of those votes is robust to a few bad
    quotes.

    This exists because the alternative - taking a live quote when the analysis
    runs - reads the spot at the wrong instant. A snapshot fetched at 15:40 and
    a quote pulled at 16:29 differed by twenty cents here, which is enough to
    shift every moneyness in the chain.
    """
    c = chain[(chain["bid"] > 0) & (chain["ask"] > 0)].copy()
    if c.empty:
        return float("nan")
    c["expiry_ts"] = pd.to_datetime(c["expiry"]).dt.tz_localize(None)
    asof = pd.Timestamp(c["fetched_at"].max()).tz_localize(None).normalize()
    c["dte"] = (c["expiry_ts"] - asof).dt.days
    if dte is None:
        live = c[c["dte"] > 0]
        if live.empty:
            return float("nan")
        dte = int(live["dte"].min())
    c = c[c["dte"] == dte]
    mid = (c["bid"] + c["ask"]) / 2.0
    rel = (c["ask"] - c["bid"]) / mid.replace(0.0, np.nan)
    w = c.assign(mid=mid, rel=rel).pivot_table(
        index="strike", columns="side", values=["mid", "rel"]).dropna()
    if w.empty or ("mid", "call") not in w or ("mid", "put") not in w:
        return float("nan")
    votes = w[("mid", "call")] - w[("mid", "put")] + w.index

    # Every strike votes, but the far ones vote badly: where one side is nearly
    # worthless its mid is quote noise rather than price, and those strikes drag
    # the median. Taking all of them here returned 71.33 while the strikes
    # around the money agreed on 70.95 - a 38-cent error, enough to break parity
    # by a quarter of a point and make puts look twenty volatility points dearer
    # than calls. So a rough pass locates the money, and the vote is retaken
    # using only strikes near it whose two-sided quotes are tight enough to mean
    # something.
    rough = float(votes.median())
    near = (np.abs(w.index - rough) < 0.05 * rough)
    tight = ((w[("mid", "call")] > 0.10) & (w[("mid", "put")] > 0.10)
             & (w[("rel", "call")] < 0.35) & (w[("rel", "put")] < 0.35))
    sel = votes[near & tight]
    return float(sel.median()) if len(sel) >= 3 else rough


def _bs_call(s: float, k: float, t: float, vol: float) -> float:
    from math import erf, log, sqrt
    if vol <= 0 or t <= 0:
        return max(0.0, s - k)
    d1 = (log(s / k) + 0.5 * vol * vol * t) / (vol * sqrt(t))
    d2 = d1 - vol * sqrt(t)
    n = lambda x: 0.5 * (1.0 + erf(x / sqrt(2.0)))
    return s * n(d1) - k * n(d2)


def solve_iv(price: float, spot: float, strike: float, years: float,
             is_call: bool, lo: float = 0.01, hi: float = 5.0) -> float:
    """Invert Black-Scholes for volatility, from the contract's own mid.

    The stored `impliedVolatility` column cannot be used: on this snapshot it
    reports 48.7% for the $71.5 call and 67.0% for the $71.5 put of the same
    expiry. Parity makes that impossible - two contracts on one strike imply one
    volatility - so the field is carrying something other than a clean
    inversion, and any premium computed from it inherits the error.

    Puts are converted to their parity-equivalent call first, so both sides go
    through one code path and agree by construction.
    """
    if not (np.isfinite(price) and price > 0 and years > 0):
        return float("nan")
    target = price if is_call else price + spot - strike     # parity
    intrinsic = max(0.0, spot - strike)
    if target <= intrinsic + 1e-9:
        return float("nan")                                  # no time value to invert
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if _bs_call(spot, strike, years, mid) < target:
            lo = mid
        else:
            hi = mid
    out = 0.5 * (lo + hi)
    return float("nan") if out < 0.011 or out > 4.99 else float(out)


def atm_iv_from_quotes(chain: pd.DataFrame, spot: float | None = None) -> dict:
    """At-the-money volatility for the nearest expiry, inverted from mids."""
    c = chain[(chain["bid"] > 0) & (chain["ask"] > 0)].copy()
    c["expiry_ts"] = pd.to_datetime(c["expiry"]).dt.tz_localize(None)
    asof = pd.Timestamp(c["fetched_at"].max()).tz_localize(None).normalize()
    c["dte"] = (c["expiry_ts"] - asof).dt.days
    live = c[c["dte"] > 0]
    if live.empty:
        return {"atm_iv": float("nan"), "dte": float("nan"), "spot": float("nan")}
    dte = int(live["dte"].min())
    if spot is None:
        spot = implied_spot(chain, dte)
    near = live[live["dte"] == dte]
    strike = float(near.iloc[(near["strike"] - spot).abs().argmin()]["strike"])
    years = dte / 365.0
    vols = []
    for _, row in near[near["strike"] == strike].iterrows():
        v = solve_iv((row["bid"] + row["ask"]) / 2.0, spot, strike, years,
                     row["side"] == "call")
        if np.isfinite(v):
            vols.append(v)
    return {"atm_iv": float(np.mean(vols)) if vols else float("nan"),
            "dte": dte, "spot": spot, "strike": strike, "n_sides": len(vols)}


TRADING_DAYS_PER_YEAR = 252
CALENDAR_DAYS_PER_YEAR = 365


def _trading_years(dte: int, asof: "pd.Timestamp") -> float:
    """Convert days-to-expiry into years of *trading* time.

    An option's variance accrues when the market is open. Pricing two calendar
    days as 2/365 of a year assigns a weekend the same variance as two sessions,
    which inflates short-dated implied volatility - here by twelve points - and
    then compares it against a realised number annualised over 252 days. The
    approximation used is the sessions the period contains, at 252 a year.
    """
    # Scaling calendar days by 252/365 and then dividing by 252 returns
    # dte/365 - the calendar figure this function exists to replace. The
    # sessions have to actually be counted.
    sessions = max(1, int(np.busday_count(
        asof.date() if hasattr(asof, "date") else asof,
        (pd.Timestamp(asof) + pd.Timedelta(days=dte)).date())))
    return sessions / TRADING_DAYS_PER_YEAR


def _atm_iv_at(c: pd.DataFrame, spot: float, near: bool, asof) -> float:
    """ATM implied volatility for the nearest or furthest expiry, from mids."""
    live = c[(c["dte"] > 0) & (c["bid"] > 0) & (c["ask"] > 0)]
    if live.empty:
        return float("nan")
    dte = int(live["dte"].min() if near else live["dte"].max())
    exp = live[live["dte"] == dte]
    strike = float(exp.iloc[(exp["strike"] - spot).abs().argmin()]["strike"])
    years = _trading_years(dte, asof)
    vols = [solve_iv((r["bid"] + r["ask"]) / 2.0, spot, strike, years,
                     r["side"] == "call")
            for _, r in exp[exp["strike"] == strike].iterrows()]
    vols = [v for v in vols if np.isfinite(v)]
    return float(np.mean(vols)) if vols else float("nan")
