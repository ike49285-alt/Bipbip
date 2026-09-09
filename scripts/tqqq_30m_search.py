"""Fifteen years of TQQQ at thirty minutes: is there a timing edge or not?

Every previous search on this symbol ran on two years of hourly bars covering a
single uninterrupted bull market. A model cannot learn to tell regimes apart in
a sample that contains one, so "no edge found" was not yet a finding. This runs
the same question over 2011-2026 - the taper tantrum, 2015, Volmageddon, March
2020, the 2022 bear - using the indicators the account's owner actually trades:
Heikin-Ashi, a fast stochastic and an Ichimoku cloud.

Three things keep the answer honest.

Costs are charged after the direction is chosen, never netted into the forward
return, because folding them in earlier hands a short a rebate instead of a
charge. Observations are non-overlapping: holding H bars while sampling every
bar makes each observation a shifted copy of the last and inflates t by the
square root of the horizon. And every result is measured against the same search
run on shuffled labels, since a search flexible enough to find an edge is
flexible enough to manufacture one.
"""
import sys, pathlib, time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np, pandas as pd

from bipbip.core.indicators import (assert_causal, atr, ema, full_stochastic,
                                    heikin_ashi, ichimoku, realised_vol,
                                    relative_volume, rsi)
from bipbip.data.store import BarStore
from bipbip.ml.discover import make_gbm

# TQQQ quotes a penny wide on ~$70: 0.01/70.86 = 1.41 bps to cross, paid on the
# way in and again on the way out, plus the SEC fee levied on the sale.
SPREAD_BPS = 1.41
SEC_FEE_BPS = 0.278
COST_BPS = 2 * SPREAD_BPS + SEC_FEE_BPS

BARS_PER_YEAR = 252 * 13


def features(b: pd.DataFrame) -> pd.DataFrame:
    """Only information available at the close of the bar being labelled."""
    c, f = b["close"], {}
    ha = heikin_ashi(b)
    for col in ha.columns:
        if col.startswith("ha_") and ha[col].dtype.kind in "fi":
            f[col] = ha[col]
    st = full_stochastic(b, 5, 1, 3)          # fast stochastic
    for col in st.columns:
        f[f"stoch_{col}"] = st[col]
    ich = ichimoku(b)
    for col in ich.columns:
        s = ich[col]
        f[f"ich_{col}"] = s.astype("float64") if s.dtype == "boolean" else s

    f["rsi"] = rsi(c, 14)
    f["atr_pct"] = atr(b, 14) / c
    f["rvol"] = relative_volume(b, 20)
    f["rv"] = realised_vol(c, 30, bars_per_year=BARS_PER_YEAR)
    for n in (1, 2, 4, 13, 26, 65):
        f[f"ret_{n}"] = c.pct_change(n)
    for n in (13, 65):
        f[f"ema_gap_{n}"] = c / ema(c, n) - 1.0
    f["bar_of_day"] = pd.Series(
        b.index.hour * 2 + (b.index.minute >= 30).astype(int), index=b.index)
    f["dow"] = pd.Series(b.index.dayofweek, index=b.index)
    return pd.DataFrame(f, index=b.index)


def forward_return(b: pd.DataFrame, horizon: int) -> pd.Series:
    """Enter at the NEXT bar's open, exit `horizon` bars later, intraday only."""
    entry = b["open"].shift(-1)
    exit_ = b["close"].shift(-horizon)
    day = pd.Series(b.index.normalize(), index=b.index)
    same_day = day.shift(-horizon) == day
    return (exit_ / entry - 1.0).where(same_day)


def walk_forward(X, fwd, dates, horizon, n_folds=6, seed=0, shuffle=False):
    """Chronological folds, embargoed, sampled every `horizon` bars.

    Sampling at the holding period is what makes each observation a distinct
    trade rather than a shifted copy of the one before it.
    """
    rng = np.random.default_rng(seed)
    y = fwd.to_numpy()
    if shuffle:
        y = rng.permutation(y)
    Xv = X.to_numpy(dtype="float32")
    n = len(X)
    edge = n // (n_folds + 1)
    trades = []
    for k in range(n_folds):
        tr_hi = edge * (k + 1)
        va_lo, va_hi = tr_hi + horizon + 13, min(edge * (k + 2), n)
        if va_hi - va_lo < 200:
            continue
        tr = np.arange(0, tr_hi)
        tr = tr[np.isfinite(y[tr])]
        if len(tr) < 2000 or len(np.unique(y[tr] > 0)) < 2:
            continue
        m = make_gbm(seed=seed).fit(Xv[tr], (y[tr] > 0).astype(float))
        va = np.arange(va_lo, va_hi, horizon)          # non-overlapping
        va = va[np.isfinite(y[va])]
        if len(va) == 0:
            continue
        p = m.predict_proba(Xv[va])[:, 1]
        # Direction from the model; the cost is charged afterwards so a short is
        # debited the spread rather than credited it.
        side = np.where(p > 0.5, 1.0, -1.0)
        conf = np.abs(p - 0.5) > 0.02
        pnl = side * y[va] * 1e4 - COST_BPS
        trades.append(pd.DataFrame({"pnl": pnl[conf], "date": dates[va][conf]}))
    return pd.concat(trades) if trades else pd.DataFrame(columns=["pnl", "date"])


def tstat(a):
    a = np.asarray(a, dtype="float64")
    return a.mean() / a.std(ddof=1) * np.sqrt(len(a)) if len(a) > 1 and a.std() else np.nan


def main():
    t0 = time.time()
    b = BarStore("data/bars").load("TQQQ", "30m").dropna()
    # Prove the features are causal by deleting the future and checking
    # nothing in the past moved, rather than trusting their names.
    X = assert_causal(features, b)
    print(f"TQQQ 30m: {len(b):,} bars  {b.index[0].date()} -> {b.index[-1].date()}"
          f"   {X.shape[1]} features")
    print(f"round-trip cost charged: {COST_BPS:.2f} bps\n")
    print(f"{'horizon':>9} {'trades':>7} {'net bps':>9} {'t':>7} {'win':>6} "
          f"{'null t (5 shuffles)':>22}")

    for h, lab in ((1, "30 min"), (2, "1 hour"), (4, "2 hours"),
                   (13, "1 session")):
        fwd = forward_return(b, h)
        ok = np.isfinite(fwd.to_numpy()) & np.isfinite(X.to_numpy()).any(axis=1)
        Xi, fi, di = X[ok], fwd[ok], b.index[ok]
        real = walk_forward(Xi, fi, di, h)
        if real.empty:
            print(f"{lab:>9}   (no usable folds)")
            continue
        nulls = [tstat(walk_forward(Xi, fi, di, h, seed=s, shuffle=True)["pnl"])
                 for s in range(1, 6)]
        nulls = [x for x in nulls if np.isfinite(x)]
        t = tstat(real["pnl"])
        print(f"{lab:>9} {len(real):>7,} {real['pnl'].mean():>+8.2f}b {t:>7.2f} "
              f"{(real['pnl'] > 0).mean():>6.1%}   "
              f"max {max(nulls):>5.2f}  mean {np.mean(nulls):>5.2f}")
        by_year = real.assign(yr=pd.DatetimeIndex(real["date"]).year).groupby("yr")["pnl"]
        summary = "  ".join(f"{y}:{v:+.0f}" for y, v in by_year.mean().items())
        print(f"{'':>9} by year  {summary}")
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
