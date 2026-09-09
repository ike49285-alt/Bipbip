"""Every feature this project has built, and both sides of the spread.

The indicators are back on and sitting alongside the raw candles, the volatility
estimators, the level distances and the clock - roughly ninety inputs where
earlier runs had twelve to fifty-nine. If anything emerges from combining what
individually measured as nothing, this is where it shows up.

The second half matters more. Every cost model here has charged 3.10 bps on the
assumption we CROSS the spread. Posting a resting order that fills EARNS it
instead: 2.82 bps of spread captured rather than paid, less the 0.278 bps sale
fee, a credit of 2.54. That is an upper bound and deliberately so - it assumes
no adverse selection, which is the one thing the execution question actually
turns on. If nothing clears even at that bound, execution cannot rescue any of
this and tomorrow's measurement is moot. That is worth knowing tonight.
"""
import sys, pathlib, time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np, pandas as pd

from bipbip.core.indicators import (assert_causal, atr, ema, full_stochastic,
                                    heikin_ashi, ichimoku, realised_vol,
                                    relative_volume, rsi)
from bipbip.data.store import BarStore
from bipbip.ml.candles import candle_features, clock_features
from bipbip.ml.discover import make_gbm

CROSS_BPS = 2 * 1.41 + 0.278      # pay the spread both ways, plus the sale fee
EARN_BPS = -(2 * 1.41) + 0.278    # capture it both ways instead
BARS_PER_YEAR = 252 * 13


def features(b: pd.DataFrame) -> pd.DataFrame:
    c, f = b["close"], {}
    for col, s in candle_features(b, 8).items():
        f[col] = s
    ha = heikin_ashi(b)
    for col in ha.columns:
        if col.startswith("ha_") and ha[col].dtype.kind in "fi":
            f[col] = ha[col]
    for col, s in full_stochastic(b, 5, 1, 3).items():
        f["st_" + col] = s
    for col, s in ichimoku(b).items():
        f["ich_" + col] = s.astype("float64") if s.dtype == "boolean" else s
    f["rsi"] = rsi(c, 14)
    f["atr_pct"] = atr(b, 14) / c
    f["rvol"] = relative_volume(b, 20)
    for w in (13, 65, 260):
        f[f"rv_{w}"] = realised_vol(c, w, bars_per_year=BARS_PER_YEAR)
    f["rv_ratio"] = f["rv_13"] / f["rv_65"]
    hl = np.log(b["high"] / b["low"])
    f["park"] = np.sqrt((hl ** 2).rolling(65, min_periods=65).mean() / (4 * np.log(2)))
    f["park_ratio"] = f["park"] / f["rv_65"]
    # Levels: how far price sits from the extremes and from the nearest round
    # number. Never given to any earlier search, which only ever saw twelve bars.
    for w in (13, 65, 260):
        f[f"hi_{w}"] = c / b["high"].rolling(w, min_periods=w).max().shift(1) - 1
        f[f"lo_{w}"] = c / b["low"].rolling(w, min_periods=w).min().shift(1) - 1
    frac = (c % 1.0)
    f["round_dist"] = np.minimum(frac, 1.0 - frac)
    f["half_dist"] = np.minimum(np.abs(frac - 0.5), 1.0 - np.abs(frac - 0.5))
    for n in (1, 2, 4, 13, 65, 260):
        f[f"ret_{n}"] = c.pct_change(n)
    for n in (13, 65, 260):
        f[f"gap_{n}"] = c / ema(c, n) - 1.0
        f[f"tstr_{n}"] = c.pct_change(n) / (
            realised_vol(c, n, bars_per_year=BARS_PER_YEAR) + 1e-9)
    out = pd.DataFrame(f, index=b.index)
    for col, s in clock_features(b.index).items():
        out[col] = s
    return out


def forward(b, h):
    entry, exit_ = b["open"].shift(-1), b["close"].shift(-h)
    day = pd.Series(b.index.normalize(), index=b.index)
    return (exit_ / entry - 1.0).where(day.shift(-h) == day)


def walk(X, y, h, cost, folds=6, seed=0, shuffle=False):
    rng = np.random.default_rng(seed)
    yy = rng.permutation(y) if shuffle else y
    Xv, n = X.to_numpy(dtype="float32"), len(X)
    edge, out = n // (folds + 1), []
    for k in range(folds):
        hi = edge * (k + 1)
        lo2, hi2 = hi + h + 13, min(edge * (k + 2), n)
        if hi2 - lo2 < 200:
            continue
        tr = np.arange(hi)
        tr = tr[np.isfinite(yy[tr])]
        if len(tr) < 2000 or len(np.unique(yy[tr] > 0)) < 2:
            continue
        m = make_gbm(seed=seed).fit(Xv[tr], (yy[tr] > 0).astype(float))
        va = np.arange(lo2, hi2, h)
        va = va[np.isfinite(yy[va])]
        if not len(va):
            continue
        p = m.predict_proba(Xv[va])[:, 1]
        out.append(np.where(p > .5, 1., -1.) * yy[va] * 1e4 - cost)
    return np.concatenate(out) if out else np.array([])


def t(a):
    a = np.asarray(a)
    return a.mean() / a.std(ddof=1) * np.sqrt(len(a)) if len(a) > 1 and a.std() else np.nan


def main():
    t0 = time.time()
    b = BarStore("data/bars").load("TQQQ", "30m").dropna()
    X = assert_causal(features, b)
    print(f"TQQQ 30m: {len(b):,} bars {b.index[0].date()}->{b.index[-1].date()}, "
          f"{X.shape[1]} features (candles + indicators + vol + levels + clock)")
    print(f"crossing the spread costs {CROSS_BPS:+.2f} bps; "
          f"earning it is {EARN_BPS:+.2f}\n")
    print(f"{'horizon':>8} {'trades':>7} {'GROSS':>9} {'t':>6}  "
          f"{'crossing':>9} {'earning':>9}   {'null gross (5x)':>17}")
    for h, lab in ((1, "30 min"), (2, "1 hour"), (4, "2 hours")):
        fwd = forward(b, h)
        ok = np.isfinite(fwd.to_numpy()) & X.notna().all(axis=1).to_numpy()
        Xi, yi = X[ok], fwd[ok].to_numpy()
        gross = walk(Xi, yi, h, 0.0)
        if not len(gross):
            continue
        nulls = [t(walk(Xi, yi, h, 0.0, seed=s, shuffle=True)) for s in range(1, 6)]
        nulls = [x for x in nulls if np.isfinite(x)]
        print(f"{lab:>8} {len(gross):>7,} {gross.mean():>+8.2f}b {t(gross):>6.2f}  "
              f"{gross.mean()-CROSS_BPS:>+8.2f}b {gross.mean()-EARN_BPS:>+8.2f}b   "
              f"mean {np.mean(nulls):>5.2f} best {max(nulls):>5.2f}")
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
