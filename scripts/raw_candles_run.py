"""Pure candles, no indicators: can the model find its own structure?

Every previous search handed the model a view - Heikin-Ashi smoothing, a
stochastic, an Ichimoku cloud. This hands it the bare chart: the last N candles
as open, high, low, close and volume, and nothing computed on top of them. If
there is structure in the tape, a gradient-boosted tree with 59 raw inputs and
fifteen years of bars has as fair a shot at it as any hand-built overlay.

Giving the model more freedom makes the honesty checks matter more, not less, so
they are unchanged: costs charged after the direction is chosen, observations
sampled at the holding period so each is a distinct trade, chronological folds
with an embargo, and the identical search re-run on shuffled labels. A search
with 59 features is more able to find an edge and equally more able to invent
one, and only the null says which happened.

The lookback is swept rather than picked, because choosing the best one after
seeing the results is itself a search that needs its own null.
"""
import sys, pathlib, time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np, pandas as pd

from bipbip.core.indicators import assert_causal
from bipbip.data.store import BarStore
from bipbip.ml.candles import candle_features, clock_features
from bipbip.ml.discover import make_gbm

SPREAD_BPS, SEC_FEE_BPS = 1.41, 0.278
COST_BPS = 2 * SPREAD_BPS + SEC_FEE_BPS


def forward_return(b: pd.DataFrame, horizon: int) -> pd.Series:
    entry, exit_ = b["open"].shift(-1), b["close"].shift(-horizon)
    day = pd.Series(b.index.normalize(), index=b.index)
    return (exit_ / entry - 1.0).where(day.shift(-horizon) == day)


def walk_forward(X, y_raw, horizon, n_folds=6, seed=0, shuffle=False):
    rng = np.random.default_rng(seed)
    y = rng.permutation(y_raw) if shuffle else y_raw
    Xv, n = X.to_numpy(dtype="float32"), len(X)
    edge, out = n // (n_folds + 1), []
    for k in range(n_folds):
        tr_hi = edge * (k + 1)
        va_lo, va_hi = tr_hi + horizon + 13, min(edge * (k + 2), n)
        if va_hi - va_lo < 200:
            continue
        tr = np.arange(tr_hi)
        tr = tr[np.isfinite(y[tr])]
        if len(tr) < 2000 or len(np.unique(y[tr] > 0)) < 2:
            continue
        m = make_gbm(seed=seed).fit(Xv[tr], (y[tr] > 0).astype(float))
        va = np.arange(va_lo, va_hi, horizon)
        va = va[np.isfinite(y[va])]
        if not len(va):
            continue
        p = m.predict_proba(Xv[va])[:, 1]
        side = np.where(p > 0.5, 1.0, -1.0)
        keep = np.abs(p - 0.5) > 0.02
        out.append((side * y[va] * 1e4 - COST_BPS)[keep])
    return np.concatenate(out) if out else np.array([])


def tstat(a):
    a = np.asarray(a, dtype="float64")
    return a.mean() / a.std(ddof=1) * np.sqrt(len(a)) if len(a) > 1 and a.std() else np.nan


def main():
    t0 = time.time()
    b = BarStore("data/bars").load("TQQQ", "30m").dropna()
    print(f"TQQQ 30m: {len(b):,} bars {b.index[0].date()} -> {b.index[-1].date()}")
    print(f"features: raw OHLCV candles only.  cost {COST_BPS:.2f} bps/round trip\n")
    print(f"{'lookback':>9} {'horizon':>8} {'feats':>6} {'trades':>7} {'net':>8} "
          f"{'t':>6}   {'null t: mean':>12} {'best':>6}  {'verdict':>12}")

    for lb in (4, 12, 26):
        Xc = assert_causal(lambda x, L=lb: candle_features(x, L), b)
        for variant, X in (("", Xc),
                           ("+clock", pd.concat([Xc, clock_features(b.index)], axis=1))):
            for h in (1, 2, 4):
                fwd = forward_return(b, h)
                ok = np.isfinite(fwd.to_numpy()) & X.notna().all(axis=1).to_numpy()
                Xi, yi = X[ok], fwd[ok].to_numpy()
                real = walk_forward(Xi, yi, h)
                if not len(real):
                    continue
                nulls = [tstat(walk_forward(Xi, yi, h, seed=s, shuffle=True))
                         for s in range(1, 6)]
                nulls = [x for x in nulls if np.isfinite(x)]
                t = tstat(real)
                beat = sum(t > x for x in nulls)
                verdict = ("SIGNAL" if beat == len(nulls) and t > 2
                           else "noise")
                lab = f"{lb}{variant}"
                print(f"{lab:>9} {h*30:>6}min {Xi.shape[1]:>6} {len(real):>7,} "
                      f"{real.mean():>+7.2f}b {t:>6.2f}   {np.mean(nulls):>12.2f} "
                      f"{max(nulls):>6.2f}  {verdict:>12}")
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
