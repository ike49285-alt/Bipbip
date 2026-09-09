"""Is the volatility gain real forecasting, or the model naming the symbol?

A pooled R-squared across twenty ETFs is easy to win dishonestly. SOXL realises
several times XLU's volatility, so most of the variance in the pooled target is
"which ETF is this", and any feature encoding price level or volume scale lets a
model score that without forecasting anything. HAR sees only the symbol's own
trailing volatility, so a model with symbol-identifying features would beat it by
a wide margin while knowing nothing extra.

Two checks separate those. Within-symbol R-squared removes the level differences
entirely - if the gain survives it, the model is forecasting rather than
labelling. And an ablation shows which features carry it: a gain that is all
time-of-day is real but not an edge, because the intraday volatility smile is
not a secret.

Also fixes the annualisation. realised() returns a per-BAR root mean square, so
the constant is sqrt(252 * 390) regardless of the forecast horizon; dividing by
the horizon understated the level roughly sevenfold and made the error in
volatility points meaningless.
"""
import sys, pathlib, time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np, pandas as pd

from bipbip.ml.discover import make_gbm_regressor
from scripts.vol_predict import BASKET, build, r2

MIN_PER_YEAR = 252 * 390


def fit_r2(Xtr, ytr, Xte, yte):
    m = make_gbm_regressor(seed=0).fit(Xtr, ytr)
    return r2(yte, m.predict(Xte)), m.predict(Xte)


def main():
    t0 = time.time()
    horizon = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    Y, H, X, S = [], [], [], []
    for sym in BASKET:
        got = build(sym, horizon)
        if got is None:
            continue
        y, h, f, _ = got
        Y.append(y); H.append(h); X.append(f); S.append(np.full(len(y), sym))
    y = np.concatenate(Y); h = np.vstack(H)
    X = pd.concat(X, ignore_index=True); syms = np.concatenate(S)

    tr = np.zeros(len(y), bool)
    for s in set(syms):
        m = np.flatnonzero(syms == s)
        tr[m[:int(len(m) * 0.7)]] = True
    te = ~tr
    print(f"{len(set(syms))} ETFs, {len(y):,} observations, {horizon}-minute horizon\n")

    A = np.column_stack([np.ones(tr.sum()), h[tr]])
    coef, *_ = np.linalg.lstsq(A, y[tr], rcond=None)
    har_all = np.column_stack([np.ones(len(y)), h]) @ coef

    sets = {
        "HAR only": None,
        "HAR + time of day": ["bar_of_day"],
        "HAR + this bar's range": ["h_0", "l_0", "o_0"],
        "HAR + volume shape": [c for c in X.columns if c.startswith("v_")],
        "everything": list(X.columns),
    }
    print(f"{'features given to the model':<30}{'pooled R2':>11}{'within-symbol R2':>19}")
    har_pool = r2(y[te], har_all[te])
    per = [r2(y[te][syms[te] == s], har_all[te][syms[te] == s]) for s in sorted(set(syms))]
    print(f"{'HAR only':<30}{har_pool:>11.3f}{np.mean(per):>19.3f}")

    base = har_all.reshape(-1, 1)
    for name, cols in sets.items():
        if cols is None:
            continue
        Xs = np.column_stack([base, X[cols].to_numpy(dtype="float32")]).astype("float32")
        pooled, pred = fit_r2(Xs[tr], y[tr], Xs[te], y[te])
        within = np.mean([r2(y[te][syms[te] == s], pred[syms[te] == s])
                          for s in sorted(set(syms))])
        print(f"{name:<30}{pooled:>11.3f}{within:>19.3f}")

    # Vol points, with the annualisation actually correct.
    Xall = np.column_stack([base, X.to_numpy(dtype="float32")]).astype("float32")
    _, pred = fit_r2(Xall[tr], y[tr], Xall[te], y[te])
    ann = np.sqrt(MIN_PER_YEAR)
    lvl = float(np.exp(np.mean(y[te])) * ann)
    for name, p in (("HAR", har_all[te]), ("model", pred)):
        rmse = float(np.sqrt(np.mean((y[te] - p) ** 2)))
        print(f"\n{name:>6}: typical error {np.expm1(rmse):.1%} of the level"
              f"  =  {np.expm1(rmse) * lvl * 100:.1f} vol points at {lvl:.0%} annualised")
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
