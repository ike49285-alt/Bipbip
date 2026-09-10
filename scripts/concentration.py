"""Does the edge live in every bar, or in a few?

Every result in this project takes a position on every bar. The +1.01 bps
measured at thirty minutes is an AVERAGE over 38,436 trades, which is
consistent with two very different worlds: a sliver of edge smeared evenly
across all of them, or a real edge in a handful diluted by noise in the rest.
They imply opposite behaviour. If it is the second, trading 5% of the time at
20 bps beats trading always at 1, and the cost hurdle stops being the binding
constraint.

The threshold is chosen on the TRAINING fold and applied to validation. Picking
"top 5%" after seeing the validation results is not selectivity, it is another
search with another null floor, and it would report the luckiest cut as a
finding.

Costs are shown both ways for the same reason as elsewhere: crossing the spread
charges 3.10 bps, earning it credits 2.54, and which world we are in is still
being measured.
"""
import sys, pathlib, time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

from bipbip.core.indicators import assert_causal
from bipbip.data.store import BarStore
from bipbip.ml.discover import make_gbm
from scripts.everything import CROSS_BPS, EARN_BPS, features, forward

KEEP = (1.00, 0.50, 0.20, 0.10, 0.05, 0.02, 0.01)


def run(X, y, h, folds=6, seed=0, shuffle=False):
    """Return, per selectivity level, the trades that survived the cut."""
    rng = np.random.default_rng(seed)
    yy = rng.permutation(y) if shuffle else y
    Xv, n = X.to_numpy(dtype="float32"), len(X)
    edge = n // (folds + 1)
    buckets = {k: [] for k in KEEP}
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
        # Confidence cut-offs come from the TRAINING predictions. Taking them
        # from validation would choose the threshold using the answers.
        conf_tr = np.abs(m.predict_proba(Xv[tr])[:, 1] - 0.5)
        va = np.arange(lo2, hi2, h)
        va = va[np.isfinite(yy[va])]
        if not len(va):
            continue
        p = m.predict_proba(Xv[va])[:, 1]
        conf = np.abs(p - 0.5)
        pnl = np.where(p > .5, 1., -1.) * yy[va] * 1e4
        for frac in KEEP:
            cut = np.quantile(conf_tr, 1.0 - frac) if frac < 1.0 else -np.inf
            sel = conf >= cut
            if sel.sum():
                buckets[frac].append(pnl[sel])
    return {k: (np.concatenate(v) if v else np.array([])) for k, v in buckets.items()}


def t(a):
    a = np.asarray(a)
    return a.mean() / a.std(ddof=1) * np.sqrt(len(a)) if len(a) > 1 and a.std() else np.nan


def main():
    t0 = time.time()
    b = BarStore("data/bars").load("TQQQ", "30m").dropna()
    X = assert_causal(features, b)
    h = 1
    fwd = forward(b, h)
    ok = np.isfinite(fwd.to_numpy()) & X.notna().all(axis=1).to_numpy()
    Xi, yi = X[ok], fwd[ok].to_numpy()
    print(f"TQQQ 30m, {len(Xi):,} usable bars, {X.shape[1]} features\n")

    real = run(Xi, yi, h)
    nulls = [run(Xi, yi, h, seed=s, shuffle=True) for s in range(1, 4)]
    print(f"{'trade':>7} {'trades':>7} {'GROSS':>9} {'t':>6} {'win':>6} "
          f"{'crossing':>9} {'earning':>9}   {'null gross':>11}")
    for frac in KEEP:
        a = real[frac]
        if len(a) < 30:
            print(f"{frac:>6.0%} (too few)")
            continue
        nm = [n[frac].mean() for n in nulls if len(n[frac]) > 30]
        print(f"{frac:>6.0%} {len(a):>7,} {a.mean():>+8.2f}b {t(a):>6.2f} "
              f"{(a > 0).mean():>6.1%} {a.mean()-CROSS_BPS:>+8.2f}b "
              f"{a.mean()-EARN_BPS:>+8.2f}b   "
              f"{np.mean(nm) if nm else float('nan'):>+10.2f}b")
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
