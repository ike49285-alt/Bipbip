"""The control the panel result never had.

The 20-ETF panel carries always-long and coin-flip benchmarks, and the model
beat always-long by +1.63 bps at paired t=2.22. It has never been run against
shuffled labels. That control was done on TQQQ alone and mattered less then;
with the rebalancing mechanism falsified there is no longer a reason the effect
should exist, so the null is carrying weight it was not carrying before.

THE SHUFFLE PRESERVES EVERYTHING EXCEPT THE ANSWER. Labels are permuted by
TIMESTAMP, with the same permutation applied to every symbol, so each moment's
cross-section stays intact - the relative outcomes across the twenty funds at a
given instant are exactly as they were, and only their attachment to the
features is destroyed. A row-wise shuffle would also break the cross-sectional
correlation, which is part of what the model is being tested on, and would make
the null easier to beat than it should be.

The number to watch is model-minus-always-long. If the null reproduces +1.63,
that margin was an artefact of the procedure rather than anything in the data.

RESULT: it survives, and the nulls are not zero.

       run     model   always long   margin   paired t
      REAL    +1.79b        +0.16b   +1.63b       2.22
    null 1    +1.67b        +0.58b   +1.10b       1.77
    null 2    +1.39b        +0.21b   +1.19b       1.94
    null 3    +1.21b        +0.51b   +0.70b       1.09
    null 4    +0.96b        +0.82b   +0.14b       0.21
    null 5    +1.42b        +0.57b   +0.85b       1.30

The real margin beats all five, so the effect is separable. But shuffled labels
carrying no information still produce +0.79 bps of margin on average, and null 2
reaches +1.19 at t=1.94 - nearly as convincing as the real run.

That is structural rather than luck. Taking max(long prediction, short
prediction) and scoring it against long-only is biased by construction: the
better of two noisy estimates beats either one even when both are noise. The
comparison was built that way and the bias was invisible until the null exposed
it.

So the margin decomposes into roughly +0.79 of procedure and +0.84 of signal.
The finding is about half the size it appeared. It still passes - seven controls
now - but +0.84 bps gross makes execution decisive rather than merely important:
crossing the spread costs four times the entire edge.
"""
import sys, pathlib, time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np, pandas as pd

from bipbip.ml.discover import make_gbm_regressor
from scripts.barrier_panel import BASKET, CROSS_BPS, load


def evaluate(df, cols, tr, L, S, H, sym, seed=0, shuffle=False):
    y_l, y_s = L.copy(), S.copy()
    if shuffle:
        rng = np.random.default_rng(seed)
        dates = np.array(sorted(df["_date"].unique()))
        perm = dict(zip(dates, rng.permutation(dates)))
        # Move every symbol's label to the same permuted timestamp, so the
        # cross-section at each moment travels together and only the link to
        # the features is cut.
        key = pd.MultiIndex.from_arrays([sym, df["_date"].to_numpy()])
        newkey = pd.MultiIndex.from_arrays(
            [sym, df["_date"].map(perm).to_numpy()])
        pos = pd.Series(np.arange(len(df)), index=key)
        take = pos.reindex(newkey).to_numpy()
        ok = np.isfinite(take)
        take = np.where(ok, take, np.arange(len(df))).astype(int)
        y_l, y_s = L[take], S[take]

    Xv = df[cols].to_numpy(dtype="float32")
    ml = make_gbm_regressor(seed=seed).fit(Xv[tr], y_l[tr])
    ms = make_gbm_regressor(seed=seed).fit(Xv[tr], y_s[tr])
    picks = []
    for s in sorted(set(sym)):
        idx = np.flatnonzero((sym == s) & ~tr)
        cur = 0
        while cur < len(idx):
            picks.append(idx[cur])
            cur += max(1, int(H[idx[cur]]))
    te = np.array(picks)
    long_side = ml.predict(Xv[te]) >= ms.predict(Xv[te])
    model = np.where(long_side, y_l[te], y_s[te])
    always = y_l[te]
    return model, always


def main():
    t0 = time.time()
    tm, sm, hold = 1.5, 1.0, 26
    frames = [f for f in (load(s, tm, sm, hold) for s in BASKET) if f is not None]
    df = pd.concat(frames, ignore_index=True)
    cols = [c for c in df.columns if not c.startswith("_")]
    dates = np.array(sorted(df["_date"].unique()))
    cut = dates[int(len(dates) * 0.7)]
    tr = (df["_date"] < cut).to_numpy()
    L, S, H = df["_L"].to_numpy(), df["_S"].to_numpy(), df["_H"].to_numpy()
    sym = df["_sym"].to_numpy()
    print(f"{df['_sym'].nunique()} ETFs, {len(df):,} bars, "
          f"test after {pd.Timestamp(cut).date()}\n")
    print(f"{'run':>10} {'trades':>8} {'model':>9} {'always long':>13} "
          f"{'model - long':>14} {'paired t':>10}")

    m, a = evaluate(df, cols, tr, L, S, H, sym)
    d = m - a
    t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    print(f"{'REAL':>10} {len(m):>8,} {m.mean():>+8.2f}b {a.mean():>+12.2f}b "
          f"{d.mean():>+13.2f}b {t:>10.2f}")

    diffs = []
    for s in range(1, 6):
        mn, an = evaluate(df, cols, tr, L, S, H, sym, seed=s, shuffle=True)
        dn = mn - an
        tn = dn.mean() / (dn.std(ddof=1) / np.sqrt(len(dn)))
        diffs.append(dn.mean())
        print(f"{'null ' + str(s):>10} {len(mn):>8,} {mn.mean():>+8.2f}b "
              f"{an.mean():>+12.2f}b {dn.mean():>+13.2f}b {tn:>10.2f}")

    print(f"\nreal margin {d.mean():+.2f} bps against nulls "
          f"mean {np.mean(diffs):+.2f}, best {max(diffs):+.2f}")
    beat = sum(d.mean() > x for x in diffs)
    print("VERDICT:", "survives the null" if beat == len(diffs) and t > 2
          else "NOT separable from the procedure itself")
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
