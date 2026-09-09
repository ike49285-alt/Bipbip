"""Not "how volatile", but "is it about to get MORE volatile than it has been".

Predicting the level of volatility is largely predicting which ticker it is -
that trap has eaten three results in this session. Predicting the CHANGE cannot
be won that way: the question is whether the next H days realise more than the
last H did, for this name, and every symbol's own level cancels out of it.

It is also the question an option buyer actually asks. A contract is priced off
prevailing volatility, so buying pays when volatility expands and selling pays
when it contracts. The level is already in the premium; the change is not.

Two baselines, because the naive one is not the honest one. The base rate is
whatever fraction of windows expand at all. But volatility mean-reverts, so a
strategy of always predicting the opposite of the current deviation from average
is a real forecaster and much harder to beat, and it is the second column.
"""
import sys, pathlib, glob, os, time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np, pandas as pd

from bipbip.ml.discover import make_gbm_regressor
from scripts.vol_horizon import CUT, MAX_SYMBOLS, build


def main():
    t0 = time.time()
    horizon = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    files = sorted(glob.glob("data/bars/*_1d.parquet"))[:MAX_SYMBOLS]
    frames = [d for d in (build(f, horizon) for f in files) if d is not None]
    df = pd.concat(frames, ignore_index=True)

    trail = "rv22" if horizon >= 20 else "rv5"
    # The target: how much bigger (or smaller) the coming window is than the one
    # just past, in logs, for this symbol. A symbol's own volatility level
    # cancels, so this cannot be won by recognising the ticker.
    df["chg"] = df["y"] - np.log(df[trail])
    # Where this symbol's current volatility sits against its own long history.
    # Mean reversion alone predicts the change from this, so it is the benchmark
    # a model has to beat rather than the coin flip.
    df["stretch"] = np.log(df[trail]) - (
        df.groupby("sym")["y"].transform(lambda s: s.expanding().mean().shift(1)))

    tr = (df["date"] < CUT).to_numpy()
    te = np.zeros(len(df), bool)
    for _, g in df.groupby("sym", sort=False):
        idx = g.index.to_numpy()[(g["date"] >= CUT).to_numpy()]
        te[idx[::horizon]] = True
    ok = np.isfinite(df["chg"]) & np.isfinite(df["stretch"])
    tr, te = tr & ok, te & ok

    feats = ["rv5", "rv22", "rv66", "park", "park_ratio", "ret22", "ret5",
             "down_share", "jump", "vol_of_vol", "vol_trend", "stretch"]
    X = df[feats].to_numpy().astype("float32")
    y = df["chg"].to_numpy()

    m = make_gbm_regressor(seed=0).fit(X[tr], y[tr])
    p = m.predict(X[te])
    yt = y[te]
    ann = np.sqrt(252)
    lvl = float(np.exp(np.mean(df["y"].to_numpy()[te])) * ann)

    print(f"{df['sym'].nunique()} symbols, {horizon}-day windows, "
          f"{int(te.sum()):,} non-overlapping tests after {CUT.date()}")
    print(f"median annualised volatility {lvl:.0%}\n")
    print(f"windows that expanded: {np.mean(yt > 0):.1%}   "
          f"mean-reversion benchmark correlation: "
          f"{np.corrcoef(-df['stretch'].to_numpy()[te], yt)[0, 1]:+.3f}")
    print(f"model correlation with the actual change: "
          f"{np.corrcoef(p, yt)[0, 1]:+.3f}\n")

    print("Sorting windows by predicted change, then measuring what happened:")
    print(f"{'predicted':>12} {'n':>7} {'expanded':>10} {'actual change':>15} "
          f"{'in vol points':>15}")
    q = pd.qcut(p, 5, labels=False, duplicates="drop")
    for b in range(int(q.max()) + 1):
        s = q == b
        chg = float(np.mean(yt[s]))
        # exp(chg) - 1 is the proportional move in volatility; times the level
        # puts it in the points an option is quoted in.
        print(f"{['lowest','low','mid','high','highest'][b]:>12} {s.sum():>7,} "
              f"{np.mean(yt[s] > 0):>9.1%} {np.expm1(chg):>+14.1%} "
              f"{np.expm1(chg) * lvl * 100:>+14.1f}")

    top, bot = q == q.max(), q == 0
    spread = np.expm1(np.mean(yt[top])) - np.expm1(np.mean(yt[bot]))
    se = np.sqrt(np.var(yt[top], ddof=1) / top.sum()
                 + np.var(yt[bot], ddof=1) / bot.sum())
    print(f"\ntop minus bottom fifth: {spread * lvl * 100:+.1f} vol points "
          f"(t={np.mean(yt[top]) - np.mean(yt[bot]):.3f}/{se:.3f} = "
          f"{(np.mean(yt[top]) - np.mean(yt[bot])) / se:.1f})")
    print(f"for comparison, the premium measured on the TQQQ chain was ~5 vol points")
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
