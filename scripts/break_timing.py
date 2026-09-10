"""When does the next big move come, and how big - beyond what volatility says?

Every search here has fixed the horizon in advance and asked what happens over
it. None has asked when the move arrives. That is a different question and the
barrier machinery already computes the answer: every trade records how many bars
until the target or stop was reached, and that column has never been used as a
label.

A "break" is defined without direction, because direction is dead: an excursion
of k times ATR either way within N bars. So this asks whether something is about
to happen, not what.

THE CONTROL IS THE WHOLE TEST. High volatility trivially predicts large moves,
so the benchmark is not the base rate - it is a model given ONLY volatility
features. Beating the base rate would just rediscover that volatile things move,
which is the trap that made an intraday volatility result look like skill
yesterday when 87% of it turned out to be the time-of-day smile. Anything above
the volatility-only model is genuine information about timing.

Two questions, separately:
  1. WILL it break within N bars - measured by AUC against the vol-only model.
  2. WHEN, given that it does - correlation of predicted with actual bars-to-
     break, again against the vol-only model.

RESULT: nothing, once the clock is in the baseline.

     k     base  AUC vol   +clock     full  lift    r clock  r full   lift
   1.0    69.6%    0.672    0.787    0.791  +0.004    0.411   0.418  +0.008
   1.5    43.5%    0.662    0.761    0.763  +0.002    0.436   0.440  +0.004
   2.0    26.0%    0.662    0.756    0.758  +0.002    0.476   0.479  +0.004
   3.0     9.4%    0.666    0.768    0.770  +0.002    0.568   0.568  -0.000

Volatility alone reaches 0.662. Adding TWO features - time of day and day of
week - reaches 0.787. Adding the other eighty-eight adds 0.002. Timing behaves
the same way: the clock explains r=0.41 to 0.57 of bars-to-break and the rest of
the feature set adds 0.004.

The reason is mechanical. A 09:30 bar has twelve bars of room before the close
and a 15:00 bar has two, so "breaks within n bars" is mostly a question about
how much session remains. Against a volatility-only baseline that looked like a
+0.10 AUC discovery.

The k=3.0 row is the one that settles it. Rare large breaks cannot be explained
by session structure the way small ones can, so a genuine effect should survive
there - and the lift is -0.000, exactly where it had to hold.

Also fixed here: the first version swept n=13 and n=26 and reported both. They
were identical to three decimals, because thirty-minute bars give thirteen per
session and the session cap binds first - the window parameter was inert and the
sweep silently tested three configurations while printing four.
"""
import sys, pathlib, time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

from bipbip.core.indicators import atr
from bipbip.data.store import BarStore
from bipbip.ml.discover import make_gbm, make_gbm_regressor
from scripts.everything import features
from scripts.barrier_panel import BASKET

VOL_ONLY = ["atr_pct", "rv_13", "rv_65", "rv_260", "rv_ratio", "park",
            "park_ratio", "rvol"]
# Bars remaining in the session decide the answer mechanically: a 09:30 bar has
# twelve bars of room before the close and a 15:00 bar has two, so "breaks
# within n bars" is largely a question about the clock. The full feature set
# carries bar_of_day and the volatility-only set does not, so without this arm
# the whole lift could be the model reading the time.
VOL_CLOCK = VOL_ONLY + ["bar_of_day", "dow"]


def break_labels(b, a, k, n):
    """Bars until price travels k*ATR in EITHER direction, and whether within n."""
    c = b["close"].to_numpy()
    hi, lo = b["high"].to_numpy(), b["low"].to_numpy()
    step = k * a.to_numpy()
    up, dn = c + step, c - step
    N = len(c)
    sess = pd.DatetimeIndex(b.index).normalize().view("int64")
    when = np.full(N, n + 1, np.int32)
    for j in range(1, n + 1):
        idx = np.minimum(np.arange(N) + j, N - 1)
        ok = (np.arange(N) + j < N) & (sess[idx] == sess) & (when > n)
        hit = ok & ((hi[idx] >= up) | (lo[idx] <= dn))
        when = np.where(hit, j, when)
    # A bar with no room left in its session cannot be labelled either way.
    room = np.zeros(N, bool)
    for j in range(1, n + 1):
        idx = np.minimum(np.arange(N) + j, N - 1)
        room |= (np.arange(N) + j < N) & (sess[idx] == sess)
    return when, (when <= n), room


def load(sym, k, n):
    try:
        b = BarStore("data/bars").load(sym, "30m").dropna()
    except Exception:
        return None
    if len(b) < 5000:
        return None
    a = atr(b, 14)
    X = features(b)
    when, broke, room = break_labels(b, a, k, n)
    ok = X.notna().all(axis=1).to_numpy() & np.isfinite(a.to_numpy()) & room
    if ok.sum() < 500:
        return None
    out = X[ok].copy()
    out["_when"], out["_broke"] = when[ok], broke[ok]
    out["_date"], out["_sym"] = b.index[ok], sym
    return out


def main():
    t0 = time.time()
    print(f"{'k x ATR':>8} {'window':>7} {'base':>10} {'AUC vol':>8} "
          f"{'+clock':>9} {'full':>8} {'lift/clock':>9}   "
          f"{'r clock':>8} {'r full':>7} {'lift':>8}")
    # n>=13 is inert at 30-minute bars: the session cap binds first.
    for k, n in ((1.0, 13), (1.5, 13), (2.0, 13), (3.0, 13)):
        frames = [f for f in (load(s, k, n) for s in BASKET) if f is not None]
        df = pd.concat(frames, ignore_index=True)
        cols = [c for c in df.columns if not c.startswith("_")]
        dates = np.array(sorted(df["_date"].unique()))
        cut = dates[int(len(dates) * 0.7)]
        tr = (df["_date"] < cut).to_numpy()
        te = ~tr
        y = df["_broke"].to_numpy().astype(float)
        Xf = df[cols].to_numpy(dtype="float32")
        Xv = df[VOL_ONLY].to_numpy(dtype="float32")

        Xc = df[VOL_CLOCK].to_numpy(dtype="float32")
        auc = {}
        for name, X in (("vol", Xv), ("clock", Xc), ("full", Xf)):
            m = make_gbm(seed=0).fit(X[tr], y[tr])
            auc[name] = roc_auc_score(y[te], m.predict_proba(X[te])[:, 1])

        # Timing, among the ones that actually broke.
        w = df["_when"].to_numpy().astype(float)
        broke = df["_broke"].to_numpy()
        rr = {}
        for name, X in (("vol", Xv), ("clock", Xc), ("full", Xf)):
            trb, teb = tr & broke, te & broke
            if trb.sum() < 2000 or teb.sum() < 500:
                rr[name] = np.nan
                continue
            m = make_gbm_regressor(seed=0).fit(X[trb], np.log(w[trb]))
            p = m.predict(X[teb])
            rr[name] = np.corrcoef(p, np.log(w[teb]))[0, 1]
        print(f"{k:>8.1f} {n:>6}b {y[te].mean():>9.1%} "
              f"{auc['vol']:>8.3f} {auc['clock']:>9.3f} {auc['full']:>8.3f} "
              f"{auc['full']-auc['clock']:>+9.3f}   "
              f"{rr['clock']:>8.3f} {rr['full']:>7.3f} "
              f"{rr['full']-rr['clock']:>+8.3f}")
    print("\nThe column that matters is lift over the +clock arm: what the full")
    print("feature set adds beyond knowing the volatility AND the time of day.")
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
