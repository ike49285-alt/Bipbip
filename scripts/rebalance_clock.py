"""Does the edge sit at the close, where levered funds actually rebalance?

The panel located the edge in levered products - +2.22 bps over always-long on
the levered longs, +4.05 on the inverse funds, -0.13 on index and sector ETFs.
The candidate mechanism is the one structural feature those products have and
the others do not: a levered fund must reset its exposure at every close, buying
when it is up and selling when it is down, and that flow is predictable.

The mechanism makes a prediction that can kill it. If this is rebalancing flow,
the edge belongs in the last bars of the session. Spread evenly across the day
it is something else, and the mechanism is wrong.

The comparison is a difference across two dimensions at once, because either
alone is confoundable. Volatility is U-shaped, so any strategy earns more per
trade near the close whether or not it knows anything - hence edge is measured
OVER ALWAYS-LONG within each bucket, not as raw P&L. And index and sector ETFs
run as the control arm: they share the clock and the volatility smile but have
no rebalancing to exploit, so a clock effect that appears in both is the
session, while one that appears only in levered products is the mechanism.
"""
import sys, pathlib, time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np, pandas as pd

from bipbip.core.indicators import atr
from bipbip.data.store import BarStore
from bipbip.ml.barriers import barrier_labels
from bipbip.ml.discover import make_gbm_regressor
from scripts.everything import features
from scripts.barrier_panel import BASKET, INVERSE, LEVERED_LONG

HOLD = 26


def load(sym):
    try:
        b = BarStore("data/bars").load(sym, "30m").dropna()
    except Exception:
        return None
    if len(b) < 5000:
        return None
    X = features(b)
    lab = barrier_labels(b, atr(b, 14), 1.5, 1.0, HOLD, cost_bps=0.0)
    ok = (X.notna().all(axis=1).to_numpy()
          & np.isfinite(lab["long_ret"].to_numpy())
          & np.isfinite(lab["short_ret"].to_numpy()))
    if ok.sum() < 500:
        return None
    out = X[ok].copy()
    out["_L"] = lab["long_ret"].to_numpy()[ok]
    out["_S"] = lab["short_ret"].to_numpy()[ok]
    out["_H"] = lab["long_held"].to_numpy()[ok]
    out["_sym"] = sym
    out["_date"] = b.index[ok]
    out["_slot"] = b.index[ok].strftime("%H:%M")
    return out


def main():
    t0 = time.time()
    frames = [f for f in (load(s) for s in BASKET) if f is not None]
    df = pd.concat(frames, ignore_index=True)
    cols = [c for c in df.columns if not c.startswith("_")]
    dates = np.array(sorted(df["_date"].unique()))
    cut = dates[int(len(dates) * 0.7)]
    tr = (df["_date"] < cut).to_numpy()
    Xv = df[cols].to_numpy(dtype="float32")
    L, S, H = df["_L"].to_numpy(), df["_S"].to_numpy(), df["_H"].to_numpy()
    sym, slot = df["_sym"].to_numpy(), df["_slot"].to_numpy()

    ml = make_gbm_regressor(seed=0).fit(Xv[tr], L[tr])
    ms = make_gbm_regressor(seed=0).fit(Xv[tr], S[tr])

    picks = []
    for s in sorted(set(sym)):
        idx = np.flatnonzero((sym == s) & ~tr)
        cur = 0
        while cur < len(idx):
            picks.append(idx[cur])
            cur += max(1, int(H[idx[cur]]))
    te = np.array(picks)
    long_side = ml.predict(Xv[te]) >= ms.predict(Xv[te])
    model = np.where(long_side, L[te], S[te])
    always = L[te]
    edge = model - always

    levered = np.isin(sym[te], list(LEVERED_LONG | INVERSE))
    print(f"{df['_sym'].nunique()} ETFs, {len(te):,} held-out trades after "
          f"{pd.Timestamp(cut).date()}")
    print(f"levered {levered.sum():,}, index+sector {(~levered).sum():,}\n")
    print("edge over always-long, by the bar the trade was ENTERED on\n")
    print(f"{'entry':>7} {'LEVERED':>22}   {'INDEX + SECTOR':>22}")
    print(f"{'':>7} {'n':>7}{'edge':>8}{'+/-95':>7}   {'n':>7}{'edge':>8}{'+/-95':>7}")

    for sl in sorted(set(slot[te])):
        line = f"{sl:>7}"
        for grp in (levered, ~levered):
            m = (slot[te] == sl) & grp
            if m.sum() < 60:
                line += f" {m.sum():>6}{'--':>8}{'':>7}  "
                continue
            e = edge[m]
            se = 1.96 * e.std(ddof=1) / np.sqrt(len(e))
            line += f" {len(e):>6}{e.mean():>+8.2f}{se:>7.2f}  "
        print(line)

    # Pooled from the raw observations rather than from the per-slot means
    # printed above: averaging slot means would weight a thin slot equally with
    # a fat one, and would not give a usable two-sample standard error.
    for name, grp in (("LEVERED", levered), ("index+sector", ~levered)):
        t_mask = (slot[te] >= "14:30") & grp
        m_mask = (slot[te] >= "10:30") & (slot[te] < "14:30") & grp
        if t_mask.sum() < 60 or m_mask.sum() < 60:
            continue
        a, b = edge[t_mask], edge[m_mask]
        se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        print(f"\n  {name:<14} last 90 min {a.mean():+.2f}  midday {b.mean():+.2f}  "
              f"difference {a.mean()-b.mean():+.2f} +/-{1.96*se:.2f}")
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
