"""Does the barrier model's edge survive a basket that did not all triple?

On TQQQ alone the result could not be separated from being long: +3.66 bps
against always-long's +1.16, a paired difference of +2.50 at t=1.17. TQQQ rose
enormously across the sample, so drift is a live explanation for most of it.

This basket removes that explanation by construction. It holds five INVERSE
levered funds - SQQQ, SOXS, TZA, SPXS, SPXU - which fell over the same period,
alongside low-drift sector funds. A strategy that wins by being quietly long
cannot win here; always-long and always-short should roughly cancel across the
panel, leaving whatever the model actually knows.

The honest limit going in: the minute archive covers ninety days, so resampled
to thirty minutes each symbol contributes about 910 bars. Pooled that is
adequate; per symbol it is thin, and the per-symbol figures below are
directional rather than conclusive.
"""
import sys, pathlib, time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np, pandas as pd

from bipbip.core.indicators import atr
from bipbip.data.store import BarStore
from bipbip.ml.barriers import barrier_labels
from bipbip.ml.discover import make_gbm_regressor
from scripts.everything import CROSS_BPS, features

BASKET = ["TQQQ", "SQQQ", "SPY", "QQQ", "IWM", "DIA", "SOXL", "SOXS", "TNA",
          "TZA", "SPXL", "SPXS", "UPRO", "SPXU", "LABU", "XLF", "XLE", "XLK",
          "XLV", "XLU"]
INVERSE = {"SQQQ", "SOXS", "TZA", "SPXS", "SPXU"}
LEVERED_LONG = {"TQQQ", "SOXL", "TNA", "SPXL", "UPRO", "LABU"}


def load(sym, tm, sm, hold):
    st = BarStore("data/bars")
    try:
        # NATIVE 30-minute bars now, reconciled against each symbol's own daily
        # record - not minute bars resampled over ninety days. Roughly 37,000
        # bars per ETF spanning 2015-2026 against the proxy's 500.
        b = st.load(sym, "30m").dropna()
    except Exception:
        return None
    if len(b) < 5000:
        return None
    X = features(b)
    lab = barrier_labels(b, atr(b, 14), target_mult=tm, stop_mult=sm,
                         max_hold=hold, cost_bps=0.0)
    ok = (X.notna().all(axis=1).to_numpy()
          & np.isfinite(lab["long_ret"].to_numpy())
          & np.isfinite(lab["short_ret"].to_numpy()))
    if ok.sum() < 200:
        return None
    out = X[ok].copy()
    out["_L"] = lab["long_ret"].to_numpy()[ok]
    out["_S"] = lab["short_ret"].to_numpy()[ok]
    out["_H"] = lab["long_held"].to_numpy()[ok]
    out["_sym"] = sym
    out["_date"] = b.index[ok]
    return out


def arm_stats(name, r):
    t = r.mean() / r.std(ddof=1) * np.sqrt(len(r)) if len(r) > 1 and r.std() else np.nan
    return (f"  {name:<14}{r.mean():>+8.2f}b{t:>7.2f}{len(r):>8,}"
            f"{r.mean()-CROSS_BPS:>+13.2f}b")


def main():
    t0 = time.time()
    tm, sm, hold = 1.5, 1.0, 26          # the BLIND pick, not the swept winner
    frames = [f for f in (load(s, tm, sm, hold) for s in BASKET) if f is not None]
    df = pd.concat(frames, ignore_index=True)
    feat_cols = [c for c in df.columns if not c.startswith("_")]
    dates = np.array(sorted(df["_date"].unique()))
    cut = dates[int(len(dates) * 0.7)]
    tr = (df["_date"] < cut).to_numpy()
    print(f"{df['_sym'].nunique()} ETFs, {len(df):,} bars, "
          f"{len(dates):,} timestamps, barriers {tm}/{sm}/{hold}")
    print(f"train before {pd.Timestamp(cut).date()}, test after "
          f"({int((~tr).sum()):,} bars)\n")

    Xv = df[feat_cols].to_numpy(dtype="float32")
    L, S, H = df["_L"].to_numpy(), df["_S"].to_numpy(), df["_H"].to_numpy()
    sym = df["_sym"].to_numpy()

    ml = make_gbm_regressor(seed=0).fit(Xv[tr], L[tr])
    ms = make_gbm_regressor(seed=0).fit(Xv[tr], S[tr])

    # Non-overlapping within each symbol, stepping by the realised hold.
    picks = []
    for s in sorted(set(sym)):
        idx = np.flatnonzero((sym == s) & ~tr)
        if len(idx) < 30:
            continue
        cur, end = 0, len(idx) - 1
        while cur <= end:
            picks.append(idx[cur])
            cur += max(1, int(H[idx[cur]]))
    te = np.array(picks)
    pick_long = ml.predict(Xv[te]) >= ms.predict(Xv[te])
    rng = np.random.default_rng(0)

    arms = {
        "model": np.where(pick_long, L[te], S[te]),
        "always long": L[te],
        "always short": S[te],
        "coin flip": np.where(rng.random(len(te)) < .5, L[te], S[te]),
    }
    print(f"  {'arm':<14}{'gross':>9}{'t':>7}{'trades':>8}{'net crossing':>14}")
    for k, v in arms.items():
        print(arm_stats(k, v))
    d = arms["model"] - arms["always long"]
    se = d.std(ddof=1) / np.sqrt(len(d))
    print(f"  model minus always-long: {d.mean():+.2f} bps +/-{1.96*se:.2f} "
          f"(paired t={d.mean()/se:.2f})")
    print(f"  went short on {np.mean(~pick_long):.1%} of trades\n")

    print(f"  {'group':<16}{'model':>9}{'long':>9}{'short':>9}{'edge vs long':>14}")
    groups = {"levered long": LEVERED_LONG, "INVERSE (fell)": INVERSE,
              "index + sector": set(BASKET) - LEVERED_LONG - INVERSE}
    for gname, members in groups.items():
        m = np.isin(sym[te], list(members))
        if m.sum() < 30:
            continue
        mm, ll, ss = arms["model"][m], arms["always long"][m], arms["always short"][m]
        print(f"  {gname:<16}{mm.mean():>+8.2f}b{ll.mean():>+8.2f}b"
              f"{ss.mean():>+8.2f}b{(mm-ll).mean():>+13.2f}b")
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
