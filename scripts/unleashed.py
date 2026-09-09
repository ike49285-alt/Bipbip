"""Every symbol, every minute, one objective: be profitable.

No primary rule, no hand-picked universe, no relative benchmark. The label is
simply whether the trade made money after costs, and the model gets every
feature the project has - including the Heikin-Ashi, fast stochastic and
Ichimoku readings a discretionary trader would use.

The one thing that is NOT relaxed is measurement. Costs are the spread measured
from each symbol's own bars rather than assumed, positions are entered at the
next bar's open, and folds are chronological with an embargo. Loosening those
would not find an edge, it would only stop the search being able to tell.
"""
import sys, pathlib, glob, os, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np, pandas as pd

from bipbip.core.panel import build_panel
from bipbip.data.sessions import restrict_to_rth
from bipbip.data.store import BarStore
from bipbip.ml.discover import _causal_features, make_gbm

SEC_FEE_BPS = 0.278


def roll_spread_bps(close: pd.Series) -> float:
    d = close.diff().dropna()
    if len(d) < 100:
        return float("nan")
    cov = float(np.cov(d.values[1:], d.values[:-1])[0, 1])
    if cov >= 0:
        return float("nan")
    return 2.0 * np.sqrt(-cov) / float(close.mean()) * 1e4


def load_everything():
    store = BarStore("data/bars")
    bars, costs = {}, {}
    for f in sorted(glob.glob("data/bars/*_1m.parquet")):
        sym = os.path.basename(f).split("_")[0]
        try:
            b = restrict_to_rth(store.load(sym, "1m")).dropna()
        except Exception:
            continue
        if len(b) < 2000:
            continue
        s = roll_spread_bps(b["close"])
        if not np.isfinite(s):
            continue
        bars[sym] = b
        costs[sym] = max(s, 0.01 / float(b["close"].iloc[-1]) * 1e4) + SEC_FEE_BPS
    return build_panel(bars), costs


def build(panel, costs, horizon):
    """Label: did this trade make money after ITS OWN symbol's spread?"""
    feats = _causal_features(panel)
    names = sorted(feats)
    o, c = panel.opens, panel.closes
    day = pd.Series(panel.dates.normalize(), index=panel.dates)

    entry = o.shift(-1)
    exit_ = c.shift(-horizon)
    fwd = exit_ / entry - 1.0
    same_day = (day.shift(-horizon) == day).to_numpy()
    fwd = fwd.where(pd.Series(same_day, index=panel.dates), np.nan)

    cost = pd.Series({s: costs.get(s, 10.0) / 1e4 for s in panel.symbols})
    net = fwd.sub(cost, axis=1)

    arrs = {n: feats[n].to_numpy(dtype="float32") for n in names}
    net_a = net.to_numpy(dtype="float64")
    ok = np.isfinite(net_a)
    rows_i, rows_j = np.nonzero(ok)
    if len(rows_i) == 0:
        raise ValueError("no labelled rows")

    X = np.column_stack([arrs[n][rows_i, rows_j] for n in names])
    keep = np.isfinite(X).any(axis=1)
    return (pd.DataFrame(X[keep], columns=names),
            net_a[rows_i, rows_j][keep],
            panel.dates[rows_i[keep]],
            np.array(panel.symbols)[rows_j[keep]])


def run(panel, costs, horizon, label, top_k=20):
    X, net, dates, syms = build(panel, costs, horizon)
    y = (net > 0).astype("float64")
    uniq = np.array(sorted(set(dates)))
    n_ts = len(uniq)
    fold = n_ts // 5
    Xv = X.to_numpy(dtype="float32")

    picks, fold_bps = [], []
    for k in range(4):
        tr_end = uniq[fold * (k + 1)]
        va_lo = tr_end + pd.Timedelta(minutes=horizon + 30)
        va_hi = uniq[min(fold * (k + 2), n_ts - 1)]
        tr = np.flatnonzero(dates <= tr_end)
        va = np.flatnonzero((dates > va_lo) & (dates <= va_hi))
        if len(tr) < 5000 or len(va) < 1000 or len(np.unique(y[tr])) < 2:
            continue
        m = make_gbm().fit(Xv[tr], y[tr])
        p = m.predict_proba(Xv[va])[:, 1]
        vd, vn = dates[va], net[va]
        got = []
        for ts in np.unique(vd)[::horizon]:
            sel = vd == ts
            if sel.sum() < top_k * 2:
                continue
            top = np.argsort(-p[sel])[:top_k]
            got.append(np.nanmean(vn[sel][top]))
        if got:
            picks.extend(got)
            fold_bps.append(np.mean(got) * 1e4)

    if not picks:
        print(f"{label:>10} (no usable folds)")
        return
    a = np.asarray(picks)
    t = a.mean() / a.std() * np.sqrt(len(a)) if a.std() > 0 else float("nan")
    print(f"{label:>10} {len(X):>10,} {a.mean()*1e4:>+9.2f}b {t:>7.2f} "
          f"{(a > 0).mean():>6.0%} {len(a):>6,}  "
          + " ".join(f"{f:+.0f}" for f in fold_bps))


def main():
    t0 = time.time()
    panel, costs = load_everything()
    print(f"{len(panel.symbols)} symbols, {len(panel):,} minute bars, "
          f"{panel.dates[0].date()} -> {panel.dates[-1].date()}")
    print(f"median measured round trip: "
          f"{np.median(list(costs.values())):.2f} bps  ({time.time()-t0:.0f}s)\n")
    print("Label: did the trade make money after its own symbol's spread?")
    print("No primary rule. No universe filter. Top 20 by predicted P&L.\n")
    print(f"{'horizon':>10} {'rows':>10} {'net P&L':>10} {'t':>7} {'win':>6} "
          f"{'trades':>6}  folds")
    for h, lab in ((5, "5 min"), (15, "15 min"), (30, "30 min"),
                   (60, "1 hour"), (120, "2 hours")):
        try:
            run(panel, costs, h, lab)
        except Exception as exc:
            print(f"{lab:>10} failed: {exc}")
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
