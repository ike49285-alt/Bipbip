"""Throw the genetic search at the minute archive.

Every rule the GA can express is one a trader could read aloud: a few threshold
conditions on features, a direction, a holding period. What matters is not the
winner's fitness but the gap between it and what the identical search achieves
on destroyed labels - on synthetic noise with only eight features, the search
alone reached t=10.5.
"""
import sys, pathlib, glob, os, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np, pandas as pd

from bipbip.core import indicators as ind
from bipbip.data.sessions import restrict_to_rth
from bipbip.data.store import BarStore
from bipbip.ml.genetic import RuleSearch, evolve_with_null

HOLDS = (5, 15, 30, 60)
SEC_FEE_BPS = 0.278


def spread_bps(close):
    d = close.diff().dropna()
    if len(d) < 100:
        return np.nan
    cov = float(np.cov(d.values[1:], d.values[:-1])[0, 1])
    return 2.0 * np.sqrt(-cov) / float(close.mean()) * 1e4 if cov < 0 else np.nan


def build(limit=200):
    store = BarStore("data/bars")
    feats, fwds, syms_out = [], {h: [] for h in HOLDS}, []
    costs_out = []
    files = sorted(glob.glob("data/bars/*_1m.parquet"))[:limit]
    for f in files:
        sym = os.path.basename(f).split("_")[0]
        b = restrict_to_rth(store.load(sym, "1m")).dropna()
        if len(b) < 3000:
            continue
        s = spread_bps(b["close"])
        cost = max(s if np.isfinite(s) else 20.0,
                   0.01 / float(b["close"].iloc[-1]) * 1e4) + SEC_FEE_BPS

        o, c, h, l = b["open"], b["close"], b["high"], b["low"]
        day = pd.Series(b.index.normalize(), index=b.index)
        lr = np.log(c / c.shift(1))
        ha = ind.heikin_ashi(b)
        st = ind.full_stochastic(b, 14, 1, 3)
        ich = ind.ichimoku(b, 9, 26, 52, 26)

        col = {
            "ret_5": np.log(c / c.shift(5)), "ret_15": np.log(c / c.shift(15)),
            "ret_30": np.log(c / c.shift(30)), "ret_60": np.log(c / c.shift(60)),
            "vol_30": lr.rolling(30, min_periods=30).std(),
            "vol_ratio": (lr.rolling(15, min_periods=15).std()
                          / lr.rolling(60, min_periods=60).std()),
            "range_pos": (c - l) / (h - l).replace(0, np.nan),
            "vol_rel": b["volume"] / b["volume"].rolling(60, min_periods=60).mean(),
            "dist_sma30": c / c.rolling(30, min_periods=30).mean() - 1.0,
            "ha_trend": ha["ha_trend"], "ha_run": ha["ha_run"],
            "ha_body": ha["ha_body_frac"],
            "stoch_k": st["stoch_k"], "stoch_kd": st["stoch_k"] - st["stoch_d"],
            "ich_cloud": (c - ich["cloud_top"]) / c,
            "ich_tk": (ich["tenkan"] - ich["kijun"]) / c,
            "minute_of_day": pd.Series(
                (b.index.hour - 9) * 60 + b.index.minute - 30, index=b.index),
        }
        F = pd.DataFrame(col)
        for hh in HOLDS:
            # GROSS only. Cost is applied by the search after direction, since
            # netting it here makes a short earn the spread instead of paying.
            fwd = (c.shift(-hh) / o.shift(-1) - 1.0) * 1e4
            fwd = fwd.where(day.shift(-hh) == day)
            fwds[hh].append(fwd.to_numpy(dtype="float64"))
        costs_out.append(np.full(len(b), cost, dtype="float64"))
        feats.append(F.to_numpy(dtype="float32"))
        syms_out.append(sym)

    X = np.vstack(feats)
    fwd_by_hold = {h: np.concatenate(v) for h, v in fwds.items()}
    return X, fwd_by_hold, list(col), syms_out, np.concatenate(costs_out)


def main():
    t0 = time.time()
    X, fwd, names, syms, cost = build()
    keep = np.isfinite(X).any(axis=1)
    X = X[keep]
    fwd = {h: v[keep] for h, v in fwd.items()}
    cost = cost[keep]
    print(f"{len(syms)} symbols, {len(X):,} samples, {len(names)} features "
          f"({time.time()-t0:.0f}s)")
    print(f"gross returns; each symbol's measured spread charged after direction\n")

    search = RuleSearch(X, fwd, names, min_trades=500, seed=7,
                        max_fraction=0.25, cost_bps=cost)
    res = evolve_with_null(search, population=60, generations=25,
                           null_runs=10, seed=7)

    print("Best rule found:")
    print(f"  {res.best.describe(names)}")
    print(f"  fires {res.trades:,} times ({res.trades/len(X):.1%} of the tape), "
          f"{res.net_bps:+.2f} bps net per trade, "
          f"t={res.t_stat:.2f}\n")
    print("What the SAME search achieves on destroyed labels:")
    print(f"  null mean t={res.null_mean:.2f}   null best t={res.null_best:.2f}")
    print(f"  p = {res.p_value:.3f}  ({len(res.history)} generations)\n")
    # The p-value floor is 1/(null_runs+1), so demanding p<=0.05 with ten runs
    # asks for something arithmetically unreachable - and the last run printed
    # "not separable" for a winner that beat all ten nulls by a factor of six.
    # Report the separation directly and say what the floor is.
    floor = 1.0 / (res.p_value and (round(1.0 / res.p_value)) or 1)
    if not np.isfinite(res.null_best):
        print("VERDICT: no usable null runs; the result means nothing.")
    elif res.fitness <= res.null_best:
        print("VERDICT: the winner is inside the noise the search itself")
        print("generates. Nothing found.")
    else:
        ratio = res.fitness / res.null_best if res.null_best > 0 else float("inf")
        print(f"VERDICT: winner beat all {len(res.history) and ''}null runs, "
              f"t={res.fitness:.1f} against a null best of {res.null_best:.1f} "
              f"({ratio:.1f}x).")
        print(f"  p={res.p_value:.3f} is the FLOOR for this many null runs, not")
        print("  a weak result. Separation is the number that matters.")
        print("  This is NOT a green light: 20 sessions, and every apparent")
        print("  edge in this project so far has been an accounting artefact.")
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
