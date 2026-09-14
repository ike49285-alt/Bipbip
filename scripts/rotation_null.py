"""Does ranking on price AND volume pick winners, or is it the search again?

The idea: hold what is running, fund it by selling what is flagging, rotate as
leadership changes. The price-only form of this is already dead here three ways
- rule-based momentum on the clean 34-ETF universe returns 8.82%/8.93% against
buy-and-hold SPY's 10.84% and turnover is not the reason (cutting it fivefold
moves 8.82% to 8.64%), and a decile sort on ~300 symbols found the ordering runs
BACKWARDS, the weakest trending decile beating the strongest by 53-105 bps a
month at every formation window.

The reason it nevertheless feels easy is worth stating: run the identical code
on the survivorship-contaminated stock list and momentum reads 26.90%, appearing
to nearly triple SPY. Only the ETF universes are clean enough to SELECT within.

What was never measured is the model form. build_cross_sectional +
evaluate_ranker ask exactly the right question - will this symbol beat the
cross-sectional median - and no result for them exists anywhere in this repo.
Both earlier runs were retracted, one for a panel bug that made hourly data
secretly daily and one for a t inflated by sqrt(horizon). Their volume input was
a single spike detector until this session.

FOUR SPOILERS, all live in the existing runners, all fixed here.

1. OVERLAP. evaluate_ranker(rebalance_every=None) is the default and
   tests/test_discover.py pins it. Holding 70 bars while rebalancing every bar
   makes consecutive observations share 69 of 70 - the same trade counted
   seventy times. Reported t walked 1.60 -> 5.55 -> 10.78 across horizons
   21/35/70, almost exactly sqrt(horizon). Here rebalance_every == horizon
   always, and it is printed.

2. UNDERPOWERED NULL. Every runner uses n=15. CLAUDE.md: five nulls cannot
   measure a spread, and on data whose true p is 0.069 five draws return
   "survives" 72% of the time. 100 here, with a Phipson-Smyth p.

3. SURVIVORSHIP. discover_run.py defaults to the `full` universe, 312 symbols,
   flagged SEVERE. A strategy that SELECTS among names cannot run there at all:
   ranking a 2026 membership list is partly ranking "did this survive to 2026".

4. BEST-OF-N LEAK, the subtle one. discover_run.py picks best_k by t across
   {3,5,10,20} and then runs its null AT THAT WINNER. The real number is a
   best-of-four and the null is not, so the comparison is biased by exactly the
   thing CLAUDE.md warns about - the scoring rule being the edge. Here the null
   repeats the WHOLE procedure, grid and argmax included.

The model does not depend on top_k - only the basket does - so each fold is fit
ONCE and every k is scored from the same probabilities. That is 4x faster and
identical by construction; tests/test_rotation.py checks the equivalence.

The bar is not "positive". It is: permutation p < 0.05, positive after costs,
beating EQUAL WEIGHT (which beat every picker in the README), and clearing the
best-of-N floor sqrt(2 ln N) for the grid actually searched.
"""
import os
# Before sklearn's OpenMP runtime initialises: single-threaded workers beat one
# multi-threaded fit, because GBM thread scaling is well short of linear.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys, pathlib, time, argparse, csv

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np, pandas as pd

from bipbip.core.costs import CostModel
from bipbip.core.panel import load_panel
from bipbip.data.store import BarStore
from bipbip.data.universe import UNIVERSES, get_universe
from bipbip.ml.discover import (CrossSectionalDataset, build_cross_sectional,
                                make_gbm)

G = {}

#: Basket sizes searched. Capped at 10 because the ranker needs at least twice
#: the basket quoting before a pick means anything, and this universe holds 34 -
#: so k=20 is silently skipped at every timestamp rather than evaluated, and
#: half the universe is not a "pick" in any case. The best-of-N floor below is
#: computed from the cells that actually produced a score, not from this tuple.
TOP_KS = (3, 5, 10)

#: Features built from volume. The ablation drops these and nothing else, so
#: the volume contribution is a measured delta rather than an assumption.
VOLUME_FEATURES = ("volume_ratio", "volume_trend", "dollar_volume",
                   "dollar_volume_ratio", "volume_price_corr",
                   "up_volume_share")


def price_only(ds: CrossSectionalDataset) -> CrossSectionalDataset:
    """The same dataset with every volume feature and its rank removed."""
    drop = {c for c in ds.X.columns
            if c in VOLUME_FEATURES or c[3:] in VOLUME_FEATURES}
    keep = [c for c in ds.X.columns if c not in drop]
    if len(keep) == len(ds.X.columns):
        raise AssertionError("ablation removed nothing; feature names changed")
    return CrossSectionalDataset(
        X=ds.X[keep], y=ds.y, excess=ds.excess, fwd=ds.fwd,
        median_fwd=ds.median_fwd, symbols=ds.symbols, dates=ds.dates,
        bar_index=ds.bar_index, feature_names=keep)


def procedure(ds, excess, cost_bps, rebalance_every, model_seed=0,
              n_splits=5, embargo_days=5, min_train=2000):
    """The WHOLE selection procedure: walk forward, score every k, take the best.

    This is the unit the null has to repeat. Returning only the winner's score
    while shuffling inside it would compare a best-of-four against a single
    draw, which is how a scoring rule becomes an edge.

    `excess` is passed separately from `ds` so a permutation can replace the
    labels without rebuilding the frame.
    """
    dates = ds.dates
    uniq = np.array(sorted(set(dates)))
    n_ts = len(uniq)
    if n_ts < n_splits * 20:
        return {"note": f"only {n_ts} timestamps"}

    Xv = ds.X.to_numpy(dtype="float32")
    y = (excess > 0).astype("float64")
    fold_size = n_ts // (n_splits + 1)
    picked = {k: [] for k in TOP_KS}

    for f in range(n_splits):
        train_end = uniq[fold_size * (f + 1)]
        val_start = train_end + pd.Timedelta(days=embargo_days)
        val_end = uniq[min(fold_size * (f + 2), n_ts - 1)]
        if val_start >= val_end:
            continue
        tr = np.flatnonzero(dates <= train_end)
        va = np.flatnonzero((dates > val_start) & (dates <= val_end))
        if len(tr) < min_train or len(va) < 200 or len(np.unique(y[tr])) < 2:
            continue

        # One fit serves every k: the model ranks, k only says how deep to go.
        model = make_gbm(seed=model_seed).fit(Xv[tr], y[tr])
        proba = model.predict_proba(Xv[va])[:, 1]
        vdates, vexcess = dates[va], excess[va]

        stamps = np.unique(vdates)
        if rebalance_every and rebalance_every > 1:
            stamps = stamps[::rebalance_every]
        for ts in stamps:
            m = vdates == ts
            p, e = proba[m], vexcess[m]
            order = np.argsort(-p)
            for k in TOP_KS:
                if m.sum() < k * 2:
                    continue
                picked[k].append(np.nanmean(e[order[:k]]))

    out = {}
    for k, vals in picked.items():
        v = np.asarray(vals, dtype="float64")
        v = v[np.isfinite(v)] - cost_bps / 1e4
        if len(v) < 20:
            continue
        t = v.mean() / v.std() * np.sqrt(len(v)) if v.std() > 0 else float("nan")
        out[k] = {"excess_bps": float(v.mean() * 1e4), "t_stat": float(t),
                  "rebalances": int(len(v))}
    if not out:
        return {"note": "no usable folds"}

    best = max(out, key=lambda k: out[k]["t_stat"])
    return {"by_k": out, "best_k": best, **out[best]}


def _shuffle_within_timestamp(excess, dates, seed):
    """Destroy the ranking inside each moment, keep the moment intact.

    A global shuffle would break the cross-section as well as the signal and
    score the model against a world where the median itself means nothing.
    """
    rng = np.random.default_rng(seed)
    out = excess.copy()
    for ts in np.unique(dates):
        m = np.flatnonzero(dates == ts)
        out[m] = out[rng.permutation(m)]
    moved = float((out != excess).mean())
    return out, moved


def _work(args):
    seed, cost, reb = args
    ds = G["ds"]
    shuffled, moved = _shuffle_within_timestamp(ds.excess, ds.dates, seed)
    r = procedure(ds, shuffled, cost, reb, model_seed=0)
    return seed, r.get("excess_bps", float("nan")), r.get("best_k", -1), moved


def equal_weight_benchmark(ds, rebalance_every):
    """Excess of holding EVERYTHING against the cross-sectional median.

    NOT zero, and the gap matters. `excess` is measured against the MEDIAN, so
    holding all of it earns the mean-minus-median spread of a right-skewed
    cross-section - about +13 bps a month on this universe. A random picker
    collects the same thing, which is why the null centres near it rather than
    on zero, and it is the reason a positive `excess_bps` is not by itself a
    finding. README found equal weight beat every strategy that tried to pick,
    so this is the bar, not zero and not SPY.
    """
    stamps = np.unique(ds.dates)
    if rebalance_every and rebalance_every > 1:
        stamps = stamps[::rebalance_every]
    vals = [np.nanmean(ds.excess[ds.dates == ts]) for ts in stamps]
    v = np.asarray(vals, dtype="float64")
    return float(np.nanmean(v[np.isfinite(v)]) * 1e4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="etf_wide")
    ap.add_argument("--bars", default="1d")
    ap.add_argument("--horizon", type=int, default=21)
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--ablation", default="both",
                    choices=["both", "volume", "price"])
    ap.add_argument("--out", default="data/rotation_null.csv")
    a = ap.parse_args()

    surv = UNIVERSES.get(a.universe, {}).get("survivorship", "?")
    if str(surv).upper() == "SEVERE":
        raise SystemExit(
            f"{a.universe} is flagged survivorship={surv}. Ranking a present-day "
            f"membership list is partly ranking 'did this survive', which is "
            f"worth 15-21 points a year here - larger than any effect measured "
            f"in this project. Use an etf_* universe.")

    t0 = time.time()
    panel = load_panel(BarStore("data/bars"), get_universe(a.universe), a.bars)
    ds_full = build_cross_sectional(panel, horizon=a.horizon, min_symbols=20)
    rt = CostModel().round_trip_cost_bps("SPY", price=100.0, shares=1.0)
    cost = rt * 0.8                       # replacing ~80% of the basket

    print(f"universe {a.universe} ({surv} survivorship), {a.bars} bars, "
          f"horizon {a.horizon}")
    print(f"panel: {len(panel.symbols)} symbols x {len(panel):,} bars")
    print(f"dataset: {ds_full.summary()}")
    print(f"cost {cost:.2f} bps per rebalance; REBALANCE_EVERY = {a.horizon} "
          f"(one observation per holding period)")
    eq = equal_weight_benchmark(ds_full, a.horizon)
    print(f"equal weight (the bar): {eq:+.2f} bps - this is the mean-minus-median "
          f"spread, which a random picker also collects\n", flush=True)

    arms = {"volume": ds_full} if a.ablation == "volume" else \
           {"price": price_only(ds_full)} if a.ablation == "price" else \
           {"price": price_only(ds_full), "volume": ds_full}

    # Write to a TEMPORARY file and rename only on success. Opening the real
    # path in "w" truncates the committed result the instant this starts, so an
    # interrupted run - a timeout, Ctrl-C, an OOM kill - leaves a partial file
    # where the evidence used to be. That happened: a 90-second cap during an
    # audit sweep cut this file from 203 rows to 40, and git was the only thing
    # that noticed. A null that takes minutes to compute should not be able to
    # destroy its own record by being stopped.
    final = pathlib.Path(a.out)
    final.parent.mkdir(parents=True, exist_ok=True)
    out = final.with_name(final.name + ".partial")
    with out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "perm_seed", "excess_bps", "best_k", "moved_share",
                    "horizon", "universe"])

        for arm, ds in arms.items():
            G["ds"] = ds
            print(f"=== {arm.upper()} arm: {len(ds.X.columns)} features ===",
                  flush=True)
            real = procedure(ds, ds.excess, cost, a.horizon, model_seed=0)
            if "by_k" not in real:
                print(f"  {real.get('note')}"); continue
            for k in sorted(real["by_k"]):
                r = real["by_k"][k]
                star = "  <- best" if k == real["best_k"] else ""
                print(f"  top_k={k:>2}  {r['excess_bps']:>+8.2f} bps  "
                      f"t={r['t_stat']:>5.2f}  {r['rebalances']:>4} rebalances{star}")
            print(f"  REAL (procedure = best of {len(TOP_KS)}): "
                  f"{real['excess_bps']:+.2f} bps at top_k={real['best_k']}\n",
                  flush=True)
            w.writerow([arm, "", f"{real['excess_bps']:.6f}", real["best_k"],
                        "", a.horizon, a.universe])

            seeds = [(s, cost, a.horizon) for s in range(1, a.runs + 1)]
            nulls = []
            import multiprocessing as mp
            with mp.get_context("fork").Pool(a.jobs) as pool:
                for i, (s, m, bk, moved) in enumerate(
                        pool.imap_unordered(_work, seeds), 1):
                    if np.isfinite(m):
                        nulls.append(m)
                    w.writerow([arm, s, f"{m:.6f}", bk, f"{moved:.4f}",
                                a.horizon, a.universe])
                    fh.flush()
                    if i % 10 == 0 or i == len(seeds):
                        el = time.time() - t0
                        print(f"  {i:>3}/{len(seeds)} nulls  "
                              f"mean {np.mean(nulls):+.2f}  "
                              f"sd {np.std(nulls, ddof=1):.2f}  "
                              f"max {max(nulls):+.2f}  "
                              f"[{el/60:.1f}m]", flush=True)

            nulls = np.asarray(nulls)
            ge = int((nulls >= real["excess_bps"]).sum())
            # Phipson-Smyth: the real run is itself a draw under H0, so p=0 is
            # not attainable and claiming it overstates.
            p = (ge + 1) / (len(nulls) + 1)
            n_cells = len(real["by_k"])          # cells that actually scored
            floor = float(np.sqrt(2 * np.log(n_cells))) if n_cells > 1 else 0.0
            print(f"\n  {'REAL':>22}{real['excess_bps']:>+10.2f}b")
            print(f"  {'null mean':>22}{nulls.mean():>+10.2f}b")
            print(f"  {'null sd':>22}{nulls.std(ddof=1):>11.2f}")
            for q in (50, 90, 95):
                print(f"  {'null p'+str(q):>22}{np.percentile(nulls, q):>+10.2f}b")
            print(f"  {'null max':>22}{nulls.max():>+10.2f}b")
            print(f"  {'nulls >= real':>22}{ge:>7} of {len(nulls)}")
            print(f"  {'permutation p':>22}{p:>11.4f}")
            print(f"  {'best-of-N floor':>22}{floor:>11.2f}  "
                  f"(t for the {n_cells}-cell grid actually searched)")
            print(f"  {'equal weight':>22}{eq:>+10.2f}b  "
                  f"(the bar; a picker must beat this, not zero)")
            print(f"  VERDICT: {'separable (p<0.05)' if p < 0.05 else 'NOT separable'}\n",
                  flush=True)

    # The run completed, so the temporary file becomes the record. os.replace
    # is atomic on the same filesystem: there is no instant at which the
    # committed result is half-written.
    os.replace(out, final)
    print(f"total {(time.time()-t0)/60:.1f}m -> {a.out}")


if __name__ == "__main__":
    main()
