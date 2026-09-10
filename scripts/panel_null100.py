"""How big is the null, really? A hundred shuffles instead of five.

The five-shuffle run established that the null is NOT zero: the scoring rule -
max(long prediction, short prediction) scored against long-only - pays about
+0.79 bps on labels carrying no information. That reframed the finding from
+1.63 bps of signal to roughly +0.84. But it left the harder question open,
because five draws cannot locate a distribution. Two limits bit:

  - A rank-based permutation p-value over five nulls cannot go below 1/6.
  - Against the correct null of +0.79 the paired t falls from 2.22 to about
    1.14, which is inside the noise.

So the effect was neither confirmed nor refuted, only underpowered. A hundred
shuffles pins the null's centre and spread and gives a permutation p-value with
resolution to 1/101.

TWO FIXES TO THE COMPARISON ITSELF. The original evaluate() passed one seed to
both the label permutation and the GBM, so every null differed from the real
run in two ways at once and the null spread carried model-seed noise the single
real point did not. Here the model seed is FIXED across all runs and only the
permutation varies, which is what a permutation test compares. Model-seed
sensitivity is then measured separately, on the real labels, so it is visible
rather than silently inflating the null.

SUPERSEDED - READ THIS FIRST. The p=0.069 below was computed against a
CONTAMINATED null. The permutation looked each (symbol, new timestamp) pair up
in an index, and the twenty funds do not share every timestamp, so 23.4% of
lookups missed and silently fell back to the row's OWN label. Nearly a quarter
of every "null" run was real signal. That inflates the null and makes the real
run look LESS exceptional than it is, so the verdict below is biased against
the finding. The corrected run uses --method common, which restricts the panel
to the 91.2% of rows on timestamps every symbol shares and raises rather than
falls back. Its numbers, and the real margin on that grid, supersede everything
in this block.

RESULT (against the contaminated null): p = 0.069.

                    margin bps
              REAL      +1.63b
         null mean      +0.33b
           null sd        0.83
          null p95      +1.71b
          null max      +2.77b

    signal above the null   +1.30 bps
    nulls at or above real  6 of 100
    permutation p           0.0693
    z against null spread   1.57

The real run does not reach the null's OWN 95th percentile of +1.71. Six
shuffles carrying no information at all beat it outright, and the best reached
+2.77 - seventy percent larger than the finding.

WHAT FIVE DRAWS GOT WRONG, AND IT WAS NOT THE CENTRE. The five-null run put the
null at +0.79; the truth is +0.33, so the procedure bias was OVERSTATED and the
surviving signal is +1.30 rather than +0.84. That part of the earlier correction
was too harsh. But five draws estimated the null's spread at sd 0.415 when the
truth is 0.829 - exactly half - and the width is what decides significance.
Resampling five draws from these hundred, 20,000 times: the null mean lands
anywhere in [-0.37, +1.04] and the sd anywhere in [0.29, 1.40]. Five draws
cannot measure either.

The verdict is the part that should worry anyone reading an old result here.
Drawing five nulls returns "survives the null" - all five below the real run -
SEVENTY-TWO PERCENT of the time on data whose true p-value is 0.069. The
five-shuffle control was not weak evidence of an edge. It was close to no
evidence at all, and it read as confirmation.

Direction is right and the point estimate is positive, so this is not a
refutation. It is a finding that does not clear the bar, on a repo whose stated
prior is that the default hypothesis is zero. Two further reasons to discount
rather than round up: the barrier configuration was chosen from a 32-cell grid,
where the best-of-N floor asks for t=2.63, and model seed alone moves the real
margin by sd 0.204.
"""
import os
# Set before sklearn's OpenMP runtime initialises: 4 single-threaded workers
# beat one 4-threaded fit, because GBM thread scaling is well short of linear.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys, pathlib, time, argparse, csv

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np, pandas as pd

from bipbip.ml.discover import make_gbm_regressor
from scripts.barrier_panel import BASKET, CROSS_BPS, load

G = {}


def shuffled_labels(df, L, S, sym, perm_seed, method="rotate"):
    """Permute labels by TIMESTAMP, the same permutation for every symbol.

    "common" is the correct one, and it requires the panel to have been
    restricted to timestamps present for EVERY symbol first. Then each moment's
    whole cross-section moves to the same new moment, every lookup resolves,
    and a miss is an error rather than a silent fallback.

    "permute" reproduces the ORIGINAL, BROKEN null for comparison. On the full
    panel the twenty funds do not share every timestamp, so 23.4% of lookups
    missed and those rows fell back to their OWN row - keeping their TRUE
    label. Nearly a quarter of every "null" run was real signal, which inflates
    the null and makes the real run look LESS exceptional than it is. The bug
    is silent: the fallback is a plausible-looking np.where.
    """
    rng = np.random.default_rng(perm_seed)
    dates = np.array(sorted(df["_date"].unique()))
    perm = dict(zip(dates, rng.permutation(dates)))
    key = pd.MultiIndex.from_arrays([sym, df["_date"].to_numpy()])
    newkey = pd.MultiIndex.from_arrays([sym, df["_date"].map(perm).to_numpy()])
    pos = pd.Series(np.arange(len(df)), index=key)
    take = pos.reindex(newkey).to_numpy()
    missing = int((~np.isfinite(take)).sum())
    if method == "common":
        # On the common grid every (symbol, permuted timestamp) pair exists, so
        # a miss means the caller did not restrict the panel. Refuse rather than
        # fall back - the fallback is what broke the original null.
        if missing:
            raise AssertionError(
                f"{missing:,} lookups missed on a grid that should be complete")
    else:
        take = np.where(np.isfinite(take), take, np.arange(len(df)))
    return L[take.astype(int)], S[take.astype(int)]


def evaluate(perm_seed=None, model_seed=0, method="rotate"):
    """Return (margin_bps, paired_t, n_trades) for one run."""
    df, Xv, tr, te, L, S, sym = (G["df"], G["Xv"], G["tr"], G["te"],
                                 G["L"], G["S"], G["sym"])
    y_l, y_s = (shuffled_labels(df, L, S, sym, perm_seed, method)
                if perm_seed is not None else (L, S))

    ml = make_gbm_regressor(seed=model_seed).fit(Xv[tr], y_l[tr])
    ms = make_gbm_regressor(seed=model_seed).fit(Xv[tr], y_s[tr])
    long_side = ml.predict(Xv[te]) >= ms.predict(Xv[te])
    d = np.where(long_side, y_l[te], y_s[te]) - y_l[te]
    t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    return float(d.mean()), float(t), len(d)


def _work(args):
    perm_seed, model_seed, method = args
    m, t, n = evaluate(perm_seed, model_seed, method)
    return perm_seed, model_seed, m, t, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--out", default="data/panel_null100.csv")
    ap.add_argument("--method", default="common",
                    choices=["common", "permute"])
    a = ap.parse_args()

    t0 = time.time()
    tm, sm, hold = 1.5, 1.0, 26
    frames = [f for f in (load(s, tm, sm, hold) for s in BASKET) if f is not None]
    df = pd.concat(frames, ignore_index=True)
    if a.method == "common":
        # Keep only moments the whole panel shares, so the permutation is total.
        n0 = len(df)
        cnt = df.groupby("_date")["_sym"].nunique()
        df = df[df["_date"].isin(cnt.index[cnt == df["_sym"].nunique()])]
        df = df.reset_index(drop=True)
        print(f"common grid: {len(df):,} of {n0:,} rows "
              f"({len(df)/n0*100:.1f}%) on timestamps every symbol shares")
    cols = [c for c in df.columns if not c.startswith("_")]
    dates = np.array(sorted(df["_date"].unique()))
    cut = dates[int(len(dates) * 0.7)]
    tr = (df["_date"] < cut).to_numpy()
    L, S, H = df["_L"].to_numpy(), df["_S"].to_numpy(), df["_H"].to_numpy()
    sym = df["_sym"].to_numpy()

    # Non-overlapping test sample: step by the REALISED hold, so no bar is
    # counted twice. Fixed once here rather than per run - it depends only on
    # the holds, which the shuffle does not touch.
    picks = []
    for s in sorted(set(sym)):
        idx = np.flatnonzero((sym == s) & ~tr)
        cur = 0
        while cur < len(idx):
            picks.append(idx[cur])
            cur += max(1, int(H[idx[cur]]))
    te = np.array(picks)

    G.update(df=df, Xv=df[cols].to_numpy(dtype="float32"), tr=tr, te=te,
             L=L, S=S, sym=sym)
    print(f"null method: {a.method}")
    print(f"{df['_sym'].nunique()} ETFs, {len(df):,} bars, {len(cols)} features, "
          f"test after {pd.Timestamp(cut).date()}, {len(te):,} trades")
    print(f"load {time.time()-t0:.0f}s\n", flush=True)

    real, real_t, n = evaluate(None, 0)
    print(f"REAL margin {real:+.3f} bps, paired t={real_t:.2f}, {n:,} trades\n",
          flush=True)

    # Model-seed sensitivity on the REAL labels, so this source of variation is
    # visible instead of hiding inside the null spread.
    seeds = [evaluate(None, s)[0] for s in range(1, 5)]
    allseeds = [real] + seeds
    print(f"real across 5 model seeds: "
          f"{', '.join(f'{x:+.2f}' for x in allseeds)}  "
          f"(sd {np.std(allseeds, ddof=1):.3f})\n", flush=True)

    import multiprocessing as mp
    jobs = [(s, 0, a.method) for s in range(1, a.runs + 1)]
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    nulls = []
    with out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["perm_seed", "model_seed", "margin_bps", "paired_t", "trades"])
        w.writerow(["", 0, f"{real:.6f}", f"{real_t:.6f}", n])  # blank = real
        with mp.get_context("fork").Pool(a.jobs) as pool:
            for i, (ps, ms_, m, t, nn) in enumerate(
                    pool.imap_unordered(_work, jobs), 1):
                nulls.append(m)
                w.writerow([ps, ms_, f"{m:.6f}", f"{t:.6f}", nn])
                fh.flush()
                if i % 10 == 0 or i == len(jobs):
                    el = time.time() - t0
                    print(f"  {i:>3}/{len(jobs)} nulls  "
                          f"mean {np.mean(nulls):+.3f}  sd {np.std(nulls, ddof=1):.3f}  "
                          f"max {max(nulls):+.3f}  [{el/60:.1f}m, "
                          f"eta {el/i*(len(jobs)-i)/60:.1f}m]", flush=True)

    nulls = np.array(nulls)
    ge = int((nulls >= real).sum())
    # The +1 is Phipson-Smyth: the real run is itself one draw under H0, so a
    # p-value of exactly zero is not attainable and claiming it is overstates.
    p = (ge + 1) / (len(nulls) + 1)
    z = (real - nulls.mean()) / nulls.std(ddof=1)

    print(f"\n{'':>22}{'margin bps':>12}")
    print(f"{'REAL':>22}{real:>+11.2f}b")
    print(f"{'null mean':>22}{nulls.mean():>+11.2f}b")
    print(f"{'null sd':>22}{nulls.std(ddof=1):>12.2f}")
    for q in (50, 90, 95, 99):
        print(f"{'null p' + str(q):>22}{np.percentile(nulls, q):>+11.2f}b")
    print(f"{'null max':>22}{nulls.max():>+11.2f}b")
    print(f"\nsignal above the null   {real - nulls.mean():+.2f} bps")
    print(f"nulls at or above real  {ge} of {len(nulls)}")
    print(f"permutation p           {p:.4f}")
    print(f"z against null spread   {z:.2f}")
    print("\nVERDICT:", "separable from the procedure (p<0.05)" if p < 0.05
          else "NOT separable from the procedure at p<0.05")
    print(f"\ntotal {(time.time()-t0)/60:.1f}m -> {a.out}")


if __name__ == "__main__":
    main()
