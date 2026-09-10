"""Evolve a READABLE expression that captures what the black box found.

The earlier symbolic run searched for alpha from scratch and returned noise -
in-sample t=1.50 against a best-of-N floor of 4.05, with shuffled labels beating
it. Running the same search harder is not the fix. Three things changed instead.

The LABEL. That run optimised fixed-horizon returns, the win condition since
shown to be broken: it scores a trade that runs +40 bps by bar five and gives it
back as a loss. Fitness here is barrier-resolved P&L, target-stop-or-clock,
which is the objective that took the gradient-boosted model from +1.73 to +3.66.

The UNIVERSE. The panel test located the edge in levered products - +2.22 bps
over always-long on the levered longs and +4.05 on the inverse funds, against
-0.13 on index and sector ETFs. Searching the dead 40% only adds noise.

The OPERATORS. cs_rank, cs_demean and cs_z let an expression say "this fund
versus the others right now", which nothing in the previous search could
express. That is the structure the panel found, and its absence may be why the
search came back empty rather than the absence of anything to find.

The point is not another alpha. There is already a model making +1.79 bps and it
is ninety features of black box. An expression a person can read is checkable by
eye, testable against the rebalancing hypothesis, and tradeable by hand.

RESULT: it still returns noise, and the way it fails is the useful part.

    best  ((cs_z((cs_z(delta(close, 3)) / close)) / high) / open)
          in-sample t=2.86 against a best-of-N floor of 4.02
          held out  +8.45 bps, t=0.80, win 49.9%
    nulls in-sample 2.87 / 0.86 / 3.59, held out 1.06 / 0.24 / -0.67

Below the floor is worse than keeping the luckiest of 3,200 coin flips, null 3
beat it in-sample on meaningless labels, and a 49.9% win rate under a +8.45 mean
is a few large outcomes rather than an edge. The winner also fails the
readability bar that motivated the rerun.

The informative part: the cross-sectional operators WERE used, by the winner and
by two of three nulls. So this is not the language being unable to express
cross-sectional structure - it can, and the structure is still not findable as a
compact formula.

Against the gradient-boosted model making +1.79 bps at t=2.22 over always-long
on the same labels, universe and data, that isolates what is doing the work:
ninety weak features combined additively succeed where one compact expression
cannot. The edge is DIFFUSE, not concentrated.

Three independent tests now agree on that shape. Selectivity found it flat from
100% of bars down to the top 1% rather than lumped in the confident trades. The
barrier grid was smooth across all 32 cells rather than spiking at one. And it
cannot be compressed. The practical consequence is that no hand-tradeable
version exists; if the effect is real it needs the model.
"""
import sys, pathlib, time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np, pandas as pd

from bipbip.core.indicators import atr
from bipbip.data.store import BarStore
from bipbip.ml.barriers import barrier_labels
from bipbip.ml.symbolic import (AlphaSearch, Node, clone, crossover, evaluate,
                                mutate, random_tree)

LEVERED = ["TQQQ", "SOXL", "TNA", "SPXL", "UPRO", "LABU",
           "SQQQ", "SOXS", "TZA", "SPXS", "SPXU"]
CROSS_BPS = 3.10


class BarrierSearch(AlphaSearch):
    """Fitness is the barrier book's t-statistic, not a forward-return book."""

    def __init__(self, panel, L, S, k=2, seed=0, max_size=10):
        self.data, self.k, self.max_size = panel, k, max_size
        self.rng = np.random.default_rng(seed)
        self.L, self.S = L, S
        self.horizon = 1

    def book(self, values, rows):
        out = []
        for t in rows:
            v = values[t]
            ok = np.isfinite(v) & np.isfinite(self.L[t]) & np.isfinite(self.S[t])
            if ok.sum() < 2 * self.k:
                continue
            idx = np.flatnonzero(ok)
            order = idx[np.argsort(v[idx])]
            # Long the top of the ranking through its LONG barrier, short the
            # bottom through its SHORT barrier - a short is not a mirrored long.
            out.append(np.mean(self.L[t][order[-self.k:]])
                       - -np.mean(self.S[t][order[:self.k]]))
        return np.asarray(out)

    def fitness(self, node, rows):
        if node.size() > self.max_size:
            return -np.inf
        try:
            v = evaluate(node, self.data)
        except (ValueError, FloatingPointError):
            return -np.inf
        if not np.isfinite(v).any():
            return -np.inf
        if np.nanmean(np.nanstd(v, axis=1)) < 1e-12:
            return -np.inf
        p = self.book(v, rows)
        if len(p) < 50 or p.std(ddof=1) == 0:
            return -np.inf
        t = p.mean() / p.std(ddof=1) * np.sqrt(len(p))
        return float(t - 0.05 * node.size())


def load(hold=26):
    st = BarStore("data/bars")
    frames = {}
    for s in LEVERED:
        try:
            b = st.load(s, "30m").dropna()
        except Exception:
            continue
        if len(b) > 5000:
            frames[s] = b
    idx = None
    for b in frames.values():
        idx = b.index if idx is None else idx.intersection(b.index)
    syms = sorted(frames)
    panel = {f: np.column_stack([frames[s][f].reindex(idx).to_numpy()
                                 for s in syms]).astype("float64")
             for f in ("open", "high", "low", "close", "volume")}
    L = np.column_stack([barrier_labels(frames[s].reindex(idx).ffill(),
                                        atr(frames[s].reindex(idx).ffill(), 14),
                                        1.5, 1.0, hold, CROSS_BPS)["long_ret"].to_numpy()
                         for s in syms])
    S = np.column_stack([barrier_labels(frames[s].reindex(idx).ffill(),
                                        atr(frames[s].reindex(idx).ffill(), 14),
                                        1.5, 1.0, hold, CROSS_BPS)["short_ret"].to_numpy()
                         for s in syms])
    return panel, L, S, idx, syms


def main():
    t0 = time.time()
    panel, L, S, idx, syms = load()
    T = len(idx)
    print(f"{len(syms)} levered ETFs, {T:,} timestamps, {T*len(syms):,} bars")
    print(f"{idx[0].date()} -> {idx[-1].date()}, barrier P&L net of {CROSS_BPS} bps\n")

    split = int(T * 0.7)
    train = np.arange(30, split, 26)
    test = np.arange(split + 60, T - 30, 26)
    n_eval = 20 * 160
    floor = np.sqrt(2 * np.log(n_eval))
    print(f"evolve on {len(train):,} rebalances, score once on {len(test):,} held out")
    print(f"~{n_eval:,} expressions searched; best-of-N floor t={floor:.2f}\n")

    ga = BarrierSearch(panel, L, S, seed=0)
    best, fit = ga.evolve(train, generations=20, population=160)
    r = ga.book(evaluate(best, panel), test)
    rt = r.mean() / r.std(ddof=1) * np.sqrt(len(r)) if len(r) > 1 else np.nan
    print(f"best expression:\n    {best}\n")
    print(f"  in-sample fitness t={fit:.2f}  (floor {floor:.2f})")
    print(f"  HELD OUT: {r.mean():+.2f} bps/rebalance  t={rt:.2f}  n={len(r):,}  "
          f"win {(r > 0).mean():.1%}\n")

    print("the same search on shuffled labels:")
    nulls = []
    for s in range(1, 4):
        rng = np.random.default_rng(200 + s)
        perm = rng.permutation(len(L))
        gn = BarrierSearch(panel, L[perm], S[perm], seed=s)
        bn, fn = gn.evolve(train, generations=20, population=160)
        rn = gn.book(evaluate(bn, panel), test)
        tn = rn.mean() / rn.std(ddof=1) * np.sqrt(len(rn)) if len(rn) > 1 else np.nan
        nulls.append(tn)
        print(f"  null {s}: in-sample t={fn:>6.2f}  held out t={tn:>6.2f}   {bn}")
    nulls = [x for x in nulls if np.isfinite(x)]
    print(f"\nheld-out t={rt:.2f} vs nulls mean {np.mean(nulls):.2f} "
          f"best {max(nulls):.2f}")
    print("VERDICT:", "readable and separable"
          if rt > 2 and all(rt > x for x in nulls) and fit > floor
          else "not separable from what the search invents")
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
