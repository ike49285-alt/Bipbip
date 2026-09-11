"""Is the barrier edge a plateau or a spike?

One configuration proves nothing. If +3.66 bps exists only at exactly 1.5 ATR
target, 1.0 ATR stop and a 26-bar cap, it is the luckiest cell of a grid nobody
had looked at yet. If the neighbours work too, the surface is describing
something about the instrument rather than about the search.

So the whole grid is printed rather than its maximum. Reading only the best cell
of 32 is a search with its own floor: the best of N draws from nothing scores
sqrt(2 ln N), which is t=2.63 here, and any single winner below that is worse
than luck. What cannot be faked as easily is a CONNECTED REGION of positive
cells - noise scatters, structure clusters.
"""
import sys, pathlib, time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

from bipbip.core.indicators import assert_causal, atr
from bipbip.data.store import BarStore
from bipbip.ml.barriers import barrier_labels
from bipbip.ml.discover import make_gbm_regressor
from scripts.everything import CROSS_BPS, features

TARGETS = (1.0, 1.5, 2.0, 3.0)
STOPS = (0.5, 1.0, 1.5, 2.0)
HOLDS = (13, 26)


def folds(n, hold, k=6):
    edge = n // (k + 1)
    for i in range(k):
        hi = edge * (i + 1)
        yield np.arange(hi), np.arange(hi + hold + 13, min(edge * (i + 2), n))


def evaluate(Xv, L, S, held, hold):
    out = []
    for tr, te in folds(len(Xv), hold):
        if len(tr) < 2000 or len(te) < 200:
            continue
        picked, cur, stop_at = [], int(te[0]), int(te[-1])
        while cur <= stop_at:
            picked.append(cur)
            cur += max(1, int(held[cur]))
        te = np.array(picked)
        if len(te) < 30:
            continue
        ml = make_gbm_regressor(seed=0).fit(Xv[tr], L[tr])
        ms = make_gbm_regressor(seed=0).fit(Xv[tr], S[tr])
        pick_long = ml.predict(Xv[te]) >= ms.predict(Xv[te])
        out.append(np.where(pick_long, L[te], S[te]))
    return np.concatenate(out) if out else np.array([])


def main():
    t0 = time.time()
    b = BarStore("data/bars").load("TQQQ", "30m").dropna()
    X = assert_causal(features, b)
    a = atr(b, 14)
    base_ok = X.notna().all(axis=1).to_numpy()
    floor = np.sqrt(2 * np.log(len(TARGETS) * len(STOPS) * len(HOLDS)))
    print(f"TQQQ 30m, {X.shape[1]} features, "
          f"{len(TARGETS)*len(STOPS)*len(HOLDS)} configurations")
    print(f"best-of-grid null floor: t={floor:.2f}\n")

    best = None
    for hold in HOLDS:
        print(f"max_hold = {hold} bars")
        print(f"{'target':>8} " + "".join(f"{f'stop {s}':>16}" for s in STOPS))
        for tm in TARGETS:
            cells = []
            for sm in STOPS:
                lab = barrier_labels(b, a, target_mult=tm, stop_mult=sm,
                                     max_hold=hold, cost_bps=0.0)
                L = lab["long_ret"].to_numpy()
                S = lab["short_ret"].to_numpy()
                held = lab["long_held"].to_numpy()
                ok = base_ok & np.isfinite(L) & np.isfinite(S)
                r = evaluate(X[ok].to_numpy(dtype="float32"), L[ok], S[ok],
                             held[ok], hold)
                if len(r) < 100:
                    cells.append("      --       ")
                    continue
                t = r.mean() / r.std(ddof=1) * np.sqrt(len(r))
                net = r.mean() - CROSS_BPS
                cells.append(f"{r.mean():>+7.2f}b t{t:>5.2f}")
                if best is None or t > best[0]:
                    best = (t, tm, sm, hold, r.mean(), net, len(r))
            print(f"{tm:>8.1f} " + "".join(f"{c:>16}" for c in cells))
        print()

    t, tm, sm, hold, g, net, n = best
    print(f"best cell: target {tm} / stop {sm} / hold {hold} -> "
          f"{g:+.2f} bps gross, t={t:.2f}, {net:+.2f} net of crossing, n={n:,}")
    print(f"{'CLEARS' if t > floor else 'DOES NOT CLEAR'} the "
          f"best-of-grid floor of {floor:.2f}")
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
