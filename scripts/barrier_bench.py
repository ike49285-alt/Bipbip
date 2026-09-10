"""Is the barrier result skill, or is it TQQQ going up?

Every one of 32 grid cells came back positive, which is what a real effect looks
like and also what a drift-harvesting artefact looks like. TQQQ compounded
enormously across this sample, so a model that is mostly long with a wide target
and a tight stop collects that drift without knowing anything.

The shuffled-label null says the MODEL adds something over a model that learned
nothing. It does not say the barrier STRUCTURE is neutral. These benchmarks do:
always long, always short, and a coin flip, all through the identical barriers
and the identical non-overlapping sampling. If always-long matches the model,
the model is a long-only strategy with extra steps.
"""
import sys, pathlib, time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np, pandas as pd

from bipbip.core.indicators import assert_causal, atr
from bipbip.data.store import BarStore
from bipbip.ml.barriers import barrier_labels
from bipbip.ml.discover import make_gbm_regressor
from scripts.everything import CROSS_BPS, features
from scripts.barrier_sweep2 import folds


def run(tm, sm, hold, label):
    b = BarStore("data/bars").load("TQQQ", "30m").dropna()
    X = assert_causal(features, b)
    a = atr(b, 14)
    lab = barrier_labels(b, a, target_mult=tm, stop_mult=sm,
                         max_hold=hold, cost_bps=0.0)
    L, S = lab["long_ret"].to_numpy(), lab["short_ret"].to_numpy()
    held = lab["long_held"].to_numpy()
    ok = X.notna().all(axis=1).to_numpy() & np.isfinite(L) & np.isfinite(S)
    Xv, L, S, held = X[ok].to_numpy(dtype="float32"), L[ok], S[ok], held[ok]

    arms = {k: [] for k in ("model", "always long", "always short", "coin flip")}
    rng = np.random.default_rng(0)
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
        pick = ml.predict(Xv[te]) >= ms.predict(Xv[te])
        arms["model"].append(np.where(pick, L[te], S[te]))
        arms["always long"].append(L[te])
        arms["always short"].append(S[te])
        arms["coin flip"].append(np.where(rng.random(len(te)) < .5, L[te], S[te]))

    print(f"\n{label}: target {tm} / stop {sm} / hold {hold}")
    print(f"  {'arm':<14}{'gross':>9}{'t':>7}{'win':>7}{'net crossing':>14}")
    for k, v in arms.items():
        r = np.concatenate(v)
        t = r.mean() / r.std(ddof=1) * np.sqrt(len(r))
        print(f"  {k:<14}{r.mean():>+8.2f}b{t:>7.2f}{(r > 0).mean():>7.1%}"
              f"{r.mean()-CROSS_BPS:>+13.2f}b")
    m = np.concatenate(arms["model"])
    l = np.concatenate(arms["always long"])
    # PAIRED. The two arms take the same trades on the same bars and differ only
    # where the model goes short, so an independent-samples interval discards
    # that pairing. It is worth reporting both, and worth recording that the
    # gain is SMALL here - 4.34 to 4.20, and 3.24 to 2.68. Pairing tightens an
    # interval in proportion to how often the arms agree, and these disagree on
    # 41 to 53 percent of trades, so most of the difference is genuinely
    # non-zero rather than cancelling. Expecting a large gain here was wrong.
    d = m - l
    se_p = d.std(ddof=1) / np.sqrt(len(d))
    se_u = np.sqrt(m.var(ddof=1) / len(m) + l.var(ddof=1) / len(l))
    shorted = np.mean(d != 0)
    print(f"  model minus always-long: {d.mean():+.2f} bps   "
          f"paired +/-{1.96*se_p:.2f} (t={d.mean()/se_p:.2f})   "
          f"unpaired +/-{1.96*se_u:.2f}")
    print(f"  the model went short on {shorted:.1%} of trades; "
          f"on those it gained {d[d != 0].mean():+.2f} bps each")


if __name__ == "__main__":
    t0 = time.time()
    run(1.5, 1.0, 26, "the blind pick")
    run(2.0, 0.5, 13, "the best grid cell")
    print(f"\ntotal {time.time()-t0:.0f}s")
