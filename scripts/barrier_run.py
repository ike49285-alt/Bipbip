"""Fixed holding period against target-and-stop, on identical features and folds.

The fixed-horizon label asks "will this be up in exactly h bars", which mislabels
the two trades that matter most: one that runs +40 bps by bar five and gives it
back is recorded as a loss, and one that sits -80 bps for twenty bars before
recovering is recorded as a win. The model is then trained to avoid the first and
sit through the second.

Barriers ask what a trader would actually do. Everything else is held constant so
the difference is the label and nothing else.

The model now REGRESSES the cost-adjusted outcome rather than classifying its
sign, because sign was never the objective: a model right 54% of the time on
small moves and wrong on large ones scores well on classification and loses
money, which is what happened twice in this project already. Trading only when
the predicted move clears the spread asks the question that decides a trade.
"""
import sys, pathlib, time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

from bipbip.core.indicators import assert_causal, atr
from bipbip.data.store import BarStore
from bipbip.ml.barriers import STOP, TARGET, TIMEOUT, barrier_labels
from bipbip.ml.discover import make_gbm, make_gbm_regressor
from scripts.everything import CROSS_BPS, EARN_BPS, features, forward

MAX_HOLD = 26


def folds(n, k=6):
    edge = n // (k + 1)
    for i in range(k):
        hi = edge * (i + 1)
        yield np.arange(hi), np.arange(hi + MAX_HOLD + 13, min(edge * (i + 2), n))


def main():
    t0 = time.time()
    b = BarStore("data/bars").load("TQQQ", "30m").dropna()
    X = assert_causal(features, b)
    a = atr(b, 14)
    lab = barrier_labels(b, a, target_mult=1.5, stop_mult=1.0,
                         max_hold=MAX_HOLD, cost_bps=0.0)   # cost added at scoring
    fixed = forward(b, 1).to_numpy() * 1e4

    ok = (X.notna().all(axis=1).to_numpy() & np.isfinite(lab["long_ret"].to_numpy())
          & np.isfinite(lab["short_ret"].to_numpy()))
    Xv = X[ok].to_numpy(dtype="float32")
    L, S = lab["long_ret"].to_numpy()[ok], lab["short_ret"].to_numpy()[ok]
    F = fixed[ok]
    held = lab["long_held"].to_numpy()[ok]
    which = lab["long_barrier"].to_numpy()[ok]

    print(f"TQQQ 30m, {ok.sum():,} labelled bars, {X.shape[1]} features")
    print(f"barriers: +1.5 ATR target, -1.0 ATR stop, {MAX_HOLD}-bar cap\n")
    print(f"how a long resolves: target {np.mean(which == TARGET):.1%}  "
          f"stop {np.mean(which == STOP):.1%}  timeout {np.mean(which == TIMEOUT):.1%}"
          f"   median hold {np.median(held):.0f} bars vs the fixed 1\n")

    n = len(Xv)
    res = {k: [] for k in ("fixed", "barrier")}
    nulls = {k: [[] for _ in range(3)] for k in ("fixed", "barrier")}
    rngs = [np.random.default_rng(100 + i) for i in range(3)]
    for tr, te in folds(n):
        if len(tr) < 2000 or len(te) < 200:
            continue
        # Non-overlapping with VARIABLE holds: take a trade, then skip forward
        # by the bars it actually consumed. Stepping by max_hold instead throws
        # away seven eighths of the sample, because the median barrier trade
        # resolves in three bars rather than twenty-six - and the power to
        # detect anything goes with it.
        picked, cur, stop_at = [], int(te[0]), int(te[-1])
        while cur <= stop_at:
            picked.append(cur)
            cur += max(1, int(held[cur]))
        te = np.array(picked)
        if len(te) < 30:
            continue
        # Fixed-horizon, classified on sign - the old setup, for comparison.
        m = make_gbm(seed=0).fit(Xv[tr], (F[tr] > 0).astype(float))
        p = m.predict_proba(Xv[te])[:, 1]
        res["fixed"].append(np.where(p > .5, 1., -1.) * F[te])

        # Barriers, regressed. Two models: a short is not a mirrored long.
        ml = make_gbm_regressor(seed=0).fit(Xv[tr], L[tr])
        ms = make_gbm_regressor(seed=0).fit(Xv[tr], S[tr])
        pl, ps = ml.predict(Xv[te]), ms.predict(Xv[te])
        take_long = pl >= ps
        res["barrier"].append(np.where(take_long, L[te], S[te]))

        # The control. Shuffling the LABELS while leaving the features and the
        # fold structure untouched destroys any relationship a model could
        # learn, so whatever the same machinery returns here is what it
        # manufactures from nothing. A barrier payoff is skewed - lose small
        # often, win big rarely - and skew alone can produce a flattering mean,
        # so this arm is not optional.
        for i, rng in enumerate(rngs):
            perm = rng.permutation(len(tr))
            mf = make_gbm(seed=0).fit(Xv[tr], (F[tr][perm] > 0).astype(float))
            pf = mf.predict_proba(Xv[te])[:, 1]
            nulls["fixed"][i].append(np.where(pf > .5, 1., -1.) * F[te])
            nl = make_gbm_regressor(seed=0).fit(Xv[tr], L[tr][perm])
            ns = make_gbm_regressor(seed=0).fit(Xv[tr], S[tr][perm])
            tl = nl.predict(Xv[te]) >= ns.predict(Xv[te])
            nulls["barrier"][i].append(np.where(tl, L[te], S[te]))

    print(f"{'label':>10} {'trades':>7} {'GROSS':>9} {'t':>6} {'win':>6} "
          f"{'crossing':>9} {'earning':>9}   {'null gross':>18}")
    for k, v in res.items():
        if not v:
            continue
        a_ = np.concatenate(v)
        t = a_.mean() / a_.std(ddof=1) * np.sqrt(len(a_)) if a_.std() else np.nan
        nm = [np.concatenate(x).mean() for x in nulls[k] if x]
        print(f"{k:>10} {len(a_):>7,} {a_.mean():>+8.2f}b {t:>6.2f} "
              f"{(a_ > 0).mean():>6.1%} {a_.mean()-CROSS_BPS:>+8.2f}b "
              f"{a_.mean()-EARN_BPS:>+8.2f}b   "
              f"mean {np.mean(nm):>+6.2f}b best {max(nm):>+6.2f}b")
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
