"""How much minute data does the model need before it starts to grow?

The useful question is not "does it work" but "is it improving with data" - a
flat curve means more history will not help and the ceiling is elsewhere, while
a rising one says keep fetching. Webull reaches fifteen years back at minute
granularity, so if the curve is still climbing at ninety days that is a cheap
thing to act on.

The comparison only means something if the training size is the ONLY thing that
changes. So the test blocks are fixed: the same five held-out stretches are
scored for every training width, with each model trained on exactly the N
sessions immediately preceding its block. Sliding the test set with the training
set instead would confound "more data" with "a different market".

Everything else is unchanged from the other searches - raw candles, cost charged
after the direction is picked, observations sampled at the holding period, and
the identical procedure re-run on shuffled labels, because a small training
window overfits harder and the null is what prices that in.
"""
import sys, pathlib, time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np, pandas as pd

from bipbip.data.store import BarStore
from bipbip.ml.candles import candle_features
from bipbip.ml.discover import make_gbm

SPREAD_BPS, SEC_FEE_BPS = 1.41, 0.278
COST_BPS = 2 * SPREAD_BPS + SEC_FEE_BPS
#: Bars in a full RTH session, and the training widths to sweep, per timeframe.
#: Ninety days of minutes can only vary training over one order of magnitude.
#: The thirty-minute archive reaches back fifteen years and varies it over
#: three, which is what actually settles whether the curve climbs.
PROFILE = {
    "1m": {"bars": 390, "test": 5,
           "train": (5, 10, 20, 30, 45), "horizons": (5, 15, 30, 60)},
    "30m": {"bars": 13, "test": 40,
            "train": (10, 40, 160, 640, 2560), "horizons": (1, 2, 4)},
}


def forward_return(b, horizon):
    entry, exit_ = b["open"].shift(-1), b["close"].shift(-horizon)
    day = pd.Series(b.index.normalize(), index=b.index)
    return (exit_ / entry - 1.0).where(day.shift(-horizon) == day)


def run(X, y, sess, blocks, train_sessions, horizon, seed=0, shuffle=False):
    """Score the fixed test blocks, training on `train_sessions` before each."""
    rng = np.random.default_rng(seed)
    yy = rng.permutation(y) if shuffle else y
    Xv = X.to_numpy(dtype="float32")
    out = []
    for lo_s in blocks:
        tr_lo, tr_hi = lo_s - train_sessions, lo_s
        tr = np.flatnonzero((sess >= tr_lo) & (sess < tr_hi) & np.isfinite(yy))
        # An embargo of one holding period, so the last training label cannot
        # overlap the first test bar.
        te = np.flatnonzero((sess >= lo_s) & (sess < lo_s + TEST_SESSIONS)
                            & np.isfinite(yy))
        # Scaled to the timeframe. A flat 500-row floor is right for minute
        # bars and silently discards every small training width at thirty
        # minutes, where ten sessions is 130 bars - the rows the curve most
        # needs in order to have a left-hand end at all.
        min_train = max(60, BARS_PER_SESSION * 2)
        if (len(tr) < min_train or len(te) < 20
                or len(np.unique(yy[tr] > 0)) < 2):
            continue
        te = te[te > tr.max() + horizon]
        if len(te) < 50:
            continue
        m = make_gbm(seed=seed).fit(Xv[tr], (yy[tr] > 0).astype(float))
        pick = te[::horizon]                       # non-overlapping trades
        p = m.predict_proba(Xv[pick])[:, 1]
        side = np.where(p > 0.5, 1.0, -1.0)
        # Every training width takes the SAME trades. A confidence gate looks
        # sensible but confounds the experiment: a model trained on more data is
        # less overconfident, so it filters more out, and the widest window ended
        # up scored on 32 trades against the narrowest one's 195. The difference
        # then measures sample size, not learning.
        out.append(side * yy[pick] * 1e4 - COST_BPS)
    return np.concatenate(out) if out else np.array([])


def tstat(a):
    a = np.asarray(a, dtype="float64")
    return a.mean() / a.std(ddof=1) * np.sqrt(len(a)) if len(a) > 1 and a.std() else np.nan


def main():
    t0 = time.time()
    tf = sys.argv[1] if len(sys.argv) > 1 else "1m"
    cfg = PROFILE[tf]
    global BARS_PER_SESSION, TEST_SESSIONS, TRAIN_SIZES
    BARS_PER_SESSION, TEST_SESSIONS = cfg["bars"], cfg["test"]
    TRAIN_SIZES = cfg["train"]
    b = BarStore("data/bars").load("TQQQ", tf).dropna()
    day = pd.DatetimeIndex([pd.Timestamp(d.date()) for d in b.index])
    per = b.groupby(day).size()
    complete = set(per[per >= BARS_PER_SESSION * 0.9].index)
    b = b[pd.Series(day, index=b.index).isin(complete).to_numpy()]
    day = pd.DatetimeIndex([pd.Timestamp(d.date()) for d in b.index])
    codes = pd.Categorical(day, categories=sorted(set(day)), ordered=True).codes
    n_sess = codes.max() + 1
    print(f"TQQQ {tf}: {len(b):,} bars over {n_sess} complete sessions "
          f"{b.index[0].date()} -> {b.index[-1].date()}")

    X = candle_features(b, 12)
    r = b["close"].pct_change().dropna() * 1e4
    print(f"features {X.shape[1]}, cost {COST_BPS:.2f} bps/round trip\n")

    max_train = max(TRAIN_SIZES)
    blocks = list(range(max_train, n_sess - TEST_SESSIONS + 1, TEST_SESSIONS))
    print(f"held-out blocks (identical for every training width): "
          f"{len(blocks)} x {TEST_SESSIONS} sessions\n")

    for horizon in cfg["horizons"]:
        fwd = forward_return(b, horizon)
        ok = np.isfinite(fwd.to_numpy()) & X.notna().all(axis=1).to_numpy()
        Xi, yi, si = X[ok], fwd[ok].to_numpy(), codes[ok]
        hurdle = 0.5 + COST_BPS / (2 * r.abs().mean() * np.sqrt(horizon))
        mins = horizon * (1 if tf == "1m" else 30)
        print(f"horizon {mins:>3} min   break-even hit rate {hurdle:.1%}")
        print(f"{'train':>7} {'bars':>9} {'trades':>7} {'net':>8} {'t':>7} "
              f"{'win':>6}   {'null net':>9} {'skill':>9}")
        for n_tr in TRAIN_SIZES:
            real = run(Xi, yi, si, blocks, n_tr, horizon)
            if not len(real):
                print(f"{n_tr:>7} (no usable folds)")
                continue
            nrs = [run(Xi, yi, si, blocks, n_tr, horizon, seed=s, shuffle=True)
                   for s in range(1, 4)]
            nrs = [a for a in nrs if len(a)]
            # The null is reported in bps as well as t. Comparing t alone
            # exaggerates the gap whenever the two have different spreads, and
            # the quantity that decides whether anything is tradeable is the
            # difference in bps against the 3.10 it costs to trade.
            nmean = np.mean([a.mean() for a in nrs]) if nrs else float("nan")
            print(f"{n_tr:>7} {n_tr * BARS_PER_SESSION:>9,} {len(real):>7,} "
                  f"{real.mean():>+7.2f}b {tstat(real):>7.2f} "
                  f"{(real > 0).mean():>6.1%}   {nmean:>+8.2f}b "
                  f"{real.mean() - nmean:>+8.2f}b")
        print()
    print("how many trades are needed to resolve an edge of a given size (t=2)")
    print(f"{'horizon':>9} {'noise/trade':>12} " +
          "".join(f"{f'{e} bps':>12}" for e in (1, 2, 5, 10)))
    for horizon in cfg["horizons"]:
        fwd = forward_return(b, horizon)
        ok = np.isfinite(fwd.to_numpy()) & X.notna().all(axis=1).to_numpy()
        sigma = float(np.nanstd(fwd[ok].to_numpy() * 1e4, ddof=1))
        per_session = max(1, BARS_PER_SESSION // horizon)
        cells = []
        for edge in (1, 2, 5, 10):
            n = (2.0 * sigma / edge) ** 2
            cells.append(f"{n / per_session:>11,.0f}d")
        mins = horizon * (1 if tf == "1m" else 30)
        print(f"{mins:>6} min {sigma:>11.1f}b " + "".join(cells))
    print("\n(sessions of HELD-OUT data required, at one non-overlapping "
          "trade per holding period)")
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
