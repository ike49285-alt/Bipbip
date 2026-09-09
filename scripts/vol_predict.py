"""Point the machinery at the thing that is actually predictable.

Returns persist at -0.008 across independent weeks; realised volatility persists
at 0.711. Every search in this project so far has been aimed at the -0.008.

The null has to change with the target. Shuffled labels are the right control
for a return search, where the honest prior is that there is nothing there. For
volatility the honest prior is the opposite - it is famously persistent, so a
model that "predicts" it is usually just rediscovering that tomorrow resembles
today, and beating shuffled labels would prove nothing at all.

So the benchmark here is HAR: forward volatility regressed on trailing
volatility at a short, medium and long window. It is the standard model in the
literature precisely because it is hard to beat, and it is what a market maker's
quote already embodies. Incremental R-squared over HAR is the only number worth
reading; the raw R-squared of any volatility model looks impressive and means
almost nothing.
"""
import sys, pathlib, time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np, pandas as pd

from bipbip.data.store import BarStore
from bipbip.ml.candles import candle_features
from bipbip.ml.discover import make_gbm_regressor

BASKET = ["TQQQ", "SQQQ", "SPY", "QQQ", "IWM", "DIA", "SOXL", "SOXS", "TNA",
          "TZA", "SPXL", "SPXS", "UPRO", "SPXU", "LABU", "XLF", "XLE", "XLK",
          "XLV", "XLU"]


def realised(r: pd.Series, w: int) -> pd.Series:
    """Root mean square return over the trailing w bars, about zero.

    About zero rather than about the mean: subtracting the sample mean removes
    exactly the drift a position is exposed to, which flatters any window that
    happens to trend.
    """
    return np.sqrt((r ** 2).rolling(w, min_periods=w).mean())


def build(sym: str, horizon: int):
    b = BarStore("data/bars").load(sym, "1m").dropna()
    if len(b) < 5000 or b.index[0].year < 2026:
        return None
    r = np.log(b["close"]).diff()
    # Drop the overnight bar rather than mask it: a rolling window returns NaN
    # for any window containing one NaN, so masking empties the whole series.
    same = (b.index.normalize()
            == pd.Series(b.index, index=b.index).shift(1).dt.normalize()).to_numpy()
    r = r.where(same).dropna()
    bb = b.loc[r.index]

    fwd = realised(r, horizon).shift(-horizon)            # the window being priced
    har = pd.DataFrame({
        "rv_short": realised(r, horizon),
        "rv_med": realised(r, horizon * 5),
        "rv_long": realised(r, horizon * 20),
    })
    feats = candle_features(bb, 8)
    feats["rv_short"], feats["rv_med"], feats["rv_long"] = (
        har["rv_short"], har["rv_med"], har["rv_long"])
    feats["rv_ratio"] = har["rv_short"] / har["rv_long"]
    feats["bar_of_day"] = bb.index.hour * 60 + bb.index.minute

    ok = (np.isfinite(fwd) & har.notna().all(axis=1)
          & feats.notna().all(axis=1) & (fwd > 0)).to_numpy()
    if ok.sum() < 2000:
        return None
    return (np.log(fwd[ok].to_numpy()), np.log(har[ok].to_numpy()),
            feats[ok], bb.index[ok])


def r2(y, p):
    ss = np.sum((y - y.mean()) ** 2)
    return 1.0 - np.sum((y - p) ** 2) / ss if ss > 0 else np.nan


def main():
    t0 = time.time()
    horizon = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    Y, H, X, S = [], [], [], []
    for sym in BASKET:
        got = build(sym, horizon)
        if got is None:
            continue
        y, h, f, idx = got
        Y.append(y); H.append(h); X.append(f)
        S.append(np.full(len(y), sym))
    if not Y:
        print("no usable symbols"); return
    y = np.concatenate(Y); h = np.vstack(H)
    X = pd.concat(X, ignore_index=True); syms = np.concatenate(S)
    n_sym = len(set(syms))
    print(f"{n_sym} ETFs, {len(y):,} observations, forecasting realised vol "
          f"over the next {horizon} minutes")
    print(f"target is log realised vol; benchmark is HAR on trailing vol "
          f"at {horizon}, {horizon*5}, {horizon*20} bars\n")

    # Chronological split per symbol, so the test period is the same calendar
    # stretch for every name and no symbol trains on another's future.
    cut = int(len(y) * 0.7)
    order = np.argsort(np.concatenate([np.arange(len(a)) for a in Y]), kind="stable")
    del order
    tr = np.zeros(len(y), bool)
    for s in set(syms):
        m = np.flatnonzero(syms == s)
        tr[m[:int(len(m) * 0.7)]] = True
    te = ~tr
    del cut

    # HAR: ordinary least squares on the three trailing volatilities.
    A = np.column_stack([np.ones(tr.sum()), h[tr]])
    coef, *_ = np.linalg.lstsq(A, y[tr], rcond=None)
    har_pred = np.column_stack([np.ones(te.sum()), h[te]]) @ coef
    har_r2 = r2(y[te], har_pred)

    # Persistence: just carry today's volatility forward, no fitting at all.
    pers_r2 = r2(y[te], h[te][:, 0])

    Xv = X.to_numpy(dtype="float32")
    m = make_gbm_regressor(seed=0).fit(Xv[tr], y[tr])
    gbm_pred = m.predict(Xv[te])
    gbm_r2 = r2(y[te], gbm_pred)

    # The model gets HAR's own forecast as a feature, so anything it adds is on
    # top of the benchmark rather than a rediscovery of it.
    Xh = np.column_stack([Xv, np.column_stack([np.ones(len(y)), h]) @ coef])
    m2 = make_gbm_regressor(seed=0).fit(Xh[tr], y[tr])
    both_r2 = r2(y[te], m2.predict(Xh[te]))

    print(f"{'model':<34}{'out-of-sample R2':>18}")
    print(f"{'persistence (carry vol forward)':<34}{pers_r2:>17.3f}")
    print(f"{'HAR (fitted, 3 windows)':<34}{har_r2:>17.3f}")
    print(f"{'raw candles + trailing vol':<34}{gbm_r2:>17.3f}")
    print(f"{'the same, given HAR as a feature':<34}{both_r2:>17.3f}")
    print(f"\nincremental over HAR: {both_r2 - har_r2:+.4f} R2")

    # What it is worth, in the units an option is quoted in.
    err_har = np.sqrt(np.mean((y[te] - har_pred) ** 2))
    err_gbm = np.sqrt(np.mean((y[te] - m2.predict(Xh[te])) ** 2))
    ann = np.sqrt(252 * 390 / horizon)
    lvl = float(np.exp(np.mean(y[te])) * ann)
    print(f"typical forecast error, HAR   : {np.expm1(err_har):.1%} of the level "
          f"(~{np.expm1(err_har)*lvl*100:.1f} vol points at {lvl:.0%} vol)")
    print(f"typical forecast error, model : {np.expm1(err_gbm):.1%} of the level "
          f"(~{np.expm1(err_gbm)*lvl*100:.1f} vol points)")
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
