"""Volatility at the horizon an option is actually priced over.

Everything measured today ran at thirty minutes or less, where the answer was
that 87% of the apparent skill was the intraday clock. Relative forecast error
shrinks as the horizon lengthens, so a five- or twenty-day forecast is a
different question, and it is the one your contract lives at.

The benchmark is HAR again - forward volatility on trailing volatility at 5, 22
and 66 days, the standard three windows - and the model is handed HAR's own
forecast as a feature so anything it adds sits on top rather than duplicating.

The metric is WITHIN-SYMBOL R-squared. Pooled R-squared across hundreds of names
is mostly "which ticker is this", and that trap has now inflated two results in
this session: HAR's own pooled 0.829 was 0.329 within symbol, and a magnitude
correlation of 0.541 was 0.283. Reporting both makes the difference visible
rather than flattering.

Test observations are sampled every H days so each is a distinct window rather
than a shifted copy of the one before it.
"""
import sys, pathlib, glob, os, time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np, pandas as pd

from bipbip.ml.discover import make_gbm_regressor

CUT = pd.Timestamp("2016-01-01")
MAX_SYMBOLS = 300


def rv(r: pd.Series, w: int) -> pd.Series:
    """Root mean square return, about zero rather than about the mean."""
    return np.sqrt((r ** 2).rolling(w, min_periods=w).mean())


def build(path: str, horizon: int):
    sym = os.path.basename(path).split("_")[0]
    try:
        b = pd.read_parquet(path)
    except Exception:
        return None
    if len(b) < 1500 or not {"high", "low", "close"} <= set(b.columns):
        return None
    b = b.dropna(subset=["close"]).sort_index()
    c = b["close"]
    r = np.log(c).diff()
    f = pd.DataFrame(index=b.index)
    f["rv5"], f["rv22"], f["rv66"] = rv(r, 5), rv(r, 22), rv(r, 66)

    # Parkinson: the high-low range carries more information about volatility
    # than the close-to-close return, because a session that travels and comes
    # back reads as quiet on closes alone. HAR never sees it.
    hl = np.log(b["high"] / b["low"])
    f["park"] = np.sqrt((hl ** 2).rolling(22, min_periods=22).mean() / (4 * np.log(2)))
    f["park_ratio"] = f["park"] / f["rv22"]
    # Leverage: volatility rises after declines, and falls after rallies. A
    # model on absolute returns alone is blind to the sign.
    f["ret22"] = np.log(c / c.shift(22))
    f["ret5"] = np.log(c / c.shift(5))
    f["down_share"] = (r < 0).rolling(22, min_periods=22).mean()
    # Jumps: one large move inflates a window's volatility without implying the
    # next window is volatile.
    f["jump"] = r.abs().rolling(22, min_periods=22).max() / f["rv22"]
    f["vol_of_vol"] = f["rv5"].rolling(66, min_periods=66).std() / f["rv22"]
    if "volume" in b.columns:
        v = b["volume"].replace(0, np.nan)
        f["vol_trend"] = np.log(v.rolling(5).mean() / v.rolling(66).mean())
    else:
        f["vol_trend"] = np.nan

    fwd = rv(r, horizon).shift(-horizon)
    ok = (np.isfinite(fwd) & (fwd > 0) & f.notna().all(axis=1)
          & (f[["rv5", "rv22", "rv66"]] > 0).all(axis=1)).to_numpy()
    if ok.sum() < 400:
        return None
    out = f[ok].copy()
    out["y"] = np.log(fwd[ok])
    out["sym"] = sym
    # The daily archive is tz-aware and the archives differ in zone, so dates
    # are flattened to naive calendar days. Comparing a tz-aware column against
    # a naive cut-off raises rather than silently mis-splitting, which is the
    # better failure, but it still has to be handled.
    out["date"] = pd.DatetimeIndex(
        [pd.Timestamp(t.date()) for t in out.index])
    return out.reset_index(drop=True)


def r2(y, p):
    ss = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - float(np.sum((y - p) ** 2)) / ss if ss > 0 else np.nan


def within(y, p, syms):
    vals = [r2(y[syms == s], p[syms == s])
            for s in np.unique(syms) if (syms == s).sum() > 30]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals))


def main():
    t0 = time.time()
    horizon = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    files = sorted(glob.glob("data/bars/*_1d.parquet"))[:MAX_SYMBOLS]
    frames = [d for d in (build(f, horizon) for f in files) if d is not None]
    df = pd.concat(frames, ignore_index=True)

    # Volatility is common across names, so what the whole market is doing is
    # information about any one symbol that its own history cannot supply.
    mkt = df.groupby("date")["rv22"].transform("mean")
    df["mkt_vol"] = np.log(df["rv22"] / mkt)
    base = ["rv5", "rv22", "rv66"]
    extras = ["park", "park_ratio", "ret22", "ret5", "down_share", "jump",
              "vol_of_vol", "vol_trend", "mkt_vol"]

    tr = (df["date"] < CUT).to_numpy()
    # Sample the test set every `horizon` days so each observation is a distinct
    # window, not a one-day shift of the previous one.
    te = np.zeros(len(df), bool)
    for s, g in df.groupby("sym", sort=False):
        idx = g.index.to_numpy()[(g["date"] >= CUT).to_numpy()]
        te[idx[::horizon]] = True

    print(f"{df['sym'].nunique()} symbols, {len(df):,} rows, {horizon}-day horizon")
    print(f"train before {CUT.date()} ({tr.sum():,}), "
          f"test after, non-overlapping ({te.sum():,})\n")

    H = np.log(df[base].to_numpy())
    y = df["y"].to_numpy()
    syms = df["sym"].to_numpy()
    A = np.column_stack([np.ones(tr.sum()), H[tr]])
    coef, *_ = np.linalg.lstsq(A, y[tr], rcond=None)
    har = np.column_stack([np.ones(len(y)), H]) @ coef

    print(f"{'model':<34}{'pooled R2':>11}{'within-symbol':>15}")
    print(f"{'persistence (carry vol forward)':<34}"
          f"{r2(y[te], H[te][:, 1]):>11.3f}{within(y[te], H[te][:, 1], syms[te]):>15.3f}")
    print(f"{'HAR (3 windows)':<34}"
          f"{r2(y[te], har[te]):>11.3f}{within(y[te], har[te], syms[te]):>15.3f}")

    har_w = within(y[te], har[te], syms[te])
    results = {}
    # The tree predicts HAR's RESIDUAL, not the target. Handing a tree the HAR
    # forecast as an input and asking for the level makes it reconstruct a
    # smooth linear function as a staircase, and it loses accuracy doing so -
    # which showed up as three of four feature sets scoring BELOW plain HAR, an
    # impossible result for a model free to ignore a feature. Predicting the
    # residual leaves HAR's forecast exactly intact and lets the tree add only
    # what HAR gets wrong.
    for name, cols in (("HAR + range (Parkinson)", ["park", "park_ratio"]),
                       ("HAR + leverage (sign)", ["ret22", "ret5", "down_share"]),
                       ("HAR + jumps", ["jump", "vol_of_vol"]),
                       ("HAR + market-wide vol", ["mkt_vol"]),
                       ("HAR + everything", extras)):
        X = df[cols].to_numpy().astype("float32")
        m = make_gbm_regressor(seed=0).fit(X[tr], (y - har)[tr])
        p = har[te] + m.predict(X[te])
        w = within(y[te], p, syms[te])
        results[name] = (r2(y[te], p), w)
        print(f"{name:<34}{results[name][0]:>11.3f}{w:>15.3f}"
              f"   ({w - har_w:+.3f})")

    best = max(results, key=lambda k: results[k][1])
    X = df[extras].to_numpy().astype("float32")
    p = har[te] + make_gbm_regressor(seed=0).fit(X[tr], (y - har)[tr]).predict(X[te])
    ann = np.sqrt(252)
    lvl = float(np.exp(np.mean(y[te])) * ann)
    print(f"\nannualised volatility of the median test window: {lvl:.0%}")
    for name, pred in (("HAR", har[te]), ("model", p)):
        rmse = float(np.sqrt(np.mean((y[te] - pred) ** 2)))
        print(f"  {name:>5}: typical error {np.expm1(rmse):>5.1%} of the level"
              f"  =  {np.expm1(rmse) * lvl * 100:>4.1f} vol points")
    # A large part of that error is the TARGET's own sampling noise, not the
    # forecast's. Realised volatility over H days is estimated from H returns,
    # so even a perfect forecast of true volatility misses the realisation by
    # roughly 1/sqrt(2H). Reporting it separates what is unforecastable from
    # what was simply not forecast.
    floor = 1.0 / np.sqrt(2.0 * horizon)
    print(f"  floor: even a perfect forecast of TRUE vol misses the realisation "
          f"by ~{floor:.0%}  =  {floor * lvl * 100:.1f} vol points")
    print(f"\nbest addition: {best} "
          f"({results[best][1] - har_w:+.3f} within-symbol R2 over HAR)")
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
