"""Ask it, four times a session: where does this close today?

The benchmark is not zero, it is the random walk - "the close will be whatever
it is right now". That is the efficient-market answer and it is famously hard to
beat, so a model that merely correlates with the close has proved nothing; it
has to beat the last print. Skill is R-squared against that forecast, which goes
negative for a model that would have done better keeping quiet.

Two questions get asked because they have different answers. The point forecast
(where) is a bet on direction, which everything in this project says is not
there. The interval (how far) is a bet on volatility, which is predictable - so
the useful answer may be "closes at 71.20, give or take 45 cents" even when the
71.20 is worth no more than the current price.

Polls are at 10:00, 11:30, 13:00 and 14:30 New York time. Every feature comes
from bars at or before the poll; the label is that session's actual close.
"""
import sys, pathlib, time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np, pandas as pd

from bipbip.data.store import BarStore
from bipbip.ml.discover import make_gbm_regressor

BASKET = ["TQQQ", "SQQQ", "SPY", "QQQ", "IWM", "DIA", "SOXL", "SOXS", "TNA",
          "TZA", "SPXL", "SPXS", "UPRO", "SPXU", "LABU", "XLF", "XLE", "XLK",
          "XLV", "XLU"]
POLLS = {"10:00": 30, "11:30": 120, "13:00": 210, "14:30": 300}
FULL = 390
FEATS = ["ret_so_far", "hi_gap", "lo_gap", "range_so_far", "rv_so_far",
         "pos_in_range", "minutes_left", "gap_open", "prev_rv",
         "vol_share", "last30_ret", "last30_rv"]


def rows_for(sym: str):
    """One row per (session, poll) holding everything knowable at that minute."""
    try:
        b = BarStore("data/bars").load(sym, "1m").dropna()
    except Exception:
        return None
    if len(b) < 5000 or b.index[0].year < 2026:
        return None
    day = pd.DatetimeIndex([pd.Timestamp(d.date()) for d in b.index])
    out, prev_close, prev_rv = [], None, None
    for d, chunk in b.groupby(day):
        if len(chunk) < FULL * 0.99:
            prev_close = float(chunk["close"].iloc[-1])
            continue
        c = chunk["close"].to_numpy()
        v = chunk["volume"].to_numpy()
        r = np.diff(np.log(c))
        close = float(c[-1])
        for label, k in POLLS.items():
            now = float(c[k])
            hi = float(chunk["high"].iloc[:k + 1].max())
            lo = float(chunk["low"].iloc[:k + 1].min())
            out.append({
                "sym": sym, "date": d, "poll": label, "now": now, "close": close,
                # A model predicting zero here IS the random walk.
                "y": float(np.log(close / now)),
                "ret_so_far": float(np.log(now / c[0])),
                "hi_gap": float(np.log(hi / now)),
                "lo_gap": float(np.log(lo / now)),
                "range_so_far": (hi - lo) / now,
                "rv_so_far": float(np.sqrt(np.mean(r[:k] ** 2))),
                "pos_in_range": (now - lo) / (hi - lo) if hi > lo else 0.5,
                "minutes_left": FULL - k,
                "gap_open": (float(np.log(c[0] / prev_close))
                             if prev_close else np.nan),
                "prev_rv": prev_rv if prev_rv is not None else np.nan,
                # Average volume per minute so far. The tempting version -
                # volume so far as a share of the session total - needs the
                # session total, which is not known until the close, and would
                # hand the model tomorrow's news today.
                "vol_share": float(np.log1p(v[:k + 1].mean())),
                "last30_ret": float(np.log(now / c[max(0, k - 30)])),
                "last30_rv": float(np.sqrt(np.mean(r[max(0, k - 30):k] ** 2))),
            })
        prev_close, prev_rv = close, float(np.sqrt(np.mean(r ** 2)))
    return pd.DataFrame(out)


def main():
    t0 = time.time()
    frames = [f for f in (rows_for(s) for s in BASKET) if f is not None and len(f)]
    df = pd.concat(frames, ignore_index=True).dropna().reset_index(drop=True)
    dates = np.array(sorted(df["date"].unique()))
    cut = dates[int(len(dates) * 0.7)]
    tr = (df["date"] < cut).to_numpy()
    te = ~tr
    print(f"{df['sym'].nunique()} ETFs, {len(dates)} sessions, {len(df):,} polls")
    print(f"train through {pd.Timestamp(cut).date()}, "
          f"test on {int(te.sum()):,} polls after it\n")

    X = df[FEATS].to_numpy(dtype="float32")
    y = df["y"].to_numpy()
    poll = df["poll"].to_numpy()
    sym = df["sym"].to_numpy()

    m = make_gbm_regressor(seed=0).fit(X[tr], y[tr])
    p = m.predict(X[te])
    yt, pt = y[te], p

    print("WHERE does it close - against 'wherever it is right now'")
    print(f"{'poll':>7} {'n':>6} {'random walk':>12} {'model':>9} "
          f"{'R2 vs walk':>12} {'direction':>10}")
    for label in POLLS:
        s = poll[te] == label
        if s.sum() < 30:
            continue
        a, b = yt[s], pt[s]
        r2 = 1.0 - np.sum((a - b) ** 2) / np.sum(a ** 2)
        print(f"{label:>7} {s.sum():>6,} {np.sqrt(np.mean(a**2))*1e4:>10.0f}b "
              f"{np.sqrt(np.mean((a-b)**2))*1e4:>8.0f}b {r2:>+12.4f} "
              f"{np.mean(np.sign(b) == np.sign(a)):>9.1%}")
    r2all = 1.0 - np.sum((yt - pt) ** 2) / np.sum(yt ** 2)
    print(f"{'ALL':>7} {te.sum():>6,} {np.sqrt(np.mean(yt**2))*1e4:>10.0f}b "
          f"{np.sqrt(np.mean((yt-pt)**2))*1e4:>8.0f}b {r2all:>+12.4f} "
          f"{np.mean(np.sign(pt) == np.sign(yt)):>9.1%}")

    print("\nHOW FAR is it from here - the size of the move still to come")
    mag = make_gbm_regressor(seed=0).fit(X[tr], np.log(np.abs(y[tr]) + 1e-6))
    pred_mag = np.exp(mag.predict(X[te]))
    flat = np.full(int(te.sum()), float(np.mean(np.abs(y[tr]))))
    # Pooled correlation across nineteen ETFs is mostly "which fund is this":
    # SOXL moves several times XLU, so a model that only learned the ranking of
    # tickers scores well without forecasting anything about today. The
    # within-symbol figure is the one that means the model knows this
    # afternoon will be quieter than usual FOR THIS NAME.
    a = np.abs(yt)
    within = [np.corrcoef(pred_mag[sym[te] == s], a[sym[te] == s])[0, 1]
              for s in sorted(set(sym[te])) if (sym[te] == s).sum() > 20]
    within = [w for w in within if np.isfinite(w)]
    print(f"  correlation with the actual distance: "
          f"pooled {np.corrcoef(pred_mag, a)[0, 1]:.3f}   "
          f"within symbol {np.mean(within):.3f}   "
          f"(range {min(within):+.2f} to {max(within):+.2f})")
    for name, band in (("model", pred_mag * 1.6), ("flat ", flat * 1.6)):
        print(f"  {name} band covers {np.mean(np.abs(yt) <= band):>5.1%} of closes, "
              f"average width +/-{np.mean(band)*1e4:>4.0f} bps")

    print("\nIn money, on the symbol actually being traded:")
    for s in ("TQQQ", "SPY"):
        k = sym[te] == s
        if k.sum() < 20:
            continue
        px = df["now"].to_numpy()[te][k]
        rw = np.mean(np.abs(np.expm1(yt[k])) * px)
        mo = np.mean(np.abs(np.expm1(yt[k] - pt[k])) * px)
        band = np.mean(np.expm1(pred_mag[k] * 1.6) * px)
        print(f"  {s:<5} avg ${px.mean():>7.2f}   "
              f"'closes where it is now' off by ${rw:.2f}   "
              f"model off by ${mo:.2f}   band +/-${band:.2f}")
    followups(df, tr, te, X, y, pt, pred_mag)
    print(f"\ntotal {time.time()-t0:.0f}s")




def followups(df, tr, te, X, y, pt, pred_mag):
    """Two checks the headline table cannot settle on its own."""
    yt = y[te]

    # 1. A hit rate is not a P&L. Being right 54% of the time while being wrong
    #    about the size on the big moves loses money, and the negative R2 says
    #    the sizes ARE wrong - so trade the sign and count the result.
    print("\nTrading the predicted direction into the close, 3.10 bps a round trip")
    print(f"{'poll':>7} {'trades':>7} {'hit':>7} {'net':>10} {'t':>7}")
    for label in list(POLLS) + ["ALL"]:
        s = (np.ones(len(yt), bool) if label == "ALL"
             else df["poll"].to_numpy()[te] == label)
        if s.sum() < 30:
            continue
        pnl = np.sign(pt[s]) * yt[s] * 1e4 - 3.10
        t = pnl.mean() / pnl.std(ddof=1) * np.sqrt(len(pnl))
        print(f"{label:>7} {s.sum():>7,} "
              f"{np.mean(np.sign(pt[s]) == np.sign(yt[s])):>6.1%} "
              f"{pnl.mean():>+9.2f}b {t:>7.2f}")

    # 2. Bands at different coverage are not comparable. Scale each to the same
    #    coverage and compare widths: the narrower one at equal coverage is the
    #    one that actually knows something.
    print("\nIntervals rescaled to equal coverage - narrower is better informed")
    flat = np.full(int(te.sum()), float(np.mean(np.abs(y[tr]))))
    print(f"{'coverage':>9} {'model width':>13} {'flat width':>12} {'tighter by':>12}")
    for target in (0.60, 0.80, 0.90):
        out = []
        for band in (pred_mag, flat):
            lo, hi = 0.1, 50.0
            for _ in range(60):                      # bisect the scale factor
                mid = (lo + hi) / 2
                if np.mean(np.abs(yt) <= band * mid) < target:
                    lo = mid
                else:
                    hi = mid
            out.append(float(np.mean(band * (lo + hi) / 2) * 1e4))
        mw, fw = out
        print(f"{target:>8.0%} {mw:>11.0f}b {fw:>10.0f}b {1 - mw/fw:>11.1%}")

if __name__ == "__main__":
    main()
