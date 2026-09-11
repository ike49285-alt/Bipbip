"""Is TQQQ's volatility predictable, and is a 5DTE option priced for it?

The implied vol below is a CONSTANT read off one quote on one day, while the
"weeks that looked like this one" band is anchored on the archive's last
session. Those two drift apart every time bars are collected, so the comparison
slowly becomes one between a stale quote and a current regime. The anchor date
is printed for that reason; `scripts/variance_premium.py` asks the same
question without a frozen quote and is the better tool for it.

Direction at thirty minutes is a coin flip - the search over fifteen years gets
50.0% against a 52.5% break-even. Volatility is a different question. It
clusters, so today's tells you something about tomorrow's, and an option is a
bet on volatility that does not require knowing which way the underlying goes.

The comparison that matters for a five-day call is forward realised volatility
against the implied volatility being charged. Twenty days of history cannot
answer it - a single passed vol spike moves that window enough to invent an
edge, which is exactly how a 67% estimate turned a losing trade into a
"+104% per trade" one earlier. Fifteen years can.
"""
import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np, pandas as pd

from bipbip.data.store import BarStore

BARS = 13                      # thirty-minute bars in a regular session
ANN = np.sqrt(252 * BARS)


def realised(r: pd.Series, bars: int) -> pd.Series:
    """Annualised realised volatility, about zero rather than about the mean.

    Subtracting the sample mean flatters a trending window: it removes exactly
    the drift a holder of the option is exposed to.
    """
    return np.sqrt((r ** 2).rolling(bars).mean()) * ANN


def main():
    b = BarStore("data/bars").load("TQQQ", "30m").dropna()
    r = np.log(b["close"]).diff()
    same_day = (b.index.normalize()
                == pd.Series(b.index, index=b.index).shift(1).dt.normalize()).to_numpy()
    # The overnight return is kept, not discarded. It is not a thirty-minute
    # move, but an option holder is exposed to it, and dropping it puts the
    # realised estimate ~10 volatility points below the close-to-close number
    # the contract is actually priced against - which is enough on its own to
    # turn a fairly-priced option into an apparently rich one. What is discarded
    # instead is the NaN at the very first bar, since a rolling window returns
    # NaN for any window containing one.
    r = r.dropna()
    del same_day

    hist = realised(r, 5 * BARS)                    # trailing week
    fwd = realised(r, 5 * BARS).shift(-5 * BARS)    # the week the option covers
    ok = np.isfinite(hist) & np.isfinite(fwd)
    h, f = hist[ok], fwd[ok]
    assert len(h) > 1000, f'only {len(h)} usable windows'
    print(f"TQQQ, {len(h):,} overlapping observations, "
          f"{b.index[0].date()} -> {b.index[-1].date()}\n")

    # Sampled weekly so each observation is a distinct week, not a shifted copy.
    step = 5 * BARS
    hs, fs = h.iloc[::step], f.iloc[::step]
    print(f"persistence of volatility (independent weeks, n={len(hs):,})")
    print(f"  corr(this week's vol, next week's vol) = {np.corrcoef(hs, fs)[0, 1]:.3f}")
    # The same persistence test on returns, as the control: if next week's
    # return were as predictable as next week's volatility, the direction search
    # would not have come back at fifty percent.
    wk_ret = r.rolling(step).sum().reindex(h.index).iloc[::step]
    fwd_ret = r.rolling(step).sum().shift(-step).reindex(h.index).iloc[::step]
    mm = np.isfinite(wk_ret) & np.isfinite(fwd_ret)
    print(f"  corr(this week's return, next week's return) = "
          f"{np.corrcoef(wk_ret[mm], fwd_ret[mm])[0, 1]:+.3f}\n")

    print("what happened NEXT week, given this week's realised vol")
    print(f"{'this week':>18} {'n':>6} {'median next':>12} {'25th':>7} {'75th':>7} "
          f"{'P(next > this)':>15}")
    edges = [0, .25, .35, .45, .60, .80, 10]
    labels = ["<25%", "25-35%", "35-45%", "45-60%", "60-80%", ">80%"]
    for lo, hi, lab in zip(edges[:-1], edges[1:], labels):
        m = (hs >= lo) & (hs < hi)
        if m.sum() < 20:
            continue
        nxt = fs[m]
        print(f"{lab:>18} {m.sum():>6,} {nxt.median():>11.1%} "
              f"{nxt.quantile(.25):>6.1%} {nxt.quantile(.75):>6.1%} "
              f"{(nxt > hs[m]).mean():>14.1%}")

    iv = 0.4603      # the quoted implied vol on the $75 5DTE call, one day
    anchor_date = h.dropna().index[-1]
    cur = float(h.iloc[-1])
    print(f"\ntrailing-week realised vol as of {anchor_date.date()}: {cur:.1%}"
          f"  (the archive's last session, not the quote's day)")
    print(f"implied vol quoted on the 5DTE call: {iv:.1%}")
    band = (hs > cur * 0.8) & (hs < cur * 1.2)
    nxt = fs[band]
    print(f"\nin {band.sum():,} past weeks that looked like this one, the FOLLOWING "
          f"week's realised vol was:")
    print(f"  median {nxt.median():.1%}   mean {nxt.mean():.1%}   "
          f"25th {nxt.quantile(.25):.1%}   75th {nxt.quantile(.75):.1%}")
    print(f"  it exceeded the {iv:.1%} being charged {(nxt > iv).mean():.1%} of the time")
    print(f"\n  buying that option wins on volatility alone in "
          f"{(nxt > iv).mean():.0%} of comparable weeks;")
    print(f"  selling it wins in {(nxt <= iv).mean():.0%}.")


if __name__ == "__main__":
    main()
