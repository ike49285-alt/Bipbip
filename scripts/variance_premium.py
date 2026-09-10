"""Is implied volatility above the volatility that actually follows?

This is the one question in this project that is NOT a search for alpha. The
variance risk premium is compensation for bearing crash risk - CLAUDE.md's own
taxonomy calls that class "real, durable, and requiring no skill". You are not
predicting anything and you do not have to be right; you are being paid to hold
something other people want to shed. That is why it can survive when fourteen
direction and ranking searches here did not.

WHY IT WAS WORTH RE-ASKING. The repo already closed "selling defined-risk
premium" as approximately zero - but measured it only on TQQQ, whose options
quote 11.2% of mid near the money against SPY's 2.3%. Converted through ATM
vega, one half-spread costs 0.1 vol points on SPY, 7.8 on TQQQ and 18.6 on
SOXL, where it consumes the entire premium. That conclusion was about the
vehicle, not the premium.

MEASUREMENT. VIX at each date against the volatility SPY actually realised over
the FOLLOWING 21 sessions - not the trailing window, which would only measure
mean reversion. VIX is a 30-calendar-day implied, so 21 sessions is the matching
trading-clock horizon.

    non-overlapping (every 21st)   n=56   mean +3.32p   sd 5.98   t=4.15
    overlapping (every day)        n=1173 mean +3.45p   sd 5.99   t=19.75

The pair is the point. Sampling daily inflates t from 4.15 to 19.75, a ratio of
4.76 against sqrt(21)=4.58 - this repo's first documented trap reproducing
almost exactly. Only the non-overlapping number means anything.

IT SURVIVES THE TEST THAT KILLED RISK PARITY.

    period          n     mean       t    negative
    full           56   +3.32p    4.15      14%
    first half     28   +2.88p    3.48      14%
    second half    28   +3.75p    2.73      14%
    2022           12   +1.01p    0.61      33%
    2023           12   +4.23p    8.65       0%
    2024           12   +3.20p    2.34      17%
    2025           12   +3.56p    1.21      17%
    2026            7   +4.51p    3.95       0%
    ex-2022        44   +3.95p    4.40

Every year is positive, and dropping the 2022 bear market makes it stronger,
not weaker. Distribution-free: positive in 48 of 56 windows, binomial
p = 2.3e-08.

THE TAIL IS NOT A FOOTNOTE, IT IS THE PRODUCT. The premium is negative in 14%
of windows and the worst is -26.0 vol points, against a mean of +3.3. That
eight-to-one ratio IS the risk being compensated. Anyone reading the +3.3 as
free money has mistaken an insurance float for an edge, and CLAUDE.md's note
that a strategy losing a fixed fraction per trade compounds to ruin applies
with full force.

WHAT THIS IS NOT. It is not a strategy P&L: turning vol points into dollars
needs a structure, and the structure pays its own spread and owns its own tail.
The window is 4.7 years and one regime - it contains 2022 but not 2008 and not
March 2020, both of which were far worse than anything here. And VIX itself is
not tradeable.

DATA. VIX from Webull. Only 2021-12 onward is usable: earlier history is gappy
(jumping 2021-12 to 2020-11 to 2018-12) and carries stale repeated closes, so it
is deliberately not used rather than quietly averaged in.
"""
import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np, pandas as pd

from bipbip.data.store import BarStore

#: VIX quotes a 30-CALENDAR-day implied vol; 21 sessions is the same span on
#: the trading clock, which is the clock variance actually accrues on.
HORIZON = 21


def forward_realised_vol(closes: pd.Series, horizon: int = HORIZON) -> pd.Series:
    """Annualised vol of the returns that FOLLOW each date, in vol points.

    Forward, not trailing. Trailing would compare implied against vol that has
    already happened, which measures mean reversion rather than a premium: in
    this archive's calmest quintile forward vol runs only +0.1 points above
    trailing, but in the top decile it runs 9.6 points BELOW it.
    """
    r = np.log(closes / closes.shift(1))
    return r.shift(-1).rolling(horizon).std().shift(-(horizon - 1)) * np.sqrt(252) * 100


def premium_frame(store: BarStore, horizon: int = HORIZON) -> pd.DataFrame:
    vix = store.load("VIX", "1d")["close"]
    spy = store.load("SPY", "1d")["close"]
    for s in (vix, spy):
        s.index = pd.DatetimeIndex([pd.Timestamp(d.date()) for d in s.index])
    df = pd.DataFrame({"vix": vix,
                       "rv_fwd": forward_realised_vol(spy, horizon)}).dropna()
    df["premium"] = df["vix"] - df["rv_fwd"]
    return df


def describe(s: pd.Series, label: str) -> str:
    t = s.mean() / s.std(ddof=1) * np.sqrt(len(s)) if s.std(ddof=1) > 0 else np.nan
    return (f"{label:<18}{len(s):>5}{s.mean():>+8.2f}p{s.std(ddof=1):>8.2f}"
            f"{t:>7.2f}{(s < 0).mean():>7.0%}")


def main():
    df = premium_frame(BarStore("data/bars"))
    over = df["premium"]
    non = df["premium"].iloc[::HORIZON]

    print(f"VIX against the SPY vol that followed, {HORIZON}-session horizon")
    print(f"{df.index.min().date()} -> {df.index.max().date()}\n")

    t_o = over.mean() / over.std(ddof=1) * np.sqrt(len(over))
    t_n = non.mean() / non.std(ddof=1) * np.sqrt(len(non))
    print(f"{'sampling':<18}{'n':>5}{'mean':>9}{'sd':>8}{'t':>7}{'neg':>7}")
    print(describe(over, "overlapping"))
    print(describe(non, "NON-overlapping"))
    print(f"\n  t ratio {t_o/t_n:.2f} against sqrt(H)={np.sqrt(HORIZON):.2f} - "
          f"the overlap inflation, exactly as documented. Only the\n"
          f"  non-overlapping row is a measurement.\n")

    print(f"{'period':<18}{'n':>5}{'mean':>9}{'sd':>8}{'t':>7}{'neg':>7}")
    print(describe(non.iloc[:len(non)//2], "first half"))
    print(describe(non.iloc[len(non)//2:], "second half"))
    for y in sorted(set(non.index.year)):
        s = non[non.index.year == y]
        if len(s) >= 4:
            print(describe(s, str(y)))
    print(describe(non[non.index.year != 2022], "ex-2022"))

    print("\nthe tail, which is the product rather than a caveat:")
    for q in (0, 5, 10, 50, 90, 100):
        lab = "min" if q == 0 else "max" if q == 100 else f"p{q}"
        print(f"  {lab:>5}{np.percentile(non, q):>+9.2f}p")
    print(f"  negative in {int((non < 0).sum())} of {len(non)} windows; "
          f"worst {non.min():+.1f}p on {non.idxmin().date()}")

    try:
        from scipy import stats
        pos = int((non > 0).sum())
        p = stats.binomtest(pos, len(non), 0.5, alternative="greater").pvalue
        print(f"\nsign test (assumes no distribution): positive in {pos}/{len(non)}, "
              f"p={p:.2e}")
    except Exception:
        pass

    print("\nThis is a RISK PREMIUM, not an edge. It pays for holding the tail,")
    print("it is not a forecast, and it does not survive a vehicle whose spread")
    print("costs more than it pays - which is why TQQQ returned approximately")
    print("zero and SPY does not.")


if __name__ == "__main__":
    main()
