"""What the $75 5DTE TQQQ call would have paid across fifteen years.

SUPERSEDED, AND ITS CONTRACT IS KNOWN TO BE WRONG. The strike and quotes below
were read off a screenshot, and the collected chain later showed the quoted
0.51/0.59 belongs to the $74 strike rather than the $75 - so this prices a
contract that did not exist at that price. `scripts/chain_edge.py` asks the
same question against real strikes and real two-sided quotes and is what should
be run instead. This is kept because the DE-DRIFTING and the estimator argument
below are the durable part, and they are what that script inherited.

Read the conditioning volatility it prints as well. The contract constants are
frozen on the day they were typed while the anchor tracks the archive, so the
sample this scores against drifts away from the quote every time bars land -
the pattern CLAUDE.md now records as a trap.

No option-pricing model: for every comparable week, take TQQQ's real forward
move, apply the contract's real payoff, compare against the real premium.

Two things decide whether the answer means anything.

Conditioning has to use a volatility estimate that is not mostly noise. Five
daily closes give one so noisy that its week-to-week wobble (0.639) is nearly
twice the intraday estimator's (0.336), and picking "weeks like today" with it
selects high-volatility weeks that flatter a call buyer. The estimator here sums
squared thirty-minute returns and keeps the overnight gap, because the contract
is exposed to the overnight move even though it is not a thirty-minute one.

And the sample has to be de-drifted. TQQQ compounded roughly 45% a year over
these fifteen years. A long call inherits every bit of that, so an unadjusted
backtest measures the drift and calls it an edge. Both versions are reported: as
it happened, and with the drift removed.
"""
import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np, pandas as pd

from bipbip.data.store import BarStore

SPOT, STRIKE = 70.86, 75.0
BID, ASK = 0.51, 0.59
MID = (BID + ASK) / 2
MULT = 100
BARS = 13


def realised_vol_daily(bars: pd.DataFrame) -> pd.Series:
    """Per-session realised variance from thirty-minute bars, overnight included."""
    r = np.log(bars["close"]).diff().dropna()
    day = pd.DatetimeIndex([pd.Timestamp(x.date()) for x in r.index])
    var = (r ** 2).groupby(day).sum()
    return np.sqrt(var.rolling(5).mean() * 252)


def payoff(fwd: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, SPOT * (1.0 + fwd) - STRIKE) * MULT


def line(name, fwd, prem):
    pay = payoff(fwd)
    pnl = pay - prem
    se = pnl.std(ddof=1) / np.sqrt(len(pnl)) if len(pnl) > 1 else np.nan
    print(f"  {name:<34} n={len(fwd):>5,}  worthless {np.mean(pay == 0):>5.1%}  "
          f"mean {pnl.mean():>+7.2f} +/- {1.96*se:>5.2f}  "
          f"({pnl.mean()/prem:>+6.1%} of premium)  win {np.mean(pnl > 0):>5.1%}")


def main():
    s = BarStore("data/bars")
    d = s.load("TQQQ", "1d")["close"]
    d.index = pd.DatetimeIndex([pd.Timestamp(x.date()) for x in d.index])
    vol = realised_vol_daily(s.load("TQQQ", "30m").dropna())
    # The archive's last row, NOT the day the constants above were quoted -
    # which is unrecoverable, because they came from a live intraday quote
    # rather than a close. Printed with the date so the gap is visible instead
    # of silent; a script whose contract is frozen and whose sample is not is
    # answering a slightly different question every time it runs.
    anchor_date = vol.dropna().index[-1]
    cur = float(vol.dropna().iloc[-1])

    lr = np.log(d).diff()
    drift = lr.mean()
    print(f"TQQQ {d.index[0].date()} -> {d.index[-1].date()}, "
          f"drift {np.expm1(drift*252):+.1%}/yr")
    print(f"contract: ${STRIKE:.0f} call, spot ${SPOT:.2f} "
          f"({STRIKE/SPOT-1:+.1%} OTM), bid {BID:.2f} / ask {ASK:.2f}")
    print(f"conditioning volatility (intraday + overnight): {cur:.1%} "
          f"as of {anchor_date.date()} (the archive's last session, not the "
          f"quote date; see the module docstring)\n")

    for days, lab in ((4, "4 sessions"), (5, "5 sessions")):
        raw = (d.shift(-days) / d - 1.0)
        # Same path with the sample's own compounding removed, so what remains
        # is the contract's payoff rather than fifteen years of a rising market.
        flat = (np.exp(np.log(d.shift(-days) / d) - drift * days) - 1.0)
        v = vol.reindex(d.index)
        like = ((v > cur * 0.85) & (v < cur * 1.15)).to_numpy()
        for prem, plab in ((MID * MULT, "mid"), (ASK * MULT, "ask")):
            print(f"{lab} to expiry, paying the {plab} (${prem/MULT:.3f}):")
            line("all weeks, as it happened", raw.dropna().to_numpy(), prem)
            line("all weeks, drift removed", flat.dropna().to_numpy(), prem)
            fr, ff = raw.to_numpy(), flat.to_numpy()
            m = like & np.isfinite(fr)
            line(f"weeks near {cur:.0%} vol, as it happened", fr[m], prem)
            line(f"weeks near {cur:.0%} vol, drift removed", ff[like & np.isfinite(ff)], prem)
            print()

    be = (STRIKE + ASK) / SPOT - 1
    f5 = (d.shift(-5) / d - 1.0).dropna()
    print(f"the underlying must gain {be:+.2%} by expiry to break even at the ask;")
    print(f"it did that in 5 sessions {np.mean(f5 > be):.1%} of weeks since 2011.")


if __name__ == "__main__":
    main()
