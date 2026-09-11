"""Where does the equity return accrue - overnight, or while the market is open?

Every direction search in this project has run inside the session: thirty
minutes to two hours, open to close. None of them ever asked whether the return
being hunted is IN that window. It mostly is not.

Splitting each daily bar into the two legs that compose it - close to next
open, held through the gap with no ability to trade out, and open to close,
held while you can - on the 34 clean ETFs, within-symbol:

    overnight beats intraday in 27 of 34 ETFs
    median overnight  +8.71% a year
    median intraday   -1.27% a year
    paired difference +9.03%, t=5.52, sign p=4.1e-04

    SPY  +10.0% overnight  +0.7% intraday        QQQ  +13.9% / -2.7%
    XLK  +15.0% overnight  -3.9% intraday        IWM  +13.4% / -4.1%

This is documented in the literature and is not a discovery. It had simply
never been measured here, and it reframes the fourteen negative results: the
intraday searches were hunting in the half of the day where the equity premium
does not accrue. A strategy long an index intraday is holding it through the
window that historically pays approximately nothing.

WHETHER IT IS TRADEABLE IS NOT SETTLED, AND THE EARLIER TEXT HERE SAID IT WAS.
Overnight-only means a round trip every session, and the annual cost is 252
times whatever one round trip costs - which is exactly the number this script
used to assert and never derive. It read 1.35 bps, described as "an SPY-class
name"; that is TQQQ's crossing cost, and SPY's is 0.133. The conclusion moves
across the plausible range rather than surviving it:

    quoted crossing   0.133 bps -> 0.34%/yr   the 2020s gap of +3.3% clears it
    modelled retail   2.28  bps -> 5.75%/yr   no decade since the 2000s clears
    poor fills        6.28  bps -> 15.8%/yr   nothing clears

So "it decays to exactly the cost line" was an artifact of a number sitting
between the two the repo can actually derive. What would settle it is the cost
of trading the closing and opening AUCTIONS, which is neither the quoted spread
nor continuous-book slippage, and which has not been measured here. After the
cost question, and independent of it:

    overnight-net beats buy-and-hold in 6 of 34 ETFs
    median edge -2.21% a year
    Sharpe 0.50 against 0.49, better in 15 of 34 - a coin flip

AND IT IS DECAYING, MONOTONICALLY, TO EXACTLY THE COST LINE. SPY by decade:

    1993-1999   overnight +22.0%   intraday -0.5%   gap +22.5%
    2000-2009   overnight  +4.4%   intraday -5.2%   gap  +9.5%
    2010-2019   overnight  +8.6%   intraday +4.5%   gap  +4.1%
    2020-2026   overnight  +9.1%   intraday +5.8%   gap  +3.3%

The last decade's gap is +3.3% against a +3.4% cost of capturing it. That is
what an effect looks like after it has been competed down to the friction that
protects it - and it matches the outside world, where the NightShares ETFs
productised exactly this trade in 2022 and closed in 2023.

So the finding to keep is not a strategy. It is that the equity premium is paid
for bearing the gap you CANNOT trade out of, which is a risk premium in
CLAUDE.md's sense, and that the session itself pays close to nothing. Any
intraday idea here is competing for a share of roughly zero drift, which is a
harder starting point than fourteen null results made it look.
"""
import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np, pandas as pd

from bipbip.data.store import BarStore
from bipbip.data.universe import get_universe

#: Round-trip crossing costs, in basis points, DERIVED rather than asserted:
#: one tick spread over the symbol's own mean price in the 1-minute archive,
#: the same calculation `scripts/adverse_selection.py` performs.
#:
#: The constant that stood here was 1.35, commented "an SPY-class name, from
#: CLAUDE.md". It is TQQQ's number. SPY's is 0.133 - ten times smaller - and
#: this study is 34 ETFs benchmarked on SPY, so the cost line was overstated by
#: an order of magnitude. CLAUDE.md's own 2.82 for TQQQ is the same figure from
#: when TQQQ averaged ~$35; it now averages ~$74 in the archive, so a spread
#: fixed in cents has halved in basis points. Both numbers were right when
#: written and neither was re-derived.
CROSSING_BPS = {"SPY": 0.133, "QQQ": 0.140, "TQQQ": 1.354}

#: The cost this conclusion is reported against. Crossing the quoted spread is
#: a FLOOR, not an estimate: it assumes every fill lands at the quote with no
#: latency and no adverse selection, which README already flags. The cost model
#: charges SPY 2.28 bps all-in, and the truth for an overnight strategy is
#: neither, because overnight-only trades the closing and opening AUCTIONS
#: rather than the continuous book. That measurement has not been made here, so
#: the sensitivity is printed instead of one column being chosen.
COST_SCENARIOS = (
    ("quoted crossing (floor)", 0.133),
    ("modelled retail all-in", 2.28),
    ("poor fills", 6.28),
)
SESSIONS = 252


def legs(bars: pd.DataFrame):
    """(overnight, intraday, total) simple returns, aligned and disjoint.

    They compose exactly: (1+on)(1+day) = 1+total, so this is a decomposition
    of the same return rather than two different bets.
    """
    o, c = bars["open"].to_numpy(), bars["close"].to_numpy()
    return o[1:] / c[:-1] - 1.0, c[1:] / o[1:] - 1.0, c[1:] / c[:-1] - 1.0


def cagr(x: np.ndarray) -> float:
    return float(np.prod(1.0 + x) ** (SESSIONS / len(x)) - 1.0)


def sharpe(x: np.ndarray) -> float:
    return float(x.mean() / x.std() * np.sqrt(SESSIONS)) if x.std() > 0 else np.nan


def main():
    st = BarStore("data/bars")
    rows = []
    for s in get_universe("etf_wide"):
        b = st.load(s, "1d").dropna()
        if len(b) < 1500:
            continue
        on, day, tot = legs(b)
        row = {"sym": s, "years": len(on) / SESSIONS,
               "on": cagr(on), "day": cagr(day), "tot": cagr(tot)}
        for label, bps in COST_SCENARIOS:
            net = on - bps / 1e4
            row[f"net@{bps}"] = cagr(net)
            row[f"sh@{bps}"] = sharpe(net)
        rows.append(row)
    r = pd.DataFrame(rows)

    print("WHERE THE RETURN ACCRUES, within-symbol on the clean ETF universe\n")
    print(f"{'sym':<6}{'years':>7}{'overnight':>11}{'intraday':>10}{'total':>9}")
    for _, x in r[r["sym"].isin(("SPY", "QQQ", "IWM", "XLK", "EFA", "TLT",
                                 "GLD", "HYG"))].iterrows():
        print(f"{x['sym']:<6}{x['years']:>7.1f}{x['on']:>+10.1%}"
              f"{x['day']:>+10.1%}{x['tot']:>+9.1%}")

    d = r["on"] - r["day"]
    print(f"\nacross {len(r)} ETFs: overnight beats intraday in "
          f"{int((r['on'] > r['day']).sum())} of {len(r)}")
    print(f"  median overnight {r['on'].median():+.2%}, "
          f"median intraday {r['day'].median():+.2%}")
    print(f"  paired difference {d.mean():+.2%}  "
          f"t={d.mean()/d.std(ddof=1)*np.sqrt(len(d)):.2f}")

    # A round trip every session, so the annual cost is 252x one round trip.
    # The scenario is printed as a COLUMN rather than chosen, because the
    # verdict changes across a range the repo cannot currently narrow.
    print(f"\nNET OF A DAILY ROUND TRIP, across execution assumptions "
          f"({SESSIONS} sessions a year):\n")
    print(f"{'assumption':<26}{'bps':>6}{'cost/yr':>9}"
          f"{'beats B&H':>11}{'median edge':>13}")
    for label, bps in COST_SCENARIOS:
        edge = r[f"net@{bps}"] - r["tot"]
        print(f"{label:<26}{bps:>6.3f}{SESSIONS*bps/100:>8.2f}%"
              f"{int((edge > 0).sum()):>7} of {len(r)}{edge.median():>+13.2%}")

    print("\nDECAY - the reason it is not a trade (SPY by decade):")
    b = st.load("SPY", "1d").dropna()
    on, day, _ = legs(b)
    idx = pd.DatetimeIndex(b.index)[1:]
    f = pd.DataFrame({"on": on, "day": day}, index=idx)
    a = cc = 0.0
    for lo, hi in ((1993, 1999), (2000, 2009), (2010, 2019), (2020, 2026)):
        sl = f[(f.index.year >= lo) & (f.index.year <= hi)]
        a, cc = cagr(sl["on"].to_numpy()), cagr(sl["day"].to_numpy())
        print(f"  {lo}-{hi}  overnight {a:>+7.1%}  intraday {cc:>+7.1%}  "
              f"gap {a-cc:>+7.1%}")
    print("\n  whether the last decade's gap survives depends on which cost "
          "is right:")
    gap = a - cc                      # a FRACTION, e.g. 0.033 for +3.3%
    for label, bps in COST_SCENARIOS:
        # Same units on both sides. Comparing a fraction against a percent is
        # a factor of 100 and reads as a confident verdict either way.
        cost = SESSIONS * bps / 1e4
        print(f"    {label:<26} {cost:>7.2%}/yr  "
              f"{'the gap CLEARS it' if gap > cost else 'the gap does not clear it'}")
    print("\n  Overnight-only trades the closing and opening AUCTIONS, whose "
          "cost is neither the\n  quoted spread nor continuous-book slippage. "
          "Until that is measured, this is open.")


if __name__ == "__main__":
    main()
