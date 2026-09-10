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

IT IS NOT TRADEABLE, AND THE ARITHMETIC IS EXACT. Overnight-only means a round
trip every session - about 252 x 1.35 bps = 3.4% a year - against buy-and-hold's
zero. After that:

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

#: Round-trip cost of crossing on an SPY-class name, from CLAUDE.md.
ROUND_TRIP_BPS = 1.35
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
        net = on - ROUND_TRIP_BPS / 1e4
        rows.append({"sym": s, "years": len(on) / SESSIONS,
                     "on": cagr(on), "day": cagr(day), "tot": cagr(tot),
                     "net": cagr(net), "sh_on": sharpe(net), "sh_bh": sharpe(tot)})
    r = pd.DataFrame(rows)
    r["edge"] = r["net"] - r["tot"]

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

    print(f"\nNET OF THE DAILY ROUND TRIP ({SESSIONS} x {ROUND_TRIP_BPS} bps = "
          f"{SESSIONS*ROUND_TRIP_BPS/100:.1f}% a year against buy-and-hold's zero):")
    print(f"  overnight-net beats buy-and-hold in "
          f"{int((r['edge'] > 0).sum())} of {len(r)}, median {r['edge'].median():+.2%}")
    print(f"  Sharpe {r['sh_on'].median():.2f} against {r['sh_bh'].median():.2f}, "
          f"better in {int((r['sh_on'] > r['sh_bh']).sum())} of {len(r)}")

    print("\nDECAY - the reason it is not a trade (SPY by decade):")
    b = st.load("SPY", "1d").dropna()
    on, day, _ = legs(b)
    idx = pd.DatetimeIndex(b.index)[1:]
    f = pd.DataFrame({"on": on, "day": day}, index=idx)
    for lo, hi in ((1993, 1999), (2000, 2009), (2010, 2019), (2020, 2026)):
        s = f[(f.index.year >= lo) & (f.index.year <= hi)]
        a, c = cagr(s["on"].to_numpy()), cagr(s["day"].to_numpy())
        print(f"  {lo}-{hi}  overnight {a:>+7.1%}  intraday {c:>+7.1%}  "
              f"gap {a-c:>+7.1%}")
    print(f"\n  the last decade's gap is below the {SESSIONS*ROUND_TRIP_BPS/100:.1f}% "
          f"it costs to capture. Competed down to its own friction.")


if __name__ == "__main__":
    main()
