"""If buying short-dated OTM calls is the wrong side, what does the right side pay?

The chain says every affordable out-of-the-money call carries a negative
expectation under any drift assumption, and the fifteen-year record says
volatility is what is predictable, not direction. Both point the same way: be
short the short-dated premium rather than long it.

Naked short calls are unbounded and unavailable on a small account, so this
prices the defined-risk version - sell one strike, buy a higher one - where the
most that can be lost is the width minus the credit, and that number is also the
margin the broker will require. Payoffs come from real two-sided quotes: the
credit is taken at the bid of the short leg and the ask of the long leg, which
is what an actual fill looks like rather than the mid.
"""
import sys, pathlib, glob

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np, pandas as pd

from bipbip.data.store import BarStore

SPOT = 71.365
MULT = 100


def main():
    s = BarStore("data/bars")
    d = s.load("TQQQ", "1d")["close"]
    d.index = pd.DatetimeIndex([pd.Timestamp(x.date()) for x in d.index])
    bars = s.load("TQQQ", "30m").dropna()
    r30 = np.log(bars["close"]).diff().dropna()
    day = pd.DatetimeIndex([pd.Timestamp(x.date()) for x in r30.index])
    vol = np.sqrt((r30 ** 2).groupby(day).sum().rolling(5).mean() * 252)
    cur = float(vol.dropna().iloc[-1])
    hist = float(np.log(d).diff().mean())

    ch = pd.read_parquet(sorted(glob.glob("data/chains/TQQQ_*.parquet"))[-1])
    ch["expiry"] = pd.to_datetime(ch["expiry"])
    calls = ch[(ch.side == "call") & (ch.bid > 0) & (ch.openInterest >= 100)]
    calls = calls.sort_values("strike").groupby(["expiry", "strike"]).first().reset_index()

    print(f"TQQQ {SPOT:.2f}, realised vol {cur:.1%}, "
          f"{len(calls)} liquid call strikes")
    print("short call spreads, credit taken at bid/ask (not mid), "
          "drift assumption +15%/yr\n")
    print(f"{'spread':>16} {'sess':>5} {'credit':>7} {'risk':>6} {'E[P&L]':>8} "
          f"{'+/-95%':>7} {'win':>6} {'on risk':>8} {'verdict':>12}")

    mu = np.log1p(0.15) / 252
    rows = []
    for exp, grp in calls.groupby("expiry"):
        lo = pd.Timestamp("2026-09-09") + pd.Timedelta(days=1)
        n = len(pd.bdate_range(lo, pd.Timestamp(exp).tz_localize(None).normalize()))
        if not 1 <= n <= 6:
            continue
        fwd = np.exp(np.log(d.shift(-n) / d) - hist * n + mu * n) - 1.0
        v = vol.reindex(d.index)
        m = ((v > cur * 0.85) & (v < cur * 1.15) & np.isfinite(fwd)).to_numpy()
        term = SPOT * (1.0 + fwd.to_numpy()[m])
        g = grp.set_index("strike")
        for short_k in g.index:
            if short_k < SPOT:                     # sell out of the money only
                continue
            for width in (1.0, 2.0, 3.0):
                long_k = short_k + width
                if long_k not in g.index:
                    continue
                credit = (g.loc[short_k, "bid"] - g.loc[long_k, "ask"]) * MULT
                risk = width * MULT - credit
                if credit <= 0 or risk <= 0:
                    continue
                loss = (np.minimum(np.maximum(term - short_k, 0.0), width) * MULT)
                pnl = credit - loss
                ci = 1.96 * pnl.std(ddof=1) / np.sqrt(len(pnl))
                verdict = ("EDGE" if pnl.mean() - ci > 0 else
                           "negative" if pnl.mean() + ci < 0 else "indistinct")
                rows.append((
                    f"{f'{short_k:.0f}/{long_k:.0f}':>16} {n:>5} "
                    f"{credit:>7.0f} {risk:>6.0f} {pnl.mean():>+8.1f} "
                    f"{ci:>7.1f} {np.mean(pnl > 0):>5.1%} "
                    f"{pnl.mean()/risk:>7.1%} {verdict:>12}",
                    pnl.mean() / risk))
    rows.sort(key=lambda r: -r[1])
    for text, _ in rows[:12]:
        print(text)
    print(f"  ... {len(rows)} spreads priced, "
          f"{sum(1 for _, v in rows if v > 0)} with positive expectation")
    print("\n('risk' is the maximum loss and the margin required; "
          "'on risk' is expected return on that margin per trade)")


if __name__ == "__main__":
    main()
