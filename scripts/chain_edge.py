"""Score every liquid short-dated TQQQ call against fifteen years of real moves.

Reconstructing a contract from a screenshot guesses at the strike and at how
many sessions "5 DTE" contains. The collected chain has neither problem: it
carries the real strikes, the real two-sided quotes, and the real expiry, and
the sessions to expiry are counted off the archive's own trading calendar
rather than assumed.

For each contract, take TQQQ's actual return over that many sessions in every
comparable week since 2011 and apply the contract's payoff. Weeks are matched on
volatility measured from thirty-minute bars with the overnight gap included,
because that is the exposure being priced. The sample's own drift is removed:
TQQQ compounded about 45% a year here, and a long call inherits all of it.
"""
import sys, pathlib, glob

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np, pandas as pd

from bipbip.data.store import BarStore

MULT = 100
SPOT_NOW = (71.36 + 71.37) / 2      # live L1 quote


US_HOLIDAYS_2026 = {"2026-11-26", "2026-12-25", "2026-11-27"}


def sessions_to_expiry(today, expiry) -> int:
    """Trading sessions strictly after today, up to and including expiry.

    Counted forward off the business-day calendar rather than off the archive:
    the archive ends yesterday, so every future expiry would otherwise score
    zero sessions and the whole chain would be silently skipped.
    """
    lo = pd.Timestamp(today).tz_localize(None).normalize() + pd.Timedelta(days=1)
    hi = pd.Timestamp(expiry).tz_localize(None).normalize()
    days = pd.bdate_range(lo, hi)
    return int(sum(d.strftime("%Y-%m-%d") not in US_HOLIDAYS_2026 for d in days))


def main():
    s = BarStore("data/bars")
    d = s.load("TQQQ", "1d")["close"]
    d.index = pd.DatetimeIndex([pd.Timestamp(x.date()) for x in d.index])
    bars = s.load("TQQQ", "30m").dropna()
    r30 = np.log(bars["close"]).diff().dropna()
    day = pd.DatetimeIndex([pd.Timestamp(x.date()) for x in r30.index])
    vol = np.sqrt((r30 ** 2).groupby(day).sum().rolling(5).mean() * 252)
    cur = float(vol.dropna().iloc[-1])
    drift = float(np.log(d).diff().mean())
    spot = SPOT_NOW      # live mid, not the archive's stale last close

    f = sorted(glob.glob("data/chains/TQQQ_*.parquet"))[-1]
    ch = pd.read_parquet(f)
    ch["expiry"] = pd.to_datetime(ch["expiry"])
    today = pd.Timestamp(ch["fetched_at"].max()).normalize()
    calls = ch[(ch.side == "call") & (ch.bid > 0) & (ch.ask > 0)
               & (ch.openInterest >= 100)].copy()
    calls["n"] = [sessions_to_expiry(today, e) for e in calls["expiry"]]
    calls = calls[calls["n"].between(1, 8)]

    print(f"TQQQ spot {spot:.2f}, realised vol (intraday+overnight) {cur:.1%}, "
          f"sample drift {np.expm1(drift*252):+.1%}/yr")
    print(f"chain {pathlib.Path(f).name}, {len(calls)} liquid calls 1-8 sessions out\n")
    print(f"{'strike':>7} {'sess':>5} {'bid':>5} {'ask':>5} {'IV':>6} {'OTM':>7} "
          f"{'spread':>7} {'E[P&L]':>8} {'+/-95%':>7} {'win':>6} {'verdict':>12}")

    for _, row in calls.sort_values(["n", "strike"]).iterrows():
        n, k = int(row["n"]), float(row["strike"])
        fwd = (np.exp(np.log(d.shift(-n) / d) - drift * n) - 1.0)
        v = vol.reindex(d.index)
        m = ((v > cur * 0.85) & (v < cur * 1.15) & np.isfinite(fwd)).to_numpy()
        if m.sum() < 100:
            continue
        pay = np.maximum(0.0, spot * (1.0 + fwd.to_numpy()[m]) - k) * MULT
        ask = row["ask"] * MULT
        e_ask = pay.mean() - ask
        # A payoff this skewed has a fat standard error; without it a +5% row
        # reads as an opportunity when it is indistinguishable from zero.
        ci = 1.96 * pay.std(ddof=1) / np.sqrt(len(pay))
        verdict = ("OVERPRICED" if e_ask + ci < 0 else
                   "cheap" if e_ask - ci > 0 else "indistinct")
        print(f"{k:>7.0f} {n:>5} {row['bid']:>5.2f} {row['ask']:>5.2f} "
              f"{row['impliedVolatility']:>5.1%} {k/spot-1:>+6.1%} "
              f"{(row['ask']-row['bid'])/((row['ask']+row['bid'])/2):>6.1%} "
              f"{e_ask/ask:>+8.1%} {ci/ask:>7.1%} "
              f"{np.mean(pay > ask):>5.1%} {verdict:>12}")

    print("\n(E[P&L] as a fraction of premium, over comparable weeks since 2011, "
          "drift removed. 'win' = finishes above the ask.)")


if __name__ == "__main__":
    main()
