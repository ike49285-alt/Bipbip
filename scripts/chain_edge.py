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
import sys, pathlib, glob, argparse

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np, pandas as pd

from bipbip.data.chain_features import implied_spot
from bipbip.data.store import BarStore

MULT = 100
#: Realised-vol band the comparable-week sample is drawn from, either side of
#: the anchor.
BAND = 0.15

#: NYSE closures, which are NOT the federal calendar: the exchange trades on
#: Columbus Day and Veterans Day and closes on Good Friday. An incomplete list
#: overstates the sessions to expiry, which lengthens the horizon every payoff
#: is scored over.
US_MARKET_HOLIDAYS = {
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25",
    "2026-06-19", "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25",
    "2027-01-01", "2027-01-18", "2027-02-15", "2027-03-26", "2027-05-31",
    "2027-06-18", "2027-07-05", "2027-09-06", "2027-11-25", "2027-12-24",
}


def sessions_to_expiry(today, expiry) -> int:
    """Trading sessions strictly after today, up to and including expiry.

    Counted forward off the business-day calendar rather than off the archive:
    the archive ends yesterday, so every future expiry would otherwise score
    zero sessions and the whole chain would be silently skipped.
    """
    lo = pd.Timestamp(today).tz_localize(None).normalize() + pd.Timedelta(days=1)
    hi = pd.Timestamp(expiry).tz_localize(None).normalize()
    days = pd.bdate_range(lo, hi)
    return int(sum(d.strftime("%Y-%m-%d") not in US_MARKET_HOLIDAYS for d in days))


def chain_files(symbol="TQQQ"):
    files = sorted(glob.glob(f"data/chains/{symbol}_*.parquet"))
    if not files:
        raise SystemExit(f"no archived chain for {symbol}")
    return files


def anchor_for(vol: pd.Series, asof: pd.Timestamp) -> float:
    """Realised vol as of the chain's own quote date, not the archive's tail.

    Reading the last bar instead re-anchors the comparable-week sample every
    time new bars land, and can read a bar from AFTER the quote. Same defect,
    same fix, as scripts/spread_edge.py.
    """
    v = vol.dropna()
    v = v[v.index <= asof]
    if v.empty:
        raise SystemExit(f"no realised vol on or before {asof.date()}")
    return float(v.iloc[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain", default=None,
                    help="PIN one snapshot; default is the newest, printed")
    ap.add_argument("--anchor", type=float, default=None)
    a = ap.parse_args()

    s = BarStore("data/bars")
    d = s.load("TQQQ", "1d")["close"]
    d.index = pd.DatetimeIndex([pd.Timestamp(x.date()) for x in d.index])
    bars = s.load("TQQQ", "30m").dropna()
    r30 = np.log(bars["close"]).diff().dropna()
    day = pd.DatetimeIndex([pd.Timestamp(x.date()) for x in r30.index])
    vol = np.sqrt((r30 ** 2).groupby(day).sum().rolling(5).mean() * 252)
    drift = float(np.log(d).diff().mean())

    f = a.chain or chain_files("TQQQ")[-1]
    ch = pd.read_parquet(f)
    ch["expiry"] = pd.to_datetime(ch["expiry"])
    # Spot from the chain's OWN put-call parity. The hardcoded live mid that
    # stood here was 71.365, taken on the 9th, while the newest snapshot is
    # from the 10th where parity says 69.213 - $2.15 out. That gap does not
    # merely shift moneyness: it FABRICATES intrinsic value, and every call
    # from the 40 to the 67 strike was consequently scored as "cheap" at up to
    # +73% of premium, because its ask sat below an intrinsic value that only
    # existed at the wrong spot. None of them is below intrinsic at parity.
    # CLAUDE.md records a separately-fetched quote being 20 cents off; a
    # constant left behind by a day is two dollars off.
    spot = implied_spot(ch)
    if not np.isfinite(spot):
        raise SystemExit(f"parity could not price {f}")
    # fetched_at is UTC and an evening pull lands on the NEXT UTC day, so it is
    # converted to exchange time before the trading date is taken - otherwise
    # every session count is short by one.
    today = (pd.Timestamp(ch["fetched_at"].max())
             .tz_convert("America/New_York").tz_localize(None).normalize())
    cur = a.anchor if a.anchor is not None else anchor_for(vol, today)
    calls = ch[(ch.side == "call") & (ch.bid > 0) & (ch.ask > 0)
               & (ch.openInterest >= 100)].copy()
    # An option cannot trade below intrinsic value - CLAUDE.md's own
    # no-arbitrage rule - so an ask under it is a stale or broken quote, not an
    # opportunity. Scoring them is how a spot error turns into a table of
    # free money, and the deep-in-the-money strikes are exactly where quotes go
    # stale. Dropped and counted rather than silently kept.
    intrinsic = np.maximum(0.0, spot - calls["strike"])
    below = calls["ask"] < intrinsic
    n_below = int(below.sum())
    calls = calls[~below]
    calls["n"] = [sessions_to_expiry(today, e) for e in calls["expiry"]]
    calls = calls[calls["n"].between(1, 8)]

    print(f"TQQQ spot {spot:.3f} (put-call parity, {today.date()}), "
          f"realised vol (intraday+overnight) {cur:.1%}, "
          f"sample drift {np.expm1(drift*252):+.1%}/yr")
    print(f"chain {pathlib.Path(f).name}, {len(calls)} liquid calls 1-8 "
          f"sessions out ({n_below} dropped for quoting below intrinsic)\n")
    print(f"{'strike':>7} {'sess':>5} {'bid':>5} {'ask':>5} {'IV':>6} {'OTM':>7} "
          f"{'spread':>7} {'E[P&L]':>8} {'+/-95%':>7} {'win':>6} {'verdict':>12}")

    for _, row in calls.sort_values(["n", "strike"]).iterrows():
        n, k = int(row["n"]), float(row["strike"])
        fwd = (np.exp(np.log(d.shift(-n) / d) - drift * n) - 1.0)
        v = vol.reindex(d.index)
        m = ((v > cur * (1 - BAND)) & (v < cur * (1 + BAND))
             & np.isfinite(fwd)).to_numpy()
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
