"""What the variance premium is worth in dollars, and what it can cost you.

`variance_premium.py` measured the premium in VOL POINTS: VIX against the
volatility SPY actually realised over the following 21 sessions, +3.32p at
t=4.15, stable across both halves and every year. Vol points are not money.
This turns them into a position: real strikes, real bid and ask from an
archived SPY chain, and the payoff evaluated against every non-overlapping
window in thirty-three years of SPY.

HORIZON. The archived chain carries only 7- and 8-day expiries, so this prices
the 6-session contract that actually exists rather than the 21-session one the
VIX study used. Stated rather than papered over: the two horizons are different
questions, and only this one has real quotes behind it.

THREE DECOMPOSITIONS THAT DECIDE WHETHER THIS IS A PREMIUM OR A DIRECTION BET.

  1. A short PUT spread is short the downside, and SPY drifts up. Its profit is
     therefore variance premium PLUS equity premium, and reporting the total as
     "selling vol works" would be claiming credit for drift. The de-drifted
     column removes the sample's mean return and leaves the variance part.
  2. A short CALL spread is short the upside, so drift works against it. The
     pair brackets the answer.
  3. The IRON CONDOR holds both, which nets most of the drift out by
     construction and is the closest thing here to a pure short-variance
     position with a capped loss.

TWO REGIMES, BECAUSE THE TAIL IS THE PRODUCT. Today's implied is 17.7% against
8.4% realised - a calm tape. Pricing today's options against thirty-three years
that include 2008 and March 2020 is not a forecast, it is a stress test, and it
is the one that matters for sizing. The vol-matched column answers "what if the
next week resembles this one"; the full-history column answers "what if it does
not". Ruin arithmetic uses the second.

SIZING. With a measured positive edge, Kelly is no longer zero for the first
time in this project - but the Kelly fraction of a capped-loss, fat-left-tail
payoff is small, and one spread is ONE BET. The output reports the growth-
optimal fraction and what a quarter of it implies at the current balance.

RESULT: the vol points do NOT become money, and the way they fail is the
useful part. SPY at 762.96, 6 sessions to 2026-09-17, 1,409 non-overlapping
windows since 1993 (459 vol-matched).

    iron condor       full history          vol-matched
    1.0% wide 5     -31.9 (-7.9% on risk)  +45.9 (+11.4%)
    1.0% wide 10    -68.5 (-8.2%)         +100.0 (+12.0%)
    2.0% wide 5     -41.5 (-8.9%)          +35.6  (+7.6%)
    3.0% wide 10    -60.6 (-6.1%)          +28.8  (+2.9%)

Every structure loses on the full sample and every one wins vol-matched, and
the gap between those columns is the whole finding: this is a bet on the regime
persisting, not on a premium.

THE VOL-MATCHED COLUMN IS NOT A BACKTEST, AND SHOULD NOT BE READ AS ONE. It
applies TODAY'S option prices to historical windows. Implied vol moves with the
regime, so in a historical calm window the option would have been priced
differently - probably cheaper, since today's 17.7% implied sits against 8.4%
trailing. The test therefore sells one specific, possibly rich, quote into a
distribution drawn from other periods. Doing it properly needs the implied vol
that prevailed at each historical date, which needs historical chains this
archive does not have. It survives a period split (early t=3.40, late t=2.87)
but a period split cannot fix a mis-specified comparison.

WHAT THE DE-DRIFT COLUMN CATCHES. Short PUT spreads look roughly break-even on
full history (-0.1% to -1.5% on risk) and their de-drifted P&L is -10 to -36
dollars. So the break-even is SPY's upward drift, not the variance premium -
the equity premium wearing a short-vol costume. Naming which of the three a
result is, before believing it, is exactly what that column is for.

THE TAIL, MEASURED. Worst window costs 64% to 95% of capital at risk, and the
condor loses in 55% of full-sample windows. Kelly on the full-sample
distribution is ZERO: the growth-optimal bet is not to place it. Even on the
favourable vol-matched arm, quarter-Kelly wants about $4,500 of capital behind
$403 of risk - against a balance of $2.10 and a planned $500.
"""
import sys, pathlib, glob, argparse

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np, pandas as pd

from bipbip.data.chain_features import implied_spot
from bipbip.data.store import BarStore

MULT = 100          # shares per contract
SEC_FEE_BPS = 0.278  # charged on the sale leg


def chain_and_spot(symbol="SPY"):
    files = sorted(glob.glob(f"data/chains/{symbol}_*.parquet"))
    if not files:
        raise SystemExit(f"no archived chain for {symbol}")
    ch = pd.read_parquet(files[-1])
    spot = implied_spot(ch)
    if not np.isfinite(spot):
        raise SystemExit("parity could not price the chain")
    asof = (pd.Timestamp(ch["fetched_at"].max())
            .tz_convert("America/New_York").tz_localize(None).normalize())
    ch["exp"] = pd.to_datetime(ch["expiry"]).dt.tz_localize(None)
    return ch, spot, asof, files[-1]


def forward_returns(closes: pd.Series, sessions: int) -> np.ndarray:
    """NON-OVERLAPPING forward returns over `sessions` bars.

    Overlapping windows would reuse the same tape `sessions` times and shrink
    every error bar below by roughly sqrt(sessions) on no new information.
    """
    r = closes.to_numpy(dtype="float64")
    idx = np.arange(0, len(r) - sessions, sessions)
    return r[idx + sessions] / r[idx] - 1.0


def spread_pnl(term_px, short_k, long_k, credit, kind):
    """P&L per contract at expiry, in dollars, for a SHORT vertical."""
    width = abs(long_k - short_k)
    if kind == "put":
        loss = np.minimum(np.maximum(short_k - term_px, 0.0), width)
    else:
        loss = np.minimum(np.maximum(term_px - short_k, 0.0), width)
    return credit - loss * MULT


def price_spread(g, short_k, long_k, kind):
    """Credit at the BID of the short leg and the ASK of the long leg."""
    try:
        s = g.loc[(kind, short_k)]
        l = g.loc[(kind, long_k)]
    except KeyError:
        return None
    credit = (float(s["bid"]) - float(l["ask"])) * MULT
    credit -= float(s["bid"]) * MULT * SEC_FEE_BPS / 1e4      # sale-side fee
    width = abs(long_k - short_k) * MULT
    risk = width - credit
    if credit <= 0 or risk <= 0:
        return None
    return credit, risk


def kelly_fraction(pnl, risk):
    """Growth-optimal fraction of capital to put at risk per trade.

    Solved numerically on the empirical payoff rather than assumed binomial,
    because the loss is capped but the distribution is nowhere near two-point.
    f is the share of capital risked; the position loses at most `risk`, so
    a bet of f puts f of the account on the worst case.
    """
    x = pnl / risk                      # return per unit of capital at risk
    if x.min() >= 0:
        return float("inf")
    lo, hi = 0.0, 0.999 / abs(x.min())
    for _ in range(200):
        mid = (lo + hi) / 2
        # derivative of E[log(1+f*x)]
        d = np.mean(x / (1.0 + mid * x))
        if d > 0:
            lo = mid
        else:
            hi = mid
    return float((lo + hi) / 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="SPY")
    ap.add_argument("--balance", type=float, default=2.10)
    a = ap.parse_args()

    ch, spot, asof, path = chain_and_spot(a.symbol)
    q = ch[(ch.bid > 0) & (ch.ask > 0)].copy()
    exp = sorted(q["exp"].unique())[-1]
    sess = int(np.busday_count(asof.date(), pd.Timestamp(exp).date()))
    g = (q[q["exp"] == exp]
         .drop_duplicates(subset=["side", "strike"])
         .set_index(["side", "strike"]).sort_index())

    d = BarStore("data/bars").load(a.symbol, "1d")["close"]
    fwd_all = forward_returns(d, sess)
    # Vol-matched: windows whose STARTING trailing vol resembles today's.
    r = np.log(d / d.shift(1))
    trail = (r.rolling(20).std() * np.sqrt(252)).to_numpy()
    idx = np.arange(0, len(d) - sess, sess)
    cur = trail[-1]
    keep = np.isfinite(trail[idx]) & (trail[idx] > cur * 0.7) & (trail[idx] < cur * 1.3)
    fwd_calm = fwd_all[keep]

    print(f"{a.symbol} {spot:.2f} (put-call parity), chain {pathlib.Path(path).name}")
    print(f"expiry {pd.Timestamp(exp).date()} = {sess} sessions; "
          f"trailing vol {cur:.1%}")
    print(f"history: {len(fwd_all):,} non-overlapping {sess}-session windows "
          f"({d.index[0].date()} -> {d.index[-1].date()}), "
          f"{len(fwd_calm):,} vol-matched\n")

    rows = []
    for kind, sgn in (("put", -1), ("call", +1)):
        for otm in (0.005, 0.01, 0.02, 0.03):
            for width in (5.0, 10.0):
                short_k = round((spot * (1 + sgn * otm)) / 5) * 5
                long_k = short_k + sgn * width
                p = price_spread(g, short_k, long_k, kind)
                if p is None:
                    continue
                credit, risk = p
                rows.append((kind, otm, short_k, long_k, credit, risk))

    if not rows:
        raise SystemExit("no priceable spreads at this expiry")

    print(f"{'structure':<22}{'credit':>8}{'risk':>8}{'E[P&L]':>9}{'+/-95':>8}"
          f"{'win':>7}{'on risk':>9}{'de-drift':>10}")
    for kind, otm, sk, lk, credit, risk in rows:
        for label, fwd in (("", fwd_all),):
            term = spot * (1 + fwd)
            pnl = spread_pnl(term, sk, lk, credit, kind)
            dd = spread_pnl(spot * (1 + fwd - fwd.mean()), sk, lk, credit, kind)
            ci = 1.96 * pnl.std(ddof=1) / np.sqrt(len(pnl))
            name = f"{kind} {sk:.0f}/{lk:.0f} ({otm:.1%})"
            print(f"{name:<22}{credit:>8.0f}{risk:>8.0f}{pnl.mean():>+9.1f}"
                  f"{ci:>8.1f}{(pnl > 0).mean():>7.0%}{pnl.mean()/risk:>8.1%}"
                  f"{dd.mean():>+10.1f}")

    # The condor: both wings, which nets most of the drift out by construction.
    puts = [r_ for r_ in rows if r_[0] == "put"]
    calls = [r_ for r_ in rows if r_[0] == "call"]
    if puts and calls:
        print(f"\n{'iron condor':<22}{'credit':>8}{'risk':>8}{'E[P&L]':>9}"
              f"{'+/-95':>8}{'win':>7}{'on risk':>9}{'de-drift':>10}")
        for (_, po, psk, plk, pc, pr) in puts:
            for (_, co, csk, clk, cc, cr) in calls:
                if abs(po - co) > 1e-9 or abs(plk - psk) != abs(clk - csk):
                    continue
                credit = pc + cc
                risk = max(pr, cr)      # only one wing can finish in the money
                for label, fwd in (("full history", fwd_all),
                                   ("vol-matched", fwd_calm)):
                    term = spot * (1 + fwd)
                    pnl = (spread_pnl(term, psk, plk, pc, "put")
                           + spread_pnl(term, csk, clk, cc, "call"))
                    dd_t = spot * (1 + fwd - fwd.mean())
                    dd = (spread_pnl(dd_t, psk, plk, pc, "put")
                          + spread_pnl(dd_t, csk, clk, cc, "call"))
                    ci = 1.96 * pnl.std(ddof=1) / np.sqrt(len(pnl))
                    name = f"{po:.1%} w{abs(plk-psk):.0f} {label}"
                    print(f"{name:<22}{credit:>8.0f}{risk:>8.0f}{pnl.mean():>+9.1f}"
                          f"{ci:>8.1f}{(pnl > 0).mean():>7.0%}"
                          f"{pnl.mean()/risk:>8.1%}{dd.mean():>+10.1f}")
                    if label == "full history":
                        _ruin(pnl, risk, a.balance)
                    else:
                        # The vol-matched arm is the only one with a positive
                        # expectation, so it is the one that has to survive a
                        # period split - the test that killed risk parity here.
                        h = len(pnl) // 2
                        for half, sl in (("early", pnl[:h]), ("late", pnl[h:])):
                            t = sl.mean()/sl.std(ddof=1)*np.sqrt(len(sl))
                            print(f"     {half:<6} n={len(sl):>4} "
                                  f"{sl.mean():>+8.1f}  t={t:>5.2f}")
                        _ruin(pnl, risk, a.balance)


def _ruin(pnl, risk, balance):
    f = kelly_fraction(pnl, risk)
    worst = pnl.min()
    print(f"     tail: worst {worst:+.0f} on {risk:.0f} at risk "
          f"({worst/risk:+.0%}), loses in {(pnl < 0).mean():.0%} of windows")
    if not np.isfinite(f) or f < 1e-4:
        print("     Kelly is zero on this distribution - the growth-optimal "
              "bet is not to place it")
        return
    need = risk / (f / 4)
    print(f"     Kelly {f:.1%}, quarter-Kelly {f/4:.1%} -> wants "
          f"${need:,.0f} behind ${risk:.0f} of risk; balance ${balance:,.2f}")


if __name__ == "__main__":
    main()
