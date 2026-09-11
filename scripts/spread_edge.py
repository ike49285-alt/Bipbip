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

READ THE SENSITIVITY TABLE BEFORE THE HEADLINE. The historical sample is
VOL-MATCHED: forward moves are drawn only from days whose realised volatility
was within +/-15% of an anchor. That anchor decides the answer outright, and
the table at the bottom prints the whole curve rather than one column of it,
because this script's committed result did not survive its own anchor moving.

What that construction actually measures is worth naming, since CLAUDE.md asks
for the return source before the number. The CREDIT comes from today's quotes,
so it carries today's IMPLIED volatility. The LOSSES come from historical days
matched on REALISED volatility. The difference between those two is the variance
risk premium, which this repo has already measured directly at +3.32 points and
already shown does not convert into money once a structure pays its own spread.
So a positive number here is not a discovery - it is that same premium arriving
by a side door, and it is largest exactly when implied has not yet followed
realised down.
"""
import sys, pathlib, glob, argparse

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np, pandas as pd

from bipbip.data.chain_features import implied_spot
from bipbip.data.store import BarStore

MULT = 100
#: The realised-vol band the historical sample is drawn from, either side of
#: the anchor. Widening it buys sample size and loses the matching.
BAND = 0.15


def chain_files(symbol="TQQQ"):
    files = sorted(glob.glob(f"data/chains/{symbol}_*.parquet"))
    if not files:
        raise SystemExit(f"no archived chain for {symbol}")
    return files


def realised_vol(store: BarStore, symbol: str) -> pd.Series:
    """Five-day realised volatility per session, annualised, from 30m bars."""
    bars = store.load(symbol, "30m").dropna()
    r30 = np.log(bars["close"]).diff().dropna()
    day = pd.DatetimeIndex([pd.Timestamp(x.date()) for x in r30.index])
    return np.sqrt((r30 ** 2).groupby(day).sum().rolling(5).mean() * 252)


def anchor_for(vol: pd.Series, asof: pd.Timestamp) -> float:
    """Realised vol AS OF THE DAY THE CHAIN WAS QUOTED, not the archive's tail.

    This is a correctness fix and not only a reproducibility one. The quotes
    being priced are from date X, so the regime they should be matched against
    is the regime standing on date X. Reading the last bar in the archive
    instead silently re-anchors the whole sample every time new bars land, and
    that is what stopped this script reproducing its own committed result: the
    answer is a MONOTONIC function of this number, running from 48 of 53
    spreads positive at an anchor of 0.25 to none of them at 0.55.
    """
    v = vol.dropna()
    v = v[v.index <= asof]
    if v.empty:
        raise SystemExit(f"no realised vol on or before {asof.date()}")
    return float(v.iloc[-1])


def price_chain(chain_file, d, vol, hist, cur, drift=0.15):
    """Every defined-risk short call spread in one snapshot, at one anchor."""
    ch = pd.read_parquet(chain_file)
    spot = implied_spot(ch)
    if not np.isfinite(spot):
        raise SystemExit(f"parity could not price {chain_file}")
    # The reference date likewise comes from the snapshot rather than a literal.
    # fetched_at is UTC and an evening pull lands on the NEXT UTC day, so it is
    # converted to exchange time before the trading date is taken.
    asof = (pd.Timestamp(ch["fetched_at"].max())
            .tz_convert("America/New_York").tz_localize(None).normalize())
    ch["expiry"] = pd.to_datetime(ch["expiry"])
    calls = ch[(ch.side == "call") & (ch.bid > 0) & (ch.openInterest >= 100)]
    calls = calls.sort_values("strike").groupby(
        ["expiry", "strike"]).first().reset_index()

    mu = np.log1p(drift) / 252
    rows, sample = [], 0
    for exp, grp in calls.groupby("expiry"):
        lo = asof + pd.Timedelta(days=1)
        n = len(pd.bdate_range(lo, pd.Timestamp(exp).tz_localize(None).normalize()))
        if not 1 <= n <= 6:
            continue
        fwd = np.exp(np.log(d.shift(-n) / d) - hist * n + mu * n) - 1.0
        v = vol.reindex(d.index)
        m = ((v > cur * (1 - BAND)) & (v < cur * (1 + BAND))
             & np.isfinite(fwd)).to_numpy()
        sample = max(sample, int(m.sum()))
        term = spot * (1.0 + fwd.to_numpy()[m])
        g = grp.set_index("strike")
        for short_k in g.index:
            if short_k < spot:                     # sell out of the money only
                continue
            for width in (1.0, 2.0, 3.0):
                long_k = short_k + width
                if long_k not in g.index:
                    continue
                credit = (g.loc[short_k, "bid"] - g.loc[long_k, "ask"]) * MULT
                risk = width * MULT - credit
                if credit <= 0 or risk <= 0:
                    continue
                loss = np.minimum(np.maximum(term - short_k, 0.0), width) * MULT
                pnl = credit - loss
                ci = 1.96 * pnl.std(ddof=1) / np.sqrt(len(pnl))
                rows.append({
                    "short": short_k, "long": long_k, "sess": n,
                    "credit": credit, "risk": risk, "mean": pnl.mean(),
                    "ci": ci, "win": float(np.mean(pnl > 0)),
                    "on_risk": pnl.mean() / risk,
                    "verdict": ("EDGE" if pnl.mean() - ci > 0 else
                                "negative" if pnl.mean() + ci < 0 else
                                "indistinct"),
                })
    return pd.DataFrame(rows), spot, asof, sample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="TQQQ")
    ap.add_argument("--chain", default=None,
                    help="PIN one snapshot. Without it every archived snapshot "
                         "is swept, because taking the newest makes the answer "
                         "depend on which file happens to be latest.")
    ap.add_argument("--anchor", type=float, default=None,
                    help="override the realised-vol anchor; default is the "
                         "vol standing on the chain's own quote date")
    ap.add_argument("--drift", type=float, default=0.15)
    a = ap.parse_args()

    s = BarStore("data/bars")
    d = s.load(a.symbol, "1d")["close"]
    d.index = pd.DatetimeIndex([pd.Timestamp(x.date()) for x in d.index])
    vol = realised_vol(s, a.symbol)
    hist = float(np.log(d).diff().mean())

    files = [a.chain] if a.chain else chain_files(a.symbol)

    print(f"{a.symbol}: {len(files)} snapshot(s), credit at bid/ask (not mid), "
          f"drift +{a.drift:.0%}/yr, vol band +/-{BAND:.0%}\n")
    print(f"{'snapshot':>26} {'spot':>7} {'anchor':>7} {'sample':>7} {'n':>4} "
          f"{'pos':>4} {'EDGE':>5} {'best on risk':>13}")

    last = None
    for f in files:
        cur = a.anchor if a.anchor is not None else None
        ch_asof = None
        if cur is None:
            # Peek at the snapshot's date to anchor on it.
            probe = pd.read_parquet(f, columns=["fetched_at"])
            ch_asof = (pd.Timestamp(probe["fetched_at"].max())
                       .tz_convert("America/New_York").tz_localize(None).normalize())
            cur = anchor_for(vol, ch_asof)
        df, spot, asof, sample = price_chain(f, d, vol, hist, cur, a.drift)
        if df.empty:
            print(f"{pathlib.Path(f).stem:>26} {spot:>7.2f} {cur:>7.3f} "
                  f"{sample:>7} {0:>4} {'-':>4} {'-':>5} {'-':>13}")
            continue
        print(f"{pathlib.Path(f).stem:>26} {spot:>7.2f} {cur:>7.3f} {sample:>7} "
              f"{len(df):>4} {int((df['on_risk'] > 0).sum()):>4} "
              f"{int((df['verdict'] == 'EDGE').sum()):>5} "
              f"{df['on_risk'].max():>12.1%}")
        last = (f, df, spot, asof, cur)

    if last is None:
        raise SystemExit("no snapshot priced any spread")

    f, df, spot, asof, cur = last
    print(f"\nbest twelve in {pathlib.Path(f).stem} "
          f"(spot {spot:.3f} by put-call parity, {asof.date()}, "
          f"anchor {cur:.1%}):\n")
    print(f"{'spread':>16} {'sess':>5} {'credit':>7} {'risk':>6} {'E[P&L]':>8} "
          f"{'+/-95%':>7} {'win':>6} {'on risk':>8} {'verdict':>12}")
    for _, r in df.sort_values("on_risk", ascending=False).head(12).iterrows():
        tag = f"{r['short']:.0f}/{r['long']:.0f}"
        print(f"{tag:>16} {int(r['sess']):>5} {r['credit']:>7.0f} "
              f"{r['risk']:>6.0f} {r['mean']:>+8.1f} {r['ci']:>7.1f} "
              f"{r['win']:>5.1%} {r['on_risk']:>7.1%} {r['verdict']:>12}")

    # The anchor is not a detail, so it is not reported as one. The sign of the
    # whole table is a monotonic function of this column; printing a single
    # value of it is what let an earlier version of this script publish
    # "selling is approximately zero" and then, on the same chain, "EDGE".
    print(f"\nsensitivity to the vol anchor, {pathlib.Path(f).stem}:")
    print(f"{'anchor':>8} {'sample':>7} {'n':>4} {'pos':>4} {'EDGE':>5} "
          f"{'best on risk':>13}")
    for probe in (0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60):
        sub, _, _, ns = price_chain(f, d, vol, hist, probe, a.drift)
        if sub.empty:
            continue
        print(f"{probe:>8.2f} {ns:>7} {len(sub):>4} "
              f"{int((sub['on_risk'] > 0).sum()):>4} "
              f"{int((sub['verdict'] == 'EDGE').sum()):>5} "
              f"{sub['on_risk'].max():>12.1%}")
    print("\n('risk' is the maximum loss and the margin required; "
          "'on risk' is expected return on that margin per trade.)")
    print("The credit carries TODAY'S implied vol and the losses come from days "
          "matched on\nREALISED vol, so a positive column is the variance risk "
          "premium arriving by a side\ndoor - already measured directly here at "
          "+3.32 points, and already shown not to\nconvert once a structure "
          "pays its own spread. It is not a separate finding.")


if __name__ == "__main__":
    main()
