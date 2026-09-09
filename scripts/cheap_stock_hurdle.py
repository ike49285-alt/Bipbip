"""Does a cheap, volatile stock beat SPY at short horizons?

The cost arithmetic that closed everything under five minutes was measured on
SPY and TQQQ, and the objection is fair: a share of SPY costs $766, so a move
worth catching is a lot of dollars, while a $12 stock moves a far larger
PERCENTAGE for the same tick.

The counter is that a spread is quoted in cents, not basis points. One cent on
SPY at $766 is 0.13 bps; the same cent on a $12 stock is 8.3 bps, sixty-four
times worse. Both effects are real and pull opposite ways, so the question is
which grows faster - and with minute bars for 304 symbols that is measurable
rather than arguable.

Spreads are ESTIMATED from the bars rather than assumed, using Roll's
covariance estimator: bid-ask bounce makes consecutive price changes negatively
autocorrelated, and the size of that covariance implies the spread. Assuming a
penny everywhere would beg the question in favour of cheap stocks, since they
are exactly the names that quote wider than a tick.
"""
import sys, pathlib, glob, os
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np, pandas as pd

from bipbip.data.sessions import restrict_to_rth
from bipbip.data.store import BarStore

SEC_FEE_BPS = 0.278          # charged on the sell leg only


def roll_spread_bps(close: pd.Series) -> float:
    """Roll (1984): effective spread implied by bid-ask bounce.

    Undefined where the covariance is positive - trending prices swamp the
    bounce - and NaN is returned rather than a fabricated number.
    """
    d = close.diff().dropna()
    if len(d) < 100:
        return float("nan")
    cov = float(np.cov(d.values[1:], d.values[:-1])[0, 1])
    if cov >= 0:
        return float("nan")
    return 2.0 * np.sqrt(-cov) / float(close.mean()) * 1e4


def analyse() -> pd.DataFrame:
    store = BarStore("data/bars")
    rows = []
    for f in sorted(glob.glob("data/bars/*_1m.parquet")):
        sym = os.path.basename(f).split("_")[0]
        try:
            b = restrict_to_rth(store.load(sym, "1m")).dropna()
        except Exception:
            continue
        if len(b) < 2000:
            continue
        px = b["close"]
        price = float(px.iloc[-1])
        spread = roll_spread_bps(px)
        if not np.isfinite(spread):
            continue
        spread = max(spread, 0.01 / price * 1e4)      # a tick is the floor
        rt = spread + SEC_FEE_BPS

        day = pd.Series(b.index.normalize(), index=b.index)
        rec = {"symbol": sym, "price": price, "spread_bps": spread,
               "round_trip_bps": rt}
        for h in (5, 30, 120):
            fwd = px.shift(-h) / px - 1.0
            r = fwd[day.shift(-h) == day].dropna()
            med = float((r.abs() * 1e4).median()) if len(r) > 200 else np.nan
            rec[f"move_{h}"] = med
            # Break-even hit rate at 2:1 reward-to-risk.
            rec[f"be2_{h}"] = ((med + rt) / (3 * med)) if med and med > 0 else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def main():
    df = analyse().dropna(subset=["move_30"])
    if df.empty:
        print("no symbols with enough minute history yet")
        return
    print(f"{len(df)} symbols with usable minute history\n")
    print("`be2` is the hit rate needed to break even at 2:1 reward-to-risk.")
    print("The best systematic strategies run 52-55%.\n")
    df["bucket"] = pd.cut(df["price"], [0, 20, 50, 150, 400, 1e9],
                          labels=["<$20", "$20-50", "$50-150", "$150-400", ">$400"])
    print(f"{'price':>10} {'n':>4} {'spread':>8} {'round trip':>11} "
          f"{'5m move':>9} {'be2 5m':>8} {'30m move':>9} {'be2 30m':>8} {'be2 2h':>8}")
    for name, g in df.groupby("bucket", observed=True):
        print(f"{str(name):>10} {len(g):>4} {g['spread_bps'].median():>7.1f}b "
              f"{g['round_trip_bps'].median():>10.1f}b "
              f"{g['move_5'].median():>8.1f}b {g['be2_5'].median()*100:>7.1f}% "
              f"{g['move_30'].median():>8.1f}b {g['be2_30'].median()*100:>7.1f}% "
              f"{g['be2_120'].median()*100:>7.1f}%")

    print("\nTen lowest 30-minute break-even rates:")
    print(f"{'symbol':>8} {'price':>9} {'spread':>8} {'30m move':>9} {'be2 30m':>9}")
    for _, r in df.nsmallest(10, "be2_30").iterrows():
        print(f"{r['symbol']:>8} {r['price']:>8.2f} {r['spread_bps']:>7.1f}b "
              f"{r['move_30']:>8.1f}b {r['be2_30']*100:>8.1f}%")

    spy = df[df["symbol"] == "SPY"]
    if not spy.empty:
        s = spy.iloc[0]
        print(f"\nSPY: ${s['price']:.2f}, spread {s['spread_bps']:.2f}b, "
              f"30m move {s['move_30']:.1f}b, be2 {s['be2_30']*100:.1f}%")
    print(f"\nCorrelation, share price vs 30-min break-even: "
          f"{df['price'].corr(df['be2_30']):+.2f}")
    print("(positive means expensive stocks are HARDER; negative, cheap ones are)")


if __name__ == "__main__":
    main()
