"""How hard is it trending, and is there anything left in it?

Not a forecast. A state estimate: sort what is already happening by strength,
then look at what followed. If "already gone" is a real thing, the most extreme
bucket underperforms the merely-strong one and the relationship bends rather
than rising all the way.

Momentum is documented at one to twelve months, which is a horizon this project
has never touched - every direction search here ran between thirty minutes and
two hours. Testing it needs the daily archive, not the minute one.

Two honest requirements. Returns are compared against buy-and-hold, not against
zero, because a strategy that is long most of the time in a rising market beats
zero without doing anything. And observations are sampled one per holding period
so overlapping windows do not inflate the result.
"""
import sys, pathlib, glob, os, time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np, pandas as pd

MAX_SYMBOLS = 300
COST_BPS = 3.10


def load(path, look, hold):
    sym = os.path.basename(path).split("_")[0]
    try:
        b = pd.read_parquet(path)
    except Exception:
        return None
    if len(b) < look + hold + 400:
        return None
    c = b.dropna(subset=["close"]).sort_index()["close"]
    if len(c) < look + hold + 400:
        return None
    d = pd.DataFrame(index=c.index)
    d["sym"] = sym
    # Trend measured to one bar ago; the position is taken after seeing it.
    d["trend"] = np.log(c.shift(1) / c.shift(look + 1))
    # Strength normalised by the symbol's own volatility, so "hard" means hard
    # for THIS name rather than "is a levered ETF".
    d["vol"] = np.log(c).diff().rolling(look).std().shift(1)
    d["tstrength"] = d["trend"] / (d["vol"] * np.sqrt(look))
    d["fwd"] = np.log(c.shift(-hold) / c)
    d["i"] = np.arange(len(d))
    d["date_i"] = d.index
    return d.dropna()


def main():
    t0 = time.time()
    print("deciles of trend strength -> mean next-period return, in bps\n")
    files = sorted(glob.glob("data/bars/*_1d.parquet"))[:MAX_SYMBOLS]
    for look, hold, lab in ((21, 21, "1 month"), (63, 21, "3 months"),
                            (126, 21, "6 months"), (252, 21, "12 months")):
        frames = [d for d in (load(f, look, hold) for f in files) if d is not None]
        df = pd.concat(frames, ignore_index=True)
        # One observation per holding period per symbol: overlapping windows
        # would inflate every interval below by sqrt(hold).
        keep = (df["i"] % hold) == 0
        df = df[keep]
        df["dec"] = pd.qcut(df["tstrength"], 10, labels=False, duplicates="drop")
        bh = df["fwd"].mean()
        cells = df.groupby("dec")["fwd"].mean() * 1e4
        top, bot = df[df.dec == 9]["fwd"], df[df.dec == 0]["fwd"]
        ninth = df[df.dec == 8]["fwd"]
        se = np.sqrt(top.var(ddof=1) / len(top) + bot.var(ddof=1) / len(bot))
        spread = (top.mean() - bot.mean()) * 1e4
        # Clustered by DATE. A market-wide selloff puts every symbol in the weak
        # decile at once and the rebound is one event, not one per symbol, so
        # treating rows as independent understates the interval badly. The
        # spread is formed cross-sectionally on each date and the interval comes
        # from how those dates vary.
        per = df.groupby("date_i").apply(
            lambda g: (g.loc[g.dec == 9, "fwd"].mean()
                       - g.loc[g.dec == 0, "fwd"].mean()), include_groups=False)
        per = per.dropna()
        cse = float(per.std(ddof=1) / np.sqrt(len(per))) * 1e4
        print(f"\n{lab:>9} {hold:>4}d   n={len(df):>7,}   "
              f"buy-and-hold {bh*1e4:+.0f} bps over the same window")
        print("   " + "  ".join(f"{v:+.0f}" for v in cells))
        print(f"   weakest decile {cells.iloc[0]:+.0f}   "
              f"9th {cells.iloc[8]:+.0f}   strongest {cells.iloc[9]:+.0f}")
        print(f"   strongest minus weakest: {spread:+.0f} bps   "
              f"naive +/-{1.96*se*1e4:.0f}   "
              f"clustered by date +/-{1.96*cse:.0f} ({len(per)} dates)")
        # "Already gone": does the very top give back against the merely-strong?
        se2 = np.sqrt(top.var(ddof=1) / len(top) + ninth.var(ddof=1) / len(ninth))
        gone = (top.mean() - ninth.mean()) * 1e4
        print(f"   strongest minus 9th: {gone:+.0f} bps +/-{1.96*se2*1e4:.0f}"
              f"   {'(exhaustion)' if gone + 1.96*se2*1e4 < 0 else '(no exhaustion)'}")
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
