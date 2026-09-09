"""Do price levels act as resistance, or is that a selection effect?

Everything searched so far is unconditional - "given these features, what is the
return". Resistance is a conditional claim of a different shape: "IF price
reaches level L, THEN it behaves differently there". The feature sets only ever
saw the last twelve bars, so prior-day highs, multi-day extremes and round
numbers are genuinely untested here rather than ruled out.

The trap is that reaching a high IS a momentum event. Condition on "touched the
20-day high" and you have selected uptrends, so any result reflects momentum
rather than the level. Two controls handle it.

Round numbers carry their own placebo: $71.50 sits in the same structural place
as $71.00 without being round, so the difference between them isolates the level
from everything else. Limit orders genuinely cluster at round prices, which is a
real microstructure mechanism rather than folklore, so this is the version worth
testing first.

For the extreme-based levels there is no such twin, so events are stratified by
the move that produced them: within buckets of recent return and volatility,
touching the level is compared against not touching it. Anything left is the
level rather than the trend that reached it.
"""
import sys, pathlib, glob, os, time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np, pandas as pd

MAX_SYMBOLS = 300
FWD = 5


def load(path):
    sym = os.path.basename(path).split("_")[0]
    try:
        b = pd.read_parquet(path)
    except Exception:
        return None
    if len(b) < 1500 or not {"high", "low", "close"} <= set(b.columns):
        return None
    b = b.dropna(subset=["close", "high", "low"]).sort_index()
    b = b[b["close"] > 1.0]                    # cents distort roundness tests
    if len(b) < 1000:
        return None
    c = b["close"]
    out = pd.DataFrame(index=b.index)
    out["sym"] = sym
    out["close"], out["high"], out["low"] = c, b["high"], b["low"]
    out["fwd"] = np.log(c.shift(-FWD) / c)
    out["ret20"] = np.log(c / c.shift(20))
    out["vol20"] = np.log(c).diff().rolling(20).std()
    out["hi20"] = b["high"].rolling(20).max().shift(1)
    out["lo20"] = b["low"].rolling(20).min().shift(1)
    out["phigh"] = b["high"].shift(1)
    return out.dropna()


def stratified(df, touched, label):
    """Compare forward returns for touch vs no-touch, matched on how we got here.

    Buckets are recent 20-day return and 20-day volatility, both in quintiles.
    Comparing inside a bucket removes the momentum that produced the touch,
    which is the whole confound.
    """
    d = df.copy()
    d["touch"] = touched
    d["rq"] = pd.qcut(d["ret20"], 5, labels=False, duplicates="drop")
    d["vq"] = pd.qcut(d["vol20"], 5, labels=False, duplicates="drop")
    diffs, ns = [], []
    for _, g in d.groupby(["rq", "vq"], observed=True):
        a, b = g.loc[g["touch"], "fwd"], g.loc[~g["touch"], "fwd"]
        if len(a) < 50 or len(b) < 50:
            continue
        diffs.append(a.mean() - b.mean())
        ns.append(len(a))
    if not diffs:
        print(f"  {label:<28} (too few matched events)")
        return
    diffs, ns = np.asarray(diffs), np.asarray(ns)
    eff = float(np.average(diffs, weights=ns))
    se = float(np.std(diffs, ddof=1) / np.sqrt(len(diffs)))
    print(f"  {label:<28} {eff*1e4:>+8.1f} bps  +/-{1.96*se*1e4:>5.1f}   "
          f"{int(ns.sum()):>8,} touches, {len(diffs)} buckets")


def main():
    t0 = time.time()
    files = sorted(glob.glob("data/bars/*_1d.parquet"))[:MAX_SYMBOLS]
    df = pd.concat([d for d in (load(f) for f in files) if d is not None],
                   ignore_index=True)
    print(f"{df['sym'].nunique()} symbols, {len(df):,} sessions, "
          f"{FWD}-day forward returns\n")

    # ---- round numbers, with their own placebo -------------------------------
    # The day's high finishing just under a level is the resistance claim: buyers
    # ran into resting supply there and stopped. $X.00 is the level under test;
    # $X.50 is the same geometry without the roundness.
    print("ROUND NUMBERS - the day's high stalling just below a level")
    print(f"  {'level':<28} {'next 5 days':>13}   sample")
    hi = df["high"].to_numpy()
    for name, offset in (("whole dollar  $X.00", 0.0), ("placebo       $X.50", 0.5)):
        frac = np.abs(((hi - offset) % 1.0))
        near = np.minimum(frac, 1.0 - frac) < 0.0015 * df["close"].to_numpy()
        # "stalled below" = high reached the level and the close fell back under
        below = near & (df["close"].to_numpy() < hi - 1e-9)
        a = df["fwd"].to_numpy()[below]
        rest = df["fwd"].to_numpy()[~below]
        se = np.sqrt(a.var(ddof=1) / len(a) + rest.var(ddof=1) / len(rest))
        print(f"  {name:<28} {(a.mean()-rest.mean())*1e4:>+8.1f} bps  "
              f"+/-{1.96*se*1e4:>5.1f}   {len(a):>8,} events")

    # ---- extremes, stratified on the move that reached them ------------------
    print("\nEXTREMES - matched on recent return and volatility")
    print(f"  {'level touched'  :<28} {'next 5 days':>13}   sample")
    stratified(df, (df["high"] >= df["hi20"]).to_numpy(), "20-day high")
    stratified(df, (df["low"] <= df["lo20"]).to_numpy(), "20-day low")
    stratified(df, (df["high"] >= df["phigh"]).to_numpy(), "prior-day high")
    # Unstratified, to show what the confound is worth on its own.
    t = (df["high"] >= df["hi20"]).to_numpy()
    raw = df["fwd"].to_numpy()[t].mean() - df["fwd"].to_numpy()[~t].mean()
    print(f"\n  for scale: 20-day high WITHOUT matching reads {raw*1e4:+.1f} bps - "
          f"that gap is the momentum, not the level")
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
