"""Resistance again, on everything, with the two controls it never had.

resistance.py answered its question with a placebo and got a clean null, but it
was run in a way that flatters ANY result, and on part of the archive:

  - It pooled every symbol into one two-sample t. A market-wide day puts all
    549 names in the same bucket; that is one event, not 549. CLAUDE.md has
    named this since the beginning and it roughly doubled the errors the one
    time it was measured.
  - It used a 5-day forward return at EVERY bar, so consecutive observations
    share four days out of five. Overlap inflates t by about sqrt(5).
  - It stopped at 300 symbols; there are 549.
  - It only ever ran on daily bars. Resting limit orders at round numbers are
    an intraday microstructure story, and the 20-ETF 30-minute panel - 15 years
    of it - was never pointed at this question.

So this is not simply "more data". Two of the four changes ADD power and two
REMOVE the spurious kind, and which wins is the thing being measured. If the
null was an artefact of a thin sample it should crack here; if it was real, the
errors widen and it stays null.

The event and its placebo are unchanged, deliberately: the day's high reaches a
level and the close falls back under it, with $X.00 the level under test and
$X.50 the same geometry stripped of roundness. Extremes are still compared
inside quintile buckets of recent return and volatility, but as fixed effects
in one clustered regression rather than an unweighted average of per-bucket
differences.

RESULT: nothing moved, and the rerun found the original test was broken.

THE ROUND-NUMBER TEST WAS NOT MEASURING ROUNDNESS. Its "near a level" condition
compares min(frac, 1-frac) - a distance from the nearest whole dollar, capped
at 0.50 by construction - against 0.0015 * close, a threshold in dollars. Above
$333 that threshold exceeds 0.50, so the condition fires on EVERY bar. The
event rate runs 2.7% below $20, 21% at $50-100, 51% at $100-333 and 100% above
it. It was a price-level selector wearing a roundness costume, and the -15 bps
it produced was the price selection, identically in both arms.

Worse, and independent of the threshold: roundness is about the number a human
types into an order ticket, and this archive is SPLIT-ADJUSTED. The inverse
levered funds have had large reverse splits, so SOXS's adjusted 2015 price is
$11 million and its maximum $8.6 billion. Those prices never traded. Testing
whether $11,522,411.17 is "near a round number" is not a microstructure test of
anything.

Fixed - a fixed 3-cent band, prices restricted to $5-$500 - roundness is flatly
nothing, which is what resistance.py concluded for the wrong reason:

    panel      real      placebo   real - placebo    t
    daily      +1.6b      +1.5b        +0.1b        0.02
    intraday   +4.4b      +0.6b        +3.8b        0.94

The placebo was doing its job all along. It caught a contaminated test and
reported the right answer, which is exactly what a structurally matched control
is for.

EXTREMES, THE ONE THING THAT SURVIVED, AND WHY IT STILL IS NOT AN EDGE.
Stalling at the 20-bar high is worth -13.7 bps at t=-2.95 on the daily panel,
and clustering plus non-overlapping sampling made it STRONGER rather than
weaker. Then it splits by period:

    1990-2004   -22.6 bps  t=-3.40
    2005-2012   -13.3 bps  t=-1.16
    2013-2019    -8.0 bps  t=-1.04
    2020-2026   -15.8 bps  t=-1.14

Every one of the last three decades is insignificant. The whole result is
carried by the earliest stretch, which is where survivorship is worst - the
archive is today's universe, everything delisted is missing, and this repo
measured that bias at 15-21 percentage points a year, LARGER than the effect
here and biasing toward exactly this sign. "Touching a high predicts lower
returns" is the fake mean reversion survivorship manufactures.

And on the twenty levered ETFs actually traded here it is -33.0 bps at t=-1.70
daily, and -1.1 bps at t=-0.24 intraday. The instruments this project trades
show nothing.

So the larger dataset moved nothing. It did surface a broken event definition
that had been sitting in a "standing result" for months.
"""
import sys, pathlib, glob, os, time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np, pandas as pd

from bipbip.core.stats import ols_cluster


def frame(b, sym, fwd, lookback=20):
    b = b.dropna(subset=["close", "high", "low"]).sort_index()
    b = b[b["close"] > 1.0]                    # cents distort roundness tests
    if len(b) < 1000:
        return None
    c = b["close"]
    out = pd.DataFrame(index=b.index)
    out["sym"] = sym
    out["close"], out["high"] = c, b["high"]
    out["fwd"] = np.log(c.shift(-fwd) / c) * 1e4
    out["ret20"] = np.log(c / c.shift(lookback))
    out["vol20"] = np.log(c).diff().rolling(lookback).std()
    # shift(1) so the level is one the bar could have been approaching, not one
    # its own high just set.
    out["hi20"] = b["high"].rolling(lookback).max().shift(1)
    out["date"] = [t.date() for t in b.index]
    return out.dropna()


def load_panel(pattern, fwd, limit=None):
    rows = []
    for p in sorted(glob.glob(pattern))[:limit]:
        try:
            b = pd.read_parquet(p)
        except Exception:
            continue
        if len(b) < 1500 or not {"high", "low", "close"} <= set(b.columns):
            continue
        f = frame(b, os.path.basename(p).split("_")[0], fwd)
        if f is not None:
            rows.append(f)
    return pd.concat(rows, ignore_index=True)


def thin(df, step):
    """One observation per holding period, keeping the whole cross-section.

    Dropping every non-multiple DATE rather than every non-multiple row keeps
    each surviving day's full panel intact, which is what the date clustering
    then treats as one event.
    """
    keep = set(sorted(df["date"].unique())[::step])
    return df[df["date"].isin(keep)]


def clustered(df, event, extra=None, label=""):
    y = df["fwd"].to_numpy()
    cols = [np.ones(len(df)), event.astype("float64")]
    if extra is not None:
        cols.extend(extra)
    r = ols_cluster(y, np.column_stack(cols), df["date"].to_numpy())
    return r["beta"][1], r["se"][1], r["t"][1], int(event.sum()), r["clusters"]


# A round number is what a human types into an order ticket, so the test only
# means anything on prices a human ever saw. Two filters enforce that: a sane
# band, and a fixed distance in CENTS rather than a fraction of price.
PRICE_LO, PRICE_HI = 5.0, 500.0
NEAR_CENTS = 0.03


def round_numbers(df, label):
    print(f"\n  ROUND NUMBERS - the high reaches a level, the close falls back under")
    print(f"    {'level':<22}{'effect':>10}{'se':>8}{'t':>7}{'events':>10}{'dates':>9}")
    out = {}
    df = df[(df["close"] >= PRICE_LO) & (df["close"] <= PRICE_HI)]
    hi, cl = df["high"].to_numpy(), df["close"].to_numpy()
    for name, off in (("whole dollar $X.00", 0.0), ("placebo      $X.50", 0.5)):
        frac = np.abs((hi - off) % 1.0)
        near = np.minimum(frac, 1.0 - frac) < NEAR_CENTS
        ev = near & (cl < hi - 1e-9)
        if ev.sum() < 200:
            print(f"    {name:<22} too few events ({ev.sum()})")
            continue
        b, se, t, n, g = clustered(df, ev)
        out[name] = (b, se)
        print(f"    {name:<22}{b:>+9.1f}b{se:>8.1f}{t:>7.2f}{n:>10,}{g:>9,}")
    if len(out) == 2:
        (b1, s1), (b2, s2) = out.values()
        d = b1 - b2
        sd = np.sqrt(s1 ** 2 + s2 ** 2)     # conservative: treats them as independent
        print(f"    {'real - placebo':<22}{d:>+9.1f}b{sd:>8.1f}{d/sd:>7.2f}"
              f"   <- roundness, isolated")


def extremes(df, label):
    """Touching the 20-bar high, compared inside return/volatility buckets."""
    d = df.copy()
    d["rq"] = pd.qcut(d["ret20"], 5, labels=False, duplicates="drop")
    d["vq"] = pd.qcut(d["vol20"], 5, labels=False, duplicates="drop")
    d = d.dropna(subset=["rq", "vq"])
    ev = (d["high"].to_numpy() >= d["hi20"].to_numpy()) & \
         (d["close"].to_numpy() < d["high"].to_numpy() - 1e-9)
    if ev.sum() < 200:
        print("\n  EXTREMES - too few events")
        return
    # Bucket fixed effects: the momentum that produced the touch is absorbed
    # rather than averaged over, and one clustered fit replaces 25 t-tests.
    bucket = (d["rq"] * 5 + d["vq"]).to_numpy().astype(int)
    dummies = [(bucket == k).astype("float64") for k in range(1, 25)]
    b, se, t, n, g = clustered(d, ev, dummies)
    print(f"\n  EXTREMES - stalling at the 20-bar high, inside return/vol buckets")
    print(f"    {'':<22}{'effect':>10}{'se':>8}{'t':>7}{'events':>10}{'dates':>9}")
    print(f"    {'20-bar high':<22}{b:>+9.1f}b{se:>8.1f}{t:>7.2f}{n:>10,}{g:>9,}")


def run(pattern, fwd, step, label, limit=None):
    t0 = time.time()
    df = load_panel(pattern, fwd, limit)
    full = len(df)
    df = thin(df, step)
    print(f"\n{'='*74}\n{label}")
    print(f"  {df['sym'].nunique()} symbols, {full:,} rows -> {len(df):,} after "
          f"thinning to 1 date in {step}, {fwd}-bar forward return")
    round_numbers(df, label)
    extremes(df, label)
    print(f"  [{time.time()-t0:.0f}s]")


def main():
    run("data/bars/*_1d.parquet", 5, 5, "DAILY - all symbols in the archive")
    # 13 thirty-minute bars is one session, so the overlap is WITHIN a day and
    # the date clustering already absorbs it. Only the spill into the next day
    # needs handling, and keeping every other date does that.
    run("data/bars/*_30m.parquet", 13, 2,
        "INTRADAY - 20 levered ETFs, 30-minute bars, never tested for this")


if __name__ == "__main__":
    main()
