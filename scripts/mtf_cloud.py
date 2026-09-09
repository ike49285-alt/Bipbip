"""Does a higher-timeframe cloud cap a lower-timeframe rally?

Every Ichimoku test in this repo has been single-timeframe: a cloud computed on
30-minute bars, fed to a model predicting 30-minute returns. "The 30-minute
cloud stopped the 5-minute rally" is a different claim entirely, and nothing run
here could have detected it.

The event: price approaches the higher-timeframe cloud from BELOW and gets
within a small distance of its underside. If the cloud is resistance, the
forward return after that approach should be worse than usual.

The null is the point of the exercise, and it is built the same way the round
number test got its answer. A placebo cloud is the real cloud series displaced
by a large, arbitrary number of bars: identical construction, identical
smoothness and scale, no causal relationship to where price actually is. If
approaching the placebo produces the same reversal, the cloud is not what is
doing the work - the approach is, and price approaching ANY level from below is
already a rally that has to stop somewhere.
"""
import sys, pathlib, time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np, pandas as pd

from bipbip.core.indicators import ichimoku
from bipbip.data.store import BarStore

# 20 bps is nothing on an instrument whose typical 30-minute move is 105 bps,
# so "approached" has to be scaled to how far this thing actually travels.
NEAR = 0.01
FWD = 13              # roughly one session at 30-minute bars


def higher_tf(b: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {"open": "first", "high": "max", "low": "min",
           "close": "last", "volume": "sum"}
    return b.resample(rule).agg(agg).dropna()


def test(b, rule, label):
    hi = higher_tf(b, rule)
    if len(hi) < 200:
        print(f"  {label:<22} (too little history)")
        return
    ich = ichimoku(hi)
    # Forward-fill the higher-timeframe cloud onto the 30-minute grid, then lag
    # one higher bar: a cloud value is only usable once its bar has CLOSED.
    bottom = ich["cloud_bottom"].shift(1).reindex(b.index, method="ffill")
    c = b["close"]
    fwd = np.log(c.shift(-FWD) / c)

    def arm(level, name):
        below = c < level
        near = (level - c) / c < NEAR
        # The approach: below the level, close to it, and rising into it.
        event = (below & near & (c > c.shift(3))).to_numpy() & np.isfinite(fwd)
        rest = (~event) & np.isfinite(fwd).to_numpy()
        a, r = fwd.to_numpy()[event], fwd.to_numpy()[rest]
        if len(a) < 100:
            return ("few", int(event.sum()))
        se = np.sqrt(a.var(ddof=1) / len(a) + r.var(ddof=1) / len(r))
        return (a.mean() - r.mean()) * 1e4, 1.96 * se * 1e4, len(a)

    real = arm(bottom, "real")
    # Placebo: the same series, displaced far enough that its alignment with
    # today's price is arbitrary.
    placebo = arm(bottom.shift(997), "placebo")
    if real is None or placebo is None or real[0] == "few" or placebo[0] == "few":
        n1 = real[1] if real and real[0] == "few" else "?"
        n2 = placebo[1] if placebo and placebo[0] == "few" else "?"
        print(f"  {label:<22} too few approaches (real {n1}, placebo {n2})")
        return
    print(f"  {label:<22} real {real[0]:>+7.1f} +/-{real[1]:>5.1f} "
          f"({real[2]:>6,})   placebo {placebo[0]:>+7.1f} +/-{placebo[1]:>5.1f} "
          f"({placebo[2]:>6,})   diff {real[0]-placebo[0]:>+6.1f}")


def main():
    t0 = time.time()
    b = BarStore("data/bars").load("TQQQ", "30m").dropna()
    print(f"TQQQ 30m, {len(b):,} bars {b.index[0].date()}->{b.index[-1].date()}")
    print(f"approaching a higher-timeframe cloud from below, "
          f"forward return over {FWD} bars (bps vs all other bars)\n")
    for rule, label in (("2h", "2-hour cloud"), ("1D", "daily cloud"),
                        ("1W", "weekly cloud")):
        test(b, rule, label)
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
