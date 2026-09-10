"""What a resting limit order actually earns, once it is filled by someone.

Every cost model here charges 3.10 bps on the assumption we CROSS the spread.
Posting instead flips the sign: you buy at the bid rather than the ask, which is
a full spread better per round trip. The barrier model's +1.79 bps gross on the
ETF panel is -1.31 net if you cross and +4.33 if you earn, so this one number
decides whether any of it is tradeable.

The catch is adverse selection. A resting bid is filled by someone choosing to
sell into it, and they are more often right than not - so the price after a
passive fill drifts against you. The half spread captured is only worth having
if it exceeds that drift.

FILL MODEL, STATED PLAINLY. Tick data is available live but the endpoint serves
only the most recent ticks, with no historical query, so a large sample cannot
be assembled without polling across hours. This uses minute bars: an order at B
is treated as filled when a subsequent bar's low reaches B.

That model is OPTIMISTIC, and in the direction that matters. In reality a resting
order sits in a queue, so price touching your level does not fill you - the
touches that DO fill you are disproportionately the ones where price traded
decisively through, which are exactly the adversely selected ones. The touch-and-
bounce cases, where passive execution wins, are the ones a real queue would deny
you. So a negative result here is conclusive; a positive one is an upper bound.
"""
import sys, pathlib, time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np, pandas as pd

from bipbip.data.store import BarStore

TICK = 0.01
HORIZONS = (1, 5, 15, 30)


def main():
    t0 = time.time()
    b = BarStore("data/bars").load("TQQQ", "1m").dropna()
    day = pd.DatetimeIndex(b.index).normalize().view("int64")
    c = b["close"].to_numpy()
    lo, hi = b["low"].to_numpy(), b["high"].to_numpy()
    n = len(c)
    half = TICK / 2.0
    half_bps = half / np.mean(c) * 1e4
    print(f"TQQQ 1m: {n:,} bars, {b.index[0].date()} -> {b.index[-1].date()}")
    print(f"tick {TICK}, half spread {half_bps:.2f} bps at ${np.mean(c):.2f}")
    print(f"crossing costs {2*half_bps:.2f} bps per round trip; "
          f"posting both sides earns it instead\n")

    print(f"{'wait':>6} {'side':>5} {'fill rate':>10} {'captured':>10} "
          f"{'drift@15':>10} {'net':>9} {'vs crossing':>13}")
    for wait in (1, 5, 15, 30):
        for side, name in ((1, "buy"), (-1, "sell")):
            # Post at the touch: a bid half a tick under the last price, or an
            # offer half a tick over it.
            level = c - side * half
            filled = np.zeros(n, bool)
            fill_at = np.full(n, -1, np.int64)
            for k in range(1, wait + 1):
                j = np.minimum(np.arange(n) + k, n - 1)
                ok = (np.arange(n) + k < n) & (day[j] == day) & ~filled
                touch = ok & ((lo[j] <= level) if side > 0 else (hi[j] >= level))
                fill_at = np.where(touch, np.arange(n) + k, fill_at)
                filled |= touch
            H = 15
            j = fill_at
            ok = filled & (j + H < n) & (day[np.minimum(j + H, n - 1)] == day)
            if ok.sum() < 500:
                continue
            idx = j[ok]
            entry = level[ok]
            # Captured: how far inside the mid the fill was, in bps.
            cap = side * (c[idx] - entry) / entry * 1e4
            # Drift: where the mid went afterwards, signed by the position.
            drift = side * (c[idx + H] - c[idx]) / entry * 1e4
            net = cap + drift
            se = net.std(ddof=1) / np.sqrt(len(net))
            print(f"{wait:>5}m {name:>5} {filled.mean():>9.1%} "
                  f"{cap.mean():>+9.2f}b {drift.mean():>+9.2f}b "
                  f"{net.mean():>+8.2f}b {net.mean()+half_bps:>+12.2f}b")
    print("\n(net = half spread captured + drift over the following 15 minutes;")
    print(f" 'vs crossing' adds the {half_bps:.2f} bps you would have PAID instead)")

    print("\ndrift after a passive BUY fill, by horizon (1-minute wait):")
    level = c - half
    filled = (np.r_[lo[1:], np.nan] <= level) & (np.r_[day[1:], -1] == day)
    print(f"{'horizon':>9} {'n':>8} {'drift':>9} {'+/-95%':>9} {'net of capture':>16}")
    for H in HORIZONS:
        j = np.arange(n) + 1
        ok = filled & (j + H < n) & (day[np.minimum(j + H, n - 1)] == day)
        if ok.sum() < 500:
            continue
        i = j[ok]
        d = (c[i + H] - c[i]) / level[ok] * 1e4
        cap = (c[i] - level[ok]) / level[ok] * 1e4
        tot = d + cap
        se = 1.96 * d.std(ddof=1) / np.sqrt(len(d))
        print(f"{H:>8}m {ok.sum():>8,} {d.mean():>+8.2f}b {se:>8.2f} "
              f"{tot.mean():>+15.2f}b")
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
