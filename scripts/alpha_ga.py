"""Let the search invent its own indicators on 20 ETFs of minute bars.

Twenty liquid ETFs - levered index long and short, broad index, sectors - give
the search a cross-section to rank rather than a single series to time, which is
the difference between "is TQQQ going up" (a bet on the market) and "is TQQQ
going up more than SPY" (a bet on something specific).

Evolution runs on the first two thirds of the tape and the winner is scored once
on the last third, which it never saw. Then the whole procedure is repeated on
shuffled labels: a search over expression trees will always return a champion,
and the only way to know whether this one means anything is to see what the same
machinery produces when the answers are noise.
"""
import sys, pathlib, glob, os, time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np, pandas as pd

from bipbip.data.store import BarStore
from bipbip.ml.symbolic import AlphaSearch, evaluate

BASKET = ["TQQQ", "SQQQ", "SPY", "QQQ", "IWM", "DIA", "SOXL", "SOXS", "TNA",
          "TZA", "SPXL", "SPXS", "UPRO", "SPXU", "LABU", "XLF", "XLE", "XLK",
          "XLV", "XLU"]
SEC_FEE_BPS = 0.278


def load(horizon: int):
    st = BarStore("data/bars")
    frames, costs = {}, {}
    for sym in BASKET:
        try:
            b = st.load(sym, "1m").dropna()
        except Exception:
            continue
        if len(b) < 5000 or b.index[0].year < 2026:
            continue          # skip anything still on the old 30-day Yahoo pull
        frames[sym] = b
        # Half a tick each way on the symbol's own price, plus the sale fee.
        tick = 0.01 / float(b["close"].iloc[-1]) * 1e4
        costs[sym] = tick + SEC_FEE_BPS

    idx = None
    for b in frames.values():
        idx = b.index if idx is None else idx.intersection(b.index)
    syms = sorted(frames)
    panel = {f: np.column_stack([frames[s][f].reindex(idx).to_numpy()
                                 for s in syms]).astype("float64")
             for f in ("open", "high", "low", "close", "volume")}
    o, c = panel["open"], panel["close"]
    entry = np.roll(o, -1, axis=0)
    exit_ = np.roll(c, -horizon, axis=0)
    fwd = exit_ / entry - 1.0
    fwd[-(horizon + 1):] = np.nan
    day = pd.DatetimeIndex(idx).normalize().to_numpy()
    same = np.roll(day, -horizon) == day
    fwd[~same] = np.nan
    return panel, fwd, np.array([costs[s] for s in syms]), idx, syms


def main():
    t0 = time.time()
    horizon = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    panel, fwd, cost, idx, syms = load(horizon)
    T = len(idx)
    print(f"{len(syms)} ETFs, {T:,} timestamps x {len(syms)} = "
          f"{T*len(syms):,} bars, {idx[0].date()} -> {idx[-1].date()}")
    print(f"holding {horizon} min, cost {cost.mean():.2f} bps/leg average\n")

    split = int(T * 0.66)
    train = np.arange(0, split, horizon)
    test = np.arange(split + horizon + 30, T, horizon)
    print(f"evolve on {len(train):,} rebalances, score once on {len(test):,} held out\n")

    # What the best of N random searches scores when there is nothing to find.
    # sqrt(2 ln N) is the expected maximum of N standard normals, so an
    # in-sample t below this floor is not a weak result - it is worse than
    # picking at random and keeping the luckiest pick.
    n_eval = 20 * 180
    floor = np.sqrt(2 * np.log(n_eval))
    print(f"search evaluates ~{n_eval:,} expressions; the best of that many "
          f"coin flips scores t={floor:.2f} on its own\n")

    ga = AlphaSearch(panel, fwd, cost, horizon, seed=0)
    best, fit = ga.evolve(train, generations=20, population=180)
    real = ga.book(evaluate(best, panel), test)
    rt = real.mean() / real.std(ddof=1) * np.sqrt(len(real)) if len(real) > 1 else np.nan
    print(f"best expression   {best}")
    print(f"  in-sample fitness t={fit:.2f}")
    print(f"  HELD OUT: {real.mean():+.2f} bps/rebalance  t={rt:.2f}  "
          f"n={len(real):,}  win {(real > 0).mean():.1%}\n")

    print("the same search, run against shuffled labels:")
    nulls = []
    for s in range(1, 6):
        rng = np.random.default_rng(100 + s)
        shuffled = fwd.copy()
        # Shuffle whole timestamp rows, which destroys any link to the features
        # while leaving each bar's cross-section intact - so the null keeps the
        # basket's real correlation structure and only the answers are wrong.
        perm = rng.permutation(len(shuffled))
        shuffled = shuffled[perm]
        gn = AlphaSearch(panel, shuffled, cost, horizon, seed=s)
        bn, fn = gn.evolve(train, generations=20, population=180)
        rn = gn.book(evaluate(bn, panel), test)
        tn = rn.mean() / rn.std(ddof=1) * np.sqrt(len(rn)) if len(rn) > 1 else np.nan
        nulls.append(tn)
        print(f"  null {s}: in-sample t={fn:>6.2f}   held out t={tn:>6.2f}   {bn}")

    nulls = [x for x in nulls if np.isfinite(x)]
    beat = sum(rt > x for x in nulls)
    print(f"\nheld-out t={rt:.2f} against nulls "
          f"mean {np.mean(nulls):.2f}, best {max(nulls):.2f}")
    print(f"in-sample t={fit:.2f} against a null floor of {floor:.2f}")
    print("VERDICT:", "separable from the search's own overfitting"
          if beat == len(nulls) and rt > 2 else
          "NOT separable - this is what the search finds in noise")
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
