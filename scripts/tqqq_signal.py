"""Look for a 30-minute signal on TQQQ, with the search separated from the test.

The minute archive holds 21 sessions of TQQQ - 252 independent 30-minute
windows, enough to resolve only a 6.3 bps edge against a 1.67 bps cost. A real
5 bps signal would be invisible in it. The hourly archive holds 730 sessions
and 5,069 independent bars, enough to resolve 4 bps.

So the hourly data does the SEARCHING and the minute data does the TESTING.
Anything found hourly is checked at 30 minutes on bars the search never saw,
which is the only out-of-sample either dataset can supply.

Every hypothesis is written down before any of them runs and the threshold is
corrected for the whole battery.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np, pandas as pd

from bipbip.core import indicators as ind
from bipbip.data.sessions import restrict_to_rth
from bipbip.data.store import BarStore

COST_BPS = 1.67          # measured from the live quote: penny wide on $72


def features(bars):
    h, l, c, v = (bars["high"], bars["low"],
                  bars["close"], bars["volume"])
    ha = ind.heikin_ashi(bars)
    st = ind.full_stochastic(bars, 14, 1, 3)
    ich = ind.ichimoku(bars, 9, 26, 52, 26)
    lr = np.log(c / c.shift(1))
    day = pd.Series(bars.index.normalize(), index=bars.index)
    f = pd.DataFrame({
        "ret_1": lr,
        "ret_3": np.log(c / c.shift(3)),
        "ret_6": np.log(c / c.shift(6)),
        "vol_20": lr.rolling(20, min_periods=20).std(),
        "vol_ratio": (lr.rolling(6, min_periods=6).std()
                      / lr.rolling(30, min_periods=30).std()),
        "range_pos": (c - l) / (h - l).replace(0, np.nan),
        "ha_run": ha["ha_run"],
        "ha_trend": ha["ha_trend"],
        "stoch_k": st["stoch_k"],
        "stoch_kd": st["stoch_k"] - st["stoch_d"],
        "ich_cloud": (c - ich["cloud_top"]) / c,
        "ich_tk": (ich["tenkan"] - ich["kijun"]) / c,
        "vol_rel": v / v.rolling(30, min_periods=30).mean(),
    }, index=bars.index)
    return f, day


def battery(f):
    """Declared in advance. Each entry maps to a +1/-1/0 position."""
    q = lambda s, p: s.quantile(p)
    tests = {
        # Reversion, which is the one effect measured as real intraday.
        "fade last bar": -np.sign(f.ret_1),
        "fade last bar, big only": -np.sign(f.ret_1) * (f.ret_1.abs() > q(f.ret_1.abs(), 0.8)),
        "fade 3-bar move": -np.sign(f.ret_3),
        "fade 6-bar move": -np.sign(f.ret_6),
        # Momentum, the opposite bet.
        "follow last bar": np.sign(f.ret_1),
        "follow 6-bar move": np.sign(f.ret_6),
        # Heikin-Ashi, as actually used.
        "buy after long red HA run": (f.ha_run <= -3).astype(float),
        "sell after long green HA run": -(f.ha_run >= 3).astype(float),
        "follow HA colour": f.ha_trend.fillna(0),
        # Stochastic.
        "buy oversold stoch": (f.stoch_k < 20).astype(float),
        "sell overbought stoch": -(f.stoch_k > 80).astype(float),
        "follow stoch crossover": np.sign(f.stoch_kd),
        # Ichimoku.
        "long above cloud": (f.ich_cloud > 0).astype(float),
        "short below cloud": -(f.ich_cloud < 0).astype(float),
        "follow tenkan-kijun": np.sign(f.ich_tk),
        # Volatility and volume conditioning.
        "long when quiet": (f.vol_ratio < q(f.vol_ratio, 0.2)).astype(float),
        "fade on high volume": -np.sign(f.ret_1) * (f.vol_rel > 1.5).astype(float),
        "buy weak close": (f.range_pos < 0.2).astype(float),
        "sell strong close": -(f.range_pos > 0.8).astype(float),
    }
    return tests


def run(bars, horizon, label, spacing=None):
    f, day = features(bars)
    c = bars["close"]
    fwd = (c.shift(-horizon) / c.shift(0) - 1.0) * 1e4
    fwd = fwd.where(day.shift(-horizon) == day)
    tests = battery(f)
    n_tests = len(tests)
    crit = 3.0 if n_tests <= 20 else 3.3
    step = spacing or horizon

    print(f"\n{'=' * 76}\n{label}\n{'=' * 76}")
    print(f"{len(tests)} hypotheses declared in advance. Bonferroni |t| > {crit:.1f}. "
          f"Cost {COST_BPS} bps.")
    print(f"{'hypothesis':<30} {'n':>6} {'net bps':>9} {'t':>7} {'hit':>6}")
    rows = []
    for name, sig in tests.items():
        s = sig.reindex(f.index).fillna(0.0)
        # Non-overlapping: take every `step`-th bar so windows do not share data.
        idx = np.arange(0, len(f), step)
        ss, ff = s.iloc[idx], fwd.iloc[idx]
        m = (ss != 0) & ff.notna()
        if m.sum() < 60:
            continue
        pnl = np.sign(ss[m]) * ff[m] - COST_BPS
        t = pnl.mean() / pnl.std() * np.sqrt(len(pnl)) if pnl.std() > 0 else 0.0
        rows.append((name, len(pnl), pnl.mean(), t, (pnl > 0).mean()))
        flag = "  <<<" if abs(t) > crit else ""
        print(f"{name:<30} {len(pnl):>6,} {pnl.mean():>+8.2f}b {t:>7.2f} "
              f"{(pnl > 0).mean():>5.0%}{flag}")
    return rows, crit


def main():
    store = BarStore("data/bars")
    hourly = store.load("TQQQ", "1h").dropna()
    rows, crit = run(hourly, 1, "SEARCH: TQQQ hourly, 1-bar horizon, 730 sessions")

    survivors = [r for r in rows if abs(r[3]) > crit]
    print()
    if not survivors:
        best = max(rows, key=lambda r: abs(r[3]))
        print(f"Nothing clears |t|>{crit}. Strongest: {best[0]!r} at t={best[3]:.2f}.")
        n_loose = sum(1 for r in rows if abs(r[3]) > 1.96)
        print(f"{n_loose} of {len(rows)} pass an UNCORRECTED test; "
              f"{len(rows) * 0.05:.1f} expected by chance.")
    else:
        for name, n, bps, t, hit in survivors:
            print(f"SURVIVES SEARCH: {name}  {bps:+.2f} bps  t={t:.2f}  n={n:,}")

    minute = restrict_to_rth(store.load("TQQQ", "1m")).dropna()
    print()
    run(minute, 30, "OUT-OF-SAMPLE: TQQQ minute bars, 30-min horizon, 21 sessions")
    print("\nThe minute panel resolves only a 6.3 bps edge, so absence of")
    print("significance there is not evidence against a smaller real effect.")


if __name__ == "__main__":
    main()
