"""The same machine at every scale, which is the claim worth testing.

Bip has been trained at two timeframes and, on the hourly one, at exactly one
holding period - seven bars, meaning a single session, which is the horizon the
cost arithmetic already says is hardest. Reporting that as "no edge at hourly"
was a single point mistaken for a survey.

So: the same model, the same features, the same validation, swept across
timeframes and horizons. If price structure is self-similar, a real signal
should leave a consistent signature rather than appearing at one arbitrary
scale. If it appears at exactly one setting and nowhere near it, that is the
signature of a coordinate, not an edge.
"""
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

from bipbip.core.costs import CostModel
from bipbip.core.panel import load_panel
from bipbip.data.store import BarStore
from bipbip.data.universe import get_universe
from bipbip.ml.discover import (build_cross_sectional, evaluate_ranker,
                                make_gbm, permutation_baseline)

# (bar size, universe, horizons in BARS, a human label for each)
PLANS = [
    ("1h", "full", [(1, "1 hour"), (7, "1 session"), (21, "3 sessions"),
                    (35, "1 week"), (70, "2 weeks"), (140, "1 month")]),
    ("1m", "etf_wide", [(15, "15 min"), (30, "30 min"), (60, "1 hour"),
                        (120, "2 hours"), (390, "1 session")]),
]


def main():
    store = BarStore("data/bars")
    costs = CostModel()
    t0 = time.time()
    best = []

    for bar_size, universe, horizons in PLANS:
        panel = load_panel(store, get_universe(universe), bar_size)
        rt = costs.round_trip_cost_bps(
            "SPY" if universe.startswith("etf") else "default",
            price=100.0, shares=1.0)
        print(f"\n{'=' * 78}")
        print(f"{bar_size} bars, universe {universe}: {len(panel.symbols)} symbols, "
              f"{len(panel):,} bars  ({panel.dates[0].date()} -> {panel.dates[-1].date()})")
        print(f"round-trip cost charged per rebalance: {rt * 0.8:.2f} bps")
        print(f"{'=' * 78}")
        print(f"{'horizon':>12} {'rows':>10} {'|excess|':>9} {'edge':>8} "
              f"{'t':>7} {'hit':>6} {'AUC':>6} {'rebal':>7}")

        for bars, label in horizons:
            try:
                ds = build_cross_sectional(panel, horizon=bars, min_symbols=15)
            except ValueError as exc:
                print(f"{label:>12} {'(' + str(exc)[:40] + ')':>50}")
                continue
            if len(ds) < 5000:
                print(f"{label:>12} {len(ds):>10,} (too few rows)")
                continue
            r = evaluate_ranker(ds, make_model=make_gbm, top_k=10,
                                cost_bps=rt * 0.8)
            if "excess_bps" not in r:
                print(f"{label:>12} {len(ds):>10,} {r.get('note','failed'):>40}")
                continue
            print(f"{label:>12} {len(ds):>10,} "
                  f"{np.nanmean(np.abs(ds.excess)) * 1e4:>8.0f}b "
                  f"{r['excess_bps']:>+7.1f}b {r['t_stat']:>7.2f} "
                  f"{r['hit_rate']:>5.0%} {r['auc']:>6.3f} {r['rebalances']:>7,}")
            best.append((r["t_stat"], bar_size, universe, bars, label, r))

    if not best:
        print("\nnothing evaluable")
        return

    best.sort(reverse=True, key=lambda x: x[0])
    t, bar_size, universe, bars, label, r = best[0]
    print(f"\n{'=' * 78}")
    print(f"Strongest: {bar_size} at {label} (t={t:.2f}, {r['excess_bps']:+.1f} bps)")
    print("Permutation test - labels shuffled WITHIN each timestamp, so the")
    print("cross-section survives and only the signal is destroyed.")
    print(f"{'=' * 78}")
    panel = load_panel(store, get_universe(universe), bar_size)
    ds = build_cross_sectional(panel, horizon=bars, min_symbols=15)
    rt = costs.round_trip_cost_bps(
        "SPY" if universe.startswith("etf") else "default", price=100.0, shares=1.0)
    pt = permutation_baseline(ds, n=15, top_k=10, cost_bps=rt * 0.8)
    for k, v in pt.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print(f"\ntotal {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
