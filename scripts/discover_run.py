"""Run the self-directed ranker on the full hourly universe."""
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np, pandas as pd

from bipbip.core.costs import CostModel
from bipbip.core.panel import load_panel
from bipbip.data.store import BarStore
from bipbip.data.universe import get_universe
from bipbip.ml.discover import (build_cross_sectional, evaluate_ranker,
                                make_gbm, permutation_baseline)


def turnover_cost(top_k: int, round_trip_bps: float, turnover: float = 0.8) -> float:
    """Cost per rebalance of replacing `turnover` of a `top_k` basket."""
    return round_trip_bps * turnover


def main():
    universe = sys.argv[1] if len(sys.argv) > 1 else "full"
    horizon = int(sys.argv[2]) if len(sys.argv) > 2 else 7

    t0 = time.time()
    store = BarStore("data/bars")
    syms = get_universe(universe)
    panel = load_panel(store, syms, "1h")
    print(f"panel: {len(panel.symbols)} symbols, {len(panel):,} bars, "
          f"{panel.dates[0].date()} -> {panel.dates[-1].date()}")

    ds = build_cross_sectional(panel, horizon=horizon, min_symbols=30)
    print(f"dataset: {ds.summary()}")
    print(f"features: {len(ds.feature_names)}  ({time.time()-t0:.0f}s)\n")

    # ETFs quote tighter than single stocks; charge the universe's own default.
    rt = CostModel().round_trip_cost_bps(
        "SPY" if universe.startswith("etf") else "default",
        price=100.0, shares=1.0)

    print(f"{'top_k':>6} {'cost/rebal':>11} {'excess':>9} {'t':>7} "
          f"{'hit':>6} {'AUC':>6} {'rebalances':>11}")
    results = {}
    for top_k in (3, 5, 10, 20):
        cost = turnover_cost(top_k, rt)
        r = evaluate_ranker(ds, make_model=make_gbm, top_k=top_k, cost_bps=cost)
        if "excess_bps" not in r:
            print(f"{top_k:>6} {r.get('note','failed')}")
            continue
        results[top_k] = r
        print(f"{top_k:>6} {cost:>10.2f}b {r['excess_bps']:>+8.1f}b "
              f"{r['t_stat']:>7.2f} {r['hit_rate']:>5.0%} {r['auc']:>6.3f} "
              f"{r['rebalances']:>11,}")
        print(f"       folds: "
              + "  ".join(f"{f['excess_bps']:+.0f}" for f in r["folds"]))

    if not results:
        print("\nnothing evaluable")
        return

    best_k = max(results, key=lambda k: results[k]["t_stat"])
    print(f"\nPermutation test at top_k={best_k}: what does this procedure")
    print("score when the labels are shuffled WITHIN each timestamp?\n")
    pt = permutation_baseline(ds, n=15, top_k=best_k,
                              cost_bps=turnover_cost(best_k, rt))
    for k, v in pt.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
