"""How much history should the model see? A hypothesis, not a knob.

Every effect measured in this project decayed: turn-of-month, post-earnings
drift, the stochastic gradient, Heikin-Ashi reversion, momentum. Each was
strong when it was worth publishing and gone within two decades. If that is the
rule rather than a run of bad luck, then a model trained on all available
history is fitting an average of regimes that no longer exist, and a short
window seeing only the live one could beat it while having a fraction of the
data.

That trades regime freshness against sample size, and the trade has an answer.
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

WINDOWS = [(30, "30 days"), (90, "90 days"), (180, "180 days"),
           (365, "1 year"), (None, "everything")]


def main():
    horizon = int(sys.argv[1]) if len(sys.argv) > 1 else 35
    t0 = time.time()
    store = BarStore("data/bars")
    panel = load_panel(store, get_universe("full"), "1h")
    rt = CostModel().round_trip_cost_bps("default", price=100.0, shares=1.0) * 0.8
    ds = build_cross_sectional(panel, horizon=horizon, min_symbols=15)

    print(f"hourly, 305 symbols, horizon {horizon} bars "
          f"(~{horizon / 7:.0f} sessions)")
    print(f"{ds.summary()}")
    print(f"cost charged: {rt:.2f} bps per rebalance\n")
    print(f"{'training window':>16} {'edge':>9} {'t':>7} {'hit':>6} "
          f"{'AUC':>6} {'rebal':>7} {'folds':>28}")

    results = {}
    for days, label in WINDOWS:
        r = evaluate_ranker(ds, make_model=make_gbm, top_k=10, cost_bps=rt,
                            train_window_days=days)
        if "excess_bps" not in r:
            print(f"{label:>16} {r.get('note', 'failed'):>40}")
            continue
        results[label] = (days, r)
        folds = " ".join(f"{f['excess_bps']:+.0f}" for f in r["folds"])
        print(f"{label:>16} {r['excess_bps']:>+8.1f}b {r['t_stat']:>7.2f} "
              f"{r['hit_rate']:>5.0%} {r['auc']:>6.3f} {r['rebalances']:>7,} "
              f"{folds:>28}")

    if not results:
        return
    best_label = max(results, key=lambda k: results[k][1]["t_stat"])
    days, r = results[best_label]
    print(f"\nStrongest: {best_label} (t={r['t_stat']:.2f}, "
          f"{r['excess_bps']:+.1f} bps). Permutation test:\n")
    pt = permutation_baseline(ds, n=15, top_k=10, cost_bps=rt,
                              train_window_days=days)
    for k, v in pt.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print(f"\ntotal {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
