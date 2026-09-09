"""Train the meta-labelling model on the daily archive, clean and contaminated.

The primary rule decides direction; the model decides whether to take the
trade. That is the one ML construction in this project with a defensible shape,
because it asks a question with a real answer - "does this particular signal
work?" - rather than trying to forecast price.

It is run on BOTH universes deliberately. Training on the stock lists is what
"train on everything" naturally means, and it is also exactly where a model
learns to pick 2026 survivors: the contamination measured elsewhere in this
repository is worth 15 to 21 points of annual return, which is more than any
model could add. Running the clean ETF universe alongside it is the only way to
tell a model that learned something from a model that learned the index.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np, pandas as pd

from bipbip.core.costs import CostModel
from bipbip.core.panel import load_panel
from bipbip.data.store import BarStore
from bipbip.data.universe import UNIVERSES, get_universe
from bipbip.ml import MODELS, permutation_test, walk_forward_evaluate
from bipbip.ml.meta import (breakout_signals, build_features, label_signals,
                            rsi_dip_signals)

START = "1993-01-29"


def build(universe, store, signal_kind):
    syms = get_universe(universe)
    panel = load_panel(store, syms, "1d").slice_from(START)
    market = panel.closes["SPY"] if "SPY" in panel.closes.columns else None
    if market is None:                       # stock lists have no index member
        spy = store.load("SPY", "1d").dropna()["close"]
        spy.index = spy.index.tz_localize(None)
        market = spy.reindex(panel.dates)

    sig = (rsi_dip_signals(panel) if signal_kind == "rsi_dip"
           else breakout_signals(panel))
    feats = build_features(panel, market=market)
    hurdle = CostModel().round_trip_cost_bps("default", price=100.0, shares=1.0)
    ds = label_signals(panel, sig, feats, cost_bps=hurdle)
    return panel, ds, hurdle


def report(name, ds, hurdle, model_name="mlp"):
    print(f"\n--- {name} ---")
    print(f"  {ds.summary()}")
    print(f"  cost hurdle baked into labels: {hurdle:.2f} bps")
    if len(ds) < 500:
        print("  too few signals to validate; skipping")
        return None

    make = MODELS[model_name]
    res = walk_forward_evaluate(make, ds.X, ds.y, ds.rets, ds.event_end,
                                n_splits=5, min_train=500, threshold=0.55)
    if not res.get("folds"):
        print(f"  {res.get('note', 'evaluation failed')}")
        return None

    o = res["overall"]
    print(f"  out-of-sample: {o['n_trades']:,} trades taken, "
          f"hit rate {o['hit_rate']:.1%}, mean {o['mean_ret_bps']:+.1f} bps, "
          f"AUC {o.get('auc', float('nan')):.3f}")
    base = ds.rets.mean() * 1e4
    print(f"  taking EVERY signal instead: {len(ds):,} trades, "
          f"mean {base:+.1f} bps")
    print(f"  the model's contribution: {o['mean_ret_bps'] - base:+.1f} bps/trade")
    return res


def main():
    store = BarStore("data/bars")
    for signal_kind in ("rsi_dip", "breakout"):
        print(f"\n{'=' * 72}\nPRIMARY SIGNAL: {signal_kind}\n{'=' * 72}")
        results = {}
        for uni in ("etf_wide", "largecap250"):
            panel, ds, hurdle = build(uni, store, signal_kind)
            bias = UNIVERSES[uni]["survivorship"]
            results[uni] = report(f"{uni} ({len(panel.symbols)} symbols, "
                                  f"survivorship {bias})", ds, hurdle)

        a, b = results.get("etf_wide"), results.get("largecap250")
        if a and b:
            print(f"\n  clean vs contaminated: "
                  f"{a['overall']['mean_ret_bps']:+.1f} vs "
                  f"{b['overall']['mean_ret_bps']:+.1f} bps per trade")

    # The clean universe is the one any claim would rest on, so it gets the
    # permutation test: what does this whole procedure score on shuffled
    # labels? A real result inside that distribution is a search artefact.
    print(f"\n{'=' * 72}\nPERMUTATION TEST (clean universe, rsi_dip)\n{'=' * 72}")
    _, ds, _ = build("etf_wide", store, "rsi_dip")
    if len(ds) >= 500:
        pt = permutation_test(MODELS["mlp"], ds.X, ds.y, ds.rets, ds.event_end,
                              n_permutations=20, n_splits=5, min_train=500,
                              threshold=0.55)
        for k, v in pt.items():
            if k in ("null", "folds", "oos_proba", "oos_idx", "frame"):
                continue
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
