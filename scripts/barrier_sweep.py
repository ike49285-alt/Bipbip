"""Does loosening the exits recover the signal the model already finds?

The meta-model picks trades sitting in windows where the market returned +77
bps against +59 across all signals - it identifies better periods. The triple
barrier then hands back +51.5: a +4% target caps the winners, a -3% stop cuts
them, a 20-bar limit closes the rest. The skill is real and the wrapper spends
it.

So the barriers are swept. The benchmark SCALES WITH THE HOLD, because a longer
leash is not free: it is compared against being long for those same, longer
windows. A configuration that makes more money per trade by holding twice as
long has not necessarily beaten anything.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

from bipbip.core.costs import CostModel
from bipbip.core.panel import load_panel
from bipbip.data.store import BarStore
from bipbip.data.universe import get_universe
from bipbip.ml import MODELS, walk_forward_evaluate
from bipbip.ml.meta import build_features, label_signals, rsi_dip_signals, breakout_signals

START = "1993-01-29"


def prepare(universe="etf_wide", signal="rsi_dip"):
    store = BarStore("data/bars")
    panel = load_panel(store, get_universe(universe), "1d").slice_from(START)
    market = panel.closes["SPY"]
    sig = rsi_dip_signals(panel) if signal == "rsi_dip" else breakout_signals(panel)
    feats = build_features(panel, market=market)
    return panel, sig, feats


def evaluate(panel, sig, feats, target, stop, hold, hurdle, use_model=True):
    """One barrier configuration, against a hold-matched benchmark."""
    ds = label_signals(panel, sig, feats, target_pct=target, stop_pct=stop,
                       max_hold=hold, cost_bps=hurdle)
    if len(ds) < 500:
        return None
    spy = panel.closes["SPY"].to_numpy(dtype="float64")

    if use_model:
        res = walk_forward_evaluate(MODELS["mlp"], ds.X, ds.y, ds.rets,
                                    ds.event_end, n_splits=5, min_train=500,
                                    threshold=0.55)
        if not res.get("folds"):
            return None
        idx = res["oos_idx"][res["oos_proba"] >= 0.55]
    else:
        idx = np.arange(len(ds))
    if len(idx) < 100:
        return None

    ep, xp = ds.entry_bar[idx], ds.exit_bar[idx]
    trade = ds.rets[idx]
    mkt = spy[xp] / spy[ep] - 1.0

    # Benchmark two ways, because they answer different questions and only the
    # second isolates the signal.
    #
    # Against SPY: "should I do this instead of holding the index?" - the
    # practical question, but it charges every trade the gap between its own
    # asset and SPY, and SPY is the third-best compounder of the 34 here. A
    # trade on USO is not competing with the index, it is being compared to it.
    #
    # Against the SAME SYMBOL over the SAME bars: "does the rule time this
    # asset better than owning it?" That subtracts the asset entirely and
    # leaves only the timing, which is the thing being tested.
    closes = panel.closes
    syms = ds.symbols[idx]
    own = np.full(len(idx), np.nan)
    for sym in np.unique(syms):
        m = syms == sym
        px = closes[sym].to_numpy(dtype="float64")
        own[m] = px[xp[m]] / px[ep[m]] - 1.0

    ok = (np.isfinite(trade) & np.isfinite(mkt) & np.isfinite(own) & (xp > ep))
    trade, mkt, own, held = trade[ok], mkt[ok], own[ok], (xp - ep)[ok]
    if len(trade) < 100:
        return None
    d_spy, d_own = trade - mkt, trade - own
    return {
        "n": len(trade), "hold": held.mean(),
        "trade_bps": trade.mean() * 1e4,
        "mkt_bps": mkt.mean() * 1e4, "own_bps": own.mean() * 1e4,
        "edge_bps": d_spy.mean() * 1e4,
        "own_edge_bps": d_own.mean() * 1e4,
        "t": d_spy.mean() / d_spy.std() * np.sqrt(len(d_spy)),
        "own_t": d_own.mean() / d_own.std() * np.sqrt(len(d_own)),
        "win": (d_own > 0).mean(),
    }


def main():
    hurdle = CostModel().round_trip_cost_bps("default", price=100.0, shares=1.0)
    for signal in ("rsi_dip", "breakout"):
        panel, sig, feats = prepare(signal=signal)
        print(f"\n{'=' * 86}\n{signal}: loosening the barriers "
              f"(model-selected trades, cost {hurdle:.1f} bps)\n{'=' * 86}")
        print(f"{'target':>7} {'stop':>6} {'hold':>6} {'n':>7} {'bars':>6} "
              f"{'trade':>8} {'hold own':>9} {'TIMING':>8} {'t':>6} "
              f"{'win':>5} {'vs SPY':>8}")

        configs = [
            (0.04, 0.03, 20),      # the original
            (0.08, 0.03, 20),      # target further away
            (0.20, 0.03, 20),      # effectively no target
            (0.04, 0.08, 20),      # stop further away
            (0.20, 0.20, 20),      # neither barrier binds
            (0.20, 0.20, 60),      # and a longer leash
            (0.20, 0.20, 120),
            (0.08, 0.06, 60),
        ]
        for tgt, stp, hold in configs:
            r = evaluate(panel, sig, feats, tgt, stp, hold, hurdle)
            if r is None:
                print(f"{tgt:>7.0%} {stp:>6.0%} {hold:>5}d {'(too few trades)':>44}")
                continue
            print(f"{tgt:>6.0%} {stp:>5.0%} {hold:>5}d {r['n']:>7,} {r['hold']:>6.1f} "
                  f"{r['trade_bps']:>+8.1f} {r['own_bps']:>+9.1f} "
                  f"{r['own_edge_bps']:>+8.1f} {r['own_t']:>6.2f} "
                  f"{r['win']:>4.0%} {r['edge_bps']:>+8.1f}")


if __name__ == "__main__":
    main()
