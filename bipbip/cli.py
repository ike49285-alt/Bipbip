"""Command line interface.

    python -m bipbip.cli fetch --symbols SPY TQQQ
    python -m bipbip.cli coverage
    python -m bipbip.cli backtest --symbol SPY --strategy orb
    python -m bipbip.cli compare --symbol SPY
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from .config import build_account, load_config
from .core import BacktestEngine, CostModel
from .core import metrics as M
from .data import BarStore, FetchError, get_fetcher, make_intraday_bars
from .strategies import REGISTRY, get_strategy
from .strategies.ml_strategy import MLStrategy


def _store(cfg) -> BarStore:
    return BarStore(cfg.get("data", {}).get("store_dir", "data/bars"))


def _engine(cfg) -> BacktestEngine:
    sess = cfg.get("session", {})
    return BacktestEngine(
        account=build_account(cfg),
        costs=CostModel.from_config(cfg),
        no_new_entries_after=sess.get("no_new_entries_after", "15:00"),
        force_flat_at=sess.get("force_flat_at", "15:55"),
    )


def cmd_fetch(args, cfg) -> int:
    """Pull the provider's trailing window and merge it into the archive."""
    store = _store(cfg)
    fetcher = get_fetcher(args.provider)
    symbols = args.symbols or cfg.get("data", {}).get("symbols", ["SPY"])
    bar_size = args.bar_size or cfg.get("data", {}).get("bar_size", "1m")

    failures = 0
    for symbol in symbols:
        try:
            bars = fetcher.fetch(symbol, bar_size=bar_size, lookback_days=args.lookback)
        except (FetchError, Exception) as exc:  # provider errors are varied
            print(f"  {symbol:6} FAILED: {exc}", file=sys.stderr)
            failures += 1
            continue
        info = store.append(symbol, bars, bar_size=bar_size)
        print(
            f"  {symbol:6} +{info['rows_added']:>6} new bars  "
            f"(archive: {info['rows_after']:,} bars, {info['start']} -> {info['end']})"
        )

    # Non-zero exit makes a silently failing scheduled job visible.
    return 1 if failures == len(symbols) else 0


def cmd_coverage(args, cfg) -> int:
    """Report how much history is archived - i.e. the honest sample size."""
    store = _store(cfg)
    symbols = args.symbols or cfg.get("data", {}).get("symbols", ["SPY"])
    bar_size = cfg.get("data", {}).get("bar_size", "1m")

    print(f"{'symbol':<8}{'bars':>10}{'sessions':>10}  range")
    print("-" * 64)
    empty = []
    for symbol in symbols:
        c = store.coverage(symbol, bar_size)
        if not c["bars"]:
            print(f"{symbol:<8}{'-':>10}{'-':>10}  (empty - run `fetch`)")
            empty.append(symbol)
            continue
        print(f"{symbol:<8}{c['bars']:>10,}{c['sessions']:>10}  {c['start'][:16]} -> {c['end'][:16]}")
        if c["sessions"] < 60:
            print(f"{'':8}{'':20}  ^ {c['sessions']} sessions is too few to validate a "
                  "one-trade-a-day system; keep the collector running.")

    if empty and args.require_data:
        # A scheduled collector whose provider is blocked otherwise reports
        # "no new bars" and exits green - a silently broken job is worse than
        # a failing one, because nobody investigates it.
        print(f"\nERROR: no archived bars for {', '.join(empty)}.", file=sys.stderr)
        print("The provider returned nothing. Check whether it is blocking "
              "the runner's IP or has changed its API.", file=sys.stderr)
        return 1
    return 0


def _load_bars(cfg, symbol, allow_synthetic):
    store = _store(cfg)
    bars = store.load(symbol, cfg.get("data", {}).get("bar_size", "1m"))
    if not bars.empty:
        return bars, False
    if not allow_synthetic:
        print(f"No archived bars for {symbol}. Run `fetch` first.", file=sys.stderr)
        raise SystemExit(2)
    print("!" * 64)
    print("! No real data. Using SYNTHETIC bars - mechanics only.")
    print("! Results below are MEANINGLESS as evidence of edge.")
    print("!" * 64)
    return make_intraday_bars(n_sessions=40, seed=11), True


def cmd_backtest(args, cfg) -> int:
    bars, synthetic = _load_bars(cfg, args.symbol, args.synthetic)
    strategy = get_strategy(args.strategy)
    result = _engine(cfg).run(args.symbol, bars, strategy)
    result.metrics = M.compute(result, bars)
    print(M.format_report(result, result.metrics, strategy.name))

    if args.trades and result.trades:
        print("\n" + result.trades_frame().to_string(index=False))
    if result.blocked:
        print(f"\n{len(result.blocked)} session(s) had a signal refused by account rules.")
        print("  (cash accounts allow one round trip per session; this is what that costs)")
    if synthetic:
        print("\nReminder: synthetic data. This proves the code runs, nothing more.")
    return 0


def cmd_compare(args, cfg) -> int:
    """Run every strategy over the same bars. The baseline is the thing to beat."""
    bars, synthetic = _load_bars(cfg, args.symbol, args.synthetic)
    costs = CostModel.from_config(cfg)
    print(f"\n{args.symbol}: round-trip cost hurdle {costs.round_trip_cost_bps(args.symbol, float(bars['close'].iloc[-1])):.2f} bps\n")

    rows = []
    for name in REGISTRY:
        strategy = get_strategy(name)
        res = _engine(cfg).run(args.symbol, bars, strategy)
        m = M.compute(res, bars)
        rows.append({
            "strategy": strategy.name,
            "trades": m.get("trades", 0),
            "return_%": round(m.get("total_return_pct", 0.0), 2),
            "sharpe": round(m.get("sharpe", float("nan")), 2),
            "max_dd_%": round(m.get("max_drawdown_pct", 0.0), 2),
            "win_%": round(m.get("win_rate_pct", 0.0), 1),
            "avg_hold_min": round(m.get("avg_hold_min", 0.0)),
            "fees_$": round(m.get("total_fees", 0.0), 2),
        })
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nBuy & hold over the same window: {(float(bars['close'].iloc[-1]) / float(bars['close'].iloc[0]) - 1) * 100:+.2f}%")
    if synthetic:
        print("\nSynthetic data: every strategy SHOULD lose here. That is the engine working.")
    return 0


def cmd_train(args, cfg) -> int:
    """Train and validate a model, then run it through the real engine.

    The sklearn metrics are diagnostics. The number that decides anything is
    the backtest at the bottom, because only that one has paid a spread and
    obeyed the settlement rules.
    """
    import numpy as np

    from .ml import MODELS, build_dataset, permutation_test, walk_forward_evaluate

    bars, synthetic = _load_bars(cfg, args.symbol, args.synthetic)
    costs = CostModel.from_config(cfg)
    hurdle = costs.round_trip_cost_bps(args.symbol, float(bars["close"].iloc[-1]))

    ds = build_dataset(bars, target_atr=args.target_atr, stop_atr=args.stop_atr,
                       cost_bps=hurdle)
    print(f"\nDataset: {ds.summary()}")
    print(f"Cost hurdle: {hurdle:.2f} bps (baked into the labels)\n")

    if ds.sessions < 60:
        print("!" * 64)
        print(f"! Only {ds.sessions} sessions. Any model fitted here is fitting noise.")
        print("! Keep the collector running; this run proves plumbing, not edge.")
        print("!" * 64 + "\n")

    kw = dict(n_splits=args.splits, min_train=args.min_train, threshold=args.threshold)
    print(f"{'model':<16}{'AUC':>8}{'trades':>8}{'hit%':>8}{'mean bps':>11}")
    print("-" * 51)
    for name in ["base_rate", "always_enter", "logistic", "mlp"]:
        r = walk_forward_evaluate(MODELS[name], ds.X, ds.y, ds.rets, ds.event_end, **kw)
        if not r.get("folds"):
            print(f"{name:<16}  {r.get('note', 'no folds')}")
            continue
        o = r["overall"]
        auc = o.get("auc", float("nan"))
        hit = o["hit_rate"] * 100 if o["n_trades"] else float("nan")
        print(f"{name:<16}{auc:>8.3f}{o['n_trades']:>8}{hit:>8.1f}{o['mean_ret_bps']:>11.2f}")

    print(f"\nPermutation test on `{args.model}` ({args.permutations} shuffles)")
    perm = permutation_test(MODELS[args.model], ds.X, ds.y, ds.rets, ds.event_end,
                            n_permutations=args.permutations, **kw)
    if "verdict" in perm:
        print(f"  observed {perm['observed_mean_ret_bps']:+.2f} bps on {perm['n_trades']} trades")
        print(f"  luck alone reached {perm['null_best_bps']:+.2f} bps (p={perm['p_value']:.3f})")
        print(f"  -> {perm['verdict']}")

    # Fit on all history and run through the engine, where costs are real.
    model = MODELS[args.model]().fit(ds.X.to_numpy(dtype="float64"), ds.y)
    strategy = MLStrategy(model, threshold=args.threshold,
                          target_atr=args.target_atr, stop_atr=args.stop_atr)
    result = _engine(cfg).run(args.symbol, bars, strategy)
    result.metrics = M.compute(result, bars)
    print("\nIn-sample backtest through the engine (NOT evidence - the model saw this data):")
    print(M.format_report(result, result.metrics, f"ml:{args.model}"))

    if synthetic:
        print("\nSynthetic data throughout. Nothing above is evidence of edge.")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="bipbip", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None, help="path to a config YAML")
    sub = p.add_subparsers(dest="command", required=True)

    f = sub.add_parser("fetch", help="fetch bars and merge into the archive")
    f.add_argument("--symbols", nargs="*", default=None)
    f.add_argument("--bar-size", default=None)
    f.add_argument("--lookback", type=int, default=None, help="calendar days to request")
    f.add_argument("--provider", default="yfinance")
    f.set_defaults(func=cmd_fetch)

    c = sub.add_parser("coverage", help="show archived history")
    c.add_argument("--symbols", nargs="*", default=None)
    c.add_argument("--require-data", action="store_true",
                   help="exit non-zero if any symbol has no archived bars")
    c.set_defaults(func=cmd_coverage)

    b = sub.add_parser("backtest", help="run one strategy")
    b.add_argument("--symbol", default="SPY")
    b.add_argument("--strategy", default="orb", choices=sorted(REGISTRY))
    b.add_argument("--trades", action="store_true", help="print the trade blotter")
    b.add_argument("--synthetic", action="store_true", help="allow synthetic bars if the archive is empty")
    b.set_defaults(func=cmd_backtest)

    k = sub.add_parser("compare", help="run every strategy side by side")
    k.add_argument("--symbol", default="SPY")
    k.add_argument("--synthetic", action="store_true")
    k.set_defaults(func=cmd_compare)

    t = sub.add_parser("train", help="train and validate a model")
    t.add_argument("--symbol", default="SPY")
    t.add_argument("--model", default="logistic", choices=["logistic", "mlp"])
    t.add_argument("--threshold", type=float, default=0.55)
    t.add_argument("--target-atr", type=float, default=1.5)
    t.add_argument("--stop-atr", type=float, default=1.0)
    t.add_argument("--splits", type=int, default=4)
    t.add_argument("--min-train", type=int, default=2000)
    t.add_argument("--permutations", type=int, default=20)
    t.add_argument("--synthetic", action="store_true")
    t.set_defaults(func=cmd_train)

    args = p.parse_args(argv)
    return args.func(args, load_config(args.config))


if __name__ == "__main__":
    raise SystemExit(main())
