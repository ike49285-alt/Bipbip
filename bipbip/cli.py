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
    for symbol in symbols:
        c = store.coverage(symbol, bar_size)
        if not c["bars"]:
            print(f"{symbol:<8}{'-':>10}{'-':>10}  (empty - run `fetch`)")
            continue
        print(f"{symbol:<8}{c['bars']:>10,}{c['sessions']:>10}  {c['start'][:16]} -> {c['end'][:16]}")
        if c["sessions"] < 60:
            print(f"{'':8}{'':20}  ^ {c['sessions']} sessions is too few to validate a "
                  "one-trade-a-day system; keep the collector running.")
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

    args = p.parse_args(argv)
    return args.func(args, load_config(args.config))


if __name__ == "__main__":
    raise SystemExit(main())
