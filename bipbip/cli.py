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
from .data.universe import UNIVERSES, get_universe, survivorship_warning
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
    if getattr(args, "universe", None):
        symbols = get_universe(args.universe)
        print(f"  universe {args.universe}: {len(symbols)} symbols")
        print(f"  {survivorship_warning(args.universe)}")
    else:
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
    return 1 if symbols and failures == len(symbols) else 0


def cmd_coverage(args, cfg) -> int:
    """Report how much history is archived - i.e. the honest sample size."""
    store = _store(cfg)
    symbols = args.symbols or cfg.get("data", {}).get("symbols", ["SPY"])
    bar_size = args.bar_size or cfg.get("data", {}).get("bar_size", "1m")

    print(f"{bar_size} bars")
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
        # Only the intraday archive gates strategy validation; daily bars are
        # context and arrive complete from the first fetch.
        if bar_size == "1m" and c["sessions"] < 60:
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

    # Imported here, not at module scope: `fetch` and `coverage` must keep
    # working when scikit-learn is absent, so a missing ML dependency can
    # never stop the collector from archiving bars.
    from .ml import MODELS, build_dataset, permutation_test, walk_forward_evaluate
    from .strategies.ml_strategy import MLStrategy

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


def _matched_null(strategy, seed, entry_rate):
    """Build a random-entry null with the SAME risk geometry as `strategy`.

    Matching matters. A null trading different stops and targets is a different
    strategy, not the same strategy with random timing, and comparing against it
    measures the wrong thing.
    """
    from .strategies.random_entry import RandomEntry
    from .strategies.opening_range import OpeningRangeBreakout
    from .strategies.vwap_reversion import VWAPReversion

    common = dict(seed=seed, entry_rate=entry_rate)
    if isinstance(strategy, OpeningRangeBreakout):
        return RandomEntry(risk_unit="or_width", stop_frac=strategy.stop_frac,
                           target_r=strategy.target_r, **common)
    if isinstance(strategy, VWAPReversion):
        # The target is session VWAP, which sits roughly `stretch_atr` ATRs
        # away at entry, so that is the honest target distance to match.
        return RandomEntry(risk_unit="atr", stop_atr=strategy.stop_atr,
                           target_atr=strategy.stretch_atr, **common)
    return RandomEntry(risk_unit="atr", stop_atr=1.0, target_atr=2.0, **common)


def cmd_significance(args, cfg) -> int:
    """Test a strategy against random entries with matched frequency and risk.

    In a falling market any strategy that trades less loses less, which reads
    as skill and is not. Holding everything constant except WHEN the entry
    happens isolates timing from exposure.
    """
    import numpy as np

    bars, synthetic = _load_bars(cfg, args.symbol, args.synthetic)
    strategy = get_strategy(args.strategy)

    res = _engine(cfg).run(args.symbol, bars, strategy)
    m = M.compute(res, bars)
    real, n_real, sessions = m["total_return_pct"], m["trades"], m["sessions"]
    if n_real == 0:
        print(f"{strategy.name} took no trades on {args.symbol}; nothing to test.")
        return 0

    rate = n_real / sessions
    null = []
    for seed in range(args.trials):
        r = _engine(cfg).run(args.symbol, bars, _matched_null(strategy, seed, rate))
        null.append(M.compute(r, bars)["total_return_pct"])
    null = np.array(null)

    beat = float((null >= real).mean())
    print(f"\n{args.symbol} / {strategy.name}   ({sessions} sessions)")
    print(f"  strategy         {real:+.2f}%   on {n_real} trades")
    print(f"  random entries   {null.mean():+.2f}% mean, {null.std():.2f}% sd, "
          f"best {null.max():+.2f}%, worst {null.min():+.2f}%")
    print(f"  matched null     {rate:.2f} trades/session, same stop/target geometry")
    print(f"  percentile       {(null < real).mean() * 100:.0f}th")
    print(f"  p-value          {beat:.3f} of random runs did as well or better")

    if n_real < 30:
        print(f"\n  VERDICT: inconclusive. {n_real} trades cannot separate skill from "
              f"luck\n  whatever the p-value says. Keep collecting.")
    elif beat > 0.05:
        print("\n  VERDICT: no evidence of timing skill.")
    else:
        print(f"\n  VERDICT: beats matched random entry (p={beat:.3f}). Necessary, not\n"
              "  sufficient - confirm on data collected AFTER today.")
    if synthetic:
        print("\n  Synthetic data. Nothing here is evidence.")
    return 0


#: Sensible sweep ranges for the hand-chosen parameters, so a sensitivity run
#: does not require guessing sane values for each one.
PARAM_SWEEPS = {
    "stretch_z": [1.0, 1.5, 2.0, 2.5, 3.0],
    "max_rsi": [25.0, 30.0, 35.0, 40.0, 45.0],
    "stop_atr": [0.8, 1.0, 1.2, 1.5, 2.0],
    "confirm_bars": [1, 2, 3, 4],
    "min_rvol": [1.0, 1.2, 1.5, 2.0],
    "min_range_bps": [5.0, 10.0, 15.0, 20.0, 30.0],
    "stop_frac": [0.25, 0.5, 0.75, 1.0],
    "target_r": [1.0, 1.5, 2.0, 3.0],
    "or_minutes": [15, 30, 45, 60],
    "min_risk_multiple": [1.0, 1.5, 2.0, 3.0],
}


def cmd_sensitivity(args, cfg) -> int:
    """Sweep one hand-chosen parameter and show the shape of the result.

    Every threshold in this project was picked by judgement, not fitted, and a
    single backtest at one setting cannot distinguish a robust effect from a
    lucky coordinate. Sweeping shows which: a real edge degrades gently either
    side of the chosen value, while an artefact is an isolated spike among
    neighbours that lose money.

    This is emphatically NOT a tuner. Picking the best cell on a 20-session
    sample is precisely the overfitting the rest of the project exists to
    avoid, so the peak is reported as a warning sign rather than a
    recommendation.
    """
    import numpy as np

    bars, synthetic = _load_bars(cfg, args.symbol, args.synthetic)
    values = args.values if args.values else PARAM_SWEEPS.get(args.param)
    if not values:
        print(f"No default sweep for {args.param!r}; pass --values.", file=sys.stderr)
        return 2

    rows = []
    for v in values:
        typed = type(PARAM_SWEEPS.get(args.param, [1.0])[0])(v)
        try:
            strategy = get_strategy(args.strategy, **{args.param: typed})
        except TypeError:
            print(f"{args.strategy} has no parameter {args.param!r}.", file=sys.stderr)
            return 2
        res = _engine(cfg).run(args.symbol, bars, strategy)
        m = M.compute(res, bars)
        rows.append({"value": typed, "trades": m["trades"],
                     "return_pct": m["total_return_pct"],
                     "win_pct": m["win_rate_pct"],
                     "sharpe": m.get("sharpe", float("nan"))})

    print(f"\n{args.symbol} / {args.strategy} - sweeping {args.param}")
    print(f"{'value':>10}{'trades':>9}{'return %':>11}{'win %':>8}{'sharpe':>9}")
    print("-" * 47)
    for r in rows:
        print(f"{r['value']:>10}{r['trades']:>9}{r['return_pct']:>11.2f}"
              f"{r['win_pct']:>8.1f}{r['sharpe']:>9.2f}")

    rets = np.array([r["return_pct"] for r in rows], dtype="float64")
    best = int(np.argmax(rets))
    neighbours = [rets[i] for i in (best - 1, best + 1) if 0 <= i < len(rets)]
    total_trades = sum(r["trades"] for r in rows)

    print()
    busiest = max((r["trades"] for r in rows), default=0)
    if total_trades == 0:
        print("  No trades at any setting. Nothing to say about robustness.")
    elif not neighbours:
        print("  Peak sits at the edge of the sweep; widen --values to see its shape.")
    else:
        nb = float(np.mean(neighbours))
        print(f"  peak {rets[best]:+.2f}% at {rows[best]['value']}, "
              f"neighbours average {nb:+.2f}%")
        if rets[best] > 0 and nb <= 0:
            print("  SHAPE: isolated spike. The settings either side average a loss, so the\n"
                  "  peak is a lucky coordinate rather than an effect. Do not adopt it.")
        elif float(np.std(rets)) < 0.25:
            print("  SHAPE: flat. The parameter barely matters here - usually because there\n"
                  "  is no signal for it to modulate.")
        else:
            print("  SHAPE: plateau. Results degrade gently either side, which is what a\n"
                  "  real effect looks like. Still needs far more data to believe.")

    if 0 < busiest < 30:
        print(f"\n  UNDERPOWERED: the busiest setting took only {busiest} trades. The shape\n"
              "  above is mostly sampling noise; re-run once the archive is deeper.")

    print("\n  Reminder: this diagnoses robustness, it does not select a value.\n"
          "  Adopting the best cell on a short sample is exactly how backtests lie.")
    if synthetic:
        print("\n  Synthetic data. Nothing here is evidence.")
    return 0


def cmd_options(args, cfg) -> int:
    """Re-express a strategy's signals as long 0DTE options.

    Prints the honest breakeven hurdle, the stock-versus-options comparison,
    and a sweep over the volatility assumption - which is modelled rather than
    observed and is the largest single source of error here.
    """
    import numpy as np

    from .options.overlay import breakeven_move_bps, express_in_options
    from .options.risk import OptionRiskModel

    bars, synthetic = _load_bars(cfg, args.symbol, args.synthetic)
    strategy = get_strategy(args.strategy)
    res = _engine(cfg).run(args.symbol, bars, strategy)
    m = M.compute(res, bars)
    if not res.trades:
        print(f"{strategy.name} took no trades on {args.symbol}; nothing to express.")
        return 0

    spot = float(bars["close"].iloc[-1])
    costs = CostModel.from_config(cfg)
    stock_hurdle = costs.round_trip_cost_bps(args.symbol, spot)

    print(f"\n{args.symbol} / {strategy.name}: 0DTE option expression")
    print("\nBreakeven underlying move, entered at the open (ATM, modelled IV):")
    print(f"{'hold':>10}{'0DTE':>12}{'stock':>12}")
    for h in (15, 30, 60, 120, 240):
        be = breakeven_move_bps(spot, spot, 390.0, args.iv_floor or 0.10, float(h))
        print(f"{h:>8}m{be:>11.1f}b{stock_hurdle:>11.2f}b")
    print("  Theta makes the option hurdle grow with holding time; the stock's\n"
          "  does not. Options only win on short holds.")

    risk = OptionRiskModel(
        target_delta=args.delta, premium_stop_pct=args.premium_stop,
        premium_target_pct=args.premium_target, max_hold_minutes=args.max_hold,
        premium_pct=args.premium_pct,
    )
    print(f"\n  risk model: {risk.describe()}")
    opts, summ = express_in_options(
        bars, res.trades, args.symbol, kind=args.kind, risk=risk,
        iv_premium=args.iv_premium, iv_floor=args.iv_floor,
    )
    moves = [abs(t.underlying_move_bps) for t in opts]
    print(f"\n  stock    {m['trades']:3d} trades  {m['total_return_pct']:+8.2f}%  "
          f"avg hold {m.get('avg_hold_min', 0):.0f}min")
    print(f"  0DTE     {summ['trades']:3d} trades  {summ['total_return_pct']:+8.2f}%  "
          f"win {summ['win_rate_pct']:.0f}%  expired worthless {summ['expired_worthless']}")
    if summ["trades"]:
        print(f"           mean win {summ['mean_win_pct']:+.0f}% / mean loss "
              f"{summ['mean_loss_pct']:+.0f}%  ratio {summ['win_loss_ratio']:.2f}  "
              f"(needs >1.0 near a 50% hit rate)")
        print(f"           median hold {summ['median_hold_min']:.0f}m  "
              f"exits {summ['exit_breakdown']}")
    if moves:
        print(f"  median |underlying move| per trade: {np.median(moves):.1f} bps")

    print(f"\n  Sensitivity to the volatility assumption (it is modelled, not observed):")
    print(f"{'IV premium':>12}{'mean IV':>12}{'0DTE return':>14}")
    for prem in [0.85, 1.0, 1.15, 1.35, 1.6]:
        _, sw = express_in_options(bars, res.trades, args.symbol, kind=args.kind,
                                   risk=risk, iv_premium=prem, iv_floor=args.iv_floor)
        from .options.iv import implied_vol
        # Mean, not median: the floor binds on most bars in a quiet sample, so
        # the median reads identically across premiums and hides why the
        # returns differ.
        series = implied_vol(bars["close"], premium=prem, symbol=args.symbol,
                             floor=args.iv_floor)
        print(f"{prem:>12.2f}{float(series.mean()):>11.1%}{sw['total_return_pct']:>13.2f}%")
    print("  If the sign flips across this range, the result is an artefact of the\n"
          "  vol assumption rather than anything about the market.")

    if summ["trades"] < 30:
        print(f"\n  Only {summ['trades']} option trades. Indicative at best.")
    print("\n  Every option price above is Black-Scholes on a MODELLED vol surface,\n"
          "  not a quote. No smile and no intraday term structure, both of which\n"
          "  flatter these numbers.")
    if synthetic:
        print("\n  Synthetic underlying data on top of that. Nothing here is evidence.")
    return 0


def cmd_regime(args, cfg) -> int:
    """Locate the recent window in the instrument's own volatility history.

    A backtest measures a strategy against the market it was run on. Knowing
    WHICH market that was decides whether the result travels: a mean-reversion
    edge found in the calmest decile is not evidence about the other nine.
    Daily bars answer this in one request, where the 30-day minute archive
    never can.
    """
    import numpy as np

    store = _store(cfg)
    symbols = args.symbols or cfg.get("data", {}).get("symbols", ["SPY"])

    for symbol in symbols:
        daily = store.load(symbol, "1d")
        if daily.empty:
            print(f"{symbol}: no daily bars. Run `fetch --bar-size 1d`.", file=sys.stderr)
            continue

        close = daily["close"]
        rets = np.log(close / close.shift(1)).dropna()
        rv = (rets.rolling(args.window).std() * np.sqrt(252)).dropna()
        if rv.empty:
            continue

        current = float(rv.iloc[-1])
        pct = float((rv < current).mean() * 100)
        years = len(close) / 252

        print(f"\n{symbol}: {len(close):,} sessions ({years:.1f} years)")
        print(f"  last {args.window} sessions realised {current:.1%} annualised vol")
        print(f"  -> {pct:.0f}th percentile of its own history")
        print(f"  {'calmest 5%':<14}{rv.quantile(0.05):>7.1%}")
        print(f"  {'median':<14}{rv.quantile(0.50):>7.1%}")
        print(f"  {'wildest 5%':<14}{rv.quantile(0.95):>7.1%}")

        if pct < 25:
            print(f"  WARNING: unusually quiet. {100 - pct:.0f}% of history was more")
            print("  volatile, so results measured here should not be assumed to")
            print("  hold in a normal market - stops and thresholds calibrated on")
            print("  this window are sized for a market that mostly does not exist.")
        elif pct > 75:
            print(f"  WARNING: unusually volatile. {pct:.0f}% of history was calmer.")
    return 0


def cmd_chains(args, cfg) -> int:
    """Fetch option chains and report implied vol against realised.

    Chains are snapshots, not history: Yahoo serves what stands right now and
    keeps nothing, so this accumulates forward from the first run. That answers
    "is this contract mispriced today" within days and never answers "would
    this have worked since 2010" - historical option data is not free.
    """
    import numpy as np

    from pathlib import Path

    import pandas as pd

    from .data.options_chain import ChainStore, fetch_chain, iv_vs_realised
    from .data.store import BarStore

    store = ChainStore(cfg.get("data", {}).get("chain_root", "data/chains"))
    bars = BarStore(cfg.get("data", {}).get("root", "data/bars"))
    failures = 0
    for symbol in args.symbols:
        try:
            chain = fetch_chain(symbol, max_expiries=args.expiries)
        except Exception as exc:
            print(f"  {symbol:6} FAILED: {exc}", file=sys.stderr)
            failures += 1
            continue
        info = store.save(symbol, chain)
        print(f"  {symbol:6} {info['contracts']:>5} contracts "
              f"across {chain['expiry'].nunique()} expiries -> {info['asof']}")

        # Say out loud what actually reached disk. The first collected snapshot
        # arrived with a date-only filename and no spot column, neither of which
        # this code path produces - so the run reported success while writing
        # something else, and nothing noticed until the file was read a session
        # later. A snapshot that silently overwrites the previous one turns four
        # observations a day into one.
        written = pd.read_parquet(info["path"])
        stamped = "T" in Path(info["path"]).stem.split("_", 1)[1]
        has_spot = "spot" in written.columns and np.isfinite(
            pd.to_numeric(written["spot"], errors="coerce")).any()
        print(f"         file {Path(info['path']).name}  "
              f"unique-per-snapshot {'yes' if stamped else 'NO'}  "
              f"spot stored {'yes' if has_spot else 'NO'}")
        if not stamped:
            print("         WARNING: date-only filename - snapshots will "
                  "overwrite each other", file=sys.stderr)

        daily = bars.load(symbol, "1d").dropna()
        if daily.empty:
            continue
        spot = float(daily["close"].iloc[-1])
        rets = np.log(daily["close"] / daily["close"].shift(1)).dropna()
        rv = float(rets.tail(60).std() * np.sqrt(252))
        cmp_ = iv_vs_realised(chain, spot, rv)
        if "near_the_money_iv" in cmp_:
            print(f"         spot ${spot:.2f}  ATM IV {cmp_['near_the_money_iv']:.1%}  "
                  f"realised {rv:.1%}  ratio {cmp_['ratio']:.2f}")
            print(f"         {cmp_['verdict']}")
        else:
            print(f"         {cmp_.get('note')}")
    return 1 if failures == len(args.symbols) else 0


def cmd_fundamentals(args, cfg) -> int:
    """Fetch non-price data, and report honestly on what is unavailable."""
    import json

    from .data.fundamentals import (FundamentalsStore, fetch_earnings_dates,
                                    fetch_sectors, probe_news)

    symbols = get_universe(args.universe)
    store = FundamentalsStore()

    if args.what in ("all", "sectors"):
        sectors = fetch_sectors(symbols)
        store.save_json("sectors.json", sectors)
        by = {}
        for v in sectors.values():
            by[v["sector"]] = by.get(v["sector"], 0) + 1
        print(f"sectors: {len(sectors)}/{len(symbols)} symbols classified")
        for k, v in sorted(by.items(), key=lambda kv: -kv[1]):
            print(f"    {k:<26}{v:>4}")

    if args.what in ("all", "earnings"):
        df = fetch_earnings_dates(symbols, limit=args.earnings_limit)
        if df.empty:
            print("earnings: nothing returned", file=sys.stderr)
        else:
            store.save_frame("earnings_dates.parquet", df)
            print(f"\nearnings: {len(df):,} dates for "
                  f"{df['symbol'].nunique()} symbols")
            print(f"    range {df['earnings_date'].min().date()} to "
                  f"{df['earnings_date'].max().date()}")
            per = df.groupby("symbol").size()
            print(f"    per symbol: median {per.median():.0f}, max {per.max()}")

    if args.what in ("all", "news"):
        report = probe_news(symbols, sample=args.news_sample)
        store.save_json("news_probe.json", report)
        print(f"\nnews probe: {report['total_items']} items across "
              f"{len(report['checked'])} symbols")
        print(f"    oldest {report.get('oldest')}   newest {report.get('newest')}")
        for row in report["checked"]:
            print(f"    {row}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="bipbip", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None, help="path to a config YAML")
    sub = p.add_subparsers(dest="command", required=True)

    f = sub.add_parser("fetch", help="fetch bars and merge into the archive")
    f.add_argument("--symbols", nargs="*", default=None)
    f.add_argument("--universe", default=None, choices=sorted(UNIVERSES),
                   help="fetch a named universe instead of individual symbols")
    f.add_argument("--bar-size", default=None)
    f.add_argument("--lookback", type=int, default=None, help="calendar days to request")
    f.add_argument("--provider", default="yfinance")
    f.set_defaults(func=cmd_fetch)

    c = sub.add_parser("coverage", help="show archived history")
    c.add_argument("--symbols", nargs="*", default=None)
    c.add_argument("--bar-size", default=None)
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

    g = sub.add_parser("significance", help="test a strategy against matched random entries")
    g.add_argument("--symbol", default="SPY")
    g.add_argument("--strategy", default="orb", choices=sorted(REGISTRY))
    g.add_argument("--trials", type=int, default=300)
    g.add_argument("--synthetic", action="store_true")
    g.set_defaults(func=cmd_significance)

    v = sub.add_parser("sensitivity", help="sweep a hand-chosen parameter to test robustness")
    v.add_argument("--symbol", default="SPY")
    v.add_argument("--strategy", default="vwap_reversion", choices=sorted(REGISTRY))
    v.add_argument("--param", required=True)
    v.add_argument("--values", nargs="*", type=float, default=None)
    v.add_argument("--synthetic", action="store_true")
    v.set_defaults(func=cmd_sensitivity)

    o = sub.add_parser("options", help="express a strategy's signals as 0DTE options")
    o.add_argument("--symbol", default="SPY")
    o.add_argument("--strategy", default="orb", choices=sorted(REGISTRY))
    o.add_argument("--kind", default="call", choices=["call", "put"])
    o.add_argument("--premium-pct", type=float, default=0.10,
                   help="fraction of equity spent on premium per trade")
    o.add_argument("--delta", type=float, default=0.75,
                   help="target strike delta; higher is further in-the-money")
    o.add_argument("--premium-stop", type=float, default=0.30)
    o.add_argument("--premium-target", type=float, default=0.50)
    o.add_argument("--max-hold", type=int, default=45,
                   help="hard time stop in minutes")
    o.add_argument("--iv-premium", type=float, default=1.15)
    o.add_argument("--iv-floor", type=float, default=None)
    o.add_argument("--synthetic", action="store_true")
    o.set_defaults(func=cmd_options)

    g2 = sub.add_parser("regime", help="where the recent window sits in its own volatility history")
    g2.add_argument("--symbols", nargs="*", default=None)
    g2.add_argument("--window", type=int, default=21)
    g2.set_defaults(func=cmd_regime)

    fu = sub.add_parser("fundamentals", help="fetch earnings dates, sectors, and probe news")
    fu.add_argument("--universe", default="largecap250", choices=sorted(UNIVERSES))

    ch = sub.add_parser("chains", help="fetch option chains and compare IV to realised")
    ch.add_argument("--symbols", nargs="+", default=["TQQQ", "SPY", "QQQ"])
    ch.add_argument("--expiries", type=int, default=6)
    ch.set_defaults(func=cmd_chains)
    fu.add_argument("--what", default="all", choices=["all", "sectors", "earnings", "news"])
    fu.add_argument("--earnings-limit", type=int, default=60)
    fu.add_argument("--news-sample", type=int, default=5)
    fu.set_defaults(func=cmd_fundamentals)

    args = p.parse_args(argv)
    return args.func(args, load_config(args.config))


if __name__ == "__main__":
    raise SystemExit(main())
