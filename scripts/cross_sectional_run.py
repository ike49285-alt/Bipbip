"""The cross-sectional strategies, on the universe that exists for them.

The archive holds 309 symbols, and almost every test in this project has used
one or two. That is a fair complaint: a universe is collected so a strategy can
RANK it, and the ranking strategies had not been re-run since two fixes that
change their results directly - dividends restored to the daily bars, and the
engine bug that made an empty target mean "hold" rather than "go to cash",
which left momentum's absolute filter inoperative for its whole life.

The second reason the universe is bigger than the tests is less comfortable.
The stock lists are 2026 index membership, so every company that failed is
absent. Running the same strategy on a clean ETF universe and a survivorship-
contaminated stock universe puts a number on what that is worth, which is the
only honest way to read any result from the stock lists at all.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))


from swing_compare import stats, START
from bipbip.core.costs import CostModel
from bipbip.core.panel import load_panel
from bipbip.core.portfolio import PortfolioEngine, PortfolioStrategy
from bipbip.data.store import BarStore
from bipbip.data.universe import UNIVERSES, get_universe
from bipbip.strategies.cross_sectional import (
    EqualWeightBuyHold, MeanReversionBasket, MomentumRanking)


class HoldOne(PortfolioStrategy):
    def __init__(self, symbol="SPY"):
        self.symbol, self.name, self.warmup_bars = symbol, f"hold_{symbol}", 1
    def target_weights(self, ctx):
        return {self.symbol: 1.0} if self.symbol in ctx.tradeable else None


def run_universe(name, store, start="1993-01-29"):
    """Run one universe from a COMMON start date.

    Aligning the start is not a detail. largecap250 reaches back to 1962 and
    the ETF list only to 1993, so comparing them as loaded would attribute
    thirty extra years of compounding to survivorship and call it a
    measurement. Both are cut to the same window.
    """
    syms = get_universe(name)
    panel = load_panel(store, syms, "1d")
    panel = panel.slice_from(start)
    have = len(panel.closes.columns)
    bias = UNIVERSES[name]["survivorship"]
    print(f"\n=== {name}: {have} symbols loaded, survivorship {bias} ===")
    print(f"{panel.dates[0].date()} -> {panel.dates[-1].date()}, "
          f"{len(panel):,} sessions")
    print(f"{'strategy':<34} {'final':>9} {'CAGR':>8} {'Shrp':>6} "
          f"{'maxDD':>7} {'trd/yr':>7}")

    # SPY is the benchmark, not a member of a stock universe, so it is priced
    # from its own series rather than required to be in the ranking list. The
    # first run silently reported $50 and 0.00% because HoldOne("SPY") could
    # never find SPY tradeable in largecap250.
    spy = store.load("SPY", "1d").dropna()["close"]
    # build_panel drops the timezone, the raw store keeps it, and comparing the
    # two raises rather than silently misaligning - which is the better failure
    # but still has to be handled.
    spy = spy.copy()
    if spy.index.tz is not None:
        spy.index = spy.index.tz_localize(None)
    lo, hi = panel.dates[0], panel.dates[-1]
    if getattr(lo, "tz", None) is not None:
        lo, hi = lo.tz_localize(None), hi.tz_localize(None)
    spy = spy[(spy.index >= lo) & (spy.index <= hi)]
    bh = START * spy / float(spy.iloc[0])
    s_bh = stats(bh, [], start=START)
    print(f"{'buy & hold SPY (benchmark)':<34} ${s_bh['final']:>8,.0f} "
          f"{s_bh['cagr']*100:>7.2f}% {s_bh['sharpe']:>6.2f} "
          f"{s_bh['mdd']*100:>6.1f}% {0.0:>7.1f}")

    # Equal-weighting a wide universe is infeasible on $50: 266 names is 19
    # cents each, below the engine's dust threshold, so every buy is refused
    # and the curve sits flat at the starting balance. That is a real
    # constraint of the account size, not a strategy result, so it is only run
    # where the slice clears the minimum.
    slice_value = START / max(have, 1)
    cands = []
    if slice_value >= 0.50:
        cands.append(("equal weight, rebalanced", EqualWeightBuyHold()))
    else:
        print(f"{'equal weight, rebalanced':<34} "
              f"{f'infeasible: ${slice_value:.2f} per name':>40}")
    cands += [
        ("momentum top5 (abs filter)", MomentumRanking(top_n=5, abs_filter=True)),
        ("momentum top5 (no filter)", MomentumRanking(top_n=5, abs_filter=False)),
        ("momentum top10 (abs filter)", MomentumRanking(top_n=10, abs_filter=True)),
        ("reversion basket", MeanReversionBasket()),
    ]
    out = {"buy & hold SPY (benchmark)": (bh, s_bh)}
    for label, strat in cands:
        res = PortfolioEngine(CostModel(), starting_equity=START,
                              settle_days=0).run(panel, strat, universe=name)
        c = res.equity_curve.dropna(); c = c[c > 0]
        if len(c) < 100:
            print(f"{label:<34} {'(insufficient history)':>40}")
            continue
        s = stats(c, res.trades, start=float(c.iloc[0]))
        out[label] = (c, s)
        print(f"{label:<34} ${s['final']:>8,.0f} {s['cagr']*100:>7.2f}% "
              f"{s['sharpe']:>6.2f} {s['mdd']*100:>6.1f}% {s['tpy']:>7.1f}")
    return out


def main():
    store = BarStore("data/bars")
    results = {}
    for uni in ("etf_wide", "largecap250"):
        results[uni] = run_universe(uni, store)

    # The survivorship gap, stated as a number.
    print("\n\n=== What survivorship is worth ===")
    print("The same strategy on a clean ETF universe and on a stock list that")
    print("is 2026 index membership. The difference is not skill.\n")
    print("Both windows now start on the same date, so the gap is not thirty")
    print("extra years of compounding wearing a survivorship label.\n")
    print(f"{'strategy':<34} {'etf_wide':>12} {'largecap250':>14} {'gap':>10}")
    for label in ("buy & hold SPY (benchmark)", "momentum top5 (abs filter)",
                  "momentum top10 (abs filter)", "reversion basket"):
        a = results["etf_wide"].get(label)
        b = results["largecap250"].get(label)
        if not a or not b:
            continue
        print(f"{label:<34} {a[1]['cagr']*100:>11.2f}% {b[1]['cagr']*100:>13.2f}% "
              f"{(b[1]['cagr']-a[1]['cagr'])*100:>+9.2f}pp")


if __name__ == "__main__":
    main()
