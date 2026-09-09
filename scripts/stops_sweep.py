"""Do protective stops improve anything? Swept, not asserted.

A stop is the most reflexively recommended risk control there is, and it has a
specific cost that recommendations skip: it converts a temporary drawdown into
a realised loss, and equity indices spend most of their drawdowns recovering.
The question is whether the losses it truncates outweigh the recoveries it
sells into, which is an empirical question with a different answer per market.

Every combination is run against the SAME strategy with stops off, so the only
difference in each row is the stop.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np, pandas as pd

from swing_compare import stats, START
from bipbip.core.costs import CostModel
from bipbip.core.panel import build_panel
from bipbip.core.portfolio import PortfolioEngine, PortfolioStrategy, StopPolicy
from bipbip.data.store import BarStore


class HoldOne(PortfolioStrategy):
    def __init__(self, symbol="SPY"):
        self.symbol, self.name, self.warmup_bars = symbol, f"hold_{symbol}", 1
    def target_weights(self, ctx):
        return {self.symbol: 1.0} if self.symbol in ctx.tradeable else None


class SmaTiming(PortfolioStrategy):
    """Monthly SMA(200), the best risk-adjusted single-asset rule here."""
    def __init__(self, symbol="SPY", window=200):
        self.symbol, self.window = symbol, window
        self.name, self.warmup_bars = f"sma{window}", window + 5

    def prepare(self, panel):
        c = panel.closes
        m = panel.dates.to_period("M")
        return {"trend": c > c.rolling(self.window, min_periods=self.window).mean(),
                "ms": pd.Series(m != np.roll(m, 1), index=panel.dates)}

    def target_weights(self, ctx):
        if not bool(ctx.indicators["ms"].iloc[ctx.i]):
            return None
        if self.symbol not in ctx.tradeable:
            return None
        return {self.symbol: 1.0} if bool(ctx.ind("trend").get(self.symbol, False)) else {}


def sweep(panel, strategy, label):
    print(f"\n=== {label} ===")
    print(f"{'stop':<28} {'final':>9} {'CAGR':>8} {'Shrp':>6} {'maxDD':>7} "
          f"{'trades':>7} {'stops':>6} {'gapped':>7}")

    def row(name, stop):
        res = PortfolioEngine(CostModel(), starting_equity=START, settle_days=0,
                              stop=stop).run(panel, strategy)
        c = res.equity_curve.dropna(); c = c[c > 0]
        s = stats(c, res.trades, start=float(c.iloc[0]))
        gapped = sum(1 for f in res.stops if f["gapped"])
        print(f"{name:<28} ${s['final']:>8,.0f} {s['cagr']*100:>7.2f}% {s['sharpe']:>6.2f} "
              f"{s['mdd']*100:>6.1f}% {len(res.trades):>7} {len(res.stops):>6} {gapped:>7}")
        return s

    base = row("none", None)
    for pct in (0.05, 0.10, 0.20):
        for lock in (0, 21, 63):
            row(f"trailing {pct:.0%}, lock {lock}d",
                StopPolicy(kind="trailing", pct=pct, lockout_days=lock))
    for pct in (0.10, 0.20):
        row(f"fixed {pct:.0%}, lock 21d",
            StopPolicy(kind="fixed", pct=pct, lockout_days=21))
    return base


def main():
    store = BarStore("data/bars")
    spy = store.load("SPY", "1d").dropna()
    panel = build_panel({"SPY": spy})
    print(f"SPY {panel.dates[0].date()} -> {panel.dates[-1].date()} "
          f"({len(panel)} sessions)")
    sweep(panel, HoldOne("SPY"), "buy & hold SPY, with stops bolted on")
    sweep(panel, SmaTiming("SPY"), "SMA200 monthly, with stops bolted on")


if __name__ == "__main__":
    main()
