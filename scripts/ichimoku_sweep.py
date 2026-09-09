"""Ichimoku and the full stochastic, against holding.

Two questions worth separating. Does the cloud beat buy-and-hold? And do the
four confirmations add anything to each other, given that all four measure
trend and therefore largely restate one another?
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np, pandas as pd

from swing_compare import stats, START
from bipbip.core.costs import CostModel
from bipbip.core.panel import build_panel, load_panel
from bipbip.core.portfolio import PortfolioEngine, PortfolioStrategy
from bipbip.data.store import BarStore
from bipbip.strategies.ichimoku import IchimokuCloud, IchimokuStochastic


class HoldOne(PortfolioStrategy):
    def __init__(self, symbol="SPY"):
        self.symbol, self.name, self.warmup_bars = symbol, f"hold_{symbol}", 1
    def target_weights(self, ctx):
        return {self.symbol: 1.0} if self.symbol in ctx.tradeable else None


def run(panel, strat, label, out=None):
    res = PortfolioEngine(CostModel(), starting_equity=START,
                          settle_days=0).run(panel, strat)
    c = res.equity_curve.dropna(); c = c[c > 0]
    s = stats(c, res.trades, start=float(c.iloc[0]))
    exposure = (res.weights.sum(axis=1) > 0.01).mean()
    print(f"{label:<34} ${s['final']:>8,.0f} {s['cagr']*100:>7.2f}% {s['sharpe']:>6.2f} "
          f"{s['mdd']*100:>6.1f}% {s['tpy']:>7.1f} {exposure*100:>7.0f}%")
    if out is not None:
        out[label] = c
    return c


HDR = (f"{'strategy':<34} {'final':>9} {'CAGR':>8} {'Shrp':>6} {'maxDD':>7} "
       f"{'trd/yr':>7} {'in mkt':>8}")


def main(symbol="SPY"):
    store = BarStore("data/bars")
    bars = store.load(symbol, "1d").dropna()
    panel = build_panel({symbol: bars})
    print(f"{symbol}: {panel.dates[0].date()} -> {panel.dates[-1].date()} "
          f"({len(panel)} sessions)\n")
    print(HDR)
    curves = {}
    run(panel, HoldOne(symbol), "buy & hold", curves)

    for n in (1, 2, 3, 4):
        run(panel, IchimokuCloud(confirmations=n, symbols=[symbol]),
            f"ichimoku, {n} confirmation" + ("s" if n > 1 else ""), curves)

    run(panel, IchimokuCloud(confirmations=4, symbols=[symbol], monthly=True),
        "ichimoku 4, monthly rebalance", curves)

    for entry in (60.0, 80.0, 100.0):
        run(panel, IchimokuStochastic(confirmations=4, symbols=[symbol],
                                      entry_below=entry),
            f"ichimoku 4 + FSTO entry<{entry:.0f}", curves)
    run(panel, IchimokuStochastic(confirmations=4, symbols=[symbol],
                                  entry_below=80.0, exit_above=80.0),
        "ichimoku 4 + FSTO exit>80", curves)

    print("\nCAGR within each period:")
    slices = [("93-99", "1993", "1999"), ("00-09", "2000", "2009"),
              ("10-19", "2010", "2019"), ("20-26", "2020", "2026")]
    print(f"{'strategy':<34}" + "".join(f"{n:>10}" for n, _, _ in slices))
    for label, c in curves.items():
        row = f"{label:<34}"
        for _, a, b in slices:
            x = c[a:b].dropna()
            if len(x) < 100 or float(x.iloc[0]) <= 0:
                row += f"{'-':>10}"; continue
            y = (x.index[-1] - x.index[0]).days / 365.25
            row += f"{((float(x.iloc[-1])/float(x.iloc[0]))**(1/y)-1)*100:>9.1f}%"
        print(row)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "SPY")
