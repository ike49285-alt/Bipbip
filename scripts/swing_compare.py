"""Honest comparison of swing candidates against buy-and-hold.

Everything runs through the SAME engine with the SAME cost model, so the only
difference between rows is the decision rule. Buy-and-hold is itself run as a
strategy rather than computed from raw prices, so it pays entry cost too.
"""
import sys, pathlib

# Every other script that imports bipbip does this; this one did not, so it
# raised ModuleNotFoundError on any invocation - `python3 scripts/...` puts
# scripts/ on sys.path, never the repo root. It took cross_sectional_run.py
# down with it, which imports this module.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np, pandas as pd

from bipbip.core.costs import CostModel
from bipbip.core.panel import load_panel
from bipbip.core.portfolio import PortfolioEngine, PortfolioStrategy
from bipbip.data.store import BarStore
from bipbip.data.universe import get_universe
from bipbip.strategies.swing_multi import DualMomentum, VolTargetTrend, _month_starts

START = 50.0


class HoldOne(PortfolioStrategy):
    """Buy one symbol on day one and never trade again."""
    def __init__(self, symbol="SPY"):
        self.symbol, self.name, self.warmup_bars = symbol, f"hold_{symbol}", 1
    def target_weights(self, ctx):
        return {self.symbol: 1.0} if self.symbol in ctx.tradeable else {}


class SmaTiming(PortfolioStrategy):
    """Faber: hold the asset while it is above its 200-day average, else cash.

    The canonical swing benchmark. If a timing rule cannot beat this, it is not
    adding anything that a single moving average does not already give away.
    """
    def __init__(self, symbol="SPY", window=200, monthly=True):
        self.symbol, self.window, self.monthly = symbol, window, monthly
        self.name = f"sma{window}_{symbol}" + ("" if monthly else "_daily")
        self.warmup_bars = window + 5
    def prepare(self, panel):
        c = panel.closes
        return {"trend": c > c.rolling(self.window, min_periods=self.window).mean(),
                "month_start": pd.DataFrame(
                    np.repeat(_month_starts(panel.dates).to_numpy()[:, None],
                              len(c.columns), axis=1), index=panel.dates, columns=c.columns)}
    def target_weights(self, ctx):
        if self.monthly and not bool(ctx.ind("month_start").iloc[0]):
            return None          # abstain between rebalances
        if self.symbol not in ctx.tradeable:
            return None
        return {self.symbol: 1.0} if bool(ctx.ind("trend").get(self.symbol, False)) else {}


def stats(curve, trades, start=START):
    curve = curve.dropna()
    if len(curve) < 2:
        return None
    years = (curve.index[-1] - curve.index[0]).days / 365.25
    final = float(curve.iloc[-1])
    r = curve.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    dd = float((curve / curve.cummax() - 1).min())
    return dict(final=final, cagr=(final / start) ** (1 / years) - 1,
                sharpe=(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0.0,
                mdd=abs(dd), years=years, tpy=len(trades) / years)


HDR = f"{'strategy':<26} {'final':>10}  {'CAGR':>7}  {'Shrp':>5}  {'maxDD':>6}  {'trd/yr':>6}"

def show(label, s):
    if s is None:
        print(f"{label:<26} {'(no data)':>10}"); return
    print(f"{label:<26} ${s['final']:>9,.0f}  {s['cagr']*100:>6.2f}%  "
          f"{s['sharpe']:>5.2f}  {s['mdd']*100:>5.1f}%  {s['tpy']:>6.1f}")


def main(universe="etf_wide"):
    panel = load_panel(BarStore("data/bars"), get_universe(universe), "1d")
    print(f"universe {universe}: {len(panel.closes.columns)} symbols, "
          f"{len(panel)} sessions, {panel.dates[0].date()} -> {panel.dates[-1].date()}")
    spy = panel.closes["SPY"].dropna()
    print(f"SPY {spy.iloc[0]:.2f} -> {spy.iloc[-1]:.2f}  "
          f"({(spy.iloc[-1]/spy.iloc[0])**(365.25/((spy.index[-1]-spy.index[0]).days))-1:+.2%}/yr price path)\n")

    cands = [("buy & hold SPY", HoldOne("SPY"), 0),
             ("SMA200 SPY (monthly)", SmaTiming("SPY"), 0),
             ("dual momentum", DualMomentum(), 0),
             ("dual momentum (cash acct)", DualMomentum(), 1),
             ("vol-target trend", VolTargetTrend(), 0),
             ("vol-target trend (cash)", VolTargetTrend(), 1)]

    print(HDR)
    curves = {}
    for label, strat, settle in cands:
        res = PortfolioEngine(CostModel(), starting_equity=START,
                              settle_days=settle).run(panel, strat, universe=universe)
        curves[label] = res.equity_curve.dropna()
        show(label, stats(curves[label], res.trades))

    # Decade stability. A rule that only worked in one decade is a fitted
    # coordinate, not an edge.
    print("\nCAGR by period (all start from the same date, so these are")
    print("annualised returns WITHIN each slice, not compounded from $50):")
    slices = [("1993-1999", "1993", "1999"), ("2000-2009", "2000", "2009"),
              ("2010-2019", "2010", "2019"), ("2020-2026", "2020", "2026")]
    print(f"{'strategy':<26}" + "".join(f"{n:>12}" for n, _, _ in slices))
    for label in curves:
        row = f"{label:<26}"
        for _, a, b in slices:
            c = curves[label][a:b].dropna()
            if len(c) < 60 or float(c.iloc[0]) <= 0:
                row += f"{'-':>12}"; continue
            yrs = (c.index[-1] - c.index[0]).days / 365.25
            row += f"{((float(c.iloc[-1])/float(c.iloc[0]))**(1/yrs)-1)*100:>11.1f}%"
        print(row)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "etf_wide")
