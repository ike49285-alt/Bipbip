"""Leveraged trend following: the one construction that scales $50.

The reason a 3x fund is a poor buy-and-hold is volatility decay, and decay is
worst exactly when volatility is high - which is when a trend filter is out of
the market anyway. So the filter and the instrument fit together: the filter
removes the periods that hurt the fund most.

The signal is measured on the UNLEVERED index and executed in the fund, which
is both how it is done in practice and the only causal way to do it.
"""
import sys
import numpy as np, pandas as pd
# Repo ROOT first so `bipbip` resolves, then scripts/ for sibling
# imports. Only the second was here, so this ran only when something
# else had already fixed the path - an accident of import order.
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

from swing_compare import stats, HDR, show, START
from bipbip.core.costs import CostModel
from bipbip.core.panel import build_panel
from bipbip.core.portfolio import PortfolioEngine, PortfolioStrategy
from bipbip.data.leveraged import simulate_leveraged
from bipbip.data.store import BarStore


class LeveredTrend(PortfolioStrategy):
    """Hold the levered fund while the INDEX is above its average, else cash."""

    def __init__(self, index="IDX", fund="LEV", window=200, monthly=False,
                 confirm=0):
        self.index, self.fund, self.window = index, fund, window
        self.monthly, self.confirm = monthly, confirm
        self.name = f"lev_trend_{window}" + ("_m" if monthly else "")
        self.warmup_bars = window + confirm + 5

    def prepare(self, panel):
        c = panel.closes[self.index]
        above = c > c.rolling(self.window, min_periods=self.window).mean()
        if self.confirm:
            # Require the signal to hold for N days before acting, which is the
            # standard defence against whipsawing in and out around the line.
            above = above.rolling(self.confirm + 1, min_periods=self.confirm + 1).min().astype(bool)
        month = panel.dates.to_period("M")
        return {"above": above,
                "month_start": pd.Series(month != np.roll(month, 1), index=panel.dates)}

    def target_weights(self, ctx):
        if self.monthly and not bool(ctx.indicators["month_start"].iloc[ctx.i]):
            return None
        if self.fund not in ctx.tradeable:
            return None
        return {self.fund: 1.0} if bool(ctx.indicators["above"].iloc[ctx.i]) else {}


class HoldOne(PortfolioStrategy):
    def __init__(self, symbol):
        self.symbol, self.name, self.warmup_bars = symbol, f"hold_{symbol}", 1
    def target_weights(self, ctx):
        return {self.symbol: 1.0} if self.symbol in ctx.tradeable else None


def run(index_symbol, leverage=3.0):
    store = BarStore("data/bars")
    idx = store.load(index_symbol, "1d").dropna()
    lev = simulate_leveraged(idx, leverage=leverage)
    panel = build_panel({"IDX": idx, "LEV": lev})
    print(f"\n=== {index_symbol}, simulated {leverage:g}x  "
          f"({panel.dates[0].date()} -> {panel.dates[-1].date()}, "
          f"{len(panel)} sessions) ===")
    print(HDR)

    rows = [(f"hold {index_symbol} (1x)", HoldOne("IDX")),
            (f"hold {leverage:g}x (no filter)", HoldOne("LEV")),
            (f"{leverage:g}x, SMA200 daily", LeveredTrend()),
            (f"{leverage:g}x, SMA200 monthly", LeveredTrend(monthly=True)),
            (f"{leverage:g}x, SMA200 +3d confirm", LeveredTrend(confirm=3)),
            (f"{leverage:g}x, SMA100 daily", LeveredTrend(window=100))]

    curves = {}
    for label, strat in rows:
        res = PortfolioEngine(CostModel(), starting_equity=START,
                             settle_days=0).run(panel, strat)
        curves[label] = res.equity_curve.dropna()
        show(label, stats(curves[label], res.trades))

    print("\nCAGR within each period:")
    slices = [("93-99", "1993", "1999"), ("00-09", "2000", "2009"),
              ("10-19", "2010", "2019"), ("20-26", "2020", "2026")]
    print(f"{'strategy':<26}" + "".join(f"{n:>10}" for n, _, _ in slices))
    for label, c in curves.items():
        row = f"{label:<26}"
        for _, a, b in slices:
            s = c[a:b].dropna()
            if len(s) < 60 or float(s.iloc[0]) <= 0:
                row += f"{'-':>10}"; continue
            y = (s.index[-1] - s.index[0]).days / 365.25
            row += f"{((float(s.iloc[-1])/float(s.iloc[0]))**(1/y)-1)*100:>9.1f}%"
        print(row)


for sym in ("SPY", "QQQ"):
    run(sym)
