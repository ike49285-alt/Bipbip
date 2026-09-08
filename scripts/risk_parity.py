"""Levered risk parity: the one construction with a theoretical reason to work.

Every other idea tested here tries to predict something. This one does not. It
rests on a documented structural fact: investors who want return but cannot or
will not borrow bid up risky assets instead, so high-volatility assets are
persistently overpriced per unit of risk and low-volatility ones underpriced.
A stock/bond blend therefore has a HIGHER Sharpe ratio than stocks alone, even
though it earns less.

That is only useful if you can borrow, because the way you convert a better
Sharpe into more money is to lever the blend up to the volatility of stocks.
The user has a margin account, so this is available.

It could not be tested before now: with dividends stripped out of the archive,
bonds appeared to return nothing at all, and the entire premise is that their
return is real.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np, pandas as pd

from swing_compare import stats, HDR, show, START
from bipbip.core.costs import CostModel
from bipbip.core.panel import load_panel
from bipbip.core.portfolio import PortfolioEngine, PortfolioStrategy
from bipbip.data.store import BarStore


def _month_start(dates):
    m = dates.to_period("M")
    return pd.Series(m != np.roll(m, 1), index=dates)


class FixedBlend(PortfolioStrategy):
    """A static weighting, rebalanced monthly. 60/40 and friends."""

    def __init__(self, weights: dict, label: str):
        self.weights, self.name, self.warmup_bars = weights, label, 5

    def prepare(self, panel):
        return {"ms": _month_start(panel.dates)}

    def target_weights(self, ctx):
        if not bool(ctx.indicators["ms"].iloc[ctx.i]):
            return None
        avail = {s: w for s, w in self.weights.items() if s in ctx.tradeable}
        if not avail:
            return None
        total = sum(avail.values())
        return {s: w / total for s, w in avail.items()}


class RiskParity(PortfolioStrategy):
    """Weight each sleeve by the inverse of its own trailing volatility.

    No forecasting: the only input is how much each asset has been moving, and
    the rule is that each should contribute the same amount of risk.
    """

    def __init__(self, sleeves, vol_window=120, label="risk_parity"):
        self.sleeves, self.vol_window = list(sleeves), vol_window
        self.name, self.warmup_bars = label, vol_window + 5

    def prepare(self, panel):
        r = np.log(panel.closes / panel.closes.shift(1))
        return {"vol": r.rolling(self.vol_window, min_periods=self.vol_window).std(),
                "ms": _month_start(panel.dates)}

    def target_weights(self, ctx):
        if not bool(ctx.indicators["ms"].iloc[ctx.i]):
            return None
        vol = ctx.ind("vol")
        ok = [s for s in self.sleeves if s in ctx.tradeable
              and np.isfinite(vol.get(s, np.nan)) and vol.get(s, 0) > 0]
        if not ok:
            return None
        raw = {s: 1.0 / float(vol[s]) for s in ok}
        t = sum(raw.values())
        return {s: w / t for s, w in raw.items()}


class HoldOne(PortfolioStrategy):
    def __init__(self, symbol):
        self.symbol, self.name, self.warmup_bars = symbol, f"hold_{symbol}", 1
    def target_weights(self, ctx):
        return {self.symbol: 1.0} if self.symbol in ctx.tradeable else None


def lever(curve, mult, borrow_rate, start=START):
    """Apply constant leverage to a realised equity curve, after financing.

    Daily-rebalanced leverage, same as a levered ETF, so volatility decay is
    included rather than assumed away. This is what a margin account actually
    does to a strategy's returns.
    """
    r = curve.pct_change().fillna(0.0)
    lr = mult * r - (mult - 1.0) * borrow_rate / 252.0
    return start * (1.0 + lr.clip(lower=-0.99)).cumprod()


def main():
    store = BarStore("data/bars")
    syms = ["SPY", "TLT", "IEF", "SHY", "GLD", "AGG"]
    panel = load_panel(store, syms, "1d")
    have = [s for s in syms if s in panel.closes.columns]
    print(f"symbols: {have}")
    print(f"window: {panel.dates[0].date()} -> {panel.dates[-1].date()} "
          f"({len(panel)} sessions)\n")

    stock = "SPY"
    bond = "TLT" if "TLT" in have else "IEF"
    cands = [
        ("buy & hold SPY", HoldOne(stock)),
        (f"hold {bond}", HoldOne(bond)),
        ("60/40", FixedBlend({stock: 0.6, bond: 0.4}, "60_40")),
        ("risk parity (2 sleeve)", RiskParity([stock, bond])),
        ("risk parity (+gold)", RiskParity([s for s in (stock, bond, "GLD") if s in have],
                                           label="rp3")),
    ]

    print(HDR)
    curves = {}
    for label, strat in cands:
        res = PortfolioEngine(CostModel(), starting_equity=START,
                              settle_days=0).run(panel, strat)
        c = res.equity_curve.dropna()
        c = c[c > 0]
        curves[label] = c
        show(label, stats(c, res.trades, start=float(c.iloc[0])))

    # The actual question: lever the best-Sharpe blend to SPY's volatility.
    bh = curves["buy & hold SPY"]
    target_vol = bh.pct_change().std()
    print("\nLevered to SPY's volatility, financed at a range of rates.")
    print("(Leverage is applied daily, so decay is included, not assumed away.)")
    for label in ("60/40", "risk parity (2 sleeve)", "risk parity (+gold)"):
        c = curves[label]
        mult = target_vol / c.pct_change().std()
        print(f"\n  {label}: needs {mult:.2f}x to match SPY's volatility")
        print(f"    {'rate':>6} {'final':>10} {'CAGR':>8} {'Shrp':>6} {'maxDD':>7}")
        for rate in (0.0, 0.02, 0.035, 0.05):
            lc = lever(c, mult, rate, start=float(c.iloc[0]))
            s = stats(lc, [], start=float(c.iloc[0]))
            if s is None:
                continue
            print(f"    {rate:>5.1%} ${s['final']:>9,.0f} {s['cagr']*100:>7.2f}% "
                  f"{s['sharpe']:>6.2f} {s['mdd']*100:>6.1f}%")
    sb = stats(bh, [], start=float(bh.iloc[0]))
    print(f"\n  buy & hold SPY for reference: ${sb['final']:,.0f}  "
          f"{sb['cagr']*100:.2f}%  Sharpe {sb['sharpe']:.2f}  maxDD {sb['mdd']*100:.1f}%")


if __name__ == "__main__":
    main()
