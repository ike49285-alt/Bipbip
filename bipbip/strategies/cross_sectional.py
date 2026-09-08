"""Cross-sectional strategies: rank the universe, hold the best.

Ranking is a materially easier statistical problem than forecasting. "Which of
these forty is strongest" only needs the ORDER to be right, and errors common
to every symbol - a market-wide move, a volatility spike - cancel out of a
ranking rather than corrupting it. Forecasting one symbol's direction has no
such cancellation, which is why 33 years of one series told us so little.

It also solves the idle-capital problem. A reversion rule that holds a position
10% of the time earns a high return while invested and almost nothing per
calendar year; run across forty names, something is nearly always triggering.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.panel import Panel
from ..core.portfolio import PortfolioContext, PortfolioStrategy


class EqualWeightBuyHold(PortfolioStrategy):
    """Hold every listed symbol, equally weighted. The benchmark to beat."""

    name = "equal_weight"
    warmup_bars = 1

    def target_weights(self, ctx: PortfolioContext) -> dict:
        n = len(ctx.tradeable)
        return {s: 1.0 / n for s in ctx.tradeable} if n else {}


class MomentumRanking(PortfolioStrategy):
    """Hold the strongest `top_n` by trailing return, skipping the last month.

    Cross-sectional momentum is among the most replicated anomalies in finance,
    documented across equities, bonds, commodities and currencies and back into
    the nineteenth century. The one-month skip is standard: the most recent
    month tends to REVERSE, so including it dilutes the signal.
    """

    name = "momentum"

    def __init__(self, lookback: int = 252, skip: int = 21, top_n: int = 5,
                 abs_filter: bool = True):
        self.lookback = lookback
        self.skip = skip
        self.top_n = top_n
        # Absolute filter: hold cash rather than the "best" of a falling market.
        self.abs_filter = abs_filter
        self.warmup_bars = lookback + skip + 1

    def prepare(self, panel: Panel) -> dict:
        c = panel.closes
        return {"mom": np.log(c.shift(self.skip) / c.shift(self.skip + self.lookback))}

    def target_weights(self, ctx: PortfolioContext) -> dict:
        mom = ctx.ind("mom").reindex(ctx.tradeable).dropna()
        if mom.empty:
            return {}
        if self.abs_filter:
            mom = mom[mom > 0]
            if mom.empty:
                return {}
        picks = mom.nlargest(min(self.top_n, len(mom))).index
        return {s: 1.0 / len(picks) for s in picks}


class MeanReversionBasket(PortfolioStrategy):
    """Buy the most oversold names that are still in an uptrend.

    The single-symbol RSI(2) rule applied across a universe. Alone it sat in
    cash 90% of the time; spread over forty symbols the same rule is nearly
    always holding something, which is what turns a high return-per-exposure
    into a return per calendar year.
    """

    name = "reversion_basket"

    def __init__(self, rsi_window: int = 2, entry_rsi: float = 15.0,
                 trend_window: int = 200, top_n: int = 5, exit_rsi: float = 60.0):
        self.rsi_window = rsi_window
        self.entry_rsi = entry_rsi
        self.trend_window = trend_window
        self.top_n = top_n
        self.exit_rsi = exit_rsi
        self.warmup_bars = trend_window + 1

    def prepare(self, panel: Panel) -> dict:
        from ..core import indicators as ind

        c = panel.closes
        rsi = c.apply(lambda s: ind.rsi(s.dropna(), self.rsi_window).reindex(s.index))
        trend = c.rolling(self.trend_window, min_periods=self.trend_window).mean()
        return {"rsi": rsi, "trend": trend, "above": c > trend}

    def target_weights(self, ctx: PortfolioContext) -> dict:
        rsi = ctx.ind("rsi").reindex(ctx.tradeable)
        above = ctx.ind("above").reindex(ctx.tradeable).fillna(False)

        # Keep existing holdings until they are no longer oversold.
        keep = [s for s, w in ctx.current_weights.items()
                if w > 1e-6 and s in ctx.tradeable
                and np.isfinite(rsi.get(s, np.nan)) and rsi[s] < self.exit_rsi]

        candidates = rsi[above & (rsi < self.entry_rsi)].dropna()
        room = self.top_n - len(keep)
        picks = list(keep)
        if room > 0 and not candidates.empty:
            for s in candidates.nsmallest(min(room, len(candidates))).index:
                if s not in picks:
                    picks.append(s)
        return {s: 1.0 / len(picks) for s in picks} if picks else {}


from .swing_multi import DualMomentum, VolTargetTrend  # noqa: E402
from .turn_of_month import TurnOfMonthRotation  # noqa: E402

REGISTRY = {
    "equal_weight": EqualWeightBuyHold,
    "turn_of_month": TurnOfMonthRotation,
    "dual_momentum": DualMomentum,
    "vol_target_trend": VolTargetTrend,
    "momentum": MomentumRanking,
    "reversion_basket": MeanReversionBasket,
}


def get_portfolio_strategy(name: str, **kw) -> PortfolioStrategy:
    if name not in REGISTRY:
        raise ValueError(f"unknown portfolio strategy {name!r}; choose from {sorted(REGISTRY)}")
    return REGISTRY[name](**kw)
