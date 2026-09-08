"""Multi-asset swing strategies with published track records.

Everything tested so far has been a variation on timing ONE asset, and all of
it lost to holding that asset. The approaches here differ structurally: they
choose among several assets, and they size positions by risk rather than
equally. Both are among the few things in the literature that have improved on
buy-and-hold out of sample rather than only in the sample that discovered them.

DUAL MOMENTUM combines relative and absolute strength. Relative picks the
stronger of two risk assets; absolute refuses to hold either when both are
weaker than cash. The second filter is what does the work - it is a rule for
being absent during the declines that dominate a compounded return.

VOLATILITY TARGETING sizes each position by the inverse of its own recent
volatility, so a calm asset gets more capital than a wild one. This is the most
consistently replicated improvement to risk-adjusted return in the literature,
and it costs nothing extra to implement.

Both rebalance monthly. That is deliberate: the cost analysis in this project
has repeatedly shown that an effect worth a few basis points cannot survive
daily trading, and monthly turnover keeps costs near 0.3% a year.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.panel import Panel
from ..core.portfolio import PortfolioContext, PortfolioStrategy


def _month_starts(dates: pd.DatetimeIndex) -> pd.Series:
    """True on the first session of each month."""
    month = dates.to_period("M")
    return pd.Series(month != np.roll(month, 1), index=dates)


class DualMomentum(PortfolioStrategy):
    """Hold the strongest risk asset, or the defensive asset if none is strong.

    Antonacci's construction. The absolute filter - refusing both risk assets
    when neither beats the defensive one - is the component that historically
    mattered, because it sidesteps the drawdowns rather than predicting them.
    """

    name = "dual_momentum"

    def __init__(self, risk_assets=("SPY", "EFA"), defensive: str = "SHY",
                 lookback: int = 252):
        self.risk_assets = list(risk_assets)
        self.defensive = defensive
        self.lookback = lookback
        self.warmup_bars = lookback + 5

    def prepare(self, panel: Panel) -> dict:
        c = panel.closes
        return {"mom": np.log(c / c.shift(self.lookback)),
                "month_start": pd.DataFrame(
                    np.repeat(_month_starts(panel.dates).to_numpy()[:, None],
                              len(c.columns), axis=1),
                    index=panel.dates, columns=c.columns)}

    def target_weights(self, ctx: PortfolioContext) -> dict | None:
        if not bool(ctx.ind("month_start").iloc[0]):
            return None  # abstain between rebalances; {} would sell to cash
        mom = ctx.ind("mom")
        avail = [a for a in self.risk_assets if a in ctx.tradeable
                 and np.isfinite(mom.get(a, np.nan))]
        if not avail:
            return {self.defensive: 1.0} if self.defensive in ctx.tradeable else {}

        best = max(avail, key=lambda a: float(mom[a]))
        defensive_mom = float(mom.get(self.defensive, 0.0)) if self.defensive in ctx.tradeable else 0.0
        if not np.isfinite(defensive_mom):
            defensive_mom = 0.0

        # Absolute filter: the winner must also beat the defensive asset.
        if float(mom[best]) > defensive_mom:
            return {best: 1.0}
        return {self.defensive: 1.0} if self.defensive in ctx.tradeable else {}


class VolTargetTrend(PortfolioStrategy):
    """Hold every asset in an uptrend, weighted by the inverse of its volatility.

    Long-only, because a cash account cannot short. Cash is held for whatever
    fraction of the book the trend filter leaves unallocated, which is itself
    the risk control: in a broad decline almost nothing qualifies.
    """

    name = "vol_target_trend"

    def __init__(self, trend_window: int = 200, vol_window: int = 60,
                 target_vol: float = 0.12, max_weight: float = 0.34):
        self.trend_window = trend_window
        self.vol_window = vol_window
        self.target_vol = target_vol
        # No single asset may dominate, however calm it looks.
        self.max_weight = max_weight
        self.warmup_bars = trend_window + 5

    def prepare(self, panel: Panel) -> dict:
        c = panel.closes
        rets = np.log(c / c.shift(1))
        return {
            "trend": c > c.rolling(self.trend_window, min_periods=self.trend_window).mean(),
            "vol": rets.rolling(self.vol_window, min_periods=self.vol_window).std() * np.sqrt(252),
            "month_start": pd.DataFrame(
                np.repeat(_month_starts(panel.dates).to_numpy()[:, None],
                          len(c.columns), axis=1),
                index=panel.dates, columns=c.columns),
        }

    def target_weights(self, ctx: PortfolioContext) -> dict | None:
        if not bool(ctx.ind("month_start").iloc[0]):
            return None
        trend = ctx.ind("trend").reindex(ctx.tradeable).fillna(False)
        vol = ctx.ind("vol").reindex(ctx.tradeable)

        picks = [s for s in ctx.tradeable
                 if bool(trend.get(s, False)) and np.isfinite(vol.get(s, np.nan))
                 and vol[s] > 0]
        if not picks:
            return {}  # nothing is trending: hold cash, which is the whole point

        # Inverse volatility, then scaled so the book targets `target_vol`.
        raw = {s: self.target_vol / float(vol[s]) for s in picks}
        raw = {s: min(w, self.max_weight) for s, w in raw.items()}
        total = sum(raw.values())
        if total > 1.0:
            raw = {s: w / total for s, w in raw.items()}
        return raw
