"""Daily swing strategies.

These are deliberately canonical rather than novel. Both have been published
and traded for decades, so their behaviour is documented elsewhere and a
surprising result here means a bug in this engine rather than a discovery.
That makes them a test of the machinery as much as of the market.

They also sidestep the constraint that has gated everything else in this
project: they run on daily bars, of which there are 8,459 going back to 1993,
rather than the 21 sessions in the minute archive.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core import indicators as ind
from ..core.strategy import Context, Strategy
from ..core.types import HOLD, Intent


class SMATrend(Strategy):
    """Hold while price is above its long moving average, otherwise cash.

    The oldest trend filter there is. It does not try to predict; it tries to
    be absent during the declines that do most of the damage to a compounded
    return.
    """

    name = "sma_trend"

    def __init__(self, window: int = 200):
        self.window = window
        self.warmup_bars = window + 1

    def prepare(self, bars: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=bars.index)
        out["sma"] = ind.sma(bars["close"], self.window)
        return out

    def on_bar(self, ctx: Context) -> Intent:
        sma = float(ctx.ind["sma"])
        if not np.isfinite(sma):
            return HOLD
        above = ctx.price > sma
        if not ctx.in_position and above:
            return Intent(action="enter", reason=f"above_sma{self.window}")
        if ctx.in_position and not above:
            return Intent(action="exit", reason=f"below_sma{self.window}")
        return HOLD


class RSI2Reversion(Strategy):
    """Buy short-term weakness inside a long-term uptrend.

    The Connors construction: a two-period RSI is an extremely fast
    oversold gauge, and the long moving average keeps the trade on the side of
    the primary trend so it is buying a dip rather than catching a collapse.
    Exits on strength rather than on a fixed target.
    """

    name = "rsi2_reversion"

    def __init__(self, rsi_window: int = 2, entry_rsi: float = 10.0,
                 trend_window: int = 200, exit_window: int = 5):
        self.rsi_window = rsi_window
        self.entry_rsi = entry_rsi
        self.trend_window = trend_window
        self.exit_window = exit_window
        self.warmup_bars = trend_window + 1

    def prepare(self, bars: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=bars.index)
        out["rsi"] = ind.rsi(bars["close"], self.rsi_window)
        out["trend"] = ind.sma(bars["close"], self.trend_window)
        out["exit_ma"] = ind.sma(bars["close"], self.exit_window)
        return out

    def on_bar(self, ctx: Context) -> Intent:
        row = ctx.ind
        rsi, trend, exit_ma = (float(row["rsi"]), float(row["trend"]), float(row["exit_ma"]))
        if not (np.isfinite(rsi) and np.isfinite(trend) and np.isfinite(exit_ma)):
            return HOLD

        price = ctx.price
        if not ctx.in_position:
            # Only buy dips inside an uptrend; the filter is what stops this
            # from averaging into a bear market.
            if price > trend and rsi < self.entry_rsi:
                return Intent(action="enter", reason=f"rsi{self.rsi_window}={rsi:.0f}")
            return HOLD

        if price > exit_ma:
            return Intent(action="exit", reason="closed_above_ma")
        return HOLD


class BuyAndHoldSwing(Strategy):
    """Buy once, hold forever. The benchmark every swing strategy must beat."""

    name = "buy_and_hold"

    def __init__(self):
        self.warmup_bars = 1

    def on_bar(self, ctx: Context) -> Intent:
        if not ctx.in_position and ctx.can_open:
            return Intent(action="enter", reason="inception")
        return HOLD
