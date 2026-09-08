"""VWAP reversion.

Session VWAP is the reference price institutional execution algorithms are
benchmarked against, which gives it genuine gravitational pull: price stretched
well below it tends to be bought back toward it. The edge is structural rather
than predictive, which is why it survives at a horizon a retail bot can reach.

The trade is only taken once the stretch stops widening. Buying a falling
market because it is "far from VWAP" is how this strategy loses money - a
trend day will stay stretched all session and stop you out on the way down.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core import indicators as ind
from ..core.strategy import Context, Strategy
from ..core.types import HOLD, Intent


class VWAPReversion(Strategy):
    name = "vwap_reversion"

    def __init__(
        self,
        stretch_atr: float = 1.5,
        atr_window: int = 30,
        rsi_window: int = 14,
        max_rsi: float = 35.0,
        stop_atr: float = 1.2,
        confirm_bars: int = 2,
        warmup: int = 30,
    ):
        """ATR spans 30 one-minute bars, not 14, so the stop sits outside
        minute-scale noise and can survive long enough to reach the target."""
        self.stretch_atr = stretch_atr
        self.atr_window = atr_window
        self.rsi_window = rsi_window
        self.max_rsi = max_rsi
        self.stop_atr = stop_atr
        self.confirm_bars = confirm_bars
        self.warmup_bars = warmup

    def prepare(self, bars: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=bars.index)
        out["vwap"] = ind.session_vwap(bars)
        out["atr"] = ind.atr(bars, self.atr_window)
        out["rsi"] = ind.rsi(bars["close"], self.rsi_window)
        return out

    def on_bar(self, ctx: Context) -> Intent:
        # Account rules are the engine's business; see opening_range.py.
        if ctx.in_position:
            return HOLD

        row = ctx.ind
        vwap, atr_v = float(row["vwap"]), float(row["atr"])
        if not (np.isfinite(vwap) and np.isfinite(atr_v)) or atr_v <= 0:
            return HOLD

        price = ctx.price
        stretch = (vwap - price) / atr_v
        if stretch < self.stretch_atr:
            return HOLD
        if float(row["rsi"]) > self.max_rsi:
            return HOLD

        # Require the last `confirm_bars` closes to be turning up: never catch
        # a falling knife on a trend day.
        closes = ctx.bars["close"].iloc[-(self.confirm_bars + 1) :]
        if len(closes) < self.confirm_bars + 1:
            return HOLD
        if not bool((closes.diff().dropna() > 0).all()):
            return HOLD

        return Intent(
            action="enter",
            reason=f"vwap_stretch_{stretch:.1f}atr",
            stop_price=price - self.stop_atr * atr_v,
            target_price=vwap,  # mean reversion targets the mean, nothing more
        )
