"""VWAP reversion.

Session VWAP is the benchmark institutional execution algorithms are measured
against, which gives it real gravitational pull: price stretched well below it
tends to be bought back toward it. The edge is structural rather than
predictive, which is why it survives at a horizon a retail bot can reach.

Displacement is measured as a VWAP Z-SCORE, not in ATRs. Normalising a
session-cumulative displacement by a one-minute ATR is a timescale error:
distance from VWAP accumulates all session while ATR is per-minute, so on real
TQQQ data the median "stretch" was 3.7 ATR and a 1.5-ATR threshold fired
almost every bar. The z-score divides by the dispersion of price around VWAP
measured on the same clock, so a threshold of 2 means the same thing at 09:45
and 15:30 and fires on roughly 9% of bars.

The trade is only taken once the stretch stops widening. Buying a falling
market because it is "far from VWAP" is how this strategy loses money - a trend
day stays stretched all session and stops you out on the way down.
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
        stretch_z: float = 2.0,
        atr_window: int = 30,
        rsi_window: int = 14,
        max_rsi: float = 35.0,
        stop_atr: float = 1.2,
        confirm_bars: int = 2,
        warmup: int = 30,
        min_risk_multiple: float = 2.0,
    ):
        self.stretch_z = stretch_z
        self.atr_window = atr_window
        self.rsi_window = rsi_window
        self.max_rsi = max_rsi
        self.stop_atr = stop_atr
        self.confirm_bars = confirm_bars
        self.warmup_bars = warmup
        self.min_risk_multiple = min_risk_multiple

    def prepare(self, bars: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=bars.index)
        bands = ind.session_vwap_bands(bars)
        out["vwap"] = bands["vwap"]
        out["vwap_z"] = ind.zscore_from_bands(bars["close"], bands)
        out["rsi"] = ind.rsi(bars["close"], self.rsi_window)
        # Floor the risk unit at a multiple of the round trip, so a stop can
        # never sit inside the cost of the trade that sets it.
        out["risk"] = ind.cost_floored_risk(
            bars["close"], ind.atr(bars, self.atr_window),
            self.cost_hurdle_bps, self.min_risk_multiple,
        )
        return out

    def on_bar(self, ctx: Context) -> Intent:
        # Account rules are the engine's business; see opening_range.py.
        if ctx.in_position:
            return HOLD

        row = ctx.ind
        z, vwap, risk = float(row["vwap_z"]), float(row["vwap"]), float(row["risk"])
        if not (np.isfinite(z) and np.isfinite(vwap) and np.isfinite(risk)) or risk <= 0:
            return HOLD

        # Negative z means below VWAP; this is a long-only mean-reversion trade.
        if z > -self.stretch_z:
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

        price = ctx.price
        return Intent(
            action="enter",
            reason=f"vwap_z={z:.1f}",
            stop_price=price - self.stop_atr * risk,
            target_price=vwap,  # mean reversion targets the mean, nothing more
        )
