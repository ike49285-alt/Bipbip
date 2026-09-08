"""Opening-range breakout.

The first thirty minutes set a high and a low on genuine two-sided volume.
Breaking that high later in the session is one of the few intraday patterns
that survives retail latency, because the signal plays out over tens of
minutes rather than milliseconds - there is nothing for a co-located firm to
front-run in a move that takes an hour.

Filters, each earning its place:
  * the range must be complete - trading a range still forming is trading noise
  * volume must confirm; a breakout on thin volume is the classic failure mode
  * price must hold above session VWAP, so buyers are in control on the day
  * the range must be wide enough in ATR terms to clear round-trip costs

With one round trip a day, selectivity is not a preference. The first
qualifying signal spends the entire day's budget, so the filters are the
strategy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core import indicators as ind
from ..core.strategy import Context, Strategy
from ..core.types import HOLD, Intent


class OpeningRangeBreakout(Strategy):
    name = "opening_range_breakout"

    def __init__(
        self,
        or_minutes: int = 30,
        atr_window: int = 30,
        stop_frac: float = 0.5,
        target_r: float = 2.0,
        min_rvol: float = 1.2,
        min_range_bps: float = 15.0,
        require_vwap: bool = True,
        min_risk_multiple: float = 2.0,
    ):
        """Risk is quoted in OPENING RANGE WIDTH, not one-minute ATR.

        This matters more than it sounds. A one-minute ATR on SPY is a few
        cents; a stop one of those away from entry is inside the noise and gets
        hit within minutes, which turns a multi-hour strategy into an expensive
        random number generator. The opening range is the natural risk unit
        here - it is measured over the same half hour whose break we are
        trading, so it scales with the day's actual volatility.
        """
        self.or_minutes = or_minutes
        self.atr_window = atr_window
        self.stop_frac = stop_frac
        self.target_r = target_r
        self.min_rvol = min_rvol
        self.min_range_bps = min_range_bps
        self.require_vwap = require_vwap
        self.min_risk_multiple = min_risk_multiple
        self.warmup_bars = or_minutes + 1

    def prepare(self, bars: pd.DataFrame) -> pd.DataFrame:
        out = ind.opening_range(bars, self.or_minutes)
        out["atr"] = ind.atr(bars, self.atr_window)
        out["vwap"] = ind.session_vwap(bars)
        out["rvol"] = ind.relative_volume(bars, 20)
        # Smallest risk unit worth trading, given what the round trip costs.
        out["min_risk"] = bars["close"] * (
            self.cost_hurdle_bps * self.min_risk_multiple / 10_000.0
        )
        return out

    def on_bar(self, ctx: Context) -> Intent:
        # Deliberately does not check ctx.can_open: the engine enforces account
        # rules and records what they refused. Gating here would hide that.
        if ctx.in_position:
            return HOLD

        row = ctx.ind
        if not bool(row["or_complete"]):
            return HOLD

        or_high, or_low = float(row["or_high"]), float(row["or_low"])
        if not (np.isfinite(or_high) and np.isfinite(or_low)):
            return HOLD

        width = or_high - or_low
        price = ctx.price
        if price <= 0 or width <= 0:
            return HOLD

        # A range too narrow in relative terms cannot produce a target that
        # clears the round trip. 15bp on SPY is roughly 7x the cost hurdle.
        if (width / price) * 10_000.0 < self.min_range_bps:
            return HOLD
        if price <= or_high:
            return HOLD
        if float(row["rvol"]) < self.min_rvol:
            return HOLD
        if self.require_vwap and price < float(row["vwap"]):
            return HOLD

        # Floored so a narrow range cannot place the stop inside the cost of
        # the trade, which produces stop-outs on the entry bar itself.
        risk = max(self.stop_frac * width, float(row["min_risk"]))
        return Intent(
            action="enter",
            reason=f"or_break@{or_high:.2f}",
            stop_price=price - risk,
            target_price=price + self.target_r * risk,
        )
