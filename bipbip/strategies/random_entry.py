"""Random-entry null strategy.

The question a short backtest cannot answer by itself is whether a strategy's
result came from TIMING or merely from EXPOSURE. In a falling market any
strategy that trades less loses less, which looks like skill and is not.

This strategy enters at a uniformly random eligible bar each session, using the
same stop and target as the strategy under test. Everything is held constant
except the decision of when to enter. Running it a few hundred times gives the
distribution of outcomes attributable to luck alone, and a real strategy has to
beat that distribution - not merely beat zero.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from ..core import indicators as ind
from ..core.strategy import Context, Strategy
from ..core.types import HOLD, Intent


class RandomEntry(Strategy):
    name = "random_entry"

    def __init__(
        self,
        seed: int = 0,
        stop_atr: float = 1.0,
        target_atr: float = 2.0,
        atr_window: int = 30,
        entry_rate: float = 1.0,
        warmup: int = 31,
        last_entry_bar: int = 330,
        risk_unit: str = "atr",
        or_minutes: int = 30,
        stop_frac: float = 0.5,
        target_r: float = 2.0,
    ):
        """`entry_rate` is the probability of trading at all in a session, so
        the null can be matched to the real strategy's selectivity rather than
        trading every day when the strategy does not."""
        self.rng = np.random.default_rng(seed)
        self.stop_atr = stop_atr
        self.target_atr = target_atr
        self.atr_window = atr_window
        self.entry_rate = entry_rate
        self.warmup_bars = warmup
        self.last_entry_bar = last_entry_bar
        # A null that trades a DIFFERENT risk geometry from the strategy under
        # test is not measuring timing skill, it is comparing two strategies.
        # "or_width" reproduces the opening-range sizing so the only difference
        # left is when the entry happens.
        self.risk_unit = risk_unit
        self.or_minutes = or_minutes
        self.stop_frac = stop_frac
        self.target_r = target_r
        self._entry_bar = None

    def prepare(self, bars: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=bars.index)
        out["atr"] = ind.atr(bars, self.atr_window)
        if self.risk_unit == "or_width":
            orng = ind.opening_range(bars, self.or_minutes)
            out["or_width"] = orng["or_high"] - orng["or_low"]
        return out

    def on_session_start(self, session_date: dt.date) -> None:
        if self.rng.random() > self.entry_rate:
            self._entry_bar = None  # sit this session out
        else:
            self._entry_bar = int(self.rng.integers(self.warmup_bars, self.last_entry_bar))

    def on_bar(self, ctx: Context) -> Intent:
        if ctx.in_position or self._entry_bar is None or ctx.i != self._entry_bar:
            return HOLD

        price = ctx.price
        if self.risk_unit == "or_width":
            width = float(ctx.ind["or_width"])
            if not np.isfinite(width) or width <= 0:
                return HOLD
            risk = self.stop_frac * width
            target_mult = self.target_r
        else:
            atr_v = float(ctx.ind["atr"])
            if not np.isfinite(atr_v) or atr_v <= 0:
                return HOLD
            risk = self.stop_atr * atr_v
            target_mult = self.target_atr / self.stop_atr

        return Intent(
            action="enter",
            reason="random",
            stop_price=price - risk,
            target_price=price + target_mult * risk,
        )
