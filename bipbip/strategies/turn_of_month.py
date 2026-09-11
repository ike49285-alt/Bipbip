"""Turn-of-month rotation.

A disproportionate share of the equity risk premium arrives in the few sessions
surrounding each month boundary. That much REPRODUCES: on the archive, SPY's
in-window sessions return +7.50 bps a day against +1.97 for the rest, and the
twelve broad ETFs checked all show it in the same direction.

WHAT THIS STRATEGY DOES WITH IT DOES NOT WORK, and the earlier version of this
docstring overstated the case in three separate ways. Each is a trap CLAUDE.md
names explicitly, so they are recorded rather than quietly deleted.

  * IT LOSES TO BUY-AND-HOLD. 2002-2026, holding SPY in the window and SHY
    outside it, 24 switches a year: 8.41% a year against SPY's 11.24% - behind
    by 2.83 points BEFORE any cost, and cost barely moves it (8.37% at SPY's
    0.133 bps crossing, 7.81% at the modelled 2.28). The per-day effect is
    real and it is not harvestable this way: being out of the market for two
    thirds of the year forfeits more drift than the window gains. The old text
    compared 7.39% against 5.17%, which is this strategy against ANOTHER
    VERSION OF ITSELF. The benchmark is buy-and-hold.
  * "EIGHT OF EIGHT IS ONE IN 256" IS NOT AN INDEPENDENT COUNT. Those eight
    assets share the same dates and most of their variance; a market-wide
    turn-of-month effect gives eight correlated draws, not eight coin flips.
    Collapsing to one observation per date takes the pooled t from 10.42 to
    2.70 - the clustering trap, to a factor of four.
  * IT HAS DECAYED. The old text claimed 9.36, 7.24, 8.52 and 7.98 bps across
    the four decades and called that stability. Measured with one observation
    per date: +5.74 (t=1.21), +14.37 (t=2.32), +4.72 (t=1.16) and +2.08
    (t=0.34). It is carried by 2000-2009 and the last six years are nothing.

What survives is narrow and worth stating exactly, because it is not nothing:
the in-window minus out-of-window spread, sampled once per in-window block so
the observations do not overlap, is +50.7 bps over 290 blocks at t=2.93, and it
holds in both halves (+52.4 and +49.0). That is a real long-short spread. It is
not a rotation strategy, because the leg you would be short is the market.

The window is counted in SESSIONS rather than calendar days so a holiday cannot
shift it off the boundary it straddles.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.panel import Panel
from ..core.portfolio import PortfolioContext, PortfolioStrategy


def turn_of_month_mask(dates: pd.DatetimeIndex, last_n: int = 4,
                       first_m: int = 4) -> pd.Series:
    """True on the last `last_n` and first `first_m` sessions of each month.

    Counted in SESSIONS, not calendar days, so holidays do not shift the window
    off the boundary it is meant to straddle.
    """
    s = pd.Series(1, index=dates)
    month = dates.to_period("M")
    pos = s.groupby(month).cumcount()
    size = s.groupby(month).transform("size")
    from_end = size - pos - 1
    return pd.Series((from_end < last_n) | (pos < first_m), index=dates)


class TurnOfMonthRotation(PortfolioStrategy):
    """Hold `risk_symbol` around the month boundary, `park_symbol` otherwise."""

    name = "turn_of_month"

    def __init__(self, risk_symbol: str = "SPY", park_symbol: str = "SHY",
                 last_n: int = 4, first_m: int = 4):
        self.risk_symbol = risk_symbol
        self.park_symbol = park_symbol
        self.last_n = last_n
        self.first_m = first_m
        self.warmup_bars = 25

    def prepare(self, panel: Panel) -> dict:
        mask = turn_of_month_mask(panel.dates, self.last_n, self.first_m)
        return {"in_window": pd.DataFrame(
            np.repeat(mask.to_numpy()[:, None], len(panel.symbols), axis=1),
            index=panel.dates, columns=panel.symbols)}

    def target_weights(self, ctx: PortfolioContext) -> dict:
        in_window = bool(ctx.ind("in_window").iloc[0])
        want = self.risk_symbol if in_window else self.park_symbol
        if want in ctx.tradeable:
            return {want: 1.0}
        # Fall back to whichever leg is available rather than going to cash,
        # which would silently change the strategy in early history.
        other = self.park_symbol if in_window else self.risk_symbol
        return {other: 1.0} if other in ctx.tradeable else {}
