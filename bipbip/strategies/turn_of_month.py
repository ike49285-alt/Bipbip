"""Turn-of-month rotation.

A disproportionate share of the equity risk premium has historically arrived in
the few sessions surrounding each month boundary. On SPY over 34 years those
days returned 8.20 bps each against 2.5 bps for the rest of the month, and the
gap is not a recent artefact: it measures 9.36, 7.24, 8.52 and 7.98 bps across
the four decades from 1993, spanning the dot-com collapse and 2008, while the
non-boundary days swing from +6.17 to -4.63.

That stability is what distinguishes this from everything else tested in this
project. The meta-labelling edge lived entirely in 1995-2002 and was gone by
2014; this one has not decayed.

Evidence, and its limits:

  * Shifting the eight-day window to any other phase of the month drops the
    return to 2.97 bps on average, best case 5.30.
  * Eight assets not used to find the effect - EFA, EEM, XLV, XLI, XLB, VNQ,
    EWJ, MDY - all show it, and in most the non-boundary days earn nothing or
    less. Eight of eight is roughly a one-in-256 coincidence.
  * A permutation test on SPY ALONE gives p = 0.081, which is not conventional
    significance. The cross-asset consistency carries this result, not the
    single-asset p-value, and that is worth remembering before sizing it.

Mechanically it is cheap: about 24 switches a year, so costs take roughly
0.28% annually rather than the 5.7% that daily trading would.

The capital is parked in short Treasuries between windows rather than left in
cash, because idle money earning nothing for two thirds of the year is what
turned the equivalent single-asset version from a 7.39% strategy into a 5.17%
one.
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
