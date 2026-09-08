"""Native option risk management.

Wrapping a stock strategy in options and keeping the stock's stops produces a
large loss out of a flat result. Measured on this project's own signals, option
winners averaged +27.6% while losers averaged -53.7% - a win/loss magnitude
ratio of 0.51, which cannot be profitable at any hit rate near 50%.

Three things caused that, and this module addresses each:

1. HOLDING TIME. Theta is the dominant cost, and it accelerates. One trade held
   152 minutes on a +33bp underlying move returned +11.7%; another held 21
   minutes on a smaller +24bp move returned +54.4%. The move was not what
   differed. A hard time stop is the single largest improvement available.

2. LOSSES RUNNING. Underlying stops sat 13-21bp away, which at 200x leverage is
   already 60-70% of premium by the time they trigger. Losses have to be capped
   in PREMIUM terms, where the risk actually lives.

3. STRIKE CHOICE. An at-the-money 0DTE contract is 100% extrinsic value, so it
   is a pure bet on theta not happening. A modestly in-the-money strike is
   mostly intrinsic: at delta 0.87 decay over 45 minutes falls from 7.3% of
   premium to 1.4%, and the payoff becomes near-symmetric. The cost is less
   leverage, which is a benefit rather than a price.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import pricing as bs


@dataclass
class OptionRiskModel:
    """Risk rules expressed in the option's own terms, not the underlying's."""

    #: Strike selection. 0.5 is at-the-money; higher is further in-the-money and
    #: carries less extrinsic value, so less to lose to time.
    target_delta: float = 0.75
    #: Exit when premium falls this fraction below entry.
    premium_stop_pct: float = 0.30
    #: Exit when premium rises this fraction above entry.
    premium_target_pct: float = 0.50
    #: Hard time stop. Theta accelerates; a stale position bleeds regardless of
    #: whether the thesis is still intact.
    max_hold_minutes: int = 45
    #: Refuse entries with less than this long to expiry - decay is fastest at
    #: the end and there is no time for the thesis to work.
    min_minutes_left: int = 90
    #: Fraction of equity spent on premium per trade. An option can expire
    #: worth exactly zero, so this is the number that decides survival.
    premium_pct: float = 0.10
    #: Strike increment for the underlying's listed chain.
    strike_increment: float = 1.0

    def describe(self) -> str:
        return (f"delta {self.target_delta:.2f}, stop -{self.premium_stop_pct:.0%}, "
                f"target +{self.premium_target_pct:.0%}, max {self.max_hold_minutes}m")


def strike_for_delta(
    S: float, T: float, sigma: float, target_delta: float,
    kind: str = bs.CALL, r: float = 0.04, increment: float = 1.0,
) -> float:
    """Nearest listed strike whose delta is closest to `target_delta`.

    Searched over the listed grid rather than solved analytically, because only
    listed strikes are tradeable and rounding an exact solution can land on a
    materially different delta for a 0DTE contract, where delta moves fast.
    """
    if T <= 0 or sigma <= 0:
        return round(S / increment) * increment

    # Widen the search with volatility and time; +-6% covers any 0DTE case.
    span = max(increment * 2, S * 0.06)
    grid = np.arange(round((S - span) / increment) * increment,
                     round((S + span) / increment) * increment + increment,
                     increment)
    grid = grid[grid > 0]
    if grid.size == 0:
        return round(S / increment) * increment

    deltas = np.abs(bs.delta(S, grid, T, r, sigma, kind))
    return float(grid[int(np.argmin(np.abs(deltas - abs(target_delta))))])
