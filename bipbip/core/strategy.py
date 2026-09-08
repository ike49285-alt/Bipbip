"""Strategy interface.

A strategy sees history up to and including the current bar's close and
returns an `Intent`. It never sees a future bar, never touches the account,
and never places an order itself - the engine owns execution, costs, and the
account rules. Keeping strategies this thin is what makes their results
comparable and their bugs shallow.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import pandas as pd

from .types import HOLD, Intent


@dataclass
class Context:
    """Everything a strategy is permitted to know at the current bar."""

    symbol: str
    bars: pd.DataFrame  # session bars up to and including the current one
    indicators: pd.DataFrame  # precomputed causal indicators, same index
    i: int  # positional index of the current bar within the session
    in_position: bool
    entry_price: float
    bars_held: int
    # False when account rules forbid a new entry. Informational only:
    # the engine enforces the rules and logs what they refused, so a
    # strategy that gates on this hides its own blocked signals.
    can_open: bool
    session_date: dt.date
    minutes_to_close: int
    prior_session_close: float = float("nan")

    @property
    def bar(self) -> pd.Series:
        return self.bars.iloc[self.i]

    @property
    def ind(self) -> pd.Series:
        return self.indicators.iloc[self.i]

    @property
    def price(self) -> float:
        return float(self.bars["close"].iloc[self.i])


class Strategy:
    """Base class. Subclasses override `on_bar` and optionally the hooks."""

    name = "base"
    #: Extra warmup bars required before the first decision of a session.
    warmup_bars = 0

    def prepare(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Compute causal indicators once for the whole history.

        Must return a frame aligned to `bars.index`. Anything non-causal here
        silently poisons every result, so keep it to functions from
        `bipbip.core.indicators`.
        """
        return pd.DataFrame(index=bars.index)

    def on_session_start(self, session_date: dt.date) -> None:
        """Reset any per-session state."""

    def on_bar(self, ctx: Context) -> Intent:
        raise NotImplementedError

    def describe(self) -> str:
        return self.name
