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
    # Full session frame. Never index past `i` - `bars` and `indicators` below
    # expose exactly the visible window and are what strategies should use.
    session_bars: pd.DataFrame
    session_indicators: pd.DataFrame
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
    def bars(self) -> pd.DataFrame:
        """Session bars up to and including the current one.

        Sliced lazily. Materialising this for every bar cost a DataFrame copy
        per call - millions of them across a significance run - while most
        strategies only need `ind` and `price`. Strategies that do need
        history pay for it; the rest no longer do.
        """
        return self.session_bars.iloc[: self.i + 1]

    @property
    def indicators(self) -> pd.DataFrame:
        """Indicator rows up to and including the current bar. Lazy, as above."""
        return self.session_indicators.iloc[: self.i + 1]

    @property
    def bar(self) -> pd.Series:
        return self.session_bars.iloc[self.i]

    @property
    def ind(self) -> pd.Series:
        return self.session_indicators.iloc[self.i]

    @property
    def price(self) -> float:
        return float(self.session_bars["close"].iat[self.i])


class Strategy:
    """Base class. Subclasses override `on_bar` and optionally the hooks."""

    name = "base"
    #: Extra warmup bars required before the first decision of a session.
    warmup_bars = 0
    #: Round-trip cost in bps, injected by the engine before `prepare` is
    #: called. Strategies should floor their risk unit at a multiple of this:
    #: a stop tighter than the round trip loses money by construction.
    cost_hurdle_bps = 3.0

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
