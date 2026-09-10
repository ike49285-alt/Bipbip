"""Higher-timeframe exit filter.

Suppresses a strategy's DISCRETIONARY exit while the higher timeframe is still
driving in the position's direction, so a two-minute retrace does not shake you
out of a thirty-minute move.

The justification is arithmetic before it is predictive. Measured on this
archive, a retrace during a driving 30-minute trend is worth perhaps 0.3bp more
than the same retrace in a flat tape - directionally consistent across SPY and
TQQQ, but well inside the error bars. An avoided exit, however, saves an entire
round trip: 2.28bp on SPY and 5.30bp on TQQQ. The cost saving is roughly ten
times the size of the predictive effect being hoped for, so at a hit rate near
50% not churning pays for itself whether or not the trend signal is real.

THE SAFETY PROPERTY IS STRUCTURAL, NOT A CONVENTION. This filter can only see
and alter intents the strategy returns. Stops, targets and the force-flat at
the closing bell are evaluated by the engine, which never consults a strategy,
so no configuration of this filter can widen a stop or carry a position
overnight. That matters because "the higher timeframe is still intact" is
exactly the sentence a trader says while a small loss becomes a large one - it
is unfalsifiable in the moment, since on a three-hour chart a six-minute trade
never looks like it is failing yet. Higher-timeframe context earns the right to
stop you fidgeting. It never earns the right to move your stop.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from ..core.strategy import Context, Strategy
from ..core.types import HOLD, Intent


def trend_zscore(close: pd.Series, window: int = 30, scale_window: int = 240) -> pd.Series:
    """Trailing `window`-minute return, normalised by its own recent scale.

    Normalising matters: a raw 30-minute return means something different at
    09:45 than at 15:30, and different again between SPY and TQQQ. The z-score
    makes one threshold portable across both.
    """
    key = np.asarray([ts.date() for ts in close.index])
    same = pd.Series(key, index=close.index).eq(pd.Series(key, index=close.index).shift(window))
    ret = np.log(close / close.shift(window)).where(same)
    scale = ret.rolling(scale_window, min_periods=max(30, scale_window // 4)).std()
    return (ret / scale.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).rename("trend_z")


class HTFExitFilter(Strategy):
    """Wraps a strategy, holding through noise while the higher timeframe drives."""

    def __init__(
        self,
        inner: Strategy,
        htf_window: int = 30,
        min_trend_z: float = 1.0,
        max_suppressed_bars: int = 30,
    ):
        self.inner = inner
        self.htf_window = htf_window
        self.min_trend_z = min_trend_z
        # A cap so a persistent trend reading cannot defer an exit indefinitely.
        # Stops and the closing bell already bound the position; this bounds the
        # filter's own influence, which is easier to reason about.
        self.max_suppressed_bars = max_suppressed_bars
        self.name = f"{inner.name}+htf"
        self.warmup_bars = max(inner.warmup_bars, htf_window + 1)
        self._suppressed = 0
        self.suppressed_events: list = []

    def prepare(self, bars: pd.DataFrame) -> pd.DataFrame:
        # The engine sets cost_hurdle_bps on this wrapper; the inner strategy
        # needs it too or its risk floor silently reverts to a default.
        self.inner.cost_hurdle_bps = self.cost_hurdle_bps
        out = self.inner.prepare(bars).copy()
        if "htf_trend_z" in out.columns:
            raise ValueError("inner strategy already defines htf_trend_z")
        out["htf_trend_z"] = trend_zscore(bars["close"], self.htf_window)
        return out

    def on_session_start(self, session_date: dt.date) -> None:
        self.inner.on_session_start(session_date)
        self._suppressed = 0

    def on_bar(self, ctx: Context) -> Intent:
        intent = self.inner.on_bar(ctx)

        if intent.action != "exit" or not ctx.in_position:
            if not ctx.in_position:
                self._suppressed = 0
            return intent

        z = float(ctx.ind.get("htf_trend_z", np.nan))
        if not np.isfinite(z):
            return intent
        # Long-only book, so only an upward higher-timeframe drive defers an exit.
        if z < self.min_trend_z:
            return intent
        if self._suppressed >= self.max_suppressed_bars:
            return intent

        self._suppressed += 1
        self.suppressed_events.append({
            "time": ctx.bars.index[ctx.i], "trend_z": z,
            "n_this_trade": self._suppressed,
        })
        return HOLD

    def describe(self) -> str:
        return (f"{self.inner.describe()} + HTF exit filter "
                f"({self.htf_window}m, z>{self.min_trend_z}, cap {self.max_suppressed_bars})")
