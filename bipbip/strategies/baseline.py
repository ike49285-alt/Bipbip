"""Baselines. Every real strategy must beat these or it is not worth running."""
from __future__ import annotations


from ..core.strategy import Context, Strategy
from ..core.types import HOLD, Intent


class SessionBuyHold(Strategy):
    """Buy the first bar of the session, sell at the close. No skill involved.

    This is the benchmark that matters for an intraday system: it captures the
    open-to-close drift while paying exactly the same round-trip costs. A
    strategy that trades all day to underperform this has negative edge.
    """

    name = "session_buy_hold"

    def on_bar(self, ctx: Context) -> Intent:
        if not ctx.in_position and ctx.i == 0:
            return Intent(action="enter", reason="session_open")
        return HOLD


class NeverTrade(Strategy):
    """Holds cash forever. The floor: any strategy losing to this is destroying money."""

    name = "never_trade"

    def on_bar(self, ctx: Context) -> Intent:
        return HOLD
