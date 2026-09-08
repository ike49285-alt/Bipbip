"""Value types shared by the engine, strategies, and reporting."""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

FLAT = "flat"
LONG = "long"


@dataclass
class Intent:
    """What a strategy wants to do, evaluated at the close of the current bar.

    The engine fills the resulting order at the NEXT bar's open, so an intent
    is a request, never a fill.
    """

    action: str  # "enter" | "exit" | "hold"
    reason: str = ""
    stop_price: Optional[float] = None
    target_price: Optional[float] = None
    size_pct: float = 1.0  # fraction of the affordable position to take


HOLD = Intent(action="hold")


@dataclass
class Trade:
    """A completed round trip."""

    symbol: str
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: pd.Timestamp
    exit_price: float
    shares: float
    pnl: float
    fees: float
    entry_reason: str = ""
    exit_reason: str = ""

    @property
    def return_pct(self) -> float:
        cost = self.shares * self.entry_price
        return self.pnl / cost if cost else 0.0

    @property
    def hold_minutes(self) -> float:
        return (self.exit_time - self.entry_time).total_seconds() / 60.0

    @property
    def is_win(self) -> bool:
        return self.pnl > 0


@dataclass
class BacktestResult:
    symbol: str
    equity_curve: pd.Series
    trades: list = field(default_factory=list)
    blocked: list = field(default_factory=list)
    starting_equity: float = 0.0
    metrics: dict = field(default_factory=dict)

    @property
    def final_equity(self) -> float:
        return float(self.equity_curve.iloc[-1]) if len(self.equity_curve) else self.starting_equity

    def trades_frame(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame(
            [
                {
                    "entry_time": t.entry_time,
                    "exit_time": t.exit_time,
                    "shares": t.shares,
                    "entry_price": round(t.entry_price, 4),
                    "exit_price": round(t.exit_price, 4),
                    "pnl": round(t.pnl, 2),
                    "return_pct": round(t.return_pct * 100, 3),
                    "hold_min": round(t.hold_minutes, 1),
                    "fees": round(t.fees, 3),
                    "entry_reason": t.entry_reason,
                    "exit_reason": t.exit_reason,
                }
                for t in self.trades
            ]
        )
