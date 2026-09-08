"""Execution cost model.

A backtest that ignores friction will show an edge at this horizon that does
not exist. Three costs are modelled:

1. Slippage - you cross the spread on entry and again on exit, and price moves
   during the 100-500ms your order spends in flight to a retail broker.
2. Regulatory fees - charged on SELLS only, by the SEC and FINRA.
3. Commission - zero at Webull for US stocks and ETFs, but kept configurable
   so the model survives a change of broker.

Rates are defaults, not guarantees. The SEC fee in particular is reset every
fiscal year; verify it before trusting live P&L.
"""
from __future__ import annotations

from dataclasses import dataclass, field

BUY = "buy"
SELL = "sell"


@dataclass
class CostModel:
    commission_per_trade: float = 0.0
    sec_fee_per_million: float = 27.80
    finra_taf_per_share: float = 0.000166
    finra_taf_max: float = 8.30
    slippage_bps: dict = field(default_factory=lambda: {"SPY": 1.0, "TQQQ": 2.5, "default": 3.0})

    def slippage_for(self, symbol: str) -> float:
        return float(self.slippage_bps.get(symbol.upper(), self.slippage_bps.get("default", 3.0)))

    def fill_price(self, side: str, reference_price: float, symbol: str) -> float:
        """Apply slippage so the fill is always worse than the reference price."""
        bps = self.slippage_for(symbol) / 10_000.0
        if side == BUY:
            return reference_price * (1.0 + bps)
        if side == SELL:
            return reference_price * (1.0 - bps)
        raise ValueError(f"side must be {BUY!r} or {SELL!r}, got {side!r}")

    def fees(self, side: str, shares: float, price: float) -> float:
        """Total non-slippage cost of a fill. Regulatory fees apply to sells only."""
        total = self.commission_per_trade
        if side == SELL:
            proceeds = shares * price
            total += proceeds * (self.sec_fee_per_million / 1_000_000.0)
            total += min(shares * self.finra_taf_per_share, self.finra_taf_max)
        return total

    def round_trip_cost_bps(self, symbol: str, price: float = 500.0, shares: float = 100.0) -> float:
        """Approximate all-in cost of one round trip, in basis points of notional.

        This is the hurdle: a strategy whose average gross edge per trade is
        below this number loses money no matter how good its hit rate looks.
        """
        notional = price * shares
        if notional <= 0:
            return 0.0
        buy_px = self.fill_price(BUY, price, symbol)
        sell_px = self.fill_price(SELL, price, symbol)
        slip = (buy_px - sell_px) * shares
        fee = self.fees(BUY, shares, buy_px) + self.fees(SELL, shares, sell_px)
        return (slip + fee) / notional * 10_000.0

    @classmethod
    def from_config(cls, cfg: dict) -> "CostModel":
        c = cfg.get("costs", cfg) or {}
        return cls(
            commission_per_trade=float(c.get("commission_per_trade", 0.0)),
            sec_fee_per_million=float(c.get("sec_fee_per_million", 27.80)),
            finra_taf_per_share=float(c.get("finra_taf_per_share", 0.000166)),
            finra_taf_max=float(c.get("finra_taf_max", 8.30)),
            slippage_bps=dict(c.get("slippage_bps", {"SPY": 1.0, "TQQQ": 2.5, "default": 3.0})),
        )
