"""Express an underlying signal as 0DTE options.

The question this answers: given a directional signal that works (or does not)
on the stock, what would it have done bought as 0DTE contracts?

Everything here is priced from the model in `synth`, not from quotes, so the
output inherits every assumption in `iv.py`. Treat the numbers as an order of
magnitude, and sweep the vol assumption before believing any of them.

Position sizing is deliberately not the stock rule. A stock position sized at
95% of settled cash is survivable; the same notional in 0DTE premium is not,
because an option that expires out of the money is worth exactly zero. Sizing
is therefore a small fraction of equity spent on PREMIUM, and that fraction is
the single most important number in live trading - far more so than the signal.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import pricing as bs
from .iv import implied_vol
from .synth import atm_strike, minutes_to_close, option_series, quote_spread

#: Regulatory pass-through per contract, each way. Webull charges no commission
#: or contract fee on options, but OCC clearing and the Options Regulatory Fee
#: are passed through. Verify before trusting live P&L; these rates change.
FEE_PER_CONTRACT = 0.05
#: Contracts are for 100 shares.
MULTIPLIER = 100


@dataclass
class OptionTrade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    kind: str
    strike: float
    contracts: int
    entry_premium: float
    exit_premium: float
    underlying_entry: float
    underlying_exit: float
    pnl: float
    fees: float
    expired_worthless: bool
    exit_reason: str

    @property
    def return_pct(self) -> float:
        cost = self.contracts * self.entry_premium * MULTIPLIER
        return self.pnl / cost if cost else 0.0

    @property
    def minutes_held(self) -> float:
        return (self.exit_time - self.entry_time).total_seconds() / 60.0

    @property
    def underlying_move_bps(self) -> float:
        return (self.underlying_exit / self.underlying_entry - 1.0) * 10_000.0


def express_in_options(
    bars: pd.DataFrame,
    trades: list,
    symbol: str,
    kind: str = bs.CALL,
    starting_equity: float = 10_000.0,
    premium_pct: float = 0.10,
    iv_premium: float = 1.15,
    iv_floor: float | None = None,
    r: float = 0.04,
    min_premium: float = bs.MIN_TRADEABLE_PREMIUM,
    spread_frac: float = 0.005,
) -> tuple:
    """Re-express each underlying trade as a long 0DTE option.

    `premium_pct` is the fraction of equity spent on premium per trade. It
    defaults low on purpose: at 200x leverage, sizing like a stock position
    turns one wrong session into a catastrophe.

    Returns ``(option_trades, summary)``.
    """
    iv = implied_vol(bars["close"], premium=iv_premium, symbol=symbol, floor=iv_floor)
    mins_left = minutes_to_close(bars.index)

    equity = starting_equity
    out: list = []
    skipped = {"too_cheap": 0, "no_size": 0, "expired_before_exit": 0}

    for t in trades:
        if t.entry_time not in bars.index or t.exit_time not in bars.index:
            continue

        S_in = float(bars["close"].loc[t.entry_time])
        S_out = float(bars["close"].loc[t.exit_time])
        K = atm_strike(S_in)

        T_in = bs.minutes_to_years(float(mins_left.loc[t.entry_time]))
        T_out = bs.minutes_to_years(float(mins_left.loc[t.exit_time]))
        sig_in = float(iv.loc[t.entry_time])
        sig_out = float(iv.loc[t.exit_time])

        mid_in = float(bs.price(S_in, K, T_in, r, sig_in, kind))
        mid_out = float(bs.price(S_out, K, T_out, r, sig_out, kind))

        # A contract this cheap cannot be traded profitably: one tick of spread
        # is already a large fraction of it.
        if mid_in < min_premium:
            skipped["too_cheap"] += 1
            continue

        half_in = float(quote_spread(np.array([mid_in]), frac=spread_frac)[0]) / 2.0
        half_out = float(quote_spread(np.array([mid_out]), frac=spread_frac)[0]) / 2.0

        entry_px = mid_in + half_in           # cross the spread to buy
        exit_px = max(mid_out - half_out, 0.0)  # and again to sell
        expired = mid_out < min_premium
        if expired:
            # Nobody buys a worthless contract back; it simply expires.
            exit_px = 0.0

        contracts = int((equity * premium_pct) // (entry_px * MULTIPLIER))
        if contracts < 1:
            skipped["no_size"] += 1
            continue

        fees = FEE_PER_CONTRACT * contracts * (1 if expired else 2)
        pnl = (exit_px - entry_px) * contracts * MULTIPLIER - fees
        equity += pnl

        out.append(OptionTrade(
            entry_time=t.entry_time, exit_time=t.exit_time, kind=kind, strike=K,
            contracts=contracts, entry_premium=entry_px, exit_premium=exit_px,
            underlying_entry=S_in, underlying_exit=S_out, pnl=pnl, fees=fees,
            expired_worthless=expired, exit_reason=t.exit_reason,
        ))

    summary = {
        "trades": len(out),
        "final_equity": equity,
        "total_return_pct": (equity / starting_equity - 1.0) * 100.0,
        "skipped": skipped,
        "expired_worthless": sum(1 for o in out if o.expired_worthless),
        "win_rate_pct": (100.0 * np.mean([o.pnl > 0 for o in out])) if out else 0.0,
        "total_fees": sum(o.fees for o in out),
        "median_entry_premium": float(np.median([o.entry_premium for o in out])) if out else 0.0,
    }
    return out, summary


def breakeven_move_bps(
    S: float, K: float, minutes_left: float, sigma: float, hold_minutes: float,
    r: float = 0.04, kind: str = bs.CALL, spread_frac: float = 0.005,
) -> float:
    """Underlying move needed to break even after theta and both spreads.

    This is the honest hurdle for an options expression, and the number to
    compare against what the signal actually predicts. If the signal's typical
    move is smaller than this, options are the wrong instrument for it however
    good the signal is.
    """
    T_in = bs.minutes_to_years(minutes_left)
    T_out = bs.minutes_to_years(max(minutes_left - hold_minutes, 0.0))
    mid_in = float(bs.price(S, K, T_in, r, sigma, kind))
    if mid_in <= 0:
        return float("nan")

    half = float(quote_spread(np.array([mid_in]), frac=spread_frac)[0]) / 2.0
    target = mid_in + 2 * half  # pay the spread on the way in and the way out

    lo, hi = 0.0, 0.05  # search up to a 5% move in the underlying
    for _ in range(60):
        mid = (lo + hi) / 2.0
        S_new = S * (1 + mid) if kind == bs.CALL else S * (1 - mid)
        if float(bs.price(S_new, K, T_out, r, sigma, kind)) < target:
            lo = mid
        else:
            hi = mid
    return hi * 10_000.0
