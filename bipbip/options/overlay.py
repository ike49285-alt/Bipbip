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

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import pricing as bs
from .iv import implied_vol
from .risk import strike_for_delta
from .synth import minutes_to_close, quote_spread

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


def _option_px(S, K, T, r, sigma, kind):
    return float(bs.price(S, K, T, r, sigma, kind))


def simulate_option_trade(
    bars: pd.DataFrame, entry_time, underlying_exit_time, kind: str,
    iv: pd.Series, mins_left: pd.Series, risk, equity: float,
    r: float = 0.04, spread_frac: float = 0.005,
    min_premium: float = bs.MIN_TRADEABLE_PREMIUM,
):
    """Walk the option's own price path, exiting on whichever comes first.

    Exits considered each bar, in this order:
      1. premium stop   - the loss cap, checked first so an ambiguous bar
                          resolves against the trader
      2. premium target
      3. time stop      - theta accelerates; a stale position bleeds
      4. the underlying signal's own exit
      5. the closing bell

    Intrabar extremes are approximated by pricing the option at the bar's low
    and high, mirroring how the equity engine treats stops, and a gap through
    the stop fills at the gapped price rather than the stop.
    """
    idx = bars.index
    if entry_time not in idx:
        return None, "entry_bar_missing"

    start = idx.get_loc(entry_time)
    minutes_at_entry = float(mins_left.iloc[start])
    if minutes_at_entry < risk.min_minutes_left:
        return None, "too_late_in_session"

    S_in = float(bars["close"].iloc[start])
    sig_in = float(iv.iloc[start])
    T_in = bs.minutes_to_years(minutes_at_entry)
    K = strike_for_delta(S_in, T_in, sig_in, risk.target_delta, kind, r,
                         risk.strike_increment)

    mid_in = _option_px(S_in, K, T_in, r, sig_in, kind)
    if mid_in < min_premium:
        return None, "too_cheap"

    half_in = float(quote_spread(np.array([mid_in]), frac=spread_frac)[0]) / 2.0
    entry_px = mid_in + half_in
    contracts = int((equity * risk.premium_pct) // (entry_px * MULTIPLIER))
    if contracts < 1:
        return None, "no_size"

    stop_px = entry_px * (1.0 - risk.premium_stop_pct)
    target_px = entry_px * (1.0 + risk.premium_target_pct)

    # The bar the position must be closed by, whatever else happens.
    session = idx[start].date()
    last = start
    while last + 1 < len(idx) and idx[last + 1].date() == session:
        last += 1
    signal_exit = idx.get_loc(underlying_exit_time) if underlying_exit_time in idx else last
    time_stop = start + int(risk.max_hold_minutes)
    hard_stop = min(last, signal_exit, time_stop)

    for j in range(start + 1, hard_stop + 1):
        T_j = bs.minutes_to_years(float(mins_left.iloc[j]))
        sig_j = float(iv.iloc[j])
        o_open = _option_px(float(bars["open"].iloc[j]), K, T_j, r, sig_j, kind)
        lo_u, hi_u = float(bars["low"].iloc[j]), float(bars["high"].iloc[j])
        # A call is worth least at the underlying's low; a put at its high.
        o_low = _option_px(lo_u if kind == bs.CALL else hi_u, K, T_j, r, sig_j, kind)
        o_high = _option_px(hi_u if kind == bs.CALL else lo_u, K, T_j, r, sig_j, kind)

        if o_low <= stop_px:
            # Gapping through the stop fills at the gap, not at the stop.
            fill = min(stop_px, o_open)
            return _close(bars, idx, start, j, K, kind, contracts, entry_px, fill,
                          spread_frac, "premium_stop", min_premium), None
        if o_high >= target_px:
            fill = max(target_px, o_open)
            return _close(bars, idx, start, j, K, kind, contracts, entry_px, fill,
                          spread_frac, "premium_target", min_premium), None

    j = hard_stop
    T_j = bs.minutes_to_years(float(mins_left.iloc[j]))
    fill = _option_px(float(bars["close"].iloc[j]), K, T_j, r, float(iv.iloc[j]), kind)
    reason = ("time_stop" if j == time_stop else
              "signal_exit" if j == signal_exit else "session_close")
    return _close(bars, idx, start, j, K, kind, contracts, entry_px, fill,
                  spread_frac, reason, min_premium), None


def _close(bars, idx, start, j, K, kind, contracts, entry_px, mid_out,
           spread_frac, reason, min_premium):
    """Build the completed trade, paying the spread on the way out."""
    expired = mid_out < min_premium
    if expired:
        exit_px = 0.0  # nobody buys back a worthless contract
    else:
        half = float(quote_spread(np.array([mid_out]), frac=spread_frac)[0]) / 2.0
        exit_px = max(mid_out - half, 0.0)

    fees = FEE_PER_CONTRACT * contracts * (1 if expired else 2)
    pnl = (exit_px - entry_px) * contracts * MULTIPLIER - fees
    return OptionTrade(
        entry_time=idx[start], exit_time=idx[j], kind=kind, strike=K,
        contracts=contracts, entry_premium=entry_px, exit_premium=exit_px,
        underlying_entry=float(bars["close"].iloc[start]),
        underlying_exit=float(bars["close"].iloc[j]),
        pnl=pnl, fees=fees, expired_worthless=expired, exit_reason=reason,
    )


def express_in_options(
    bars: pd.DataFrame,
    trades: list,
    symbol: str,
    kind: str = bs.CALL,
    starting_equity: float = 10_000.0,
    risk=None,
    iv_premium: float = 1.15,
    iv_floor: float | None = None,
    r: float = 0.04,
    min_premium: float = bs.MIN_TRADEABLE_PREMIUM,
    spread_frac: float = 0.005,
) -> tuple:
    """Re-express each underlying signal as a long option, managed natively.

    The underlying strategy supplies only the ENTRY. Exits are the risk model's,
    because stops sized in the underlying's ATRs are already 60-70% of premium
    by the time they trigger at this leverage.
    """
    from .risk import OptionRiskModel

    risk = risk or OptionRiskModel()
    iv = implied_vol(bars["close"], premium=iv_premium, symbol=symbol, floor=iv_floor)
    mins_left = minutes_to_close(bars.index)

    equity = starting_equity
    out: list = []
    skipped: dict = {}

    for t in trades:
        trade, why = simulate_option_trade(
            bars, t.entry_time, t.exit_time, kind, iv, mins_left, risk,
            equity, r, spread_frac, min_premium,
        )
        if trade is None:
            skipped[why] = skipped.get(why, 0) + 1
            continue
        equity += trade.pnl
        out.append(trade)

    exits: dict = {}
    for o in out:
        exits[o.exit_reason] = exits.get(o.exit_reason, 0) + 1
    rets = np.array([o.return_pct for o in out]) if out else np.array([])
    wins, losses = rets[rets > 0], rets[rets <= 0]

    summary = {
        "trades": len(out),
        "final_equity": equity,
        "total_return_pct": (equity / starting_equity - 1.0) * 100.0,
        "skipped": skipped,
        "expired_worthless": sum(1 for o in out if o.expired_worthless),
        "win_rate_pct": float(100.0 * np.mean(rets > 0)) if out else 0.0,
        "total_fees": sum(o.fees for o in out),
        "median_entry_premium": float(np.median([o.entry_premium for o in out])) if out else 0.0,
        "mean_win_pct": float(wins.mean() * 100) if len(wins) else 0.0,
        "mean_loss_pct": float(losses.mean() * 100) if len(losses) else 0.0,
        "win_loss_ratio": float(abs(wins.mean() / losses.mean())) if len(wins) and len(losses) and losses.mean() else float("nan"),
        "median_hold_min": float(np.median([o.minutes_held for o in out])) if out else 0.0,
        "exit_breakdown": exits,
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
