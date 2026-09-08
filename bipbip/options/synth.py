"""Synthetic 0DTE option series built from underlying bars.

Everything here is modelled, not observed. The output looks like market data
and is not: it is Black-Scholes evaluated on an estimated vol surface. Results
computed from it are worth exactly as much as the assumptions in `iv.py`.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from ..data.sessions import RTH_CLOSE
from . import pricing as bs
from .iv import implied_vol


def minutes_to_close(index: pd.DatetimeIndex) -> pd.Series:
    """Trading minutes remaining until the 16:00 expiry of each bar's session.

    0DTE options expire at the close of the session they belong to, so this is
    the option's entire remaining life.
    """
    close_t = (RTH_CLOSE.hour * 60) + RTH_CLOSE.minute
    now = np.asarray([ts.hour * 60 + ts.minute for ts in index], dtype="float64")
    return pd.Series(np.maximum(close_t - now, 0.0), index=index, name="minutes_left")


def atm_strike(price: float, increment: float = 1.0) -> float:
    """Nearest listed strike. SPY lists $1 increments; 0DTE is penny-quoted."""
    return round(price / increment) * increment


def option_series(
    bars: pd.DataFrame,
    strike: float,
    kind: str = bs.CALL,
    iv: pd.Series | None = None,
    r: float = 0.04,
    session: dt.date | None = None,
) -> pd.DataFrame:
    """Price one 0DTE contract across the bars of its session.

    Returns mid price, greeks, leverage and minutes remaining.
    """
    if session is not None:
        bars = bars[[ts.date() == session for ts in bars.index]]
    if bars.empty:
        return pd.DataFrame()

    S = bars["close"].to_numpy(dtype="float64")
    mins = minutes_to_close(bars.index).to_numpy()
    T = bs.minutes_to_years(mins)
    sigma = (implied_vol(bars["close"]) if iv is None else iv.reindex(bars.index)).to_numpy()

    out = pd.DataFrame(index=bars.index)
    out["underlying"] = S
    out["minutes_left"] = mins
    out["iv"] = sigma
    out["mid"] = bs.price(S, strike, T, r, sigma, kind)
    out["delta"] = bs.delta(S, strike, T, r, sigma, kind)
    out["gamma"] = bs.gamma(S, strike, T, r, sigma)
    out["theta_per_min"] = bs.theta_per_minute(S, strike, T, r, sigma, kind)
    out["leverage"] = bs.leverage(S, strike, T, r, sigma, kind)
    return out


def quote_spread(mid: np.ndarray, tick: float = 0.01, frac: float = 0.005) -> np.ndarray:
    """Modelled bid-ask spread in dollars of premium.

    Never tighter than one tick, and proportionally punishing on cheap options:
    a penny spread on a $0.10 contract is 10%, which is exactly why buying
    nearly-worthless 0DTE lottery tickets loses money even when directionally
    right. Spreads also widen into the close and away from the money; neither
    is modelled, so this is an optimistic floor.
    """
    return np.maximum(tick, frac * np.asarray(mid, dtype="float64"))
