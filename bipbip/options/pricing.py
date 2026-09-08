"""Black-Scholes pricing and greeks, on a TRADING-TIME clock.

The clock matters more than anything else here, and it is where a naive
implementation goes wrong.

Volatility estimated from one-minute bars is annualised over TRADING minutes
(252 sessions x 390 minutes). Time to expiry must therefore be measured on the
same clock, or the two disagree by the ratio of calendar to trading hours.
Pricing an ATM 0DTE call on SPY at $640 with 13% vol gives about $0.91 on a
calendar clock and about $2.10 on a trading clock. Real SPY 0DTE options trade
near the latter, because markets accumulate variance while open and very little
overnight. Calendar time systematically under-prices short-dated options, and
an options backtest built on it would look far too profitable.

All functions are vectorised over numpy arrays and clamp at expiry, where the
option is worth exactly its intrinsic value.
"""
from __future__ import annotations

import numpy as np

#: Minutes in a trading year - the clock everything here shares.
TRADING_MINUTES_PER_YEAR = 252 * 390
#: One regular session, in trading-time years.
SESSION_YEARS = 390 / TRADING_MINUTES_PER_YEAR

CALL, PUT = "call", "put"


def _norm_cdf(x):
    from scipy.special import ndtr

    return ndtr(x)


def _norm_pdf(x):
    return np.exp(-0.5 * np.square(x)) / np.sqrt(2.0 * np.pi)


def minutes_to_years(minutes) -> np.ndarray:
    """Convert minutes remaining into trading-time years."""
    return np.asarray(minutes, dtype="float64") / TRADING_MINUTES_PER_YEAR


def _d1_d2(S, K, T, r, sigma):
    S = np.asarray(S, dtype="float64")
    K = np.asarray(K, dtype="float64")
    T = np.maximum(np.asarray(T, dtype="float64"), 0.0)
    sigma = np.maximum(np.asarray(sigma, dtype="float64"), 1e-9)

    sqrt_T = np.sqrt(T)
    denom = np.where(sqrt_T > 0, sigma * sqrt_T, np.nan)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / denom
    return d1, d1 - sigma * sqrt_T, T, sqrt_T


def price(S, K, T, r=0.04, sigma=0.15, kind=CALL) -> np.ndarray:
    """Option price. At or past expiry this is exactly intrinsic value."""
    d1, d2, T, _ = _d1_d2(S, K, T, r, sigma)
    S = np.asarray(S, dtype="float64")
    K = np.asarray(K, dtype="float64")
    disc = K * np.exp(-r * T)

    if kind == CALL:
        live = S * _norm_cdf(d1) - disc * _norm_cdf(d2)
        intrinsic = np.maximum(S - K, 0.0)
    elif kind == PUT:
        live = disc * _norm_cdf(-d2) - S * _norm_cdf(-d1)
        intrinsic = np.maximum(K - S, 0.0)
    else:
        raise ValueError(f"kind must be {CALL!r} or {PUT!r}, got {kind!r}")

    return np.where(T > 0, np.nan_to_num(live, nan=intrinsic), intrinsic)


def delta(S, K, T, r=0.04, sigma=0.15, kind=CALL) -> np.ndarray:
    d1, _, T, _ = _d1_d2(S, K, T, r, sigma)
    S = np.asarray(S, dtype="float64")
    K = np.asarray(K, dtype="float64")
    if kind == CALL:
        expired = (S > K).astype("float64")
        return np.where(T > 0, np.nan_to_num(_norm_cdf(d1), nan=expired), expired)
    expired = -(S < K).astype("float64")
    return np.where(T > 0, np.nan_to_num(_norm_cdf(d1) - 1.0, nan=expired), expired)


def gamma(S, K, T, r=0.04, sigma=0.15) -> np.ndarray:
    """Same for calls and puts. Enormous for 0DTE near the strike, which is why
    a position's delta can invert within minutes."""
    d1, _, T, sqrt_T = _d1_d2(S, K, T, r, sigma)
    S = np.asarray(S, dtype="float64")
    sigma = np.maximum(np.asarray(sigma, dtype="float64"), 1e-9)
    out = _norm_pdf(d1) / (S * sigma * sqrt_T)
    return np.where(T > 0, np.nan_to_num(out, nan=0.0, posinf=0.0), 0.0)


def theta_per_minute(S, K, T, r=0.04, sigma=0.15, kind=CALL) -> np.ndarray:
    """Decay per TRADING MINUTE - the unit an intraday holder actually pays.

    Reported per minute rather than per day because a 0DTE position is held for
    minutes to hours, and a per-day figure invites the mistake of thinking the
    decay is spread evenly across the session. It is not: it accelerates as the
    square root of remaining time collapses.
    """
    d1, d2, T, sqrt_T = _d1_d2(S, K, T, r, sigma)
    S = np.asarray(S, dtype="float64")
    K = np.asarray(K, dtype="float64")
    sigma = np.maximum(np.asarray(sigma, dtype="float64"), 1e-9)

    term = -(S * _norm_pdf(d1) * sigma) / (2.0 * sqrt_T)
    disc = K * np.exp(-r * T)
    annual = term - r * disc * _norm_cdf(d2) if kind == CALL else term + r * disc * _norm_cdf(-d2)
    per_min = annual / TRADING_MINUTES_PER_YEAR
    return np.where(T > 0, np.nan_to_num(per_min, nan=0.0), 0.0)


def vega(S, K, T, r=0.04, sigma=0.15) -> np.ndarray:
    """Sensitivity to a 1.00 (100 point) change in vol; divide by 100 for 1%."""
    d1, _, T, sqrt_T = _d1_d2(S, K, T, r, sigma)
    S = np.asarray(S, dtype="float64")
    out = S * _norm_pdf(d1) * sqrt_T
    return np.where(T > 0, np.nan_to_num(out, nan=0.0), 0.0)


#: Below this premium a contract is not meaningfully tradeable: one tick of
#: spread is already a double-digit percentage of its value, and ratios like
#: leverage stop carrying information.
MIN_TRADEABLE_PREMIUM = 0.05


def leverage(S, K, T, r=0.04, sigma=0.15, kind=CALL,
             min_premium: float = MIN_TRADEABLE_PREMIUM) -> np.ndarray:
    """Elasticity: percent move in the option per percent move in the underlying.

    This is the number that makes 0DTE dangerous rather than merely expensive.
    An ATM 0DTE call runs hundreds of times the underlying's percentage move,
    so position sizing dominates signal quality.

    NaN below `min_premium`. As a contract decays toward zero the ratio diverges
    - it reached 21,000 on real data, and over 1e26 for the spread percentage -
    which is arithmetic rather than an opportunity. A contract that cheap cannot
    be traded profitably at any leverage.
    """
    p = price(S, K, T, r, sigma, kind)
    d = delta(S, K, T, r, sigma, kind)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.abs(d) * np.asarray(S, dtype="float64") / p
    out = np.where(np.asarray(p) >= min_premium, out, np.nan)
    return np.nan_to_num(out, nan=np.nan, posinf=np.nan)
