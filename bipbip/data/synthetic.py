"""Synthetic intraday bars for testing the engine without a data provider.

This exists to verify mechanics - settlement, fills, force-flat, accounting -
not to evaluate strategies. Numbers produced from synthetic bars say nothing
about whether an edge is real. The generator reproduces the intraday features
that break naive engine code: overnight gaps, a U-shaped volume profile,
volatility clustering, and an elevated opening range.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .sessions import EXCHANGE_TZ


def make_intraday_bars(
    n_sessions: int = 30,
    start_date: str = "2025-01-02",
    start_price: float = 500.0,
    annual_drift: float = 0.08,
    annual_vol: float = 0.18,
    seed: int = 7,
    bars_per_session: int = 390,
) -> pd.DataFrame:
    """Generate `n_sessions` of one-minute OHLCV bars on RTH timestamps."""
    rng = np.random.default_rng(seed)
    minutes_per_year = 252 * bars_per_session
    drift = annual_drift / minutes_per_year
    base_sigma = annual_vol / np.sqrt(minutes_per_year)

    # Business days only; the index must look like a real trading calendar so
    # that T+1 settlement across a weekend is genuinely exercised.
    days = pd.bdate_range(start=start_date, periods=n_sessions, tz=EXCHANGE_TZ)

    # U-shaped intraday volatility: open and close are far livelier than midday.
    x = np.linspace(0, 1, bars_per_session)
    shape = 1.0 + 1.8 * np.exp(-x / 0.08) + 1.1 * np.exp(-(1 - x) / 0.10)

    price = start_price
    frames = []
    vol_state = 1.0
    for day in days:
        # Volatility clusters across sessions rather than resetting daily.
        vol_state = 0.85 * vol_state + 0.15 * rng.lognormal(0.0, 0.35)
        sigma = base_sigma * shape * np.clip(vol_state, 0.4, 3.0)

        # Overnight gap, applied before the first bar of the session.
        price *= float(np.exp(rng.normal(0, base_sigma * 9)))

        rets = rng.normal(drift, sigma)
        closes = price * np.exp(np.cumsum(rets))
        opens = np.concatenate([[price], closes[:-1]])

        # Wicks scale with that bar's own volatility.
        wick = np.abs(rng.normal(0, sigma * 0.6, bars_per_session)) * closes
        highs = np.maximum(opens, closes) + wick
        lows = np.minimum(opens, closes) - np.abs(rng.normal(0, sigma * 0.6, bars_per_session)) * closes
        lows = np.minimum(lows, np.minimum(opens, closes))

        volume = (shape * rng.lognormal(11.5, 0.4, bars_per_session)).round()

        idx = pd.date_range(
            start=day.normalize() + pd.Timedelta(hours=9, minutes=30),
            periods=bars_per_session,
            freq="1min",
        )
        frames.append(
            pd.DataFrame(
                {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volume},
                index=idx,
            )
        )
        price = float(closes[-1])

    out = pd.concat(frames)
    out.index.name = "timestamp"
    return out
