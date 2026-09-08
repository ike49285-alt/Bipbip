"""Intraday indicators.

Every function here is causal: the value at bar `i` uses only bars `<= i`.
That property is what keeps lookahead bias out of the strategies, so any new
indicator added to this module must preserve it. Anything using `.shift(-n)`,
a centred window, or a whole-session aggregate computed up front does not
belong here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _session_key(index: pd.DatetimeIndex) -> np.ndarray:
    return np.asarray([ts.date() for ts in index])


def session_vwap(df: pd.DataFrame) -> pd.Series:
    """Volume-weighted average price, re-anchored at each session open."""
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    key = _session_key(df.index)
    pv = (typical * df["volume"]).groupby(key).cumsum()
    vol = df["volume"].groupby(key).cumsum()
    return (pv / vol.replace(0, np.nan)).rename("vwap")


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    ranges = pd.concat(
        [df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()],
        axis=1,
    )
    return ranges.max(axis=1)


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Average true range - the unit position sizing and stops are quoted in."""
    return true_range(df).ewm(alpha=1.0 / window, adjust=False).mean().rename("atr")


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100.0 - 100.0 / (1.0 + rs)).fillna(50.0).rename("rsi")


def opening_range(df: pd.DataFrame, minutes: int = 30) -> pd.DataFrame:
    """Running high/low of the first `minutes` of each session.

    Before the range completes, the values are the running extremes so far and
    must not be traded on; `or_complete` marks when the window has closed.
    """
    key = _session_key(df.index)
    elapsed = df.groupby(key).cumcount()
    in_window = elapsed < minutes

    high = df["high"].where(in_window)
    low = df["low"].where(in_window)
    out = pd.DataFrame(index=df.index)
    out["or_high"] = high.groupby(key).cummax().groupby(key).ffill()
    out["or_low"] = low.groupby(key).cummin().groupby(key).ffill()
    out["or_complete"] = elapsed >= minutes
    return out


def relative_volume(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Volume against its own trailing average - a proxy for participation."""
    avg = df["volume"].rolling(window, min_periods=window).mean()
    return (df["volume"] / avg.replace(0, np.nan)).rename("rvol")


def realised_vol(series: pd.Series, window: int = 30, bars_per_year: int = 252 * 390) -> pd.Series:
    """Annualised realised volatility from log returns."""
    r = np.log(series / series.shift(1))
    return (r.rolling(window, min_periods=window).std() * np.sqrt(bars_per_year)).rename("rvol_ann")


def minutes_since_open(index: pd.DatetimeIndex) -> pd.Series:
    """Bars elapsed in the session - the natural intraday clock."""
    key = _session_key(index)
    return pd.Series(index, index=index).groupby(key).cumcount().rename("bar_of_day")
