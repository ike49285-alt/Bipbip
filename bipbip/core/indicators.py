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


def session_vwap_bands(df: pd.DataFrame, min_bars: int = 15) -> pd.DataFrame:
    """Session VWAP with its volume-weighted standard deviation.

    This exists because normalising the distance from VWAP by a one-minute ATR
    is dimensionally wrong. Displacement from session VWAP accumulates over the
    whole session and grows roughly with the square root of elapsed time, while
    ATR is a per-minute quantity. Dividing one by the other produced a median
    "stretch" of 3.7 ATR on real TQQQ data - so a 1.5-ATR threshold fired
    almost constantly, and readings of 11 ATR were ordinary rather than
    extreme.

    Sigma here is the dispersion of price around VWAP measured on the same
    clock as VWAP itself, so ``(price - vwap) / sigma`` is a genuine z-score
    that means the same thing at 09:45 and at 15:30.

    Computed from cumulative volume-weighted moments, so it stays causal.
    Sigma is NaN until `min_bars` have accumulated, because a standard
    deviation over three bars is noise pretending to be a statistic.
    """
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    key = _session_key(df.index)
    vol = df["volume"]

    cum_v = vol.groupby(key).cumsum()
    cum_pv = (typical * vol).groupby(key).cumsum()
    cum_p2v = (typical * typical * vol).groupby(key).cumsum()

    safe_v = cum_v.replace(0, np.nan)
    vwap = cum_pv / safe_v
    # Var(X) = E[X^2] - E[X]^2, volume-weighted. Clipped at zero: floating point
    # can make this marginally negative when price barely moves.
    var = (cum_p2v / safe_v) - vwap**2
    sigma = np.sqrt(var.clip(lower=0.0))

    n = pd.Series(1, index=df.index).groupby(key).cumsum()
    sigma = sigma.where(n >= min_bars)

    out = pd.DataFrame(index=df.index)
    out["vwap"] = vwap
    out["vwap_sigma"] = sigma
    return out


def zscore_from_bands(close: pd.Series, bands: pd.DataFrame,
                      min_sigma_bps: float = 1.0) -> pd.Series:
    """Z-score from bands already computed, so callers holding them do not
    recompute - and cannot drift from the flooring rule below.

    `min_sigma_bps` floors sigma so a near-motionless stretch of tape cannot
    blow the ratio up; a dispersion below a basis point of price is not a
    tradeable dislocation whatever the arithmetic says.
    """
    sigma = bands["vwap_sigma"].clip(lower=close * (min_sigma_bps / 10_000.0))
    return ((close - bands["vwap"]) / sigma).rename("vwap_z")


def vwap_zscore(df: pd.DataFrame, min_bars: int = 15, min_sigma_bps: float = 1.0) -> pd.Series:
    """Signed z-score of price relative to session VWAP."""
    return zscore_from_bands(df["close"], session_vwap_bands(df, min_bars=min_bars),
                             min_sigma_bps=min_sigma_bps)


def cost_floored_risk(close: pd.Series, atr_series: pd.Series, hurdle_bps: float,
                      min_multiple: float = 2.0) -> pd.Series:
    """Risk unit for stops and targets, floored at a multiple of trading cost.

    A stop closer than the round trip is a guaranteed loss when hit. On real
    SPY data 37% of bars had a one-minute ATR smaller than the 2.3bp round-trip
    cost, so an unfloored ATR stop was inside the noise a third of the time -
    which is what produced trades that entered and stopped out on the same bar.
    """
    floor = close * (hurdle_bps * min_multiple / 10_000.0)
    return atr_series.clip(lower=floor)


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
