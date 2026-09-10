"""Macro context derived from bond and commodity ETFs already in the archive.

No new data is needed. The shape of the yield curve, the direction of rates and
the behaviour of real assets are all readable from funds this project already
holds daily history for, and they carry information no single stock's price
series contains.

Bond ETF prices move INVERSELY to yields, so a rising TLT means falling
long-term rates. The features below are stated in price terms and named so that
the direction is unambiguous, because getting the sign backwards here would
silently invert every macro signal.

TLT tracks 20+ year Treasuries, IEF 7-10 year, SHY 1-3 year. The ratio of long
to short bond performance is a usable proxy for the slope of the curve: when
long bonds outperform short ones the curve is flattening, which historically
accompanies tightening conditions and a late-cycle economy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MACRO_COLUMNS = [
    "rate_trend_long",
    "rate_trend_short",
    "curve_slope_proxy",
    "curve_slope_change",
    "credit_spread_proxy",
    "gold_trend",
    "dollar_proxy",
]


def _trend(series: pd.Series, window: int = 60) -> pd.Series:
    return np.log(series / series.shift(window))


def build_macro(store, dates: pd.DatetimeIndex, window: int = 60) -> pd.DataFrame:
    """Macro features aligned to `dates`, using only completed daily bars."""
    def load(symbol: str) -> pd.Series:
        bars = store.load(symbol, "1d")
        if bars.empty:
            return pd.Series(dtype="float64")
        s = pd.Series(bars["close"].to_numpy(dtype="float64"),
                      index=pd.DatetimeIndex([pd.Timestamp(t.date()) for t in bars.index]))
        return s[~s.index.duplicated(keep="last")].reindex(dates).ffill()

    tlt, shy = load("TLT"), load("SHY")
    lqd, hyg = load("LQD"), load("HYG")
    gld, uso = load("GLD"), load("USO")

    out = pd.DataFrame(index=dates)

    # Rising bond PRICE means falling yields. Named accordingly.
    out["rate_trend_long"] = -_trend(tlt, window)
    out["rate_trend_short"] = -_trend(shy, window)

    # Long versus short bond performance: a proxy for the curve's slope.
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = np.log(tlt / shy.replace(0, np.nan))
    out["curve_slope_proxy"] = slope
    out["curve_slope_change"] = slope - slope.shift(window)

    # High yield versus investment grade: credit stress, which leads equity
    # trouble more reliably than equity prices lead themselves.
    with np.errstate(divide="ignore", invalid="ignore"):
        out["credit_spread_proxy"] = np.log(hyg / lqd.replace(0, np.nan))

    out["gold_trend"] = _trend(gld, window)
    # Oil against gold stands in for the growth/inflation mix without needing
    # an actual dollar index, which is not available as a free daily series.
    with np.errstate(divide="ignore", invalid="ignore"):
        out["dollar_proxy"] = -np.log(uso / gld.replace(0, np.nan))

    return out[MACRO_COLUMNS].replace([np.inf, -np.inf], np.nan)
