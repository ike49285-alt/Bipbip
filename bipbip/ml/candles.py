"""Raw candles as features: open, high, low, close, volume, and nothing else.

No Heikin-Ashi, no stochastic, no Ichimoku, no moving averages. The model gets
the same thing a trader looking at a bare chart gets - the last N candles - and
has to find its own structure in them rather than being handed someone's.

Two normalisations are unavoidable, and neither is an indicator.

Prices are expressed relative to the current close. TQQQ ranges over a factor of
200 across this archive, so a tree splitting on absolute price learns "in 2012,
when this traded near $5, do X" - a rule about the calendar wearing a rule about
price. Dividing by the latest close removes the level and keeps the shape, which
is exactly what makes two candles at different prices comparable on a chart.

Volume is expressed against its own recent median for the same reason: TQQQ's
daily share volume grew by orders of magnitude, so raw counts encode the year.
The median is taken over the trailing window only, so nothing from the future
enters it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

OHLC = ("open", "high", "low", "close")


def candle_features(bars: pd.DataFrame, lookback: int = 12,
                    volume: bool = True) -> pd.DataFrame:
    """The last `lookback` candles, made comparable across price levels.

    Column `c_3` is the close three bars ago as a fraction of the current close;
    `v_0` is this bar's volume against the window's median. Lag 0 is the bar the
    row describes, so every value is known when that bar closes and nothing
    reaches forward.
    """
    if lookback < 1:
        raise ValueError("lookback must be at least one bar")
    close = bars["close"]
    out = {}
    for lag in range(lookback):
        for col in OHLC:
            # c_0 would be close/close - 1, identically zero for every row; a
            # constant column teaches nothing and only pads the feature count.
            if lag == 0 and col == "close":
                continue
            out[f"{col[0]}_{lag}"] = bars[col].shift(lag) / close - 1.0
    if volume:
        v = bars["volume"]
        # min_periods is the full window: a partial median early in the series
        # is computed from fewer bars than later ones, which quietly makes the
        # first rows a different feature from the rest.
        med = v.rolling(lookback, min_periods=lookback).median()
        for lag in range(lookback):
            out[f"v_{lag}"] = np.log1p(v.shift(lag) / med.replace(0.0, np.nan))
    return pd.DataFrame(out, index=bars.index)


def clock_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Where the bar sits in the session and the week - the chart's x axis.

    Kept separate from the candles so its contribution can be measured rather
    than assumed: a result that only appears with these present is a result
    about the clock, not about price.
    """
    return pd.DataFrame(
        {"bar_of_day": index.hour * 2 + (index.minute >= 30).astype(int),
         "dow": index.dayofweek},
        index=index)
