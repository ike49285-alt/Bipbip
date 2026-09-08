"""Daily regime context for intraday bars.

Daily history supplies what a 30-day minute archive structurally cannot: where
the current market sits in decades of its own behaviour. That is genuinely new
information - an intraday bar cannot know whether today is calm or wild by the
standards of thirty years.

THE LAG IS THE WHOLE PROBLEM. A session on date D may only see the daily bar
for D-1, because D's own daily bar contains D's close - the very thing a
strategy is trying to predict. Every feature here is therefore shifted by one
session before being mapped onto intraday bars, and the tests attack that
directly. This project has already been bitten once by daily bars carrying the
wrong date; the same mistake here would put tomorrow's close into today's
features and flatter every result invisibly.

The one exception is the overnight gap, which compares today's OPEN to
yesterday's close. That is known the moment the session starts and is legitimate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REGIME_COLUMNS = [
    "regime_vol_pct",
    "regime_vol_ratio",
    "daily_trend_20",
    "daily_atr_pct",
    "overnight_gap",
]


def daily_features(daily: pd.DataFrame, vol_window: int = 21) -> pd.DataFrame:
    """Per-session context, computed from completed daily bars.

    Values are stated AS OF each session's close; `attach_to_intraday` performs
    the shift that makes them usable.
    """
    close = daily["close"]
    rets = np.log(close / close.shift(1))

    out = pd.DataFrame(index=daily.index)
    rv = rets.rolling(vol_window).std() * np.sqrt(252)
    out["realised_vol"] = rv
    # Expanding rank, so the percentile at any point uses only prior history and
    # never the full sample - a full-sample rank would leak the future.
    out["regime_vol_pct"] = rv.expanding(min_periods=252).apply(
        lambda w: (w[:-1] < w[-1]).mean(), raw=True
    )
    out["regime_vol_ratio"] = rv / rv.expanding(min_periods=252).median()
    out["daily_trend_20"] = np.log(close / close.shift(20))

    prev_close = close.shift(1)
    tr = pd.concat([
        daily["high"] - daily["low"],
        (daily["high"] - prev_close).abs(),
        (daily["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    out["daily_atr_pct"] = tr.ewm(alpha=1 / 14, adjust=False).mean() / close
    out["daily_close"] = close
    return out


def attach_to_intraday(
    intraday_index: pd.DatetimeIndex,
    session_opens: pd.Series,
    daily: pd.DataFrame,
    vol_window: int = 21,
) -> pd.DataFrame:
    """Map lagged daily context onto every intraday bar.

    `session_opens` is each session's opening price, keyed by date, and is used
    only for the overnight gap.
    """
    feats = daily_features(daily, vol_window)
    # THE LAG. Session D sees the daily bar completed on D-1.
    lagged = feats.shift(1)
    lagged.index = pd.Index([ts.date() for ts in lagged.index], name="session")
    lagged = lagged[~lagged.index.duplicated(keep="last")]

    dates = pd.Index([ts.date() for ts in intraday_index], name="session")
    mapped = lagged.reindex(dates)

    # Overnight gap: today's open against yesterday's close. Known at the bell.
    opens = session_opens.reindex(pd.Index(sorted(set(dates))))
    gap = (opens / lagged["daily_close"].reindex(opens.index) - 1.0)
    mapped["overnight_gap"] = gap.reindex(dates).to_numpy()

    out = mapped[REGIME_COLUMNS].copy()
    out.index = intraday_index
    return out


def session_open_prices(bars: pd.DataFrame) -> pd.Series:
    """First open of each session, keyed by date."""
    key = pd.Index([ts.date() for ts in bars.index], name="session")
    return bars["open"].groupby(key).first()
