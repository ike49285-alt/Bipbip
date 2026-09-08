"""Regular-trading-hours session handling.

Everything downstream assumes bars are tz-aware in the exchange timezone and
restricted to RTH. Pre/post-market bars are thin, wide-spread, and would make
any backtest look better than reality, so they are dropped at the boundary.
"""
from __future__ import annotations

import datetime as dt

import pandas as pd

EXCHANGE_TZ = "America/New_York"
RTH_OPEN = dt.time(9, 30)
RTH_CLOSE = dt.time(16, 0)


def to_exchange_tz(df: pd.DataFrame, tz: str = EXCHANGE_TZ) -> pd.DataFrame:
    """Return `df` with a tz-aware DatetimeIndex in the exchange timezone."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(f"expected a DatetimeIndex, got {type(df.index).__name__}")
    idx = df.index
    idx = idx.tz_localize("UTC") if idx.tz is None else idx
    out = df.copy()
    out.index = idx.tz_convert(tz)
    return out


def restrict_to_rth(df: pd.DataFrame) -> pd.DataFrame:
    """Drop bars outside 09:30-16:00 exchange time.

    The closing bar is stamped 15:59 for one-minute data, so the window is
    half-open on the right: a bar stamped exactly 16:00 belongs to the auction
    and is excluded.
    """
    if df.empty:
        return df
    t = df.index.time
    return df[(t >= RTH_OPEN) & (t < RTH_CLOSE)]


def session_dates(df: pd.DataFrame) -> pd.Index:
    """Unique trading dates present in `df`, ascending."""
    return pd.Index(sorted({ts.date() for ts in df.index}))


def iter_sessions(df: pd.DataFrame):
    """Yield ``(date, session_bars)`` for each trading day, in order.

    Grouping on the calendar date is what makes "flat by the close" and the
    cash account's one-round-trip-per-session rule expressible.
    """
    if df.empty:
        return
    for day, bars in df.groupby(df.index.date, sort=True):
        yield day, bars


def next_session_date(dates: pd.Index, after) -> dt.date | None:
    """First trading date strictly after `after`, or None if there is none.

    Used to settle cash T+1 against the real trading calendar rather than
    calendar arithmetic, which would wrongly settle a Friday sale on Saturday.
    """
    for d in dates:
        if d > after:
            return d
    return None
