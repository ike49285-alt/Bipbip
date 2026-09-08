"""Regular-trading-hours session handling.

Everything downstream assumes bars are tz-aware in the exchange timezone and
restricted to RTH. Pre/post-market bars are thin, wide-spread, and would make
any backtest look better than reality, so they are dropped at the boundary.
"""
from __future__ import annotations

import datetime as dt

import pandas as pd

EXCHANGE_TZ = "America/New_York"

#: Intervals that sit INSIDE a session, and to which the RTH window applies.
#: A daily or weekly bar is stamped at midnight, so filtering it to 09:30-16:00
#: silently discards every row - which is what happened the first time daily
#: bars were added.
INTRADAY_INTERVALS = frozenset({"1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"})


def is_intraday(bar_size: str) -> bool:
    return str(bar_size).lower() in INTRADAY_INTERVALS


RTH_OPEN = dt.time(9, 30)
RTH_CLOSE = dt.time(16, 0)


def to_session_dates(df: pd.DataFrame, tz: str = EXCHANGE_TZ) -> pd.DataFrame:
    """Stamp daily-or-coarser bars at midnight on their own session date.

    Providers return daily bars with a NAIVE timestamp that already IS the
    session date. Localising that to UTC and converting to exchange time lands
    it at 19:00 or 20:00 on the PREVIOUS calendar day, so every bar ends up
    labelled with the wrong date - the bar stamped 2026-09-01 actually held the
    2026-09-02 session.

    That is not a cosmetic problem. Joining such an index to intraday bars by
    date silently pairs each session with TOMORROW's daily bar, which is
    lookahead bias of the worst kind: invisible, and flattering.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(f"expected a DatetimeIndex, got {type(df.index).__name__}")

    idx = df.index
    # A naive stamp is already the session date. A tz-aware one is normalised in
    # UTC first, since that is the convention providers use for daily data.
    dates = idx.date if idx.tz is None else idx.tz_convert("UTC").date

    out = df.copy()
    out.index = pd.DatetimeIndex(
        [pd.Timestamp(d).tz_localize(tz) for d in dates], name=df.index.name
    )
    return out


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
