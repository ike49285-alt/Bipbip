"""Ingest for bars pulled through the Webull MCP tools.

Webull reaches roughly fifteen years back at minute granularity, where Yahoo
caps out at thirty days. The catch is that the two feeds are adjusted
differently: Webull's *daily* bars are split- and dividend-adjusted, but its
*intraday* bars are raw. TQQQ on 2011-09-09 closes at 61.79 on the minute feed
and 0.307424 on the daily one, a factor of 201.

Splicing raw intraday history onto an adjusted archive would plant a fake 50%
gap at every split. Backwards, too: a 2:1 split makes the raw series halve
overnight, so a naive reader sees a crash that never happened and any model
learns to buy it. `rescale_to_adjusted` removes that by re-expressing the raw
bars in the same units as the adjusted daily series we already store.
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd

COLUMNS = ["open", "high", "low", "close", "volume"]


def parse_bars(text: str, tz: str = "America/New_York") -> pd.DataFrame:
    """Read `iso,open,high,low,close,volume` rows into a UTC-naive-free frame.

    Webull stamps bars in UTC. The rest of the project indexes intraday bars in
    exchange-local time, so convert rather than store two conventions.
    """
    rows = [ln for ln in (l.strip() for l in text.splitlines()) if ln and not ln.startswith("#")]
    if not rows:
        return pd.DataFrame(columns=COLUMNS, index=pd.DatetimeIndex([], tz=tz))
    df = pd.read_csv(io.StringIO("\n".join(rows)), header=None,
                     names=["time"] + COLUMNS)
    idx = pd.to_datetime(df["time"], utc=True, format="mixed").dt.tz_convert(tz)
    out = df[COLUMNS].astype("float64")
    out.index = pd.DatetimeIndex(idx)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out.index.name = "time"
    return out


def adjustment_factors(raw: pd.DataFrame, adjusted_daily: pd.Series) -> pd.Series:
    """Per-session multiplier carrying raw intraday prices onto adjusted ones.

    Uses each session's last raw bar against that session's adjusted close. The
    result should be piecewise constant, stepping only on splits and ex-dividend
    dates; `verify_factors` is what checks that, because a factor that wanders
    daily means the two feeds are not the same instrument.
    """
    last = raw["close"].groupby(raw.index.normalize()).last()
    last.index = pd.DatetimeIndex([pd.Timestamp(d.date()) for d in last.index])
    adj = adjusted_daily.copy()
    adj.index = pd.DatetimeIndex([pd.Timestamp(d.date()) for d in adj.index])
    common = last.index.intersection(adj.index)
    if len(common) == 0:
        raise ValueError("no overlapping sessions between raw and adjusted series")
    return (adj.loc[common] / last.loc[common]).sort_index()


def robust_factors(raw: pd.Series | pd.DataFrame, adjusted_daily: pd.Series,
                   window: int = 11) -> pd.Series:
    """Adjustment factors that a handful of corrupt sessions cannot move.

    The naive factor is `adjusted_close / raw_close` for each session, but that
    is circular here: the raw close is exactly the value the after-hours leak
    corrupts, so a bad session sets its own factor and rescales its twelve good
    bars to hide the damage.

    The true factor is a step function - constant between corporate actions - so
    a median over neighbouring sessions recovers it and discards the outliers.
    Splits are found first and the median is taken strictly within each
    split-free segment, because a window straddling a 2:1 boundary would average
    two plateaus into a value correct on neither side.
    """
    raw_close = raw["close"] if isinstance(raw, pd.DataFrame) else raw
    last = raw_close.groupby(raw_close.index.normalize()).last()
    last.index = pd.DatetimeIndex([pd.Timestamp(d.date()) for d in last.index])
    adj = adjusted_daily.copy()
    adj.index = pd.DatetimeIndex([pd.Timestamp(d.date()) for d in adj.index])
    common = last.index.intersection(adj.index)
    if len(common) == 0:
        raise ValueError("no overlapping sessions between raw and adjusted series")
    ratio = (adj.loc[common] / last.loc[common]).sort_index()

    step = np.log(ratio / ratio.shift(1)).abs()
    segment = (step > 0.10).cumsum()          # splits only; dividends are tiny
    return ratio.groupby(segment).transform(
        lambda s: s.rolling(window, center=True, min_periods=1).median())


def verify_factors(factors: pd.Series, tol: float = 0.02) -> pd.DataFrame:
    """Report every day the factor moved by more than `tol`.

    Genuine splits and dividends appear here and are expected. What must NOT
    appear is a long tail of small daily wobbles: that would mean the raw and
    adjusted series disagree about prices themselves, and rescaling would be
    laundering a mismatch rather than removing an artefact.
    """
    step = factors / factors.shift(1) - 1.0
    hits = step[step.abs() > tol]
    return pd.DataFrame({"factor": factors.reindex(hits.index), "step": hits})


def rescale_to_adjusted(raw: pd.DataFrame, adjusted_daily: pd.Series) -> pd.DataFrame:
    """Express raw intraday bars in the adjusted series' units.

    Sessions with no adjusted close to anchor against are dropped rather than
    carried at the previous factor: an unanchored stretch spanning a split is
    precisely the seam this function exists to prevent.
    """
    factors = adjustment_factors(raw, adjusted_daily)
    day = pd.DatetimeIndex([pd.Timestamp(d.date()) for d in raw.index])
    f = factors.reindex(day).to_numpy()
    keep = np.isfinite(f)
    out = raw.loc[keep].copy()
    f = f[keep]
    for c in ("open", "high", "low", "close"):
        out[c] = out[c].to_numpy() * f
    out["volume"] = out["volume"].to_numpy() / f
    return out


def reconcile_sessions(raw: pd.DataFrame, daily: pd.DataFrame,
                       factors: pd.Series, tol: float = 0.005,
                       check_close: bool = True) -> pd.DataFrame:
    """Check each session's intraday bars against the independent daily record.

    Webull tags the final thirty-minute bar of a session as RTH but sometimes
    lets it run past the closing bell. On a quiet day the after-hours range sits
    inside the regular one and nothing shows. On a violent day it does not: the
    2020-02-27 15:30 bar closes at 68.74 when the session actually closed at
    75.82, because 68.74 is an after-hours print - and it is where the stock
    opens the following morning.

    That is lookahead, concentrated in exactly the sessions whose returns
    dominate a backtest. This function locates it by rescaling each session onto
    the daily series and comparing closes, highs and lows against a source that
    knows nothing about the intraday feed.
    """
    day = pd.DatetimeIndex([pd.Timestamp(d.date()) for d in raw.index])
    grp = raw.groupby(day)
    intraday = pd.DataFrame({
        "open": grp["open"].first(), "high": grp["high"].max(),
        "low": grp["low"].min(), "close": grp["close"].last(),
    })
    d = daily.copy()
    d.index = pd.DatetimeIndex([pd.Timestamp(x.date()) for x in d.index])
    common = intraday.index.intersection(d.index).intersection(factors.index)
    intraday, d = intraday.loc[common], d.loc[common]
    f = factors.loc[common]

    scaled = intraday.mul(f, axis=0)
    out = pd.DataFrame(index=common)
    out["close_err"] = (scaled["close"] / d["close"] - 1.0).abs()
    # A session's intraday extremes must sit inside the daily bar's. Exceeding
    # them means the feed saw a price the regular session never traded at.
    out["above_high"] = (scaled["high"] / d["high"] - 1.0).clip(lower=0)
    out["below_low"] = (1.0 - scaled["low"] / d["low"]).clip(lower=0)
    # Range containment is the test that survives editing. Once the bar holding
    # the closing print has been removed, the session's last price is no longer
    # meant to equal the daily close, so demanding it would condemn every
    # session the repair just fixed. What must always hold is that no remaining
    # bar trades outside the range the regular session actually printed.
    escaped = (out["above_high"] > tol) | (out["below_low"] > tol)
    out["bad"] = (escaped | (out["close_err"] > tol)) if check_close else escaped
    return out


def drop_leaked_closes(raw: pd.DataFrame, report: pd.DataFrame) -> pd.DataFrame:
    """Remove the final bar of every session the daily record contradicts.

    Only the last bar is dropped, not the session: the leak is confined to the
    bar that straddles the bell, and discarding twelve good bars to remove one
    bad one would cost the volatile sessions the search most needs. Sessions are
    re-checked after the drop, so one that still disagrees is removed entirely.
    """
    day = pd.DatetimeIndex([pd.Timestamp(d.date()) for d in raw.index])
    bad_days = set(report.index[report["bad"]])
    if not bad_days:
        return raw
    is_last = pd.Series(day, index=raw.index).groupby(day).transform(
        lambda s: pd.Series(s.index == s.index.max(), index=s.index))
    drop = pd.Series(day, index=raw.index).isin(bad_days) & is_last
    return raw.loc[~drop.to_numpy()]
