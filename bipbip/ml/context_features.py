"""Features built from non-price data: earnings timing, sector, and macro.

These exist because everything else in the model is a transformation of OHLCV,
and the meta-labelling folds showed that family of signal decaying to nothing
after the early 2000s. Price-derived patterns in liquid US large caps are
picked over; a real-world event the price series does not encode is a different
kind of input.

EARNINGS TIMING is the most promising of the three. A large share of an
individual stock's variance arrives on report day, and a setup forming three
days beforehand is a bet on an announcement rather than on a pattern. The
features are signed - days until the next report, days since the last - so the
model can separate the two rather than seeing "near earnings" as one state.

CAUSALITY NEEDS CARE HERE. Scheduled earnings dates are known in advance, so
"days until the next report" is legitimately knowable today. But a date fetched
TODAY for a report that happened years ago is only knowable in hindsight if the
schedule was revised, so the features are built to use the next SCHEDULED date
as of each bar, and nothing about the outcome.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CONTEXT_COLUMNS = [
    "days_to_earnings",
    "days_since_earnings",
    "in_earnings_window",
    "sector_rel_strength",
    "sector_breadth",
]


def _as_day_numbers(index: pd.DatetimeIndex) -> np.ndarray:
    """Whole days since the epoch, independent of the index's time resolution."""
    return np.asarray(index.to_numpy(), dtype="datetime64[D]").astype("int64")


def earnings_features(dates: pd.DatetimeIndex, symbols, earnings: pd.DataFrame) -> dict:
    """Days to the next report and days since the last, per symbol.

    Capped at 60 days: beyond that the distance carries no information and a
    raw count would dominate a standardised model.
    """
    to_next = pd.DataFrame(np.nan, index=dates, columns=list(symbols))
    since = pd.DataFrame(np.nan, index=dates, columns=list(symbols))

    if earnings is None or earnings.empty:
        return {"days_to_earnings": to_next, "days_since_earnings": since,
                "in_earnings_window": to_next.copy()}

    # Convert to whole days via datetime64[D] rather than dividing raw integers
    # by a hardcoded nanoseconds-per-day constant. pandas 3 defaults to
    # MICROSECOND resolution, so that constant silently collapsed every date to
    # the same integer and turned all of these features into constants.
    day_ints = _as_day_numbers(dates)
    for sym, grp in earnings.groupby("symbol"):
        if sym not in to_next.columns:
            continue
        ed = pd.DatetimeIndex(sorted(pd.to_datetime(grp["earnings_date"]).dt.normalize().unique()))
        if len(ed) == 0:
            continue
        e_ints = _as_day_numbers(ed)

        nxt = np.searchsorted(e_ints, day_ints, side="left")
        prv = nxt - 1
        d_next = np.where(nxt < len(e_ints), e_ints[np.clip(nxt, 0, len(e_ints) - 1)] - day_ints, np.nan)
        d_prev = np.where(prv >= 0, day_ints - e_ints[np.clip(prv, 0, len(e_ints) - 1)], np.nan)

        to_next[sym] = np.clip(d_next, 0, 60)
        since[sym] = np.clip(d_prev, 0, 60)

    # A single flag for "inside the window where the report dominates".
    window = ((to_next <= 5) | (since <= 2)).astype("float64")
    window[to_next.isna() & since.isna()] = np.nan
    return {"days_to_earnings": to_next, "days_since_earnings": since,
            "in_earnings_window": window}


def sector_features(closes: pd.DataFrame, sectors: dict, window: int = 60) -> dict:
    """How a stock is doing against its OWN sector, and how broad that move is.

    Ranking a name against every other name mixes unrelated businesses.
    Relative strength within a sector isolates what is specific to the company
    from what is happening to its whole industry, and sector breadth says
    whether a move is one stock or the entire group.
    """
    mom = np.log(closes / closes.shift(window))
    rel = pd.DataFrame(np.nan, index=closes.index, columns=closes.columns)
    breadth = pd.DataFrame(np.nan, index=closes.index, columns=closes.columns)

    groups: dict = {}
    for sym in closes.columns:
        sec = (sectors.get(sym) or {}).get("sector") if sectors else None
        if sec:
            groups.setdefault(sec, []).append(sym)

    for sec, members in groups.items():
        if len(members) < 3:
            continue
        sub = mom[members]
        sector_mean = sub.mean(axis=1)
        for sym in members:
            rel[sym] = sub[sym] - sector_mean
        # Fraction of the sector that is advancing over the window.
        frac = (sub > 0).sum(axis=1) / sub.notna().sum(axis=1).replace(0, np.nan)
        for sym in members:
            breadth[sym] = frac

    return {"sector_rel_strength": rel, "sector_breadth": breadth}


def build_context(closes: pd.DataFrame, symbols, earnings: pd.DataFrame | None,
                  sectors: dict | None, macro: pd.DataFrame | None = None) -> dict:
    """Assemble every non-price feature, broadcasting macro across symbols."""
    feats: dict = {}
    feats.update(earnings_features(closes.index, symbols, earnings))
    feats.update(sector_features(closes, sectors or {}))

    if macro is not None and not macro.empty:
        aligned = macro.reindex(closes.index).ffill()
        for col in aligned.columns:
            feats[col] = pd.DataFrame(
                np.repeat(aligned[col].to_numpy(dtype="float64")[:, None],
                          len(closes.columns), axis=1),
                index=closes.index, columns=closes.columns)
    return feats
