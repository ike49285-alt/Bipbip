"""Aligned multi-symbol price data.

A cross-sectional strategy needs every symbol on one calendar, with missing
data explicit rather than silently forward-filled. Symbols list at different
times - the sector SPDRs in 1998, TQQQ in 2010 - so a naive join either
truncates history to the youngest member or invents prices before inception.

Neither is acceptable, so a symbol is simply NOT TRADEABLE on dates it has no
bar for, and the engine skips it there.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

FIELDS = ["open", "high", "low", "close", "volume"]


@dataclass
class Panel:
    """OHLCV across symbols, aligned on a shared date index."""

    opens: pd.DataFrame
    highs: pd.DataFrame
    lows: pd.DataFrame
    closes: pd.DataFrame
    volumes: pd.DataFrame

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.closes.index

    @property
    def symbols(self) -> list:
        return list(self.closes.columns)

    def tradeable(self, i: int) -> pd.Index:
        """Symbols with a usable bar at position `i`.

        A symbol that has not yet listed, or has a gap, must not be selected -
        otherwise the backtest buys something that did not trade.
        """
        row = self.closes.iloc[i]
        return row[row.notna() & (row > 0)].index

    def slice_from(self, start) -> "Panel":
        """A view of this panel beginning at `start`.

        Comparing two universes only means something when they cover the same
        period. The stock lists reach back to 1962 and the ETF list to 1993, so
        running each over whatever history it happens to have would credit one
        of them with thirty extra years of compounding and read the difference
        as an effect.
        """
        ts = pd.Timestamp(start)
        if self.dates.tz is not None and ts.tz is None:
            ts = ts.tz_localize(self.dates.tz)
        keep = self.dates >= ts
        if not keep.any():
            raise ValueError(f"no bars at or after {start}")
        return Panel(opens=self.opens[keep], highs=self.highs[keep],
                     lows=self.lows[keep], closes=self.closes[keep],
                     volumes=self.volumes[keep])

    def __len__(self) -> int:
        return len(self.closes)

    def summary(self) -> str:
        first = {s: self.closes[s].first_valid_index() for s in self.symbols}
        listed = [d for d in first.values() if d is not None]
        return (f"{len(self.symbols)} symbols, {len(self):,} dates "
                f"({self.dates[0].date()} to {self.dates[-1].date()}); "
                f"earliest listing {min(listed).date()}, latest {max(listed).date()}")


def _is_intraday_index(idx: pd.DatetimeIndex) -> bool:
    """True when the stamps carry a time of day, not just a session date."""
    if len(idx) == 0:
        return False
    t = idx.time
    return any((x.hour or x.minute or x.second) for x in t)


def build_panel(bars_by_symbol: dict) -> Panel:
    """Align per-symbol frames onto the union of their timestamps.

    Values are NOT forward-filled across gaps. A missing bar means the symbol
    was not tradeable at that moment, and `tradeable()` reports it as such;
    filling would let a strategy trade a price that never existed.

    Daily bars are keyed by session DATE, which is what they are. Intraday bars
    keep their full timestamp - and that distinction is load-bearing. This
    function truncated every stamp to its date unconditionally and then dropped
    duplicates keeping the LAST, which for hourly data silently discarded six
    of every seven bars and left only the 15:30 close. A 305-symbol hourly
    panel came back with 730 rows instead of 5,079, and reported itself as
    hourly throughout.
    """
    frames = {f: {} for f in FIELDS}
    for symbol, bars in bars_by_symbol.items():
        if bars is None or bars.empty:
            continue
        if _is_intraday_index(bars.index):
            idx = bars.index
        else:
            idx = pd.DatetimeIndex([pd.Timestamp(ts.date()) for ts in bars.index])
        for f in FIELDS:
            s = pd.Series(bars[f].to_numpy(dtype="float64"), index=idx)
            frames[f][symbol] = s[~s.index.duplicated(keep="last")]

    if not frames["close"]:
        raise ValueError("no usable symbols supplied")

    aligned = {f: pd.DataFrame(frames[f]).sort_index() for f in FIELDS}
    index = aligned["close"].index
    for f in FIELDS:
        aligned[f] = aligned[f].reindex(index)

    return Panel(opens=aligned["open"], highs=aligned["high"], lows=aligned["low"],
                 closes=aligned["close"], volumes=aligned["volume"])


def load_panel(store, symbols, bar_size: str = "1d") -> Panel:
    """Load a universe out of the bar archive."""
    out = {}
    for s in symbols:
        bars = store.load(s, bar_size)
        if not bars.empty:
            out[s] = bars
    return build_panel(out)
