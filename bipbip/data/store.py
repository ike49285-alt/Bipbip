"""Append-only local bar archive.

Free intraday data providers only serve a short trailing window (yfinance gives
roughly 30 days of one-minute bars). A strategy validated on 21 sessions is not
validated at all. The fix is to never throw a fetch away: every download is
merged into a local parquet archive, so running `fetch` on a schedule
accumulates history far past any single provider window.

Merges are last-writer-wins on timestamp, which lets a later fetch correct a
provisional bar without duplicating it.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .sessions import restrict_to_rth, to_exchange_tz

OHLCV = ["open", "high", "low", "close", "volume"]


class BarStore:
    """Parquet-backed archive of OHLCV bars, keyed by symbol and bar size."""

    def __init__(self, root: str | Path = "data/bars"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, symbol: str, bar_size: str = "1m") -> Path:
        return self.root / f"{symbol.upper()}_{bar_size}.parquet"

    def load(self, symbol: str, bar_size: str = "1m") -> pd.DataFrame:
        """Return the archived bars, or an empty frame if nothing is stored."""
        path = self.path_for(symbol, bar_size)
        if not path.exists():
            return pd.DataFrame(columns=OHLCV, index=pd.DatetimeIndex([], name="timestamp"))
        return pd.read_parquet(path)

    def append(self, symbol: str, bars: pd.DataFrame, bar_size: str = "1m") -> dict:
        """Merge `bars` into the archive. Returns a summary of what changed."""
        incoming = self._normalise(bars)
        existing = self.load(symbol, bar_size)

        before = len(existing)
        if existing.empty:
            merged = incoming
        else:
            # Incoming last so that duplicated(keep="last") prefers the fresh copy.
            merged = pd.concat([existing, incoming])
            merged = merged[~merged.index.duplicated(keep="last")]
        merged = merged.sort_index()

        self.path_for(symbol, bar_size).parent.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(self.path_for(symbol, bar_size))

        return {
            "symbol": symbol.upper(),
            "rows_before": before,
            "rows_after": len(merged),
            "rows_added": len(merged) - before,
            "start": merged.index.min() if len(merged) else None,
            "end": merged.index.max() if len(merged) else None,
        }

    @staticmethod
    def _normalise(bars: pd.DataFrame) -> pd.DataFrame:
        """Coerce a provider frame into the archive's canonical shape."""
        if bars.empty:
            return pd.DataFrame(columns=OHLCV, index=pd.DatetimeIndex([], name="timestamp"))

        df = bars.copy()
        # Providers vary: flatten a MultiIndex from a multi-symbol download.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]

        missing = [c for c in OHLCV if c not in df.columns]
        if missing:
            raise ValueError(f"bars are missing required columns: {missing}")

        df = df[OHLCV].astype("float64")
        df = to_exchange_tz(df)
        df = restrict_to_rth(df)
        df = df[~df.index.duplicated(keep="last")].sort_index()
        df.index.name = "timestamp"
        # A zero-volume bar is a provider gap-fill, not a tradeable minute.
        return df[df["volume"] > 0]

    def coverage(self, symbol: str, bar_size: str = "1m") -> dict:
        """Describe how much history is archived - the honest sample size."""
        df = self.load(symbol, bar_size)
        if df.empty:
            return {"symbol": symbol.upper(), "bars": 0, "sessions": 0}
        dates = {ts.date() for ts in df.index}
        return {
            "symbol": symbol.upper(),
            "bars": len(df),
            "sessions": len(dates),
            "start": str(df.index.min()),
            "end": str(df.index.max()),
        }
