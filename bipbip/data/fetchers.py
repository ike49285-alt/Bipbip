"""Market data providers.

Only `yfinance` is wired up, because the answer to the data question was
"free only". Its one-minute history is limited to a trailing ~30 days, which
is why `BarStore` archives every fetch. `AlpacaFetcher` is a documented seam
for when that limit starts to hurt.
"""
from __future__ import annotations

import pandas as pd

# yfinance's own hard limits on intraday history, in calendar days.
# One minute is capped at 29 rather than 30 on purpose: Yahoo rejects a
# request whose start is exactly 30 days back ("must be within the last 30
# days"), which silently drops the oldest chunk and costs about a week of
# history on every run.
YF_MAX_DAYS = {"1m": 29, "2m": 59, "5m": 59, "15m": 59, "30m": 59, "60m": 729}


class FetchError(RuntimeError):
    """Raised when a provider returns nothing usable."""


class YFinanceFetcher:
    """Fetch intraday bars from Yahoo Finance via `yfinance`."""

    name = "yfinance"

    def fetch(self, symbol: str, bar_size: str = "1m", lookback_days: int | None = None) -> pd.DataFrame:
        try:
            import yfinance as yf
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise FetchError("yfinance is not installed; run `pip install -r requirements.txt`") from exc

        cap = YF_MAX_DAYS.get(bar_size, 30)
        days = cap if lookback_days is None else min(lookback_days, cap)

        # Yahoo rejects a 1m request spanning more than 8 days, so walk the
        # window in chunks and let the store stitch them together.
        chunk = 7 if bar_size == "1m" else 30
        frames = []
        for start in range(0, days, chunk):
            end = min(start + chunk, days)
            df = yf.download(
                symbol,
                period=None,
                interval=bar_size,
                start=(pd.Timestamp.utcnow() - pd.Timedelta(days=days - start)).date(),
                end=(pd.Timestamp.utcnow() - pd.Timedelta(days=days - end)).date() + pd.Timedelta(days=1),
                progress=False,
                auto_adjust=False,
                prepost=False,
                threads=False,
            )
            if df is not None and not df.empty:
                frames.append(df)

        if not frames:
            raise FetchError(
                f"no {bar_size} bars returned for {symbol}. "
                "Yahoo serves intraday history only for a trailing window, and "
                "returns nothing on weekends before the first session of the week."
            )
        return pd.concat(frames)


class AlpacaFetcher:
    """Placeholder for Alpaca's free tier (years of 1-minute IEX bars).

    Left unimplemented on purpose: it needs an API key, and the current answer
    is free-only. Implementing `fetch` with the same signature is the entire
    migration - nothing downstream knows which provider it is talking to.
    """

    name = "alpaca"

    def fetch(self, symbol: str, bar_size: str = "1m", lookback_days: int | None = None) -> pd.DataFrame:
        raise NotImplementedError(
            "AlpacaFetcher is a stub. Add credentials and implement fetch() to "
            "extend history beyond the ~30 sessions Yahoo will serve."
        )


def get_fetcher(name: str = "yfinance"):
    fetchers = {"yfinance": YFinanceFetcher, "alpaca": AlpacaFetcher}
    if name not in fetchers:
        raise ValueError(f"unknown fetcher {name!r}; choose from {sorted(fetchers)}")
    return fetchers[name]()
