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
YF_MAX_DAYS = {"1m": 29, "2m": 59, "5m": 59, "15m": 59, "30m": 59, "60m": 729, "1h": 729}

#: Daily and coarser intervals have no trailing-window limit; Yahoo serves the
#: instrument's full history. SPY reaches back to 1993 and TQQQ to 2010, which
#: is decades of regime context the minute archive can never contain.
UNLIMITED_INTERVALS = frozenset({"1d", "5d", "1wk", "1mo", "3mo"})

#: Intervals Yahoo will serve in ONE request spanning their whole window, given
#: a `period` rather than explicit dates. Walking 729 days of hourly bars in
#: 30-day chunks costs 25 requests per symbol - 7,625 across a 305-symbol
#: universe, which is enough to get throttled and slow enough to time the job
#: out. The same history arrives in a single request per symbol.
PERIOD_FETCHABLE = {"1h": "730d", "60m": "730d"}


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

        if bar_size in UNLIMITED_INTERVALS:
            # auto_adjust=True gives a TOTAL-RETURN series: OHLC back-adjusted
            # for splits and dividends. Without it a bond ETF looks like it
            # returned nothing for twenty years, because its entire return
            # arrives as coupons. SHY read 81.01 -> 81.67 across 24 years on
            # unadjusted data, which made "does the risk asset beat cash?"
            # compare against a cash proxy earning zero.
            #
            # Every daily fetch pulls period="max", so the whole series is
            # re-adjusted on each run and the archive stays internally
            # consistent as new dividends are paid.
            df = yf.download(symbol, period="max", interval=bar_size,
                             progress=False, auto_adjust=True, threads=False)
            if df is None or df.empty:
                raise FetchError(f"no {bar_size} bars returned for {symbol}")
            return df

        # One request for the whole window where Yahoo allows it. Falls through
        # to the chunked walk below if it comes back empty, so a change at the
        # provider degrades to the slow path rather than to no data.
        if bar_size in PERIOD_FETCHABLE and lookback_days is None:
            df = yf.download(symbol, period=PERIOD_FETCHABLE[bar_size],
                             interval=bar_size, progress=False,
                             auto_adjust=False, prepost=False, threads=False)
            if df is not None and not df.empty:
                return df

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
                # Deliberately unadjusted, unlike the daily path: the intraday
                # archive is stitched from many partial fetches that are never
                # rewritten, so back-adjusting new bars would leave a seam at
                # every dividend. Over a trailing window the distortion is
                # under 0.5%, and it is the price you would actually trade at.
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
