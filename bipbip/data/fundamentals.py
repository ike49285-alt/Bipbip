"""Non-price data: earnings dates, sector membership, and news.

Everything the model has seen so far is derived from open, high, low, close and
volume - the same numbers on every retail charting platform. Any edge in a
combination of those has been competed away, which is exactly the decay pattern
the meta-labelling folds showed. Information the price series does not encode is
the only place a durable edge could come from.

Three things are worth reaching for, and they differ sharply in how obtainable
they are:

  EARNINGS DATES are the biggest gap. A large share of an individual stock's
  movement clusters around its report, and a breakout three days before
  earnings is a different bet from one three days after. The model currently
  cannot tell them apart.

  SECTOR is cheap and useful. Ranking Apple against Johnson & Johnson is much
  weaker than ranking it against its own sector.

  NEWS is the one people assume is available and is not, at least historically.
  Providers serve RECENT headlines; training needs years of them, timestamped,
  per ticker. This module fetches what exists and reports honestly on the rest
  rather than quietly backfilling a gap that cannot be filled.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pandas as pd


class FundamentalsStore:
    """Local cache for non-price data, which changes slowly or not at all."""

    def __init__(self, root: str | Path = "data/fundamentals"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, name: str) -> Path:
        return self.root / name

    def save_json(self, name: str, obj) -> None:
        self.path(name).write_text(json.dumps(obj, indent=2, sort_keys=True, default=str))

    def load_json(self, name: str, default=None):
        p = self.path(name)
        if not p.exists():
            return default
        return json.loads(p.read_text())

    def save_frame(self, name: str, df: pd.DataFrame) -> None:
        df.to_parquet(self.path(name))

    def load_frame(self, name: str) -> pd.DataFrame:
        p = self.path(name)
        return pd.read_parquet(p) if p.exists() else pd.DataFrame()


def fetch_sectors(symbols) -> dict:
    """Sector and industry per symbol.

    Uses the current classification, which is a mild anachronism when applied
    to decades of history - a company's sector can be reclassified - but the
    effect is small next to the survivorship bias already present.
    """
    import yfinance as yf

    out = {}
    for sym in symbols:
        try:
            info = yf.Ticker(sym).info or {}
            sector = info.get("sector")
            if sector:
                out[sym] = {"sector": sector, "industry": info.get("industry")}
        except Exception:
            continue
    return out


def fetch_earnings_dates(symbols, limit: int = 60) -> pd.DataFrame:
    """Historical and upcoming earnings dates.

    Coverage is the open question: providers typically serve only recent
    quarters, so the caller must check the date range rather than assume
    decades are available.
    """
    import yfinance as yf

    rows = []
    for sym in symbols:
        try:
            df = yf.Ticker(sym).get_earnings_dates(limit=limit)
        except Exception:
            continue
        if df is None or df.empty:
            continue
        for ts in df.index:
            rows.append({"symbol": sym, "earnings_date": pd.Timestamp(ts).tz_localize(None)
                         if pd.Timestamp(ts).tzinfo else pd.Timestamp(ts)})
    return pd.DataFrame(rows)


def probe_news(symbols, sample: int = 5) -> dict:
    """Report what news is actually retrievable, without pretending otherwise.

    Returns counts and the observed date range. If the range covers days rather
    than years, the data can support live signals but cannot support training,
    and that distinction decides whether news is usable at all here.
    """
    import yfinance as yf

    report = {"checked": [], "total_items": 0, "oldest": None, "newest": None}
    for sym in list(symbols)[:sample]:
        try:
            items = yf.Ticker(sym).news or []
        except Exception as exc:
            report["checked"].append({"symbol": sym, "error": str(exc)[:120]})
            continue

        stamps = []
        for it in items:
            content = it.get("content", it)
            for key in ("pubDate", "providerPublishTime", "displayTime"):
                v = content.get(key)
                if v:
                    try:
                        stamps.append(pd.Timestamp(v, unit="s") if isinstance(v, (int, float))
                                      else pd.Timestamp(v).tz_localize(None))
                    except Exception:
                        pass
                    break
        report["checked"].append({"symbol": sym, "items": len(items),
                                  "dated": len(stamps)})
        report["total_items"] += len(items)
        if stamps:
            lo, hi = min(stamps), max(stamps)
            report["oldest"] = str(min(pd.Timestamp(report["oldest"]), lo)) if report["oldest"] else str(lo)
            report["newest"] = str(max(pd.Timestamp(report["newest"]), hi)) if report["newest"] else str(hi)
    return report
