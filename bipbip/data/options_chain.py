"""Option chains from the same free source as everything else.

`yfinance.Ticker.option_chain()` returns strikes, bid, ask, volume, open
interest and IMPLIED VOLATILITY per contract. That last column is the one that
matters: the whole TQQQ question - whether its options trade near 40% while the
underlying realises 67% - is a direct comparison the moment a chain is stored,
and it needs no model of ours to answer.

What this CANNOT do is backtest an options strategy. Yahoo serves the chain as
it stands right now and keeps no history, so chains only accumulate forward
from the first fetch. Historical option data is not free anywhere reachable
here. The distinction matters: "is this contract mispriced today" is answerable
within days, "would this options strategy have worked since 2010" is not
answerable at all on free data.

Greeks are absent from the feed and are not worth faking from a different
model than the one the market used; implied vol plus the pricing module
reproduces them consistently instead.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

#: Columns worth keeping. `impliedVolatility` is the point of the exercise.
KEEP = ["contractSymbol", "strike", "lastPrice", "bid", "ask", "volume",
        "openInterest", "impliedVolatility", "inTheMoney"]


class ChainStore:
    """Parquet archive of option-chain snapshots, keyed by symbol and date.

    Chains are snapshots rather than bars, so unlike the bar archive a fetch
    never corrects an earlier one - each day is its own row set and they are
    kept side by side.
    """

    def __init__(self, root: str | Path = "data/chains"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, symbol: str, asof: dt.date) -> Path:
        return self.root / f"{symbol.upper()}_{asof.isoformat()}.parquet"

    def save(self, symbol: str, frame: pd.DataFrame,
             asof: dt.date | None = None) -> dict:
        asof = asof or dt.date.today()
        path = self.path_for(symbol, asof)
        frame.to_parquet(path)
        return {"symbol": symbol.upper(), "asof": str(asof),
                "contracts": len(frame), "path": str(path)}

    def load(self, symbol: str, asof: dt.date | None = None) -> pd.DataFrame:
        if asof is not None:
            p = self.path_for(symbol, asof)
            return pd.read_parquet(p) if p.exists() else pd.DataFrame()
        found = sorted(self.root.glob(f"{symbol.upper()}_*.parquet"))
        if not found:
            return pd.DataFrame()
        return pd.read_parquet(found[-1])

    def dates(self, symbol: str) -> list:
        out = []
        for p in sorted(self.root.glob(f"{symbol.upper()}_*.parquet")):
            try:
                out.append(dt.date.fromisoformat(p.stem.split("_", 1)[1]))
            except ValueError:
                continue
        return out


def fetch_chain(symbol: str, max_expiries: int = 6) -> pd.DataFrame:
    """Pull the nearest `max_expiries` expiries for `symbol`.

    Every row carries its expiry and side, so calls and puts across several
    dates live in one frame and a later comparison does not have to re-join
    anything.
    """
    import yfinance as yf

    ticker = yf.Ticker(symbol)
    expiries = list(ticker.options or [])
    if not expiries:
        raise ValueError(f"no expiries returned for {symbol}")

    frames = []
    for expiry in expiries[:max_expiries]:
        try:
            chain = ticker.option_chain(expiry)
        except Exception:
            continue
        for side, df in (("call", chain.calls), ("put", chain.puts)):
            if df is None or df.empty:
                continue
            keep = [c for c in KEEP if c in df.columns]
            part = df[keep].copy()
            part["expiry"] = expiry
            part["side"] = side
            frames.append(part)

    if not frames:
        raise ValueError(f"no usable chains for {symbol}")
    out = pd.concat(frames, ignore_index=True)
    out["fetched_at"] = pd.Timestamp.utcnow().isoformat()
    return out


def iv_vs_realised(chain: pd.DataFrame, spot: float, realised_vol: float,
                   within_pct: float = 0.05) -> dict:
    """Compare near-the-money implied vol against what the underlying does.

    This is the whole question in one number. Options normally trade ABOVE
    realised volatility - the variance risk premium is why selling them is the
    crowded side - so implied sitting materially below realised would be a real
    and unusual edge for a buyer, and worth checking carefully rather than
    believing.
    """
    if chain.empty or "impliedVolatility" not in chain:
        return {"note": "no implied volatility in the chain"}
    near = chain[(chain["strike"] - spot).abs() <= spot * within_pct]
    near = near[near["impliedVolatility"] > 0]
    if near.empty:
        return {"note": "no near-the-money contracts"}
    iv = float(near["impliedVolatility"].median())
    return {
        "near_the_money_iv": iv,
        "realised_vol": realised_vol,
        "premium": iv - realised_vol,
        "ratio": iv / realised_vol if realised_vol > 0 else float("nan"),
        "contracts": int(len(near)),
        "verdict": ("implied BELOW realised - unusual, check the quote"
                    if iv < realised_vol else
                    "implied above realised, as normal"),
    }
