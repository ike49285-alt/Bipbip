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

import numpy as np
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

    def path_for(self, symbol: str, asof: dt.date | dt.datetime) -> Path:
        # Four snapshots a day means the DATE alone is not a key; a
        # date-stamped filename would have each run silently overwrite the last
        # and leave one chain a day, which is the whole point lost.
        if isinstance(asof, dt.datetime):
            stamp = asof.strftime("%Y-%m-%dT%H%M")
        else:
            stamp = asof.isoformat()
        return self.root / f"{symbol.upper()}_{stamp}.parquet"

    def save(self, symbol: str, frame: pd.DataFrame,
             asof: "dt.date | dt.datetime | None" = None) -> dict:
        asof = asof or dt.datetime.now(dt.timezone.utc)
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
            stamp = p.stem.split("_", 1)[1]
            try:
                out.append(dt.datetime.fromisoformat(stamp)
                           if "T" in stamp else dt.date.fromisoformat(stamp))
            except ValueError:
                continue
        return out


def fetch_chain(symbol: str, max_expiries: int = 6,
                spot: float | None = None) -> pd.DataFrame:
    """Pull the nearest `max_expiries` expiries for `symbol`.

    Every row carries its expiry, side, the UNDERLYING PRICE at the moment of
    the snapshot, and the timestamp. Storing the spot alongside is what makes
    repeated snapshots usable: an option price is meaningless without knowing
    where the underlying was when it was quoted, and it is the pair that lets
    delta be measured rather than modelled. It also makes the collector robust
    to scheduling drift - a snapshot that fires eleven minutes late is still
    exact, because it records where the stock actually was.
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
    stamp = pd.Timestamp.now(tz="UTC")
    out["fetched_at"] = stamp.isoformat()
    if spot is None:
        try:
            hist = ticker.history(period="1d", interval="1m")
            spot = float(hist["Close"].iloc[-1]) if not hist.empty else float("nan")
        except Exception:
            spot = float("nan")
    out["spot"] = spot
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


def add_greeks(chain: pd.DataFrame, asof: pd.Timestamp | None = None,
               rate: float = 0.04) -> pd.DataFrame:
    """Derive greeks from the implied vol the market actually quoted.

    This is the legitimate half of "extrapolate what is not supplied". The feed
    carries no greeks, but it carries IV, and delta/gamma/theta/vega are
    functions of IV rather than independent facts - computing them from the
    market's own IV reproduces what the broker shows rather than substituting a
    private opinion about volatility.

    What is NOT legitimate, and is deliberately absent from this module: making
    up option prices between snapshots, or before collection began. Those are
    not extrapolation, they are invention, and a backtest run over them tests
    the interpolator rather than the market. Snapshots are observations; the
    gaps between them stay empty.
    """
    from ..options import pricing as bs

    if chain.empty:
        return chain
    asof = asof or pd.Timestamp.now(tz="UTC")
    out = chain.copy()

    expiry = pd.to_datetime(out["expiry"]).dt.tz_localize("UTC")
    # Trading minutes remaining, floored at zero - the clock this project
    # prices on, since a weekend contributes no decay.
    days = (expiry - asof).dt.total_seconds() / 86400.0
    minutes = (days.clip(lower=0.0) * (252.0 / 365.0) * 390.0).to_numpy()
    T = bs.minutes_to_years(minutes)

    S = out["spot"].to_numpy(dtype="float64")
    K = out["strike"].to_numpy(dtype="float64")
    iv = out["impliedVolatility"].to_numpy(dtype="float64")
    is_call = (out["side"] == "call").to_numpy()

    delta = np.where(is_call,
                     bs.delta(S, K, T, rate, iv, bs.CALL),
                     bs.delta(S, K, T, rate, iv, bs.PUT))
    theta = np.where(is_call,
                     bs.theta_per_minute(S, K, T, rate, iv, bs.CALL),
                     bs.theta_per_minute(S, K, T, rate, iv, bs.PUT))
    out["delta"] = delta
    out["gamma"] = bs.gamma(S, K, T, rate, iv)
    out["theta_per_day"] = theta * 390.0
    out["vega"] = bs.vega(S, K, T, rate, iv)
    out["minutes_to_expiry"] = minutes
    return out


def empirical_delta(snapshots: list) -> pd.DataFrame:
    """Measure delta from consecutive snapshots instead of assuming it.

    With the spot stored alongside each chain, the change in a contract's mid
    against the change in the underlying IS its realised delta over that
    interval. That is worth having precisely because it needs no model: if the
    measured value disagrees with Black-Scholes, the disagreement is
    information about the quote rather than an error to reconcile away.

    Returns one row per contract per consecutive pair.
    """
    if len(snapshots) < 2:
        return pd.DataFrame()

    def mid(df):
        m = (df["bid"] + df["ask"]) / 2.0
        return m.where((df["bid"] > 0) & (df["ask"] > 0))

    rows = []
    for before, after in zip(snapshots, snapshots[1:]):
        a = before.set_index("contractSymbol")
        b = after.set_index("contractSymbol")
        shared = a.index.intersection(b.index)
        if not len(shared):
            continue
        ds = float(b["spot"].iloc[0]) - float(a["spot"].iloc[0])
        if abs(ds) < 1e-9:
            continue
        dp = (mid(b.loc[shared]) - mid(a.loc[shared]))
        rows.append(pd.DataFrame({
            "contractSymbol": shared,
            "d_spot": ds,
            "d_mid": dp.to_numpy(),
            "empirical_delta": (dp / ds).to_numpy(),
            "modelled_delta": b.loc[shared]["delta"].to_numpy()
            if "delta" in b.columns else np.nan,
            "from": a["fetched_at"].iloc[0],
            "to": b["fetched_at"].iloc[0],
        }))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
