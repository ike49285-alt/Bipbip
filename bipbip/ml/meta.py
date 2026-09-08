"""Meta-labelling.

A model asked to predict direction has to solve the hard problem badly. A
meta-labelling model is asked something much easier: given that a simple rule
has already fired, is THIS instance of the setup one of the good ones?

That reframing helps in three concrete ways.

  * The dataset is smaller and cleaner. Instead of every bar, it contains only
    bars where the primary rule fired, and the label is the outcome of a trade
    that would actually have been taken.
  * The classes are closer to balanced. Predicting direction on financial data
    means separating two nearly identical distributions; predicting "did this
    setup work" is a genuine classification problem with signal in it.
  * It cannot do catastrophic damage. The secondary model may only SKIP or SIZE
    DOWN a trade the primary proposed. It never chooses a direction, so a bad
    secondary model costs opportunity, not capital.

The last point is why this is the right ML to attempt here. Every previous
model in this project could, in principle, put money into a position for
reasons nobody could inspect. This one can only decline.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..core import indicators as ind
from ..core.panel import Panel


@dataclass
class MetaDataset:
    X: pd.DataFrame
    y: np.ndarray
    rets: np.ndarray
    event_end: np.ndarray
    symbols: np.ndarray
    dates: pd.DatetimeIndex

    def __len__(self) -> int:
        return len(self.X)

    def summary(self) -> str:
        return (f"{len(self):,} signals across {len(set(self.symbols))} symbols, "
                f"{self.dates.min().date()} to {self.dates.max().date()} | "
                f"{self.y.mean():.1%} profitable | "
                f"mean net {self.rets.mean() * 100:+.2f}%")


def rsi_dip_signals(panel: Panel, rsi_window: int = 2, entry_rsi: float = 15.0,
                    trend_window: int = 200) -> pd.DataFrame:
    """Primary rule: an oversold reading inside an uptrend.

    Deliberately crude. The primary's job is to propose plausible moments, not
    to be right - the secondary model exists to sort the proposals.
    """
    c = panel.closes
    rsi = c.apply(lambda s: ind.rsi(s.dropna(), rsi_window).reindex(s.index))
    trend = c.rolling(trend_window, min_periods=trend_window).mean()
    return (rsi < entry_rsi) & (c > trend)


def breakout_signals(panel: Panel, window: int = 50, trend_window: int = 200) -> pd.DataFrame:
    """Primary rule: a new N-day high inside an uptrend."""
    c = panel.closes
    high = c.rolling(window, min_periods=window).max()
    trend = c.rolling(trend_window, min_periods=trend_window).mean()
    return (c >= high) & (c > trend)


def build_features(panel: Panel, market: pd.Series | None = None) -> dict:
    """Causal per-symbol features, plus cross-sectional and market context.

    Every column uses data available at the close of the bar it is indexed on.
    Cross-sectional ranks are computed across symbols WITHIN a date, which uses
    no future information: on any given day, every symbol's own history is
    already known.
    """
    c, h, l, v = panel.closes, panel.highs, panel.lows, panel.volumes
    f: dict = {}

    f["rsi2"] = c.apply(lambda s: ind.rsi(s.dropna(), 2).reindex(s.index))
    f["rsi14"] = c.apply(lambda s: ind.rsi(s.dropna(), 14).reindex(s.index))

    sma200 = c.rolling(200, min_periods=200).mean()
    sma50 = c.rolling(50, min_periods=50).mean()
    f["dist_sma200"] = (c / sma200 - 1.0)
    f["dist_sma50"] = (c / sma50 - 1.0)

    for n in (5, 20, 60, 252):
        f[f"mom_{n}"] = np.log(c / c.shift(n))

    vol20 = np.log(c / c.shift(1)).rolling(20, min_periods=20).std() * np.sqrt(252)
    vol60 = np.log(c / c.shift(1)).rolling(60, min_periods=60).std() * np.sqrt(252)
    f["vol20"] = vol20
    f["vol_ratio"] = vol20 / vol60.replace(0, np.nan)

    high52 = c.rolling(252, min_periods=100).max()
    f["dist_52w_high"] = c / high52 - 1.0
    f["range_pct"] = (h - l) / c
    f["volume_ratio"] = v / v.rolling(20, min_periods=20).mean().replace(0, np.nan)

    # Cross-sectional: where this symbol sits among its peers TODAY.
    f["rank_mom_60"] = f["mom_60"].rank(axis=1, pct=True)
    f["rank_vol20"] = vol20.rank(axis=1, pct=True)
    f["rank_rsi2"] = f["rsi2"].rank(axis=1, pct=True)

    if market is not None:
        m = market.reindex(c.index).ffill()
        m_sma = m.rolling(200, min_periods=200).mean()
        m_vol = np.log(m / m.shift(1)).rolling(21, min_periods=21).std() * np.sqrt(252)
        # Broadcast market state onto every symbol column.
        f["mkt_above_sma"] = pd.DataFrame(
            np.repeat((m > m_sma).to_numpy()[:, None], len(c.columns), axis=1),
            index=c.index, columns=c.columns).astype("float64")
        f["mkt_vol_pct"] = pd.DataFrame(
            np.repeat(m_vol.expanding(min_periods=252).apply(
                lambda w: (w[:-1] < w[-1]).mean(), raw=True).to_numpy()[:, None],
                len(c.columns), axis=1),
            index=c.index, columns=c.columns)
    return f


def label_signals(panel: Panel, signals: pd.DataFrame, features: dict,
                  target_pct: float = 0.04, stop_pct: float = 0.03,
                  max_hold: int = 20, cost_bps: float = 5.0) -> MetaDataset:
    """Label each primary signal by the outcome of the trade it implies.

    Entry is the NEXT bar's open, exactly as the engine fills. The trade
    resolves at whichever comes first: the target, the stop, or the holding
    limit. Costs are subtracted, so a "win" is a win after paying to trade.
    Stops are checked before targets on a bar that touches both.
    """
    opens, highs, lows, closes = panel.opens, panel.highs, panel.lows, panel.closes
    dates = panel.dates
    cost = cost_bps / 10_000.0

    rows, ys, rets, ends, syms, when = [], [], [], [], [], []
    cols = list(signals.columns)
    feat_names = list(features)

    for sym in cols:
        sig = signals[sym].to_numpy()
        o = opens[sym].to_numpy(dtype="float64")
        hi = highs[sym].to_numpy(dtype="float64")
        lo = lows[sym].to_numpy(dtype="float64")
        cl = closes[sym].to_numpy(dtype="float64")
        fvals = {k: features[k][sym].to_numpy(dtype="float64") for k in feat_names}

        idxs = np.flatnonzero(sig)
        for i in idxs:
            j0 = i + 1
            if j0 >= len(dates) or not np.isfinite(o[j0]) or o[j0] <= 0:
                continue
            entry = o[j0]
            tgt, stp = entry * (1 + target_pct), entry * (1 - stop_pct)

            end, outcome, r = min(j0 + max_hold, len(dates) - 1), 0, np.nan
            for j in range(j0, end + 1):
                if not np.isfinite(cl[j]):
                    continue
                if np.isfinite(lo[j]) and lo[j] <= stp:
                    outcome, end, r = 0, j, (stp / entry - 1) - cost
                    break
                if np.isfinite(hi[j]) and hi[j] >= tgt:
                    outcome, end, r = 1, j, (tgt / entry - 1) - cost
                    break
            else:
                last = cl[end] if np.isfinite(cl[end]) else entry
                r = (last / entry - 1) - cost
                outcome = 1 if r > 0 else 0

            row = [fvals[k][i] for k in feat_names]
            if not np.any(np.isfinite(row)):
                continue
            rows.append(row); ys.append(outcome); rets.append(r)
            ends.append(end); syms.append(sym); when.append(dates[i])

    if not rows:
        raise ValueError("the primary rule produced no usable signals")

    X = pd.DataFrame(rows, columns=feat_names)
    X = X.replace([np.inf, -np.inf], np.nan).clip(lower=-20.0, upper=20.0)

    order = np.argsort(np.asarray(when))
    X = X.iloc[order].reset_index(drop=True)
    when_sorted = pd.DatetimeIndex(np.asarray(when)[order])

    # event_end as a POSITION in the sorted sample, so purging can measure
    # overlap between training and validation folds.
    ends_sorted = np.asarray(ends)[order]
    pos_of_date = {d: k for k, d in enumerate(when_sorted)}
    remap = np.array([
        pos_of_date.get(dates[e], k) if e < len(dates) else k
        for k, e in enumerate(ends_sorted)
    ])
    remap = np.maximum(remap, np.arange(len(remap)))

    return MetaDataset(
        X=X, y=np.asarray(ys, dtype="float64")[order],
        rets=np.asarray(rets, dtype="float64")[order],
        event_end=remap, symbols=np.asarray(syms)[order], dates=when_sorted,
    )
