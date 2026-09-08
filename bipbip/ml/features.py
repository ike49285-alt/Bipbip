"""Feature engineering.

Every column here must be computable at bar `i` from bars `<= i`. That
constraint is enforced by `tests/test_lookahead.py::test_features_are_causal`,
which recomputes the frame on truncated history and demands the overlap match.
A single non-causal feature invalidates every result downstream, so new
features belong here only if they pass that test.

The design intent: hand-crafted signals as features, letting the model learn
WHEN each setup pays rather than rediscovering technical analysis from raw
prices. With a few hundred usable observations, giving the model structure is
the difference between learning and memorising.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core import indicators as ind

FEATURE_COLUMNS = [
    "vwap_stretch_atr",
    "rsi",
    "atr_pct",
    "rvol",
    "or_position",
    "or_broken",
    "bars_since_or_break",
    "or_width_bps",
    "ret_1",
    "ret_5",
    "ret_15",
    "range_pct",
    "close_in_bar",
    "dist_from_high_atr",
    "dist_from_low_atr",
    "bar_of_day",
    "vol_of_vol",
]


def build_features(bars: pd.DataFrame, or_minutes: int = 30) -> pd.DataFrame:
    """Return the causal feature frame aligned to `bars.index`."""
    key = np.asarray([ts.date() for ts in bars.index])
    close, high, low = bars["close"], bars["high"], bars["low"]

    vwap = ind.session_vwap(bars)
    atr = ind.atr(bars, 30)
    orng = ind.opening_range(bars, or_minutes)
    atr_safe = atr.replace(0, np.nan)

    f = pd.DataFrame(index=bars.index)

    # Where price sits relative to the institutional benchmark, in risk units.
    f["vwap_stretch_atr"] = (close - vwap) / atr_safe
    f["rsi"] = ind.rsi(close, 14)
    f["atr_pct"] = atr / close
    f["rvol"] = ind.relative_volume(bars, 20)

    # Opening-range geometry: position within the range, whether it has broken,
    # and how long ago - a break twenty minutes stale is not a fresh signal.
    width = (orng["or_high"] - orng["or_low"]).replace(0, np.nan)
    f["or_position"] = (close - orng["or_low"]) / width
    broken = ((close > orng["or_high"]) & orng["or_complete"]).astype("float64")
    f["or_broken"] = broken
    # Bars since the most recent break, reset each session.
    grp = broken.groupby(key)
    since = grp.apply(lambda s: _bars_since(s.to_numpy()))
    f["bars_since_or_break"] = np.concatenate(since.to_list()) if len(since) else 0.0
    f["or_width_bps"] = (width / close) * 10_000.0

    # Momentum over several horizons.
    for n in (1, 5, 15):
        f[f"ret_{n}"] = np.log(close / close.shift(n))

    # Bar shape: where the close sits in its own range is a crude order-flow proxy.
    bar_range = (high - low).replace(0, np.nan)
    f["range_pct"] = bar_range / close
    f["close_in_bar"] = (close - low) / bar_range

    # Distance from the session's running extremes, in risk units.
    sess_high = high.groupby(key).cummax()
    sess_low = low.groupby(key).cummin()
    f["dist_from_high_atr"] = (sess_high - close) / atr_safe
    f["dist_from_low_atr"] = (close - sess_low) / atr_safe

    # Time of day matters enormously intraday; normalised to [0, 1].
    bar_of_day = ind.minutes_since_open(bars.index).astype("float64")
    f["bar_of_day"] = bar_of_day / 390.0

    # Volatility of volatility - regime instability.
    f["vol_of_vol"] = f["atr_pct"].rolling(60, min_periods=30).std()

    return f[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)


def _bars_since(flags: np.ndarray) -> np.ndarray:
    """Bars elapsed since `flags` was last 1, capped; large when never seen."""
    out = np.full(len(flags), 999.0)
    last = -1
    for i, v in enumerate(flags):
        if v:
            last = i
        out[i] = (i - last) if last >= 0 else 999.0
    return out
