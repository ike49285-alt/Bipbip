"""Simulate a leveraged ETF from its underlying index.

Backtesting TQQQ on TQQQ's own history is close to worthless. It listed in
February 2010, so its record covers one of the strongest technology bull runs
ever recorded and misses the dot-com collapse entirely. QQQ fell about 83% from
2000 to 2002; a 3x fund tracking it would have been destroyed. Any strategy
tuned on TQQQ's actual history has never seen the event that would have killed
it.

Simulating the fund from the index instead extends the test back to QQQ's 1999
inception, which includes that collapse and 2008. The simulation is honest
about the three costs that make leveraged funds different from leverage:

  * DAILY RESET. The fund targets 3x the DAILY return, not 3x the period
    return. Compounding daily returns is not the same as compounding the
    period, and in choppy markets the difference is a persistent drag known as
    volatility decay. Simulating day by day reproduces it exactly rather than
    approximating it.
  * BORROWING. Two of the three dollars are borrowed, at roughly the short
    rate plus a spread.
  * EXPENSES. Around 0.95% a year for these funds, an order of magnitude above
    a plain index ETF.
"""
from __future__ import annotations

import pandas as pd

#: Typical expense ratio for a 3x US equity ETF.
DEFAULT_EXPENSE_RATIO = 0.0095
#: Financing spread over the short rate on the borrowed portion.
DEFAULT_FINANCING_SPREAD = 0.006


def simulate_leveraged(
    index_bars: pd.DataFrame,
    leverage: float = 3.0,
    expense_ratio: float = DEFAULT_EXPENSE_RATIO,
    short_rate: float = 0.02,
    financing_spread: float = DEFAULT_FINANCING_SPREAD,
    start_price: float = 100.0,
) -> pd.DataFrame:
    """Build a synthetic leveraged fund series from `index_bars`.

    Returns an OHLCV frame on the index's dates. Intraday high and low are
    approximated by scaling the index's own intraday range, which is adequate
    for stop placement and honest about being an approximation.
    """
    close = index_bars["close"].astype("float64")
    ret = close.pct_change()

    daily_cost = (expense_ratio + (leverage - 1.0) * (short_rate + financing_spread)) / 252.0
    lev_ret = leverage * ret - daily_cost

    # A fund cannot lose more than everything in a day.
    lev_ret = lev_ret.clip(lower=-0.99)

    nav = start_price * (1.0 + lev_ret.fillna(0.0)).cumprod()

    # Scale the index's own intraday range onto the levered NAV.
    hi_frac = (index_bars["high"] / close).astype("float64")
    lo_frac = (index_bars["low"] / close).astype("float64")
    op_frac = (index_bars["open"] / close).astype("float64")

    out = pd.DataFrame(index=index_bars.index)
    out["close"] = nav
    out["open"] = nav * (1.0 + leverage * (op_frac - 1.0))
    out["high"] = nav * (1.0 + leverage * (hi_frac - 1.0))
    out["low"] = nav * (1.0 + leverage * (lo_frac - 1.0)).clip(lower=0.01)
    out["volume"] = index_bars["volume"].astype("float64")
    out = out[["open", "high", "low", "close", "volume"]]
    return out.dropna()


def decay_report(index_bars: pd.DataFrame, leverage: float = 3.0) -> dict:
    """Quantify what daily resetting costs against naive leverage."""
    sim = simulate_leveraged(index_bars, leverage)
    idx_total = float(index_bars["close"].iloc[-1] / index_bars["close"].iloc[0])
    lev_total = float(sim["close"].iloc[-1] / sim["close"].iloc[0])
    return {
        "index_multiple": idx_total,
        "naive_multiple": idx_total ** leverage if idx_total > 0 else float("nan"),
        "levered_multiple": lev_total,
        "years": len(index_bars) / 252,
    }
