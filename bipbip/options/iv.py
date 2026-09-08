"""Implied volatility estimation.

Historical intraday option chains are not free, so implied vol cannot be
observed here - it has to be modelled. That is the single largest source of
error in everything downstream, and it is stated loudly rather than buried:
every option price this project produces is a MODEL OUTPUT, not a quote.

The estimate is realised volatility scaled by a variance risk premium. Index
options persistently trade above subsequent realised vol - sellers demand
payment for carrying gap risk - so implied runs roughly 10-25% above realised
for SPY. The premium is a parameter precisely so its effect can be swept, and
it should be: if a strategy's profitability flips over that range, the
strategy is trading the assumption rather than the market.

Two known limitations, unmodelled and both flattering:
  * No smile. Real OTM strikes trade above ATM vol, so away-from-the-money
    options here are priced too cheaply.
  * No intraday term structure. 0DTE implied vol typically firms into the
    afternoon rather than following realised vol down.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .pricing import TRADING_MINUTES_PER_YEAR

#: Typical ratio of implied to subsequent realised vol for SPY index options.
DEFAULT_VARIANCE_RISK_PREMIUM = 1.15

#: Empirical floors on implied vol, by symbol. These are ESTIMATES, and after
#: the modelled IV itself they are the largest assumption in the options code.
#:
#: A flat multiplier on realised vol is the wrong shape, because the variance
#: risk premium expands when markets are quiet: VIX rarely prints below about
#: 11 even in stretches where SPY realises 5%. Measured over this project's own
#: archive SPY realised 6.2% annualised (consistent across 1, 5, 15 and 30
#: minute sampling, so not a microstructure artefact) while its options would
#: never have been quoted anywhere near that. Without a floor the model prices
#: an ATM 0DTE call at $0.86 when the real quote is $2-4, and any backtest
#: built on it invents free money.
#:
#: TQQQ is a 3x leveraged fund, so its vol runs roughly three times the index's.
IV_FLOORS = {"SPY": 0.10, "QQQ": 0.12, "TQQQ": 0.32, "default": 0.15}


def iv_floor_for(symbol: str | None) -> float:
    """Floor for `symbol`, falling back to a conservative default."""
    if not symbol:
        return IV_FLOORS["default"]
    return IV_FLOORS.get(symbol.upper(), IV_FLOORS["default"])


def realised_vol(close: pd.Series, window: int = 60) -> pd.Series:
    """Annualised realised volatility from one-minute log returns.

    Annualised on the TRADING clock, so it is consistent with the time-to-expiry
    convention in `pricing`. Mixing the two silently mis-prices everything.
    """
    r = np.log(close / close.shift(1))
    return (r.rolling(window, min_periods=max(10, window // 3)).std()
            * np.sqrt(TRADING_MINUTES_PER_YEAR)).rename("realised_vol")


def implied_vol(
    close: pd.Series,
    window: int = 60,
    premium: float = DEFAULT_VARIANCE_RISK_PREMIUM,
    symbol: str | None = None,
    floor: float | None = None,
    cap: float = 1.50,
) -> pd.Series:
    """Model an implied vol series from realised vol, a risk premium and a floor.

    The floor is not a numerical safeguard but a statement about how options are
    actually priced: implied vol does not follow realised vol all the way down,
    because sellers still demand payment for gap risk in quiet markets. Omitting
    it under-prices every option and manufactures profit that does not exist.

    Pass `symbol` to pick the instrument's floor, or `floor` to set it directly.
    Both this and `premium` should be swept before any result is believed.
    """
    rv = realised_vol(close, window)
    iv = (rv * premium).ffill()
    lower = iv_floor_for(symbol) if floor is None else floor
    return iv.clip(lower=lower, upper=cap).rename("implied_vol")
