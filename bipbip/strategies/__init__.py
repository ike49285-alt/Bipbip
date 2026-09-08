"""Strategy registry."""
from ..core.strategy import Strategy
from .baseline import NeverTrade, SessionBuyHold
from .exit_filter import HTFExitFilter, trend_zscore
from .opening_range import OpeningRangeBreakout
from .vwap_reversion import VWAPReversion

REGISTRY = {
    "buy_hold": SessionBuyHold,
    "never_trade": NeverTrade,
    "orb": OpeningRangeBreakout,
    "vwap_reversion": VWAPReversion,
}


def get_strategy(name: str, **kwargs) -> Strategy:
    if name not in REGISTRY:
        raise ValueError(f"unknown strategy {name!r}; choose from {sorted(REGISTRY)}")
    return REGISTRY[name](**kwargs)


__all__ = ["REGISTRY", "get_strategy", "Strategy", "SessionBuyHold", "NeverTrade",
           "OpeningRangeBreakout", "VWAPReversion", "HTFExitFilter", "trend_zscore"]
