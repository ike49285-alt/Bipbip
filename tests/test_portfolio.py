"""Cross-sectional portfolio engine."""
import numpy as np
import pandas as pd
import pytest

from bipbip.core.costs import CostModel
from bipbip.core.panel import Panel, build_panel
from bipbip.core.portfolio import PortfolioEngine, PortfolioStrategy


def _series(n, start, seed, gap=None):
    rng = np.random.default_rng(seed)
    close = start * np.exp(np.cumsum(rng.normal(0.0003, 0.011, n)))
    idx = pd.DatetimeIndex(pd.bdate_range("2015-01-05", periods=n))
    df = pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99,
                       "close": close, "volume": 1e7}, index=idx)
    if gap:
        df.iloc[gap[0]:gap[1]] = np.nan
    return df


def _panel(gap=None):
    return build_panel({"AAA": _series(600, 100, 1),
                        "BBB": _series(600, 50, 2),
                        "CCC": _series(600, 200, 3, gap=gap)})


class HoldAll(PortfolioStrategy):
    name = "hold_all"
    warmup_bars = 1

    def target_weights(self, ctx):
        n = len(ctx.tradeable)
        return {s: 1.0 / n for s in ctx.tradeable} if n else {}


def test_a_data_gap_does_not_poison_the_equity_curve():
    """Regression: one missing bar turned the whole curve into NaN.

    A held symbol with no quote still has to be VALUED. Marking it at zero
    would show a fake loss; refusing to value it made equity non-finite, and
    the metrics then reported a Sharpe ratio for a curve ending in NaN.
    """
    res = PortfolioEngine(CostModel(), starting_equity=50.0).run(_panel(gap=(300, 340)), HoldAll())
    assert np.isfinite(res.equity_curve).all()
    assert (res.equity_curve > 0).all()


def test_rebalance_band_suppresses_daily_churn():
    """Regression: naming the same target every day still traded every day.

    Prices drift the realised weights off target overnight, so an unbanded
    engine trades continuously to correct rounding noise - 24,001 trades over
    8,459 sessions on a 34-symbol universe, all of it paying spread.
    """
    panel = _panel()
    tight = PortfolioEngine(CostModel(), starting_equity=50.0, rebalance_band=0.0,
                            min_trade_value=0.0).run(panel, HoldAll())
    banded = PortfolioEngine(CostModel(), starting_equity=50.0,
                             rebalance_band=0.10).run(panel, HoldAll())
    assert len(banded.trades) < len(tight.trades) / 5


def test_symbols_are_not_traded_before_they_list():
    """A backtest must not buy something that had not started trading."""
    late = _series(600, 80, 4)
    late.iloc[:400] = np.nan
    panel = build_panel({"AAA": _series(600, 100, 1), "LATE": late})

    res = PortfolioEngine(CostModel(), starting_equity=50.0).run(panel, HoldAll())
    first = [t for t in res.trades if t["symbol"] == "LATE"]
    assert first, "expected the late lister to be traded eventually"
    assert min(t["date"] for t in first) >= panel.dates[400]


def test_buys_are_limited_by_settled_cash():
    """A cash account cannot sell one name and buy another the same day: the
    proceeds are unsettled until T+1."""
    class Rotate(PortfolioStrategy):
        name = "rotate"
        warmup_bars = 1

        def target_weights(self, ctx):
            # Alternate the entire book between two names every day.
            pick = "AAA" if ctx.i % 2 == 0 else "BBB"
            return {pick: 0.98}

    res = PortfolioEngine(CostModel(), starting_equity=50.0,
                          rebalance_band=0.01).run(_panel(), Rotate())
    assert res.unfunded, "same-day rotation should hit the settlement constraint"
    assert np.isfinite(res.equity_curve).all()


def test_weights_never_exceed_the_invested_cap():
    class Greedy(PortfolioStrategy):
        name = "greedy"
        warmup_bars = 1

        def target_weights(self, ctx):
            return {s: 1.0 for s in ctx.tradeable}   # 300% requested

    res = PortfolioEngine(CostModel(), starting_equity=50.0,
                          max_invested=0.98).run(_panel(), Greedy())
    assert res.weights.sum(axis=1).max() <= 1.02


def test_panel_reports_only_tradeable_symbols():
    panel = _panel(gap=(100, 150))
    assert "CCC" in panel.tradeable(50)
    assert "CCC" not in panel.tradeable(120)
    assert "CCC" in panel.tradeable(200)


def test_panel_does_not_forward_fill_across_gaps():
    """Filling would let a strategy trade a price that never existed."""
    panel = _panel(gap=(100, 150))
    assert panel.closes["CCC"].iloc[100:150].isna().all()
