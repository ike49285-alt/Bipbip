"""Cross-sectional portfolio engine."""
import numpy as np
import pandas as pd

from bipbip.core.costs import CostModel
from bipbip.core.panel import build_panel
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


def test_instant_settlement_funds_a_switch_in_the_same_rebalance():
    """Regression: settle_days=0 was not instant.

    Sale proceeds were routed through a pending queue drained at the START of a
    bar, so a sale at bar i could not fund a purchase at bar i even with zero
    settlement days. A rotation strategy was therefore permanently unable to
    fund its own switch, and margin behaviour was never actually modelled -
    both account types produced identical, crippled results.
    """
    class Rotate(PortfolioStrategy):
        name = "rotate"
        warmup_bars = 1

        def target_weights(self, ctx):
            pick = "AAA" if (ctx.i // 20) % 2 == 0 else "BBB"
            return {pick: 0.98}

    panel = _panel()
    cash = PortfolioEngine(CostModel(), starting_equity=50.0,
                           rebalance_band=0.10, settle_days=1).run(panel, Rotate())
    margin = PortfolioEngine(CostModel(), starting_equity=50.0,
                             rebalance_band=0.10, settle_days=0).run(panel, Rotate())

    assert len(margin.unfunded) < len(cash.unfunded), (
        "instant settlement must fund switches the cash account cannot")
    # And the margin book should actually reach its target weight.
    assert margin.weights.sum(axis=1).max() > 0.9


class GoFlatAfter(PortfolioStrategy):
    """Fully invested, then asks for cash from bar `cut` onward."""
    name = "go_flat"
    warmup_bars = 1

    def __init__(self, cut=300):
        self.cut = cut

    def target_weights(self, ctx):
        if ctx.i >= self.cut:
            return {}
        n = len(ctx.tradeable)
        return {s: 1.0 / n for s in ctx.tradeable} if n else {}


class AbstainAfter(GoFlatAfter):
    """Identical, except it abstains instead of asking for cash."""
    name = "abstain"

    def target_weights(self, ctx):
        if ctx.i >= self.cut:
            return None
        return super().target_weights(ctx)


def test_an_empty_target_means_cash_not_hold():
    """Regression: `{}` was read as "no opinion", so go-to-cash never fired.

    The engine tested the target for truthiness, and an empty dict is falsey,
    so a strategy asking to hold nothing was left fully invested instead. Every
    downside rule in the project was silently inoperative because of it -
    momentum's absolute filter and the trend filter both scored an identical
    2008 to buy-and-hold, which is what gave it away.
    """
    panel = _panel()
    res = PortfolioEngine(CostModel(), starting_equity=50.0,
                          settle_days=0).run(panel, GoFlatAfter(cut=300))
    late = res.weights.iloc[350:]
    assert late.abs().max().max() < 1e-6, "still holding after asking for cash"


def test_none_means_hold_the_existing_book():
    """The other half of the contract: abstaining must NOT liquidate."""
    panel = _panel()
    res = PortfolioEngine(CostModel(), starting_equity=50.0,
                          settle_days=0).run(panel, AbstainAfter(cut=300))
    late = res.weights.iloc[350:]
    assert late.sum(axis=1).min() > 0.5, "abstaining sold the book off"


def test_going_flat_preserves_capital_in_a_crash():
    """The behaviour the contract exists to make possible."""
    n = 600
    idx = pd.DatetimeIndex(pd.bdate_range("2015-01-05", periods=n))
    path = np.concatenate([np.full(300, 100.0), np.linspace(100.0, 40.0, n - 300)])
    df = pd.DataFrame({"open": path, "high": path, "low": path,
                       "close": path, "volume": 1e7}, index=idx)
    panel = build_panel({"AAA": df})

    flat = PortfolioEngine(CostModel(), starting_equity=50.0,
                           settle_days=0).run(panel, GoFlatAfter(cut=300))
    held = PortfolioEngine(CostModel(), starting_equity=50.0,
                           settle_days=0).run(panel, AbstainAfter(cut=300))
    assert flat.final_equity > held.final_equity * 1.5
