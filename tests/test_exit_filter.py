"""Higher-timeframe exit filter, and above all what it must never do."""
import numpy as np
import pandas as pd
import pytest

from bipbip.core import BacktestEngine, CashAccount, CostModel
from bipbip.core.strategy import Strategy
from bipbip.core.types import HOLD, Intent
from bipbip.data import make_intraday_bars
from bipbip.strategies import HTFExitFilter, get_strategy
from bipbip.strategies.exit_filter import trend_zscore


class AlwaysExit(Strategy):
    """Enters on the first eligible bar, then asks to exit on every bar after.

    Isolates the filter: every exit in a run is discretionary, so anything that
    survives came from the engine rather than the strategy.
    """

    name = "always_exit"
    warmup_bars = 35

    def __init__(self, stop=None, target=None):
        self.stop, self.target = stop, target

    def prepare(self, bars):
        from bipbip.core import indicators as ind
        out = pd.DataFrame(index=bars.index)
        out["atr"] = ind.atr(bars, 30)
        return out

    def on_bar(self, ctx):
        if not ctx.in_position:
            atr = float(ctx.ind["atr"])
            if not np.isfinite(atr) or atr <= 0:
                return HOLD
            price = ctx.price
            return Intent(action="enter", reason="test",
                          stop_price=price - (self.stop or 1.0) * atr,
                          target_price=price + (self.target or 50.0) * atr)
        return Intent(action="exit", reason="always")


def test_filter_defers_discretionary_exits_while_the_trend_drives():
    bars = make_intraday_bars(n_sessions=20, seed=301)
    filt = HTFExitFilter(AlwaysExit(), min_trend_z=0.5)
    BacktestEngine(CashAccount(10_000.0), CostModel()).run("SPY", bars, filt)
    assert filt.suppressed_events, "filter never engaged"
    for e in filt.suppressed_events:
        assert e["trend_z"] >= 0.5


def test_filter_cannot_defer_a_stop():
    """The safety property, and it is structural rather than conventional.

    The filter only sees intents the strategy returns; stops are evaluated by
    the engine, which never consults a strategy. So no configuration can widen
    a stop. This test pins that, because "the higher timeframe is still intact"
    is exactly the sentence that turns a small loss into a large one.
    """
    bars = make_intraday_bars(n_sessions=25, seed=302, annual_vol=0.40)
    # Suppress everything the filter is capable of suppressing.
    filt = HTFExitFilter(AlwaysExit(stop=0.5), min_trend_z=-99.0,
                         max_suppressed_bars=10_000)
    res = BacktestEngine(CashAccount(10_000.0), CostModel()).run("SPY", bars, filt)

    stops = [t for t in res.trades if t.exit_reason == "stop"]
    assert stops, "expected stops to fire despite total suppression"
    for t in stops:
        assert t.exit_price < t.entry_price


def test_filter_cannot_carry_a_position_overnight():
    bars = make_intraday_bars(n_sessions=20, seed=303)
    filt = HTFExitFilter(AlwaysExit(stop=99.0), min_trend_z=-99.0,
                         max_suppressed_bars=10_000)
    res = BacktestEngine(CashAccount(10_000.0), CostModel()).run("SPY", bars, filt)
    assert res.trades
    for t in res.trades:
        assert t.entry_time.date() == t.exit_time.date()


def test_suppression_is_capped_per_trade():
    """Bounds the filter's own influence, so a persistent reading cannot defer
    an exit indefinitely."""
    bars = make_intraday_bars(n_sessions=20, seed=304)
    cap = 3
    filt = HTFExitFilter(AlwaysExit(), min_trend_z=-99.0, max_suppressed_bars=cap)
    BacktestEngine(CashAccount(10_000.0), CostModel()).run("SPY", bars, filt)
    assert filt.suppressed_events
    assert max(e["n_this_trade"] for e in filt.suppressed_events) <= cap


def test_filter_forwards_the_cost_hurdle_to_the_inner_strategy():
    """Otherwise the inner strategy's risk floor silently reverts to a default."""
    bars = make_intraday_bars(n_sessions=10, seed=305)
    inner = get_strategy("vwap_reversion")
    filt = HTFExitFilter(inner)
    BacktestEngine(CashAccount(10_000.0), CostModel()).run("SPY", bars, filt)
    assert inner.cost_hurdle_bps == filt.cost_hurdle_bps
    assert inner.cost_hurdle_bps != Strategy.cost_hurdle_bps


def test_filter_rejects_an_indicator_name_collision():
    class Colliding(Strategy):
        name = "colliding"

        def prepare(self, bars):
            return pd.DataFrame({"htf_trend_z": 0.0}, index=bars.index)

        def on_bar(self, ctx):
            return HOLD

    with pytest.raises(ValueError, match="htf_trend_z"):
        BacktestEngine(CashAccount(10_000.0), CostModel()).run(
            "SPY", make_intraday_bars(n_sessions=3), HTFExitFilter(Colliding()))


def test_trend_zscore_is_causal_and_scale_free():
    bars = make_intraday_bars(n_sessions=8, seed=306)
    cut = int(len(bars) * 0.7)
    pd.testing.assert_series_equal(
        trend_zscore(bars["close"]).iloc[:cut],
        trend_zscore(bars["close"].iloc[:cut]),
        check_exact=False, rtol=1e-9,
    )
    z = trend_zscore(bars["close"]).dropna()
    assert 0.5 < z.std() < 2.5, "a z-score should be near unit scale"


def test_ml_signal_exit_uses_hysteresis():
    """Without a gap between entry and exit thresholds a probability hovering
    at the boundary would round-trip on alternate bars, paying costs each time."""
    from bipbip.ml import MODELS, build_dataset
    from bipbip.strategies.ml_strategy import MLStrategy

    bars = make_intraday_bars(n_sessions=20, seed=307)
    ds = build_dataset(bars)
    model = MODELS["logistic"]().fit(ds.X.to_numpy(dtype="float64"), ds.y)

    strat = MLStrategy(model, threshold=0.55, exit_threshold=0.45)
    assert strat.exit_threshold < strat.threshold
    res = BacktestEngine(CashAccount(10_000.0), CostModel()).run("SPY", bars, strat)
    per_day = {}
    for t in res.trades:
        per_day[t.entry_time.date()] = per_day.get(t.entry_time.date(), 0) + 1
    assert not per_day or max(per_day.values()) <= 1


def test_ml_signal_exit_can_be_disabled():
    from bipbip.ml import MODELS, build_dataset
    from bipbip.strategies.ml_strategy import MLStrategy

    bars = make_intraday_bars(n_sessions=15, seed=308)
    ds = build_dataset(bars)
    model = MODELS["logistic"]().fit(ds.X.to_numpy(dtype="float64"), ds.y)
    res = BacktestEngine(CashAccount(10_000.0), CostModel()).run(
        "SPY", bars, MLStrategy(model, threshold=0.55, exit_threshold=None))
    for t in res.trades:
        assert not t.exit_reason.startswith("ml_decayed")
