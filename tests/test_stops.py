"""Protective stops, and the ways a stop backtest lies.

A stop is the single easiest thing to make look free in a simulation. Fill it
at the stop price every time and it becomes perfect insurance at zero premium,
which is what a naive engine reports and what no broker delivers.
"""
import numpy as np
import pandas as pd
import pytest

from bipbip.core.costs import CostModel
from bipbip.core.panel import build_panel
from bipbip.core.portfolio import PortfolioEngine, PortfolioStrategy, StopPolicy


def _bars(closes, highs=None, lows=None, opens=None, start="2015-01-05"):
    closes = np.asarray(closes, dtype=float)
    opens = closes if opens is None else np.asarray(opens, dtype=float)
    highs = closes if highs is None else np.asarray(highs, dtype=float)
    lows = closes if lows is None else np.asarray(lows, dtype=float)
    idx = pd.DatetimeIndex(pd.bdate_range(start, periods=len(closes)))
    return pd.DataFrame({"open": opens, "high": highs, "low": lows,
                         "close": closes, "volume": 1e7}, index=idx)


class HoldAAA(PortfolioStrategy):
    name = "hold_aaa"
    warmup_bars = 1

    def target_weights(self, ctx):
        return {"AAA": 1.0} if "AAA" in ctx.tradeable else None


def _run(bars, stop=None, **kw):
    return PortfolioEngine(CostModel(), starting_equity=100.0, settle_days=0,
                           stop=stop, **kw).run(build_panel({"AAA": bars}), HoldAAA())


# --------------------------------------------------------------------------
# Does it fire at all, and does it cap the loss?
# --------------------------------------------------------------------------

def test_a_stop_exits_and_caps_the_loss():
    """Flat, then a slide to a third of the price."""
    path = np.concatenate([np.full(20, 100.0), np.linspace(100.0, 33.0, 80)])
    stopped = _run(_bars(path), stop=StopPolicy(pct=0.10, lockout_days=10_000))
    held = _run(_bars(path))

    assert stopped.stops, "the stop never fired during a 67% decline"
    assert stopped.final_equity > held.final_equity * 2


def test_no_stop_means_no_stop():
    path = np.concatenate([np.full(20, 100.0), np.linspace(100.0, 33.0, 80)])
    assert _run(_bars(path)).stops == []


# --------------------------------------------------------------------------
# The lie: filling at the stop price on a gap.
# --------------------------------------------------------------------------

def test_a_gap_through_the_stop_fills_at_the_open_not_the_level():
    """The detail that decides whether stops look free.

    A stop becomes a MARKET order when touched. On a bar that opens far below
    the level there is no trade at the level to be had, so the fill is the
    open. Filling at the level regardless is how a backtest invents downside
    protection on precisely the days it is supposed to be tested.
    """
    closes = np.concatenate([np.full(20, 100.0), np.full(10, 60.0)])
    opens = closes.copy()
    lows = closes.copy()
    # Bar 20 gaps from 100 straight to 60, well through a 90 stop.
    res = _run(_bars(closes, lows=lows, opens=opens),
               stop=StopPolicy(pct=0.10, lockout_days=10_000))

    assert len(res.stops) == 1
    fired = res.stops[0]
    assert fired["gapped"] is True
    assert fired["fill"] < 61.0, (
        f"filled at {fired['fill']:.2f}; a gap to 60 cannot fill near the 90 stop")


def test_an_intraday_touch_fills_at_the_stop_level():
    """The other side: dipping to the level and closing above fills AT it."""
    closes = np.full(40, 100.0)
    lows = closes.copy()
    lows[25] = 89.0          # touches through a 90 stop, closes back at 100
    res = _run(_bars(closes, lows=lows),
               stop=StopPolicy(pct=0.10, lockout_days=10_000))

    assert len(res.stops) == 1
    assert res.stops[0]["gapped"] is False
    assert 89.0 <= res.stops[0]["fill"] <= 90.1


# --------------------------------------------------------------------------
# The trailing reference must not look ahead.
# --------------------------------------------------------------------------

def test_the_trailing_stop_does_not_use_the_current_bar_high():
    """Lookahead dressed as risk management.

    On a bar that makes a new high AND a deep low, using that high to set the
    level the low is tested against assumes an intraday ordering the data does
    not contain. The reference must come from the previous bar.
    """
    closes = np.full(40, 100.0)
    highs = closes.copy()
    lows = closes.copy()
    # One wide bar: high 120, low 91. Against the prior reference of 100 the
    # stop is 90 and 91 does NOT trigger. Against this bar's own high of 120
    # the stop would be 108 and it would.
    highs[25], lows[25] = 120.0, 91.0
    bars = _bars(closes, highs=highs, lows=lows)
    res = _run(bars, stop=StopPolicy(kind="trailing", pct=0.10, lockout_days=10_000))

    wide_bar = bars.index[25]
    fired_same_bar = [f for f in res.stops if f["date"] == wide_bar]
    assert not fired_same_bar, "the stop used the same bar's high to test its own low"

    # It SHOULD fire on the next bar: the 120 high legitimately lifts the level
    # to 108, and price is back at 100. That is the stop working, one bar later
    # than a lookahead version would have it.
    assert res.stops and res.stops[0]["date"] > wide_bar
    assert res.stops[0]["level"] == pytest.approx(108.0)


def test_the_trailing_stop_ratchets_up_after_a_rally():
    closes = np.concatenate([np.linspace(100.0, 200.0, 40), np.full(20, 175.0)])
    # 175 is far above a fixed 10% stop from the 100 entry (90), and below a
    # trailing one measured from the 200 high (180).
    trailing = _run(_bars(closes), stop=StopPolicy(kind="trailing", pct=0.10,
                                                   lockout_days=10_000))
    fixed = _run(_bars(closes), stop=StopPolicy(kind="fixed", pct=0.10,
                                                lockout_days=10_000))
    assert trailing.stops, "a trailing stop should follow price up and fire at 175"
    assert fixed.stops == [], "a fixed stop from 100 must not fire at 175"


# --------------------------------------------------------------------------
# Re-entry, which is where a stop turns into pure cost.
# --------------------------------------------------------------------------

def test_the_lockout_prevents_buying_straight_back_in():
    closes = np.full(60, 100.0)
    lows = closes.copy()
    lows[20] = 85.0
    res = _run(_bars(closes, lows=lows), stop=StopPolicy(pct=0.10, lockout_days=15))
    buys_after = [t for t in res.trades
                  if t["side"] == "buy" and t["date"] > res.stops[0]["date"]]
    assert buys_after, "the position never came back after the lockout"
    gap = (buys_after[0]["date"] - res.stops[0]["date"]).days
    assert gap >= 15, f"re-entered after {gap} days despite a 15-bar lockout"


def test_without_a_lockout_a_choppy_market_churns():
    """A stop with instant re-entry is a spread-paying round trip, not insurance.

    This is why `lockout_days` exists and why it is the parameter that decides
    whether stops help: sell the dip, buy it back tomorrow, repeat.
    """
    rng = np.random.default_rng(7)
    closes = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.02, 400)))
    bars = _bars(closes, highs=closes * 1.02, lows=closes * 0.97)

    churn = _run(bars, stop=StopPolicy(pct=0.05, lockout_days=0))
    patient = _run(bars, stop=StopPolicy(pct=0.05, lockout_days=21))
    assert len(churn.trades) > len(patient.trades) * 1.5


def test_locked_out_weight_is_held_as_cash_not_redistributed():
    """Otherwise a stop quietly becomes a concentration rule."""
    class HoldBoth(PortfolioStrategy):
        name = "both"
        warmup_bars = 1

        def target_weights(self, ctx):
            return {s: 0.5 for s in ("AAA", "BBB") if s in ctx.tradeable}

    flat = np.full(60, 100.0)
    dip = flat.copy()
    lows = flat.copy()
    lows[20] = 80.0
    panel = build_panel({"AAA": _bars(dip, lows=lows), "BBB": _bars(flat)})
    res = PortfolioEngine(CostModel(), starting_equity=100.0, settle_days=0,
                          stop=StopPolicy(pct=0.10, lockout_days=20)).run(panel, HoldBoth())
    assert res.stops
    after = res.weights.loc[res.stops[0]["date"]:].iloc[1:10]
    assert after["BBB"].max() < 0.60, "the stopped-out weight was piled into BBB"


def test_stop_policy_rejects_nonsense():
    with pytest.raises(ValueError):
        StopPolicy(kind="sideways")
    with pytest.raises(ValueError):
        StopPolicy(pct=10.0)          # 10.0 is not 10%


def test_a_tight_stop_can_produce_a_DEEPER_drawdown_than_holding():
    """The finding that makes stops worth testing rather than assuming.

    On real SPY history a 5% trailing stop with immediate re-entry produced a
    73.0% maximum drawdown against 54.9% for holding, and against 55.2% for the
    index itself. The stop did not fail to prevent the drawdown - it MANUFACTURED
    eighteen points of drawdown that the market never had.

    The mechanism is reproduced here. In a market that repeatedly dips and
    recovers, each stop realises a loss and then re-enters higher. The asset
    ends up, and the account that kept stopping out of it ends down, having
    ratcheted its own equity curve below anything the price did.
    """
    # Sawtooth: dip ~6%, recover, drift upward. The asset gains overall.
    cycle = [100.0, 94.0, 101.0]
    closes = np.array([v * (1.0025 ** i) for i in range(60) for v in cycle])
    bars = _bars(closes, highs=closes * 1.005, lows=closes * 0.995)

    held = _run(bars)
    churned = _run(bars, stop=StopPolicy(kind="trailing", pct=0.05, lockout_days=0))

    px = bars["close"]
    asset_dd = abs(float((px / px.cummax() - 1).min()))
    def dd(res):
        c = res.equity_curve.dropna()
        return abs(float((c / c.cummax() - 1).min()))

    assert held.final_equity > 100.0, "the asset itself should have gained"
    assert churned.final_equity < held.final_equity, "the stop should have cost money here"
    assert dd(churned) > dd(held)
    assert dd(churned) > asset_dd, (
        f"stop drawdown {dd(churned):.1%} should exceed the asset's own "
        f"{asset_dd:.1%} - that is the whole point of the test")


def test_a_stop_wide_enough_to_be_harmless_almost_never_fires():
    """Why the 'good' settings are not evidence of anything.

    Sweeping stop width on SPY, the only settings that do not damage returns
    are the ones so wide they fire twice in thirty-four years. A Sharpe ratio
    that improves on the back of two events is two data points, not a property.
    """
    rng = np.random.default_rng(11)
    closes = 100.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.011, 3000)))
    bars = _bars(closes, highs=closes * 1.008, lows=closes * 0.992)

    tight = _run(bars, stop=StopPolicy(kind="trailing", pct=0.05, lockout_days=21))
    wide = _run(bars, stop=StopPolicy(kind="trailing", pct=0.30, lockout_days=21))
    held = _run(bars)

    assert len(tight.stops) > 5 * max(len(wide.stops), 1)
    # The wide stop is close to holding precisely because it barely acts.
    assert abs(wide.final_equity - held.final_equity) < abs(tight.final_equity - held.final_equity)


def test_readme_stop_table_still_shows_stops_losing():
    """Guards the conclusion, not the prose.

    If a later edit reorders or trims this table into something that reads as
    an endorsement, the numbers stop matching the sweep. In particular the
    tightest stop must still show a drawdown WORSE than holding, since that is
    the section's whole point.
    """
    import pathlib
    import re

    readme = pathlib.Path(__file__).resolve().parents[1] / "README.md"
    if not readme.exists():
        pytest.skip("README not present")
    section = readme.read_text().split("**Protective stops.**")
    assert len(section) == 2, "stops section missing from README"
    body = section[1]

    rows = re.findall(
        r"^\| ([^|]+?) \| \$([\d,]+) \| (\d+\.\d+)% \| (\d\.\d+) \| (\d+\.\d+)% \| (\d+) \|$",
        body, re.M)
    assert len(rows) >= 4, f"expected the stop sweep table, parsed {len(rows)} rows"

    by_name = {r[0].strip(): r for r in rows}
    none = by_name.get("none")
    assert none, "the unstopped baseline row is missing"
    none_final = float(none[1].replace(",", ""))
    none_dd = float(none[4])
    assert int(none[5]) == 0, "the baseline row should fire no stops"

    stopped = [r for r in rows if r[0].strip() != "none"]
    assert all(float(r[1].replace(",", "")) < none_final for r in stopped), \
        "a stop row now beats holding on dollars; recheck the sweep"
    assert any(float(r[4]) > none_dd for r in stopped), \
        "no stop row shows a worse drawdown than holding - the finding was edited out"
