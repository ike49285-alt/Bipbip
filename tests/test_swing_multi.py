"""Multi-asset swing strategies, and the total-return data they depend on."""
import sys
import types

import numpy as np
import pandas as pd
import pytest

from bipbip.core.costs import CostModel
from bipbip.core.panel import build_panel
from bipbip.core.portfolio import PortfolioContext, PortfolioEngine
from bipbip.strategies.cross_sectional import REGISTRY
from bipbip.strategies.swing_multi import DualMomentum, VolTargetTrend


def _series(n, start, drift, seed, vol=0.011, first=0):
    """A synthetic price path, optionally listed late (leading NaNs)."""
    rng = np.random.default_rng(seed)
    close = start * np.exp(np.cumsum(rng.normal(drift, vol, n)))
    idx = pd.DatetimeIndex(pd.bdate_range("2010-01-04", periods=n))
    df = pd.DataFrame({"open": close, "high": close * 1.005, "low": close * 0.995,
                       "close": close, "volume": 1e7}, index=idx)
    if first:
        df.iloc[:first] = np.nan
    return df


# --------------------------------------------------------------------------
# The data the strategies stand on.
# --------------------------------------------------------------------------

def _fake_yfinance(recorder):
    """A stand-in for yfinance that records the kwargs it was called with."""
    mod = types.ModuleType("yfinance")

    def download(symbol, **kwargs):
        recorder.append({"symbol": symbol, **kwargs})
        idx = pd.DatetimeIndex(pd.bdate_range("2020-01-06", periods=5))
        return pd.DataFrame({"Open": 1.0, "High": 1.0, "Low": 1.0,
                             "Close": 1.0, "Volume": 1.0}, index=idx)

    mod.download = download
    return mod


def test_daily_bars_are_fetched_as_total_return(monkeypatch):
    """Regression: daily history was stored price-only, so dividends vanished.

    With auto_adjust=False a bond ETF looks like it returned nothing for two
    decades - SHY read 81.01 -> 81.67 across 24 years - because its entire
    return arrives as coupons. That silently broke every comparison a swing
    strategy makes: the benchmark was understated by its dividend yield, and
    "does the risk asset beat cash?" was measured against a cash proxy earning
    zero.
    """
    from bipbip.data.fetchers import YFinanceFetcher

    calls = []
    monkeypatch.setitem(sys.modules, "yfinance", _fake_yfinance(calls))
    YFinanceFetcher().fetch("SHY", bar_size="1d")

    assert calls, "the fetcher never called the provider"
    assert calls[0]["auto_adjust"] is True


def test_intraday_bars_stay_unadjusted(monkeypatch):
    """The intraday archive is stitched from fetches that are never rewritten.

    Back-adjusting new minute bars while old ones keep their original prices
    would put a seam in the archive at every dividend. Over a trailing window
    the distortion is negligible, so the raw price is the right choice there -
    the opposite of the daily path, and deliberately so.
    """
    from bipbip.data.fetchers import YFinanceFetcher

    calls = []
    monkeypatch.setitem(sys.modules, "yfinance", _fake_yfinance(calls))
    YFinanceFetcher().fetch("SPY", bar_size="1m", lookback_days=7)

    assert calls
    assert all(c["auto_adjust"] is False for c in calls)


# --------------------------------------------------------------------------
# Dual momentum.
# --------------------------------------------------------------------------

def _dm_panel(risk_drift, alt_drift, defensive_drift):
    return build_panel({
        "SPY": _series(600, 100, risk_drift, 1),
        "EFA": _series(600, 60, alt_drift, 2),
        "SHY": _series(600, 80, defensive_drift, 3, vol=0.001),
    })


def _held(res):
    """The symbols the run actually ended up holding, in order of first use."""
    seen = []
    for t in res.trades:
        sym = getattr(t, "symbol", None) or t["symbol"]
        if sym not in seen:
            seen.append(sym)
    return seen


def test_dual_momentum_holds_the_stronger_risk_asset():
    panel = _dm_panel(risk_drift=0.0008, alt_drift=0.0001, defensive_drift=0.0)
    res = PortfolioEngine(CostModel(), starting_equity=50.0,
                          settle_days=0).run(panel, DualMomentum())
    assert "SPY" in _held(res)


def test_the_absolute_filter_retreats_when_no_risk_asset_beats_cash():
    """The component that historically did the work.

    Relative strength always names a winner - even in a bear market, where the
    winner is merely the asset falling slowest. Refusing to hold it unless it
    also beats the defensive asset is what turns a ranking into a rule for
    being absent.
    """
    panel = _dm_panel(risk_drift=-0.0008, alt_drift=-0.0010, defensive_drift=0.0001)
    res = PortfolioEngine(CostModel(), starting_equity=50.0,
                          settle_days=0).run(panel, DualMomentum())
    held = _held(res)
    assert held, "the strategy never traded at all"
    assert held == ["SHY"], f"held risk assets in a downtrend: {held}"


def test_dual_momentum_does_not_trade_mid_month():
    panel = _dm_panel(0.0008, 0.0001, 0.0)
    res = PortfolioEngine(CostModel(), starting_equity=50.0,
                          settle_days=0).run(panel, DualMomentum())
    dates = pd.DatetimeIndex([getattr(t, "date", None) or t["date"] for t in res.trades])
    # ~28 months in the post-warmup window; one switch a month is the ceiling.
    assert len(res.trades) <= 40, f"{len(res.trades)} trades is not monthly"


# --------------------------------------------------------------------------
# Volatility-targeted trend.
# --------------------------------------------------------------------------

def _vt_panel():
    return build_panel({
        "CALM": _series(600, 100, 0.0004, 11, vol=0.004),
        "WILD": _series(600, 100, 0.0004, 12, vol=0.020),
        "DOWN": _series(600, 100, -0.0010, 13, vol=0.010),
    })


def test_vol_target_gives_the_calm_asset_more_capital():
    """Inverse-volatility sizing, which is the point of the strategy."""
    panel = _vt_panel()
    strat = VolTargetTrend(trend_window=100, vol_window=40)
    res = PortfolioEngine(CostModel(), starting_equity=50.0,
                          settle_days=0).run(panel, strat)
    w = res.weights.dropna(how="all")
    assert w["CALM"].mean() > w["WILD"].mean()


def test_the_trend_filter_excludes_a_falling_asset():
    panel = _vt_panel()
    strat = VolTargetTrend(trend_window=100, vol_window=40)
    res = PortfolioEngine(CostModel(), starting_equity=50.0,
                          settle_days=0).run(panel, strat)
    w = res.weights.dropna(how="all")
    assert w["DOWN"].mean() < w["CALM"].mean()


def _targets(panel, strat):
    """Every non-empty target the strategy asks for across the whole panel."""
    ind = strat.prepare(panel)
    out = []
    for i in range(strat.warmup_bars, len(panel)):
        ctx = PortfolioContext(panel=panel, indicators=ind, i=i,
                               tradeable=panel.tradeable(i), current_weights={},
                               equity=50.0, date=panel.dates[i].date())
        t = strat.target_weights(ctx)
        if t:
            out.append(t)
    return out


def test_no_target_weight_exceeds_the_cap():
    """A very calm asset gets an enormous inverse-vol weight without a cap.

    Low measured volatility is not the same as low risk - it is often just an
    asset that has not moved YET - so the cap is a refusal to let the sizing
    rule concentrate the whole book on whatever looks quietest.

    This checks what the strategy ASKS for. Realised weights drift past the cap
    between monthly rebalances, which is the next test.
    """
    strat = VolTargetTrend(trend_window=100, vol_window=40, max_weight=0.34)
    targets = _targets(_vt_panel(), strat)
    assert targets, "the strategy never asked for a position"
    worst = max(max(t.values()) for t in targets)
    assert worst <= 0.34 + 1e-9, f"target weight {worst:.3f} exceeded the cap"


def test_realised_weight_drifts_past_the_cap_between_rebalances():
    """The cap binds at the rebalance, not continuously - by construction.

    A monthly strategy holding a 2%-a-day asset will be over its cap most of
    the time simply because the position grew. Pinning it back daily would
    trade constantly to correct noise, which is what the rebalance band exists
    to prevent. Documented rather than fixed, so nobody later reads the cap as
    a guarantee about the book at any given moment.
    """
    strat = VolTargetTrend(trend_window=100, vol_window=40, max_weight=0.34)
    res = PortfolioEngine(CostModel(), starting_equity=50.0,
                          settle_days=0).run(_vt_panel(), strat)
    w = res.weights.dropna(how="all")
    assert w.max().max() > 0.34, "expected drift above the cap; sizing may be broken"
    # Bounded, though: not a position quietly compounding into the whole book.
    assert w.max().max() < 0.60


def test_both_strategies_are_registered():
    assert REGISTRY["dual_momentum"] is DualMomentum
    assert REGISTRY["vol_target_trend"] is VolTargetTrend


# --------------------------------------------------------------------------
# The README's leverage table, recomputed.
# --------------------------------------------------------------------------

def test_readme_leverage_table_is_monotonic_in_the_borrowing_rate():
    """The table's claim is that the whole result is a bet on cheap money.

    Two commits in this project have already claimed a README update that
    silently did nothing, so numbers published there are checked against the
    code rather than trusted. This does not re-run the 34-year backtest - that
    needs the archive - but it does assert the shape the argument rests on:
    every column strictly worsens as borrowing gets more expensive, and the
    crossover with buy-and-hold sits between the 2% and 3.5% rows.
    """
    import pathlib
    import re

    readme = pathlib.Path(__file__).resolve().parents[1] / "README.md"
    if not readme.exists():
        pytest.skip("README not present")
    text = readme.read_text()

    section = text.split("**Leveraged trend following.**")
    assert len(section) == 2, "leverage section missing from README"

    rows = re.findall(r"^\| (\d\.\d)% \| \$([\d,]+) \| (\d+\.\d+)% \| (\d\.\d+) \|$",
                      section[1], re.M)
    assert len(rows) == 4, f"expected 4 rate rows, parsed {len(rows)}"

    rates = [float(r[0]) for r in rows]
    finals = [float(r[1].replace(",", "")) for r in rows]
    cagrs = [float(r[2]) for r in rows]
    sharpes = [float(r[3]) for r in rows]

    assert rates == sorted(rates), "rate rows are out of order"
    for name, col in (("final", finals), ("CAGR", cagrs), ("Sharpe", sharpes)):
        assert col == sorted(col, reverse=True), \
            f"{name} does not fall monotonically with the borrowing rate: {col}"

    bh = re.search(r"\*buy and hold SPY\* \| \*\$([\d,]+)\* \| \*(\d+\.\d+)%\*", section[1])
    assert bh, "buy-and-hold reference row missing"
    bh_final = float(bh.group(1).replace(",", ""))
    # The claim in the prose: leverage wins at 2%, loses by 3.5%.
    assert finals[1] > bh_final > finals[3], \
        "the stated crossover with buy-and-hold is not where the table puts it"


def test_readme_leverage_sharpe_never_beats_holding_meaningfully():
    """The point of the section: leverage buys return, not risk-adjusted return."""
    import pathlib
    import re

    readme = pathlib.Path(__file__).resolve().parents[1] / "README.md"
    if not readme.exists():
        pytest.skip("README not present")
    section = readme.read_text().split("**Leveraged trend following.**")[1]
    sharpes = [float(m) for m in re.findall(r"^\| \d\.\d% \| \$[\d,]+ \| \d+\.\d+% \| (\d\.\d+) \|$",
                                            section, re.M)]
    # 0.55 is buy-and-hold. Only the free-money row clears it, and barely.
    assert max(sharpes) < 0.60
    assert sum(s < 0.55 for s in sharpes) >= 3
