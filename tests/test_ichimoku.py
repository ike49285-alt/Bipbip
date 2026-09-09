"""Ichimoku and the full stochastic, with emphasis on the time shifts.

Ichimoku is drawn with two displacements and a backtest can get either one
backwards. The cloud is shifted FORWARD, which is safe - the value over bar i
was computed at bar i-26. The lagging span is shifted BACKWARD, and reading it
at row i hands the strategy a price 26 bars in the future.
"""
import numpy as np
import pandas as pd
import pytest

from bipbip.core import indicators as ind


def _bars(n=400, seed=3, drift=0.0004):
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(drift, 0.012, n)))
    idx = pd.DatetimeIndex(pd.bdate_range("2015-01-05", periods=n))
    return pd.DataFrame({"open": close, "high": close * 1.008, "low": close * 0.992,
                         "close": close, "volume": 1e7}, index=idx)


# --------------------------------------------------------------------------
# Causality: the property that makes the rest meaningful.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("col", ["tenkan", "kijun", "senkou_a", "senkou_b",
                                 "cloud_top", "cloud_bottom", "above_cloud",
                                 "below_cloud", "chikou_above", "cloud_bull"])
def test_every_ichimoku_column_is_causal(col):
    """Truncating the future must not change any past value."""
    bars = _bars()
    cut = 300
    full = ind.ichimoku(bars)[col].iloc[:cut]
    truncated = ind.ichimoku(bars.iloc[:cut])[col]
    pd.testing.assert_series_equal(full, truncated, check_exact=False, rtol=1e-10)


@pytest.mark.parametrize("col", ["stoch_k", "stoch_d", "stoch_raw_k"])
def test_full_stochastic_is_causal(col):
    bars = _bars()
    cut = 300
    full = ind.full_stochastic(bars)[col].iloc[:cut]
    truncated = ind.full_stochastic(bars.iloc[:cut])[col]
    pd.testing.assert_series_equal(full, truncated, check_exact=False, rtol=1e-10)


def test_a_future_price_spike_cannot_move_an_earlier_ichimoku_reading():
    """The direct form of the same check, aimed at the lagging span.

    `chikou_above` is the one column derived from a line the chart draws in the
    past. Implemented as `close.shift(-26)` it would read a price 26 bars ahead
    and this spike would reach backwards.
    """
    bars = _bars()
    cut = 300
    tampered = bars.copy()
    tampered.iloc[cut:] *= 3.0

    a = ind.ichimoku(bars).iloc[:cut]
    b = ind.ichimoku(tampered).iloc[:cut]
    pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-10)


def test_chikou_above_compares_today_against_the_bar_26_back():
    """State the definition, so a later 'fix' to shift(-26) fails loudly."""
    bars = _bars()
    ich = ind.ichimoku(bars, displacement=26)
    expected = (bars["close"] > bars["close"].shift(26))
    # Undefined for the first 26 bars, where there is no bar to look back at.
    assert ich["chikou_above"].iloc[:26].isna().all()
    pd.testing.assert_series_equal(
        ich["chikou_above"].iloc[26:].astype(bool), expected.iloc[26:],
        check_names=False)


# --------------------------------------------------------------------------
# The cloud's forward shift means what it looks like.
# --------------------------------------------------------------------------

def test_the_cloud_over_a_bar_was_computed_a_displacement_earlier():
    bars = _bars()
    ich = ind.ichimoku(bars, tenkan=9, kijun=26, displacement=26)
    raw_a = (ich["tenkan"] + ich["kijun"]) / 2.0
    # Row i of senkou_a is raw_a from 26 rows earlier.
    assert ich["senkou_a"].iloc[100] == pytest.approx(raw_a.iloc[74])


def test_cloud_top_is_never_below_cloud_bottom():
    ich = ind.ichimoku(_bars())
    both = ich[["cloud_top", "cloud_bottom"]].dropna()
    assert (both["cloud_top"] >= both["cloud_bottom"]).all()


def test_above_and_below_the_cloud_are_mutually_exclusive():
    ich = ind.ichimoku(_bars()).dropna(subset=["cloud_top", "cloud_bottom"])
    assert not (ich["above_cloud"] & ich["below_cloud"]).any()


@pytest.mark.parametrize("col", ["above_cloud", "below_cloud", "chikou_above",
                                 "cloud_bull"])
def test_a_flag_is_missing_where_its_inputs_are_missing(col):
    """Regression: `a > b` with a NaN operand is False, not unknown.

    Senkou B needs 52 bars plus a 26-bar displacement, so the cloud is
    undefined for 78 bars of otherwise-valid price history. Returning False
    there asserts "price is not above the cloud" about a cloud that does not
    exist yet - the same bug that once had a market filter reading as
    "below its average" for thirty years it had no data for.
    """
    ich = ind.ichimoku(_bars())
    assert ich[col].dtype == "boolean"
    assert ich[col].isna().any(), f"{col} never reports missing during warmup"
    # And where it IS defined it must be a real bool, not NA leaking through.
    assert ich[col].dropna().isin([True, False]).all()


def test_the_cloud_flags_go_missing_for_the_full_warmup():
    ich = ind.ichimoku(_bars(), tenkan=9, kijun=26, senkou_b=52, displacement=26)
    # 52 bars to form Senkou B, then 26 more before it is drawn.
    assert ich["cloud_bull"].iloc[:52 + 26 - 1].isna().all()
    assert ich["cloud_bull"].iloc[52 + 26:].notna().all()


def test_a_steady_uptrend_puts_price_above_the_cloud():
    n = 300
    close = 100.0 * (1.003 ** np.arange(n))
    idx = pd.DatetimeIndex(pd.bdate_range("2015-01-05", periods=n))
    bars = pd.DataFrame({"open": close, "high": close, "low": close,
                         "close": close, "volume": 1e7}, index=idx)
    ich = ind.ichimoku(bars).dropna(subset=["cloud_top", "senkou_b"])
    assert ich["above_cloud"].mean() > 0.95
    assert ich["cloud_bull"].mean() > 0.95


# --------------------------------------------------------------------------
# Stochastic definition and the flat-range case.
# --------------------------------------------------------------------------

def test_stochastic_is_bounded_and_reads_high_at_the_top_of_the_range():
    bars = _bars()
    st = ind.full_stochastic(bars).dropna()
    assert st["stoch_k"].between(0, 100).all()
    assert st["stoch_d"].between(0, 100).all()

    n = 60
    close = np.linspace(100.0, 200.0, n)      # closes at its highest ever
    idx = pd.DatetimeIndex(pd.bdate_range("2015-01-05", periods=n))
    rising = pd.DataFrame({"open": close, "high": close, "low": close,
                           "close": close, "volume": 1e7}, index=idx)
    assert ind.full_stochastic(rising)["stoch_k"].dropna().iloc[-1] > 95


def test_a_flat_range_yields_no_reading_rather_than_a_neutral_one():
    """0/0 is missing information, not a reading of 50.

    Filling it with the midpoint invents a 'perfectly neutral' oscillator value
    out of a tape that did not move, and a crossover rule downstream will
    happily trade it.
    """
    n = 60
    close = np.full(n, 100.0)
    idx = pd.DatetimeIndex(pd.bdate_range("2015-01-05", periods=n))
    flat = pd.DataFrame({"open": close, "high": close, "low": close,
                         "close": close, "volume": 1e7}, index=idx)
    st = ind.full_stochastic(flat)
    assert st["stoch_k"].isna().all()
    assert not (st["stoch_k"] == 50).any()


def test_more_smoothing_is_less_jumpy():
    bars = _bars()
    fast = ind.full_stochastic(bars, 14, 1, 3)["stoch_k"].dropna()
    full = ind.full_stochastic(bars, 14, 3, 3)["stoch_k"].dropna()
    assert full.diff().abs().mean() < fast.diff().abs().mean()


def test_fast_stochastic_k_is_the_raw_oscillator():
    bars = _bars()
    st = ind.full_stochastic(bars, 14, 1, 3)
    pd.testing.assert_series_equal(st["stoch_k"], st["stoch_raw_k"],
                                   check_names=False)


# --------------------------------------------------------------------------
# The strategies built on them.
# --------------------------------------------------------------------------

from bipbip.core.costs import CostModel                        # noqa: E402
from bipbip.core.panel import build_panel                      # noqa: E402
from bipbip.core.portfolio import PortfolioEngine              # noqa: E402
from bipbip.strategies.cross_sectional import REGISTRY         # noqa: E402
from bipbip.strategies.ichimoku import (                       # noqa: E402
    IchimokuCloud, IchimokuStochastic)


def _panel(n=600, seed=5, drift=0.0004):
    return build_panel({"AAA": _bars(n=n, seed=seed, drift=drift)})


def _run(strat, panel=None):
    return PortfolioEngine(CostModel(), starting_equity=50.0,
                           settle_days=0).run(panel or _panel(), strat)


def test_more_confirmations_means_less_time_in_the_market():
    """The mechanism behind the sweep: the votes are not independent.

    All four Ichimoku conditions measure trend, so requiring more of them does
    not gather more evidence - it just narrows the window. Exposure must fall
    monotonically as the threshold rises.
    """
    panel = _panel()
    exposure = []
    for n in (1, 2, 3, 4):
        res = _run(IchimokuCloud(confirmations=n, symbols=["AAA"]), panel)
        exposure.append((res.weights.sum(axis=1) > 0.01).mean())
    assert exposure == sorted(exposure, reverse=True), exposure


def test_a_downtrend_is_sat_out_entirely():
    panel = _panel(drift=-0.0012, seed=17)
    res = _run(IchimokuCloud(confirmations=4, symbols=["AAA"]), panel)
    assert (res.weights.sum(axis=1) > 0.01).mean() < 0.15


def test_an_uptrend_is_mostly_held():
    panel = _panel(drift=0.0012, seed=19)
    res = _run(IchimokuCloud(confirmations=1, symbols=["AAA"]), panel)
    assert (res.weights.sum(axis=1) > 0.01).mean() > 0.60


def test_a_stochastic_entry_gate_at_100_is_a_no_op():
    """Sanity check on the filter: admitting everything must change nothing.

    Without this, a bug in the gate looks like an effect rather than a bug.
    """
    panel = _panel()
    plain = _run(IchimokuCloud(confirmations=4, symbols=["AAA"]), panel)
    ungated = _run(IchimokuStochastic(confirmations=4, symbols=["AAA"],
                                      entry_below=100.0), panel)
    assert ungated.final_equity == pytest.approx(plain.final_equity, rel=1e-9)


def test_a_tighter_stochastic_gate_reduces_exposure():
    panel = _panel()
    loose = _run(IchimokuStochastic(confirmations=4, symbols=["AAA"],
                                    entry_below=90.0), panel)
    tight = _run(IchimokuStochastic(confirmations=4, symbols=["AAA"],
                                    entry_below=40.0), panel)
    assert ((tight.weights.sum(axis=1) > 0.01).mean()
            < (loose.weights.sum(axis=1) > 0.01).mean())


def test_the_strategy_never_holds_before_the_cloud_exists():
    """Warmup is 78 bars of cloud plus slack; trading earlier would be trading
    on a comparison against a NaN."""
    panel = _panel()
    res = _run(IchimokuCloud(confirmations=1, symbols=["AAA"]), panel)
    early = res.weights.iloc[:52 + 26]
    assert early.abs().max().max() < 1e-9


def test_confirmations_are_bounded():
    with pytest.raises(ValueError):
        IchimokuCloud(confirmations=0)
    with pytest.raises(ValueError):
        IchimokuCloud(confirmations=5)


def test_both_are_registered():
    assert REGISTRY["ichimoku"] is IchimokuCloud
    assert REGISTRY["ichimoku_stoch"] is IchimokuStochastic


def test_readme_ichimoku_table_stays_monotonic():
    """The section's claim is an ordering, so the ordering is what is checked."""
    import pathlib
    import re

    readme = pathlib.Path(__file__).resolve().parents[1] / "README.md"
    if not readme.exists():
        pytest.skip("README not present")
    section = readme.read_text().split("**Ichimoku cloud, and the full stochastic.**")
    assert len(section) == 2, "ichimoku section missing from README"
    # Bound at the next section; splitting on the heading alone leaves every
    # later section in scope and their tables get parsed as this one's rows.
    body = section[1].split("\n\n**")[0]

    rows = re.findall(
        r"^\| ([^|]+?) \| \$([\d,]+) \| (\d+\.\d+)% \| (\d\.\d+) \| (\d+\.\d+)% \| (\d+)% \|$",
        body, re.M)
    assert len(rows) == 5, f"expected 5 rows in the confirmation table, got {len(rows)}"

    exposure = [int(r[5]) for r in rows]
    assert exposure == sorted(exposure, reverse=True), \
        "exposure should fall as confirmations are added"
    # Confirmations 2..4 must each be worse than holding on BOTH counts.
    hold_final, hold_sharpe = float(rows[0][1].replace(",", "")), float(rows[0][3])
    for r in rows[2:]:
        assert float(r[1].replace(",", "")) < hold_final
        assert float(r[3]) < hold_sharpe

    decay = re.search(r"\| basis points per day \|(.+)\|$", body, re.M)
    assert decay, "the stochastic decay table is missing"
    bps = [float(c.strip().lstrip("+")) for c in decay.group(1).split("|")]
    assert bps == sorted(bps, reverse=True), f"the decay is no longer monotonic: {bps}"
