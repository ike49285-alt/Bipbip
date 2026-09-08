"""Cost model and data archive."""
import pandas as pd
import pytest

from bipbip.core.costs import BUY, SELL, CostModel
from bipbip.data import BarStore, make_intraday_bars
from bipbip.data.sessions import restrict_to_rth


def test_slippage_always_moves_the_fill_against_you():
    c = CostModel()
    assert c.fill_price(BUY, 100.0, "SPY") > 100.0
    assert c.fill_price(SELL, 100.0, "SPY") < 100.0


def test_leveraged_etf_is_modelled_as_more_expensive_than_spy():
    """TQQQ's wider relative spread must not be modelled as SPY's."""
    c = CostModel()
    assert c.slippage_for("TQQQ") > c.slippage_for("SPY")
    assert c.slippage_for("SOXL") == c.slippage_for("default")


def test_regulatory_fees_are_charged_on_sells_only():
    """SEC and FINRA fees hit the sell side; Webull charges no commission."""
    c = CostModel()
    assert c.fees(BUY, 100, 500.0) == pytest.approx(0.0)
    assert c.fees(SELL, 100, 500.0) > 0.0


def test_finra_taf_is_capped():
    """TAF is per-share but capped per trade; an uncapped model would badly
    overstate costs on large, low-priced orders."""
    c = CostModel()
    shares, price = 10_000_000, 1.0
    sec_fee = shares * price * (c.sec_fee_per_million / 1e6)
    taf = c.fees(SELL, shares, price) - sec_fee
    assert taf == pytest.approx(c.finra_taf_max)
    assert shares * c.finra_taf_per_share > c.finra_taf_max  # cap really binds


def test_round_trip_hurdle_is_reported_in_bps():
    c = CostModel()
    spy = c.round_trip_cost_bps("SPY", 640.0, 100)
    tqqq = c.round_trip_cost_bps("TQQQ", 95.0, 100)
    assert 0 < spy < tqqq < 100


def test_store_append_is_idempotent(tmp_path):
    """Re-fetching an overlapping window must not duplicate bars."""
    store = BarStore(tmp_path)
    bars = make_intraday_bars(n_sessions=4, seed=31)

    first = store.append("SPY", bars)
    second = store.append("SPY", bars)

    assert first["rows_added"] == len(bars)
    assert second["rows_added"] == 0
    assert store.load("SPY").index.is_unique


def test_store_merges_new_history_with_old(tmp_path):
    """The point of the archive: windows accumulate rather than replace."""
    store = BarStore(tmp_path)
    bars = make_intraday_bars(n_sessions=10, seed=32)
    store.append("SPY", bars.iloc[:1000])
    store.append("SPY", bars.iloc[800:])

    stored = store.load("SPY")
    assert len(stored) == len(bars)
    assert stored.index.is_monotonic_increasing


def test_store_prefers_the_freshest_copy_of_a_bar(tmp_path):
    """A provider revising a provisional bar should correct, not duplicate."""
    store = BarStore(tmp_path)
    bars = make_intraday_bars(n_sessions=2, seed=33)
    store.append("SPY", bars)

    revised = bars.iloc[:5].copy()
    revised.loc[:, "close"] = 999.0
    store.append("SPY", revised)

    assert store.load("SPY")["close"].iloc[0] == pytest.approx(999.0)
    assert len(store.load("SPY")) == len(bars)


def test_store_rejects_frames_missing_columns(tmp_path):
    store = BarStore(tmp_path)
    bad = make_intraday_bars(n_sessions=1).drop(columns=["volume"])
    with pytest.raises(ValueError, match="missing required columns"):
        store.append("SPY", bad)


def test_extended_hours_bars_are_dropped():
    """Pre/post-market bars are thin and wide; trading them flatters a backtest."""
    bars = make_intraday_bars(n_sessions=2, seed=34)
    pre = bars.iloc[:3].copy()
    pre.index = pre.index - pd.Timedelta(hours=2)  # 07:30, pre-market
    combined = pd.concat([pre, bars]).sort_index()

    kept = restrict_to_rth(combined)
    assert len(kept) == len(bars)
    assert all(t.hour >= 9 for t in kept.index)


def test_zero_volume_padding_bars_are_dropped(tmp_path):
    """Providers pad gaps with zero-volume bars; those minutes were untradeable."""
    store = BarStore(tmp_path)
    bars = make_intraday_bars(n_sessions=2, seed=35).copy()
    bars.iloc[10:20, bars.columns.get_loc("volume")] = 0.0
    store.append("SPY", bars)
    assert len(store.load("SPY")) == len(bars) - 10
