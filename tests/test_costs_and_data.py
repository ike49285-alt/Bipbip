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


def test_git_diff_does_not_see_untracked_files(tmp_path):
    """Documents the trap that made the collector discard its own archive.

    `git diff --quiet -- <path>` reports NO changes for brand-new untracked
    files, so a "commit if changed" guard written that way silently skips the
    very first run - exactly when every file is new. Staging first and checking
    the index catches new files and modifications alike.
    """
    import subprocess

    def git(*args):
        return subprocess.run(["git", *args], cwd=tmp_path, capture_output=True, text=True)

    git("init", "-q")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "t")
    (tmp_path / "seed.txt").write_text("seed")
    git("add", "seed.txt")
    git("commit", "-qm", "seed")

    bars_dir = tmp_path / "data" / "bars"
    bars_dir.mkdir(parents=True)
    (bars_dir / "SPY_1m.parquet").write_bytes(b"not really parquet, but untracked")

    # The buggy check: sees nothing, so the archive would be thrown away.
    assert git("diff", "--quiet", "--", "data/bars").returncode == 0

    # The fix: stage, then inspect the index.
    git("add", "data/bars")
    assert git("diff", "--cached", "--quiet", "--", "data/bars").returncode != 0


def test_one_minute_lookback_stays_inside_yahoo_limit():
    """Requesting exactly 30 days back is rejected and costs a week of history."""
    from bipbip.data.fetchers import YF_MAX_DAYS

    assert YF_MAX_DAYS["1m"] < 30


def test_daily_bars_survive_the_rth_filter(tmp_path):
    """Regression: the RTH window is meaningless for daily bars.

    Daily bars are stamped at midnight, so applying the 09:30-16:00 intraday
    filter to them discards every row. The store reported zero bars and the
    collector's --require-data guard would have failed the run, for data that
    was fetched correctly.
    """
    import pandas as pd
    from bipbip.data.sessions import EXCHANGE_TZ

    idx = pd.date_range("2024-01-02", periods=300, freq="B", tz=EXCHANGE_TZ)
    daily = pd.DataFrame({"open": 400.0, "high": 404.0, "low": 398.0,
                          "close": 402.0, "volume": 7e7}, index=idx)

    store = BarStore(tmp_path)
    info = store.append("SPY", daily, bar_size="1d")
    assert info["rows_after"] == len(daily)
    assert store.coverage("SPY", "1d")["bars"] == len(daily)


def test_intraday_and_daily_archives_are_kept_separate(tmp_path):
    """Different bar sizes must not overwrite one another."""
    import pandas as pd
    from bipbip.data.sessions import EXCHANGE_TZ

    store = BarStore(tmp_path)
    store.append("SPY", make_intraday_bars(n_sessions=3, seed=81), bar_size="1m")
    idx = pd.date_range("2024-01-02", periods=100, freq="B", tz=EXCHANGE_TZ)
    store.append("SPY", pd.DataFrame({"open": 1.0, "high": 2.0, "low": 0.5,
                                      "close": 1.5, "volume": 10.0}, index=idx),
                 bar_size="1d")

    assert store.coverage("SPY", "1m")["bars"] > 1000
    assert store.coverage("SPY", "1d")["bars"] == 100
    assert store.path_for("SPY", "1m") != store.path_for("SPY", "1d")


def test_is_intraday_classification():
    from bipbip.data.sessions import is_intraday

    for size in ("1m", "5m", "30m", "1h"):
        assert is_intraday(size)
    for size in ("1d", "1wk", "1mo"):
        assert not is_intraday(size)


def test_daily_bars_keep_their_own_session_date(tmp_path):
    """Regression for silent lookahead bias.

    Providers stamp daily bars naive, and the naive value already IS the
    session date. Localising to UTC and converting to exchange time moved every
    bar onto the previous calendar day: the bar labelled 2026-09-01 held the
    2026-09-02 session. Joining that to intraday bars by date pairs each
    session with TOMORROW's daily bar - lookahead bias that is invisible and
    flatters every result built on it.
    """
    import pandas as pd

    dates = pd.to_datetime(["2026-09-01", "2026-09-02", "2026-09-03"])
    daily = pd.DataFrame({"open": [1.0, 2.0, 3.0], "high": [1.5, 2.5, 3.5],
                          "low": [0.5, 1.5, 2.5], "close": [1.2, 2.2, 3.2],
                          "volume": [10.0, 20.0, 30.0]}, index=dates)

    store = BarStore(tmp_path)
    store.append("SPY", daily, bar_size="1d")
    stored = store.load("SPY", "1d")

    assert [t.date().isoformat() for t in stored.index] == [
        "2026-09-01", "2026-09-02", "2026-09-03"]
    # The close on each date must be the one the provider gave for that date.
    assert float(stored["close"].iloc[0]) == pytest.approx(1.2)
    assert float(stored["close"].iloc[2]) == pytest.approx(3.2)


def test_daily_and_intraday_agree_on_the_same_date(tmp_path):
    """The join that the date shift would have silently corrupted."""
    import pandas as pd

    intraday = make_intraday_bars(n_sessions=5, seed=91)
    sessions = sorted({t.date() for t in intraday.index})
    closes = [float(intraday[[t.date() == d for t in intraday.index]]["close"].iloc[-1])
              for d in sessions]
    daily = pd.DataFrame(
        {"open": closes, "high": closes, "low": closes, "close": closes,
         "volume": [1e6] * len(closes)},
        index=pd.to_datetime([str(d) for d in sessions]),
    )

    store = BarStore(tmp_path)
    store.append("SPY", intraday, bar_size="1m")
    store.append("SPY", daily, bar_size="1d")

    d = store.load("SPY", "1d")
    m = store.load("SPY", "1m")
    for day, expected in zip(sessions, closes):
        row = d[[t.date() == day for t in d.index]]
        assert len(row) == 1, f"no daily bar on {day}"
        assert float(row["close"].iloc[0]) == pytest.approx(expected)

        # The intraday half of "agree". Comparing the daily bar only against
        # `closes` re-checks a list this test built before storing anything;
        # the corruption being guarded against is a date shift on the JOIN, so
        # the last intraday bar of the session has to be read back out of the
        # store and matched to the daily bar sharing its date.
        session = m[[t.date() == day for t in m.index]]
        assert not session.empty, f"no intraday bars on {day}"
        assert float(session["close"].iloc[-1]) == pytest.approx(
            float(row["close"].iloc[0])), (
            f"daily close and last intraday close disagree on {day}")


def test_store_rejects_a_bar_that_has_volume_but_no_prices(tmp_path):
    """A counted-but-unsettled session must never enter the archive.

    A fetch once wrote a volume-only row into all 540 daily files, which made
    QQQ's total return NaN.
    """
    store = BarStore(tmp_path)
    bars = make_intraday_bars(n_sessions=2, seed=34)
    bars.iloc[3, bars.columns.get_indexer(["open", "high", "low", "close"])] = float("nan")

    store.append("SPY", bars)

    stored = store.load("SPY")
    assert len(stored) == len(bars) - 1
    assert stored[["open", "high", "low", "close"]].notna().all().all()


def test_an_unpriced_bar_cannot_overwrite_a_good_one(tmp_path):
    """Last-writer-wins makes this destructive, not merely useless."""
    store = BarStore(tmp_path)
    bars = make_intraday_bars(n_sessions=2, seed=35)
    store.append("SPY", bars)
    good = store.load("SPY")["close"].iloc[3]

    unpriced = bars.iloc[3:4].copy()
    unpriced.loc[:, ["open", "high", "low", "close"]] = float("nan")
    store.append("SPY", unpriced)

    assert store.load("SPY")["close"].iloc[3] == pytest.approx(good)


# ---------------------------------------------------------------------------
# The cost model's arithmetic. Mutation testing left 6 of 22 alive here, and
# the live one that mattered was `proceeds = shares * price` becoming
# `shares / price`: the SEC fee's BASE could be wrong by orders of magnitude
# with the whole suite green. Cost is subtracted from every result this repo
# produces, so an error here moves every number at once and in one direction.
# ---------------------------------------------------------------------------

def test_the_sec_fee_is_charged_on_notional_not_on_some_other_combination():
    """27.80 per million of PROCEEDS, on sells. At 100 shares of $500 that is
    $50,000 of notional and $1.39 of fee - computed here from the published
    rate rather than from the function under test."""
    c = CostModel(commission_per_trade=0.0, finra_taf_per_share=0.0)
    fee = c.fees("sell", 100.0, 500.0)
    assert fee == pytest.approx(50_000.0 * 27.80 / 1e6)
    assert fee == pytest.approx(1.39)


def test_the_sec_fee_scales_with_both_shares_and_price():
    """`shares / price` happens to be close to `shares * price` for no inputs
    at all, but a single test at one size cannot tell them apart if the number
    is never checked against an independent calculation. Doubling each input
    independently must double the fee."""
    c = CostModel(commission_per_trade=0.0, finra_taf_per_share=0.0)
    base = c.fees("sell", 100.0, 500.0)
    assert c.fees("sell", 200.0, 500.0) == pytest.approx(2 * base)
    assert c.fees("sell", 100.0, 1000.0) == pytest.approx(2 * base)


def test_the_finra_fee_is_per_share_and_capped():
    """Per SHARE, not per dollar - so it is unchanged by price - and capped, or
    a large order would be charged without limit."""
    c = CostModel(commission_per_trade=0.0, sec_fee_per_million=0.0,
                  finra_taf_per_share=0.000166, finra_taf_max=8.30)
    assert c.fees("sell", 1000.0, 10.0) == pytest.approx(0.166)
    assert c.fees("sell", 1000.0, 900.0) == pytest.approx(0.166)   # price-blind
    assert c.fees("sell", 10_000_000.0, 10.0) == pytest.approx(8.30)  # capped


def test_regulatory_fees_are_charged_on_sells_only():
    """The asymmetry is the point: charging buys too would roughly double the
    modelled regulatory cost of every round trip."""
    c = CostModel(commission_per_trade=0.0)
    assert c.fees("buy", 100.0, 500.0) == 0.0
    assert c.fees("sell", 100.0, 500.0) > 0.0


def test_commission_is_charged_on_both_sides():
    c = CostModel(commission_per_trade=1.25, sec_fee_per_million=0.0,
                  finra_taf_per_share=0.0)
    assert c.fees("buy", 10.0, 100.0) == pytest.approx(1.25)
    assert c.fees("sell", 10.0, 100.0) == pytest.approx(1.25)


def test_a_round_trip_counts_both_legs_of_slippage_and_both_fees():
    """Recomputed from the parts rather than trusting the aggregate, because
    the aggregate is the number every result in this repo is charged."""
    c = CostModel(commission_per_trade=0.50)
    px, sh = 500.0, 100.0
    buy = c.fill_price("buy", px, "SPY")
    sell = c.fill_price("sell", px, "SPY")
    expected = ((buy - sell) * sh
                + c.fees("buy", sh, buy) + c.fees("sell", sh, sell)) / (px * sh) * 1e4
    assert c.round_trip_cost_bps("SPY", px, sh) == pytest.approx(expected, rel=1e-12)


def test_the_round_trip_is_never_free_and_grows_with_the_spread():
    c = CostModel()
    spy = c.round_trip_cost_bps("SPY")
    tqqq = c.round_trip_cost_bps("TQQQ")
    other = c.round_trip_cost_bps("SOMETHING_UNLISTED")
    assert 0 < spy < tqqq < other      # 1.0 < 2.5 < 3.0 bps of slippage


def test_a_zero_or_negative_notional_costs_nothing_rather_than_dividing_by_it():
    c = CostModel()
    assert c.round_trip_cost_bps("SPY", 0.0, 100.0) == 0.0
    assert c.round_trip_cost_bps("SPY", 500.0, 0.0) == 0.0
    assert c.round_trip_cost_bps("SPY", -10.0, 100.0) == 0.0


def test_a_fill_is_always_worse_than_the_reference_price():
    """The direction of slippage. Inverting it turns a cost into a subsidy, and
    a backtest would simply look better."""
    c = CostModel()
    assert c.fill_price("buy", 100.0, "SPY") > 100.0
    assert c.fill_price("sell", 100.0, "SPY") < 100.0


def test_an_unknown_side_is_refused_rather_than_treated_as_a_buy():
    with pytest.raises(ValueError, match="side must be"):
        CostModel().fill_price("hold", 100.0, "SPY")


def test_slippage_lookup_is_case_insensitive_and_falls_back_to_default():
    c = CostModel(slippage_bps={"SPY": 1.0, "default": 7.0})
    assert c.slippage_for("spy") == 1.0
    assert c.slippage_for("SPY") == 1.0
    assert c.slippage_for("ZZZZ") == 7.0


def test_from_config_reads_a_nested_costs_block_or_a_flat_one():
    """Both shapes are accepted, so both are pinned: `{"costs": {...}}` as the
    config file supplies it, and a bare dict as callers pass it."""
    nested = CostModel.from_config({"costs": {"commission_per_trade": 2.0}})
    flat = CostModel.from_config({"commission_per_trade": 2.0})
    assert nested.commission_per_trade == 2.0
    assert flat.commission_per_trade == 2.0


def test_from_config_on_an_empty_or_missing_block_uses_the_documented_defaults():
    """A config with no costs section must not silently produce a zero-cost
    model, which would make every backtest look profitable."""
    for cfg in ({}, {"costs": None}, {"costs": {}}):
        c = CostModel.from_config(cfg)
        assert c.sec_fee_per_million == 27.80
        assert c.finra_taf_per_share == 0.000166
        assert c.slippage_for("TQQQ") == 2.5
        assert c.round_trip_cost_bps("SPY") > 0


def test_a_sub_dollar_notional_still_costs_something():
    """The guard is `notional <= 0`, not `<= 1`, and the distinction is real
    here: the account this project is sized for holds $2.10, so a fractional
    order worth less than a dollar is an ordinary case rather than a corner.
    Treating it as free would make the smallest trades look the cheapest."""
    c = CostModel()
    assert c.round_trip_cost_bps("SPY", price=0.50, shares=1.0) > 0
    assert c.round_trip_cost_bps("SPY", price=1.00, shares=0.5) > 0


# ---------------------------------------------------------------------------
# The cost model's arithmetic. Mutation testing left 6 of 22 alive here, and
# the live one that mattered was `proceeds = shares * price` becoming
# `shares / price`: the SEC fee's BASE could be wrong by orders of magnitude
# with the whole suite green. Cost is subtracted from every result this repo
# produces, so an error here moves every number at once and in one direction.
# ---------------------------------------------------------------------------

def test_the_sec_fee_is_charged_on_notional_not_on_some_other_combination():
    """27.80 per million of PROCEEDS, on sells. At 100 shares of $500 that is
    $50,000 of notional and $1.39 of fee - computed here from the published
    rate rather than from the function under test."""
    c = CostModel(commission_per_trade=0.0, finra_taf_per_share=0.0)
    fee = c.fees("sell", 100.0, 500.0)
    assert fee == pytest.approx(50_000.0 * 27.80 / 1e6)
    assert fee == pytest.approx(1.39)


def test_the_sec_fee_scales_with_both_shares_and_price():
    """`shares / price` happens to be close to `shares * price` for no inputs
    at all, but a single test at one size cannot tell them apart if the number
    is never checked against an independent calculation. Doubling each input
    independently must double the fee."""
    c = CostModel(commission_per_trade=0.0, finra_taf_per_share=0.0)
    base = c.fees("sell", 100.0, 500.0)
    assert c.fees("sell", 200.0, 500.0) == pytest.approx(2 * base)
    assert c.fees("sell", 100.0, 1000.0) == pytest.approx(2 * base)


def test_the_finra_fee_is_per_share_and_capped():
    """Per SHARE, not per dollar - so it is unchanged by price - and capped, or
    a large order would be charged without limit."""
    c = CostModel(commission_per_trade=0.0, sec_fee_per_million=0.0,
                  finra_taf_per_share=0.000166, finra_taf_max=8.30)
    assert c.fees("sell", 1000.0, 10.0) == pytest.approx(0.166)
    assert c.fees("sell", 1000.0, 900.0) == pytest.approx(0.166)   # price-blind
    assert c.fees("sell", 10_000_000.0, 10.0) == pytest.approx(8.30)  # capped


def test_regulatory_fees_are_charged_on_sells_only():
    """The asymmetry is the point: charging buys too would roughly double the
    modelled regulatory cost of every round trip."""
    c = CostModel(commission_per_trade=0.0)
    assert c.fees("buy", 100.0, 500.0) == 0.0
    assert c.fees("sell", 100.0, 500.0) > 0.0


def test_commission_is_charged_on_both_sides():
    c = CostModel(commission_per_trade=1.25, sec_fee_per_million=0.0,
                  finra_taf_per_share=0.0)
    assert c.fees("buy", 10.0, 100.0) == pytest.approx(1.25)
    assert c.fees("sell", 10.0, 100.0) == pytest.approx(1.25)


def test_a_round_trip_counts_both_legs_of_slippage_and_both_fees():
    """Recomputed from the parts rather than trusting the aggregate, because
    the aggregate is the number every result in this repo is charged."""
    c = CostModel(commission_per_trade=0.50)
    px, sh = 500.0, 100.0
    buy = c.fill_price("buy", px, "SPY")
    sell = c.fill_price("sell", px, "SPY")
    expected = ((buy - sell) * sh
                + c.fees("buy", sh, buy) + c.fees("sell", sh, sell)) / (px * sh) * 1e4
    assert c.round_trip_cost_bps("SPY", px, sh) == pytest.approx(expected, rel=1e-12)


def test_the_round_trip_is_never_free_and_grows_with_the_spread():
    c = CostModel()
    spy = c.round_trip_cost_bps("SPY")
    tqqq = c.round_trip_cost_bps("TQQQ")
    other = c.round_trip_cost_bps("SOMETHING_UNLISTED")
    assert 0 < spy < tqqq < other      # 1.0 < 2.5 < 3.0 bps of slippage


def test_a_zero_or_negative_notional_costs_nothing_rather_than_dividing_by_it():
    c = CostModel()
    assert c.round_trip_cost_bps("SPY", 0.0, 100.0) == 0.0
    assert c.round_trip_cost_bps("SPY", 500.0, 0.0) == 0.0
    assert c.round_trip_cost_bps("SPY", -10.0, 100.0) == 0.0


def test_a_fill_is_always_worse_than_the_reference_price():
    """The direction of slippage. Inverting it turns a cost into a subsidy, and
    a backtest would simply look better."""
    c = CostModel()
    assert c.fill_price("buy", 100.0, "SPY") > 100.0
    assert c.fill_price("sell", 100.0, "SPY") < 100.0


def test_an_unknown_side_is_refused_rather_than_treated_as_a_buy():
    with pytest.raises(ValueError, match="side must be"):
        CostModel().fill_price("hold", 100.0, "SPY")


def test_slippage_lookup_is_case_insensitive_and_falls_back_to_default():
    c = CostModel(slippage_bps={"SPY": 1.0, "default": 7.0})
    assert c.slippage_for("spy") == 1.0
    assert c.slippage_for("SPY") == 1.0
    assert c.slippage_for("ZZZZ") == 7.0


def test_from_config_reads_a_nested_costs_block_or_a_flat_one():
    """Both shapes are accepted, so both are pinned: `{"costs": {...}}` as the
    config file supplies it, and a bare dict as callers pass it."""
    nested = CostModel.from_config({"costs": {"commission_per_trade": 2.0}})
    flat = CostModel.from_config({"commission_per_trade": 2.0})
    assert nested.commission_per_trade == 2.0
    assert flat.commission_per_trade == 2.0


def test_from_config_on_an_empty_or_missing_block_uses_the_documented_defaults():
    """A config with no costs section must not silently produce a zero-cost
    model, which would make every backtest look profitable."""
    for cfg in ({}, {"costs": None}, {"costs": {}}):
        c = CostModel.from_config(cfg)
        assert c.sec_fee_per_million == 27.80
        assert c.finra_taf_per_share == 0.000166
        assert c.slippage_for("TQQQ") == 2.5
        assert c.round_trip_cost_bps("SPY") > 0
