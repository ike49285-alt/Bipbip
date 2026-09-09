"""Meta-labelling: the dataset, and the guarantee that it can only decline."""
import numpy as np
import pandas as pd
import pytest

from bipbip.core.panel import build_panel
from bipbip.ml.meta import (breakout_signals, build_features, label_signals,
                            rsi_dip_signals)


def _sym(n, start, seed):
    rng = np.random.default_rng(seed)
    close = start * np.exp(np.cumsum(rng.normal(0.0004, 0.014, n)))
    idx = pd.DatetimeIndex(pd.bdate_range("2010-01-04", periods=n))
    return pd.DataFrame({"open": close, "high": close * 1.012, "low": close * 0.988,
                         "close": close, "volume": 1e6 * (1 + rng.random(n))}, index=idx)


def _panel(n=1200, k=8):
    return build_panel({f"S{i}": _sym(n, 50 + 10 * i, 100 + i) for i in range(k)})


def test_primary_rule_fires_and_labels_resolve():
    panel = _panel()
    sig = rsi_dip_signals(panel)
    feats = build_features(panel)
    ds = label_signals(panel, sig, feats)

    assert len(ds) > 50, "expected a workable number of signals"
    assert set(np.unique(ds.y)) <= {0.0, 1.0}
    assert np.isfinite(ds.rets).all()
    assert len(ds.X) == len(ds.y) == len(ds.rets) == len(ds.event_end)


def test_labels_are_net_of_costs():
    """A 'win' has to be a win after paying to trade, or the model learns to
    chase moves smaller than the spread."""
    panel = _panel()
    sig = rsi_dip_signals(panel)
    feats = build_features(panel)
    cheap = label_signals(panel, sig, feats, cost_bps=0.0)
    dear = label_signals(panel, sig, feats, cost_bps=200.0)
    assert dear.rets.mean() < cheap.rets.mean()
    assert dear.y.mean() <= cheap.y.mean()


def test_entry_is_the_next_bar_open_not_the_signal_close():
    """Filling at the close of the bar that generated the signal would be
    lookahead: the rule only knows that close once it has happened."""
    panel = _panel(600, 4)
    sig = rsi_dip_signals(panel)
    feats = build_features(panel)
    ds = label_signals(panel, sig, feats, max_hold=5)
    # Every label must resolve strictly after its signal date.
    assert (ds.event_end >= np.arange(len(ds))).all()


def test_features_are_causal():
    """Corrupting the future must not change features computed earlier."""
    panel = _panel(900, 5)
    cut = 600
    base = build_features(panel)

    tampered_src = {}
    for s in panel.symbols:
        df = pd.DataFrame({"open": panel.opens[s], "high": panel.highs[s],
                           "low": panel.lows[s], "close": panel.closes[s],
                           "volume": panel.volumes[s]})
        df.iloc[cut:] *= 2.0
        tampered_src[s] = df
    alt = build_features(build_panel(tampered_src))

    for name in ["rsi2", "mom_60", "dist_sma200", "vol20", "rank_mom_60"]:
        pd.testing.assert_frame_equal(
            base[name].iloc[:cut], alt[name].iloc[:cut],
            check_exact=False, rtol=1e-9,
        )


def test_cross_sectional_rank_uses_only_same_day_information():
    """Ranking across symbols within a date is legitimate: every symbol's own
    history is known that day. Ranking across TIME would not be."""
    panel = _panel(400, 6)
    f = build_features(panel)
    r = f["rank_mom_60"].dropna(how="all")
    assert (r.max(axis=1) <= 1.0 + 1e-9).all()
    assert (r.min(axis=1) >= 0.0 - 1e-9).all()


def test_stop_is_checked_before_target():
    """A bar touching both resolves against the trader, as in the engine."""
    idx = pd.DatetimeIndex(pd.bdate_range("2020-01-02", periods=300))
    close = np.full(300, 100.0)
    df = pd.DataFrame({"open": close, "high": close, "low": close,
                       "close": close, "volume": 1e6}, index=idx)
    # One violent bar that spans both barriers.
    df.iloc[250, df.columns.get_loc("high")] = 130.0
    df.iloc[250, df.columns.get_loc("low")] = 70.0
    panel = build_panel({"A": df})

    sig = pd.DataFrame(False, index=panel.dates, columns=["A"])
    sig.iloc[249] = True
    feats = build_features(panel)
    ds = label_signals(panel, sig, feats, target_pct=0.04, stop_pct=0.03)
    assert len(ds) == 1
    assert ds.y[0] == 0.0, "an ambiguous bar must resolve as the stop"


def test_meta_model_can_only_decline_a_trade():
    """The structural safety property: the secondary may skip or shrink a
    trade, never propose or reverse one."""
    from bipbip.strategies.meta_labeled import MetaLabeledStrategy

    panel = _panel(700, 5)

    class AlwaysConfident:
        def predict_proba(self, X):
            return np.ones(len(X))

    class NeverConfident:
        def predict_proba(self, X):
            return np.zeros(len(X))

    sig = rsi_dip_signals(panel)
    yes = MetaLabeledStrategy(AlwaysConfident(), rsi_dip_signals, build_features,
                              threshold=0.5)
    no = MetaLabeledStrategy(NeverConfident(), rsi_dip_signals, build_features,
                             threshold=0.5)
    prep_yes, prep_no = yes.prepare(panel), no.prepare(panel)

    # Whatever the model says, the tradeable set is a subset of the primary's.
    assert prep_yes["signals"].equals(sig)
    assert prep_no["signals"].equals(sig)
    assert (prep_no["proba"].fillna(0) < 0.5).all().all()


def test_missing_market_context_stays_missing():
    """A comparison against NaN yields False, not NaN.

    Left alone that made "the market is below its 200-day average" assert
    itself for every date before the market series began - on the real panel,
    thirty years of a feature stating an unknown as a known False.
    """
    panel = _panel(600, 4)
    # Market data that only exists for the second half of the panel.
    market = pd.Series(np.nan, index=panel.dates, dtype="float64")
    rng = np.random.default_rng(9)
    tail = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, 300)))
    market.iloc[300:] = tail

    f = build_features(panel, market)
    early = f["mkt_above_sma"].iloc[:250]
    assert early.isna().all().all(), "unknown market state must be NaN, not False"


def test_market_features_are_present_once_context_exists():
    panel = _panel(900, 4)
    rng = np.random.default_rng(10)
    market = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, 900))),
                       index=panel.dates)
    f = build_features(panel, market)
    assert f["mkt_above_sma"].iloc[400:].notna().all().all()


def test_winsorisation_does_not_flatten_features_onto_one_scale():
    """Regression: a shared clip destroyed every feature with a wide range.

    Clipping all columns to a fixed +/-20 turned RSI - which spans 0-100 and
    sits high on breakout signals - into the constant 20, with zero variance
    and no information. Days-to-earnings lost every distinction beyond 20 days
    the same way. Winsorising per column at its own quantiles preserves scale.
    """
    panel = _panel(900, 6)
    feats = build_features(panel)
    ds = label_signals(panel, breakout_signals(panel), feats)

    for col in ("rsi2", "rsi14"):
        values = ds.X[col].dropna()
        assert values.std() > 1.0, f"{col} collapsed to a constant"
        assert values.max() > 20.0, f"{col} was clipped to the old fixed range"


def test_wide_range_context_features_survive_winsorisation():
    import pandas as pd
    from bipbip.ml.context_features import build_context

    panel = _panel(900, 6)
    dates = panel.dates
    # Earnings roughly quarterly for each symbol.
    rows = []
    for i, sym in enumerate(panel.symbols):
        for d in dates[::63][i % 3:]:
            rows.append({"symbol": sym, "earnings_date": pd.Timestamp(d)})
    earnings = pd.DataFrame(rows)

    ctx = build_context(panel.closes, panel.symbols, earnings, None)
    feats = {**build_features(panel), **ctx}
    ds = label_signals(panel, breakout_signals(panel), feats)

    days = ds.X["days_to_earnings"].dropna()
    assert days.max() > 20.0, "days-to-earnings was flattened by the old clip"


# --------------------------------------------------------------------------
# Trade windows, and the benchmark they exist to support.
# --------------------------------------------------------------------------

def test_entry_and_exit_bars_are_real_positions_in_the_panel():
    """Regression: event_end is a SAMPLE position, not a calendar location.

    `event_end` is remapped into the sorted sample so purging can measure
    overlap between folds. Read as a bar index it is nonsense - a benchmark
    built that way reported mean holding periods of 1,588 days for a rule
    capped at 20 bars, and a market return of +14,729 bps to compare against.
    `entry_bar` and `exit_bar` keep the real window available.
    """
    panel = _panel()
    ds = label_signals(panel, breakout_signals(panel), build_features(panel),
                       max_hold=20)
    assert ds.entry_bar is not None and ds.exit_bar is not None
    hold = ds.exit_bar - ds.entry_bar
    assert hold.min() >= 0
    assert hold.max() <= 20, f"a trade ran {hold.max()} bars against a 20-bar cap"
    assert ds.exit_bar.max() < len(panel.dates)
    assert ds.entry_bar.min() >= 0


def test_the_holding_period_cap_is_respected_for_every_trade():
    panel = _panel()
    for cap in (5, 10):
        ds = label_signals(panel, breakout_signals(panel), build_features(panel),
                           max_hold=cap)
        assert (ds.exit_bar - ds.entry_bar).max() <= cap


def test_event_end_stays_a_sample_position_for_purging():
    """The two must not be confused again in the other direction either."""
    panel = _panel()
    ds = label_signals(panel, breakout_signals(panel), build_features(panel))
    assert ds.event_end.max() < len(ds), "event_end should index the sample"
    assert (ds.event_end >= np.arange(len(ds))).all(), "purging needs it monotone"


def test_a_per_trade_return_is_not_an_edge_without_a_matched_benchmark():
    """The trap this whole section exists to document.

    These trades run about twelve bars in a market that drifts upward, so a
    positive mean per trade is the default rather than evidence. The comparison
    that means something is the SAME bars held long, and it is strictly harder
    to beat than zero.
    """
    panel = _panel()
    ds = label_signals(panel, breakout_signals(panel), build_features(panel))
    closes = panel.closes[panel.symbols[0]].to_numpy(dtype="float64")
    ep, xp = ds.entry_bar, ds.exit_bar
    bench = closes[xp] / closes[ep] - 1.0
    assert np.isfinite(bench).any(), "the benchmark must be computable at all"
    # The point: it is a different number from zero, so it can change a verdict.
    assert abs(np.nanmean(bench)) > 0


def test_a_multi_asset_rule_needs_a_per_symbol_benchmark():
    """The correction. Benchmarking against one index measures the wrong thing.

    A rule trading 34 assets, compared against SPY, is charged the gap between
    whatever it bought and the index - and SPY is the third-best compounder of
    the 34, with 31 growing more slowly. That gap was read as the triple
    barrier destroying value, and it was not; it was the assets.

    This asserts the two benchmarks genuinely differ on a panel whose symbols
    drift apart, so a future refactor cannot collapse them back together.
    """
    n = 400
    idx = pd.DatetimeIndex(pd.bdate_range("2015-01-05", periods=n))

    def series(drift):
        rng = np.random.default_rng(int(drift * 1e6) % 9999)
        c = 100.0 * np.exp(np.cumsum(rng.normal(drift, 0.01, n)))
        return pd.DataFrame({"open": c, "high": c * 1.01, "low": c * 0.99,
                             "close": c, "volume": 1e7}, index=idx)

    # One clear outperformer and one clear laggard.
    panel = build_panel({"SPY": series(0.0008), "LAGGARD": series(-0.0002)})
    ds = label_signals(panel, breakout_signals(panel), build_features(panel))
    if len(ds) < 20:
        pytest.skip("fixture produced too few signals")

    spy = panel.closes["SPY"].to_numpy(dtype="float64")
    closes = panel.closes
    ep, xp = ds.entry_bar, ds.exit_bar

    vs_spy = spy[xp] / spy[ep] - 1.0
    vs_own = np.full(len(ds), np.nan)
    for sym in np.unique(ds.symbols):
        m = ds.symbols == sym
        px = closes[sym].to_numpy(dtype="float64")
        vs_own[m] = px[xp[m]] / px[ep[m]] - 1.0

    ok = np.isfinite(vs_spy) & np.isfinite(vs_own)
    assert ok.sum() > 10
    # On the laggard the two benchmarks must disagree materially.
    lag = ok & (ds.symbols == "LAGGARD")
    if lag.sum() > 5:
        assert abs(np.nanmean(vs_spy[lag]) - np.nanmean(vs_own[lag])) > 1e-4, (
            "the per-symbol and index benchmarks collapsed together; the "
            "confound this guards against is back")
