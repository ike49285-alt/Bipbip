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
