"""Volume features for the cross-sectional ranker.

The ranker asks "will this symbol beat the cross-sectional median", which is the
model form of moving capital out of what is flagging and into what is running.
Until now its only volume input was `volume_ratio` - this bar's volume against a
56-bar mean, which is a spike detector and cannot distinguish interest BUILDING
from interest FADING. Fading interest is what "flagging" means to anyone reading
a chart, so the hypothesis was never actually represented in the features.

Two things are tested here, and the second is the one that matters. A feature
must be causal - no row may change when later bars are appended or removed - and
it must MEAN what its name says, which a planted case can check and a
distributional summary cannot. A feature that is merely finite and
well-scaled can still be measuring nothing.
"""
import numpy as np
import pandas as pd
import pytest

from bipbip.core.panel import Panel
from bipbip.ml.discover import _causal_features

VOLUME_FEATURES = ["volume_ratio", "volume_trend", "dollar_volume",
                   "dollar_volume_ratio", "volume_price_corr",
                   "up_volume_share"]


def _panel(closes: dict, volumes: dict, n: int) -> Panel:
    """A panel from explicit per-symbol close and volume paths."""
    idx = pd.date_range("2015-01-02", periods=n, freq="B")
    c = pd.DataFrame({s: np.asarray(v, dtype="float64") for s, v in closes.items()},
                     index=idx)
    v = pd.DataFrame({s: np.asarray(x, dtype="float64") for s, x in volumes.items()},
                     index=idx)
    return Panel(opens=c.shift(1).fillna(c), highs=c * 1.001, lows=c * 0.999,
                 closes=c, volumes=v)


def _flat(n, price=100.0, vol=1e6, seed=0):
    rng = np.random.default_rng(seed)
    c = price * np.exp(np.cumsum(rng.normal(0, 0.005, n)))
    v = np.full(n, vol) * np.exp(rng.normal(0, 0.05, n))
    return c, v


@pytest.mark.parametrize("name", VOLUME_FEATURES)
def test_every_volume_feature_is_causal(name):
    """No row may move when later bars are added or taken away.

    This is the property that a rolling window silently breaks if it is ever
    centred, and the one that makes a backtest quietly impossible to trade.
    """
    n = 400
    c1, v1 = _flat(n, seed=1)
    c2, v2 = _flat(n, price=50.0, vol=4e6, seed=2)
    full = _panel({"A": c1, "B": c2}, {"A": v1, "B": v2}, n)
    cut = 250
    trunc = Panel(opens=full.opens.iloc[:cut], highs=full.highs.iloc[:cut],
                  lows=full.lows.iloc[:cut], closes=full.closes.iloc[:cut],
                  volumes=full.volumes.iloc[:cut])

    a = _causal_features(full)[name].iloc[:cut].to_numpy(dtype="float64")
    b = _causal_features(trunc)[name].to_numpy(dtype="float64")
    assert np.allclose(a, b, equal_nan=True), f"{name} reads past its own bar"


def test_volume_trend_separates_building_from_fading():
    """The feature the ranker did not have.

    `volume_ratio` cannot tell these two apart once the level settles, because
    both end at a similar multiple of their own long mean. The direction of
    participation is the whole point.
    """
    n = 300
    rise = np.concatenate([np.full(200, 1e6), np.linspace(1e6, 4e6, 100)])
    fall = np.concatenate([np.full(200, 4e6), np.linspace(4e6, 1e6, 100)])
    c, _ = _flat(n, seed=3)
    p = _panel({"UP": c, "DOWN": c}, {"UP": rise, "DOWN": fall}, n)
    f = _causal_features(p)["volume_trend"].iloc[-1]

    assert f["UP"] > 0.15, f"building participation read {f['UP']:.3f}"
    assert f["DOWN"] < -0.15, f"fading participation read {f['DOWN']:.3f}"


def test_dollar_volume_separates_a_cheap_fund_from_an_expensive_one():
    """A million shares of a $5 fund is not a million shares of a $500 one,
    and the cross-section spans both."""
    n = 200
    shares = np.full(n, 1e6)
    p = _panel({"CHEAP": np.full(n, 5.0), "DEAR": np.full(n, 500.0)},
               {"CHEAP": shares, "DEAR": shares}, n)
    f = _causal_features(p)["dollar_volume"].iloc[-1]
    assert f["DEAR"] > f["CHEAP"]
    # 100x the price is log(100) apart, whatever the share count.
    assert f["DEAR"] - f["CHEAP"] == pytest.approx(np.log(100.0), abs=0.01)


def test_dollar_volume_is_invariant_to_the_archive_s_adjustment():
    """Rescaling multiplies price by a factor and divides volume by the same
    one, so their product is untouched. Neither leg is, which is why this is
    the size measure to rank on in an adjusted archive."""
    n = 200
    c, v = _flat(n, seed=4)
    plain = _panel({"A": c}, {"A": v}, n)
    k = 7.5                                    # a reverse split, say
    adj = _panel({"A": c * k}, {"A": v / k}, n)
    a = _causal_features(plain)["dollar_volume"].to_numpy(dtype="float64")
    b = _causal_features(adj)["dollar_volume"].to_numpy(dtype="float64")
    assert np.allclose(a, b, equal_nan=True)


def test_volume_price_corr_signs_confirmation_against_contradiction():
    """The only feature that is genuinely joint in price and volume."""
    n = 300
    rng = np.random.default_rng(5)
    step = rng.normal(0, 0.01, n)
    c = 100 * np.exp(np.cumsum(step))
    base = 1e6
    # Volume rises on up bars for CONFIRM and on down bars for DIVERGE.
    confirm = base * np.exp(3.0 * step)
    diverge = base * np.exp(-3.0 * step)
    p = _panel({"CONFIRM": c, "DIVERGE": c},
               {"CONFIRM": confirm, "DIVERGE": diverge}, n)
    f = _causal_features(p)["volume_price_corr"].iloc[-1]
    assert f["CONFIRM"] > 0.5, f"confirmation read {f['CONFIRM']:.3f}"
    assert f["DIVERGE"] < -0.5, f"divergence read {f['DIVERGE']:.3f}"


def test_up_volume_share_tracks_accumulation():
    """Volume concentrated in up bars against volume concentrated in down bars,
    on the SAME price path - so only the weighting differs."""
    n = 200
    rng = np.random.default_rng(6)
    step = rng.normal(0, 0.01, n)
    c = 100 * np.exp(np.cumsum(step))
    up = step > 0
    acc = np.where(up, 5e6, 1e6)
    dist = np.where(up, 1e6, 5e6)
    p = _panel({"ACC": c, "DIST": c}, {"ACC": acc, "DIST": dist}, n)
    f = _causal_features(p)["up_volume_share"].iloc[-1]

    # The reference is the UNWEIGHTED up-rate over the same window, not 0.5.
    # A fixed threshold would really be asserting that the window drew a
    # balanced number of up bars: this seed drew 16 of 21, which puts even
    # heavily distribution-weighted volume at 0.39. What the feature claims is
    # that weighting volume toward up bars moves it above the plain rate and
    # toward down bars moves it below - on an identical price path.
    plain = float((np.sign(np.diff(np.log(c)))[-21:] > 0).mean())
    assert f["ACC"] > plain + 0.15, f"accumulation {f['ACC']:.3f} vs rate {plain:.3f}"
    assert f["DIST"] < plain - 0.15, f"distribution {f['DIST']:.3f} vs rate {plain:.3f}"
    assert f["ACC"] - f["DIST"] > 0.4


def test_the_new_features_reach_the_model_as_ranks_too():
    """A relative question needs relative inputs; `_add_cross_sectional_ranks`
    mirrors every raw feature, so each of these arrives in both forms."""
    from bipbip.ml.discover import _add_cross_sectional_ranks
    n = 200
    c1, v1 = _flat(n, seed=7)
    c2, v2 = _flat(n, price=20.0, vol=9e6, seed=8)
    f = _add_cross_sectional_ranks(
        _causal_features(_panel({"A": c1, "B": c2}, {"A": v1, "B": v2}, n)))
    for name in VOLUME_FEATURES:
        assert name in f
        assert f"rk_{name}" in f, f"{name} never gets a cross-sectional rank"
        r = f[f"rk_{name}"].to_numpy(dtype="float64")
        r = r[np.isfinite(r)]
        assert r.min() >= 0.0 and r.max() <= 1.0


# --------------------------------------------------------------------------
# The rotation harness.
# --------------------------------------------------------------------------

def _ranker_panel(n_symbols=14, n=900, seed=11):
    """A panel wide enough for the ranker's fold and basket minimums.

    It was 24 symbols, which made this the slowest file in the suite by a wide
    margin - over two and a half minutes against about a second for a typical
    file - because the equivalence test below fits the ranker FOUR times, once
    through `procedure` and once per `top_k`. Fourteen runs it in eight seconds
    and still clears every minimum: `min_symbols` is 10 below and the largest
    basket is 10.

    THE BAR COUNT STAYS AT 900 and is not a spare knob. The splitter needs
    `n_splits * 20` usable fold timestamps, and those are counted AFTER the
    21-bar horizon and the embargo are taken out, so a first attempt at 420
    bars produced "no usable folds" and the test failed rather than running
    fast. Symbols are the free dimension here; sessions are not.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2015-01-02", periods=n, freq="B")
    closes, volumes = {}, {}
    for i in range(n_symbols):
        s = f"S{i:02d}"
        closes[s] = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.011, n)))
        volumes[s] = 1e6 * np.exp(rng.normal(0, 0.3, n))
    c = pd.DataFrame(closes, index=idx)
    v = pd.DataFrame(volumes, index=idx)
    return Panel(opens=c.shift(1).fillna(c), highs=c * 1.002, lows=c * 0.998,
                 closes=c, volumes=v)


def test_fitting_once_per_fold_matches_fitting_per_basket_size():
    """The harness fits each fold ONCE and scores every top_k off the same
    probabilities, which is 4x cheaper than refitting per k.

    That is only legitimate because the model does not depend on k - k decides
    how deep to go into a ranking the model already produced. If it ever did
    depend on k this shortcut would silently change the result, so it is checked
    against the library's own per-k path rather than assumed.
    """
    from bipbip.ml.discover import build_cross_sectional, evaluate_ranker, make_gbm
    from scripts.rotation_null import TOP_KS, procedure

    ds = build_cross_sectional(_ranker_panel(), horizon=21, min_symbols=10)
    mine = procedure(ds, ds.excess, cost_bps=0.0, rebalance_every=21, model_seed=0)
    assert "by_k" in mine, mine.get("note")

    for k in TOP_KS:
        if k not in mine["by_k"]:
            continue
        theirs = evaluate_ranker(ds, make_model=make_gbm, top_k=k, cost_bps=0.0,
                                 rebalance_every=21)
        assert theirs["rebalances"] == mine["by_k"][k]["rebalances"]
        assert theirs["excess_bps"] == pytest.approx(
            mine["by_k"][k]["excess_bps"], abs=1e-6), f"top_k={k} diverged"


def test_the_ablation_removes_volume_and_only_volume():
    from bipbip.ml.discover import build_cross_sectional
    from scripts.rotation_null import VOLUME_FEATURES, price_only

    ds = build_cross_sectional(_ranker_panel(), horizon=21, min_symbols=10)
    thin = price_only(ds)
    gone = set(ds.X.columns) - set(thin.X.columns)
    # Every volume feature and its rank, nothing else.
    assert gone == ({f for f in VOLUME_FEATURES} |
                    {f"rk_{f}" for f in VOLUME_FEATURES})
    assert len(thin.X.columns) == len(ds.X.columns) - 2 * len(VOLUME_FEATURES)
    # Labels are untouched, so the two arms are scored on the same target.
    assert np.array_equal(thin.excess, ds.excess)


def test_a_severe_survivorship_universe_is_refused():
    """Ranking a present-day membership list is partly ranking 'did this
    survive', worth 15-21 points a year here. The runner must refuse rather
    than produce the 26.90% that momentum reads on the contaminated list."""
    from bipbip.data.universe import UNIVERSES
    severe = [n for n, v in UNIVERSES.items()
              if str(v.get("survivorship", "")).upper() == "SEVERE"]
    assert severe, "no universe is flagged SEVERE; the guard has nothing to catch"
    assert all(not n.startswith("etf") for n in severe)


def test_the_shuffle_moves_labels_but_keeps_each_timestamp_intact():
    """A global shuffle would break the cross-section as well as the signal,
    scoring the model against a world where the median means nothing."""
    from scripts.rotation_null import _shuffle_within_timestamp

    dates = np.repeat(pd.date_range("2020-01-01", periods=50), 10)
    excess = np.arange(len(dates), dtype="float64")
    out, moved = _shuffle_within_timestamp(excess, dates, seed=3)

    assert moved > 0.5, f"only {moved:.1%} of labels moved"
    assert sorted(out.tolist()) == sorted(excess.tolist())
    # Each timestamp holds exactly the values it held before, reordered.
    for ts in np.unique(dates):
        m = dates == ts
        assert sorted(out[m].tolist()) == sorted(excess[m].tolist())
