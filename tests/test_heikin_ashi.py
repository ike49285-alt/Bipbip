"""Heikin-Ashi candles, and the one way they are dangerous in a backtest."""
import numpy as np
import pandas as pd
import pytest

from bipbip.core import indicators as ind


def _bars(n=200, seed=2, drift=0.0004):
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(drift, 0.012, n)))
    idx = pd.DatetimeIndex(pd.bdate_range("2015-01-05", periods=n))
    high = close * (1 + np.abs(rng.normal(0, 0.004, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.004, n)))
    op = np.concatenate([[close[0]], close[:-1]])
    return pd.DataFrame({"open": op, "high": np.maximum(high, np.maximum(op, close)),
                         "low": np.minimum(low, np.minimum(op, close)),
                         "close": close, "volume": 1e7}, index=idx)


def test_the_definition():
    bars = _bars(50)
    ha = ind.heikin_ashi(bars)
    expected_close = (bars["open"] + bars["high"] + bars["low"] + bars["close"]) / 4
    pd.testing.assert_series_equal(ha["ha_close"], expected_close, check_names=False)
    # HA open is the running average of the previous HA candle.
    for i in range(1, 20):
        assert ha["ha_open"].iloc[i] == pytest.approx(
            (ha["ha_open"].iloc[i - 1] + ha["ha_close"].iloc[i - 1]) / 2)


def test_heikin_ashi_is_causal():
    """Recursive is not the same as forward-looking; this proves it."""
    bars = _bars(300)
    full = ind.heikin_ashi(bars).iloc[:200]
    truncated = ind.heikin_ashi(bars.iloc[:200])
    pd.testing.assert_frame_equal(full, truncated, rtol=1e-10)


def test_ha_prices_are_not_real_prices():
    """The reason these are signal-only.

    An HA close averages four numbers and an HA open averages two earlier
    averages, so neither ever traded. Because the construction SMOOTHS, the
    synthetic price is systematically kinder than the real one - filling an
    order at it is not optimism, it is invention.
    """
    bars = _bars(400)
    ha = ind.heikin_ashi(bars)
    gap = (ha["ha_close"] - bars["close"]).abs()
    assert gap.mean() > 0, "HA close should differ from the traded close"

    # And it is smoother: less bar-to-bar movement than the real series.
    assert ha["ha_close"].diff().abs().mean() < bars["close"].diff().abs().mean()


def test_the_ha_series_is_smoother_and_runs_longer_than_raw_direction():
    """What the technique is actually for: fewer, longer, cleaner runs."""
    bars = _bars(600, seed=7)
    ha = ind.heikin_ashi(bars)
    raw_flips = int((np.sign(bars["close"].diff()).diff() != 0).sum())
    ha_flips = int((ha["ha_trend"].diff() != 0).sum())
    assert ha_flips < raw_flips


def test_run_length_is_signed_and_resets_on_a_colour_change():
    bars = _bars(300, seed=11)
    ha = ind.heikin_ashi(bars)
    run, trend = ha["ha_run"].to_numpy(), ha["ha_trend"].to_numpy()
    for i in range(1, len(run)):
        if np.isfinite(trend[i]) and trend[i] != 0:
            assert np.sign(run[i]) == np.sign(trend[i])
            if trend[i] != trend[i - 1]:
                assert abs(run[i]) == 1, "a colour change must restart the count"


def test_run_length_is_not_all_nan():
    """Regression: _signed_run returned a default RangeIndex.

    Assigned into a frame keyed by timestamps, pandas aligned on the index and
    produced a column of NaN - which reads as a feature that simply had no
    data, rather than as a bug.
    """
    ha = ind.heikin_ashi(_bars(200))
    assert ha["ha_run"].notna().mean() > 0.9
    assert ha["ha_run"].abs().max() >= 2


def test_ha_high_and_low_bound_the_candle():
    ha = ind.heikin_ashi(_bars(300))
    assert (ha["ha_high"] >= ha[["ha_open", "ha_close"]].max(axis=1) - 1e-9).all()
    assert (ha["ha_low"] <= ha[["ha_open", "ha_close"]].min(axis=1) + 1e-9).all()


def test_body_fraction_stays_in_range():
    ha = ind.heikin_ashi(_bars(300))
    v = ha["ha_body_frac"].dropna()
    assert v.between(0.0, 1.0 + 1e-9).all()


def test_the_model_gets_ha_readings_but_never_ha_prices():
    """The guard that matters: no synthetic price reaches the feature set."""
    from bipbip.core.panel import build_panel
    from bipbip.ml.discover import _discretionary_features

    panel = build_panel({s: _bars(300, seed=i) for i, s in enumerate("ABCDE")})
    feats = _discretionary_features(panel)
    for banned in ("ha_open", "ha_close", "ha_high", "ha_low"):
        assert banned not in feats, f"{banned} is a fabricated price, not a signal"
    for wanted in ("ha_trend", "ha_run", "ha_body"):
        assert wanted in feats
