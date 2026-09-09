"""Tests for the raw-candle feature set: no indicators, nothing from the future."""
import numpy as np
import pandas as pd
import pytest

from bipbip.core.indicators import assert_causal
from bipbip.ml.candles import candle_features, clock_features


def _bars(n=200, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.DatetimeIndex(pd.date_range("2024-01-02 09:30", periods=n, freq="30min",
                                         tz="America/New_York"))
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    # Bar geometry has to vary bar to bar. A fixture with a fixed high/close
    # ratio makes h_0 a constant column and fails a check that is really about
    # the fixture rather than the feature.
    up = 1.0 + np.abs(rng.normal(0, 0.004, n))
    dn = 1.0 - np.abs(rng.normal(0, 0.004, n))
    op = close * (1.0 + rng.normal(0, 0.003, n))
    return pd.DataFrame({"open": op,
                         "high": np.maximum(close * up, op),
                         "low": np.minimum(close * dn, op), "close": close,
                         "volume": rng.integers(1000, 5000, n).astype(float)},
                        index=idx)


def test_features_are_causal():
    assert_causal(lambda b: candle_features(b, 12), _bars(400))


def test_no_constant_columns():
    f = candle_features(_bars(300), 8)
    assert not [c for c in f.columns if f[c].nunique(dropna=True) <= 1]


def test_scale_free_across_price_levels():
    """The same shape at $5 and at $500 must produce the same features."""
    b = _bars(120)
    a = candle_features(b, 6).dropna()
    scaled = candle_features(b * pd.Series({"open": 100.0, "high": 100.0,
                                            "low": 100.0, "close": 100.0,
                                            "volume": 1.0}), 6).dropna()
    common = a.index.intersection(scaled.index)
    assert len(common) > 50
    np.testing.assert_allclose(a.loc[common].to_numpy(),
                               scaled.loc[common].to_numpy(), atol=1e-9)


def test_volume_is_scale_free_too():
    """Volume that grew by orders of magnitude must not encode the year."""
    b = _bars(200)
    big = b.copy()
    big["volume"] = big["volume"] * 1000.0
    a = candle_features(b, 10).dropna()
    c = candle_features(big, 10).dropna()
    common = a.index.intersection(c.index)
    vcols = [x for x in a.columns if x.startswith("v_")]
    np.testing.assert_allclose(a.loc[common, vcols].to_numpy(),
                               c.loc[common, vcols].to_numpy(), atol=1e-9)


def test_lag_zero_close_is_not_emitted():
    f = candle_features(_bars(60), 4)
    assert "c_0" not in f.columns
    assert "o_0" in f.columns and "h_0" in f.columns


def test_lookback_controls_width():
    for lb in (1, 4, 12):
        f = candle_features(_bars(120), lb)
        assert f.shape[1] == 5 * lb - 1


def test_rejects_zero_lookback():
    with pytest.raises(ValueError):
        candle_features(_bars(50), 0)


def test_clock_features_are_separate_from_price():
    b = _bars(80)
    c = clock_features(b.index)
    assert list(c.columns) == ["bar_of_day", "dow"]
    assert not set(c.columns) & set(candle_features(b, 4).columns)
