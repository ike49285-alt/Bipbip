"""The self-directed ranker, and the panel bug that made its first run fiction."""
import numpy as np
import pandas as pd
import pytest

from bipbip.core.panel import build_panel, load_panel
from bipbip.ml.discover import build_cross_sectional


def _intraday(sym_drift, n_days=40, bars=7, seed=1):
    """A small hourly panel: `bars` stamps per session, real times of day."""
    rng = np.random.default_rng(seed)
    out = {}
    for k, (sym, drift) in enumerate(sym_drift.items()):
        ts, rows, px = [], [], 100.0
        for d in range(n_days):
            day = pd.Timestamp("2025-01-06", tz="America/New_York") + pd.Timedelta(days=d)
            for b in range(bars):
                o = px
                px *= np.exp(rng.normal(drift, 0.004))
                rows.append({"open": o, "high": max(o, px) * 1.001,
                             "low": min(o, px) * 0.999, "close": px,
                             "volume": 1e6})
                ts.append(day + pd.Timedelta(hours=9, minutes=30) + pd.Timedelta(hours=b))
        out[sym] = pd.DataFrame(rows, index=pd.DatetimeIndex(ts, name="timestamp"))
    return out


def test_an_intraday_panel_keeps_every_bar():
    """Regression: build_panel truncated stamps to their DATE and kept the last.

    Correct for daily bars, which ARE session dates. For hourly data it
    discarded six of every seven bars, leaving only the 15:30 close - a
    305-symbol panel came back with 730 rows instead of 5,087 and described
    itself as hourly the whole way. The first cross-sectional result was
    therefore daily closes wearing an hourly label.
    """
    bars_by_symbol = _intraday({"AAA": 0.0002, "BBB": -0.0001}, n_days=40, bars=7)
    panel = build_panel(bars_by_symbol)
    assert len(panel) == 40 * 7, f"expected 280 rows, got {len(panel)}"
    # And the stamps must still carry a time of day.
    assert len({t.hour for t in panel.dates.time if True} if False else
               {ts.hour for ts in panel.dates}) > 1


def test_a_daily_panel_is_still_keyed_by_session_date():
    """The other half: daily behaviour must not change."""
    idx = pd.DatetimeIndex(pd.bdate_range("2020-01-06", periods=50))
    frame = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0,
                          "close": np.linspace(100, 120, 50), "volume": 1e6},
                         index=idx)
    panel = build_panel({"AAA": frame, "BBB": frame * 1.01})
    assert len(panel) == 50
    assert all(ts.hour == 0 for ts in panel.dates)


def test_duplicate_intraday_stamps_are_still_collapsed():
    """Keeping timestamps must not reintroduce duplicate rows."""
    bars = _intraday({"AAA": 0.0}, n_days=5, bars=7)["AAA"]
    doubled = pd.concat([bars, bars]).sort_index()
    panel = build_panel({"AAA": doubled})
    assert len(panel) == 35
    assert not panel.dates.duplicated().any()


# --------------------------------------------------------------------------
# The dataset's defining property: it is measured against the cross-section.
# --------------------------------------------------------------------------

def test_the_label_is_relative_so_the_base_rate_is_a_coin_flip():
    """Why this framing exists.

    Every per-trade metric elsewhere in this project was contaminated by market
    drift - shuffled labels "earned" +31.8 bps because everything rises. A
    label defined against the cross-sectional MEDIAN cannot be: if all symbols
    gain, the median gains with them, and half the sample is still below it.
    """
    panel = build_panel(_intraday({c: 0.0008 for c in "ABCDEFGHIJ"},
                                  n_days=60, bars=7, seed=3))
    ds = build_cross_sectional(panel, horizon=7, min_symbols=6)
    assert len(ds) > 500
    # A universe where EVERY symbol drifts up hard must still be ~50/50.
    assert 0.42 < ds.y.mean() < 0.58, f"base rate {ds.y.mean():.1%} is not neutral"
    assert abs(np.nanmedian(ds.excess)) < 1e-3


def test_features_and_labels_are_causal():
    """Truncating the future must not change any earlier feature row."""
    full_bars = _intraday({c: 0.0003 for c in "ABCDEFG"}, n_days=60, bars=7, seed=5)
    cut = {s: b.iloc[:len(b) // 2] for s, b in full_bars.items()}

    a = build_cross_sectional(build_panel(full_bars), horizon=7, min_symbols=4)
    b = build_cross_sectional(build_panel(cut), horizon=7, min_symbols=4)
    overlap = min(len(b) // 2, 200)
    assert overlap > 20
    np.testing.assert_allclose(
        a.X.iloc[:overlap].to_numpy(dtype="float64"),
        b.X.iloc[:overlap].to_numpy(dtype="float64"),
        rtol=1e-6, equal_nan=True)


def test_entry_is_the_next_bar_not_the_signal_bar():
    """Nothing is measured that could not be traded."""
    panel = build_panel(_intraday({c: 0.0002 for c in "ABCDEF"}, n_days=40, bars=7))
    ds = build_cross_sectional(panel, horizon=3, min_symbols=4)
    o, c = panel.opens, panel.closes
    for row in range(0, min(len(ds), 50), 7):
        i, sym = int(ds.bar_index[row]), ds.symbols[row]
        expected = c[sym].iloc[i + 3] / o[sym].iloc[i + 1] - 1.0
        if np.isfinite(expected) and np.isfinite(ds.fwd[row]):
            assert ds.fwd[row] == pytest.approx(expected, rel=1e-9)


def test_ranks_are_cross_sectional_not_time_series():
    panel = build_panel(_intraday({c: 0.0002 for c in "ABCDEFGH"}, n_days=40, bars=7))
    ds = build_cross_sectional(panel, horizon=7, min_symbols=5)
    rank_cols = [c for c in ds.X.columns if c.startswith("rk_")]
    assert rank_cols, "no cross-sectional rank features were built"
    for col in rank_cols[:5]:
        v = ds.X[col].dropna()
        assert v.min() >= 0.0 and v.max() <= 1.0


def test_overlapping_rebalances_inflate_the_t_statistic():
    """The flaw that made a 2-week edge look like t=10.78.

    Rebalancing every bar while holding for `horizon` bars makes consecutive
    observations share horizon-1 of their bars - nearly the same trade counted
    `horizon` times. A t-statistic over those assumes an independence they do
    not have, and inflates by roughly sqrt(horizon). Reported t rose 1.60,
    5.55, 10.78 across horizons of 21, 35 and 70 bars, which is almost exactly
    that curve rather than a strengthening edge.

    This asserts the mechanism directly on a synthetic series: spacing the
    rebalances must reduce the observation count by about the horizon.
    """
    from bipbip.ml.discover import evaluate_ranker

    rng = np.random.default_rng(3)
    n_days, bars, horizon = 200, 7, 35
    panel = build_panel({
        s: pd.DataFrame(
            {"open": p, "high": p * 1.002, "low": p * 0.998, "close": p,
             "volume": 1e6},
            index=pd.DatetimeIndex(
                [pd.Timestamp("2024-01-08", tz="America/New_York")
                 + pd.Timedelta(days=d) + pd.Timedelta(hours=9, minutes=30)
                 + pd.Timedelta(hours=b)
                 for d in range(n_days) for b in range(bars)]))
        for s, p in {
            c: 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.004, n_days * bars)))
            for c in "ABCDEFGHIJKLMNOPQRST"}.items()})

    ds = build_cross_sectional(panel, horizon=horizon, min_symbols=10)
    if len(ds) < 5000:
        pytest.skip("fixture too small")

    overlapped = evaluate_ranker(ds, top_k=5, n_splits=3)
    spaced = evaluate_ranker(ds, top_k=5, n_splits=3, rebalance_every=horizon)
    if "rebalances" not in overlapped or "rebalances" not in spaced:
        pytest.skip("no usable folds in the fixture")

    # Spacing must cut the observation count by roughly the horizon.
    ratio = overlapped["rebalances"] / max(spaced["rebalances"], 1)
    assert ratio > horizon * 0.5, (
        f"spacing only reduced observations {ratio:.1f}x for a {horizon}-bar "
        "hold; the rebalances are not actually being spaced")


def test_spacing_is_off_by_default_so_the_change_is_explicit():
    """A default that silently changed every earlier number would be worse."""
    import inspect

    from bipbip.ml.discover import evaluate_ranker
    sig = inspect.signature(evaluate_ranker)
    assert sig.parameters["rebalance_every"].default is None
