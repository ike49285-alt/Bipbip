"""Non-price features: earnings timing, sector relatives, macro broadcast."""
import numpy as np
import pandas as pd
import pytest

from bipbip.ml.context_features import (CONTEXT_COLUMNS, build_context,
                                        earnings_features, sector_features)


def _dates(n=400):
    return pd.DatetimeIndex(pd.bdate_range("2020-01-02", periods=n))


def _closes(dates, symbols, seed=1):
    rng = np.random.default_rng(seed)
    data = {s: 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, len(dates))))
            for s in symbols}
    return pd.DataFrame(data, index=dates)


def _earnings(symbols, dates, every=63):
    rows = []
    for s in symbols:
        for i in range(30, len(dates), every):
            rows.append({"symbol": s, "earnings_date": dates[i]})
    return pd.DataFrame(rows)


def test_days_to_earnings_counts_down_then_resets():
    dates = _dates()
    syms = ["AAA", "BBB"]
    f = earnings_features(dates, syms, _earnings(syms, dates))
    d = f["days_to_earnings"]["AAA"].dropna()

    assert (d >= 0).all(), "days until a future date cannot be negative"
    assert d.max() <= 60, "capped so a raw count cannot dominate a scaled model"
    # Somewhere the countdown must reach zero: the report day itself.
    assert (d == 0).any()


def test_days_since_and_days_to_are_distinguishable():
    """Signed timing is the point: a setup before a report is a bet ON the
    announcement, one after is a reaction to it."""
    dates = _dates()
    syms = ["AAA"]
    f = earnings_features(dates, syms, _earnings(syms, dates))
    to_next, since = f["days_to_earnings"]["AAA"], f["days_since_earnings"]["AAA"]
    both = pd.concat([to_next, since], axis=1).dropna()
    # They must not be the same series.
    assert not np.allclose(both.iloc[:, 0], both.iloc[:, 1])
    assert (both >= 0).all().all()


def test_earnings_features_survive_missing_data():
    dates = _dates()
    f = earnings_features(dates, ["AAA"], pd.DataFrame())
    assert f["days_to_earnings"].isna().all().all()
    assert set(f) == {"days_to_earnings", "days_since_earnings", "in_earnings_window"}


def test_earnings_window_flag_marks_only_the_window():
    dates = _dates()
    syms = ["AAA"]
    f = earnings_features(dates, syms, _earnings(syms, dates))
    win = f["in_earnings_window"]["AAA"].dropna()
    assert set(np.unique(win)) <= {0.0, 1.0}
    # It should be a minority of days, not most of them.
    assert 0.0 < win.mean() < 0.5


def test_sector_relative_strength_is_measured_within_a_sector():
    dates = _dates()
    syms = [f"T{i}" for i in range(6)]
    closes = _closes(dates, syms)
    sectors = {s: {"sector": "Tech" if i < 3 else "Health"}
               for i, s in enumerate(syms)}

    f = sector_features(closes, sectors)
    rel = f["sector_rel_strength"].dropna()
    # Within each sector the relative strengths are deviations from that
    # sector's own mean, so they sum to roughly zero across its members.
    tech = rel[["T0", "T1", "T2"]].sum(axis=1)
    assert np.allclose(tech.to_numpy(), 0.0, atol=1e-9)


def test_sector_breadth_is_a_fraction():
    dates = _dates()
    syms = [f"T{i}" for i in range(5)]
    closes = _closes(dates, syms, seed=4)
    sectors = {s: {"sector": "Tech"} for s in syms}
    b = sector_features(closes, sectors)["sector_breadth"].dropna()
    assert (b >= 0).all().all() and (b <= 1).all().all()


def test_sectors_with_too_few_members_are_skipped():
    """Relative strength against one other company is noise, not a sector."""
    dates = _dates()
    closes = _closes(dates, ["A", "B"])
    f = sector_features(closes, {"A": {"sector": "X"}, "B": {"sector": "X"}})
    assert f["sector_rel_strength"].isna().all().all()


def test_macro_is_broadcast_to_every_symbol():
    dates = _dates()
    syms = ["AAA", "BBB", "CCC"]
    closes = _closes(dates, syms)
    macro = pd.DataFrame({"rate_trend_long": np.linspace(-0.1, 0.1, len(dates))},
                         index=dates)
    ctx = build_context(closes, syms, None, None, macro)
    assert "rate_trend_long" in ctx
    row = ctx["rate_trend_long"].iloc[100]
    assert row.nunique() == 1, "a macro value is identical across symbols on a date"


def test_context_features_are_causal():
    """Corrupting later prices must not move earlier sector features."""
    dates = _dates(500)
    syms = [f"T{i}" for i in range(5)]
    closes = _closes(dates, syms, seed=7)
    sectors = {s: {"sector": "Tech"} for s in syms}

    base = sector_features(closes, sectors)["sector_rel_strength"]
    tampered = closes.copy()
    tampered.iloc[300:] *= 2.0
    alt = sector_features(tampered, sectors)["sector_rel_strength"]

    pd.testing.assert_frame_equal(base.iloc[:300], alt.iloc[:300],
                                  check_exact=False, rtol=1e-9)


def test_day_arithmetic_is_independent_of_datetime_resolution():
    """Regression for a silent, total failure.

    pandas 3 defaults to MICROSECOND datetime resolution. Converting dates to
    day numbers by dividing the raw integer by a hardcoded nanoseconds-per-day
    constant collapsed every date to the same value, so every earnings feature
    became a constant - present, plausible-looking, and carrying no information
    whatsoever.
    """
    base = pd.DatetimeIndex(pd.bdate_range("2020-01-02", periods=200))
    rows = [{"symbol": "AAA", "earnings_date": base[i]} for i in range(30, 200, 45)]
    earnings = pd.DataFrame(rows)

    for unit in ("datetime64[ns]", "datetime64[us]", "datetime64[ms]"):
        dates = pd.DatetimeIndex(base.to_numpy().astype(unit))
        f = earnings_features(dates, ["AAA"], earnings)
        d = f["days_to_earnings"]["AAA"].dropna()
        assert d.nunique() > 5, f"{unit}: feature collapsed to {d.nunique()} value(s)"
        assert d.max() > 5, f"{unit}: countdown never exceeds {d.max()}"
