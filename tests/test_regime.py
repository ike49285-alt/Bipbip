"""Daily regime context, and above all that it never leaks the future."""
import numpy as np
import pandas as pd
import pytest

from bipbip.data import BarStore, make_intraday_bars
from bipbip.data.regime import (REGIME_COLUMNS, attach_to_intraday, daily_features,
                                session_open_prices)
from bipbip.data.sessions import EXCHANGE_TZ
from bipbip.ml.features import build_features


def _daily_from(intraday: pd.DataFrame) -> pd.DataFrame:
    """Build a plausible daily frame covering the intraday sessions, plus enough
    prior history for the expanding percentile to be defined."""
    sessions = sorted({t.date() for t in intraday.index})
    rng = np.random.default_rng(11)
    hist = pd.bdate_range(end=pd.Timestamp(sessions[0]) - pd.Timedelta(days=1), periods=400)
    all_dates = list(hist.date) + sessions
    close = 400 * np.exp(np.cumsum(rng.normal(0, 0.01, len(all_dates))))
    return pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99,
         "close": close, "volume": 1e7},
        index=pd.DatetimeIndex([pd.Timestamp(d).tz_localize(EXCHANGE_TZ) for d in all_dates]),
    )


def test_session_never_sees_its_own_daily_bar():
    """The core guarantee.

    A session on date D must use the daily bar completed on D-1. D's own daily
    bar contains D's close - the thing a strategy is trying to predict - so
    using it is lookahead bias of the most damaging kind: invisible, and it
    flatters every downstream result.
    """
    intraday = make_intraday_bars(n_sessions=10, seed=401)
    daily = _daily_from(intraday)
    sessions = sorted({t.date() for t in intraday.index})
    target = sessions[5]

    base = attach_to_intraday(intraday.index, session_open_prices(intraday), daily)

    tampered = daily.copy()
    mask = [t.date() >= target for t in tampered.index]
    tampered.loc[mask, ["open", "high", "low", "close"]] *= 3.0
    alt = attach_to_intraday(intraday.index, session_open_prices(intraday), tampered)

    sel = [t.date() == target for t in base.index]
    cols = ["regime_vol_pct", "regime_vol_ratio", "daily_trend_20", "daily_atr_pct"]
    pd.testing.assert_frame_equal(base.loc[sel, cols], alt.loc[sel, cols])


def test_corrupting_the_future_leaves_earlier_sessions_untouched():
    intraday = make_intraday_bars(n_sessions=12, seed=402)
    daily = _daily_from(intraday)
    sessions = sorted({t.date() for t in intraday.index})
    cut = sessions[7]

    base = attach_to_intraday(intraday.index, session_open_prices(intraday), daily)
    tampered = daily.copy()
    tampered.loc[[t.date() >= cut for t in tampered.index], "close"] *= 5.0
    alt = attach_to_intraday(intraday.index, session_open_prices(intraday), tampered)

    before = [t.date() < cut for t in base.index]
    cols = ["regime_vol_pct", "daily_trend_20", "daily_atr_pct"]
    pd.testing.assert_frame_equal(base.loc[before, cols], alt.loc[before, cols])


def test_volatility_percentile_uses_only_prior_history():
    """A full-sample rank would tell each day where it sits among days that had
    not happened yet."""
    idx = pd.DatetimeIndex([pd.Timestamp(d).tz_localize(EXCHANGE_TZ)
                            for d in pd.bdate_range("2020-01-01", periods=900).date])
    rng = np.random.default_rng(12)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx))))
    daily = pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99,
                          "close": close, "volume": 1e6}, index=idx)

    full = daily_features(daily)["regime_vol_pct"]
    truncated = daily_features(daily.iloc[:600])["regime_vol_pct"]
    pd.testing.assert_series_equal(full.iloc[:600], truncated, check_exact=False, rtol=1e-9)


def test_overnight_gap_is_known_at_the_open():
    """The one legitimate same-day value: today's open against yesterday's close."""
    intraday = make_intraday_bars(n_sessions=8, seed=403)
    daily = _daily_from(intraday)
    ctx = attach_to_intraday(intraday.index, session_open_prices(intraday), daily)
    gaps = ctx["overnight_gap"].dropna()
    assert len(gaps) > 0
    # Constant within a session: it cannot change once the bell has rung.
    per_day = ctx.groupby([t.date() for t in ctx.index])["overnight_gap"].nunique(dropna=True)
    assert (per_day <= 1).all()


def test_features_gain_regime_columns_only_when_daily_is_supplied():
    intraday = make_intraday_bars(n_sessions=8, seed=404)
    daily = _daily_from(intraday)
    without = build_features(intraday)
    with_daily = build_features(intraday, daily=daily)
    assert with_daily.shape[1] == without.shape[1] + len(REGIME_COLUMNS)
    for col in REGIME_COLUMNS:
        assert col in with_daily.columns
        assert col not in without.columns


@pytest.mark.parametrize("column", REGIME_COLUMNS)
def test_regime_features_are_causal(column):
    """Same guarantee the intraday features carry, applied to daily context."""
    intraday = make_intraday_bars(n_sessions=10, seed=405)
    daily = _daily_from(intraday)
    cut = int(len(intraday) * 0.6)

    full = build_features(intraday, daily=daily)[column].iloc[:cut]
    trunc = build_features(intraday.iloc[:cut], daily=daily)[column]
    pd.testing.assert_series_equal(full, trunc, check_exact=False, rtol=1e-9)


def test_regime_note_flags_a_calm_window():
    from bipbip.core.metrics import regime_note

    store = BarStore("data/bars")
    daily = store.load("SPY", "1d")
    if daily.empty:
        pytest.skip("no daily archive")
    note = regime_note(daily, 21)
    assert "percentile" in note
    assert "vol" in note
