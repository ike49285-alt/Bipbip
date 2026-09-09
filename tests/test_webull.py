"""Tests for the Webull ingest path.

Each test encodes a defect that was actually present in the fifteen-year TQQQ
pull, because those are the ones that will recur on the next symbol.
"""
import numpy as np
import pandas as pd
import pytest

from bipbip.core.indicators import assert_causal
from bipbip.data.webull import (adjustment_factors, drop_leaked_closes,
                                parse_bars, reconcile_sessions,
                                rescale_to_adjusted, robust_factors,
                                verify_factors)

TZ = "America/New_York"


def _session(day: str, closes, opens=None, highs=None, lows=None):
    start = pd.Timestamp(f"{day} 09:30", tz=TZ)
    idx = pd.DatetimeIndex([start + pd.Timedelta(minutes=30 * i)
                            for i in range(len(closes))])
    c = np.asarray(closes, dtype=float)
    return pd.DataFrame(
        {"open": c if opens is None else opens,
         "high": c if highs is None else highs,
         "low": c if lows is None else lows,
         "close": c, "volume": np.full(len(c), 1000.0)}, index=idx)


def _panel(days, level=100.0, n=4):
    frames = [_session(d, np.full(n, level)) for d in days]
    return pd.concat(frames)


def _daily(days, closes):
    idx = pd.DatetimeIndex([pd.Timestamp(d) for d in days])
    c = np.asarray(closes, dtype=float)
    return pd.DataFrame({"open": c, "high": c, "low": c, "close": c}, index=idx)


def test_parse_bars_converts_utc_to_exchange_time():
    df = parse_bars("2024-03-01T14:30:00.000+0000,1,2,0.5,1.5,100")
    assert len(df) == 1
    assert str(df.index.tz) == TZ
    assert df.index[0].hour == 9 and df.index[0].minute == 30


def test_parse_bars_deduplicates_and_sorts():
    text = ("2024-03-01T15:00:00.000+0000,1,1,1,1,10\n"
            "2024-03-01T14:30:00.000+0000,2,2,2,2,20\n"
            "2024-03-01T15:00:00.000+0000,3,3,3,3,30\n")
    df = parse_bars(text)
    assert len(df) == 2
    assert df.index.is_monotonic_increasing
    assert df["close"].iloc[-1] == 3.0     # later duplicate wins


def test_rescale_puts_raw_prices_on_the_adjusted_scale():
    days = ["2024-03-01", "2024-03-04"]
    raw = _panel(days, level=200.0)
    daily = _daily(days, [2.0, 2.0])
    out = rescale_to_adjusted(raw, daily["close"])
    assert np.allclose(out["close"], 2.0)
    # Volume moves the other way so that price times volume is preserved.
    assert np.allclose(out["volume"], 1000.0 * 100.0)


def test_rescale_removes_the_split_seam():
    """A 2:1 split must not appear as a 50% overnight crash."""
    days = ["2024-03-01", "2024-03-04"]
    raw = pd.concat([_session(days[0], np.full(4, 100.0)),
                     _session(days[1], np.full(4, 50.0))])
    daily = _daily(days, [50.0, 50.0])       # adjusted: flat across the split
    out = rescale_to_adjusted(raw, daily["close"])
    closes = out["close"].to_numpy()
    assert np.allclose(closes, 50.0), closes


def test_robust_factors_ignore_a_single_corrupt_close():
    """One bad session must not set its own adjustment factor."""
    days = pd.bdate_range("2024-01-01", periods=25).strftime("%Y-%m-%d").tolist()
    raw = _panel(days, level=100.0)
    closes = np.full(len(days), 1.0)
    naive_days = days[:]
    # Corrupt one session's last bar the way an after-hours leak does.
    bad = days[12]
    mask = pd.DatetimeIndex([pd.Timestamp(x.date()) for x in raw.index]) == pd.Timestamp(bad)
    last_of_bad = np.flatnonzero(mask)[-1]
    raw.iloc[last_of_bad, raw.columns.get_loc("close")] = 70.0

    daily = _daily(naive_days, closes)
    naive = adjustment_factors(raw, daily["close"])
    robust = robust_factors(raw, daily["close"])
    t = pd.Timestamp(bad)
    assert naive[t] == pytest.approx(1.0 / 70.0)       # corrupted
    assert robust[t] == pytest.approx(0.01)            # neighbours win


def test_robust_factors_do_not_smear_across_a_split():
    """A window straddling a split must not average the two plateaus."""
    days = pd.bdate_range("2024-01-01", periods=20).strftime("%Y-%m-%d").tolist()
    lv = [100.0] * 10 + [50.0] * 10
    raw = pd.concat([_session(d, np.full(4, v)) for d, v in zip(days, lv)])
    daily = _daily(days, np.full(20, 50.0))
    f = robust_factors(raw, daily["close"])
    assert f.iloc[:10].to_numpy() == pytest.approx(0.5)
    assert f.iloc[10:].to_numpy() == pytest.approx(1.0)
    assert (np.abs(np.log(f / f.shift(1))) > 0.10).sum() == 1


def test_reconcile_flags_a_bar_that_trades_outside_the_daily_range():
    days = ["2024-03-01", "2024-03-04"]
    good = _session(days[0], np.full(4, 100.0))
    leaked = _session(days[1], [100.0, 100.0, 100.0, 80.0],
                      lows=[100.0, 100.0, 100.0, 80.0])
    raw = pd.concat([good, leaked])
    daily = _daily(days, [100.0, 100.0])
    f = pd.Series(1.0, index=pd.DatetimeIndex([pd.Timestamp(d) for d in days]))
    rep = reconcile_sessions(raw, daily, f)
    assert not rep.loc[pd.Timestamp(days[0]), "bad"]
    assert rep.loc[pd.Timestamp(days[1]), "bad"]


def test_drop_leaked_closes_removes_only_the_final_bar():
    days = ["2024-03-01", "2024-03-04"]
    raw = pd.concat([_session(days[0], np.full(4, 100.0)),
                     _session(days[1], [100.0, 100.0, 100.0, 80.0])])
    rep = pd.DataFrame({"bad": [False, True]},
                       index=pd.DatetimeIndex([pd.Timestamp(d) for d in days]))
    out = drop_leaked_closes(raw, rep)
    assert len(out) == len(raw) - 1
    assert 80.0 not in set(out["close"])
    assert (pd.DatetimeIndex([pd.Timestamp(x.date()) for x in out.index])
            == pd.Timestamp(days[1])).sum() == 3


def test_reconcile_close_check_is_skippable_after_repair():
    """Post-repair the session no longer holds the close, and must not be failed for it."""
    days = ["2024-03-01"]
    raw = _session(days[0], [100.0, 100.0, 100.0])       # last bar already removed
    daily = _daily(days, [95.0])                         # true close, never reached
    daily.loc[:, "high"] = 100.0                         # the day's real range
    daily.loc[:, "low"] = 95.0
    f = pd.Series(1.0, index=pd.DatetimeIndex([pd.Timestamp(days[0])]))
    assert reconcile_sessions(raw, daily, f, check_close=True)["bad"].all()
    assert not reconcile_sessions(raw, daily, f, check_close=False)["bad"].any()


def test_verify_factors_reports_only_real_steps():
    idx = pd.DatetimeIndex(pd.bdate_range("2024-01-01", periods=10))
    f = pd.Series([1.0] * 5 + [2.0] * 5, index=idx)
    steps = verify_factors(f)
    assert len(steps) == 1
    assert steps["step"].iloc[0] == pytest.approx(1.0)


def test_assert_causal_catches_a_leaking_feature():
    idx = pd.DatetimeIndex(pd.bdate_range("2024-01-01", periods=200))
    bars = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0,
                         "close": np.arange(200.0), "volume": 1.0}, index=idx)

    def leaky(b):
        return pd.DataFrame({"future": b["close"].shift(-5)}, index=b.index)

    def causal(b):
        return pd.DataFrame({"past": b["close"].shift(5)}, index=b.index)

    with pytest.raises(AssertionError, match="future"):
        assert_causal(leaky, bars)
    assert_causal(causal, bars)          # must not raise


def test_assert_causal_does_not_flag_a_rolling_maximum():
    """NaN != NaN made every warmup row look like a leak, condemning a rolling max."""
    idx = pd.DatetimeIndex(pd.bdate_range("2024-01-01", periods=200))
    bars = pd.DataFrame({"open": 1.0, "high": np.arange(200.0), "low": 0.0,
                         "close": np.arange(200.0), "volume": 1.0}, index=idx)

    def rolling_max(b):
        return pd.DataFrame({"hi": b["high"].rolling(26, min_periods=26).max()},
                            index=b.index)

    assert_causal(rolling_max, bars)


def test_spacing_detects_granularity():
    """Dumps of every granularity share one directory; only the gaps distinguish them."""
    import importlib.util, pathlib
    spec = importlib.util.spec_from_file_location(
        "wi", pathlib.Path(__file__).resolve().parents[1] / "scripts" / "webull_ingest.py")
    wi = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wi)

    def rows(minutes, n=10):
        base = pd.Timestamp("2026-09-09T13:30:00+0000")
        return [{"time": (base + pd.Timedelta(minutes=minutes * i)).isoformat()}
                for i in range(n)]

    assert wi._spacing_minutes(rows(1)) == pytest.approx(1)
    assert wi._spacing_minutes(rows(30)) == pytest.approx(30)
    assert wi._spacing_minutes(rows(60)) == pytest.approx(60)
    # A single bar cannot reveal its own granularity, and must not guess.
    assert not np.isfinite(wi._spacing_minutes(rows(30, n=1)))


def test_spacing_survives_a_session_gap():
    """Overnight gaps are larger than the bar interval; the mode must ignore them."""
    import importlib.util, pathlib
    spec = importlib.util.spec_from_file_location(
        "wi2", pathlib.Path(__file__).resolve().parents[1] / "scripts" / "webull_ingest.py")
    wi = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wi)
    day1 = [{"time": (pd.Timestamp("2026-09-08T13:30:00+0000")
                      + pd.Timedelta(minutes=30 * i)).isoformat()} for i in range(13)]
    day2 = [{"time": (pd.Timestamp("2026-09-09T13:30:00+0000")
                      + pd.Timedelta(minutes=30 * i)).isoformat()} for i in range(13)]
    assert wi._spacing_minutes(day1 + day2) == pytest.approx(30)
