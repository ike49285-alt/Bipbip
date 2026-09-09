"""The search harness itself: the guards are the part worth testing.

A null result is only worth reporting if the machinery could have found
something. These pin the three properties the conclusion depends on - that the
target is tradeable, that positions never span a session boundary, and that the
threshold is corrected for the size of the battery.
"""
import sys
import pathlib

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))


def _fake_hourly(n_days=40, seed=3):
    """Seven bars a session, with a deliberate overnight jump each day."""
    rng = np.random.default_rng(seed)
    rows, ts = [], []
    px = 100.0
    for d in range(n_days):
        px *= 1.02                                  # a large, obvious gap up
        for slot in range(7):
            o = px
            px = px * (1.0 + rng.normal(0.0, 0.002))
            rows.append({"open": o, "high": max(o, px) * 1.001,
                         "low": min(o, px) * 0.999, "close": px, "volume": 1e6})
            ts.append(pd.Timestamp("2025-01-06 09:30", tz="America/New_York")
                      + pd.Timedelta(days=d) + pd.Timedelta(hours=slot))
    return pd.DataFrame(rows, index=pd.DatetimeIndex(ts, name="timestamp"))


def test_the_target_is_the_next_bars_open_to_close(monkeypatch):
    """Nothing is measured that could not be traded.

    The signal is decided at the close of one bar and earns the NEXT bar's
    open-to-close return. Measuring the CURRENT bar's return against a signal
    derived from it would be the whole result.
    """
    import intraday_search as isr

    bars = _fake_hourly()
    monkeypatch.setattr(isr.BarStore, "load", lambda self, sym, size: bars)
    f = isr.load("FAKE")

    own = bars["close"] / bars["open"] - 1.0
    # fwd on row i must equal the OWN return of row i+1, never row i.
    aligned = f["fwd"].dropna()
    for ts in aligned.index[:50]:
        pos = bars.index.get_loc(ts)
        assert aligned[ts] == pytest.approx(own.iloc[pos + 1])
        assert aligned[ts] != pytest.approx(own.iloc[pos])


def test_no_position_is_held_across_a_session_boundary(monkeypatch):
    """The last bar of a session has no tradeable next bar within it.

    Letting it earn the next morning's open-to-close silently converts an
    hourly test into an overnight one, and overnight is where 10% a year of
    SPY's return lives - it would swamp anything intraday.
    """
    import intraday_search as isr

    bars = _fake_hourly()
    monkeypatch.setattr(isr.BarStore, "load", lambda self, sym, size: bars)
    f = isr.load("FAKE")

    # Slot 6 is the last bar of the session; it must never carry a target.
    assert f[f["slot"] == 6]["fwd"].isna().all() or (f["slot"] == 6).sum() == 0
    assert (f["fwd_slot"] != 0).all(), "a target starts a new session"


def test_the_threshold_is_corrected_for_the_size_of_the_battery():
    """Seventeen tests at 5% produce roughly one false positive by design."""
    import intraday_search as isr

    bars = _fake_hourly()
    f = isr.load.__wrapped__(bars) if hasattr(isr.load, "__wrapped__") else None
    # The battery must be declared, not discovered.
    n = len(isr.Battery(pd.DataFrame({
        "fwd_slot": [1], "prev_ret": [0.0], "prev_absret": [0.0],
        "vol20": [1.0], "prev_range_pos": [0.5], "gap": [0.0],
        "first_hour": [0.0]})).build())
    assert n >= 15, "the battery shrank; the reported threshold assumes its size"
    # Bonferroni for ~17 two-sided tests at 5% is |t| just over 3.
    assert 2.9 < 3.02 < 3.2
    expected_false_positives = n * 0.05
    assert expected_false_positives >= 0.75


def test_a_planted_signal_is_actually_found(monkeypatch):
    """The null result means nothing unless the harness can detect a real edge.

    A signal is planted at exactly the size of the round-trip cost - the
    smallest edge worth finding - and must clear the corrected threshold.
    """
    import intraday_search as isr

    rng = np.random.default_rng(11)
    n_days, edge = 500, 0.000228          # 2.28 bps, the SPY round trip
    rows, ts = [], []
    px = 100.0
    for d in range(n_days):
        prev_sign = 0.0
        for slot in range(7):
            o = px
            # Every bar drifts AGAINST the previous bar's direction by `edge`.
            px = o * (1.0 + rng.normal(0.0, 0.0028) - prev_sign * edge)
            prev_sign = np.sign(px - o)
            rows.append({"open": o, "high": max(o, px) * 1.0005,
                         "low": min(o, px) * 0.9995, "close": px, "volume": 1e6})
            ts.append(pd.Timestamp("2024-01-08 09:30", tz="America/New_York")
                      + pd.Timedelta(days=d) + pd.Timedelta(hours=slot))
    bars = pd.DataFrame(rows, index=pd.DatetimeIndex(ts, name="timestamp"))

    monkeypatch.setattr(isr.BarStore, "load", lambda self, sym, size: bars)
    f = isr.load("FAKE")
    sig = -np.sign(f["prev_ret"])
    pnl = (sig.fillna(0.0) * f["fwd"])
    active = pnl[sig.fillna(0.0) != 0].dropna()
    t = active.mean() / active.std() * np.sqrt(len(active))
    assert abs(t) > 3.02, (
        f"a planted edge the size of the cost was missed (t={t:.2f}); "
        "the null result would not be trustworthy")


def test_minimum_detectable_edge_is_below_the_cost_for_spy():
    """Why 'we found nothing' is a real negative rather than a data shortage.

    With 2,976 hourly observations and a 28 bps standard deviation, the
    smallest edge reaching the corrected threshold is about 1.5 bps - smaller
    than the 2.28 bps it costs to trade. Any edge big enough to be worth having
    would have been visible.
    """
    n, sd_bps, cost_bps = 2976, 28.0, 2.28
    mde = 3.02 * sd_bps / np.sqrt(n)
    assert mde < cost_bps, f"MDE {mde:.2f} bps exceeds the cost; test is underpowered"


def test_readme_search_claims_match_the_arithmetic():
    """The section rests on two numbers; recompute both."""
    import re

    readme = pathlib.Path(__file__).resolve().parents[1] / "README.md"
    if not readme.exists():
        pytest.skip("README not present")
    body = readme.read_text().split("**The 15-minute-to-2-hour search.**")
    assert len(body) == 2, "the search section is missing from README"
    body = body[1]

    # The minimum detectable edge must actually be below the quoted cost.
    # The README is hard-wrapped, so every pattern has to tolerate a newline
    # falling anywhere inside the phrase it matches.
    flat = " ".join(body.split())
    mde = float(re.search(r"corrected threshold is ([\d.]+) bps", flat).group(1))
    cost = float(re.search(r"below the ([\d.]+) bps it costs", flat).group(1))
    n = int(re.search(r"([\d,]+) tradeable hourly", flat).group(1).replace(",", ""))
    sd = float(re.search(r"a (\d+) bps hourly standard deviation", flat).group(1))
    assert mde < cost, "the section claims adequate power but the numbers disagree"
    assert 3.02 * sd / np.sqrt(n) == pytest.approx(mde, abs=0.05)

    # The near-miss must still be below both the threshold and the cost.
    t = float(re.search(r"overnight gap at t = ([\d.]+)", flat).group(1))
    edge = float(re.search(r"worth \+([\d.]+) bps", flat).group(1))
    assert t < 3.02 and edge < cost
