"""Tests for barrier resolution.

The positive controls matter more than usual here: a resolver that never fires
returns timeouts for everything, which looks exactly like a working resolver on
a quiet series. Each case plants an outcome and checks it is recovered.
"""
import numpy as np
import pandas as pd
import pytest

from bipbip.ml.barriers import (STOP, TARGET, TIMEOUT, UNRESOLVED,
                                barrier_labels, first_touch)


def _bars(closes, highs=None, lows=None, opens=None, day="2026-01-05"):
    n = len(closes)
    idx = pd.DatetimeIndex([pd.Timestamp(f"{day} 09:30") + pd.Timedelta(minutes=30 * i)
                            for i in range(n)])
    c = np.asarray(closes, float)
    return pd.DataFrame(
        {"open": c if opens is None else np.asarray(opens, float),
         "high": c if highs is None else np.asarray(highs, float),
         "low": c if lows is None else np.asarray(lows, float),
         "close": c, "volume": np.full(n, 1000.0)}, index=idx)


def _call(b, target, stop, max_hold, side=1):
    entry = b["open"].shift(-1).to_numpy()
    sess = pd.DatetimeIndex(b.index).normalize().view("int64")
    return first_touch(b["high"].to_numpy(), b["low"].to_numpy(),
                       b["close"].to_numpy(), entry,
                       np.full(len(b), target, float),
                       np.full(len(b), stop, float),
                       max_hold, sess, side=side)


def test_target_is_found_when_planted():
    """Positive control: a run that reaches the target must resolve as TARGET."""
    b = _bars([100, 100, 103, 100, 100], highs=[100, 100, 103, 100, 100])
    ret, which, held = _call(b, target=102.0, stop=98.0, max_hold=3)
    assert which[0] == TARGET
    assert held[0] == 2                      # entry at bar 1, target on bar 2
    assert ret[0] == pytest.approx(102.0 / 100.0 - 1.0)


def test_stop_is_found_when_planted():
    b = _bars([100, 100, 97, 100, 100], lows=[100, 100, 97, 100, 100])
    ret, which, held = _call(b, target=102.0, stop=98.0, max_hold=3)
    assert which[0] == STOP
    assert ret[0] == pytest.approx(98.0 / 100.0 - 1.0)


def test_timeout_exits_at_the_close():
    """max_hold is bars HELD: entering at bar 1 with max_hold=3 exits at bar 3."""
    b = _bars([100, 100, 100.5, 100.7, 101.0])
    ret, which, held = _call(b, target=105.0, stop=95.0, max_hold=3)
    assert which[0] == TIMEOUT
    assert held[0] == 3
    assert ret[0] == pytest.approx(100.7 / 100.0 - 1.0)


def test_stop_wins_a_bar_that_touches_both():
    """The whole point. OHLC cannot order two touches inside one bar.

    Awarding that bar to the target is how a barrier backtest pays itself, so
    the ambiguous case must resolve as the stop.
    """
    b = _bars([100, 100, 100], highs=[100, 100, 105], lows=[100, 100, 95])
    ret, which, _ = _call(b, target=102.0, stop=98.0, max_hold=3)
    assert which[0] == STOP
    assert ret[0] < 0


def test_earlier_barrier_wins():
    """A target at bar 1 must not be overwritten by a stop at bar 3."""
    b = _bars([100, 100, 103, 100, 96],
              highs=[100, 100, 103, 100, 96], lows=[100, 100, 103, 100, 96])
    _, which, held = _call(b, target=102.0, stop=98.0, max_hold=4)
    assert which[0] == TARGET and held[0] == 2


def test_position_closes_at_the_session_end():
    """Nothing is carried overnight, even if the barrier would hit tomorrow."""
    a = _bars([100, 100, 100], day="2026-01-05")
    c = _bars([100, 110, 110], day="2026-01-06")
    b = pd.concat([a, c])
    _, which, held = _call(b, target=105.0, stop=95.0, max_hold=10)
    assert which[0] == TIMEOUT          # the 110s belong to the next session
    assert held[0] <= 2


def test_last_bar_has_no_trade():
    b = _bars([100, 100])
    _, which, _ = _call(b, target=101.0, stop=99.0, max_hold=3)
    assert which[-1] == UNRESOLVED


def test_short_is_not_the_mirror_of_long():
    """A short's stop is above and its target below, so it resolves elsewhere."""
    b = _bars([100, 100, 103, 97, 100],
              highs=[100, 100, 103, 97, 100], lows=[100, 100, 103, 97, 100])
    _, long_which, _ = _call(b, target=102.0, stop=98.0, max_hold=4, side=1)
    _, short_which, _ = _call(b, target=98.0, stop=102.0, max_hold=4, side=-1)
    assert long_which[0] == TARGET      # up to 103 first
    assert short_which[0] == STOP       # the same move stops the short out


def test_barrier_labels_charges_cost_to_both_sides():
    b = _bars(np.full(40, 100.0))
    atr = pd.Series(np.full(40, 1.0), index=b.index)
    out = barrier_labels(b, atr, cost_bps=5.0)
    flat = out["long_ret"].dropna()
    assert len(flat) > 0
    # A perfectly flat market times out at the entry price, so the only thing
    # left is the cost - and it must be charged to a short as well as a long.
    assert flat.iloc[0] == pytest.approx(-5.0)
    assert out["short_ret"].dropna().iloc[0] == pytest.approx(-5.0)
