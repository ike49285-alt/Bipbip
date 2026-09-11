"""Tests for barrier resolution.

The positive controls matter more than usual here: a resolver that never fires
returns timeouts for everything, which looks exactly like a working resolver on
a quiet series. Each case plants an outcome and checks it is recovered.

MUTATION COVERAGE. This module produces the labels behind the panel result, so
it was mutation-tested: break the source, confirm something goes red. The first
run killed 19 of 57 mutants - every comparison in `first_touch`, the entire
barrier-distance formula, the basis-point conversion and the entry offset could
each be changed with the file still green, because every case called
`first_touch` directly with hand-written absolute levels and cleared its
barrier by a wide margin. It now kills 38.

The other 19 are EQUIVALENT MUTANTS, not gaps, and are listed here so they are
not chased again:

  line 25    TARGET/STOP/UNRESOLVED values. Compared by imported name
             everywhere, so any distinct values work; the collision that WOULD
             matter (TIMEOUT becoming 1) is caught by the distinctness test.
  lines 41-43  `max_hold + 1` and `-1` sentinels. Any value still above
             max_hold, or still below zero, means the same thing.
  line 47    the session lookup is clamped behind an `ok` mask that already
             requires `j < n`, so the clamp never decides anything.
  line 50    `side > 0` vs `>= 0`: side is only ever +1 or -1.
  lines 56-57  the first-touch latch initialises above max_hold and the loop
             ends at max_hold, so `>` and `>=` cannot diverge.
  lines 68-74  `last < 0` vs `<= 0`: last is -1 or k >= 1, never 0. The clip
             bounds only bind on rows that are already NaN.
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


# ---------------------------------------------------------------------------
# Boundary conditions. Every case above clears its barrier by a wide margin
# (103 against a target of 102), so none of them can tell >= from >. Mutation
# testing found every comparison in first_touch unconstrained: nine operators
# could each be flipped with the whole file still green.
# ---------------------------------------------------------------------------

def test_a_high_exactly_at_the_target_counts_as_a_touch():
    """A limit order resting AT the target fills when the print is at it, so
    the boundary is inclusive. Off by one tick here moves every label."""
    b = _bars([100, 100, 100, 100], highs=[100, 100, 102.0, 100])
    _, which, _ = _call(b, target=102.0, stop=98.0, max_hold=3)
    assert which[0] == TARGET


def test_a_high_a_tick_below_the_target_is_not_a_touch():
    b = _bars([100, 100, 100, 100], highs=[100, 100, 101.99, 100])
    _, which, _ = _call(b, target=102.0, stop=98.0, max_hold=3)
    assert which[0] == TIMEOUT


def test_a_low_exactly_at_the_stop_counts_as_a_touch():
    b = _bars([100, 100, 100, 100], lows=[100, 100, 98.0, 100])
    _, which, _ = _call(b, target=102.0, stop=98.0, max_hold=3)
    assert which[0] == STOP


def test_a_low_a_tick_above_the_stop_is_not_a_touch():
    b = _bars([100, 100, 100, 100], lows=[100, 100, 98.01, 100])
    _, which, _ = _call(b, target=102.0, stop=98.0, max_hold=3)
    assert which[0] == TIMEOUT


def test_a_short_touches_its_barriers_on_the_opposite_sides():
    """For a short the target is BELOW and the stop ABOVE, so both comparisons
    invert. Tested at the boundary for the same reason as the long."""
    hit_t = _bars([100, 100, 100, 100], lows=[100, 100, 98.0, 100])
    _, which, _ = _call(hit_t, target=98.0, stop=102.0, max_hold=3, side=-1)
    assert which[0] == TARGET

    hit_s = _bars([100, 100, 100, 100], highs=[100, 100, 102.0, 100])
    _, which, _ = _call(hit_s, target=98.0, stop=102.0, max_hold=3, side=-1)
    assert which[0] == STOP


def test_a_barrier_touched_exactly_on_the_last_allowed_bar_still_counts():
    """max_hold is inclusive: the loop runs to max_hold, so a touch on that
    bar resolves rather than timing out. One off here silently shortens every
    holding period in the panel."""
    b = _bars([100, 100, 100, 102.0, 100], highs=[100, 100, 100, 102.0, 100])
    _, which, held = _call(b, target=102.0, stop=98.0, max_hold=3)
    assert which[0] == TARGET and held[0] == 3

    # ...and one bar further out does not, at the same max_hold.
    b2 = _bars([100, 100, 100, 100, 102.0], highs=[100, 100, 100, 100, 102.0])
    _, which2, _ = _call(b2, target=102.0, stop=98.0, max_hold=3)
    assert which2[0] == TIMEOUT


# ---------------------------------------------------------------------------
# barrier_labels itself. Every test above calls first_touch DIRECTLY with
# hand-written absolute levels, so the function that turns an ATR into those
# levels was exercised by exactly one test - on a flat series, where the
# return is zero and `r * 1e4` and `r / 1e4` are indistinguishable.
# ---------------------------------------------------------------------------

def test_the_target_level_is_entry_plus_target_mult_times_atr():
    """Planted so the high sits a tick either side of the implied level.

    entry = next open = 100, atr = 2, target_mult = 1.5 -> target = 103.
    """
    b = _bars([100, 100, 100, 100, 100], highs=[100, 100, 102.99, 100, 100],
              opens=[100, 100, 100, 100, 100])
    atr = pd.Series(2.0, index=b.index)
    inside = barrier_labels(b, atr, target_mult=1.5, stop_mult=1.0,
                            max_hold=3, cost_bps=0.0)
    assert inside["long_barrier"].iloc[0] == TIMEOUT     # 102.99 < 103

    b2 = _bars([100, 100, 100, 100, 100], highs=[100, 100, 103.0, 100, 100],
               opens=[100, 100, 100, 100, 100])
    out = barrier_labels(b2, atr, target_mult=1.5, stop_mult=1.0,
                         max_hold=3, cost_bps=0.0)
    assert out["long_barrier"].iloc[0] == TARGET         # 103.0 == 103


def test_the_stop_level_is_entry_minus_stop_mult_times_atr():
    """entry = 100, atr = 2, stop_mult = 1.0 -> stop = 98."""
    b = _bars([100, 100, 100, 100, 100], lows=[100, 100, 98.01, 100, 100],
              opens=[100, 100, 100, 100, 100])
    atr = pd.Series(2.0, index=b.index)
    assert barrier_labels(b, atr, target_mult=1.5, stop_mult=1.0, max_hold=3,
                          cost_bps=0.0)["long_barrier"].iloc[0] == TIMEOUT

    b2 = _bars([100, 100, 100, 100, 100], lows=[100, 100, 98.0, 100, 100],
               opens=[100, 100, 100, 100, 100])
    assert barrier_labels(b2, atr, target_mult=1.5, stop_mult=1.0, max_hold=3,
                          cost_bps=0.0)["long_barrier"].iloc[0] == STOP


def test_a_wider_multiple_moves_the_barrier_further_away():
    """The multiplier must SCALE the distance, not merely be present. The same
    bars resolve as a hit at 1.5x and a timeout at 3x."""
    b = _bars([100, 100, 100, 100, 100], highs=[100, 100, 103.0, 100, 100],
              opens=[100, 100, 100, 100, 100])
    atr = pd.Series(2.0, index=b.index)
    near = barrier_labels(b, atr, target_mult=1.5, max_hold=3, cost_bps=0.0)
    far = barrier_labels(b, atr, target_mult=3.0, max_hold=3, cost_bps=0.0)
    assert near["long_barrier"].iloc[0] == TARGET        # 103 >= 103
    assert far["long_barrier"].iloc[0] == TIMEOUT        # 103 <  106


def test_the_return_is_reported_in_basis_points():
    """The UNIT of the headline number. A 3% move is 300 bps, and the one
    existing barrier_labels test ran on a flat series where the return is zero
    and the scaling could not be observed at all."""
    b = _bars([100, 100, 103.0, 103.0, 103.0],
              highs=[100, 100, 103.0, 103.0, 103.0],
              opens=[100, 100, 100, 103.0, 103.0])
    atr = pd.Series(2.0, index=b.index)
    out = barrier_labels(b, atr, target_mult=1.5, stop_mult=1.0, max_hold=3,
                         cost_bps=0.0)
    # entry 100, target 103, touched -> +3% -> +300 bps
    assert out["long_ret"].iloc[0] == pytest.approx(300.0, abs=1e-6)


def test_cost_is_subtracted_in_the_same_units_as_the_return():
    """Charging cost in percent against a return in bps would be a 10,000x
    error and still look like a plausible small number."""
    b = _bars([100, 100, 103.0, 103.0, 103.0],
              highs=[100, 100, 103.0, 103.0, 103.0],
              opens=[100, 100, 100, 103.0, 103.0])
    atr = pd.Series(2.0, index=b.index)
    free = barrier_labels(b, atr, max_hold=3, cost_bps=0.0)["long_ret"].iloc[0]
    charged = barrier_labels(b, atr, max_hold=3,
                             cost_bps=3.10)["long_ret"].iloc[0]
    assert free - charged == pytest.approx(3.10)


def test_entry_is_the_next_bars_open():
    """Not this bar's, and not two bars out. A drifting entry would change
    every label in the panel without changing any test."""
    b = _bars([100, 100, 100, 100, 100],
              highs=[100, 100, 100, 100, 100],
              opens=[999.0, 100.0, 555.0, 100.0, 100.0])
    atr = pd.Series(2.0, index=b.index)
    out = barrier_labels(b, atr, target_mult=1.5, stop_mult=1.0, max_hold=3,
                         cost_bps=0.0)
    # Row 0 must fill at bar 1's open (100), so a flat tape returns 0. Filling
    # at bar 0's 999 or bar 2's 555 would be a large negative instead.
    assert out["long_ret"].iloc[0] == pytest.approx(0.0, abs=1e-9)


def test_the_documented_defaults_are_the_ones_in_the_signature():
    """max_hold=26 is one 30-minute session, and the panel's holding period is
    quoted from it. cost_bps=3.10 is the figure CLAUDE.md records."""
    import inspect
    d = {p.name: p.default
         for p in inspect.signature(barrier_labels).parameters.values()}
    assert d["max_hold"] == 26
    assert d["target_mult"] == 1.5
    assert d["stop_mult"] == 1.0
    assert d["cost_bps"] == 3.10


def test_the_four_outcome_sentinels_are_distinct():
    """They are compared by identity everywhere, so a collision would be
    invisible: setting TIMEOUT to 1 makes every timeout read as a TARGET, and
    every test that imports the names still passes."""
    assert len({TARGET, STOP, TIMEOUT, UNRESOLVED}) == 4


def test_a_stop_touched_exactly_on_the_last_allowed_bar_still_resolves():
    """The target's boundary is covered above; the stop has its own comparison
    and its own `<= max_hold` resolution test, which flipping to `<` would
    silently turn into a timeout."""
    b = _bars([100, 100, 100, 98.0, 100], lows=[100, 100, 100, 98.0, 100])
    _, which, held = _call(b, target=102.0, stop=98.0, max_hold=3)
    assert which[0] == STOP and held[0] == 3


def test_a_non_unit_stop_multiple_scales_the_stop_distance():
    """Every other stop test uses stop_mult=1.0, where multiplying and
    dividing by it are the same thing - so the multiplier itself was never
    constrained. entry 100, atr 2, stop_mult 2.0 -> stop at 96."""
    b = _bars([100, 100, 100, 100, 100], lows=[100, 100, 96.5, 100, 100],
              opens=[100, 100, 100, 100, 100])
    atr = pd.Series(2.0, index=b.index)
    tight = barrier_labels(b, atr, stop_mult=1.0, max_hold=3, cost_bps=0.0)
    wide = barrier_labels(b, atr, stop_mult=2.0, max_hold=3, cost_bps=0.0)
    assert tight["long_barrier"].iloc[0] == STOP       # 96.5 <= 98
    assert wide["long_barrier"].iloc[0] == TIMEOUT     # 96.5 >  96


def test_the_short_leg_uses_the_opposite_side_not_a_larger_one():
    """`side` enters the barrier distance as a multiplier, so a short written
    as -2 rather than -1 would place both barriers twice as far out and read
    as a quieter instrument rather than an error."""
    # entry 100, atr 2, target_mult 1.5 -> short target at 97, not 94.
    b = _bars([100, 100, 100, 100, 100], lows=[100, 100, 97.0, 100, 100],
              opens=[100, 100, 100, 100, 100])
    atr = pd.Series(2.0, index=b.index)
    out = barrier_labels(b, atr, target_mult=1.5, stop_mult=1.0, max_hold=3,
                         cost_bps=0.0)
    assert out["short_barrier"].iloc[0] == TARGET
    assert out["short_ret"].iloc[0] == pytest.approx(300.0, abs=1e-6)


def test_an_unresolvable_row_returns_nan_and_zero_held_not_a_number():
    """A row with no future bar in its own session has no trade. Returning 0.0
    instead of NaN would put a real zero into the panel's mean; returning a
    non-zero `held` would corrupt the holding-period statistics."""
    b = _bars([100, 100, 100])
    ret, which, held = _call(b, target=102.0, stop=98.0, max_hold=3)
    assert which[-1] == UNRESOLVED
    assert np.isnan(ret[-1])
    assert held[-1] == 0


def test_a_timeout_exit_reads_the_close_of_the_bar_it_was_held_to():
    """The exit index is entry bar + bars held, clipped into range. An
    off-by-one reads the wrong close and a wrong clip reads the last bar of
    the array - both plausible-looking numbers."""
    b = _bars([100, 100, 101.0, 107.0, 100])
    ret, which, held = _call(b, target=999.0, stop=1.0, max_hold=2)
    assert which[0] == TIMEOUT and held[0] == 2
    # entry at bar 1's open (100), held 2 bars -> close of bar 2, which is 101.
    assert ret[0] == pytest.approx(101.0 / 100.0 - 1.0)


def test_a_barrier_touched_on_the_very_first_bar_after_entry_resolves():
    """An immediate stop-out is the most consequential case there is, and no
    test planted one: every case above resolves on bar 2 or later, so the
    scan's starting index was unconstrained and could skip k=1 entirely.
    """
    # Row 0 fills at bar 1's OPEN, so k=1 is bar 1 itself: the touch has to
    # sit there, not at bar 2, or the case being tested is k=2 again.
    b = _bars([100, 100, 100, 100, 100], lows=[100, 98.0, 100, 100, 100],
              opens=[100, 100, 100, 100, 100])
    _, which, held = _call(b, target=102.0, stop=98.0, max_hold=3)
    assert which[0] == STOP and held[0] == 1

    up = _bars([100, 100, 100, 100, 100], highs=[100, 102.0, 100, 100, 100],
               opens=[100, 100, 100, 100, 100])
    _, which_up, held_up = _call(up, target=102.0, stop=98.0, max_hold=3)
    assert which_up[0] == TARGET and held_up[0] == 1


def test_the_default_side_is_long():
    """Every call in this file passes `side` explicitly, so the default was
    never exercised - and it scales the realised return, so a wrong one
    doubles or inverts every label rather than failing loudly."""
    b = _bars([100, 100, 103.0, 100, 100], highs=[100, 100, 103.0, 100, 100],
              opens=[100, 100, 100, 100, 100])
    entry = b["open"].shift(-1).to_numpy()
    sess = pd.DatetimeIndex(b.index).normalize().view("int64")
    ret, which, _ = first_touch(
        b["high"].to_numpy(), b["low"].to_numpy(), b["close"].to_numpy(),
        entry, np.full(len(b), 102.0), np.full(len(b), 98.0), 3, sess)
    assert which[0] == TARGET
    assert ret[0] == pytest.approx(102.0 / 100.0 - 1.0)
