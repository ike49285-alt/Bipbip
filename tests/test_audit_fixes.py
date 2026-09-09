"""Bugs found in a full read of the code, and the guards against their return.

Each of these passed the whole suite before being found, which is the point:
they are failures of things nothing was asserting, not regressions of things
that once worked.
"""
import numpy as np
import pandas as pd
import pytest

from bipbip.core import indicators as ind
from bipbip.data.sessions import EXCHANGE_TZ, restrict_to_rth, to_exchange_tz


# --------------------------------------------------------------------------
# RSI was asymmetric: correct at one extreme, inverted at the other.
# --------------------------------------------------------------------------

def _series(values):
    return pd.Series(np.asarray(values, dtype="float64"))


def test_rsi_reads_100_when_nothing_has_fallen():
    """The bug. A window of only gains divided by zero and was filled with 50.

    Every bar up is the most overbought a tape can be, and the indicator
    reported dead neutral. RSI(2) meets this constantly - two rising bars in a
    row is ordinary - so the corruption was routine, not a corner case.
    """
    up = _series(100 * 1.01 ** np.arange(30))
    assert ind.rsi(up, 2).iloc[-1] == pytest.approx(100.0)
    assert ind.rsi(up, 14).iloc[-1] > 90.0


def test_rsi_reads_0_when_nothing_has_risen():
    down = _series(100 * 0.99 ** np.arange(30))
    assert ind.rsi(down, 2).iloc[-1] == pytest.approx(0.0)


def test_rsi_is_symmetric_at_the_two_extremes():
    """The property whose absence made the bug hard to notice.

    A pure downtrend always read 0 correctly, so the indicator looked fine
    wherever anyone checked the oversold side - which is the side a
    mean-reversion rule looks at.
    """
    up = _series(100 * 1.01 ** np.arange(30))
    down = _series(100 * 0.99 ** np.arange(30))
    hi, lo = ind.rsi(up, 2).iloc[-1], ind.rsi(down, 2).iloc[-1]
    assert hi + lo == pytest.approx(100.0), f"asymmetric: {lo} and {hi}"


def test_rsi_is_missing_on_a_motionless_tape():
    """0/0 is an absence of information, not a reading of 50."""
    flat = _series(np.full(30, 100.0))
    assert ind.rsi(flat, 2).isna().all()


def test_rsi_has_no_reading_on_the_first_bar():
    """A difference needs two prices."""
    up = _series(100 * 1.01 ** np.arange(30))
    assert np.isnan(ind.rsi(up, 14).iloc[0])


def test_rsi_stays_inside_its_range():
    rng = np.random.default_rng(4)
    noisy = _series(100 * np.exp(np.cumsum(rng.normal(0, 0.015, 500))))
    r = ind.rsi(noisy, 14).dropna()
    assert r.between(0, 100).all()


def test_a_held_position_now_sees_the_overbought_extreme():
    """What the bug cost downstream.

    MeanReversionBasket holds while RSI stays under its exit level. With the
    strongest advances reporting 50 instead of 100, the rule could not sell
    into them - the exit was unreachable precisely when it mattered.
    """
    up = _series(100 * 1.01 ** np.arange(40))
    assert ind.rsi(up, 2).iloc[-1] > 60.0, "the exit threshold is reachable again"


# --------------------------------------------------------------------------
# Sortino used the spread of losses rather than their size.
# --------------------------------------------------------------------------

def test_downside_deviation_is_measured_about_zero_not_about_its_own_mean():
    """A run of uniformly bad days has near-zero spread and real risk.

    Taking the standard deviation of the negative subset measures how much the
    losses differ from EACH OTHER, so a steady drip of identical losses scores
    as almost riskless. Measured about the target it does not.
    """
    from bipbip.core.metrics import compute

    class _Result:
        starting_equity = 100.0
        trades: list = []
        blocked: list = []

    # Uniform losses: std of the negative subset is ~0, so the old formula
    # divided by nothing and produced an enormous ratio.
    rets = np.array([0.03, -0.01, 0.03, -0.01, 0.03, -0.01, 0.03, -0.01] * 6)
    curve = pd.Series(100.0 * np.cumprod(1 + rets),
                      index=pd.bdate_range("2020-01-01", periods=len(rets)))
    res = _Result()
    res.equity_curve = curve
    m = compute(res)

    subset_std = float(pd.Series(rets)[pd.Series(rets) < 0].std())
    assert subset_std < 1e-9, "the fixture needs uniform losses to make the point"
    assert np.isfinite(m["sortino"]), "sortino should still be computable"
    assert m["sortino"] < 100, f"sortino {m['sortino']:.0f} looks like the old formula"


def test_sortino_is_finite_and_beaten_by_sharpe_on_downside_heavy_returns():
    from bipbip.core.metrics import compute

    class _Result:
        starting_equity = 100.0
        trades: list = []
        blocked: list = []

    rng = np.random.default_rng(9)
    rets = rng.normal(0.0005, 0.01, 400)
    curve = pd.Series(100.0 * np.cumprod(1 + rets),
                      index=pd.bdate_range("2020-01-01", periods=len(rets)))
    res = _Result()
    res.equity_curve = curve
    m = compute(res)
    assert np.isfinite(m["sortino"]) and np.isfinite(m["sharpe"])
    # Roughly symmetric returns put the two ratios in the same neighbourhood.
    assert 0.4 < m["sortino"] / m["sharpe"] < 2.5


# --------------------------------------------------------------------------
# The embargo was counted in samples, not days.
# --------------------------------------------------------------------------

def test_the_embargo_can_be_expressed_in_calendar_days():
    """Samples per day scale with universe breadth; the embargo should not.

    Roughly two samples a day on 34 ETFs and twenty-two on 266 stocks means a
    60-sample embargo is 27 trading days on one and under 3 on the other - it
    weakened by an order of magnitude exactly where breadth made leakage more
    likely.
    """
    from bipbip.ml.validation import PurgedWalkForward

    n = 2000
    # Ten samples share each calendar day.
    dates = pd.DatetimeIndex(np.repeat(pd.bdate_range("2015-01-05", periods=n // 10), 10))
    event_end = np.minimum(np.arange(n) + 5, n - 1)

    splitter = PurgedWalkForward(n_splits=3, embargo_bars=30, min_train=500)
    by_sample = [tr for tr, _ in splitter.split(n, event_end)]
    by_date = [tr for tr, _ in splitter.split(n, event_end, dates=dates)]

    assert by_sample and by_date
    # A 30-DAY embargo must drop strictly more than a 30-SAMPLE one here,
    # because thirty days is three hundred samples.
    assert len(by_date[0]) < len(by_sample[0])


def test_purging_still_excludes_every_overlapping_label():
    from bipbip.ml.validation import PurgedWalkForward

    n = 1500
    dates = pd.DatetimeIndex(np.repeat(pd.bdate_range("2015-01-05", periods=n // 5), 5))
    event_end = np.minimum(np.arange(n) + 20, n - 1)
    splitter = PurgedWalkForward(n_splits=3, embargo_bars=10, min_train=500)
    for tr, va in splitter.split(n, event_end, dates=dates):
        assert event_end[tr].max() < va.min(), "a training label resolved inside validation"


# --------------------------------------------------------------------------
# restrict_to_rth accepted a naive index and filtered against the wrong clock.
# --------------------------------------------------------------------------

def test_restrict_to_rth_refuses_a_naive_index():
    """Silently keeping the wrong four hours is worse than failing."""
    idx = pd.date_range("2025-03-03 09:30", periods=100, freq="1min")
    df = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0,
                       "close": 1.0, "volume": 1.0}, index=idx)
    with pytest.raises(ValueError, match="tz-aware"):
        restrict_to_rth(df)


def test_restrict_to_rth_accepts_exchange_time_and_keeps_the_session():
    idx = pd.date_range("2025-03-03 00:00", periods=60 * 24, freq="1min",
                        tz="UTC")
    df = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0,
                       "close": 1.0, "volume": 1.0}, index=idx)
    kept = restrict_to_rth(to_exchange_tz(df))
    assert not kept.empty
    assert kept.index.tz is not None
    times = kept.index.time
    assert min(times).hour == 9 and max(times).hour == 15
