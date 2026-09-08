"""ML layer: causality, leakage, and the anti-self-deception machinery."""
import numpy as np
import pandas as pd
import pytest

from bipbip.core import BacktestEngine, CashAccount, CostModel
from bipbip.core import indicators as ind
from bipbip.data import make_intraday_bars
from bipbip.ml import MODELS, build_dataset, build_features, permutation_test
from bipbip.ml.features import FEATURE_COLUMNS
from bipbip.ml.labels import triple_barrier_labels
from bipbip.ml.validation import PurgedWalkForward, walk_forward_evaluate
from bipbip.strategies.ml_strategy import MLStrategy


# -- causality ---------------------------------------------------------------

@pytest.mark.parametrize("column", FEATURE_COLUMNS)
def test_features_are_causal(column):
    """Every feature at bar i must depend only on bars <= i.

    One non-causal feature silently invalidates every result downstream, so
    each column is checked individually rather than in aggregate.
    """
    bars = make_intraday_bars(n_sessions=6, seed=91)
    cut = int(len(bars) * 0.7)

    full = build_features(bars)[column].iloc[:cut]
    truncated = build_features(bars.iloc[:cut])[column]

    pd.testing.assert_series_equal(full, truncated, check_exact=False, rtol=1e-9)


def test_labels_never_cross_a_session_boundary():
    """A label resolved by tomorrow's bars is unusable: the engine force-flats."""
    bars = make_intraday_bars(n_sessions=5, seed=92)
    lab = triple_barrier_labels(bars, ind.atr(bars, 30))
    days = np.asarray([ts.date() for ts in bars.index])

    resolved = lab["event_end"].to_numpy().astype(int)
    for i in range(len(bars) - 1):
        if np.isfinite(lab["label"].to_numpy()[i]):
            assert days[resolved[i]] == days[i], f"label at {bars.index[i]} escaped its session"


def test_labels_are_cost_aware():
    """A 'win' must clear the round trip, or it is a loss wearing a disguise."""
    bars = make_intraday_bars(n_sessions=3, seed=93)
    atr = ind.atr(bars, 30)
    expensive = triple_barrier_labels(bars, atr, cost_bps=200.0)
    cheap = triple_barrier_labels(bars, atr, cost_bps=0.1)
    # Raising costs cannot create winners.
    assert expensive["label"].mean() <= cheap["label"].mean()
    assert expensive["ret"].mean() < cheap["ret"].mean()


# -- leakage -----------------------------------------------------------------

def test_purging_removes_every_overlapping_training_sample():
    """The core anti-leakage guarantee.

    A training sample whose label resolves inside the validation window has
    effectively seen validation data. Without purging, a model with no edge
    routinely reports an encouraging AUC.
    """
    n, embargo = 3000, 60
    rng = np.random.default_rng(0)
    event_end = np.arange(n) + rng.integers(1, 120, n)
    event_end = np.minimum(event_end, n - 1)

    splitter = PurgedWalkForward(n_splits=4, embargo_bars=embargo, min_train=1000)
    splits = list(splitter.split(n, event_end))
    assert splits, "expected usable folds"

    for train_idx, val_idx in splits:
        val_start = val_idx[0]
        assert train_idx.max() < val_start, "training data must precede validation"
        assert (event_end[train_idx] < val_start - embargo).all(), "leaked label window"


def test_splits_are_chronological_and_never_shuffled():
    n = 3000
    event_end = np.arange(n)
    for train_idx, val_idx in PurgedWalkForward(4, 10, 1000).split(n, event_end):
        assert (np.diff(train_idx) > 0).all()
        assert (np.diff(val_idx) > 0).all()
        assert train_idx.max() < val_idx.min()


def test_validation_refuses_to_run_on_too_little_history():
    """Silence is the wrong response to an inadequate sample."""
    with pytest.raises(ValueError, match="not enough history"):
        list(PurgedWalkForward(min_train=500).split(300, np.arange(300)))


# -- self-deception guards ---------------------------------------------------

def test_permutation_test_rejects_a_model_fitted_to_pure_noise():
    """Regression test for a real false positive.

    An earlier verdict threshold called random labels 'worth a second look'.
    Labels here are pure coin flips, so any verdict claiming significance is
    the tool failing at its one job.
    """
    bars = make_intraday_bars(n_sessions=30, seed=94)
    ds = build_dataset(bars)
    rng = np.random.default_rng(5)
    noise_y = rng.integers(0, 2, len(ds)).astype("float64")

    result = permutation_test(
        MODELS["logistic"], ds.X, noise_y, ds.rets, ds.event_end,
        n_permutations=10, n_splits=3, min_train=1500, threshold=0.55,
    )
    assert "survives" not in result["verdict"], result["verdict"]


def test_verdict_is_withheld_when_too_few_trades():
    """A handful of out-of-sample trades cannot support any conclusion,
    however good the p-value looks."""
    bars = make_intraday_bars(n_sessions=30, seed=95)
    ds = build_dataset(bars)
    result = permutation_test(
        MODELS["logistic"], ds.X, ds.y, ds.rets, ds.event_end,
        n_permutations=5, n_splits=3, min_train=1500,
        threshold=0.99,  # so selective almost nothing is taken
    )
    if result.get("n_trades", 0) < 30:
        assert "inconclusive" in result["verdict"]


def test_baselines_lose_exactly_the_costs_on_random_data():
    """Sanity anchor: on a random walk, trading everything loses the spread.
    If this ever comes out positive, the cost model has broken."""
    bars = make_intraday_bars(n_sessions=40, seed=96)
    ds = build_dataset(bars, cost_bps=2.3)
    r = walk_forward_evaluate(MODELS["always_enter"], ds.X, ds.y, ds.rets,
                              ds.event_end, n_splits=3, min_train=2000, threshold=0.5)
    assert r["overall"]["mean_ret_bps"] < 0


# -- engine integration ------------------------------------------------------

def test_ml_strategy_obeys_the_same_engine_invariants():
    """A model gets no exemption from settlement rules or the closing bell."""
    bars = make_intraday_bars(n_sessions=20, seed=97)
    ds = build_dataset(bars)
    model = MODELS["logistic"]().fit(ds.X.to_numpy(dtype="float64"), ds.y)

    acct = CashAccount(starting_equity=10_000.0)
    res = BacktestEngine(acct, CostModel()).run("SPY", bars, MLStrategy(model, threshold=0.5))

    per_day = {}
    for t in res.trades:
        assert t.entry_time.date() == t.exit_time.date(), "held overnight"
        per_day[t.entry_time.date()] = per_day.get(t.entry_time.date(), 0) + 1
    assert not per_day or max(per_day.values()) <= 1, "broke one round trip per session"
    assert acct.position.shares >= 0


def test_ml_strategy_scores_causally_through_the_engine():
    """Scoring every bar up front is a performance choice, not a peek."""
    bars = make_intraday_bars(n_sessions=10, seed=98)
    ds = build_dataset(bars)
    model = MODELS["logistic"]().fit(ds.X.to_numpy(dtype="float64"), ds.y)
    cut = int(len(bars) * 0.6)

    tampered = bars.copy()
    for col in ("open", "high", "low", "close"):
        tampered.iloc[cut:, tampered.columns.get_loc(col)] *= 1.8

    a = BacktestEngine(CashAccount(10_000.0), CostModel()).run("SPY", bars, MLStrategy(model))
    b = BacktestEngine(CashAccount(10_000.0), CostModel()).run("SPY", tampered, MLStrategy(model))

    ts = bars.index[cut]
    pd.testing.assert_series_equal(
        a.equity_curve[a.equity_curve.index < ts],
        b.equity_curve[b.equity_curve.index < ts],
        check_exact=False, rtol=1e-9,
    )
