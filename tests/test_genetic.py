"""The genetic rule search, and the null that makes its output mean anything."""
import numpy as np
import pytest

from bipbip.ml.genetic import Rule, RuleSearch, evolve_with_null


def _planted(n=20000, seed=0, edge_bps=30.0):
    """Feature 0 above its 80th percentile pays; everything else is noise."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6))
    base = rng.normal(0.0, 60.0, n)
    fwd = np.where(X[:, 0] > 0.84, base + edge_bps, base)
    return X, {5: fwd, 15: fwd.copy()}, [f"f{i}" for i in range(6)]


def test_the_search_finds_a_planted_edge():
    X, fwd, names = _planted()
    s = RuleSearch(X, fwd, names, min_trades=100, seed=1)
    best, fit, _ = s.evolve(population=30, generations=12)
    assert fit > 4.0, f"failed to find a planted 30 bps edge (t={fit:.2f})"
    # It should be conditioning on feature 0, which is where the edge lives.
    assert any(i == 0 for i, _, _ in best.conditions), best.describe(names)
    assert best.direction == 1


def test_the_search_overfits_pure_noise_and_the_null_measures_it():
    """The reason every GA result here carries a null.

    With no signal at all the search still returns a winner, and its fitness is
    large - that is what searching thousands of rules buys you. A GA reporting
    only its winner has reported the size of its own search.
    """
    rng = np.random.default_rng(3)
    X = rng.normal(size=(8000, 6))
    fwd = {5: rng.normal(0.0, 60.0, 8000)}
    s = RuleSearch(X, fwd, [f"f{i}" for i in range(6)], min_trades=100, seed=2)
    _, fit, _ = s.evolve(population=30, generations=12)
    assert fit > 1.5, "a search over noise should still find something flattering"


def test_a_planted_edge_beats_its_own_null():
    X, fwd, names = _planted(seed=5)
    res = evolve_with_null(RuleSearch(X, fwd, names, min_trades=100, seed=5),
                           population=30, generations=10, null_runs=5, seed=5)
    assert res.fitness > res.null_best, (
        f"real t={res.fitness:.2f} did not beat null best {res.null_best:.2f}")
    assert res.p_value <= 0.2


def test_pure_noise_does_not_beat_its_own_null():
    """The test that matters: no signal must not produce a significant result."""
    rng = np.random.default_rng(11)
    X = rng.normal(size=(8000, 6))
    fwd = {5: rng.normal(0.0, 60.0, 8000)}
    res = evolve_with_null(RuleSearch(X, fwd, [f"f{i}" for i in range(6)],
                                      min_trades=100, seed=11),
                           population=25, generations=8, null_runs=8, seed=11)
    assert res.p_value > 0.05, (
        f"a search over pure noise reported p={res.p_value:.3f}; the null is "
        "not doing its job")


def test_fitness_is_a_t_statistic_not_a_mean_return():
    """Selecting on average return breeds rules that fire nine times.

    A rule with an enormous mean over a handful of trades is not a strategy,
    and a fitness function that rewards it will fill the population with them.
    """
    rng = np.random.default_rng(7)
    X = rng.normal(size=(5000, 4))
    fwd = {5: rng.normal(0.0, 50.0, 5000)}
    s = RuleSearch(X, fwd, [f"f{i}" for i in range(4)], min_trades=500, seed=7)

    rare = Rule([(0, ">", 0.95), (1, ">", 0.95), (2, ">", 0.95)], 1, 5)
    fit, n, _, _ = s.evaluate(rare)
    assert n < 500 and fit == -np.inf, "a rule firing too rarely must score -inf"


def test_thresholds_are_percentiles_so_features_are_comparable():
    """Raw thresholds would make one gene mean different things per feature."""
    X = np.column_stack([np.arange(1000.0), np.arange(1000.0) * 1e6])
    fwd = {5: np.zeros(1000)}
    s = RuleSearch(X, fwd, ["small", "huge"], min_trades=1, seed=0)
    a = s.mask(Rule([(0, ">", 0.5)], 1, 5)).sum()
    b = s.mask(Rule([(1, ">", 0.5)], 1, 5)).sum()
    assert abs(int(a) - int(b)) < 20, "the same percentile gene selected "\
        "very different fractions of two features"


def test_a_rule_describes_itself_in_words():
    _, _, names = _planted()
    r = Rule([(0, ">", 0.8), (2, "<", 0.2)], -1, 15)
    text = r.describe(names)
    assert "SHORT" in text and "f0 > p80" in text and "f2 < p20" in text
