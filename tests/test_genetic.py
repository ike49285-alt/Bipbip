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


def test_always_trading_scores_zero_after_demeaning():
    """The bug the first real run walked straight into.

    Forward returns have a non-zero unconditional mean, so a rule firing on
    most of the sample inherits it without predicting anything. The first GA
    run returned a rule covering 73% of the tape and scored t=675; shuffling
    does not change an unconditional mean, so the null scored t=560 and
    correctly refused to call it significant. Demeaning makes that whole
    strategy worth exactly nothing.
    """
    rng = np.random.default_rng(21)
    n = 20000
    X = rng.normal(size=(n, 5))
    # A strong upward drift and no relationship to any feature whatsoever.
    fwd = {5: rng.normal(40.0, 60.0, n)}
    s = RuleSearch(X, fwd, [f"f{i}" for i in range(5)], min_trades=100,
                   seed=21, max_fraction=1.0)
    everything = Rule([(0, ">", 0.05)], 1, 5)
    _, n_fired, mean_bps, _ = s.evaluate(everything)
    assert n_fired > n * 0.8, "fixture should fire on nearly everything"
    assert abs(mean_bps) < 3.0, (
        f"a near-universal rule kept {mean_bps:.1f} bps of drift; demeaning "
        "is not working")


def test_a_rule_covering_most_of_the_tape_is_rejected():
    """Breadth is not a strategy: cap how much a candidate may claim."""
    rng = np.random.default_rng(22)
    X = rng.normal(size=(10000, 4))
    fwd = {5: rng.normal(0.0, 50.0, 10000)}
    s = RuleSearch(X, fwd, [f"f{i}" for i in range(4)], min_trades=50,
                   seed=22, max_fraction=0.30)
    broad = Rule([(0, ">", 0.05)], 1, 5)
    fit, n_fired, _, _ = s.evaluate(broad)
    assert n_fired > 0.30 * 10000
    assert fit == -np.inf, "a rule covering most of the sample must not score"


def test_mean_is_returned_in_the_units_supplied():
    """Regression: evaluate() scaled by 1e4 on inputs already in basis points,
    and reported +82,473 bps a trade for what was +8.2."""
    # The feature needs variance or the percentile threshold selects nothing
    # and the rule is rejected for firing too rarely rather than evaluated.
    X = np.column_stack([np.arange(1000.0), np.arange(1000.0)])
    fwd = {5: np.full(1000, 25.0)}          # 25 bps, already in bps
    s = RuleSearch(X, fwd, ["a", "b"], min_trades=10, seed=0,
                   max_fraction=1.0, demean=False)
    _, _, mean_bps, _ = s.evaluate(Rule([(0, ">", 0.05)], 1, 5))
    assert mean_bps == pytest.approx(25.0), f"got {mean_bps}, expected 25 bps"


def test_cost_is_charged_after_direction_not_before():
    """The bug that produced an 'edge' at every hour of the day.

    Netting cost into the forward return and then flipping the sign for a
    short turns `gross - cost` into `-gross + cost`: the short is PAID the
    spread instead of charged it. That handed every short a 4.34 bps rebate,
    which showed up as +8.58 bps of profit in the first hour, at midday, in the
    afternoon and into the close - uniform, because a bug does not care what
    time it is. Gross open-to-close over five bars is -0.12 bps.
    """
    X = np.column_stack([np.arange(1000.0)] * 2)
    gross = {5: np.zeros(1000)}          # no edge whatsoever
    s = RuleSearch(X, gross, ["a", "b"], min_trades=10, seed=0,
                   max_fraction=1.0, demean=False,
                   cost_bps=np.full(1000, 4.3))

    for direction in (1, -1):
        _, _, mean_bps, _ = s.evaluate(Rule([(0, ">", 0.05)], direction, 5))
        assert mean_bps == pytest.approx(-4.3), (
            f"direction {direction} scored {mean_bps:+.2f} bps on zero gross; "
            "both sides must pay the spread")


def test_a_short_cannot_profit_from_a_flat_tape():
    """The same property stated as the thing a trader would notice."""
    rng = np.random.default_rng(31)
    X = rng.normal(size=(5000, 3))
    gross = {5: rng.normal(0.0, 40.0, 5000)}     # symmetric, zero mean
    s = RuleSearch(X, gross, ["a", "b", "c"], min_trades=100, seed=31,
                   max_fraction=1.0, demean=False,
                   cost_bps=np.full(5000, 5.0))
    _, _, longs, _ = s.evaluate(Rule([(0, ">", 0.05)], 1, 5))
    _, _, shorts, _ = s.evaluate(Rule([(0, ">", 0.05)], -1, 5))
    assert longs < 0 and shorts < 0, (
        f"long {longs:+.2f}, short {shorts:+.2f} - one of them is being paid "
        "to trade a coin flip")


def test_a_uniform_result_across_all_conditions_is_a_bug_signature():
    """Recorded as a diagnostic, because it is how the last two runs were caught.

    A real effect is conditional - it appears under some circumstances and not
    others. An effect present identically everywhere is arithmetic leaking, and
    the GA found two of those before it found anything else.
    """
    rng = np.random.default_rng(41)
    n = 30000
    X = rng.normal(size=(n, 4))
    gross = {5: rng.normal(0.0, 40.0, n)}
    s = RuleSearch(X, gross, [f"f{i}" for i in range(4)], min_trades=200,
                   seed=41, max_fraction=1.0, demean=False,
                   cost_bps=np.full(n, 4.0))
    # Slice the sample four ways; a correct engine loses the cost in each.
    means = []
    for lo in (0.05, 0.30, 0.55, 0.80):
        _, _, m, _ = s.evaluate(Rule([(0, ">", lo)], -1, 5))
        means.append(m)
    assert all(x < 0 for x in means), f"a slice showed free money: {means}"
    assert max(means) - min(means) < 3.0, "slices should differ only by noise"
