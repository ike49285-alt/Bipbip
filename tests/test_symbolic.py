"""Tests for the symbolic alpha search.

The guards are the module. An expression search without them finds an edge in
anything, so each one gets a test that fails if it is removed.
"""
import numpy as np

from bipbip.ml.symbolic import (AlphaSearch, Node, clone, crossover, evaluate,
                                mutate, random_tree)


def _panel(T=400, N=6, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, .002, (T, N)), axis=0))
    # The bar range has to vary per symbol and per bar. A fixed high/low ratio
    # makes every candle the same shape, so (high - low) / close is a single
    # constant across the whole panel - and a "planted signal" built from it
    # vanishes entirely under cross-sectional demeaning, failing the positive
    # control for a reason that has nothing to do with the search.
    up = np.abs(rng.normal(0, .004, (T, N)))
    dn = np.abs(rng.normal(0, .004, (T, N)))
    return {"open": close * (1 + rng.normal(0, .0005, (T, N))),
            "high": close * (1 + up), "low": close * (1 - dn),
            "close": close, "volume": rng.uniform(1e5, 5e5, (T, N))}


def _leaf(name):
    return Node("base", name=name)


def test_evaluate_base_and_const():
    d = _panel(20, 3)
    assert np.allclose(evaluate(_leaf("close"), d), d["close"])
    assert np.allclose(evaluate(Node("const", value=2.5), d), 2.5)


def test_evaluate_unary_and_binary():
    d = _panel(60, 3)
    m = evaluate(Node("unary", name="mean", window=5, kids=[_leaf("close")]), d)
    assert np.isnan(m[:4]).all() and np.isfinite(m[10]).all()
    s = evaluate(Node("binary", name="-", kids=[_leaf("high"), _leaf("low")]), d)
    assert (s > 0).all()


def test_safe_division_never_explodes():
    """An unguarded ratio wins by finding a near-zero denominator on a few bars."""
    d = _panel(50, 3)
    d["volume"][10, 1] = 0.0
    d["volume"][20, 2] = 1e-15
    out = evaluate(Node("binary", name="/", kids=[_leaf("close"), _leaf("volume")]), d)
    assert not np.isinf(out).any()
    assert np.isnan(out[10, 1]) and np.isnan(out[20, 2])


def test_forward_returns_are_demeaned_across_symbols():
    """Without this the winner is always 'be long', which is the market."""
    d = _panel(200, 5)
    fwd = np.full((200, 5), 0.01)          # every symbol up 1%: pure market
    ga = AlphaSearch(d, fwd, np.zeros(5), horizon=5)
    assert np.allclose(np.nan_to_num(ga.fwd), 0.0)


def test_cost_is_charged_after_the_direction():
    """A book with zero gross must lose exactly the cost, never earn it."""
    d = _panel(200, 6)
    fwd = np.zeros((200, 6))
    cost = np.full(6, 2.0)
    ga = AlphaSearch(d, fwd, cost, horizon=5, k=2)
    v = np.tile(np.arange(6.0), (200, 1))
    pnl = ga.book(v, np.arange(0, 200, 5))
    assert len(pnl) > 10
    assert np.allclose(pnl, -4.0)          # 2 bps per leg, both legs pay


def test_a_constant_expression_is_rejected():
    """It ranks every symbol equally; its 'book' is tie ordering, not a decision."""
    d = _panel(300, 6)
    rng = np.random.default_rng(1)
    fwd = rng.normal(0, .002, (300, 6))
    ga = AlphaSearch(d, fwd, np.zeros(6), horizon=5, k=2)
    assert ga.fitness(Node("const", value=1.0), np.arange(0, 300, 5)) == -np.inf


def test_oversized_trees_are_rejected():
    d = _panel(200, 6)
    ga = AlphaSearch(d, np.zeros((200, 6)), np.zeros(6), horizon=5, max_size=5)
    big = _leaf("close")
    for _ in range(8):
        big = Node("binary", name="+", kids=[big, _leaf("high")])
    assert ga.fitness(big, np.arange(0, 200, 5)) == -np.inf


def test_book_skips_timestamps_with_too_few_live_symbols():
    d = _panel(120, 6)
    fwd = np.full((120, 6), np.nan)
    fwd[::5, :2] = 0.01                    # only two symbols ever priced
    ga = AlphaSearch(d, fwd, np.zeros(6), horizon=5, k=3)
    assert len(ga.book(np.tile(np.arange(6.0), (120, 1)), np.arange(0, 120, 5))) == 0


def test_mutate_and_crossover_return_evaluable_trees():
    d = _panel(120, 4)
    rng = np.random.default_rng(3)
    for i in range(25):
        a, b = random_tree(rng, 2), random_tree(rng, 2)
        for child in (mutate(a, rng), crossover(a, b, rng)):
            out = evaluate(child, d)
            assert out.shape == d["close"].shape


def test_clone_is_deep():
    a = Node("binary", name="+", kids=[_leaf("close"), _leaf("high")])
    b = clone(a)
    b.kids[0].name = "volume"
    assert a.kids[0].name == "close"


def test_evolution_finds_a_planted_signal():
    """Sanity: if a real relationship exists, the search must actually find it."""
    d = _panel(600, 6, seed=7)
    rng = np.random.default_rng(9)
    # Forward return follows the high/low spread, plus noise.
    signal = (d["high"] - d["low"]) / d["close"]
    fwd = (signal - signal.mean()) * 4.0 + rng.normal(0, 1e-4, signal.shape)
    ga = AlphaSearch(d, fwd, np.zeros(6), horizon=5, k=2, seed=0)
    rows = np.arange(60, 600, 5)
    _, fit = ga.evolve(rows, generations=6, population=40, elite=6)
    assert fit > 3.0, f"failed to recover a planted signal (fitness {fit:.2f})"
