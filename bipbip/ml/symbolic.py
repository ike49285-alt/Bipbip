"""Evolve the indicators themselves, instead of choosing them.

Every search so far handed the model features somebody picked - a stochastic, an
Ichimoku cloud, or twelve raw candles. This searches the space those features
live in. An expression is a small tree over the raw series and a handful of
operators, so `(high - close) / (high - low)` and `close / mean(close, 20) - 1`
are simply two points in it. So is RSI, and so is every indicator anyone has
published; the search is not restricted to the ones that have names.

That freedom is the whole risk. A space this large contains an expression that
fits any set of labels, including labels with nothing in them, so three things
constrain it and none is optional:

Forward returns are demeaned across symbols at each timestamp. Without that the
winner is always "be long", which on a basket of levered index ETFs in an up
market scores brilliantly and is not a signal - a previous run of this project's
rule GA found exactly that, fired on 73% of the tape, and reported t=675.

Cost is charged after the direction is chosen, never folded into the label.

And the identical evolution is re-run against shuffled labels. A search this
flexible always finds something; the shuffled run says how large "something"
usually is when there is provably nothing there.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BASE = ("open", "high", "low", "close", "volume")
WINDOWS = (3, 5, 10, 20, 60)


class Node:
    """One node of an expression tree.

    `kind` is "base" (a raw series), "const", a unary op with a window, or a
    binary op. Kept deliberately small: deep trees fit noise and are unreadable
    afterwards, and an alpha nobody can read is an alpha nobody can check.
    """

    __slots__ = ("kind", "name", "window", "kids", "value")

    def __init__(self, kind, name=None, window=None, kids=(), value=None):
        self.kind, self.name = kind, name
        self.window, self.kids, self.value = window, list(kids), value

    def size(self) -> int:
        return 1 + sum(k.size() for k in self.kids)

    def __str__(self) -> str:
        if self.kind == "base":
            return self.name
        if self.kind == "const":
            return f"{self.value:g}"
        if self.kind == "unary":
            return f"{self.name}({self.kids[0]}, {self.window})"
        if self.kind == "cross":
            return f"{self.name}({self.kids[0]})"
        return f"({self.kids[0]} {self.name} {self.kids[1]})"


def _roll(a: np.ndarray, w: int, fn: str) -> np.ndarray:
    """Rolling op down the time axis of a (time x symbol) array."""
    df = pd.DataFrame(a)
    r = df.rolling(w, min_periods=w)
    out = {"mean": r.mean, "std": r.std, "max": r.max, "min": r.min}[fn]()
    return out.to_numpy()


UNARY = {
    "mean": lambda a, w: _roll(a, w, "mean"),
    "std": lambda a, w: _roll(a, w, "std"),
    "max": lambda a, w: _roll(a, w, "max"),
    "min": lambda a, w: _roll(a, w, "min"),
    "delta": lambda a, w: a - np.roll(a, w, axis=0),
    "zscore": lambda a, w: (a - _roll(a, w, "mean")) / (_roll(a, w, "std") + 1e-12),
}


def _cs(a: np.ndarray, how: str) -> np.ndarray:
    """Operators that compare a symbol against the others AT THE SAME INSTANT.

    Everything else in this file runs down one symbol's own history, which means
    the search had no way to express "this fund versus the rest right now" - and
    that is exactly the structure the panel test found mattered, since the edge
    appeared on levered products relative to each other and not on index funds
    at all. Real alpha expressions are mostly rank across a universe; without
    these the search could not write one.

    Row-wise, so nothing crosses time: the value for a symbol at bar i depends
    only on the other symbols at bar i.
    """
    if how == "rank":
        order = np.argsort(np.argsort(np.where(np.isfinite(a), a, -np.inf), axis=1),
                           axis=1).astype("float64")
        n = np.isfinite(a).sum(axis=1, keepdims=True).clip(min=1)
        out = order / np.maximum(n - 1, 1)
        return np.where(np.isfinite(a), out, np.nan)
    mu = np.nanmean(a, axis=1, keepdims=True)
    if how == "demean":
        return a - mu
    sd = np.nanstd(a, axis=1, keepdims=True)
    return (a - mu) / np.where(sd > 1e-12, sd, np.nan)


CROSS = {"cs_rank": lambda a: _cs(a, "rank"),
         "cs_demean": lambda a: _cs(a, "demean"),
         "cs_z": lambda a: _cs(a, "zscore")}


def _safe_div(x, y):
    """Division that cannot manufacture an outlier from a near-zero denominator.

    An unguarded ratio is the classic way a symbolic search wins: it finds a
    denominator that is almost zero on three bars, those bars dominate the
    fitness, and the expression is really an indicator variable for three dates.
    """
    out = np.divide(x, y, out=np.full_like(x, np.nan, dtype="float64"),
                    where=np.abs(y) > 1e-9)
    return out


BINARY = {
    "+": np.add, "-": np.subtract, "*": np.multiply, "/": _safe_div,
}


def evaluate(node: Node, data: dict) -> np.ndarray:
    if node.kind == "base":
        return data[node.name]
    if node.kind == "const":
        return np.full_like(data["close"], node.value, dtype="float64")
    if node.kind == "unary":
        return UNARY[node.name](evaluate(node.kids[0], data), node.window)
    if node.kind == "cross":
        return CROSS[node.name](evaluate(node.kids[0], data))
    return BINARY[node.name](evaluate(node.kids[0], data),
                             evaluate(node.kids[1], data))


def random_tree(rng, depth=2) -> Node:
    if depth <= 0 or rng.random() < 0.25:
        if rng.random() < 0.12:
            return Node("const", value=float(rng.choice([1.0, 2.0, 0.5])))
        return Node("base", name=str(rng.choice(BASE)))
    r = rng.random()
    if r < 0.22:
        return Node("cross", name=str(rng.choice(list(CROSS))),
                    kids=[random_tree(rng, depth - 1)])
    if r < 0.55:
        return Node("unary", name=str(rng.choice(list(UNARY))),
                    window=int(rng.choice(WINDOWS)),
                    kids=[random_tree(rng, depth - 1)])
    return Node("binary", name=str(rng.choice(list(BINARY))),
                kids=[random_tree(rng, depth - 1), random_tree(rng, depth - 1)])


def _nodes(node: Node, acc=None):
    acc = [] if acc is None else acc
    acc.append(node)
    for k in node.kids:
        _nodes(k, acc)
    return acc


def clone(node: Node) -> Node:
    return Node(node.kind, node.name, node.window,
                [clone(k) for k in node.kids], node.value)


def mutate(node: Node, rng, depth=2) -> Node:
    out = clone(node)
    spots = _nodes(out)
    target = spots[rng.integers(len(spots))]
    fresh = random_tree(rng, depth)
    target.kind, target.name = fresh.kind, fresh.name
    target.window, target.kids, target.value = fresh.window, fresh.kids, fresh.value
    return out


def crossover(a: Node, b: Node, rng) -> Node:
    out, donor = clone(a), clone(b)
    spots, gifts = _nodes(out), _nodes(donor)
    t = spots[rng.integers(len(spots))]
    g = gifts[rng.integers(len(gifts))]
    t.kind, t.name, t.window, t.kids, t.value = g.kind, g.name, g.window, g.kids, g.value
    return out


class AlphaSearch:
    """Evolve expressions, score them as a cross-sectional long/short book.

    Fitness is the t-statistic of net P&L rather than the mean, because an
    expression that earns its whole return on four bars is not a signal even
    when the average looks good. Trees are penalised for size so the winner is
    something a person can read and argue with.
    """

    def __init__(self, panel: dict, fwd: np.ndarray, cost_bps: np.ndarray,
                 horizon: int, k: int = 3, seed: int = 0, max_size: int = 12):
        self.data = panel
        self.horizon, self.k, self.max_size = horizon, k, max_size
        self.rng = np.random.default_rng(seed)
        self.cost = cost_bps
        # Demean across symbols at each timestamp. What survives is the part of
        # a move that is specific to the symbol; the part shared by the whole
        # basket is the market, and rediscovering the market is not alpha.
        self.fwd = fwd - np.nanmean(fwd, axis=1, keepdims=True)

    def book(self, values: np.ndarray, rows: np.ndarray) -> np.ndarray:
        """Net P&L per rebalance from going long the top k and short the bottom k."""
        out = []
        for t in rows:
            v, f = values[t], self.fwd[t]
            ok = np.isfinite(v) & np.isfinite(f)
            if ok.sum() < 2 * self.k:
                continue
            idx = np.flatnonzero(ok)
            order = idx[np.argsort(v[idx])]
            lo, hi = order[:self.k], order[-self.k:]
            # Cost is charged here, after the sides are chosen - never netted
            # into fwd, where it would credit a short instead of debiting it.
            gross = np.mean(f[hi]) - np.mean(f[lo])
            out.append(gross * 1e4 - np.mean(self.cost[hi]) - np.mean(self.cost[lo]))
        return np.asarray(out)

    def fitness(self, node: Node, rows: np.ndarray) -> float:
        if node.size() > self.max_size:
            return -np.inf
        try:
            v = evaluate(node, self.data)
        except (ValueError, FloatingPointError):
            return -np.inf
        if not np.isfinite(v).any():
            return -np.inf
        # A constant ranks every symbol equally and its "book" is an artefact of
        # tie ordering, not a decision.
        spread = np.nanstd(v, axis=1)
        if np.nanmean(spread) < 1e-12:
            return -np.inf
        pnl = self.book(v, rows)
        if len(pnl) < 30 or pnl.std(ddof=1) == 0:
            return -np.inf
        t = pnl.mean() / pnl.std(ddof=1) * np.sqrt(len(pnl))
        return float(t - 0.02 * node.size())        # parsimony

    def evolve(self, rows, generations=20, population=180, elite=12):
        pop = [random_tree(self.rng, 2) for _ in range(population)]
        best, best_fit = None, -np.inf
        for _ in range(generations):
            scored = sorted(((self.fitness(n, rows), n) for n in pop),
                            key=lambda p: -p[0])
            if scored[0][0] > best_fit:
                best_fit, best = scored[0][0], clone(scored[0][1])
            parents = [n for _, n in scored[:elite]]
            pop = [clone(p) for p in parents]
            while len(pop) < population:
                a = parents[self.rng.integers(len(parents))]
                if self.rng.random() < 0.6:
                    b = parents[self.rng.integers(len(parents))]
                    pop.append(crossover(a, b, self.rng))
                else:
                    pop.append(mutate(a, self.rng))
        return best, best_fit
