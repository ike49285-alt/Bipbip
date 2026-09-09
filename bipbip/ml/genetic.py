"""A genetic search over trading rules, with its own noise floor attached.

A GA is a search, and a search over N candidates finds the best of N even when
none of them is real: the top of 1,000 random rules scores about t=3.1 on pure
noise, and the top of 10,000 about t=3.7. So a GA that reports its winner's
fitness has reported nothing. The number that means something is the winner's
fitness MINUS what the identical search achieves on data whose labels have been
destroyed.

That is why `evolve` takes a `null_runs` argument and why the result carries
`p_value`. Running the whole GA - same population, same generations, same
operators - against shuffled forward returns measures exactly the overfitting
the search induces, and nothing else in this module is trustworthy without it.

Rules are deliberately legible. Each individual is a handful of feature
thresholds combined with AND, a direction, and a holding period, so a survivor
can be read as a sentence rather than a weight matrix. If the GA finds
something, the point is to understand it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: Comparison operators a gene may use.
OPS = ("<", ">")


@dataclass
class Rule:
    """`conditions` are (feature index, op, threshold as a percentile 0-1)."""

    conditions: list
    direction: int          # +1 long, -1 short
    hold: int               # bars

    def describe(self, names) -> str:
        parts = [f"{names[i]} {op} p{int(q * 100)}" for i, op, q in self.conditions]
        side = "LONG" if self.direction > 0 else "SHORT"
        return f"{side} {hold_str(self.hold)} when " + " AND ".join(parts)


def hold_str(bars: int) -> str:
    return f"{bars} bars"


@dataclass
class GAResult:
    best: Rule
    fitness: float
    trades: int
    net_bps: float
    t_stat: float
    null_best: float = float("nan")
    null_mean: float = float("nan")
    p_value: float = float("nan")
    history: list = field(default_factory=list)
    feature_names: list = field(default_factory=list)


class RuleSearch:
    """Evolves rules over a precomputed feature matrix.

    `X` is (n_samples, n_features) and already causal. `fwd[h]` maps a holding
    period to that sample's forward return, net of cost, so the fitness
    function never has to touch prices.
    """

    def __init__(self, X, fwd_by_hold: dict, feature_names,
                 min_trades: int = 200, seed: int = 0,
                 max_conditions: int = 3):
        self.X = X
        self.fwd = fwd_by_hold
        self.holds = sorted(fwd_by_hold)
        self.names = list(feature_names)
        self.min_trades = min_trades
        self.max_conditions = max_conditions
        self.rng = np.random.default_rng(seed)
        # Thresholds are expressed as PERCENTILES of each feature, not raw
        # values, so a gene means the same thing across features on wildly
        # different scales and stays meaningful if the data is re-fetched.
        self.qs = np.nanpercentile(X, np.arange(5, 100, 5), axis=0)

    # -- genetic operators -------------------------------------------------

    def random_rule(self) -> Rule:
        n = self.rng.integers(1, self.max_conditions + 1)
        return Rule(
            conditions=[self._random_condition() for _ in range(n)],
            direction=int(self.rng.choice((-1, 1))),
            hold=int(self.rng.choice(self.holds)),
        )

    def _random_condition(self):
        return (int(self.rng.integers(len(self.names))),
                str(self.rng.choice(OPS)),
                float(self.rng.choice(np.arange(5, 100, 5)) / 100.0))

    def mutate(self, rule: Rule) -> Rule:
        conds = list(rule.conditions)
        roll = self.rng.random()
        if roll < 0.35 and conds:
            conds[self.rng.integers(len(conds))] = self._random_condition()
        elif roll < 0.55 and len(conds) < self.max_conditions:
            conds.append(self._random_condition())
        elif roll < 0.70 and len(conds) > 1:
            conds.pop(int(self.rng.integers(len(conds))))
        direction = rule.direction * (-1 if self.rng.random() < 0.15 else 1)
        hold = int(self.rng.choice(self.holds)) if self.rng.random() < 0.25 else rule.hold
        return Rule(conds, direction, hold)

    def crossover(self, a: Rule, b: Rule) -> Rule:
        pool = a.conditions + b.conditions
        k = max(1, min(self.max_conditions, (len(a.conditions) + len(b.conditions)) // 2))
        idx = self.rng.choice(len(pool), size=k, replace=False)
        return Rule([pool[i] for i in idx],
                    a.direction if self.rng.random() < 0.5 else b.direction,
                    a.hold if self.rng.random() < 0.5 else b.hold)

    # -- fitness -----------------------------------------------------------

    def mask(self, rule: Rule) -> np.ndarray:
        m = np.ones(len(self.X), dtype=bool)
        for i, op, q in rule.conditions:
            thr = self.qs[int(round(q * 100 / 5)) - 1, i]
            col = self.X[:, i]
            m &= (col < thr) if op == "<" else (col > thr)
            m &= np.isfinite(col)
        return m

    def evaluate(self, rule: Rule, fwd_by_hold=None) -> tuple:
        fwd = (fwd_by_hold or self.fwd)[rule.hold]
        m = self.mask(rule) & np.isfinite(fwd)
        n = int(m.sum())
        if n < self.min_trades:
            return -np.inf, n, 0.0, 0.0
        pnl = rule.direction * fwd[m]
        mean = float(np.mean(pnl))
        sd = float(np.std(pnl))
        t = mean / sd * np.sqrt(n) if sd > 0 else 0.0
        # Fitness is the t-statistic, not the raw return: a rule that fires
        # nine times for a huge average is not a strategy, and selecting on
        # mean return alone breeds exactly those.
        return t, n, mean * 1e4, t

    # -- the search --------------------------------------------------------

    def evolve(self, population: int = 60, generations: int = 25,
               elite: int = 6, fwd_by_hold=None) -> tuple:
        pop = [self.random_rule() for _ in range(population)]
        history = []
        best, best_fit = None, -np.inf
        for _ in range(generations):
            scored = sorted(((self.evaluate(r, fwd_by_hold)[0], r) for r in pop),
                            key=lambda kv: kv[0], reverse=True)
            if scored[0][0] > best_fit:
                best_fit, best = scored[0][0], scored[0][1]
            history.append(scored[0][0])
            survivors = [r for _, r in scored[:elite]]
            children = []
            while len(children) < population - elite:
                a, b = (survivors[int(self.rng.integers(len(survivors)))],
                        survivors[int(self.rng.integers(len(survivors)))])
                child = self.crossover(a, b)
                if self.rng.random() < 0.6:
                    child = self.mutate(child)
                children.append(child)
            pop = survivors + children
        return best, best_fit, history


def evolve_with_null(search: RuleSearch, population: int = 60,
                     generations: int = 25, null_runs: int = 10,
                     seed: int = 0) -> GAResult:
    """Run the search, then run the SAME search against destroyed labels.

    The null shuffles each holding period's forward returns independently, so
    the features and the trade counts are untouched and only the relationship
    between them is gone. Whatever the GA still achieves there is the price of
    the search itself, and the real run has to beat it.
    """
    best, fit, hist = search.evolve(population, generations)
    t, n, bps, tstat = search.evaluate(best)

    rng = np.random.default_rng(seed + 1)
    nulls = []
    for _ in range(null_runs):
        shuffled = {h: v[rng.permutation(len(v))] for h, v in search.fwd.items()}
        _, nfit, _ = search.evolve(population, generations, fwd_by_hold=shuffled)
        nulls.append(nfit)
    nulls = np.asarray([x for x in nulls if np.isfinite(x)], dtype="float64")

    return GAResult(
        best=best, fitness=fit, trades=n, net_bps=bps, t_stat=tstat,
        null_best=float(nulls.max()) if len(nulls) else float("nan"),
        null_mean=float(nulls.mean()) if len(nulls) else float("nan"),
        p_value=(float((np.sum(nulls >= fit) + 1) / (len(nulls) + 1))
                 if len(nulls) else float("nan")),
        history=hist, feature_names=search.names,
    )
