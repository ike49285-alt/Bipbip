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
    #: Null runs that actually produced a finite fitness. The permutation
    #: p-value cannot go below 1/(null_runs + 1), and callers need the count
    #: to say so - it is not recoverable from `p_value` alone, and `history`
    #: is per-generation fitness, not null runs.
    null_runs: int = 0
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
                 max_conditions: int = 3, max_fraction: float = 0.30,
                 demean: bool = True, cost_bps=None):
        self.X = X
        # Cost is charged AFTER direction, never before. Netting it into the
        # forward return first and then flipping the sign for a short turns
        # `gross - cost` into `-gross + cost`: the short is PAID the spread
        # instead of charged it. That handed every short a 4.34 bps rebate and
        # produced an "edge" of +8.58 bps that existed at every hour of the
        # day, which is what it looks like when a bug is uniform. Gross
        # open-to-close over five bars is -0.12 bps; there was nothing there in
        # either direction.
        self.cost = (np.zeros(len(X), dtype="float64") if cost_bps is None
                     else np.asarray(cost_bps, dtype="float64"))
        # DEMEAN each holding period. Without this the search does not need a
        # signal: forward returns have a non-zero unconditional mean, so a rule
        # firing on most of the sample simply inherits it. The first real run
        # returned "SHORT when vol_30 < p75 AND vol_ratio < p95", which fired
        # on 73% of all samples - "always short" wearing two conditions - and
        # scored t=675. Shuffling does not change an unconditional mean, so the
        # null scored t=560 too and correctly refused to call it significant.
        # Demeaning makes "always trade" worth exactly zero, so the only way to
        # score is to be SELECTIVE.
        self.fwd = ({h: v - np.nanmean(v) for h, v in fwd_by_hold.items()}
                    if demean else dict(fwd_by_hold))
        self.holds = sorted(self.fwd)
        self.names = list(feature_names)
        self.min_trades = min_trades
        self.max_conditions = max_conditions
        # A rule that fires on most of the tape is not a rule. This caps how
        # much of the sample a candidate may claim.
        self.max_fraction = max_fraction
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
        if n > self.max_fraction * len(fwd):
            # Too broad to be a rule; it is market exposure with extra steps.
            return -np.inf, n, 0.0, 0.0
        pnl = rule.direction * fwd[m] - self.cost[m]
        mean = float(np.mean(pnl))
        sd = float(np.std(pnl))
        t = mean / sd * np.sqrt(n) if sd > 0 else 0.0
        # Fitness is the t-statistic, not the raw return: a rule that fires
        # nine times for a huge average is not a strategy, and selecting on
        # mean return alone breeds exactly those.
        #
        # `mean` is returned in whatever units the caller supplied. An earlier
        # version multiplied by 1e4 here while callers already passed basis
        # points, and reported +82,473 bps a trade for what was +8.2.
        return t, n, mean, t

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
        null_runs=int(len(nulls)),
        history=hist, feature_names=search.names,
    )
