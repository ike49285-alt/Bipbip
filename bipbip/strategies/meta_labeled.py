"""A primary rule filtered and sized by a meta-labelling model.

The division of labour is strict and is the point of the design:

  PRIMARY decides direction. It is a fixed, inspectable rule.
  SECONDARY decides whether to act, and how large.

The secondary model can only decline a trade or take a smaller one. It cannot
propose a trade the primary did not, and it cannot reverse one. A secondary
model that has learnt nothing therefore costs opportunity, never capital -
which is not true of any other model in this project.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.panel import Panel
from ..core.portfolio import PortfolioContext, PortfolioStrategy


class MetaLabeledStrategy(PortfolioStrategy):
    """Take the primary's signals only where the model is confident enough."""

    name = "meta_labeled"

    def __init__(self, model, signal_fn, feature_fn, threshold: float = 0.55,
                 top_n: int = 5, max_hold: int = 20, size_by_confidence: bool = True,
                 market: pd.Series | None = None):
        self.model = model
        self.signal_fn = signal_fn
        self.feature_fn = feature_fn
        self.threshold = threshold
        self.top_n = top_n
        self.max_hold = max_hold
        # Confidence-weighted sizing is the second half of meta-labelling: the
        # model says how good a setup looks, so a marginal one gets less money.
        self.size_by_confidence = size_by_confidence
        self.market = market
        self.warmup_bars = 260
        self._held: dict = {}

    def prepare(self, panel: Panel) -> dict:
        signals = self.signal_fn(panel)
        feats = self.feature_fn(panel, self.market)
        names = list(feats)

        # Score every (date, symbol) once. The features are causal, so scoring
        # up front is a performance choice rather than a peek.
        stacked = np.stack([feats[k].to_numpy(dtype="float64") for k in names], axis=-1)
        flat = stacked.reshape(-1, len(names))
        proba = np.full(len(flat), np.nan)
        usable = np.isfinite(flat).any(axis=1)
        if usable.any():
            proba[usable] = self.model.predict_proba(
                np.nan_to_num(flat[usable], nan=0.0, posinf=0.0, neginf=0.0))
        proba = proba.reshape(stacked.shape[:-1])

        return {"signals": signals,
                "proba": pd.DataFrame(proba, index=panel.dates, columns=panel.symbols)}

    def target_weights(self, ctx: PortfolioContext) -> dict:
        sig = ctx.ind("signals").reindex(ctx.tradeable).fillna(False)
        proba = ctx.ind("proba").reindex(ctx.tradeable)

        # Age out existing holdings.
        held = {s: age + 1 for s, age in self._held.items()
                if s in ctx.tradeable and age + 1 <= self.max_hold
                and ctx.current_weights.get(s, 0.0) > 1e-6}

        fresh = proba[sig.astype(bool) & (proba >= self.threshold)].dropna()
        room = self.top_n - len(held)
        if room > 0 and not fresh.empty:
            for s in fresh.nlargest(min(room, len(fresh))).index:
                if s not in held:
                    held[s] = 0
        self._held = held
        if not held:
            return {}

        if not self.size_by_confidence:
            return {s: 1.0 / len(held) for s in held}

        conf = {}
        for s in held:
            p = float(proba.get(s, np.nan))
            # A held name whose score is stale still gets the base weight.
            conf[s] = max(0.0, (p - 0.5)) if np.isfinite(p) else 0.05
        total = sum(conf.values())
        if total <= 0:
            return {s: 1.0 / len(held) for s in held}
        return {s: w / total for s, w in conf.items()}


class PrimaryOnlyStrategy(PortfolioStrategy):
    """The primary rule with no model at all - the benchmark meta-labelling
    must beat, or the model is adding nothing but complexity."""

    name = "primary_only"

    def __init__(self, signal_fn, top_n: int = 5, max_hold: int = 20):
        self.signal_fn = signal_fn
        self.top_n = top_n
        self.max_hold = max_hold
        self.warmup_bars = 260
        self._held: dict = {}

    def prepare(self, panel: Panel) -> dict:
        return {"signals": self.signal_fn(panel)}

    def target_weights(self, ctx: PortfolioContext) -> dict:
        sig = ctx.ind("signals").reindex(ctx.tradeable).fillna(False)
        held = {s: age + 1 for s, age in self._held.items()
                if s in ctx.tradeable and age + 1 <= self.max_hold
                and ctx.current_weights.get(s, 0.0) > 1e-6}
        fresh = [s for s in ctx.tradeable if bool(sig.get(s, False)) and s not in held]
        for s in fresh[: max(0, self.top_n - len(held))]:
            held[s] = 0
        self._held = held
        return {s: 1.0 / len(held) for s in held} if held else {}
