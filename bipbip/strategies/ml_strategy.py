"""A trained model, wrapped as an ordinary Strategy.

This matters more than it looks. A model evaluated only with sklearn metrics
has never paid a spread, never been refused an entry by a settlement rule, and
never been forced flat at 15:55. Running it through the same engine as every
hand-written strategy is what turns "72% accuracy" into a number denominated in
dollars.

The barriers here MUST match the ones used to label the training data - the
model was trained to answer a specific question, and deploying different
barriers asks it a different one.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core import indicators as ind
from ..core.strategy import Context, Strategy
from ..core.types import HOLD, Intent
from ..ml.features import FEATURE_COLUMNS, build_features


class MLStrategy(Strategy):
    """Enters when the model's probability of a profitable trade clears
    `threshold`.

    With one round trip per session, the threshold is the whole risk policy:
    the first bar to clear it spends the entire day's budget. Set it too low
    and the day is wasted on the first mediocre setup of the morning.
    """

    name = "ml"

    def __init__(
        self,
        model,
        threshold: float = 0.6,
        target_atr: float = 1.5,
        stop_atr: float = 1.0,
        or_minutes: int = 30,
        warmup: int = 60,
    ):
        self.model = model
        self.threshold = threshold
        self.target_atr = target_atr
        self.stop_atr = stop_atr
        self.or_minutes = or_minutes
        self.warmup_bars = warmup

    def prepare(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Compute features and score every bar in one pass.

        Scoring here rather than per-bar is a performance choice, not a
        shortcut: `build_features` is causal, so the score at bar i depends
        only on bars <= i either way. The lookahead suite verifies this.
        """
        feats = build_features(bars, or_minutes=self.or_minutes)
        out = feats.copy()
        out["atr"] = ind.atr(bars, 30)

        X = feats.to_numpy(dtype="float64")
        proba = np.full(len(feats), np.nan)
        valid = np.isfinite(X).any(axis=1)
        if valid.any():
            proba[valid] = self.model.predict_proba(X[valid])
        out["proba"] = proba
        return out

    def on_bar(self, ctx: Context) -> Intent:
        # Account rules are the engine's business; see opening_range.py.
        if ctx.in_position:
            return HOLD

        row = ctx.ind
        p, atr_v = float(row["proba"]), float(row["atr"])
        if not np.isfinite(p) or not np.isfinite(atr_v) or atr_v <= 0:
            return HOLD
        if p < self.threshold:
            return HOLD

        price = ctx.price
        return Intent(
            action="enter",
            reason=f"ml_p={p:.2f}",
            stop_price=price - self.stop_atr * atr_v,
            target_price=price + self.target_atr * atr_v,
        )
