"""Assemble the supervised learning problem from raw bars."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..core import indicators as ind
from .features import FEATURE_COLUMNS, build_features
from .labels import triple_barrier_labels


@dataclass
class Dataset:
    X: pd.DataFrame
    y: np.ndarray
    rets: np.ndarray
    event_end: np.ndarray
    index: pd.DatetimeIndex

    def __len__(self) -> int:
        return len(self.X)

    @property
    def sessions(self) -> int:
        return len({ts.date() for ts in self.index})

    def summary(self) -> str:
        return (
            f"{len(self):,} samples over {self.sessions} sessions | "
            f"positive rate {self.y.mean():.1%} | "
            f"mean net return {self.rets.mean() * 10_000:+.2f} bps"
        )


def build_dataset(
    bars: pd.DataFrame,
    target_atr: float = 1.5,
    stop_atr: float = 1.0,
    max_hold_bars: int = 120,
    cost_bps: float = 2.3,
    or_minutes: int = 30,
    warmup_bars: int = 60,
    daily: pd.DataFrame | None = None,
) -> Dataset:
    """Build features and labels, dropping warmup and unresolved rows.

    Barrier parameters are shared with `MLStrategy` at deployment time. They
    must match: a model trained to predict a 1.5-ATR target reached before a
    1.0-ATR stop is answering a different question from one deployed with
    different barriers, and the mismatch shows up as unexplained live
    underperformance.
    """
    features = build_features(bars, or_minutes=or_minutes, daily=daily)
    atr = ind.atr(bars, 30)
    labels = triple_barrier_labels(
        bars, atr, target_atr=target_atr, stop_atr=stop_atr,
        max_hold_bars=max_hold_bars, cost_bps=cost_bps,
    )

    # Drop the warmup window per session: indicators are not yet meaningful.
    bar_of_day = ind.minutes_since_open(bars.index).to_numpy()
    usable = (bar_of_day >= warmup_bars) & labels["label"].notna().to_numpy()
    # A feature row that is entirely NaN carries no information.
    usable &= features.notna().any(axis=1).to_numpy()

    X = features.loc[usable]
    lab = labels.loc[usable]

    # event_end is stored as an absolute position; remap onto the filtered rows
    # so purging still measures overlap correctly after filtering.
    positions = np.flatnonzero(usable)
    remap = np.searchsorted(positions, lab["event_end"].to_numpy())
    remap = np.clip(remap, 0, len(positions) - 1)

    return Dataset(
        X=X,
        y=lab["label"].to_numpy(dtype="float64"),
        rets=lab["ret"].to_numpy(dtype="float64"),
        event_end=remap,
        index=X.index,
    )
