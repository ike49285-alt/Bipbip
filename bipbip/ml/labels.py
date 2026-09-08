"""Labelling.

The label answers the only question that matters: had I entered here, would the
trade have made money AFTER costs, before the session forced me flat?

Two decisions worth stating.

First, labels are cost-aware. A model trained on "did price rise" learns to
predict moves smaller than the spread, then loses money in production being
technically correct. The target barrier must clear the round-trip hurdle.

Second, labels use the triple-barrier method rather than a fixed-horizon
return, because that is how the strategy actually exits: a stop, a target, or
the closing bell, whichever comes first. Training on fixed-horizon returns
while deploying stops means the model is answering a question nobody asked.

Barriers are evaluated pessimistically and never cross a session boundary,
matching the engine's force-flat rule.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def triple_barrier_labels(
    bars: pd.DataFrame,
    atr: pd.Series,
    target_atr: float = 1.5,
    stop_atr: float = 1.0,
    max_hold_bars: int = 120,
    cost_bps: float = 2.3,
) -> pd.DataFrame:
    """Label each bar with the outcome of a long entered at the next open.

    Returns a frame with:
      ``label``      1 if the target was reached before the stop, else 0
      ``event_end``  positional index where the outcome resolved, for purging
      ``ret``        realised return net of costs
    """
    n = len(bars)
    open_, high, low, close = (bars[c].to_numpy(dtype="float64") for c in ("open", "high", "low", "close"))
    atr_arr = atr.to_numpy(dtype="float64")
    day = np.asarray([ts.date() for ts in bars.index])

    labels = np.full(n, np.nan)
    event_end = np.arange(n)
    rets = np.full(n, np.nan)
    cost = cost_bps / 10_000.0

    for i in range(n - 1):
        a = atr_arr[i]
        if not np.isfinite(a) or a <= 0:
            continue

        # The final bar of a session has no tradeable next open: the engine
        # force-flats before the close and cannot carry risk overnight.
        # Labelling it against tomorrow's bars would leak overnight
        # information into training on a bar that can never be traded.
        if day[i + 1] != day[i]:
            continue

        entry = open_[i + 1]  # fills at the next open, exactly as the engine does
        if not np.isfinite(entry) or entry <= 0:
            continue

        # The target must clear the round trip, or a "win" is still a loss.
        target = entry + max(target_atr * a, entry * cost * 1.5)
        stop = entry - stop_atr * a

        # Never look past the session's final bar: the engine force-flats.
        last = i + 1
        while last + 1 < n and day[last + 1] == day[i] and (last - i) < max_hold_bars:
            last += 1

        outcome, end = 0, last
        for j in range(i + 1, last + 1):
            # Stop first: a bar touching both resolves against the trader.
            if low[j] <= stop:
                outcome, end = 0, j
                rets[i] = (stop / entry - 1.0) - cost
                break
            if high[j] >= target:
                outcome, end = 1, j
                rets[i] = (target / entry - 1.0) - cost
                break
        else:
            # Neither barrier hit: exit at the last available close.
            rets[i] = (close[last] / entry - 1.0) - cost
            outcome = 1 if rets[i] > 0 else 0

        labels[i] = outcome
        event_end[i] = end

    return pd.DataFrame(
        {"label": labels, "event_end": event_end, "ret": rets}, index=bars.index
    )
