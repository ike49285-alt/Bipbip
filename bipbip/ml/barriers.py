"""Resolve a trade at whichever comes first: the target, the stop, or the clock.

A fixed holding period asks the wrong question. A trade that runs +40 bps by bar
five and gives it all back by bar thirty is labelled a loss, and the model is
then trained to avoid the setup that produced it - while a trade that sits at
-80 bps for twenty bars before recovering is labelled a win, and the model
learns to sit through drawdowns no one would actually sit through. Neither
label describes a trade anybody would take.

Barriers describe the trade you would really place: a target, a stop, and a
limit on patience. The outcome is whichever the price reaches first.

THE DETAIL THAT DECIDES EVERYTHING: on a bar whose high clears the target AND
whose low breaks the stop, OHLC data cannot say which came first - the path
inside the bar is not recorded. Assuming the target is the optimistic lie that
inflates every barrier backtest, because it hands the strategy the good half of
every ambiguous bar. Stops are checked first here. That is conservative by
construction and it is the only defensible choice without tick data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TARGET, STOP, TIMEOUT, UNRESOLVED = 1, -1, 0, -9


def first_touch(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                entry: np.ndarray, target: np.ndarray, stop: np.ndarray,
                max_hold: int, session: np.ndarray,
                side: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (realised return, which barrier, bars held) for an entry at each bar.

    `entry` is the fill price for a position opened at bar i (normally the next
    bar's open). `target` and `stop` are absolute price levels already oriented
    for `side`: for a long the target is above and the stop below, for a short
    the reverse. `session` marks each bar's day so a position is closed at the
    last bar of its own session rather than carried overnight.
    """
    n = len(close)
    hit_t = np.full(n, max_hold + 1, dtype=np.int32)
    hit_s = np.full(n, max_hold + 1, dtype=np.int32)
    last = np.full(n, -1, dtype=np.int32)          # final bar available intraday

    for k in range(1, max_hold + 1):
        j = np.arange(n) + k
        ok = (j < n) & (session[np.minimum(j, n - 1)] == session)
        jj = np.minimum(j, n - 1)
        last = np.where(ok, k, last)
        if side > 0:
            t_now = ok & (high[jj] >= target)
            s_now = ok & (low[jj] <= stop)
        else:
            t_now = ok & (low[jj] <= target)
            s_now = ok & (high[jj] >= stop)
        hit_t = np.where((hit_t > max_hold) & t_now, k, hit_t)
        hit_s = np.where((hit_s > max_hold) & s_now, k, hit_s)

    resolved_t = hit_t <= max_hold
    resolved_s = hit_s <= max_hold
    # Stop first on a tie: a bar that touches both is ambiguous, and awarding it
    # to the target is how a barrier backtest quietly pays itself.
    stop_wins = resolved_s & (hit_s <= hit_t)
    tgt_wins = resolved_t & ~stop_wins

    which = np.where(stop_wins, STOP, np.where(tgt_wins, TARGET, TIMEOUT))
    held = np.where(stop_wins, hit_s, np.where(tgt_wins, hit_t, last))
    which = np.where(last < 0, UNRESOLVED, which)

    idx = np.clip(np.arange(n) + held, 0, n - 1)
    exit_px = np.where(stop_wins, stop, np.where(tgt_wins, target, close[idx]))
    ret = side * (exit_px / entry - 1.0)
    ret = np.where(last < 0, np.nan, ret)
    return ret, which, np.where(last < 0, 0, held)


def barrier_labels(bars: pd.DataFrame, atr: pd.Series, target_mult: float = 1.5,
                   stop_mult: float = 1.0, max_hold: int = 26,
                   cost_bps: float = 3.10) -> pd.DataFrame:
    """Cost-adjusted barrier outcomes for a long and a short at every bar.

    Barriers are set in units of the symbol's own recent range rather than fixed
    percentages, so the same parameters mean the same thing on a quiet day and a
    violent one, and on a $5 fund and a $500 one.

    A short is NOT the negative of a long under barriers - its stop sits above
    and its target below, so the two resolve on different bars and must be
    computed separately. Treating one as the mirror of the other is a modelling
    error that silently doubles any apparent edge.
    """
    entry = bars["open"].shift(-1).to_numpy()
    high, low, close = (bars[c].to_numpy() for c in ("high", "low", "close"))
    session = pd.DatetimeIndex(bars.index).normalize().view("int64")
    a = atr.to_numpy()

    out = {}
    for side, name in ((1, "long"), (-1, "short")):
        tgt = entry + side * target_mult * a
        stp = entry - side * stop_mult * a
        r, which, held = first_touch(high, low, close, entry, tgt, stp,
                                     max_hold, session, side=side)
        out[f"{name}_ret"] = r * 1e4 - cost_bps
        out[f"{name}_barrier"] = which
        out[f"{name}_held"] = held
    return pd.DataFrame(out, index=bars.index)
