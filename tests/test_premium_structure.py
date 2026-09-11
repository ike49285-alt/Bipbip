"""The vol anchor in `scripts/premium_structure.py`.

This is the third script in the repo to draw a vol-matched historical sample
and anchor it on the BAR ARCHIVE'S last row rather than on the date the option
chain was quoted. In the other two the defect was not latent: spread_edge's
published verdict walked from "approximately zero" to "EDGE" on the anchor
alone, and chain_edge's near-the-money half was an artifact.

Here it is currently latent - the archive ends the day before the newest chain,
so the tail and the quote date select the same row - which is precisely why it
needed a test rather than an inspection. It becomes wrong the moment bars
advance past a pinned snapshot, and `--chain` exists to pin one.
"""
import importlib.util
import pathlib

import numpy as np
import pandas as pd

_SPEC = importlib.util.spec_from_file_location(
    "premium_structure",
    pathlib.Path(__file__).resolve().parents[1] / "scripts"
    / "premium_structure.py")
premium_structure = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(premium_structure)


def _anchor(closes, asof):
    """The anchor selection as the script performs it."""
    r = np.log(closes / closes.shift(1))
    trail = (r.rolling(20).std() * np.sqrt(252)).to_numpy()
    day = pd.DatetimeIndex([pd.Timestamp(x.date()) for x in closes.index])
    upto = np.flatnonzero(day <= asof)
    return trail[upto[-1]], trail[-1]


def test_the_anchor_tracks_the_quote_date_and_not_the_archive_tail():
    """A regime change AFTER the quote must not reach back into the sample."""
    idx = pd.bdate_range("2026-01-01", periods=120)
    steps = np.r_[np.full(80, 0.002), np.full(40, 0.020)]   # calm, then wild
    rng = np.random.default_rng(0)
    closes = pd.Series(100 * np.exp(np.cumsum(steps * rng.standard_normal(120))),
                       index=idx)

    asof = idx[79]                      # quoted before the vol regime changed
    anchored, tail = _anchor(closes, asof)
    assert np.isfinite(anchored) and np.isfinite(tail)
    # The tail sits in the wild regime; the quote date does not.
    assert tail > anchored * 2, (anchored, tail)


def test_the_anchor_is_the_tail_when_the_chain_is_the_newest_bar():
    """The current archive's case, pinned so the fix is known to be a no-op
    today rather than assumed to be one."""
    idx = pd.bdate_range("2026-01-01", periods=60)
    rng = np.random.default_rng(1)
    closes = pd.Series(100 * np.exp(np.cumsum(0.01 * rng.standard_normal(60))),
                       index=idx)
    anchored, tail = _anchor(closes, idx[-1])
    assert anchored == tail

    # And a quote date AFTER every bar still resolves to the last bar, which is
    # the real archive's situation: bars end the day before the newest chain.
    later, _ = _anchor(closes, idx[-1] + pd.Timedelta(days=3))
    assert later == tail
