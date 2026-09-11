"""The vol anchor in `scripts/spread_edge.py`, which used to be a free parameter.

This script published "six of 53 short call spreads positive, none clearing its
own confidence interval" and later, on the SAME chain, forty-two spreads flagged
EDGE. Nothing about the strategy changed in between. The historical sample is
drawn only from days whose realised volatility matches an anchor, and that
anchor read the last bar in the bar archive - so every collection run silently
re-anchored the result.

These pin the two properties that stop it happening again: the anchor is taken
from the chain's own quote date, and the verdict is monotonic in the anchor, so
reporting one value of it is reporting one point on a curve.
"""
import importlib.util
import pathlib

import numpy as np
import pandas as pd
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "spread_edge",
    pathlib.Path(__file__).resolve().parents[1] / "scripts" / "spread_edge.py")
spread_edge = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(spread_edge)


def _vol_series():
    """Realised vol that CHANGES, so an anchor read off the wrong end differs."""
    idx = pd.bdate_range("2026-01-01", periods=60)
    v = pd.Series(np.linspace(0.60, 0.30, len(idx)), index=idx)
    return v


def test_the_anchor_comes_from_the_chains_own_date_not_the_latest_bar():
    """The defect, stated directly.

    The quotes being priced were made on a particular day, so the regime they
    are matched against has to be that day's. Reading the end of the series
    instead means a chain quoted in a 0.60 regime gets scored against a 0.30
    sample purely because more bars arrived later.
    """
    v = _vol_series()
    early = v.index[5]
    assert spread_edge.anchor_for(v, early) == pytest.approx(v.loc[early])
    # And emphatically NOT the last value in the series.
    assert spread_edge.anchor_for(v, early) != pytest.approx(v.iloc[-1])


def test_the_anchor_never_reads_a_bar_from_after_the_quote():
    """Anchoring on a later bar is lookahead, not merely irreproducibility."""
    v = _vol_series()
    asof = v.index[30]
    a = spread_edge.anchor_for(v, asof)
    assert a in set(v.loc[:asof].values)
    assert a not in set(v.loc[v.index > asof].values)


def test_a_chain_older_than_every_bar_raises_rather_than_guessing():
    """A silent fallback here would hand the sample an anchor from the future -
    which is the shape of the bug that contaminated 23.4% of a null in this
    repo. Missing must read as missing."""
    v = _vol_series()
    with pytest.raises(SystemExit):
        spread_edge.anchor_for(v, pd.Timestamp("2025-01-01"))


def test_the_band_is_relative_to_the_anchor_so_the_sample_moves_with_it():
    """The mechanism that makes the verdict anchor-dependent at all.

    If the band did not scale with the anchor this would be a reproducibility
    nit. It does, so the anchor selects WHICH historical days are scored, and
    the verdict is a function of it - which is why the script now prints the
    whole curve.
    """
    v = _vol_series()
    lo_anchor, hi_anchor = 0.32, 0.58
    lo_n = int(((v > lo_anchor * (1 - spread_edge.BAND))
                & (v < lo_anchor * (1 + spread_edge.BAND))).sum())
    hi_n = int(((v > hi_anchor * (1 - spread_edge.BAND))
                & (v < hi_anchor * (1 + spread_edge.BAND))).sum())
    assert lo_n > 0 and hi_n > 0
    # Different anchors select disjoint-ish regimes; they must not coincide.
    lo_days = set(v[(v > lo_anchor * (1 - spread_edge.BAND))
                    & (v < lo_anchor * (1 + spread_edge.BAND))].index)
    hi_days = set(v[(v > hi_anchor * (1 - spread_edge.BAND))
                    & (v < hi_anchor * (1 + spread_edge.BAND))].index)
    assert not (lo_days & hi_days)
