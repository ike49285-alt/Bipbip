"""The label permutation behind the headline null.

`scripts/panel_null100.py` produces the only number in this project that a
reader is likely to quote, and the bug that contaminated it lived in exactly
one function: a `(symbol, permuted timestamp)` lookup that missed on 23.4% of
rows and silently handed each of them back its OWN label. Nearly a quarter of
every "null" run was real signal, and it survived for months because nothing
here exercised it.

These tests pin the three properties that make the null a null:

  - labels actually MOVE,
  - the permutation is the SAME for every symbol, so a moment's whole
    cross-section travels together and the correlation structure the model
    reads is preserved,
  - a lookup that cannot resolve RAISES instead of falling back.

The contaminated `--method permute` path is deliberately kept in the script to
reproduce the superseded p=0.069, so it is tested for the contamination it is
known to have rather than asserted clean.
"""
import sys
import pathlib

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from scripts.panel_null100 import MAX_FIXED_DATES, shuffled_labels


def _panel(symbols=("A", "B", "C"), n_dates=40, drop=None):
    """A complete (symbol x date) grid, optionally punched full of holes.

    `drop` removes (symbol, date) pairs, which is what the real archive looks
    like: the twenty funds do not all trade every 30-minute stamp.
    """
    dates = pd.date_range("2020-01-01", periods=n_dates, freq="30min")
    rows = [(s, d) for s in symbols for d in dates
            if not (drop and (s, d) in drop)]
    df = pd.DataFrame(rows, columns=["_sym", "_date"])
    # A label unique to each cell, so a moved label is identifiable by value.
    df["_L"] = np.arange(len(df), dtype="float64")
    df["_S"] = -df["_L"]
    return df


def _call(df, method, seed=1):
    return shuffled_labels(df, df["_L"].to_numpy(), df["_S"].to_numpy(),
                           df["_sym"].to_numpy(), seed, method)


def test_a_common_grid_shuffle_actually_moves_labels():
    df = _panel()
    y_l, _ = _call(df, "common")
    kept = float((y_l == df["_L"].to_numpy()).mean())
    # Expected fixed points is 1 for any panel size, so the expected row share
    # is ~1/n_dates. Allow a generous multiple of that, not a flat percentage.
    assert kept <= 5.0 / 40, f"{kept:.1%} of rows kept their own label"


def test_the_shuffle_is_a_permutation_not_a_resample():
    """Every label must survive exactly once; none invented, none duplicated."""
    df = _panel()
    y_l, y_s = _call(df, "common")
    assert sorted(y_l.tolist()) == sorted(df["_L"].to_numpy().tolist())
    assert sorted(y_s.tolist()) == sorted(df["_S"].to_numpy().tolist())


def test_long_and_short_labels_move_together():
    """A row must not take its long leg from one moment and its short from
    another - that would break the pairing the margin is computed over."""
    df = _panel()
    y_l, y_s = _call(df, "common")
    assert np.allclose(y_s, -y_l)


def test_every_symbol_gets_the_same_date_permutation():
    """The cross-section has to travel together.

    If each symbol were permuted independently, a moment's twenty funds would
    be assembled from twenty unrelated moments, destroying the correlation the
    model reads and making the null easier to beat than the real run.
    """
    df = _panel()
    y_l, _ = _call(df, "common")
    df = df.assign(_new=y_l)
    # Recover, per symbol, the date each date's label came from.
    src = {}
    lookup = {(s, d): v for s, d, v in
              zip(df["_sym"], df["_date"], df["_L"])}
    back = {v: (s, d) for (s, d), v in lookup.items()}
    for s, d, v in zip(df["_sym"], df["_date"], df["_new"]):
        src.setdefault(s, {})[d] = back[v][1]
    reference = src[df["_sym"].iloc[0]]
    for sym, mapping in src.items():
        assert mapping == reference, f"{sym} used a different date permutation"


@pytest.mark.parametrize("n_dates", [40, 200, 1000])
def test_the_guard_does_not_depend_on_panel_size(n_dates):
    """One flat ceiling has to hold at every panel size.

    Fixed points are Poisson(1) for any D, so the same limit works whether the
    panel has forty dates or a thousand. A ceiling on the fixed-row SHARE would
    not: that share is ~1/D, so it is 2.5% on a forty-date panel and 0.05% on
    the real archive, and tuning it to one would mis-set it for the other.
    """
    df = _panel(n_dates=n_dates)
    for seed in range(1, 6):
        y_l, _ = _call(df, "common", seed=seed)          # must not raise
        assert not np.array_equal(y_l, df["_L"].to_numpy())
    assert MAX_FIXED_DATES == 10


def test_an_incomplete_grid_raises_instead_of_falling_back():
    """The actual bug. A missed lookup must be an error, not a default."""
    dates = pd.date_range("2020-01-01", periods=40, freq="30min")
    df = _panel(drop={("B", dates[3]), ("B", dates[9]), ("C", dates[15])})
    with pytest.raises(AssertionError, match="lookups missed"):
        _call(df, "common")


def test_a_degenerate_permutation_is_caught():
    """A shuffle that does not shuffle must fail loudly rather than score."""
    df = _panel(n_dates=40)

    class _Identity:
        def permutation(self, x):
            return np.asarray(x)

    import scripts.panel_null100 as pn
    real = pn.np.random.default_rng
    pn.np.random.default_rng = lambda *a, **k: _Identity()
    try:
        with pytest.raises(AssertionError, match="did not shuffle"):
            _call(df, "common")
    finally:
        pn.np.random.default_rng = real


def test_the_permute_method_is_contaminated_as_documented():
    """`--method permute` is kept only to reproduce the superseded p=0.069.

    It must stay reachable, and it must stay obviously broken: on a panel whose
    symbols do not share every timestamp it hands rows back their true labels.
    """
    dates = pd.date_range("2020-01-01", periods=40, freq="30min")
    holes = {("B", d) for d in dates[::2]} | {("C", d) for d in dates[::3]}
    df = _panel(drop=holes)
    y_l, _ = _call(df, "permute")          # must not raise
    kept = float((y_l == df["_L"].to_numpy()).mean())
    assert kept > 0.10, (
        "the permute path no longer reproduces the contaminated null; it is "
        "kept in the script precisely to reproduce p=0.069")
