"""Cluster-robust inference."""
import numpy as np
import pytest

from bipbip.core.stats import ols_cluster


def _design(n, rng):
    x = rng.normal(size=n)
    return np.column_stack([np.ones(n), x]), x


def test_clustering_widens_the_error_when_observations_repeat_within_a_cluster():
    """The trap it exists for: correlated rows counted as independent ones."""
    rng = np.random.default_rng(0)
    # 40 clusters of 25 rows; the whole cluster shares one shock, so the
    # effective sample is 40, not 1,000.
    clusters = np.repeat(np.arange(40), 25)
    x = rng.normal(size=40)[clusters]
    y = rng.normal(size=40)[clusters] + 0.05 * rng.normal(size=1000)
    X = np.column_stack([np.ones(1000), x])

    naive = ols_cluster(y, X, np.arange(1000))
    clustered = ols_cluster(y, X, clusters)

    assert clustered["se"][1] > 3 * naive["se"][1]
    assert clustered["clusters"] == 40


def test_one_observation_per_cluster_reduces_to_white_hc0():
    rng = np.random.default_rng(1)
    n = 500
    X, x = _design(n, rng)
    y = 2.0 + 0.5 * x + rng.normal(size=n)

    got = ols_cluster(y, X, np.arange(n))

    beta = np.linalg.pinv(X.T @ X) @ (X.T @ y)
    u = y - X @ beta
    xtx_inv = np.linalg.pinv(X.T @ X)
    hc0 = xtx_inv @ (X.T @ (X * u[:, None] ** 2)) @ xtx_inv
    c = (n / (n - 1)) * ((n - 1) / (n - X.shape[1]))
    assert got["se"] == pytest.approx(np.sqrt(np.diag(c * hc0)), rel=1e-10)


def test_recovers_a_known_slope():
    rng = np.random.default_rng(2)
    n = 4000
    X, x = _design(n, rng)
    y = 1.0 - 0.75 * x + 0.1 * rng.normal(size=n)

    got = ols_cluster(y, X, rng.integers(0, 100, n))

    assert got["beta"][0] == pytest.approx(1.0, abs=0.02)
    assert got["beta"][1] == pytest.approx(-0.75, abs=0.02)


@pytest.mark.parametrize("y_n,x_n,c_n", [(9, 10, 10), (10, 10, 9), (10, 9, 10)])
def test_mismatched_lengths_are_rejected(y_n, x_n, c_n):
    """Each of the three, separately, and on the MESSAGE.

    Asserting only `ValueError` passed whether or not the guard existed: with
    it deleted, numpy reaches `np.add.at` and raises "array is not
    broadcastable to correct shape", which is the same exception type. The test
    could not fail for its stated purpose, and mutation testing is what showed
    it - flipping the guard's `or` to `and` left every test green.
    """
    with pytest.raises(ValueError, match="same length"):
        ols_cluster(np.zeros(y_n), np.ones((x_n, 2)), np.arange(c_n))


def test_t_is_beta_over_se():
    """The formula behind every significance claim in this repo, and nothing
    asserted on it. `beta / se` could be changed to `beta * se` and all of
    test_stats passed - the `t` key was never read by any test."""
    rng = np.random.default_rng(7)
    n = 300
    X, x = _design(n, rng)
    y = 0.4 + 1.3 * x + rng.normal(size=n)

    got = ols_cluster(y, X, rng.integers(0, 30, n))

    assert got["t"] == pytest.approx(got["beta"] / got["se"], rel=1e-12)
    # ...and it is genuinely large here, so a near-zero t would not pass by
    # coincidence on a slope this strong.
    assert abs(got["t"][1]) > 5


def test_an_exactly_zero_standard_error_yields_zero_t_not_infinity():
    """The `se > 0` guard, exercised where it actually fires."""
    got = ols_cluster(np.zeros(8), np.ones((8, 1)), np.arange(8))
    assert got["se"][0] == 0
    assert got["t"][0] == 0 and np.isfinite(got["t"][0])


def test_a_degenerate_design_reports_an_absurd_t_rather_than_infinity():
    """DOCUMENTS A WEAKNESS rather than asserting the behaviour is right.

    The `se > 0` guard only catches an EXACTLY zero standard error, and
    floating point almost never produces one. A perfectly fitted design leaves
    residuals around 1e-15 instead, so se is ~3e-13 and t comes back near
    1e14 - which is not infinity, and therefore passes every finiteness check,
    while reading downstream as overwhelming significance.

    A human notices a t of 2e14. An automated significance filter does not,
    and this repo's whole direction is automation. The guard wants a scale-
    relative tolerance rather than `> 0`; that is a live proposal, not a
    change made quietly here, because it alters what the function returns.
    """
    n = 50
    X = np.column_stack([np.ones(n), np.arange(n, dtype="float64")])
    y = 3.0 + 2.0 * np.arange(n)           # exact fit up to rounding

    got = ols_cluster(y, X, np.arange(n))

    assert np.all(got["se"] > 0)           # NOT exactly zero - the guard misses
    assert np.all(np.isfinite(got["t"]))
    assert np.max(np.abs(got["t"])) > 1e9  # ...and this is the problem


def test_the_finite_sample_correction_is_the_documented_formula():
    """Cameron-Gelbach-Miller: (g/(g-1)) * ((n-1)/(n-k)). Recomputed here
    independently, because the constants inside it are otherwise only pinned
    for the one-row-per-cluster case where g == n."""
    rng = np.random.default_rng(11)
    n, k = 600, 2
    X, x = _design(n, rng)
    clusters = rng.integers(0, 37, n)
    y = 1.0 + 0.5 * x + rng.normal(size=n)

    got = ols_cluster(y, X, clusters)
    g = len(np.unique(clusters))
    assert got["clusters"] == g

    xtx_inv = np.linalg.pinv(X.T @ X)
    beta = xtx_inv @ (X.T @ y)
    u = y - X @ beta
    _, inv = np.unique(clusters, return_inverse=True)
    scores = np.zeros((g, k))
    np.add.at(scores, inv, X * u[:, None])
    c = (g / (g - 1)) * ((n - 1) / (n - k))
    cov = c * (xtx_inv @ (scores.T @ scores) @ xtx_inv)

    assert got["se"] == pytest.approx(np.sqrt(np.diag(cov)), rel=1e-10)


def test_a_single_cluster_does_not_divide_by_zero():
    """g == 1 makes the correction's (g-1) denominator zero, which is what the
    max(..., 1) floor is for. One cluster is a real input - it is what a
    single-symbol panel with date clustering collapses to on one date."""
    rng = np.random.default_rng(13)
    n = 40
    X, x = _design(n, rng)
    y = 1.0 + 0.5 * x + rng.normal(size=n)

    got = ols_cluster(y, X, np.zeros(n, dtype=int))

    assert got["clusters"] == 1
    assert np.all(np.isfinite(got["se"]))
    assert np.all(np.isfinite(got["t"]))


def test_a_saturated_regression_does_not_divide_by_zero():
    """n == k makes (n - k) zero, the other floor. A design with as many
    parameters as observations is degenerate but must not produce inf."""
    n = 3
    X = np.eye(n)
    y = np.array([1.0, 2.0, 3.0])

    got = ols_cluster(y, X, np.arange(n))

    assert np.all(np.isfinite(got["se"]))
    assert np.all(np.isfinite(got["t"]))


def _mtf_probe(base, build, probes=60):
    """Count rows where a higher-timeframe series changes once the future lands.

    The only assertion that means anything: recompute with the history ENDING
    at each probe bar and check the value there is unchanged. Truncating just
    the tail is not enough - it perturbs one bar and leaves the rest identical
    whether the construction is causal or not.

    Probe generously. A boolean output collapses a leak whenever the leaked
    value lands on the same side of the threshold, so a handful of probes can
    miss a series that really is peeking.
    """
    import pandas as pd
    full = build(base)
    bad, n = 0, len(base)
    step = max(1, (n // 2) // probes)
    for i in range(n - probes * step, n, step):
        a, b = full.iloc[i], build(base.iloc[:i + 1]).iloc[i]
        same = (a is pd.NA and b is pd.NA) or (
            a == b if not (pd.isna(a) or pd.isna(b)) else pd.isna(a) and pd.isna(b))
        bad += 0 if same else 1
    return bad


def _synthetic_30m(n=20000):
    import pandas as pd
    rng = np.random.default_rng(7)
    idx = pd.date_range("2024-01-02 09:30", periods=n, freq="30min",
                        tz="America/New_York")
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
    return pd.DataFrame({"open": close, "high": close * 1.002,
                         "low": close * 0.998, "close": close,
                         "volume": 1e6}, index=idx)


def test_higher_timeframe_colour_never_uses_an_unclosed_bar():
    """The lookahead that would fake any multi-timeframe result.

    A daily series forward-filled onto 30-minute bars can put the whole of
    today's range onto 10am, which is tomorrow's news priced this morning. The
    cloud colour escapes that not because of the one-bar lag but because
    Ichimoku's senkou spans are already displaced 26 higher bars: the colour at
    daily row j is computed from bars around j-26, which closed long ago.
    """
    from scripts.cloud_inversion import colour_pair

    base = _synthetic_30m()
    assert _mtf_probe(base, lambda b: colour_pair(b, "1D")[1]) == 0


def test_the_lookahead_probe_catches_a_series_that_peeks():
    """Positive control, without which the test above proves nothing."""
    from scripts.cloud_inversion import AGG

    base = _synthetic_30m()

    def leaky(b):
        hi = b.resample("1D").agg(AGG).dropna()
        # Today's close, forward-filled onto every bar of today.
        return hi["close"].reindex(b.index, method="ffill")

    assert _mtf_probe(base, leaky) > 30
