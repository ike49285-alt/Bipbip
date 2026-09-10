"""Inference helpers for series whose observations are not independent.

CLAUDE.md names date clustering as a standing trap - a market-wide move puts
every symbol in the same bucket on one date, and bars within a session are
correlated with each other - but until now every script rolled its own
two-sample t-test and none of them clustered. Clustering roughly doubled the
standard errors the one time it was measured here.
"""
from __future__ import annotations

import numpy as np


def ols_cluster(y: np.ndarray, X: np.ndarray, clusters: np.ndarray) -> dict:
    """OLS with cluster-robust (Liang-Zeger) standard errors.

    The sandwich replaces the independence assumption with independence
    BETWEEN clusters only, letting observations inside a cluster correlate
    arbitrarily. With one observation per cluster it reduces to White's HC0.

    Returns coefficients, standard errors, t-statistics and the cluster count.
    """
    y = np.asarray(y, dtype="float64").ravel()
    X = np.asarray(X, dtype="float64")
    if X.ndim == 1:
        X = X[:, None]
    n, k = X.shape
    if len(y) != n or len(clusters) != n:
        raise ValueError("y, X and clusters must have the same length")

    xtx_inv = np.linalg.pinv(X.T @ X)
    beta = xtx_inv @ (X.T @ y)
    resid = y - X @ beta

    # Sum the outer products of per-cluster score vectors, not per-observation
    # ones. That is the whole difference: within a cluster the residuals are
    # allowed to move together instead of being assumed to cancel.
    codes, inv = np.unique(np.asarray(clusters), return_inverse=True)
    g = len(codes)
    scores = np.zeros((g, k))
    np.add.at(scores, inv, X * resid[:, None])
    meat = scores.T @ scores

    # Standard finite-sample correction (Cameron-Gelbach-Miller / Stata).
    c = (g / max(g - 1, 1)) * ((n - 1) / max(n - k, 1))
    cov = c * (xtx_inv @ meat @ xtx_inv)
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(se > 0, beta / se, 0.0)
    return {"beta": beta, "se": se, "t": t, "n": n, "clusters": g}
