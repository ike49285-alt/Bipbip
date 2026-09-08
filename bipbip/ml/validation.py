"""Purged walk-forward validation.

Standard k-fold cross-validation is catastrophically wrong for this problem,
in two ways that both inflate results:

1. It shuffles, so the model trains on the future and predicts the past.
2. Even chronological splits leak, because a label at bar `i` is resolved by
   bars up to `event_end[i]`. A training sample whose outcome resolves inside
   the validation window has effectively seen validation data.

So splits are chronological and PURGED: any training sample whose label window
reaches into the validation fold is dropped, plus an embargo of further bars to
account for serial correlation. This is the López de Prado construction, and
without it a model with no edge routinely reports an encouraging AUC.

`permutation_test` is the other half of the defence. It shuffles the labels and
re-runs the whole pipeline, showing what score the procedure produces on data
with no signal whatsoever. If the real score sits inside that distribution,
there is nothing there - however good it looked.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class PurgedWalkForward:
    """Chronological expanding-window splits with purging and an embargo."""

    def __init__(self, n_splits: int = 5, embargo_bars: int = 60, min_train: int = 500):
        self.n_splits = n_splits
        self.embargo_bars = embargo_bars
        self.min_train = min_train

    def split(self, n: int, event_end: np.ndarray):
        """Yield ``(train_idx, val_idx)`` position arrays."""
        if n <= self.min_train:
            raise ValueError(
                f"{n} samples is fewer than min_train={self.min_train}; "
                "there is not enough history to validate anything yet"
            )

        fold_size = (n - self.min_train) // self.n_splits
        if fold_size <= 0:
            raise ValueError("not enough data for the requested number of splits")

        for k in range(self.n_splits):
            val_start = self.min_train + k * fold_size
            val_end = val_start + fold_size if k < self.n_splits - 1 else n
            val_idx = np.arange(val_start, val_end)

            # Purge: drop training samples whose label resolves at or after the
            # validation window opens, less an embargo for serial correlation.
            cutoff = val_start - self.embargo_bars
            candidate = np.arange(0, val_start)
            train_idx = candidate[event_end[candidate] < cutoff]

            if len(train_idx) < 50:
                continue
            yield train_idx, val_idx


def trade_metrics(proba: np.ndarray, y: np.ndarray, rets: np.ndarray, threshold: float) -> dict:
    """Score by money, not by accuracy.

    Accuracy and AUC are diagnostics. What decides whether a model is worth
    deploying is the net return of the trades it would actually have taken at
    its operating threshold.
    """
    take = proba >= threshold
    n = int(take.sum())
    if n == 0:
        return {"n_trades": 0, "hit_rate": float("nan"), "mean_ret_bps": 0.0,
                "total_ret_bps": 0.0, "selectivity": 0.0}
    taken_rets = rets[take]
    return {
        "n_trades": n,
        "hit_rate": float(np.nanmean(y[take])),
        "mean_ret_bps": float(np.nanmean(taken_rets) * 10_000),
        "total_ret_bps": float(np.nansum(taken_rets) * 10_000),
        "selectivity": n / len(proba),
    }


def walk_forward_evaluate(
    make_model,
    X: pd.DataFrame,
    y: np.ndarray,
    rets: np.ndarray,
    event_end: np.ndarray,
    n_splits: int = 5,
    embargo_bars: int = 60,
    min_train: int = 500,
    threshold: float = 0.5,
) -> dict:
    """Run the model across purged folds; report out-of-sample results only."""
    from sklearn.metrics import roc_auc_score

    splitter = PurgedWalkForward(n_splits, embargo_bars, min_train)
    Xv = X.to_numpy(dtype="float64")

    fold_rows, oos_proba, oos_idx = [], [], []
    for fold, (tr, va) in enumerate(splitter.split(len(X), event_end)):
        if len(np.unique(y[tr])) < 2:
            continue  # a single-class fold cannot train a classifier
        model = make_model().fit(Xv[tr], y[tr])
        p = model.predict_proba(Xv[va])
        oos_proba.append(p)
        oos_idx.append(va)

        row = {"fold": fold, "n_train": len(tr), "n_val": len(va)}
        row.update(trade_metrics(p, y[va], rets[va], threshold))
        try:
            row["auc"] = float(roc_auc_score(y[va], p)) if len(np.unique(y[va])) > 1 else float("nan")
        except ValueError:
            row["auc"] = float("nan")
        fold_rows.append(row)

    if not fold_rows:
        return {"folds": [], "note": "no usable folds - not enough history"}

    proba = np.concatenate(oos_proba)
    idx = np.concatenate(oos_idx)
    overall = trade_metrics(proba, y[idx], rets[idx], threshold)
    try:
        overall["auc"] = float(roc_auc_score(y[idx], proba)) if len(np.unique(y[idx])) > 1 else float("nan")
    except ValueError:
        overall["auc"] = float("nan")

    return {"folds": fold_rows, "overall": overall,
            "oos_proba": proba, "oos_idx": idx, "frame": pd.DataFrame(fold_rows)}


def permutation_test(
    make_model, X, y, rets, event_end, n_permutations: int = 20, seed: int = 0, **kw
) -> dict:
    """Re-run the whole evaluation on shuffled labels.

    This measures what the PROCEDURE scores on pure noise. A real score inside
    this distribution means the apparent edge is an artefact of the search, not
    a property of the market. It is the cheapest insurance against fooling
    yourself that exists.
    """
    rng = np.random.default_rng(seed)
    real = walk_forward_evaluate(make_model, X, y, rets, event_end, **kw)
    if not real.get("folds"):
        return {"note": real.get("note", "evaluation failed")}

    null_scores = []
    for _ in range(n_permutations):
        perm = rng.permutation(len(y))
        r = walk_forward_evaluate(make_model, X, y[perm], rets[perm], event_end, **kw)
        if r.get("folds"):
            null_scores.append(r["overall"]["mean_ret_bps"])

    null = np.array(null_scores, dtype="float64")
    observed = real["overall"]["mean_ret_bps"]
    n_trades = real["overall"]["n_trades"]

    if len(null) == 0:
        return {"note": "no usable permutations"}

    # The +1 convention: a permutation test can never legitimately report p=0,
    # and pretending otherwise overstates significance at small permutation
    # counts. The finest p-value resolvable is 1/(n+1).
    p_value = float((1 + (null >= observed).sum()) / (1 + len(null)))
    null_std = float(null.std())
    z = (observed - float(null.mean())) / null_std if null_std > 0 else 0.0
    null_best = float(null.max())

    # Verdicts are deliberately hard to earn. Every clause below exists because
    # its absence produced a false positive on data known to contain no signal.
    if n_trades < 30:
        verdict = (
            f"inconclusive: only {n_trades} out-of-sample trades. A threshold "
            "this selective picks lucky samples rather than predicting."
        )
    elif observed <= null_best:
        verdict = (
            f"indistinguishable from noise: shuffled labels reached "
            f"{null_best:+.2f} bps by luck alone, beating the observed "
            f"{observed:+.2f} bps."
        )
    elif p_value > 0.05:
        verdict = f"indistinguishable from noise (p={p_value:.3f})."
    elif len(null) < 50:
        verdict = (
            f"p={p_value:.3f} on only {len(null)} permutations, which cannot "
            "resolve significance this fine. Re-run with more before believing it."
        )
    else:
        verdict = (
            f"survives the permutation test (p={p_value:.3f}, z={z:+.1f}). "
            "Necessary, not sufficient - confirm out of sample on fresh data."
        )

    return {
        "observed_mean_ret_bps": observed,
        "n_trades": n_trades,
        "null_mean_bps": float(null.mean()),
        "null_std_bps": null_std,
        "null_best_bps": null_best,
        "z_score": z,
        "p_value": p_value,
        "n_permutations": len(null),
        "verdict": verdict,
    }
