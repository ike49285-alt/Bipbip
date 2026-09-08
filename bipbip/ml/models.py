"""Models and the baselines they must beat.

The net is deliberately small. With a few hundred usable observations, a large
network does not learn a market - it memorises one. Capacity here is set by the
data, not by ambition, and the regularisation is heavy on purpose.

The baselines are not decoration. A model is only interesting if it beats
"always trade" and "never trade", and on a trending sample "always trade" is
a surprisingly strong opponent. Reporting a model's accuracy without them is
how a 55%-accurate coin flip gets mistaken for an edge.
"""
from __future__ import annotations

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class Model:
    """Common interface: fit, then predict a probability of a profitable trade."""

    name = "base"

    def fit(self, X, y):
        raise NotImplementedError

    def predict_proba(self, X) -> np.ndarray:
        raise NotImplementedError


class SklearnModel(Model):
    """Wraps a sklearn pipeline. Imputation and scaling are fitted on the
    training fold only - fitting them on all data leaks the validation
    distribution and is one of the quietest ways to fake a good result."""

    def __init__(self, estimator, name):
        self.name = name
        self.pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", estimator),
        ])

    def fit(self, X, y):
        self.pipe.fit(X, y)
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self.pipe.predict_proba(X)[:, 1]


def make_mlp(hidden=(16, 8), alpha=1.0, seed=0) -> SklearnModel:
    """A small, heavily regularised MLP.

    `alpha=1.0` is strong L2 by sklearn's defaults (0.0001). Early stopping
    holds out the last 15% of the training fold, which stays chronological
    because the data is passed in time order and shuffle is off.
    """
    return SklearnModel(
        MLPClassifier(
            hidden_layer_sizes=hidden,
            alpha=alpha,
            max_iter=800,
            early_stopping=True,
            n_iter_no_change=20,
            validation_fraction=0.15,
            shuffle=False,
            random_state=seed,
        ),
        name=f"mlp{list(hidden)}",
    )


def make_logistic(C=0.1, seed=0) -> SklearnModel:
    """The baseline the net has to justify itself against. It very often wins
    at this sample size, and that result is a finding, not a failure."""
    return SklearnModel(
        LogisticRegression(C=C, max_iter=2000, random_state=seed), name="logistic"
    )


class AlwaysEnter(Model):
    """Takes every opportunity. Strong in a trending sample; the honest floor."""

    name = "always_enter"

    def fit(self, X, y):
        return self

    def predict_proba(self, X) -> np.ndarray:
        return np.ones(len(X))


class BaseRate(Model):
    """Predicts the training base rate for everything - zero discrimination."""

    name = "base_rate"

    def fit(self, X, y):
        self.rate = float(np.mean(y)) if len(y) else 0.5
        return self

    def predict_proba(self, X) -> np.ndarray:
        return np.full(len(X), getattr(self, "rate", 0.5))


MODELS = {
    "mlp": make_mlp,
    "logistic": make_logistic,
    "always_enter": lambda **kw: AlwaysEnter(),
    "base_rate": lambda **kw: BaseRate(),
}
