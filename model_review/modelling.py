"""The fixed evaluation primitives every check and fixture shares.

A methodology review is only meaningful if the reviewer's own numbers are stable.
Everything in this module is deterministic: fixed seeds, fixed fold counts, one
estimator per job, and no tuning. That way the difference between a reported score
and a recomputed score is a difference in the *evaluation*, which is what the
toolkit is reviewing, and never a difference in the reviewer's estimator.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Every split in the toolkit uses these. They are stated here rather than inline so
# a reader can see the whole protocol at once and disagree with it in one place.
HOLDOUT_TEST_SIZE = 0.30
SEED = 0
CV_FOLDS = 5
FOREST_TREES = 250
PERMUTATION_REPEATS = 20


def linear_model():
    """A scaled logistic regression, the estimator used for scoring checks.

    One fixed model keeps a comparison between a flawed evaluation and an honest
    one a comparison of the evaluation rather than of the estimator. It has no
    hyperparameters that matter here and it trains in milliseconds.
    """
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))


def forest():
    """The estimator used by the ranking check, fixed for reproducibility."""
    return RandomForestClassifier(n_estimators=FOREST_TREES, random_state=SEED)


def stratified_folds(n_splits: int = CV_FOLDS) -> StratifiedKFold:
    """The cross validation splitter used by every fold based check."""
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)


def holdout_indices(y: np.ndarray, test_size: float = HOLDOUT_TEST_SIZE):
    """Row indices for the fixed stratified holdout every single split report uses.

    The split is defined once, on indices, so the scorer and the overlap check
    cannot disagree about which rows were held out.
    """
    y = np.asarray(y)
    index = np.arange(y.shape[0])
    train, test = train_test_split(
        index, test_size=test_size, random_state=SEED, stratify=y
    )
    return train, test


def split_holdout(X: np.ndarray, y: np.ndarray, test_size: float = HOLDOUT_TEST_SIZE):
    """The fixed stratified holdout, applied to a design matrix and its labels."""
    train, test = holdout_indices(y, test_size)
    return X[train], X[test], y[train], y[test]


def rank_by_correlation(X: np.ndarray, y: np.ndarray, keep: int) -> np.ndarray:
    """Indices of the ``keep`` columns most correlated with a binary label.

    This is the univariate screen a project runs when it selects features. It is
    written as a plain correlation so the same rule can be applied to any subset of
    rows, which is what lets a check redo the selection inside a fold.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    centred = X - X.mean(axis=0)
    labels = y - y.mean()
    spread = np.sqrt((centred**2).sum(axis=0) * (labels**2).sum())
    spread[spread == 0] = np.inf
    correlation = np.abs(centred.T @ labels / spread)
    return np.argsort(-correlation)[:keep]
