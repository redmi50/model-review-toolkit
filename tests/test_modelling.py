"""The shared evaluation primitives.

The whole toolkit rests on these being deterministic and on the holdout being
defined once, as indices, so that a scorer and a structural check cannot disagree
about which rows were held out. Each function here is small enough that the test
can state its contract outright.
"""

from __future__ import annotations

import numpy as np

from model_review.modelling import (
    CV_FOLDS,
    HOLDOUT_TEST_SIZE,
    SEED,
    holdout_indices,
    linear_model,
    rank_by_correlation,
    split_holdout,
    stratified_folds,
)


def problem(n: int = 300, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """A small balanced problem with a real, non trivial signal."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, size=(n, 4))
    y = (1.5 * X[:, 0] - X[:, 1] + rng.normal(0, 1.0, n) > 0).astype(int)
    return X, y


class TestHoldout:
    def test_the_holdout_is_the_documented_size(self):
        _, y = problem()
        _, test = holdout_indices(y)
        assert len(test) == round(len(y) * HOLDOUT_TEST_SIZE)

    def test_the_two_sides_do_not_overlap(self):
        _, y = problem()
        train, test = holdout_indices(y)
        assert set(train).isdisjoint(set(test))
        assert len(train) + len(test) == len(y)

    def test_the_holdout_preserves_the_class_balance(self):
        _, y = problem()
        train, test = holdout_indices(y)
        assert abs(y[train].mean() - y[test].mean()) < 0.05

    def test_the_holdout_is_the_same_every_call(self):
        _, y = problem()
        first_train, first_test = holdout_indices(y)
        second_train, second_test = holdout_indices(y)
        assert np.array_equal(first_train, second_train)
        assert np.array_equal(first_test, second_test)

    def test_splitting_returns_the_arrays_on_the_same_rows(self):
        X, y = problem()
        X_train, X_test, y_train, y_test = split_holdout(X, y)
        train, test = holdout_indices(y)
        assert np.array_equal(X_train, X[train])
        assert np.array_equal(X_test, X[test])
        assert np.array_equal(y_test, y[test])

    def test_the_holdout_moves_when_the_seed_would_have(self):
        """A different problem, not a different seed, is the source of variation."""
        _, y_a = problem(seed=1)
        _, y_b = problem(seed=2)
        assert not np.array_equal(holdout_indices(y_a)[1], holdout_indices(y_b)[1])


class TestFolds:
    def test_the_fold_count_is_the_documented_one(self):
        assert stratified_folds().get_n_splits() == CV_FOLDS

    def test_every_row_is_tested_exactly_once_across_the_folds(self):
        X, y = problem()
        tested = np.zeros(len(y), dtype=int)
        for _, test in stratified_folds().split(X, y):
            tested[test] += 1
        assert (tested == 1).all()

    def test_the_folds_are_the_same_every_call(self):
        X, y = problem()
        first = [test.tolist() for _, test in stratified_folds().split(X, y)]
        second = [test.tolist() for _, test in stratified_folds().split(X, y)]
        assert first == second

    def test_the_folds_are_stratified(self):
        X, y = problem()
        overall = y.mean()
        for _, test in stratified_folds().split(X, y):
            assert abs(y[test].mean() - overall) < 0.10


class TestEstimators:
    def test_the_linear_model_learns_the_signal(self):
        X, y = problem()
        model = linear_model().fit(X, y)
        accuracy = (model.predict(X) == y).mean()
        assert accuracy > 0.65

    def test_the_linear_model_produces_probabilities(self):
        X, y = problem()
        probabilities = linear_model().fit(X, y).predict_proba(X)[:, 1]
        assert probabilities.min() >= 0.0
        assert probabilities.max() <= 1.0

    def test_the_linear_model_is_the_same_every_time(self):
        X, y = problem()
        first = linear_model().fit(X, y).predict_proba(X)[:, 1]
        second = linear_model().fit(X, y).predict_proba(X)[:, 1]
        assert np.array_equal(first, second)

    def test_the_seed_is_fixed(self):
        assert SEED == 0


class TestRankByCorrelation:
    def test_it_returns_the_requested_number_of_columns(self):
        X, y = problem()
        assert rank_by_correlation(X, y, 2).shape == (2,)

    def test_it_returns_indices_into_the_columns(self):
        X, y = problem()
        indices = rank_by_correlation(X, y, 2)
        assert indices.min() >= 0
        assert indices.max() < X.shape[1]

    def test_it_finds_a_column_that_is_a_copy_of_the_label(self):
        X, y = problem()
        with_leak = np.column_stack([X, y + 0.001])
        assert rank_by_correlation(with_leak, y, 1)[0] == X.shape[1]

    def test_it_orders_the_columns_by_the_strength_of_the_relationship(self):
        rng = np.random.default_rng(3)
        y = rng.integers(0, 2, size=400)
        noise = rng.normal(0, 1, size=(400, 1))
        strong = (y * 2 - 1).reshape(-1, 1) * 3.0
        X = np.column_stack([noise, strong])
        assert rank_by_correlation(X, y, 2)[0] == 1

    def test_a_constant_column_does_not_break_the_ranking(self):
        X, y = problem()
        with_constant = np.column_stack([X, np.zeros(len(y))])
        indices = rank_by_correlation(with_constant, y, 2)
        assert X.shape[1] not in indices

    def test_it_can_be_applied_to_a_subset_of_rows(self):
        """This is what lets a check redo a selection inside a fold."""
        X, y = problem()
        subset = np.arange(0, len(y), 2)
        indices = rank_by_correlation(X[subset], y[subset], 2)
        assert indices.shape == (2,)

    def test_the_ranking_can_differ_between_two_subsets_of_rows(self):
        """A selection made on every row is not the selection made inside a fold."""
        rng = np.random.default_rng(4)
        n, p = 40, 500
        X = rng.normal(0, 1, size=(n, p))
        y = rng.integers(0, 2, size=n)
        every_row = set(rank_by_correlation(X, y, 10).tolist())
        first_half = set(rank_by_correlation(X[: n // 2], y[: n // 2], 10).tolist())
        assert every_row != first_half
