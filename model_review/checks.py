"""The checks.

Each check recomputes something the project reported, or measures a structural
property of how the project was evaluated, and turns the difference into a
finding. Nothing here reads a project's prose. A check decides whether it applies
from the project's stored evaluation settings, then measures, and the severity
comes from the measured quantity.

Two kinds of quantity are used. An *optimism gap* is the amount by which the
reported score exceeded the score from the correct evaluation; it drives the
scoring checks. A *share* is a proportion of something that should not have it,
such as the fraction of test rows that also appear in training; it drives the
structural checks. Both are mapped through the shared severity bands in
``findings``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

from .datasets import Project
from .findings import Finding, severity_for_gap, severity_for_share
from .modelling import (
    PERMUTATION_REPEATS,
    forest,
    holdout_indices,
    linear_model,
    rank_by_correlation,
    split_holdout,
    stratified_folds,
)

# A feature is treated as a leak when it separates the label this cleanly on its
# own. The threshold is high on purpose: the check reports a leak only when a
# single column is close to a copy of the outcome, which is a defect rather than a
# strong predictor.
LEAK_CORRELATION = 0.98

# A model must clear the no skill score by this much to be worth reporting. The
# shortfall below it is what the weak baseline check measures.
MIN_LIFT = 0.05

# The positive rate under which an accuracy headline is treated as uninformative.
RARE_EVENT_RATE = 0.10

# Bands for the structural checks. Each is stated next to the quantity it applies
# to, and the reason for the number is that the quantity itself is decisive well
# below the band: a test set where most rows were memorised is not a test set.
REPLICATION_BANDS = {"high": 0.50, "medium": 0.05}
IMPORTANCE_BANDS = {"high": 0.25, "medium": 0.10}
MISSED_EVENT_BANDS = {"high": 0.50, "medium": 0.20}

# Bands for the weak baseline check, stated as the shortfall below MIN_LIFT. A
# model a hair above the no skill line falls the whole way short, so that is the
# high band, and half the way short is the medium band. Stating the bands this way
# keeps the high band reachable: a fixed shortfall of the full minimum would never
# fire for any score above the no skill line.
BASELINE_BANDS = {"high": MIN_LIFT / 2, "medium": MIN_LIFT / 10}

# A reported score is published to three decimal places, so a recomputation that
# agrees to within half of the last published digit agrees exactly as far as the
# report can show. Differences below that are rounding, not a discrepancy, and are
# not findings. The bands above the tolerance are stated in the same units.
PUBLICATION_TOLERANCE = 0.0005
REPRODUCTION_BANDS = {"high": 0.02, "medium": 0.002}


def _estimator(project: Project):
    """The estimator the project used, so a check never scores a different model."""
    return forest if project.estimator == "forest" else linear_model


def score_with_protocol(project: Project, X: np.ndarray) -> dict[str, float]:
    """Run the project's own protocol on a design matrix and return its metrics.

    This is the toolkit's own evaluation, so it is held to the same protocol the
    project used: a holdout stays a holdout, cross validation stays cross
    validation, and the estimator is the one the project chose. The only thing a
    caller changes is which columns are handed in, which is how the leakage check
    removes the suspicious column and the selection check redoes the screen.
    """
    estimator = _estimator(project)
    if project.protocol == "holdout":
        X_train, X_test, y_train, y_test = split_holdout(X, project.y)
        fitted = estimator().fit(X_train, y_train)
        return {
            "roc_auc": float(roc_auc_score(y_test, fitted.predict_proba(X_test)[:, 1])),
            "accuracy": float(accuracy_score(y_test, fitted.predict(X_test))),
        }
    auc = cross_val_score(
        estimator(), X, project.y, cv=stratified_folds(), scoring="roc_auc"
    )
    accuracy = cross_val_score(
        estimator(), X, project.y, cv=stratified_folds(), scoring="accuracy"
    )
    return {"roc_auc": float(auc.mean()), "accuracy": float(accuracy.mean())}


def _reported(project: Project, metric: str) -> float:
    """The project's published value for one metric."""
    return float(project.reported_metrics[metric])


def reproduce_reported_score(project: Project) -> list[Finding]:
    """Re-run the project's own protocol and check the published number.

    This runs first for a reason. If the published figure cannot be reproduced
    under the protocol the project describes, the review has found an arithmetic
    or reporting problem, and everything else the review says about methodology is
    beside the point until that is resolved.
    """
    findings: list[Finding] = []
    recomputed_all = score_with_protocol(project, project.X)
    for metric in sorted(project.reported_metrics):
        if metric not in recomputed_all:
            continue
        recomputed = recomputed_all[metric]
        difference = abs(recomputed - _reported(project, metric))
        if difference <= PUBLICATION_TOLERANCE:
            continue
        severity = severity_for_share(difference, **REPRODUCTION_BANDS)
        if severity is None:
            continue
        findings.append(
            Finding(
                check="reproduce_reported_score",
                severity=severity,
                summary=(
                    f"The reported {metric} of {_reported(project, metric):.3f} does "
                    f"not reproduce: the stated protocol gives "
                    f"{recomputed:.3f}."
                ),
                recommendation=(
                    "Reproduce the published figure from the stored code and data "
                    "before any methodology question is settled, and correct the "
                    "number if the two disagree."
                ),
                evidence={
                    "metric": metric,
                    "reported": _reported(project, metric),
                    "recomputed": round(recomputed, 4),
                    "difference": round(difference, 4),
                    "protocol": project.protocol,
                    "estimator": project.estimator,
                },
            )
        )
    return findings


def leakage(project: Project) -> list[Finding]:
    """Look for a single column that is a near copy of the label, then measure it.

    The check is deliberately narrow. It computes each column's correlation with
    the label, and only when one exceeds the leak threshold does it recompute the
    project's score without that column. The size of the drop is the evidence, so
    the finding is a measured quantity rather than a judgement about which fields
    look suspicious.
    """
    X = project.X
    correlations = []
    for j in range(X.shape[1]):
        column = X[:, j].astype(float)
        if column.std() == 0:
            correlations.append(0.0)
            continue
        correlations.append(abs(float(np.corrcoef(column, project.y)[0, 1])))
    correlations = np.asarray(correlations)
    suspects = np.flatnonzero(correlations >= LEAK_CORRELATION)
    if suspects.size == 0:
        return []

    suspected = [project.feature_names[int(j)] for j in suspects]
    keep = np.flatnonzero(correlations < LEAK_CORRELATION)

    metric = "roc_auc"
    if metric not in project.reported_metrics:
        return []
    reported = _reported(project, metric)
    if keep.size == 0:
        return []
    without = score_with_protocol(project, X[:, keep])[metric]
    gap = reported - without
    severity = severity_for_gap(gap)
    if severity is None:
        return []

    return [
        Finding(
            check="leakage",
            severity=severity,
            summary=(
                f"{', '.join(suspected)} separates the label almost perfectly on its "
                f"own, and the reported {metric} of {reported:.3f} falls to "
                f"{without:.3f} without it."
            ),
            recommendation=(
                "Drop the flagged column, confirm with the data owner that it is "
                "recorded after the outcome, and rebuild the score from predictors "
                "that are known at scoring time."
            ),
            evidence={
                "metric": metric,
                "suspected_features": suspected,
                "correlations": {
                    name: round(float(correlations[int(j)]), 4)
                    for name, j in zip(suspected, suspects)
                },
                "reported": reported,
                "without_suspects": round(without, 4),
                "gap": round(gap, 4),
            },
        )
    ]


def temporal_evaluation(project: Project) -> list[Finding]:
    """Compare a shuffled split with a forward chaining split on ordered rows.

    Only projects whose rows are ordered in time are checked, and the honest
    number is the one a forward chaining split gives, because that is the only
    split that matches how the model will be used. The gap between the two is the
    amount of future the shuffled split let the model read.
    """
    if project.split != "temporal":
        return []

    metric = "roc_auc"
    if metric not in project.reported_metrics:
        return []
    reported = _reported(project, metric)
    shuffled = score_with_protocol(project, project.X)[metric]
    forward_scores = cross_val_score(
        _estimator(project)(),
        project.X,
        project.y,
        cv=TimeSeriesSplit(5),
        scoring="roc_auc",
    )
    forward = float(forward_scores.mean())
    gap = reported - forward
    severity = severity_for_gap(gap)
    if severity is None:
        return []

    return [
        Finding(
            check="temporal_evaluation",
            severity=severity,
            summary=(
                f"Rows are ordered in time and were shuffled before the split. The "
                f"reported {metric} of {reported:.3f} compares with {forward:.3f} "
                f"under a forward chaining split and {shuffled:.3f} under the "
                f"shuffled split."
            ),
            recommendation=(
                "Evaluate with a forward chaining split such as TimeSeriesSplit, "
                "which trains on earlier rows and scores later ones, and report the "
                "result the production ordering would produce."
            ),
            evidence={
                "metric": metric,
                "reported": reported,
                "shuffled_cv": round(shuffled, 4),
                "forward_chaining": round(forward, 4),
                "gap": round(gap, 4),
                "fold_scores": [round(float(v), 4) for v in forward_scores],
            },
        )
    ]


def selection_outside_cv(project: Project) -> list[Finding]:
    """Redo a global feature selection inside each training fold.

    The project selected columns using every row and then cross validated the
    survivors, so the test rows influenced which columns were kept. The check
    repeats the same selection rule inside each fold, where the test rows are not
    visible, and measures the difference.
    """
    if project.selection != "global" or project.candidates is None:
        return []

    metric = "roc_auc"
    if metric not in project.reported_metrics:
        return []
    reported = _reported(project, metric)
    candidates = project.candidates
    honest: list[float] = []
    for train, test in stratified_folds().split(candidates, project.y):
        inner = rank_by_correlation(candidates[train], project.y[train], project.n_selected)
        fitted = linear_model().fit(candidates[train][:, inner], project.y[train])
        honest.append(
            float(
                roc_auc_score(
                    project.y[test],
                    fitted.predict_proba(candidates[test][:, inner])[:, 1],
                )
            )
        )
    honest_mean = float(np.mean(honest))
    gap = reported - honest_mean
    severity = severity_for_gap(gap)
    if severity is None:
        return []

    return [
        Finding(
            check="selection_outside_cv",
            severity=severity,
            summary=(
                f"{project.n_selected} features were chosen from "
                f"{candidates.shape[1]} candidates using every row, then scored on "
                f"the same rows. The reported {metric} of {reported:.3f} compares "
                f"with {honest_mean:.3f} when the same selection is redone inside "
                f"each fold."
            ),
            recommendation=(
                "Move the feature selection inside the cross validation loop, or into "
                "a pipeline, so each fold selects its columns from its training rows "
                "only."
            ),
            evidence={
                "metric": metric,
                "reported": reported,
                "in_fold_selection": round(honest_mean, 4),
                "gap": round(gap, 4),
                "candidate_features": int(candidates.shape[1]),
                "selected_features": project.n_selected,
                "fold_scores": [round(v, 4) for v in honest],
            },
        )
    ]


def row_replication(project: Project) -> list[Finding]:
    """Measure how much of the test set is a repeat of the training set.

    This is a structural check rather than a scoring one. Duplicated rows do not
    move an area under the curve much, because a repeated row is still ranked
    correctly, so a score gap would understate the problem. What matters is the
    proportion of test rows whose feature vector was already seen in training: on
    those rows the test set measures recall of memorised rows, not generalisation.
    """
    X = project.X
    y = project.y
    if project.protocol == "holdout":
        folds = [holdout_indices(y)]
    else:
        folds = list(stratified_folds().split(X, y))

    test_rows = 0
    overlapping = 0
    for index_train, index_test in folds:
        seen = {tuple(row) for row in X[index_train]}
        test_rows += len(index_test)
        overlapping += sum(1 for row in X[index_test] if tuple(row) in seen)

    share = overlapping / test_rows
    severity = severity_for_share(share, **REPLICATION_BANDS)
    if severity is None:
        return []

    return [
        Finding(
            check="row_replication",
            severity=severity,
            summary=(
                f"{share:.1%} of the test rows carry feature values that already "
                f"appear in the training rows ({overlapping} of {test_rows}), so "
                f"the score is partly a measure of memorisation."
            ),
            recommendation=(
                "Deduplicate before splitting, or split by the unit that was "
                "duplicated, so no feature vector is present on both sides. If "
                "balancing is needed, do it inside the training fold only."
            ),
            evidence={
                "test_rows": int(test_rows),
                "overlapping_rows": int(overlapping),
                "overlap_share": round(share, 4),
                "protocol": project.protocol,
            },
        )
    ]


def class_imbalance_metric(project: Project) -> list[Finding]:
    """Check whether an accuracy headline can hide a rare class failure.

    The check applies when accuracy is the reported metric and the positive class
    is rare. It reports the base rate, the accuracy a constant predictor achieves,
    and the recall of the actual model, because together those three numbers show
    whether the headline distinguishes a working model from a rule of thumb. The
    severity comes from the share of positive cases the model misses.
    """
    if "accuracy" not in project.reported_metrics:
        return []
    if project.positive_rate > RARE_EVENT_RATE:
        return []

    reported = _reported(project, "accuracy")
    X_train, X_test, y_train, y_test = split_holdout(project.X, project.y)
    fitted = _estimator(project)().fit(X_train, y_train)
    predictions = fitted.predict(X_test)
    model_accuracy = float(accuracy_score(y_test, predictions))
    majority = int(y_train.mean() > 0.5)
    constant_accuracy = float(accuracy_score(y_test, np.full_like(y_test, majority)))
    missed = 1.0 - float(recall_score(y_test, predictions, zero_division=0))
    severity = severity_for_share(missed, **MISSED_EVENT_BANDS)
    if severity is None:
        return []

    return [
        Finding(
            check="class_imbalance_metric",
            severity=severity,
            summary=(
                f"Accuracy is the only reported metric on a class that is "
                f"{project.positive_rate:.1%} positive. The model scores "
                f"{model_accuracy:.3f} while a constant prediction of the majority "
                f"class scores {constant_accuracy:.3f}, and it misses {missed:.1%} of "
                f"the positive cases the metric is meant to summarise."
            ),
            recommendation=(
                "Report recall, precision and the confusion matrix alongside "
                "accuracy, choose a decision threshold against the cost of a miss, "
                "and state the base rate next to any headline rate."
            ),
            evidence={
                "reported_accuracy": reported,
                "base_rate": round(project.positive_rate, 4),
                "model_accuracy": round(model_accuracy, 4),
                "constant_predictor_accuracy": round(constant_accuracy, 4),
                "missed_positive_share": round(missed, 4),
            },
        )
    ]


def importance_method(project: Project) -> list[Finding]:
    """Compare impurity importance with permutation importance on held out rows.

    Impurity importance credits any column that can carve up the training sample,
    so a high cardinality identifier collects weight without carrying signal.
    Permutation importance on rows the model has not seen does not credit it. The
    check measures the share of the published importance weight held by columns
    whose permutation importance is at or below zero.
    """
    if project.ranking_method != "impurity":
        return []
    if project.estimator != "forest":
        return []

    X_train, X_test, y_train, y_test = split_holdout(project.X, project.y)
    fitted = forest().fit(X_train, y_train)
    impurity = fitted.feature_importances_
    permutation = permutation_importance(
        fitted,
        X_test,
        y_test,
        n_repeats=PERMUTATION_REPEATS,
        random_state=0,
        scoring="roc_auc",
    ).importances_mean

    unearned = np.flatnonzero(permutation <= 0.0)
    share = float(impurity[unearned].sum())
    severity = severity_for_share(share, **IMPORTANCE_BANDS)
    if severity is None:
        return []

    names = [project.feature_names[int(j)] for j in unearned]
    published = [name for name in project.published_drivers if name in names]
    summary = (
        f"{share:.1%} of the published importance weight went to columns that do "
        f"not improve the score on held out rows: {', '.join(names)}."
    )
    if published:
        summary += (
            f" The published driver list includes {', '.join(published)}, whose "
            f"permutation importance is at or below zero."
        )

    return [
        Finding(
            check="importance_method",
            severity=severity,
            summary=summary,
            recommendation=(
                "Report permutation importance on held out rows, and if a ranking must "
                "come from the forest, drop identifiers and high cardinality keys "
                "first. State which importance was used next to the ranking."
            ),
            evidence={
                "importance_method": project.ranking_method,
                "unearned_share": round(share, 4),
                "unearned_features": names,
                "published_drivers_without_measured_effect": published,
                "impurity_importance": {
                    name: round(float(impurity[j]), 4)
                    for j, name in enumerate(project.feature_names)
                },
                "permutation_importance": {
                    name: round(float(permutation[j]), 4)
                    for j, name in enumerate(project.feature_names)
                },
            },
        )
    ]


def baseline_lift(project: Project) -> list[Finding]:
    """Check that a reported ranking score clears the no skill score.

    A model that cannot beat the no skill score is not a model, and this check
    raises a finding when the reported score is within a defined margin of it. It
    is here so the toolkit can also say that a number is fine, rather than
    returning findings for everything it looks at.
    """
    if "roc_auc" not in project.reported_metrics:
        return []

    reported = _reported(project, "roc_auc")
    lift = reported - 0.5
    shortfall = max(0.0, MIN_LIFT - lift)
    severity = severity_for_share(shortfall, **BASELINE_BANDS)
    if severity is None:
        return []

    return [
        Finding(
            check="baseline_lift",
            severity=severity,
            summary=(
                f"The reported roc_auc of {reported:.3f} is only {lift:.3f} above the "
                f"no skill score of 0.500."
            ),
            recommendation=(
                "Compare against a no skill baseline and a simple rules based "
                "baseline before presenting the score, and establish that the model "
                "adds measurable value."
            ),
            evidence={
                "reported": reported,
                "no_skill_score": 0.5,
                "lift": round(lift, 4),
                "minimum_lift": MIN_LIFT,
            },
        )
    ]


@dataclass(frozen=True)
class Check:
    """One named check plus the predicate that decides whether it applies."""

    name: str
    description: str
    run: Callable[[Project], list[Finding]]


CHECKS: tuple[Check, ...] = (
    Check(
        "reproduce_reported_score",
        "Re-run the stated protocol and confirm the published figure.",
        reproduce_reported_score,
    ),
    Check(
        "leakage",
        "Find a column that is a near copy of the label and measure its effect.",
        leakage,
    ),
    Check(
        "temporal_evaluation",
        "Compare a shuffled split with a forward chaining split on ordered rows.",
        temporal_evaluation,
    ),
    Check(
        "selection_outside_cv",
        "Redo a global feature selection inside each training fold.",
        selection_outside_cv,
    ),
    Check(
        "row_replication",
        "Measure how much of the test set repeats the training set.",
        row_replication,
    ),
    Check(
        "class_imbalance_metric",
        "Check whether an accuracy headline can hide a rare class failure.",
        class_imbalance_metric,
    ),
    Check(
        "importance_method",
        "Compare impurity importance with permutation importance on held out rows.",
        importance_method,
    ),
    Check(
        "baseline_lift",
        "Check that a reported ranking score clears the no skill score.",
        baseline_lift,
    ),
)
