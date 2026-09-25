"""The fixture projects the checks are demonstrated on.

Each fixture is a small supervised learning problem with a documented flaw in how
it was evaluated, the score the original analyst reported, and everything a check
needs to recompute that score honestly. The data is generated from a fixed seed,
so every number the checks produce is reproducible on any machine.

The reported scores are stored as literals rather than computed at import time.
That is deliberate: a check compares the report against its own honest
recomputation, and the test suite asserts that the stored literal is what the
flawed evaluation actually produces. If either drifts, a test fails rather than a
report quietly becoming wrong.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.datasets import load_breast_cancer

from .modelling import rank_by_correlation


@dataclass
class Project:
    """One evaluated modelling project under review.

    Attributes:
        project_id: stable identifier used in reports and tests.
        title: human readable name.
        summary: what the project set out to do.
        X: the design matrix as it was presented to the model.
        y: the binary label.
        reported_metrics: the numbers the analyst published, keyed by metric name.
        reported_evaluation: a sentence describing how the score was produced.
        protocol: ``holdout`` or ``cross_validation``, as the analyst ran it.
        estimator: ``linear`` or ``forest``, the estimator the analyst used.
        split: ``random`` or ``temporal``, as the analyst ran it.
        time_index: a sortable position per row, when the data is ordered.
        feature_names: column names, for importance reporting.
        selection: ``global`` when features were chosen before the split.
        n_selected: how many features the selection kept.
        candidates: the full candidate matrix, when selection reduced it.
        ranking_method: how a published driver ranking was produced.
        published_drivers: the top names of a published driver ranking.
        notes: anything the report should state up front.
    """

    project_id: str
    title: str
    summary: str
    X: np.ndarray
    y: np.ndarray
    reported_metrics: dict[str, float]
    reported_evaluation: str
    protocol: str = "cross_validation"
    estimator: str = "linear"
    split: str = "random"
    time_index: np.ndarray | None = None
    feature_names: list[str] = field(default_factory=list)
    selection: str = "none"
    n_selected: int = 0
    candidates: np.ndarray | None = None
    ranking_method: str | None = None
    published_drivers: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.X = np.asarray(self.X, dtype=float)
        self.y = np.asarray(self.y, dtype=int)
        if self.candidates is not None:
            self.candidates = np.asarray(self.candidates, dtype=float)
            if self.candidates.shape[0] != self.y.shape[0]:
                raise ValueError(
                    f"Project {self.project_id!r} has {self.candidates.shape[0]} "
                    f"candidate rows but {self.y.shape[0]} labels."
                )
        if self.X.shape[0] != self.y.shape[0]:
            raise ValueError(
                f"Project {self.project_id!r} has {self.X.shape[0]} rows of features "
                f"but {self.y.shape[0]} labels."
            )
        if not self.feature_names:
            self.feature_names = [f"x{i}" for i in range(self.X.shape[1])]
        if len(self.feature_names) != self.X.shape[1]:
            raise ValueError(
                f"Project {self.project_id!r} names {len(self.feature_names)} features "
                f"but has {self.X.shape[1]} columns."
            )
        if self.protocol not in {"holdout", "cross_validation"}:
            raise ValueError(
                f"Unknown protocol {self.protocol!r}. Expected holdout or "
                "cross_validation."
            )
        if self.estimator not in {"linear", "forest"}:
            raise ValueError(
                f"Unknown estimator {self.estimator!r}. Expected linear or forest."
            )
        if self.split not in {"random", "temporal"}:
            raise ValueError(
                f"Unknown split {self.split!r}. Expected random or temporal."
            )
        if self.split == "temporal" and self.time_index is None:
            raise ValueError(
                f"Project {self.project_id!r} uses a temporal split but has no "
                "time_index."
            )
        if self.selection not in {"none", "global"}:
            raise ValueError(
                f"Unknown selection {self.selection!r}. Expected none or global."
            )
        if self.selection == "global" and self.candidates is None:
            raise ValueError(
                f"Project {self.project_id!r} used global selection but stores no "
                "candidate matrix, so the selection cannot be redone inside a fold."
            )
        if self.ranking_method not in {None, "impurity"}:
            raise ValueError(
                f"Unknown ranking method {self.ranking_method!r}. Expected None or "
                "impurity."
            )
        if self.ranking_method is not None and not self.published_drivers:
            raise ValueError(
                f"Project {self.project_id!r} published a driver ranking but lists no "
                "drivers."
            )
        unknown = [n for n in self.published_drivers if n not in self.feature_names]
        if unknown:
            raise ValueError(
                f"Project {self.project_id!r} publishes drivers that are not "
                f"features: {unknown}."
            )

    @property
    def n_rows(self) -> int:
        """Number of rows."""
        return int(self.X.shape[0])

    @property
    def positive_rate(self) -> float:
        """Fraction of rows in the positive class."""
        return float(self.y.mean())

    def as_dict(self) -> dict[str, object]:
        """A plain dictionary describing the project, without the arrays."""
        return {
            "project_id": self.project_id,
            "title": self.title,
            "summary": self.summary,
            "rows": self.n_rows,
            "features": int(self.X.shape[1]),
            "positive_rate": self.positive_rate,
            "reported_metrics": dict(self.reported_metrics),
            "reported_evaluation": self.reported_evaluation,
            "protocol": self.protocol,
            "estimator": self.estimator,
            "split": self.split,
            "selection": self.selection,
            "n_selected": self.n_selected,
            "ranking_method": self.ranking_method,
            "published_drivers": list(self.published_drivers),
            "notes": list(self.notes),
        }


def _clean_baseline_project() -> Project:
    """A project that was evaluated carefully, as a control.

    Nothing here is flawed on purpose. The holdout is taken once with a fixed
    seed, no column is a copy of the label, no selection ran outside the split, no
    row was replicated, the classes are balanced, and the score clears the no
    skill line. The toolkit should find nothing, and a test asserts that it does.
    Without this project the checks could pass by firing on everything.
    """
    rng = np.random.default_rng(11)
    n = 1500
    X = rng.normal(0, 1, size=(n, 5))
    logit = 0.9 * X[:, 0] + 0.7 * X[:, 1] - 0.5 * X[:, 2]
    y = (logit + rng.normal(0, 1.4, n) > 0).astype(int)
    return Project(
        project_id="clean_baseline",
        title="Retention model evaluated with a fixed holdout",
        summary=(
            "Predict a balanced retention outcome from five behavioural signals, "
            "with a single stratified holdout taken before any modelling."
        ),
        X=X,
        y=y,
        reported_metrics={"roc_auc": 0.810},
        reported_evaluation=(
            "A single stratified holdout, split before any preprocessing, reporting "
            "the area under the ROC curve."
        ),
        protocol="holdout",
        feature_names=["usage_trend", "support_tickets", "tenure", "channel_a", "channel_b"],
        notes=[
            "This project is a control: it is included so the review can show that a "
            "careful evaluation raises nothing.",
        ],
    )


def _leaky_feature_project() -> Project:
    """A project whose best feature is only known after the outcome occurs.

    The honest signal is deliberately weak: six noisy predictors carry a modest
    relationship to the label. The leaked column is the label plus a small amount
    of noise, which is what a field such as a closure date looks like once it has
    been encoded. Cross validation cannot see the problem, because the leaked
    value is present in every fold.
    """
    rng = np.random.default_rng(42)
    n = 2000
    X = rng.normal(0, 1, size=(n, 6))
    logit = 0.7 * X[:, 0] + 0.5 * X[:, 1]
    y = (logit + rng.normal(0, 1.6, n) > 0).astype(int)
    leaky = y + rng.normal(0, 0.05, n)
    X_full = np.column_stack([X, leaky])
    names = [f"signal_{i}" for i in range(6)] + ["account_closed_date"]
    return Project(
        project_id="leaky_feature",
        title="Churn model with a closure date feature",
        summary=(
            "Predict account churn from six behavioural predictors and an account "
            "closure date that is populated once an account has already closed."
        ),
        X=X_full,
        y=y,
        reported_metrics={"roc_auc": 1.000},
        reported_evaluation=(
            "Five fold stratified cross validation over all rows, reporting the mean "
            "area under the ROC curve."
        ),
        feature_names=names,
        notes=[
            "The closure date is recorded when an account closes, so it cannot be "
            "known at the moment a live account is scored.",
        ],
    )


def _temporal_drift_project() -> Project:
    """A project that shuffled time ordered rows.

    The label is a function of a smoothly drifting quantity, and a column records
    where in time each row sits. A random split puts late rows in training, which
    lets the model read the future. A forward chaining split reproduces the
    production situation, where only the past is available.
    """
    rng = np.random.default_rng(9)
    n = 1200
    position = np.arange(n) / n
    y = (position + rng.normal(0, 0.18, n) > 0.5).astype(int)
    X = np.column_stack(
        [
            position,
            rng.normal(0, 1, n),
            rng.normal(0, 1, n),
            rng.normal(0, 1, n),
        ]
    )
    return Project(
        project_id="temporal_drift",
        title="Demand model evaluated on shuffled time",
        summary=(
            "Predict a demand flag from a week index and three operational signals, "
            "evaluated with a shuffled split over eighteen months of history."
        ),
        X=X,
        y=y,
        reported_metrics={"roc_auc": 0.936},
        reported_evaluation=(
            "Five fold stratified cross validation with rows shuffled before the "
            "split, reporting the mean area under the ROC curve."
        ),
        split="temporal",
        time_index=np.arange(n),
        feature_names=["week_index", "ops_a", "ops_b", "ops_c"],
        notes=[
            "The rows are ordered in time and the label drifts with the week index.",
        ],
    )


def _selection_outside_cv_project() -> Project:
    """A project that chose its features using every row, then cross validated.

    With three thousand candidate columns and ninety rows, the best fifteen
    columns by univariate correlation are the ones that best fit the whole sample,
    including the rows they will later be tested on. Cross validating that subset
    measures how well the choice fits this sample, not how well it generalises.
    """
    rng = np.random.default_rng(21)
    n, p = 90, 3000
    X = rng.normal(0, 1, size=(n, p))
    y = (X[:, 0] + 0.8 * X[:, 1] + rng.normal(0, 1.2, n) > 0).astype(int)
    keep = rank_by_correlation(X, y, 15)
    selected = X[:, keep]
    return Project(
        project_id="selection_outside_cv",
        title="Biomarker screen with global feature selection",
        summary=(
            "Rank three thousand candidate biomarkers by correlation with the "
            "outcome, keep the best fifteen, and cross validate a classifier on them."
        ),
        X=selected,
        y=y,
        reported_metrics={"roc_auc": 0.928},
        reported_evaluation=(
            "The top fifteen biomarkers were chosen on the full dataset, then a five "
            "fold stratified cross validation was run on that subset."
        ),
        feature_names=[f"marker_{int(j)}" for j in keep],
        selection="global",
        n_selected=15,
        candidates=X,
        notes=[
            "There are ninety rows and three thousand candidate columns, and the "
            "selection step was allowed to see every row.",
        ],
    )


def _duplicate_rows_project() -> Project:
    """A project whose rows were replicated before the split.

    Replication is a common way to balance a rare class, and it is usually done to
    the whole table before the split is taken. The same feature vector then lands
    in both training and test, so the test set partly measures memorisation.
    """
    X_all, y_all = load_breast_cancer(return_X_y=True)
    rng = np.random.default_rng(23)
    positive = np.flatnonzero(y_all == 1)
    negative = np.flatnonzero(y_all == 0)
    take = min(len(positive), len(negative))
    index = np.concatenate(
        [
            rng.choice(positive, size=take, replace=False),
            rng.choice(negative, size=take, replace=False),
        ]
    )
    X_base = X_all[index]
    y_base = y_all[index]
    copies = 5
    repeated = np.concatenate([np.arange(len(y_base))] * copies)
    return Project(
        project_id="duplicate_rows",
        title="Tumour classifier trained on a replicated table",
        summary=(
            "Balance a two class tumour problem by replicating every row five times, "
            "then take a single stratified holdout."
        ),
        X=X_base[repeated],
        y=y_base[repeated],
        reported_metrics={"roc_auc": 0.997},
        reported_evaluation=(
            "Every row was replicated five times to balance the classes, then a "
            "single seventy percent holdout was scored."
        ),
        protocol="holdout",
        notes=[
            "The replication was applied to the whole table before the split.",
        ],
    )


def _imbalanced_metric_project() -> Project:
    """A project that measured a rare class problem with one number.

    The positive class is a little under three percent of the data, the reported
    metric is accuracy, and the positive class is the one that matters. Nothing
    here needs to be recomputed across folds to show the problem: the base rate
    and the behaviour of a constant predictor settle it. The predictor below is
    given a modest but real signal, so the flaw is the metric rather than a model
    that does not work.
    """
    rng = np.random.default_rng(31)
    n = 3000
    X = rng.normal(0, 1, size=(n, 5))
    logit = 0.35 * X[:, 0] + 0.4 * X[:, 1]
    y = (logit + rng.normal(0, 1.0, n) > 2.05).astype(int)
    return Project(
        project_id="imbalanced_metric",
        title="Rare event classifier reported on accuracy",
        summary=(
            "Flag a rare adverse event where under three percent of records are "
            "positive, reported to stakeholders as 97.4 percent accuracy."
        ),
        X=X,
        y=y,
        reported_metrics={"accuracy": 0.974},
        reported_evaluation=(
            "A single stratified holdout, reporting accuracy as the headline metric."
        ),
        protocol="holdout",
        notes=[
            "Accuracy is the only metric reported and the positive class is the one "
            "that matters operationally.",
        ],
    )


def _importance_method_project() -> Project:
    """A project that explained its model with impurity importance alone.

    A random forest is fit on a mixture of informative columns, six uninformative
    gaussian columns, and a high cardinality identifier drawn from four hundred
    values. Impurity importance splits credit across any column that can carve up
    the training sample, so the identifier collects weight it has not earned.
    Permutation importance on held out rows does not.
    """
    rng = np.random.default_rng(13)
    n = 900
    informative = rng.normal(0, 1, size=(n, 2))
    filler = rng.normal(0, 1, size=(n, 4))
    y = (informative[:, 0] + 0.6 * informative[:, 1] + rng.normal(0, 1.0, n) > 0).astype(int)
    identifier = rng.integers(0, 400, size=n).astype(float)
    binary_noise = rng.integers(0, 2, size=n).astype(float)
    names = [f"gauss_{i}" for i in range(6)] + ["account_id_400", "binary_noise"]
    X = np.column_stack([informative, filler, identifier, binary_noise])
    return Project(
        project_id="importance_method",
        title="Retention model explained by impurity importance",
        summary=(
            "Rank the drivers of retention with a random forest's built in feature "
            "importance, on a table that includes a four hundred level account id."
        ),
        X=X,
        y=y,
        reported_metrics={"roc_auc": 0.823},
        reported_evaluation=(
            "A single holdout, with the driver ranking taken from the forest's "
            "impurity based feature importances."
        ),
        protocol="holdout",
        estimator="forest",
        feature_names=names,
        ranking_method="impurity",
        published_drivers=["gauss_0", "gauss_1", "gauss_4"],
        notes=[
            "A four hundred level identifier carries no information about the label "
            "but offers many ways to split the training rows.",
        ],
    )


BUILDERS = {
    "clean_baseline": _clean_baseline_project,
    "leaky_feature": _leaky_feature_project,
    "temporal_drift": _temporal_drift_project,
    "selection_outside_cv": _selection_outside_cv_project,
    "duplicate_rows": _duplicate_rows_project,
    "imbalanced_metric": _imbalanced_metric_project,
    "importance_method": _importance_method_project,
}

PROJECT_IDS = tuple(BUILDERS)


def load_project(project_id: str) -> Project:
    """Build one fixture project by id."""
    if project_id not in BUILDERS:
        raise KeyError(
            f"Unknown project {project_id!r}. Expected one of {list(PROJECT_IDS)}."
        )
    return BUILDERS[project_id]()


def load_projects() -> list[Project]:
    """Build every fixture project, in a stable order."""
    return [BUILDERS[project_id]() for project_id in PROJECT_IDS]
