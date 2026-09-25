"""The fixture projects: their shape, their stored numbers and their controls.

The stored reported metrics are the crux of this repository. A report compares a
published figure against the toolkit's own recomputation, so if a fixture drifts
the report becomes quietly wrong. These tests recompute every stored metric from
the fixture under the same protocol the checks use, and fail if the two disagree.
"""

from __future__ import annotations

import numpy as np
import pytest

from model_review.checks import score_with_protocol
from model_review.datasets import PROJECT_IDS, Project, load_project, load_projects


def make_project(**overrides) -> Project:
    """A minimal valid project, with individual fields overridden per test."""
    fields: dict[str, object] = {
        "project_id": "broken",
        "title": "t",
        "summary": "s",
        "X": np.zeros((4, 2)),
        "y": np.zeros(4, dtype=int),
        "reported_metrics": {"roc_auc": 0.7},
        "reported_evaluation": "e",
    }
    fields.update(overrides)
    return Project(**fields)


class TestProjectValidation:
    def test_every_contractual_project_can_be_built(self, projects):
        assert [project.project_id for project in projects] == list(PROJECT_IDS)

    def test_a_mismatch_between_rows_and_labels_is_rejected(self):
        with pytest.raises(ValueError, match="labels"):
            make_project(X=np.zeros((4, 2)), y=np.zeros(3, dtype=int))

    def test_a_mismatch_between_columns_and_names_is_rejected(self):
        with pytest.raises(ValueError, match="names"):
            make_project(feature_names=["only_one"])

    def test_an_unknown_protocol_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown protocol"):
            make_project(protocol="bootstrap")

    def test_an_unknown_estimator_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown estimator"):
            make_project(estimator="gradient_boosting")

    def test_a_temporal_split_without_a_time_index_is_rejected(self):
        with pytest.raises(ValueError, match="time_index"):
            make_project(split="temporal")

    def test_global_selection_without_candidates_is_rejected(self):
        with pytest.raises(ValueError, match="candidate matrix"):
            make_project(selection="global", n_selected=2)

    def test_a_published_driver_must_be_a_feature(self):
        with pytest.raises(ValueError, match="not.*features"):
            make_project(
                feature_names=["a", "b"],
                ranking_method="impurity",
                published_drivers=["c"],
            )

    def test_an_unknown_ranking_method_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown ranking method"):
            make_project(ranking_method="shap", published_drivers=[])

    def test_an_unknown_project_id_is_rejected(self):
        with pytest.raises(KeyError, match="Unknown project"):
            load_project("does_not_exist")


class TestFixtureShape:
    @pytest.mark.parametrize(
        "project_id,rows,features",
        [
            ("clean_baseline", 1500, 5),
            ("leaky_feature", 2000, 7),
            ("temporal_drift", 1200, 4),
            ("selection_outside_cv", 90, 15),
            ("duplicate_rows", 2120, 30),
            ("imbalanced_metric", 3000, 5),
            ("importance_method", 900, 8),
        ],
    )
    def test_documented_shape(self, projects_by_id, project_id, rows, features):
        project = projects_by_id[project_id]
        assert project.n_rows == rows
        assert project.X.shape[1] == features

    def test_every_fixture_is_deterministic(self):
        for project_id in PROJECT_IDS:
            first = load_project(project_id)
            second = load_project(project_id)
            assert np.array_equal(first.X, second.X)
            assert np.array_equal(first.y, second.y)

    def test_every_project_states_a_binary_problem(self):
        for project in load_projects():
            assert set(np.unique(project.y)) <= {0, 1}

    def test_the_control_project_is_balanced_and_uncontaminated(self, projects_by_id):
        project = projects_by_id["clean_baseline"]
        assert 0.4 < project.positive_rate < 0.6
        assert project.selection == "none"
        assert project.split == "random"
        assert project.ranking_method is None


class TestStoredMetricsAreReproducible:
    """Each published figure must equal what the fixture's own evaluation produces.

    This is the guarantee the reports rest on. A check compares a published number
    against a recomputation; these tests make the same comparison and fail when a
    fixture or its stored literal drifts, so a review can never quietly become
    wrong.
    """

    @pytest.mark.parametrize("project_id", PROJECT_IDS)
    def test_published_metrics_match_the_stated_protocol(self, projects_by_id, project_id):
        project = projects_by_id[project_id]
        recomputed = score_with_protocol(project, project.X)
        for metric, published in project.reported_metrics.items():
            assert metric in recomputed, f"{project_id} publishes an unknown {metric}"
            assert recomputed[metric] == pytest.approx(published, abs=0.0005), (
                f"{project_id} publishes {metric} {published} but its stated protocol "
                f"gives {recomputed[metric]:.4f}"
            )

    def test_the_leaky_column_is_a_copy_of_the_label(self, projects_by_id):
        project = projects_by_id["leaky_feature"]
        correlations = {
            name: abs(float(np.corrcoef(project.X[:, j], project.y)[0, 1]))
            for j, name in enumerate(project.feature_names)
        }
        assert correlations["account_closed_date"] > 0.99
        for name, value in correlations.items():
            if name != "account_closed_date":
                assert value < 0.7

    def test_the_temporal_project_drifts_with_time(self, projects_by_id):
        project = projects_by_id["temporal_drift"]
        order = np.argsort(project.time_index)
        first_half = project.y[order][: project.n_rows // 2].mean()
        second_half = project.y[order][project.n_rows // 2 :].mean()
        assert abs(second_half - first_half) > 0.2

    def test_the_selection_project_stores_its_full_candidate_matrix(
        self, projects_by_id
    ):
        project = projects_by_id["selection_outside_cv"]
        assert project.candidates is not None
        assert project.candidates.shape[1] == 3000
        assert project.X.shape[1] == project.n_selected

    def test_the_replicated_project_actually_repeats_rows(self, projects_by_id):
        project = projects_by_id["duplicate_rows"]
        unique = np.unique(project.X, axis=0)
        assert len(unique) == project.n_rows // 5

    def test_the_rare_event_project_really_is_rare(self, projects_by_id):
        project = projects_by_id["imbalanced_metric"]
        assert project.positive_rate < 0.05
        assert "accuracy" in project.reported_metrics

    def test_the_importance_project_has_an_uninformative_identifier(
        self, projects_by_id
    ):
        project = projects_by_id["importance_method"]
        assert "account_id_400" in project.feature_names
        j = project.feature_names.index("account_id_400")
        coefficient = abs(float(np.corrcoef(project.X[:, j], project.y)[0, 1]))
        assert coefficient < 0.1
