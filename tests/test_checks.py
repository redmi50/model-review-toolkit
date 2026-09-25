"""The checks, one class each, plus the applicability rules and the bands.

Every assertion here is on a measured quantity, not on prose. The control project
is tested first and most insistently: a review suite that raises a finding for
everything it looks at is useless, so the tests pin down that a carefully
evaluated project comes back clean, and that each flawed project raises exactly
the one check that names its flaw.
"""

from __future__ import annotations

import numpy as np
import pytest

from model_review import checks as checks_module
from model_review.checks import (
    LEAK_CORRELATION,
    PUBLICATION_TOLERANCE,
    baseline_lift,
    class_imbalance_metric,
    importance_method,
    leakage,
    reproduce_reported_score,
    row_replication,
    score_with_protocol,
    selection_outside_cv,
    temporal_evaluation,
)
from model_review.datasets import Project
from model_review.findings import Finding, severity_for_gap, severity_for_share


def synthetic(n: int = 200, seed: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """A small balanced problem, large enough for the five fold protocol."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, size=(n, 2))
    y = (1.2 * X[:, 0] + rng.normal(0, 1.2, n) > 0).astype(int)
    return X, y


X_DEFAULT, Y_DEFAULT = synthetic()


def make_project(**overrides) -> Project:
    """A minimal valid project, with individual fields overridden per test."""
    fields: dict[str, object] = {
        "project_id": "broken",
        "title": "t",
        "summary": "s",
        "X": X_DEFAULT,
        "y": Y_DEFAULT,
        "reported_metrics": {"roc_auc": 0.5},
        "reported_evaluation": "e",
    }
    fields.update(overrides)
    return Project(**fields)


def project_reporting_an_offset(offset: float) -> Project:
    """A project whose published roc_auc is the honest one plus an offset.

    The offset is applied to the score the protocol actually produces, so the
    reproduction check is exercised against a known distance rather than a
    literal that happens to sit some distance away.
    """
    base = make_project()
    honest = score_with_protocol(base, base.X)["roc_auc"]
    return make_project(reported_metrics={"roc_auc": honest + offset})


def only(findings) -> Finding:
    """Assert exactly one finding and return it."""
    assert len(findings) == 1, [f.check for f in findings]
    return findings[0]


class TestSeverityBands:
    """The bands decide every severity in the report, so they are tested directly."""

    @pytest.mark.parametrize(
        "gap,expected",
        [
            (-0.5, None),
            (0.0, None),
            (0.001, "low"),
            (0.029, "low"),
            (0.03, "medium"),
            (0.099, "medium"),
            (0.10, "high"),
            (0.9, "high"),
        ],
    )
    def test_gap_bands(self, gap, expected):
        assert severity_for_gap(gap) == expected

    def test_a_share_of_zero_is_not_a_finding(self):
        assert severity_for_share(0.0, high=0.5, medium=0.1) is None

    def test_share_bands_are_the_ones_passed_in(self):
        assert severity_for_share(0.05, high=0.5, medium=0.1) == "low"
        assert severity_for_share(0.2, high=0.5, medium=0.1) == "medium"
        assert severity_for_share(0.6, high=0.5, medium=0.1) == "high"

    def test_an_unknown_severity_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown severity"):
            Finding(check="c", severity="critical", summary="s", recommendation="r")


class TestControlProject:
    """The control is why the other findings mean anything."""

    def test_the_control_project_raises_no_findings(self, clean_report):
        assert clean_report.findings == []
        assert clean_report.weight == 0
        assert clean_report.passed

    def test_the_control_project_reproduces_its_own_score(self, projects_by_id):
        assert reproduce_reported_score(projects_by_id["clean_baseline"]) == []

    def test_the_control_project_has_no_leaky_column(self, projects_by_id):
        assert leakage(projects_by_id["clean_baseline"]) == []

    def test_the_control_project_does_not_trigger_the_structural_checks(
        self, projects_by_id
    ):
        project = projects_by_id["clean_baseline"]
        assert row_replication(project) == []
        assert importance_method(project) == []

    def test_the_control_is_the_only_project_that_reaches_zero_weight(
        self, reports_by_id, clean_report
    ):
        assert clean_report.project_id == "clean_baseline"
        for project_id, report in reports_by_id.items():
            if project_id != "clean_baseline":
                assert report.weight > 0, project_id


class TestReproduction:
    def test_a_reported_score_that_matches_the_protocol_is_not_a_finding(
        self, projects_by_id
    ):
        for project in projects_by_id.values():
            assert reproduce_reported_score(project) == [], project.project_id

    def test_a_reported_score_that_cannot_be_reproduced_is_flagged(self):
        finding = only(reproduce_reported_score(project_reporting_an_offset(0.20)))
        assert finding.check == "reproduce_reported_score"
        assert finding.severity == "high"
        assert finding.evidence["difference"] == pytest.approx(0.20, abs=0.001)

    def test_a_difference_within_rounding_is_not_a_finding(self):
        """A three decimal publication and a four decimal recomputation agree."""
        assert reproduce_reported_score(project_reporting_an_offset(0.0004)) == []
        assert PUBLICATION_TOLERANCE < 0.001

    def test_a_small_but_real_difference_is_a_medium_finding(self):
        finding = only(reproduce_reported_score(project_reporting_an_offset(0.005)))
        assert finding.severity == "medium"
        assert finding.evidence["difference"] == pytest.approx(0.005, abs=0.001)

    def test_the_finding_states_the_protocol_and_the_estimator(self):
        project = project_reporting_an_offset(0.20)
        finding = only(reproduce_reported_score(project))
        assert finding.evidence["protocol"] == project.protocol
        assert finding.evidence["estimator"] == project.estimator

    def test_an_unknown_metric_is_ignored_rather_than_crashing(self):
        assert reproduce_reported_score(make_project(reported_metrics={"f1": 0.9})) == []


class TestLeakage:
    def test_the_closure_date_is_flagged_with_the_score_it_inflates(
        self, projects_by_id
    ):
        finding = only(leakage(projects_by_id["leaky_feature"]))
        assert finding.check == "leakage"
        assert finding.severity == "high"
        assert finding.evidence["suspected_features"] == ["account_closed_date"]
        assert finding.evidence["reported"] == 1.0
        assert finding.evidence["gap"] > 0.20

    def test_the_reported_evidence_names_the_column_and_the_drop(self, projects_by_id):
        finding = only(leakage(projects_by_id["leaky_feature"]))
        assert "account_closed_date" in finding.summary
        assert "falls to" in finding.summary

    def test_the_correlation_clears_the_stated_threshold(self, projects_by_id):
        finding = only(leakage(projects_by_id["leaky_feature"]))
        correlation = finding.evidence["correlations"]["account_closed_date"]
        assert correlation >= LEAK_CORRELATION

    def test_a_project_with_no_copy_of_the_label_is_not_flagged(self, projects_by_id):
        assert leakage(projects_by_id["temporal_drift"]) == []
        assert leakage(projects_by_id["imbalanced_metric"]) == []


class TestTemporalEvaluation:
    def test_the_shuffled_split_is_compared_with_forward_chaining(self, projects_by_id):
        finding = only(temporal_evaluation(projects_by_id["temporal_drift"]))
        assert finding.check == "temporal_evaluation"
        assert finding.severity == "high"
        evidence = finding.evidence
        assert evidence["reported"] == 0.936
        assert evidence["forward_chaining"] < 0.70
        assert evidence["gap"] > 0.30
        assert len(evidence["fold_scores"]) == 5

    def test_the_shuffled_number_agrees_with_the_reported_one(self, projects_by_id):
        finding = only(temporal_evaluation(projects_by_id["temporal_drift"]))
        assert finding.evidence["shuffled_cv"] == pytest.approx(0.936, abs=0.01)

    def test_a_project_that_did_not_shuffle_time_is_not_checked(self, projects_by_id):
        for project_id in ("clean_baseline", "leaky_feature", "duplicate_rows"):
            assert temporal_evaluation(projects_by_id[project_id]) == []


class TestSelectionOutsideCv:
    def test_the_global_screen_is_redone_inside_each_fold(self, projects_by_id):
        finding = only(selection_outside_cv(projects_by_id["selection_outside_cv"]))
        assert finding.check == "selection_outside_cv"
        assert finding.severity == "high"
        assert finding.evidence["candidate_features"] == 3000
        assert finding.evidence["selected_features"] == 15
        assert finding.evidence["gap"] > 0.30

    def test_the_in_fold_score_is_what_a_rerun_produces(self, projects_by_id):
        finding = only(selection_outside_cv(projects_by_id["selection_outside_cv"]))
        folds = np.asarray(finding.evidence["fold_scores"])
        assert folds.shape == (5,)
        assert finding.evidence["in_fold_selection"] == pytest.approx(
            folds.mean(), abs=0.0001
        )

    def test_a_project_that_selected_nothing_is_not_checked(self, projects_by_id):
        assert selection_outside_cv(projects_by_id["clean_baseline"]) == []


class TestRowReplication:
    def test_the_shared_test_rows_are_reported_as_a_share(self, projects_by_id):
        finding = only(row_replication(projects_by_id["duplicate_rows"]))
        assert finding.check == "row_replication"
        assert finding.severity == "high"
        assert finding.evidence["overlap_share"] > 0.95
        assert (
            finding.evidence["overlapping_rows"] <= finding.evidence["test_rows"]
        )

    def test_the_share_equals_the_counts_it_reports(self, projects_by_id):
        evidence = only(row_replication(projects_by_id["duplicate_rows"])).evidence
        assert evidence["overlap_share"] == pytest.approx(
            evidence["overlapping_rows"] / evidence["test_rows"], abs=0.0001
        )

    def test_a_project_with_distinct_rows_is_not_flagged(self, projects_by_id):
        assert row_replication(projects_by_id["clean_baseline"]) == []

    def test_the_check_reads_the_protocol_to_choose_its_folds(self, projects_by_id):
        finding = only(row_replication(projects_by_id["duplicate_rows"]))
        assert finding.evidence["protocol"] == "holdout"
        assert row_replication(projects_by_id["imbalanced_metric"]) == []


class TestClassImbalanceMetric:
    def test_the_base_rate_the_constant_and_the_missed_share_are_reported(
        self, projects_by_id
    ):
        finding = only(class_imbalance_metric(projects_by_id["imbalanced_metric"]))
        assert finding.check == "class_imbalance_metric"
        assert finding.severity == "high"
        evidence = finding.evidence
        assert evidence["base_rate"] < 0.05
        assert evidence["constant_predictor_accuracy"] > 0.95
        assert evidence["missed_positive_share"] > 0.50

    def test_the_headline_does_not_beat_a_constant_prediction(self, projects_by_id):
        evidence = only(
            class_imbalance_metric(projects_by_id["imbalanced_metric"])
        ).evidence
        assert (
            evidence["model_accuracy"] - evidence["constant_predictor_accuracy"] < 0.01
        )

    def test_a_project_that_reports_something_other_than_accuracy_is_not_checked(
        self, projects_by_id
    ):
        assert class_imbalance_metric(projects_by_id["duplicate_rows"]) == []

    def test_a_common_positive_class_is_not_checked(self, projects_by_id):
        assert class_imbalance_metric(projects_by_id["clean_baseline"]) == []


class TestImportanceMethod:
    def test_the_weight_on_columns_with_no_effect_is_measured(self, projects_by_id):
        finding = only(importance_method(projects_by_id["importance_method"]))
        assert finding.check == "importance_method"
        assert finding.severity == "high"
        evidence = finding.evidence
        assert evidence["unearned_share"] > 0.25
        assert "account_id_400" in evidence["unearned_features"]

    def test_the_identifier_collects_impurity_weight_without_earning_it(
        self, projects_by_id
    ):
        evidence = only(
            importance_method(projects_by_id["importance_method"])
        ).evidence
        assert evidence["impurity_importance"]["account_id_400"] > 0.0
        assert evidence["permutation_importance"]["account_id_400"] <= 0.0

    def test_a_published_driver_without_a_measured_effect_is_named(
        self, projects_by_id
    ):
        finding = only(importance_method(projects_by_id["importance_method"]))
        assert finding.evidence["published_drivers_without_measured_effect"]
        assert "permutation importance is at or below zero" in finding.summary

    def test_a_project_that_did_not_rank_by_impurity_is_not_checked(
        self, projects_by_id
    ):
        assert importance_method(projects_by_id["clean_baseline"]) == []


class TestBaselineLift:
    def test_a_score_on_the_no_skill_line_is_a_finding(self):
        finding = only(baseline_lift(make_project(reported_metrics={"roc_auc": 0.501})))
        assert finding.check == "baseline_lift"
        assert finding.severity == "high"
        assert finding.evidence["lift"] < 0.01

    def test_a_score_just_under_the_minimum_lift_is_a_medium_finding(self):
        finding = only(baseline_lift(make_project(reported_metrics={"roc_auc": 0.53})))
        assert finding.severity == "medium"

    def test_a_score_that_clears_the_minimum_lift_is_not_a_finding(self):
        assert baseline_lift(make_project(reported_metrics={"roc_auc": 0.70})) == []

    def test_a_project_that_reports_no_ranking_score_is_not_checked(self):
        assert baseline_lift(make_project(reported_metrics={"accuracy": 0.9})) == []

    def test_the_verified_projects_all_clear_the_no_skill_line(self, projects_by_id):
        for project in projects_by_id.values():
            assert baseline_lift(project) == [], project.project_id


class TestCheckRegistry:
    def test_the_registry_names_match_the_functions(self):
        names = [check.name for check in checks_module.CHECKS]
        assert names == [
            "reproduce_reported_score",
            "leakage",
            "temporal_evaluation",
            "selection_outside_cv",
            "row_replication",
            "class_imbalance_metric",
            "importance_method",
            "baseline_lift",
        ]

    def test_every_registered_check_is_callable_on_every_project(self, projects_by_id):
        for project in projects_by_id.values():
            for check in checks_module.CHECKS:
                assert isinstance(check.run(project), list)

    def test_a_check_that_does_not_apply_returns_nothing_rather_than_raising(self):
        project = make_project(
            selection="global",
            n_selected=2,
            candidates=np.zeros_like(X_DEFAULT),
            feature_names=["a", "b"],
            ranking_method="impurity",
            published_drivers=["a"],
        )
        assert temporal_evaluation(project) == []
