"""The reports a person reads, and the CLI that writes them.

The markdown and JSON reports are two views of the same review, so these tests
check that they agree: a finding that appears in the JSON must appear in the
markdown with the same numbers, and a clean project must say so rather than
producing an empty section. The CLI is exercised end to end into a temporary
directory, so the documented command in the README is the command under test.
"""

from __future__ import annotations

import json

import pytest

from model_review.__main__ import DEFAULT_OUT, main
from model_review.checks import CHECKS
from model_review.datasets import PROJECT_IDS, load_project, load_projects
from model_review.report import render_json, render_markdown, review, review_all


class TestReview:
    def test_every_project_is_reviewed_once_in_a_stable_order(self, reports):
        assert [report.project_id for report in reports] == list(PROJECT_IDS)

    def test_every_check_is_recorded_as_having_run(self, clean_report):
        assert clean_report.checks_run == [check.name for check in CHECKS]

    def test_a_review_carries_the_project_notes(self, clean_report):
        project = load_project("clean_baseline")
        assert clean_report.notes == list(project.notes)

    def test_a_review_can_be_restricted_to_a_subset_of_checks(self, projects_by_id):
        subset = (CHECKS[0],)
        report = review(projects_by_id["leaky_feature"], checks=subset)
        assert report.checks_run == ["reproduce_reported_score"]
        assert report.findings == []

    def test_a_subset_review_says_which_project_it_is(self, projects_by_id):
        report = review(projects_by_id["leaky_feature"])
        assert report.project_id == "leaky_feature"
        assert report.title == projects_by_id["leaky_feature"].title

    def test_review_all_accepts_an_explicit_selection(self):
        reports = review_all(("leaky_feature", "duplicate_rows"))
        assert [report.project_id for report in reports] == [
            "leaky_feature",
            "duplicate_rows",
        ]


class TestSeverityAggregation:
    def test_the_control_review_passes(self, clean_report):
        assert clean_report.passed
        assert clean_report.weight == 0
        assert clean_report.by_severity == {"high": 0, "medium": 0, "low": 0}

    @pytest.mark.parametrize("project_id", [p for p in PROJECT_IDS if p != "clean_baseline"])
    def test_a_flawed_project_does_not_pass(self, reports_by_id, project_id):
        report = reports_by_id[project_id]
        assert not report.passed
        assert report.by_severity["high"] >= 1

    def test_each_flawed_project_raises_exactly_one_high_finding(self, reports_by_id):
        for project_id, report in reports_by_id.items():
            if project_id == "clean_baseline":
                continue
            assert report.by_severity["high"] == 1, project_id
            assert len(report.findings) == 1, project_id

    def test_the_weight_counts_three_per_high_finding(self, reports_by_id):
        for report in reports_by_id.values():
            assert report.weight == 3 * report.by_severity["high"]

    def test_findings_are_ordered_most_severe_first(self):
        report = review_all(("leaky_feature",))[0]
        severities = [finding.weight for finding in report.sorted_findings]
        assert severities == sorted(severities, reverse=True)


class TestMarkdownReport:
    def test_the_report_opens_with_a_summary_table(self, reports, projects):
        text = render_markdown(reports, projects)
        assert text.startswith("# Model methodology review")
        assert "| Project | Rows | Findings | High | Weight |" in text

    def test_every_project_appears_as_its_own_section(self, reports, projects):
        text = render_markdown(reports, projects)
        for project in projects:
            assert f"## {project.project_id}: {project.title}" in text

    def test_a_clean_project_states_that_it_is_clean(self, reports, projects):
        text = render_markdown(reports, projects)
        assert "No findings." in text

    def test_the_findings_are_rendered_with_their_numbers(self, reports, projects):
        text = render_markdown(reports, projects)
        assert "account_closed_date" in text
        assert "0.2648" in text or "0.265" in text
        assert "### leakage (high)" in text

    def test_the_checks_are_listed_at_the_end(self, reports, projects):
        text = render_markdown(reports, projects)
        assert "## Checks" in text
        for check in CHECKS:
            assert check.name in text

    def test_the_totals_line_agrees_with_the_projects(self, reports, projects):
        text = render_markdown(reports, projects)
        weight = sum(report.weight for report in reports)
        highs = sum(report.by_severity["high"] for report in reports)
        assert f"{len(reports)} projects reviewed" in text
        assert f"{weight} total severity weight" in text
        assert f"{highs} high severity findings." in text

    def test_the_report_has_no_em_or_en_dashes(self, reports, projects):
        text = render_markdown(reports, projects)
        assert "\u2014" not in text
        assert "\u2013" not in text


class TestJsonReport:
    def test_the_payload_holds_every_project_with_its_review(self, reports, projects):
        payload = json.loads(render_json(reports, projects))
        assert len(payload["projects"]) == len(projects)
        for entry in payload["projects"]:
            assert "project" in entry
            assert "review" in entry
            assert entry["review"]["project_id"] == entry["project"]["project_id"]

    def test_the_totals_agree_with_the_reviews(self, reports, projects):
        payload = json.loads(render_json(reports, projects))
        assert payload["totals"]["projects"] == len(reports)
        assert payload["totals"]["findings"] == sum(
            len(report.findings) for report in reports
        )
        assert payload["totals"]["weight"] == sum(report.weight for report in reports)

    def test_the_evidence_travels_with_the_finding(self, reports, projects):
        payload = json.loads(render_json(reports, projects))
        entries = {
            entry["review"]["project_id"]: entry for entry in payload["projects"]
        }
        finding = entries["leaky_feature"]["review"]["findings"][0]
        assert finding["check"] == "leakage"
        assert finding["evidence"]["suspected_features"] == ["account_closed_date"]
        assert finding["evidence"]["gap"] == pytest.approx(0.2648, abs=0.001)

    def test_the_project_settings_are_recorded_alongside(self, reports, projects):
        payload = json.loads(render_json(reports, projects))
        entries = {
            entry["project"]["project_id"]: entry["project"]
            for entry in payload["projects"]
        }
        assert entries["temporal_drift"]["split"] == "temporal"
        assert entries["importance_method"]["ranking_method"] == "impurity"
        assert entries["selection_outside_cv"]["n_selected"] == 15

    def test_the_json_is_valid_for_an_empty_review_list(self, projects):
        payload = json.loads(render_json([], projects))
        assert payload["totals"]["projects"] == 0
        assert payload["projects"] == []


class TestCommandLine:
    def test_list_projects_names_every_fixture(self, capsys):
        assert main(["list-projects"]) == 0
        out = capsys.readouterr().out
        for project_id in PROJECT_IDS:
            assert project_id in out
        assert out.startswith(f"{len(PROJECT_IDS)} projects")

    def test_list_checks_names_every_check(self, capsys):
        assert main(["list-checks"]) == 0
        out = capsys.readouterr().out
        for check in CHECKS:
            assert check.name in out

    def test_run_writes_both_reports_to_the_requested_directory(self, tmp_path, capsys):
        assert main(["run", "--out", str(tmp_path)]) == 0
        markdown = (tmp_path / "review.md").read_text(encoding="utf-8")
        payload = json.loads((tmp_path / "reviews.json").read_text(encoding="utf-8"))
        assert "Model methodology review" in markdown
        assert payload["totals"]["projects"] == len(PROJECT_IDS)

    def test_run_reports_the_findings_it_wrote(self, tmp_path, capsys):
        main(["run", "--out", str(tmp_path)])
        out = capsys.readouterr().out
        assert f"Projects reviewed: {len(PROJECT_IDS)}" in out
        assert f"Checks run per project: {len(CHECKS)}" in out
        assert "leaky_feature" in out
        assert "leakage" in out

    def test_run_accepts_a_subset_of_projects(self, tmp_path):
        assert main(["run", "--projects", "leaky_feature", "--out", str(tmp_path)]) == 0
        payload = json.loads((tmp_path / "reviews.json").read_text(encoding="utf-8"))
        assert [entry["review"]["project_id"] for entry in payload["projects"]] == [
            "leaky_feature"
        ]

    def test_run_rejects_an_unknown_project_without_writing(self, tmp_path, capsys):
        assert main(["run", "--projects", "nope", "--out", str(tmp_path)]) == 2
        assert "Unknown projects" in capsys.readouterr().err
        assert not (tmp_path / "review.md").exists()

    def test_the_default_output_directory_is_documented(self):
        assert DEFAULT_OUT == "reports"

    def test_report_re_renders_markdown_from_saved_json(self, tmp_path, capsys):
        main(["run", "--out", str(tmp_path)])
        capsys.readouterr()
        saved = tmp_path / "reviews.json"
        target = tmp_path / "again.md"
        assert main(["report", "--reviews", str(saved), "--out", str(target)]) == 0
        text = target.read_text(encoding="utf-8")
        assert "account_closed_date" in text
        assert "leaky_feature" in text

    def test_report_prints_to_stdout_without_an_output_path(self, tmp_path, capsys):
        main(["run", "--out", str(tmp_path)])
        capsys.readouterr()
        assert main(["report", "--reviews", str(tmp_path / "reviews.json")]) == 0
        assert "## Summary" in capsys.readouterr().out

    def test_report_rejects_a_missing_file(self, tmp_path, capsys):
        assert main(["report", "--reviews", str(tmp_path / "nope.json")]) == 2
        assert "not found" in capsys.readouterr().err
