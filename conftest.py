"""Pytest configuration.

Placing this file at the repository root puts the repository root on ``sys.path``
so that ``model_review`` imports without an install step, and it gives the tests a
single shared set of fixture projects and reports built once per session.
"""

from __future__ import annotations

import pytest

from model_review.datasets import load_projects
from model_review.report import review, review_all


@pytest.fixture(scope="session")
def projects():
    """Every fixture project, built once per session."""
    return load_projects()


@pytest.fixture(scope="session")
def projects_by_id(projects):
    """The fixture projects keyed by id."""
    return {project.project_id: project for project in projects}


@pytest.fixture(scope="session")
def reports(projects):
    """A review of every fixture project, run once per session."""
    return review_all()


@pytest.fixture(scope="session")
def reports_by_id(reports):
    """The reviews keyed by project id."""
    return {report.project_id: report for report in reports}


@pytest.fixture(scope="session")
def clean_report(projects_by_id):
    """The review of the control project, which is expected to raise nothing."""
    return review(projects_by_id["clean_baseline"])
