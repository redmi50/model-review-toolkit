"""A toolkit for reviewing how a predictive model was evaluated.

The review is a measurement, not an opinion. Each check recomputes the project's
reported score under the protocol the project described, or under the protocol it
should have used, and reports the difference. Severity is a function of that
difference, so a project that was evaluated carefully produces no findings.

Everything runs offline against fixed seed fixtures, so the numbers in the report
are the same on every machine.
"""

from .checks import CHECKS, Check
from .datasets import PROJECT_IDS, Project, load_project, load_projects
from .findings import Finding, ReviewReport, severity_for_gap
from .report import render_json, render_markdown, review, review_all

__all__ = [
    "CHECKS",
    "PROJECT_IDS",
    "Check",
    "Finding",
    "Project",
    "ReviewReport",
    "load_project",
    "load_projects",
    "render_json",
    "render_markdown",
    "review",
    "review_all",
    "severity_for_gap",
]
