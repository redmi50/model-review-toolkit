"""Turning findings into a review a person can read.

The markdown report is ordered so the most expensive thing to be wrong about comes
first: whether the published number reproduces, then what the checks found, then
the checks that found nothing. The JSON report carries the same content with the
full evidence dictionaries, so a script downstream does not have to parse prose.
"""

from __future__ import annotations

import json

from .checks import CHECKS, Check
from .datasets import Project, load_project, load_projects
from .findings import Finding, ReviewReport


def review(project: Project, checks: tuple[Check, ...] = CHECKS) -> ReviewReport:
    """Run every check against one project and collect the findings."""
    findings: list[Finding] = []
    for check in checks:
        findings.extend(check.run(project))

    return ReviewReport(
        project_id=project.project_id,
        title=project.title,
        findings=findings,
        checks_run=[check.name for check in checks],
        notes=list(project.notes),
    )


def review_all(
    project_ids: tuple[str, ...] | None = None,
    checks: tuple[Check, ...] = CHECKS,
) -> list[ReviewReport]:
    """Review every fixture project, or a selected subset, in a stable order."""
    projects = (
        load_projects() if project_ids is None else [load_project(p) for p in project_ids]
    )
    return [review(project, checks) for project in projects]


def render_markdown(reports: list[ReviewReport], projects: list[Project]) -> str:
    """Render the reviews as one markdown document."""
    lines: list[str] = ["# Model methodology review", ""]
    lines.append(
        "Each project below is reviewed by recomputing what it reported rather than "
        "by reading what it said. Severity comes from the size of the measured "
        "problem: a high finding means the reported score or the published ranking "
        "changes materially when the evaluation is done correctly."
    )
    lines.append("")

    total = sum(report.weight for report in reports)
    highs = sum(report.by_severity["high"] for report in reports)
    lines.append("## Summary")
    lines.append("")
    lines.append("| Project | Rows | Findings | High | Weight |")
    lines.append("| --- | --- | --- | --- | --- |")
    for project, report in zip(projects, reports):
        lines.append(
            f"| {project.project_id} | {project.n_rows} | {len(report.findings)} "
            f"| {report.by_severity['high']} | {report.weight} |"
        )
    lines.append("")
    lines.append(
        f"{len(reports)} projects reviewed, {total} total severity weight, {highs} "
        f"high severity findings."
    )
    lines.append("")

    for project, report in zip(projects, reports):
        lines.append(f"## {project.project_id}: {project.title}")
        lines.append("")
        lines.append(project.summary)
        lines.append("")
        lines.append(f"- Reported: {_metrics(project.reported_metrics)}")
        lines.append(f"- Reported evaluation: {project.reported_evaluation}")
        for note in report.notes:
            lines.append(f"- {note}")
        lines.append("")

        if not report.findings:
            lines.append(
                "No findings. The reported score reproduces and no check raised an "
                "issue."
            )
            lines.append("")
            continue

        lines.append("| Check | Severity | Finding |")
        lines.append("| --- | --- | --- |")
        for finding in report.sorted_findings:
            lines.append(
                f"| {finding.check} | {finding.severity} | "
                f"{finding.summary} |"
            )
        lines.append("")
        for finding in report.sorted_findings:
            lines.append(f"### {finding.check} ({finding.severity})")
            lines.append("")
            lines.append(finding.summary)
            lines.append("")
            lines.append(f"Recommendation: {finding.recommendation}")
            lines.append("")
            lines.append("| Quantity | Value |")
            lines.append("| --- | --- |")
            for key, value in finding.evidence.items():
                lines.append(f"| {key} | {_value(value)} |")
            lines.append("")

    lines.append("## Checks")
    lines.append("")
    for check in CHECKS:
        lines.append(f"- {check.name}: {check.description}")
    lines.append("")
    return "\n".join(lines)


def _metrics(metrics: dict[str, float]) -> str:
    """Render a metrics dictionary as one readable clause."""
    return ", ".join(f"{name} {value:.3f}" for name, value in sorted(metrics.items()))


def _value(value: object) -> str:
    """Render one evidence value without hiding its structure."""
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, dict):
        return ", ".join(f"{k} {v}" for k, v in value.items())
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return str(value)


def render_json(reports: list[ReviewReport], projects: list[Project]) -> str:
    """Render the reviews as JSON, with the projects' own settings alongside."""
    payload = {
        "projects": [
            {"project": project.as_dict(), "review": report.as_dict()}
            for project, report in zip(projects, reports)
        ],
        "totals": {
            "projects": len(reports),
            "findings": sum(len(report.findings) for report in reports),
            "weight": sum(report.weight for report in reports),
            "high": sum(report.by_severity["high"] for report in reports),
        },
    }
    return json.dumps(payload, indent=2, sort_keys=False)
