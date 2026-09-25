"""Command line interface.

    python -m model_review list-projects
    python -m model_review list-checks
    python -m model_review run
    python -m model_review run --projects leaky_feature,selection_outside_cv
    python -m model_review report --reviews reports/reviews.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .checks import CHECKS
from .datasets import PROJECT_IDS, load_project, load_projects
from .report import render_json, render_markdown, review, review_all

DEFAULT_OUT = "reports"


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser."""
    parser = argparse.ArgumentParser(
        prog="model_review",
        description=(
            "Review how a predictive model was evaluated by recomputing what it "
            "reported."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list-projects", help="Print every fixture project.")
    subparsers.add_parser("list-checks", help="Print every check.")

    run_parser = subparsers.add_parser("run", help="Review projects and write reports.")
    run_parser.add_argument(
        "--projects",
        default=None,
        help=(
            "Comma separated project ids. Defaults to every fixture. "
            f"Available: {', '.join(PROJECT_IDS)}."
        ),
    )
    run_parser.add_argument("--out", default=DEFAULT_OUT, help="Output directory.")

    report_parser = subparsers.add_parser(
        "report", help="Re-render a saved review from its JSON."
    )
    report_parser.add_argument("--reviews", required=True, help="Path to reviews.json.")
    report_parser.add_argument("--out", default=None, help="Where to write the markdown.")

    return parser


def cmd_list_projects() -> int:
    """Print the fixture projects and the settings that decide which checks apply."""
    projects = load_projects()
    print(f"{len(projects)} projects\n")
    for project in projects:
        print(f"{project.project_id}: {project.title}")
        print(f"  {project.summary}")
        print(
            f"  rows {project.n_rows}, features {project.X.shape[1]}, "
            f"positive rate {project.positive_rate:.4f}"
        )
        print(
            f"  protocol {project.protocol}, estimator {project.estimator}, "
            f"split {project.split}, selection {project.selection}"
        )
        print(
            f"  reported {', '.join(f'{k} {v:.3f}' for k, v in sorted(project.reported_metrics.items()))}"
        )
        for note in project.notes:
            print(f"  note: {note}")
        print()
    return 0


def cmd_list_checks() -> int:
    """Print the checks and what each one measures."""
    print(f"{len(CHECKS)} checks\n")
    for check in CHECKS:
        print(f"{check.name}")
        print(f"  {check.description}")
    return 0


def cmd_run(args) -> int:
    """Review the selected projects and write the markdown and JSON reports."""
    project_ids = None
    if args.projects:
        project_ids = tuple(
            name.strip() for name in args.projects.split(",") if name.strip()
        )
        unknown = [name for name in project_ids if name not in PROJECT_IDS]
        if unknown:
            print(f"Unknown projects: {', '.join(unknown)}", file=sys.stderr)
            return 2

    reports = review_all(project_ids)
    projects = (
        load_projects() if project_ids is None else [load_project(p) for p in project_ids]
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = out_dir / "review.md"
    json_path = out_dir / "reviews.json"
    markdown_path.write_text(render_markdown(reports, projects), encoding="utf-8")
    json_path.write_text(render_json(reports, projects), encoding="utf-8")

    print(f"Projects reviewed: {len(reports)}")
    print(f"Checks run per project: {len(CHECKS)}")
    print()
    for project, report in zip(projects, reports):
        counts = report.by_severity
        print(
            f"{project.project_id:22s} findings {len(report.findings)}  "
            f"high {counts['high']}  medium {counts['medium']}  low {counts['low']}  "
            f"weight {report.weight}"
        )
        for finding in report.sorted_findings:
            print(f"    {finding.severity:6s} {finding.check}: {finding.summary}")
    print()
    print(f"Wrote {markdown_path}")
    print(f"Wrote {json_path}")
    return 0


def cmd_report(args) -> int:
    """Re-render the markdown from a saved reviews.json."""
    path = Path(args.reviews)
    if not path.is_file():
        print(f"Review file not found: {path}", file=sys.stderr)
        return 2

    payload = json.loads(path.read_text(encoding="utf-8"))
    rendered = _render_saved(payload)
    if args.out:
        Path(args.out).write_text(rendered, encoding="utf-8")
        print(f"Wrote {args.out}")
    else:
        print(rendered)
    return 0


def _render_saved(payload: dict) -> str:
    """Render a saved review without re-running any check."""
    totals = payload.get("totals", {})
    lines = ["# Model methodology review", ""]
    lines.append("## Summary")
    lines.append("")
    lines.append("| Project | Findings | High | Weight |")
    lines.append("| --- | --- | --- | --- |")
    for entry in payload.get("projects", []):
        review_payload = entry["review"]
        lines.append(
            f"| {review_payload['project_id']} | {len(review_payload['findings'])} "
            f"| {review_payload['by_severity']['high']} | {review_payload['weight']} |"
        )
    lines.append("")
    lines.append(
        f"{totals.get('projects', 0)} projects reviewed, "
        f"{totals.get('weight', 0)} total severity weight, "
        f"{totals.get('high', 0)} high severity findings."
    )
    lines.append("")
    for entry in payload.get("projects", []):
        project = entry["project"]
        review_payload = entry["review"]
        lines.append(f"## {review_payload['project_id']}: {review_payload['title']}")
        lines.append("")
        for finding in review_payload["findings"]:
            lines.append(f"### {finding['check']} ({finding['severity']})")
            lines.append("")
            lines.append(finding["summary"])
            lines.append("")
        if not review_payload["findings"]:
            lines.append(
                "No findings. The reported score reproduces and no check raised an "
                "issue."
            )
            lines.append("")
        lines.append(f"Rows: {project['rows']}. Checks run: {len(review_payload['checks_run'])}.")
        lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:
    """Entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "list-projects":
        return cmd_list_projects()
    if args.command == "list-checks":
        return cmd_list_checks()
    if args.command == "run":
        return cmd_run(args)
    if args.command == "report":
        return cmd_report(args)

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
