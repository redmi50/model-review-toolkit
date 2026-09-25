"""Findings, severity and the review report.

A finding is one thing the toolkit has to say about a project. It is not an
opinion: every finding carries the evidence that produced it, and the severity is
derived from a measured quantity rather than set by hand. A review that finds
nothing is a legitimate outcome, and a review that finds only low severity notes
is the common case for a project that was built carefully.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Severity is derived from a measured gap between the evaluation as it was run
# and the evaluation done correctly. The bands are stated in full so a reader can
# disagree with the threshold rather than having to guess at it.
HIGH_GAP = 0.10
MEDIUM_GAP = 0.03

SEVERITY_WEIGHTS = {"high": 3, "medium": 2, "low": 1}


@dataclass(frozen=True)
class Finding:
    """One issue raised by one check.

    ``evidence`` holds the numbers the check computed. It is rendered into the
    report verbatim, so a reader can retrace the finding without re-running
    anything, and a test can assert on the same values the report shows.
    """

    check: str
    severity: str
    summary: str
    recommendation: str
    evidence: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.severity not in SEVERITY_WEIGHTS:
            raise ValueError(
                f"Unknown severity {self.severity!r}. Expected one of "
                f"{sorted(SEVERITY_WEIGHTS)}."
            )

    @property
    def weight(self) -> int:
        """Severity as a number, for ordering and aggregation."""
        return SEVERITY_WEIGHTS[self.severity]

    def as_dict(self) -> dict[str, object]:
        """A plain dictionary for JSON output."""
        return {
            "check": self.check,
            "severity": self.severity,
            "summary": self.summary,
            "recommendation": self.recommendation,
            "evidence": self.evidence,
        }


def severity_for_share(
    share: float, *, high: float, medium: float
) -> str | None:
    """Map a measured share onto a severity, or None if the share is zero.

    Some checks measure a proportion rather than a score gap: the fraction of
    test rows that also appear in training, or the fraction of published
    importance weight held by columns with no measurable effect. The bands for
    those quantities differ by check, so they are passed in by the caller and
    stated next to the check that uses them. A share of zero is not a finding.
    """
    if share >= high:
        return "high"
    if share >= medium:
        return "medium"
    if share > 0:
        return "low"
    return None


def severity_for_gap(gap: float) -> str | None:
    """Map a measured optimism gap onto a severity, or None if there is no issue.

    The gap is the amount by which the reported score overstated the score from
    the correct evaluation. A gap at or below zero is not a finding: it means the
    evaluation as run was not optimistic, whatever the method was called.
    """
    return severity_for_share(gap, high=HIGH_GAP, medium=MEDIUM_GAP)


@dataclass
class ReviewReport:
    """Every finding raised for one project, plus the checks that were run."""

    project_id: str
    title: str
    findings: list[Finding] = field(default_factory=list)
    checks_run: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def weight(self) -> int:
        """Total severity weight across all findings."""
        return sum(finding.weight for finding in self.findings)

    @property
    def by_severity(self) -> dict[str, int]:
        """Counts of findings per severity, including zeroes."""
        counts = {name: 0 for name in SEVERITY_WEIGHTS}
        for finding in self.findings:
            counts[finding.severity] += 1
        return counts

    @property
    def passed(self) -> bool:
        """True when no finding reached the high band."""
        return not any(finding.severity == "high" for finding in self.findings)

    @property
    def sorted_findings(self) -> list[Finding]:
        """Findings ordered most severe first, then by check name."""
        return sorted(self.findings, key=lambda f: (-f.weight, f.check))

    def as_dict(self) -> dict[str, object]:
        """A plain dictionary for JSON output."""
        return {
            "project_id": self.project_id,
            "title": self.title,
            "weight": self.weight,
            "passed": self.passed,
            "by_severity": self.by_severity,
            "checks_run": list(self.checks_run),
            "notes": list(self.notes),
            "findings": [finding.as_dict() for finding in self.sorted_findings],
        }
