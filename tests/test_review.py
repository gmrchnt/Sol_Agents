import pytest
from pydantic import ValidationError

from orchestrator.review import ReviewRejected, validate_review
from schemas.requirements import Specification
from schemas.review import Review

SPEC = Specification(
    summary="s",
    requirements=[
        {"id": "REQ-001", "description": "d", "acceptance_criteria": ["x"],
         "source": "explicit", "source_text": "add multiply"},
        {"id": "REQ-002", "description": "d", "acceptance_criteria": ["y"],
         "source": "explicit", "source_text": "add divide"},
    ],
)


def _result(rid: str, status: str = "pass") -> dict:
    return {"requirement_id": rid, "status": status, "evidence": "seen in diff"}


def _finding(fid: str = "F-001", severity: str = "low", requirement_id: str | None = None) -> dict:
    return {"id": fid, "severity": severity, "category": "bug",
            "requirement_id": requirement_id, "claim": "c", "evidence": "e"}


def _review(verdict: str = "approved", results: list | None = None,
            findings: list | None = None) -> Review:
    return Review(
        verdict=verdict,
        requirements=results if results is not None
        else [_result("REQ-001"), _result("REQ-002")],
        findings=findings or [],
    )


def test_accepts_complete_approved_review() -> None:
    validate_review(_review(), SPEC, changes_made=True)


def test_accepts_changes_required_with_uncertain_results() -> None:
    results = [_result("REQ-001", "uncertain"), _result("REQ-002", "fail")]
    validate_review(_review("changes_required", results), SPEC, changes_made=True)


def test_allows_approval_with_only_low_findings() -> None:
    validate_review(_review(findings=[_finding(severity="low")]), SPEC, changes_made=True)


def test_rejects_missing_requirement() -> None:
    with pytest.raises(ReviewRejected, match="no status for: REQ-002"):
        validate_review(_review(results=[_result("REQ-001")]), SPEC, changes_made=True)


def test_rejects_unknown_requirement() -> None:
    results = [_result("REQ-001"), _result("REQ-002"), _result("REQ-003")]
    with pytest.raises(ReviewRejected, match="not in the specification"):
        validate_review(_review(results=results), SPEC, changes_made=True)


def test_rejects_requirement_reported_twice() -> None:
    results = [_result("REQ-001"), _result("REQ-001"), _result("REQ-002")]
    with pytest.raises(ReviewRejected, match="more than once"):
        validate_review(_review(results=results), SPEC, changes_made=True)


def test_rejects_approval_while_requirement_uncertain() -> None:
    results = [_result("REQ-001"), _result("REQ-002", "uncertain")]
    with pytest.raises(ReviewRejected, match="did not pass"):
        validate_review(_review(results=results), SPEC, changes_made=True)


@pytest.mark.parametrize("severity", ["critical", "high"])
def test_rejects_approval_with_blocking_finding(severity: str) -> None:
    with pytest.raises(ReviewRejected, match="blocking findings"):
        validate_review(_review(findings=[_finding(severity=severity)]), SPEC, changes_made=True)


def test_rejects_duplicate_finding_ids() -> None:
    findings = [_finding("F-001"), _finding("F-001")]
    with pytest.raises(ReviewRejected, match="unique"):
        validate_review(_review("changes_required", findings=findings), SPEC, changes_made=True)


def test_rejects_finding_about_unknown_requirement() -> None:
    findings = [_finding(requirement_id="REQ-009")]
    with pytest.raises(ReviewRejected, match="unknown requirement"):
        validate_review(_review("changes_required", findings=findings), SPEC, changes_made=True)


def test_schema_rejects_invalid_status() -> None:
    with pytest.raises(ValidationError):
        _review(results=[_result("REQ-001", "maybe")])


def test_schema_rejects_empty_evidence() -> None:
    with pytest.raises(ValidationError):
        _review(results=[{"requirement_id": "REQ-001", "status": "pass", "evidence": ""}])


def test_schema_rejects_malformed_finding_id() -> None:
    with pytest.raises(ValidationError):
        _review(findings=[_finding(fid="BUG-1")])

def test_rejects_pass_when_project_has_no_changes() -> None:
    with pytest.raises(ReviewRejected, match="no changes"):
        validate_review(_review(), SPEC, changes_made=False)


def test_accepts_uncertain_when_project_has_no_changes() -> None:
    results = [_result("REQ-001", "uncertain"), _result("REQ-002", "uncertain")]
    validate_review(_review("changes_required", results), SPEC, changes_made=False)


def test_rejects_finding_repeated_with_new_id() -> None:
    first = _finding("F-001")
    repeat = {**_finding("F-002"), "claim": "  C  "}
    with pytest.raises(ReviewRejected, match="F-002 repeats F-001"):
        validate_review(_review("changes_required", findings=[first, repeat]), SPEC,
                        changes_made=True)


def test_allows_distinct_findings() -> None:
    findings = [_finding("F-001"), {**_finding("F-002"), "claim": "different"}]
    validate_review(_review("changes_required", findings=findings), SPEC, changes_made=True)


@pytest.mark.parametrize("evidence", [
    "This is consistent with the spec and is not a defect per the approved spec.",
    "Behaves as Python does; Not A Bug.",
])
def test_rejects_finding_that_says_it_is_not_a_defect(evidence: str) -> None:
    finding = {**_finding("F-001"), "evidence": evidence}
    with pytest.raises(ReviewRejected, match="must be a defect"):
        validate_review(_review("changes_required", findings=[finding]), SPEC,
                        changes_made=True)
