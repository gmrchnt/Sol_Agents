from schemas.requirements import Specification
from schemas.review import Review

BLOCKING_SEVERITIES = {"critical", "high"}

# Phrases a finding uses when its own evidence says it isn't a defect.
NOT_A_DEFECT_PHRASES = (
    "not a defect",
    "not a bug",
    "no defect",
    "not an issue",
)


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


class ReviewRejected(ValueError):
    """The model produced a review that failed deterministic validation."""


def validate_review(
    review: Review,
    specification: Specification,
    *,
    changes_made: bool,
) -> None:
    expected = [req.id for req in specification.requirements]
    reported = [result.requirement_id for result in review.requirements]

    duplicates = sorted({rid for rid in reported if reported.count(rid) > 1})
    if duplicates:
        raise ReviewRejected(
            f"Requirements reported more than once: {', '.join(duplicates)}"
        )

    unknown = sorted(set(reported) - set(expected))
    if unknown:
        raise ReviewRejected(
            f"Review names requirements not in the specification: {', '.join(unknown)}"
        )

    missing = [rid for rid in expected if rid not in reported]
    if missing:
        raise ReviewRejected(f"Review gives no status for: {', '.join(missing)}")

    finding_ids = [finding.id for finding in review.findings]
    if len(finding_ids) != len(set(finding_ids)):
        raise ReviewRejected("Finding IDs must be unique.")

    not_reviewer = [fid for fid in finding_ids if not fid.startswith("F-")]
    if not_reviewer:
        raise ReviewRejected(
            f"Your findings must be numbered F-001, F-002, ...: {', '.join(not_reviewer)}"
        )

    claims = {_normalize(f.claim): f.id for f in review.findings}
    for suggestion in review.suggestions:
        twin = claims.get(_normalize(suggestion.description))
        if twin:
            raise ReviewRejected(
                f"{suggestion.id} repeats {twin}. A defect is a finding; a suggestion is "
                f"only an optional improvement or addition. Report it once."
            )

    first_with_claim: dict[str, str] = {}
    for finding in review.findings:
        claim = _normalize(finding.claim)
        if claim in first_with_claim:
            raise ReviewRejected(
                f"{finding.id} repeats {first_with_claim[claim]}. Report each issue once."
            )
        first_with_claim[claim] = finding.id

    for finding in review.findings:
        text = _normalize(f"{finding.claim} {finding.evidence}")
        phrase = next((p for p in NOT_A_DEFECT_PHRASES if p in text), None)
        if phrase:
            raise ReviewRejected(
                f"{finding.id} says {phrase!r} about itself. A finding must be a defect; "
                f"leave out anything your own evidence shows is acceptable."
            )

    for finding in review.findings:
        if finding.requirement_id is not None and finding.requirement_id not in expected:
            raise ReviewRejected(
                f"{finding.id} refers to unknown requirement {finding.requirement_id}."
            )

    if not changes_made:
        passed = [r.requirement_id for r in review.requirements if r.status == "pass"]
        if passed:
            raise ReviewRejected(
                f"The project has no changes (git diff is empty), so nothing "
                f"can pass yet: {', '.join(passed)}. Mark these uncertain or fail."
            )

    if review.verdict == "approved":
        unresolved = [r.requirement_id for r in review.requirements if r.status != "pass"]
        if unresolved:
            raise ReviewRejected(
                f"Verdict is approved but these did not pass: {', '.join(unresolved)}"
            )
        blocking = [f.id for f in review.findings if f.severity in BLOCKING_SEVERITIES]
        if blocking:
            raise ReviewRejected(
                f"Verdict is approved despite blocking findings: {', '.join(blocking)}"
            )