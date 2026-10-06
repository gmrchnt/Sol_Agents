from schemas.decisions import DeveloperResponse, HumanDecision
from schemas.review import Finding, Review

MUST_ADDRESS = {"approve", "instruct"}


class ResponseRejected(ValueError):
    """The Developer's adjudication failed deterministic validation."""


def findings_to_adjudicate(review: Review, human_decisions: list[HumanDecision]) -> list[Finding]:
    rejected = {d.finding_id for d in human_decisions if d.action == "reject"}
    return [finding for finding in review.findings if finding.id not in rejected]


def validate_developer_response(
    response: DeveloperResponse,
    review: Review,
    human_decisions: list[HumanDecision],
) -> None:
    expected = [f.id for f in findings_to_adjudicate(review, human_decisions)]
    given = [d.finding_id for d in response.decisions]

    duplicates = sorted({fid for fid in given if given.count(fid) > 1})
    if duplicates:
        raise ResponseRejected(f"Decided more than once: {', '.join(duplicates)}")

    unknown = sorted(set(given) - set(expected))
    if unknown:
        raise ResponseRejected(
            f"Decisions given for findings that need none: {', '.join(unknown)}. "
            f"Only decide: {', '.join(expected) or 'nothing'}."
        )

    missing = [fid for fid in expected if fid not in given]
    if missing:
        raise ResponseRejected(f"No decision for: {', '.join(missing)}")

    human = {d.finding_id: d.action for d in human_decisions}
    for decision in response.decisions:
        fid = decision.finding_id
        if decision.decision == "reject":
            if human.get(fid) in MUST_ADDRESS:
                raise ResponseRejected(
                    f"{fid} was approved by the human. You may accept or partially "
                    f"accept it, but not reject it."
                )
            if not (decision.evidence or "").strip():
                raise ResponseRejected(
                    f"Rejecting {fid} needs evidence: a counterexample, an invariant, "
                    f"a passing test or the code path that shows the claim is false."
                )
        elif not (decision.action or "").strip():
            raise ResponseRejected(
                f"{decision.decision} on {fid} needs an action describing the fix you will make."
            )
