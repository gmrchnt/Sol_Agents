from orchestrator.console import ConsoleUI
from schemas.decisions import DeveloperResponse, HumanDecision
from schemas.review import Finding, Review

REVIEW_CHOICES = {"a": "approve", "r": "reject", "i": "investigate", "c": "instruct"}


def describe_finding(finding: Finding) -> str:
    location = finding.file or ""
    if finding.file and finding.line:
        location += f":{finding.line}"

    lines = [f"{finding.id}  [{finding.severity} / {finding.category}]  {location}".rstrip()]
    if finding.requirement_id:
        lines.append(f"  Requirement:   {finding.requirement_id}")
    lines.append(f"  Claim:         {finding.claim}")
    lines.append(f"  Evidence:      {finding.evidence}")
    if finding.reproduction:
        lines.append(f"  Reproduce:     {finding.reproduction}")
    if finding.suggested_fix:
        lines.append(f"  Suggested fix: {finding.suggested_fix}")
    return "\n".join(lines)


def human_review(ui: ConsoleUI, review: Review) -> list[HumanDecision]:
    """Section 9 checkpoint: the human rules on each finding before the Developer sees it."""
    decisions = []
    for finding in review.findings:
        ui.say()
        ui.say(describe_finding(finding))
        key = ui.choose(
            "[a]pprove  [r]eject  [i]nvestigate  [c]ustom instructions  "
            "(Enter = investigate): ",
            REVIEW_CHOICES,
            default="i",
        )
        action = REVIEW_CHOICES[key]

        note = None
        if action == "instruct":
            note = ui.ask_required("Instructions for the Developer: ")
        elif action == "reject":
            note = ui.ask("Reason (optional): ").strip() or None

        decisions.append(HumanDecision(finding_id=finding.id, action=action, note=note))
    return decisions


def resolve_disputes(
    ui: ConsoleUI, review: Review, response: DeveloperResponse
) -> tuple[list[str], list[Finding]]:
    """The human settles each finding the Developer rejected.

    Returns (finding ids the Developer must now fix, findings dismissed).
    """
    by_id = {finding.id: finding for finding in review.findings}
    overridden: list[str] = []
    dismissed: list[Finding] = []

    for decision in response.decisions:
        if decision.decision != "reject":
            continue
        finding = by_id[decision.finding_id]
        ui.say()
        ui.say(describe_finding(finding))
        ui.say(f"  Developer rejects it: {decision.reasoning}")
        ui.say(f"  Developer's evidence: {decision.evidence}")
        key = ui.choose(
            "[u]phold the rejection  [o]verride - the Developer must fix it  "
            "(Enter = uphold): ",
            {"u": "uphold", "o": "override"},
            default="u",
        )
        if key == "o":
            overridden.append(finding.id)
        else:
            dismissed.append(finding)

    return overridden, dismissed
