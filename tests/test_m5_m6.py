import builtins

import pytest

from orchestrator.console import ConsoleUI
from orchestrator.decisions import (
    ResponseRejected,
    findings_to_adjudicate,
    validate_developer_response,
)
from orchestrator.human import human_review, resolve_disputes
from schemas.decisions import DeveloperResponse, FindingDecision, HumanDecision
from schemas.review import Review


class ScriptedUI(ConsoleUI):
    def __init__(self, answers: list[str]) -> None:
        self.answers = list(answers)
        self.said: list[str] = []

    def say(self, text: str = "") -> None:
        self.said.append(text)

    def ask(self, prompt: str) -> str:
        return self.answers.pop(0)


def _finding(fid: str, severity: str = "high") -> dict:
    return {"id": fid, "severity": severity, "category": "bug", "claim": f"claim {fid}",
            "evidence": f"evidence {fid}"}


REVIEW = Review(
    verdict="changes_required",
    requirements=[{"requirement_id": "REQ-001", "status": "fail", "evidence": "e"}],
    findings=[_finding("F-001"), _finding("F-002"), _finding("F-003", "low")],
)


def _human(*pairs: tuple[str, str]) -> list[HumanDecision]:
    return [HumanDecision(finding_id=fid, action=action) for fid, action in pairs]


def _decide(fid: str, decision: str = "accept", evidence: str | None = None,
            action: str | None = "fix it") -> FindingDecision:
    return FindingDecision(finding_id=fid, decision=decision, reasoning="because",
                           evidence=evidence, action=action)


ALL_INVESTIGATE = _human(("F-001", "investigate"), ("F-002", "investigate"),
                         ("F-003", "investigate"))


# --- console ---------------------------------------------------------------

@pytest.mark.parametrize("answer,default,expected", [
    ("", False, False), ("", True, True), ("y", False, True), ("no", True, False),
])
def test_confirm_uses_default_on_enter(answer: str, default: bool, expected: bool) -> None:
    assert ScriptedUI([answer]).confirm("Run?", default=default) is expected


def test_confirm_reasks_on_nonsense() -> None:
    ui = ScriptedUI(["maybe", "y"])
    assert ui.confirm("Run?") is True
    assert "Please answer y or n." in ui.said


def test_console_reads_from_input(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(builtins, "input", lambda prompt: "y")
    assert ConsoleUI().confirm("Run?") is True


def test_choose_reasks_until_valid() -> None:
    ui = ScriptedUI(["x", "b"])
    assert ui.choose("pick: ", {"a": "A", "b": "B"}) == "b"


# --- human review (M5) -----------------------------------------------------

def test_human_review_records_each_choice() -> None:
    ui = ScriptedUI(["a", "r", "not relevant", "c", "use integers only"])
    decisions = human_review(ui, REVIEW)
    assert [(d.finding_id, d.action, d.note) for d in decisions] == [
        ("F-001", "approve", None),
        ("F-002", "reject", "not relevant"),
        ("F-003", "instruct", "use integers only"),
    ]


def test_enter_means_investigate() -> None:
    decisions = human_review(ScriptedUI(["", "", ""]), REVIEW)
    assert {d.action for d in decisions} == {"investigate"}


def test_custom_instructions_cannot_be_blank() -> None:
    ui = ScriptedUI(["c", "  ", "be strict", "", ""])
    assert human_review(ui, REVIEW)[0].note == "be strict"


def test_finding_details_are_shown() -> None:
    ui = ScriptedUI(["", "", ""])
    human_review(ui, REVIEW)
    assert any("Claim:         claim F-001" in line for line in ui.said)


# --- adjudication (M6) -----------------------------------------------------

def test_human_rejected_findings_are_not_adjudicated() -> None:
    human = _human(("F-001", "reject"), ("F-002", "approve"), ("F-003", "investigate"))
    assert [f.id for f in findings_to_adjudicate(REVIEW, human)] == ["F-002", "F-003"]


def test_complete_response_is_accepted() -> None:
    response = DeveloperResponse(decisions=[
        _decide("F-001"), _decide("F-002", "partially_accept"),
        _decide("F-003", "reject", evidence="add(2, 3) returns 5 in test_add", action=None),
    ])
    validate_developer_response(response, REVIEW, ALL_INVESTIGATE)


def test_missing_decision_is_rejected() -> None:
    response = DeveloperResponse(decisions=[_decide("F-001"), _decide("F-002")])
    with pytest.raises(ResponseRejected, match="No decision for: F-003"):
        validate_developer_response(response, REVIEW, ALL_INVESTIGATE)


def test_decision_on_human_rejected_finding_is_rejected() -> None:
    human = _human(("F-001", "reject"), ("F-002", "investigate"), ("F-003", "investigate"))
    response = DeveloperResponse(decisions=[_decide("F-001"), _decide("F-002"),
                                            _decide("F-003")])
    with pytest.raises(ResponseRejected, match="need none: F-001"):
        validate_developer_response(response, REVIEW, human)


def test_duplicate_decisions_are_rejected() -> None:
    response = DeveloperResponse(decisions=[_decide("F-001"), _decide("F-001"),
                                            _decide("F-002"), _decide("F-003")])
    with pytest.raises(ResponseRejected, match="more than once"):
        validate_developer_response(response, REVIEW, ALL_INVESTIGATE)


@pytest.mark.parametrize("action", ["approve", "instruct"])
def test_developer_cannot_reject_what_the_human_approved(action: str) -> None:
    human = _human(("F-001", action), ("F-002", "investigate"), ("F-003", "investigate"))
    response = DeveloperResponse(decisions=[
        _decide("F-001", "reject", evidence="e", action=None),
        _decide("F-002"), _decide("F-003"),
    ])
    with pytest.raises(ResponseRejected, match="approved by the human"):
        validate_developer_response(response, REVIEW, human)


def test_rejection_needs_evidence() -> None:
    response = DeveloperResponse(decisions=[
        _decide("F-001", "reject", evidence="  ", action=None),
        _decide("F-002"), _decide("F-003"),
    ])
    with pytest.raises(ResponseRejected, match="needs evidence"):
        validate_developer_response(response, REVIEW, ALL_INVESTIGATE)


@pytest.mark.parametrize("decision", ["accept", "partially_accept"])
def test_acceptance_needs_an_action(decision: str) -> None:
    response = DeveloperResponse(decisions=[
        _decide("F-001", decision, action=None), _decide("F-002"), _decide("F-003"),
    ])
    with pytest.raises(ResponseRejected, match="needs an action"):
        validate_developer_response(response, REVIEW, ALL_INVESTIGATE)


# --- disputes (section 9) --------------------------------------------------

def test_disputes_default_to_upholding_the_rejection() -> None:
    response = DeveloperResponse(decisions=[
        _decide("F-001", "reject", evidence="e", action=None),
        _decide("F-002"),
    ])
    overridden, dismissed = resolve_disputes(ScriptedUI([""]), REVIEW, response)
    assert overridden == []
    assert [f.id for f in dismissed] == ["F-001"]


def test_human_can_override_a_rejection() -> None:
    response = DeveloperResponse(decisions=[
        _decide("F-001", "reject", evidence="e", action=None),
        _decide("F-003", "reject", evidence="e", action=None),
    ])
    overridden, dismissed = resolve_disputes(ScriptedUI(["o", "u"]), REVIEW, response)
    assert overridden == ["F-001"]
    assert [f.id for f in dismissed] == ["F-003"]
