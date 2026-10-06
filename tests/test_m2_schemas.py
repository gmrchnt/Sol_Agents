import pytest
from pydantic import ValidationError

from orchestrator.structured import StructuredOutputError, request_structured
from schemas.decisions import DeveloperResponse, FindingDecision, HumanDecision
from schemas.requirements import Specification
from schemas.state import ProjectState
from schemas.validation import CheckResult, ValidationReport

SPEC = Specification(
    summary="s",
    requirements=[{"id": "REQ-001", "description": "d", "acceptance_criteria": ["x"],
                   "source": "explicit", "source_text": "add multiply"}],
)


def _report(**overrides: CheckResult) -> ValidationReport:
    checks = {name: CheckResult(status="pass") for name in
              ("build", "tests", "lint", "static_analysis")}
    checks.update(overrides)
    return ValidationReport(**checks)


class ScriptedLLM:
    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.prompts: list[str] = []

    def chat(self, messages, *, schema=None, temperature=None):
        self.prompts.append(messages[-1]["content"])
        return self.replies.pop(0)


class Rejected(ValueError):
    pass


def test_acceptance_criteria_items_cannot_be_empty() -> None:
    with pytest.raises(ValidationError):
        Specification(summary="s", requirements=[
            {"id": "REQ-001", "description": "d", "acceptance_criteria": [""],
             "source": "explicit", "source_text": "add multiply"}])


def test_human_decision_rejects_unknown_action() -> None:
    with pytest.raises(ValidationError):
        HumanDecision(finding_id="F-001", action="ignore")


def test_finding_decision_requires_reasoning() -> None:
    with pytest.raises(ValidationError):
        FindingDecision(finding_id="F-001", decision="accept", reasoning="")


def test_finding_ids_are_checked_in_decisions() -> None:
    with pytest.raises(ValidationError):
        FindingDecision(finding_id="BUG-1", decision="accept", reasoning="r")


def test_developer_response_may_be_empty() -> None:
    assert DeveloperResponse().decisions == []


def test_validation_report_lists_failures() -> None:
    report = _report(tests=CheckResult(status="fail", output="1 failed"))
    assert report.failures == ["tests failed:\n1 failed"]


def test_not_run_and_not_configured_are_not_failures() -> None:
    report = _report(tests=CheckResult(status="not_run"),
                     lint=CheckResult(status="not_configured"))
    assert report.failures == []


def test_unexpected_files_are_failures() -> None:
    report = _report()
    report.unexpected_files.append("old.py")
    assert report.failures == ["unexpected change: old.py"]


def test_project_state_round_trips_through_json() -> None:
    state = ProjectState(run_id="r1", user_prompt="p", specification=SPEC,
                         project_path="D:/x", original_branch="master",
                         run_branch="sol/run-r1", base_commit="abc")
    state.validation = _report()
    assert ProjectState.model_validate_json(state.model_dump_json()) == state


def test_project_state_rejects_unknown_status() -> None:
    with pytest.raises(ValidationError):
        ProjectState(run_id="r1", user_prompt="p", specification=SPEC,
                     project_path="D:/x", original_branch="master",
                     run_branch="sol/run-r1", base_commit="abc", status="done")


def test_structured_request_retries_with_the_reason() -> None:
    llm = ScriptedLLM(['{"decisions": [{"finding_id": "nope"}]}', '{"decisions": []}'])
    result = request_structured(llm, "sys", "msg", DeveloperResponse,
                                lambda r: None, Rejected, "response")
    assert result == DeveloperResponse()
    assert "rejected by deterministic validation" in llm.prompts[1]


def test_structured_request_retries_custom_rejection() -> None:
    def validate(response: DeveloperResponse) -> None:
        if not response.decisions:
            raise Rejected("need a decision")

    good = '{"decisions": [{"finding_id": "F-001", "decision": "accept", "reasoning": "r"}]}'
    llm = ScriptedLLM(['{"decisions": []}', good])
    result = request_structured(llm, "sys", "msg", DeveloperResponse, validate,
                                Rejected, "response")
    assert result.decisions[0].finding_id == "F-001"
    assert "need a decision" in llm.prompts[1]


def test_structured_request_gives_up_after_three_attempts() -> None:
    llm = ScriptedLLM(["not json"] * 3)
    with pytest.raises(StructuredOutputError, match="3 attempts"):
        request_structured(llm, "sys", "msg", DeveloperResponse, lambda r: None,
                           Rejected, "response")
